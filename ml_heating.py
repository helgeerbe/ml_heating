#!/usr/bin/env python3
"""ml_heating: Online-learning controller helper for heat pump."""

import os
import time
import warnings
from datetime import datetime, timezone, timedelta
from typing import Any, List, Optional, Dict
import pickle

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
import joblib
from influxdb_client import InfluxDBClient
import requests
from dotenv import load_dotenv

load_dotenv()

# --- Home Assistant API ---
HASS_URL: str = os.getenv("HASS_URL", "https://home.erbehome.de")
HASS_TOKEN: str = os.getenv("HASS_TOKEN", "").strip()
HASS_HEADERS: dict[str, str] = {
    "Authorization": f"Bearer {HASS_TOKEN}",
    "Content-Type": "application/json",
}

# --- InfluxDB ---
INFLUX_URL: str = os.getenv("INFLUX_URL", "https://influxdb.erbehome.de")
INFLUX_TOKEN: str = os.getenv("INFLUX_TOKEN", "")
INFLUX_ORG: str = os.getenv("INFLUX_ORG", "erbehome")
INFLUX_BUCKET: str = os.getenv("INFLUX_BUCKET", "home_assistant/autogen")

# Model and state files
MODEL_FILE: str = os.getenv("MODEL_FILE", "/opt/ml_heating/ml_model.pkl")
STATE_FILE: str = os.getenv("STATE_FILE", "/opt/ml_heating/ml_state.pkl")
HISTORY_STEPS: int = int(os.getenv("HISTORY_STEPS", "6"))
HISTORY_STEP_MINUTES: int = int(os.getenv("HISTORY_STEP_MINUTES", "10"))
PREDICTION_HORIZON_STEPS: int = int(os.getenv("PREDICTION_HORIZON_STEPS", "6"))
PREDICTION_HORIZON_MINUTES: int = PREDICTION_HORIZON_STEPS * HISTORY_STEP_MINUTES

# Target indoor temp
TARGET_INDOOR_TEMP: float = float(os.getenv("TARGET_INDOOR_TEMP", "21.0"))
INDOOR_TEMP_ENTITY_ID: str = os.getenv("INDOOR_TEMP_ENTITY_ID", "sensor.kuche_temperatur")
ACTUAL_OUTLET_TEMP_ENTITY_ID: str = os.getenv("ACTUAL_OUTLET_TEMP_ENTITY_ID", "sensor.hp_outlet_temp")
DHW_STATUS_ENTITY_ID: str = os.getenv("DHW_STATUS_ENTITY_ID", "binary_sensor.hp_dhw_heating_status")
TV_STATUS_ENTITY_ID: str = os.getenv("TV_STATUS_ENTITY_ID", "input_boolean.fernseher")

# Additional sensor IDs
OUTDOOR_TEMP_ENTITY_ID: str = os.getenv("OUTDOOR_TEMP_ENTITY_ID", "sensor.thermometer_waermepume_kompensiert")
PV1_POWER_ENTITY_ID: str = os.getenv("PV1_POWER_ENTITY_ID", "sensor.amperestorage_pv1_power")
PV2_POWER_ENTITY_ID: str = os.getenv("PV2_POWER_ENTITY_ID", "sensor.amperestorage_pv2_power")
PV3_POWER_ENTITY_ID: str = os.getenv("PV3_POWER_ENTITY_ID", "sensor.solarmax_pv_power")
HEATING_STATUS_ENTITY_ID: str = os.getenv("HEATING_STATUS_ENTITY_ID", "climate.heizung_2")

# --- Debug ---
DEBUG: bool = os.getenv("DEBUG", "0") == "1"

# InfluxDB client
client: InfluxDBClient = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
query_api: Any = client.query_api()

# Load or create model
try:
    model: Any = joblib.load(MODEL_FILE)
    print("Model loaded")
except FileNotFoundError:
    model = RandomForestRegressor(
        n_estimators=100, warm_start=True, max_depth=10, min_samples_split=5
    )
    print("New model created")


# -----------------------------
# Helper Functions
# -----------------------------

def get_ha_state(entity_id: str, is_binary: bool = False) -> Optional[Any]:
    url = f"{HASS_URL}/api/states/{entity_id}"
    try:
        resp = requests.get(url, headers=HASS_HEADERS, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        warnings.warn(f"HA request error {entity_id}: {exc}")
        return None

    data = resp.json()
    state = data.get("state")
    if state in (None, "unknown", "unavailable"):
        return None

    if is_binary:
        return data

    try:
        return float(state)
    except (TypeError, ValueError):
        return state


def set_ha_state(entity_id: str, value: float) -> None:
    url = f"{HASS_URL}/api/states/{entity_id}"
    payload = {"state": round(value, 1)}
    try:
        requests.post(url, headers=HASS_HEADERS, json=payload, timeout=10)
    except requests.RequestException as exc:
        warnings.warn(f"HA state set failed for {entity_id}: {exc}")


def get_hourly_forecast() -> List[float]:
    svc_url = f"{HASS_URL}/api/services/weather/get_forecasts?return_response"
    body = {"entity_id": ["weather.openweathermap"], "type": "hourly"}

    try:
        resp = requests.post(svc_url, headers=HASS_HEADERS, json=body, timeout=10)
        resp.raise_for_status()
        data = resp.json()["service_response"]
    except Exception:
        return [0.0, 0.0, 0.0, 0.0]

    try:
        forecast_list = data["weather.openweathermap"]["forecast"]
    except (KeyError, TypeError):
        return [0.0, 0.0, 0.0, 0.0]

    result = []
    for entry in forecast_list[:4]:
        temp = entry.get("temperature") if isinstance(entry, dict) else None
        result.append(round(temp, 2) if isinstance(temp, (int, float)) else 0.0)
    return result


def get_pv_forecast() -> List[float]:
    debug = os.getenv("PV_DEBUG", "0") == "1"

    flux_query = """
        import "experimental"

        stop = experimental.addDuration(d: 4h, to: now())

        from(bucket: "home_assistant/autogen")
        |> range(start: -1h, stop: stop)
        |> filter(fn: (r) => r["_measurement"] == "W")
        |> filter(fn: (r) => r["_field"] == "value")
        |> filter(fn: (r) => r["entity_id"] == "pvForecastWattsPV1" or r["entity_id"] == "pvForecastWattsPV2")
        |> group()
        |> pivot(rowKey: ["_time"], columnKey: ["entity_id"], valueColumn: "_value")
        |> map(fn: (r) => ({
            _time: r._time,
            total: (if exists r["pvForecastWattsPV1"] then r["pvForecastWattsPV1"] else 0.0) +
                   (if exists r["pvForecastWattsPV2"] then r["pvForecastWattsPV2"] else 0.0)
        }))
        |> sort(columns: ["_time"])
        |> yield(name: "4h_total_forecast")
    """

    try:
        raw = query_api.query_data_frame(flux_query)
    except Exception:
        return [0.0, 0.0, 0.0, 0.0]

    df = pd.concat(raw, ignore_index=True) if isinstance(raw, list) else raw
    if df.empty or "_time" not in df.columns or "total" not in df.columns:
        return [0.0, 0.0, 0.0, 0.0]

    df["_time"] = pd.to_datetime(df["_time"], utc=True)
    df.sort_values("_time", inplace=True)

    now_utc = pd.Timestamp(datetime.now(timezone.utc))
    first_anchor = now_utc.ceil("h")
    anchors = pd.date_range(start=first_anchor, periods=4, freq="h", tz="UTC")

    series = df.set_index("_time")["total"].sort_index()
    matched = series.reindex(anchors, method="nearest", tolerance=pd.Timedelta("30min"))
    results = [float(x) if pd.notna(x) else 0.0 for x in matched.tolist()]

    return results


def fetch_outlet_history(steps: int = HISTORY_STEPS) -> List[float]:
    minutes = steps * HISTORY_STEP_MINUTES
    entity_id = ACTUAL_OUTLET_TEMP_ENTITY_ID.split('.', 1)[-1]
    flux_query = f"""
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -{minutes}m)
      |> filter(fn: (r) => r["entity_id"] == "{entity_id}")
      |> filter(fn: (r) => r["_field"] == "value")
    |> aggregateWindow(every: {HISTORY_STEP_MINUTES}m, fn: mean, createEmpty: false)
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> keep(columns:["_time","value"])
      |> sort(columns:["_time"])
      |> tail(n: {steps})
    """
    try:
        df = query_api.query_data_frame(flux_query)
        df = pd.concat(df, ignore_index=True) if isinstance(df, list) else df
        df["value"] = df["value"].ffill().bfill()
        values = df["value"].tolist()
        # Ensure we have enough data points, otherwise pad with last known value
        if len(values) < steps:
            padding = [values[-1] if values else 40.0] * (steps - len(values))
            values.extend(padding)
        
        # Simple downsampling if more data than steps
        step = max(1, len(values) // steps)
        result = []
        for i in range(steps):
            chunk = values[i*step:(i+1)*step]
            result.append(float(np.mean(chunk)) if chunk else 40.0)
        return result[-steps:]
    except Exception:
        return [40.0]*steps


def fetch_indoor_history(steps: int = HISTORY_STEPS) -> List[float]:
    minutes = steps * HISTORY_STEP_MINUTES
    entity_id = INDOOR_TEMP_ENTITY_ID.split('.', 1)[-1]
    flux_query = f"""
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -{minutes}m)
      |> filter(fn: (r) => r["entity_id"] == "{entity_id}")
      |> filter(fn: (r) => r["_field"] == "value")
    |> aggregateWindow(every: {HISTORY_STEP_MINUTES}m, fn: mean, createEmpty: false)
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> keep(columns:["_time","value"])
      |> sort(columns:["_time"])
      |> tail(n: {steps})
    """
    try:
        df = query_api.query_data_frame(flux_query)
        df = pd.concat(df, ignore_index=True) if isinstance(df, list) else df
        df["value"] = df["value"].ffill().bfill()
        values = df["value"].tolist()
        # Ensure we have enough data points, otherwise pad with last known value
        if len(values) < steps:
            padding = [values[-1] if values else 21.0] * (steps - len(values))
            values.extend(padding)

        # Simple downsampling if more data than steps
        step = max(1, len(values) // steps)
        result = []
        for i in range(steps):
            chunk = values[i*step:(i+1)*step]
            result.append(float(np.mean(chunk)) if chunk else 21.0)
        return result[-steps:]
    except Exception:
        return [21.0]*steps

def load_state(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def save_state(path: str, data: Dict[str, Any]) -> None:
    try:
        with open(path, "wb") as f:
            pickle.dump(data, f)
    except Exception:
        pass


def find_best_outlet_temp(model: Any, features: List[float], target_temp: float, baseline_indoor: float) -> float:
    min_temp, max_temp, step = 30.0, 55.0, 0.5
    best_temp, min_diff = 40.0, float("inf")

    # --- Define feature indices based on the structure in main() ---
    # Structure: [base (3)] + [outlet_hist (H)] + [indoor_hist (H)] + [delta_outlet (H)] + ...
    outlet_hist_start = 3
    outlet_hist_end = outlet_hist_start + HISTORY_STEPS
    delta_outlet_start = outlet_hist_end + HISTORY_STEPS
    delta_outlet_end = delta_outlet_start + HISTORY_STEPS

    if DEBUG:
        print("--- Finding Best Outlet Temp ---")
        print(f"Target indoor temp: {target_temp}°C")

    for temp_candidate in np.arange(min_temp, max_temp+step, step):
        X_candidate_list = features[:]  # Work on a copy to avoid side effects

        # 1. Shift the outlet history forward and append the candidate temperature
        current_outlet_history = X_candidate_list[outlet_hist_start:outlet_hist_end]
        new_outlet_history = current_outlet_history[1:] + [temp_candidate]
        X_candidate_list[outlet_hist_start:outlet_hist_end] = new_outlet_history

        # 2. Recalculate the delta values based on the updated history
        new_delta_outlet = np.diff(new_outlet_history, prepend=new_outlet_history[0]).tolist()
        X_candidate_list[delta_outlet_start:delta_outlet_end] = new_delta_outlet

        if DEBUG and temp_candidate in (30.0, 40.0, 50.0):
            print(f"    Outlet history tail: {new_outlet_history[-5:]}")
            print(f"    Delta history tail: {new_delta_outlet[-5:]}")
        
        X_candidate = np.array(X_candidate_list).reshape(1, -1)
        try:
            predicted_delta = model.predict(X_candidate)[0]
            predicted_indoor = baseline_indoor + predicted_delta
            if DEBUG:
                print(f"  - Testing outlet {temp_candidate:.1f}°C -> Predicted delta: {predicted_delta:.3f}°C, indoor: {predicted_indoor:.2f}°C")
        except Exception:
            continue
        diff = abs(predicted_indoor - target_temp)
        if diff < min_diff:
            min_diff, best_temp = diff, float(temp_candidate)
    
    if DEBUG:
        print(f"--- Best outlet temp found: {best_temp:.1f}°C ---")
    return best_temp


def initial_train_model(model: Any, lookback_hours: int = 6) -> Any:
    """
    Train the model on the last `lookback_hours` of data to predict the change in indoor
    temperature `PREDICTION_HORIZON_STEPS` steps (~`PREDICTION_HORIZON_MINUTES` minutes) ahead.
    """
    hp_outlet_temp_id = ACTUAL_OUTLET_TEMP_ENTITY_ID.split('.', 1)[-1]
    kuche_temperatur_id = INDOOR_TEMP_ENTITY_ID.split('.', 1)[-1]
    fernseher_id = TV_STATUS_ENTITY_ID.split('.', 1)[-1]
    outdoor_temp_id = OUTDOOR_TEMP_ENTITY_ID.split('.', 1)[-1]
    pv1_power_id = PV1_POWER_ENTITY_ID.split('.', 1)[-1]
    pv2_power_id = PV2_POWER_ENTITY_ID.split('.', 1)[-1]
    pv3_power_id = PV3_POWER_ENTITY_ID.split('.', 1)[-1]
    
    flux_query = f"""
        from(bucket: "{INFLUX_BUCKET}")
        |> range(start: -{lookback_hours}h)
        |> filter(fn: (r) => r["_field"]=="value")
        |> filter(fn: (r) => r["entity_id"]=="{hp_outlet_temp_id}"
                              or r["entity_id"]=="{kuche_temperatur_id}"
                              or r["entity_id"]=="{outdoor_temp_id}"
                              or r["entity_id"]=="{pv1_power_id}"
                              or r["entity_id"]=="{pv2_power_id}"
                              or r["entity_id"]=="{pv3_power_id}"
                              or r["entity_id"]=="{fernseher_id}")
        |> aggregateWindow(every:5m, fn:mean, createEmpty:false)
        |> pivot(rowKey:["_time"], columnKey:["entity_id"], valueColumn:"_value")
        |> sort(columns:["_time"])
    """
    try:
        raw = query_api.query_data_frame(flux_query)
        df = pd.concat(raw, ignore_index=True) if isinstance(raw, list) else raw
        df["_time"] = pd.to_datetime(df["_time"], utc=True)
        df.sort_values("_time", inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
    except Exception:
        return model

    features_list, labels_list = [], []
    outlet_steps = indoor_steps = HISTORY_STEPS
    start_idx = max(outlet_steps, indoor_steps) - 1

    for idx in range(start_idx, len(df) - PREDICTION_HORIZON_STEPS):
        row = df.iloc[idx]
        next_row = df.iloc[idx + PREDICTION_HORIZON_STEPS]
        outdoor = row.get(outdoor_temp_id, np.nan)
        if pd.isna(outdoor):
            continue
        current_indoor_val = row.get(kuche_temperatur_id, np.nan)
        future_indoor_val = next_row.get(kuche_temperatur_id, np.nan)
        if pd.isna(current_indoor_val) or pd.isna(future_indoor_val):
            continue
        pv_now = sum(row.get(k, 0.0) for k in [pv1_power_id, pv2_power_id, pv3_power_id])
        tv_on = 1.0 if row.get(fernseher_id, 0.0)==1 else 0.0

        outlet_hist_series = df.iloc[idx-outlet_steps+1:idx+1][hp_outlet_temp_id].ffill().bfill()
        indoor_hist_series = df.iloc[idx-indoor_steps+1:idx+1][kuche_temperatur_id].ffill().bfill()

        outlet_hist = outlet_hist_series.tolist()
        indoor_hist = indoor_hist_series.tolist()

        if len(outlet_hist) < outlet_steps:
            outlet_hist = ([outlet_hist[0]] * (outlet_steps - len(outlet_hist))) + outlet_hist
        if len(indoor_hist) < indoor_steps:
            indoor_hist = ([indoor_hist[0]] * (indoor_steps - len(indoor_hist))) + indoor_hist

        outlet_hist = outlet_hist[-outlet_steps:]
        indoor_hist = indoor_hist[-indoor_steps:]

        delta_outlet = np.diff(outlet_hist, prepend=outlet_hist[0]).tolist()
        delta_indoor = np.diff(indoor_hist, prepend=indoor_hist[0]).tolist()

        pv_forecast = get_pv_forecast()
        temp_forecast = get_hourly_forecast()

        features = [outdoor, pv_now, tv_on] + outlet_hist + indoor_hist + delta_outlet + delta_indoor + pv_forecast + temp_forecast
        label_delta = float(future_indoor_val) - float(current_indoor_val)
        features_list.append(features)
        labels_list.append(label_delta)

    if features_list and labels_list:
        try:
            X = np.array(features_list)
            y = np.array(labels_list)
            if DEBUG:
                print("--- Initial Training Stats ---")
                print(f"Samples: {len(y)} | Delta mean: {y.mean():.3f} | Delta std: {y.std():.3f}")
                print(f"First sample deltas (head 5): {y[:5].tolist()}")
            model.fit(X, y.ravel())
            if DEBUG:
                try:
                    importances = model.feature_importances_
                    top_idx = np.argsort(importances)[-5:][::-1]
                    print("Top feature importances:")
                    for idx in top_idx:
                        print(f"  idx {int(idx)} -> {importances[idx]:.4f}")
                except AttributeError:
                    print("Model does not expose feature_importances_")
            joblib.dump(model, MODEL_FILE)
            print(f"Initial training done on last {lookback_hours}h of data ({len(features_list)} samples).")
        except Exception:
            pass

    return model


def main(poll_interval_seconds: int = 300) -> None:
    while True:
        try:
            dhw_state_data = get_ha_state(DHW_STATUS_ENTITY_ID, is_binary=True)
            if dhw_state_data and dhw_state_data.get("state") == 'on':
                print("DHW active, skipping cycle.")
                time.sleep(poll_interval_seconds)
                continue

            # Get states needed for both training and prediction
            last_state = load_state(STATE_FILE)
            actual_indoor = get_ha_state(INDOOR_TEMP_ENTITY_ID)
            now_utc = datetime.now(timezone.utc)

            # --- Online training ---
            # Only train if heating is on
            heating_state = get_ha_state(HEATING_STATUS_ENTITY_ID)
            horizon_delta = timedelta(minutes=PREDICTION_HORIZON_MINUTES)
            should_train = False
            train_features = None
            baseline_for_training: Optional[float] = None

            if heating_state == 'heat' and last_state and actual_indoor is not None:
                last_ts_str = last_state.get("timestamp")
                if last_ts_str:
                    try:
                        last_ts = datetime.fromisoformat(last_ts_str)
                        if now_utc - last_ts >= horizon_delta:
                            train_features = np.array(last_state.get("features", [])).reshape(1, -1)
                            baseline_candidate = last_state.get("baseline_indoor")
                            if train_features.size > 0 and baseline_candidate is not None:
                                baseline_for_training = float(baseline_candidate)
                                should_train = True
                    except ValueError:
                        pass

            if should_train:
                try:
                    delta_label = float(actual_indoor) - baseline_for_training
                    y_train = np.array([delta_label]).ravel()
                    if DEBUG:
                        print("--- Online Training ---")
                        print(f"Baseline indoor (saved): {baseline_for_training:.2f}°C")
                        print(f"Training with delta label (+{PREDICTION_HORIZON_MINUTES}m): {y_train[0]:.3f}°C")
                        print(f"Feature vector: {train_features.flatten().tolist()}")
                    model.fit(train_features, y_train)
                    if DEBUG:
                        try:
                            importances = model.feature_importances_
                            top_idx = np.argsort(importances)[-5:][::-1]
                            print("Top feature importances after online update:")
                            for idx in top_idx:
                                print(f"  idx {int(idx)} -> {importances[idx]:.4f}")
                        except AttributeError:
                            pass
                    joblib.dump(model, MODEL_FILE)
                    print("Model retrained with new data.")
                except Exception as e:
                    print(f"Error during retraining: {e}")
            # --- End of online training ---

            # Get current features for prediction
            tv_state_raw = get_ha_state(TV_STATUS_ENTITY_ID, is_binary=True)
            tv_on = 1.0 if tv_state_raw and tv_state_raw.get("state") == 'on' else 0.0

            outdoor_temp = get_ha_state(OUTDOOR_TEMP_ENTITY_ID)
            pv1 = get_ha_state(PV1_POWER_ENTITY_ID)
            pv2 = get_ha_state(PV2_POWER_ENTITY_ID)
            pv3 = get_ha_state(PV3_POWER_ENTITY_ID)
            pv_now = sum(v for v in (pv1, pv2, pv3) if v is not None)

            pv_forecasts = get_pv_forecast()
            temp_forecasts = get_hourly_forecast()
            outlet_history = fetch_outlet_history(HISTORY_STEPS)
            indoor_history = fetch_indoor_history(HISTORY_STEPS)

            if actual_indoor is not None:
                indoor_history.pop(0)
                indoor_history.append(float(actual_indoor))

            current_features_list = [float(outdoor_temp or 0.0), pv_now, tv_on] + outlet_history + indoor_history + \
                                    np.diff(outlet_history, prepend=outlet_history[0]).tolist() + \
                                    np.diff(indoor_history, prepend=indoor_history[0]).tolist() + \
                                    pv_forecasts + temp_forecasts
            
            if DEBUG:
                print("--- Prediction ---")
                print(f"Feature vector for prediction: {current_features_list}")

            # Check model
            try:
                model.n_features_in_
                is_fitted = True
            except (AttributeError, NotFittedError):
                is_fitted = False

            baseline_current = None
            if actual_indoor is not None:
                baseline_current = float(actual_indoor)
            elif indoor_history:
                baseline_current = float(indoor_history[-1])
            else:
                baseline_current = TARGET_INDOOR_TEMP

            if not is_fitted or model.n_features_in_ != len(current_features_list):
                suggested_temp = 40.0
                predicted_indoor = baseline_current
            else:
                X_val = np.array(current_features_list).reshape(1, -1)
                predicted_delta = float(model.predict(X_val)[0])
                predicted_indoor = baseline_current + predicted_delta
                suggested_temp = find_best_outlet_temp(model, current_features_list, TARGET_INDOOR_TEMP, baseline_current)

            set_ha_state("sensor.ml_vorlauftemperatur", suggested_temp)
            set_ha_state("sensor.ml_predicted_indoor_temp", predicted_indoor)

            # Save current state
            save_state(STATE_FILE, {"features": np.array(current_features_list).reshape(1, -1),
                                    "timestamp": now_utc.isoformat(),
                                    "baseline_indoor": float(actual_indoor) if actual_indoor is not None else None})

            print(f"{now_utc.isoformat()} - Target: {TARGET_INDOOR_TEMP}°C | Suggested: {suggested_temp:.1f}°C | Predicted: {predicted_indoor:.2f}°C | Actual: {actual_indoor or 'N/A'}")

        except Exception as exc:
            print("Error in main loop:", exc)

        time.sleep(poll_interval_seconds)


if __name__ == "__main__":
    model = initial_train_model(model, lookback_hours=6)
    main()
