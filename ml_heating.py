#!/usr/bin/env python3
"""ml_heating: Online-learning controller helper for heat pump."""

import os
import time
import warnings
from datetime import datetime, timezone
from typing import Any, List, Optional, Dict
import pickle

import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
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
SCALER_FILE: str = os.getenv("SCALER_FILE", "/opt/ml_heating/ml_scaler.pkl")
STATE_FILE: str = os.getenv("STATE_FILE", "/opt/ml_heating/ml_state.pkl")
HISTORY_STEPS: int = int(os.getenv("HISTORY_STEPS", "6"))
HISTORY_STEP_MINUTES: int = int(os.getenv("HISTORY_STEP_MINUTES", "10"))
PREDICTION_HORIZON_STEPS: int = int(os.getenv("PREDICTION_HORIZON_STEPS", "6"))
PREDICTION_HORIZON_MINUTES: int = (
    PREDICTION_HORIZON_STEPS * HISTORY_STEP_MINUTES
)

# Target indoor temp
TARGET_INDOOR_TEMP_ENTITY_ID: str = os.getenv(
    "TARGET_INDOOR_TEMP_ENTITY_ID", "input_number.hp_auto_correct_target"
)
INDOOR_TEMP_ENTITY_ID: str = os.getenv(
    "INDOOR_TEMP_ENTITY_ID", "sensor.kuche_temperatur"
)
ACTUAL_OUTLET_TEMP_ENTITY_ID: str = os.getenv(
    "ACTUAL_OUTLET_TEMP_ENTITY_ID", "sensor.hp_outlet_temp"
)
DHW_STATUS_ENTITY_ID: str = os.getenv(
    "DHW_STATUS_ENTITY_ID", "binary_sensor.hp_dhw_heating_status"
)
TV_STATUS_ENTITY_ID: str = os.getenv(
    "TV_STATUS_ENTITY_ID", "input_boolean.fernseher"
)

# Additional sensor IDs
OUTDOOR_TEMP_ENTITY_ID: str = os.getenv(
    "OUTDOOR_TEMP_ENTITY_ID", "sensor.thermometer_waermepume_kompensiert"
)
PV1_POWER_ENTITY_ID: str = os.getenv(
    "PV1_POWER_ENTITY_ID", "sensor.amperestorage_pv1_power"
)
PV2_POWER_ENTITY_ID: str = os.getenv(
    "PV2_POWER_ENTITY_ID", "sensor.amperestorage_pv2_power"
)
PV3_POWER_ENTITY_ID: str = os.getenv(
    "PV3_POWER_ENTITY_ID", "sensor.solarmax_pv_power"
)
HEATING_STATUS_ENTITY_ID: str = os.getenv(
    "HEATING_STATUS_ENTITY_ID", "climate.heizung_2"
)
OPENWEATHERMAP_TEMP_ENTITY_ID: str = os.getenv(
    "OPENWEATHERMAP_TEMP_ENTITY_ID", "sensor.openweathermap_temperature"
)

# --- Debug ---
DEBUG: bool = os.getenv("DEBUG", "0") == "1"

# InfluxDB client
client: InfluxDBClient = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
query_api: Any = client.query_api()

# Load or create model and scaler
try:
    model: Any = joblib.load(MODEL_FILE)
    print("Model loaded")
except FileNotFoundError:
    model = Ridge(alpha=0.1, random_state=42)
    print("New Ridge model created")

try:
    scaler: StandardScaler = joblib.load(SCALER_FILE)
    print("Scaler loaded")
except FileNotFoundError:
    scaler = StandardScaler()
    print("New Scaler created")


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
        resp = requests.post(
            svc_url, headers=HASS_HEADERS, json=body, timeout=10
        )
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
        result.append(
            round(temp, 2) if isinstance(temp, (int, float)) else 0.0
        )
    return result


def get_pv_forecast() -> List[float]:
    flux_query = """
        import "experimental"

        stop = experimental.addDuration(d: 4h, to: now())

        from(bucket: "home_assistant/autogen")
        |> range(start: -1h, stop: stop)
        |> filter(fn: (r) => r["_measurement"] == "W")
        |> filter(fn: (r) => r["_field"] == "value")
        |> filter(fn: (r) =>
            r["entity_id"] == "pvForecastWattsPV1" or
            r["entity_id"] == "pvForecastWattsPV2"
        )
        |> group()
        |> pivot(
            rowKey: ["_time"],
            columnKey: ["entity_id"],
            valueColumn: "_value"
        )
        |> map(fn: (r) => ({
            _time: r._time,
            total: (
                if exists r["pvForecastWattsPV1"] then r["pvForecastWattsPV1"]
                else 0.0
            ) + (
                if exists r["pvForecastWattsPV2"] then r["pvForecastWattsPV2"]
                else 0.0
            )
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
    matched = series.reindex(
        anchors, method="nearest", tolerance=pd.Timedelta("30min")
    )
    results = [float(x) if pd.notna(x) else 0.0 for x in matched.tolist()]

    return results


def fetch_outlet_history(steps: int = HISTORY_STEPS) -> List[float]:
    minutes = steps * HISTORY_STEP_MINUTES
    entity_id = ACTUAL_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1]
    flux_query = f"""
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -{minutes}m)
      |> filter(fn: (r) => r["entity_id"] == "{entity_id}")
      |> filter(fn: (r) => r["_field"] == "value")
    |> aggregateWindow(
        every: {HISTORY_STEP_MINUTES}m,
        fn: mean,
        createEmpty: false
    )
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
            chunk = values[i * step:(i + 1) * step]
            result.append(float(np.mean(chunk)) if chunk else 40.0)
        return result[-steps:]
    except Exception:
        return [40.0]*steps


def fetch_indoor_history(steps: int = HISTORY_STEPS) -> List[float]:
    minutes = steps * HISTORY_STEP_MINUTES
    entity_id = INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    flux_query = f"""
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -{minutes}m)
      |> filter(fn: (r) => r["entity_id"] == "{entity_id}")
      |> filter(fn: (r) => r["_field"] == "value")
    |> aggregateWindow(
        every: {HISTORY_STEP_MINUTES}m,
        fn: mean,
        createEmpty: false
    )
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
            chunk = values[i * step:(i + 1) * step]
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


def calculate_dynamic_boost(
    error: float, outdoor_temp: float, max_boost: float = 10.0
) -> float:
    """
    Calculates a dynamic boost for the outlet temperature based on the
    indoor temperature error and the outdoor temperature.

    The boost is more aggressive when it's colder outside.

    Args:
        error: The difference between target and actual indoor temperature.
        outdoor_temp: The current outdoor temperature.
        max_boost: The maximum allowable boost in degrees.

    Returns:
        The calculated boost value in degrees.
    """
    if error <= 0.2:
        return 0.0

    # Define the range for the dynamic boost factor
    temp_low, factor_high = 0.0, 4.0  # Colder outside -> higher boost
    temp_high, factor_low = 15.0, 2.0  # Milder outside -> lower boost

    # Clamp outdoor_temp for stable interpolation
    clamped_temp = max(temp_low, min(temp_high, outdoor_temp))

    # Linear interpolation to find the dynamic factor
    temp_range = temp_high - temp_low
    factor_range = factor_high - factor_low

    # Position is 1.0 at temp_low (cold), 0.0 at temp_high (mild)
    position = (
        (temp_high - clamped_temp) / temp_range
        if temp_range > 0
        else 1.0
    )
    dynamic_boost_factor = factor_low + (position * factor_range)

    # Calculate and clamp the final boost
    boost = error * dynamic_boost_factor
    final_boost = min(boost, max_boost)

    if DEBUG:
        print("--- Dynamic Boost Calculation ---")
        print(
            f"  Outdoor Temp: {outdoor_temp:.1f}°C, "
            f"Error: {error:.2f}°C"
        )
        print(f"  Dynamic Factor: {dynamic_boost_factor:.2f}")
        print(
            f"  Calculated Boost: {boost:.2f}°C -> "
            f"Final: {final_boost:.2f}°C"
        )

    return final_boost


def calculate_baseline_outlet_temp() -> float:
    """
    Calculates the target outlet temperature based on the user's
    heating curve formula.
    """
    # --- Part 1: Calculate the base heating curve value ---
    # Define the two points for the linear heating curve
    x1, y1 = -15.0, 64.0
    x2, y2 = 18.0, 31.0
    
    delta_y = y1 - y2
    delta_x = x1 - x2
    m = delta_y / delta_x
    b = y2 - (m * x2)

    # --- Part 2: Calculate the target temperature input for the curve ---
    outdoor_temp_actual = get_ha_state(OUTDOOR_TEMP_ENTITY_ID)
    owm_temp_actual = get_ha_state(OPENWEATHERMAP_TEMP_ENTITY_ID)
    forecast_temps = get_hourly_forecast()

    # Ensure we have the necessary data
    if (
        outdoor_temp_actual is None or
        owm_temp_actual is None or
        len(forecast_temps) < 3
    ):
        return 40.0  # Return a safe default if data is missing

    # Calculate the forecast delta (2h forecast - current)
    temp_forecast_delta = forecast_temps[2] - owm_temp_actual
    
    # Weighted average of current and forecast temp
    target_temp_for_curve = (
        outdoor_temp_actual * 0.6
        + (outdoor_temp_actual + temp_forecast_delta) * 0.4
    )

    # --- Part 3: Final Calculation ---
    # Apply the formula: y = mx + b
    outlet = m * target_temp_for_curve + b
    
    # Clamp the final result between 16.0 and 65.0
    final_outlet = max(16.0, min(65.0, outlet))
    rounded_outlet = round(final_outlet)

    if DEBUG:
        print("--- Baseline Calculation ---")
        print(
            f"  Outdoor Temp: {outdoor_temp_actual:.2f}°C, "
            f"OWM Temp: {owm_temp_actual:.2f}°C"
        )
        print(
            f"  2h Forecast: {forecast_temps[2]:.2f}°C -> "
            f"Delta: {temp_forecast_delta:.2f}°C"
        )
        print(f"  Target for Curve: {target_temp_for_curve:.2f}°C")
        print(
            f"  Calculated: {outlet:.2f}°C -> Clamped: {final_outlet:.1f}°C"
            f" -> Rounded: {rounded_outlet:.1f}°C"
        )

    return float(rounded_outlet)


def get_feature_names() -> List[str]:
    """Generate the list of feature names."""
    history_features = [
        f"{name}_hist_{i}" for name in ("outlet", "indoor")
        for i in range(HISTORY_STEPS)
    ]
    delta_features = [
        f"delta_{name}_{i}" for name in ("outlet", "indoor")
        for i in range(HISTORY_STEPS)
    ]
    forecast_features = [
        f"{name}_forecast_{i+1}h" for name in ("pv", "temp")
        for i in range(4)
    ]
    base_features = [
        "outdoor_temp",
        "pv_now",
        "tv_on",
        "temp_diff",
        "hour_sin",
        "hour_cos",
        "error_target_vs_actual",
    ]
    return base_features + history_features + delta_features + forecast_features


def find_best_outlet_temp(
    model: Any,
    scaler: StandardScaler,
    features: pd.DataFrame,
    target_temp: float,
    baseline_indoor: float,
) -> float:
    # --- Get baseline from user's formula ---
    baseline_outlet = calculate_baseline_outlet_temp()
    best_temp, min_diff = baseline_outlet, float("inf")

    # --- Define a smart search range around the baseline ---
    search_radius = 5.0
    min_temp = baseline_outlet - search_radius
    max_temp = baseline_outlet + search_radius
    step = 0.5

    # Clamp the search range to reasonable values
    min_temp = max(18.0, min_temp)
    max_temp = min(65.0, max_temp)

    outlet_hist_cols = [f"outlet_hist_{i}" for i in range(HISTORY_STEPS)]
    delta_outlet_cols = [f"delta_outlet_{i}" for i in range(HISTORY_STEPS)]

    if DEBUG:
        print("--- Finding Best Outlet Temp ---")
        print(f"Target indoor temp: {target_temp}°C")
        print(f"Baseline outlet from formula: {baseline_outlet:.1f}°C")
        print(f"Searching from {min_temp:.1f}°C to {max_temp:.1f}°C")

    # If the search range is invalid, just return the baseline
    if min_temp > max_temp:
        if DEBUG:
            print("Invalid search range, returning baseline.")
        return baseline_outlet

    for temp_candidate in np.arange(min_temp, max_temp + step, step):
        X_candidate_df = features.copy()

        # 1. Shift outlet history and append candidate
        new_outlet_history = (
            X_candidate_df[outlet_hist_cols].values[0, 1:].tolist()
            + [temp_candidate]
        )
        X_candidate_df[outlet_hist_cols] = new_outlet_history

        # 2. Recalculate deltas
        new_delta_outlet = np.diff(
            new_outlet_history, prepend=new_outlet_history[0]
        ).tolist()
        X_candidate_df[delta_outlet_cols] = new_delta_outlet

        if DEBUG and temp_candidate in (30.0, 40.0, 50.0):
            print(f"    Outlet history tail: {new_outlet_history[-5:]}")
            print(f"    Delta history tail: {new_delta_outlet[-5:]}")

        try:
            X_candidate_scaled = scaler.transform(X_candidate_df)
            predicted_delta = model.predict(X_candidate_scaled)[0]
            predicted_indoor = baseline_indoor + predicted_delta
            if DEBUG:
                print(
                    f"  - Testing outlet {temp_candidate:.1f}°C -> "
                    f"Predicted delta: {predicted_delta:.3f}°C, "
                    f"indoor: {predicted_indoor:.2f}°C"
                )
        except Exception:
            continue
        diff = abs(predicted_indoor - target_temp)
        if diff < min_diff:
            min_diff, best_temp = diff, float(temp_candidate)
    
    if DEBUG:
        print(f"--- Optimal float temp found: {best_temp:.1f}°C ---")

    # --- Smart Rounding to nearest integer ---
    floor_temp = np.floor(best_temp)
    ceil_temp = np.ceil(best_temp)

    if floor_temp == ceil_temp:
        final_temp = best_temp
    else:
        # Predict outcome for both floor and ceil temperatures
        temps_to_check = [floor_temp, ceil_temp]
        predictions = []

        for temp_candidate in temps_to_check:
            X_candidate_df = features.copy()
            new_outlet_history = (
                X_candidate_df[outlet_hist_cols].values[0, 1:].tolist()
                + [temp_candidate]
            )
            X_candidate_df[outlet_hist_cols] = new_outlet_history
            new_delta_outlet = np.diff(
                new_outlet_history, prepend=new_outlet_history[0]
            ).tolist()
            X_candidate_df[delta_outlet_cols] = new_delta_outlet
            
            try:
                X_candidate_scaled = scaler.transform(X_candidate_df)
                predicted_delta = model.predict(X_candidate_scaled)[0]
                predicted_indoor = baseline_indoor + predicted_delta
                predictions.append((temp_candidate, predicted_indoor))
            except Exception:
                # If prediction fails, skip this candidate
                continue

        if not predictions:
            # Fallback to simple rounding if both predictions fail
            final_temp = round(best_temp)
        else:
            # Choose the integer temp that gets closer to the target
            best_int_temp, min_int_diff = (
                predictions[0][0],
                abs(predictions[0][1] - target_temp),
            )
            for temp, indoor in predictions:
                diff = abs(indoor - target_temp)
                if diff < min_int_diff:
                    min_int_diff = diff
                    best_int_temp = temp
            final_temp = best_int_temp

            if DEBUG:
                print("--- Smart Rounding ---")
                for temp, indoor in predictions:
                    print(
                        f"  - Candidate {temp}°C -> Predicted indoor: "
                        f"{indoor:.2f}°C (Diff: {abs(indoor - target_temp):.2f})"
                    )
                print(f"  -> Chose: {final_temp}°C")

    return float(final_temp)


def initial_train_model(
    model: Any, scaler: StandardScaler, lookback_hours: int = 6
) -> tuple[Any, StandardScaler]:
    """
    Train the model on the last `lookback_hours` of data to predict the
    change in indoor temperature `PREDICTION_HORIZON_STEPS` steps.
    """
    feature_names = get_feature_names()
    hp_outlet_temp_id = ACTUAL_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1]
    kuche_temperatur_id = INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    fernseher_id = TV_STATUS_ENTITY_ID.split(".", 1)[-1]
    outdoor_temp_id = OUTDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    pv1_power_id = PV1_POWER_ENTITY_ID.split(".", 1)[-1]
    pv2_power_id = PV2_POWER_ENTITY_ID.split(".", 1)[-1]
    pv3_power_id = PV3_POWER_ENTITY_ID.split(".", 1)[-1]
    
    flux_query = f"""
        from(bucket: "{INFLUX_BUCKET}")
        |> range(start: -{lookback_hours}h)
        |> filter(fn: (r) => r["_field"] == "value")
        |> filter(fn: (r) =>
            r["entity_id"] == "{hp_outlet_temp_id}" or
            r["entity_id"] == "{kuche_temperatur_id}" or
            r["entity_id"] == "{outdoor_temp_id}" or
            r["entity_id"] == "{pv1_power_id}" or
            r["entity_id"] == "{pv2_power_id}" or
            r["entity_id"] == "{pv3_power_id}" or
            r["entity_id"] == "{fernseher_id}"
        )
        |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
        |> pivot(
            rowKey:["_time"],
            columnKey:["entity_id"],
            valueColumn:"_value"
        )
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
        return model, scaler

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
        pv_now = sum(
            row.get(k, 0.0) for k in [
                pv1_power_id, pv2_power_id, pv3_power_id
            ]
        )
        tv_on = 1.0 if row.get(fernseher_id, 0.0) == 1 else 0.0

        # --- New features ---
        temp_diff = outdoor - current_indoor_val
        hour = row["_time"].hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        # This is a simplification for training; in production, we'd use the live target.
        # This is a simplification for training. In production, we'd use the
        # live target. For historical training, we assume the target was the
        # setpoint at that time. A more advanced version could store
        # historical targets. For now, we'll use a placeholder or a
        # reasonable assumption. Let's assume the target was slightly above
        # the actual, e.g., 21.5
        # For historical training, we don't know the real target temp.
        # Assuming it was the same as the actual temp makes the error 0,
        # preventing the model from learning from faulty error data.
        # The real error will be used during live prediction.
        assumed_target_temp = current_indoor_val
        error_target_vs_actual = assumed_target_temp - current_indoor_val

        outlet_hist_series = df.iloc[idx - outlet_steps + 1: idx + 1][
            hp_outlet_temp_id
        ].ffill().bfill()
        indoor_hist_series = df.iloc[idx - indoor_steps + 1: idx + 1][
            kuche_temperatur_id
        ].ffill().bfill()

        outlet_hist = outlet_hist_series.tolist()
        indoor_hist = indoor_hist_series.tolist()

        if len(outlet_hist) < outlet_steps:
            padding = [outlet_hist[0]] * (outlet_steps - len(outlet_hist))
            outlet_hist = padding + outlet_hist
        if len(indoor_hist) < indoor_steps:
            padding = [indoor_hist[0]] * (indoor_steps - len(indoor_hist))
            indoor_hist = padding + indoor_hist

        outlet_hist = outlet_hist[-outlet_steps:]
        indoor_hist = indoor_hist[-indoor_steps:]

        delta_outlet = np.diff(outlet_hist, prepend=outlet_hist[0]).tolist()
        delta_indoor = np.diff(indoor_hist, prepend=indoor_hist[0]).tolist()

        pv_forecast = get_pv_forecast()
        temp_forecast = get_hourly_forecast()

        features_values = (
            [
                outdoor,
                pv_now,
                tv_on,
                temp_diff,
                hour_sin,
                hour_cos,
                error_target_vs_actual,
            ]
            + outlet_hist
            + indoor_hist
            + delta_outlet
            + delta_indoor
            + pv_forecast
            + temp_forecast
        )
        
        # Ensure feature vector length matches feature names
        if len(features_values) != len(feature_names):
            if DEBUG:
                print(
                    "Feature length mismatch: "
                    f"expected {len(feature_names)}, got {len(features_values)}. "
                    "Skipping sample."
                )
            continue

        label_delta = float(future_indoor_val) - float(current_indoor_val)
        features_list.append(features_values)
        labels_list.append(label_delta)

    if features_list and labels_list:
        try:
            X = pd.DataFrame(features_list, columns=feature_names)
            y = np.array(labels_list)

            # Scale features
            X_scaled = scaler.fit_transform(X)
            joblib.dump(scaler, SCALER_FILE)
            print("Scaler fitted and saved.")

            if DEBUG:
                print("--- Initial Training Stats ---")
                print(
                    f"Samples: {len(y)} | Delta mean: {y.mean():.3f} | "
                    f"Delta std: {y.std():.3f}"
                )
                print(f"First sample deltas (head 5): {y[:5].tolist()}")

            model.fit(X_scaled, y.ravel())

            if DEBUG:
                try:
                    # For Ridge model, we look at coefficients
                    if hasattr(model, "coef_"):
                        importances = model.coef_
                        feature_importance = pd.DataFrame(
                            {"feature": X.columns, "importance": importances}
                        )
                        feature_importance[
                            "abs_importance"
                        ] = feature_importance["importance"].abs()
                        sorted_importance = feature_importance.sort_values(
                            by="abs_importance", ascending=False
                        )
                        print("Top 5 feature coefficients (absolute value):")
                        print(sorted_importance.head(5))
                    # Fallback for other models like LightGBM
                    elif hasattr(model, "feature_importances_"):
                        importances = model.feature_importances_
                        top_idx = np.argsort(importances)[-5:][::-1]
                        print("Top feature importances:")
                        for idx in top_idx:
                            print(
                                f"  idx {int(idx)} -> {importances[idx]:.4f}"
                            )
                except AttributeError:
                    print(
                        "Model does not expose feature importances or "
                        "coefficients."
                    )

            joblib.dump(model, MODEL_FILE)
            print(
                "Initial training done on last "
                f"{lookback_hours}h of data ({len(features_list)} samples)."
            )
        except Exception:
            pass

    return model, scaler


def main(poll_interval_seconds: int = 300) -> None:
    while True:
        try:
            dhw_state_data = get_ha_state(
                DHW_STATUS_ENTITY_ID, is_binary=True
            )
            if dhw_state_data and dhw_state_data.get("state") == 'on':
                print("DHW active, skipping cycle.")
                time.sleep(poll_interval_seconds)
                continue

            # --- Get critical sensor states with validation ---
            target_indoor_temp = get_ha_state(TARGET_INDOOR_TEMP_ENTITY_ID)
            actual_indoor = get_ha_state(INDOOR_TEMP_ENTITY_ID)
            outdoor_temp = get_ha_state(OUTDOOR_TEMP_ENTITY_ID)

            critical_sensors = {
                "target_indoor": target_indoor_temp,
                "actual_indoor": actual_indoor,
                "outdoor_temp": outdoor_temp,
            }

            invalid_sensors = [
                name
                for name, value in critical_sensors.items()
                if not isinstance(value, (int, float))
            ]

            if invalid_sensors:
                print(
                    f"Invalid sensor data for: {', '.join(invalid_sensors)}. "
                    "Skipping cycle."
                )
                time.sleep(poll_interval_seconds)
                continue

            now_utc = datetime.now(timezone.utc)
            error_target_vs_actual = target_indoor_temp - actual_indoor

            # --- Online training ---
            # Note: Online training for Ridge with a scaler is more complex.
            # It would involve either partial_fit (if the model supports it)
            # or collecting a batch of new data and refitting both the scaler
            # and the model. For simplicity, we will rely on the periodic
            # `initial_train_model` call, which now handles the scaler.
            # The logic for LGBM online training is kept for reference but
            # is not active with the Ridge model.
            # --- End of online training ---

            # --- ML-based prediction path (with fallback) ---
            try:
                check_is_fitted(model)
                check_is_fitted(scaler)
                is_fitted = True
            except NotFittedError:
                is_fitted = False

            # Fallback to baseline if model is not fitted
            if not is_fitted:
                print("Model not fitted, falling back to baseline.")
                suggested_temp = calculate_baseline_outlet_temp()
                predicted_indoor = actual_indoor or target_indoor_temp
            else:
                # --- Gather all features for ML model ---
                tv_state_raw = get_ha_state(
                    TV_STATUS_ENTITY_ID, is_binary=True
                )
                is_tv_on = tv_state_raw and tv_state_raw.get("state") == "on"
                tv_on = 1.0 if is_tv_on else 0.0

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

                temp_diff = outdoor_temp - actual_indoor
                hour = now_utc.hour
                hour_sin = np.sin(2 * np.pi * hour / 24)
                hour_cos = np.cos(2 * np.pi * hour / 24)

                feature_names = get_feature_names()
                current_features_list = [
                    float(outdoor_temp),
                    pv_now,
                    tv_on,
                    temp_diff,
                    hour_sin,
                    hour_cos,
                    error_target_vs_actual,
                ] + (
                    outlet_history
                    + indoor_history
                    + np.diff(
                        outlet_history, prepend=outlet_history[0]
                    ).tolist()
                    + np.diff(
                        indoor_history, prepend=indoor_history[0]
                    ).tolist()
                    + pv_forecasts
                    + temp_forecasts
                )

                # Fallback if feature lengths don't match
                is_mismatched = hasattr(
                    model, "n_features_in_"
                ) and model.n_features_in_ != len(current_features_list)
                if len(current_features_list) != len(
                    feature_names
                ) or is_mismatched:
                    print(
                        "Feature length mismatch, falling back to baseline."
                    )
                    suggested_temp = calculate_baseline_outlet_temp()
                    predicted_indoor = actual_indoor or target_indoor_temp
                else:
                    # --- This is the successful ML path ---
                    current_features_df = pd.DataFrame(
                        [current_features_list], columns=feature_names
                    )
                    baseline_current = actual_indoor or indoor_history[-1]

                    suggested_temp = find_best_outlet_temp(
                        model,
                        scaler,
                        current_features_df,
                        target_indoor_temp,
                        baseline_current,
                    )

                    # Predict final indoor temp for logging
                    scaled_features = scaler.transform(current_features_df)
                    predicted_delta = float(model.predict(scaled_features)[0])
                    predicted_indoor = baseline_current + predicted_delta

                    # Save state only after a successful ML prediction
                    state_to_save = {
                        "features": np.array(
                            current_features_list
                        ).reshape(1, -1),
                        "timestamp": now_utc.isoformat(),
                        "baseline_indoor": (
                            float(actual_indoor)
                            if actual_indoor is not None
                            else None
                        ),
                    }
                    save_state(STATE_FILE, state_to_save)

            # --- Final calculation with boost and clamping ---
            error_boost = calculate_dynamic_boost(
                error=error_target_vs_actual, outdoor_temp=outdoor_temp
            )
            boosted_temp = suggested_temp + error_boost
            final_temp = min(boosted_temp, 65.0)

            if DEBUG:
                print("--- Final Temp Calculation ---")
                print(f"  Model Suggested: {suggested_temp:.1f}°C")
                print(f"  Dynamic Boost Applied: {error_boost:.2f}°C")
                print(f"  Boosted Temp: {boosted_temp:.1f}°C")
                print(f"  Final Temp (clamped at 65°C): {final_temp:.1f}°C")

            set_ha_state("sensor.ml_vorlauftemperatur", final_temp)
            set_ha_state("sensor.ml_predicted_indoor_temp", predicted_indoor)

            print(
                f"{now_utc.isoformat()} - Target: {target_indoor_temp}°C"
                f" | Suggested: {final_temp:.1f}°C"
                f" | Predicted: {predicted_indoor:.2f}°C"
                f" | Actual: {actual_indoor or 'N/A'}"
            )

        except Exception as exc:
            print("Error in main loop:", exc)

        time.sleep(poll_interval_seconds)


def check_is_fitted(estimator):
    """Checks if an estimator is fitted."""
    if not hasattr(estimator, "n_features_in_"):
        raise NotFittedError(f"{type(estimator).__name__} is not fitted yet.")


if __name__ == "__main__":
    model, scaler = initial_train_model(model, scaler, lookback_hours=24)
    main()
