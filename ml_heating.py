#!/usr/bin/env python3
"""ml_heating: Online-learning controller helper for heat pump."""

import os
import time
import warnings
from collections import deque
from datetime import datetime, timezone
from typing import Any, List, Optional, Dict
import pickle

import pandas as pd
import numpy as np
from influxdb_client import InfluxDBClient
import requests
from dotenv import load_dotenv
from river import compose, preprocessing, ensemble, metrics, drift

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
PREDICTION_HORIZON_STEPS: int = int(
    os.getenv("PREDICTION_HORIZON_STEPS", "24")
)  # 24 steps * 5 min = 120 min
PREDICTION_HORIZON_MINUTES: int = (
    PREDICTION_HORIZON_STEPS * 5  # Now based on 5-minute intervals
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
DEFROST_STATUS_ENTITY_ID: str = os.getenv(
    "DEFROST_STATUS_ENTITY_ID", "binary_sensor.hp_defrosting_status"
)
DISINFECTION_STATUS_ENTITY_ID: str = os.getenv(
    "DISINFECTION_STATUS_ENTITY_ID",
    "binary_sensor.hp_dhw_tank_disinfection_status",
)
DHW_BOOST_HEATER_STATUS_ENTITY_ID: str = os.getenv(
    "DHW_BOOST_HEATER_STATUS_ENTITY_ID",
    "binary_sensor.hp_dhw_boost_heater_status",
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
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.2"))

# InfluxDB client
client: InfluxDBClient = InfluxDBClient(
    url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG
)
query_api: Any = client.query_api()

# Prediction smoothing
prediction_history: deque = deque(maxlen=5)

# Load or create model
try:
    with open(MODEL_FILE, "rb") as f:
        saved_data = pickle.load(f)
        if isinstance(saved_data, dict):
            model = saved_data['model']
            mae = saved_data.get('mae', metrics.MAE())
            rmse = saved_data.get('rmse', metrics.RMSE())
        else:
            # Handle old model format
            model = saved_data
            mae = metrics.MAE()
            rmse = metrics.RMSE()
    print("Model and metrics loaded")
except (FileNotFoundError, pickle.UnpicklingError):
    unscaled_features = [
        "outlet_temp",
        "outlet_temp_sq",
        "outlet_temp_cub",
        "outlet_temp_change_from_last",
        "outlet_indoor_diff",
    ]
    
    # The pipeline will be created in initial_train_model
    model = None
    mae = metrics.MAE()
    rmse = metrics.RMSE()
    print("New model and metrics created")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    mae = metrics.MAE()
    rmse = metrics.RMSE()


# -----------------------------
# Helper Functions
# -----------------------------


def get_all_ha_states() -> Dict[str, Any]:
    """Fetches all states from Home Assistant in a single API call."""
    url = f"{HASS_URL}/api/states"
    try:
        resp = requests.get(url, headers=HASS_HEADERS, timeout=10)
        resp.raise_for_status()
        # Create a dictionary mapping entity_id to its state object
        return {entity["entity_id"]: entity for entity in resp.json()}
    except requests.RequestException as exc:
        warnings.warn(f"HA request error for all states: {exc}")
        return {}


def get_ha_state(
    entity_id: str,
    states_cache: Optional[Dict[str, Any]] = None,
    is_binary: bool = False,
) -> Optional[Any]:
    """
    Retrieves the state of a Home Assistant entity, using a cache if provided.
    """
    if states_cache is None:
        # Fallback to individual request if no cache is provided
        url = f"{HASS_URL}/api/states/{entity_id}"
        try:
            resp = requests.get(url, headers=HASS_HEADERS, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            warnings.warn(f"HA request error {entity_id}: {exc}")
            return None
    else:
        data = states_cache.get(entity_id)

    if data is None:
        return None

    state = data.get("state")
    if state in (None, "unknown", "unavailable"):
        return None

    if is_binary:
        return data

    try:
        return float(state)
    except (TypeError, ValueError):
        return state


def set_ha_state(
    entity_id: str,
    value: float,
    attributes: Optional[Dict[str, Any]] = None,
    round_digits: Optional[int] = 1,
) -> None:
    """Posts a state to the Home Assistant API, with optional rounding."""
    url = f"{HASS_URL}/api/states/{entity_id}"

    # Round the value only if round_digits is specified
    if round_digits is not None:
        state_value = round(value, round_digits)
    else:
        state_value = value

    payload = {"state": state_value, "attributes": attributes or {}}
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
        return [40.0] * steps


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
        return [21.0] * steps


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


def calculate_baseline_outlet_temp(states_cache: Dict[str, Any]) -> float:
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
    outdoor_temp_actual = get_ha_state(
        OUTDOOR_TEMP_ENTITY_ID, states_cache
    )
    owm_temp_actual = get_ha_state(
        OPENWEATHERMAP_TEMP_ENTITY_ID, states_cache
    )
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
        f"{name}_hist_{i}"
        for name in ("outlet", "indoor")
        for i in range(HISTORY_STEPS)
    ]
    forecast_features = [
        f"{name}_forecast_{i+1}h" for name in ("pv", "temp") for i in range(4)
    ]
    lag_features = [
        "indoor_temp_lag_10m",
        "indoor_temp_lag_30m",
        "indoor_temp_lag_60m",
        "outlet_temp_lag_10m",
        "outlet_temp_lag_30m",
        "outlet_temp_lag_60m",
    ]
    delta_features = [
        "indoor_temp_delta_10m",
        "indoor_temp_delta_30m",
        "indoor_temp_delta_60m",
        "outlet_temp_delta_10m",
        "outlet_temp_delta_30m",
        "outlet_temp_delta_60m",
    ]
    gradient_features = [
        "indoor_temp_gradient",
        "outlet_temp_gradient",
    ]
    binary_features = [
        "dhw_heating",
        "defrosting",
        "dhw_disinfection",
        "dhw_boost_heater",
    ]
    base_features = [
        "temp_diff_indoor_outdoor",
        "outlet_temp",
        "outdoor_temp",
        "pv_now",
        "tv_on",
        "hour_sin",
        "hour_cos",
        "outlet_temp_change_from_last",
        "outlet_indoor_diff",
        "outlet_temp_sq",
        "outlet_temp_cub",
        "month_sin",
        "month_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "is_weekend",
    ]
    return (
        base_features
        + history_features
        + forecast_features
        + lag_features
        + delta_features
        + gradient_features
        + binary_features
    )


def find_best_outlet_temp(
    model: Any,
    features: pd.DataFrame,
    target_temp: float,
    baseline_indoor: float,
    states_cache: Dict[str, Any],
) -> tuple[float, float]:
    # --- Get baseline from user's formula ---
    baseline_outlet = calculate_baseline_outlet_temp(
        states_cache
    )
    best_temp, min_diff = baseline_outlet, float("inf")

    # --- Define a smart search range around the baseline ---
    search_radius = 20.0
    min_temp = baseline_outlet - search_radius
    max_temp = baseline_outlet + search_radius
    step = 0.5

    # Clamp the search range to reasonable values
    min_temp = max(18.0, min_temp)
    max_temp = min(65.0, max_temp)

    if DEBUG:
        print("--- Finding Best Outlet Temp ---")
        print(f"Target indoor temp: {target_temp}°C")
        print(f"Baseline outlet from formula: {baseline_outlet:.1f}°C")
        print(f"Searching from {min_temp:.1f}°C to {max_temp:.1f}°C")

    # If the search range is invalid, just return the baseline
    if min_temp > max_temp:
        if DEBUG:
            print("Invalid search range, returning baseline.")
        return baseline_outlet, 0.0

    # --- Isolate features for the prediction loop ---
    last_outlet_temp = features["outlet_hist_5"].iloc[0]
    # The indoor_temp is not a feature anymore, but we need it for
    # calculations
    indoor_temp = baseline_indoor

    # --- Confidence Monitoring ---
    # Get predictions from all trees in the forest for a baseline temp
    X_baseline_dict = features.iloc[0].to_dict()
    X_baseline_dict["outlet_temp"] = baseline_outlet
    
    regressor = model._last_step
    tree_predictions = [
        tree.predict_one(X_baseline_dict)
        for tree in regressor.models
    ]
    confidence = np.std(tree_predictions)
    set_ha_state(
        "sensor.ml_model_confidence",
        confidence,
        {"state_class": "measurement"},
        round_digits=None,
    )

    # If confidence is low (high std dev), fall back to baseline
    if confidence > CONFIDENCE_THRESHOLD:
        if DEBUG:
            print(
                f"Model confidence low ({confidence:.3f}), "
                "falling back to baseline."
            )
        return baseline_outlet, confidence

    for temp_candidate in np.arange(min_temp, max_temp + step, step):
        predicted_indoor = 0.0
        
        # --- Direct ML model for both heating and cooling ---
        X_candidate_dict = features.iloc[0].to_dict()
        
        # Update features based on the temperature candidate
        X_candidate_dict["outlet_temp"] = temp_candidate
        X_candidate_dict["outlet_temp_sq"] = temp_candidate**2
        X_candidate_dict["outlet_temp_cub"] = temp_candidate**3
        X_candidate_dict["outlet_temp_change_from_last"] = (
            temp_candidate - last_outlet_temp
        )
        X_candidate_dict["outlet_indoor_diff"] = temp_candidate - indoor_temp

        try:
            # The model now directly predicts ΔT
            predicted_delta = model.predict_one(X_candidate_dict)
            predicted_indoor = baseline_indoor + predicted_delta
            if DEBUG:
                print(
                    f"  - Testing outlet {temp_candidate:.1f}°C -> "
                    f"Predicted ΔT: {predicted_delta:.3f}°C, "
                    f"Indoor: {predicted_indoor:.2f}°C"
                )
        except Exception:
            continue
        diff = abs(predicted_indoor - target_temp)
        if diff < min_diff:
            min_diff, best_temp = diff, float(temp_candidate)
        elif diff == min_diff:
            # If the difference is the same, prefer the temperature
            # closer to the baseline to promote stability.
            if abs(temp_candidate - baseline_outlet) < abs(
                best_temp - baseline_outlet
            ):
                best_temp = float(temp_candidate)
    
    if DEBUG:
        print(f"--- Optimal float temp found: {best_temp:.1f}°C ---")

    # --- Prediction Smoothing (Exponential Moving Average) ---
    if not prediction_history:
        prediction_history.append(best_temp)
    else:
        last_smoothed = prediction_history[-1]
        # Alpha can be tuned, 0.5 gives equal weight to new and old
        alpha = 0.5
        smoothed_temp = alpha * best_temp + (1 - alpha) * last_smoothed
        prediction_history.append(smoothed_temp)

    if DEBUG:
        print("--- Prediction Smoothing ---")
        print(f"  History: {[f'{t:.1f}' for t in prediction_history]}")
        print(f"  Smoothed Temp: {prediction_history[-1]:.1f}°C")
    best_temp = prediction_history[-1]

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
            # Use the ML model for heating/cooling
            X_candidate_dict = features.iloc[0].to_dict()
            X_candidate_dict["outlet_temp"] = temp_candidate
            X_candidate_dict["outlet_temp_change_from_last"] = (
                temp_candidate - last_outlet_temp
            )
            X_candidate_dict["outlet_indoor_diff"] = (
                temp_candidate - indoor_temp
            )

            try:
                predicted_delta = model.predict_one(X_candidate_dict)
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
                        f"  - Candidate {temp}°C -> "
                        f"Predicted: {indoor:.2f}°C "
                        f"(Diff: {abs(indoor - target_temp):.2f})"
                    )
                print(f"  -> Chose: {final_temp}°C")

    return float(final_temp), confidence


def get_feature_importances(model: Any, feature_names: List[str]) -> Dict[str, float]:
    """
    Get feature importances from a River model by traversing the trees.
    This is more robust for young models than relying on the internal
    `feature_importances_` attribute.
    """
    if not isinstance(model, compose.Pipeline):
        if DEBUG:
            print("DEBUG: Model is not a River pipeline, cannot get importances.")
        return {}

    regressor = model._last_step
    if not hasattr(regressor, "models"):
        if DEBUG:
            print("DEBUG: Regressor has no 'models' attribute.")
        return {}

    if DEBUG:
        print(f"DEBUG: Traversing {len(regressor.models)} trees for feature importances.")

    total_importances: Dict[str, int] = {}
    for i, tree_model in enumerate(regressor.models):
        if hasattr(tree_model, "regressor"):
            actual_tree = tree_model.regressor
            if hasattr(actual_tree, "_root"):
                def traverse(node):
                    if node is None:
                        return
                    if hasattr(node, "feature"):
                        feature = node.feature
                        total_importances[feature] = total_importances.get(feature, 0) + 1
                        if hasattr(node, "children"):
                            for child in node.children:
                                traverse(child)
                traverse(actual_tree._root)

    if not total_importances:
        if DEBUG:
            print("DEBUG: No feature splits were found in any tree.")
        return {}

    if DEBUG:
        print(f"DEBUG: Raw feature split counts: {total_importances}")

    # Normalize the counts to get a percentage-like importance score
    total_splits = sum(total_importances.values())
    if total_splits > 0:
        normalized_importances = {
            feature: count / total_splits
            for feature, count in total_importances.items()
        }
        return normalized_importances
    else:
        if DEBUG:
            print("DEBUG: Total feature splits is 0, cannot normalize.")
        return {}


def initial_train_model(model: Any, lookback_hours: int = 168) -> Any:
    """
    Train the model on the last `lookback_hours` of data to predict the
    change in indoor temperature `PREDICTION_HORIZON_STEPS` steps.
    """
    feature_names = get_feature_names()
    if model is None:
        unscaled_features = [
            "outlet_temp",
            "outlet_temp_sq",
            "outlet_temp_cub",
            "outlet_temp_change_from_last",
            "outlet_indoor_diff",
        ]
        
        # Create a pipeline that scales all features except the unscaled ones
        scaler = compose.Select(
            *[f for f in feature_names if f not in unscaled_features]
        ) | preprocessing.StandardScaler()

        model = compose.Pipeline(
            ("features", scaler),
            (
                "learn",
                ensemble.AdaptiveRandomForestRegressor(
                    n_models=10,
                    max_depth=10,
                    seed=42,
                    drift_detector=drift.PageHinkley(),
                    warning_detector=drift.ADWIN(),
                ),
            ),
        )
    hp_outlet_temp_id = ACTUAL_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1]
    kuche_temperatur_id = INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    fernseher_id = TV_STATUS_ENTITY_ID.split(".", 1)[-1]
    dhw_status_id = DHW_STATUS_ENTITY_ID.split(".", 1)[-1]
    defrost_status_id = DEFROST_STATUS_ENTITY_ID.split(".", 1)[-1]
    disinfection_status_id = DISINFECTION_STATUS_ENTITY_ID.split(".", 1)[-1]
    dhw_boost_heater_status_id = DHW_BOOST_HEATER_STATUS_ENTITY_ID.split(".", 1)[-1]
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
            r["entity_id"] == "{dhw_status_id}" or
            r["entity_id"] == "{defrost_status_id}" or
            r["entity_id"] == "{disinfection_status_id}" or
            r["entity_id"] == "{dhw_boost_heater_status_id}" or
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
        # If the database query fails, we can't train. Return None for the model.
        return model, mae, rmse
    features_list, labels_list = [], []
    outlet_steps = indoor_steps = HISTORY_STEPS
    start_idx = max(outlet_steps, indoor_steps, 12)  # 12 * 5m = 60m

    pv_forecast = get_pv_forecast()
    temp_forecast = get_hourly_forecast()

    for idx in range(start_idx, len(df) - PREDICTION_HORIZON_STEPS):
        row = df.iloc[idx]
        # --- Skip if DHW, defrosting, or disinfection is active ---
        if (
            row.get(dhw_status_id, 0.0) == 1.0
            or row.get(defrost_status_id, 0.0) == 1.0
            or row.get(disinfection_status_id, 0.0) == 1.0
            or row.get(dhw_boost_heater_status_id, 0.0) == 1.0
        ):
            continue
        next_row = df.iloc[idx + PREDICTION_HORIZON_STEPS]
        outdoor = row.get(outdoor_temp_id, np.nan)
        if pd.isna(outdoor):
            continue
        current_indoor_val = row.get(kuche_temperatur_id, np.nan)
        outlet_temp_val = row.get(hp_outlet_temp_id, np.nan)
        future_indoor_val = next_row.get(kuche_temperatur_id, np.nan)
        if (
            pd.isna(current_indoor_val)
            or pd.isna(future_indoor_val)
            or pd.isna(outlet_temp_val)
        ):
            continue
        pv_now = sum(
            row.get(k, 0.0) for k in [
                pv1_power_id, pv2_power_id, pv3_power_id
            ]
        )
        tv_on = 1.0 if row.get(fernseher_id, 0.0) == 1 else 0.0

        # --- New features ---
        temp_diff_indoor_outdoor = current_indoor_val - outdoor
        outlet_indoor_diff = outlet_temp_val - current_indoor_val
        time = row["_time"]
        hour = time.hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        month = time.month
        month_sin = np.sin(2 * np.pi * (month - 1) / 12)
        month_cos = np.cos(2 * np.pi * (month - 1) / 12)
        day_of_week = time.weekday()
        day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)
        is_weekend = 1.0 if day_of_week >= 5 else 0.0
        # This is a simplification for training; in production, we'd use the live target.
        # This is a simplification for training. In production, we'd use the
        # This is a simplification for training. In production, we'd use
        # the live target. For historical training, we assume the target
        # was the setpoint at that time. A more advanced version could
        # store historical targets. For now, we'll use a placeholder or a
        # reasonable assumption. Let's assume the target was slightly
        # above the actual, e.g., 21.5
        # For historical training, we don't know the real target temp.
        # Assuming it was the same as the actual temp makes the error 0,
        # preventing the model from learning from faulty error data. The
        # real error will be used during live prediction.
        
        # Get the previous outlet temperature to calculate the change
        previous_outlet_temp_val = df.iloc[idx - 1].get(
            hp_outlet_temp_id, outlet_temp_val
        )
        outlet_temp_change_from_last = (
            outlet_temp_val - previous_outlet_temp_val
        )
        outlet_temp_sq = outlet_temp_val**2
        outlet_temp_cub = outlet_temp_val**3

        # Lag features (10m, 30m, 60m)
        indoor_temp_lag_10m = df.iloc[idx - 2].get(
            kuche_temperatur_id, current_indoor_val
        )
        indoor_temp_lag_30m = df.iloc[idx - 6].get(
            kuche_temperatur_id, current_indoor_val
        )
        indoor_temp_lag_60m = df.iloc[idx - 12].get(
            kuche_temperatur_id, current_indoor_val
        )
        outlet_temp_lag_10m = df.iloc[idx - 2].get(
            hp_outlet_temp_id, outlet_temp_val
        )
        outlet_temp_lag_30m = df.iloc[idx - 6].get(
            hp_outlet_temp_id, outlet_temp_val
        )
        outlet_temp_lag_60m = df.iloc[idx - 12].get(
            hp_outlet_temp_id, outlet_temp_val
        )

        # Delta features
        indoor_temp_delta_10m = current_indoor_val - indoor_temp_lag_10m
        indoor_temp_delta_30m = current_indoor_val - indoor_temp_lag_30m
        indoor_temp_delta_60m = current_indoor_val - indoor_temp_lag_60m
        outlet_temp_delta_10m = outlet_temp_val - outlet_temp_lag_10m
        outlet_temp_delta_30m = outlet_temp_val - outlet_temp_lag_30m
        outlet_temp_delta_60m = outlet_temp_val - outlet_temp_lag_60m

        # Gradient features
        indoor_temp_gradient = (current_indoor_val - indoor_temp_lag_60m) / 60
        outlet_temp_gradient = (outlet_temp_val - outlet_temp_lag_60m) / 60

        # Binary features
        dhw_heating = 1.0 if row.get(dhw_status_id, 0.0) == 1.0 else 0.0
        defrosting = 1.0 if row.get(defrost_status_id, 0.0) == 1.0 else 0.0
        dhw_disinfection = (
            1.0 if row.get(disinfection_status_id, 0.0) == 1.0 else 0.0
        )
        dhw_boost_heater = (
            1.0 if row.get(dhw_boost_heater_status_id, 0.0) == 1.0 else 0.0
        )

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

        # Forecasts are fetched outside the loop now

        features_values = (
            [
                temp_diff_indoor_outdoor,
                outlet_temp_val,
                outdoor,
                pv_now,
                tv_on,
                hour_sin,
                hour_cos,
                outlet_temp_change_from_last,
                outlet_indoor_diff,
                outlet_temp_sq,
                outlet_temp_cub,
                month_sin,
                month_cos,
                day_of_week_sin,
                day_of_week_cos,
                is_weekend,
            ]
            + outlet_hist
            + indoor_hist
            + pv_forecast
            + temp_forecast
            + [
                indoor_temp_lag_10m,
                indoor_temp_lag_30m,
                indoor_temp_lag_60m,
                outlet_temp_lag_10m,
                outlet_temp_lag_30m,
                outlet_temp_lag_60m,
            ]
            + [
                indoor_temp_delta_10m,
                indoor_temp_delta_30m,
                indoor_temp_delta_60m,
                outlet_temp_delta_10m,
                outlet_temp_delta_30m,
                outlet_temp_delta_60m,
            ]
            + [
                indoor_temp_gradient,
                outlet_temp_gradient,
            ]
            + [
                dhw_heating,
                defrosting,
                dhw_disinfection,
                dhw_boost_heater,
            ]
        )

        # Ensure feature vector length matches feature names
        if len(features_values) != len(feature_names):
            if DEBUG:
                print(
                    "Feature length mismatch: "
                    f"expected {len(feature_names)}, "
                    f"got {len(features_values)}. Skipping sample."
                )
            continue

        # --- Label ---
        # The model now directly predicts the change in temperature (ΔT).
        actual_delta = float(future_indoor_val) - float(current_indoor_val)

        features_list.append(features_values)
        labels_list.append(actual_delta)

    if features_list and labels_list:
        try:
            X = pd.DataFrame(features_list, columns=feature_names)
            y = np.array(labels_list)

            for i in range(len(X)):
                features = X.iloc[i].to_dict()
                label = y[i]
                model.learn_one(features, label)

            # After training, calculate metrics
            for i in range(len(X)):
                features = X.iloc[i].to_dict()
                label = y[i]
                prediction = model.predict_one(features)
                mae.update(y_true=label, y_pred=prediction)
                rmse.update(y_true=label, y_pred=prediction)

            with open(MODEL_FILE, "wb") as f:
                pickle.dump({'model': model, 'mae': mae, 'rmse': rmse}, f)

            if DEBUG:
                print(f"MAE after training: {mae.get():.4f}")
                print(f"RMSE after training: {rmse.get():.4f}")

            print(
                "Initial training done on last "
                f"{lookback_hours}h of data "
                f"({len(features_list)} samples)."
            )
        except Exception:
            # If training fails, the model will remain None.
            # We'll return it at the end.
            pass

    # This single return statement handles all cases.
    # It returns the trained model if successful, or None if any step failed.
    return model, mae, rmse


def main(
    model: Optional[Any],
    mae: metrics.MAE,
    rmse: metrics.RMSE,
    poll_interval_seconds: int = 300,
) -> None:
    # If initial training failed, models will be None.
    # The loop will handle this by falling back to baseline.
    while True:
        try:
            confidence = 0.0
            # --- Batch fetch all states from Home Assistant ---
            all_states = get_all_ha_states()
            if not all_states:
                print("Could not fetch states from HA, skipping cycle.")
                time.sleep(poll_interval_seconds)
                continue

            dhw_state_data = get_ha_state(
                DHW_STATUS_ENTITY_ID, all_states, is_binary=True
            )
            if dhw_state_data and dhw_state_data.get("state") == "on":
                print("DHW active, skipping cycle.")
                time.sleep(poll_interval_seconds)
                continue

            defrost_state_data = get_ha_state(
                DEFROST_STATUS_ENTITY_ID, all_states, is_binary=True
            )
            if defrost_state_data and defrost_state_data.get("state") == "on":
                print("Defrosting active, skipping cycle.")
                time.sleep(poll_interval_seconds)
                continue

            disinfection_state_data = get_ha_state(
                DISINFECTION_STATUS_ENTITY_ID, all_states, is_binary=True
            )
            if disinfection_state_data and disinfection_state_data.get("state") == "on":
                print("DHW disinfection active, skipping cycle.")
                time.sleep(poll_interval_seconds)
                continue

            dhw_boost_heater_state_data = get_ha_state(
                DHW_BOOST_HEATER_STATUS_ENTITY_ID, all_states, is_binary=True
            )
            if (
                dhw_boost_heater_state_data
                and dhw_boost_heater_state_data.get("state") == "on"
            ):
                print("DHW boost heater active, skipping cycle.")
                time.sleep(poll_interval_seconds)
                continue

            # --- Get critical sensor states with validation ---
            target_indoor_temp = get_ha_state(
                TARGET_INDOOR_TEMP_ENTITY_ID, all_states
            )
            actual_indoor = get_ha_state(INDOOR_TEMP_ENTITY_ID, all_states)
            outdoor_temp = get_ha_state(OUTDOOR_TEMP_ENTITY_ID, all_states)

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
                    f"Invalid sensor data for: "
                    f"{', '.join(invalid_sensors)}. Skipping cycle."
                )
                time.sleep(poll_interval_seconds)
                continue

            now_utc = datetime.now(timezone.utc)
            error_target_vs_actual = target_indoor_temp - actual_indoor

            # --- Online training ---
            last_state = load_state(STATE_FILE)
            if last_state:
                print("Last state found, performing online training.")
                try:
                    last_features = last_state["features"]
                    last_timestamp = pd.to_datetime(last_state["timestamp"])
                    last_baseline_indoor = last_state["baseline_indoor"]

                    # Calculate the time difference in minutes
                    time_diff_minutes = (now_utc - last_timestamp).total_seconds() / 60

                    # Check if the time difference is close to the prediction horizon
                    if abs(time_diff_minutes - PREDICTION_HORIZON_MINUTES) < 10:
                        # The actual change is the current indoor temp minus
                        # the baseline from the past
                        actual_delta = actual_indoor - last_baseline_indoor

                        # Get the prediction that was made for this moment
                        predicted_delta = model.predict_one(last_features)

                        # Update the MAE metric
                        mae.update(y_true=actual_delta, y_pred=predicted_delta)
                        rmse.update(y_true=actual_delta, y_pred=predicted_delta)

                        # learn from the last state
                        model.learn_one(last_features, actual_delta)

                        with open(MODEL_FILE, "wb") as f:
                            pickle.dump({"model": model, "mae": mae, "rmse": rmse}, f)

                except Exception as e:
                    print(f"Error during online training: {e}")
            # --- End of online training ---

            # --- ML-based prediction path (with fallback) ---
            # Check if models exist and are fitted
            if model is None:
                is_fitted = False
                confidence = 0.0
            else:
                is_fitted = True

            # Fallback to baseline if models are not fitted
            if not is_fitted:
                print("Model not fitted, falling back to baseline.")
                suggested_temp = calculate_baseline_outlet_temp(
                    all_states
                )
                predicted_indoor = actual_indoor or target_indoor_temp
            else:
                # --- Log feature importances ---
                if DEBUG and isinstance(model, compose.Pipeline):
                    feature_importances = get_feature_importances(
                        model, get_feature_names()
                    )
                    if feature_importances:
                        print("--- Feature Importances ---")
                        sorted_importances = sorted(
                            feature_importances.items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )
                        # Prepare attributes for Home Assistant
                        importance_attributes = {
                            "state_class": "measurement",
                            "unit_of_measurement": "%",
                            "top_features": {
                                f: round(v * 100, 2) for f, v in sorted_importances[:10]
                            },
                            "last_updated": now_utc.isoformat(),
                        }
                        # Set a dummy state, the real data is in the attributes
                        set_ha_state(
                            "sensor.ml_feature_importance",
                            len(sorted_importances),
                            importance_attributes,
                            round_digits=None,
                        )

                        for feature, importance in sorted_importances[:10]:
                            print(f"  - {feature}: {importance:.4f}")

                # --- Gather all features for ML model ---
                tv_state_raw = get_ha_state(
                    TV_STATUS_ENTITY_ID, all_states, is_binary=True
                )
                is_tv_on = tv_state_raw and tv_state_raw.get("state") == "on"
                tv_on = 1.0 if is_tv_on else 0.0

                pv1 = get_ha_state(PV1_POWER_ENTITY_ID, all_states)
                pv2 = get_ha_state(PV2_POWER_ENTITY_ID, all_states)
                pv3 = get_ha_state(PV3_POWER_ENTITY_ID, all_states)
                pv_now = sum(v for v in (pv1, pv2, pv3) if v is not None)

                pv_forecasts = get_pv_forecast()
                temp_forecasts = get_hourly_forecast()
                outlet_history = fetch_outlet_history(HISTORY_STEPS)
                indoor_history = fetch_indoor_history(HISTORY_STEPS)

                if actual_indoor is not None:
                    indoor_history.pop(0)
                    indoor_history.append(float(actual_indoor))

                temp_diff_indoor_outdoor = actual_indoor - outdoor_temp
                hour = now_utc.hour
                hour_sin = np.sin(2 * np.pi * hour / 24)
                hour_cos = np.cos(2 * np.pi * hour / 24)
                month = now_utc.month
                month_sin = np.sin(2 * np.pi * (month - 1) / 12)
                month_cos = np.cos(2 * np.pi * (month - 1) / 12)
                day_of_week = now_utc.weekday()
                day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
                day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)
                is_weekend = 1.0 if day_of_week >= 5 else 0.0

                feature_names = get_feature_names()
                outlet_temp = get_ha_state(
                    ACTUAL_OUTLET_TEMP_ENTITY_ID, all_states
                ) or outlet_history[-1]
                
                outlet_temp_change_from_last = outlet_temp - outlet_history[-1]

                # Lag features
                indoor_temp_lag_10m = indoor_history[-2]
                indoor_temp_lag_30m = indoor_history[-4]
                indoor_temp_lag_60m = indoor_history[0]
                outlet_temp_lag_10m = outlet_history[-2]
                outlet_temp_lag_30m = outlet_history[-4]
                outlet_temp_lag_60m = outlet_history[0]

                # Delta features
                indoor_temp_delta_10m = actual_indoor - indoor_temp_lag_10m
                indoor_temp_delta_30m = actual_indoor - indoor_temp_lag_30m
                indoor_temp_delta_60m = actual_indoor - indoor_temp_lag_60m
                outlet_temp_delta_10m = outlet_temp - outlet_temp_lag_10m
                outlet_temp_delta_30m = outlet_temp - outlet_temp_lag_30m
                outlet_temp_delta_60m = outlet_temp - outlet_temp_lag_60m

                # Gradient features
                indoor_temp_gradient = (actual_indoor - indoor_temp_lag_60m) / 60
                outlet_temp_gradient = (outlet_temp - outlet_temp_lag_60m) / 60

                # Binary features
                dhw_heating_state = get_ha_state(
                    DHW_STATUS_ENTITY_ID, all_states, is_binary=True
                )
                dhw_heating = 1.0 if dhw_heating_state and dhw_heating_state.get("state") == "on" else 0.0

                defrosting_state = get_ha_state(
                    DEFROST_STATUS_ENTITY_ID, all_states, is_binary=True
                )
                defrosting = 1.0 if defrosting_state and defrosting_state.get("state") == "on" else 0.0

                dhw_disinfection_state = get_ha_state(
                    DISINFECTION_STATUS_ENTITY_ID, all_states, is_binary=True
                )
                dhw_disinfection = 1.0 if dhw_disinfection_state and dhw_disinfection_state.get("state") == "on" else 0.0

                dhw_boost_heater_state = get_ha_state(
                    DHW_BOOST_HEATER_STATUS_ENTITY_ID, all_states, is_binary=True
                )
                dhw_boost_heater = 1.0 if dhw_boost_heater_state and dhw_boost_heater_state.get("state") == "on" else 0.0


                current_features_list = (
                    [
                        temp_diff_indoor_outdoor,
                        outlet_temp,
                        float(outdoor_temp),
                        pv_now,
                        tv_on,
                        hour_sin,
                        hour_cos,
                        outlet_temp_change_from_last,
                        outlet_temp - actual_indoor,
                        outlet_temp**2,
                        outlet_temp**3,
                        month_sin,
                        month_cos,
                        day_of_week_sin,
                        day_of_week_cos,
                        is_weekend,
                    ]
                    + outlet_history
                    + indoor_history
                    + pv_forecasts
                    + temp_forecasts
                    + [
                        indoor_temp_lag_10m,
                        indoor_temp_lag_30m,
                        indoor_temp_lag_60m,
                        outlet_temp_lag_10m,
                        outlet_temp_lag_30m,
                        outlet_temp_lag_60m,
                    ]
                    + [
                        indoor_temp_delta_10m,
                        indoor_temp_delta_30m,
                        indoor_temp_delta_60m,
                        outlet_temp_delta_10m,
                        outlet_temp_delta_30m,
                        outlet_temp_delta_60m,
                    ]
                    + [
                        indoor_temp_gradient,
                        outlet_temp_gradient,
                    ]
                    + [
                        dhw_heating,
                        defrosting,
                        dhw_disinfection,
                        dhw_boost_heater,
                    ]
                )

                # Fallback if feature lengths don't match
                if len(current_features_list) != len(feature_names):
                    print("Feature length mismatch, " "falling back to baseline.")
                    suggested_temp = calculate_baseline_outlet_temp(all_states)
                    predicted_indoor = actual_indoor or target_indoor_temp
                else:
                    # --- This is the successful ML path ---
                    current_features_df = pd.DataFrame(
                        [current_features_list], columns=feature_names
                    )
                    baseline_current = actual_indoor or indoor_history[-1]

                    suggested_temp, confidence = find_best_outlet_temp(
                        model,
                        current_features_df,
                        target_indoor_temp,
                        baseline_current,
                        all_states,
                    )

                    # Predict final indoor temp for logging
                    final_features = current_features_df.iloc[0].to_dict()
                    final_features["outlet_temp"] = suggested_temp
                    final_features["outlet_temp_sq"] = suggested_temp**2
                    final_features["outlet_temp_cub"] = suggested_temp**3
                    final_features["outlet_temp_change_from_last"] = (
                        suggested_temp - outlet_history[-1]
                    )
                    final_features["outlet_indoor_diff"] = (
                        suggested_temp - baseline_current
                    )

                    predicted_delta = model.predict_one(final_features)
                    predicted_indoor = baseline_current + predicted_delta

                    # Save state only after a successful ML prediction
                    state_to_save = {
                        "features": final_features,
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
            final_temp = max(final_temp, 18.0)

            if DEBUG:
                print("--- Final Temp Calculation ---")
                print(f"  Model Suggested: {suggested_temp:.1f}°C")
                print(f"  Dynamic Boost Applied: {error_boost:.2f}°C")
                print(f"  Boosted Temp: {boosted_temp:.1f}°C")
                print(f"  Final Temp (clamped at 65°C): {final_temp:.1f}°C")

            temp_attributes = {
                "state_class": "measurement",
                "unit_of_measurement": "°C",
                "device_class": "temperature",
            }
            set_ha_state(
                "sensor.ml_vorlauftemperatur", final_temp, temp_attributes
            )
            set_ha_state(
                "sensor.ml_predicted_indoor_temp",
                predicted_indoor,
                temp_attributes,
            )
            set_ha_state(
                "sensor.ml_model_mae",
                mae.get(),
                {"state_class": "measurement"},
                round_digits=None,
            )
            set_ha_state(
                "sensor.ml_model_rmse",
                rmse.get(),
                {"state_class": "measurement"},
                round_digits=None,
            )

            print(
                f"{now_utc.isoformat()} - Target: {target_indoor_temp}°C"
                f" | Suggested: {final_temp:.1f}°C"
                f" | Predicted: {predicted_indoor:.2f}°C"
                f" | Actual: {actual_indoor or 'N/A'}"
                f" | Confidence: {confidence:.3f}"
                f" | MAE: {mae.get():.3f}"
                f" | RMSE: {rmse.get():.3f}"
            )

        except Exception as exc:
            print("Error in main loop:", exc)

        time.sleep(poll_interval_seconds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ML Heating Controller")
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Run the initial training and then exit.",
    )
    args = parser.parse_args()

    # The models are now created inside the training function
    model, mae, rmse = initial_train_model(model, lookback_hours=168)

    if args.train_only:
        print("Training complete. Exiting as requested by --train-only flag.")
    else:
        # Pass the potentially newly created models to main
        main(model=model, mae=mae, rmse=rmse)
