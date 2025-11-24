"""
This module is responsible for creating the feature set used by the ML model.

Feature engineering is a critical step in machine learning. This module
gathers raw data from various sources (Home Assistant, InfluxDB), calculates
new features, and assembles them into a structured format (a pandas DataFrame)
that the model can consume for both training and prediction.
"""
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Support both package-relative and direct import for notebooks
try:
    from . import config  # Package-relative import
    from .ha_client import HAClient
except ImportError:
    import config  # Direct import fallback for notebooks
    from ha_client import HAClient
from .influx_service import InfluxService


def get_feature_names() -> List[str]:
    """
    Returns a definitive list of all feature names used by the model.

    This function is crucial for ensuring consistency between the features
    created during training and prediction. It centralizes the feature
    definitions, making the system easier to maintain and debug.
    """
    # Features related to future forecasts (PV power and temperature).
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
        "fireplace_on",
        "defrost_recent",
        "defrost_count",
        "defrost_age_min",
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
    new_aggregated_features = [
        "outlet_hist_mean",
        "outlet_hist_std",
        "outlet_hist_trend",
        "outlet_hist_min",
        "outlet_hist_max",
        "outlet_hist_q25",
        "outlet_hist_q75",
        "indoor_hist_mean",
        "indoor_hist_std",
        "indoor_hist_trend",
        "indoor_hist_min",
        "indoor_hist_max",
        "indoor_hist_q25",
        "indoor_hist_q75",
        "outdoor_temp_x_outlet_temp",
    ]
    return (
        base_features
        + forecast_features
        + lag_features
        + delta_features
        + gradient_features
        + binary_features
        + new_aggregated_features
    )


def build_features(
    ha_client: HAClient,
    influx_service: InfluxService,
    all_states: Dict[str, Any],
    target_indoor_temp: float,
) -> Optional[Tuple[pd.DataFrame, list[float]]]:
    """
    Constructs the feature DataFrame for a real-time prediction.

    This function is called during each main loop cycle. It fetches live data,
    historical data, and forecasts to build a single-row DataFrame
    representing the current state of the environment.

    Args:
        ha_client: The Home Assistant client instance.
        influx_service: The InfluxDB service instance.
        all_states: A dictionary of all current states from Home Assistant.

    Returns:
        A tuple containing the feature DataFrame and the outlet history,
        or (None, None) if feature building fails.
    """
    now = datetime.now(timezone.utc)

    # --- 1. Gather Raw Data ---
    # Fetch current sensor values from Home Assistant's state cache.
    actual_indoor = ha_client.get_state(config.INDOOR_TEMP_ENTITY_ID, all_states)
    outdoor_temp = ha_client.get_state(config.OUTDOOR_TEMP_ENTITY_ID, all_states)
    outlet_temp = ha_client.get_state(config.ACTUAL_OUTLET_TEMP_ENTITY_ID, all_states)
    tv_state_raw = ha_client.get_state(
        config.TV_STATUS_ENTITY_ID, all_states, is_binary=True
    )
    tv_on = 1.0 if tv_state_raw else 0.0

    pv1 = ha_client.get_state(config.PV1_POWER_ENTITY_ID, all_states) or 0.0
    pv2 = ha_client.get_state(config.PV2_POWER_ENTITY_ID, all_states) or 0.0
    pv3 = (
        ha_client.get_state(config.PV3_POWER_ENTITY_ID, all_states) or 0.0
    )
    pv_now = pv1 + pv2 + pv3

    # --- Get historical and forecast data ---
    # Use Home Assistant PV forecast sensor (15-min 'watts' attribute) exclusively.
    # Build 4 hourly anchors starting at the next full hour (UTC) and compute
    # the mean watts within each hour. If an hour has no samples, assume 0.0.
    pv_entity = config.PV_FORECAST_ENTITY_ID
    pv_forecasts = [0.0, 0.0, 0.0, 0.0]

    entity_state = all_states.get(pv_entity) if all_states else None
    attrs = entity_state.get("attributes", {}) if isinstance(entity_state, dict) else {}

    now_utc = pd.Timestamp(datetime.now(timezone.utc))
    first_anchor = now_utc.ceil("h")
    anchors = pd.date_range(start=first_anchor, periods=4, freq="h", tz="UTC")

    watts_map = attrs.get("watts") or {}
    if isinstance(watts_map, dict) and watts_map:
        try:
            s = pd.Series(
                list(watts_map.values()),
                index=pd.to_datetime(list(watts_map.keys()), utc=True),
            ).sort_index()
            hourly = []
            for a in anchors:
                start = a
                end = a + pd.Timedelta("1h")
                slice_vals = s[(s.index >= start) & (s.index < end)]
                hourly.append(float(slice_vals.mean()) if not slice_vals.empty else 0.0)
            pv_forecasts = hourly
        except Exception:
            pv_forecasts = [0.0, 0.0, 0.0, 0.0]
    temp_forecasts = ha_client.get_hourly_forecast()
    outlet_history = influx_service.fetch_outlet_history(config.HISTORY_STEPS)
    indoor_history = influx_service.fetch_indoor_history(config.HISTORY_STEPS)

    if None in [actual_indoor, outdoor_temp, outlet_temp, target_indoor_temp]:
        logging.error("Missing critical sensor data. Cannot build features.")
        return None, None

    # --- 2. Engineer Features ---
    # Create new features from the raw data to provide more context for the model.

    # Temperature difference between inside and outside.
    temp_diff_indoor_outdoor = actual_indoor - outdoor_temp

    # Cyclical time-based features to capture daily, monthly, and weekly patterns.
    hour = now.hour
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month = now.month
    month_sin = np.sin(2 * np.pi * (month - 1) / 12)
    month_cos = np.cos(2 * np.pi * (month - 1) / 12)
    day_of_week = now.weekday()
    day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)
    is_weekend = 1.0 if day_of_week >= 5 else 0.0

    if not outlet_history:
        logging.error("Outlet history is empty, cannot build features.")
        return None, None
    outlet_temp_change_from_last = outlet_temp - outlet_history[-1]

    # --- 3. Advanced Feature Engineering ---

    # Statistical features from historical data to capture trends,
    # variability, and momentum.
    outlet_hist_series = pd.Series(outlet_history)
    indoor_hist_series = pd.Series(indoor_history)

    outlet_hist_mean = outlet_hist_series.mean()
    outlet_hist_std = outlet_hist_series.std()
    outlet_hist_trend = (
        (outlet_hist_series.iloc[-1] - outlet_hist_series.iloc[0])
        / len(outlet_hist_series)
        if len(outlet_hist_series) > 1
        else 0
    )
    outlet_hist_min = outlet_hist_series.min()
    outlet_hist_max = outlet_hist_series.max()
    outlet_hist_q25 = outlet_hist_series.quantile(0.25)
    outlet_hist_q75 = outlet_hist_series.quantile(0.75)

    indoor_hist_mean = indoor_hist_series.mean()
    indoor_hist_std = indoor_hist_series.std()
    indoor_hist_trend = (
        (indoor_hist_series.iloc[-1] - indoor_hist_series.iloc[0])
        / len(indoor_hist_series)
        if len(indoor_hist_series) > 1
        else 0
    )
    indoor_hist_min = indoor_hist_series.min()
    indoor_hist_max = indoor_hist_series.max()
    indoor_hist_q25 = indoor_hist_series.quantile(0.25)
    indoor_hist_q75 = indoor_hist_series.quantile(0.75)

    # Interaction features capture combined effects.
    outdoor_temp_x_outlet_temp = outdoor_temp * outlet_temp

    # Lag features provide a snapshot of past states.
    # Delta features show the change over different time windows.
    # Gradient features calculate the rate of change.
    indoor_temp_lag_10m = indoor_history[-1]
    indoor_temp_lag_30m = indoor_history[-3]
    indoor_temp_lag_60m = indoor_history[0]
    outlet_temp_lag_10m = outlet_history[-1]
    outlet_temp_lag_30m = outlet_history[-3]
    outlet_temp_lag_60m = outlet_history[0]

    indoor_temp_delta_10m = actual_indoor - indoor_temp_lag_10m
    indoor_temp_delta_30m = actual_indoor - indoor_temp_lag_30m
    indoor_temp_delta_60m = actual_indoor - indoor_temp_lag_60m
    outlet_temp_delta_10m = outlet_temp - outlet_temp_lag_10m
    outlet_temp_delta_30m = outlet_temp - outlet_temp_lag_30m
    outlet_temp_delta_60m = outlet_temp - outlet_temp_lag_60m

    indoor_temp_gradient = (
        actual_indoor - indoor_history[0]
    ) / (config.HISTORY_STEPS * config.HISTORY_STEP_MINUTES)
    outlet_temp_gradient = (
        outlet_temp - outlet_history[0]
    ) / (config.HISTORY_STEPS * config.HISTORY_STEP_MINUTES)

    # Binary flags for system states (e.g., defrosting).
    # For binary sensors we fetch a history using max-aggregation so short
    # pulses remain 1.0 after aggregation, then use the most recent value.
    binary_entities = {
        "dhw_heating": config.DHW_STATUS_ENTITY_ID,
        "defrosting": config.DEFROST_STATUS_ENTITY_ID,
        "dhw_disinfection": config.DISINFECTION_STATUS_ENTITY_ID,
        "dhw_boost_heater": config.DHW_BOOST_HEATER_STATUS_ENTITY_ID,
        "fireplace_on": config.FIREPLACE_STATUS_ENTITY_ID,
    }
    binary_features_values = {}
    for name, entity_id in binary_entities.items():
        try:
            hist = influx_service.fetch_binary_history(
                entity_id, config.HISTORY_STEPS
            )
            recent_val = hist[-1] if hist else 0.0
            binary_features_values[name] = 1.0 if recent_val and recent_val > 0.1 else 0.0
        except Exception:
            # Fallback to live HA snapshot if Influx query fails.
            state_data = ha_client.get_state(entity_id, all_states, is_binary=True)
            binary_features_values[name] = 1.0 if state_data else 0.0

    # --- Defrost history-derived features ---
    # Short defrost events may be missed in the live snapshot; query Influx history
    # for the defrost entity using the same history window used elsewhere.
    try:
        # Use max-aggregation for binary signals so short pulses remain 1.0
        defrost_history = influx_service.fetch_binary_history(
            config.DEFROST_STATUS_ENTITY_ID, config.HISTORY_STEPS
        )
    except Exception:
        defrost_history = [0.0] * config.HISTORY_STEPS

    defrost_count = int(sum(1 for v in defrost_history if v and v > 0.1))
    defrost_recent = 1.0 if defrost_count > 0 else 0.0

    # Minutes since last defrost sample in the history window. If none, use a large sentinel.
    defrost_age_min = (
        config.HISTORY_STEPS * config.HISTORY_STEP_MINUTES
    )
    for i in range(len(defrost_history) - 1, -1, -1):
        if defrost_history[i] and defrost_history[i] > 0.1:
            defrost_age_min = (len(defrost_history) - 1 - i) * config.HISTORY_STEP_MINUTES
            break

    logging.debug(
        "Defrost history for %s: %s -> defrost_count=%d defrost_recent=%s defrost_age_min=%s",
        config.DEFROST_STATUS_ENTITY_ID,
        defrost_history,
        defrost_count,
        defrost_recent,
        defrost_age_min,
    )

    binary_features_values["defrost_recent"] = defrost_recent
    binary_features_values["defrost_count"] = float(defrost_count)
    binary_features_values["defrost_age_min"] = float(defrost_age_min)

    # --- 4. Assemble the DataFrame ---
    # Combine all engineered features into a list in the correct order.
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
        + list(binary_features_values.values())
        + [
            outlet_hist_mean,
            outlet_hist_std,
            outlet_hist_trend,
            outlet_hist_min,
            outlet_hist_max,
            outlet_hist_q25,
            outlet_hist_q75,
            indoor_hist_mean,
            indoor_hist_std,
            indoor_hist_trend,
            indoor_hist_min,
            indoor_hist_max,
            indoor_hist_q25,
            indoor_hist_q75,
            outdoor_temp_x_outlet_temp,
        ]
    )

    feature_names = get_feature_names()
    if len(current_features_list) != len(feature_names):
        logging.error(
            "Feature length mismatch: expected %d, got %d. "
            "Returning empty DataFrame.",
            len(feature_names),
            len(current_features_list),
        )
        return None, None

    # Create the final DataFrame with named columns.
    return pd.DataFrame([current_features_list], columns=feature_names), outlet_history


def build_features_for_training(
    df: pd.DataFrame,
    idx: int,
) -> Optional[Dict[str, Any]]:
    """
    Builds a feature dictionary for a single time step from a historical DataFrame.

    This function is used during the initial training process. It iterates
    through a large historical dataset and constructs a feature set for each
    row, similar to `build_features` but operating on historical data rather
    than live data.

    Args:
        df: The historical data DataFrame from InfluxDB.
        idx: The index of the current row to process.

    Returns:
        A dictionary of features for the given time step, or None if not possible.
    """
    if idx < config.HISTORY_STEPS:
        # Need enough historical data to build lag/delta features.
        return None

    row = df.iloc[idx]
    now = pd.to_datetime(df.index[idx])

    # --- Get entity IDs ---
    actual_indoor_id = config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    outdoor_temp_id = config.OUTDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    outlet_temp_id = config.ACTUAL_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1]
    tv_id = config.TV_STATUS_ENTITY_ID.split(".", 1)[-1]
    pv1_id = config.PV1_POWER_ENTITY_ID.split(".", 1)[-1]
    pv2_id = config.PV2_POWER_ENTITY_ID.split(".", 1)[-1]
    pv3_id = config.PV3_POWER_ENTITY_ID.split(".", 1)[-1]
    defrost_status_id = config.DEFROST_STATUS_ENTITY_ID.split(".", 1)[-1]

    # --- Get current values from the row ---
    actual_indoor = row.get(actual_indoor_id)
    outdoor_temp = row.get(outdoor_temp_id)
    outlet_temp = row.get(outlet_temp_id)
    tv_on = 1.0 if row.get(tv_id, "off") == "on" else 0.0
    pv_now = row.get(pv1_id, 0.0) + row.get(pv2_id, 0.0) + row.get(pv3_id, 0.0)

    if pd.isna(actual_indoor) or pd.isna(outdoor_temp) or pd.isna(outlet_temp):
        return None

    # --- Time-based features ---
    hour = now.hour
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month = now.month
    month_sin = np.sin(2 * np.pi * (month - 1) / 12)
    month_cos = np.cos(2 * np.pi * (month - 1) / 12)
    day_of_week = now.weekday()
    day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)
    is_weekend = 1.0 if day_of_week >= 5 else 0.0

    # --- History and Lag Features ---
    history_slice = df.iloc[idx - config.HISTORY_STEPS:idx]
    outlet_history = history_slice[outlet_temp_id].tolist()
    indoor_history = history_slice[actual_indoor_id].tolist()

    if not outlet_history or not indoor_history:
        return None

    outlet_temp_change_from_last = outlet_temp - outlet_history[-1]

    # --- Aggregated History Features ---
    outlet_hist_series = pd.Series(outlet_history)
    indoor_hist_series = pd.Series(indoor_history)

    # --- Lag, Delta, and Gradient features ---
    indoor_temp_lag_10m = indoor_history[-1]
    indoor_temp_lag_30m = indoor_history[-3]
    indoor_temp_lag_60m = indoor_history[0]
    outlet_temp_lag_10m = outlet_history[-1]
    outlet_temp_lag_30m = outlet_history[-3]
    outlet_temp_lag_60m = outlet_history[0]

    indoor_temp_delta_10m = actual_indoor - indoor_temp_lag_10m
    indoor_temp_delta_30m = actual_indoor - indoor_temp_lag_30m
    indoor_temp_delta_60m = actual_indoor - indoor_temp_lag_60m
    outlet_temp_delta_10m = outlet_temp - outlet_temp_lag_10m
    outlet_temp_delta_30m = outlet_temp - outlet_temp_lag_30m
    outlet_temp_delta_60m = outlet_temp - outlet_temp_lag_60m

    indoor_temp_gradient = (
        actual_indoor - indoor_history[0]
    ) / (config.HISTORY_STEPS * config.HISTORY_STEP_MINUTES)
    outlet_temp_gradient = (
        outlet_temp - outlet_history[0]
    ) / (config.HISTORY_STEPS * config.HISTORY_STEP_MINUTES)

    # --- Binary features ---
    binary_entities = {
        "dhw_heating": config.DHW_STATUS_ENTITY_ID,
        "defrosting": config.DEFROST_STATUS_ENTITY_ID,
        "dhw_disinfection": config.DISINFECTION_STATUS_ENTITY_ID,
        "dhw_boost_heater": config.DHW_BOOST_HEATER_STATUS_ENTITY_ID,
        "fireplace_on": config.FIREPLACE_STATUS_ENTITY_ID,
    }
    binary_features_values = {}
    for name, entity_id in binary_entities.items():
        entity_short_id = entity_id.split(".", 1)[-1]
        binary_features_values[name] = (
            1.0 if row.get(entity_short_id) == "on" else 0.0
        )

    # Derive defrost history metrics from the history slice so training features
    # match the live feature builder.
    try:
        defrost_history = history_slice[defrost_status_id].tolist()
    except Exception:
        defrost_history = [0.0] * config.HISTORY_STEPS

    defrost_count = int(sum(1 for v in defrost_history if v and v > 0.1))
    defrost_recent = 1.0 if defrost_count > 0 else 0.0

    defrost_age_min = config.HISTORY_STEPS * config.HISTORY_STEP_MINUTES
    for i in range(len(defrost_history) - 1, -1, -1):
        if defrost_history[i] and defrost_history[i] > 0.1:
            defrost_age_min = (len(defrost_history) - 1 - i) * config.HISTORY_STEP_MINUTES
            break

    binary_features_values["defrost_recent"] = defrost_recent
    binary_features_values["defrost_count"] = float(defrost_count)
    binary_features_values["defrost_age_min"] = float(defrost_age_min)

    # --- Forecasts (assuming they are in the DataFrame) ---
    pv_forecasts = [
        row.get(f"pv_forecast_{i+1}h", 0.0) for i in range(4)
    ]
    temp_forecasts = [
        row.get(f"temp_forecast_{i+1}h", outdoor_temp) for i in range(4)
    ]

    # --- Assemble feature dictionary ---
    features = {
        "temp_diff_indoor_outdoor": actual_indoor - outdoor_temp,
        "outlet_temp": outlet_temp,
        "outdoor_temp": outdoor_temp,
        "pv_now": pv_now,
        "tv_on": tv_on,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "outlet_temp_change_from_last": outlet_temp_change_from_last,
        "outlet_indoor_diff": outlet_temp - actual_indoor,
        "outlet_temp_sq": outlet_temp**2,
        "outlet_temp_cub": outlet_temp**3,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "day_of_week_sin": day_of_week_sin,
        "day_of_week_cos": day_of_week_cos,
        "is_weekend": is_weekend,
        "outlet_hist_mean": outlet_hist_series.mean(),
        "outlet_hist_std": outlet_hist_series.std(),
        "outlet_hist_trend": (
            (outlet_hist_series.iloc[-1] - outlet_hist_series.iloc[0])
            / len(outlet_hist_series)
            if len(outlet_hist_series) > 1
            else 0
        ),
        "outlet_hist_min": outlet_hist_series.min(),
        "outlet_hist_max": outlet_hist_series.max(),
        "outlet_hist_q25": outlet_hist_series.quantile(0.25),
        "outlet_hist_q75": outlet_hist_series.quantile(0.75),
        "indoor_hist_mean": indoor_hist_series.mean(),
        "indoor_hist_std": indoor_hist_series.std(),
        "indoor_hist_trend": (
            (indoor_hist_series.iloc[-1] - indoor_hist_series.iloc[0])
            / len(indoor_hist_series)
            if len(indoor_hist_series) > 1
            else 0
        ),
        "indoor_hist_min": indoor_hist_series.min(),
        "indoor_hist_max": indoor_hist_series.max(),
        "indoor_hist_q25": indoor_hist_series.quantile(0.25),
        "indoor_hist_q75": indoor_hist_series.quantile(0.75),
        "outdoor_temp_x_outlet_temp": outdoor_temp * outlet_temp,
        "indoor_temp_lag_10m": indoor_temp_lag_10m,
        "indoor_temp_lag_30m": indoor_temp_lag_30m,
        "indoor_temp_lag_60m": indoor_temp_lag_60m,
        "outlet_temp_lag_10m": outlet_temp_lag_10m,
        "outlet_temp_lag_30m": outlet_temp_lag_30m,
        "outlet_temp_lag_60m": outlet_temp_lag_60m,
        "indoor_temp_delta_10m": indoor_temp_delta_10m,
        "indoor_temp_delta_30m": indoor_temp_delta_30m,
        "indoor_temp_delta_60m": indoor_temp_delta_60m,
        "outlet_temp_delta_10m": outlet_temp_delta_10m,
        "outlet_temp_delta_30m": outlet_temp_delta_30m,
        "outlet_temp_delta_60m": outlet_temp_delta_60m,
        "indoor_temp_gradient": indoor_temp_gradient,
        "outlet_temp_gradient": outlet_temp_gradient,
    }
    features.update(binary_features_values)
    for i, val in enumerate(pv_forecasts):
        features[f"pv_forecast_{i+1}h"] = val
    for i, val in enumerate(temp_forecasts):
        features[f"temp_forecast_{i+1}h"] = val

    # Ensure all feature names are present
    expected_features = get_feature_names()
    for f in expected_features:
        if f not in features:
            logging.warning(f"Feature '{f}' missing in training build.")
            features[f] = 0.0  # Add with a default value

    return {f: features[f] for f in expected_features}
