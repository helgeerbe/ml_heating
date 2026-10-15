"""
Centralized Configuration for the ML Heating Script.

This file consolidates all user-configurable parameters for the application.
It uses the `dotenv` library to load settings from a `.env` file, allowing for
easy management of sensitive information like API keys and environment-specific
settings without hardcoding them into the source.

The configuration is organized into logical sections:
- API Credentials and Endpoints
- File Paths for persistent data
- Model & History Parameters for feature engineering
- Home Assistant Entity IDs (Core, Blocking, and Additional Sensors)
- Tuning & Debug Parameters for runtime behavior
- Metrics Entity IDs for performance monitoring

It is crucial to create a `.env` file and customize these settings, especially
the `HASS_URL`, `HASS_TOKEN`, and all `*_ENTITY_ID` variables, to match your
specific Home Assistant setup.
"""
import os
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists.
load_dotenv()

# --- API Credentials and Endpoints ---
# This section contains the connection details for external services.


# Detect if running in Home Assistant addon environment
def _is_addon_environment():
    """Detect if running in Home Assistant addon environment."""
    return os.getenv("SUPERVISOR_TOKEN") is not None


# Detect if running in a notebook/analysis environment
def _is_notebook_environment():
    """Detect if running in a Jupyter notebook or analysis script."""
    # Check for common notebook indicators or explicit env var
    return (
        os.getenv("ML_HEATING_ENV") == "notebook"
        or "ipykernel" in str(os.environ.get("modules", ""))
    )


# For Home Assistant addon, uses internal supervisor API;
# for standalone, uses .env
if _is_addon_environment():
    HASS_URL: str = os.getenv("HASS_URL", "http://supervisor/core")
    HASS_TOKEN: str = os.getenv("SUPERVISOR_TOKEN", "").strip()
else:
    # Default to localhost, but allow override for notebooks
    default_url = "http://localhost:8123"
    if _is_notebook_environment():
        # In notebooks, we might want to fail gracefully or warn if not set
        pass

    HASS_URL = os.getenv("HASS_URL", default_url)
    HASS_TOKEN = os.getenv("HASS_TOKEN", "").strip()


HASS_HEADERS: dict[str, str] = {
    "Authorization": f"Bearer {HASS_TOKEN}",
    "Content-Type": "application/json",
}
INFLUX_URL: str = os.getenv("INFLUX_URL", "https://influxdb.erbehome.de")
INFLUX_TOKEN: str = os.getenv("INFLUX_TOKEN", "")
INFLUX_ORG: str = os.getenv("INFLUX_ORG", "erbehome")
INFLUX_BUCKET: str = os.getenv("INFLUX_BUCKET", "home_assistant/autogen")

INFLUX_FEATURES_BUCKET: str = os.getenv(
    "INFLUX_FEATURES_BUCKET", "ml_heating_features"
)

# --- File Paths ---
UNIFIED_STATE_FILE: str = os.getenv(
    "UNIFIED_STATE_FILE", "/opt/ml_heating/unified_thermal_state.json"
)
CALIBRATION_BASELINE_FILE: str = os.getenv(
    "CALIBRATION_BASELINE_FILE", "/opt/ml_heating/calibrated_baseline.json"
)

# --- Model & History Parameters ---
# These parameters control time windows for feature creation and prediction.
# HISTORY_STEPS: Number of historical time slices to use for features.
# HISTORY_STEP_MINUTES: The interval in minutes between each history step.
# PREDICTION_HORIZON_STEPS: How many steps into the future the model
# predicts.
HISTORY_STEPS: int = int(os.getenv("HISTORY_STEPS", "6"))
HISTORY_STEP_MINUTES: int = int(os.getenv("HISTORY_STEP_MINUTES", "10"))
# Prediction horizon used during calibration to determine future target.
PREDICTION_HORIZON_STEPS: int = int(
    os.getenv("PREDICTION_HORIZON_STEPS", "24")
)
# The number of hours of historical data to use for initial training.
TRAINING_LOOKBACK_HOURS: int = int(os.getenv("TRAINING_LOOKBACK_HOURS", "168"))

# --- Core Entity IDs ---
# These are the most critical entities for the script's operation.
# **It is essential to update these to match your Home Assistant setup.**
# TARGET_INDOOR_TEMP_ENTITY_ID: The desired indoor temperature (e.g., from
# a thermostat).
# INDOOR_TEMP_ENTITY_ID: The current actual indoor temperature.
# ACTUAL_OUTLET_TEMP_ENTITY_ID: The current measured boiler outlet
# temperature.
# TARGET_OUTLET_TEMP_ENTITY_ID: The sensor this script will create/update
# with its calculated temperature.
TARGET_INDOOR_TEMP_ENTITY_ID: str = os.getenv(
    "TARGET_INDOOR_TEMP_ENTITY_ID",
    "input_number.hp_auto_correct_target",
)
INDOOR_TEMP_ENTITY_ID: str = os.getenv(
    "INDOOR_TEMP_ENTITY_ID", "sensor.kuche_temperatur"
)
ACTUAL_OUTLET_TEMP_ENTITY_ID: str = os.getenv(
    "ACTUAL_OUTLET_TEMP_ENTITY_ID", "sensor.hp_outlet_temp"
)
INLET_TEMP_ENTITY_ID: str = os.getenv(
    "INLET_TEMP_ENTITY_ID", "sensor.hp_inlet_temp"
)
FLOW_RATE_ENTITY_ID: str = os.getenv(
    "FLOW_RATE_ENTITY_ID", "sensor.hp_current_flow_rate"
)
POWER_CONSUMPTION_ENTITY_ID: str = os.getenv(
    "POWER_CONSUMPTION_ENTITY_ID", "sensor.power_wp"
)
SPECIFIC_HEAT_CAPACITY: float = float(
    os.getenv("SPECIFIC_HEAT_CAPACITY", "4.186")
)
# The entity the script will write the final calculated temperature to.
TARGET_OUTLET_TEMP_ENTITY_ID: str = os.getenv(
    "TARGET_OUTLET_TEMP_ENTITY_ID", "sensor.ml_vorlauftemperatur"
)
# The entity to read what outlet temperature was actually set (for learning).
# In active mode: should match TARGET_OUTLET_TEMP_ENTITY_ID
# In shadow mode: reads the heat curve's setting
ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID: str = os.getenv(
    "ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID", "sensor.hp_target_temp_circuit1"
)
# The heat pump's own heat-curve target outlet, independent of the ML command
# entity. Used to source an uncontaminated heat-curve anchor for feature
# injection (heat_curve_temp) and shadow-mode deviation metrics, even in
# active mode where ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID points at the ML's
# own entity (sensor.ml_vorlauftemperatur) — which would create a feedback
# loop. Falls back to ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID for backward
# compatibility (shadow mode where both resolve to the same entity).
HEAT_CURVE_OUTLET_ENTITY_ID: str = os.getenv(
    "HEAT_CURVE_OUTLET_ENTITY_ID"
) or os.getenv(
    "ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID", "sensor.hp_target_temp_circuit1"
)

# --- Blocking & Status Entity IDs ---
# These binary sensors can pause the script's operation. For example, if the
# system is busy with Domestic Hot Water (DHW) heating, the script will wait.
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

# --- Additional Sensor IDs ---
# These entities provide extra context to the model as features. The more
# relevant data the model has, the better its predictions can be.
TV_STATUS_ENTITY_ID: str = os.getenv(
    "TV_STATUS_ENTITY_ID", "input_boolean.fernseher"
)
OUTDOOR_TEMP_ENTITY_ID: str = os.getenv(
    "OUTDOOR_TEMP_ENTITY_ID", "sensor.thermometer_waermepume_kompensiert"
)
PV_POWER_ENTITY_ID: str = os.getenv(
    "PV_POWER_ENTITY_ID", "sensor.power_pv"
)

# PV forecast sensor (HA attributes 'watts' available in 15-min steps)
PV_FORECAST_ENTITY_ID: str = os.getenv(
    "PV_FORECAST_ENTITY_ID", "sensor.energy_production_today_4"
)
HEATING_STATUS_ENTITY_ID: str = os.getenv(
    "HEATING_STATUS_ENTITY_ID", "climate.heizung_2"
)
OPENWEATHERMAP_TEMP_ENTITY_ID: str = os.getenv(
    "OPENWEATHERMAP_TEMP_ENTITY_ID", "sensor.openweathermap_temperature"
)
AVG_OTHER_ROOMS_TEMP_ENTITY_ID: str = os.getenv(
    "AVG_OTHER_ROOMS_TEMP_ENTITY_ID", "sensor.avg_other_rooms_temp"
)
FIREPLACE_STATUS_ENTITY_ID: str = os.getenv(
    "FIREPLACE_STATUS_ENTITY_ID", "binary_sensor.fireplace_active"
)

# --- Tuning & Debug Parameters ---
# DEBUG: Set to "1" to enable verbose logging for development.
# CONFIDENCE_THRESHOLD: A critical tuning parameter. The thermal model's
#   learning_confidence operates on a 0.1-5.0 scale (initialized to 3.0, clipped
#   to [0.1, 5.0], multiplicatively boosted x1.1 / decayed x0.99 per prediction
#   based on prediction error). If learning_confidence drops below this
#   threshold, the trust gate demotes the model to heat-curve passthrough.
#   Default 2.0 is conservative for the 0.1-5.0 scale; lower it to tolerate more
#   model uncertainty, raise it to demand higher confidence before ML engages.
# HEAT_BALANCE_MODE: Enable the intelligent heat balance controller that uses
#   trajectory prediction instead of smoothing. Uses 3-phase control:
#   Charging (>0.5°C error), Balancing (0.2-0.5°C), Maintenance (<0.2°C).
# TRAJECTORY_STEPS: Number of hours to predict in trajectory optimization.
# CYCLE_INTERVAL_MINUTES: The time in minutes between each full cycle of
#   learning and prediction. A longer interval (e.g., 10-15 mins) provides a
#   clearer learning signal, while a shorter one is more responsive.
# MAX_TEMP_CHANGE_PER_CYCLE: The maximum allowable integer change (in degrees)
#   for the outlet temperature setpoint in a single cycle. This prevents
#   abrupt changes and is required as the heatpump only accepts full degrees.
DEBUG: bool = os.getenv("DEBUG", "0") == "1"
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "2.0"))

# --- T0.7: Self-Calibration Trust Gate ---
# The ML heating system is always enabled. While the thermal model is not yet
# trusted, it commands the heat-pump's own curve (passthrough) and learns in the
# background. Each cycle it evaluates a compound gate; when all criteria are met
# it auto-promotes itself to "calibrated" and starts commanding the ML-predicted
# outlet. If accuracy later degrades it auto-demotes back to passthrough.
# These thresholds are conservative defaults — calibrate them from one cold
# season of production telemetry (logged every cycle as promotion_progress).
AUTO_CAL_MIN_PREDICTIONS: int = int(
    os.getenv("AUTO_CAL_MIN_PREDICTIONS", "200")
)
# AUTO_CAL_MAE_THRESHOLD: The 24h-mean-absolute-error gate (in °C) for
#   trust-gate promotion/demotion. Default 1.0 °C. This is distinct from
#   ERROR_THRESHOLD_MEDIUM (0.2 °C, in thermal_constants.py), which triggers
#   the adaptive-learning rate boost — a finer-grained signal, not a gate.
AUTO_CAL_MAE_THRESHOLD: float = float(
    os.getenv("AUTO_CAL_MAE_THRESHOLD", "1.0")
)
AUTO_CAL_BIAS_THRESHOLD: float = float(
    os.getenv("AUTO_CAL_BIAS_THRESHOLD", "0.5")
)
AUTO_CAL_GOOD_CONTROL_PCT: float = float(
    os.getenv("AUTO_CAL_GOOD_CONTROL_PCT", "80.0")
)
# Max std/range of a learned parameter (fraction of its allowed bounds) over
# the last AUTO_CAL_PARAM_STABILITY_WINDOW history records for the model to be
# considered "converged". Blocks promotion when parameters are still drifting
# (the key inversion-safety guard: a model that looks accurate but has not
# converged can invert dangerously).
AUTO_CAL_PARAM_STABILITY_FRAC: float = float(
    os.getenv("AUTO_CAL_PARAM_STABILITY_FRAC", "0.05")
)
AUTO_CAL_PARAM_STABILITY_WINDOW: int = int(
    os.getenv("AUTO_CAL_PARAM_STABILITY_WINDOW", "20")
)
# A trusted model is demoted to passthrough when accuracy breaches the
# promotion thresholds multiplied by this factor (sustained). It re-promotes
# when accuracy recovers below the promotion thresholds again.
DEMOTION_FACTOR: float = float(os.getenv("DEMOTION_FACTOR", "1.5"))
# Number of consecutive cycles the demotion condition must hold before
# demoting, to avoid reacting to transient spikes.
DEMOTION_SUSTAINED_CYCLES: int = int(
    os.getenv("DEMOTION_SUSTAINED_CYCLES", "6")
)
# Conservative fallback outlet (°C) used in passthrough when the heat-curve
# sensor reading is unavailable. Lower than the old 45°C ceiling to avoid
# overheating from an untrained model.
CONSERVATIVE_FALLBACK_TEMP: float = float(
    os.getenv("CONSERVATIVE_FALLBACK_TEMP", "35.0")
)
# Minimum plausible heat-curve outlet (°C). Values below this in passthrough
# are treated as sensor garbage and replaced with the conservative fallback.
# Set low enough to accept legitimate mild-weather curves (≈20°C) but high
# enough to reject zero/negative readings from a stale or offline sensor.
AUTO_CAL_PASSTHROUGH_MIN_OUTLET: float = float(
    os.getenv("AUTO_CAL_PASSTHROUGH_MIN_OUTLET", "15.0")
)
# Maximum safe heat-curve outlet (°C) for passthrough. Values above this
# are treated as non-physical / feedback-loop contamination and replaced
# with the conservative fallback. Set high enough for genuine cold-weather
# curves (up to ~55 °C at -15 °C outdoor) but low enough to reject the
# 53 °C spike that occurs when the ML system feeds back into the heat-curve
# entity. Default 50 °C; tune via env var if needed.
PASSTHROUGH_MAX_SAFE_OUTLET: float = float(
    os.getenv("PASSTHROUGH_MAX_SAFE_OUTLET", "50.0")
)

# --- Saturation Guard ---
# When the binary search saturates at the absolute clamp ceiling (heating) or
# floor (cooling) and the predicted indoor temperature *still* cannot reach
# the target, the model's outlet→indoor coupling is too weak to be trustworthy
# (typically outlet-effectiveness drift after a long heating-off period). In
# that case the ML must NOT command the clamp extreme; instead it falls back
# toward the heat-pump's own curve (mirroring the curve's deliberate +shift
# self-limit) and accepts the residual error. This directly prevents the
# OFF→ON transition spike (e.g. 65 °C commanded vs ~42 °C heat curve, +23 °C
# deviation) observed when stale outlet effectiveness makes the search
# unable to reach target even at the ceiling.
SATURATION_GUARD_ENABLED: bool = (
    os.getenv("SATURATION_GUARD_ENABLED", "true").lower() == "true"
)
# How close to the clamp (°C) counts as "saturated". The binary search result
# is considered saturated if it is within this margin of CLAMP_MAX_ABS
# (heating) or CLAMP_MIN_ABS (cooling).
SATURATION_GUARD_MARGIN: float = float(
    os.getenv("SATURATION_GUARD_MARGIN", "0.5")
)
# Maximum deviation (°C) the guarded outlet may exceed the heat-curve outlet
# when saturated (heating) / fall below it (cooling). Mirrors the heat curve's
# own +shift cap so the ML is no more aggressive than the curve it replaces.
SATURATION_GUARD_MAX_DELTA: float = float(
    os.getenv("SATURATION_GUARD_MAX_DELTA", "10.0")
)

# OE delta mean-reversion safety net (drift recurrence fix).
# ``outlet_effectiveness_delta`` (persisted in unified_thermal_state.json) is
# a pure accumulator: every cycle adds the new OE adjustment with no reversion
# path, so any biased OE signal drifts it permanently across a season (the
# 2026-10-12 manual reset fixed the value, not the mechanism). The restoring
# force is the equilibrium-based OE gradient, but it is gated out of
# active-heating cycles by the 0.15 C/cycle stability threshold, while the
# biased trajectory gradient still fires on those cycles -> one-sided drift.
# When there is no clean OE signal (fully-passive window OR zero stable active
# records) the persisted delta is decayed toward 0.0 so the runtime OE
# self-heals to baseline on the next restart instead of ratcheting away. On
# cycles that carry a clean OE signal (>=1 stable active record) the decay is
# skipped entirely -- behaviour is byte-identical to the legacy accumulator.
# Disable to revert to pure-accumulator behaviour for debugging/A-B testing.
OE_DELTA_MEAN_REVERSION_ENABLED: bool = (
    os.getenv("OE_DELTA_MEAN_REVERSION_ENABLED", "true").lower() == "true"
)
# Fraction of the accumulated OE delta reverted toward 0 per adaptation cycle
# when mean-reversion is active. 1e-4/cycle is ~0.7%/day at 30-min cycles --
# slow enough that it never opposes a genuine clean-signal update (which
# writes the full gradient step, not the decay), fast enough to heal a
# season-scale drift in a few days.
OE_DELTA_MEAN_REVERSION_RATE: float = float(
    os.getenv("OE_DELTA_MEAN_REVERSION_RATE", "0.0001")
)

TRAJECTORY_STEPS: int = int(os.getenv("TRAJECTORY_STEPS", "4"))
CYCLE_INTERVAL_MINUTES: int = int(os.getenv("CYCLE_INTERVAL_MINUTES", "10"))
MAX_TEMP_CHANGE_PER_CYCLE: int = int(
    os.getenv("MAX_TEMP_CHANGE_PER_CYCLE", "2")
)
# Maximum minutes to wait during the grace period after blocking ends.
GRACE_PERIOD_MAX_MINUTES: int = int(
    os.getenv("GRACE_PERIOD_MAX_MINUTES", "15")
)

# How often (seconds) to poll blocking entities during the idle period.
# A value of 60 means we check the blocking state once per minute.
BLOCKING_POLL_INTERVAL_SECONDS: int = int(
    os.getenv("BLOCKING_POLL_INTERVAL_SECONDS", "60")
)

# --- Metrics Entity IDs ---
# These entities are created in Home Assistant to allow real-time monitoring
# of the model's performance and health.
# Note: Model confidence is now provided via sensor.ml_heating_learning.state
MAE_ENTITY_ID: str = os.getenv("MAE_ENTITY_ID", "sensor.ml_model_mae")
RMSE_ENTITY_ID: str = os.getenv("RMSE_ENTITY_ID", "sensor.ml_model_rmse")

# --- Multi-Lag Learning Configuration ---
# Enable time-delayed learning for external heat sources (PV, fireplace, TV)
# to capture realistic time delays (e.g., PV warming peaks 60-90min
# after production)
ENABLE_MULTI_LAG_LEARNING: bool = (
    os.getenv("ENABLE_MULTI_LAG_LEARNING", "true").lower() == "true"
)
PV_LAG_STEPS: int = int(os.getenv("PV_LAG_STEPS", "4"))
FIREPLACE_LAG_STEPS: int = int(os.getenv("FIREPLACE_LAG_STEPS", "4"))
TV_LAG_STEPS: int = int(os.getenv("TV_LAG_STEPS", "2"))

# --- Seasonal Adaptation Configuration ---
# Enable automatic seasonal learning to eliminate need for recalibration
# between winter and summer. Uses cos/sin modulation.
ENABLE_SEASONAL_ADAPTATION: bool = (
    os.getenv("ENABLE_SEASONAL_ADAPTATION", "true").lower() == "true"
)
SEASONAL_LEARNING_RATE: float = float(
    os.getenv("SEASONAL_LEARNING_RATE", "0.01")
)
MIN_SEASONAL_SAMPLES: int = int(os.getenv("MIN_SEASONAL_SAMPLES", "100"))

# --- Summer Learning Configuration ---
# Enable learning from periods when HVAC is off (typically summer) for
# cleaner signal of external source effects
ENABLE_SUMMER_LEARNING: bool = (
    os.getenv("ENABLE_SUMMER_LEARNING", "true").lower() == "true"
)

# --- Shadow Mode Configuration ---
# SHADOW_MODE: When true, ML runs in observation mode without affecting heating
# - ML predictions are calculated but not sent to heating system
# - No HA sensors are updated (target temp, confidence, MAE, RMSE, state)
# - System learns from heat curve's actual control decisions
# - Performance comparison logging between ML vs heat curve
SHADOW_MODE: bool = os.getenv("SHADOW_MODE", "false").lower() == "true"

# --- ML Heating Control Entity ---
# ML_HEATING_CONTROL_ENTITY_ID: HA input_boolean to enable/disable ML control
# - When ON: ML actively controls heating (Active Mode)
# - When OFF: Shadow mode (ML observes only, doesn't control)
# - Note: SHADOW_MODE environment variable overrides this setting
ML_HEATING_CONTROL_ENTITY_ID: str = os.getenv(
    "ML_HEATING_CONTROL_ENTITY_ID",
    "input_boolean.ml_heating"
)

# --- Thermal Equilibrium Model Parameters ---
# DEPRECATED: These parameters are now managed by the unified
# ThermalParameterManager and are sourced from src/thermal_config.py.
# Environment variables set for these will be loaded by the manager.

# Adaptive Learning Parameters (Priority 3 - Advanced Tuning)
# Error analysis window size
RECENT_ERRORS_WINDOW: int = int(os.getenv("RECENT_ERRORS_WINDOW", "10"))

# --- Hybrid Learning Strategy (Phase 2 Enhancement) ---
# Enable intelligent learning phase classification with weighted periods
HYBRID_LEARNING_ENABLED: bool = (
    os.getenv("HYBRID_LEARNING_ENABLED", "true").lower() == "true"
)
STABILITY_CLASSIFICATION_ENABLED: bool = (
    os.getenv("STABILITY_CLASSIFICATION_ENABLED", "true").lower() == "true"
)
HIGH_CONFIDENCE_WEIGHT: float = float(
    os.getenv("HIGH_CONFIDENCE_WEIGHT", "1.0")
)
LOW_CONFIDENCE_WEIGHT: float = float(
    os.getenv("LOW_CONFIDENCE_WEIGHT", "0.3")
)
LEARNING_PHASE_SKIP_WEIGHT: float = float(
    os.getenv("LEARNING_PHASE_SKIP_WEIGHT", "0.0")
)

# --- MAE/RMSE Tracking System (Phase 2 Enhancement) ---
# Enable comprehensive prediction accuracy tracking
PREDICTION_METRICS_ENABLED: bool = (
    os.getenv("PREDICTION_METRICS_ENABLED", "true").lower() == "true"
)
METRICS_WINDOW_1H: int = int(os.getenv("METRICS_WINDOW_1H", "12"))
METRICS_WINDOW_6H: int = int(os.getenv("METRICS_WINDOW_6H", "72"))
METRICS_WINDOW_24H: int = int(os.getenv("METRICS_WINDOW_24H", "288"))
PREDICTION_ACCURACY_THRESHOLD: float = float(
    os.getenv("PREDICTION_ACCURACY_THRESHOLD", "0.3")
)

# --- Trajectory Prediction Enhancement (Phase 2) ---
# Advanced trajectory prediction with forecast integration
TRAJECTORY_PREDICTION_ENABLED: bool = (
    os.getenv("TRAJECTORY_PREDICTION_ENABLED", "true").lower() == "true"
)
WEATHER_FORECAST_INTEGRATION: bool = (
    os.getenv("WEATHER_FORECAST_INTEGRATION", "true").lower() == "true"
)
PV_FORECAST_INTEGRATION: bool = (
    os.getenv("PV_FORECAST_INTEGRATION", "true").lower() == "true"
)
OVERSHOOT_DETECTION_ENABLED: bool = (
    os.getenv("OVERSHOOT_DETECTION_ENABLED", "true").lower() == "true"
)

# --- Historical Calibration System (Phase 0) ---
# Physics-based historical parameter optimization
STABILITY_TEMP_CHANGE_THRESHOLD: float = float(
    os.getenv("STABILITY_TEMP_CHANGE_THRESHOLD", "0.1")
)
MIN_STABLE_PERIOD_MINUTES: int = int(
    os.getenv("MIN_STABLE_PERIOD_MINUTES", "30")
)
OPTIMIZATION_METHOD: str = os.getenv("OPTIMIZATION_METHOD", "L-BFGS-B")

# --- Delta Temperature Forecast Calibration ---
# Enable local calibration of weather forecasts using measured temperature
# offset. This corrects for systematic biases between weather station and
# actual location.
ENABLE_DELTA_FORECAST_CALIBRATION: bool = (
    os.getenv("ENABLE_DELTA_FORECAST_CALIBRATION", "true").lower() == "true"
)
# Maximum allowed temperature offset to prevent unrealistic corrections.
DELTA_CALIBRATION_MAX_OFFSET: float = float(
    os.getenv("DELTA_CALIBRATION_MAX_OFFSET", "10.0")
)

# Absolute clamp values for outlet temperature
CLAMP_MIN_ABS: float = float(os.getenv("CLAMP_MIN_ABS", "20.0"))
CLAMP_MAX_ABS: float = float(os.getenv("CLAMP_MAX_ABS", "55.0"))

# Thermal Model Parameters
PV_HEAT_WEIGHT: float = float(os.getenv("PV_HEAT_WEIGHT", "0.002"))
FIREPLACE_HEAT_WEIGHT: float = float(os.getenv("FIREPLACE_HEAT_WEIGHT", "5.0"))
TV_HEAT_WEIGHT: float = float(os.getenv("TV_HEAT_WEIGHT", "0.2"))
THERMAL_TIME_CONSTANT: float = float(os.getenv("THERMAL_TIME_CONSTANT", "4.0"))
EQUILIBRIUM_RATIO: float = float(os.getenv("EQUILIBRIUM_RATIO", "0.17"))
TOTAL_CONDUCTANCE: float = float(os.getenv("TOTAL_CONDUCTANCE", "0.24"))
ADAPTIVE_LEARNING_RATE: float = float(
    os.getenv("ADAPTIVE_LEARNING_RATE", "0.01")
)
LEARNING_CONFIDENCE: float = float(os.getenv("LEARNING_CONFIDENCE", "3.0"))
MIN_LEARNING_RATE: float = float(os.getenv("MIN_LEARNING_RATE", "0.001"))
MAX_LEARNING_RATE: float = float(os.getenv("MAX_LEARNING_RATE", "0.1"))
SOLAR_LAG_MINUTES: float = float(os.getenv("SOLAR_LAG_MINUTES", "45.0"))
