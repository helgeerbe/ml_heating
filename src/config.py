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
# For Home Assistant addon, uses internal supervisor API by default.
HASS_URL: str = os.getenv("HASS_URL", "http://supervisor/core")
HASS_TOKEN: str = os.getenv("SUPERVISOR_TOKEN", "").strip()
HASS_HEADERS: dict[str, str] = {
    "Authorization": f"Bearer {HASS_TOKEN}",
    "Content-Type": "application/json",
}
INFLUX_URL: str = os.getenv("INFLUX_URL", "https://influxdb.erbehome.de")
INFLUX_TOKEN: str = os.getenv("INFLUX_TOKEN", "")
INFLUX_ORG: str = os.getenv("INFLUX_ORG", "erbehome")
INFLUX_BUCKET: str = os.getenv("INFLUX_BUCKET", "home_assistant/autogen")

INFLUX_FEATURES_BUCKET: str = os.getenv("INFLUX_FEATURES_BUCKET", "ml_heating_features")

# --- File Paths ---
# Defines where the application's persistent data is stored.
# For Home Assistant addon, uses /data/ directory structure by default.
# MODEL_FILE: The trained machine learning model.
# STATE_FILE: The application's state, like prediction history.
MODEL_FILE: str = os.getenv("MODEL_FILE_PATH", "/data/models/ml_model.pkl")
STATE_FILE: str = os.getenv("STATE_FILE_PATH", "/data/models/ml_state.pkl")

# --- Model & History Parameters ---
# These parameters control the time windows for feature creation and prediction.
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
TRAINING_LOOKBACK_HOURS: int = int(
    os.getenv("TRAINING_LOOKBACK_HOURS", "168")
)

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
# CONFIDENCE_THRESHOLD: A critical tuning parameter. If the model's
#   normalized confidence score (0-1) drops below this threshold, the system
#   will fall back to the baseline temperature. A lower threshold means the
#   system is more tolerant of model uncertainty.
# HEAT_BALANCE_MODE: Enable the intelligent heat balance controller that uses
#   trajectory prediction instead of smoothing. Uses 3-phase control:
#   Charging (>0.5°C error), Balancing (0.2-0.5°C), Maintenance (<0.2°C).
# CHARGING_MODE_THRESHOLD: Temperature error threshold for entering charging
#   mode vs balancing mode. Higher values make charging mode more aggressive.
# MAINTENANCE_MODE_THRESHOLD: Temperature error threshold for entering
#   maintenance mode vs balancing mode. Lower values activate maintenance sooner.
# TRAJECTORY_STEPS: Number of hours to predict in trajectory optimization.
# OSCILLATION_PENALTY_WEIGHT: Penalty applied for temperature direction changes
#   in trajectory scoring. Higher values prioritize stability over accuracy.
# FINAL_DESTINATION_WEIGHT: Weight given to final trajectory endpoint in
#   scoring. Higher values prioritize long-term stability.
# CYCLE_INTERVAL_MINUTES: The time in minutes between each full cycle of
#   learning and prediction. A longer interval (e.g., 10-15 mins) provides a
#   clearer learning signal, while a shorter one is more responsive.
# MAX_TEMP_CHANGE_PER_CYCLE: The maximum allowable integer change (in degrees)
#   for the outlet temperature setpoint in a single cycle. This prevents
#   abrupt changes and is required as the heatpump only accepts full degrees.
DEBUG: bool = os.getenv("DEBUG", "0") == "1"
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.2"))
HEAT_BALANCE_MODE: bool = os.getenv("HEAT_BALANCE_MODE", "true").lower() == "true"
CHARGING_MODE_THRESHOLD: float = float(os.getenv("CHARGING_MODE_THRESHOLD", "0.2"))
MAINTENANCE_MODE_THRESHOLD: float = float(os.getenv("MAINTENANCE_MODE_THRESHOLD", "0.1"))
TRAJECTORY_STEPS: int = int(os.getenv("TRAJECTORY_STEPS", "4"))
OSCILLATION_PENALTY_WEIGHT: float = float(os.getenv("OSCILLATION_PENALTY_WEIGHT", "0.3"))
FINAL_DESTINATION_WEIGHT: float = float(os.getenv("FINAL_DESTINATION_WEIGHT", "2.0"))
CYCLE_INTERVAL_MINUTES: int = int(os.getenv("CYCLE_INTERVAL_MINUTES", "10"))
MAX_TEMP_CHANGE_PER_CYCLE: int = int(
    os.getenv("MAX_TEMP_CHANGE_PER_CYCLE", "2")
)
# Maximum minutes to wait during the grace period after blocking ends.
GRACE_PERIOD_MAX_MINUTES: int = int(
    os.getenv("GRACE_PERIOD_MAX_MINUTES", "30")
)

# How often (seconds) to poll blocking entities during the idle period.
# A value of 60 means we check the blocking state once per minute.
BLOCKING_POLL_INTERVAL_SECONDS: int = int(
    os.getenv("BLOCKING_POLL_INTERVAL_SECONDS", "60")
)

# --- Metrics Entity IDs ---
# These entities are created in Home Assistant to allow real-time monitoring
# of the model's performance and health.
CONFIDENCE_ENTITY_ID: str = os.getenv(
    "CONFIDENCE_ENTITY_ID", "sensor.ml_model_confidence"
)
MAE_ENTITY_ID: str = os.getenv("MAE_ENTITY_ID", "sensor.ml_model_mae")
RMSE_ENTITY_ID: str = os.getenv("RMSE_ENTITY_ID", "sensor.ml_model_rmse")

# --- Clamping (absolute) ---
# These define the absolute allowed range for any ML-proposed outlet temperature.
# Use environment variables CLAMP_MIN_ABS and CLAMP_MAX_ABS to override defaults.
CLAMP_MIN_ABS: float = float(os.getenv("CLAMP_MIN_ABS", "14.0"))
CLAMP_MAX_ABS: float = float(os.getenv("CLAMP_MAX_ABS", "65.0"))

# --- Multi-Lag Learning Configuration ---
# Enable time-delayed learning for external heat sources (PV, fireplace, TV)
# to capture realistic time delays (e.g., PV warming peaks 60-90min after production)
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
