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
# It is critical to set these in your .env file.
HASS_URL: str = os.getenv("HASS_URL", "https://home.erbehome.de")
HASS_TOKEN: str = os.getenv("HASS_TOKEN", "").strip()
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
# MODEL_FILE: The trained machine learning model.
# STATE_FILE: The application's state, like prediction history.
MODEL_FILE: str = os.getenv("MODEL_FILE", "/opt/ml_heating/ml_model.pkl")
STATE_FILE: str = os.getenv("STATE_FILE", "/opt/ml_heating/ml_state.pkl")

# --- Model & History Parameters ---
# These parameters control the time windows for feature creation and prediction.
# HISTORY_STEPS: Number of historical time slices to use for features.
# HISTORY_STEP_MINUTES: The interval in minutes between each history step.
# PREDICTION_HORIZON_STEPS: How many steps into the future the model
# predicts.
HISTORY_STEPS: int = int(os.getenv("HISTORY_STEPS", "6"))
HISTORY_STEP_MINUTES: int = int(os.getenv("HISTORY_STEP_MINUTES", "10"))
# Defines the prediction horizon for the model. Not currently used in the core logic.
PREDICTION_HORIZON_STEPS: int = int(
    os.getenv("PREDICTION_HORIZON_STEPS", "24")
)
PREDICTION_HORIZON_MINUTES: int = (
    PREDICTION_HORIZON_STEPS * 5
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
# PREDICTED_INDOOR_TEMP_ENTITY_ID: The sensor for the model's indoor temp
# prediction.
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
# The entity for the predicted indoor temperature sensor.
PREDICTED_INDOOR_TEMP_ENTITY_ID: str = os.getenv(
    "PREDICTED_INDOOR_TEMP_ENTITY_ID", "sensor.ml_predicted_indoor_temp"
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
# SMOOTHING_ALPHA: The smoothing factor for the exponential moving average
#   applied to the model's temperature predictions. A lower value (e.g., 0.1)
#   results in more aggressive smoothing and less volatile output. A higher
#   value (e.g., 0.8) makes the output more responsive to the latest raw
#   prediction.
# CYCLE_INTERVAL_MINUTES: The time in minutes between each full cycle of
#   learning and prediction. A longer interval (e.g., 10-15 mins) provides a
#   clearer learning signal, while a shorter one is more responsive.
# MAX_TEMP_CHANGE_PER_CYCLE: The maximum allowable integer change (in degrees)
#   for the outlet temperature setpoint in a single cycle. This prevents
#   abrupt changes and is required as the heatpump only accepts full degrees.
DEBUG: bool = os.getenv("DEBUG", "0") == "1"
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.2"))
SMOOTHING_ALPHA: float = float(os.getenv("SMOOTHING_ALPHA", "0.3"))
CYCLE_INTERVAL_MINUTES: int = int(os.getenv("CYCLE_INTERVAL_MINUTES", "10"))
MAX_TEMP_CHANGE_PER_CYCLE: int = int(
    os.getenv("MAX_TEMP_CHANGE_PER_CYCLE", "2")
)

# --- Metrics Entity IDs ---
# These entities are created in Home Assistant to allow real-time monitoring
# of the model's performance and health.
CONFIDENCE_ENTITY_ID: str = os.getenv(
    "CONFIDENCE_ENTITY_ID", "sensor.ml_model_confidence"
)
MAE_ENTITY_ID: str = os.getenv("MAE_ENTITY_ID", "sensor.ml_model_mae")
RMSE_ENTITY_ID: str = os.getenv("RMSE_ENTITY_ID", "sensor.ml_model_rmse")
