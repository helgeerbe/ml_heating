"""Configuration for the ML Heating script."""
import os
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
TARGET_OUTLET_TEMP_ENTITY_ID: str = os.getenv(
    "TARGET_OUTLET_TEMP_ENTITY_ID", "sensor.ml_vorlauftemperatur"
)
PREDICTED_INDOOR_TEMP_ENTITY_ID: str = os.getenv(
    "PREDICTED_INDOOR_TEMP_ENTITY_ID", "sensor.ml_predicted_indoor_temp"
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
CONFIDENCE_ENTITY_ID: str = os.getenv(
    "CONFIDENCE_ENTITY_ID", "sensor.ml_model_confidence"
)
MAE_ENTITY_ID: str = os.getenv("MAE_ENTITY_ID", "sensor.ml_model_mae")
RMSE_ENTITY_ID: str = os.getenv("RMSE_ENTITY_ID", "sensor.ml_model_rmse")
