import logging
import sys
import os
from unittest.mock import MagicMock, patch
# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.heating_controller import (  # noqa: E402
    BlockingStateManager,
    SensorDataManager,
    HeatingSystemStateChecker
)
from src import config  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def debug_main_loop_logic():
    """
    Simulate the main loop logic to identify why target temperature
    calculation might be skipped.
    """
    print("\n=== Debugging Main Loop Logic ===\n")

    # Mock HA Client
    ha_client = MagicMock()
    
    # Mock States
    all_states = {
        config.ML_HEATING_CONTROL_ENTITY_ID: "on",
        config.HEATING_STATUS_ENTITY_ID: "heat",
        config.DHW_STATUS_ENTITY_ID: "off",
        config.DEFROST_STATUS_ENTITY_ID: "off",
        config.DISINFECTION_STATUS_ENTITY_ID: "off",
        config.DHW_BOOST_HEATER_STATUS_ENTITY_ID: "off",
        config.INDOOR_TEMP_ENTITY_ID: "20.5",
        config.TARGET_INDOOR_TEMP_ENTITY_ID: "21.0",
        config.ACTUAL_OUTLET_TEMP_ENTITY_ID: "35.0",
        config.TARGET_OUTLET_TEMP_ENTITY_ID: "35.0",
        config.OUTDOOR_TEMP_ENTITY_ID: "5.0",
        config.INLET_TEMP_ENTITY_ID: "30.0",
        config.FLOW_RATE_ENTITY_ID: "15.0",
        config.POWER_CONSUMPTION_ENTITY_ID: "1.5",
        config.AVG_OTHER_ROOMS_TEMP_ENTITY_ID: "20.0",
        config.FIREPLACE_STATUS_ENTITY_ID: "off",
        config.OPENWEATHERMAP_TEMP_ENTITY_ID: "5.0"
    }

    def get_state_side_effect(entity_id, states=None, is_binary=False):
        val = all_states.get(entity_id)
        if is_binary:
            return val == "on"
        return val

    ha_client.get_state.side_effect = get_state_side_effect
    ha_client.get_all_states.return_value = all_states

    # Mock State Manager
    state = MagicMock()
    state.last_is_blocking = False
    state.last_blocking_end_time = None
    state.last_final_temp = 35.0
    state.get.return_value = None  # Default get behavior

    # 1. Check Blocking State
    print("--- Checking Blocking State ---")
    blocking_manager = BlockingStateManager()
    is_blocking, blocking_reasons = blocking_manager.check_blocking_state(
        ha_client, all_states
    )
    print(f"Is Blocking: {is_blocking}")
    print(f"Blocking Reasons: {blocking_reasons}")

    # 2. Check Grace Period
    print("\n--- Checking Grace Period ---")
    # Simulate no shadow mode
    is_grace_period = blocking_manager.handle_grace_period(
        ha_client, state, shadow_mode=False
    )
    print(f"Is Grace Period: {is_grace_period}")

    # 3. Check Heating Active
    print("\n--- Checking Heating Active ---")
    heating_checker = HeatingSystemStateChecker()
    is_heating_active = heating_checker.check_heating_active(
        ha_client, all_states
    )
    print(f"Is Heating Active: {is_heating_active}")

    # 4. Check Sensor Data
    print("\n--- Checking Sensor Data ---")
    sensor_manager = SensorDataManager()
    sensor_data, missing_sensors = sensor_manager.get_sensor_data(
        ha_client, cycle_number=1
    )
    print(f"Missing Sensors: {missing_sensors}")
    if sensor_data:
        print("Sensor Data Retrieved Successfully")

    # 5. Simulate Feature Building (Mocked)
    print("\n--- Simulating Feature Building ---")
    with patch('src.main.build_physics_features') as mock_build_features:
        # Return valid features
        mock_build_features.return_value = ({'feature1': 1}, {})

        features, _ = mock_build_features(ha_client, MagicMock(), MagicMock())
        if features is not None:
            print("Features Built Successfully")
        else:
            print("Feature Building Failed")

    # 6. Simulate Prediction (Mocked)
    print("\n--- Simulating Prediction ---")
    with patch('src.main.simplified_outlet_prediction') as mock_prediction:
        mock_prediction.return_value = (40.0, 1.0, {})

        suggested_temp, confidence, metadata = mock_prediction(
            features, 20.5, 21.0
        )
        print(f"Suggested Temp: {suggested_temp}")
        print(f"Confidence: {confidence}")

    print("\n=== Conclusion ===")
    if (not is_blocking and not is_grace_period and is_heating_active and
            not missing_sensors and features is not None):
        print("✅ Logic flow seems correct for normal operation.")
    else:
        print("❌ Logic flow interrupted.")


if __name__ == "__main__":
    debug_main_loop_logic()
