import sys
import os
import logging
import time
from unittest.mock import MagicMock, patch

# Add root to path to allow importing from src
sys.path.append(os.getcwd())

from src.heating_controller import BlockingStateManager  # noqa: E402
from src.state_manager import SystemState  # noqa: E402
import src.config as config  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def reproduce_dhw_overshoot():
    print("\n--- Reproducing DHW Overshoot ---")

    # 1. Setup Mocks
    ha_client = MagicMock()

    # Mock sensor states
    # Scenario: Cold day, DHW just finished, house is slightly under target
    current_indoor = 20.0
    target_indoor = 21.0
    outdoor_temp = 5.0
    actual_outlet = 35.0  # Was idling or cooling down

    def get_state_side_effect(entity_id, all_states=None, is_binary=False):
        if entity_id == config.INDOOR_TEMP_ENTITY_ID:
            return current_indoor
        elif entity_id == config.TARGET_INDOOR_TEMP_ENTITY_ID:
            return target_indoor
        elif entity_id == config.OUTDOOR_TEMP_ENTITY_ID:
            return outdoor_temp
        elif entity_id == config.ACTUAL_OUTLET_TEMP_ENTITY_ID:
            return actual_outlet
        elif entity_id == config.DHW_STATUS_ENTITY_ID:
            return False  # Blocking ended
        elif entity_id == config.DEFROST_STATUS_ENTITY_ID:
            return False
        elif entity_id == config.DISINFECTION_STATUS_ENTITY_ID:
            return False
        elif entity_id == config.DHW_BOOST_HEATER_STATUS_ENTITY_ID:
            return False
        return None

    ha_client.get_state.side_effect = get_state_side_effect
    ha_client.get_all_states.return_value = {}  # Just needs to return a dict

    # Mock Model Wrapper to return a high temperature
    # This simulates the model thinking it needs to blast heat to recover
    mock_wrapper = MagicMock()
    # Simulate a high requirement (e.g. 65C)
    mock_wrapper._calculate_required_outlet_temp.return_value = 65.0
    mock_wrapper._extract_thermal_features.return_value = {"some": "features"}

    # 2. Setup BlockingStateManager
    bsm = BlockingStateManager()

    # 3. Setup SystemState
    # Simulate that we were blocking in the last cycle
    state = SystemState()
    state.last_is_blocking = True
    state.last_blocking_end_time = time.time() - 60  # Ended 1 minute ago
    state.last_final_temp = 40.0  # Previous setpoint before blocking

    # 4. Run handle_grace_period
    print("Initial Conditions:")
    print(f"  Indoor: {current_indoor}°C")
    print(f"  Target: {target_indoor}°C")
    print(f"  Outdoor: {outdoor_temp}°C")
    print(f"  Actual Outlet: {actual_outlet}°C")
    print(f"  Last Final Temp: {state.last_final_temp}°C")

    # Create a mock DataFrame for features
    mock_df = MagicMock()
    mock_df.empty = False
    mock_df.iloc = MagicMock()
    mock_df.iloc[0].to_dict.return_value = {"some": "features"}

    # Patch the imports where they are defined
    with patch(
        'src.model_wrapper.get_enhanced_model_wrapper',
        return_value=mock_wrapper
    ), patch(
        'src.physics_features.build_physics_features',
        return_value=(mock_df, MagicMock())
    ), patch(
        'src.influx_service.create_influx_service',
        return_value=MagicMock()
    ), patch(
        'src.config.MAX_TEMP_CHANGE_PER_CYCLE',
        2.0
    ):

        # We need to mock _wait_for_grace_target to avoid infinite loop
        with patch.object(bsm, '_wait_for_grace_target') as mock_wait:  # noqa: F841
            print("\nExecuting handle_grace_period...")
            bsm.handle_grace_period(ha_client, state, shadow_mode=False)

            # Check what target was set
            # ha_client.set_state is called with
            # (entity_id, value, attributes, round_digits)
            calls = ha_client.set_state.call_args_list
            target_set = None
            for call in calls:
                if call[0][0] == config.TARGET_OUTLET_TEMP_ENTITY_ID:
                    target_set = call[0][1]
                    print(f"\n[!] TARGET SET TO: {target_set}°C")

            if target_set is None:
                print("\n[?] No target set.")
            elif target_set > state.last_final_temp + 5.0:
                print(
                    f"\n[FAIL] Overshoot detected! Target {target_set}°C is "
                    f"much higher than previous {state.last_final_temp}°C"
                )
                print("The gradual control logic was bypassed.")
            else:
                print(
                    f"\n[PASS] Target {target_set}°C is within reasonable "
                    "range."
                )


if __name__ == "__main__":
    reproduce_dhw_overshoot()
