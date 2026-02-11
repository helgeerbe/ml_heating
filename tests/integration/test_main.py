
from unittest.mock import patch, MagicMock
from src import config
from src.state_manager import SystemState
from src.heating_controller import BlockingStateManager

# Since main.py is the entry point, we will test it in an
# integration-style manner. We will mock external dependencies and assert
# that the main function behaves as expected.


@patch("src.main.get_sensor_attributes")
@patch("src.main.build_physics_features", return_value=({}, []))
@patch("src.main.create_ha_client")
@patch("src.main.create_influx_service")
@patch("src.main.simplified_outlet_prediction")
@patch("src.main.load_state")
@patch("src.main.save_state")
def test_main_loop_runs_once(
    mock_save_state,
    mock_load_state,
    mock_simplified_outlet_prediction,
    mock_create_influx_service,
    mock_create_ha_client,
    mock_build_features,
    mock_get_attributes,
):
    """Test that the main loop runs once and calls expected functions."""
    # Arrange
    mock_ha_instance = MagicMock()
    mock_create_ha_client.return_value = mock_ha_instance

    # Ensure get_sensor_attributes returns a fresh dict each time to avoid
    # shared state between calls (e.g. between status sensor and outlet sensor)
    mock_get_attributes.side_effect = lambda *args: {}

    # Mock InfluxService
    mock_influx_instance = MagicMock()
    mock_create_influx_service.return_value = mock_influx_instance

    # Define all states for the HA client to return
    all_states = {
        config.HEATING_STATUS_ENTITY_ID: {"state": "heat"},
        config.ML_HEATING_CONTROL_ENTITY_ID: {"state": "on"},
        config.TARGET_INDOOR_TEMP_ENTITY_ID: {"state": "21.0"},
        config.INDOOR_TEMP_ENTITY_ID: {"state": "20.5"},
        config.OUTDOOR_TEMP_ENTITY_ID: {"state": "10.0"},
        config.ACTUAL_OUTLET_TEMP_ENTITY_ID: {"state": "45.0"},
        config.AVG_OTHER_ROOMS_TEMP_ENTITY_ID: {"state": "20.0"},
        config.FIREPLACE_STATUS_ENTITY_ID: {"state": "off"},
        config.OPENWEATHERMAP_TEMP_ENTITY_ID: {"state": "9.0"},
        config.DHW_STATUS_ENTITY_ID: {"state": "off"},
        config.DEFROST_STATUS_ENTITY_ID: {"state": "off"},
        config.DISINFECTION_STATUS_ENTITY_ID: {"state": "off"},
        config.DHW_BOOST_HEATER_STATUS_ENTITY_ID: {"state": "off"},
        config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID: {"state": "35.0"},
        config.TV_STATUS_ENTITY_ID: {"state": "off"},
        config.PV_POWER_ENTITY_ID: {"state": "0.0"},
        config.PV_FORECAST_ENTITY_ID: {"state": "0.0", "attributes": {}},
    }

    # Mock get_all_states to return our dictionary
    # We use side_effect to raise KeyboardInterrupt on the 3rd call
    # to break the infinite loop in main()
    # Calls:
    # 1. Main loop start
    # 2. sensor_manager.get_sensor_data()
    # 3. Next loop start -> KeyboardInterrupt
    mock_ha_instance.get_all_states.side_effect = [
        all_states,
        all_states,
        KeyboardInterrupt("End of test loop"),
    ]

    # Correctly mock get_state to look up from the all_states dictionary
    def get_state_side_effect(entity_id, states_dict, is_binary=False):
        entity_info = states_dict.get(entity_id)
        if not entity_info:
            return None
        state = entity_info.get("state")
        if is_binary:
            return state == "on"
        try:
            return float(state)
        except (ValueError, TypeError):
            return state

    mock_ha_instance.get_state.side_effect = get_state_side_effect

    mock_load_state.return_value = SystemState()
    mock_simplified_outlet_prediction.return_value = (
        35.0,
        0.9,
        {"predicted_indoor": 21.1},
    )

    # Act
    from src import main

    # We need to patch BlockingStateManager.poll_for_blocking to avoid
    # infinite loop or long sleep, but we use the real class for other methods
    with patch.object(main, "time") as mock_time, patch(
        "src.model_wrapper.get_enhanced_model_wrapper"
    ) as mock_get_wrapper, patch.object(
        BlockingStateManager, "poll_for_blocking"
    ) as mock_poll_blocking:

        mock_wrapper = MagicMock()
        mock_get_wrapper.return_value = mock_wrapper
        mock_wrapper.predict_indoor_temp.return_value = 21.0

        # Mock time.time() to advance
        mock_time.time.side_effect = [1000.0, 1001.0, 1002.0, 1003.0, 1004.0]
        
        # Mock sleep to do nothing
        mock_time.sleep.return_value = None

        # Run main() - it should catch KeyboardInterrupt and exit
        with patch("sys.argv", ["main.py"]):
            try:
                main.main()
            except KeyboardInterrupt:
                pass

    # Assert
    # load_state is called at the start of each loop.
    # 1st call: First iteration (successful)
    # 2nd call: Second iteration (starts, then KeyboardInterrupt at get_all_states)
    assert mock_load_state.call_count == 2
    mock_simplified_outlet_prediction.assert_called_once()
    mock_create_influx_service.assert_called_once()
    assert mock_create_ha_client.call_count > 0
    
    # Verify save_state was called with expected values
    # This confirms data flowed through SensorDataManager to main logic
    mock_save_state.assert_called()
    call_kwargs = mock_save_state.call_args[1]
    assert call_kwargs.get("last_indoor_temp") == 20.5
    assert call_kwargs.get("last_final_temp") == 35.0
    
    # Verify set_state was called to update the target temperature
    # This confirms the prediction result was used
    # Note: The actual call might have attributes populated by get_sensor_attributes
    # so we check for the main arguments and ignore the exact attributes content
    # or use ANY for attributes if needed.
    # Based on main.py:
    # ha_client.set_state(
    #     config.TARGET_OUTLET_TEMP_ENTITY_ID,
    #     smart_rounded_temp,
    #     get_sensor_attributes(config.TARGET_OUTLET_TEMP_ENTITY_ID),
    #     round_digits=None,  # No additional rounding needed
    # )
    mock_ha_instance.set_state.assert_any_call(
        config.TARGET_OUTLET_TEMP_ENTITY_ID,
        35,  # Expect integer 35 (from round(35.0))
        {},  # attributes (mocked get_sensor_attributes returns {})
        round_digits=None
    )

    mock_poll_blocking.assert_called_once()
    mock_build_features.assert_called_once()
