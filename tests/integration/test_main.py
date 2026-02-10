
import pytest
from unittest.mock import patch, MagicMock
from src import config

# Since main.py is the entry point, we will test it in an
# integration-style manner. We will mock external dependencies and assert
# that the main function behaves as expected.


@patch("src.main.get_sensor_attributes", return_value={})
@patch("src.main.build_physics_features", return_value=({}, []))
@patch("src.main.poll_for_blocking")
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
    mock_poll_for_blocking,
    mock_build_features,
    mock_get_attributes,
):
    """Test that the main loop runs once and calls expected functions."""
    # Arrange
    mock_ha_instance = MagicMock()
    mock_create_ha_client.return_value = mock_ha_instance

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
    }
    mock_ha_instance.get_all_states.return_value = all_states

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

    mock_load_state.return_value = {}
    mock_simplified_outlet_prediction.return_value = (
        35.0,
        0.9,
        {"predicted_indoor": 21.1},
    )
    mock_poll_for_blocking.side_effect = InterruptedError

    # Act
    from src import main

    with patch.object(main, "time") as mock_time, patch(
        "src.model_wrapper.get_enhanced_model_wrapper"
    ) as mock_get_wrapper:
        mock_wrapper = MagicMock()
        mock_get_wrapper.return_value = mock_wrapper
        mock_wrapper.predict_indoor_temp.return_value = 21.0

        mock_time.time.side_effect = [1000.0, 1001.0, 1002.0, 1003.0]
        with pytest.raises(InterruptedError):
            with patch("sys.argv", ["main.py"]):
                main.main()

    # Assert
    mock_load_state.assert_called_once()
    mock_simplified_outlet_prediction.assert_called_once()
    mock_create_influx_service.assert_called_once()
    assert mock_create_ha_client.call_count > 0
    mock_save_state.assert_called_once()
    mock_poll_for_blocking.assert_called_once()
    mock_build_features.assert_called_once()
