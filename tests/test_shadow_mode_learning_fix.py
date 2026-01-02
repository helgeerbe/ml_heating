"""
Test suite for shadow mode learning fix.

This test validates that shadow mode correctly learns from heat curve decisions
rather than ML's own predictions.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime


@pytest.fixture
def mock_ha_client():
    """Mock Home Assistant client with test data."""
    client = Mock()
    # Simulate shadow mode: ML calculates 45°C, HC applies 48°C
    client.get_state = Mock(side_effect=lambda entity, states=None, is_binary=False: {
        "sensor.target_indoor_temp": 21.0,
        "sensor.indoor_temp": 20.5,
        "sensor.actual_outlet_temp": 47.0,
        "sensor.avg_other_rooms_temp": 20.3,
        "binary_sensor.fireplace": False,
        "sensor.outdoor_temp": 5.0,
        "sensor.owm_temp": 5.2,
        # Key: Heat curve applied 48°C while ML calculated 45°C
        "number.target_outlet_temp": 48.0,  # Heat curve's setting
        "input_boolean.ml_heating": False,  # Shadow mode active
        "climate.heating": "heat",
        "binary_sensor.dhw": False,
        "binary_sensor.defrost": False,
        "binary_sensor.disinfection": False,
        "binary_sensor.dhw_boost": False,
    }.get(entity, None))
    
    client.get_all_states = Mock(return_value={"mock": "states"})
    client.set_state = Mock()
    return client


@pytest.fixture
def mock_state():
    """Mock previous cycle state with ML's calculated temperature."""
    return {
        "last_run_features": {
            "outdoor_temp": 5.0,
            "pv_now": 100.0,
            "fireplace_on": 0.0,
            "tv_on": 0.0,
        },
        "last_indoor_temp": 20.0,
        "last_final_temp": 45.0,  # ML calculated 45°C
        "last_is_blocking": False,
        "last_blocking_reasons": [],
    }


@pytest.fixture
def mock_wrapper():
    """Mock enhanced model wrapper for thermal model predictions."""
    wrapper = Mock()
    
    # Mock thermal model with predict_equilibrium_temperature method
    thermal_model = Mock()
    
    def mock_predict_equilibrium(outlet_temp, outdoor_temp, current_indoor, 
                                pv_power=0.0, fireplace_on=0.0, tv_on=0.0, 
                                _suppress_logging=False):
        """
        Mock thermal model prediction.
        
        Returns different predictions based on outlet temp:
        - 45°C (ML's calculation): would predict 20.8°C
        - 48°C (HC's setting): would predict 21.3°C
        """
        if outlet_temp == 45.0:
            return 20.8  # ML's prediction for its own calculation
        elif outlet_temp == 48.0:
            return 21.3  # What heat curve's setting would achieve
        else:
            return 21.0  # Default
    
    thermal_model.predict_equilibrium_temperature = mock_predict_equilibrium
    wrapper.thermal_model = thermal_model
    wrapper.cycle_count = 1
    wrapper.learn_from_prediction_feedback = Mock()
    
    return wrapper


@patch('src.main.create_ha_client')
@patch('src.main.load_state')
@patch('src.main.save_state')
@patch('src.main.create_influx_service')
@patch('src.main.build_physics_features')
@patch('src.main.simplified_outlet_prediction')
@patch('src.model_wrapper.get_enhanced_model_wrapper')
def test_shadow_mode_learning_uses_heat_curve_prediction(
    mock_get_wrapper,
    mock_simplified_prediction,
    mock_build_features,
    mock_influx,
    mock_save_state,
    mock_load_state,
    mock_create_ha_client,
    mock_ha_client,
    mock_state,
    mock_wrapper
):
    """
    Test that shadow mode learning correctly predicts based on heat curve's
    outlet setting rather than ML's calculation.
    """
    # Setup mocks
    mock_create_ha_client.return_value = mock_ha_client
    mock_load_state.return_value = mock_state
    mock_get_wrapper.return_value = mock_wrapper
    
    # Mock feature building to return basic features
    mock_features = pd.DataFrame([{
        "outdoor_temp": 5.0,
        "pv_now": 100.0,
        "fireplace_on": 0.0,
        "tv_on": 0.0,
    }])
    mock_build_features.return_value = (mock_features, None)
    
    # Mock simplified prediction (ML's calculation)
    mock_simplified_prediction.return_value = (45.0, 3.0, {"predicted_indoor": 20.8})
    
    # Mock influx service
    mock_influx.return_value = Mock()
    
    # Import and run main function (single iteration)
    from src.main import main
    from unittest.mock import Mock as MockArgs
    
    args = MockArgs()
    args.debug = False
    args.calibrate_physics = False
    args.validate_physics = False
    
    # Override the infinite loop for testing
    with patch('src.main.time.sleep') as mock_sleep:
        # Make sleep raise exception to exit after first cycle
        mock_sleep.side_effect = KeyboardInterrupt("Test complete")
        
        try:
            main(args)
        except KeyboardInterrupt:
            pass  # Expected to exit after first cycle
    
    # Verify that learning was called with correct parameters
    mock_wrapper.learn_from_prediction_feedback.assert_called_once()
    
    # Get the call arguments
    call_args = mock_wrapper.learn_from_prediction_feedback.call_args
    predicted_temp = call_args[1]['predicted_temp']  # kwargs
    actual_temp = call_args[1]['actual_temp']
    prediction_context = call_args[1]['prediction_context']
    
    # Key assertion: predicted_temp should be 21.3°C (heat curve's 48°C prediction)
    # NOT 20.8°C (ML's 45°C prediction)
    assert predicted_temp == 21.3, (
        f"Shadow mode should predict based on heat curve's 48°C setting "
        f"(21.3°C), not ML's 45°C calculation (20.8°C). Got: {predicted_temp}°C"
    )
    
    # Verify actual temperature is current indoor temp
    assert actual_temp == 20.5, f"Expected actual_temp=20.5°C, got {actual_temp}°C"
    
    # Verify learning mode is correctly identified
    assert prediction_context['learning_mode'] == "shadow_mode_hc_observation", (
        f"Expected shadow_mode_hc_observation, got {prediction_context['learning_mode']}"
    )
    
    # Verify shadow mode cycle detection
    assert prediction_context['was_shadow_mode_cycle'] is True, (
        "Should detect shadow mode cycle when ML calculated 45°C but HC applied 48°C"
    )
    
    # Verify temperature context
    assert prediction_context['ml_calculated_temp'] == 45.0, (
        f"Expected ML calculated 45°C, got {prediction_context['ml_calculated_temp']}°C"
    )
    assert prediction_context['hc_applied_temp'] == 48.0, (
        f"Expected HC applied 48°C, got {prediction_context['hc_applied_temp']}°C"
    )


@patch('src.main.create_ha_client')
@patch('src.main.load_state')
@patch('src.main.save_state')
@patch('src.main.create_influx_service')
@patch('src.main.build_physics_features')
@patch('src.main.simplified_outlet_prediction')
@patch('src.model_wrapper.get_enhanced_model_wrapper')
def test_active_mode_learning_uses_ml_prediction(
    mock_get_wrapper,
    mock_simplified_prediction,
    mock_build_features,
    mock_influx,
    mock_save_state,
    mock_load_state,
    mock_create_ha_client,
    mock_ha_client,
    mock_state,
    mock_wrapper
):
    """
    Test that active mode learning correctly predicts based on ML's own
    outlet setting (unchanged behavior).
    """
    # Modify HA client for active mode: ML controls heating
    def active_mode_get_state(entity, states=None, is_binary=False):
        base_values = {
            "sensor.target_indoor_temp": 21.0,
            "sensor.indoor_temp": 20.5,
            "sensor.actual_outlet_temp": 47.0,
            "sensor.avg_other_rooms_temp": 20.3,
            "binary_sensor.fireplace": False,
            "sensor.outdoor_temp": 5.0,
            "sensor.owm_temp": 5.2,
            # Key: In active mode, ML's calculation is applied
            "number.target_outlet_temp": 45.0,  # ML's setting applied
            "input_boolean.ml_heating": True,  # Active mode
            "climate.heating": "heat",
            "binary_sensor.dhw": False,
            "binary_sensor.defrost": False,
            "binary_sensor.disinfection": False,
            "binary_sensor.dhw_boost": False,
        }
        return base_values.get(entity, None)
    
    mock_ha_client.get_state.side_effect = active_mode_get_state
    
    # Setup other mocks
    mock_create_ha_client.return_value = mock_ha_client
    mock_load_state.return_value = mock_state
    mock_get_wrapper.return_value = mock_wrapper
    
    # Mock feature building
    mock_features = pd.DataFrame([{
        "outdoor_temp": 5.0,
        "pv_now": 100.0,
        "fireplace_on": 0.0,
        "tv_on": 0.0,
    }])
    mock_build_features.return_value = (mock_features, None)
    
    # Mock simplified prediction
    mock_simplified_prediction.return_value = (45.0, 3.0, {"predicted_indoor": 20.8})
    
    # Mock influx service
    mock_influx.return_value = Mock()
    
    # Import and run main function (single iteration)
    from src.main import main
    from unittest.mock import Mock as MockArgs
    
    args = MockArgs()
    args.debug = False
    args.calibrate_physics = False
    args.validate_physics = False
    
    # Override the infinite loop for testing
    with patch('src.main.time.sleep') as mock_sleep:
        mock_sleep.side_effect = KeyboardInterrupt("Test complete")
        
        try:
            main(args)
        except KeyboardInterrupt:
            pass
    
    # Verify that learning was called
    mock_wrapper.learn_from_prediction_feedback.assert_called_once()
    
    # Get the call arguments
    call_args = mock_wrapper.learn_from_prediction_feedback.call_args
    predicted_temp = call_args[1]['predicted_temp']
    prediction_context = call_args[1]['prediction_context']
    
    # Key assertion: predicted_temp should be 20.8°C (ML's own 45°C prediction)
    assert predicted_temp == 20.8, (
        f"Active mode should predict based on ML's own 45°C setting "
        f"(20.8°C). Got: {predicted_temp}°C"
    )
    
    # Verify learning mode is correctly identified as active mode
    assert prediction_context['learning_mode'] == "active_mode_ml_feedback", (
        f"Expected active_mode_ml_feedback, got {prediction_context['learning_mode']}"
    )
    
    # Verify active mode cycle detection
    assert prediction_context['was_shadow_mode_cycle'] is False, (
        "Should detect active mode cycle when ML calculated and applied same temp"
    )


def test_shadow_mode_cycle_detection():
    """
    Test the logic for detecting whether a cycle was in shadow mode.
    """
    # Shadow mode: ML calculated 45°C, HC applied 48°C
    ml_calculated = 45.0
    hc_applied = 48.0
    was_shadow_mode = (hc_applied != ml_calculated)
    assert was_shadow_mode is True, "Should detect shadow mode when temperatures differ"
    
    # Active mode: ML calculated 45°C, ML applied 45°C
    ml_calculated = 45.0
    ml_applied = 45.0
    was_active_mode = (ml_applied == ml_calculated)
    assert was_active_mode is True, "Should detect active mode when temperatures match"


if __name__ == "__main__":
    pytest.main([__file__])
