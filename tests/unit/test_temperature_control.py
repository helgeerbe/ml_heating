
import pytest
from unittest.mock import MagicMock, patch
from src.temperature_control import (OnlineLearning, TemperatureControlManager, GradualTemperatureControl, SmartRounding, TemperaturePredictor)


@pytest.fixture
def gradual_control():
    return GradualTemperatureControl()


def test_gradual_control_no_change(gradual_control):
    """Test gradual control when the change is within the allowed limit."""
    final_temp = gradual_control.apply_gradual_control(41.0, 40.0, {})
    assert final_temp == 41.0


def test_gradual_control_positive_clamp(gradual_control):
    """Test gradual control when the change exceeds the positive limit."""
    with patch('src.temperature_control.config.MAX_TEMP_CHANGE_PER_CYCLE', 2.0):
        final_temp = gradual_control.apply_gradual_control(45.0, 40.0, {})
        assert final_temp == 42.0


def test_gradual_control_negative_clamp(gradual_control):
    """Test gradual control when the change exceeds the negative limit."""
    with patch('src.temperature_control.config.MAX_TEMP_CHANGE_PER_CYCLE', 2.0):
        final_temp = gradual_control.apply_gradual_control(37.0, 40.0, {})
        assert final_temp == 38.0


def test_gradual_control_with_last_final_temp(gradual_control):
    """Test that gradual control uses last_final_temp from state as baseline."""
    with patch('src.temperature_control.config.MAX_TEMP_CHANGE_PER_CYCLE', 2.0):
        state = {"last_final_temp": 42.0}
        final_temp = gradual_control.apply_gradual_control(45.0, 40.0, state)
        assert final_temp == 44.0


@pytest.fixture
def smart_rounding():
    return SmartRounding()


@patch('src.temperature_control.get_enhanced_model_wrapper')
def test_smart_rounding_chooses_floor(mock_get_wrapper, smart_rounding):
    """Test smart rounding when the floor temperature is a better fit."""
    mock_wrapper = MagicMock()
    mock_wrapper.cycle_aligned_forecast = {}
    mock_wrapper.predict_indoor_temp.side_effect = [20.9, 21.2]
    mock_get_wrapper.return_value = mock_wrapper

    rounded_temp = smart_rounding.apply_smart_rounding(42.4, 21.0)
    assert rounded_temp == 42


@patch('src.temperature_control.get_enhanced_model_wrapper')
def test_smart_rounding_chooses_ceiling(mock_get_wrapper, smart_rounding):
    """Test smart rounding when the ceiling temperature is a better fit."""
    mock_wrapper = MagicMock()
    mock_wrapper.cycle_aligned_forecast = {}
    mock_wrapper.predict_indoor_temp.side_effect = [21.2, 20.9]
    mock_get_wrapper.return_value = mock_wrapper

    rounded_temp = smart_rounding.apply_smart_rounding(42.6, 21.0)
    assert rounded_temp == 43


@pytest.fixture
def temperature_predictor():
    return TemperaturePredictor()


@patch('src.temperature_control.simplified_outlet_prediction')
def test_predict_optimal_temperature(mock_simplified_prediction, temperature_predictor):
    """Test the TemperaturePredictor class."""
    mock_simplified_prediction.return_value = (45.0, 0.9, {})
    features = {'some_feature': 1}
    predicted_temp, confidence, metadata = temperature_predictor.predict_optimal_temperature(
        features, 20.0, 21.0
    )
    assert predicted_temp == 45.0
    assert confidence == 0.9
    mock_simplified_prediction.assert_called_once_with(features, 20.0, 21.0)


@pytest.fixture
def online_learning():
    return OnlineLearning()


def test_online_learning_skip_no_previous_data(online_learning):
    """Test that learning is skipped if there is no previous data."""
    mock_ha_client = MagicMock()
    with patch('src.temperature_control.get_enhanced_model_wrapper') as mock_get_wrapper:
        online_learning.learn_from_previous_cycle({}, mock_ha_client, {})
        mock_get_wrapper.return_value.learn_from_prediction_feedback.assert_not_called()


@patch('src.temperature_control.get_enhanced_model_wrapper')
def test_online_learning_active_mode(mock_get_wrapper, online_learning):
    """Test online learning in active mode."""
    mock_wrapper = MagicMock()
    mock_get_wrapper.return_value = mock_wrapper
    mock_ha_client = MagicMock()
    mock_ha_client.get_state.side_effect = [43.0, 21.5]  # applied_temp, current_indoor
    state = {
        "last_run_features": {"outdoor_temp": 5.0},
        "last_indoor_temp": 21.0,
        "last_final_temp": 43.0,
    }

    with patch('src.temperature_control.config.SHADOW_MODE', False):
        online_learning.learn_from_previous_cycle(state, mock_ha_client, {})

    mock_wrapper.learn_from_prediction_feedback.assert_called_once()
    args, kwargs = mock_wrapper.learn_from_prediction_feedback.call_args
    assert kwargs['actual_temp'] == 21.5
    assert kwargs['prediction_context']['outlet_temp'] == 43.0


@patch('src.temperature_control.get_enhanced_model_wrapper')
def test_online_learning_shadow_mode(mock_get_wrapper, online_learning):
    """Test online learning in shadow mode."""
    mock_wrapper = MagicMock()
    mock_get_wrapper.return_value = mock_wrapper
    mock_ha_client = MagicMock()
    # In shadow mode, applied temp (from heat curve) is different from ML's prediction
    mock_ha_client.get_state.side_effect = [45.0, 21.5]  # applied_temp, current_indoor
    state = {
        "last_run_features": {"outdoor_temp": 5.0},
        "last_indoor_temp": 21.0,
        "last_final_temp": 43.0,  # ML's prediction
    }

    with patch('src.temperature_control.config.SHADOW_MODE', True):
        online_learning.learn_from_previous_cycle(state, mock_ha_client, {})

    mock_wrapper.learn_from_prediction_feedback.assert_called_once()
    args, kwargs = mock_wrapper.learn_from_prediction_feedback.call_args
    assert kwargs['actual_temp'] == 21.5
    assert kwargs['prediction_context']['outlet_temp'] == 45.0


@pytest.fixture
def manager():
    return TemperatureControlManager()


@patch('src.temperature_control.TemperaturePredictor.predict_optimal_temperature')
@patch('src.temperature_control.GradualTemperatureControl.apply_gradual_control')
@patch('src.temperature_control.SmartRounding.apply_smart_rounding')
def test_temperature_control_manager_full_cycle(
    mock_smart_rounding, mock_gradual_control, mock_predictor, manager
):
    """Test the full temperature control cycle of the manager."""
    mock_predictor.return_value = (45.0, 0.9, {})
    mock_gradual_control.return_value = 44.0
    mock_smart_rounding.return_value = 44

    features = {'outdoor_temp': 5.0}
    final_temp, confidence, metadata, smart_rounded_temp = manager.execute_temperature_control_cycle(
        features, 20.0, 21.0, 43.0, 5.0, False, {}
    )

    mock_predictor.assert_called_once_with(features, 20.0, 21.0)
    mock_gradual_control.assert_called_once_with(45.0, 43.0, {})
    mock_smart_rounding.assert_called_once_with(44.0, 21.0)
    assert final_temp == 44.0
    assert smart_rounded_temp == 44


@patch('src.temperature_control.get_enhanced_model_wrapper')
def test_online_learning_corrupted_features(mock_get_wrapper, online_learning):
    """Test that learning can handle corrupted (string) features."""
    mock_wrapper = MagicMock()
    mock_get_wrapper.return_value = mock_wrapper
    mock_ha_client = MagicMock()
    mock_ha_client.get_state.side_effect = [43.0, 21.5]
    state = {
        "last_run_features": '{"outdoor_temp": 5.0}',
        "last_indoor_temp": 21.0,
        "last_final_temp": 43.0,
    }

    with patch('src.temperature_control.config.SHADOW_MODE', False):
        online_learning.learn_from_previous_cycle(state, mock_ha_client, {})

    mock_wrapper.learn_from_prediction_feedback.assert_called_once()


@patch('src.temperature_control.OnlineLearning._export_shadow_benchmark_data')
def test_shadow_mode_logging(mock_export, online_learning):
    """Test that shadow mode comparison is logged."""
    with patch('src.temperature_control.config.SHADOW_MODE', True):
        online_learning._log_shadow_mode_comparison(45.0, 43.0)
        mock_export.assert_called_once()


@patch('src.temperature_control.get_enhanced_model_wrapper')
def test_calculate_ml_benchmark_prediction(mock_get_wrapper, online_learning):
    """Test the ML benchmark prediction calculation."""
    mock_wrapper = MagicMock()
    mock_wrapper.calculate_optimal_outlet_temp.return_value = {'optimal_outlet_temp': 42.0}
    mock_get_wrapper.return_value = mock_wrapper

    prediction = online_learning.calculate_ml_benchmark_prediction(
        21.0, 20.0, {'outdoor_temp': 5.0}
    )
    assert prediction == 42.0


def test_determine_prediction_indoor_temp_fireplace_on(manager):
    """Test that the average room temperature is used when the fireplace is on."""
    temp = manager.determine_prediction_indoor_temp(True, 21.0, 20.0)
    assert temp == 20.0


def test_determine_prediction_indoor_temp_fireplace_off(manager):
    """Test that the main indoor temperature is used when the fireplace is off."""
    temp = manager.determine_prediction_indoor_temp(False, 21.0, 20.0)
    assert temp == 21.0


@patch('src.temperature_control.build_physics_features')
def test_build_features_failure(mock_build_features, manager):
    """Test that None is returned when feature building fails."""
    mock_build_features.return_value = (None, None)
    features, history = manager.build_features(MagicMock(), MagicMock())
    assert features is None
    assert history is None


@patch('src.temperature_control.get_enhanced_model_wrapper')
def test_smart_rounding_fallback(mock_get_wrapper, smart_rounding):
    """Test that smart rounding falls back to standard rounding on error."""
    mock_get_wrapper.side_effect = Exception("Test error")
    rounded_temp = smart_rounding.apply_smart_rounding(42.7, 21.0)
    assert rounded_temp == 43


def test_gradual_control_no_actual_outlet_temp(gradual_control):
    """Test that gradual control returns the final temp if no actual outlet is available."""
    final_temp = gradual_control.apply_gradual_control(45.0, None, {})
    assert final_temp == 45.0


@patch('src.temperature_control.get_enhanced_model_wrapper')
def test_smart_rounding_none_prediction(mock_get_wrapper, smart_rounding):
    """Test fallback when prediction returns None."""
    mock_wrapper = MagicMock()
    mock_wrapper.cycle_aligned_forecast = {}
    mock_wrapper.predict_indoor_temp.side_effect = [None, 21.2]  # Simulate a None return
    mock_get_wrapper.return_value = mock_wrapper

    rounded_temp = smart_rounding.apply_smart_rounding(42.4, 21.0)
    assert rounded_temp == round(42.4)  # Should use standard rounding

