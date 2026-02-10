
import pytest
from src.prediction_context import UnifiedPredictionContext, PredictionContextManager


def test_create_prediction_context_with_forecasts():
    """Test creating a prediction context with forecast data."""
    features = {
        'temp_forecast_1h': 10, 'temp_forecast_2h': 11, 'temp_forecast_3h': 12, 'temp_forecast_4h': 13,
        'pv_forecast_1h': 100, 'pv_forecast_2h': 200, 'pv_forecast_3h': 300, 'pv_forecast_4h': 400,
    }
    thermal_features = {'fireplace_on': 1, 'tv_on': 0}
    context = UnifiedPredictionContext.create_prediction_context(features, 8, 50, thermal_features)
    assert context['use_forecasts'] is True
    assert context['avg_outdoor'] == 11.5
    assert context['avg_pv'] == 250


def test_create_prediction_context_without_forecasts():
    """Test creating a prediction context without forecast data."""
    thermal_features = {'fireplace_on': 0, 'tv_on': 1}
    context = UnifiedPredictionContext.create_prediction_context({}, 8, 50, thermal_features)
    assert context['use_forecasts'] is False
    assert context['avg_outdoor'] == 8
    assert context['avg_pv'] == 50


def test_get_thermal_model_params():
    """Test getting thermal model parameters from the context."""
    context = {'avg_outdoor': 10, 'avg_pv': 100, 'fireplace_on': 1, 'tv_on': 0}
    params = UnifiedPredictionContext.get_thermal_model_params(context)
    assert params['outdoor_temp'] == 10
    assert params['pv_power'] == 100


@pytest.fixture
def manager():
    return PredictionContextManager()


def test_prediction_context_manager(manager):
    """Test the PredictionContextManager class."""
    features = {'temp_forecast_1h': 10}
    thermal_features = {'fireplace_on': 1}
    manager.set_features(features)
    manager.create_context(8, 50, thermal_features)
    assert manager.get_context() is not None
    assert manager.get_thermal_model_params()['outdoor_temp'] == 8.5
    assert manager.get_forecast_arrays()[0][0] == 10
    assert manager.uses_forecasts() is True

