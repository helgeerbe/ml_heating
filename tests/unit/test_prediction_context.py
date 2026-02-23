import pytest
from unittest.mock import patch
import src.prediction_context
from src.prediction_context import UnifiedPredictionContext, PredictionContextManager


def test_create_prediction_context_with_forecasts_short_cycle():
    """Test creating a prediction context with forecast data (short cycle)."""
    features = {
        'temp_forecast_1h': 10, 'temp_forecast_2h': 11, 'temp_forecast_3h': 12, 'temp_forecast_4h': 13,
        'pv_forecast_1h': 100, 'pv_forecast_2h': 200, 'pv_forecast_3h': 300, 'pv_forecast_4h': 400,
    }
    thermal_features = {'fireplace_on': 1, 'tv_on': 0}

    # Test with 30 minute cycle
    with patch.object(src.prediction_context.config, 'CYCLE_INTERVAL_MINUTES', 30):
        context = UnifiedPredictionContext.create_prediction_context(features, 8, 50, thermal_features)
        
        assert context['use_forecasts'] is True
        
        # Expected: Linear interpolation for 30 mins (0.5h)
        # Weight = 0.5 (updated to align with trajectory optimizer)
        # Outdoor = 8 * 0.5 + 10 * 0.5 = 4 + 5 = 9.0
        assert context['avg_outdoor'] == 9.0
        
        # PV = 50 * 0.5 + 100 * 0.5 = 25 + 50 = 75.0
        assert context['avg_pv'] == 75.0


def test_create_prediction_context_with_forecasts_long_cycle():
    """Test creating a prediction context with forecast data (long cycle)."""
    features = {
        'temp_forecast_1h': 10, 'temp_forecast_2h': 11, 'temp_forecast_3h': 12, 'temp_forecast_4h': 13,
        'pv_forecast_1h': 100, 'pv_forecast_2h': 200, 'pv_forecast_3h': 300, 'pv_forecast_4h': 400,
    }
    thermal_features = {'fireplace_on': 1, 'tv_on': 0}
    
    # Test with 90 minute cycle (1.5 hours) -> Should use 1h forecast
    with patch.object(src.prediction_context.config, 'CYCLE_INTERVAL_MINUTES', 90):
        context = UnifiedPredictionContext.create_prediction_context(features, 8, 50, thermal_features)
        
        assert context['use_forecasts'] is True
        assert context['avg_outdoor'] == 10  # 1h forecast
        assert context['avg_pv'] == 100      # 1h forecast


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

    # Mock 30 min cycle for consistent results
    with patch.object(src.prediction_context.config, 'CYCLE_INTERVAL_MINUTES', 30):
        manager.create_context(8, 50, thermal_features)
        
        assert manager.get_context() is not None
        # 9.0 as calculated in short_cycle test (updated from 8.5)
        assert manager.get_thermal_model_params()['outdoor_temp'] == 9.0
        assert manager.get_forecast_arrays()[0][0] == 10
        assert manager.uses_forecasts() is True
