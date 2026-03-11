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
        # Weight = 0.5 / 2.0 = 0.25 (midpoint of cycle)
        # Outdoor = 8 * 0.75 + 10 * 0.25 = 6 + 2.5 = 8.5
        assert context['avg_outdoor'] == 8.5
        
        # PV = 50 (Step interpolation / zero-order hold for conservative estimate)
        assert context['avg_pv'] == 50


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
        # 8.5 as calculated in short_cycle test
        assert manager.get_thermal_model_params()['outdoor_temp'] == 8.5
        # The forecast array now includes current temp at index 0
        assert manager.get_forecast_arrays()[0][0] == 8
        assert manager.uses_forecasts() is True


def test_morning_drop_prevention():
    """
    Regression test for 'morning drop' issue.
    
    Ensures that for short cycles (e.g. 30 mins), the forecast blending weight
    targets the cycle midpoint (15 mins) rather than a hardcoded 0.5 (30 mins).
    
    Scenario:
    - Current Outdoor: 6.0°C
    - 1h Forecast: 10.0°C (Rising temperature, e.g. sunrise)
    - Cycle: 30 mins (0.5h)
    
    Calculation:
    - Midpoint = 15 mins = 0.25h
    - Weight = 0.25 / 1.0 = 0.25
    - Effective Temp = 6.0 * (1 - 0.25) + 10.0 * 0.25
                     = 6.0 * 0.75 + 2.5
                     = 4.5 + 2.5 = 7.0°C
                     
    (Buggy behavior would use weight 0.5 -> 8.0°C)
    """
    features = {
        'temp_forecast_1h': 10.0,
        'temp_forecast_2h': 12.0,
        'temp_forecast_3h': 14.0,
        'temp_forecast_4h': 15.0,
        'pv_forecast_1h': 100,
        'pv_forecast_2h': 200,
        'pv_forecast_3h': 300,
        'pv_forecast_4h': 400,
    }
    thermal_features = {'fireplace_on': 0, 'tv_on': 0}
    current_outdoor = 6.0
    
    with patch.object(src.prediction_context.config, 'CYCLE_INTERVAL_MINUTES', 30):
        context = UnifiedPredictionContext.create_prediction_context(
            features,
            current_outdoor,
            0,
            thermal_features
        )
        
        expected_temp = 7.0
        assert context['avg_outdoor'] == expected_temp
