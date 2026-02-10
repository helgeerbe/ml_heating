
import pytest
from src.forecast_analytics import (
    analyze_forecast_quality,
    calculate_thermal_forecast_impact,
    get_forecast_fallback_strategy,
    calculate_forecast_accuracy_metrics,
)


@pytest.mark.parametrize(
    "weather_forecasts, pv_forecasts, expected",
    [
        ([10, 11, 12, 13], [100, 200, 300, 400], {"overall_confidence": 1.0}),
        ([None, 10, 12, 13], [None, 200, 300, 400], {"weather_availability": 0.75, "pv_availability": 0.75}),
        ([50, 10, 12, 13], [16000, 200, 300, 400], {"weather_confidence": 0.75, "pv_confidence": 0.75}),
    ],
)
def test_analyze_forecast_quality(weather_forecasts, pv_forecasts, expected):
    """Test the analysis of forecast data quality."""
    result = analyze_forecast_quality(weather_forecasts, pv_forecasts)
    for key, value in expected.items():
        assert result[key] == pytest.approx(value)


@pytest.mark.parametrize(
    "temp_forecasts, pv_forecasts, outdoor_temp, pv_power, expected_keys",
    [
        ([10, 9, 8, 7], [100, 50, 20, 10], 12, 120, ["weather_cooling_trend"]),
        ([10, 11, 12, 13], [100, 150, 200, 250], 9, 80, ["weather_heating_trend", "pv_warming_trend"]),
    ],
)
def test_calculate_thermal_forecast_impact(temp_forecasts, pv_forecasts, outdoor_temp, pv_power, expected_keys):
    """Test the calculation of the thermal impact of forecasts."""
    result = calculate_thermal_forecast_impact(temp_forecasts, pv_forecasts, outdoor_temp, pv_power)
    for key in expected_keys:
        assert result[key] > 0


def test_get_forecast_fallback_strategy():
    """Test the forecast fallback strategy."""
    quality_metrics = {"overall_confidence": 0.4}
    current_conditions = {"outdoor_temp": 5.0, "pv_now": 50.0}
    result = get_forecast_fallback_strategy(quality_metrics, current_conditions)
    assert result["fallback_reason"] == "low_confidence"


@pytest.mark.parametrize(
    "predicted, actual, forecast_type, expected_mae, expected_score",
    [
        ([10, 11, 12], [10.5, 11.5, 12.5], "temperature", 0.5, 0.75),
        ([100, 200, 300], [150, 250, 350], "pv", 50, 0.9),
    ],
)
def test_calculate_forecast_accuracy_metrics(predicted, actual, forecast_type, expected_mae, expected_score):
    """Test the calculation of forecast accuracy metrics."""
    result = calculate_forecast_accuracy_metrics(predicted, actual, forecast_type)
    assert result["mae"] == pytest.approx(expected_mae)
    assert result["accuracy_score"] == pytest.approx(expected_score)

