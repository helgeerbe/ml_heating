"""
Test forecast analytics module for Week 4.
"""
import unittest
from unittest.mock import patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from forecast_analytics import (
        analyze_forecast_quality,
        calculate_thermal_forecast_impact,
        get_forecast_fallback_strategy,
        calculate_forecast_accuracy_metrics
    )
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class TestForecastAnalytics(unittest.TestCase):
    """Test forecast analytics functionality."""

    def test_analyze_forecast_quality_good_forecasts(self):
        """Test forecast quality analysis with good forecasts."""
        weather_forecasts = [10.0, 8.0, 6.0, 4.0]  # Valid temperatures
        pv_forecasts = [1000.0, 800.0, 600.0, 400.0]  # Valid PV values
        
        quality = analyze_forecast_quality(weather_forecasts, pv_forecasts)
        
        # Should have high availability and confidence
        self.assertEqual(quality['weather_availability'], 1.0)
        self.assertEqual(quality['pv_availability'], 1.0) 
        self.assertEqual(quality['combined_availability'], 1.0)
        self.assertEqual(quality['weather_confidence'], 1.0)
        self.assertEqual(quality['pv_confidence'], 1.0)
        self.assertEqual(quality['overall_confidence'], 1.0)

    def test_analyze_forecast_quality_poor_forecasts(self):
        """Test forecast quality analysis with poor forecasts."""
        weather_forecasts = [None, 0.0, -50.0, 100.0]  # Poor quality
        pv_forecasts = [None, -100.0, 20000.0, 0.0]  # Poor quality
        
        quality = analyze_forecast_quality(weather_forecasts, pv_forecasts)
        
        # Should have low availability and confidence
        self.assertLessEqual(quality['weather_availability'], 0.5)
        self.assertLess(quality['pv_availability'], 0.8)
        self.assertLessEqual(quality['combined_availability'], 0.65)
        self.assertLessEqual(quality['overall_confidence'], 0.5)

    def test_thermal_forecast_impact_cooling_trend(self):
        """Test thermal impact calculation with cooling trend."""
        temp_forecasts = [10.0, 8.0, 6.0, 4.0]  # Cooling trend
        pv_forecasts = [1000.0, 800.0, 600.0, 200.0]  # Decreasing PV
        current_outdoor_temp = 12.0
        current_pv_power = 1200.0
        
        impact = calculate_thermal_forecast_impact(
            temp_forecasts, pv_forecasts, current_outdoor_temp, current_pv_power
        )
        
        # Should detect cooling trend (increases heating demand)
        self.assertGreater(impact['weather_cooling_trend'], 0.0)
        self.assertEqual(impact['weather_heating_trend'], 0.0)
        self.assertGreater(impact['net_thermal_trend'], 0.0)  # Net heating needed
        self.assertGreater(impact['thermal_load_forecast'], 0.0)

    def test_thermal_forecast_impact_warming_trend(self):
        """Test thermal impact calculation with warming trend."""
        temp_forecasts = [10.0, 12.0, 14.0, 16.0]  # Warming trend
        pv_forecasts = [500.0, 800.0, 1200.0, 1500.0]  # Increasing PV
        current_outdoor_temp = 8.0
        current_pv_power = 300.0
        
        impact = calculate_thermal_forecast_impact(
            temp_forecasts, pv_forecasts, current_outdoor_temp, current_pv_power
        )
        
        # Should detect warming trend (reduces heating demand)
        self.assertEqual(impact['weather_cooling_trend'], 0.0)
        self.assertGreater(impact['weather_heating_trend'], 0.0)
        self.assertGreater(impact['pv_warming_trend'], 0.0)
        self.assertLess(impact['net_thermal_trend'], 0.0)  # Net cooling effect

    def test_fallback_strategy_low_confidence(self):
        """Test fallback strategy for low confidence forecasts."""
        quality_metrics = {
            'overall_confidence': 0.3,  # Low confidence
            'combined_availability': 0.8  # Good availability
        }
        current_conditions = {
            'outdoor_temp': 15.0,
            'pv_now': 500.0
        }
        
        fallback = get_forecast_fallback_strategy(quality_metrics, current_conditions)
        
        # Should use conservative fallback
        self.assertEqual(fallback['fallback_reason'], 'low_confidence')
        # All temperatures should be current temp (conservative)
        for i in range(1, 5):
            self.assertEqual(fallback[f'temp_forecast_{i}h'], 15.0)

    @patch('forecast_analytics.datetime')
    def test_fallback_strategy_low_availability(self, mock_datetime):
        """Test fallback strategy for low availability forecasts."""
        # Mock current time as 14:00 (daytime)
        mock_datetime.now.return_value.hour = 14
        
        quality_metrics = {
            'overall_confidence': 0.8,  # Good confidence
            'combined_availability': 0.3  # Low availability
        }
        current_conditions = {
            'outdoor_temp': 20.0,
            'pv_now': 1000.0
        }
        
        fallback = get_forecast_fallback_strategy(quality_metrics, current_conditions)
        
        # Should use seasonal trend fallback
        self.assertEqual(fallback['fallback_reason'], 'low_availability')
        # Should have some temperature variations
        temps = [fallback[f'temp_forecast_{i}h'] for i in range(1, 5)]
        self.assertTrue(any(t != 20.0 for t in temps))  # Some variation expected

    def test_forecast_accuracy_metrics(self):
        """Test forecast accuracy calculation."""
        predicted = [10.0, 11.0, 12.0, 13.0]
        actual = [10.5, 11.2, 11.8, 12.9]  # Small errors
        
        accuracy = calculate_forecast_accuracy_metrics(
            predicted, actual, "temperature"
        )
        
        # Should have reasonable accuracy
        self.assertGreater(accuracy['accuracy_score'], 0.8)  # Good accuracy
        self.assertLess(accuracy['mae'], 1.0)  # Small error
        self.assertEqual(accuracy['sample_size'], 4)

    def test_forecast_accuracy_metrics_poor_prediction(self):
        """Test forecast accuracy with poor predictions."""
        predicted = [10.0, 11.0, 12.0, 13.0]
        actual = [15.0, 16.0, 17.0, 18.0]  # Large errors
        
        accuracy = calculate_forecast_accuracy_metrics(
            predicted, actual, "temperature"
        )
        
        # Should have poor accuracy
        self.assertLess(accuracy['accuracy_score'], 0.5)  # Poor accuracy
        self.assertGreater(accuracy['mae'], 4.0)  # Large error

    def test_empty_forecasts_handling(self):
        """Test handling of empty or None forecasts."""
        # Empty forecasts
        quality = analyze_forecast_quality([], [])
        self.assertEqual(quality['combined_availability'], 0.0)
        
        # Thermal impact with empty forecasts
        impact = calculate_thermal_forecast_impact([], [], 10.0)
        self.assertEqual(impact['net_thermal_trend'], 0.0)
        
        # Accuracy with mismatched lengths
        accuracy = calculate_forecast_accuracy_metrics([1, 2], [1])
        self.assertEqual(accuracy['sample_size'], 0)


if __name__ == '__main__':
    unittest.main()
