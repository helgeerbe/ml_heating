#!/usr/bin/env python3
"""
Unit tests for cycle-time-based forecast alignment functionality.

Tests the timing alignment between control cycles and forecast horizons to
prevent evening/overnight over-heating predictions due to forecast timing
mismatch.
"""
import sys
import unittest
from unittest.mock import patch

# Add src to path for testing
sys.path.insert(0, 'src')

from model_wrapper import EnhancedModelWrapper


class TestCycleTimeForecastAlignment(unittest.TestCase):
    """Test cycle-time forecast alignment functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.wrapper = EnhancedModelWrapper()
        
        # Mock features with forecast data
        self.test_features = {
            'indoor_temp_lag_30m': 20.0,
            'target_temp': 21.0,
            'outdoor_temp': 2.0,
            'pv_now': 0.0,
            'fireplace_on': 0,
            'tv_on': 0,
            # Forecast data showing temperature drop (typical evening scenario)
            'temp_forecast_1h': 1.5,
            'temp_forecast_2h': 1.0,
            'temp_forecast_3h': 0.5,
            'temp_forecast_4h': 0.0,
            'pv_forecast_1h': 0.0,
            'pv_forecast_2h': 0.0,
            'pv_forecast_3h': 0.0,
            'pv_forecast_4h': 0.0,
        }
    
    def test_cycle_time_validation_within_limit(self):
        """Test that normal cycle times are accepted."""
        # Test 30min cycle (normal)
        with patch('config.CYCLE_INTERVAL_MINUTES', 30):
            outdoor, pv, _, _ = self.wrapper._get_forecast_conditions(
                2.0, 0.0, {'pv_power': 0.0}
            )
            # Should not raise any warnings
            self.assertIsInstance(outdoor, float)
            self.assertIsInstance(pv, float)

    def test_cycle_time_validation_exceeds_limit(self):
        """Test that excessive cycle times are capped at 180min."""
        # Test 240min cycle (excessive - should be capped)
        with patch('config.CYCLE_INTERVAL_MINUTES', 240):
            with patch('model_wrapper.logging') as mock_logging:
                outdoor, pv, _, _ = self.wrapper._get_forecast_conditions(
                    2.0, 0.0, {'pv_power': 0.0}
                )
                # Should log a warning
                mock_logging.warning.assert_called()
                warning_msg = mock_logging.warning.call_args[0][0]
                self.assertIn("240min exceeds", warning_msg)
                self.assertIn("180min", warning_msg)

    def test_30min_cycle_interpolation(self):
        """Test forecast interpolation for 30min cycles."""
        # Set up features with forecast data
        self.wrapper._current_features = self.test_features
        
        with patch('config.CYCLE_INTERVAL_MINUTES', 30):
            outdoor, pv, _, _ = self.wrapper._get_forecast_conditions(
                2.0, 0.0, {'pv_power': 0.0}
            )
            
            # For 30min cycle (0.5h), should interpolate between current (2.0)
            # and 1h forecast (1.5)
            # Expected: 2.0 + (1.5 - 2.0) * 0.5 = 1.75
            expected_outdoor = 2.0 + (1.5 - 2.0) * 0.5
            self.assertAlmostEqual(outdoor, expected_outdoor, places=2)

    def test_60min_cycle_uses_1h_forecast(self):
        """Test that 60min cycles use 1h forecast directly."""
        self.wrapper._current_features = self.test_features
        
        with patch('config.CYCLE_INTERVAL_MINUTES', 60):
            outdoor, pv, _, _ = self.wrapper._get_forecast_conditions(
                2.0, 0.0, {'pv_power': 0.0}
            )
            
            # For 60min cycle (1.0h), uses 1h forecast directly
            self.assertEqual(outdoor, 1.5)

    def test_120min_cycle_2h_forecast(self):
        """Test 2h forecast usage for 120min cycles."""
        self.wrapper._current_features = self.test_features
        
        with patch('config.CYCLE_INTERVAL_MINUTES', 120):
            outdoor, pv, _, _ = self.wrapper._get_forecast_conditions(
                2.0, 0.0, {'pv_power': 0.0}
            )
            
            # For 120min cycle (2.0h), should use 2h forecast directly
            self.assertEqual(outdoor, 1.0)

    def test_trajectory_interpolation_sub_hour(self):
        """Test trajectory interpolation for sub-hour cycle checking."""
        # Mock trajectory with temperature progression
        mock_trajectory = {
            'trajectory': [20.8, 21.2, 21.5, 22.0],  # Temps at 1h, 2h, 3h, 4h
            'reaches_target_at': None
        }
        
        current_indoor = 20.0
        target_indoor = 21.0
        cycle_minutes = 30
        cycle_hours = 0.5
        
        # Test interpolation calculation
        future_temp_1h = mock_trajectory['trajectory'][0]  # 20.8
        interpolated_temp = current_indoor + (future_temp_1h - current_indoor) * cycle_hours
        # Expected: 20.0 + (20.8 - 20.0) * 0.5 = 20.4
        expected_temp = 20.4
        
        self.assertAlmostEqual(interpolated_temp, expected_temp, places=1)
        
        # Check if target would be reached (20.4 vs 21.0 target)
        target_reached = abs(interpolated_temp - target_indoor) <= 0.1
        self.assertFalse(target_reached)  # 20.4 is not close enough to 21.0

    def test_no_forecast_data_fallback(self):
        """Test fallback to current conditions when no forecast data."""
        # No forecast features
        self.wrapper._current_features = {}
        
        with patch('config.CYCLE_INTERVAL_MINUTES', 30):
            outdoor, pv, _, _ = self.wrapper._get_forecast_conditions(
                2.0, 0.0, {'pv_power': 0.0}
            )
            
            # Should fallback to current values
            self.assertEqual(outdoor, 2.0)
            self.assertEqual(pv, 0.0)

    def test_multi_horizon_logging_with_cycle_aligned(self):
        """Test that multi-horizon logging includes cycle-aligned prediction."""
        self.wrapper._current_features = self.test_features
        
        with patch('config.CYCLE_INTERVAL_MINUTES', 30):
            with patch('model_wrapper.logging') as mock_logging:
                # Call the multi-horizon logging method
                thermal_features = {
                    'pv_power': 0.0, 'fireplace_on': 0, 'tv_on': 0
                }
                self.wrapper._log_multi_horizon_predictions(
                    current_indoor=20.0,
                    target_indoor=21.0,
                    outdoor_temp=2.0,
                    thermal_features=thermal_features
                )
                
                # Check that cycle-aligned prediction is logged
                logged_calls = [call[0][0] for call in 
                              mock_logging.info.call_args_list]
                cycle_logged = any("cycle(30min)" in call for call in 
                                 logged_calls)
                self.assertTrue(cycle_logged, 
                              "Cycle-aligned prediction should be logged")


if __name__ == "__main__":
    unittest.main()
