"""
Test enhanced trajectory prediction with thermal momentum.
"""
import unittest
from unittest.mock import Mock
import pandas as pd
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from enhanced_trajectory import (
        predict_thermal_trajectory_enhanced,
        evaluate_trajectory_stability_enhanced,
        apply_thermal_momentum_correction
    )
    from physics_model import RealisticPhysicsModel
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class TestEnhancedTrajectory(unittest.TestCase):
    """Test enhanced trajectory prediction functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock physics model
        self.mock_model = Mock(spec=RealisticPhysicsModel)
        self.mock_model.predict_one.return_value = 0.1  # Mild heating
        
        # Sample features with thermal momentum
        self.sample_features = pd.DataFrame([{
            'outlet_temp': 35.0,
            'indoor_temp_lag_30m': 21.0,
            'target_temp': 21.5,
            'outdoor_temp': 10.0,
            'temp_forecast_1h': 9.0,
            'temp_forecast_2h': 8.0,
            'temp_forecast_3h': 7.0,
            'temp_forecast_4h': 6.0,
            'pv_forecast_1h': 0.0,
            'pv_forecast_2h': 0.0,
            'pv_forecast_3h': 0.0,
            'pv_forecast_4h': 0.0,
            'indoor_temp_gradient': 0.2,  # Mild warming trend
            'temp_diff_indoor_outdoor': 11.0,
            'outlet_indoor_diff': 14.0,
            'indoor_temp_delta_10m': 0.1,
            'indoor_temp_delta_30m': 0.3,
        }])

    def test_enhanced_trajectory_prediction(self):
        """Test enhanced trajectory prediction function."""
        trajectory = predict_thermal_trajectory_enhanced(
            self.mock_model, 
            self.sample_features, 
            outlet_temp=35.0, 
            steps=4
        )
        
        # Should return 4 temperature predictions
        self.assertEqual(len(trajectory), 4)
        
        # All temperatures should be reasonable (15-25°C range)
        for temp in trajectory:
            self.assertGreater(temp, 15.0)
            self.assertLess(temp, 25.0)
        
        # Model should be called 4 times (once per step)
        self.assertEqual(self.mock_model.predict_one.call_count, 4)

    def test_trajectory_stability_evaluation(self):
        """Test enhanced trajectory stability evaluation."""
        # Test stable trajectory (gradually approaching target)
        stable_trajectory = [21.1, 21.2, 21.3, 21.4]
        target_temp = 21.5
        current_temp = 21.0
        
        thermal_momentum = {
            'indoor_temp_gradient': 0.1,  # Gentle warming
            'temp_diff_indoor_outdoor': 11.0
        }
        
        stability_score = evaluate_trajectory_stability_enhanced(
            stable_trajectory, target_temp, current_temp, thermal_momentum
        )
        
        # Should get a reasonable stability score
        self.assertIsInstance(stability_score, float)
        self.assertGreater(stability_score, 0.0)
        self.assertLess(stability_score, 20.0)  # Should be reasonable

    def test_oscillating_trajectory_penalty(self):
        """Test that oscillating trajectory gets penalized."""
        # Oscillating trajectory
        oscillating = [21.0, 21.6, 21.2, 21.8]
        stable = [21.1, 21.2, 21.3, 21.4]
        target_temp = 21.5
        
        oscillating_score = evaluate_trajectory_stability_enhanced(
            oscillating, target_temp
        )
        stable_score = evaluate_trajectory_stability_enhanced(
            stable, target_temp
        )
        
        # Oscillating should have higher (worse) score
        self.assertGreater(oscillating_score, stable_score)

    def test_thermal_momentum_correction(self):
        """Test thermal momentum correction function."""
        thermal_momentum = {
            'indoor_temp_gradient': 0.5,  # Strong warming
            'temp_diff_indoor_outdoor': 10.0
        }
        
        # Test prediction opposing strong momentum
        opposing_delta = -0.3  # Cooling prediction
        corrected = apply_thermal_momentum_correction(
            opposing_delta, thermal_momentum, 21.5, 21.0
        )
        
        # Should reduce the opposing correction
        self.assertGreater(corrected, opposing_delta)  # Less negative
        self.assertLess(abs(corrected), abs(opposing_delta))

    def test_empty_trajectory_handling(self):
        """Test handling of empty trajectory."""
        score = evaluate_trajectory_stability_enhanced(
            [], target_temp=21.5
        )
        
        # Should return infinity for empty trajectory
        self.assertEqual(score, float('inf'))

    def test_thermal_momentum_correction_no_momentum(self):
        """Test momentum correction with no momentum data."""
        corrected = apply_thermal_momentum_correction(
            0.2, None, 21.5, 21.0
        )
        
        # Should return original prediction when no momentum
        self.assertEqual(corrected, 0.2)


if __name__ == '__main__':
    unittest.main()
