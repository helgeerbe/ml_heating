"""
Unit tests for trajectory prediction functionality.
"""
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_wrapper import predict_thermal_trajectory
from physics_model import RealisticPhysicsModel


class TestTrajectoryPrediction(unittest.TestCase):
    """Test cases for trajectory prediction function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = RealisticPhysicsModel()
        
        # Create mock features similar to production scenario
        self.features_dict = {
            'indoor_temp_lag_30m': 20.8,
            'outdoor_temp': 5.0,
            'outlet_temp': 46.0,
            'outlet_temp_sq': 46.0 ** 2,
            'outlet_temp_cub': 46.0 ** 3,
            'outlet_temp_change_from_last': 0.0,
            'outlet_indoor_diff': 25.2,
            'outdoor_temp_x_outlet_temp': 5.0 * 46.0,
            'target_temp': 21.0,
            'dhw_heating': 0.0,
            'defrosting': 0.0,
            'dhw_disinfection': 0.0,
            'dhw_boost_heater': 0.0,
            'fireplace_on': 0.0,
            'pv_now': 190.5,
            'tv_on': 0.0,
            'temp_forecast_1h': 6.5,
            'temp_forecast_2h': 6.4,
            'temp_forecast_3h': 6.1,
            'temp_forecast_4h': 5.7,
            'pv_forecast_1h': 0.0,
            'pv_forecast_2h': 0.0,
            'pv_forecast_3h': 0.0,
            'pv_forecast_4h': 0.0,
        }
        
        self.features_df = pd.DataFrame([self.features_dict])
    
    def test_different_outlet_temps_produce_different_trajectories(self):
        """Test that different outlet temperatures produce different trajectories."""
        # Test with low and high outlet temperatures
        traj_low = predict_thermal_trajectory(self.model, self.features_df, 20.0, steps=4)
        traj_high = predict_thermal_trajectory(self.model, self.features_df, 65.0, steps=4)
        
        # Trajectories should be different
        self.assertNotEqual(traj_low, traj_high, 
                           "Different outlet temperatures should produce different trajectories")
        
        # Both should have 4 steps
        self.assertEqual(len(traj_low), 4, "Trajectory should have 4 steps")
        self.assertEqual(len(traj_high), 4, "Trajectory should have 4 steps")
        
        # All values should be finite numbers
        for temp in traj_low + traj_high:
            self.assertTrue(np.isfinite(temp), "All trajectory temperatures should be finite")
    
    def test_trajectory_length_matches_steps(self):
        """Test that trajectory length matches requested steps."""
        for steps in [1, 2, 3, 4, 6]:
            trajectory = predict_thermal_trajectory(self.model, self.features_df, 35.0, steps=steps)
            self.assertEqual(len(trajectory), steps, 
                           f"Trajectory should have {steps} steps but has {len(trajectory)}")
    
    def test_trajectory_with_zero_steps(self):
        """Test trajectory prediction with zero steps."""
        trajectory = predict_thermal_trajectory(self.model, self.features_df, 35.0, steps=0)
        self.assertEqual(len(trajectory), 0, "Zero steps should produce empty trajectory")
    
    def test_outlet_temp_variation_affects_trajectory(self):
        """Test that varying outlet temperature produces varying trajectories."""
        test_temps = [20.0, 30.0, 40.0, 50.0, 60.0]
        trajectories = []
        
        for outlet_temp in test_temps:
            trajectory = predict_thermal_trajectory(self.model, self.features_df, outlet_temp, steps=4)
            trajectories.append(trajectory)
        
        # Check that not all trajectories are identical
        first_trajectory = trajectories[0]
        all_identical = all(traj == first_trajectory for traj in trajectories)
        
        self.assertFalse(all_identical, 
                        "Not all outlet temperatures should produce identical trajectories")
    
    def test_trajectory_values_are_reasonable(self):
        """Test that trajectory values are within reasonable temperature ranges."""
        trajectory = predict_thermal_trajectory(self.model, self.features_df, 45.0, steps=4)
        
        # Indoor temperatures should be reasonable (e.g., 15-30°C)
        for temp in trajectory:
            self.assertGreaterEqual(temp, 10.0, "Indoor temperature should be >= 10°C")
            self.assertLessEqual(temp, 35.0, "Indoor temperature should be <= 35°C")
            self.assertTrue(np.isfinite(temp), "Temperature should be finite")
    
    def test_features_df_not_modified(self):
        """Test that the original features DataFrame is not modified."""
        original_features = self.features_df.copy()
        
        predict_thermal_trajectory(self.model, self.features_df, 45.0, steps=4)
        
        # Features should be unchanged
        pd.testing.assert_frame_equal(self.features_df, original_features,
                                    "Original features DataFrame should not be modified")


if __name__ == '__main__':
    unittest.main()
