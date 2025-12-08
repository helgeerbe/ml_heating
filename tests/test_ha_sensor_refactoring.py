"""
Unit tests for HA sensor refactoring.

Tests the new clean sensor schema implementation that eliminates redundancy:
- ml_heating_learning: Learning confidence + thermal parameters only
- ml_model_mae: Enhanced with time-windowed attributes  
- ml_model_rmse: Enhanced with error distribution attributes
- ml_prediction_accuracy: 24h control quality (no redundant MAE/RMSE)
"""
import sys
import os
import unittest
from unittest.mock import Mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ha_client import HAClient


class TestHASensorRefactoring(unittest.TestCase):
    """Test cases for the refactored HA sensor schema."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ha_client = HAClient("http://test:8123", "test_token")
        
        # Mock the set_state method to capture calls
        self.ha_client.set_state = Mock()
        
        # Sample learning metrics for testing
        self.sample_metrics = {
            # Core thermal parameters
            "thermal_time_constant": 6.5,
            "heat_loss_coefficient": 0.08,
            "outlet_effectiveness": 0.12,
            "learning_confidence": 3.8,
            
            # Learning progress
            "cycle_count": 150,
            "parameter_updates": 25,
            "model_health": "good",
            "is_improving": True,
            "improvement_percentage": 12.5,
            "total_predictions": 1500,
            
            # MAE metrics
            "mae_1h": 0.15,
            "mae_6h": 0.18,
            "mae_24h": 0.22,
            "mae_all_time": 0.25,
            
            # RMSE metrics  
            "rmse_all_time": 0.35,
            "recent_max_error": 0.45,
            
            # Accuracy breakdowns
            "perfect_accuracy_pct": 65.0,
            "tolerable_accuracy_pct": 25.0,
            "poor_accuracy_pct": 10.0,
            "prediction_count_24h": 144,
            "excellent_accuracy_pct": 80.0,
            "good_accuracy_pct": 90.0,
            "good_control_pct": 85.0
        }
    
    def test_ml_heating_learning_sensor_attributes(self):
        """Test ml_heating_learning sensor has correct attributes."""
        self.ha_client.log_adaptive_learning_metrics(self.sample_metrics)
        
        # Find the call for ml_heating_learning sensor
        learning_call = None
        for call in self.ha_client.set_state.call_args_list:
            if call[0][0] == "sensor.ml_heating_learning":
                learning_call = call
                break
        
        self.assertIsNotNone(learning_call, 
                           "ml_heating_learning sensor should be called")
        
        # Check state is learning confidence
        self.assertEqual(learning_call[0][1], 3.8)
        
        # Check attributes (third positional argument)
        attrs = learning_call.args[2]
        
        # Should have thermal parameters
        self.assertEqual(attrs["thermal_time_constant"], 6.5)
        self.assertEqual(attrs["heat_loss_coefficient"], 0.08)
        self.assertEqual(attrs["outlet_effectiveness"], 0.12)
        
        # Should have learning progress
        self.assertEqual(attrs["cycle_count"], 150)
        self.assertEqual(attrs["parameter_updates"], 25)
        self.assertEqual(attrs["model_health"], "good")
        self.assertEqual(attrs["learning_progress"], 1.0)  # min(1.0, 150/100)
        self.assertEqual(attrs["is_improving"], True)
        self.assertEqual(attrs["improvement_percentage"], 12.5)
        self.assertEqual(attrs["total_predictions"], 1500)
        
        # Should NOT have learning_confidence attribute (redundant with state)
        self.assertNotIn("learning_confidence", attrs)
        
        # Should NOT have MAE/RMSE attributes (moved to dedicated sensors)
        self.assertNotIn("mae_1h", attrs)
        self.assertNotIn("mae_all_time", attrs)
        self.assertNotIn("rmse_all_time", attrs)
    
    def test_ml_model_mae_sensor_attributes(self):
        """Test ml_model_mae sensor has enhanced time-windowed attributes."""
        self.ha_client.log_adaptive_learning_metrics(self.sample_metrics)
        
        # Find the call for ml_model_mae sensor
        mae_call = None
        for call in self.ha_client.set_state.call_args_list:
            if call[0][0] == "sensor.ml_model_mae":
                mae_call = call
                break
        
        self.assertIsNotNone(mae_call, "ml_model_mae sensor should be called")
        
        # Check state is all-time MAE
        self.assertEqual(mae_call[0][1], 0.25)
        
        # Check enhanced attributes
        attrs = mae_call.args[2]
        
        # Should have time-windowed MAE
        self.assertEqual(attrs["mae_1h"], 0.15)
        self.assertEqual(attrs["mae_6h"], 0.18)
        self.assertEqual(attrs["mae_24h"], 0.22)
        
        # Should have trend direction
        self.assertEqual(attrs["trend_direction"], "improving")
        self.assertEqual(attrs["prediction_count"], 1500)
        self.assertIn("last_updated", attrs)
    
    def test_ml_model_rmse_sensor_attributes(self):
        """Test ml_model_rmse sensor has enhanced error distribution attributes."""
        self.ha_client.log_adaptive_learning_metrics(self.sample_metrics)
        
        # Find the call for ml_model_rmse sensor
        rmse_call = None
        for call in self.ha_client.set_state.call_args_list:
            if call[0][0] == "sensor.ml_model_rmse":
                rmse_call = call
                break
        
        self.assertIsNotNone(rmse_call, "ml_model_rmse sensor should be called")
        
        # Check state is all-time RMSE
        self.assertEqual(rmse_call[0][1], 0.35)
        
        # Check enhanced attributes
        attrs = rmse_call.args[2]
        
        # Should have error distribution metrics
        self.assertEqual(attrs["recent_max_error"], 0.45)
        self.assertIn("std_error", attrs)  # Calculated from MAE/RMSE
        self.assertIn("mean_bias", attrs)  # Placeholder for now
        self.assertEqual(attrs["prediction_count"], 1500)
        self.assertIn("last_updated", attrs)
    
    def test_ml_prediction_accuracy_sensor_attributes(self):
        """Test ml_prediction_accuracy sensor has clean 24h control quality."""
        self.ha_client.log_adaptive_learning_metrics(self.sample_metrics)
        
        # Find the call for ml_prediction_accuracy sensor
        accuracy_call = None
        for call in self.ha_client.set_state.call_args_list:
            if call[0][0] == "sensor.ml_prediction_accuracy":
                accuracy_call = call
                break
        
        self.assertIsNotNone(accuracy_call, 
                           "ml_prediction_accuracy sensor should be called")
        
        # Check state is good control percentage (24h)
        self.assertEqual(accuracy_call[0][1], 85.0)
        
        # Check clean attributes
        attrs = accuracy_call.args[2]
        
        # Should have 24h breakdown
        self.assertEqual(attrs["perfect_accuracy_pct"], 65.0)
        self.assertEqual(attrs["tolerable_accuracy_pct"], 25.0)
        self.assertEqual(attrs["poor_accuracy_pct"], 10.0)
        self.assertEqual(attrs["prediction_count_24h"], 144)
        
        # Should have all-time reference
        self.assertEqual(attrs["excellent_all_time_pct"], 80.0)
        self.assertEqual(attrs["good_all_time_pct"], 90.0)
        
        # Should NOT have redundant MAE/RMSE attributes
        self.assertNotIn("mae_current", attrs)
        self.assertNotIn("rmse_current", attrs)
        
        # Should NOT have redundant good_accuracy_pct attribute (same as state)
        self.assertNotIn("good_accuracy_pct", attrs)
    
    def test_mae_trend_calculation(self):
        """Test MAE trend direction calculation."""
        # Test improving trend
        metrics_improving = {"improvement_percentage": 10.0}
        trend = self.ha_client._get_mae_trend(metrics_improving)
        self.assertEqual(trend, "improving")
        
        # Test degrading trend
        metrics_degrading = {"improvement_percentage": -10.0}
        trend = self.ha_client._get_mae_trend(metrics_degrading)
        self.assertEqual(trend, "degrading")
        
        # Test stable trend
        metrics_stable = {"improvement_percentage": 2.0}
        trend = self.ha_client._get_mae_trend(metrics_stable)
        self.assertEqual(trend, "stable")
    
    def test_std_error_calculation(self):
        """Test standard deviation error calculation from MAE/RMSE."""
        metrics = {"mae_all_time": 0.2, "rmse_all_time": 0.3}
        std_error = self.ha_client._calculate_std_error(metrics)
        
        # std_error = sqrt(rmse^2 - mae^2) = sqrt(0.09 - 0.04) = sqrt(0.05)
        expected = round((0.09 - 0.04)**0.5, 4)
        self.assertEqual(std_error, expected)
    
    def test_no_redundant_attributes(self):
        """Test that redundant attributes are eliminated."""
        self.ha_client.log_adaptive_learning_metrics(self.sample_metrics)
        
        all_calls = {call[0][0]: call.args[2]
                     for call in self.ha_client.set_state.call_args_list}
        
        # ml_heating_learning should NOT have learning_confidence attribute
        learning_attrs = all_calls["sensor.ml_heating_learning"]
        self.assertNotIn("learning_confidence", learning_attrs)
        
        # ml_prediction_accuracy should NOT have good_accuracy_pct attribute
        accuracy_attrs = all_calls["sensor.ml_prediction_accuracy"]
        self.assertNotIn("good_accuracy_pct", accuracy_attrs)
        
        # Only ml_heating_learning should have model_health
        mae_attrs = all_calls["sensor.ml_model_mae"]
        rmse_attrs = all_calls["sensor.ml_model_rmse"]
        self.assertNotIn("model_health", mae_attrs)
        self.assertNotIn("model_health", rmse_attrs)
        self.assertNotIn("model_health", accuracy_attrs)
        self.assertIn("model_health", learning_attrs)
    
    def test_sensor_state_values(self):
        """Test that each sensor has the correct state value."""
        self.ha_client.log_adaptive_learning_metrics(self.sample_metrics)
        
        # Extract state values from calls
        state_values = {}
        for call in self.ha_client.set_state.call_args_list:
            entity_id = call[0][0]
            state_value = call[0][1]
            state_values[entity_id] = state_value
        
        # Check each sensor has correct state
        self.assertEqual(state_values["sensor.ml_heating_learning"], 3.8)
        self.assertEqual(state_values["sensor.ml_model_mae"], 0.25)
        self.assertEqual(state_values["sensor.ml_model_rmse"], 0.35)
        self.assertEqual(state_values["sensor.ml_prediction_accuracy"], 85.0)
    
    def test_learning_progress_calculation(self):
        """Test learning progress calculation."""
        # Test early learning phase
        early_metrics = self.sample_metrics.copy()
        early_metrics["cycle_count"] = 30
        
        self.ha_client.log_adaptive_learning_metrics(early_metrics)
        
        learning_call = None
        for call in self.ha_client.set_state.call_args_list:
            if call[0][0] == "sensor.ml_heating_learning":
                learning_call = call
                break
        
        attrs = learning_call.args[2]
        self.assertEqual(attrs["learning_progress"], 0.3)  # 30/100
        
        # Test mature model (should cap at 1.0)
        self.ha_client.set_state.reset_mock()
        mature_metrics = self.sample_metrics.copy()
        mature_metrics["cycle_count"] = 200
        
        self.ha_client.log_adaptive_learning_metrics(mature_metrics)
        
        learning_call = None
        for call in self.ha_client.set_state.call_args_list:
            if call[0][0] == "sensor.ml_heating_learning":
                learning_call = call
                break
        
        attrs = learning_call.args[2]
        self.assertEqual(attrs["learning_progress"], 1.0)  # min(1.0, 200/100)
    
    def test_all_sensors_called(self):
        """Test that all 4 sensors are called with refactored implementation."""
        self.ha_client.log_adaptive_learning_metrics(self.sample_metrics)
        
        called_sensors = {call[0][0] for call in self.ha_client.set_state.call_args_list}
        
        expected_sensors = {
            "sensor.ml_heating_learning",
            "sensor.ml_model_mae", 
            "sensor.ml_model_rmse",
            "sensor.ml_prediction_accuracy"
        }
        
        self.assertEqual(called_sensors, expected_sensors)
        self.assertEqual(len(self.ha_client.set_state.call_args_list), 4)


if __name__ == '__main__':
    unittest.main()
