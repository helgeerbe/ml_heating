"""
Test suite for shadow mode online learning functionality.

Tests the core shadow mode physics learning and ML vs heat curve benchmarking system.
"""

import unittest
from unittest.mock import Mock, patch, call
import sys
import os
from datetime import datetime

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import with proper package structure
from src.thermal_equilibrium_model import ThermalEquilibriumModel
from src.temperature_control import OnlineLearning
from src.influx_service import get_influx_service


class TestShadowModePhysicsLearning(unittest.TestCase):
    """Test shadow mode physics learning logic."""
    
    def setUp(self):
        """Set up test environment."""
        self.thermal_model = ThermalEquilibriumModel()
        self.thermal_model.prediction_history = []
        self.thermal_model.parameter_history = []
        self.online_learning = OnlineLearning()
    
    @patch('src.config.SHADOW_MODE', True)
    def test_shadow_mode_learning_ignores_target(self):
        """Test that shadow mode learning proceeds without a target temperature."""
        prediction_context = {
            'outlet_temp': 45.0, 'outdoor_temp': 10.0, 'pv_power': 0.0,
            'fireplace_on': 0.0, 'tv_on': 0.0
        }
        
        try:
            self.thermal_model.update_prediction_feedback(
                predicted_temp=21.0, actual_temp=21.5,
                prediction_context=prediction_context,
                timestamp=datetime.now().isoformat()
            )
            self.assertEqual(len(self.thermal_model.prediction_history), 1)
        except KeyError as e:
            self.fail(f"Shadow mode learning failed due to missing target temp: {e}")
    
    @patch('src.config.SHADOW_MODE', True)
    @patch('src.temperature_control.get_enhanced_model_wrapper')
    def test_shadow_mode_uses_heat_curve_outlet(self, mock_get_wrapper):
        """Test that shadow mode learns from the actual heat curve outlet settings."""
        mock_wrapper = mock_get_wrapper.return_value
        heat_curve_outlet = 42.0
        
        learning_features = {
            'outdoor_temp': 8.0, 'pv_now': 100.0, 'fireplace_on': 0.0, 'tv_on': 0.0
        }
        
        self.online_learning._perform_online_learning(
            learning_features, heat_curve_outlet, 0.3, 21.3
        )
        
        mock_wrapper.learn_from_prediction_feedback.assert_called_once()
        call_args = mock_wrapper.learn_from_prediction_feedback.call_args
        self.assertEqual(call_args.kwargs['prediction_context']['outlet_temp'], heat_curve_outlet)


class TestShadowModeBenchmarking(unittest.TestCase):
    """Test the shadow mode benchmarking system."""
    
    def setUp(self):
        self.online_learning = OnlineLearning()
    
    @patch('src.temperature_control.get_enhanced_model_wrapper')
    def test_ml_prediction_calculation(self, mock_get_wrapper):
        """Test correct ML outlet prediction for a given target temperature."""
        mock_model = mock_get_wrapper.return_value
        mock_model.calculate_optimal_outlet_temp.return_value = {
            'optimal_outlet_temp': 43.5
        }
        
        context = {
            'outdoor_temp': 5.0, 'pv_power': 200.0, 
            'fireplace_on': 0.0, 'tv_on': 0.0
        }
        
        result = self.online_learning.calculate_ml_benchmark_prediction(
            22.0, 21.0, context
        )
        
        self.assertEqual(result, 43.5)
        mock_model.calculate_optimal_outlet_temp.assert_called_once_with(
            target_indoor=22.0, current_indoor=21.0, outdoor_temp=5.0,
            pv_power=200.0, fireplace_on=0.0, tv_on=0.0
        )
    
    @patch('src.temperature_control.logging.info')
    def test_benchmark_logging_format(self, mock_logging_info):
        """Test that benchmark comparison is logged in the correct format."""
        self.online_learning._log_shadow_mode_comparison(45.0, 42.0)
        
        expected_call = call(
            "🎯 Shadow Benchmark: ML would predict %.1f°C, Heat Curve set %.1f°C (difference: %+.1f°C)",
            42.0, 45.0, 3.0
        )
        mock_logging_info.assert_has_calls([expected_call])
    
    @patch.object(OnlineLearning, '_export_shadow_benchmark_data')
    def test_efficiency_advantage_calculation(self, mock_export):
        """Test that the efficiency advantage is calculated and exported correctly."""
        self.online_learning._log_shadow_mode_comparison(45.0, 40.0)
        
        mock_export.assert_called_once()
        call_kwargs = mock_export.call_args.kwargs
        self.assertEqual(call_kwargs.get('efficiency_advantage'), 5.0)


class TestShadowModeCompleteCycle(unittest.TestCase):
    """Integration tests for complete shadow mode cycles."""
    
    def setUp(self):
        self.thermal_model = ThermalEquilibriumModel()
        self.thermal_model.prediction_history = []
    
    @patch('src.config.SHADOW_MODE', True)
    def test_fresh_startup_scenario(self):
        """Test that shadow mode works correctly on a fresh start."""
        prediction_context = {
            'outlet_temp': 40.0, 'outdoor_temp': 10.0, 'pv_power': 0.0,
            'fireplace_on': 0.0, 'tv_on': 0.0
        }
        
        self.thermal_model.update_prediction_feedback(
            predicted_temp=21.0, actual_temp=21.2,
            prediction_context=prediction_context,
            timestamp=datetime.now().isoformat()
        )
        self.assertEqual(len(self.thermal_model.prediction_history), 1)
    
    @patch.object(OnlineLearning, '_perform_online_learning')
    @patch.object(OnlineLearning, '_log_shadow_mode_comparison')
    def test_full_cycle_learning_and_benchmarking(self, mock_benchmark, mock_learning):
        """Test that a full cycle triggers both learning and benchmarking."""
        online_learning = OnlineLearning()
        state = {
            'last_run_features': {
                'outdoor_temp': 5.0, 'pv_now': 100.0, 
                'fireplace_on': 0.0, 'tv_on': 0.0
            },
            'last_indoor_temp': 21.0,
            'last_final_temp': 42.0
        }
        
        mock_ha_client = Mock()
        mock_ha_client.get_state.side_effect = [45.0, 21.3]
        
        online_learning.learn_from_previous_cycle(state, mock_ha_client, {})
        
        mock_learning.assert_called_once()
        mock_benchmark.assert_called_once_with(45.0, 42.0)


if __name__ == '__main__':
    unittest.main()
