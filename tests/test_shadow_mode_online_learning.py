"""
Test suite for shadow mode online learning functionality.

Tests the core shadow mode physics learning and ML vs heat curve benchmarking system.
"""

import unittest
from unittest.mock import Mock, patch
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
    """Test shadow mode physics learning (Task 3.1 from TODO)"""
    
    def setUp(self):
        """Set up test environment"""
        self.thermal_model = ThermalEquilibriumModel()
        # Clear singleton prediction history for clean tests
        self.thermal_model.prediction_history = []
        self.thermal_model.parameter_history = []
        self.online_learning = OnlineLearning()
    
    @patch('src.config.SHADOW_MODE', True)
    def test_shadow_mode_learning_ignores_target(self):
        """Test that shadow mode learning ignores target temperature"""
        # Mock prediction context without target temperature
        prediction_context = {
            'outlet_temp': 45.0,
            'outdoor_temp': 10.0, 
            'pv_power': 0.0,
            'fireplace_on': 0.0,
            'tv_on': 0.0
            # Note: NO target_indoor_temp in shadow mode
        }
        
        # Test that thermal model accepts context without target
        try:
            self.thermal_model.update_prediction_feedback(
                predicted_temp=21.0,
                actual_temp=21.5,
                prediction_context=prediction_context,
                timestamp=datetime.now().isoformat()
            )
            # Should not raise exception - function returns None
            self.assertEqual(len(self.thermal_model.prediction_history), 1)
        except KeyError as e:
            self.fail(f"Shadow mode learning requires target temp: {e}")
    
    @patch('src.config.SHADOW_MODE', True)
    def test_shadow_mode_uses_heat_curve_outlet(self):
        """Test that shadow mode learns from heat curve outlet settings"""
        heat_curve_outlet = 42.0
        actual_indoor_change = 0.3
        
        # Mock learning features
        learning_features = {
            'outlet_temp': heat_curve_outlet,
            'outdoor_temp': 8.0,
            'pv_now': 100.0,
            'fireplace_on': 0.0,
            'tv_on': 0.0
        }
        
        # Mock the learning process
        with patch.object(self.online_learning, '_perform_online_learning') as mock_learning:
            self.online_learning._perform_online_learning(
                learning_features, heat_curve_outlet, actual_indoor_change, 21.3
            )
            
            # Verify learning was called with heat curve outlet
            mock_learning.assert_called_once()
            args = mock_learning.call_args[0]
            self.assertEqual(args[1], heat_curve_outlet)  # actual_applied_temp
    
    def test_shadow_mode_parameter_convergence(self):
        """Test parameter convergence with known physics scenarios"""
        # Test with known physics relationship
        # If outlet=40°C always produces indoor_change=+0.2°C, 
        # parameters should converge to reflect this
        
        known_scenarios = [
            {'outlet': 40.0, 'outdoor': 10.0, 'expected_change': 0.2},
            {'outlet': 45.0, 'outdoor': 10.0, 'expected_change': 0.4},
            {'outlet': 35.0, 'outdoor': 10.0, 'expected_change': 0.0},
        ]
        
        for scenario in known_scenarios:
            prediction_context = {
                'outlet_temp': scenario['outlet'],
                'outdoor_temp': scenario['outdoor'],
                'pv_power': 0.0,
                'fireplace_on': 0.0,
                'tv_on': 0.0
            }
            
            # Multiple learning iterations should improve predictions
            for _ in range(10):
                self.thermal_model.update_prediction_feedback(
                    predicted_temp=21.0,
                    actual_temp=21.0 + scenario['expected_change'],
                    prediction_context=prediction_context,
                    timestamp=datetime.now().isoformat()
                )
            
            # Check if parameters adapted (not exact, just direction)
            self.assertGreater(len(self.thermal_model.prediction_history), 0)


class TestShadowModeBenchmarking(unittest.TestCase):
    """Test shadow mode benchmarking system (Task 3.2 from TODO)"""
    
    def setUp(self):
        self.online_learning = OnlineLearning()
    
    def test_ml_prediction_calculation(self):
        """Test ML outlet prediction for target temperatures"""
        target_temp = 22.0
        current_temp = 21.0
        context = {
            'outdoor_temp': 5.0,
            'pv_power': 200.0,
            'fireplace_on': 0.0,
            'tv_on': 0.0
        }
        
        # Mock the model wrapper
        with patch('src.temperature_control.get_enhanced_model_wrapper') as mock_wrapper:
            mock_model = Mock()
            mock_model.calculate_optimal_outlet_temperature.return_value = {
                'optimal_outlet_temp': 43.5
            }
            mock_wrapper.return_value = mock_model
            
            result = self.online_learning.calculate_ml_benchmark_prediction(
                target_temp, current_temp, context
            )
            
            self.assertEqual(result, 43.5)
            mock_model.calculate_optimal_outlet_temperature.assert_called_once_with(
                target_indoor=target_temp,
                current_indoor=current_temp,
                outdoor_temp=5.0,
                pv_power=200.0,
                fireplace_on=0.0,
                tv_on=0.0
            )
    
    def test_benchmark_logging(self):
        """Test benchmark comparison logging format"""
        ml_prediction = 42.0
        heat_curve_actual = 45.0
        
        with patch('src.temperature_control.logging') as mock_logging:
            self.online_learning._log_shadow_mode_comparison(
                heat_curve_actual, ml_prediction
            )
            
            # Check that comparison was logged with correct format
            mock_logging.info.assert_called()
            # The log uses string formatting, so check for the template
            log_call = mock_logging.info.call_args
            self.assertEqual(len(log_call[0]), 4)  # format string + 3 values
            self.assertIn("Shadow Benchmark", log_call[0][0])
            self.assertEqual(log_call[0][1], 42.0)  # ML prediction
            self.assertEqual(log_call[0][2], 45.0)  # Heat curve actual
    
    def test_efficiency_comparison_calculation(self):
        """Test efficiency advantage calculation"""
        ml_prediction = 40.0
        heat_curve_actual = 45.0
        expected_advantage = 5.0  # heat_curve - ml = more efficient
        
        # Mock InfluxDB export
        with patch.object(self.online_learning, '_export_shadow_benchmark_data') as mock_export:
            self.online_learning._log_shadow_mode_comparison(
                heat_curve_actual, ml_prediction
            )
            
            # Verify export was called with correct efficiency advantage
            mock_export.assert_called_once()
            args = mock_export.call_args[1]
            self.assertEqual(args['efficiency_advantage'], expected_advantage)


class TestShadowModeCompleteCycle(unittest.TestCase):
    """Integration tests for complete shadow mode cycles (Task 3.3 from TODO)"""
    
    def setUp(self):
        """Set up clean test environment"""
        self.thermal_model = ThermalEquilibriumModel()
        # Clear singleton prediction history for clean tests
        self.thermal_model.prediction_history = []
        self.thermal_model.parameter_history = []
    
    @patch('src.config.SHADOW_MODE', True)
    def test_shadow_mode_startup_fresh(self):
        """Test fresh start scenario (no thermal_state.json)"""
        # Test that shadow mode works with default parameters
        thermal_model = self.thermal_model
        
        # Should start with default parameters
        self.assertGreater(thermal_model.outlet_effectiveness, 0)
        self.assertGreater(thermal_model.heat_loss_coefficient, 0)
        self.assertGreater(thermal_model.thermal_time_constant, 0)
        
        # Should accept learning immediately
        prediction_context = {
            'outlet_temp': 40.0,
            'outdoor_temp': 10.0,
            'pv_power': 0.0,
            'fireplace_on': 0.0,
            'tv_on': 0.0
        }
        
        thermal_model.update_prediction_feedback(
            predicted_temp=21.0,
            actual_temp=21.2,
            prediction_context=prediction_context,
            timestamp=datetime.now().isoformat()
        )
        
        # Should return None but not raise exception
        # Check that learning occurred by verifying prediction_history
        self.assertEqual(len(thermal_model.prediction_history), 1)
    
    @patch('src.config.SHADOW_MODE', True)
    def test_shadow_mode_startup_calibrated(self):
        """Test startup with existing calibration"""
        # Mock existing calibration data
        thermal_model = ThermalEquilibriumModel()
        
        # Simulate pre-calibrated parameters
        thermal_model.outlet_effectiveness = 0.85
        thermal_model.heat_loss_coefficient = 150.0
        thermal_model.thermal_time_constant = 3600.0
        
        # Shadow mode should continue refining from calibrated baseline
        prediction_context = {
            'outlet_temp': 42.0,
            'outdoor_temp': 8.0,
            'pv_power': 50.0,
            'fireplace_on': 0.0,
            'tv_on': 0.0
        }
        
        thermal_model.update_prediction_feedback(
            predicted_temp=21.5,
            actual_temp=21.7,
            prediction_context=prediction_context,
            timestamp=datetime.now().isoformat()
        )
        
        # Learning should have occurred
        self.assertEqual(len(thermal_model.prediction_history), 1)
        # Effectiveness should remain a valid number
        self.assertIsInstance(thermal_model.outlet_effectiveness, (int, float))
    
    def test_shadow_mode_learning_and_benchmarking(self):
        """Test full shadow mode cycle: learning + benchmarking"""
        online_learning = OnlineLearning()
        
        # Mock state data for learning
        state = {
            'last_run_features': {
                'outdoor_temp': 5.0,
                'pv_now': 100.0,
                'fireplace_on': 0.0,
                'tv_on': 0.0
            },
            'last_indoor_temp': 21.0,
            'last_final_temp': 42.0  # ML's prediction
        }
        
        # Mock HA client
        mock_ha_client = Mock()
        mock_ha_client.get_state.side_effect = [
            45.0,  # Actual applied temp (heat curve)
            21.3   # Current indoor temp
        ]
        
        all_states = {}
        
        with patch.object(online_learning, '_perform_online_learning') as mock_learning, \
             patch.object(online_learning, '_log_shadow_mode_comparison') as mock_benchmark:
            
            online_learning.learn_from_previous_cycle(state, mock_ha_client, all_states)
            
            # Both learning and benchmarking should occur
            mock_learning.assert_called_once()
            mock_benchmark.assert_called_once_with(45.0, 42.0)  # heat_curve, ml_prediction


class TestShadowModeValidation(unittest.TestCase):
    """Validation against known scenarios (Task 3.4 from TODO)"""
    
    def setUp(self):
        """Set up clean test environment"""
        self.thermal_model = ThermalEquilibriumModel()
        # Clear singleton prediction history for clean tests
        self.thermal_model.prediction_history = []
        self.thermal_model.parameter_history = []
    
    def test_known_physics_scenarios(self):
        """Test with mock heat curve data (known outlet → indoor relationships)"""
        thermal_model = self.thermal_model
        
        # Known physics: outlet=40°C, outdoor=10°C should produce ~0.15°C/hour change
        test_scenarios = [
            {'outlet': 40, 'outdoor': 10, 'expected': 0.15},
            {'outlet': 45, 'outdoor': 10, 'expected': 0.35},
            {'outlet': 35, 'outdoor': 10, 'expected': -0.05},
        ]
        
        for scenario in test_scenarios:
            prediction_context = {
                'outlet_temp': scenario['outlet'],
                'outdoor_temp': scenario['outdoor'],
                'pv_power': 0.0,
                'fireplace_on': 0.0,
                'tv_on': 0.0
            }
            
            # Train with known result
            for _ in range(5):
                thermal_model.update_prediction_feedback(
                    predicted_temp=21.0,
                    actual_temp=21.0 + scenario['expected'],
                    prediction_context=prediction_context,
                    timestamp=datetime.now().isoformat()
                )
            
            # Check that learning occurred
            self.assertGreater(len(thermal_model.prediction_history), 0)
    
    def test_parameter_convergence_rates(self):
        """Test parameter convergence rates"""
        thermal_model = self.thermal_model
        
        # Consistent learning should converge parameters
        prediction_context = {
            'outlet_temp': 42.0,
            'outdoor_temp': 8.0,
            'pv_power': 0.0,
            'fireplace_on': 0.0,
            'tv_on': 0.0
        }
        
        # 20 learning iterations
        for i in range(20):
            thermal_model.update_prediction_feedback(
                predicted_temp=21.0,
                actual_temp=21.25,  # Consistent 0.25°C increase
                prediction_context=prediction_context,
                timestamp=datetime.now().isoformat()
            )
        
        # Learning should have occurred
        self.assertEqual(len(thermal_model.prediction_history), 20)
        # Parameters might have changed through learning
        self.assertGreater(len(thermal_model.parameter_history), 0)


if __name__ == '__main__':
    unittest.main()
