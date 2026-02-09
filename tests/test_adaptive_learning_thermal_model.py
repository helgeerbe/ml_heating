"""
Unit tests for the enhanced ThermalEquilibriumModel with adaptive learning capabilities.

Tests cover:
- Real-time parameter adaptation
- Prediction error feedback
- Learning rate scheduling
- Parameter stability monitoring
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from datetime import datetime
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.thermal_equilibrium_model import ThermalEquilibriumModel, _thermal_equilibrium_model_instance


class TestAdaptiveLearningThermalModel(unittest.TestCase):
    """Test suite for adaptive learning features in ThermalEquilibriumModel."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset singleton to ensure fresh instance for each test
        global _thermal_equilibrium_model_instance
        _thermal_equilibrium_model_instance = None

        # Mock the state manager to prevent file system dependencies
        self.mock_state_manager = MagicMock()
        self.mock_state_manager.get_current_parameters.return_value = {
            "baseline_parameters": {
                "source": "calibrated",
                "thermal_time_constant": 5.0,
                "heat_loss_coefficient": 0.1,
                "outlet_effectiveness": 0.8,
            },
            "learning_state": {},
        }
        self.mock_state_manager.state = {
            "baseline_parameters": {
                "source": "calibrated",
                "thermal_time_constant": 5.0,
                "heat_loss_coefficient": 0.1,
                "outlet_effectiveness": 0.8,
            },
            "learning_state": {
                "parameter_adjustments": {
                    "thermal_time_constant_delta": 0.0,
                    "heat_loss_coefficient_delta": 0.0,
                    "outlet_effectiveness_delta": 0.0,
                }
            }
        }

        # Patch where get_thermal_state_manager is looked up.
        self.patcher = patch(
            'src.unified_thermal_state.get_thermal_state_manager',
            return_value=self.mock_state_manager
        )
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

        self.model = ThermalEquilibriumModel()
        # Since loading is mocked, explicitly set parameters for tests
        self.model.heat_loss_coefficient = 0.1
        self.model.outlet_effectiveness = 0.8
        self.model.thermal_time_constant = 5.0
        self.model.reset_adaptive_learning()  # Clean state for each test

        # Standard test scenario
        self.test_context = {
            'outlet_temp': 40.0,
            'outdoor_temp': 10.0,
            'pv_power': 1000.0,
            'current_indoor': 20.0
        }
        
    def test_adaptive_learning_initialization(self):
        """Test that adaptive learning is properly initialized."""
        self.assertTrue(self.model.adaptive_learning_enabled)
        self.assertEqual(len(self.model.prediction_history), 0)
        self.assertEqual(len(self.model.parameter_history), 0)
        self.assertEqual(self.model.learning_confidence, 3.0)
        self.assertEqual(self.model.recent_errors_window, 10)
        
        # Test parameter bounds
        self.assertEqual(self.model.thermal_time_constant_bounds, (3.0, 8.0))
        self.assertEqual(self.model.heat_loss_coefficient_bounds, (0.01, 0.5))
        self.assertEqual(self.model.outlet_effectiveness_bounds, (0.5, 1.0))
        
    def test_prediction_feedback_basic(self):
        """Test basic prediction feedback functionality."""
        self.model.update_prediction_feedback(
            predicted_temp=20.5,
            actual_temp=20.8,
            prediction_context=self.test_context,
            timestamp="2025-12-02T12:00:00"
        )
        
        self.assertEqual(len(self.model.prediction_history), 1)
        prediction = self.model.prediction_history[0]
        
        self.assertEqual(prediction['predicted'], 20.5)
        self.assertEqual(prediction['actual'], 20.8)
        self.assertAlmostEqual(prediction['error'], 0.3, places=1)
        self.assertEqual(prediction['timestamp'], "2025-12-02T12:00:00")
        self.assertIn('outlet_temp', prediction['context'])
        
    def test_prediction_feedback_disabled(self):
        """Test that feedback is ignored when adaptive learning is disabled."""
        self.model.adaptive_learning_enabled = False
        
        self.model.update_prediction_feedback(
            predicted_temp=20.5,
            actual_temp=20.8,
            prediction_context=self.test_context,
            timestamp="2025-12-02T12:00:00"
        )
        
        self.assertEqual(len(self.model.prediction_history), 0)
        
    def test_learning_confidence_evolution(self):
        """Test that learning confidence evolves based on prediction accuracy."""
        initial_confidence = self.model.learning_confidence
        
        for i in range(10):
            error = 1.0 - (i * 0.08)
            actual_temp = 20.0 + error
            
            self.model.update_prediction_feedback(
                predicted_temp=20.0,
                actual_temp=actual_temp,
                prediction_context=self.test_context,
                timestamp=f"2025-12-02T12:{i:02d}:00"
            )
        
        final_confidence = self.model.learning_confidence
        self.assertGreater(final_confidence, initial_confidence)
        
    def test_parameter_adaptation_threshold(self):
        """Test that parameters adapt only after sufficient prediction history."""
        for i in range(self.model.recent_errors_window - 1):
            self.model.update_prediction_feedback(
                predicted_temp=20.0,
                actual_temp=20.5,
                prediction_context=self.test_context,
                timestamp=f"2025-12-02T12:{i:02d}:00"
            )
        
        self.assertEqual(len(self.model.parameter_history), 0)
        
        self.model.update_prediction_feedback(
            predicted_temp=20.0,
            actual_temp=20.5,
            prediction_context=self.test_context,
            timestamp="2025-12-02T12:20:00"
        )
        
        self.assertGreaterEqual(len(self.model.parameter_history), 1)
            
    def tearDown(self):
        """Clean up after each test."""
        state_file = "/opt/ml_heating/thermal_state.json"
        if os.path.exists(state_file):
            os.remove(state_file)

    def test_parameter_bounds_enforcement(self):
        """Test that parameters stay within defined bounds during adaptation."""
        for i in range(50):
            self.model.update_prediction_feedback(
                predicted_temp=19.0,
                actual_temp=21.0,
                prediction_context=self.test_context,
                timestamp=f"2025-12-02T{i // 60:02d}:{i % 60:02d}:00"
            )

        self.assertGreaterEqual(
            self.model.thermal_time_constant,
            self.model.thermal_time_constant_bounds[0]
        )
        self.assertLessEqual(
            self.model.thermal_time_constant,
            self.model.thermal_time_constant_bounds[1]
        )
        
        self.assertGreaterEqual(
            self.model.heat_loss_coefficient, 
            self.model.heat_loss_coefficient_bounds[0]
        )
        self.assertLessEqual(
            self.model.heat_loss_coefficient, 
            self.model.heat_loss_coefficient_bounds[1]
        )
        
        self.assertGreaterEqual(
            self.model.outlet_effectiveness, 
            self.model.outlet_effectiveness_bounds[0]
        )
        self.assertLessEqual(
            self.model.outlet_effectiveness, 
            self.model.outlet_effectiveness_bounds[1]
        )
                            
    def test_adaptive_learning_rate_calculation(self):
        """Test adaptive learning rate calculation."""
        for _ in range(10):
            self.model.parameter_history.append({
                'timestamp': datetime.now(),
                'thermal_time_constant': 5.0,
                'heat_loss_coefficient': 0.1,
                'outlet_effectiveness': 0.8,
                'learning_rate': 0.01,
                'learning_confidence': 1.0,
                'avg_recent_error': 0.1,
                'gradients': {},
                'changes': {}
            })
        
        adaptive_rate = self.model._calculate_adaptive_learning_rate()
        self.assertLessEqual(
            adaptive_rate,
            self.model.learning_rate * self.model.learning_confidence
        )
        
    def test_gradient_calculation_thermal_time_constant(self):
        """Test numerical gradient calculation for thermal time constant."""
        recent_predictions = []
        for i in range(5):
            recent_predictions.append({
                'error': 0.5,
                'context': {
                    'outlet_temp': 35.0,
                    'outdoor_temp': 8.0,
                    'current_indoor': 20.0,
                    'pv_power': 500.0
                },
                'timestamp': '2023-01-01T00:00:00Z'
            })
        
        gradient = self.model._calculate_thermal_time_constant_gradient(recent_predictions)
        self.assertTrue(np.isfinite(gradient))
        
    def test_gradient_calculation_heat_loss_coefficient(self):
        """Test numerical gradient calculation for heat loss coefficient."""
        recent_predictions = []
        for i in range(5):
            recent_predictions.append({
                'error': -0.3,
                'context': {
                    'outlet_temp': 40.0,
                    'outdoor_temp': 12.0,
                    'current_indoor': 20.0,
                    'pv_power': 800.0
                },
                'timestamp': '2023-01-01T00:00:00Z'
            })
        
        gradient = self.model._calculate_heat_loss_coefficient_gradient(recent_predictions)
        self.assertTrue(np.isfinite(gradient))
        
    def test_gradient_calculation_outlet_effectiveness(self):
        """Test numerical gradient calculation for outlet effectiveness."""
        recent_predictions = []
        for i in range(5):
            recent_predictions.append({
                'error': 0.8,
                'context': {
                    'outlet_temp': 45.0,
                    'outdoor_temp': 5.0,
                    'current_indoor': 20.0,
                    'pv_power': 200.0
                },
                'timestamp': '2023-01-01T00:00:00Z'
            })
        
        gradient = self.model._calculate_outlet_effectiveness_gradient(recent_predictions)
        self.assertTrue(np.isfinite(gradient))
        
    def test_adaptive_learning_metrics(self):
        """Test adaptive learning metrics calculation."""
        for i in range(25):
            error = 1.0 - (i * 0.03)
            self.model.update_prediction_feedback(
                predicted_temp=20.0,
                actual_temp=20.0 + error,
                prediction_context=self.test_context,
                timestamp=f"2025-12-02T12:{i:02d}:00"
            )

        metrics = self.model.get_adaptive_learning_metrics()

        self.assertFalse(metrics.get('insufficient_data', False))
        self.assertIn('total_predictions', metrics)
        self.assertIn('avg_recent_error', metrics)
        self.assertIn('error_improvement_trend', metrics)
        self.assertIn('learning_confidence', metrics)
        self.assertIn('current_learning_rate', metrics)
        self.assertIn('current_parameters', metrics)
        self.assertGreater(metrics['error_improvement_trend'], 0)
        
    def test_learning_metrics_insufficient_data(self):
        """Test learning metrics with insufficient data."""
        metrics = self.model.get_adaptive_learning_metrics()
        self.assertTrue(metrics.get('insufficient_data', False))
        
    def test_reset_adaptive_learning(self):
        """Test resetting adaptive learning state."""
        for i in range(10):
            self.model.update_prediction_feedback(
                predicted_temp=20.0,
                actual_temp=21.0,
                prediction_context=self.test_context,
                timestamp=f"2025-12-02T12:{i:02d}:00"
            )
        
        self.assertGreater(len(self.model.prediction_history), 0)
        self.model.reset_adaptive_learning()
        self.assertEqual(len(self.model.prediction_history), 0)
        self.assertEqual(len(self.model.parameter_history), 0)
        self.assertEqual(self.model.learning_confidence, 3.0)
        
    def test_prediction_history_size_management(self):
        """Test that prediction history size is managed properly."""
        for i in range(250):
            self.model.update_prediction_feedback(
                predicted_temp=20.0,
                actual_temp=20.2,
                prediction_context=self.test_context,
                timestamp=f"2025-12-02T{i // 60:02d}:{i % 60:02d}:00"
            )
        self.assertLessEqual(len(self.model.prediction_history), 200)
        
    def test_parameter_history_size_management(self):
        """Test that parameter history size is managed properly."""
        for i in range(600):
            self.model.update_prediction_feedback(
                predicted_temp=20.0,
                actual_temp=20.0 + (0.5 if i % 2 == 0 else -0.5),
                prediction_context=self.test_context,
                timestamp=f"2025-12-02T{i // 60:02d}:{i % 60:02d}:00"
            )
        self.assertLessEqual(len(self.model.parameter_history), 500)

    def test_learning_convergence_detection(self):
        """Test detection of learning convergence."""
        for i in range(25):
            self.model.update_prediction_feedback(
                predicted_temp=20.0,
                actual_temp=20.1,
                prediction_context=self.test_context,
                timestamp=f"2025-12-02T12:{i:02d}:00"
            )

        stable_params = {
            'thermal_time_constant': 5.0,
            'heat_loss_coefficient': 0.1,
            'outlet_effectiveness': 0.8
        }
        
        for i in range(10):
            self.model.parameter_history.append({
                'timestamp': datetime.now(),
                **stable_params,
                'learning_rate': 0.01,
                'learning_confidence': 1.0,
                'avg_recent_error': 0.05,
                'gradients': {},
                'changes': {}
            })
        
        metrics = self.model.get_adaptive_learning_metrics()
        
        self.assertFalse(metrics.get('insufficient_data', False))
        self.assertIn('total_predictions', metrics)
        self.assertIn('current_parameters', metrics)
        
        if 'thermal_time_constant_stability' in metrics:
            self.assertLess(metrics['thermal_time_constant_stability'], 0.1)
            self.assertLess(metrics['heat_loss_coefficient_stability'], 0.001)
            self.assertLess(metrics['outlet_effectiveness_stability'], 0.01)

if __name__ == '__main__':
    unittest.main(verbosity=2)
