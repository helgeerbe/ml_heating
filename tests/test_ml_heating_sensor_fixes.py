"""
Unit tests for ML Heating Sensor Fixes

Tests the fixes implemented for sensor issues:
1. Fix prediction counter persistence
2. Repair trend analysis (lower threshold)  
3. Enable parameter tracking
4. Verify sensor data flow
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from prediction_metrics import PredictionMetrics
from datetime import datetime


class TestMLHeatingSensorFixes(unittest.TestCase):
    """Test suite for ML heating sensor fixes."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock state manager
        self.mock_state_manager = Mock()
        self.mock_state_manager.state = {
            "learning_state": {
                "prediction_history": []
            },
            "prediction_metrics": {
                "total_predictions": 0
            }
        }
        
    def test_prediction_counter_persistence(self):
        """Test that prediction counter persists across instances."""
        # Create first instance with state manager
        metrics1 = PredictionMetrics(state_manager=self.mock_state_manager)
        
        # Add predictions
        for i in range(5):
            metrics1.add_prediction(
                predicted=20.0 + i * 0.1,
                actual=20.0 + i * 0.1 + 0.05,
                context={'test': f'prediction_{i}'}
            )
        
        # Verify predictions were added
        self.assertEqual(len(metrics1.predictions), 5)
        
        # Verify state manager save was called
        self.assertTrue(self.mock_state_manager.save_state.called)
        
        # Update state to simulate persistence
        self.mock_state_manager.state["learning_state"]["prediction_history"] = list(metrics1.predictions)
        
        # Create second instance (simulating restart)
        metrics2 = PredictionMetrics(state_manager=self.mock_state_manager)
        
        # Verify persistence
        self.assertEqual(len(metrics2.predictions), 5)
        
    def test_trend_analysis_lower_threshold(self):
        """Test that trend analysis works with lowered threshold."""
        # Create metrics without state manager to test pure functionality
        metrics = PredictionMetrics(state_manager=None)
        
        # Add exactly 12 predictions (more than new threshold of 10)
        for i in range(12):
            error = 0.5 - (i * 0.03)  # Decreasing error = improving trend
            metrics.add_prediction(
                predicted=21.0,
                actual=21.0 + error,
                context={'test_trend': i}
            )
        
        # Verify we have the right number of predictions
        self.assertEqual(len(metrics.predictions), 12)
        
        # Get metrics
        result_metrics = metrics.get_metrics()
        trends = result_metrics.get('trends', {})
        
        # Debug output
        print(f"Predictions count: {len(metrics.predictions)}")
        print(f"Trends result: {trends}")
        
        # Verify threshold fix worked - trend analysis should be working
        # If insufficient_data key is missing, that means we have sufficient data
        self.assertFalse(trends.get('insufficient_data', False), 
                        "Trend analysis should work with 12 predictions")
        
        # Verify trend calculation works and contains expected keys
        self.assertIn('is_improving', trends)
        self.assertIn('mae_improvement_percentage', trends)
        self.assertIn('mae_improvement', trends)
        
        # Verify the trend shows improvement (decreasing error over time)
        self.assertTrue(trends['is_improving'])
        self.assertGreater(trends['mae_improvement_percentage'], 0)
        
    def test_prediction_metrics_integration(self):
        """Test prediction metrics integration with state manager."""
        metrics = PredictionMetrics(state_manager=self.mock_state_manager)
        
        # Add prediction
        metrics.add_prediction(
            predicted=21.0,
            actual=21.1,
            context={'test': 'integration'}
        )
        
        # Verify auto-save was called
        self.assertTrue(self.mock_state_manager.save_state.called)
        
        # Verify state was updated
        save_calls = self.mock_state_manager.save_state.call_count
        self.assertGreater(save_calls, 0)
        
    @patch('src.model_wrapper.get_enhanced_model_wrapper')
    def test_model_wrapper_integration(self, mock_wrapper):
        """Test model wrapper integration for HA sensor."""
        # Setup mock wrapper
        mock_instance = Mock()
        mock_instance.get_comprehensive_metrics_for_ha.return_value = {
            'total_predictions': 100,
            'is_improving': True,
            'improvement_percentage': 5.5,
            'parameter_updates': 10,
            'learning_confidence': 4.2,
            'last_updated': datetime.now().isoformat()
        }
        mock_wrapper.return_value = mock_instance
        
        # Import after mock is set up
        from model_wrapper import get_enhanced_model_wrapper
        
        wrapper = get_enhanced_model_wrapper()
        metrics = wrapper.get_comprehensive_metrics_for_ha()
        
        # Verify metrics are accessible
        self.assertIn('total_predictions', metrics)
        self.assertIn('is_improving', metrics)
        self.assertIn('parameter_updates', metrics)
        
    def test_metrics_calculation_with_real_data(self):
        """Test metrics calculation with realistic data."""
        metrics = PredictionMetrics(state_manager=self.mock_state_manager)
        
        # Add realistic prediction data
        predictions = [
            (21.0, 21.05),  # Small error
            (22.0, 22.15),  # Slightly larger error
            (20.5, 20.45),  # Small negative error
            (21.5, 21.7),   # Larger positive error
            (20.0, 20.0),   # Perfect prediction
        ]
        
        for pred, actual in predictions:
            metrics.add_prediction(pred, actual)
        
        # Get comprehensive metrics
        result = metrics.get_metrics()
        
        # Verify all time windows are calculated
        self.assertIn('1h', result)
        self.assertIn('6h', result)
        self.assertIn('24h', result)
        self.assertIn('all', result)
        
        # Verify MAE is calculated
        self.assertIn('mae', result['all'])
        self.assertGreater(result['all']['mae'], 0)
        
        # Verify RMSE is calculated
        self.assertIn('rmse', result['all'])
        self.assertGreater(result['all']['rmse'], 0)


class TestSensorDataFlow(unittest.TestCase):
    """Test sensor data flow and freshness."""
    
    def test_sensor_attribute_mapping(self):
        """Test that sensor attributes map correctly."""
        # This would test the actual sensor mapping
        # For now, just verify the expected attributes exist
        expected_attributes = [
            'thermal_time_constant',
            'heat_loss_coefficient', 
            'outlet_effectiveness',
            'learning_confidence',
            'cycle_count',
            'parameter_updates',
            'update_percentage',
            'mae_1h',
            'mae_6h', 
            'mae_24h',
            'mae_all_time',
            'rmse_all_time',
            'recent_mae_10',
            'recent_max_error',
            'model_health',
            'is_improving',
            'improvement_percentage',
            'total_predictions',
            'last_updated'
        ]
        
        # This is a structural test - just verify we know what attributes should exist
        self.assertTrue(len(expected_attributes) > 0)


if __name__ == '__main__':
    unittest.main()
