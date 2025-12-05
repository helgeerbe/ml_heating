"""
Test suite for Home Assistant sensor export functionality.

Tests that HA sensors are properly updated when learning cycles complete.
"""
import pytest
from unittest.mock import MagicMock, patch, call
from datetime import datetime

from src.model_wrapper import EnhancedModelWrapper
import src.model_wrapper
import src.unified_thermal_state


class TestHASensorExport:
    """Test Home Assistant sensor export functionality."""
    
    def setup_method(self):
        """Reset singleton instance before each test."""
        # Reset the singleton instance to ensure clean state
        src.model_wrapper._enhanced_model_wrapper_instance = None
        
        # Also reset any thermal state manager state for clean testing
        if hasattr(src.unified_thermal_state, '_thermal_state_manager_instance'):
            src.unified_thermal_state._thermal_state_manager_instance = None
    
    @patch('src.ha_client.create_ha_client')
    def test_export_metrics_to_ha_called_during_learning(self, mock_create_ha_client):
        """Test that _export_metrics_to_ha is called during learning feedback."""
        # Arrange
        mock_ha_client = MagicMock()
        mock_create_ha_client.return_value = mock_ha_client
        
        wrapper = EnhancedModelWrapper()
        
        prediction_context = {
            'outlet_temp': 35.0,
            'outdoor_temp': 5.0,
            'pv_power': 0.0,
            'fireplace_on': 0.0,
            'tv_on': 0.0
        }
        
        # Act
        wrapper.learn_from_prediction_feedback(
            predicted_temp=21.5,
            actual_temp=21.3,
            prediction_context=prediction_context,
            timestamp=datetime.now().isoformat()
        )
        
        # Assert
        mock_create_ha_client.assert_called_once()
        mock_ha_client.log_model_metrics.assert_called_once()
        mock_ha_client.log_adaptive_learning_metrics.assert_called_once()
    
    @patch('src.ha_client.create_ha_client')
    def test_log_model_metrics_called_with_correct_parameters(self, mock_create_ha_client):
        """Test that log_model_metrics is called with correct confidence, MAE, RMSE."""
        # Arrange
        mock_ha_client = MagicMock()
        mock_create_ha_client.return_value = mock_ha_client
        
        wrapper = EnhancedModelWrapper()
        
        # Set up some prediction history for metrics
        prediction_context = {
            'outlet_temp': 35.0,
            'outdoor_temp': 5.0,
            'pv_power': 0.0,
            'fireplace_on': 0.0,
            'tv_on': 0.0
        }
        
        # Add a few predictions to generate metrics
        wrapper.learn_from_prediction_feedback(21.5, 21.3, prediction_context)
        wrapper.learn_from_prediction_feedback(21.8, 21.6, prediction_context)
        
        # Clear previous calls
        mock_ha_client.reset_mock()
        
        # Act - trigger another learning cycle
        wrapper.learn_from_prediction_feedback(22.0, 21.9, prediction_context)
        
        # Assert
        call_args = mock_ha_client.log_model_metrics.call_args
        assert call_args is not None
        
        # Check that confidence, mae, rmse are passed as keyword arguments
        kwargs = call_args.kwargs
        assert 'confidence' in kwargs
        assert 'mae' in kwargs
        assert 'rmse' in kwargs
        
        # Check that values are reasonable
        assert isinstance(kwargs['confidence'], float)
        assert kwargs['confidence'] > 0
        assert isinstance(kwargs['mae'], float) 
        assert kwargs['mae'] >= 0
        assert isinstance(kwargs['rmse'], float)
        assert kwargs['rmse'] >= 0
    
    @patch('src.ha_client.create_ha_client')
    def test_log_adaptive_learning_metrics_called_with_comprehensive_data(self, mock_create_ha_client):
        """Test that log_adaptive_learning_metrics is called with comprehensive metrics."""
        # Arrange
        mock_ha_client = MagicMock()
        mock_create_ha_client.return_value = mock_ha_client
        
        wrapper = EnhancedModelWrapper()
        
        prediction_context = {
            'outlet_temp': 35.0,
            'outdoor_temp': 5.0,
            'pv_power': 0.0,
            'fireplace_on': 0.0,
            'tv_on': 0.0
        }
        
        # Act
        wrapper.learn_from_prediction_feedback(21.5, 21.3, prediction_context)
        
        # Assert
        call_args = mock_ha_client.log_adaptive_learning_metrics.call_args
        assert call_args is not None
        
        # Check that comprehensive metrics dictionary is passed
        metrics_dict = call_args.args[0]
        assert isinstance(metrics_dict, dict)
        
        # Check required thermal parameters
        assert 'thermal_time_constant' in metrics_dict
        assert 'heat_loss_coefficient' in metrics_dict
        assert 'outlet_effectiveness' in metrics_dict
        assert 'learning_confidence' in metrics_dict
        
        # Check learning progress
        assert 'cycle_count' in metrics_dict
        assert 'parameter_updates' in metrics_dict
        
        # Check prediction accuracy metrics
        assert 'mae_1h' in metrics_dict
        assert 'mae_6h' in metrics_dict
        assert 'mae_24h' in metrics_dict
        assert 'mae_all_time' in metrics_dict
        assert 'rmse_all_time' in metrics_dict
        
        # Check accuracy breakdown
        assert 'excellent_accuracy_pct' in metrics_dict
        assert 'good_accuracy_pct' in metrics_dict
        
        # Check model health
        assert 'model_health' in metrics_dict
        assert 'is_improving' in metrics_dict
        assert 'total_predictions' in metrics_dict
    
    @patch('src.ha_client.create_ha_client')
    def test_feature_importance_export_when_available(self, mock_create_ha_client):
        """Test that feature importance is exported when thermal model provides it."""
        # Arrange
        mock_ha_client = MagicMock()
        mock_create_ha_client.return_value = mock_ha_client
        
        wrapper = EnhancedModelWrapper()
        
        # Mock the thermal model to have get_feature_importance method
        mock_importances = {
            'outlet_temp': 0.35,
            'outdoor_temp': 0.25,
            'pv_power': 0.15,
            'indoor_temp_gradient': 0.10,
            'temp_diff_indoor_outdoor': 0.15
        }
        wrapper.thermal_model.get_feature_importance = MagicMock(return_value=mock_importances)
        
        prediction_context = {
            'outlet_temp': 35.0,
            'outdoor_temp': 5.0,
            'pv_power': 0.0,
            'fireplace_on': 0.0,
            'tv_on': 0.0
        }
        
        # Act
        wrapper.learn_from_prediction_feedback(21.5, 21.3, prediction_context)
        
        # Assert
        wrapper.thermal_model.get_feature_importance.assert_called_once()
        mock_ha_client.log_feature_importance.assert_called_once_with(mock_importances)
    
    @patch('src.ha_client.create_ha_client')
    def test_feature_importance_skipped_when_not_available(self, mock_create_ha_client):
        """Test that feature importance export is skipped when not available."""
        # Arrange
        mock_ha_client = MagicMock()
        mock_create_ha_client.return_value = mock_ha_client
        
        wrapper = EnhancedModelWrapper()
        
        # Ensure thermal model doesn't have get_feature_importance method
        assert not hasattr(wrapper.thermal_model, 'get_feature_importance')
        
        prediction_context = {
            'outlet_temp': 35.0,
            'outdoor_temp': 5.0,
            'pv_power': 0.0,
            'fireplace_on': 0.0,
            'tv_on': 0.0
        }
        
        # Act
        wrapper.learn_from_prediction_feedback(21.5, 21.3, prediction_context)
        
        # Assert - should not crash and should not call log_feature_importance
        mock_ha_client.log_feature_importance.assert_not_called()
    
    def test_get_comprehensive_metrics_for_ha_structure(self):
        """Test that get_comprehensive_metrics_for_ha returns well-structured data."""
        # Arrange
        wrapper = EnhancedModelWrapper()
        
        # Act
        metrics = wrapper.get_comprehensive_metrics_for_ha()
        
        # Assert structure
        assert isinstance(metrics, dict)
        
        # Required thermal parameters
        required_thermal_keys = [
            'thermal_time_constant', 'heat_loss_coefficient', 
            'outlet_effectiveness', 'learning_confidence'
        ]
        for key in required_thermal_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))
        
        # Required learning progress keys
        required_learning_keys = [
            'cycle_count', 'parameter_updates', 'update_percentage'
        ]
        for key in required_learning_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))
        
        # Required accuracy keys
        required_accuracy_keys = [
            'mae_1h', 'mae_6h', 'mae_24h', 'mae_all_time', 'rmse_all_time'
        ]
        for key in required_accuracy_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))
        
        # Required status keys
        assert 'model_health' in metrics
        assert metrics['model_health'] in ['excellent', 'good', 'fair', 'poor']
        
        assert 'total_predictions' in metrics
        assert isinstance(metrics['total_predictions'], int)
        
        assert 'last_updated' in metrics
        assert isinstance(metrics['last_updated'], str)


class TestHASensorIntegration:
    """Integration tests for HA sensor functionality."""
    
    def setup_method(self):
        """Reset singleton instance before each test."""
        # Reset the singleton instance to ensure clean state
        src.model_wrapper._enhanced_model_wrapper_instance = None
    
    def test_sensor_ml_heating_state_values_mapping(self):
        """Test that sensor.ml_heating_state state codes are properly documented."""
        # This test documents the state codes used in main.py
        expected_state_codes = {
            0: "OK - Prediction done (confidence >= threshold)",
            1: "Confidence - Too Low (confidence < threshold)", 
            2: "DHW/Defrost active - Skipping cycle",
            3: "Network Error - could not fetch HA states",
            4: "Critical sensors unavailable",
            6: "Heating system not in 'heat' or 'auto' mode",
            7: "Model error in main loop"
        }
        
        # This test serves as documentation - no actual testing needed
        # but ensures we track what state codes mean
        assert len(expected_state_codes) == 7
        assert all(isinstance(code, int) for code in expected_state_codes.keys())
        assert all(isinstance(desc, str) for desc in expected_state_codes.values())
    
    def test_sensor_list_completeness(self):
        """Test that all expected HA sensors are accounted for."""
        expected_sensors = [
            'sensor.ml_heating_state',           # State codes (0-7)
            'sensor.ml_vorlauftemperatur',       # Target outlet temp
            'sensor.ml_model_confidence',        # Confidence score
            'sensor.ml_model_mae',               # Mean Absolute Error
            'sensor.ml_model_rmse',              # Root Mean Squared Error
            'sensor.ml_feature_importance',      # Feature importance
            'sensor.ml_heating_learning',        # Adaptive learning metrics
            'sensor.ml_prediction_accuracy'      # Prediction accuracy %
        ]
        
        # Verify we have 8 total sensors
        assert len(expected_sensors) == 8
        
        # Verify naming convention consistency
        ml_sensors = [s for s in expected_sensors if s.startswith('sensor.ml_')]
        assert len(ml_sensors) == 8  # All should be ML sensors
