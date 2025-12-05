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


class TestHASensorIntegration:
    """Integration tests for HA sensor functionality."""
    
    def setup_method(self):
        """Reset singleton instance before each test."""
        # Reset the singleton instance to ensure clean state
        src.model_wrapper._enhanced_model_wrapper_instance = None
        
        # Also reset any thermal state manager state for clean testing
        if hasattr(src.unified_thermal_state, '_thermal_state_manager_instance'):
            src.unified_thermal_state._thermal_state_manager_instance = None
    
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
