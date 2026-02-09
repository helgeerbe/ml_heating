"""
Unit tests for parameter corruption detection in ThermalEquilibriumModel.

These tests ensure the system can detect when thermal parameters have been
corrupted and need to be reset to prevent catastrophic prediction errors.

TDD Approach: Tests written FIRST, then implementation follows.
"""

import pytest
from unittest.mock import patch

# Import the model we're testing
from src.thermal_equilibrium_model import ThermalEquilibriumModel, _thermal_equilibrium_model_instance


class TestParameterCorruptionDetection:
    """Test parameter corruption detection functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create fresh model instance for each test
        global _thermal_equilibrium_model_instance
        _thermal_equilibrium_model_instance = None
        self.model = ThermalEquilibriumModel()
        
        # Set known good parameters as baseline
        self.model.heat_loss_coefficient = 0.1
        self.model.outlet_effectiveness = 0.8
        self.model.thermal_time_constant = 5.0
        self.model.learning_confidence = 3.0
        
        # Mock thermal state loading to prevent file system dependencies
        with patch('src.unified_thermal_state.get_thermal_state_manager'):
            pass
    
    def test_detect_corrupted_heat_loss_coefficient_too_low(self):
        """Test detection of heat_loss_coefficient corruption - value too low."""
        self.model.heat_loss_coefficient = 0.005
        assert self.model._detect_parameter_corruption() is True

    def test_detect_corrupted_heat_loss_coefficient_too_high(self):
        """Test detection of heat_loss_coefficient corruption - value too high."""
        self.model.heat_loss_coefficient = 0.6
        assert self.model._detect_parameter_corruption() is True

    def test_detect_corrupted_outlet_effectiveness_too_low(self):
        """Test detection of outlet_effectiveness corruption - value too low."""
        self.model.outlet_effectiveness = 0.4
        assert self.model._detect_parameter_corruption() is True

    def test_detect_corrupted_outlet_effectiveness_too_high(self):
        """Test detection of outlet_effectiveness corruption - value too high."""
        self.model.outlet_effectiveness = 1.1
        assert self.model._detect_parameter_corruption() is True

    def test_detect_corrupted_thermal_time_constant_too_low(self):
        """Test detection of thermal_time_constant corruption - value too low."""
        self.model.thermal_time_constant = 2.0
        assert self.model._detect_parameter_corruption() is True

    def test_detect_corrupted_thermal_time_constant_too_high(self):
        """Test detection of thermal_time_constant corruption - value too high."""
        self.model.thermal_time_constant = 9.0
        assert self.model._detect_parameter_corruption() is True
    
    def test_detect_corrupted_learning_confidence_too_low(self):
        """Test detection of learning_confidence corruption - system gave up."""
        self.model.learning_confidence = 0.0
        assert self.model._detect_parameter_corruption() is True
    
    def test_corruption_detection_with_valid_parameters(self):
        """Test that valid parameters are NOT flagged as corrupted."""
        assert self.model._detect_parameter_corruption() is False
    
    def test_corruption_detection_boundary_values_valid(self):
        """Test boundary values that should be considered valid."""
        self.model.heat_loss_coefficient = 0.01
        self.model.outlet_effectiveness = 0.5
        self.model.thermal_time_constant = 3.0
        self.model.learning_confidence = 0.01
        assert self.model._detect_parameter_corruption() is False
        
        self.model.heat_loss_coefficient = 0.5
        self.model.outlet_effectiveness = 1.0
        self.model.thermal_time_constant = 8.0
        assert self.model._detect_parameter_corruption() is False

    def test_corruption_detection_boundary_values_invalid(self):
        """Test boundary values that should be considered corrupted."""
        self.model.heat_loss_coefficient = 0.009
        assert self.model._detect_parameter_corruption() is True
        self.model.heat_loss_coefficient = 0.1

        self.model.outlet_effectiveness = 0.49
        assert self.model._detect_parameter_corruption() is True
        self.model.outlet_effectiveness = 0.8
        
        self.model.thermal_time_constant = 2.9
        assert self.model._detect_parameter_corruption() is True
        self.model.thermal_time_constant = 5.0
        
        self.model.learning_confidence = 0.009
        assert self.model._detect_parameter_corruption() is True
        self.model.learning_confidence = 3.0

        self.model.heat_loss_coefficient = 0.51
        assert self.model._detect_parameter_corruption() is True
        self.model.heat_loss_coefficient = 0.1

        self.model.outlet_effectiveness = 1.01
        assert self.model._detect_parameter_corruption() is True
        self.model.outlet_effectiveness = 0.8

        self.model.thermal_time_constant = 8.1
        assert self.model._detect_parameter_corruption() is True
        self.model.thermal_time_constant = 5.0

class TestCorruptionDetectionIntegration:
    """Test corruption detection integration with learning system."""
    
    def setup_method(self):
        """Set up test fixtures before each test method.""" 
        global _thermal_equilibrium_model_instance
        _thermal_equilibrium_model_instance = None
        self.model = ThermalEquilibriumModel()
        
        # Mock thermal state loading to prevent file dependencies
        with patch('src.unified_thermal_state.get_thermal_state_manager'):
            pass
    
    def test_corruption_detection_called_in_update_prediction_feedback(self):
        """Test that corruption detection is called during learning updates."""
        # Arrange: Mock the corruption detection method
        mock_detect_path = '_detect_parameter_corruption'
        with patch.object(self.model, mock_detect_path,
                          return_value=True) as mock_detect:
            
            # Prepare prediction feedback data
            prediction_context = {
                'outlet_temp': 44.0,
                'current_indoor': 20.8,
                'outdoor_temp': 3.5,
                'pv_power': 0,
                'fireplace_on': 0,
                'tv_on': 0
            }
            
            # Act: Try to update prediction feedback
            self.model.update_prediction_feedback(
                predicted_temp=21.0,
                actual_temp=20.8,
                prediction_context=prediction_context,
                is_blocking_active=False
            )
            
            # Assert: Corruption detection should have been called
            mock_detect.assert_called_once()
    
    def test_learning_skipped_when_corruption_detected(self):
        """Test that learning is skipped when parameters are corrupted."""
        # Arrange: Set corrupted parameters
        self.model.heat_loss_coefficient = 0.001  # Corrupted value
        
        # Mock the parameter adaptation method to verify it's not called
        mock_adapt_path = '_adapt_parameters_from_recent_errors'
        with patch.object(self.model, mock_adapt_path) as mock_adapt:
            
            prediction_context = {
                'outlet_temp': 44.0,
                'current_indoor': 20.8, 
                'outdoor_temp': 3.5,
                'pv_power': 0,
                'fireplace_on': 0,
                'tv_on': 0
            }
            
            # Act: Try to update prediction feedback with corrupted parameters
            self.model.update_prediction_feedback(
                predicted_temp=21.0,
                actual_temp=20.8,
                prediction_context=prediction_context,
                is_blocking_active=False
            )
            
            # Assert: Parameter adaptation should NOT have been called
            mock_adapt.assert_not_called()
    
    def test_learning_proceeds_when_parameters_valid(self):
        """Test that learning proceeds normally when parameters are valid."""
        # Arrange: Set valid parameters
        self.model.heat_loss_coefficient = 0.1
        self.model.outlet_effectiveness = 0.8
        self.model.thermal_time_constant = 5.0
        self.model.learning_confidence = 3.0
        
        # Enable adaptive learning
        self.model.adaptive_learning_enabled = True
        
        # Add some prediction history to enable parameter updates
        prediction_entry = {
            'error': 0.2, 
            'context': {
                'outlet_temp': 44.0, 
                'current_indoor': 20.8, 
                'outdoor_temp': 3.5
            },
            'timestamp': '2023-01-01T00:00:00Z'
        }
        self.model.prediction_history = [prediction_entry] * self.model.recent_errors_window
        
        # Mock the parameter adaptation method to verify it IS called
        mock_adapt_path = '_adapt_parameters_from_recent_errors'
        with patch.object(self.model, mock_adapt_path) as mock_adapt:
            
            prediction_context = {
                'outlet_temp': 44.0,
                'current_indoor': 20.8,
                'outdoor_temp': 3.5,
                'pv_power': 0,
                'fireplace_on': 0, 
                'tv_on': 0
            }
            
            # Act: Update prediction feedback with valid parameters
            self.model.update_prediction_feedback(
                predicted_temp=21.0,
                actual_temp=20.8,
                prediction_context=prediction_context,
                is_blocking_active=False
            )
            
            # Assert: Parameter adaptation should have been called
            mock_adapt.assert_called_once()


if __name__ == '__main__':
    # Run the tests
    pytest.main([__file__, '-v'])
