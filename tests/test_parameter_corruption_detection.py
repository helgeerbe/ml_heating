"""
Unit tests for parameter corruption detection in ThermalEquilibriumModel.

These tests ensure the system can detect when thermal parameters have been
corrupted and need to be reset to prevent catastrophic prediction errors.

TDD Approach: Tests written FIRST, then implementation follows.
"""

import pytest
from unittest.mock import patch

# Import the model we're testing
from src.thermal_equilibrium_model import ThermalEquilibriumModel


class TestParameterCorruptionDetection:
    """Test parameter corruption detection functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create fresh model instance for each test
        global _thermal_equilibrium_model_instance
        _thermal_equilibrium_model_instance = None
        self.model = ThermalEquilibriumModel()
        
        # Set known good parameters as baseline
        self.model.equilibrium_ratio = 0.408
        self.model.total_conductance = 0.194
        self.model.learning_confidence = 3.0
        
        # Mock thermal state loading to prevent file system dependencies
        with patch('src.unified_thermal_state.get_thermal_state_manager'):
            pass
    
    def test_detect_corrupted_equilibrium_ratio_too_low(self):
        """Test detection of equilibrium_ratio corruption - value too low."""
        # Arrange: Set corrupted parameter (known bad value from real issue)
        self.model.equilibrium_ratio = 0.1  # Should be ~0.8, this is corrupted
        
        # Act & Assert: Should detect corruption
        assert self.model._detect_parameter_corruption() is True
    
    def test_detect_corrupted_equilibrium_ratio_too_high(self):
        """Test detection of equilibrium_ratio corruption - value too high."""
        # Arrange: Set unrealistic high value
        # Above reasonable physics bounds
        self.model.equilibrium_ratio = 0.95
        
        # Act & Assert: Should detect corruption
        assert self.model._detect_parameter_corruption() is True
    
    def test_detect_corrupted_total_conductance_too_low(self):
        """Test detection of total_conductance corruption - value too low."""
        # Arrange: Set unrealistically low conductance
        # Too low for realistic building
        self.model.total_conductance = 0.01
        
        # Act & Assert: Should detect corruption
        assert self.model._detect_parameter_corruption() is True
    
    def test_detect_corrupted_total_conductance_too_high(self):
        """Test detection of total_conductance corruption - value too high."""
        # Arrange: Set corrupted parameter (known bad value from real issue)
        # Should be ~0.05, this is corrupted
        self.model.total_conductance = 0.266
        
        # Act & Assert: Should detect corruption
        assert self.model._detect_parameter_corruption() is True
    
    def test_detect_corrupted_learning_confidence_too_low(self):
        """Test detection of learning_confidence corruption - system gave up."""
        # Arrange: Set corrupted confidence (known bad value from real issue)
        self.model.learning_confidence = 0.0  # System gave up learning
        
        # Act & Assert: Should detect corruption
        assert self.model._detect_parameter_corruption() is True
    
    def test_corruption_detection_with_valid_parameters(self):
        """Test that valid parameters are NOT flagged as corrupted."""
        # Arrange: Parameters are already set to good values in setup
        # equilibrium_ratio = 0.408, total_conductance = 0.194,
        # learning_confidence = 3.0
        
        # Act & Assert: Should NOT detect corruption
        assert self.model._detect_parameter_corruption() is False
    
    def test_corruption_detection_boundary_values_valid(self):
        """Test boundary values that should be considered valid."""
        # Test minimum valid values
        # Just above corruption threshold
        self.model.equilibrium_ratio = 0.3
        # Just above corruption threshold
        self.model.total_conductance = 0.02
        # Just above corruption threshold
        self.model.learning_confidence = 0.1
        
        # Act & Assert: Should NOT detect corruption
        assert self.model._detect_parameter_corruption() is False
        
        # Test maximum valid values
        self.model.equilibrium_ratio = 0.9  # Just below corruption threshold
        self.model.total_conductance = 0.3  # Just below corruption threshold
        self.model.learning_confidence = 10.0  # High but valid confidence
        
        # Act & Assert: Should NOT detect corruption
        assert self.model._detect_parameter_corruption() is False
    
    def test_corruption_detection_boundary_values_invalid(self):
        """Test boundary values that should be considered corrupted."""
        # Test values just below minimum thresholds
        self.model.equilibrium_ratio = 0.29  # Just below valid threshold
        assert self.model._detect_parameter_corruption() is True
        
        # Reset to good values and test next parameter
        self.model.equilibrium_ratio = 0.408
        self.model.total_conductance = 0.019  # Just below valid threshold
        assert self.model._detect_parameter_corruption() is True
        
        # Reset to good values and test next parameter  
        self.model.total_conductance = 0.194
        self.model.learning_confidence = 0.009  # Just below valid threshold
        assert self.model._detect_parameter_corruption() is True
        
        # Reset to good values and test upper bounds
        self.model.learning_confidence = 3.0
        self.model.equilibrium_ratio = 0.91  # Just above valid threshold
        assert self.model._detect_parameter_corruption() is True
        
        # Reset and test conductance upper bound
        self.model.equilibrium_ratio = 0.408  
        self.model.total_conductance = 0.31  # Just above valid threshold
        assert self.model._detect_parameter_corruption() is True
    
    def test_real_world_corrupted_values_detection(self):
        """Test detection using actual corrupted values from production issue."""
        # Arrange: Set exact corrupted values that caused original problem
        self.model.equilibrium_ratio = 0.1      # Original corrupted value
        self.model.total_conductance = 0.266    # Original corrupted value
        self.model.learning_confidence = 0.0    # Original corrupted value
        
        # Act & Assert: Should definitely detect this corruption
        assert self.model._detect_parameter_corruption() is True
    
    def test_known_good_calibrated_values_not_corrupted(self):
        """Test that known good calibrated values are not flagged as corrupt."""
        # Arrange: Set the exact calibrated values that work correctly
        # Known good calibrated value
        self.model.equilibrium_ratio = 0.408
        # Known good calibrated value
        self.model.total_conductance = 0.194
        # Known good calibrated value
        self.model.learning_confidence = 3.0
        
        # Act & Assert: Should NOT detect corruption
        assert self.model._detect_parameter_corruption() is False


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
        self.model.equilibrium_ratio = 0.1  # Corrupted value
        
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
        self.model.equilibrium_ratio = 0.408
        self.model.total_conductance = 0.194
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
            }
        }
        self.model.prediction_history = [
            prediction_entry,
            {'error': 0.1, 'context': prediction_entry['context']},
            {'error': -0.1, 'context': prediction_entry['context']},
        ] * 4  # Enough entries to trigger parameter adaptation
        
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
