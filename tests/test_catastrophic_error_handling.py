"""
Unit tests for catastrophic error handling in ThermalEquilibriumModel.

These tests ensure the system stops learning when prediction errors are too
large (>5°C), indicating the model is fundamentally broken and should not
continue adapting parameters.

TDD Approach: Tests written FIRST, then implementation follows.
"""

import pytest
from unittest.mock import patch

# Import the model we're testing
from src.thermal_equilibrium_model import ThermalEquilibriumModel


class TestCatastrophicErrorHandling:
    """Test catastrophic error detection and learning shutdown."""
    
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
        
        # Enable adaptive learning for testing
        self.model.adaptive_learning_enabled = True
        
        # Mock thermal state loading to prevent file system dependencies
        manager_path = 'src.unified_thermal_state.get_thermal_state_manager'
        with patch(manager_path):
            pass
    
    def test_learning_disabled_for_errors_over_5_degrees(self):
        """Test that learning rate becomes 0 for errors > 5°C."""
        # Arrange: Add catastrophic error to history
        self.model.prediction_history = [
            {'error': 6.0, 'context': {'outlet_temp': 44.0}},  # Catastrophic
        ]
        
        # Act: Calculate learning rate with catastrophic error
        learning_rate = self.model._calculate_adaptive_learning_rate()
        
        # Assert: Learning rate should be zero
        assert learning_rate == 0.0
    
    def test_learning_disabled_for_errors_over_10_degrees(self):
        """Test that learning rate becomes 0 for errors > 10°C."""
        # Arrange: Add massive catastrophic error to history
        self.model.prediction_history = [
            {'error': 12.0, 'context': {'outlet_temp': 44.0}},  # Massive error
        ]
        
        # Act: Calculate learning rate with massive error
        learning_rate = self.model._calculate_adaptive_learning_rate()
        
        # Assert: Learning rate should be zero
        assert learning_rate == 0.0
    
    def test_learning_rate_zero_for_catastrophic_errors(self):
        """Test exact boundary case where errors = 5.0°C."""
        # Arrange: Add exactly 5.0°C error to history
        self.model.prediction_history = [
            {'error': 5.0, 'context': {'outlet_temp': 44.0}},  # Exactly 5.0°C
        ]
        
        # Act: Calculate learning rate at boundary
        learning_rate = self.model._calculate_adaptive_learning_rate()
        
        # Assert: Learning rate should be zero (5.0 should trigger shutdown)
        assert learning_rate == 0.0
    
    def test_normal_learning_for_small_errors(self):
        """Test that normal learning proceeds for small errors < 5°C."""
        # Arrange: Add normal small errors to history
        self.model.prediction_history = [
            {'error': 0.5, 'context': {'outlet_temp': 44.0}},  # Small error
            {'error': 1.2, 'context': {'outlet_temp': 44.0}},  # Small error
            {'error': -0.8, 'context': {'outlet_temp': 44.0}},  # Small error
        ]
        
        # Act: Calculate learning rate with normal errors
        learning_rate = self.model._calculate_adaptive_learning_rate()
        
        # Assert: Learning rate should be positive (normal learning)
        assert learning_rate > 0.0
    
    def test_learning_disabled_for_mixed_errors_with_catastrophic(self):
        """Test that any catastrophic error disables learning."""
        # Arrange: Mix of normal and catastrophic errors
        self.model.prediction_history = [
            {'error': 0.5, 'context': {'outlet_temp': 44.0}},  # Normal
            {'error': 8.0, 'context': {'outlet_temp': 44.0}},  # Catastrophic
            {'error': 1.0, 'context': {'outlet_temp': 44.0}},  # Normal
        ]
        
        # Act: Calculate learning rate with mixed errors
        learning_rate = self.model._calculate_adaptive_learning_rate()
        
        # Assert: Should disable learning due to catastrophic error
        assert learning_rate == 0.0
    
    def test_boundary_case_just_under_catastrophic_threshold(self):
        """Test that errors just under 5.0°C still allow learning."""
        # Arrange: Add error just under the threshold
        self.model.prediction_history = [
            {'error': 4.9, 'context': {'outlet_temp': 44.0}},  # Just under 5.0
        ]
        
        # Act: Calculate learning rate
        learning_rate = self.model._calculate_adaptive_learning_rate()
        
        # Assert: Should still allow learning
        assert learning_rate > 0.0
    
    def test_real_world_catastrophic_error_from_production(self):
        """Test with actual 12°C error that occurred in production."""
        # Arrange: Use the actual error values from the production issue
        # Model predicted ~8.8°C, actual was ~20.8°C = 12°C error
        self.model.prediction_history = [
            {'error': 12.0, 'context': {'outlet_temp': 65.0}},  # Real error
        ]
        
        # Act: Calculate learning rate with real production error
        learning_rate = self.model._calculate_adaptive_learning_rate()
        
        # Assert: Should completely disable learning
        assert learning_rate == 0.0


class TestCatastrophicErrorLogging:
    """Test that catastrophic errors are properly logged."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        global _thermal_equilibrium_model_instance
        _thermal_equilibrium_model_instance = None
        self.model = ThermalEquilibriumModel()
        
        # Set known good parameters
        self.model.equilibrium_ratio = 0.408
        self.model.total_conductance = 0.194
        self.model.learning_confidence = 3.0
        
        # Mock thermal state loading
        with patch('src.unified_thermal_state.get_thermal_state_manager'):
            pass
    
    def test_catastrophic_error_logged_with_warning(self):
        """Test that catastrophic errors generate warning logs."""
        # Arrange: Prepare catastrophic error
        self.model.prediction_history = [
            {'error': 8.5, 'context': {'outlet_temp': 44.0}},
        ]
        
        # Mock logging to capture warning
        with patch('src.thermal_equilibrium_model.logging.warning') as mock_log:
            # Act: Calculate learning rate (should trigger warning)
            self.model._calculate_adaptive_learning_rate()
            
            # Assert: Should have logged a catastrophic error warning
            mock_log.assert_called_once()
            call_args = mock_log.call_args[0][0]  # Get the log message
            assert 'Catastrophic error' in call_args
            assert '8.5' in call_args
            assert 'learning DISABLED' in call_args
    
    def test_normal_errors_do_not_generate_warnings(self):
        """Test that normal errors don't generate catastrophic warnings."""
        # Arrange: Prepare normal errors
        self.model.prediction_history = [
            {'error': 1.2, 'context': {'outlet_temp': 44.0}},
        ]
        
        # Mock logging to capture warnings
        with patch('src.thermal_equilibrium_model.logging.warning') as mock_log:
            # Act: Calculate learning rate (should not trigger warning)
            self.model._calculate_adaptive_learning_rate()
            
            # Assert: Should not have logged any catastrophic error warnings
            # Filter out any calls that aren't about catastrophic errors
            catastrophic_calls = [
                call for call in mock_log.call_args_list 
                if 'Catastrophic error' in str(call)
            ]
            assert len(catastrophic_calls) == 0


class TestCatastrophicErrorIntegration:
    """Test catastrophic error handling integration with learning system."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        global _thermal_equilibrium_model_instance
        _thermal_equilibrium_model_instance = None
        self.model = ThermalEquilibriumModel()
        
        # Enable adaptive learning
        self.model.adaptive_learning_enabled = True
        
        # Mock thermal state loading
        with patch('src.unified_thermal_state.get_thermal_state_manager'):
            pass
    
    def test_parameter_updates_blocked_by_catastrophic_errors(self):
        """Test that parameter updates are blocked when catastrophic errors exist."""
        # Arrange: Set up prediction history with catastrophic error
        self.model.prediction_history = [
            {'error': 10.0, 'context': {'outlet_temp': 44.0}},  # Catastrophic
        ] * 12  # Enough entries to normally trigger parameter updates
        
        # Mock parameter adaptation to verify it's not called
        with patch.object(self.model, '_adapt_parameters_from_recent_errors') as mock_adapt:
            
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
            
            # Assert: Parameter adaptation should not be called
            mock_adapt.assert_not_called()
    
    def test_parameter_updates_proceed_with_normal_errors(self):
        """Test that parameter updates proceed normally with small errors."""
        # Arrange: Set up prediction history with normal errors
        self.model.prediction_history = [
            {'error': 0.5, 'context': {'outlet_temp': 44.0}},
            {'error': 1.2, 'context': {'outlet_temp': 44.0}},
            {'error': -0.8, 'context': {'outlet_temp': 44.0}},
        ] * 4  # Enough entries to trigger parameter updates
        
        # Set valid parameters to pass corruption detection
        self.model.equilibrium_ratio = 0.408
        self.model.total_conductance = 0.194
        self.model.learning_confidence = 3.0
        
        # Mock parameter adaptation to verify it IS called
        with patch.object(self.model, '_adapt_parameters_from_recent_errors') as mock_adapt:
            
            prediction_context = {
                'outlet_temp': 44.0,
                'current_indoor': 20.8,
                'outdoor_temp': 3.5,
                'pv_power': 0,
                'fireplace_on': 0,
                'tv_on': 0
            }
            
            # Act: Update prediction feedback with normal errors
            self.model.update_prediction_feedback(
                predicted_temp=21.0,
                actual_temp=20.8,
                prediction_context=prediction_context,
                is_blocking_active=False
            )
            
            # Assert: Parameter adaptation should be called
            mock_adapt.assert_called_once()
    
    def test_learning_recovery_after_catastrophic_errors_clear(self):
        """Test that learning can resume after catastrophic errors clear."""
        # Arrange: Start with catastrophic errors
        self.model.prediction_history = [
            {'error': 12.0, 'context': {'outlet_temp': 44.0}},  # Catastrophic
        ]
        
        # Verify learning is disabled
        assert self.model._calculate_adaptive_learning_rate() == 0.0
        
        # Clear catastrophic errors and add normal ones
        self.model.prediction_history = [
            {'error': 0.5, 'context': {'outlet_temp': 44.0}},   # Normal
            {'error': 1.0, 'context': {'outlet_temp': 44.0}},   # Normal
        ]
        
        # Act: Calculate learning rate after clearing errors
        learning_rate = self.model._calculate_adaptive_learning_rate()
        
        # Assert: Learning should resume
        assert learning_rate > 0.0


if __name__ == '__main__':
    # Run the tests
    pytest.main([__file__, '-v'])
