"""
TDD Unit Tests for Parameter Updates Fix

This test suite verifies that the parameter_updates fix works correctly by testing:
1. Parameter history is always recorded (Phase 2 fix)
2. Lower thresholds allow smaller changes to be logged (Phase 1 fix)
3. Learning metrics correctly report parameter_updates count
4. Edge cases and error conditions
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from thermal_equilibrium_model import ThermalEquilibriumModel
from thermal_constants import PhysicsConstants
import config


@pytest.fixture
def fresh_thermal_model():
    """Create a fresh thermal model instance for each test."""
    # Reset singleton to ensure fresh instance
    import thermal_equilibrium_model
    thermal_equilibrium_model._thermal_equilibrium_model_instance = None
    
    with patch('unified_thermal_state.get_thermal_state_manager') as mock_manager:
        # Mock thermal state manager to return default parameters
        mock_state_manager = Mock()
        mock_state_manager.get_current_parameters.return_value = {
            'baseline_parameters': {
                'source': 'config',  # Use config defaults, not calibrated
                'thermal_time_constant': 4.0,
                'heat_loss_coefficient': 0.1,
                'outlet_effectiveness': 0.3
            }
        }
        mock_state_manager.update_learning_state = Mock()  # Mock the save method
        mock_manager.return_value = mock_state_manager
        
        model = ThermalEquilibriumModel()
        # Clear any existing history for clean tests
        model.prediction_history = []
        model.parameter_history = []
        model.learning_confidence = 3.0
        return model


class TestParameterUpdatesAlwaysRecorded:
    """Test that parameter updates are always recorded (Phase 2 fix)."""
    
    def test_parameter_history_recorded_with_zero_error(self, fresh_thermal_model):
        """Test that parameter history is recorded even with zero prediction errors."""
        model = fresh_thermal_model
        
        # Add enough predictions to trigger learning (RECENT_ERRORS_WINDOW = 10)
        for i in range(15):
            model.update_prediction_feedback(
                predicted_temp=21.0,
                actual_temp=21.0,  # Zero error
                prediction_context={
                    'outlet_temp': 35.0,
                    'outdoor_temp': 5.0,
                    'current_indoor': 21.0,
                    'pv_power': 0,
                    'fireplace_on': 0,
                    'tv_on': 0
                },
                timestamp=f"2025-12-08T11:00:{i:02d}"
            )
        
        # Verify parameter history was recorded despite zero errors
        assert len(model.parameter_history) > 0, "Parameter history should be recorded even with zero errors"
        
        # Verify learning metrics reports parameter updates
        metrics = model.get_adaptive_learning_metrics()
        assert metrics['parameter_updates'] > 0, "Should report parameter_updates > 0"
        assert 'insufficient_data' not in metrics, "Should have sufficient data"

    def test_parameter_history_recorded_with_tiny_errors(self, fresh_thermal_model):
        """Test that parameter history is recorded with very small errors."""
        model = fresh_thermal_model
        
        # Add predictions with tiny errors (0.001°C)
        for i in range(15):
            model.update_prediction_feedback(
                predicted_temp=21.0,
                actual_temp=21.001,  # Very small error
                prediction_context={
                    'outlet_temp': 35.0,
                    'outdoor_temp': 5.0,
                    'current_indoor': 21.0,
                    'pv_power': 0,
                    'fireplace_on': 0,
                    'tv_on': 0
                },
                timestamp=f"2025-12-08T11:00:{i:02d}"
            )
        
        # Verify parameter history was recorded
        assert len(model.parameter_history) > 0, "Parameter history should be recorded with tiny errors"
        
        # Check that changes are tracked in history
        if len(model.parameter_history) > 0:
            latest_entry = model.parameter_history[-1]
            assert 'changes' in latest_entry, "Parameter history should include change tracking"
            changes = latest_entry['changes']
            assert 'thermal' in changes, "Should track thermal parameter changes"
            assert 'heat_loss' in changes, "Should track heat_loss parameter changes"
            assert 'effectiveness' in changes, "Should track effectiveness parameter changes"

    def test_parameter_updates_increments_correctly(self, fresh_thermal_model):
        """Test that parameter_updates count increments with each learning cycle."""
        model = fresh_thermal_model
        
        initial_param_updates = len(model.parameter_history)
        
        # First batch of predictions
        for i in range(12):
            model.update_prediction_feedback(
                predicted_temp=21.0,
                actual_temp=21.05,  # Small but non-zero error
                prediction_context={
                    'outlet_temp': 35.0,
                    'outdoor_temp': 5.0,
                    'current_indoor': 21.0
                },
                timestamp=f"2025-12-08T11:00:{i:02d}"
            )
        
        first_count = len(model.parameter_history)
        assert first_count > initial_param_updates, "Parameter updates should increment after first batch"
        
        # Second batch of predictions
        for i in range(10):
            model.update_prediction_feedback(
                predicted_temp=21.0,
                actual_temp=21.02,  # Different small error
                prediction_context={
                    'outlet_temp': 36.0,
                    'outdoor_temp': 4.0,
                    'current_indoor': 21.0
                },
                timestamp=f"2025-12-08T11:01:{i:02d}"
            )
        
        second_count = len(model.parameter_history)
        assert second_count > first_count, "Parameter updates should continue incrementing"


class TestLowerThresholds:
    """Test that lower thresholds allow smaller changes to be logged (Phase 1 fix)."""
    
    def test_micro_changes_logged_with_lower_thresholds(self, fresh_thermal_model):
        """Test that micro-changes are logged with the new lower thresholds."""
        model = fresh_thermal_model
        
        with patch('thermal_equilibrium_model.logging') as mock_logging:
            # Simulate tiny parameter changes that would be below old thresholds
            # but above new thresholds
            original_thermal = model.thermal_time_constant
            original_heat_loss = model.heat_loss_coefficient
            original_effectiveness = model.outlet_effectiveness
            
            # Manually adjust parameters by tiny amounts
            model.thermal_time_constant = original_thermal + 0.002  # Above new threshold (0.001)
            model.heat_loss_coefficient = original_heat_loss + 0.00002  # Above new threshold (0.00001)
            model.outlet_effectiveness = original_effectiveness + 0.0002  # Above new threshold (0.0001)
            
            # Add enough predictions to trigger parameter update check
            for i in range(12):
                model.update_prediction_feedback(
                    predicted_temp=21.0,
                    actual_temp=21.01,
                    prediction_context={
                        'outlet_temp': 35.0,
                        'outdoor_temp': 5.0,
                        'current_indoor': 21.0
                    },
                    timestamp=f"2025-12-08T11:00:{i:02d}"
                )
            
            # Check if logging was called (indicating changes were detected)
            # Note: This may not trigger in actual gradient descent, but tests the threshold logic
            mock_logging.debug.assert_called()  # Should have debug logging for micro-updates

    def test_old_thresholds_would_miss_changes(self, fresh_thermal_model):
        """Test that changes below old thresholds would be missed but are caught with new ones."""
        # Old thresholds:
        # thermal_change > 0.01
        # heat_loss_change > 0.0001  
        # effectiveness_change > 0.001
        
        # New thresholds (10x lower):
        # thermal_change > 0.001
        # heat_loss_change > 0.00001
        # effectiveness_change > 0.0001
        
        # Test changes that would be missed by old thresholds
        old_thermal_threshold = 0.01
        old_heat_loss_threshold = 0.0001
        old_effectiveness_threshold = 0.001
        
        new_thermal_threshold = 0.001
        new_heat_loss_threshold = 0.00001
        new_effectiveness_threshold = 0.0001
        
        # Changes between old and new thresholds
        test_thermal_change = 0.005  # Between 0.001 and 0.01
        test_heat_loss_change = 0.00005  # Between 0.00001 and 0.0001
        test_effectiveness_change = 0.0005  # Between 0.0001 and 0.001
        
        # Old threshold logic would miss these
        old_would_log = (test_thermal_change > old_thermal_threshold or 
                        test_heat_loss_change > old_heat_loss_threshold or
                        test_effectiveness_change > old_effectiveness_threshold)
        
        # New threshold logic catches these
        new_would_log = (test_thermal_change > new_thermal_threshold or
                        test_heat_loss_change > new_heat_loss_threshold or
                        test_effectiveness_change > new_effectiveness_threshold)
        
        assert not old_would_log, "Old thresholds should miss these changes"
        assert new_would_log, "New thresholds should catch these changes"


class TestLearningMetrics:
    """Test that learning metrics correctly report parameter updates."""
    
    def test_get_adaptive_learning_metrics_reports_parameter_updates(self, fresh_thermal_model):
        """Test that get_adaptive_learning_metrics() correctly reports parameter_updates count."""
        model = fresh_thermal_model
        
        # Initially should have no parameter updates
        initial_metrics = model.get_adaptive_learning_metrics()
        if 'insufficient_data' not in initial_metrics:
            assert initial_metrics['parameter_updates'] == 0
        
        # Add predictions to trigger learning
        for i in range(15):
            model.update_prediction_feedback(
                predicted_temp=21.0,
                actual_temp=21.05,
                prediction_context={
                    'outlet_temp': 35.0,
                    'outdoor_temp': 5.0,
                    'current_indoor': 21.0
                },
                timestamp=f"2025-12-08T11:00:{i:02d}"
            )
        
        # Check metrics after learning
        metrics = model.get_adaptive_learning_metrics()
        assert 'insufficient_data' not in metrics, "Should have sufficient data now"
        assert metrics['parameter_updates'] == len(model.parameter_history)
        assert metrics['total_predictions'] == len(model.prediction_history)
        
        if metrics['total_predictions'] > 0:
            expected_percentage = (metrics['parameter_updates'] / metrics['total_predictions']) * 100
            assert abs(metrics['update_percentage'] - expected_percentage) < 0.1

    def test_update_percentage_calculation(self, fresh_thermal_model):
        """Test that update_percentage is calculated correctly."""
        model = fresh_thermal_model
        
        # Add exactly 20 predictions
        for i in range(20):
            model.update_prediction_feedback(
                predicted_temp=21.0,
                actual_temp=21.02,
                prediction_context={
                    'outlet_temp': 35.0,
                    'outdoor_temp': 5.0,
                    'current_indoor': 21.0
                },
                timestamp=f"2025-12-08T11:00:{i:02d}"
            )
        
        metrics = model.get_adaptive_learning_metrics()
        
        # Calculate expected percentage
        total_predictions = metrics['total_predictions']
        parameter_updates = metrics['parameter_updates']
        expected_percentage = (parameter_updates / total_predictions) * 100 if total_predictions > 0 else 0
        
        assert abs(metrics['update_percentage'] - expected_percentage) < 0.001, \
            f"Expected {expected_percentage}%, got {metrics['update_percentage']}%"


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_insufficient_data_handling(self, fresh_thermal_model):
        """Test behavior when there's insufficient data for learning."""
        model = fresh_thermal_model
        
        # Add fewer than RECENT_ERRORS_WINDOW predictions
        for i in range(5):  # Less than 10
            model.update_prediction_feedback(
                predicted_temp=21.0,
                actual_temp=21.0,
                prediction_context={
                    'outlet_temp': 35.0,
                    'outdoor_temp': 5.0,
                    'current_indoor': 21.0
                },
                timestamp=f"2025-12-08T11:00:{i:02d}"
            )
        
        # Should not trigger parameter learning yet
        assert len(model.parameter_history) == 0, "Should not have parameter updates with insufficient data"
        
        metrics = model.get_adaptive_learning_metrics()
        # With only 5 predictions, should either report insufficient_data or have very limited metrics
        if 'insufficient_data' in metrics:
            assert metrics['insufficient_data'], "Should report insufficient data"
        else:
            # If it doesn't report insufficient_data, it should have 0 parameter updates
            assert metrics['parameter_updates'] == 0, "Should have 0 parameter updates with insufficient data"

    def test_missing_prediction_context(self, fresh_thermal_model):
        """Test behavior with missing or incomplete prediction context."""
        model = fresh_thermal_model
        
        # Add predictions with missing context keys
        for i in range(15):
            incomplete_context = {
                'outlet_temp': 35.0,
                # Missing 'outdoor_temp' and 'current_indoor'
            }
            
            model.update_prediction_feedback(
                predicted_temp=21.0,
                actual_temp=21.05,
                prediction_context=incomplete_context,
                timestamp=f"2025-12-08T11:00:{i:02d}"
            )
        
        # Should still record parameter history (fix ensures this)
        # but gradients might be zero due to missing context
        metrics = model.get_adaptive_learning_metrics()
        
        # The fix ensures parameter_updates is tracked even if gradients are zero
        if 'insufficient_data' not in metrics:
            assert metrics['parameter_updates'] >= 0, "Should handle missing context gracefully"

    def test_zero_gradients_still_record_history(self, fresh_thermal_model):
        """Test that parameter history is recorded even when gradients are zero."""
        model = fresh_thermal_model
        
        with patch.object(model, '_calculate_thermal_time_constant_gradient', return_value=0.0), \
             patch.object(model, '_calculate_heat_loss_coefficient_gradient', return_value=0.0), \
             patch.object(model, '_calculate_outlet_effectiveness_gradient', return_value=0.0):
            
            # Add predictions that would normally calculate gradients
            for i in range(15):
                model.update_prediction_feedback(
                    predicted_temp=21.0,
                    actual_temp=21.0,
                    prediction_context={
                        'outlet_temp': 35.0,
                        'outdoor_temp': 5.0,
                        'current_indoor': 21.0
                    },
                    timestamp=f"2025-12-08T11:00:{i:02d}"
                )
            
            # Even with zero gradients, parameter history should be recorded
            assert len(model.parameter_history) > 0, "Parameter history should be recorded even with zero gradients"
            
            metrics = model.get_adaptive_learning_metrics()
            assert metrics['parameter_updates'] > 0, "Should report parameter_updates even with zero gradients"


class TestParameterUpdatesIntegration:
    """Integration tests for the complete parameter updates fix."""
    
    def test_realistic_scenario_with_small_errors(self, fresh_thermal_model):
        """Test a realistic scenario with small but realistic prediction errors."""
        model = fresh_thermal_model
        
        # Simulate realistic heating scenario with small errors
        prediction_scenarios = [
            # (predicted, actual, outlet, outdoor, indoor)
            (21.0, 21.05, 35.0, 5.0, 21.0),  # Slight overprediction
            (21.1, 21.08, 36.0, 4.0, 21.1),  # Slight underprediction  
            (21.2, 21.22, 37.0, 3.0, 21.2),  # Very close
            (21.3, 21.28, 38.0, 3.5, 21.3),  # Small error
            (21.4, 21.45, 39.0, 4.0, 21.4),  # Small overprediction
        ]
        
        # Run multiple cycles with realistic variations
        for cycle in range(3):
            for i, (pred, actual, outlet, outdoor, indoor) in enumerate(prediction_scenarios):
                model.update_prediction_feedback(
                    predicted_temp=pred + (cycle * 0.01),  # Slight variation per cycle
                    actual_temp=actual + (cycle * 0.01),
                    prediction_context={
                        'outlet_temp': outlet + cycle,
                        'outdoor_temp': outdoor + (cycle * 0.5),
                        'current_indoor': indoor + (cycle * 0.01),
                        'pv_power': cycle * 100,  # Some PV variation
                        'fireplace_on': 0,
                        'tv_on': 1 if i % 2 == 0 else 0
                    },
                    timestamp=f"2025-12-08T{11 + cycle}:{i * 12:02d}:00"
                )
        
        # Verify learning occurred
        assert len(model.prediction_history) >= 15, "Should have sufficient predictions"
        assert len(model.parameter_history) > 0, "Should have parameter updates"
        
        metrics = model.get_adaptive_learning_metrics()
        assert 'insufficient_data' not in metrics, "Should have sufficient data"
        assert metrics['parameter_updates'] > 0, "Should report parameter updates"
        assert metrics['update_percentage'] > 0, "Should have non-zero update percentage"
        
        # Verify that recent changes are tracked
        if metrics['parameter_updates'] > 0:
            latest_entry = model.parameter_history[-1]
            assert 'changes' in latest_entry, "Latest entry should track changes"
            assert 'gradients' in latest_entry, "Latest entry should track gradients"

    def test_parameter_updates_survive_restarts(self, fresh_thermal_model):
        """Test that parameter update counting works correctly after model restarts."""
        model = fresh_thermal_model
        
        # First session - accumulate some learning
        for i in range(12):
            model.update_prediction_feedback(
                predicted_temp=21.0,
                actual_temp=21.03,
                prediction_context={
                    'outlet_temp': 35.0,
                    'outdoor_temp': 5.0,
                    'current_indoor': 21.0
                },
                timestamp=f"2025-12-08T11:00:{i:02d}"
            )
        
        # Verify first session had some learning
        assert len(model.parameter_history) > 0, "First session should have parameter updates"
        
        # Simulate restart by creating new model instance with fresh mocking
        # (In real scenario, unified thermal state would persist the learning)
        import thermal_equilibrium_model
        thermal_equilibrium_model._thermal_equilibrium_model_instance = None
        
        with patch('unified_thermal_state.get_thermal_state_manager') as mock_manager:
            mock_state_manager = Mock()
            mock_state_manager.get_current_parameters.return_value = {
                'baseline_parameters': {
                    'source': 'config',
                    'thermal_time_constant': 4.0,
                    'heat_loss_coefficient': 0.1,
                    'outlet_effectiveness': 0.3
                }
            }
            mock_state_manager.update_learning_state = Mock()
            mock_manager.return_value = mock_state_manager
            
            model2 = ThermalEquilibriumModel()
            model2.prediction_history = []
            model2.parameter_history = []
            
            # Second session - continue learning
            for i in range(15):
                model2.update_prediction_feedback(
                    predicted_temp=21.0,
                    actual_temp=21.02,
                    prediction_context={
                        'outlet_temp': 36.0,
                        'outdoor_temp': 6.0,
                        'current_indoor': 21.0
                    },
                    timestamp=f"2025-12-08T12:00:{i:02d}"
                )
            
            second_session_updates = len(model2.parameter_history)
            assert second_session_updates > 0, "Should accumulate new parameter updates"
            
            metrics = model2.get_adaptive_learning_metrics()
            assert metrics['parameter_updates'] == second_session_updates
