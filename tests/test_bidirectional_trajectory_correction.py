"""
Unit tests for bidirectional trajectory correction enhancement.

Tests the enhanced PRIORITY 3 logic that detects both temperature drops 
and rises using sensor-aligned 0.1°C boundaries.
"""
import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_wrapper import get_enhanced_model_wrapper


class TestBidirectionalTrajectoryCorrection:
    """Test enhanced PRIORITY 3 bidirectional trajectory correction."""
    
    def setup_method(self):
        """Set up test environment."""
        self.wrapper = get_enhanced_model_wrapper()
        self.base_thermal_features = {
            'pv_power': 0.0,
            'fireplace_on': 0.0,
            'tv_on': 0.0
        }
    
    @patch('model_wrapper.config.TRAJECTORY_PREDICTION_ENABLED', True)
    def test_temperature_drop_detection_boundary_exact(self):
        """Test detection of temperature drop at exact boundary (20.9°C)."""
        # Mock trajectory with temperature dropping to exactly 20.9°C
        mock_trajectory = {
            'reaches_target_at': None,
            'equilibrium_temp': 21.0,
            'trajectory': [21.0, 20.95, 20.9, 20.9]  # min = 20.9
        }
        
        with patch.object(self.wrapper.thermal_model, 'predict_thermal_trajectory', return_value=mock_trajectory):
            result = self.wrapper._verify_trajectory_and_correct(
                outlet_temp=35.0,
                current_indoor=21.0,
                target_indoor=21.0,
                outdoor_temp=5.0,
                thermal_features=self.base_thermal_features
            )
            
        # Should trigger correction since 20.9 <= 21.0 - 0.1
        assert result > 35.0, "Should apply correction for temperature drop to boundary"
    
    @patch('model_wrapper.config.TRAJECTORY_PREDICTION_ENABLED', True)
    def test_temperature_drop_detection_below_boundary(self):
        """Test detection of temperature drop below boundary (20.8°C)."""
        # Mock trajectory with temperature dropping below boundary
        mock_trajectory = {
            'reaches_target_at': None,
            'equilibrium_temp': 21.0,
            'trajectory': [21.0, 20.9, 20.8, 20.8]  # min = 20.8
        }
        
        with patch.object(self.wrapper.thermal_model, 'predict_thermal_trajectory', return_value=mock_trajectory):
            result = self.wrapper._verify_trajectory_and_correct(
                outlet_temp=35.0,
                current_indoor=21.0,
                target_indoor=21.0,
                outdoor_temp=5.0,
                thermal_features=self.base_thermal_features
            )
            
        # Should trigger correction since 20.8 <= 21.0 - 0.1
        assert result > 35.0, "Should apply correction for temperature drop below boundary"
    
    @patch('model_wrapper.config.TRAJECTORY_PREDICTION_ENABLED', True)
    def test_temperature_rise_detection_boundary_exact(self):
        """Test detection of temperature rise at exact boundary (21.1°C)."""
        # Mock trajectory with temperature rising to exactly 21.1°C
        mock_trajectory = {
            'reaches_target_at': None,
            'equilibrium_temp': 21.0,
            'trajectory': [21.0, 21.05, 21.1, 21.1]  # max = 21.1
        }
        
        with patch.object(self.wrapper.thermal_model, 'predict_thermal_trajectory', return_value=mock_trajectory):
            result = self.wrapper._verify_trajectory_and_correct(
                outlet_temp=35.0,
                current_indoor=21.0,
                target_indoor=21.0,
                outdoor_temp=5.0,
                thermal_features=self.base_thermal_features
            )
            
        # Should trigger correction since 21.1 >= 21.0 + 0.1
        assert result < 35.0, "Should apply negative correction for temperature rise to boundary"
    
    @patch('model_wrapper.config.TRAJECTORY_PREDICTION_ENABLED', True)
    def test_temperature_rise_detection_above_boundary(self):
        """Test detection of temperature rise above boundary (21.2°C)."""
        # Mock trajectory with temperature rising above boundary
        mock_trajectory = {
            'reaches_target_at': None,
            'equilibrium_temp': 21.0,
            'trajectory': [21.0, 21.1, 21.2, 21.2]  # max = 21.2
        }
        
        with patch.object(self.wrapper.thermal_model, 'predict_thermal_trajectory', return_value=mock_trajectory):
            result = self.wrapper._verify_trajectory_and_correct(
                outlet_temp=35.0,
                current_indoor=21.0,
                target_indoor=21.0,
                outdoor_temp=5.0,
                thermal_features=self.base_thermal_features
            )
            
        # Should trigger correction since 21.2 >= 21.0 + 0.1
        assert result < 35.0, "Should apply negative correction for temperature rise above boundary"
    
    @patch('model_wrapper.config.TRAJECTORY_PREDICTION_ENABLED', True)
    def test_no_correction_within_boundaries(self):
        """Test no correction when temperature stays within boundaries."""
        # Mock trajectory with temperature staying within ±0.1°C
        mock_trajectory = {
            'reaches_target_at': None,
            'equilibrium_temp': 21.0,
            'trajectory': [21.0, 20.95, 21.05, 21.0]  # min=20.95, max=21.05
        }
        
        with patch.object(self.wrapper.thermal_model, 'predict_thermal_trajectory', return_value=mock_trajectory):
            result = self.wrapper._verify_trajectory_and_correct(
                outlet_temp=35.0,
                current_indoor=21.0,
                target_indoor=21.0,
                outdoor_temp=5.0,
                thermal_features=self.base_thermal_features
            )
            
        # Should NOT trigger correction since within boundaries
        assert result == 35.0, "Should not apply correction when within boundaries"
    
    @patch('model_wrapper.config.TRAJECTORY_PREDICTION_ENABLED', True)
    def test_priority_1_overrides_boundary_checks_when_fast(self):
        """Test that PRIORITY 1 (target reached within 1h) overrides boundary checks."""
        # Mock trajectory that reaches target quickly (should override boundary checks)
        mock_trajectory = {
            'reaches_target_at': 0.8,  # Target reached at 0.8 hours (fast enough)
            'equilibrium_temp': 21.0,
            'trajectory': [21.0, 20.8, 20.9, 21.0]  # Contains boundary violation
        }
        
        with patch.object(self.wrapper.thermal_model, 'predict_thermal_trajectory', return_value=mock_trajectory):
            result = self.wrapper._verify_trajectory_and_correct(
                outlet_temp=35.0,
                current_indoor=21.0,
                target_indoor=21.0,
                outdoor_temp=5.0,
                thermal_features=self.base_thermal_features
            )
            
        # Should NOT apply correction due to PRIORITY 1
        assert result == 35.0, "PRIORITY 1 should override boundary checks when target reached quickly"
    
    @patch('model_wrapper.config.TRAJECTORY_PREDICTION_ENABLED', True)
    def test_priority_1_does_not_override_when_too_slow(self):
        """Test that PRIORITY 1 does NOT override when target reached too slowly (>1h)."""
        # Mock trajectory that reaches target slowly (should NOT override boundary checks)
        mock_trajectory = {
            'reaches_target_at': 2.5,  # Target reached at 2.5 hours (too slow)
            'equilibrium_temp': 21.0,
            'trajectory': [21.0, 20.8, 20.8, 20.8]  # Contains boundary violation
        }
        
        with patch.object(self.wrapper.thermal_model, 'predict_thermal_trajectory', return_value=mock_trajectory):
            result = self.wrapper._verify_trajectory_and_correct(
                outlet_temp=35.0,
                current_indoor=21.0,
                target_indoor=21.0,
                outdoor_temp=5.0,
                thermal_features=self.base_thermal_features
            )
            
        # Should apply correction because target reached too slowly
        assert result > 35.0, "Should apply correction when target reached too slowly"
    
    @patch('model_wrapper.config.TRAJECTORY_PREDICTION_ENABLED', True)
    def test_priority_2_handles_boundary_violations(self):
        """Test that PRIORITY 2 handles both unreachable targets and boundary violations."""
        # Mock trajectory with unreachable target AND boundary violation
        mock_trajectory = {
            'reaches_target_at': None,
            'equilibrium_temp': 20.5,  # Significantly below target
            'trajectory': [21.0, 21.2, 21.1, 20.5]  # Also has rise and drop violations
        }
        
        with patch.object(self.wrapper.thermal_model, 'predict_thermal_trajectory', return_value=mock_trajectory):
            result = self.wrapper._verify_trajectory_and_correct(
                outlet_temp=35.0,
                current_indoor=21.0,
                target_indoor=21.0,
                outdoor_temp=5.0,
                thermal_features=self.base_thermal_features
            )
            
        # Should apply PRIORITY 2 correction (large increase for unreachable target)
        assert result > 40.0, "PRIORITY 2 should handle unreachable targets with strong correction"
    
    @patch('model_wrapper.config.TRAJECTORY_PREDICTION_ENABLED', True)
    def test_internal_precision_with_sensor_boundaries(self):
        """Test internal precision values with sensor-aligned boundaries."""
        # Mock trajectory with high-precision internal values
        mock_trajectory = {
            'reaches_target_at': None,
            'equilibrium_temp': 21.0,
            'trajectory': [21.0, 20.876, 20.923, 20.987]  # min = 20.876 (high precision)
        }
        
        with patch.object(self.wrapper.thermal_model, 'predict_thermal_trajectory', return_value=mock_trajectory):
            result = self.wrapper._verify_trajectory_and_correct(
                outlet_temp=35.0,
                current_indoor=21.0,
                target_indoor=21.0,
                outdoor_temp=5.0,
                thermal_features=self.base_thermal_features
            )
            
        # Should trigger correction since 20.876 <= 20.9 (21.0 - 0.1)
        assert result > 35.0, "Should handle internal precision with sensor boundaries"
    
    @patch('model_wrapper.config.TRAJECTORY_PREDICTION_ENABLED', True)
    def test_correction_reason_messages(self):
        """Test correct reason messages for different scenarios."""
        # Test temperature drop reason
        mock_trajectory_drop = {
            'reaches_target_at': None,
            'equilibrium_temp': 21.0,
            'trajectory': [21.0, 20.8]  # Drop scenario
        }
        
        # Test temperature rise reason  
        mock_trajectory_rise = {
            'reaches_target_at': None,
            'equilibrium_temp': 21.0,
            'trajectory': [21.0, 21.2]  # Rise scenario
        }
        
        # We can't easily test logging output, but we can verify the method executes
        # without errors for both scenarios
        with patch.object(self.wrapper.thermal_model, 'predict_thermal_trajectory', return_value=mock_trajectory_drop):
            result_drop = self.wrapper._verify_trajectory_and_correct(
                outlet_temp=35.0, current_indoor=21.0, target_indoor=21.0,
                outdoor_temp=5.0, thermal_features=self.base_thermal_features
            )
            
        with patch.object(self.wrapper.thermal_model, 'predict_thermal_trajectory', return_value=mock_trajectory_rise):
            result_rise = self.wrapper._verify_trajectory_and_correct(
                outlet_temp=35.0, current_indoor=21.0, target_indoor=21.0,
                outdoor_temp=5.0, thermal_features=self.base_thermal_features
            )
            
        assert result_drop > 35.0, "Drop scenario should increase outlet temperature"
        assert result_rise < 35.0, "Rise scenario should decrease outlet temperature"
    
    @patch('model_wrapper.config.TRAJECTORY_PREDICTION_ENABLED', True)
    def test_different_target_temperatures(self):
        """Test boundary logic with different target temperatures."""
        test_cases = [
            (20.0, 19.8, True),   # Target 20.0°C, min 19.8°C → correction
            (20.0, 19.9, True),   # Target 20.0°C, min 19.9°C → correction (boundary)
            (20.0, 20.0, False),  # Target 20.0°C, min 20.0°C → no correction
            (22.5, 22.3, True),   # Target 22.5°C, min 22.3°C → correction
            (22.5, 22.4, True),   # Target 22.5°C, min 22.4°C → correction (boundary)
            (22.5, 22.5, False),  # Target 22.5°C, min 22.5°C → no correction
        ]
        
        for target, min_temp, should_correct in test_cases:
            mock_trajectory = {
                'reaches_target_at': None,
                'equilibrium_temp': target,
                'trajectory': [target, min_temp]
            }
            
            with patch.object(self.wrapper.thermal_model, 'predict_thermal_trajectory', return_value=mock_trajectory):
                result = self.wrapper._verify_trajectory_and_correct(
                    outlet_temp=35.0,
                    current_indoor=target,
                    target_indoor=target,
                    outdoor_temp=5.0,
                    thermal_features=self.base_thermal_features
                )
                
            if should_correct:
                assert result != 35.0, f"Should correct for target={target}, min={min_temp}"
            else:
                assert result == 35.0, f"Should not correct for target={target}, min={min_temp}"


class TestBidirectionalCorrectionIntegration:
    """Integration tests for bidirectional correction in full control flow."""
    
    def setup_method(self):
        """Set up test environment."""
        self.wrapper = get_enhanced_model_wrapper()
    
    @patch('model_wrapper.config.TRAJECTORY_PREDICTION_ENABLED', True)
    def test_full_outlet_calculation_with_trajectory_correction(self):
        """Test full outlet temperature calculation with trajectory correction."""
        features = {
            'indoor_temp_lag_30m': 21.0,
            'target_temp': 21.0,
            'outdoor_temp': 5.0,
            'pv_now': 0.0,
            'fireplace_on': 0,
            'tv_on': 0
        }
        
        # Mock binary search convergence
        with patch.object(self.wrapper.thermal_model, 'predict_equilibrium_temperature', return_value=21.0):
            # Mock trajectory with boundary violation
            mock_trajectory = {
                'reaches_target_at': None,
                'equilibrium_temp': 21.0,
                'trajectory': [21.0, 20.8]  # Temperature drop
            }
            
            with patch.object(self.wrapper.thermal_model, 'predict_thermal_trajectory', return_value=mock_trajectory):
                outlet_temp, metadata = self.wrapper.calculate_optimal_outlet_temp(features)
                
        # Should return corrected outlet temperature
        assert outlet_temp is not None
        assert isinstance(outlet_temp, float)
        assert metadata['prediction_method'] == 'thermal_equilibrium_single_prediction'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
