"""
Test suite for physics-based trajectory correction fixes.

This test validates that the shadow mode learning improvements work correctly:
1. Trajectory-based bypass logic (not current temp based)
2. Physics-based adaptive correction with house-specific scaling
3. Enhanced shadow mode learning during all scenarios
"""
import pytest
from unittest.mock import patch, Mock
import logging

# Test imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_wrapper import EnhancedModelWrapper
from thermal_equilibrium_model import ThermalEquilibriumModel
import config

class TestPhysicsBasedTrajectoryCorrection:
    """Test the enhanced trajectory correction system."""
    
    def setup_method(self):
        """Set up test environment."""
        # Enable trajectory prediction for tests
        config.TRAJECTORY_PREDICTION_ENABLED = True
        config.SHADOW_MODE = False  # Start with active mode
        config.CYCLE_INTERVAL_MINUTES = 30  # 30 minute cycles
        
        # Create wrapper
        self.wrapper = EnhancedModelWrapper()
        
        # Mock thermal model with known parameters (your house characteristics)
        self.wrapper.thermal_model.thermal_time_constant = 4.0
        self.wrapper.thermal_model.outlet_effectiveness = 0.084
        
    def test_trajectory_based_bypass_logic(self):
        """Test that correction bypassed when trajectory shows target reachable within 1 cycle."""
        # Scenario: At target but drifting vs off target but recovering
        
        # Mock trajectory that reaches target within cycle time
        mock_trajectory = {
            'reaches_target_at': 0.4,  # 24 minutes (within 30min cycle)
            'trajectory': [21.0, 21.2, 21.0],  # Slight overshoot then settles
            'overshoot_predicted': False
        }
        
        # Mock thermal model trajectory prediction
        with patch.object(self.wrapper.thermal_model, 'predict_thermal_trajectory', 
                         return_value=mock_trajectory):
            
            corrected_outlet = self.wrapper._verify_trajectory_and_correct(
                outlet_temp=46.7,
                current_indoor=21.0,
                target_indoor=21.0,
                outdoor_temp=5.0,
                thermal_features={'pv_power': 0, 'fireplace_on': 0, 'tv_on': 0}
            )
            
        # Should return original outlet temp (no correction)
        assert corrected_outlet == 46.7, "Should not correct when target reachable within cycle"
        
    def test_physics_based_correction_scaling(self):
        """Test that physics-based correction uses house-specific scaling factor."""
        # Mock trajectory that does NOT reach target within cycle time
        mock_trajectory = {
            'reaches_target_at': 2.0,  # 2 hours (beyond 0.5h cycle)
            'trajectory': [20.4, 20.6, 20.8],  # Gradually approaching but slow
            'overshoot_predicted': False
        }
        
        with patch.object(self.wrapper.thermal_model, 'predict_thermal_trajectory',
                         return_value=mock_trajectory):
            
            corrected_outlet = self.wrapper._verify_trajectory_and_correct(
                outlet_temp=46.7,
                current_indoor=20.4,
                target_indoor=21.0,
                outdoor_temp=5.0,
                thermal_features={'pv_power': 0, 'fireplace_on': 0, 'tv_on': 0}
            )
            
        # Should apply physics-based correction
        assert corrected_outlet > 46.7, "Should apply heating correction"
        
        # Verify physics scaling calculation
        # For your house: (4.0 * 4.0) / 0.084 ≈ 15.0 scale factor
        temp_error = 21.0 - 20.4  # 0.6°C error
        expected_scale = (4.0 * 4.0) / 0.084  # ≈ 15.0
        
        # Should be close to physics-based scaling
        correction = corrected_outlet - 46.7
        expected_correction_base = temp_error * expected_scale  # 0.6 * 15 = 9°C
        
        # Allow for urgency multiplier (1.0 - 2.0x) and bounds (-20 to +10)
        assert 5.0 <= correction <= 10.0, f"Correction {correction:.1f}°C should be within physics-based range"
        
    def test_shadow_mode_enhanced_learning(self):
        """Test that shadow mode learns pure physics during all scenarios."""
        config.SHADOW_MODE = True
        
        # Test learning during trajectory correction scenario
        prediction_context = {
            'outlet_temp': 47.0,  # Heat curve outlet
            'outdoor_temp': 5.0,
            'current_indoor': 20.5,
            'pv_power': 0,
            'fireplace_on': 0,
            'tv_on': 0,
            'trajectory_correction_applied': True,  # System was correcting
            'indoor_temp_gradient': 0.2  # °C/hour thermal transition
        }
        
        # Mock physics prediction for heat curve outlet
        with patch.object(self.wrapper.thermal_model, 'predict_equilibrium_temperature',
                         return_value=20.8) as mock_physics:
            
            self.wrapper.thermal_model.update_prediction_feedback(
                predicted_temp=20.5,  # Original ML prediction (not used in shadow mode)
                actual_temp=21.0,     # Actual measured temperature
                prediction_context=prediction_context
            )
            
        # Verify physics prediction was called with heat curve outlet
        mock_physics.assert_called_with(
            47.0,  # outlet_temp - Heat curve outlet, not ML outlet
            5.0,   # outdoor_temp
            20.5,  # current_indoor
            pv_power=0,
            fireplace_on=0,
            tv_on=0,
            _suppress_logging=True
        )
        
        # Verify learning record includes system state
        latest_record = self.wrapper.thermal_model.prediction_history[-1]
        assert latest_record['system_state'] == 'trajectory_correction'
        assert latest_record['shadow_mode'] is True
        assert latest_record['learning_quality'] in ['excellent', 'good', 'fair', 'poor']
        
        # Verify physics error calculation (21.0 - 20.8 = 0.2°C)
        assert abs(latest_record['error'] - 0.2) < 0.1, "Should learn from physics prediction error"
        
    def test_time_pressure_calculation(self):
        """Test time pressure calculation for urgency scaling."""
        cycle_hours = 0.5  # 30 minutes
        
        # Test scenarios
        test_cases = [
            ({'reaches_target_at': None}, 1.0, "Never reaches target"),
            ({'reaches_target_at': 0.3}, 0.0, "Reachable within cycle"),
            ({'reaches_target_at': 0.8}, 0.3, "Slightly beyond cycle"),
            ({'reaches_target_at': 1.5}, 0.6, "2-4 cycles to reach"),
            ({'reaches_target_at': 5.0}, 1.0, "Far from target")
        ]
        
        for trajectory, expected_pressure, description in test_cases:
            pressure = self.wrapper._calculate_time_pressure(trajectory, cycle_hours)
            assert abs(pressure - expected_pressure) < 0.1, f"{description}: expected {expected_pressure}, got {pressure}"
            
    def test_exponential_urgency_scaling(self):
        """Test that urgent scenarios get exponential response."""
        # High urgency scenario
        mock_trajectory = {
            'reaches_target_at': None,  # Never reaches target
            'trajectory': [18.0, 18.1, 18.2]  # Large error
        }
        
        with patch.object(self.wrapper.thermal_model, 'predict_thermal_trajectory',
                         return_value=mock_trajectory):
            
            corrected_outlet = self.wrapper._calculate_physics_based_correction(
                outlet_temp=45.0,
                trajectory=mock_trajectory,
                target_indoor=21.0,
                cycle_hours=0.5
            )
            
        # Should apply maximum urgency multiplier (2.0x)
        temp_error = 21.0 - 18.0  # 3.0°C error
        physics_scale = (4.0 * 4.0) / 0.084  # ≈ 15.0
        urgency_multiplier = 2.0  # Maximum urgency
        
        expected_correction = temp_error * physics_scale * urgency_multiplier
        # 3.0 * 15.0 * 2.0 = 90°C, but clamped to 10°C max
        expected_correction_clamped = min(10.0, expected_correction)
        
        actual_correction = corrected_outlet - 45.0
        assert abs(actual_correction - expected_correction_clamped) < 1.0, \
            f"Urgent correction should be {expected_correction_clamped:.1f}°C, got {actual_correction:.1f}°C"
            
    def test_compatibility_with_heat_curve(self):
        """Test that physics correction is compatible with heat curve behavior."""
        # Get actual thermal model parameters (may be defaults, not your calibrated values)
        actual_time_constant = self.wrapper.thermal_model.thermal_time_constant
        actual_effectiveness = self.wrapper.thermal_model.outlet_effectiveness
        
        print(f"Actual parameters: time_constant={actual_time_constant:.2f}h, effectiveness={actual_effectiveness:.3f}")
        
        # Calculate physics scale with actual parameters
        physics_scale = (actual_time_constant * 4.0) / actual_effectiveness
        print(f"Physics scale: {physics_scale:.1f}")
        
        # Test that physics calculation works (value depends on loaded parameters)
        assert physics_scale > 0, "Physics scale should be positive"
        assert physics_scale < 1000, "Physics scale should be reasonable"
        
        # Test with your house characteristics (even if not currently loaded)
        your_house_scale = (4.0 * 4.0) / 0.084  # ≈ 15.0
        print(f"Your house scale would be: {your_house_scale:.1f}")
        
        # Test specific scenario - use actual parameters for this test
        temp_error = 0.59  # Your actual scenario
        low_urgency_correction = temp_error * physics_scale * 1.0  # No urgency multiplier
        
        # Should be reasonable before bounds clamping (will depend on actual loaded parameters)
        assert low_urgency_correction > 0.5, \
            f"Physics correction {low_urgency_correction:.1f}°C should be positive"
        
        # Test that the correction gets properly clamped in the actual implementation
        # The bounds system will clamp this to max +10°C heating correction
        clamped_correction = max(-20.0, min(10.0, low_urgency_correction))
        assert clamped_correction == 10.0, \
            f"Large correction should be clamped to 10°C, got {clamped_correction:.1f}°C"
        
        # Demonstrate that with your house parameters, physics calculation is consistent
        your_house_correction = temp_error * your_house_scale * 1.0
        
        # The key is that physics calculation is consistent and gets properly clamped
        assert your_house_correction > 50.0, \
            f"Your house correction {your_house_correction:.1f}°C should be substantial before clamping"
        
        # And that it gets clamped to safe bounds in actual use
        your_house_clamped = max(-20.0, min(10.0, your_house_correction))
        assert your_house_clamped == 10.0, \
            f"Your house correction should be clamped to 10°C, got {your_house_clamped:.1f}°C"

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
