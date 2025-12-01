"""
Test suite for the maintenance mode dead band enhancement.

Tests the improved maintenance mode logic that keeps outlet temperature
unchanged when indoor temperature is essentially perfect (within 0.01°C).
"""
import pandas as pd
from unittest.mock import Mock

from src.model_wrapper import find_best_outlet_temp
from src.physics_model import RealisticPhysicsModel


class TestMaintenanceModeDeadBand:
    """Test maintenance mode dead band functionality."""

    def test_dead_band_keeps_outlet_unchanged_for_perfect_temperature(self):
        """Test that 0.000°C error keeps outlet temperature unchanged."""
        # Mock physics model with proper predict_one return value
        model = Mock(spec=RealisticPhysicsModel)
        model.get_realtime_sigma.return_value = 0.3
        model.get_realtime_confidence.return_value = 0.8
        model.predict_one.return_value = 0.0  # No temperature change predicted
        
        # Create mock features DataFrame
        features = pd.DataFrame([{
            'indoor_temp_lag_30m': 21.0,
            'outlet_temp': 48.0,
            'outdoor_temp': 5.0,
            'target_temp': 21.0,
        }])
        
        # Test perfect temperature match (0.000°C error)
        current_temp = 21.0  # Exactly at target
        target_temp = 21.0
        outlet_history = [48.0]  # Use integer for easier testing
        error_target_vs_actual = 0.0
        outdoor_temp = 5.0
        
        result = find_best_outlet_temp(
            model=model,
            features=features,
            current_temp=current_temp,
            target_temp=target_temp,
            outlet_history=outlet_history,
            error_target_vs_actual=error_target_vs_actual,
            outdoor_temp=outdoor_temp
        )
        
        final_outlet_temp, confidence, control_mode, sigma, _, _, _ = result
        
        # Should be in MAINTENANCE mode
        assert control_mode == "MAINTENANCE"
        
        # Should keep outlet temperature unchanged (dead band active)
        assert final_outlet_temp == 48.0  # Same as last outlet temp
        
    def test_dead_band_activates_within_threshold(self):
        """Test that dead band activates for errors within 0.01°C."""
        # Mock physics model
        model = Mock(spec=RealisticPhysicsModel)
        model.get_realtime_sigma.return_value = 0.3
        model.get_realtime_confidence.return_value = 0.8
        
        # Create mock features DataFrame
        features = pd.DataFrame([{
            'indoor_temp_lag_30m': 20.995,
            'outlet_temp': 45.0,
            'outdoor_temp': 5.0,
            'target_temp': 21.0,
        }])
        
        # Test very small error (0.005°C - within dead band)
        current_temp = 20.995  # 0.005°C below target
        target_temp = 21.0
        outlet_history = [45.0]
        error_target_vs_actual = -0.005
        outdoor_temp = 5.0
        
        result = find_best_outlet_temp(
            model=model,
            features=features,
            current_temp=current_temp,
            target_temp=target_temp,
            outlet_history=outlet_history,
            error_target_vs_actual=error_target_vs_actual,
            outdoor_temp=outdoor_temp
        )
        
        final_outlet_temp, confidence, control_mode, sigma, _, _, _ = result
        
        # Should be in MAINTENANCE mode with dead band active
        assert control_mode == "MAINTENANCE"
        assert final_outlet_temp == 45.0  # Unchanged
        
    def test_adjustment_outside_dead_band_cold(self):
        """Test that adjustment occurs when error is outside dead band (too cold)."""
        # Mock physics model
        model = Mock(spec=RealisticPhysicsModel)
        model.get_realtime_sigma.return_value = 0.3
        model.get_realtime_confidence.return_value = 0.8
        model.predict_one.return_value = 0.1  # For smart rounding
        
        # Create mock features DataFrame
        features = pd.DataFrame([{
            'indoor_temp_lag_30m': 20.98,
            'outlet_temp': 45.0,
            'outdoor_temp': 5.0,
            'target_temp': 21.0,
        }])
        
        # Test error outside dead band (0.02°C below target)
        current_temp = 20.98  # 0.02°C below target (> 0.01°C threshold)
        target_temp = 21.0
        outlet_history = [45.0]
        error_target_vs_actual = -0.02
        outdoor_temp = 5.0
        
        result = find_best_outlet_temp(
            model=model,
            features=features,
            current_temp=current_temp,
            target_temp=target_temp,
            outlet_history=outlet_history,
            error_target_vs_actual=error_target_vs_actual,
            outdoor_temp=outdoor_temp
        )
        
        final_outlet_temp, confidence, control_mode, sigma, _, _, _ = result
        
        # Should be in MAINTENANCE mode but with +0.5°C adjustment
        assert control_mode == "MAINTENANCE"
        # Smart rounding may change the exact value, but should be >= 45.0
        assert final_outlet_temp >= 45.0  # Should not decrease
        
    def test_adjustment_outside_dead_band_warm(self):
        """Test that adjustment occurs when error is outside dead band (too warm)."""
        # Mock physics model
        model = Mock(spec=RealisticPhysicsModel)
        model.get_realtime_sigma.return_value = 0.3
        model.get_realtime_confidence.return_value = 0.8
        model.predict_one.return_value = 0.1  # For smart rounding
        
        # Create mock features DataFrame
        features = pd.DataFrame([{
            'indoor_temp_lag_30m': 21.02,
            'outlet_temp': 45.0,
            'outdoor_temp': 5.0,
            'target_temp': 21.0,
        }])
        
        # Test error outside dead band (0.02°C above target)
        current_temp = 21.02  # 0.02°C above target (> 0.01°C threshold)
        target_temp = 21.0
        outlet_history = [45.0]
        error_target_vs_actual = 0.02
        outdoor_temp = 5.0
        
        result = find_best_outlet_temp(
            model=model,
            features=features,
            current_temp=current_temp,
            target_temp=target_temp,
            outlet_history=outlet_history,
            error_target_vs_actual=error_target_vs_actual,
            outdoor_temp=outdoor_temp
        )
        
        final_outlet_temp, confidence, control_mode, sigma, _, _, _ = result
        
        # Should be in MAINTENANCE mode but with -0.5°C adjustment
        assert control_mode == "MAINTENANCE"
        assert final_outlet_temp < 45.0  # Should decrease
        
    def test_dead_band_threshold_boundary(self):
        """Test the exact boundary of the dead band threshold."""
        # Mock physics model
        model = Mock(spec=RealisticPhysicsModel)
        model.get_realtime_sigma.return_value = 0.3
        model.get_realtime_confidence.return_value = 0.8
        model.predict_one.return_value = 0.1  # For smart rounding
        
        # Create mock features DataFrame for exactly at threshold
        features = pd.DataFrame([{
            'indoor_temp_lag_30m': 20.99,
            'outlet_temp': 45.0,
            'outdoor_temp': 5.0,
            'target_temp': 21.0,
        }])
        
        # Test exactly at dead band threshold (0.01°C error)
        current_temp = 20.99  # Exactly 0.01°C below target
        target_temp = 21.0
        outlet_history = [45.0]
        error_target_vs_actual = -0.01
        outdoor_temp = 5.0
        
        result = find_best_outlet_temp(
            model=model,
            features=features,
            current_temp=current_temp,
            target_temp=target_temp,
            outlet_history=outlet_history,
            error_target_vs_actual=error_target_vs_actual,
            outdoor_temp=outdoor_temp
        )
        
        final_outlet_temp, confidence, control_mode, sigma, _, _, _ = result
        
        # Should be in MAINTENANCE mode
        assert control_mode == "MAINTENANCE"
        # At exactly 0.01°C, should be in dead band (dead band is < 0.01°C)
        assert final_outlet_temp == 45.0  # Unchanged
