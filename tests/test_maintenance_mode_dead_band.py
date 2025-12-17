"""
Test suite for the maintenance mode dead band enhancement.

Tests that the simplified outlet prediction provides reasonable results
for different temperature scenarios.
"""
import pandas as pd

from src.model_wrapper import simplified_outlet_prediction
from src import config


class TestMaintenanceModeDeadBand:
    """Test simplified outlet prediction functionality."""

    def test_perfect_temperature_scenario(self):
        """Test outlet prediction when at target temperature."""
        
        # Create features DataFrame
        features = pd.DataFrame([{
            'indoor_temp_lag_30m': 21.0,
            'outlet_temp': 48.0,
            'outdoor_temp': 5.0,
            'target_temp': 21.0,
        }])
        
        # Test perfect temperature match (0.000°C error)
        current_temp = 21.0  # Exactly at target
        target_temp = 21.0
        
        final_outlet_temp, confidence, metadata = simplified_outlet_prediction(
            features,
            current_temp=current_temp,
            target_temp=target_temp
        )
        
        prediction_method = metadata.get('prediction_method', 'SIMPLIFIED')
        
        # Should return reasonable outlet temperature
        assert config.CLAMP_MIN_ABS <= final_outlet_temp <= config.CLAMP_MAX_ABS
        assert confidence > 0.0  # Should have positive confidence
        assert prediction_method in ['SIMPLIFIED', 'ThermalEquilibrium', 
                                   'thermal_equilibrium_single_prediction']
        
    def test_small_temperature_difference(self):
        """Test outlet prediction for small temperature differences."""
        
        # Create features DataFrame
        features = pd.DataFrame([{
            'indoor_temp_lag_30m': 20.995,
            'outlet_temp': 45.0,
            'outdoor_temp': 5.0,
            'target_temp': 21.0,
        }])
        
        # Test very small error (0.005°C below target)
        current_temp = 20.995
        target_temp = 21.0
        
        final_outlet_temp, confidence, metadata = simplified_outlet_prediction(
            features,
            current_temp=current_temp,
            target_temp=target_temp
        )
        
        # Should return reasonable values
        assert config.CLAMP_MIN_ABS <= final_outlet_temp <= config.CLAMP_MAX_ABS
        assert confidence > 0.0
        
    def test_heating_scenario(self):
        """Test outlet prediction when heating is needed."""
        
        # Create features DataFrame
        features = pd.DataFrame([{
            'indoor_temp_lag_30m': 20.0,
            'outlet_temp': 35.0,
            'outdoor_temp': 0.0,
            'target_temp': 21.0,
        }])
        
        # Test heating scenario (1.0°C below target)
        current_temp = 20.0
        target_temp = 21.0
        
        final_outlet_temp, confidence, metadata = simplified_outlet_prediction(
            features,
            current_temp=current_temp,
            target_temp=target_temp
        )
        
        # Should return reasonable heating temperature
        assert config.CLAMP_MIN_ABS <= final_outlet_temp <= config.CLAMP_MAX_ABS
        assert confidence > 0.0
        # For heating scenario, might expect higher outlet temperature
        assert final_outlet_temp >= 25.0  # Should be reasonable for heating
        
    def test_cooling_scenario(self):
        """Test outlet prediction when cooling is needed."""
        
        # Create features DataFrame
        features = pd.DataFrame([{
            'indoor_temp_lag_30m': 22.0,
            'outlet_temp': 45.0,
            'outdoor_temp': 20.0,
            'target_temp': 21.0,
        }])
        
        # Test cooling scenario (1.0°C above target)
        current_temp = 22.0
        target_temp = 21.0
        
        final_outlet_temp, confidence, metadata = simplified_outlet_prediction(
            features,
            current_temp=current_temp,
            target_temp=target_temp
        )
        
        # Should return reasonable cooling temperature
        assert config.CLAMP_MIN_ABS <= final_outlet_temp <= config.CLAMP_MAX_ABS
        assert confidence > 0.0
        
    def test_boundary_temperature_scenarios(self):
        """Test outlet prediction at temperature boundaries."""
        
        # Test different scenarios
        scenarios = [
            {'current': 20.99, 'target': 21.0, 'outdoor': 5.0},
            {'current': 21.01, 'target': 21.0, 'outdoor': 5.0},
            {'current': 19.5, 'target': 21.0, 'outdoor': -5.0},
            {'current': 22.5, 'target': 21.0, 'outdoor': 25.0},
        ]
        
        for scenario in scenarios:
            features = pd.DataFrame([{
                'indoor_temp_lag_30m': scenario['current'],
                'outlet_temp': 40.0,
                'outdoor_temp': scenario['outdoor'],
                'target_temp': scenario['target'],
            }])
            
            final_outlet_temp, confidence, metadata = simplified_outlet_prediction(
                features,
                current_temp=scenario['current'],
                target_temp=scenario['target']
            )
            
            # All scenarios should return reasonable results
            assert config.CLAMP_MIN_ABS <= final_outlet_temp <= config.CLAMP_MAX_ABS
            assert confidence > 0.0
            assert isinstance(metadata, dict)
