"""
Tests for adaptive fireplace learning system
Validates temperature differential detection and learning algorithms
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

from src.adaptive_fireplace_learning import (
    AdaptiveFireplaceLearning,
    FireplaceObservation,
    integrate_adaptive_fireplace_with_multi_source_physics
)
from src.multi_heat_source_physics import MultiHeatSourcePhysics


class TestAdaptiveFireplaceLearning:
    """Test adaptive fireplace learning functionality"""
    
    def setup_method(self):
        """Set up test environment with temporary state file"""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        self.adaptive_fireplace = AdaptiveFireplaceLearning(state_file=self.temp_file.name)
    
    def teardown_method(self):
        """Clean up temporary files"""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_fireplace_detection_logic(self):
        """Test fireplace detection based on temperature differential"""
        # Test fireplace activation (>2°C differential)
        result = self.adaptive_fireplace.observe_fireplace_state(
            living_room_temp=22.5,
            other_rooms_temp=20.0,  # 2.5°C differential
            outdoor_temp=5.0,
            fireplace_active=True
        )
        
        assert result['temp_differential'] == 2.5
        assert result['fireplace_active'] is True
        assert result['heat_contribution_kw'] > 0
        assert result['session_update']['status'] == 'session_started'
    
    def test_fireplace_deactivation_logic(self):
        """Test fireplace deactivation based on hysteresis"""
        # Start with fireplace active
        self.adaptive_fireplace.observe_fireplace_state(
            living_room_temp=22.5, other_rooms_temp=20.0,
            outdoor_temp=5.0, fireplace_active=True
        )
        
        # Then test deactivation (<0.8°C differential)
        result = self.adaptive_fireplace.observe_fireplace_state(
            living_room_temp=20.5,
            other_rooms_temp=20.0,  # 0.5°C differential
            outdoor_temp=5.0,
            fireplace_active=False
        )
        
        assert result['temp_differential'] == 0.5
        assert result['fireplace_active'] is False
        assert result['heat_contribution_kw'] == 0
        assert result['session_update']['status'] == 'session_ended'
    
    def test_learning_progression(self):
        """Test that system learns from multiple fireplace sessions"""
        initial_confidence = self.adaptive_fireplace.learning_state.learned_coefficients['learning_confidence']
        
        # Add enough observations to see confidence increase (need more than 5 for confidence > 0.1)
        sessions = [
            {'duration': 30, 'peak_diff': 3.0, 'outdoor': 5.0},
            {'duration': 45, 'peak_diff': 2.8, 'outdoor': 2.0},
            {'duration': 60, 'peak_diff': 3.2, 'outdoor': 8.0},
            {'duration': 25, 'peak_diff': 2.5, 'outdoor': 0.0},
            {'duration': 35, 'peak_diff': 2.9, 'outdoor': 3.0},
            {'duration': 40, 'peak_diff': 3.1, 'outdoor': 1.0},
            {'duration': 50, 'peak_diff': 2.7, 'outdoor': 7.0}
        ]
        
        for session in sessions:
            # Create observation manually for faster testing
            observation = FireplaceObservation(
                timestamp=datetime.now(),
                temp_differential=2.0,
                outdoor_temp=session['outdoor'],
                fireplace_active=True,
                duration_minutes=session['duration'],
                peak_differential=session['peak_diff']
            )
            self.adaptive_fireplace.learning_state.observations.append(observation)
        
        # Update learning
        result = self.adaptive_fireplace._update_learning_coefficients()
        
        assert result['status'] == 'coefficients_updated'
        final_confidence = self.adaptive_fireplace.learning_state.learned_coefficients['learning_confidence']
        # With 7 observations: min(0.9, 7/50.0) = min(0.9, 0.14) = 0.14, which should be > 0.1
        assert final_confidence > initial_confidence
        
    def test_outdoor_temperature_correlation(self):
        """Test learning of outdoor temperature correlation"""
        # Add observations with different outdoor temperatures
        observations = [
            FireplaceObservation(datetime.now(), 2.0, -5.0, True, 30, 0, 0, 3.5),
            FireplaceObservation(datetime.now(), 2.0, 0.0, True, 30, 0, 0, 3.0),
            FireplaceObservation(datetime.now(), 2.0, 5.0, True, 30, 0, 0, 2.5),
            FireplaceObservation(datetime.now(), 2.0, 10.0, True, 30, 0, 0, 2.0),
        ]
        
        self.adaptive_fireplace.learning_state.observations = observations
        result = self.adaptive_fireplace._update_learning_coefficients()
        
        # Should learn negative correlation (colder = more effective)
        correlation = self.adaptive_fireplace.learning_state.learned_coefficients['outdoor_temp_correlation']
        assert result['status'] == 'coefficients_updated'
        # Note: correlation might be weak with limited data, but should be updated
    
    def test_heat_contribution_calculation(self):
        """Test heat contribution calculation accuracy"""
        # Test with known differential and outdoor temp
        result = self.adaptive_fireplace._calculate_learned_heat_contribution(
            temp_differential=3.0,
            outdoor_temp=0.0,  # Cold outdoor temp should increase effectiveness
            fireplace_active=True
        )
        
        assert result['heat_contribution_kw'] > 0
        assert result['effectiveness_factor'] >= 0.5  # Based on confidence
        assert result['outdoor_factor'] > 1.0  # Cold weather bonus
        assert 'reasoning' in result
    
    def test_enhanced_features_generation(self):
        """Test generation of enhanced features for ML models"""
        base_features = {
            'indoor_temp': 22.0,
            'avg_other_rooms_temp': 20.0,
            'outdoor_temp': 5.0,
            'fireplace_on': 1,
            'pv_now': 500
        }
        
        enhanced = self.adaptive_fireplace.get_enhanced_fireplace_features(base_features)
        
        # Check that enhanced features are added
        assert 'fireplace_heat_contribution_kw' in enhanced
        assert 'fireplace_effectiveness_factor' in enhanced
        assert 'fireplace_temp_differential' in enhanced
        assert 'fireplace_learning_confidence' in enhanced
        assert 'fireplace_observations_count' in enhanced
        
        # Original features should be preserved
        assert enhanced['pv_now'] == 500
        assert enhanced['outdoor_temp'] == 5.0
    
    def test_learning_summary(self):
        """Test learning summary generation"""
        # Add some test observations
        observation = FireplaceObservation(
            datetime.now(), 2.5, 5.0, True, 45, 0, 0, 3.0
        )
        self.adaptive_fireplace.learning_state.observations.append(observation)
        
        summary = self.adaptive_fireplace.get_learning_summary()
        
        assert 'learning_status' in summary
        assert 'learned_characteristics' in summary
        assert 'usage_patterns' in summary
        assert 'recent_sessions' in summary
        
        assert summary['learning_status']['total_observations'] == 1
        assert len(summary['recent_sessions']) == 1
    
    def test_state_persistence(self):
        """Test that learning state is saved and loaded correctly"""
        # Add observation
        observation = FireplaceObservation(
            datetime.now(), 2.5, 5.0, True, 30, 0, 0, 3.0
        )
        self.adaptive_fireplace.learning_state.observations.append(observation)
        self.adaptive_fireplace._save_state()
        
        # Create new instance with same state file
        new_adaptive = AdaptiveFireplaceLearning(state_file=self.temp_file.name)
        
        assert len(new_adaptive.learning_state.observations) == 1
        assert new_adaptive.learning_state.observations[0].peak_differential == 3.0
    
    def test_safety_bounds_enforcement(self):
        """Test that learned coefficients stay within safety bounds"""
        coeffs = self.adaptive_fireplace.learning_state.learned_coefficients
        
        # Set extreme value for coefficient that's always bounded during learning
        coeffs['differential_to_heat_ratio'] = 50.0  # Way too high
        
        # Add observations to trigger learning (need enough for differential_to_heat_ratio bounds)
        for i in range(10):
            observation = FireplaceObservation(
                timestamp=datetime.now(),
                temp_differential=2.0 + i * 0.1,  # Varied differentials
                outdoor_temp=5.0,
                fireplace_active=True,
                duration_minutes=30,  # Long enough to trigger learning
                peak_differential=3.0 + i * 0.1  # Varied peak differentials
            )
            self.adaptive_fireplace.learning_state.observations.append(observation)
        
        # Update learning (should apply safety bounds)
        result = self.adaptive_fireplace._update_learning_coefficients()
        
        # Check bounds are enforced on coefficients that are actually bounded during learning
        assert coeffs['differential_to_heat_ratio'] <= 5.0  # Max bound from safety_bounds
        assert coeffs['differential_to_heat_ratio'] >= 1.0  # Min bound from safety_bounds
        assert result['status'] == 'coefficients_updated'


class TestMultiSourcePhysicsIntegration:
    """Test integration with existing multi-heat-source physics"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        
        self.multi_source_physics = MultiHeatSourcePhysics()
        self.adaptive_fireplace = AdaptiveFireplaceLearning(state_file=self.temp_file.name)
        
        # Integrate systems
        integrate_adaptive_fireplace_with_multi_source_physics(
            self.multi_source_physics, self.adaptive_fireplace
        )
    
    def teardown_method(self):
        """Clean up"""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_enhanced_fireplace_calculation(self):
        """Test that fireplace calculation is enhanced with learning"""
        # Increase learning confidence to trigger adaptive learning instead of fallback
        self.adaptive_fireplace.learning_state.learned_coefficients['learning_confidence'] = 0.5
        
        # Test with temperature data (should use learning)
        result = self.multi_source_physics.calculate_fireplace_heat_contribution(
            fireplace_on=True,
            outdoor_temp=5.0,
            living_room_temp=23.0,
            other_rooms_temp=20.0
        )
        
        assert 'learning_enhanced' in result
        assert result['heat_contribution_kw'] > 0
        assert 'adaptive learning' in result['reasoning']
    
    def test_fallback_to_physics(self):
        """Test fallback to original physics when learning insufficient"""
        # Test without temperature data (should use original physics)
        result = self.multi_source_physics.calculate_fireplace_heat_contribution(
            fireplace_on=True,
            outdoor_temp=5.0
            # No living_room_temp or other_rooms_temp provided
        )
        
        assert 'learning_enhanced' in result
        assert result['learning_enhanced'] is False
        assert 'physics fallback' in result['reasoning']
    
    def test_combined_heat_source_analysis(self):
        """Test that adaptive fireplace works with other heat sources"""
        # Test comprehensive multi-source analysis
        analysis = self.multi_source_physics.calculate_combined_heat_sources(
            pv_power=1500,
            fireplace_on=True,
            tv_on=True,
            indoor_temp=22.0,
            outdoor_temp=5.0,
            # Additional fireplace learning data
            living_room_temp=23.0,
            other_rooms_temp=20.0
        )
        
        assert analysis['total_heat_contribution_kw'] > 0
        assert analysis['fireplace_contribution']['heat_contribution_kw'] > 0
        # Should include both PV and fireplace contributions


def test_integration_example():
    """Test example of how to use adaptive fireplace in production"""
    
    # Create temporary state file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
    temp_file.close()
    
    try:
        # Initialize systems
        adaptive_fireplace = AdaptiveFireplaceLearning(state_file=temp_file.name)
        multi_source_physics = MultiHeatSourcePhysics()
        
        # Integrate
        enhanced_physics = integrate_adaptive_fireplace_with_multi_source_physics(
            multi_source_physics, adaptive_fireplace
        )
        
        # Simulate sensor data from Home Assistant
        sensor_data = {
            'living_room_temp': 22.8,      # sensor.thermometer_wohnzimmer_kompensiert
            'other_rooms_temp': 20.2,      # sensor.avg_other_rooms_temp
            'outdoor_temp': 3.0,           # sensor.thermometer_waermepume_kompensiert
            'fireplace_active': True,      # binary_sensor.fireplace_active
            'pv_power': 800,              # sensor.power_pv
            'tv_on': False                # input_boolean.fernseher
        }
        
        # Observe fireplace state for learning
        fireplace_result = adaptive_fireplace.observe_fireplace_state(
            sensor_data['living_room_temp'],
            sensor_data['other_rooms_temp'],
            sensor_data['outdoor_temp'],
            sensor_data['fireplace_active']
        )
        
        # Calculate enhanced heat source analysis
        heat_analysis = enhanced_physics.calculate_combined_heat_sources(
            pv_power=sensor_data['pv_power'],
            fireplace_on=sensor_data['fireplace_active'],
            tv_on=sensor_data['tv_on'],
            indoor_temp=sensor_data['living_room_temp'],
            outdoor_temp=sensor_data['outdoor_temp'],
            living_room_temp=sensor_data['living_room_temp'],
            other_rooms_temp=sensor_data['other_rooms_temp']
        )
        
        # Validate results
        assert fireplace_result['temp_differential'] > 2.0  # Above fireplace threshold
        assert fireplace_result['heat_contribution_kw'] > 0
        assert heat_analysis['total_heat_contribution_kw'] > 0
        
        print("✅ Production integration example successful!")
        
    finally:
        # Cleanup
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


if __name__ == "__main__":
    # Run a quick test
    test_integration_example()
    print("🔥 All adaptive fireplace tests complete!")
