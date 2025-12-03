"""
Comprehensive tests for Multi-Heat-Source Physics Integration

Tests the sophisticated heat source coordination algorithms that transform
the ML Heating System from single-variable control to multi-variable optimization.

Test Coverage:
- PV solar warming calculations
- Fireplace heat equivalent estimation  
- TV/electronics and occupancy heat analysis
- System state impact calculations
- Combined multi-source optimization
- Outlet temperature optimization
- Feature enhancement integration
"""

import unittest
import math
from datetime import datetime
from unittest.mock import Mock, patch

# Import the modules under test
from src.multi_heat_source_physics import (
    MultiHeatSourcePhysics,
    enhance_physics_features_with_heat_sources,
    _encode_heat_source
)


class TestMultiHeatSourcePhysics(unittest.TestCase):
    """Test the core MultiHeatSourcePhysics class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.physics = MultiHeatSourcePhysics()
        self.test_indoor_temp = 21.0
        self.test_outdoor_temp = 5.0
    
    def test_pv_heat_contribution_minimal_power(self):
        """Test PV heat contribution with minimal power generation."""
        result = self.physics.calculate_pv_heat_contribution(
            pv_power=30,  # Below 50W threshold
            indoor_temp=self.test_indoor_temp,
            outdoor_temp=self.test_outdoor_temp
        )
        
        self.assertEqual(result['heat_contribution_kw'], 0.0)
        self.assertEqual(result['outlet_temp_reduction'], 0.0)
        self.assertEqual(result['thermal_effectiveness'], 0.0)
        self.assertIn('Minimal PV', result['reasoning'])
    
    def test_pv_heat_contribution_significant_power(self):
        """Test PV heat contribution with significant power generation."""
        result = self.physics.calculate_pv_heat_contribution(
            pv_power=2000,  # 2kW generation
            indoor_temp=self.test_indoor_temp,
            outdoor_temp=self.test_outdoor_temp,
            time_of_day=12  # Solar noon
        )
        
        # Should have meaningful heat contribution
        self.assertGreater(result['heat_contribution_kw'], 0.0)
        self.assertGreater(result['outlet_temp_reduction'], 0.0)
        self.assertGreater(result['thermal_effectiveness'], 0.0)
        
        # Time effectiveness should be high at solar noon
        self.assertGreater(result['hour_effectiveness'], 0.9)
        
        # Reasoning should include power and effectiveness details
        self.assertIn('2000W', result['reasoning'])
        self.assertIn('kW', result['reasoning'])
    
    def test_pv_heat_contribution_time_of_day_variations(self):
        """Test PV thermal effectiveness varies with time of day."""
        pv_power = 1500
        
        # Test solar noon (high effectiveness)
        noon_result = self.physics.calculate_pv_heat_contribution(
            pv_power, self.test_indoor_temp, self.test_outdoor_temp, 12
        )
        
        # Test midnight (low effectiveness)
        midnight_result = self.physics.calculate_pv_heat_contribution(
            pv_power, self.test_indoor_temp, self.test_outdoor_temp, 0
        )
        
        # Noon should be more effective than midnight
        self.assertGreater(
            noon_result['hour_effectiveness'],
            midnight_result['hour_effectiveness']
        )
        self.assertGreater(
            noon_result['heat_contribution_kw'],
            midnight_result['heat_contribution_kw']
        )
    
    def test_fireplace_heat_contribution_off(self):
        """Test fireplace heat contribution when off."""
        result = self.physics.calculate_fireplace_heat_contribution(
            fireplace_on=False
        )
        
        self.assertEqual(result['heat_contribution_kw'], 0.0)
        self.assertEqual(result['outlet_temp_reduction'], 0.0)
        self.assertEqual(result['heat_distribution_factor'], 0.0)
        self.assertIn('Fireplace off', result['reasoning'])
    
    def test_fireplace_heat_contribution_on(self):
        """Test fireplace heat contribution when active."""
        result = self.physics.calculate_fireplace_heat_contribution(
            fireplace_on=True,
            zone_factor=0.8,
            outdoor_temp=self.test_outdoor_temp,
            duration_hours=1.5
        )
        
        # Should have significant heat contribution
        self.assertGreater(result['heat_contribution_kw'], 1.0)
        self.assertGreater(result['outlet_temp_reduction'], 0.0)
        
        # Zone factor should be reflected
        self.assertEqual(result['heat_distribution_factor'], 0.8)
        
        # Thermal buildup should be partial for 1.5 hours
        self.assertLess(result['thermal_buildup_factor'], 1.0)
        self.assertGreater(result['thermal_buildup_factor'], 0.5)
    
    def test_fireplace_thermal_buildup(self):
        """Test fireplace thermal buildup over time."""
        # Short duration
        short_result = self.physics.calculate_fireplace_heat_contribution(
            True, outdoor_temp=0, duration_hours=0.5
        )
        
        # Long duration 
        long_result = self.physics.calculate_fireplace_heat_contribution(
            True, outdoor_temp=0, duration_hours=3.0
        )
        
        # Long duration should have higher thermal buildup
        self.assertGreater(
            long_result['thermal_buildup_factor'],
            short_result['thermal_buildup_factor']
        )
        self.assertGreater(
            long_result['heat_contribution_kw'],
            short_result['heat_contribution_kw']
        )
    
    def test_electronics_occupancy_heat_off(self):
        """Test electronics/occupancy heat when TV is off."""
        result = self.physics.calculate_electronics_occupancy_heat(
            tv_on=False
        )
        
        self.assertEqual(result['heat_contribution_kw'], 0.0)
        self.assertEqual(result['outlet_temp_reduction'], 0.0)
        self.assertEqual(result['electronics_heat'], 0.0)
        self.assertEqual(result['occupancy_heat'], 0.0)
        self.assertIn('TV off', result['reasoning'])
    
    def test_electronics_occupancy_heat_on(self):
        """Test electronics/occupancy heat when TV is on."""
        result = self.physics.calculate_electronics_occupancy_heat(
            tv_on=True,
            estimated_occupancy=2,
            activity_level='normal'
        )
        
        # Should have both electronics and occupancy heat
        self.assertGreater(result['electronics_heat'], 0.0)
        self.assertGreater(result['occupancy_heat'], 0.0)
        self.assertGreater(result['heat_contribution_kw'], 0.0)
        self.assertEqual(result['estimated_occupancy'], 2)
        self.assertEqual(result['activity_factor'], 1.0)  # Normal activity
    
    def test_electronics_activity_level_variations(self):
        """Test different activity levels affect occupancy heat."""
        low_result = self.physics.calculate_electronics_occupancy_heat(
            tv_on=True, estimated_occupancy=2, activity_level='low'
        )
        
        high_result = self.physics.calculate_electronics_occupancy_heat(
            tv_on=True, estimated_occupancy=2, activity_level='high'
        )
        
        # High activity should produce more heat
        self.assertGreater(
            high_result['activity_factor'],
            low_result['activity_factor']
        )
        self.assertGreater(
            high_result['heat_contribution_kw'],
            low_result['heat_contribution_kw']
        )
    
    def test_system_state_impacts_normal(self):
        """Test system state impacts under normal operation."""
        result = self.physics.calculate_system_state_impacts(
            dhw_heating=False,
            dhw_disinfection=False,
            dhw_boost_heater=False,
            defrosting=False
        )
        
        self.assertEqual(result['capacity_reduction_percent'], 0.0)
        self.assertEqual(result['auxiliary_heat_kw'], 0.0)
        self.assertEqual(result['net_outlet_adjustment'], 0.0)
        self.assertEqual(result['active_states'], [])
        self.assertIn('Normal operation', result['reasoning'])
    
    def test_system_state_impacts_dhw_active(self):
        """Test system state impacts with DHW heating active."""
        result = self.physics.calculate_system_state_impacts(
            dhw_heating=True,
            dhw_disinfection=False,
            dhw_boost_heater=False,
            defrosting=False
        )
        
        # Should reduce capacity
        self.assertGreater(result['capacity_reduction_percent'], 0.0)
        self.assertGreater(result['net_outlet_adjustment'], 0.0)
        self.assertEqual(len(result['active_states']), 1)
        self.assertIn('DHW', result['active_states'][0])
    
    def test_system_state_impacts_defrost_active(self):
        """Test system state impacts with defrost active."""
        result = self.physics.calculate_system_state_impacts(
            dhw_heating=False,
            dhw_disinfection=False,
            dhw_boost_heater=False,
            defrosting=True
        )
        
        # Defrost should have significant capacity reduction
        self.assertGreater(result['capacity_reduction_percent'], 20.0)
        self.assertGreater(result['net_outlet_adjustment'], 0.0)
        self.assertIn('Defrost', result['active_states'][0])
    
    def test_system_state_impacts_boost_heater(self):
        """Test system state impacts with boost heater active."""
        result = self.physics.calculate_system_state_impacts(
            dhw_heating=False,
            dhw_disinfection=False,
            dhw_boost_heater=True,
            defrosting=False
        )
        
        # Boost heater adds auxiliary heat (reduces outlet need)
        self.assertGreater(result['auxiliary_heat_kw'], 0.0)
        self.assertLess(result['net_outlet_adjustment'], 0.0)  # Negative = reduction
        self.assertIn('Boost heater', result['active_states'][0])
    
    def test_combined_heat_sources_no_sources(self):
        """Test combined heat sources with no active sources."""
        result = self.physics.calculate_combined_heat_sources(
            pv_power=0,
            fireplace_on=False,
            tv_on=False,
            indoor_temp=self.test_indoor_temp,
            outdoor_temp=self.test_outdoor_temp
        )
        
        self.assertEqual(result['total_heat_contribution_kw'], 0.0)
        self.assertEqual(result['heat_source_diversity'], 0)
        self.assertEqual(result['diversity_factor'], 1.0)  # Baseline
    
    def test_combined_heat_sources_multiple_active(self):
        """Test combined heat sources with multiple sources active."""
        result = self.physics.calculate_combined_heat_sources(
            pv_power=1500,  # Significant PV
            fireplace_on=True,
            tv_on=True,
            indoor_temp=self.test_indoor_temp,
            outdoor_temp=self.test_outdoor_temp
        )
        
        # Should have heat from all sources
        self.assertGreater(result['total_heat_contribution_kw'], 0.0)
        self.assertEqual(result['heat_source_diversity'], 3)  # PV + fireplace + TV
        self.assertGreater(result['diversity_factor'], 1.0)  # Diversity bonus
        
        # Individual contributions should be present
        self.assertGreater(result['pv_contribution']['heat_contribution_kw'], 0.0)
        self.assertGreater(result['fireplace_contribution']['heat_contribution_kw'], 0.0)
        self.assertGreater(result['electronics_contribution']['heat_contribution_kw'], 0.0)
    
    def test_combined_heat_sources_coordination_analysis(self):
        """Test heat source coordination analysis."""
        result = self.physics.calculate_combined_heat_sources(
            pv_power=2000,  # High PV
            fireplace_on=True,
            tv_on=True,
            indoor_temp=self.test_indoor_temp,
            outdoor_temp=self.test_outdoor_temp
        )
        
        coordination = result['coordination_analysis']
        
        # Should identify dominant source
        self.assertIn(coordination['dominant_source'], ['PV', 'Fireplace', 'Electronics', 'System'])
        self.assertGreater(coordination['dominant_heat_kw'], 0.0)
        
        # Heat distribution percentages should sum to ~100%
        distribution = coordination['heat_distribution']
        total_percentage = sum(distribution.values())
        self.assertAlmostEqual(total_percentage, 100.0, places=1)
        
        # Should have thermal balance classification
        self.assertIn(coordination['thermal_balance'], ['balanced', 'dominated'])
    
    def test_outlet_optimization_no_heat_sources(self):
        """Test outlet temperature optimization with no heat sources."""
        base_outlet = 45.0
        
        # Mock heat source analysis with no contributions
        mock_analysis = {
            'total_heat_contribution_kw': 0.0,
            'total_outlet_temp_reduction': 0.0,
            'heat_source_diversity': 0,
            'diversity_factor': 1.0,
            'pv_contribution': {'heat_contribution_kw': 0.0, 'outlet_temp_reduction': 0.0},
            'fireplace_contribution': {'heat_contribution_kw': 0.0, 'outlet_temp_reduction': 0.0},
            'electronics_contribution': {'heat_contribution_kw': 0.0, 'outlet_temp_reduction': 0.0},
            'system_impacts': {'net_outlet_adjustment': 0.0}
        }
        
        result = self.physics.calculate_optimized_outlet_temperature(
            base_outlet, mock_analysis
        )
        
        # With no heat sources, should stay close to base
        self.assertAlmostEqual(result['optimized_outlet_temp'], base_outlet, places=1)
        self.assertAlmostEqual(result['optimization_amount'], 0.0, places=1)
    
    def test_outlet_optimization_with_heat_sources(self):
        """Test outlet temperature optimization with active heat sources."""
        base_outlet = 45.0
        
        # Mock heat source analysis with significant contributions
        mock_analysis = {
            'total_heat_contribution_kw': 2.5,
            'total_outlet_temp_reduction': 5.0,  # 5°C reduction possible
            'heat_source_diversity': 3,
            'diversity_factor': 1.3,
            'pv_contribution': {'heat_contribution_kw': 1.0, 'outlet_temp_reduction': 2.0},
            'fireplace_contribution': {'heat_contribution_kw': 1.2, 'outlet_temp_reduction': 2.5},
            'electronics_contribution': {'heat_contribution_kw': 0.3, 'outlet_temp_reduction': 0.5},
            'system_impacts': {'net_outlet_adjustment': 0.0}
        }
        
        result = self.physics.calculate_optimized_outlet_temperature(
            base_outlet, mock_analysis
        )
        
        # Should optimize outlet temperature down from base
        self.assertLess(result['optimized_outlet_temp'], base_outlet)
        self.assertGreater(result['optimization_amount'], 0.0)
        self.assertGreater(result['optimization_percentage'], 0.0)
        
        # Should respect safety bounds
        self.assertGreaterEqual(result['optimized_outlet_temp'], 18.0)  # min + safety
        self.assertLessEqual(result['optimized_outlet_temp'], 63.0)  # max - safety


class TestFeatureEnhancement(unittest.TestCase):
    """Test the feature enhancement integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.physics = MultiHeatSourcePhysics()
    
    def test_enhance_physics_features_with_heat_sources(self):
        """Test enhancing existing features with heat source analysis."""
        # Mock existing features (subset of actual physics features)
        existing_features = {
            'pv_now': 1500,
            'fireplace_on': 1,
            'tv_on': 1,
            'indoor_temp_lag_30m': 21.0,
            'outdoor_temp': 5.0,
            'dhw_heating': 0,
            'dhw_disinfection': 0,
            'dhw_boost_heater': 0,
            'defrosting': 0,
            'outlet_temp': 45.0
        }
        
        enhanced = enhance_physics_features_with_heat_sources(
            existing_features, self.physics
        )
        
        # Should contain all original features
        for key, value in existing_features.items():
            self.assertEqual(enhanced[key], value)
        
        # Should add new multi-heat-source features
        expected_new_features = [
            'pv_heat_contribution_kw',
            'fireplace_heat_contribution_kw',
            'electronics_heat_contribution_kw',
            'total_auxiliary_heat_kw',
            'pv_outlet_reduction',
            'fireplace_outlet_reduction',
            'electronics_outlet_reduction',
            'total_outlet_reduction',
            'heat_source_diversity',
            'heat_source_diversity_factor',
            'dominant_heat_source',
            'thermal_balance_score',
            'pv_thermal_effectiveness',
            'fireplace_thermal_buildup',
            'electronics_occupancy_factor'
        ]
        
        for feature in expected_new_features:
            self.assertIn(feature, enhanced)
        
        # Heat contributions should be positive for active sources
        self.assertGreater(enhanced['pv_heat_contribution_kw'], 0.0)
        self.assertGreater(enhanced['fireplace_heat_contribution_kw'], 0.0)
        self.assertGreater(enhanced['electronics_heat_contribution_kw'], 0.0)
    
    def test_encode_heat_source(self):
        """Test heat source encoding for ML features."""
        self.assertEqual(_encode_heat_source('PV'), 1.0)
        self.assertEqual(_encode_heat_source('Fireplace'), 2.0)
        self.assertEqual(_encode_heat_source('Electronics'), 3.0)
        self.assertEqual(_encode_heat_source('System'), 4.0)
        self.assertEqual(_encode_heat_source('Unknown'), 0.0)


if __name__ == '__main__':
    unittest.main()
