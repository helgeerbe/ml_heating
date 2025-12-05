"""
Integration tests for thermal constants system with thermal equilibrium model.

This test validates that the standardized units system properly integrates
with the thermal model and provides meaningful validation.
"""

import unittest
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from thermal_constants import (
    PhysicsConstants, ThermalUnits, ThermalParameterValidator,
    validate_thermal_parameters, format_thermal_state
)
from thermal_equilibrium_model import ThermalEquilibriumModel


class TestThermalConstantsIntegration(unittest.TestCase):
    """Test integration between thermal constants and thermal model."""
    
    def setUp(self):
        """Set up test model and validator."""
        self.model = ThermalEquilibriumModel()
        self.validator = ThermalParameterValidator()
        
    def test_model_parameters_within_physics_bounds(self):
        """Test that model parameters are within defined physics bounds."""
        # Get model parameters
        parameters = {
            'thermal_time_constant': self.model.thermal_time_constant,
            'heat_loss_coefficient': self.model.heat_loss_coefficient,
            'outlet_effectiveness': self.model.outlet_effectiveness,
            'pv_heat_weight': self.model.external_source_weights['pv'],
            'fireplace_heat_weight': self.model.external_source_weights['fireplace'],
            'tv_heat_weight': self.model.external_source_weights['tv']
        }
        
        # Validate all parameters
        validation_results = ThermalUnits.validate_parameters(parameters)
        
        # All should be valid
        for param_name, is_valid in validation_results.items():
            self.assertTrue(
                is_valid, 
                f"Parameter {param_name}={parameters[param_name]} "
                f"outside valid range {ThermalUnits.get_range(param_name)}"
            )
    
    def test_heat_balance_validation(self):
        """Test comprehensive heat balance parameter validation."""
        is_valid = self.validator.validate_heat_balance_parameters(
            heat_loss_coeff=self.model.heat_loss_coefficient,
            outlet_effectiveness=self.model.outlet_effectiveness,
            external_weights=self.model.external_source_weights
        )
        
        self.assertTrue(is_valid, f"Heat balance validation failed: {self.validator.get_validation_report()}")
    
    def test_temperature_input_validation(self):
        """Test temperature input validation with realistic values."""
        # Valid temperature scenario
        is_valid = self.validator.validate_temperature_inputs(
            indoor=21.5,    # Normal indoor temp
            outdoor=5.0,    # Cool outdoor temp
            outlet=45.0     # Normal heat pump outlet
        )
        
        self.assertTrue(is_valid, f"Temperature validation failed: {self.validator.get_validation_report()}")
        
        # Invalid temperature scenario - outlet below outdoor while heating
        is_invalid = self.validator.validate_temperature_inputs(
            indoor=22.0,    # Indoor warmer than outdoor
            outdoor=10.0,   # Outdoor temp
            outlet=8.0      # Outlet below outdoor - physically impossible for heating
        )
        
        self.assertFalse(is_invalid, "Should detect invalid outlet temperature")
    
    def test_parameter_formatting(self):
        """Test parameter formatting with units."""
        # Test temperature formatting
        temp_formatted = ThermalUnits.format_parameter('indoor_temperature', 21.5)
        self.assertEqual(temp_formatted, "21.5 °C")
        
        # Test coefficient formatting
        coeff_formatted = ThermalUnits.format_parameter('heat_loss_coefficient', 0.25)
        self.assertEqual(coeff_formatted, "0.250 °C/thermal_unit")
        
        # Test power formatting
        power_formatted = ThermalUnits.format_parameter('pv_power', 1500.7)
        self.assertEqual(power_formatted, "1501 W")
    
    def test_thermal_state_formatting(self):
        """Test complete thermal state formatting."""
        state = {
            'thermal_time_constant': self.model.thermal_time_constant,
            'heat_loss_coefficient': self.model.heat_loss_coefficient,
            'outlet_effectiveness': self.model.outlet_effectiveness,
            'indoor_temperature': 21.5,
            'outdoor_temperature': 5.0,
            'pv_power': 1500
        }
        
        formatted = format_thermal_state(state)
        
        # Should contain all parameters with units
        self.assertIn("thermal_time_constant:", formatted)
        self.assertIn("hours", formatted)
        self.assertIn("°C", formatted)
        self.assertIn("W", formatted)
    
    def test_physics_constants_bounds(self):
        """Test that physics constants define reasonable bounds."""
        # Temperature bounds should be realistic
        self.assertEqual(PhysicsConstants.MIN_BUILDING_TEMP, -20.0)
        self.assertEqual(PhysicsConstants.MAX_BUILDING_TEMP, 50.0)
        self.assertEqual(PhysicsConstants.MIN_OUTLET_TEMP, 25.0)
        self.assertEqual(PhysicsConstants.MAX_OUTLET_TEMP, 70.0)
        
        # Time constants should cover realistic building response times
        self.assertEqual(PhysicsConstants.MIN_TIME_CONSTANT, 0.5)  # 30 minutes
        self.assertEqual(PhysicsConstants.MAX_TIME_CONSTANT, 24.0)  # 24 hours
    
    def test_unit_definitions_completeness(self):
        """Test that all important thermal parameters have unit definitions."""
        required_units = [
            'thermal_time_constant',
            'heat_loss_coefficient', 
            'outlet_effectiveness',
            'pv_heat_weight',
            'fireplace_heat_weight',
            'tv_heat_weight',
            'indoor_temperature',
            'outdoor_temperature',
            'outlet_temperature',
            'pv_power'
        ]
        
        for param in required_units:
            unit = ThermalUnits.get_unit(param)
            self.assertNotEqual(unit, 'unknown', f"Missing unit definition for {param}")
    
    def test_validation_error_reporting(self):
        """Test that validation errors are properly reported."""
        # Test with invalid heat loss coefficient
        validator = ThermalParameterValidator()
        
        # Negative heat loss coefficient should fail
        is_valid = validator.validate_heat_balance_parameters(
            heat_loss_coeff=-1.0,  # Invalid negative value
            outlet_effectiveness=0.5,
            external_weights={'pv': 0.002, 'fireplace': 5.0, 'tv': 0.2}
        )
        
        self.assertFalse(is_valid)
        
        report = validator.get_validation_report()
        self.assertIn("❌", report)  # Should contain error indicator
        self.assertIn("heat loss coefficient must be positive", report.lower())
    
    def test_model_equilibrium_with_validation(self):
        """Test that model equilibrium calculations work with parameter validation."""
        # Valid scenario
        outdoor_temp = 5.0
        outlet_temp = 45.0
        pv_power = 1000
        
        # Validate inputs first
        temp_valid = self.validator.validate_temperature_inputs(
            indoor=20.0,  # Expected result
            outdoor=outdoor_temp,
            outlet=outlet_temp
        )
        self.assertTrue(temp_valid)
        
        # Calculate equilibrium
        equilibrium = self.model.predict_equilibrium_temperature(
            outlet_temp=outlet_temp, 
            outdoor_temp=outdoor_temp, 
            current_indoor=20.0, 
            pv_power=pv_power
        )
        
        # Result should be valid temperature
        self.assertTrue(ThermalUnits.validate_parameter('indoor_temperature', equilibrium))
        
        # Should be physically reasonable
        self.assertGreater(equilibrium, outdoor_temp)  # Indoor warmer than outdoor
        self.assertLess(equilibrium, outlet_temp)      # Indoor cooler than outlet
    
    def test_external_source_weight_validation(self):
        """Test validation of external heat source weights."""
        # Test each external source weight is in valid range
        pv_weight = self.model.external_source_weights['pv']
        self.assertTrue(ThermalUnits.validate_parameter('pv_heat_weight', pv_weight))
        
        fireplace_weight = self.model.external_source_weights['fireplace']
        self.assertTrue(ThermalUnits.validate_parameter('fireplace_heat_weight', fireplace_weight))
        
        tv_weight = self.model.external_source_weights['tv']
        self.assertTrue(ThermalUnits.validate_parameter('tv_heat_weight', tv_weight))


class TestThermalConstantsUsagePatterrns(unittest.TestCase):
    """Test common usage patterns for thermal constants."""
    
    def test_quick_parameter_validation(self):
        """Test convenience function for quick validation."""
        valid_params = {
            'indoor_temperature': 21.0,
            'outdoor_temperature': 5.0,
            'pv_power': 1500,
            'thermal_time_constant': 4.0
        }
        
        self.assertTrue(validate_thermal_parameters(valid_params))
        
        # Test with invalid parameter
        invalid_params = {
            'indoor_temperature': 100.0,  # Too hot
            'outdoor_temperature': 5.0
        }
        
        self.assertFalse(validate_thermal_parameters(invalid_params))
    
    def test_parameter_range_queries(self):
        """Test querying parameter ranges."""
        # Test specific ranges
        temp_range = ThermalUnits.get_range('indoor_temperature')
        self.assertEqual(temp_range, (-20.0, 50.0))
        
        pv_range = ThermalUnits.get_range('pv_heat_weight')
        self.assertEqual(pv_range, (0.0, 0.01))
        
        # Test unknown parameter
        unknown_range = ThermalUnits.get_range('unknown_parameter')
        self.assertEqual(unknown_range, (-float('inf'), float('inf')))
    
    def test_unit_system_consistency(self):
        """Test that the unit system is internally consistent."""
        # Temperature units should all be °C
        temp_params = ['indoor_temperature', 'outdoor_temperature', 'outlet_temperature']
        for param in temp_params:
            unit = ThermalUnits.get_unit(param)
            self.assertEqual(unit, '°C')
        
        # PV should be in watts
        self.assertEqual(ThermalUnits.get_unit('pv_power'), 'W')
        
        # PV weight should be temperature per power
        self.assertEqual(ThermalUnits.get_unit('pv_heat_weight'), '°C/W')


if __name__ == '__main__':
    unittest.main(verbosity=2)
