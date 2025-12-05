"""
Physics Validation Tests for Thermal Equilibrium Model.

These tests validate that the thermal model follows fundamental physics principles:
- Energy conservation
- Second Law of Thermodynamics  
- Unit consistency
- Physical bounds and realistic behavior

Created as part of Phase 2: Implementation Quality Fixes
"""

import unittest
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from thermal_equilibrium_model import ThermalEquilibriumModel


class TestThermalPhysics(unittest.TestCase):
    """Physics validation tests for thermal equilibrium model."""
    
    def setUp(self):
        """Set up test model with realistic parameters."""
        self.model = ThermalEquilibriumModel()
        
        # Set realistic test parameters
        self.model.thermal_time_constant = 4.0  # 4 hours
        self.model.heat_loss_coefficient = 1.2  # °C per unit heat input
        self.model.outlet_effectiveness = 0.75  # 75% heat transfer efficiency
        
        # Test scenarios
        self.outdoor_temp = 5.0
        self.target_indoor = 21.0
        self.outlet_temp = 45.0

    def test_energy_conservation_at_equilibrium(self):
        """
        Test that energy is conserved at equilibrium.
        
        At equilibrium: Heat Input = Heat Loss
        Q_in = Q_out = heat_loss_coefficient * (T_indoor - T_outdoor)
        """
        pv_power = 1000  # 1kW PV
        
        # Calculate equilibrium temperature
        equilibrium_temp = self.model.predict_equilibrium_temperature(
            self.outlet_temp, self.outdoor_temp, pv_power=pv_power
        )
        
        # Calculate heat input
        heat_from_outlet = self.outlet_temp * self.model.outlet_effectiveness
        heat_from_pv = pv_power * self.model.external_source_weights['pv']
        total_heat_input = heat_from_outlet + heat_from_pv
        
        # Calculate heat loss at equilibrium
        heat_loss = self.model.heat_loss_coefficient * (equilibrium_temp - self.outdoor_temp)
        
        # Energy conservation: heat input should equal heat loss
        self.assertAlmostEqual(
            total_heat_input, heat_loss, places=2,
            msg=f"Energy not conserved: input={total_heat_input:.3f}, "
                f"loss={heat_loss:.3f}"
        )

    def test_second_law_thermodynamics_indoor_outdoor_relationship(self):
        """
        Test Second Law: Indoor temperature cannot be below outdoor
        when heat is being added to the system.
        """
        # Test with positive heat input
        equilibrium_temp = self.model.predict_equilibrium_temperature(
            self.outlet_temp, self.outdoor_temp, pv_power=0
        )
        
        # Indoor should be warmer than outdoor when heating
        self.assertGreater(
            equilibrium_temp, self.outdoor_temp,
            msg="Indoor temperature below outdoor despite heat input"
        )
        
        # Test with no heat input (outlet temp = 0)
        equilibrium_no_heat = self.model.predict_equilibrium_temperature(
            0.0, self.outdoor_temp, pv_power=0
        )
        
        # With no heat, indoor should equal outdoor
        self.assertAlmostEqual(
            equilibrium_no_heat, self.outdoor_temp, places=1,
            msg="Indoor temperature differs from outdoor with no heat input"
        )

    def test_unit_consistency_across_calculations(self):
        """
        Test that all calculations use consistent units.
        
        Expected units:
        - Temperatures: °C
        - Heat coefficients: °C per unit heat
        - PV power: W (watts)
        - PV weight: °C/W
        """
        pv_power = 2000  # 2kW in watts
        
        equilibrium_temp = self.model.predict_equilibrium_temperature(
            self.outlet_temp, self.outdoor_temp, pv_power=pv_power
        )
        
        # Result should be a reasonable temperature in °C
        self.assertGreater(equilibrium_temp, -50, "Temperature unrealistically low")
        self.assertLess(equilibrium_temp, 100, "Temperature unrealistically high")
        self.assertIsInstance(equilibrium_temp, float, "Temperature not a float")

    def test_physical_bounds_indoor_between_outdoor_and_source(self):
        """
        Test that indoor temperature is physically bounded.
        
        Indoor temperature should be:
        - Above outdoor temperature (when heating)
        - Below the effective heat source temperature
        - Realistic for building environments
        """
        # Test normal heating scenario
        equilibrium_temp = self.model.predict_equilibrium_temperature(
            self.outlet_temp, self.outdoor_temp, pv_power=0
        )
        
        # Indoor should be between outdoor and a reasonable upper bound
        self.assertGreater(equilibrium_temp, self.outdoor_temp)
        self.assertLess(equilibrium_temp, self.outlet_temp)
        
        # Should be in realistic building temperature range
        self.assertGreater(equilibrium_temp, -20, "Temperature too cold for buildings")
        self.assertLess(equilibrium_temp, 50, "Temperature too hot for buildings")

    def test_linearity_of_heat_loss_with_temperature_difference(self):
        """
        Test that heat loss is linear with temperature difference.
        
        Q_loss = k * (T_indoor - T_outdoor) should be linear in ΔT
        """
        outdoor_temps = [0, 5, 10, 15, 20]
        equilibrium_temps = []
        
        for outdoor_temp in outdoor_temps:
            eq_temp = self.model.predict_equilibrium_temperature(
                self.outlet_temp, outdoor_temp, pv_power=0
            )
            equilibrium_temps.append(eq_temp)
        
        # Calculate temperature differences
        temp_differences = [eq - out for eq, out in zip(equilibrium_temps, outdoor_temps)]
        
        # All temperature differences should be approximately equal
        # (since heat input is constant)
        expected_diff = temp_differences[0]
        for i, diff in enumerate(temp_differences[1:], 1):
            self.assertAlmostEqual(
                diff, expected_diff, places=1,
                msg=f"Non-linear heat loss detected at outdoor={outdoor_temps[i]}°C: "
                    f"expected_diff={expected_diff:.2f}, actual_diff={diff:.2f}"
            )

    def test_external_heat_source_contribution_additivity(self):
        """
        Test that external heat sources contribute additively.
        
        Total heat = outlet_heat + pv_heat + fireplace_heat + tv_heat
        """
        # Test individual contributions
        eq_baseline = self.model.predict_equilibrium_temperature(
            self.outlet_temp, self.outdoor_temp, pv_power=0, fireplace_on=0, tv_on=0
        )
        
        eq_with_pv = self.model.predict_equilibrium_temperature(
            self.outlet_temp, self.outdoor_temp, pv_power=1000, fireplace_on=0, tv_on=0
        )
        
        eq_with_fireplace = self.model.predict_equilibrium_temperature(
            self.outlet_temp, self.outdoor_temp, pv_power=0, fireplace_on=1, tv_on=0
        )
        
        eq_with_both = self.model.predict_equilibrium_temperature(
            self.outlet_temp, self.outdoor_temp, pv_power=1000, fireplace_on=1, tv_on=0
        )
        
        # Calculate individual contributions
        pv_contribution = eq_with_pv - eq_baseline
        fireplace_contribution = eq_with_fireplace - eq_baseline
        
        # Combined effect should equal sum of individual effects
        expected_combined = eq_baseline + pv_contribution + fireplace_contribution
        
        self.assertAlmostEqual(
            eq_with_both, expected_combined, places=2,
            msg=f"Heat sources not additive: expected={expected_combined:.3f}, "
                f"actual={eq_with_both:.3f}"
        )

    def test_heat_loss_coefficient_physical_meaning(self):
        """
        Test that heat loss coefficient has correct physical meaning.
        
        Coefficient should represent the heat loss rate per degree
        temperature difference.
        """
        # Test with known temperature difference
        outdoor_temp = 10.0
        target_indoor = 20.0  # 10°C difference
        
        # Calculate required heat input to maintain this difference
        required_heat_loss = self.model.heat_loss_coefficient * (target_indoor - outdoor_temp)
        
        # Calculate what outlet temperature would be needed
        required_outlet_temp = required_heat_loss / self.model.outlet_effectiveness
        
        # Verify that this outlet temperature actually produces the target indoor
        actual_equilibrium = self.model.predict_equilibrium_temperature(
            required_outlet_temp, outdoor_temp, pv_power=0
        )
        
        self.assertAlmostEqual(
            actual_equilibrium, target_indoor, places=1,
            msg=f"Heat loss coefficient incorrect: expected={target_indoor}°C, "
                f"actual={actual_equilibrium:.3f}°C"
        )

    def test_thermal_time_constant_not_in_equilibrium(self):
        """
        Test that thermal time constant does not affect equilibrium calculations.
        
        Equilibrium temperature should be independent of time constants.
        """
        # Save original time constant
        original_time_constant = self.model.thermal_time_constant
        
        # Calculate equilibrium with original time constant
        eq_original = self.model.predict_equilibrium_temperature(
            self.outlet_temp, self.outdoor_temp, pv_power=0
        )
        
        # Change time constant dramatically
        self.model.thermal_time_constant = 10.0  # Much slower system
        
        eq_different_time = self.model.predict_equilibrium_temperature(
            self.outlet_temp, self.outdoor_temp, pv_power=0
        )
        
        # Restore original
        self.model.thermal_time_constant = original_time_constant
        
        # Equilibrium should be identical
        self.assertAlmostEqual(
            eq_original, eq_different_time, places=3,
            msg="Thermal time constant affects equilibrium calculation"
        )


class TestThermalPhysicsEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def setUp(self):
        """Set up test model."""
        self.model = ThermalEquilibriumModel()
        
    def test_zero_heat_loss_coefficient(self):
        """Test behavior with zero or very small heat loss coefficient."""
        self.model.heat_loss_coefficient = 0.0
        
        # Should return outdoor temperature (no heat retention)
        equilibrium = self.model.predict_equilibrium_temperature(45.0, 10.0)
        self.assertEqual(equilibrium, 10.0, "Zero heat loss should equal outdoor temp")
        
    def test_extreme_temperatures(self):
        """Test behavior with extreme temperature inputs."""
        # Very cold outdoor
        eq_cold = self.model.predict_equilibrium_temperature(45.0, -30.0)
        self.assertIsInstance(eq_cold, float, "Failed with extreme cold")
        
        # Very hot outdoor  
        eq_hot = self.model.predict_equilibrium_temperature(45.0, 40.0)
        self.assertIsInstance(eq_hot, float, "Failed with extreme heat")
        
    def test_high_pv_power(self):
        """Test behavior with very high PV power input."""
        eq_high_pv = self.model.predict_equilibrium_temperature(
            45.0, 10.0, pv_power=10000  # 10kW
        )
        
        # Should handle gracefully
        self.assertIsInstance(eq_high_pv, float, "Failed with high PV power")
        self.assertGreater(eq_high_pv, 10.0, "High PV should increase temperature")


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
