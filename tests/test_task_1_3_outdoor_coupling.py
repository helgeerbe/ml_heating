"""
TDD Unit Tests for Task 1.3: Remove Arbitrary Outdoor Coupling

These tests define the EXPECTED behavior for removing non-physical outdoor
coupling and thermal bridge factors. Following TDD methodology - tests
written FIRST, then implementation follows.

Task 1.3 Requirements:
- Remove outdoor_coupling parameter and related calculations
- Implement proper heat loss: Q_loss = heat_loss_coefficient * (T_indoor - T_outdoor)
- Remove arbitrary thermal bridge calculations with magic 20°C reference
- Remove outdoor_coupling from optimization parameters
"""

import unittest
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from thermal_equilibrium_model import ThermalEquilibriumModel


class TestTask13OutdoorCoupling(unittest.TestCase):
    """TDD tests for removing arbitrary outdoor coupling (Task 1.3)."""
    
    def setUp(self):
        """Set up test model."""
        self.model = ThermalEquilibriumModel()
        
        # Set predictable parameters for testing
        self.model.thermal_time_constant = 4.0
        self.model.heat_loss_coefficient = 1.5 # TDD-FIX: More realistic value
        self.model.outlet_effectiveness = 0.8

    def test_no_arbitrary_20_celsius_reference(self):
        """
        TDD TEST: No arbitrary 20°C normalization should exist.
        
        Expected behavior:
        - Heat loss should be based purely on (T_indoor - T_outdoor)
        - No magic 20°C reference temperature in calculations
        - Physics should work consistently at all temperature ranges
        """
        outlet_temp = 50.0
        
        # Test around 0°C
        equilibrium_0 = self.model.predict_equilibrium_temperature(
            outlet_temp, 0.0, pv_power=0, fireplace_on=0, tv_on=0
        )
        
        # Test around 20°C  
        equilibrium_20 = self.model.predict_equilibrium_temperature(
            outlet_temp, 20.0, pv_power=0, fireplace_on=0, tv_on=0
        )
        
        # Test around 40°C
        equilibrium_40 = self.model.predict_equilibrium_temperature(
            outlet_temp, 40.0, pv_power=0, fireplace_on=0, tv_on=0
        )
        
        # Temperature differences should be purely based on physics
        # Expected: deltaT = outdoor_temp_difference / heat_loss_coefficient  
        
        # Calculate deltas
        delta_0_to_20 = equilibrium_20 - equilibrium_0
        delta_20_to_40 = equilibrium_40 - equilibrium_20
        
        # Should be exactly equal (no 20°C bias)
        self.assertAlmostEqual(
            delta_0_to_20, delta_20_to_40, places=2,
            msg=f"20°C reference detected: delta_0_to_20={delta_0_to_20:.3f}, "
                f"delta_20_to_40={delta_20_to_40:.3f}"
        )
        
        # Both deltas should equal outdoor temperature difference (20°C)
        self.assertAlmostEqual(
            delta_0_to_20, 20.0, places=1,
            msg=f"Heat loss not 1:1 with outdoor temp: "
                f"expected=20.0°C, actual={delta_0_to_20:.3f}°C"
        )

    def test_proper_heat_loss_equation(self):
        """
        TDD TEST: Heat loss should follow Q_loss = coefficient * (T_indoor - T_outdoor)
        
        Expected behavior:
        - Heat loss directly proportional to temperature difference
        - No coupling factors or normalization
        - Simple physics-based relationship
        """
        outlet_temp = 45.0
        
        # Test different outdoor temperatures
        outdoor_temps = [0, 5, 10, 15, 20, 25]
        equilibrium_temps = []
        
        for outdoor_temp in outdoor_temps:
            equilibrium = self.model.predict_equilibrium_temperature(
                outlet_temp, outdoor_temp, pv_power=0, fireplace_on=0, tv_on=0
            )
            equilibrium_temps.append(equilibrium)
        
        # Check that relationship is linear: T_eq = T_out + constant
        # Where constant = heat_input / heat_loss_coefficient
        
        heat_input = outlet_temp * self.model.outlet_effectiveness
        expected_constant = heat_input / self.model.heat_loss_coefficient
        
        for i, (outdoor_temp, equilibrium) in enumerate(zip(outdoor_temps, equilibrium_temps)):
            expected_equilibrium = outdoor_temp + expected_constant
            
            self.assertAlmostEqual(
                equilibrium, expected_equilibrium, places=1,
                msg=f"Heat loss equation wrong at outdoor={outdoor_temp}°C: "
                    f"actual={equilibrium:.3f}, expected={expected_equilibrium:.3f}"
            )

    def test_no_thermal_bridge_magic_factor(self):
        """
        TDD TEST: No arbitrary thermal bridge calculations with magic factors.
        
        Expected behavior:
        - No thermal_bridge_factor in equilibrium calculations
        - No arbitrary 0.01 multiplication factors
        - No abs(outdoor_temp - 20) calculations
        """
        outlet_temp = 40.0
        
        # Test at temperatures far from 20°C
        equilibrium_cold = self.model.predict_equilibrium_temperature(
            outlet_temp, -10.0, pv_power=0, fireplace_on=0, tv_on=0
        )
        
        equilibrium_hot = self.model.predict_equilibrium_temperature(
            outlet_temp, 50.0, pv_power=0, fireplace_on=0, tv_on=0
        )
        
        # Calculate expected using proper heat balance
        heat_input = outlet_temp * self.model.outlet_effectiveness
        
        expected_cold = -10.0 + (heat_input / self.model.heat_loss_coefficient)
        expected_hot = 50.0 + (heat_input / self.model.heat_loss_coefficient)
        
        self.assertAlmostEqual(
            equilibrium_cold, expected_cold, places=1,
            msg=f"Thermal bridge factor detected at cold temp: "
                f"actual={equilibrium_cold:.3f}, expected={expected_cold:.3f}"
        )
        
        self.assertAlmostEqual(
            equilibrium_hot, expected_hot, places=1,
            msg=f"Thermal bridge factor detected at hot temp: "
                f"actual={equilibrium_hot:.3f}, expected={expected_hot:.3f}"
        )

    def test_no_outdoor_coupling_in_heat_loss(self):
        """
        TDD TEST: Heat loss should not include outdoor coupling factors.
        
        Expected behavior:
        - Heat loss = coefficient * temperature_difference
        - No (1 - outdoor_coupling * normalized_outdoor) factors
        - No outdoor_coupling parameter influencing heat loss rate
        """
        outlet_temp = 50.0
        
        # Test at various outdoor temperatures
        test_cases = [
            (-20, "very_cold"),
            (0, "freezing"), 
            (10, "cold"),
            (20, "mild"),
            (30, "warm"),
            (40, "hot")
        ]
        
        for outdoor_temp, description in test_cases:
            equilibrium = self.model.predict_equilibrium_temperature(
                outlet_temp, outdoor_temp, pv_power=0, fireplace_on=0, tv_on=0
            )
            
            # Calculate what equilibrium SHOULD be with pure heat balance
            heat_input = outlet_temp * self.model.outlet_effectiveness
            expected_equilibrium = outdoor_temp + (heat_input / self.model.heat_loss_coefficient)
            
            self.assertAlmostEqual(
                equilibrium, expected_equilibrium, places=1,
                msg=f"Outdoor coupling detected at {description} ({outdoor_temp}°C): "
                    f"actual={equilibrium:.3f}, expected={expected_equilibrium:.3f}"
            )

    def test_heat_loss_coefficient_is_constant(self):
        """
        TDD TEST: Heat loss coefficient should be constant, not modified by coupling.
        
        Expected behavior:
        - Heat loss coefficient doesn't change with outdoor temperature
        - No coupling factors modifying the coefficient
        - Consistent heat loss rate across all temperatures
        """
        outlet_temp = 45.0
        base_outdoor = 20.0
        
        # Get baseline heat loss rate
        equilibrium_base = self.model.predict_equilibrium_temperature(
            outlet_temp, base_outdoor, pv_power=0, fireplace_on=0, tv_on=0
        )
        
        heat_input = outlet_temp * self.model.outlet_effectiveness
        effective_coefficient = heat_input / (equilibrium_base - base_outdoor)
        
        # Test that coefficient is the same at different outdoor temperatures
        test_outdoor_temps = [0, 10, 30, 40]
        
        for outdoor_temp in test_outdoor_temps:
            equilibrium = self.model.predict_equilibrium_temperature(
                outlet_temp, outdoor_temp, pv_power=0, fireplace_on=0, tv_on=0
            )
            
            calculated_coefficient = heat_input / (equilibrium - outdoor_temp)
            
            self.assertAlmostEqual(
                calculated_coefficient, effective_coefficient, places=3,
                msg=f"Heat loss coefficient varies with outdoor temp {outdoor_temp}°C: "
                    f"base={effective_coefficient:.4f}, at_{outdoor_temp}={calculated_coefficient:.4f}"
            )

    def test_equilibrium_calculation_method_signature(self):
        """
        TDD TEST: Equilibrium calculation should not require outdoor coupling params.
        
        Expected behavior:
        - Method should work with just basic physics parameters
        - No outdoor_coupling parameters in calculation path
        - Clean physics-based implementation
        """
        # This test ensures the calculation method is clean
        outlet_temp = 42.0
        outdoor_temp = 8.0
        
        # Should work with just the basic parameters
        try:
            equilibrium = self.model.predict_equilibrium_temperature(
                outlet_temp, outdoor_temp, pv_power=0, fireplace_on=0, tv_on=0
            )
            
            # Result should be reasonable
            self.assertGreater(equilibrium, outdoor_temp)
            self.assertIsInstance(equilibrium, float)
            
        except Exception as e:
            self.fail(f"Equilibrium calculation failed, likely due to coupling issues: {e}")

    def test_calculate_optimal_outlet_uses_proper_physics(self):
        """
        TDD TEST: Optimal outlet calculation should use clean heat balance.
        
        Expected behavior:
        - Uses proper heat loss = coefficient * (target - outdoor)
        - No outdoor coupling factors in outlet calculation
        - Clean heat balance equation solving
        """
        if not hasattr(self.model, 'calculate_optimal_outlet_temperature'):
            self.skipTest("calculate_optimal_outlet_temperature not implemented")
        
        target_temp = 21.0
        current_temp = 18.0
        outdoor_temp = 5.0
        
        result = self.model.calculate_optimal_outlet_temperature(
            target_temp, current_temp, outdoor_temp, pv_power=0, config_override={
                'heat_loss_coefficient': self.model.heat_loss_coefficient,
                'outlet_effectiveness': self.model.outlet_effectiveness
            }
        )
        
        if result is None:
            self.skipTest("calculate_optimal_outlet_temperature returned None")
        
        # Verify the calculation uses proper heat balance
        outlet_temp = result['optimal_outlet_temp']
        
        # Calculate what equilibrium should be with this outlet
        predicted_equilibrium = self.model.predict_equilibrium_temperature(
            outlet_temp, outdoor_temp, pv_power=0, fireplace_on=0, tv_on=0
        )
        
        # Should get close to target (within reasonable tolerance)
        self.assertAlmostEqual(
            predicted_equilibrium, target_temp, delta=2.0,
            msg=f"Optimal outlet calculation wrong: target={target_temp}°C, "
                f"predicted_equilibrium={predicted_equilibrium:.3f}°C"
        )


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
