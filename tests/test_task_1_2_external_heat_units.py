"""
TDD Unit Tests for Task 1.2: Correct External Heat Source Units

These tests define the EXPECTED behavior for external heat source unit corrections.
Following TDD methodology - tests written FIRST, then implementation follows.

Task 1.2 Requirements:
- Standardize fireplace/tv units to °C (direct temperature contribution)
- Standardize PV units to °C/kW (temperature rise per kilowatt)  
- Update equilibrium calculations to use consistent units
- Add unit validation in thermal config
"""

import unittest
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from thermal_equilibrium_model import ThermalEquilibriumModel


class TestTask12ExternalHeatUnits(unittest.TestCase):
    """TDD tests for external heat source unit corrections (Task 1.2)."""
    
    def setUp(self):
        """Set up test model."""
        self.model = ThermalEquilibriumModel()
        
        # Set predictable parameters for testing
        self.model.thermal_time_constant = 4.0
        self.model.heat_loss_coefficient = 0.05
        self.model.outlet_effectiveness = 0.8

    def test_pv_units_are_celsius_per_kilowatt(self):
        """
        TDD TEST: PV units should be °C/kW (temperature rise per kilowatt).
        
        Expected behavior:
        - 1000W (1kW) PV should contribute X°C to equilibrium temperature
        - 2000W (2kW) PV should contribute 2X°C to equilibrium temperature
        - Linear relationship between PV power and temperature contribution
        """
        outlet_temp = 40.0
        outdoor_temp = 10.0
        
        # Test with 1kW PV
        equilibrium_1kw = self.model.predict_equilibrium_temperature(
            outlet_temp, outdoor_temp, pv_power=1000, fireplace_on=0, tv_on=0
        )
        
        # Test with 2kW PV
        equilibrium_2kw = self.model.predict_equilibrium_temperature(
            outlet_temp, outdoor_temp, pv_power=2000, fireplace_on=0, tv_on=0
        )
        
        # Test without PV
        equilibrium_no_pv = self.model.predict_equilibrium_temperature(
            outlet_temp, outdoor_temp, pv_power=0, fireplace_on=0, tv_on=0
        )
        
        # Calculate actual contributions
        pv_contribution_1kw = equilibrium_1kw - equilibrium_no_pv
        pv_contribution_2kw = equilibrium_2kw - equilibrium_no_pv
        
        # Test linear relationship (2kW should give exactly 2x the contribution)
        self.assertAlmostEqual(
            pv_contribution_2kw, 2 * pv_contribution_1kw, places=2,
            msg=f"PV units not linear: 1kW={pv_contribution_1kw:.3f}°C, 2kW={pv_contribution_2kw:.3f}°C"
        )
        
        # Test that PV weight is in reasonable °C/kW range (0.001-0.01 °C/W)
        pv_weight = self.model.external_source_weights['pv']
        expected_contribution_per_kw = (1000 * pv_weight) / self.model.heat_loss_coefficient
        
        self.assertAlmostEqual(
            pv_contribution_1kw, expected_contribution_per_kw, places=1,
            msg=f"PV contribution calculation wrong: actual={pv_contribution_1kw:.3f}, expected={expected_contribution_per_kw:.3f}"
        )

    def test_fireplace_units_are_direct_celsius(self):
        """
        TDD TEST: Fireplace units should be °C (direct temperature contribution).
        
        Expected behavior:
        - fireplace_on=1 should add X°C directly to equilibrium
        - fireplace_on=2 should add 2X°C directly to equilibrium  
        - Linear relationship independent of PV power
        """
        outlet_temp = 35.0
        outdoor_temp = 5.0
        
        # Test with fireplace on level 1
        equilibrium_fire_1 = self.model.predict_equilibrium_temperature(
            outlet_temp, outdoor_temp, pv_power=0, fireplace_on=1, tv_on=0
        )
        
        # Test with fireplace on level 2
        equilibrium_fire_2 = self.model.predict_equilibrium_temperature(
            outlet_temp, outdoor_temp, pv_power=0, fireplace_on=2, tv_on=0
        )
        
        # Test without fireplace
        equilibrium_no_fire = self.model.predict_equilibrium_temperature(
            outlet_temp, outdoor_temp, pv_power=0, fireplace_on=0, tv_on=0
        )
        
        # Calculate contributions
        fire_contribution_1 = equilibrium_fire_1 - equilibrium_no_fire
        fire_contribution_2 = equilibrium_fire_2 - equilibrium_no_fire
        
        # Test linear relationship
        self.assertAlmostEqual(
            fire_contribution_2, 2 * fire_contribution_1, places=2,
            msg=f"Fireplace units not linear: level_1={fire_contribution_1:.3f}°C, level_2={fire_contribution_2:.3f}°C"
        )
        
        # Test direct contribution calculation
        fire_weight = self.model.external_source_weights['fireplace']
        expected_contribution = fire_weight / self.model.heat_loss_coefficient
        
        self.assertAlmostEqual(
            fire_contribution_1, expected_contribution, places=1,
            msg=f"Fireplace contribution wrong: actual={fire_contribution_1:.3f}, expected={expected_contribution:.3f}"
        )

    def test_tv_units_are_direct_celsius(self):
        """
        TDD TEST: TV units should be °C (direct temperature contribution).
        
        Expected behavior:
        - tv_on=1 should add a small but measurable temperature increase
        - Multiple TVs (tv_on=2) should add proportionally more
        """
        outlet_temp = 30.0
        outdoor_temp = 15.0
        
        # Test with TV on
        equilibrium_tv_on = self.model.predict_equilibrium_temperature(
            outlet_temp, outdoor_temp, pv_power=0, fireplace_on=0, tv_on=1
        )
        
        # Test without TV
        equilibrium_tv_off = self.model.predict_equilibrium_temperature(
            outlet_temp, outdoor_temp, pv_power=0, fireplace_on=0, tv_on=0
        )
        
        # Calculate TV contribution
        tv_contribution = equilibrium_tv_on - equilibrium_tv_off
        
        # TV should contribute something measurable but small (0.1-2°C)
        self.assertGreater(tv_contribution, 0.05, msg="TV contribution too small")
        self.assertLess(tv_contribution, 5.0, msg="TV contribution too large")
        
        # Test direct contribution calculation
        tv_weight = self.model.external_source_weights['tv']
        expected_contribution = tv_weight / self.model.heat_loss_coefficient
        
        self.assertAlmostEqual(
            tv_contribution, expected_contribution, places=2,
            msg=f"TV contribution wrong: actual={tv_contribution:.3f}, expected={expected_contribution:.3f}"
        )

    def test_external_sources_are_additive(self):
        """
        TDD TEST: All external heat sources should be properly additive.
        
        Expected behavior:
        - Total contribution = PV + Fireplace + TV contributions
        - No interference between different heat sources
        """
        outlet_temp = 45.0
        outdoor_temp = 8.0
        pv_power = 1500  # 1.5kW
        fireplace_level = 1
        tv_count = 1
        
        # Test individual contributions
        equilibrium_baseline = self.model.predict_equilibrium_temperature(
            outlet_temp, outdoor_temp, pv_power=0, fireplace_on=0, tv_on=0
        )
        
        equilibrium_pv_only = self.model.predict_equilibrium_temperature(
            outlet_temp, outdoor_temp, pv_power=pv_power, fireplace_on=0, tv_on=0
        )
        
        equilibrium_fire_only = self.model.predict_equilibrium_temperature(
            outlet_temp, outdoor_temp, pv_power=0, fireplace_on=fireplace_level, tv_on=0
        )
        
        equilibrium_tv_only = self.model.predict_equilibrium_temperature(
            outlet_temp, outdoor_temp, pv_power=0, fireplace_on=0, tv_on=tv_count
        )
        
        # Test all combined
        equilibrium_all = self.model.predict_equilibrium_temperature(
            outlet_temp, outdoor_temp, pv_power=pv_power, fireplace_on=fireplace_level, tv_on=tv_count
        )
        
        # Calculate individual contributions
        pv_contribution = equilibrium_pv_only - equilibrium_baseline
        fire_contribution = equilibrium_fire_only - equilibrium_baseline
        tv_contribution = equilibrium_tv_only - equilibrium_baseline
        
        # Calculate expected total
        expected_total_contribution = pv_contribution + fire_contribution + tv_contribution
        actual_total_contribution = equilibrium_all - equilibrium_baseline
        
        self.assertAlmostEqual(
            actual_total_contribution, expected_total_contribution, places=1,
            msg=f"External sources not additive: actual={actual_total_contribution:.3f}, expected={expected_total_contribution:.3f}"
        )

    def test_unit_consistency_across_calculations(self):
        """
        TDD TEST: Units should be consistent across all equilibrium calculations.
        
        Expected behavior:
        - All external source contributions should be in same units (°C effect)
        - Heat balance equation should use consistent units throughout
        """
        # Test with realistic values
        outlet_temp = 50.0
        outdoor_temp = 0.0  # Use 0°C for easy math
        
        equilibrium_temp = self.model.predict_equilibrium_temperature(
            outlet_temp, outdoor_temp, 
            pv_power=1000,  # 1kW
            fireplace_on=1, # Level 1
            tv_on=1         # 1 TV
        )
        
        # Calculate expected using heat balance
        heat_from_outlet = outlet_temp * self.model.outlet_effectiveness
        heat_from_pv = 1000 * self.model.external_source_weights['pv']
        heat_from_fireplace = 1 * self.model.external_source_weights['fireplace']
        heat_from_tv = 1 * self.model.external_source_weights['tv']
        
        total_heat_input = heat_from_outlet + heat_from_pv + heat_from_fireplace + heat_from_tv
        expected_equilibrium = outdoor_temp + (total_heat_input / self.model.heat_loss_coefficient)
        
        self.assertAlmostEqual(
            equilibrium_temp, expected_equilibrium, places=1,
            msg=f"Unit consistency failed: actual={equilibrium_temp:.3f}, expected={expected_equilibrium:.3f}"
        )

    def test_external_source_weights_are_physically_reasonable(self):
        """
        TDD TEST: External source weights should be in physically reasonable ranges.
        
        Expected ranges:
        - PV: 0.001-0.01 °C/W (1-10°C per kW)
        - Fireplace: 2-10°C (significant heating)
        - TV: 0.1-2°C (small but measurable)
        """
        pv_weight = self.model.external_source_weights['pv']
        fire_weight = self.model.external_source_weights['fireplace']
        tv_weight = self.model.external_source_weights['tv']
        
        # Test PV weight is in reasonable range (°C/W)
        self.assertGreaterEqual(pv_weight, 0.0005, msg=f"PV weight too small: {pv_weight}")
        self.assertLessEqual(pv_weight, 0.02, msg=f"PV weight too large: {pv_weight}")
        
        # Test fireplace weight is reasonable (°C)
        self.assertGreaterEqual(fire_weight, 1.0, msg=f"Fireplace weight too small: {fire_weight}")
        self.assertLessEqual(fire_weight, 15.0, msg=f"Fireplace weight too large: {fire_weight}")
        
        # Test TV weight is reasonable (°C)
        self.assertGreaterEqual(tv_weight, 0.05, msg=f"TV weight too small: {tv_weight}")
        self.assertLessEqual(tv_weight, 3.0, msg=f"TV weight too large: {tv_weight}")


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
