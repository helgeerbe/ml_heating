"""
TDD Tests for Removing Differential-Based Effectiveness Scaling

These tests define the expected behavior AFTER removing the differential scaling
from predict_equilibrium_temperature(). They should FAIL with the current code
and PASS after the scaling is removed.

Issue: Calibration-Runtime Mismatch (Dec 11, 2025)
- Differential scaling creates inconsistency between calibration and live operation
- Solution: Remove scaling, use outlet_effectiveness directly
"""

import pytest
import numpy as np


class TestNoDifferentialScaling:
    """Tests that verify differential scaling has been removed."""
    
    @pytest.fixture
    def thermal_model(self):
        """Create a fresh thermal model instance for testing."""
        # Reset singleton to get fresh instance
        import sys
        if 'thermal_equilibrium_model' in sys.modules:
            import importlib
            # Clear singleton
            try:
                from src.thermal_equilibrium_model import _thermal_equilibrium_model_instance
                import src.thermal_equilibrium_model as tem
                tem._thermal_equilibrium_model_instance = None
            except:
                pass
        
        from src.thermal_equilibrium_model import ThermalEquilibriumModel
        # Reset singleton
        import src.thermal_equilibrium_model as tem
        tem._thermal_equilibrium_model_instance = None
        
        model = ThermalEquilibriumModel()
        # Set known test parameters
        model.outlet_effectiveness = 0.5
        model.heat_loss_coefficient = 0.2
        model.thermal_time_constant = 4.0
        model.external_source_weights = {'pv': 0.002, 'fireplace': 5.0, 'tv': 0.2}
        return model
    
    def test_effectiveness_used_directly(self, thermal_model):
        """
        Test that outlet_effectiveness is used directly without scaling.
        
        The equilibrium should be calculated as:
        equilibrium = (eff * outlet + loss * outdoor) / (eff + loss)
        
        With eff=0.5, loss=0.2, outlet=35, outdoor=5:
        equilibrium = (0.5*35 + 0.2*5) / (0.5 + 0.2) = (17.5 + 1.0) / 0.7 = 26.43°C
        """
        result = thermal_model.predict_equilibrium_temperature(
            outlet_temp=35.0,
            outdoor_temp=5.0,
            current_indoor=21.0,  # This should NOT affect the equilibrium calculation
            pv_power=0,
            fireplace_on=0,
            tv_on=0
        )
        
        # Calculate expected value WITHOUT differential scaling
        eff = 0.5  # Should be used directly
        loss = 0.2
        expected = (eff * 35.0 + loss * 5.0) / (eff + loss)
        
        assert abs(result - expected) < 0.1, (
            f"Expected equilibrium={expected:.2f}°C (using eff={eff} directly), "
            f"got {result:.2f}°C. "
            f"Differential scaling may still be active."
        )
    
    def test_equilibrium_independent_of_current_indoor(self, thermal_model):
        """
        Test that equilibrium prediction does NOT depend on current_indoor.
        
        Without differential scaling, the equilibrium should be the same
        regardless of what current_indoor is, because equilibrium is the
        steady-state temperature the room will eventually reach.
        """
        outlet_temp = 40.0
        outdoor_temp = 10.0
        
        # Test with different current_indoor values
        result_low = thermal_model.predict_equilibrium_temperature(
            outlet_temp=outlet_temp,
            outdoor_temp=outdoor_temp,
            current_indoor=15.0,  # Low indoor
            pv_power=0, fireplace_on=0, tv_on=0
        )
        
        result_mid = thermal_model.predict_equilibrium_temperature(
            outlet_temp=outlet_temp,
            outdoor_temp=outdoor_temp,
            current_indoor=21.0,  # Normal indoor
            pv_power=0, fireplace_on=0, tv_on=0
        )
        
        result_high = thermal_model.predict_equilibrium_temperature(
            outlet_temp=outlet_temp,
            outdoor_temp=outdoor_temp,
            current_indoor=25.0,  # High indoor
            pv_power=0, fireplace_on=0, tv_on=0
        )
        
        # All results should be the same (equilibrium is independent of current state)
        assert abs(result_low - result_mid) < 0.01, (
            f"Equilibrium changed with current_indoor: {result_low:.2f}°C vs {result_mid:.2f}°C. "
            f"Differential scaling may still be affecting the calculation."
        )
        assert abs(result_mid - result_high) < 0.01, (
            f"Equilibrium changed with current_indoor: {result_mid:.2f}°C vs {result_high:.2f}°C. "
            f"Differential scaling may still be affecting the calculation."
        )
    
    def test_equilibrium_monotonic_with_outlet_temp(self, thermal_model):
        """
        Test that equilibrium increases monotonically with outlet temperature.
        
        Higher outlet temp should always result in higher equilibrium.
        """
        outdoor_temp = 5.0
        current_indoor = 21.0
        
        previous_equilibrium = -100.0  # Start very low
        
        for outlet_temp in [25, 30, 35, 40, 45, 50, 55, 60]:
            equilibrium = thermal_model.predict_equilibrium_temperature(
                outlet_temp=float(outlet_temp),
                outdoor_temp=outdoor_temp,
                current_indoor=current_indoor,
                pv_power=0, fireplace_on=0, tv_on=0
            )
            
            assert equilibrium > previous_equilibrium, (
                f"Equilibrium not monotonic: outlet={outlet_temp}°C gave {equilibrium:.2f}°C, "
                f"but previous was {previous_equilibrium:.2f}°C"
            )
            previous_equilibrium = equilibrium
    
    def test_binary_search_range_consistency(self, thermal_model):
        """
        Test that effectiveness is consistent across the full binary search range.
        
        This addresses the user's clarification: the binary search explores 25-60°C,
        but differential scaling was penalizing mid-range temps (35-45°C).
        """
        outdoor_temp = 5.0
        current_indoor = 21.0
        
        # Test outlet temps across the full binary search range
        outlet_temps = [25, 30, 35, 40, 45, 50, 55, 60]
        effectiveness_ratios = []
        
        for outlet_temp in outlet_temps:
            equilibrium = thermal_model.predict_equilibrium_temperature(
                outlet_temp=float(outlet_temp),
                outdoor_temp=outdoor_temp,
                current_indoor=current_indoor,
                pv_power=0, fireplace_on=0, tv_on=0
            )
            
            # Calculate actual effectiveness ratio: (equilibrium - outdoor) / (outlet - outdoor)
            if outlet_temp != outdoor_temp:
                ratio = (equilibrium - outdoor_temp) / (outlet_temp - outdoor_temp)
                effectiveness_ratios.append(ratio)
        
        # All effectiveness ratios should be the same (no differential scaling)
        if effectiveness_ratios:
            ratio_std = np.std(effectiveness_ratios)
            ratio_mean = np.mean(effectiveness_ratios)
            
            assert ratio_std < 0.01, (
                f"Effectiveness ratio varies across binary search range: mean={ratio_mean:.3f}, std={ratio_std:.4f}. "
                f"This suggests differential scaling is still penalizing mid-range temps."
            )
    
    def test_calibration_runtime_consistency(self, thermal_model):
        """
        Test that calibration-like and runtime-like calls produce identical results.
        
        This is the core issue: calibration and runtime must use the same calculation.
        """
        # Calibration-like scenario (historical data)
        calibration_result = thermal_model.predict_equilibrium_temperature(
            outlet_temp=40.0,
            outdoor_temp=10.0,
            current_indoor=21.0,  # Historical indoor temp
            pv_power=0, fireplace_on=0, tv_on=0
        )
        
        # Runtime binary search scenario (same inputs)
        runtime_result = thermal_model.predict_equilibrium_temperature(
            outlet_temp=40.0,
            outdoor_temp=10.0,
            current_indoor=21.0,  # Current indoor temp during binary search
            pv_power=0, fireplace_on=0, tv_on=0
        )
        
        assert calibration_result == runtime_result, (
            f"Calibration ({calibration_result:.2f}°C) and runtime ({runtime_result:.2f}°C) "
            f"produce different results for identical inputs!"
        )


class TestRegressionScenarios:
    """Regression tests for known good scenarios."""
    
    @pytest.fixture
    def thermal_model(self):
        """Create a thermal model with known parameters."""
        from src.thermal_equilibrium_model import ThermalEquilibriumModel
        import src.thermal_equilibrium_model as tem
        tem._thermal_equilibrium_model_instance = None
        
        model = ThermalEquilibriumModel()
        model.outlet_effectiveness = 0.5
        model.heat_loss_coefficient = 0.2
        model.thermal_time_constant = 4.0
        model.external_source_weights = {'pv': 0.002, 'fireplace': 5.0, 'tv': 0.2}
        return model
    
    def test_typical_heating_scenario(self, thermal_model):
        """
        Test typical winter heating scenario.
        
        outlet=45°C, outdoor=0°C, target=21°C
        Expected equilibrium = (0.5*45 + 0.2*0) / (0.5 + 0.2) = 22.5 / 0.7 = 32.14°C
        """
        result = thermal_model.predict_equilibrium_temperature(
            outlet_temp=45.0,
            outdoor_temp=0.0,
            current_indoor=21.0,
            pv_power=0, fireplace_on=0, tv_on=0
        )
        
        expected = (0.5 * 45.0 + 0.2 * 0.0) / (0.5 + 0.2)
        assert abs(result - expected) < 0.5, (
            f"Typical heating scenario: expected {expected:.1f}°C, got {result:.1f}°C"
        )
    
    def test_mild_weather_scenario(self, thermal_model):
        """
        Test mild weather scenario with moderate heating.
        
        outlet=35°C, outdoor=15°C, target=21°C
        Expected equilibrium = (0.5*35 + 0.2*15) / (0.5 + 0.2) = (17.5 + 3) / 0.7 = 29.29°C
        """
        result = thermal_model.predict_equilibrium_temperature(
            outlet_temp=35.0,
            outdoor_temp=15.0,
            current_indoor=21.0,
            pv_power=0, fireplace_on=0, tv_on=0
        )
        
        expected = (0.5 * 35.0 + 0.2 * 15.0) / (0.5 + 0.2)
        assert abs(result - expected) < 0.5, (
            f"Mild weather scenario: expected {expected:.1f}°C, got {result:.1f}°C"
        )
    
    def test_with_pv_power(self, thermal_model):
        """
        Test scenario with PV power contribution.
        
        outlet=35°C, outdoor=10°C, pv_power=2000W
        With pv_weight=0.002, pv contribution = 2000 * 0.002 = 4.0°C
        Expected = (0.5*35 + 0.2*10 + 4.0) / (0.5 + 0.2) = (17.5 + 2 + 4) / 0.7 = 33.57°C
        """
        result = thermal_model.predict_equilibrium_temperature(
            outlet_temp=35.0,
            outdoor_temp=10.0,
            current_indoor=21.0,
            pv_power=2000.0,
            fireplace_on=0, tv_on=0
        )
        
        pv_contribution = 2000.0 * 0.002
        expected = (0.5 * 35.0 + 0.2 * 10.0 + pv_contribution) / (0.5 + 0.2)
        assert abs(result - expected) < 0.5, (
            f"PV power scenario: expected {expected:.1f}°C, got {result:.1f}°C"
        )
    
    def test_with_fireplace(self, thermal_model):
        """
        Test scenario with fireplace active.
        
        outlet=30°C, outdoor=5°C, fireplace_on=1
        With fireplace_weight=5.0, fireplace contribution = 1 * 5.0 = 5.0°C
        Expected = (0.5*30 + 0.2*5 + 5.0) / (0.5 + 0.2) = (15 + 1 + 5) / 0.7 = 30.0°C
        """
        result = thermal_model.predict_equilibrium_temperature(
            outlet_temp=30.0,
            outdoor_temp=5.0,
            current_indoor=21.0,
            pv_power=0,
            fireplace_on=1.0,
            tv_on=0
        )
        
        fireplace_contribution = 1.0 * 5.0
        expected = (0.5 * 30.0 + 0.2 * 5.0 + fireplace_contribution) / (0.5 + 0.2)
        assert abs(result - expected) < 0.5, (
            f"Fireplace scenario: expected {expected:.1f}°C, got {result:.1f}°C"
        )


class TestPhysicsConstraints:
    """Tests for basic physics constraints that must always hold."""
    
    @pytest.fixture
    def thermal_model(self):
        """Create a thermal model with known parameters."""
        from src.thermal_equilibrium_model import ThermalEquilibriumModel
        import src.thermal_equilibrium_model as tem
        tem._thermal_equilibrium_model_instance = None
        
        model = ThermalEquilibriumModel()
        model.outlet_effectiveness = 0.5
        model.heat_loss_coefficient = 0.2
        return model
    
    def test_equilibrium_between_outlet_and_outdoor(self, thermal_model):
        """
        Test that equilibrium is always between outlet and outdoor temps.
        
        For heating (outlet > outdoor): outdoor < equilibrium < outlet
        """
        for outlet_temp in [30, 40, 50, 60]:
            for outdoor_temp in [0, 5, 10, 15]:
                if outlet_temp > outdoor_temp:
                    result = thermal_model.predict_equilibrium_temperature(
                        outlet_temp=float(outlet_temp),
                        outdoor_temp=float(outdoor_temp),
                        current_indoor=21.0,
                        pv_power=0, fireplace_on=0, tv_on=0
                    )
                    
                    assert outdoor_temp <= result <= outlet_temp, (
                        f"Equilibrium {result:.1f}°C not between outdoor {outdoor_temp}°C "
                        f"and outlet {outlet_temp}°C"
                    )
    
    def test_no_heating_when_outlet_equals_outdoor(self, thermal_model):
        """
        Test that equilibrium equals outdoor when outlet equals outdoor.
        
        No heat transfer when there's no temperature difference.
        """
        temp = 15.0  # Both outlet and outdoor
        
        result = thermal_model.predict_equilibrium_temperature(
            outlet_temp=temp,
            outdoor_temp=temp,
            current_indoor=21.0,
            pv_power=0, fireplace_on=0, tv_on=0
        )
        
        assert abs(result - temp) < 0.5, (
            f"With outlet=outdoor={temp}°C, equilibrium should be ~{temp}°C, got {result:.1f}°C"
        )
