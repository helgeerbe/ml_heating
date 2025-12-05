"""
Test suite for corrected thermal equilibrium physics.

This validates the fundamental physics correction that fixes the critical bug
where heating systems predicted to cool the house.
"""

import pytest
import numpy as np
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from thermal_equilibrium_model import ThermalEquilibriumModel

class TestCorrectedThermalPhysics:
    """Test the corrected thermal equilibrium physics formula."""
    
    def setup_method(self):
        """Set up test model with known parameters."""
        self.model = ThermalEquilibriumModel()
        # Use realistic parameters for testing
        self.model.outlet_effectiveness = 0.1  # 10% heat transfer efficiency
        self.model.heat_loss_coefficient = 0.08  # Moderate heat loss
        
    def test_heating_never_cools_house(self):
        """CRITICAL: Heating with outlet > indoor should never predict cooling."""
        current_indoor = 20.0
        outdoor_temp = 5.0
        outlet_temp = 45.0  # Reasonable heat pump outlet
        
        predicted = self.model.predict_equilibrium_temperature(
            outlet_temp=outlet_temp,
            outdoor_temp=outdoor_temp,
            current_indoor=current_indoor
        )
        
        # Fundamental physics: heating should never cool
        assert predicted >= current_indoor, f"Heating predicted to cool: {predicted}°C < {current_indoor}°C"
        assert predicted >= outdoor_temp, f"Indoor below outdoor: {predicted}°C < {outdoor_temp}°C"
        
    def test_weighted_average_equilibrium(self):
        """Equilibrium should be weighted average of outlet and outdoor temperatures."""
        outdoor_temp = 5.0
        outlet_temp = 45.0
        current_indoor = 20.0
        
        eff = self.model.outlet_effectiveness  # 0.1
        loss = self.model.heat_loss_coefficient  # 0.08
        
        # Expected weighted average (no external heat)
        expected = (eff * outlet_temp + loss * outdoor_temp) / (eff + loss)
        expected = (0.1 * 45 + 0.08 * 5) / (0.1 + 0.08)
        expected = (4.5 + 0.4) / 0.18
        expected = 27.22  # °C
        
        predicted = self.model.predict_equilibrium_temperature(
            outlet_temp=outlet_temp,
            outdoor_temp=outdoor_temp,
            current_indoor=current_indoor
        )
        
        assert abs(predicted - expected) < 0.1, f"Expected {expected:.1f}°C, got {predicted:.1f}°C"
        
    def test_external_heat_additivity(self):
        """External heat sources should add to equilibrium temperature."""
        outdoor_temp = 5.0
        outlet_temp = 35.0
        current_indoor = 20.0
        
        # Baseline without external heat
        baseline = self.model.predict_equilibrium_temperature(
            outlet_temp=outlet_temp,
            outdoor_temp=outdoor_temp,
            current_indoor=current_indoor
        )
        
        # With PV power (should increase temperature)
        with_pv = self.model.predict_equilibrium_temperature(
            outlet_temp=outlet_temp,
            outdoor_temp=outdoor_temp,
            current_indoor=current_indoor,
            pv_power=1000  # 1kW
        )
        
        # With fireplace (should increase temperature more)
        with_fireplace = self.model.predict_equilibrium_temperature(
            outlet_temp=outlet_temp,
            outdoor_temp=outdoor_temp,
            current_indoor=current_indoor,
            fireplace_on=1
        )
        
        assert with_pv > baseline, "PV should increase equilibrium temperature"
        assert with_fireplace > baseline, "Fireplace should increase equilibrium temperature"
        
    def test_realistic_scenario_validation(self):
        """Test the specific scenario from the bug report."""
        # Scenario from bug report
        outdoor_temp = 4.2
        outlet_temp = 65.0  # Maximum heat pump outlet
        current_indoor = 20.6
        pv_power = 589.2  # Watts
        
        predicted = self.model.predict_equilibrium_temperature(
            outlet_temp=outlet_temp,
            outdoor_temp=outdoor_temp,
            current_indoor=current_indoor,
            pv_power=pv_power
        )
        
        # With 65°C outlet, should predict significant heating
        assert predicted > current_indoor, f"65°C outlet should heat above current {current_indoor}°C"
        assert predicted > 25.0, f"65°C outlet should achieve >25°C, got {predicted:.1f}°C"
        assert predicted < outlet_temp, f"Equilibrium {predicted:.1f}°C should be below outlet {outlet_temp}°C"
        
    def test_physics_bounds_enforcement(self):
        """Equilibrium must respect physical bounds."""
        test_cases = [
            # (outlet, outdoor, expected_min, expected_max)
            (45.0, 5.0, 5.0, 45.0),   # Heating mode
            (25.0, 30.0, 25.0, 30.0), # Cooling mode (rare)
            (20.0, 20.0, 20.0, 20.0), # No temperature difference
        ]
        
        for outlet, outdoor, min_bound, max_bound in test_cases:
            predicted = self.model.predict_equilibrium_temperature(
                outlet_temp=outlet,
                outdoor_temp=outdoor,
                current_indoor=20.0
            )
            
            assert min_bound <= predicted <= max_bound, \
                f"Outlet={outlet}°C, Outdoor={outdoor}°C: predicted {predicted:.1f}°C outside bounds [{min_bound}, {max_bound}]"
                
    def test_effectiveness_sensitivity(self):
        """Higher effectiveness should result in temperature closer to outlet."""
        outdoor_temp = 5.0
        outlet_temp = 45.0
        current_indoor = 20.0
        
        # Test with low effectiveness
        self.model.outlet_effectiveness = 0.05
        low_eff_temp = self.model.predict_equilibrium_temperature(
            outlet_temp=outlet_temp,
            outdoor_temp=outdoor_temp,
            current_indoor=current_indoor
        )
        
        # Test with high effectiveness
        self.model.outlet_effectiveness = 0.2
        high_eff_temp = self.model.predict_equilibrium_temperature(
            outlet_temp=outlet_temp,
            outdoor_temp=outdoor_temp,
            current_indoor=current_indoor
        )
        
        assert high_eff_temp > low_eff_temp, "Higher effectiveness should result in higher equilibrium temperature"
        
    def test_zero_division_protection(self):
        """Model should handle edge cases without crashing."""
        # Zero effectiveness and heat loss
        self.model.outlet_effectiveness = 0.0
        self.model.heat_loss_coefficient = 0.0
        
        predicted = self.model.predict_equilibrium_temperature(
            outlet_temp=45.0,
            outdoor_temp=5.0,
            current_indoor=20.0
        )
        
        # Should fallback to outdoor temperature
        assert predicted == 5.0, f"Expected fallback to outdoor temp 5.0°C, got {predicted}°C"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
