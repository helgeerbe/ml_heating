#!/usr/bin/env python3

"""
Test thermal equilibrium model behavior to debug why it always suggests 65°C
"""

import sys
sys.path.insert(0, 'src')

from thermal_equilibrium_model import ThermalEquilibriumModel

def test_thermal_equilibrium_behavior():
    """Test the thermal equilibrium model with different parameters"""
    
    print("=== THERMAL EQUILIBRIUM DEBUGGING ===")
    
    # Test with learned parameters (causing 65°C suggestions)
    print("\n1. Testing with LEARNED parameters (outlet_effectiveness=0.200):")
    model_learned = ThermalEquilibriumModel()
    model_learned.outlet_effectiveness = 0.200  # Learned value
    model_learned.heat_loss_coefficient = 0.2500
    model_learned.thermal_time_constant = 24.0
    
    test_scenarios = [
        (21.0, 21.1, 5.0),   # Current scenario: target 21°C, current 21.1°C, outdoor 5°C
        (21.0, 20.5, 5.0),   # Need heating: target 21°C, current 20.5°C, outdoor 5°C
        (21.0, 21.5, 5.0),   # Need cooling: target 21°C, current 21.5°C, outdoor 5°C
        (21.0, 21.0, 10.0),  # Perfect temp, warmer outdoor
        (22.0, 21.0, 0.0),   # Higher target, cold outdoor
    ]
    
    for target, current, outdoor in test_scenarios:
        # Test different outlet temperatures to see equilibrium
        print(f"\n  Scenario: target={target}°C, current={current}°C, outdoor={outdoor}°C")
        for outlet_temp in [35, 40, 45, 50, 55, 60]:
            equilibrium = model_learned.predict_equilibrium_temperature(
                outlet_temp=outlet_temp,
                outdoor_temp=outdoor,
                pv_power=0,
                fireplace_on=0,
                tv_on=0
            )
            temp_gap = target - equilibrium
            print(f"    Outlet {outlet_temp}°C → Equilibrium {equilibrium:.2f}°C (Gap: {temp_gap:+.2f}°C)")
    
    print("\n2. Testing with DEFAULT parameters (outlet_effectiveness=0.513):")
    model_default = ThermalEquilibriumModel()
    model_default.outlet_effectiveness = 0.513  # Default value
    model_default.heat_loss_coefficient = 0.2500
    model_default.thermal_time_constant = 24.0
    
    target, current, outdoor = 21.0, 21.1, 5.0  # Current scenario
    print(f"\n  Scenario: target={target}°C, current={current}°C, outdoor={outdoor}°C")
    for outlet_temp in [35, 40, 45, 50, 55, 60]:
        equilibrium = model_default.predict_equilibrium_temperature(
            outlet_temp=outlet_temp,
            outdoor_temp=outdoor,
            pv_power=0,
            fireplace_on=0,
            tv_on=0
        )
        temp_gap = target - equilibrium
        print(f"    Outlet {outlet_temp}°C → Equilibrium {equilibrium:.2f}°C (Gap: {temp_gap:+.2f}°C)")
    
    print("\n3. Testing with CORRECTED parameters (outlet_effectiveness=0.800):")
    model_corrected = ThermalEquilibriumModel()
    model_corrected.outlet_effectiveness = 0.800  # Corrected higher value
    model_corrected.heat_loss_coefficient = 0.2500
    model_corrected.thermal_time_constant = 24.0
    
    print(f"\n  Scenario: target={target}°C, current={current}°C, outdoor={outdoor}°C")
    for outlet_temp in [35, 40, 45, 50, 55, 60]:
        equilibrium = model_corrected.predict_equilibrium_temperature(
            outlet_temp=outlet_temp,
            outdoor_temp=outdoor,
            pv_power=0,
            fireplace_on=0,
            tv_on=0
        )
        temp_gap = target - equilibrium
        print(f"    Outlet {outlet_temp}°C → Equilibrium {equilibrium:.2f}°C (Gap: {temp_gap:+.2f}°C)")
        
        # Find what outlet temperature would achieve target
        if abs(temp_gap) < 0.5:  # Close to target
            print(f"      ✅ GOOD: Outlet {outlet_temp}°C achieves target!")

if __name__ == "__main__":
    test_thermal_equilibrium_behavior()
