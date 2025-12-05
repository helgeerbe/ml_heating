#!/usr/bin/env python3
"""
Parameter Update Script for Corrected Physics Formula

The corrected physics formula requires recalibration of parameters because
the current parameters were tuned to compensate for the broken formula.

Current broken behavior: 65°C outlet → 14.82°C (cooling!)
Target behavior: 65°C outlet → 35-45°C (proper heating)
"""

import sys
import os
sys.path.insert(0, 'src')

from thermal_equilibrium_model import ThermalEquilibriumModel

def analyze_current_parameters():
    """Analyze what the current parameters predict."""
    print("=" * 60)
    print("CURRENT PARAMETERS ANALYSIS")
    print("=" * 60)
    
    model = ThermalEquilibriumModel()
    
    print(f"Current Parameters:")
    print(f"  outlet_effectiveness = {model.outlet_effectiveness}")
    print(f"  heat_loss_coefficient = {model.heat_loss_coefficient}")
    print(f"  pv_weight = {model.external_source_weights['pv']}")
    
    # Test scenarios
    scenarios = [
        {"outlet": 35.0, "outdoor": 5.0, "indoor": 20.0, "pv": 0, "name": "Moderate heating"},
        {"outlet": 45.0, "outdoor": 5.0, "indoor": 20.0, "pv": 0, "name": "Strong heating"},  
        {"outlet": 65.0, "outdoor": 4.2, "indoor": 20.6, "pv": 589.2, "name": "Bug scenario"},
    ]
    
    print(f"\nPrediction Tests:")
    for scenario in scenarios:
        predicted = model.predict_equilibrium_temperature(
            outlet_temp=scenario["outlet"],
            outdoor_temp=scenario["outdoor"], 
            current_indoor=scenario["indoor"],
            pv_power=scenario["pv"]
        )
        change = predicted - scenario["indoor"]
        status = "✅ OK" if change > 0 else "❌ BROKEN"
        print(f"  {scenario['name']}: {scenario['outlet']}°C outlet → {predicted:.1f}°C ({change:+.1f}°C) {status}")

def test_parameter_ranges():
    """Test different parameter combinations to find realistic values."""
    print("\n" + "=" * 60)
    print("PARAMETER RANGE TESTING")
    print("=" * 60)
    
    # Test different effectiveness values with fixed heat_loss
    model = ThermalEquilibriumModel()
    model.heat_loss_coefficient = 0.1  # Reasonable heat loss
    
    print(f"\nTesting outlet_effectiveness (heat_loss = 0.1):")
    effectiveness_values = [0.05, 0.08, 0.12, 0.15, 0.20]
    
    for eff in effectiveness_values:
        model.outlet_effectiveness = eff
        predicted = model.predict_equilibrium_temperature(
            outlet_temp=45.0, outdoor_temp=5.0, current_indoor=20.0
        )
        change = predicted - 20.0
        print(f"  effectiveness = {eff:.2f}: 45°C → {predicted:.1f}°C ({change:+.1f}°C)")
    
    print(f"\nBug scenario with different effectiveness:")
    for eff in effectiveness_values:
        model.outlet_effectiveness = eff
        predicted = model.predict_equilibrium_temperature(
            outlet_temp=65.0, outdoor_temp=4.2, current_indoor=20.6, pv_power=589.2
        )
        change = predicted - 20.6
        status = "✅" if change > 0 else "❌"
        print(f"  effectiveness = {eff:.2f}: {predicted:.1f}°C ({change:+.1f}°C) {status}")

def suggest_new_parameters():
    """Suggest physically reasonable parameters."""
    print("\n" + "=" * 60)
    print("SUGGESTED PARAMETER UPDATE")
    print("=" * 60)
    
    # Physics-based reasoning for parameters:
    # - outlet_effectiveness: How much outlet temp contributes (0.05-0.15 reasonable)
    # - heat_loss_coefficient: Building heat loss rate (0.05-0.15 reasonable)
    # - Balance between them determines indoor temperature
    
    suggested_params = [
        {"eff": 0.08, "loss": 0.12, "name": "Conservative (more outdoor influence)"},
        {"eff": 0.12, "loss": 0.08, "name": "Aggressive (more outlet influence)"},
        {"eff": 0.10, "loss": 0.10, "name": "Balanced"},
    ]
    
    model = ThermalEquilibriumModel()
    
    print("Testing suggested parameter combinations:")
    for params in suggested_params:
        model.outlet_effectiveness = params["eff"]
        model.heat_loss_coefficient = params["loss"]
        
        print(f"\n{params['name']} (eff={params['eff']}, loss={params['loss']}):")
        
        # Test bug scenario
        predicted = model.predict_equilibrium_temperature(
            outlet_temp=65.0, outdoor_temp=4.2, current_indoor=20.6, pv_power=589.2
        )
        change = predicted - 20.6
        status = "✅ FIXED" if change > 5.0 else "⚠️ MARGINAL" if change > 0 else "❌ BROKEN"
        print(f"  Bug scenario: 65°C → {predicted:.1f}°C ({change:+.1f}°C) {status}")
        
        # Test normal scenario  
        predicted2 = model.predict_equilibrium_temperature(
            outlet_temp=45.0, outdoor_temp=5.0, current_indoor=20.0
        )
        change2 = predicted2 - 20.0
        print(f"  Normal heating: 45°C → {predicted2:.1f}°C ({change2:+.1f}°C)")

def generate_config_update():
    """Generate the config.py update needed."""
    print("\n" + "=" * 60)
    print("CONFIG.PY UPDATE NEEDED")
    print("=" * 60)
    
    print("Add these lines to your .env file or update config.py:")
    print()
    print("# CORRECTED PHYSICS PARAMETERS")
    print("OUTLET_EFFECTIVENESS=0.10")
    print("HEAT_LOSS_COEFFICIENT=0.10") 
    print()
    print("Or for more conservative heating:")
    print("OUTLET_EFFECTIVENESS=0.08")
    print("HEAT_LOSS_COEFFICIENT=0.12")

if __name__ == "__main__":
    analyze_current_parameters()
    test_parameter_ranges()
    suggest_new_parameters()
    generate_config_update()
