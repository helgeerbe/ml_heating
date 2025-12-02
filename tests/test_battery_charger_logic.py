#!/usr/bin/env python3
"""
Test the new Battery Charger Logic in Heat Balance Controller

This script tests the two-stage optimization:
1. Stage 1: Maximum Progress Mode (when far from target)
2. Stage 2: Fine Balancing Mode (when close to target)

Tests both heating and cooling scenarios to verify proper outlet temperature selection.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_wrapper import find_best_outlet_temp
from physics_model import RealisticPhysicsModel

# Configure logging to see the optimization logic
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_test_features(current_indoor, target_indoor, outdoor_temp, outlet_temp):
    """Create test features for Heat Balance Controller"""
    return pd.DataFrame([{
        'outlet_temp': outlet_temp,
        'indoor_temp_lag_30m': current_indoor,
        'target_temp': target_indoor,
        'outdoor_temp': outdoor_temp,
        'dhw_heating': 0.0,
        'dhw_disinfection': 0.0,
        'dhw_boost_heater': 0.0,
        'defrosting': 0.0,
        'pv_now': 0.0,
        'fireplace_on': 0.0,
        'tv_on': 0.0,
        'temp_forecast_1h': outdoor_temp,
        'temp_forecast_2h': outdoor_temp,
        'temp_forecast_3h': outdoor_temp,
        'temp_forecast_4h': outdoor_temp,
        'pv_forecast_1h': 0.0,
        'pv_forecast_2h': 0.0,
        'pv_forecast_3h': 0.0,
        'pv_forecast_4h': 0.0,
    }])

def test_battery_charger_logic():
    """Test the battery charger optimization logic"""
    
    print("🔋 TESTING BATTERY CHARGER LOGIC IN HEAT BALANCE CONTROLLER")
    print("=" * 70)
    
    # Initialize model
    model = RealisticPhysicsModel()
    outlet_history = [45.0]  # Starting outlet temperature
    outdoor_temp = 5.0  # Winter conditions
    
    print(f"\n📊 TEST CONDITIONS:")
    print(f"   - Outdoor temperature: {outdoor_temp:.1f}°C")
    print(f"   - Starting outlet: {outlet_history[0]:.1f}°C")
    print(f"   - Precision threshold: 0.2°C (for stage switching)")
    
    test_scenarios = [
        {
            "name": "LARGE HEATING GAP",
            "current": 19.5,
            "target": 21.0,
            "gap": 1.5,
            "expected_stage": "Stage 1: Maximum Progress",
            "expected_behavior": "Select maximum heating (65°C)"
        },
        {
            "name": "LARGE COOLING GAP", 
            "current": 22.8,
            "target": 21.0,
            "gap": -1.8,
            "expected_stage": "Stage 1: Maximum Progress",
            "expected_behavior": "Select maximum cooling (14°C)"
        },
        {
            "name": "SMALL HEATING GAP",
            "current": 20.85,
            "target": 21.0,
            "gap": 0.15,
            "expected_stage": "Stage 2: Fine Balancing",
            "expected_behavior": "Precise optimization for target"
        },
        {
            "name": "SMALL COOLING GAP",
            "current": 21.18,
            "target": 21.0,
            "gap": -0.18,
            "expected_stage": "Stage 2: Fine Balancing", 
            "expected_behavior": "Precise optimization for target"
        },
    ]
    
    results = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🧪 TEST {i}: {scenario['name']}")
        print(f"   Current: {scenario['current']:.2f}°C")
        print(f"   Target:  {scenario['target']:.1f}°C")
        print(f"   Gap:     {scenario['gap']:+.2f}°C")
        print(f"   Expected: {scenario['expected_behavior']}")
        print("-" * 50)
        
        # Create features
        features = create_test_features(
            scenario['current'], 
            scenario['target'], 
            outdoor_temp,
            outlet_history[0]
        )
        
        # Run Heat Balance Controller
        result = find_best_outlet_temp(
            model=model,
            features=features,
            current_temp=scenario['current'],
            target_temp=scenario['target'],
            outlet_history=outlet_history,
            error_target_vs_actual=scenario['gap'],
            outdoor_temp=outdoor_temp
        )
        
        selected_outlet, confidence, control_mode, sigma, score, trajectory, outlet_range = result
        
        # Analyze results
        stage_detected = "Stage 1" if abs(scenario['gap']) > 0.2 else "Stage 2"
        
        # Verify correct logic
        if abs(scenario['gap']) > 0.2:  # Stage 1: Maximum Progress
            if scenario['gap'] > 0:  # Need heating
                correct_logic = selected_outlet >= 60.0  # Should select high temp
                logic_desc = f"Heating: {selected_outlet:.0f}°C ({'✅ CORRECT' if correct_logic else '❌ WRONG - should be ≥60°C'})"
            else:  # Need cooling
                correct_logic = selected_outlet <= 20.0  # Should select low temp  
                logic_desc = f"Cooling: {selected_outlet:.0f}°C ({'✅ CORRECT' if correct_logic else '❌ WRONG - should be ≤20°C'})"
        else:  # Stage 2: Fine Balancing
            correct_logic = 30.0 <= selected_outlet <= 55.0  # Reasonable precision range
            logic_desc = f"Precision: {selected_outlet:.0f}°C ({'✅ CORRECT' if correct_logic else '❌ WRONG - unexpected value'})"
        
        # Store results
        results.append({
            'scenario': scenario['name'],
            'gap': scenario['gap'],
            'stage': stage_detected,
            'outlet': selected_outlet,
            'correct': correct_logic,
            'mode': control_mode
        })
        
        print(f"🎯 RESULT: {logic_desc}")
        print(f"   Mode: {control_mode}")
        print(f"   Stage: {stage_detected}")
        print(f"   Confidence: {confidence:.3f}")
        
    # Summary
    print("\n" + "=" * 70)
    print("📈 BATTERY CHARGER LOGIC TEST SUMMARY")
    print("=" * 70)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['correct'])
    
    for result in results:
        status = "✅ PASS" if result['correct'] else "❌ FAIL"
        print(f"{status} {result['scenario']}: {result['outlet']:.0f}°C (Gap: {result['gap']:+.2f}°C, {result['stage']})")
    
    print(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.0f}%)")
    
    if passed_tests == total_tests:
        print("🏆 SUCCESS: Battery charger logic working correctly!")
        print("   ✅ Maximum progress mode for large gaps")
        print("   ✅ Fine balancing mode for small gaps")
        print("   ✅ Proper heating/cooling outlet selection")
    else:
        print("❌ ISSUES DETECTED: Some tests failed - check optimization logic")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = test_battery_charger_logic()
    sys.exit(0 if success else 1)
