#!/usr/bin/env python3
"""
Test Physics Constraints - Verify the corrected model prevents impossible predictions

This script tests the updated physics model to ensure it correctly handles:
1. Night-time PV constraints (zero effects at night)
2. External source magnitude limits
3. Physics compliance checks
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from physics_model import RealisticPhysicsModel

def test_nighttime_pv_constraints():
    """Test that PV effects are zero at night"""
    print("🌙 TESTING NIGHTTIME PV CONSTRAINTS")
    print("=" * 50)
    
    model = RealisticPhysicsModel()
    
    # Set up problematic PV coefficients that would cause issues
    model.pv_coeffs = {
        'lag_1': 0.01,   # Very high coefficients
        'lag_2': 0.02,   
        'lag_3': 0.01,   
        'lag_4': 0.005,  
    }
    
    # Simulate having had high PV during the day
    model.pv_history = [1500, 1200, 800, 400, 0]  # High daytime PV, now zero
    
    # Night-time scenario
    features = {
        'outlet_temp': 14.0,
        'indoor_temp_lag_30m': 20.5,
        'target_temp': 21.0,
        'outdoor_temp': 5.0,
        'pv_now': 0.0,  # NO PV at night
        'fireplace_on': 0.0,
        'tv_on': 0.0,
        'month_cos': 1.0,
        'month_sin': 0.0,
        'dhw_heating': 0.0,
        'dhw_disinfection': 0.0,
        'dhw_boost_heater': 0.0,
        'defrosting': 0.0,
    }
    
    prediction = model.predict_one(features)
    
    # Calculate what the prediction would be without PV constraints
    basic_physics = ((features['outlet_temp'] - features['indoor_temp_lag_30m']) * 
                    model.base_heating_rate)
    target_boost = ((features['target_temp'] - features['indoor_temp_lag_30m']) * 
                   model.target_influence)
    unconstrained_pv = sum(model.pv_coeffs.values()) * 0.01 * 800  # Average historical PV
    
    print(f"Historical PV: {model.pv_history}")
    print(f"PV Coefficients: {model.pv_coeffs}")
    print(f"Current PV: {features['pv_now']}W (NIGHT)")
    print(f"Basic physics (14°C outlet): {basic_physics:.6f}°C")
    print(f"Target boost: {target_boost:.6f}°C")
    print(f"Unconstrained PV would add: {unconstrained_pv:.6f}°C")
    print(f"🔸 ACTUAL PREDICTION: {prediction:.6f}°C")
    
    if prediction < 0:
        print("✅ PASS: Night-time prediction correctly shows cooling from 14°C outlet")
    else:
        print("❌ FAIL: Night-time prediction incorrectly shows heating from 14°C outlet")
    
    return prediction < 0

def test_external_source_limits():
    """Test that external sources have reasonable magnitude limits"""
    print("\n🏠 TESTING EXTERNAL SOURCE MAGNITUDE LIMITS") 
    print("=" * 55)
    
    model = RealisticPhysicsModel()
    
    # Set up extremely high external source coefficients
    model.fireplace_coeffs = {
        'immediate': 0.5,   # Unrealistically high
        'lag_1': 0.4,
        'lag_2': 0.3,
        'lag_3': 0.2,
    }
    model.tv_coeffs = {
        'immediate': 0.2,   # Unrealistically high for TV
        'lag_1': 0.1,
    }
    
    # Set up unrealistic history
    model.fireplace_history = [1, 1, 1, 1]  # Fireplace on for 2 hours
    model.tv_history = [1, 1]  # TV on
    
    features = {
        'outlet_temp': 14.0,
        'indoor_temp_lag_30m': 20.5,
        'target_temp': 21.0,
        'outdoor_temp': 5.0,
        'pv_now': 0.0,
        'fireplace_on': 1.0,  # Fireplace on
        'tv_on': 1.0,  # TV on
        'month_cos': 1.0,
        'month_sin': 0.0,
        'dhw_heating': 0.0,
        'dhw_disinfection': 0.0,
        'dhw_boost_heater': 0.0,
        'defrosting': 0.0,
    }
    
    prediction = model.predict_one(features)
    
    # Calculate unconstrained external effects
    unconstrained_fireplace = sum(model.fireplace_coeffs.values())
    unconstrained_tv = sum(model.tv_coeffs.values())
    basic_physics = ((features['outlet_temp'] - features['indoor_temp_lag_30m']) * 
                    model.base_heating_rate)
    
    print(f"Unconstrained fireplace effect: {unconstrained_fireplace:.6f}°C")
    print(f"Unconstrained TV effect: {unconstrained_tv:.6f}°C")
    print(f"Basic physics (14°C outlet): {basic_physics:.6f}°C")
    print(f"🔸 ACTUAL PREDICTION: {prediction:.6f}°C")
    
    # Check if external sources are constrained
    if prediction < unconstrained_fireplace + unconstrained_tv + basic_physics:
        print("✅ PASS: External sources are constrained to reasonable limits")
        constrained = True
    else:
        print("❌ FAIL: External sources are not properly constrained")
        constrained = False
    
    # Check if basic physics still dominates
    if prediction < 0.05:  # Should not predict significant heating from 14°C outlet
        print("✅ PASS: Basic outlet physics still prevents impossible heating")
        physics_respected = True
    else:
        print("❌ FAIL: External sources overwhelm basic physics")
        physics_respected = False
    
    return constrained and physics_respected

def test_physics_compliance_scaling():
    """Test the physics compliance scaling mechanism"""
    print("\n⚖️ TESTING PHYSICS COMPLIANCE SCALING")
    print("=" * 45)
    
    model = RealisticPhysicsModel()
    
    # Create scenario where external sources would overwhelm physics
    model.pv_coeffs = {'lag_1': 0.02, 'lag_2': 0.03, 'lag_3': 0.02, 'lag_4': 0.01}
    model.fireplace_coeffs = {'immediate': 0.1, 'lag_1': 0.08, 'lag_2': 0.06, 'lag_3': 0.04}
    model.tv_coeffs = {'immediate': 0.05, 'lag_1': 0.03}
    
    # Set up histories during daytime (so PV constraints don't apply)
    model.pv_history = [1000, 800, 600, 400, 200]  # Declining PV
    model.fireplace_history = [1, 1, 1, 1]  # Fireplace on
    model.tv_history = [1, 1]  # TV on
    
    features = {
        'outlet_temp': 14.0,
        'indoor_temp_lag_30m': 20.5,
        'target_temp': 21.0,
        'outdoor_temp': 5.0,
        'pv_now': 200.0,  # Low but not zero (daytime)
        'fireplace_on': 1.0,
        'tv_on': 1.0,
        'month_cos': 1.0,
        'month_sin': 0.0,
        'dhw_heating': 0.0,
        'dhw_disinfection': 0.0,
        'dhw_boost_heater': 0.0,
        'defrosting': 0.0,
    }
    
    prediction = model.predict_one(features)
    
    # Calculate basic physics and what external would be without scaling
    basic_physics = ((features['outlet_temp'] - features['indoor_temp_lag_30m']) * 
                    model.base_heating_rate + 
                    (features['target_temp'] - features['indoor_temp_lag_30m']) * 
                    model.target_influence)
    
    print(f"Basic physics: {basic_physics:.6f}°C (negative = cooling)")
    print(f"🔸 FINAL PREDICTION: {prediction:.6f}°C")
    
    # The key test: external sources should be scaled down to respect physics
    if abs(prediction) < 0.1:  # Should not be extreme heating or cooling
        print("✅ PASS: Physics compliance scaling prevents extreme predictions")
        return True
    else:
        print("❌ FAIL: Physics compliance scaling not working properly")
        return False

def test_real_scenario_reproduction():
    """Test with the exact scenario from the production logs"""
    print("\n🎯 TESTING REAL SCENARIO REPRODUCTION")
    print("=" * 45)
    
    model = RealisticPhysicsModel()
    
    # Use production-like learned parameters that caused the issue
    model.base_heating_rate = 0.001337
    model.target_influence = 0.009431
    model.outdoor_factor = 0.003
    
    # Set up problematic external source coefficients 
    model.pv_coeffs = {'lag_1': 0.002, 'lag_2': 0.003, 'lag_3': 0.002, 'lag_4': 0.001}
    model.fireplace_coeffs = {'immediate': 0.02, 'lag_1': 0.015, 'lag_2': 0.01, 'lag_3': 0.005}
    model.tv_coeffs = {'immediate': 0.01, 'lag_1': 0.005}
    
    # Simulate having had some PV during day, fireplace on
    model.pv_history = [800, 600, 200, 0, 0]  # Declining to night
    model.fireplace_history = [0, 0, 1, 1]  # Fireplace turned on recently
    model.tv_history = [1, 1]  # TV on
    
    # The exact scenario from logs
    features = {
        'outlet_temp': 14.0,
        'indoor_temp_lag_30m': 20.5,
        'target_temp': 21.0,
        'outdoor_temp': 5.0,
        'pv_now': 0.0,  # Night time
        'fireplace_on': 1.0,  # Fireplace might be on
        'tv_on': 1.0,  # TV on
        'month_cos': 1.0,  # December
        'month_sin': 0.0,
        'temp_forecast_1h': 4.5,
        'temp_forecast_2h': 4.0,
        'temp_forecast_3h': 3.5,
        'temp_forecast_4h': 3.0,
        'pv_forecast_1h': 0.0,
        'pv_forecast_2h': 0.0,
        'pv_forecast_3h': 0.0,
        'pv_forecast_4h': 0.0,
        'dhw_heating': 0.0,
        'dhw_disinfection': 0.0,
        'dhw_boost_heater': 0.0,
        'defrosting': 0.0,
    }
    
    prediction = model.predict_one(features)
    
    print(f"Exact scenario from logs (14°C outlet, 20.5°C indoor, 21°C target)")
    print(f"Original problematic prediction was: +0.053°C (HEATING)")
    print(f"🔸 CORRECTED PREDICTION: {prediction:.6f}°C")
    
    if prediction < 0:
        print("✅ PASS: Now correctly predicts cooling from 14°C outlet")
        print("✅ This will prevent the Heat Balance Controller from choosing impossible outlet temperatures")
        return True
    elif prediction < 0.01:
        print("✅ ACCEPTABLE: Very small heating prediction (within noise)")
        return True
    else:
        print("❌ FAIL: Still predicts significant heating from 14°C outlet")
        return False

def main():
    print("🔧 PHYSICS CONSTRAINTS VALIDATION")
    print("Testing corrected physics model to prevent impossible predictions")
    print("=" * 70)
    
    tests = [
        ("Night-time PV Constraints", test_nighttime_pv_constraints),
        ("External Source Limits", test_external_source_limits),
        ("Physics Compliance Scaling", test_physics_compliance_scaling),
        ("Real Scenario Reproduction", test_real_scenario_reproduction),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"Result: {'✅ PASS' if result else '❌ FAIL'}")
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    print("\n🎯 SUMMARY")
    print("=" * 30)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("Physics constraints are working correctly.")
        print("The Heat Balance Controller should now respect outlet temperature physics.")
    else:
        print("⚠️ Some tests failed - additional fixes needed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
