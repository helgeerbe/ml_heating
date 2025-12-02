#!/usr/bin/env python3
"""
Analyze Log Discrepancy - Find why production logs show different prediction

The production model correctly predicts cooling (-0.007°C) but logs show heating (+0.053°C).
This script investigates what's causing this discrepancy.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model_wrapper import load_model

def test_various_feature_combinations():
    """Test different feature combinations to reproduce the +0.053°C prediction"""
    
    print("🔍 TESTING FEATURE COMBINATIONS TO REPRODUCE +0.053°C")
    print("=" * 65)
    
    model, mae, rmse = load_model()
    
    base_features = {
        'outlet_temp': 14.0,
        'indoor_temp_lag_30m': 20.5,
        'target_temp': 21.0,
        'outdoor_temp': 5.0,
        'dhw_heating': 0.0,
        'dhw_disinfection': 0.0,
        'dhw_boost_heater': 0.0,
        'defrosting': 0.0,
        'pv_now': 0.0,
        'fireplace_on': 0.0,
        'tv_on': 0.0,
    }
    
    test_cases = [
        {
            'name': "Base case (night time)",
            'features': {**base_features,
                        'month_cos': 1.0,
                        'month_sin': 0.0,
                        'temp_forecast_1h': 4.5,
                        'temp_forecast_2h': 4.0,
                        'temp_forecast_3h': 3.5,
                        'temp_forecast_4h': 3.0,
                        'pv_forecast_1h': 0.0,
                        'pv_forecast_2h': 0.0,
                        'pv_forecast_3h': 0.0,
                        'pv_forecast_4h': 0.0}
        },
        {
            'name': "With historical PV (simulating daytime history)",
            'features': {**base_features,
                        'month_cos': 1.0,
                        'month_sin': 0.0,
                        'temp_forecast_1h': 4.5,
                        'temp_forecast_2h': 4.0,
                        'temp_forecast_3h': 3.5,
                        'temp_forecast_4h': 3.0,
                        'pv_forecast_1h': 0.0,
                        'pv_forecast_2h': 0.0,
                        'pv_forecast_3h': 0.0,
                        'pv_forecast_4h': 0.0}
        },
        {
            'name': "With TV on",
            'features': {**base_features,
                        'tv_on': 1.0,  # TV is on
                        'month_cos': 1.0,
                        'month_sin': 0.0,
                        'temp_forecast_1h': 4.5,
                        'temp_forecast_2h': 4.0,
                        'temp_forecast_3h': 3.5,
                        'temp_forecast_4h': 3.0,
                        'pv_forecast_1h': 0.0,
                        'pv_forecast_2h': 0.0,
                        'pv_forecast_3h': 0.0,
                        'pv_forecast_4h': 0.0}
        },
        {
            'name': "With fireplace on",
            'features': {**base_features,
                        'fireplace_on': 1.0,  # Fireplace is on
                        'month_cos': 1.0,
                        'month_sin': 0.0,
                        'temp_forecast_1h': 4.5,
                        'temp_forecast_2h': 4.0,
                        'temp_forecast_3h': 3.5,
                        'temp_forecast_4h': 3.0,
                        'pv_forecast_1h': 0.0,
                        'pv_forecast_2h': 0.0,
                        'pv_forecast_3h': 0.0,
                        'pv_forecast_4h': 0.0}
        },
        {
            'name': "With different seasonal (actual December)",
            'features': {**base_features,
                        'month_cos': 0.866,  # cos(2π*11/12) for December
                        'month_sin': 0.5,    # sin(2π*11/12) for December
                        'temp_forecast_1h': 4.5,
                        'temp_forecast_2h': 4.0,
                        'temp_forecast_3h': 3.5,
                        'temp_forecast_4h': 3.0,
                        'pv_forecast_1h': 0.0,
                        'pv_forecast_2h': 0.0,
                        'pv_forecast_3h': 0.0,
                        'pv_forecast_4h': 0.0}
        },
        {
            'name': "With warmer forecasts (less cooling predicted)",
            'features': {**base_features,
                        'month_cos': 1.0,
                        'month_sin': 0.0,
                        'temp_forecast_1h': 8.0,   # Warmer forecasts
                        'temp_forecast_2h': 10.0,
                        'temp_forecast_3h': 12.0,
                        'temp_forecast_4h': 15.0,
                        'pv_forecast_1h': 0.0,
                        'pv_forecast_2h': 0.0,
                        'pv_forecast_3h': 0.0,
                        'pv_forecast_4h': 0.0}
        }
    ]
    
    target_prediction = 0.053
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 TEST {i}: {test_case['name']}")
        
        # Simulate PV history if needed
        if "historical PV" in test_case['name']:
            # Simulate having had PV during the day
            model.pv_history = [1000, 800, 400, 100, 0]  # Declining PV from day
            # Set some learned multi-lag coefficients
            model.pv_coeffs = {
                'lag_1': 0.001,   # 30 min ago
                'lag_2': 0.002,   # 60 min ago  
                'lag_3': 0.001,   # 90 min ago
                'lag_4': 0.0005,  # 120 min ago
            }
        else:
            model.pv_history = [0, 0, 0, 0, 0]  # No PV history
        
        prediction = model.predict_one(test_case['features'])
        difference = abs(prediction - target_prediction)
        
        print(f"   Prediction: {prediction:.6f}°C")
        print(f"   Target:     {target_prediction:.6f}°C")
        print(f"   Difference: {difference:.6f}°C")
        print(f"   Match? {'✅' if difference < 0.01 else '❌'}")
        
        if difference < 0.01:
            print(f"   🎯 FOUND MATCH! This reproduces the log prediction!")
            
        if prediction > 0.04:
            print(f"   🚨 PROBLEM: This shows significant heating from cold outlet!")

def check_model_state_issues():
    """Check if there are model state issues causing discrepancies"""
    
    print("\n🔧 CHECKING MODEL STATE ISSUES")
    print("=" * 50)
    
    model, mae, rmse = load_model()
    
    # Check for any unusual state
    print(f"📊 MODEL STATE ANALYSIS:")
    print(f"   Training count: {getattr(model, 'training_count', 'N/A')}")
    print(f"   Prediction errors count: {len(getattr(model, 'prediction_errors', []))}")
    
    if hasattr(model, 'pv_history'):
        print(f"   PV history length: {len(model.pv_history)}")
        print(f"   PV history: {model.pv_history}")
    
    if hasattr(model, 'fireplace_history'):
        print(f"   Fireplace history: {model.fireplace_history}")
    
    if hasattr(model, 'tv_history'):
        print(f"   TV history: {model.tv_history}")
    
    # Check for any non-zero multi-lag coefficients
    suspicious_coeffs = {}
    
    if hasattr(model, 'pv_coeffs'):
        for lag, coeff in model.pv_coeffs.items():
            if abs(coeff) > 0.001:
                suspicious_coeffs[f'pv_{lag}'] = coeff
    
    if hasattr(model, 'fireplace_coeffs'):
        for lag, coeff in model.fireplace_coeffs.items():
            if abs(coeff) > 0.001:
                suspicious_coeffs[f'fireplace_{lag}'] = coeff
    
    if hasattr(model, 'tv_coeffs'):
        for lag, coeff in model.tv_coeffs.items():
            if abs(coeff) > 0.001:
                suspicious_coeffs[f'tv_{lag}'] = coeff
    
    if suspicious_coeffs:
        print(f"\n🚨 SUSPICIOUS COEFFICIENTS:")
        for name, coeff in suspicious_coeffs.items():
            print(f"   {name}: {coeff:.6f}")
    else:
        print(f"\n✅ No suspicious multi-lag coefficients found")

def investigate_different_models():
    """Check if there might be different model versions in use"""
    
    print("\n🔍 INVESTIGATING DIFFERENT MODEL VERSIONS")
    print("=" * 55)
    
    # Try to load model multiple times to see if it changes
    for i in range(3):
        try:
            model, mae, rmse = load_model()
            print(f"\nLoad {i+1}:")
            print(f"   base_heating_rate: {model.base_heating_rate:.6f}")
            print(f"   target_influence: {model.target_influence:.6f}")
            print(f"   Training count: {getattr(model, 'training_count', 'N/A')}")
            
            # Test prediction consistency
            test_features = {
                'outlet_temp': 14.0,
                'indoor_temp_lag_30m': 20.5,
                'target_temp': 21.0,
                'outdoor_temp': 5.0,
                'dhw_heating': 0.0,
                'pv_now': 0.0,
                'fireplace_on': 0.0,
                'tv_on': 0.0,
                'month_cos': 1.0,
                'month_sin': 0.0,
            }
            
            prediction = model.predict_one(test_features)
            print(f"   Test prediction: {prediction:.6f}°C")
            
        except Exception as e:
            print(f"   Error on load {i+1}: {e}")

def main():
    print("🚨 INVESTIGATING LOG DISCREPANCY")
    print("Production model predicts -0.007°C but logs show +0.053°C")
    print("=" * 70)
    
    test_various_feature_combinations()
    check_model_state_issues()
    investigate_different_models()
    
    print("\n🎯 CONCLUSIONS:")
    print("1. If no test case reproduces +0.053°C:")
    print("   - Different model file may be in use during production")
    print("   - Model state may change between predictions")
    print("   - Input features may be different than expected")
    print("\n2. If historical PV case shows high prediction:")
    print("   - Multi-lag PV coefficients are the culprit")
    print("   - Need to reset/limit these coefficients")
    print("\n3. If TV/fireplace cases show high prediction:")
    print("   - External heat source coefficients too large")
    print("   - Need to reduce these effects")

if __name__ == "__main__":
    main()
