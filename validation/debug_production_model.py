#!/usr/bin/env python3
"""
Debug Production Model - Load actual model and trace prediction

This script loads the actual production model with learned parameters
and traces why it predicts +0.053°C heating from 14°C outlet air.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from model_wrapper import load_model


def create_exact_log_features():
    """Create the exact scenario from the production logs"""
    return {
        # Core temperatures (from logs)
        'outlet_temp': 14.0,
        'indoor_temp_lag_30m': 20.5,
        'target_temp': 21.0,
        'outdoor_temp': 5.0,
        
        # System states (night time)
        'dhw_heating': 0.0,
        'dhw_disinfection': 0.0,
        'dhw_boost_heater': 0.0,
        'defrosting': 0.0,
        
        # External sources (from your system at 23:20)
        'pv_now': 0.0,  # NO PV at night
        'fireplace_on': 0.0,  # Likely no fireplace
        'tv_on': 0.0,  # TV might be on but heat should be minimal
        
        # Seasonal context (December)
        'month_cos': 1.0,   # December ≈ cos(2π*11/12) ≈ 0.866
        'month_sin': 0.0,   # December ≈ sin(2π*11/12) ≈ 0.5
        
        # Weather forecasts (approximate from typical conditions)
        'temp_forecast_1h': 4.5,
        'temp_forecast_2h': 4.0,
        'temp_forecast_3h': 3.5,
        'temp_forecast_4h': 3.0,
        
        # PV forecasts (zero at night)
        'pv_forecast_1h': 0.0,
        'pv_forecast_2h': 0.0,
        'pv_forecast_3h': 0.0,
        'pv_forecast_4h': 0.0,
    }


def trace_production_prediction():
    """Load production model and trace the actual prediction"""
    
    print("🚨 PRODUCTION MODEL PHYSICS VIOLATION ANALYSIS")
    print("Loading actual production model with learned parameters")
    print("=" * 70)
    print()
    
    try:
        # Load the actual production model
        model, mae, rmse = load_model()
        
        print(f"✅ Loaded production model successfully")
        print(f"   MAE: {mae.get():.6f}°C")
        print(f"   RMSE: {rmse.get():.6f}°C")
        print()
        
        # Print actual learned parameters
        print("📊 ACTUAL LEARNED PARAMETERS:")
        print(f"   base_heating_rate: {model.base_heating_rate:.6f}")
        print(f"   target_influence: {model.target_influence:.6f}")
        print(f"   outdoor_factor: {model.outdoor_factor:.6f}")
        print()
        
        print("🔋 EXTERNAL SOURCE COEFFICIENTS:")
        print(f"   pv_warming_coefficient: {model.pv_warming_coefficient:.6f}")
        print(f"   fireplace_heating_rate: {model.fireplace_heating_rate:.6f}")
        print(f"   tv_heat_contribution: {model.tv_heat_contribution:.6f}")
        print()
        
        if hasattr(model, 'pv_coeffs'):
            print("⚡ PV MULTI-LAG COEFFICIENTS:")
            for lag, coeff in model.pv_coeffs.items():
                print(f"   {lag}: {coeff:.6f}")
            print()
        
        if hasattr(model, 'fireplace_coeffs'):
            print("🔥 FIREPLACE MULTI-LAG COEFFICIENTS:")
            for lag, coeff in model.fireplace_coeffs.items():
                print(f"   {lag}: {coeff:.6f}")
            print()
        
        if hasattr(model, 'tv_coeffs'):
            print("📺 TV MULTI-LAG COEFFICIENTS:")
            for lag, coeff in model.tv_coeffs.items():
                print(f"   {lag}: {coeff:.6f}")
            print()
        
        # Create the exact test case
        features = create_exact_log_features()
        
        # Make prediction with production model
        print("🔍 PRODUCTION MODEL PREDICTION:")
        prediction = model.predict_one(features)
        
        print(f"   🔸 ACTUAL PREDICTION: {prediction:.6f}°C")
        print(f"   📝 Expected from logs: +0.053°C")
        print(f"   🎯 Match? {'✅' if abs(prediction - 0.053) < 0.01 else '❌'}")
        print()
        
        # Now let's manually trace the components like in the logs
        print("🧮 MANUAL COMPONENT BREAKDOWN:")
        
        # Basic physics
        temp_gap = features['target_temp'] - features['indoor_temp_lag_30m']
        outlet_effect = features['outlet_temp'] - features['indoor_temp_lag_30m']
        outdoor_penalty = max(0, 10 - features['outdoor_temp']) / 15
        
        base_heating = outlet_effect * model.base_heating_rate
        target_boost = temp_gap * model.target_influence
        weather_adjustment = base_heating * outdoor_penalty * model.outdoor_factor
        
        print(f"   Basic heating: {base_heating:.6f}°C")
        print(f"   Target boost: {target_boost:.6f}°C") 
        print(f"   Weather adjustment: {weather_adjustment:.6f}°C")
        basic_total = base_heating + target_boost + weather_adjustment
        print(f"   Basic total: {basic_total:.6f}°C")
        print()
        
        # Check PV history and effects
        if hasattr(model, 'pv_history'):
            print(f"🔋 PV ANALYSIS:")
            print(f"   PV History: {model.pv_history}")
            print(f"   Current PV: {features['pv_now']}W")
            
            # Calculate PV contribution manually
            if len(model.pv_history) >= 5:
                month_cos = features['month_cos']
                month_sin = features['month_sin']
                
                pv_seasonal = 1.0 + (
                    model.pv_seasonal_cos * month_cos +
                    model.pv_seasonal_sin * month_sin
                )
                pv_seasonal = max(0.5, min(1.5, pv_seasonal))
                
                pv_effect = 0.0
                pv_effect += (model.pv_history[-2] * 0.01 * model.pv_coeffs['lag_1'])
                pv_effect += (model.pv_history[-3] * 0.01 * model.pv_coeffs['lag_2'])
                if len(model.pv_history) >= 4:
                    pv_effect += (model.pv_history[-4] * 0.01 * model.pv_coeffs['lag_3'])
                if len(model.pv_history) >= 5:
                    pv_effect += (model.pv_history[-5] * 0.01 * model.pv_coeffs['lag_4'])
                
                pv_contribution = pv_effect * pv_seasonal
                print(f"   PV seasonal factor: {pv_seasonal:.3f}")
                print(f"   PV multi-lag effect: {pv_contribution:.6f}°C")
                print()
            
        # The key insight: what's the difference?
        missing_contribution = prediction - basic_total
        print(f"❗ MISSING CONTRIBUTION: {missing_contribution:.6f}°C")
        print(f"   This extra heating is coming from somewhere!")
        print()
        
        # Check if it's forecast effects
        forecast_effect = model._calculate_forecast_adjustment(features)
        print(f"🌤️ Forecast adjustment: {forecast_effect:.6f}°C")
        
        remaining = missing_contribution - forecast_effect
        print(f"🔍 Still unaccounted: {remaining:.6f}°C")
        print()
        
        # Export all learned metrics to see what's unusual
        metrics = model.export_learning_metrics()
        
        print("🎯 SUSPICIOUS LEARNED VALUES:")
        for key, value in metrics.items():
            if abs(value) > 0.01:  # Focus on significant values
                print(f"   {key}: {value:.6f}")
        
        print()
        print("🚨 DIAGNOSIS:")
        if prediction > 0.04:
            print(f"   ❌ Production model predicts HEATING from 14°C outlet!")
            print(f"   ❌ This is physically impossible!")
            print(f"   🔧 Need to fix external source coefficients")
        else:
            print(f"   ✅ Model correctly predicts cooling from 14°C outlet")
            print(f"   🤔 Discrepancy with logs - check feature inputs")
            
    except Exception as e:
        print(f"❌ Error loading production model: {e}")
        print("   Make sure model.pkl exists and is accessible")


if __name__ == "__main__":
    trace_production_prediction()
