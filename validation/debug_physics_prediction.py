#!/usr/bin/env python3
"""
Debug Physics Prediction - Root Cause Analysis

This script traces exactly how the model predicts +0.053°C heating 
from 14°C outlet air into a 20.5°C room, which is physically impossible.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from physics_model import RealisticPhysicsModel

def create_test_features():
    """Create the exact scenario from the logs at 23:20"""
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
        
        # External sources (night time - should be ZERO/minimal)
        'pv_now': 0.0,  # NO PV at 23:20!
        'fireplace_on': 0.0,  # Probably no fireplace at 23:20
        'tv_on': 0.0,  # Minimal TV heat
        
        # Seasonal context (December)
        'month_cos': 0.0,  # December ≈ cos(2π*12/12) = cos(2π) ≈ 1.0
        'month_sin': -1.0,  # December ≈ sin(2π*12/12) = sin(2π) ≈ 0.0
        
        # Weather forecasts (not critical for this analysis)
        'temp_forecast_1h': 4.5,
        'temp_forecast_2h': 4.0,
        'temp_forecast_3h': 3.5,
        'temp_forecast_4h': 3.0,
        
        # PV forecasts (should be ZERO at night)
        'pv_forecast_1h': 0.0,
        'pv_forecast_2h': 0.0,
        'pv_forecast_3h': 0.0,
        'pv_forecast_4h': 0.0,
    }

def trace_prediction_components(model, features):
    """Trace each component of the physics prediction"""
    
    print("🔍 TRACING PHYSICS PREDICTION COMPONENTS")
    print("=" * 60)
    
    # Extract key values
    outlet_temp = features['outlet_temp']
    indoor_temp = features['indoor_temp_lag_30m']
    target_temp = features['target_temp']
    outdoor_temp = features['outdoor_temp']
    
    print(f"📊 INPUT CONDITIONS:")
    print(f"   Outlet Temperature: {outlet_temp}°C")
    print(f"   Indoor Temperature: {indoor_temp}°C")
    print(f"   Target Temperature: {target_temp}°C")
    print(f"   Outdoor Temperature: {outdoor_temp}°C")
    print(f"   Time: 23:20 (NIGHT - no PV expected)")
    print()
    
    # 1. Basic physics calculations
    temp_gap = target_temp - indoor_temp
    outlet_effect = outlet_temp - indoor_temp
    outdoor_penalty = max(0, 10 - outdoor_temp) / 15
    
    base_heating = outlet_effect * model.base_heating_rate
    target_boost = temp_gap * model.target_influence
    weather_adjustment = base_heating * outdoor_penalty * model.outdoor_factor
    
    print(f"🧮 BASIC PHYSICS CALCULATIONS:")
    print(f"   Temperature gap (target - indoor): {temp_gap:.3f}°C")
    print(f"   Outlet effect (outlet - indoor): {outlet_effect:.3f}°C (NEGATIVE = COOLING)")
    print(f"   Outdoor penalty factor: {outdoor_penalty:.3f}")
    print()
    print(f"   Base heating (outlet effect × {model.base_heating_rate:.6f}): {base_heating:.6f}°C")
    print(f"   Target boost (gap × {model.target_influence:.6f}): {target_boost:.6f}°C")
    print(f"   Weather adjustment: {weather_adjustment:.6f}°C")
    
    basic_total = base_heating + target_boost + weather_adjustment
    print(f"   🔸 BASIC PHYSICS TOTAL: {basic_total:.6f}°C")
    print()
    
    # 2. External heat sources
    pv_now = features['pv_now']
    fireplace_on = features['fireplace_on']
    tv_on = features['tv_on']
    month_cos = features['month_cos']
    month_sin = features['month_sin']
    
    print(f"🏠 EXTERNAL HEAT SOURCES:")
    print(f"   PV Power Now: {pv_now}W")
    print(f"   Fireplace On: {fireplace_on}")
    print(f"   TV On: {tv_on}")
    print(f"   Month cos/sin: {month_cos:.3f}, {month_sin:.3f}")
    print()
    
    # Update histories for multi-lag (this modifies the model state)
    model._update_histories(pv_now, fireplace_on, tv_on)
    
    # Calculate multi-lag contributions
    print(f"🔄 MULTI-LAG CALCULATIONS:")
    
    # PV multi-lag
    if hasattr(model, 'pv_history') and len(model.pv_history) >= 5:
        print(f"   PV History (last 5): {model.pv_history}")
        print(f"   PV Coefficients: {model.pv_coeffs}")
        
        pv_seasonal = 1.0 + (model.pv_seasonal_cos * month_cos + model.pv_seasonal_sin * month_sin)
        pv_seasonal = max(0.5, min(1.5, pv_seasonal))
        print(f"   PV Seasonal factor: {pv_seasonal:.3f}")
        
        pv_effect = 0.0
        pv_effect += (model.pv_history[-2] * 0.01 * model.pv_coeffs['lag_1'])
        pv_effect += (model.pv_history[-3] * 0.01 * model.pv_coeffs['lag_2'])
        pv_effect += (model.pv_history[-4] * 0.01 * model.pv_coeffs['lag_3'])
        pv_effect += (model.pv_history[-5] * 0.01 * model.pv_coeffs['lag_4'])
        pv_contribution = pv_effect * pv_seasonal
        print(f"   🔸 PV Multi-lag Contribution: {pv_contribution:.6f}°C")
    else:
        # Fallback to simple PV
        pv_contribution = pv_now * model.pv_warming_coefficient * 0.01
        print(f"   🔸 PV Simple Contribution: {pv_contribution:.6f}°C")
    
    # Fireplace multi-lag  
    if hasattr(model, 'fireplace_history') and len(model.fireplace_history) >= 4:
        print(f"   Fireplace History: {model.fireplace_history}")
        print(f"   Fireplace Coefficients: {model.fireplace_coeffs}")
        
        fireplace_effect = 0.0
        fireplace_effect += (model.fireplace_history[-1] * model.fireplace_coeffs['immediate'])
        fireplace_effect += (model.fireplace_history[-2] * model.fireplace_coeffs['lag_1'])
        fireplace_effect += (model.fireplace_history[-3] * model.fireplace_coeffs['lag_2'])
        fireplace_effect += (model.fireplace_history[-4] * model.fireplace_coeffs['lag_3'])
        fireplace_contribution = fireplace_effect
        print(f"   🔸 Fireplace Multi-lag Contribution: {fireplace_contribution:.6f}°C")
    else:
        fireplace_contribution = fireplace_on * model.fireplace_heating_rate
        print(f"   🔸 Fireplace Simple Contribution: {fireplace_contribution:.6f}°C")
    
    # TV multi-lag
    if hasattr(model, 'tv_history') and len(model.tv_history) >= 2:
        print(f"   TV History: {model.tv_history}")
        print(f"   TV Coefficients: {model.tv_coeffs}")
        
        tv_seasonal = 1.0 + (model.tv_seasonal_cos * month_cos + model.tv_seasonal_sin * month_sin)
        tv_seasonal = max(0.7, min(1.3, tv_seasonal))
        
        tv_effect = 0.0
        tv_effect += model.tv_history[-1] * model.tv_coeffs['immediate']
        tv_effect += model.tv_history[-2] * model.tv_coeffs['lag_1']
        tv_contribution = tv_effect * tv_seasonal
        print(f"   🔸 TV Multi-lag Contribution: {tv_contribution:.6f}°C")
    else:
        tv_contribution = tv_on * model.tv_heat_contribution
        print(f"   🔸 TV Simple Contribution: {tv_contribution:.6f}°C")
    
    print()
    
    # 3. Forecast adjustments
    print(f"🌤️ FORECAST ADJUSTMENTS:")
    current_outdoor = features['outdoor_temp']
    current_pv = features['pv_now']
    
    weather_adjustment_forecast = 0.0
    for i in range(4):
        forecast_temp = features.get(f'temp_forecast_{i+1}h', current_outdoor)
        temp_change = forecast_temp - current_outdoor
        decay = model.forecast_decay[i]
        
        if temp_change > 1.5:  # Significant warming
            contrib = -temp_change * model.weather_forecast_coeff * decay * 0.1
            weather_adjustment_forecast += contrib
            print(f"   Hour {i+1}: {forecast_temp}°C (Δ{temp_change:+.1f}°C) → {contrib:.6f}°C")
        elif temp_change < -1.5:  # Significant cooling  
            contrib = -temp_change * model.weather_forecast_coeff * decay * 0.06
            weather_adjustment_forecast += contrib
            print(f"   Hour {i+1}: {forecast_temp}°C (Δ{temp_change:+.1f}°C) → {contrib:.6f}°C")
    
    pv_adjustment_forecast = 0.0
    for i in range(4):
        forecast_pv = features.get(f'pv_forecast_{i+1}h', 0.0)
        pv_increase = max(0, forecast_pv - current_pv)
        decay = model.forecast_decay[i]
        
        if pv_increase > 200:  # Significant solar expected
            contrib = -pv_increase * model.pv_forecast_coeff * decay * 0.001
            pv_adjustment_forecast += contrib
            print(f"   PV Hour {i+1}: {forecast_pv}W (+{pv_increase}W) → {contrib:.6f}°C")
    
    forecast_effect = weather_adjustment_forecast + pv_adjustment_forecast
    print(f"   🔸 TOTAL FORECAST EFFECT: {forecast_effect:.6f}°C")
    print()
    
    # 4. Final calculation
    total_external = pv_contribution + fireplace_contribution + tv_contribution
    total_all = basic_total + total_external + forecast_effect
    
    print(f"📊 FINAL BREAKDOWN:")
    print(f"   Basic Physics: {basic_total:.6f}°C")
    print(f"   PV Contribution: {pv_contribution:.6f}°C")
    print(f"   Fireplace Contribution: {fireplace_contribution:.6f}°C") 
    print(f"   TV Contribution: {tv_contribution:.6f}°C")
    print(f"   Forecast Effect: {forecast_effect:.6f}°C")
    print(f"   ─" * 40)
    print(f"   🔸 TOTAL PREDICTION: {total_all:.6f}°C")
    print()
    
    # 5. Physics bounds
    bounded_result = np.clip(total_all, model.min_prediction, model.max_prediction)
    print(f"🚧 PHYSICS BOUNDS:")
    print(f"   Min/Max allowed: [{model.min_prediction:.3f}, {model.max_prediction:.3f}]°C")
    print(f"   🔸 FINAL BOUNDED RESULT: {bounded_result:.6f}°C")
    print()
    
    return bounded_result, {
        'basic_physics': basic_total,
        'pv_contribution': pv_contribution,
        'fireplace_contribution': fireplace_contribution,
        'tv_contribution': tv_contribution,
        'forecast_effect': forecast_effect,
        'total_external': total_external,
        'total_prediction': total_all,
        'final_result': bounded_result
    }

def main():
    print("🚨 PHYSICS VIOLATION DEBUG ANALYSIS")
    print("Investigating why 14°C outlet air predicts +0.053°C heating")
    print("=" * 70)
    print()
    
    # Create model with realistic parameters (from production)
    model = RealisticPhysicsModel()
    
    # Set realistic learned parameters (approximate from logs)
    model.base_heating_rate = 0.001337
    model.target_influence = 0.009431
    model.outdoor_factor = 0.003
    
    # Initialize some external source effects (will be updated based on findings)
    model.pv_warming_coefficient = 0.001
    model.fireplace_heating_rate = 0.01
    model.tv_heat_contribution = 0.005
    
    # Initialize multi-lag buffers with some test data
    # Simulate daytime PV history that shouldn't affect night predictions
    model.pv_history = [0, 0, 0, 0, 0]  # All zeros for night time
    model.fireplace_history = [0, 0, 0, 0]  # No fireplace
    model.tv_history = [0, 0]  # No TV heat
    
    # Set potentially problematic multi-lag coefficients
    model.pv_coeffs = {
        'lag_1': 0.001,   # 30 min ago
        'lag_2': 0.002,   # 60 min ago  
        'lag_3': 0.001,   # 90 min ago
        'lag_4': 0.0005,  # 120 min ago
    }
    
    model.fireplace_coeffs = {
        'immediate': 0.01,
        'lag_1': 0.008,
        'lag_2': 0.005,
        'lag_3': 0.002,
    }
    
    model.tv_coeffs = {
        'immediate': 0.003,
        'lag_1': 0.002,
    }
    
    # Create test scenario
    features = create_test_features()
    
    # Trace the prediction
    result, breakdown = trace_prediction_components(model, features)
    
    print("🎯 ANALYSIS RESULTS:")
    print("=" * 50)
    
    if breakdown['pv_contribution'] > 0.001:
        print(f"❌ PROBLEM: PV adding {breakdown['pv_contribution']:.6f}°C at NIGHT!")
        print(f"   This is physically impossible - no solar gain at 23:20")
        
    if breakdown['total_external'] > 0.01:
        print(f"❌ PROBLEM: External sources adding {breakdown['total_external']:.6f}°C")
        print(f"   This overwhelms basic outlet physics ({breakdown['basic_physics']:.6f}°C)")
        
    if result > 0 and breakdown['basic_physics'] < 0:
        print(f"❌ CRITICAL: Net heating (+{result:.6f}°C) from cooling outlet!")
        print(f"   14°C air into 20.5°C room should cool, not heat!")
        
    print()
    print("🔧 RECOMMENDED FIXES:")
    if breakdown['pv_contribution'] > 0.001:
        print("   1. Add time-of-day validation - zero PV effects at night")
        print("   2. Reduce PV multi-lag coefficients - effects too large")
        
    if breakdown['total_external'] > abs(breakdown['basic_physics']) * 2:
        print("   3. Limit external source magnitude - can't override outlet physics")
        print("   4. Physics-first controller - respect outlet temperature reality")
        
    print()
    print(f"🎯 CONTROLLER IMPACT:")
    print(f"   Current logic: Choose 14°C because total effect is +{result:.3f}°C")
    print(f"   Should choose: Higher outlet temp because 14°C should COOL by {breakdown['basic_physics']:.3f}°C")
    
if __name__ == "__main__":
    main()
