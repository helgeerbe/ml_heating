#!/usr/bin/env python3
"""
Fix Calibration System - Replace weak online learning with proper scipy optimization
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None
    print("❌ scipy not available - cannot run optimization")
    exit(1)

# Import necessary modules
import src.config as config
from src.thermal_equilibrium_model import ThermalEquilibriumModel
from src.influx_service import InfluxService
from src.unified_thermal_state import get_thermal_state_manager

def run_proper_calibration():
    """Run REAL calibration with scipy optimization"""
    
    print("🔧 FIXING CALIBRATION SYSTEM")
    print("=" * 50)
    
    # Step 1: Fetch historical data
    print("Step 1: Fetching historical data...")
    influx = InfluxService(
        url=config.INFLUX_URL,
        token=config.INFLUX_TOKEN,
        org=config.INFLUX_ORG
    )
    
    df = influx.get_training_data(lookback_hours=config.TRAINING_LOOKBACK_HOURS)
    
    if df.empty or len(df) < 1000:
        print("❌ Insufficient historical data")
        return False
    
    print(f"✅ Retrieved {len(df)} samples")
    
    # Step 2: Filter for stable periods
    print("Step 2: Filtering stable periods...")
    stable_periods = filter_stable_periods_simple(df)
    
    if len(stable_periods) < 100:
        print(f"❌ Insufficient stable periods: {len(stable_periods)}")
        return False
    
    print(f"✅ Found {len(stable_periods)} stable periods")
    
    # Step 3: Run scipy optimization
    print("Step 3: Running scipy L-BFGS-B optimization...")
    optimized_params = run_scipy_optimization(stable_periods)
    
    if optimized_params is None:
        print("❌ Optimization failed")
        return False
    
    print("✅ Optimization completed successfully!")
    
    # Step 4: Save results
    print("Step 4: Saving calibrated parameters...")
    save_calibrated_params(optimized_params, len(stable_periods))
    
    print("\n🎉 CALIBRATION FIXED AND COMPLETED!")
    return True


def filter_stable_periods_simple(df, max_temp_change=0.2):
    """Simple but effective stable period filtering"""
    
    # Get column names
    indoor_col = config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    outlet_col = config.ACTUAL_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1]
    outdoor_col = config.OUTDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    pv_col = config.PV_POWER_ENTITY_ID.split(".", 1)[-1]
    fireplace_col = config.FIREPLACE_STATUS_ENTITY_ID.split(".", 1)[-1]
    tv_col = config.TV_STATUS_ENTITY_ID.split(".", 1)[-1]
    
    # Blocking state columns  
    dhw_col = config.DHW_STATUS_ENTITY_ID.split(".", 1)[-1]
    defrost_col = config.DEFROST_STATUS_ENTITY_ID.split(".", 1)[-1]
    
    stable_periods = []
    window_size = 12  # 1 hour window at 5min intervals
    
    for i in range(window_size, len(df) - window_size):
        # Get window around this point
        window = df.iloc[i-window_size//2:i+window_size//2]
        center_row = df.iloc[i]
        
        # Skip if missing critical data
        indoor_temps = window[indoor_col].dropna()
        if len(indoor_temps) < window_size * 0.8:
            continue
            
        # Check temperature stability
        temp_range = indoor_temps.max() - indoor_temps.min()
        if temp_range > max_temp_change:
            continue
            
        # Skip blocking states
        if dhw_col in window.columns and window[dhw_col].sum() > 0:
            continue
        if defrost_col in window.columns and window[defrost_col].sum() > 0:
            continue
            
        # Skip extreme outlet temperatures
        outlet_temp = center_row.get(outlet_col)
        if pd.isna(outlet_temp) or outlet_temp < 20 or outlet_temp > 65:
            continue
            
        # Extract period data
        period = {
            'indoor_temp': center_row[indoor_col],
            'outlet_temp': outlet_temp,
            'outdoor_temp': center_row[outdoor_col],
            'pv_power': center_row.get(pv_col, 0.0),
            'fireplace_on': center_row.get(fireplace_col, 0.0),
            'tv_on': center_row.get(tv_col, 0.0)
        }
        
        # Skip if any critical data is missing
        if any(pd.isna(v) for v in [period['indoor_temp'], period['outdoor_temp']]):
            continue
            
        stable_periods.append(period)
    
    return stable_periods


def run_scipy_optimization(stable_periods):
    """Run proper scipy optimization on stable periods"""
    
    # Current parameters from config as starting point
    initial_params = [
        config.THERMAL_TIME_CONSTANT,      # 4.0h
        config.HEAT_LOSS_COEFFICIENT,     # 0.25
        config.OUTLET_EFFECTIVENESS       # 0.55
    ]
    
    # Parameter bounds - realistic thermal physics constraints
    bounds = [
        (2.0, 12.0),    # thermal_time_constant: 2-12 hours
        (0.1, 0.4),     # heat_loss_coefficient: 0.1-0.4
        (0.3, 0.8)      # outlet_effectiveness: 30-80%
    ]
    
    print(f"Starting optimization from: {initial_params}")
    print(f"Parameter bounds: {bounds}")
    print(f"Optimizing on {len(stable_periods)} stable periods...")
    
    def objective_function(params):
        """Calculate MAE for given thermal parameters"""
        thermal_time_constant, heat_loss_coefficient, outlet_effectiveness = params
        
        total_error = 0.0
        valid_predictions = 0
        
        for period in stable_periods:
            try:
                # Create thermal model with test parameters
                test_model = ThermalEquilibriumModel()
                test_model.thermal_time_constant = thermal_time_constant
                test_model.heat_loss_coefficient = heat_loss_coefficient  
                test_model.outlet_effectiveness = outlet_effectiveness
                
                # Predict equilibrium temperature
                predicted_temp = test_model.predict_equilibrium_temperature(
                    outlet_temp=period['outlet_temp'],
                    outdoor_temp=period['outdoor_temp'],
                    pv_power=period['pv_power'],
                    fireplace_on=period['fireplace_on'],
                    tv_on=period['tv_on']
                )
                
                # Calculate prediction error
                actual_temp = period['indoor_temp']
                error = abs(predicted_temp - actual_temp)
                
                # Skip extreme errors (bad parameters)
                if error > 10.0:
                    continue
                    
                total_error += error
                valid_predictions += 1
                
            except Exception:
                continue
        
        if valid_predictions < 10:
            return 1000.0  # Penalize invalid parameter sets
            
        mae = total_error / valid_predictions
        return mae
    
    # Run scipy optimization
    try:
        result = minimize(
            objective_function,
            x0=initial_params,
            bounds=bounds,
            method='L-BFGS-B',
            options={
                'maxiter': 50,
                'ftol': 1e-6
            }
        )
        
        if result.success:
            optimized_params = {
                'thermal_time_constant': float(result.x[0]),
                'heat_loss_coefficient': float(result.x[1]),
                'outlet_effectiveness': float(result.x[2]),
                'mae': float(result.fun),
                'optimization_success': True
            }
            
            print("\n=== OPTIMIZATION RESULTS ===")
            print(f"thermal_time_constant: {initial_params[0]:.2f} → {result.x[0]:.2f}h")
            print(f"heat_loss_coefficient: {initial_params[1]:.3f} → {result.x[1]:.3f}")  
            print(f"outlet_effectiveness: {initial_params[2]:.3f} → {result.x[2]:.3f}")
            print(f"Final MAE: {result.fun:.3f}°C")
            print(f"Function evaluations: {result.nfev}")
            
            return optimized_params
            
        else:
            print(f"❌ Optimization failed: {result.message}")
            return None
            
    except Exception as e:
        print(f"❌ Optimization error: {e}")
        return None


def save_calibrated_params(optimized_params, num_periods):
    """Save optimized parameters to unified thermal state"""
    
    try:
        state_manager = get_thermal_state_manager()
        
        # Prepare calibrated parameters
        calibrated_params = {
            'thermal_time_constant': optimized_params['thermal_time_constant'],
            'heat_loss_coefficient': optimized_params['heat_loss_coefficient'],
            'outlet_effectiveness': optimized_params['outlet_effectiveness'],
            'pv_heat_weight': config.PV_HEAT_WEIGHT,
            'fireplace_heat_weight': config.FIREPLACE_HEAT_WEIGHT,
            'tv_heat_weight': config.TV_HEAT_WEIGHT
        }
        
        # Save as calibrated baseline
        state_manager.set_calibrated_baseline(calibrated_params, calibration_cycles=num_periods)
        
        print("✅ Calibrated parameters saved to thermal_state.json")
        print("✅ Parameters will be used immediately in operations")
        
        # Show final comparison
        print("\n=== FINAL PARAMETER COMPARISON ===")
        print(f"thermal_time_constant: {config.THERMAL_TIME_CONSTANT} → {optimized_params['thermal_time_constant']:.2f}h")
        print(f"heat_loss_coefficient: {config.HEAT_LOSS_COEFFICIENT} → {optimized_params['heat_loss_coefficient']:.3f}")
        print(f"outlet_effectiveness: {config.OUTLET_EFFECTIVENESS} → {optimized_params['outlet_effectiveness']:.3f}")
        print(f"MAE improvement: {optimized_params['mae']:.3f}°C")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to save parameters: {e}")
        return False


if __name__ == "__main__":
    success = run_proper_calibration()
    exit(0 if success else 1)
