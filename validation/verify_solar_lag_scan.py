
import logging
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
import sys
import os
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from physics_calibration import optimize_thermal_parameters, build_optimization_params, calculate_mae_for_params
from thermal_equilibrium_model import ThermalEquilibriumModel
from thermal_config import ThermalParameterConfig

# Configure logging
logging.basicConfig(level=logging.INFO)

def create_synthetic_data(true_lag_minutes=60.0, true_pv_weight=0.002):
    """
    Create synthetic stable periods where the indoor temperature 
    is clearly driven by lagged solar.
    """
    periods = []
    
    # Create a "day" of data
    # PV peaks at noon (index 36 of 72 10-min intervals)
    # Indoor temp should peak at noon + lag
    
    steps = 100
    for i in range(steps):
        # Time in minutes
        t = i * 10
        
        # PV curve (Gaussian-ish)
        # Peak at t=300 (5 hours in)
        pv = 1000 * np.exp(-((t - 300)**2) / (2 * 60**2))
        if pv < 10: pv = 0
        
        # Create history for this point
        # History needs to go back ~3 hours (18 steps)
        history = []
        for h in range(19): # 0 to 18
            t_hist = t - h * 10
            pv_hist = 1000 * np.exp(-((t_hist - 300)**2) / (2 * 60**2))
            if pv_hist < 10: pv_hist = 0
            history.insert(0, pv_hist) # Oldest first
            
        # Calculate "True" effective solar with lag
        # Simple rolling average approximation for ground truth generation
        lag_steps = int(true_lag_minutes / 10)
        if lag_steps > 0:
            effective_pv = np.mean(history[-lag_steps:])
        else:
            effective_pv = history[-1]
            
        # Indoor temp response
        # Baseline 20C + solar effect
        # Ignore other factors for simplicity (outlet=indoor, outdoor=indoor)
        indoor = 20.0 + effective_pv * true_pv_weight
        
        period = {
            'indoor_temp': indoor,
            'outlet_temp': 20.0, # No heating
            'outdoor_temp': 20.0, # No loss
            'pv_power': pv,
            'pv_power_history': history,
            'fireplace_on': 0,
            'tv_on': 0,
            'thermal_power_kw': 0,
            'timestamp': f"2023-01-01T{t//60:02d}:{t%60:02d}:00"
        }
        periods.append(period)
        
    return periods

def scan_lag_parameter():
    print("Scanning MAE vs Solar Lag...")
    true_lag = 90.0
    true_weight = 0.005
    periods = create_synthetic_data(true_lag_minutes=true_lag, true_pv_weight=true_weight)
    
    model = ThermalEquilibriumModel()
    # Fix other parameters to "true" values to isolate lag
    model.heat_loss_coefficient = 0.4 # Doesn't matter as deltaT=0
    model.outlet_effectiveness = 0.5
    model.external_source_weights['pv'] = true_weight
    
    lags = np.linspace(0, 180, 37) # 0, 5, 10, ... 180
    maes = []
    
    print(f"{'Lag (min)':<10} | {'MAE':<10}")
    print("-" * 25)
    
    for lag in lags:
        model.solar_lag_minutes = lag
        total_error = 0
        count = 0
        
        for p in periods:
            # Use PV history
            pv_input = p['pv_power_history']
            
            pred = model.predict_equilibrium_temperature(
                outlet_temp=p['outlet_temp'],
                outdoor_temp=p['outdoor_temp'],
                current_indoor=p['indoor_temp'], # Not used in equilibrium calc usually
                pv_power=pv_input,
                fireplace_on=0,
                tv_on=0,
                thermal_power=None,
                _suppress_logging=True
            )
            
            error = abs(pred - p['indoor_temp'])
            total_error += error
            count += 1
            
        mae = total_error / count
        maes.append(mae)
        print(f"{lag:<10.1f} | {mae:<10.4f}")
        
    best_lag = lags[np.argmin(maes)]
    print(f"\nBest Lag found by scan: {best_lag} min (True: {true_lag})")

if __name__ == "__main__":
    scan_lag_parameter()
