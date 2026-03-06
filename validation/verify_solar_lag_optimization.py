
import logging
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
import sys
import os

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

def test_optimization():
    print("Creating synthetic data with True Lag = 90 min...")
    periods = create_synthetic_data(true_lag_minutes=90.0, true_pv_weight=0.005)
    
    # Mock config to ensure we optimize lag
    with patch('src.physics_calibration.ThermalParameterConfig') as mock_config:
        # Setup default behavior for config
        def get_default(param):
            defaults = {
                'thermal_time_constant': 4.0,
                'heat_loss_coefficient': 0.4,
                'outlet_effectiveness': 0.5,
                'pv_heat_weight': 0.002, # Start wrong
                'fireplace_heat_weight': 5.0,
                'tv_heat_weight': 0.2,
                'solar_lag_minutes': 30.0 # Start wrong (True is 90)
            }
            return defaults.get(param, 0.0)
            
        def get_bounds(param):
            bounds = {
                'thermal_time_constant': (1.0, 10.0),
                'heat_loss_coefficient': (0.1, 1.0),
                'outlet_effectiveness': (0.1, 1.0),
                'pv_heat_weight': (0.0005, 0.01),
                'fireplace_heat_weight': (1.0, 10.0),
                'tv_heat_weight': (0.1, 1.0),
                'solar_lag_minutes': (0.0, 180.0)
            }
            return bounds.get(param, (0.0, 1.0))

        mock_config.get_default.side_effect = get_default
        mock_config.get_bounds.side_effect = get_bounds
        
        # Run optimization
        print("Running optimization...")
        # We need to patch minimize to use the real one, but we are importing it inside physics_calibration
        # physics_calibration imports minimize from scipy.optimize
        # So we don't need to patch it if we want the real one.
        
        # However, we need to make sure data availability checks pass
        # The synthetic data has PV > 0, so it should pass.
        
        result = optimize_thermal_parameters(periods)
        
        if result:
            print("\nOptimization Result:")
            print(f"  Solar Lag: {result['solar_lag_minutes']:.2f} min (Target: 90.00)")
            print(f"  PV Weight: {result['pv_heat_weight']:.5f} (Target: 0.00500)")
            print(f"  MAE: {result['mae']:.4f}")
        else:
            print("Optimization failed.")

if __name__ == "__main__":
    test_optimization()
