import sys
import os
import logging
from unittest.mock import MagicMock
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from model_wrapper import EnhancedModelWrapper
from thermal_equilibrium_model import ThermalEquilibriumModel
from thermal_config import ThermalParameterConfig as config

# Configure logging
logging.basicConfig(level=logging.INFO)

def compare_parameters():
    """
    Compare outlet temp calculation with Calibrated vs Default parameters.
    """
    print("--- Comparing Calibrated vs Default Parameters ---")
    
    # 1. Setup Inputs (approximate from user description)
    # "predicted temperature dropped suddenly from ~43 to ~34"
    # Let's assume conditions that produced ~43°C with calibrated params.
    
    current_indoor = 20.0
    target_indoor = 21.0
    outdoor_temp = 5.0
    thermal_features = {"pv_power": 0.0, "fireplace_on": 0, "tv_on": 0}
    
    # 2. Calibrated Parameters (from unified_thermal_state.json)
    calibrated_params = {
        "thermal_time_constant": 20.90083045985179,
        "equilibrium_ratio": 0.17,
        "total_conductance": 0.8,
        "heat_loss_coefficient": 0.5399558410933942,
        "outlet_effectiveness": 0.48249043849150236,
        "pv_heat_weight": 0.0005,
        "fireplace_heat_weight": 5.0,
        "tv_heat_weight": 0.19129427501188587
    }
    
    # 3. Default Parameters (from ThermalParameterConfig)
    default_params = config.get_all_defaults()
    
    print(f"Inputs: Indoor={current_indoor}, Target={target_indoor}, Outdoor={outdoor_temp}")
    
    # 4. Run Calculation with Calibrated
    wrapper_cal = EnhancedModelWrapper()
    # Inject params
    for k, v in calibrated_params.items():
        setattr(wrapper_cal.thermal_model, k, v)
        
    # Mock forecast to be simple
    wrapper_cal._get_forecast_conditions = MagicMock(return_value=(
        outdoor_temp, 0.0, [outdoor_temp]*5, [0.0]*5
    ))
    
    res_cal = wrapper_cal._calculate_required_outlet_temp(
        current_indoor, target_indoor, outdoor_temp, thermal_features
    )
    print(f"Calibrated Result: {res_cal:.2f}°C")
    
    # 5. Run Calculation with Defaults
    wrapper_def = EnhancedModelWrapper()
    # Inject params
    for k, v in default_params.items():
        if hasattr(wrapper_def.thermal_model, k):
            setattr(wrapper_def.thermal_model, k, v)
            
    # Mock forecast
    wrapper_def._get_forecast_conditions = MagicMock(return_value=(
        outdoor_temp, 0.0, [outdoor_temp]*5, [0.0]*5
    ))
    
    res_def = wrapper_def._calculate_required_outlet_temp(
        current_indoor, target_indoor, outdoor_temp, thermal_features
    )
    print(f"Default Result: {res_def:.2f}°C")
    
    diff = res_cal - res_def
    print(f"Difference: {diff:.2f}°C")
    
    if abs(diff) > 10.0:
        print("SIGNIFICANT DIFFERENCE: Reverting to defaults explains the drop!")
    else:
        print("Difference is small. Parameter reversion might not be the sole cause.")

if __name__ == "__main__":
    compare_parameters()
