
import logging
import sys
import os
from datetime import datetime

# Add root to path to allow package imports
sys.path.append(os.getcwd())

from src.thermal_equilibrium_model import ThermalEquilibriumModel
from src.prediction_context import UnifiedPredictionContext

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_pv_sensitivity():
    print("--- Testing PV Sensitivity ---")
    model = ThermalEquilibriumModel()
    
    # Force parameters to what we saw in logs
    model.heat_loss_coefficient = 0.52
    model.outlet_effectiveness = 0.48
    model.thermal_time_constant = 20.9
    
    # Test with Default PV Weight
    model.pv_heat_weight = 0.002
    print(f"\nScenario 1: Default PV Weight ({model.pv_heat_weight})")
    
    outdoor = 10.0
    outlet = 35.0
    indoor = 20.0
    
    # No PV
    pred_no_pv = model.predict_equilibrium_temperature(
        outlet_temp=outlet,
        outdoor_temp=outdoor,
        current_indoor=indoor,
        pv_power=0
    )
    print(f"PV=0W: Eq Temp = {pred_no_pv:.2f}°C")
    
    # High PV
    pred_high_pv = model.predict_equilibrium_temperature(
        outlet_temp=outlet,
        outdoor_temp=outdoor,
        current_indoor=indoor,
        pv_power=2000
    )
    print(f"PV=2000W: Eq Temp = {pred_high_pv:.2f}°C")
    print(f"Delta: {pred_high_pv - pred_no_pv:.2f}°C")
    
    # Test with Max PV Weight (from logs)
    model.pv_heat_weight = 0.005
    print(f"\nScenario 2: Max PV Weight ({model.pv_heat_weight})")
    
    # No PV
    pred_no_pv = model.predict_equilibrium_temperature(
        outlet_temp=outlet,
        outdoor_temp=outdoor,
        current_indoor=indoor,
        pv_power=0
    )
    print(f"PV=0W: Eq Temp = {pred_no_pv:.2f}°C")
    
    # High PV
    pred_high_pv = model.predict_equilibrium_temperature(
        outlet_temp=outlet,
        outdoor_temp=outdoor,
        current_indoor=indoor,
        pv_power=2000
    )
    print(f"PV=2000W: Eq Temp = {pred_high_pv:.2f}°C")
    print(f"Delta: {pred_high_pv - pred_no_pv:.2f}°C")

if __name__ == "__main__":
    test_pv_sensitivity()
