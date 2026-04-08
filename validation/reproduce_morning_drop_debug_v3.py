
import sys
import os
import logging
import time

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from src.prediction_context import UnifiedPredictionContext
from src.model_wrapper import EnhancedModelWrapper
from src import config

def reproduce_morning_drop_v3_debug():
    print("=== Reproducing Morning Drop Scenario V3 (Debug) ===")
    
    # 1. Setup Scenario
    current_outdoor = 5.0
    forecast_1h_outdoor = 10.0 
    
    current_pv = 0.0
    forecast_1h_pv = 2000.0 
    
    target_temp = 21.0
    current_indoor = 20.96 
    
    # Mock features
    features = {
        "temp_forecast_1h": forecast_1h_outdoor,
        "temp_forecast_2h": forecast_1h_outdoor + 2,
        "temp_forecast_3h": forecast_1h_outdoor + 3,
        "temp_forecast_4h": forecast_1h_outdoor + 4,
        "pv_forecast_1h": forecast_1h_pv,
        "pv_forecast_2h": forecast_1h_pv,
        "pv_forecast_3h": forecast_1h_pv,
        "pv_forecast_4h": forecast_1h_pv,
        "indoor_temp_lag_30m": current_indoor,
        "target_temp": target_temp,
        "outdoor_temp": current_outdoor,
        "pv_now": current_pv,
        "fireplace_on": 0,
        "tv_on": 0,
        "thermal_power_kw": 0,
        "indoor_temp_gradient": 0,
        "temp_diff_indoor_outdoor": target_temp - current_outdoor,
        "outlet_indoor_diff": 0,
        "pv_power_history": None 
    }
    
    print(f"\n--- Checking Model Wrapper Output ---")
    wrapper = EnhancedModelWrapper()
    
    # Force parameters
    wrapper.thermal_model.heat_loss_coefficient = 1.0
    wrapper.thermal_model.outlet_effectiveness = 0.5
    wrapper.thermal_model.pv_heat_weight = 0.005 
    wrapper.thermal_model.solar_lag_minutes = 60 
    
    # Calculate optimal outlet temp
    outlet_temp, meta = wrapper.calculate_optimal_outlet_temp(features)
    
    print(f"Optimal Outlet Temp: {outlet_temp:.2f}°C")
    
    # Compare with current conditions only
    features_current_only = features.copy()
    features_current_only["temp_forecast_1h"] = current_outdoor
    features_current_only["pv_forecast_1h"] = current_pv
    features_current_only["pv_forecast_2h"] = current_pv
    features_current_only["pv_forecast_3h"] = current_pv
    features_current_only["pv_forecast_4h"] = current_pv
    
    outlet_temp_current, _ = wrapper.calculate_optimal_outlet_temp(features_current_only)
    print(f"Outlet Temp (Current Conditions Only): {outlet_temp_current:.2f}°C")
    
    drop = outlet_temp_current - outlet_temp
    print(f"Morning Drop Magnitude: {drop:.2f}°C")

    # ISOLATION TEST: Check if drop persists with constant outdoor temp but rising PV
    print("\n--- ISOLATION TEST: Constant Outdoor, Rising PV ---")
    features_iso_pv = features.copy()
    features_iso_pv["temp_forecast_1h"] = current_outdoor
    features_iso_pv["temp_forecast_2h"] = current_outdoor
    features_iso_pv["temp_forecast_3h"] = current_outdoor
    features_iso_pv["temp_forecast_4h"] = current_outdoor
    
    outlet_temp_iso_pv, _ = wrapper.calculate_optimal_outlet_temp(features_iso_pv)
    print(f"Outlet Temp (Iso PV): {outlet_temp_iso_pv:.2f}°C")
    drop_iso_pv = outlet_temp_current - outlet_temp_iso_pv
    print(f"Drop due to PV only: {drop_iso_pv:.2f}°C")

    # ISOLATION TEST: Check if drop persists with constant PV but rising Outdoor
    print("\n--- ISOLATION TEST: Constant PV, Rising Outdoor ---")
    features_iso_outdoor = features.copy()
    features_iso_outdoor["pv_forecast_1h"] = current_pv
    features_iso_outdoor["pv_forecast_2h"] = current_pv
    features_iso_outdoor["pv_forecast_3h"] = current_pv
    features_iso_outdoor["pv_forecast_4h"] = current_pv
    
    outlet_temp_iso_outdoor, _ = wrapper.calculate_optimal_outlet_temp(features_iso_outdoor)
    print(f"Outlet Temp (Iso Outdoor): {outlet_temp_iso_outdoor:.2f}°C")
    drop_iso_outdoor = outlet_temp_current - outlet_temp_iso_outdoor
    print(f"Drop due to Outdoor only: {drop_iso_outdoor:.2f}°C")

if __name__ == "__main__":
    reproduce_morning_drop_v3_debug()
