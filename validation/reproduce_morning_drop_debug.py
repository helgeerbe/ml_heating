
import logging
import sys
import os
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

# Force logging to stdout and DEBUG level
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s:%(name)s:%(message)s',
    stream=sys.stdout,
    force=True
)

from model_wrapper import EnhancedModelWrapper
from thermal_equilibrium_model import ThermalEquilibriumModel

def reproduce_morning_drop():
    print("=== Reproducing Morning Drop Scenario ===")
    
    # 1. Setup Model Wrapper
    wrapper = EnhancedModelWrapper()
    
    # Mock the thermal model to behave predictably
    wrapper.thermal_model = ThermalEquilibriumModel()
    wrapper.thermal_model.solar_lag_minutes = 60
    # Make PV very influential to show the bug clearly
    wrapper.thermal_model.pv_heat_weight = 0.01 
    # Realistic heat loss so target is reachable
    wrapper.thermal_model.heat_loss_coefficient = 2.5 
    wrapper.thermal_model.thermal_time_constant = 40.0
    wrapper.thermal_model.outlet_effectiveness = 1.0
    
    print(f"Model Weights: {wrapper.thermal_model.external_source_weights}")
    
    # 2. Define Scenario Data
    current_indoor = 21.0
    target_indoor = 21.2 # Slight demand
    outdoor_temp = 5.0
    current_pv = 200.0
    forecast_pv_1h = 2000.0
    
    # Create a history that reflects "just woke up" - mostly 0
    pv_history = [0.0] * 10 + [50.0, 100.0, 200.0] 
    
    # Mock features
    features_with_history = {
        "indoor_temp_lag_30m": current_indoor,
        "target_temp": target_indoor,
        "outdoor_temp": outdoor_temp,
        "pv_now": current_pv,
        "pv_power_history": pv_history,
        "pv_forecast_1h": forecast_pv_1h,
        "temp_forecast_1h": outdoor_temp + 1,
        "fireplace_on": 0,
        "tv_on": 0,
        "thermal_power_kw": 0,
        "indoor_temp_gradient": 0,
        "temp_diff_indoor_outdoor": 16,
        "outlet_indoor_diff": 0,
    }
    
    features_without_history = features_with_history.copy()
    features_without_history["pv_power_history"] = None # Simulate missing history
    
    # Mock PredictionContextManager to return high avg_pv (blended)
    # avg_pv = 1359.0 (from logs)
    
    with patch("src.model_wrapper.prediction_context_manager") as mock_pcm:
        # Setup mock context
        mock_context = {
            'avg_outdoor': outdoor_temp,
            'avg_pv': 1359.0, # The value from the log
            'outdoor_forecast': [outdoor_temp]*4,
            'pv_forecast': [forecast_pv_1h]*4,
            'fireplace_on': 0.0,
            'tv_on': 0.0,
            'use_forecasts': True,
            'current_outdoor': outdoor_temp,
            'current_pv': current_pv,
            'target_temp': target_indoor,
            'current_temp': current_indoor,
        }
        mock_pcm.create_context.return_value = mock_context
        
        # Force the pre-check to pass by mocking predict_equilibrium_temperature
        # We want to force it into the binary search where the PV logic is used
        original_predict = wrapper.thermal_model.predict_equilibrium_temperature
        
        def mock_predict(*args, **kwargs):
            # If checking bounds (outlet_min/max), return values that allow search
            outlet = kwargs.get('outlet_temp')
            if outlet == 25.0: # Min bound
                return 10.0 # Too cold
            if outlet == 65.0: # Max bound
                return 30.0 # Too hot
            
            # Otherwise use real logic
            return original_predict(*args, **kwargs)
            
        wrapper.thermal_model.predict_equilibrium_temperature = mock_predict

        # Run Case A: With History
        print("\n--- Case A: With PV History ---")
        outlet_a, meta_a = wrapper.calculate_optimal_outlet_temp(features_with_history)
        print(f"Outlet Temp (With History): {outlet_a:.2f}°C")
        
        # Run Case B: Without History
        print("\n--- Case B: Without PV History ---")
        outlet_b, meta_b = wrapper.calculate_optimal_outlet_temp(features_without_history)
        print(f"Outlet Temp (Without History): {outlet_b:.2f}°C")
        
        diff = outlet_a - outlet_b
        print(f"\nDifference: {diff:.2f}°C")
        
        if outlet_b < outlet_a - 2.0:
            print("FAILURE REPRODUCED: Missing history caused significant drop in outlet temp.")
            print("Explanation: Without history, model used scalar avg_pv (1359W) as instant heat,")
            print("bypassing solar lag. With history, it correctly saw low effective solar.")
        else:
            print("Could not reproduce significant drop.")

if __name__ == "__main__":
    reproduce_morning_drop()
