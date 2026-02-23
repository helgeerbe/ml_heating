import sys
import os
import logging
from unittest.mock import MagicMock, patch
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from model_wrapper import EnhancedModelWrapper
from thermal_config import ThermalParameterConfig as config

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def reproduce_unreachable_condition():
    """
    Simulate conditions where the model might clamp to 14.0°C (outlet_min).
    """
    print("--- Reproducing Unreachable Condition ---")
    
    wrapper = EnhancedModelWrapper()
    
    # Mock the thermal model's predict_equilibrium_temperature
    # Scenario: Even at min outlet (14°C), the house temp is predicted to be HIGH (e.g. 22°C)
    # If target is 20°C, then 20 < 22, so it's "unreachable" (cannot cool down enough).
    # The code should return outlet_min (14°C).
    
    # We need to mock _get_forecast_conditions to avoid complex context setup
    wrapper._get_forecast_conditions = MagicMock(return_value=(
        10.0, # avg_outdoor
        0.0,  # avg_pv
        [10.0]*5, # outdoor_forecast
        [0.0]*5   # pv_forecast
    ))
    
    # Mock thermal_model.predict_equilibrium_temperature
    # We want min_prediction (at 14°C outlet) to be HIGHER than target.
    target_indoor = 20.0
    predicted_at_min_outlet = 22.0 
    predicted_at_max_outlet = 40.0
    
    def mock_predict(outlet_temp, **kwargs):
        if outlet_temp <= 14.0:
            return predicted_at_min_outlet
        return predicted_at_max_outlet

    wrapper.thermal_model.predict_equilibrium_temperature = MagicMock(side_effect=mock_predict)
    
    # Inputs
    current_indoor = 21.0 # House is warm
    outdoor_temp = 10.0
    thermal_features = {"pv_power": 0.0}
    
    print(f"Inputs: Target={target_indoor}°C, Current={current_indoor}°C")
    print(f"Mock Model Prediction at Min Outlet (14°C): {predicted_at_min_outlet}°C")
    
    # Run calculation
    result = wrapper._calculate_required_outlet_temp(
        current_indoor,
        target_indoor,
        outdoor_temp,
        thermal_features
    )
    
    print(f"Resulting Outlet Temp: {result}°C")
    
    if result == 14.0:
        print("SUCCESS: Reproduced clamping to 14.0°C due to 'unreachable' target.")
    else:
        print(f"FAILURE: Result was {result}°C, expected 14.0°C")

if __name__ == "__main__":
    reproduce_unreachable_condition()
