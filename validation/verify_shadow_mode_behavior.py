import logging
import sys
import os
from unittest.mock import MagicMock, patch
import pandas as pd

# Add project root to path
sys.path.append(os.getcwd())

from src.model_wrapper import simplified_outlet_prediction, EnhancedModelWrapper
from src import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def test_shadow_mode_prediction():
    print("Testing simplified_outlet_prediction...")
    
    # Mock features
    features = pd.DataFrame([{
        "pv_now": 0.0,
        "fireplace_on": 0,
        "tv_on": 0,
        "thermal_power_kw": 0.0,
        "indoor_temp_lag_30m": 20.0,
        "target_temp": 21.0,
        "outdoor_temp": 10.0,
        "indoor_temp_gradient": 0.0,
        "temp_diff_indoor_outdoor": 10.0,
        "outlet_indoor_diff": 10.0
    }])
    
    current_temp = 20.0
    target_temp = 21.0
    
    # Mock EnhancedModelWrapper to avoid actual thermal model complexity if needed,
    # but we want to test the actual flow if possible.
    # However, ThermalEquilibriumModel might need state.
    # Let's try to run it with the actual wrapper but mocked internal model if it's easier,
    # or just let it run (it initializes defaults).
    
    # We need to mock get_thermal_state_manager to avoid file operations
    with patch('src.model_wrapper.get_thermal_state_manager') as mock_get_manager:
        mock_manager = MagicMock()
        mock_manager.get_learning_metrics.return_value = {"current_cycle_count": 1}
        mock_get_manager.return_value = mock_manager
        
        # Also mock InfluxDB and HA client to avoid connection errors
        with patch('src.model_wrapper.create_influx_service'), \
             patch('src.model_wrapper.create_ha_client'), \
             patch('src.model_wrapper.AdaptiveFireplaceLearning'):
            
            outlet, confidence, metadata = simplified_outlet_prediction(
                features, current_temp, target_temp
            )
            
            print(f"Outlet: {outlet}")
            print(f"Confidence: {confidence}")
            print(f"Metadata keys: {metadata.keys()}")
            
            if "predicted_indoor" in metadata:
                print(f"✅ predicted_indoor found in metadata: {metadata['predicted_indoor']}")
            else:
                print("❌ predicted_indoor NOT found in metadata")

if __name__ == "__main__":
    test_shadow_mode_prediction()
