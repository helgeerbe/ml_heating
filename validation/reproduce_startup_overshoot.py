import sys
import os
import logging
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from physics_features import build_physics_features
from model_wrapper import simplified_outlet_prediction, EnhancedModelWrapper
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)

def reproduce_startup_overshoot():
    print("\n=== Reproducing Startup Overshoot (65°C Jump) ===\n")

    # 1. Setup Mocks
    mock_ha_client = MagicMock()
    mock_influx_service = MagicMock()

    # 2. Define Scenario
    # Scenario: Service restarts, InfluxDB returns defaults (21.0), but actual house is colder (19.0)
    # This creates a fake massive temperature drop (21.0 -> 19.0) implying a huge negative gradient.
    
    current_indoor_temp = 19.0
    target_indoor_temp = 21.0
    outdoor_temp = 5.0
    default_history_temp = 21.0  # What InfluxService returns on failure/missing data

    print(f"Scenario Configuration:")
    print(f"  Current Indoor Temp: {current_indoor_temp}°C")
    print(f"  Target Indoor Temp:  {target_indoor_temp}°C")
    print(f"  Outdoor Temp:        {outdoor_temp}°C")
    print(f"  InfluxDB History:    Missing/Failed (Defaults to {default_history_temp}°C)")

    # Mock HA Client responses
    # get_all_states returns a dict, get_state parses it
    mock_ha_client.get_all_states.return_value = {}
    
    def get_state_side_effect(entity_id, all_states, is_binary=False):
        if entity_id == config.INDOOR_TEMP_ENTITY_ID:
            return current_indoor_temp
        elif entity_id == config.OUTDOOR_TEMP_ENTITY_ID:
            return outdoor_temp
        elif entity_id == config.ACTUAL_OUTLET_TEMP_ENTITY_ID:
            return 30.0
        elif entity_id == config.TARGET_INDOOR_TEMP_ENTITY_ID:
            return target_indoor_temp
        elif entity_id == config.INLET_TEMP_ENTITY_ID:
            return 25.0
        elif entity_id == config.FLOW_RATE_ENTITY_ID:
            return 15.0
        elif entity_id == config.POWER_CONSUMPTION_ENTITY_ID:
            return 500.0
        elif entity_id == config.PV_POWER_ENTITY_ID:
            return 0.0
        return 0.0 if not is_binary else False

    mock_ha_client.get_state.side_effect = get_state_side_effect
    mock_ha_client.get_hourly_forecast.return_value = [outdoor_temp] * 4

    # Mock InfluxService responses (The Critical Part)
    # fetch_indoor_history returns defaults (21.0) when it fails or has no data
    steps = max(18, config.HISTORY_STEPS)
    mock_influx_service.fetch_indoor_history.return_value = [default_history_temp] * steps
    mock_influx_service.fetch_outlet_history.return_value = [30.0] * steps
    mock_influx_service.fetch_pv_history.return_value = [0.0] * steps

    # 3. Build Features
    print("\nBuilding Physics Features...")
    features_df, _ = build_physics_features(mock_ha_client, mock_influx_service)
    
    if features_df is None:
        print("Error: Failed to build features.")
        return

    # Extract the gradient feature
    # features_df is a dict in the current implementation of build_physics_features? 
    # No, the type hint says Tuple[Optional[pd.DataFrame], list[float]]
    # But let's check the implementation I read earlier.
    # It returns `features` which is a dict at the end of the function, but the return type says DataFrame?
    # Wait, let me check `src/physics_features.py` again.
    # Line 430: return pd.DataFrame([features]), []
    
    # features_df is a dict in the current implementation of build_physics_features?
    # Wait, looking at the code I read earlier:
    # 349 |     features = { ... }
    # It builds a dict.
    # I need to check the return statement of build_physics_features.
    # I'll assume it returns a DataFrame based on the type hint and usage in model_wrapper.
    
    # Let's just print the type to be sure in the script if I was running it interactively, 
    # but here I'll assume it's a DataFrame or dict.
    # Actually, simplified_outlet_prediction expects a DataFrame.
    
    # Let's look at the features
    if isinstance(features_df, pd.DataFrame):
        features_dict = features_df.iloc[0].to_dict()
    else:
        features_dict = features_df # Fallback if it returns a dict (unlikely given type hint)

    indoor_gradient = features_dict.get('indoor_temp_gradient')
    print(f"\nCalculated Features:")
    print(f"  indoor_temp_gradient: {indoor_gradient:.4f} (Expected negative)")
    print(f"  indoor_temp_lag_30m:  {features_dict.get('indoor_temp_lag_30m')}")
    
    # 4. Run Prediction
    print("\nRunning Simplified Outlet Prediction...")
    
    # We need to patch get_enhanced_model_wrapper to return a fresh instance or mock
    # For this test, we want the real logic, so we'll let it create a real instance,
    # but we might need to mock internal calls if they hit the network/disk.
    # The EnhancedModelWrapper uses ThermalEquilibriumModel.
    
    # We need to make sure we don't actually write to InfluxDB or HA during prediction
    with patch('model_wrapper.get_enhanced_model_wrapper') as mock_get_wrapper:
        # We'll use a real wrapper but mock its dependencies if needed
        real_wrapper = EnhancedModelWrapper()
        
        # Manually override parameters to REPRODUCE the issue with low effectiveness
        # This simulates the corrupted state found in logs
        real_wrapper.thermal_model.heat_loss_coefficient = 0.8
        real_wrapper.thermal_model.outlet_effectiveness = 0.019
        real_wrapper.thermal_model.total_conductance = 1.0
        real_wrapper.thermal_model.equilibrium_ratio = 0.5
        real_wrapper.thermal_model.pv_heat_weight = 1.0
        real_wrapper.thermal_model.tv_heat_weight = 1.0
        real_wrapper.thermal_model.fireplace_heat_weight = 1.0
        real_wrapper.thermal_model.solar_lag_hours = 1.0
        real_wrapper.thermal_model.thermal_time_constant = 4.0
        print("   [Test] Manually set PROBLEMATIC thermal parameters (simulating corrupted state)")

        # VERIFY FIX: Check if corruption detection catches this
        is_corrupt = real_wrapper.thermal_model._detect_parameter_corruption()
        print(f"   [Test] Corruption Detected: {is_corrupt}")
        
        if is_corrupt:
            print("   [Test] ✅ Corruption detection working! Simulating reset/clamping...")
            # Simulate the fix: Reset to defaults (as done in _load_thermal_parameters)
            # When corruption is detected, the system resets to safe defaults (eff=0.5)
            print("   [Test] Simulating reset to defaults (eff=0.5)...")
            real_wrapper.thermal_model.outlet_effectiveness = 0.5
            print(f"   [Test] Reset Outlet Effectiveness: {real_wrapper.thermal_model.outlet_effectiveness}")

        mock_get_wrapper.return_value = real_wrapper
        
        outlet_temp, confidence, metadata = simplified_outlet_prediction(
            features_df, current_indoor_temp, target_indoor_temp
        )

        print(f"\nPrediction Result (Gap 2.0°C):")
        print(f"  Predicted Outlet Temp: {outlet_temp:.2f}°C")
        print(f"  Confidence: {confidence}")
        print(f"  Metadata Method: {metadata.get('method')}")

        # Test Case 2: Small Gap
        print("\n--- Test Case 2: Small Gap (20.5°C vs 21.0°C) ---")
        current_indoor_small = 20.5
        features_dict_small = features_dict.copy()
        features_dict_small['indoor_temp_gradient'] = 0.0 # Still 0 gradient
        # IMPORTANT: Update lag feature to match current temp for this test case
        # In real app, this would come from history, but for test isolation we assume steady state
        features_dict_small['indoor_temp_lag_30m'] = current_indoor_small

        outlet_temp_small, _, _ = simplified_outlet_prediction(
            pd.DataFrame([features_dict_small]), current_indoor_small, target_indoor_temp
        )
        print(f"Prediction Result (Gap 0.5°C):")
        print(f"  Predicted Outlet Temp: {outlet_temp_small:.2f}°C")

        if outlet_temp >= 55.0:
            print("\n⚠️  High prediction (>=55°C) for 2°C gap.")
            if outlet_temp_small < 45.0:
                print("✅  But small gap prediction is reasonable. The 60°C might be normal aggressive heating.")
            else:
                print("❌  Small gap also predicts high temp! Something else is forcing max output.")
        else:
            print("\n✅  Prediction is within normal range.")

if __name__ == "__main__":
    reproduce_startup_overshoot()
