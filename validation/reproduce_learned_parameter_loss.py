import os
import sys
import json
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from thermal_equilibrium_model import ThermalEquilibriumModel
from unified_thermal_state import get_thermal_state_manager, UNIFIED_STATE_FILE

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def test_learned_parameter_persistence():
    print("\n--- Test: Learned Parameter Persistence ---")
    
    # 1. Setup: Ensure we start with a known state
    manager = get_thermal_state_manager()
    # Reset to defaults to ensure clean slate
    manager.reset_learning_state(keep_baseline=True)
    
    # 2. Initialize model
    print("Initializing Model 1...")
    model1 = ThermalEquilibriumModel()
    
    # 3. Simulate learning: Apply an adjustment
    print("Applying adjustment to Model 1...")
    # Simulate what happens during learning
    original_delta = -0.05
    
    # We need to calculate what the new parameter value would be
    # The model stores the absolute parameter value, not the delta
    # The delta is calculated during save
    
    # Get current baseline (loaded from config or state)
    baseline_ratio = model1.heat_loss_coefficient # Using heat_loss_coefficient as proxy for ratio since ratio isn't a direct property
    
    # Let's use heat_loss_coefficient for the test as it's a direct property
    original_hlc = model1.heat_loss_coefficient
    target_hlc = original_hlc + original_delta
    
    print(f"Model 1 Heat Loss: {original_hlc:.4f}")
    print(f"Target Heat Loss: {target_hlc:.4f} (Delta: {original_delta})")
    
    # Update the model parameter
    model1.heat_loss_coefficient = target_hlc
    
    # 4. Save state
    print("Saving Model 1 state...")
    # We need to trigger the save mechanism. 
    # The model usually saves during _save_learning_to_thermal_state which is called by update_prediction_feedback
    # We can call _save_learning_to_thermal_state directly if we access it (it's protected but accessible in python)
    
    # Calculate the adjustment needed
    adjustment = target_hlc - original_hlc
    
    # Call the save method
    model1._save_learning_to_thermal_state(
        new_thermal_adjustment=0.0,
        new_heat_loss_coefficient_adjustment=adjustment,
        new_outlet_effectiveness_adjustment=0.0
    )
    
    # Verify file content
    with open(UNIFIED_STATE_FILE, 'r') as f:
        data = json.load(f)
        saved_delta = data.get("learning_state", {}).get("parameter_adjustments", {}).get("heat_loss_coefficient_delta")
        print(f"Saved Delta in JSON: {saved_delta}")
        
    if abs(saved_delta - original_delta) > 0.0001:
        print(f"FAILURE: Delta was not saved correctly to JSON. Expected {original_delta}, got {saved_delta}")
        return

    # 5. Simulate Restart
    print("\nSimulating Restart...")
    # Create new model instance
    model2 = ThermalEquilibriumModel()
    
    # 6. Verify loaded state
    # The model should load baseline + delta
    # Note: We need to know what the baseline is. 
    # If the model loaded from config defaults initially, the baseline is the config default.
    # If it loaded from a calibrated state, the baseline is in the state file.
    
    # Let's check what model2 loaded
    loaded_hlc = model2.heat_loss_coefficient
    print(f"Model 2 Heat Loss: {loaded_hlc:.4f}")
    
    if abs(loaded_hlc - target_hlc) < 0.0001:
        print("SUCCESS: Learned parameters persisted across restart.")
    else:
        print(f"FAILURE: Learned parameters LOST. Expected {target_hlc:.4f}, got {loaded_hlc:.4f}")
        print("This confirms the issue: The model is not loading the saved adjustments correctly.")

if __name__ == "__main__":
    try:
        test_learned_parameter_persistence()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
