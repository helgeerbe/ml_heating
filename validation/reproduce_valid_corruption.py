import os
import json
import logging
from src.thermal_equilibrium_model import ThermalEquilibriumModel
from src.unified_thermal_state import ThermalStateManager
from src.thermal_config import ThermalParameterConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_STATE_FILE = "validation_test_state.json"

def cleanup():
    if os.path.exists(TEST_STATE_FILE):
        os.remove(TEST_STATE_FILE)

def reproduce_valid_corruption():
    print("--- Reproducing Valid Parameter Corruption ---")
    
    # 1. Setup State Manager with test file
    state_manager = ThermalStateManager(state_file=TEST_STATE_FILE)
    
    # 2. Define "Valid" but "Corrupt" parameters
    # Bounds: HL (0.1, 1.5), Eff (0.3, 1.0)
    # Corruption Trigger: HL > 0.8 AND Eff < 0.35
    
    valid_high_hl = 1.0  # Valid (< 1.5) but > 0.8
    valid_low_eff = 0.32 # Valid (> 0.3) but < 0.35
    
    print(f"Testing with parameters:")
    print(f"  Heat Loss Coefficient: {valid_high_hl} (Max allowed: {ThermalParameterConfig.get_bounds('heat_loss_coefficient')[1]})")
    print(f"  Outlet Effectiveness: {valid_low_eff} (Min allowed: {ThermalParameterConfig.get_bounds('outlet_effectiveness')[0]})")
    
    # 3. Save these parameters to state
    # We need to simulate how the model saves them. 
    # The model usually saves 'learning_state' with deltas.
    # But _load_thermal_parameters also looks at 'parameters' key if set_calibrated_baseline was used.
    
    # Let's use set_calibrated_baseline to set the base values directly, 
    # as this is the clearest way to set the "current" parameters.
    state_manager.set_calibrated_baseline({
        "heat_loss_coefficient": valid_high_hl,
        "outlet_effectiveness": valid_low_eff,
        "thermal_mass": 5000.0 # Standard value
    })
    
    # Verify what's in the file
    with open(TEST_STATE_FILE, 'r') as f:
        data = json.load(f)
        saved_hl = data['parameters']['heat_loss_coefficient']
        saved_eff = data['parameters']['outlet_effectiveness']
        print(f"Saved to file - HL: {saved_hl}, Eff: {saved_eff}")
        
    # 4. Initialize Model (which loads state)
    # We need to patch the state manager used by the model or pass the file path if possible.
    # ThermalEquilibriumModel uses get_thermal_state_manager() which uses a singleton or default file.
    # We'll temporarily patch the UNIFIED_STATE_FILE in the module if needed, 
    # but ThermalEquilibriumModel doesn't take a state manager in __init__.
    # It calls load_thermal_state() -> get_thermal_state_manager().
    
    # Hack: We will instantiate the model, and then force it to load from our test manager
    # effectively by mocking the internal state loading or just relying on the fact 
    # that we can't easily change the file path for the singleton without monkeypatching.
    
    # Better approach: Monkeypatch the constant in unified_thermal_state
    import src.unified_thermal_state
    original_file = src.unified_thermal_state.UNIFIED_STATE_FILE
    src.unified_thermal_state.UNIFIED_STATE_FILE = TEST_STATE_FILE
    
    try:
        model = ThermalEquilibriumModel()
        # The model loads state in __init__ -> _load_thermal_parameters
        
        print(f"Model loaded parameters:")
        print(f"  Heat Loss Coefficient: {model.heat_loss_coefficient}")
        print(f"  Outlet Effectiveness: {model.outlet_effectiveness}")
        
        # 5. Check if reset occurred
        default_hl = ThermalParameterConfig.get_default('heat_loss_coefficient')
        default_eff = ThermalParameterConfig.get_default('outlet_effectiveness')
        
        if model.heat_loss_coefficient == default_hl and model.outlet_effectiveness == default_eff:
            print("\n[SUCCESS] Reproduction Successful: Parameters were reset to defaults!")
            print("The valid parameters triggered the corruption check.")
        elif model.heat_loss_coefficient == valid_high_hl and model.outlet_effectiveness == valid_low_eff:
            print("\n[FAILURE] Reproduction Failed: Parameters were loaded correctly.")
            print("The corruption check did not trigger.")
        else:
            print("\n[UNCLEAR] Parameters are neither defaults nor the test values.")
            
    finally:
        # Restore constant
        src.unified_thermal_state.UNIFIED_STATE_FILE = original_file
        cleanup()

if __name__ == "__main__":
    try:
        reproduce_valid_corruption()
    except Exception as e:
        print(f"An error occurred: {e}")
        cleanup()
