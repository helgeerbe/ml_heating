import sys
import os
import json
import logging

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)

def check_validity():
    print("--- Checking Unified Thermal State Validity ---")
    
    state_file = 'unified_thermal_state.json'
    if not os.path.exists(state_file):
        print(f"ERROR: {state_file} does not exist.")
        return

    try:
        with open(state_file, 'r') as f:
            state = json.load(f)
            
        print("State loaded successfully.")
        
        # Try to import validator
        try:
            from thermal_state_validator import validate_thermal_state_safely
            print("Validator imported.")
            
            is_valid = validate_thermal_state_safely(state)
            print(f"Validation Result: {is_valid}")
            
            if not is_valid:
                print("WARNING: State file is INVALID. This would cause a reset to defaults.")
            else:
                print("State file is VALID. Parameters should load correctly.")
                
        except ImportError:
            print("Could not import validator.")
            
    except Exception as e:
        print(f"Error checking validity: {e}")

if __name__ == "__main__":
    check_validity()
