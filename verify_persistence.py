
import logging
import sys
import os
from src.unified_thermal_state import get_thermal_state_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def verify_persistence():
    logging.info("=== Verifying Thermal Time Constant Persistence ===")
    
    # 1. Get the manager
    manager = get_thermal_state_manager()
    
    # 2. Read current value
    current_params = manager.get_computed_parameters()
    initial_tau = current_params.get('thermal_time_constant')
    logging.info(f"Initial thermal_time_constant: {initial_tau}")
    
    # 3. Set a new calibrated baseline
    new_tau = 15.5
    logging.info(f"Setting new calibrated baseline with thermal_time_constant = {new_tau}...")
    
    # Create a dummy set of parameters (preserving others if possible, or just setting what we need)
    # We'll just use the current ones and update tau
    new_params = current_params.copy()
    new_params['thermal_time_constant'] = new_tau
    
    # We need to pass these to set_calibrated_baseline
    # Note: set_calibrated_baseline expects a dictionary of parameters
    manager.set_calibrated_baseline(new_params)
    
    # 4. Force a reload from disk to ensure it was saved and can be loaded
    # We'll create a new manager instance to be sure
    logging.info("Reloading state from disk...")
    new_manager = get_thermal_state_manager()
    # Force reload explicitly just in case singleton kept state (though set_calibrated_baseline saves to disk)
    new_manager.load_state()
    
    # 5. Check the value
    loaded_params = new_manager.get_computed_parameters()
    loaded_tau = loaded_params.get('thermal_time_constant')
    logging.info(f"Loaded thermal_time_constant: {loaded_tau}")
    
    if loaded_tau == new_tau:
        logging.info("✅ SUCCESS: thermal_time_constant was correctly persisted!")
        return True
    else:
        logging.error(f"❌ FAILURE: Expected {new_tau}, got {loaded_tau}")
        return False

if __name__ == "__main__":
    success = verify_persistence()
    sys.exit(0 if success else 1)
