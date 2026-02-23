import sys
import os
import logging
from unittest.mock import MagicMock, patch
import json

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Import modules to ensure they are loaded and we know their names
import thermal_equilibrium_model
import unified_thermal_state
import thermal_state_validator
from thermal_config import ThermalParameterConfig as config

# Configure logging
logging.basicConfig(level=logging.INFO)

def reproduce_validation_fallback():
    print("--- Reproducing Validation Fallback Issue ---")
    
    # 1. Mock unified_thermal_state to return CALIBRATED params
    calibrated_state = {
        "baseline_parameters": {
            "thermal_time_constant": 20.0,
            "heat_loss_coefficient": 0.5,
            "outlet_effectiveness": 0.4,
            "source": "calibrated",
            "pv_heat_weight": 0.0,
            "fireplace_heat_weight": 0.0,
            "tv_heat_weight": 0.0
        },
        "learning_state": {}
    }
    
    # 2. Mock State Manager
    mock_manager = MagicMock()
    mock_manager.get_current_parameters.return_value = calibrated_state
    
    # 3. Mock Validator to FAIL
    # We patch the modules as they are known to sys.modules (top-level because src is in path)
    
    print(f"Patching unified_thermal_state.get_thermal_state_manager...")
    print(f"Patching thermal_state_validator.validate_thermal_state_safely...")

    with patch('unified_thermal_state.get_thermal_state_manager', return_value=mock_manager) as mock_get_manager, \
         patch('thermal_state_validator.validate_thermal_state_safely', return_value=False) as mock_validate:
        
        model = thermal_equilibrium_model.ThermalEquilibriumModel()
        
        print(f"Mock Manager called: {mock_get_manager.called}")
        print(f"Mock Validate called: {mock_validate.called}")

        # Check if we have calibrated or default params
        default_hlc = config.get_default("heat_loss_coefficient")
        
        print(f"Calibrated HLC in state: {calibrated_state['baseline_parameters']['heat_loss_coefficient']}")
        print(f"Default HLC in config: {default_hlc}")
        print(f"Model HLC after load: {model.heat_loss_coefficient}")
        
        if model.heat_loss_coefficient == 0.5:
            print("SUCCESS: Model retained calibrated params despite validation failure.")
        elif model.heat_loss_coefficient == default_hlc:
            print("FAILURE: Model fell back to defaults (Issue reproduced).")
        else:
            print(f"FAILURE: Model has unexpected value: {model.heat_loss_coefficient}")

if __name__ == "__main__":
    reproduce_validation_fallback()
