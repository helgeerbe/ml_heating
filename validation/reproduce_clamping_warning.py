import sys
import os
import logging
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Configure logging to show warnings
logging.basicConfig(level=logging.INFO)


def reproduce_clamping():
    print("--- Reproducing Heat Loss Coefficient Clamping ---")

    # Import here to avoid import errors if paths are tricky
    try:
        from thermal_equilibrium_model import ThermalEquilibriumModel
        from thermal_config import ThermalParameterConfig
    except ImportError:
        # Fallback for running from root
        sys.path.append('src')
        from thermal_equilibrium_model import ThermalEquilibriumModel
        from thermal_config import ThermalParameterConfig

    # 1. Check current bounds
    bounds = ThermalParameterConfig.get_bounds("heat_loss_coefficient")
    print(f"Current config bounds: {bounds}")

    # 2. Create a state with a value INSIDE the new bounds (e.g. 0.9)
    # This should now pass validation and NOT be clamped
    valid_value = 0.9
    print(f"Attempting to load state with heat_loss_coefficient = "
          f"{valid_value}")

    # Mock the state manager
    mock_state_manager = MagicMock()
    mock_state_manager.load_state.return_value = True

    state = {
        "baseline_parameters": {
            "thermal_time_constant": 4.0,
            "heat_loss_coefficient": valid_value,
            "outlet_effectiveness": 0.5,
            "pv_heat_weight": 0.002,
            "fireplace_heat_weight": 5.0,
            "tv_heat_weight": 0.2,
            "source": "calibrated"  # Must be calibrated to trigger loading logic
        },
        "learning_state": {
            "learning_confidence": 3.0,
            "prediction_history": [],
            "parameter_history": [],
            "parameter_adjustments": {}
        },
        "prediction_metrics": {},
        "operational_state": {},
        "metadata": {}
    }
    mock_state_manager.get_current_parameters.return_value = state
    mock_state_manager.state = state  # For the logging access

    with patch('unified_thermal_state.get_thermal_state_manager',
               return_value=mock_state_manager):
        print("Loading model with mocked state...")
        model = ThermalEquilibriumModel()

        print(f"Model heat_loss_coefficient after loading: "
              f"{model.heat_loss_coefficient}")

        if model.heat_loss_coefficient == valid_value:
            print("SUCCESS: Value was NOT clamped.")
        else:
            print(f"FAILURE: Value was clamped to {model.heat_loss_coefficient}")


if __name__ == "__main__":
    reproduce_clamping()
