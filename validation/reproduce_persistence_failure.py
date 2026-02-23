import os
import json
import logging
import sys
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

# Import modules after path setup
from thermal_equilibrium_model import ThermalEquilibriumModel  # noqa: E402
from config import UNIFIED_STATE_FILE  # noqa: E402

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')


def create_corrupted_state():
    """Create a thermal state file with a corrupted baseline."""
    state = {
        "metadata": {
            "version": "1.0",
            "last_updated": datetime.now().isoformat()
        },
        "baseline_parameters": {
            "thermal_time_constant": 4.0,
            "equilibrium_ratio": 0.17,
            "total_conductance": 0.8,
            "heat_loss_coefficient": 0.9,  # Borderline high value
            "outlet_effectiveness": 0.3,   # Borderline low value
            "pv_heat_weight": 0.002,
            "fireplace_heat_weight": 5.0,
            "tv_heat_weight": 0.2,
            "source": "calibrated",        # Mark as calibrated so it loads
            "calibration_date": datetime.now().isoformat(),
            "calibration_cycles": 10
        },
        "learning_state": {
            "cycle_count": 100,
            "learning_confidence": 3.0,
            "learning_enabled": True,
            "parameter_adjustments": {
                "equilibrium_ratio_delta": 0.0,
                "total_conductance_delta": 0.0,
                "heat_loss_coefficient_delta": 0.0,
                "outlet_effectiveness_delta": 0.0
            },
            "prediction_history": [],
            "parameter_history": []
        },
        "prediction_metrics": {
            "total_predictions": 0,
            "accuracy_stats": {}
        },
        "operational_state": {}
    }

    with open(UNIFIED_STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

    print(f"Created corrupted state file at {UNIFIED_STATE_FILE}")
    print(f"Baseline Heat Loss: "
          f"{state['baseline_parameters']['heat_loss_coefficient']}")
    print(f"Baseline Effectiveness: "
          f"{state['baseline_parameters']['outlet_effectiveness']}")


def test_persistence_failure():
    print("\n--- Test Run 1: Initial Load ---")
    # 1. Create corrupted state
    create_corrupted_state()

    # 2. Initialize model (should detect corruption and reset in memory)
    model = ThermalEquilibriumModel()

    print(f"Model Heat Loss (in memory): {model.heat_loss_coefficient}")
    print(f"Model Effectiveness (in memory): {model.outlet_effectiveness}")

    # Check if in-memory values are defaults (0.4 and 0.5)
    if (abs(model.heat_loss_coefficient - 0.4) < 0.01 and
            abs(model.outlet_effectiveness - 0.5) < 0.01):
        print("SUCCESS: Model reset to defaults in memory.")
    else:
        print("FAILURE: Model did not reset to defaults.")

    # 3. Check what's on disk
    with open(UNIFIED_STATE_FILE, 'r') as f:
        disk_state = json.load(f)

    disk_baseline_hl = disk_state['baseline_parameters'][
        'heat_loss_coefficient']
    print(f"Disk Baseline Heat Loss: {disk_baseline_hl}")

    if disk_baseline_hl > 0.8:
        print("OBSERVATION: Bad baseline persists on disk.")
    else:
        print("OBSERVATION: Baseline was reset on disk.")

    print("\n--- Test Run 2: Restart ---")
    # 4. Initialize model again (simulating restart)
    model2 = ThermalEquilibriumModel()

    print(f"Model 2 Heat Loss (in memory): {model2.heat_loss_coefficient}")
    
    # Calculate prediction with these parameters
    # Target 20, Outdoor 0
    target = 20.0
    outdoor = 0.0
    optimal_outlet = model2.calculate_optimal_outlet_temperature(
        target_indoor=target,
        current_indoor=target,  # Maintain
        outdoor_temp=outdoor,
        time_available_hours=1.0
    )

    print(f"Prediction with current params: {optimal_outlet}")

    if disk_baseline_hl > 0.8:
        print("CONCLUSION: The fix resets in-memory state but leaves the "
              "corrupted baseline on disk.")
        print("Every restart re-loads the corrupted baseline.")
        if model2.heat_loss_coefficient > 0.8:
            print("CRITICAL: The model ACCEPTED the bad baseline because "
                  "it passed validation!")
    else:
        print("CONCLUSION: The baseline was fixed on disk.")


if __name__ == "__main__":
    try:
        test_persistence_failure()
    except Exception as e:
        print(f"An error occurred: {e}")
