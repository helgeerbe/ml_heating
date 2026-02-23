import sys
import os
import numpy as np
import traceback
from unittest.mock import MagicMock
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())

from src.thermal_equilibrium_model import ThermalEquilibriumModel

def test_learning_jump():
    print("Initializing model...")
    model = ThermalEquilibriumModel()
    
    # Force a specific state
    model.pv_heat_weight = 0.001
    
    # Mock the gradient calculation to return a large NEGATIVE value
    # This should drive the weight UP.
    # Update = learning_rate * gradient
    # New_weight = Old_weight - Update
    # So if gradient is negative, Update is negative, so we subtract a negative -> add.
    
    model._calculate_adaptive_learning_rate = MagicMock(return_value=1.0)
    model._calculate_pv_heat_weight_gradient = MagicMock(return_value=-0.1)
    
    # Mock other gradients to 0 to isolate PV
    model._calculate_thermal_time_constant_gradient = MagicMock(return_value=0.0)
    model._calculate_heat_loss_coefficient_gradient = MagicMock(return_value=0.0)
    model._calculate_outlet_effectiveness_gradient = MagicMock(return_value=0.0)
    model._calculate_tv_heat_weight_gradient = MagicMock(return_value=0.0)
    
    # Mock history to allow the method to run
    # Use string timestamp to avoid JSON serialization error if it tries to save
    model.prediction_history = [{'error': 1.0, 'features': {}, 'timestamp': "2023-01-01T00:00:00"}] * 10
    
    print(f"Initial PV Weight: {model.pv_heat_weight}")
    
    # Run the adaptation
    model._assess_learning_quality = MagicMock(return_value=True)
    
    try:
        model._adapt_parameters_from_recent_errors()
    except Exception:
        traceback.print_exc()
        return

    print(f"Final PV Weight: {model.pv_heat_weight}")
    
    change = model.pv_heat_weight - 0.001
    print(f"Change: {change}")
    
    # Max bound is 0.005.
    # If it jumps to 0.005, the change is 0.004.
    if model.pv_heat_weight >= 0.005:
        print("FAIL: Parameter jumped to MAX bound in a single step!")
    elif change > 0.001:
        print("WARNING: Parameter changed significantly.")
    else:
        print("PASS: Parameter update was constrained.")

if __name__ == "__main__":
    test_learning_jump()
