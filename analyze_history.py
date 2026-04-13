import json
import numpy as np

with open('unified_thermal_state.json', 'r') as f:
    data = json.load(f)

history = data.get('learning_state', {}).get('prediction_history', [])
if not history:
    print("No history found.")
    exit()

errors = [h.get('error', 0) for h in history]
print(f"Total predictions: {len(errors)}")
print(f"Mean Error (Bias): {np.mean(errors):.4f}")
print(f"Mean Absolute Error (MAE): {np.mean(np.abs(errors)):.4f}")
print(f"Max Error: {np.max(np.abs(errors)):.4f}")

# Check parameter history
param_history = data.get('learning_state', {}).get('parameter_history', [])
print(f"\nParameter updates: {len(param_history)}")
if param_history:
    print("First update:")
    print(param_history[0])
    print("Last update:")
    print(param_history[-1])

