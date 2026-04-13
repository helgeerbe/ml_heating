import json
import numpy as np

with open('unified_thermal_state.json', 'r') as f:
    data = json.load(f)

param_history = data.get('learning_state', {}).get('parameter_history', [])
if not param_history:
    print("No parameter history found.")
    exit()

print("--- Parameter Evolution ---")
for i, p in enumerate(param_history):
    print(f"[{i}] HLC: {p.get('heat_loss_coefficient'):.4f} | OE: {p.get('outlet_effectiveness'):.4f} | "
          f"PV: {p.get('pv_heat_weight'):.4f} | TV: {p.get('tv_heat_weight'):.4f} | "
          f"LR: {p.get('learning_rate'):.4f} | Conf: {p.get('learning_confidence'):.1f}")

print("\n--- Gradients ---")
for i, p in enumerate(param_history):
    g = p.get('gradients', {})
    print(f"[{i}] HLC_g: {g.get('heat_loss_coefficient'):.4f} | OE_g: {g.get('outlet_effectiveness'):.4f} | "
          f"PV_g: {g.get('pv_heat_weight'):.4f}")

