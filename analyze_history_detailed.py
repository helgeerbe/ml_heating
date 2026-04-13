import json

with open('unified_thermal_state.json', 'r') as f:
    data = json.load(f)

history = data.get('learning_state', {}).get('prediction_history', [])
if not history:
    print("No history found.")
    exit()

print("--- Detailed Prediction History ---")
for i, h in enumerate(history):
    ctx = h.get('context', {})
    print(f"[{i}] Error: {h.get('error'):.4f} | Pred: {h.get('predicted'):.4f} | Act: {h.get('actual'):.4f} | "
          f"Out: {ctx.get('outdoor_temp'):.1f} | In: {ctx.get('current_indoor'):.1f} | "
          f"Outlet: {ctx.get('outlet_temp'):.1f} | PV: {ctx.get('pv_power'):.1f} | "
          f"Mode: {ctx.get('learning_mode')}")

