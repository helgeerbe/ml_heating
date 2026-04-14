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
    pv = ctx.get('pv_power')
    pv_str = f"{pv:.1f}" if pv is not None else "None"
    out_temp = ctx.get('outdoor_temp')
    out_str = f"{out_temp:.1f}" if out_temp is not None else "None"
    in_temp = ctx.get('current_indoor')
    in_str = f"{in_temp:.1f}" if in_temp is not None else "None"
    outlet = ctx.get('outlet_temp')
    outlet_str = f"{outlet:.1f}" if outlet is not None else "None"
    
    print(f"[{i}] Error: {h.get('error'):.4f} | Pred: {h.get('predicted'):.4f} | Act: {h.get('actual'):.4f} | "
          f"Out: {out_str} | In: {in_str} | "
          f"Outlet: {outlet_str} | PV: {pv_str} | "
          f"Mode: {ctx.get('learning_mode')}")
