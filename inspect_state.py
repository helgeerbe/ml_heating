import json
import os

files = ['unified_thermal_state.json', 'thermal_state.json']
for f in files:
    if os.path.exists(f):
        print(f"--- {f} ---")
        with open(f, 'r') as file:
            try:
                data = json.load(file)
                print(f"Baseline: {data.get('baseline_parameters', {})}")
                learning = data.get('learning_state', {})
                print(f"Adjustments: {learning.get('parameter_adjustments', {})}")
                print(f"Confidence: {learning.get('learning_confidence')}")
                metrics = data.get('prediction_metrics', {})
                print(f"Metrics: {metrics}")
                history = learning.get('prediction_history', [])
                print(f"History length: {len(history)}")
                if history:
                    print(f"Last 5 errors: {[h.get('error') for h in history[-5:]]}")
                    print(f"Last 5 contexts: {[h.get('context') for h in history[-5:]]}")
            except Exception as e:
                print(f"Error reading {f}: {e}")
    else:
        print(f"--- {f} not found ---")
