import json
import os

def main():
    try:
        with open('unified_thermal_state.json', 'r') as f:
            state = json.load(f)
            print("Unified Thermal State:")
            print(f"  Baseline: {state.get('baseline', {})}")
            print(f"  Adjustments: {state.get('adjustments', {})}")
            print(f"  Learning State: {state.get('learning_state', {})}")
    except Exception as e:
        print(f"Error reading state: {e}")

main()
