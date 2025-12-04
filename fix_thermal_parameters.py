#!/usr/bin/env python3

"""
Fix thermal parameters by correcting the learned outlet effectiveness
"""

import sys
sys.path.insert(0, 'src')

from state_manager import load_state, save_state

def fix_thermal_parameters():
    """Fix the outlet effectiveness to a realistic value"""
    
    print("=== FIXING THERMAL PARAMETERS ===")
    
    # Load current state
    try:
        state = load_state()
        print(f"Current state loaded successfully")
        
        if 'thermal_learning_state' in state:
            thermal_state = state['thermal_learning_state']
            print(f"Current thermal parameters:")
            print(f"  - Thermal time constant: {thermal_state.get('thermal_time_constant', 'N/A')}")
            print(f"  - Heat loss coefficient: {thermal_state.get('heat_loss_coefficient', 'N/A')}")
            print(f"  - Outlet effectiveness: {thermal_state.get('outlet_effectiveness', 'N/A')} ❌ TOO LOW")
            print(f"  - Learning confidence: {thermal_state.get('learning_confidence', 'N/A')}")
            
            # Fix the outlet effectiveness to a realistic value
            thermal_state['outlet_effectiveness'] = 0.550  # Realistic value between default 0.513 and too high 0.800
            thermal_state['learning_confidence'] = 4.0  # Reset confidence but keep some learning history
            
            # Keep the parameter and prediction history but truncate to prevent bad learning
            if 'parameter_history' in thermal_state and thermal_state['parameter_history']:
                thermal_state['parameter_history'] = thermal_state['parameter_history'][-10:]  # Keep last 10
            
            if 'prediction_history' in thermal_state and thermal_state['prediction_history']:
                thermal_state['prediction_history'] = thermal_state['prediction_history'][-20:]  # Keep last 20
            
            # Save the corrected state
            state['thermal_learning_state'] = thermal_state
            save_state(**state)
            
            print(f"\n✅ FIXED thermal parameters:")
            print(f"  - Thermal time constant: {thermal_state['thermal_time_constant']} (kept)")
            print(f"  - Heat loss coefficient: {thermal_state['heat_loss_coefficient']} (kept)")
            print(f"  - Outlet effectiveness: {thermal_state['outlet_effectiveness']} ✅ CORRECTED")
            print(f"  - Learning confidence: {thermal_state['learning_confidence']} (reset)")
            print(f"  - Parameter history: {len(thermal_state.get('parameter_history', []))} entries")
            print(f"  - Prediction history: {len(thermal_state.get('prediction_history', []))} entries")
            
        else:
            print("❌ No thermal learning state found in saved state")
            return False
            
    except Exception as e:
        print(f"❌ Error loading state: {e}")
        return False
    
    return True

if __name__ == "__main__":
    if fix_thermal_parameters():
        print(f"\n🎯 Thermal parameters fixed! Restart the system to use corrected values.")
    else:
        print(f"\n❌ Failed to fix thermal parameters.")
