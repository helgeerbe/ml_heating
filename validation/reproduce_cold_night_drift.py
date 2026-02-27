import sys
import os
import numpy as np
import logging
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.getcwd())

from src.thermal_equilibrium_model import ThermalEquilibriumModel
from src.thermal_constants import PhysicsConstants

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def simulate_cold_night():
    print("Initializing model...")
    model = ThermalEquilibriumModel()
    
    # Reset to known state
    model.heat_loss_coefficient = 0.6  # Start with a reasonable value
    # PROBLEM SCENARIO: Model underestimates thermal mass
    model.thermal_time_constant = 15.0 # Model thinks house reacts fast
    model.outlet_effectiveness = 0.8
    model.learning_confidence = 2.0
    model.prediction_history = []
    model.parameter_history = []
    
    # Force learning enabled
    model.adaptive_learning_enabled = True
    
    print(f"Initial Heat Loss Coefficient: {model.heat_loss_coefficient}")
    print(f"Initial Thermal Time Constant: {model.thermal_time_constant}")
    
    # Simulation parameters
    start_time = datetime(2024, 1, 15, 22, 0, 0) # 22:00
    duration_hours = 12 # Until 10:00 (longer duration to see drift)
    steps = int(duration_hours * 4) # 15 minute intervals
    
    # Initial conditions
    indoor_temp = 20.0
    outdoor_temp = 5.0
    outlet_temp = 25.0 # Heating is low/off or maintaining
    
    # True physics (what we are simulating against)
    # We simulate a house that actually loses heat, but has high inertia
    true_hlc = 0.6
    # REALITY: House has high thermal mass (slow reaction)
    true_ttc = 40.0
    
    history = []
    
    print("\nStarting simulation (22:00 -> 04:00)...")
    print(f"{'Time':<20} | {'Outdoor':<8} | {'Indoor':<8} | {'Pred':<8} | {'Error':<8} | {'HLC':<8} | {'TTC':<8}")
    print("-" * 100)
    
    prev_indoor = indoor_temp
    prev_outdoor = outdoor_temp
    prev_outlet = outlet_temp

    for i in range(steps):
        current_time = start_time + timedelta(minutes=i*15)
        
        # 1. Update Environment
        # Outdoor temp drops from 5C to -2C over 6 hours
        outdoor_temp = 5.0 - (7.0 * (i / steps))
        
        # 2. Simulate Physics (True House Behavior)
        # Simple Euler integration for true indoor temp
        # dT/dt = (1/tau) * (T_eq - T_current)
        # T_eq = T_out + (Power / HLC) -> Power is small here, maybe just maintenance
        # Let's assume outlet_temp provides some heat
        # Power ~ (outlet - indoor) * effectiveness * flow_factor? 
        # Using model's equation for simplicity but with fixed "True" params
        
        # Calculate equilibrium for the "True" house
        # Using the model's formula structure for the "truth" simulation
        total_conductance = true_hlc + model.outlet_effectiveness # Assuming effectiveness is correct
        
        # Simplified: T_eq = (eff * T_outlet + HLC * T_outdoor) / (HLC + eff)
        true_equilibrium = (model.outlet_effectiveness * outlet_temp + true_hlc * outdoor_temp) / total_conductance
        
        # Decay towards equilibrium
        # time_step = 0.25 hours
        dt = 0.25
        approach = 1 - np.exp(-dt / true_ttc)
        indoor_temp = indoor_temp + (true_equilibrium - indoor_temp) * approach
        
        # 3. Model Prediction (before seeing the new indoor_temp)
        # The model predicts what indoor_temp *should* be based on its CURRENT parameters
        # We need to predict from the PREVIOUS step's indoor temp.
        
        # Predict what indoor temp would be now, given previous state and current parameters
        prediction_result = model.predict_thermal_trajectory(
            current_indoor=prev_indoor,
            target_indoor=prev_indoor, # Irrelevant for trajectory
            outlet_temp=prev_outlet,
            outdoor_temp=prev_outdoor,
            time_horizon_hours=0.25,
            time_step_minutes=15
        )
        predicted_temp = prediction_result['trajectory'][0]
        
        # 4. Feedback Loop
        # The model sees the *actual* indoor_temp we just simulated
        context = {
            "outlet_temp": prev_outlet,
            "current_indoor": prev_indoor,
            "outdoor_temp": prev_outdoor,
            "indoor_temp_gradient": (indoor_temp - prev_indoor) / 0.25, # Simple gradient
            "pv_power": 0,
            "fireplace_on": 0,
            "tv_on": 0
        }
        
        error = model.update_prediction_feedback(
            predicted_temp=predicted_temp,
            actual_temp=indoor_temp,
            prediction_context=context,
            timestamp=current_time.isoformat()
        )
        
        # Store for next step
        prev_indoor = indoor_temp
        prev_outdoor = outdoor_temp
        prev_outlet = outlet_temp
        
        history.append({
            "time": current_time,
            "outdoor": outdoor_temp,
            "indoor": indoor_temp,
            "predicted": predicted_temp,
            "error": error,
            "hlc": model.heat_loss_coefficient,
            "ttc": model.thermal_time_constant,
            "oe": model.outlet_effectiveness
        })
        
        print(f"{current_time.strftime('%H:%M')} | {outdoor_temp:6.2f} | {indoor_temp:6.2f} | {predicted_temp:6.2f} | {error:6.3f} | {model.heat_loss_coefficient:6.4f} | {model.thermal_time_constant:6.2f}")

    # Analysis
    initial_hlc = history[0]['hlc']
    final_hlc = history[-1]['hlc']
    hlc_drift = final_hlc - initial_hlc

    # Also check Outlet Effectiveness drift
    # If effectiveness drifts UP, it means the model thinks radiators are BETTER than they are,
    # which also leads to lower target temperatures (undershoot).
    initial_oe = history[0]['oe']
    final_oe = history[-1]['oe']
    oe_drift = final_oe - initial_oe
    
    print("\nAnalysis:")
    print(f"Initial HLC: {initial_hlc:.4f}")
    print(f"Final HLC:   {final_hlc:.4f}")
    print(f"HLC Drift:   {hlc_drift:.4f}")
    print(f"Initial OE:  {initial_oe:.4f}")
    print(f"Final OE:    {final_oe:.4f}")
    print(f"OE Drift:    {oe_drift:.4f}")
    
    failed = False
    if hlc_drift < -0.05:
        print("FAIL: Significant downward drift in Heat Loss Coefficient detected!")
        failed = True
    
    if oe_drift > 0.01:
        print("FAIL: Significant upward drift in Outlet Effectiveness detected!")
        print("The model is learning that radiators are more effective than reality,")
        print("which will cause it to request lower water temperatures.")
        failed = True

    if not failed:
        print("PASS: Parameters remained stable.")

if __name__ == "__main__":
    simulate_cold_night()
