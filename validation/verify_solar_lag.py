import matplotlib.pyplot as plt
import numpy as np
from src.thermal_equilibrium_model import ThermalEquilibriumModel

def simulate_morning_spike():
    """
    Simulate a morning PV spike scenario to verify the solar lag effect.
    """
    model = ThermalEquilibriumModel()
    model.solar_lag_minutes = 60.0
    model.external_source_weights["pv"] = 0.005 # Stronger effect for visibility
    
    # Scenario: 3 hours (18 steps of 10 mins)
    # Hour 1: Night (0 PV)
    # Hour 2: Sunrise (Ramp to 1000)
    # Hour 3: Full Sun (1000)
    
    steps = 36 # 6 hours total
    pv_profile = []
    
    # 0-60 mins: 0
    pv_profile.extend([0.0] * 6)
    
    # 60-120 mins: Ramp 0 -> 1000
    pv_profile.extend(np.linspace(0, 1000, 6).tolist())
    
    # 120-360 mins: 1000
    pv_profile.extend([1000.0] * 24)
    
    # Simulation
    outlet = 30.0
    outdoor = 10.0
    indoor = 20.0
    
    results_lag = []
    results_no_lag = []
    
    # Rolling buffer for lag simulation
    pv_buffer = [0.0] * 18 # 3 hours history
    
    print("Step | PV (W) | Eff PV (W) | Teq (Lag) | Teq (No Lag)")
    print("-" * 50)
    
    for i, pv_now in enumerate(pv_profile):
        # Update buffer
        pv_buffer.append(pv_now)
        if len(pv_buffer) > 18:
            pv_buffer.pop(0)
            
        # Predict with lag
        model.solar_lag_minutes = 60.0
        teq_lag = model.predict_equilibrium_temperature(
            outlet, outdoor, indoor, pv_power=pv_buffer
        )
        results_lag.append(teq_lag)
        
        # Predict without lag (instantaneous)
        model.solar_lag_minutes = 0.0
        teq_no_lag = model.predict_equilibrium_temperature(
            outlet, outdoor, indoor, pv_power=pv_now # Pass scalar for instant
        )
        results_no_lag.append(teq_no_lag)
        
        # Calculate effective PV for display
        eff_pv = np.mean(pv_buffer[-6:]) # 60 min lag = last 6 steps
        
        if i % 3 == 0:
            print(f"{i:4d} | {pv_now:6.1f} | {eff_pv:10.1f} | {teq_lag:9.2f} | {teq_no_lag:12.2f}")

    # Verification
    # The lagged temperature should be consistently LOWER than the no-lag temperature
    # during the ramp-up phase.
    
    ramp_start = 6
    ramp_end = 12
    
    diffs = np.array(results_no_lag) - np.array(results_lag)
    avg_diff_ramp = np.mean(diffs[ramp_start:ramp_end+6]) # Check ramp + 1 hour
    
    print(f"\nAverage Temp Difference during Ramp (No Lag - Lag): {avg_diff_ramp:.3f}°C")
    
    if avg_diff_ramp > 0.1:
        print("✅ SUCCESS: Solar lag is effectively smoothing the temperature rise.")
    else:
        print("❌ FAILURE: Solar lag effect is negligible.")

if __name__ == "__main__":
    simulate_morning_spike()
