# Bugfix: Startup Overshoot (65°C Jump)

## Issue Description
After a system restart, the heating controller would sometimes immediately request the maximum possible outlet temperature (65°C), even when the indoor temperature was close to the target (e.g., 19.0°C vs 21.0°C). This caused significant overheating and discomfort.

## Root Cause Analysis
The issue was traced to **thermal parameter corruption** that persisted across restarts.

1.  **Corrupted State**: The `thermal_state.json` file contained a physically impossible combination of parameters:
    *   `heat_loss_coefficient`: **0.8** (Extremely high, implying open windows/doors)
    *   `outlet_effectiveness`: **0.019** (Extremely low, implying radiators are wrapped in blankets)

2.  **The "Death Spiral"**:
    *   The high heat loss coefficient caused the model to believe the house was losing heat rapidly.
    *   The low outlet effectiveness caused the model to believe the radiators were incredibly inefficient.
    *   **Result**: To maintain equilibrium against the perceived massive heat loss using "inefficient" radiators, the model calculated that it needed an infinite amount of heat, capping out at the maximum safety limit (65°C).

3.  **Persistence**: The system correctly loaded these values from the state file on startup. While there were some bounds checks, this specific *combination* (High Loss + Low Effectiveness) was not explicitly forbidden, allowing the "poisoned" state to survive restarts.

## Fix Implementation

### 1. Enhanced Corruption Detection
We updated `src/thermal_equilibrium_model.py` to detect this specific drift pattern.

```python
# src/thermal_equilibrium_model.py

def _detect_parameter_corruption(self) -> bool:
    # ... existing bounds checks ...

    # NEW: Check for physically impossible combinations (drift detection)
    
    # 1. Extreme heat loss check
    if self.heat_loss_coefficient > 1.8:
        logging.warning("⚠️ Extreme heat loss detected (%.2f)...", self.heat_loss_coefficient)
        return True

    # 2. The "Death Spiral" combo check
    # High heat loss (>0.6) AND Low effectiveness (<0.35)
    if self.heat_loss_coefficient > 0.6 and self.outlet_effectiveness < 0.35:
        logging.warning(
            "⚠️ Parameter drift detected: Moderate heat loss (%.2f) "
            "with very low effectiveness (%.2f) causes extreme predictions",
            self.heat_loss_coefficient,
            self.outlet_effectiveness,
        )
        return True
        
    return False
```

### 2. Automatic State Reset
When corruption is detected during the loading process (`_load_thermal_parameters`), the system now:
1.  **Wipes the corrupted state**: It clears the learning adjustments and, if necessary, the baseline parameters.
2.  **Resets to defaults**: It reloads the safe default configuration (e.g., HLC=0.32, Eff=0.5).
3.  **Prevents Bad Learning**: It disables learning for the current cycle to prevent re-learning the bad parameters immediately.

## Verification
A reproduction script (`validation/reproduce_startup_overshoot.py`) was created to simulate the corrupted state.

**Before Fix:**
```
Predicted Outlet Temp: 65.00°C (Capped)
```

**After Fix:**
```
[Test] Corruption Detected: True
[Test] ✅ Corruption detection working! Simulating reset/clamping...
[Test] Reset Outlet Effectiveness: 0.5
Predicted Outlet Temp: 43.20°C (Reasonable)
```

The system now correctly identifies the invalid state on startup, resets to safe defaults, and produces a reasonable outlet temperature prediction.
