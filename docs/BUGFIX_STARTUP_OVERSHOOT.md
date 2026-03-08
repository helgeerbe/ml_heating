# Bugfix: Startup Overshoot (65°C Spike)

## Problem Description
After a system restart, the heating controller would sometimes immediately request the maximum possible outlet temperature (65°C), even if the indoor temperature was close to the target. This caused significant overheating and discomfort.

## Root Cause Analysis
The issue was traced to **"State Poisoning"** where the thermal parameters in the persistent state file (`unified_thermal_state.json`) became corrupted.

1.  **Corrupted State**: The state file contained a physically impossible combination of parameters:
    *   **Heat Loss Coefficient (HLC)**: Extremely high (> 0.8, sometimes ~1.2)
    *   **Outlet Effectiveness**: Extremely low (< 0.35, sometimes ~0.019)

2.  **Mechanism**:
    *   The high HLC made the model believe the house was losing heat rapidly.
    *   The low Effectiveness made the model believe the radiators were extremely inefficient.
    *   **Result**: To maintain equilibrium, the model calculated that an incredibly high outlet temperature was required.

3.  **Persistence**:
    *   Previous fixes detected this corruption in memory but didn't aggressively clean up the disk file.
    *   On the next restart, the system would reload the corrupted file, re-applying the bad parameters.

## Solution Implementation

The fix involves a multi-layered defense strategy to detect, reject, and permanently remove corrupted state data.

### 1. Enhanced Corruption Detection
Updated `src/thermal_equilibrium_model.py` to detect specific "toxic combinations" of parameters, not just individual bounds violations.

```python
# In ThermalEquilibriumModel._detect_parameter_corruption
if (self.heat_loss_coefficient > 0.6 and self.outlet_effectiveness < 0.35):
    logging.warning("⚠️ Parameter drift detected: Moderate heat loss with very low effectiveness...")
    return True
```

### 2. Aggressive State Reset
Modified `src/thermal_equilibrium_model.py` to take decisive action when corruption is detected during loading:

*   **Wipe Baseline**: If the baseline parameters themselves are corrupted, the entire state is wiped.
*   **Reset to Defaults**: The system immediately reverts to safe, hardcoded configuration defaults.
*   **Prevent Use**: The loading function returns early to ensure the corrupted values are never used in calculations.

### 3. Atomic State Overwrite
Modified `src/unified_thermal_state.py` to ensure that when the system falls back to defaults, it **overwrites the corrupted file on disk**.

```python
# In ThermalStateManager.load_state
logging.warning("🧹 Overwriting corrupted state file with fresh defaults...")
self.state = self._get_default_state()
self.save_state() # Forces atomic write of clean state
```

This prevents "zombie states" where a bad file persists on disk and gets reloaded later.

## Verification

### Reproduction Script
A reproduction script (`validation/reproduce_startup_overshoot.py`) was created to simulate the corrupted state:
1.  Manually sets HLC=0.8 and Effectiveness=0.019.
2.  Runs the prediction logic.
3.  Verifies that the model predicts ~65°C.
4.  Verifies that the new detection logic catches the corruption.
5.  Verifies that resetting to defaults (Effectiveness=0.5) restores a sane prediction (~30-40°C).

### Results
*   **Before Fix**: Prediction = 65.0°C (Clamped Max)
*   **After Fix**: Corruption detected -> Reset to Defaults -> Prediction = ~38.5°C (Normal)

## Related Files
*   `src/thermal_equilibrium_model.py`
*   `src/unified_thermal_state.py`
*   `validation/reproduce_startup_overshoot.py`
