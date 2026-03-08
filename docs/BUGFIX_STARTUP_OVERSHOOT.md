# Bugfix: Startup Overshoot (65°C Jump)

## Issue Description
After a service restart, the system would occasionally predict an extremely high required outlet temperature (often hitting the 65°C safety cap), even when the indoor temperature was close to the target. This caused uncomfortable overheating and inefficient operation.

### Symptoms
- **Sudden Spikes**: Outlet temperature target jumping from ~35°C to 65°C immediately after a restart.
- **Corrupted Parameters**: Logs showed physically impossible thermal parameters, specifically:
    - `heat_loss_coefficient` > 0.8 (indicating an open window or hole in the wall)
    - `outlet_effectiveness` < 0.2 (indicating radiators were barely working)
- **Persistence**: This "poisoned" state would persist across restarts because the corrupted parameters were saved to `unified_thermal_state.json`.

### Root Cause
The root cause was a "death spiral" in the adaptive learning logic during transient states (like startup or after a crash):
1.  The model would make a poor prediction due to missing history or sensor noise.
2.  The learning algorithm would attempt to compensate by drastically adjusting parameters.
3.  It would find a "mathematical" solution that minimized error but was physically nonsensical (e.g., "The house is losing heat instantly, but the radiators are useless").
4.  These corrupted parameters were saved to disk.
5.  On the next restart, the system loaded these broken parameters, leading to extreme predictions (65°C) to compensate for the perceived "useless" radiators.

## The Fix

### 1. Enhanced Corruption Detection
We implemented a rigorous check in `ThermalEquilibriumModel._detect_parameter_corruption()` to identify this specific failure mode.

```python
# New detection logic
if self.heat_loss_coefficient > 0.6 and self.outlet_effectiveness < 0.35:
    logging.warning("⚠️ Parameter drift detected: Moderate heat loss with very low effectiveness")
    return True
```

### 2. Automatic State Recovery
We modified `ThermalEquilibriumModel._load_thermal_parameters()` to handle corruption gracefully.
- **Before**: If parameters were loaded, they were used blindly.
- **After**: If loaded parameters are detected as corrupted, the system **immediately resets to safe defaults** (HLC=0.32, Eff=0.50) and clears the learning history.

### 3. Persistent State Repair
We updated `ThermalStateManager.load_state()` to prevent "zombie" states.
- If the system falls back to defaults due to corruption, it now **overwrites the corrupted JSON file** with the clean default state. This ensures that the next restart doesn't accidentally reload the bad parameters.

## Verification
The fix was verified using `validation/reproduce_startup_overshoot.py`, which simulated the corrupted state and confirmed that:
1.  The system detects the corruption.
2.  It resets parameters to defaults.
3.  The resulting prediction is reasonable (~43°C) instead of extreme (65°C).

## Recommendations
After applying this fix, it is highly recommended to run a fresh physics calibration:
```bash
python3 src/physics_calibration.py --days 7
```
This will establish a healthy, accurate baseline for your specific home, replacing the generic safety defaults.
