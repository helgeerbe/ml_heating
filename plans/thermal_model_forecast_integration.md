# Thermal Model Forecast Integration Plan

## Objective
Fix the "Average vs. Instant" mismatch in the heating control logic to prevent overheating. The system currently averages solar/outdoor conditions over 4 hours, causing it to underestimate immediate solar gain and overheat the house. The fix involves passing full forecast arrays to the thermal model instead of scalar averages.

## Current Status
- **`src/thermal_equilibrium_model.py`**: Updated `predict_thermal_trajectory` to accept list/array inputs for `outdoor_temp` and `pv_power`. Added interpolation logic to map forecasts to simulation time steps.
- **`src/model_wrapper.py`**: Updated `_calculate_required_outlet_temp` to pass full `outdoor_forecast` and `pv_forecast` arrays to the thermal model.

## Implementation Plan

### 1. Verification of Code Changes
- [ ] **Audit `src/thermal_equilibrium_model.py`**:
    - Confirm `predict_thermal_trajectory` correctly distinguishes between scalar (legacy) and list (new) inputs.
    - Verify interpolation logic (`np.interp`) handles different array lengths correctly.
    - Check for potential "list of lists" bugs if `model_wrapper` passes a list wrapped in another list.
- [ ] **Audit `src/model_wrapper.py`**:
    - Confirm `_get_forecast_conditions` returns flat lists, not nested lists.
    - Verify `outdoor_forecast` and `pv_forecast` are passed to `predict_thermal_trajectory` as keyword arguments `outdoor_temp` and `pv_power`.

### 2. Interface Validation (Scripted)
- [ ] **Create `verify_trajectory_inputs.py`**:
    - A standalone script to import `ThermalEquilibriumModel`.
    - Test Case 1: Scalar inputs (Legacy behavior).
    - Test Case 2: List inputs (New behavior).
    - Test Case 3: Numpy array inputs.
    - **Success Criteria**: List inputs produce a different (more accurate) trajectory than scalar inputs for variable weather.

### 3. Integration Testing
- [ ] **Run `verify_trajectory_inputs.py`**: Execute the script and analyze output.
- [ ] **Check Logs**: Monitor `journalctl` or application logs for `predict_thermal_trajectory` errors (e.g., `TypeError`, `ValueError`).
- [ ] **Monitor System**: Observe if the "Equilibrium physics" log (instant) and "Multi-horizon predictions" (forecast) align better during sunny periods.

### 4. Calibration Adjustment
- [ ] **Lower Heat Loss Coefficient**:
    - The current coefficient (0.621) is too high, contributing to overheating.
    - Action: Ensure configuration is set to ~0.4 and verify the model respects this baseline or adapts downwards.

## Technical Details

### `predict_thermal_trajectory` Refactor
The method now supports polymorphism for `outdoor_temp` and `pv_power`:
```python
def predict_thermal_trajectory(..., outdoor_temp, pv_power, ...):
    if isinstance(outdoor_temp, list):
        # Interpolate
    else:
        # Use constant
```

### `model_wrapper` Integration
The wrapper now extracts forecasts and passes them directly:
```python
trajectory_result = self.thermal_model.predict_thermal_trajectory(
    ...
    outdoor_temp=outdoor_forecast,  # [T+1, T+2, T+3, T+4]
    pv_power=pv_forecast,          # [P+1, P+2, P+3, P+4]
    ...
)
```

## Rollback Strategy
If verification fails or runtime errors occur:
1. Revert `src/model_wrapper.py` to calculate averages (`sum(forecast)/len(forecast)`) and pass scalars.
2. Revert `src/thermal_equilibrium_model.py` to expect scalars only.
