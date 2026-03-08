# Bug Fix: Morning Drop & DHW Overshoot

## 1. Sunrise Temperature Drop Fix

### Issue Description
Users reported a significant drop in indoor temperature (e.g., 20.3°C -> 19.7°C) occurring exactly at sunrise. The heating system would throttle down aggressively, anticipating solar gain that had not yet warmed the house.

### Root Cause Analysis
Two main factors contributed to this behavior:
1.  **Over-Optimistic Differential Scaling**: The thermal model included a logic that artificially boosted `outlet_effectiveness` when the difference between outlet and indoor temperature was high. This caused the model to believe it was delivering more heat than it actually was, leading it to request lower outlet temperatures.
2.  **Instant Solar Impact**: The model assumed that PV power converted to indoor heat instantly (`solar_gain_factor = 1.0`). In reality, there is a significant thermal lag (heating the floor/structure) before solar energy affects air temperature.

### Resolution
The fix involved two key changes in `src/thermal_equilibrium_model.py`:
1.  **Disabled Differential Scaling**: The scaling factor was set to `0.0`, forcing the model to rely on the base `outlet_effectiveness`. This results in higher, more realistic outlet temperature requests during the morning ramp-up.
2.  **Implemented Solar Lag**: A `solar_lag_minutes` parameter (default 45 mins) was introduced. The `_calculate_effective_solar` method now uses a rolling average of PV power over this window to smooth the solar input, correctly modeling the delay between radiation and indoor heating.

### Verification
-   **Reproduction Script**: `validation/verify_sunrise_drop.py` demonstrated that the fix reduced the target drop from ~8°C to ~2°C, which is physically realistic.
-   **Unit Tests**: Updated `tests/unit/test_thermal_equilibrium_model.py` to assert the new behavior.

---

## 2. DHW Overshoot Prevention

### Issue Description
After a Domestic Hot Water (DHW) cycle, the heating system would sometimes jump to the maximum possible temperature (e.g., 65°C) instead of resuming gently. This caused discomfort and energy waste.

### Root Cause Analysis
The `handle_grace_period` method in `src/heating_controller.py` calculated a new target temperature using the model wrapper but applied it directly to the thermostat. This bypassed the `GradualTemperatureControl` safety mechanisms that normally limit temperature changes (e.g., `MAX_TEMP_CHANGE_PER_CYCLE`).

### Resolution
The fix involved modifying `src/heating_controller.py`:
1.  **Integration of Gradual Control**: The `handle_grace_period` method now imports and uses `GradualTemperatureControl`.
2.  **Clamping**: The calculated grace target is passed through `apply_gradual_control` before being set. This ensures that the target temperature only increases by a safe amount (e.g., +2°C) from the previous setpoint, even if the model requests a much higher temperature.

### Verification
-   **Reproduction Script**: `validation/reproduce_dhw_overshoot.py` confirmed that the overshoot is prevented and the target is clamped to a safe value.
-   **Regression Test**: Added `test_grace_period_clamps_overshoot` to `tests/unit/test_heating_controller.py`.
