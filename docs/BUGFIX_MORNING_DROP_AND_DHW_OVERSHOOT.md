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

---

## 3. Morning Drop Regression (Forecast Blending)

### Issue Description
Despite the previous fix for solar lag, a "morning drop" was still observed where the system would throttle down prematurely as soon as the solar forecast predicted sun, even if the current PV output was zero.

### Root Cause Analysis
The `UnifiedPredictionContext` calculates an `avg_pv` value by blending the current PV sensor reading with the 1-hour forecast. This blended value is a scalar (single float).
When `src/model_wrapper.py` passed this scalar `avg_pv` to the thermal model's `predict_equilibrium_temperature` method, the model treated it as an "instant" heat source, bypassing the solar lag calculation (which requires a history list to calculate a rolling average).
Consequently, the model reacted to the *forecasted* solar gain (blended into `avg_pv`) as if it were immediate heat, causing the target temperature to drop before the sun actually had an effect.

### Resolution
The fix involved modifying `src/model_wrapper.py` to:
1.  **Extract PV History**: Explicitly extract `pv_power_history` from the input features.
2.  **Prioritize History**: Pass the `pv_power_history` list to `predict_equilibrium_temperature` instead of the scalar `avg_pv` whenever available. This ensures the thermal model's `_calculate_effective_solar` method can correctly apply the lag logic using the historical data.
3.  **Correct Trajectory Initialization**: Updated `predict_thermal_trajectory` to use the history list for initializing the lag buffer, while still using the forecast array for future steps.

### Verification
-   **Reproduction Script**: `validation/reproduce_morning_drop_fix.py` confirmed that the model was previously receiving a scalar (blended forecast) and now receives the full PV history list (zeros at night/morning), preventing the premature drop.

---

## 4. Morning Drop Regression (Dynamic Horizon & History Fix)

### Issue Description
Despite previous fixes, a significant temperature drop was still observed in the morning when the forecast predicted high solar gain, even if current PV was zero. The system would "coast" on the expected future heat, letting the house cool down.

### Root Cause Analysis
1.  **Incorrect History Initialization**: When `pv_power_history` was missing (e.g., at startup), the code used `avg_pv` (a blended average of current + forecast) to initialize the history. This caused the thermal model to "hallucinate" past solar gain, bypassing the solar lag mechanism.
2.  **Optimization Horizon**: The 4-hour optimization horizon allowed the model to see future solar gains and reduce immediate heating, even when the house was currently cold.

### Resolution
The fix involved two key changes in `src/model_wrapper.py`:
1.  **Correct History Init**: Used `current_pv` (sensor reading) instead of `avg_pv` for history initialization.
2.  **Dynamic Horizon**: Implemented logic to shorten the optimization horizon to 1.0h when the house is cold (>0.3°C below target) to prioritize immediate recovery, while keeping it at 4.0h for stability when close to target.

### Verification
-   **Reproduction Script**: `validation/reproduce_morning_drop_v2.py` demonstrated that the fix reduced the drop from ~15°C to 0°C.
-   **Regression Test**: Added `tests/unit/test_morning_drop.py` with comprehensive tests for both the history initialization and the dynamic horizon logic.
