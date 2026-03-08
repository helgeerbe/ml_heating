# Plan for Adding Regression Tests for DHW Overshoot

## Objective
Ensure that the heating controller correctly clamps the target outlet temperature after a blocking event (like DHW heating) to prevent sudden temperature spikes (overshoot), even if the thermal model predicts a high requirement.

## Analysis
The `BlockingStateManager.handle_grace_period` method in `src/heating_controller.py` is responsible for restoring the temperature after a blocking event. It uses `GradualTemperatureControl` to limit the rate of change.
The key logic is:
1. Calculate `raw_grace_target` from the model.
2. Apply `gradual_ctrl.apply_gradual_control(raw_grace_target, ..., state)`.
3. `apply_gradual_control` uses `state.last_final_temp` as a baseline and limits the change to `config.MAX_TEMP_CHANGE_PER_CYCLE`.

## Proposed Test
We will add a new test case `test_grace_period_clamps_overshoot` to `tests/unit/test_heating_controller.py`.

### Test Setup
1.  **Mock `HAClient`**: To simulate sensor readings.
2.  **Mock `SystemState`**:
    *   `last_is_blocking = True`
    *   `last_blocking_end_time` = recent timestamp
    *   `last_final_temp` = 40.0°C (Baseline)
3.  **Mock Model Wrapper**:
    *   `_calculate_required_outlet_temp` returns 65.0°C (High prediction).
    *   `_extract_thermal_features` returns dummy features.
4.  **Mock Configuration**:
    *   `MAX_TEMP_CHANGE_PER_CYCLE` = 2.0°C.

### Execution
Call `blocking_manager.handle_grace_period(ha_client, state)`.

### Verification
Assert that `ha_client.set_state` is called for `TARGET_OUTLET_TEMP_ENTITY_ID` with a value of **42.0°C** (40.0 + 2.0), verifying that the 65.0°C prediction was clamped.

## Implementation Steps
1.  Edit `tests/unit/test_heating_controller.py`.
2.  Add the `test_grace_period_clamps_overshoot` method to the `TestBlockingStateManager` class.
3.  Run the tests using `pytest`.
