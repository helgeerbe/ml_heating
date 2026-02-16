# Implementation Plan: Adaptive Learning Threshold Adjustment (0.5°C -> 0.2°C)

## Objective
Lower the adaptive learning error threshold from 0.5°C to 0.2°C to improve thermal control precision. This involves updating constants, refining metric definitions, and ensuring the learning logic targets this tighter bound.

## 1. Configuration & Constants Updates

### `src/thermal_constants.py`
- [ ] **Update `ERROR_THRESHOLD_MEDIUM`**: Change from `0.5` to `0.2`.
    - This is the primary trigger for the learning rate boost.
- [ ] **Define `ERROR_THRESHOLD_LOW`**: Ensure a constant exists for the "excellent" range (currently implied as 0.1 or 0.25 in various places).
    - Set `ERROR_THRESHOLD_LOW = 0.1` (Target precision).
    - Set `ERROR_THRESHOLD_CONFIDENCE = 0.2` (Threshold for boosting model confidence).

### `src/thermal_equilibrium_model.py`
- [ ] **Refactor `update_prediction_feedback`**:
    - Replace hardcoded `0.25` check with `PhysicsConstants.ERROR_THRESHOLD_CONFIDENCE` (0.2).
    - Logic: `if abs(error) < PhysicsConstants.ERROR_THRESHOLD_CONFIDENCE: boost_confidence()`
- [ ] **Verify `_calculate_adaptive_learning_rate`**:
    - Ensure it uses `PhysicsConstants.ERROR_THRESHOLD_MEDIUM` (which will now be 0.2).
    - Logic: `if avg_error > PhysicsConstants.ERROR_THRESHOLD_MEDIUM: apply_boost()`

## 2. Metric Definitions Updates

### `src/prediction_metrics.py`
- [ ] **Align Accuracy Categories**:
    - `EXCELLENT_THRESHOLD`: 0.1 (Unchanged)
    - `VERY_GOOD_THRESHOLD`: 0.2 (Unchanged, but now the primary learning target)
    - `GOOD_THRESHOLD`: 0.5 (Retain as "acceptable comfort" bound, but learning will be aggressive below this).

## 3. Validation & Testing

### Unit Tests
- [ ] **Update `tests/unit/test_adaptive_learning_boost.py`**:
    - Modify tests to assert that learning rate boost triggers at >0.2°C error (previously >0.5°C).
    - Verify that errors <0.2°C do *not* trigger the boost.
- [ ] **Update `tests/unit/test_thermal_equilibrium_model.py`**:
    - Verify confidence boosting logic respects the new 0.2°C constant.

### Integration Verification
- [ ] **Run `tests/integration/test_adaptive_learning.py`**: Ensure the full learning loop remains stable with tighter thresholds.

## 4. Rollout Strategy
1.  Apply constant changes.
2.  Refactor hardcoded values in `thermal_equilibrium_model.py`.
3.  Run unit tests.
4.  Run integration tests.
