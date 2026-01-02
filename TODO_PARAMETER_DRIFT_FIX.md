# TODO: Parameter Drift Fix - ML Heating System

**Issue Date:** January 2, 2026  
**Priority:** CRITICAL  
**Status:** In Progress  

## Problem Summary

### Original Issue
- **Symptoms:** ML prediction accuracy 0.0%, MAE 12.5°C, RMSE 12.97°C
- **Behavior:** Target outlet always 65°C while heat curve uses 48°C
- **Prediction Error:** Model predicts ~8.8°C when actual is ~20.8°C (12°C error)

### Root Cause Analysis ✅
**IDENTIFIED:** Corrupted thermal parameters in `thermal_state.json`:
- `equilibrium_ratio`: 0.1 (should be ~0.8) 
- `total_conductance`: 0.266 (should be ~0.05)
- `learning_confidence`: 0.0 (system gave up learning)

### Temporary Fix Applied ✅
- Used `--calibrate-physics` to restore reasonable parameters
- System now predicts 21.30°C vs target 21.0°C (0.3°C error)
- **Problem:** Parameters will drift back to corruption due to flawed learning algorithm

## Critical Learning Algorithm Flaws Found

### 1. Learning from Catastrophic Errors
**File:** `src/thermal_equilibrium_model.py`  
**Method:** `_calculate_adaptive_learning_rate()`  
**Issue:** System still learns from 12°C prediction errors at reduced rate, but should STOP learning entirely

### 2. Gradient Calculation During Corruption
**Method:** `_calculate_parameter_gradient()`  
**Issue:** Calculates gradients using corrupted model state, amplifying garbage predictions

### 3. Insufficient Parameter Bounds
**Method:** `_adapt_parameters_from_recent_errors()`  
**Issue:** Bounds applied AFTER gradient updates, gradients calculated from corrupted states

### 4. Delta Accumulation Corruption  
**Method:** `_save_learning_to_thermal_state()`  
**Issue:** Accumulates corrupted deltas over time, compounding the problem

## TDD Implementation Plan

### Phase 1: Unit Tests (WRITE FIRST)

#### A. New Test Files to Create
- [x] `tests/test_parameter_corruption_detection.py` ✅ **COMPLETED**
  - [x] `test_detect_corrupted_equilibrium_ratio()` (13 comprehensive tests)
  - [x] `test_detect_corrupted_total_conductance()` 
  - [x] `test_detect_corrupted_learning_confidence()`
  - [x] `test_corruption_detection_with_valid_parameters()`
  - [x] Integration tests with learning system

- [x] `tests/test_catastrophic_error_handling.py` ✅ **COMPLETED**
  - [x] `test_learning_disabled_for_errors_over_5_degrees()` (12 comprehensive tests)
  - [x] `test_learning_disabled_for_errors_over_10_degrees()`
  - [x] `test_learning_rate_zero_for_catastrophic_errors()`
  - [x] `test_normal_learning_for_small_errors()`
  - [x] Integration tests with learning system
  - [x] Logging validation tests

- [ ] `tests/test_parameter_bounds_validation.py`
  - [ ] `test_parameter_validation_before_gradient_calculation()`
  - [ ] `test_reject_gradient_calculation_with_corrupted_state()`
  - [ ] `test_bounds_enforcement_before_updates()`

- [ ] `tests/test_learning_rate_stability.py`
  - [ ] `test_conservative_learning_rates()`
  - [ ] `test_oscillation_detection_reduces_learning()`
  - [ ] `test_parameter_stability_threshold_enforcement()`

- [ ] `tests/test_automatic_calibration_reset.py`
  - [ ] `test_auto_reset_when_corruption_detected()`
  - [ ] `test_preserve_good_parameters_no_reset()`
  - [ ] `test_reset_restores_calibrated_values()`

#### B. Existing Tests to Review/Update
- [ ] Review `tests/test_thermal_equilibrium_model.py`
  - [ ] Update for new stability controls
  - [ ] Ensure existing functionality preserved
  
- [ ] Review `tests/test_adaptive_learning_*.py`
  - [ ] Update learning algorithm expectations
  - [ ] Adjust for conservative learning rates
  
- [ ] Review `tests/test_parameter_*.py`
  - [ ] Update parameter update behavior tests
  - [ ] Ensure bounds checking works

- [ ] Review `tests/test_prediction_*.py`
  - [ ] Ensure prediction accuracy maintained
  - [ ] Update error handling expectations

### Phase 2: Implementation (AFTER TESTS)

#### A. Emergency Stability Controls
**File:** `src/thermal_equilibrium_model.py`

- [ ] Add `_detect_parameter_corruption()` method
  ```python
  def _detect_parameter_corruption(self) -> bool:
      """Detect if parameters are in corrupted state."""
      if self.equilibrium_ratio < 0.3 or self.equilibrium_ratio > 0.9:
          return True
      if self.total_conductance < 0.02 or self.total_conductance > 0.3:
          return True
      if self.learning_confidence < 0.01:
          return True
      return False
  ```

- [ ] Add corruption check to `update_prediction_feedback()`
  ```python
  # STABILITY FIX: Skip learning if parameters corrupted
  if self._detect_parameter_corruption():
      logging.warning("🛑 Parameter corruption detected - learning DISABLED")
      return
  ```

- [ ] Enhance `_calculate_adaptive_learning_rate()` 
  ```python
  # Don't learn from catastrophically wrong predictions  
  if last_error > 5.0:  # Catastrophic errors
      base_rate = 0.0
      logging.warning(f"🛑 Catastrophic error ({last_error:.1f}°C) - learning DISABLED")
  ```

#### B. Learning Algorithm Fixes

- [ ] Add parameter validation to `_adapt_parameters_from_recent_errors()`
  ```python
  # Validate model state before gradient calculation
  if self._detect_parameter_corruption():
      logging.warning("🛑 Skipping parameter updates - model corrupted")
      return
  ```

- [ ] Conservative learning rate defaults in `_initialize_learning_attributes()`
  ```python
  self.learning_rate = 0.001  # Much more conservative
  self.min_learning_rate = 0.0001  # Lower minimum
  self.max_learning_rate = 0.01  # Lower maximum
  ```

- [ ] Enhanced parameter bounds checking
  ```python
  # Apply stricter bounds BEFORE gradient calculation
  if not self._validate_parameters_for_learning():
      return
  ```

#### C. Automatic Reset Mechanism

- [ ] Add `reset_to_calibrated_parameters()` method
  ```python
  def reset_to_calibrated_parameters(self):
      """Reset to calibrated parameters if corruption detected."""
      # Load baseline parameters from unified thermal state
      # Reset learning confidence to reasonable value
      # Clear corrupted prediction history
  ```

- [ ] Add corruption detection trigger
  ```python
  # In update_prediction_feedback()
  if self._detect_parameter_corruption():
      logging.error("🚨 Parameter corruption detected - auto-resetting to calibrated values")
      self.reset_to_calibrated_parameters()
  ```

#### D. Long-term Monitoring

- [ ] Add parameter drift alerts
  ```python
  def _check_parameter_drift(self):
      """Monitor for gradual parameter drift from calibrated values."""
      # Compare current vs baseline parameters
      # Alert if drift exceeds thresholds
      # Log parameter change trends
  ```

- [ ] Enhanced logging in `_save_learning_to_thermal_state()`
  ```python
  # Log parameter changes with drift analysis
  # Track cumulative drift from calibrated baseline
  # Alert when approaching corruption thresholds
  ```

### Phase 3: Integration & Validation

#### A. Integration Tests
- [ ] `tests/test_calibration_to_stability_flow.py`
  - [ ] Test: Calibration → Learning → Parameters Stay Stable
  - [ ] Test: Corruption Detection → Auto-Reset → Recovery
  - [ ] Test: Parameter Drift Prevention Over 100+ Cycles

#### B. Regression Tests  
- [ ] `tests/test_corruption_prevention_regression.py`
  - [ ] Test with original corrupted parameter values
  - [ ] Ensure corruption detection triggers immediately
  - [ ] Test gradient calculations refuse to run with bad parameters
  - [ ] Test auto-reset restores good prediction accuracy

#### C. End-to-End Validation
- [ ] Run system in shadow mode for 24 hours
- [ ] Verify parameters remain stable
- [ ] Ensure prediction accuracy maintained
- [ ] Confirm no regression in existing functionality

### Phase 4: Test Execution & Validation

#### A. TDD Test Cycle
```bash
# 1. Run tests (should fail initially)
python -m pytest tests/test_parameter_corruption_detection.py -v

# 2. Implement minimal code to pass tests
# 3. Refactor for clean code
# 4. Repeat for each component

# 5. Final validation
python -m pytest tests/ -v  # All tests must pass
```

#### B. Success Criteria Checklist
- [ ] All new unit tests pass
- [ ] All existing unit tests pass (after appropriate updates)  
- [ ] No regression in prediction accuracy (MAE < 0.5°C maintained)
- [ ] Parameters remain stable after calibration over 100+ cycles
- [ ] Corruption detection triggers correctly with test scenarios
- [ ] Auto-reset functionality works when corruption detected
- [ ] Learning rate appropriately reduced for large errors
- [ ] Gradient calculations refuse to run with corrupted parameters

## Implementation Priority Order

### Immediate (Day 1)
1. ✅ Create this TODO document
2. Write corruption detection tests
3. Implement corruption detection
4. Write catastrophic error handling tests  
5. Implement learning rate emergency controls

### Short-term (Day 2-3)
6. Write parameter bounds validation tests
7. Implement enhanced bounds checking
8. Write learning rate stability tests
9. Implement conservative learning parameters

### Medium-term (Day 4-5)
10. Write auto-reset tests
11. Implement automatic calibration reset
12. Update existing tests for compatibility
13. Integration testing

### Final Validation (Day 6-7)
14. End-to-end testing
15. Regression testing
16. Performance validation
17. Documentation updates

## Risk Mitigation

### Backup Strategy
- Keep backup of current working calibrated `thermal_state.json`
- Implement rollback mechanism if fixes cause issues
- Test all changes in shadow mode first

### Testing Safety
- Run all tests in isolated environment
- Use test-specific thermal state files
- Don't modify production parameters during testing

### Deployment Safety  
- Deploy fixes incrementally (corruption detection first)
- Monitor system behavior after each change
- Have immediate rollback plan ready

## Notes & Insights

### Key Learning from Root Cause Analysis
1. **Never learn from garbage predictions** - if MAE > 5°C, something is fundamentally wrong
2. **Validate model state before gradients** - corrupted parameters produce corrupted gradients
3. **Conservative learning is safer** - aggressive learning leads to instability
4. **Bounds checking must come BEFORE calculation** - not after

### Original Parameter Values (Corrupted)
```json
{
  "equilibrium_ratio": 0.1,
  "total_conductance": 0.266,
  "learning_confidence": 0.0
}
```

### Good Parameter Values (After Calibration)  
```json
{
  "equilibrium_ratio": 0.408,
  "total_conductance": 0.194,
  "learning_confidence": 3.0
}
```

### Files to Monitor During Implementation
- `src/thermal_equilibrium_model.py` - Main learning algorithm
- `src/unified_thermal_state.py` - Parameter persistence
- `thermal_state.json` - Runtime parameter values
- All test files in `tests/` directory

## Status Tracking

**Last Updated:** January 2, 2026 10:05 AM  
**Current Phase:** Phase 1 - Unit Test Creation  
**Next Milestone:** Complete corruption detection tests  
**Estimated Completion:** January 8, 2026  

---
*This document will be updated as progress is made on each item.*
