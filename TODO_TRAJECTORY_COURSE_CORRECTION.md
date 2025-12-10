# Trajectory Course Correction Implementation - TDD Plan

## Issue: Overnight Temperature Drop (Dec 10, 2025)

### Problem Statement
Indoor temperature dropped from 21.2°C to 20.2°C overnight while ML system maintained confidence at 5.0 and reported improving MAE. The target was 21.0°C, but the system consistently suggested only 25.8°C outlet temperature, which was insufficient to maintain the target.

### Root Cause Analysis

**CRITICAL FINDING**: The trajectory prediction system was **planned but never integrated** into the main control loop.

#### Evidence:

1. **Method exists but never called:**
   ```bash
   grep -rn "predict_thermal_trajectory" /opt/ml_heating/src/ --include="*.py"
   # Result: Only found at line 539 in thermal_equilibrium_model.py (definition)
   # NO calls to this method anywhere in the codebase!
   ```

2. **Config flag defined but never used:**
   ```bash
   grep -rn "TRAJECTORY_PREDICTION_ENABLED" /opt/ml_heating/src/ --include="*.py"
   # Result: Only found in config.py lines 307-308 (definition)
   # NEVER checked anywhere in the control loop!
   ```

3. **Control loop uses one-shot binary search:**
   - `main.py` line 835: `simplified_outlet_prediction()` is called
   - This uses `_calculate_required_outlet_temp()` which does binary search
   - Binary search finds equilibrium outlet temp but **never validates trajectory**
   - No feedback loop to correct course when temperature drifts from target

### Technical Gap

| Component | Status | Issue |
|-----------|--------|-------|
| `predict_thermal_trajectory()` | ✅ Implemented | Method exists in thermal_equilibrium_model.py |
| `TRAJECTORY_PREDICTION_ENABLED` | ✅ Defined | Config flag set to `true` by default |
| Trajectory validation test | ✅ Exists | validation/validate_trajectory_prediction.py |
| **Integration into control loop** | ❌ **MISSING** | Never called in main.py or model_wrapper.py |
| **Course correction logic** | ❌ **MISSING** | No adjustment when trajectory shows target won't be reached |

---

## TDD Implementation Plan

### Phase 1: Unit Tests for Trajectory-Based Course Correction

#### Test 1.1: Trajectory Verification After Binary Search
```python
def test_trajectory_verification_triggers_correction():
    """
    When binary search finds outlet temp that equilibrium model says is correct,
    but trajectory prediction shows target won't be reached in time,
    the system should increase outlet temperature.
    """
    # Given: Binary search returns 25.8°C as optimal
    # And: Trajectory prediction shows temperature will drop to 20.2°C in 4 hours
    # When: Trajectory verification is performed
    # Then: Outlet temperature should be increased to reach target
```

#### Test 1.2: Course Correction When Drifting Away From Target
```python
def test_course_correction_when_drifting():
    """
    When current temperature is below target and trajectory shows it will
    continue to drop, course correction should boost outlet temperature.
    """
    # Given: current_indoor=20.5°C, target=21.0°C
    # And: trajectory predicts 20.2°C in 4 hours with current outlet
    # When: course correction is applied
    # Then: outlet temperature should increase proportionally to error
```

#### Test 1.3: No Over-Correction When Trajectory Is Good
```python
def test_no_overcorrection_when_trajectory_good():
    """
    When trajectory prediction shows target will be reached,
    no additional correction should be applied.
    """
    # Given: Binary search returns 35°C as optimal
    # And: Trajectory shows temperature will reach 21°C in 2 hours
    # When: Trajectory verification is performed
    # Then: Original outlet temperature should be maintained
```

#### Test 1.4: Trajectory Correction Respects Temperature Bounds
```python
def test_trajectory_correction_respects_bounds():
    """
    Course correction should not exceed CLAMP_MAX_ABS even when
    trajectory shows aggressive correction is needed.
    """
    # Given: Large temperature gap requiring high outlet temp
    # When: Course correction calculates required outlet
    # Then: Result should be clamped to config.CLAMP_MAX_ABS
```

### Phase 2: Integration Tests

#### Test 2.1: Full Control Loop With Trajectory Verification
```python
def test_control_loop_uses_trajectory_verification():
    """
    The main control loop should verify trajectory after binary search
    and apply course correction when needed.
    """
    # Given: TRAJECTORY_PREDICTION_ENABLED=true
    # When: simplified_outlet_prediction() is called
    # Then: trajectory verification should be performed
    # And: outlet temp should be adjusted if trajectory shows target unreachable
```

#### Test 2.2: Overnight Scenario Simulation
```python
def test_overnight_temperature_maintenance():
    """
    Simulate overnight conditions where temperature naturally drops.
    The system should proactively increase outlet temperature to maintain target.
    """
    # Given: Overnight conditions (outdoor=9°C, no PV, no internal gains)
    # And: target=21°C, current=21.2°C
    # When: Multiple control cycles are simulated
    # Then: Outlet temperature should increase to prevent temperature drop
```

### Phase 3: Implementation Tasks

#### Task 3.1: Create Trajectory Verification Method
- [ ] Add `verify_trajectory_and_correct()` method to `EnhancedModelWrapper`
- [ ] Use `predict_thermal_trajectory()` to validate binary search result
- [ ] Return corrected outlet temperature if trajectory shows target unreachable

#### Task 3.2: Integrate Into Binary Search
- [ ] Modify `_calculate_required_outlet_temp()` to include trajectory verification
- [ ] Check `config.TRAJECTORY_PREDICTION_ENABLED` before verification
- [ ] Log trajectory verification results for debugging

#### Task 3.3: Add Course Correction Algorithm
- [ ] Calculate correction factor based on trajectory deviation
- [ ] Implement proportional correction: `correction = 1.0 + (error * factor)`
- [ ] Add bounds checking for correction magnitude

#### Task 3.4: Update Logging
- [ ] Add trajectory verification logging
- [ ] Log when course correction is applied
- [ ] Log trajectory prediction metrics to InfluxDB

---

## Implementation Details

### Proposed Code Changes

#### File: `src/model_wrapper.py`

Add trajectory verification to `_calculate_required_outlet_temp()`:

```python
def _calculate_required_outlet_temp(self, current_indoor: float, target_indoor: float, 
                                  outdoor_temp: float, thermal_features: Dict) -> float:
    # ... existing binary search code ...
    
    # NEW: Trajectory verification and course correction
    if config.TRAJECTORY_PREDICTION_ENABLED:
        final_outlet = self._verify_trajectory_and_correct(
            outlet_temp=final_outlet,
            current_indoor=current_indoor,
            target_indoor=target_indoor,
            outdoor_temp=outdoor_temp,
            thermal_features=thermal_features
        )
    
    return final_outlet


def _verify_trajectory_and_correct(self, outlet_temp: float, current_indoor: float,
                                   target_indoor: float, outdoor_temp: float,
                                   thermal_features: Dict) -> float:
    """
    Verify that the calculated outlet temperature will actually reach the target
    using trajectory prediction, and apply course correction if needed.
    """
    # Get trajectory prediction
    trajectory = self.thermal_model.predict_thermal_trajectory(
        current_indoor=current_indoor,
        target_indoor=target_indoor,
        outlet_temp=outlet_temp,
        outdoor_temp=outdoor_temp,
        time_horizon_hours=config.TRAJECTORY_STEPS,
        pv_power=thermal_features.get('pv_power', 0.0)
    )
    
    # Check if trajectory reaches target
    if trajectory['reaches_target_at'] is None:
        # Target won't be reached - apply course correction
        final_temp_predicted = trajectory['equilibrium_temp']
        temp_error = target_indoor - final_temp_predicted
        
        if temp_error > 0.3:  # Need more heating
            # Proportional correction
            correction_factor = 1.0 + (temp_error * 0.15)  # 15% per degree error
            corrected_outlet = outlet_temp * correction_factor
            
            # Apply bounds
            corrected_outlet = min(corrected_outlet, config.CLAMP_MAX_ABS)
            
            logging.info(f"🎯 Trajectory correction: {outlet_temp:.1f}°C → {corrected_outlet:.1f}°C "
                        f"(predicted {final_temp_predicted:.1f}°C, target {target_indoor:.1f}°C)")
            
            return corrected_outlet
    
    return outlet_temp
```

---

## Test File Location

Create new test file: `tests/test_trajectory_course_correction.py`

---

## Acceptance Criteria

1. ✅ `predict_thermal_trajectory()` is called in the main control loop
2. ✅ `TRAJECTORY_PREDICTION_ENABLED` config flag is checked and respected
3. ✅ Course correction is applied when trajectory shows target unreachable
4. ✅ Overnight scenario maintains temperature within ±0.3°C of target
5. ✅ All unit tests pass
6. ✅ No regression in existing tests (294+ tests)

---

## Timeline

| Phase | Task | Estimated Time |
|-------|------|----------------|
| 1 | Write unit tests | 30 min |
| 2 | Implement trajectory verification | 30 min |
| 3 | Integration and testing | 30 min |
| 4 | Validation with production scenarios | 30 min |

**Total: ~2 hours**

---

## References

- `src/thermal_equilibrium_model.py:539` - `predict_thermal_trajectory()` definition
- `src/config.py:307-308` - `TRAJECTORY_PREDICTION_ENABLED` definition
- `validation/validate_trajectory_prediction.py` - Existing trajectory validation tests
- Overnight logs (Dec 9-10, 2025) showing temperature drop pattern

---

**Created**: December 10, 2025
**Status**: Ready for TDD implementation
