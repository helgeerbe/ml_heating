# Thermal Model Comprehensive Fix Plan

**Based on**: THERMAL_MODEL_DEEP_ANALYSIS.md  
**Created**: December 5, 2025  
**Status**: Implementation Ready  
**Priority**: Critical - Physics Correctness Issues

---

## Executive Summary

This plan addresses **fundamental physics errors** in the thermal equilibrium model identified in the deep analysis. The current 1.56°C MAE is achieved despite incorrect physics due to parameter compensation. This plan will fix the root causes while maintaining or improving performance.

---

## Phase 0: Project Setup (Priority: CRITICAL - FIRST)

### 0.1 Update .gitignore ✅ COMPLETED
**Issue**: State files and backups should not be tracked  
**Location**: `.gitignore`  
**Impact**: Clean repository management

**Tasks**:
- [x] Add `thermal_state.json` to .gitignore
- [x] Add `*_backup_*.py` pattern to .gitignore  
- [x] Add `*.json` state files to .gitignore (except config files)
- [x] Commit .gitignore changes

**Success Criteria**:
- No sensitive state data in repository
- Clean working directory

### 0.2 Implement Git Workflow Setup
**Issue**: Proper branching strategy needed for TDD  
**Location**: Git repository  
**Impact**: Clean development workflow

**Tasks**:
- [ ] Commit all current staged changes to `feature/heat-balance-controller`
- [ ] Add and commit thermal_state.json changes and analysis documents
- [ ] Create `fix/thermal-physics-corrections` branch from feature branch
- [ ] Set up TDD workflow for remaining phases

**Success Criteria**:
- Clean feature branch with all current work committed
- Fix branch ready for TDD implementation

---

## Phase 1: Critical Physics Fixes (Priority: CRITICAL) ✅ COMPLETED

### 1.1 Fix Equilibrium Equation Physics ✅ COMPLETED
**Issue**: Thermal time constant incorrectly affects equilibrium temperature  
**Location**: `src/thermal_equilibrium_model.py` lines 167-172  
**Impact**: Fundamental physics violation

**Tasks**:
- [x] Write failing unit tests for correct equilibrium physics
- [x] Remove `thermal_insulation_multiplier` from equilibrium calculations
- [x] Implement correct heat balance equation: `T_eq = T_outdoor + Q_in / heat_loss_coefficient`
- [x] Preserve thermal time constant usage ONLY in trajectory prediction
- [x] Update equilibrium method signature for external heat sources
- [x] Ensure all tests pass (160 passed, 1 skipped)

**Success Criteria**: 
- [x] Thermal time constant does not appear in equilibrium calculation
- [x] Energy conservation verified in tests
- [x] MAE remains ≤ 2.0°C after fix

### 1.2 Correct External Heat Source Units ✅ COMPLETED
**Issue**: Inconsistent units for heat source weights  
**Location**: `src/thermal_config.py` UNITS dictionary  
**Impact**: Physics calculations meaningless

**Tasks**:
- [x] Standardize fireplace/tv units to °C (direct temperature contribution)
- [x] Standardize PV units to °C/W (temperature rise per watt)
- [x] Update equilibrium calculations to use consistent units
- [x] Add unit validation in thermal config

**Success Criteria**:
- [x] All external heat sources have physically meaningful units
- [x] Unit conversions are explicit and documented

### 1.3 Remove Arbitrary Outdoor Coupling ✅ COMPLETED
**Issue**: Non-physical outdoor temperature coupling term  
**Location**: `src/thermal_equilibrium_model.py` line 161  
**Impact**: Violates heat transfer principles

**Tasks**:
- [x] Remove `outdoor_coupling` parameter and related calculations
- [x] Implement proper heat loss: `Q_loss = heat_loss_coefficient × (T_indoor - T_outdoor)`
- [x] Update gradient calculations to exclude outdoor_coupling
- [x] Remove outdoor_coupling from optimization parameters
- [x] TDD tests validate clean physics implementation

**Success Criteria**:
- [x] Heat loss follows proper temperature difference relationship
- [x] No arbitrary normalization around 20°C

---

## Phase 2: Implementation Quality Fixes (Priority: HIGH)

### 2.1 Fix Thermal Bridge Implementation
**Issue**: Unphysical thermal bridge calculation  
**Location**: Multiple locations in equilibrium model

**Tasks**:
- [ ] Research proper thermal bridge physics
- [ ] Implement area-based thermal bridge losses
- [ ] Remove arbitrary 20°C reference temperature
- [ ] Remove magic 0.01 multiplication factor

**Success Criteria**:
- Thermal bridge calculation follows building physics standards
- Parameters have clear physical meaning

### 2.2 Implement Physics Validation Tests
**Issue**: No tests verify physics correctness  
**Location**: `tests/` directory

**Tasks**:
- [ ] Create `test_thermal_physics.py`
- [ ] Test energy conservation at equilibrium
- [ ] Test Second Law of Thermodynamics compliance
- [ ] Test unit consistency across calculations
- [ ] Test physical bounds (indoor temp between outdoor and source)

**Success Criteria**:
- All physics tests pass
- Violations of thermodynamics caught automatically

### 2.3 Standardize Units System
**Issue**: Inconsistent units throughout codebase

**Tasks**:
- [ ] Create `src/thermal_constants.py` with unit definitions
- [ ] Define ThermalUnits class with all parameter units
- [ ] Add unit validation to parameter loading
- [ ] Update documentation with unit explanations

**Success Criteria**:
- All parameters have clearly defined units
- Unit mismatches caught at runtime

---

## Phase 3: Code Quality Improvements (Priority: MEDIUM)

### 3.1 Refactor Gradient Calculations
**Issue**: Duplicated code in _FIXED gradient methods

**Tasks**:
- [ ] Extract common finite-difference logic
- [ ] Create generic `_calculate_parameter_gradient()` method
- [ ] Reduce epsilon values (current 2.0 is too large)
- [ ] Add gradient calculation tests

**Success Criteria**:
- No duplicate gradient calculation code
- Configurable epsilon values for different parameters

### 3.2 Replace Magic Numbers with Named Constants
**Issue**: Unexplained magic numbers throughout code

**Tasks**:
- [ ] Define PhysicsConstants class in thermal_constants.py
- [ ] Replace all hardcoded values with named constants
- [ ] Add explanatory comments for each constant
- [ ] Make constants configurable where appropriate

**Success Criteria**:
- No unexplained numbers in calculations
- Constants have clear physical meaning

### 3.3 Improve Error Handling and Bounds
**Issue**: Overly restrictive sanity bounds

**Tasks**:
- [ ] Replace hard bounds with physics-based limits
- [ ] Add proper error handling for invalid inputs
- [ ] Implement graceful degradation for edge cases
- [ ] Add parameter validation

**Success Criteria**:
- Bounds based on physics principles
- Graceful handling of edge cases

---

## Phase 4: Validation and Testing (Priority: HIGH)

### 4.1 Comprehensive Physics Testing
**Tasks**:
- [ ] Test corrected equilibrium equation against known scenarios
- [ ] Validate trajectory prediction still works correctly
- [ ] Test external heat source contributions
- [ ] Verify parameter optimization still converges

**Success Criteria**:
- All physics tests pass
- Model performance ≤ 2.0°C MAE
- Parameters converge reliably

### 4.2 Regression Testing
**Tasks**:
- [ ] Run existing test suite
- [ ] Test on historical data
- [ ] Compare before/after performance
- [ ] Validate state persistence still works

**Success Criteria**:
- No regression in existing functionality
- Performance maintained or improved

### 4.3 Integration Testing
**Tasks**:
- [ ] Test with adaptive learning system
- [ ] Test with dashboard integration
- [ ] Test with InfluxDB data export
- [ ] Test with Home Assistant integration

**Success Criteria**:
- All integrations still function
- No breaking changes to APIs

---

## Phase 5: Smart Rounding and Prediction Logic Fixes (Priority: CRITICAL)

**5.1. Root Cause Analysis: Flawed Equilibrium Physics**
*   **Issue**: The `predict_equilibrium_temperature` function in `thermal_equilibrium_model.py` uses a physically incorrect formula, causing `None` returns and unpredictable behavior in the binary search.
*   **Impact**: This is the root cause of the `TypeError` in smart rounding and the model's failure to aim for the target temperature.
*   **Recommendation**: This issue will be resolved by the fixes planned in **Phase 1: Critical Physics Fixes**. No separate action is needed here, but it's critical to understand this is the source of the problem.

**5.2. Fix Smart Rounding Logic**
*   **Issue**: The smart rounding logic in `main.py` crashes with a `TypeError` when `predict_indoor_temp` returns `None`.
*   **Task**:
    *   Add a check in the `smart_rounding` logic in `main.py` to handle the case where `wrapper.predict_indoor_temp` returns `None`.
    *   If `predict_indoor_temp` returns `None`, fallback to simple rounding (`round(final_temp)`) and log a warning.
*   **Success Criteria**: The smart rounding logic no longer crashes and gracefully handles prediction failures.

**5.3. Fix Binary Search in `model_wrapper.py`**
*   **Issue**: The binary search in `_calculate_required_outlet_temp` gets stuck and doesn't converge to the correct outlet temperature.
*   **Task**:
    *   This is a direct symptom of the incorrect physics. Once the equilibrium equation is fixed in Phase 1, this binary search will function correctly.
    *   Add logging to the binary search loop to output the `outlet_mid`, `predicted_indoor`, and `error` at each iteration. This will help verify the fix.
*   **Success Criteria**: The binary search converges to the correct outlet temperature, and the model correctly aims for the target indoor temperature.

---

## Implementation Strategy

### Approach: Test-Driven Development with Git Workflow

1. **Git Workflow**:
   - Update .gitignore to exclude thermal_state.json and backup files
   - Commit all current changes to `feature/heat-balance-controller` branch
   - Create `fix/thermal-physics-corrections` branch from feature branch
   - Commit after each completed task with descriptive messages
   - Merge completed fixes back into feature branch when done

2. **Test-Driven Development Process**:
   - **Red**: Write failing unit tests FIRST for each fix
   - **Green**: Implement minimal code to make tests pass
   - **Refactor**: Clean up code while keeping tests green
   - All physics fixes must have corresponding unit tests
   - Run full test suite before each commit

3. **Validation Strategy**:
   - Unit tests validate physics correctness and prevent regression
   - Integration tests ensure system compatibility
   - Performance tests monitor MAE ≤ 2.0°C continuously
   - Automated test execution before each commit

### Risk Mitigation

1. **Regression Risk**: Comprehensive unit tests prevent breaking existing functionality
2. **Performance Risk**: Automated tests validate MAE after each change
3. **Physics Risk**: Dedicated physics validation tests catch thermodynamics violations
4. **Rollback Risk**: Git commit history provides clean rollback points for each task

---

## Timeline Estimate

| Phase | Estimated Time | Dependencies |
|-------|----------------|--------------|
| Phase 1 | 2-3 days | Critical foundation |
| Phase 2 | 1-2 days | Requires Phase 1 |
| Phase 3 | 1 day | Can parallel Phase 2 |
| Phase 4 | 1 day | Requires all phases |
| Phase 5 | 1 day | Requires Phase 1 |

**Total**: 5-8 days

---

## Success Metrics

### Critical Success Criteria
- [ ] Thermal time constant removed from equilibrium calculations
- [ ] Energy conservation verified in tests
- [ ] Units consistent across all parameters
- [ ] MAE ≤ 2.0°C maintained

### Quality Success Criteria
- [ ] No magic numbers in code
- [ ] Physics validation tests pass
- [ ] Code duplication eliminated
- [ ] Clear documentation of physics principles

### Integration Success Criteria
- [ ] All existing tests pass
- [ ] Dashboard continues to function
- [ ] State persistence preserved
- [ ] No breaking API changes

---

## Files to Modify

### Primary Files
- `src/thermal_equilibrium_model.py` (major changes)
- `src/thermal_config.py` (units update)
- `src/physics_calibration.py` (parameter updates)

### New Files
- `src/thermal_constants.py` (constants and units)
- `tests/test_thermal_physics.py` (physics validation)

---

## Next Steps

1. **Setup Git Workflow**: Update .gitignore and commit current work to feature branch
2. **Create Fix Branch**: `fix/thermal-physics-corrections` from feature branch
3. **Start TDD with Phase 1.1**: Write failing tests first, then fix equilibrium physics
4. **Commit Incrementally**: After each completed task with descriptive messages
5. **Monitor Performance**: Track MAE throughout process with automated tests
6. **Document Changes**: Update memory bank with learnings and merge back to feature branch

---

**Plan Status**: Ready for Implementation  
**Review Required**: Yes - before starting Phase 1  
**Approval Needed**: Proceed with Phase 1.1 first
