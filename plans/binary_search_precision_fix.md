# Binary Search Precision and Grace Period Clamping Fix Plan

## Objective
Resolve the anomalous temperature fluctuations (spiking to 45.0°C) caused by premature binary search convergence and amplified by the grace period logic.

## Root Cause
1. **The 45.0°C Midpoint Trap**: The binary search in `model_wrapper.py` starts at the midpoint of 25.0°C and 65.0°C (which is 45.0°C).
2. **Premature Convergence**: The search exits early if the predicted error is within a `0.1°C` tolerance. Due to high thermal mass, the 45.0°C guess often falls within this tolerance immediately, causing the search to converge in 1 iteration.
3. **Grace Period Amplification**: The `heating_controller.py` grace period logic locks onto this artificially high target without sufficient rate-of-change clamping from the pre-blocking baseline.

## Test-Driven Development (TDD) Steps

### Phase 1: Binary Search Precision
- [ ] **Write failing test**: Create or update a unit test in `tests/unit/test_model_wrapper.py` that simulates the conditions where the binary search prematurely converges on the first iteration (45.0°C).
- [ ] **Implement fix**: Update `src/model_wrapper.py` (`_calculate_required_outlet_temp`) to tighten the binary search precision. Reduce the `tolerance` from `0.1` to `0.01` (or remove the early exit condition and rely on range collapse `range_size < 0.05`).
- [ ] **Verify test passes**: Ensure the unit test now correctly iterates and finds the true optimal temperature instead of stopping at 45.0°C.

### Phase 2: Grace Period Target Clamping
- [ ] **Write failing test**: Create or update a unit test in `tests/unit/test_heating_controller.py` (or `test_overshoot_logic.py`) to simulate a grace period starting with a drastically different target temperature, verifying that it should be clamped.
- [ ] **Implement fix**: Update `src/heating_controller.py` (`_execute_grace_period`) to enforce a strict rate-of-change limit (e.g., max ±2.0°C deviation from the pre-blocking baseline) for the initial grace period target.
- [ ] **Verify test passes**: Ensure the unit test confirms the target is properly clamped.

### Phase 3: Validation and Documentation
- [ ] **Run full test suite**: Execute `pytest` to ensure all existing and new tests pass successfully.
- [ ] **Update documentation**: Document the fix in a new or existing bugfix markdown file (e.g., `docs/BUGFIX_BINARY_SEARCH_PRECISION.md`) explaining the midpoint trap and the implemented solutions.
- [ ] **Update CHANGELOG**: Add an entry for the bug fix.