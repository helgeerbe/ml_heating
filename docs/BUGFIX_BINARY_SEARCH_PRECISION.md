# Bugfix: Binary Search Precision and Grace Period Clamping

## Issue Description
The system experienced a degradation in model performance (accuracy dropped to 45.5%) and anomalous temperature fluctuations, specifically spiking from 25°C to 50°C and abruptly returning to 25°C.

## Root Cause Analysis
The root cause was a combination of two issues:
1. **The 45.0°C Midpoint Trap**: The binary search for the optimal outlet temperature started at 45.0°C (the midpoint of the 25°C - 65°C bounds). Due to a loose tolerance (`0.1°C`) and the high thermal mass of the system, the predicted error over a 2-hour horizon was within the tolerance, causing the search to prematurely converge on the first iteration.
2. **Grace Period Amplification**: The grace period logic locked onto this artificially high target without properly clamping the rate of change from the previous baseline, causing the system to spike to the high target.

## Resolution
1. **Tightened Binary Search Tolerance**: The tolerance in `src/model_wrapper.py` was tightened from `0.1` to `0.01` to force proper convergence and prevent premature exit on the first iteration.
2. **Grace Period Clamping**: The grace period logic in `src/heating_controller.py` was updated to use `state.last_final_temp` as the baseline for `apply_gradual_control` instead of `actual_outlet_temp_start`. This ensures that the rate of change is properly clamped relative to the previous setpoint, preventing sudden spikes.

## Testing
- Added `test_binary_search_precision_midpoint_trap` to `tests/unit/test_model_wrapper.py` to verify the binary search doesn't prematurely converge on the 45.0°C midpoint.
- Verified existing grace period clamping tests pass with the new logic.
- Full test suite passes successfully.
