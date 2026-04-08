# Bugfix: Dynamic Horizon for Cooling Scenarios

## Issue Description
The system exhibited an anomaly where the outlet temperature unexpectedly jumped from 14°C to 43°C while the indoor temperature was still above the target setpoint (meaning it should be actively cooling).

## Root Cause Analysis
The root cause was traced to the dynamic optimization horizon logic in `src/model_wrapper.py`. The logic was asymmetrical, providing "Aggressive Recovery" (1.0h horizon) and "Moderate Recovery" (2.0h horizon) only for heating scenarios (when the room is too cold). 

When the room was too hot (`temp_diff < 0`), the system defaulted to the `else` block, which assigned a 4.0-hour "Stability" horizon. Because the system was looking 4 hours into the future, and the outdoor temperature was cold, the model predicted that applying 14°C water would cause the room to overcool (drop below the target) by the end of the 4-hour window. To prevent this long-term overcooling, the binary search raised the outlet temperature to ~43°C to land exactly on the target temperature in 4 hours.

This behavior prioritized long-term stability over immediate temperature reduction, contradicting the user's expectation of active cooling when the room is too warm.

## Solution
The dynamic horizon logic was updated to be symmetrical, introducing an "Aggressive Cooling" horizon:

```python
temp_diff = target_indoor - current_indoor

if temp_diff > 0.3:
    # Cold (>0.3°C gap): Focus on next 1.0 hour for aggressive heating
    optimization_horizon = 1.0
elif temp_diff > 0.0:
    # Cool (>0.0°C gap): Focus on next 2.0 hours for moderate heating
    optimization_horizon = 2.0
elif temp_diff < -0.2:
    # Hot (<-0.2°C gap): Focus on next 1.0 hour for aggressive cooling
    optimization_horizon = 1.0
else:
    # Maintenance (Within -0.2 to 0.0): Focus on 4.0h for maximum stability
    optimization_horizon = 4.0
```

This ensures that if the room is significantly above the target, the system will prioritize immediate temperature reduction (1.0h horizon), resulting in the expected 14°C outlet temperature instead of jumping to 43°C.

## Verification
Unit tests for `model_wrapper.py` and `precheck_logic.py` were run and passed successfully, confirming that the changes did not break existing functionality.
