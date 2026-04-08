# Bugfix: Pre-check Logic Short-Circuiting

## Issue Description
The system exhibited extreme control fluctuations where the applied outlet temperature spiked to 65°C and then drastically dropped to 24°C, causing the indoor temperature to climb significantly above the target.

## Root Cause
The root cause was identified in the `_calculate_required_outlet_temp` function within `src/model_wrapper.py`. The "unreachable target" pre-check evaluated equilibrium reachability but ignored the current room state. 

For example, if the room was 22.40°C (target 21.2°C), the model calculated that with max heat (65°C), the equilibrium would only be 21.08°C due to cold outdoor temps. Because 21.2°C > 21.08°C, the pre-check short-circuited and applied 65°C to try and reach the unreachable target, completely ignoring that the room was already at 22.40°C and needed to cool down. This caused the massive temperature spike.

## Fix
The pre-check logic was updated to include a guard condition based on the current heating/cooling needs. 
- If the target is below the minimum capability, it will only short-circuit to `outlet_min` if the system is NOT trying to heat.
- If the target is above the maximum capability, it will only short-circuit to `outlet_max` if the system is NOT trying to cool.

This ensures that the system does not apply maximum heating when the room is already above the target temperature, and instead proceeds to the binary search to find the optimal outlet temperature for the current trajectory.
