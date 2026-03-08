# Morning Drop Fix Plan

## Problem Description
The heating system exhibits a "morning drop" where the target outlet temperature decreases too early as the sun rises, potentially leading to under-heating before the solar gain actually warms the house.

## Root Cause Analysis
Investigation into `src/prediction_context.py` revealed that the `UnifiedPredictionContext` calculates the "effective" outdoor temperature for the current control cycle by blending the current sensor reading with the 1-hour forecast.

For a standard 30-minute cycle (`cycle_hours = 0.5`), the current code uses a hardcoded weight of **0.5**:

```python
if cycle_hours <= 0.5:
    # ...
    weight = 0.5  # Increased weight to capture rapid solar rise
    avg_outdoor = outdoor_temp * (1 - weight) + forecast_1h_outdoor * weight
```

This calculation effectively sets the "average" temperature to the value at `t=30min` (assuming linear change). However, for a 30-minute cycle, the system should be reacting to the average conditions *during* that cycle, which is best approximated by the temperature at the midpoint, `t=15min`.

Using a weight of 0.5 means the system "jumps the gun" by 15 minutes, anticipating warmth that hasn't arrived yet. Combined with the natural thermal lag of the building (which is not fully captured by the outdoor temp sensor), this causes the premature drop in heating output.

## Proposed Solution

Adjust the weighting logic to strictly follow linear interpolation for the cycle midpoint.

**Current Logic:**
- Cycle <= 30m: Weight = 0.5 (Fixed)
- Cycle 30-60m: Weight = `cycle_hours / 2.0`

**New Logic:**
- Unified formula: `weight = cycle_hours / 2.0`

For a 30-minute cycle (`0.5` hours):
- `weight = 0.5 / 2.0 = 0.25`

This means the effective outdoor temperature will be:
`75% Current + 25% Forecast_1h`

This represents the temperature at `t=15min`, which is the correct average for the upcoming 30-minute interval.

## Implementation Steps

1.  **Reproduction Script:** Create `validation/reproduce_morning_drop_context.py` to simulate a morning temperature rise (e.g., 0°C -> 4°C) and verify that the current code calculates an aggressively high effective temperature (2.0°C instead of ~1.0°C).
2.  **Code Fix:** Modify `src/prediction_context.py` to use the dynamic weighting formula for short cycles.
3.  **Verification:** Run the reproduction script to confirm the effective temperature is now physically realistic.
4.  **Regression Testing:** Run existing unit tests to ensure no side effects.

## Expected Impact
-   **Morning:** The heating target will drop more gradually, maintaining comfort as the sun rises.
-   **Evening:** The heating target will rise more gradually as the sun sets (less "panic" heating).
-   **General:** Better alignment between the model's "effective" environment and the physical reality of the control cycle.
