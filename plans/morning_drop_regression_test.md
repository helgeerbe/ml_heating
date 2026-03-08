# Regression Test Plan

## Objective
Add a regression test for the "morning drop" issue where the system over-anticipates rising outdoor temperatures due to incorrect forecast blending weights.

## Context
The "morning drop" bug caused the system to use a hardcoded 0.5 weight for the 1-hour forecast when the cycle interval was less than 1 hour. For a 30-minute cycle, the correct weight should be based on the midpoint (15 minutes), which is 0.25. This caused the system to "see" a warmer outdoor temperature than reality during sunrise, leading to underheating.

## Changes
1.  **File:** `tests/unit/test_prediction_context.py`
2.  **Action:** Add a new test method `test_morning_drop_prevention`.
3.  **Logic:**
    *   Mock `CYCLE_INTERVAL_MINUTES` to 30.
    *   Set current outdoor temp to 6.0°C.
    *   Set 1h forecast to 10.0°C (rising temp).
    *   Calculate expected effective temp:
        *   Cycle = 30 mins = 0.5h
        *   Midpoint = 15 mins = 0.25h
        *   Weight = 0.25 / 1.0 = 0.25
        *   Expected = 6.0 * 0.75 + 10.0 * 0.25 = 4.5 + 2.5 = 7.0°C
    *   Assert that the calculated `avg_outdoor` matches 7.0°C.
    *   (The buggy behavior would yield 8.0°C).

## Verification
Run `pytest tests/unit/test_prediction_context.py` to ensure the new test passes.
