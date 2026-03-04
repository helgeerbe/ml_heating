# Bug Fix: State Poisoning During Grace Periods

## Issue Description
Users reported sudden drops in the heating target temperature (e.g., from 41°C to 25°C) following Domestic Hot Water (DHW) or Defrost cycles. These drops persisted into the next heating cycle, causing the house to cool down significantly.

## Root Cause Analysis
The issue was identified as a **"State Poisoning" bug** in the control logic within `src/main.py`.

1.  **Trigger:** When the heat pump switches to DHW or Defrost, the heating circuit is idle. The *actual* outlet temperature naturally drops (e.g., to ~25°C).
2.  **Mechanism:** Upon returning to heating mode, the system enters a "Grace Period" to allow temperatures to stabilize.
3.  **The Bug:** During this Grace Period, the system was overwriting the persistent `last_final_temp` state variable with the *current actual* outlet temperature (25°C).
4.  **Consequence:**
    *   The system saved 25°C as the "last valid target".
    *   In the subsequent cycle, this corrupted value was used as the baseline or reported as the `ml_calculated_temp`.
    *   This also corrupted the learning dataset, teaching the ML model that 25°C was sufficient to maintain indoor temperature, leading to long-term degradation of prediction accuracy (MAE/RMSE).

## Verification
*   **Physics Validation:** A reproduction script (`validation/reproduce_0304_drop.py`) confirmed that the physics model correctly calculated a required target of **43.0°C** for the reported conditions, proving the 25.0°C value in the logs was a control artifact, not a calculation error.
*   **Logic Verification:** Code analysis confirmed that `ml_calculated_temp` in the logs is populated directly from the stored `last_final_temp`, explaining the discrepancy.

## Resolution
The fix was implemented in `src/main.py`:
*   **Logic Change:** During Grace Periods, the system now explicitly **preserves the previous valid `last_final_temp`** instead of overwriting it with the current actual temperature.
*   **Learning Protection:** These cycles are marked as `grace_period_passthrough` to prevent the ML model from learning from these non-representative states.

## Impact
*   **Immediate:** Prevents sudden target drops after DHW/Defrost cycles.
*   **Long-term:** Improves ML model accuracy by preventing "poisoned" low-temperature data points from entering the training set.
