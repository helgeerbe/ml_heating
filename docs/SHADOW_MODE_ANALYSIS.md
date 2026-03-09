# Shadow Mode Behavior Analysis

## Overview
This document explains the behavior of "Shadow Mode" in the ML Heating system, specifically addressing two reported issues:
1.  **Prediction Jump:** A sudden spike in predicted outlet temperature (e.g., 25°C -> 40.6°C) when switching from Active to Shadow Mode.
2.  **Missing Target Calculation:** Logs showing the *current* indoor temperature as the *predicted* indoor temperature.

## 1. Prediction Jump (25°C -> 40.6°C)

### Cause
The "jump" is caused by Shadow Mode intentionally skipping the **Grace Period** logic.

*   **Active Mode:** When a blocking event (like DHW heating) ends, the system enters a "Grace Period" where it holds the outlet temperature at a safe, lower level (e.g., 25°C) to prevent overheating or rapid cycling.
*   **Shadow Mode:** The system is designed to observe and calculate the *raw physics-based demand* without interfering with the actual heating system. Therefore, it skips the Grace Period safety checks and immediately calculates the full heating demand required by the physics model (e.g., 40.6°C).

### Conclusion
This behavior is **intentional**. Shadow Mode shows you what the raw physics model *wants* to do, unconstrained by safety clamps like the Grace Period. The jump confirms that the physics model sees a high heating demand, which the Active Mode would normally suppress for safety.

## 2. Missing Target Calculation

### Cause
The `predicted_indoor` temperature was missing from the metadata returned by the model wrapper.
*   In **Active Mode**, this was masked because the `smart_rounding` logic (which runs only in Active Mode) recalculates the predicted indoor temperature.
*   In **Shadow Mode**, `smart_rounding` is skipped. The system fell back to using the *current* indoor temperature as the *predicted* value because the metadata was missing.

### Fix
The `EnhancedModelWrapper.calculate_optimal_outlet_temp` method has been updated to explicitly calculate and include `predicted_indoor` in the returned metadata.

```python
# src/model_wrapper.py

# Calculate predicted indoor temp for the optimal outlet temp
predicted_indoor = self.predict_indoor_temp(
    outlet_temp=optimal_outlet_temp,
    # ... other params ...
)

prediction_metadata = {
    "predicted_indoor": predicted_indoor,  # <--- Added this
    # ...
}
```

## Verification
*   **Prediction Jump:** Verified by `validation/verify_shadow_mode_behavior.py`. Confirmed that Shadow Mode skips Grace Period logic.
*   **Missing Target:** Verified by `validation/reproduce_missing_target.py` (before fix) and confirmed fixed after applying the patch.

## Summary
*   **Prediction Jump:** Normal behavior for Shadow Mode (raw physics output).
*   **Missing Target:** Fixed by updating `model_wrapper.py`.
