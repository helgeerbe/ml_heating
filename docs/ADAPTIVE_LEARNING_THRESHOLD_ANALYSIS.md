# Adaptive Learning Threshold Analysis: 0.5°C to 0.2°C

## Executive Summary
The adaptive learning error threshold has been lowered from **0.5°C** to **0.2°C**. This change is designed to increase the model's sensitivity to minor thermal drifts, enabling tighter temperature control and faster convergence to the optimal thermal parameters. This document analyzes the implications, risks, and mitigation strategies associated with this change.

## 1. Motivation
*   **Precision Control**: The previous threshold of 0.5°C allowed for temperature drifts of up to half a degree before the adaptive learning mechanism would trigger a "boost" in learning rate. This resulted in a "lazy" model that was slow to correct minor but persistent offsets.
*   **High-Performance Target**: Modern heat pumps and well-insulated homes can achieve stability within ±0.1°C. A 0.5°C threshold is too coarse for this goal.
*   **Faster Convergence**: By reacting to smaller errors, the model can fine-tune `heat_loss_coefficient` and `thermal_time_constant` more continuously, rather than waiting for a large deviation.

## 2. Technical Changes

### 2.1. Threshold Adjustments (`src/thermal_constants.py`)
*   `ERROR_THRESHOLD_MEDIUM`: Lowered from **0.5°C** to **0.2°C**.
    *   *Effect*: Errors > 0.2°C now trigger `ERROR_BOOST_FACTOR_LOW` (1.5x learning rate).
*   `ERROR_THRESHOLD_LOW`: Set to **0.1°C**.
    *   *Effect*: Defines the new "target" precision. Errors below this are considered negligible noise.
*   `ERROR_THRESHOLD_CONFIDENCE`: Set to **0.2°C**.
    *   *Effect*: Predictions within ±0.2°C now boost the model's "confidence" score, whereas previously this might have required a larger deviation to affect confidence negatively or positively.

### 2.2. Metric Re-calibration (`src/prediction_metrics.py`)
The accuracy classification standards have been tightened:
*   **Excellent**: < 0.1°C (Previously < 0.2°C)
*   **Very Good**: < 0.2°C (Previously < 0.5°C)
*   **Good**: < 0.5°C (Previously < 1.0°C)

## 3. Implications & Risks

### 3.1. Risk: Chasing Sensor Noise
*   **Issue**: If the temperature sensors have a noise floor > 0.1°C, the model might interpret random noise as "error" and constantly adjust parameters.
*   **Analysis**:
    *   `src/sensor_buffer.py` implements a 15-minute rolling average for indoor temperature.
    *   Typical DS18B20 or similar digital sensors have a resolution of 0.0625°C but accuracy of ±0.5°C. However, *precision* (repeatability) is usually high.
    *   **Mitigation**: The 15-minute smoothing significantly reduces high-frequency noise. The `ERROR_THRESHOLD_LOW` of 0.1°C acts as a deadband.

### 3.2. Risk: Parameter Oscillation
*   **Issue**: Aggressive learning at small errors could cause parameters (like `heat_loss_coefficient`) to oscillate around the true value.
*   **Analysis**:
    *   `src/thermal_equilibrium_model.py` includes oscillation detection (checking standard deviation of recent parameters).
    *   If oscillation is detected (`heat_loss_coefficient_std > 0.02`), the learning rate is slashed by 99%.
    *   **Mitigation**: Existing stability mechanisms are robust enough to handle increased sensitivity.

### 3.3. Impact on Control Logic
*   **Benefit**: The `HeatingController` relies on the model's `predict_thermal_trajectory`. A more accurate model (tuned to ±0.2°C) allows the controller to make better decisions about when to start/stop heating to hit the target exactly.
*   **Behavior**: We expect to see more frequent, smaller adjustments to the learning parameters, leading to a more stable `thermal_time_constant` over time.

## 4. Validation Plan

### 4.1. Monitoring
*   **Metric**: Monitor `mae_improvement_percentage` in `src/prediction_metrics.py`.
*   **Success Criteria**:
    *   MAE should trend towards < 0.2°C.
    *   Parameter oscillation flags in logs should remain low.
    *   "Confidence" score should remain high (> 0.8).

### 4.2. Rollback Trigger
If the following are observed, revert `ERROR_THRESHOLD_MEDIUM` to 0.5°C:
1.  **Instability**: Indoor temperature oscillates by > 0.5°C.
2.  **Parameter Drift**: `heat_loss_coefficient` changes by > 10% per day without weather changes.
3.  **Low Confidence**: Model confidence drops below 0.3 and stays there.

## 5. Conclusion
Lowering the threshold to 0.2°C is a necessary step to achieve high-precision thermal control. The risks of noise amplification are mitigated by sensor smoothing and existing stability checks. This change moves the system from "coarse" regulation to "fine" tuning.
