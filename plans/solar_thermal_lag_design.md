# Solar Thermal Lag Design

## 1. Problem Statement
The current thermal model assumes instantaneous heat transfer from PV/Solar sources to the indoor air. This causes a "morning drop" regression where the model predicts a temperature rise as soon as the sun comes up, leading to premature heating cutoff, while the actual house temperature lags behind due to thermal inertia.

## 2. Solution Overview
Implement a "Solar Thermal Lag" feature that buffers PV power inputs using a rolling average or time-delay mechanism. This aligns the model's "effective" solar input with the physical reality of the house.

## 3. Detailed Design

### 3.1 Configuration
*   **File:** `src/thermal_config.py`
*   **New Parameter:** `solar_lag_minutes`
*   **Default:** `45.0` minutes (based on typical building physics)
*   **Bounds:** `0.0` to `180.0` minutes

### 3.2 Data Pipeline
*   **InfluxDB (`src/influx_service.py`):**
    *   Need to fetch historical PV data for the lag window (e.g., last 3 hours).
    *   Update `get_latest_sensor_data` or similar to include a history trace for PV.
*   **Feature Builder (`src/physics_features.py`):**
    *   Pass this historical PV trace into the feature dictionary.
    *   Key: `pv_power_history` (list of `(timestamp, value)` tuples).

### 3.3 Model Logic (`src/thermal_equilibrium_model.py`)

#### A. Effective Solar Calculation
Create a helper method `_calculate_effective_solar(current_pv, pv_history, lag_minutes)`:
*   If `lag_minutes` is small (< 5), return `current_pv`.
*   Otherwise, calculate a **weighted rolling average** of the PV power over the window `[now - lag_minutes, now]`.
*   Alternatively, a simple "delayed" value (value at `now - lag_minutes`) could be used, but a rolling average is more physically realistic for thermal mass (smoothing effect).
*   **Decision:** Use **Rolling Average** over the lag window. It simulates the "charging" of the thermal mass.

#### B. Equilibrium Prediction
Update `predict_equilibrium_temperature`:
*   Accept `pv_history` as an optional argument.
*   Calculate `effective_pv = self._calculate_effective_solar(...)`.
*   Use `effective_pv` in the heat balance equation instead of raw `pv_power`.

#### C. Trajectory Prediction
Update `predict_thermal_trajectory`:
*   This is trickier because we need to simulate the lag into the future.
*   We need a buffer that combines `pv_history` (past) and `pv_forecast` (future).
*   At each simulation step `t`:
    *   The "effective" PV is the average of the buffer from `t - lag` to `t`.
    *   This ensures the forecast also respects the lag (e.g., peak heat hits 45 mins after peak sun).

### 3.4 Learning Strategy

#### A. Calibration
*   In `physics_calibration.py`, add `solar_lag_minutes` to the optimization vector.
*   The optimizer will adjust the lag to minimize error during morning ramp-ups.

#### B. Online Adaptation
*   In `thermal_equilibrium_model.py`, add a gradient rule for `solar_lag_minutes`.
*   **Rule:**
    *   If `Error (Actual - Predicted) < 0` (Over-prediction) AND `PV > 0` AND `Morning`: **Increase Lag**.
    *   If `Error > 0` (Under-prediction) AND `PV > 0`: **Decrease Lag**.

## 4. Implementation Plan

1.  **Config:** Add `solar_lag_minutes` to `ThermalParameterConfig`.
2.  **Model:** Implement `_calculate_effective_solar` and update `predict_equilibrium_temperature`.
3.  **Features:** Update `physics_features.py` to pass dummy history (initially) or real history.
4.  **Trajectory:** Update `predict_thermal_trajectory` to handle the rolling buffer.
5.  **Tests:** Add unit tests for the lag logic.
