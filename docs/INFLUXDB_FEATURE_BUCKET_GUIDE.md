# InfluxDB Data Dictionary: `ml_heating_feature` Bucket

This document serves as a comprehensive technical guide for the `ml_heating_feature` InfluxDB bucket. It details the exported measurements used for the machine learning-based heating control system, providing semantic descriptions, observability significance, and ML feature context for DevOps engineers and Data Scientists.

## Overview

The `ml_heating_feature` bucket is the central repository for:
1.  **Generated ML Features:** Synthetic metrics derived from raw sensor data.
2.  **Model Performance Metrics:** Real-time tracking of prediction accuracy (MAE, RMSE).
3.  **System State:** Internal states of the thermal equilibrium model and adaptive learning engine.
4.  **Benchmarking:** Comparisons between ML control and legacy heat curve logic.

---

## Measurement: `ml_prediction_metrics`

Tracks the accuracy and performance of the core temperature prediction model over various time windows.

| Field Key | Data Type | Semantic Description | Observability Significance | ML Feature Context |
| :--- | :--- | :--- | :--- | :--- |
| `mae_1h` | Float | Mean Absolute Error of predictions over the last 1 hour. | Immediate indicator of model drift or sudden environmental changes. | Target variable for meta-learning (error prediction). |
| `rmse_1h` | Float | Root Mean Square Error over the last 1 hour. | Penalizes larger errors more heavily than MAE; high RMSE vs MAE indicates outliers. | Loss function metric. |
| `mae_6h` | Float | Mean Absolute Error over the last 6 hours. | Smoothed accuracy metric, less sensitive to transient noise. | Trend analysis for model stability. |
| `mae_24h` | Float | Mean Absolute Error over the last 24 hours. | Daily performance baseline. | Long-term model health indicator. |
| `accuracy_excellent_pct` | Float | Percentage of predictions within ±0.1°C of actual. | "Bullseye" metric; high values indicate highly tuned physics parameters. | Success metric for reinforcement learning. |
| `accuracy_acceptable_pct` | Float | Percentage of predictions within ±1.0°C of actual. | Minimum viability metric; drops below 95% indicate system failure or sensor fault. | System health check. |
| `mae_improvement_pct` | Float | Percentage improvement in MAE compared to the previous window. | Positive values indicate the adaptive learning is successfully reducing error. | Reward signal for adaptive learning agents. |
| `is_improving` | Boolean | True if the error trend is decreasing. | Quick boolean check for dashboard status lights (Green/Red). | State indicator for learning rate adjustment. |
| `total_predictions` | Integer | Total number of predictions made in the current window. | Verifies the inference engine is running at the expected frequency. | Data density validation. |

---

## Measurement: `ml_thermal_parameters`

Logs the internal physics parameters of the `ThermalEquilibriumModel`. These parameters are dynamically adjusted by the adaptive learning engine.

| Field Key | Data Type | Semantic Description | Observability Significance | ML Feature Context |
| :--- | :--- | :--- | :--- | :--- |
| `outlet_effectiveness` | Float | Efficiency factor of the heat distribution system (radiators/floor). | Drops may indicate hydraulic balancing issues or air in the system. | Core physics parameter (coefficient) in the heat equation. |
| `heat_loss_coefficient` | Float | Rate of heat loss from the building envelope (W/K). | Increases indicate open windows, insulation failure, or extreme wind. | Core physics parameter defining building thermal retention. |
| `thermal_time_constant` | Float | System inertia; time required to reach 63.2% of a temperature change. | Changes suggest alterations in thermal mass (e.g., furniture changes, renovation). | Lag parameter determining system responsiveness. |
| `learning_confidence` | Float | Statistical confidence (0.0-1.0) in the current parameter set. | Low confidence triggers "safe mode" or conservative control strategies. | Weighting factor for parameter updates. |
| `current_learning_rate` | Float | The step size used for parameter updates. | High values indicate the model is aggressively searching for a new optimum. | Hyperparameter for the gradient descent optimizer. |
| `outlet_effectiveness_correction_pct` | Float | Percentage deviation of the current effectiveness from the baseline. | Large deviations (>20%) suggest the baseline physics model is invalid. | Anomaly detection feature. |
| `parameter_updates_24h` | Integer | Number of times parameters were updated in the last 24 hours. | Frequent updates in stable weather suggest "chasing noise" (overfitting). | Stability metric. |

---

## Measurement: `ml_learning_phase`

Tracks the state of the Hybrid Learning Strategy, which switches between different learning modes based on data quality and system stability.

| Field Key | Data Type | Semantic Description | Observability Significance | ML Feature Context |
| :--- | :--- | :--- | :--- | :--- |
| `current_learning_phase` | String | Current mode: `high_confidence`, `low_confidence`, or `skip`. | Explains *why* the system is (or isn't) learning. | Categorical state variable. |
| `stability_score` | Float | Composite score (0-1) representing signal-to-noise ratio and steady state. | Low scores prevent learning during chaotic periods (e.g., rapid valve cycling). | Gating mechanism for training data ingestion. |
| `learning_weight_applied` | Float | The actual weight (0.0-1.0) applied to the most recent update. | Zero values confirm that "skip" logic is functioning correctly. | Multiplier for gradient application. |
| `stable_period_duration_min` | Integer | Duration of the current stable operation window in minutes. | Long stable periods are ideal for calibration; short ones are useless. | Data quality filter. |
| `high_confidence_updates_24h` | Integer | Count of updates made in `high_confidence` mode. | Primary KPI for the adaptive learning system's effectiveness. | Cumulative learning opportunity metric. |
| `false_learning_prevention_pct` | Float | Percentage of potential updates rejected due to safety checks. | High values indicate the safety filters are actively protecting the model. | Safety system performance metric. |

---

## Measurement: `ml_trajectory_prediction`

Monitors the system's ability to forecast future temperature trajectories (multi-step prediction).

| Field Key | Data Type | Semantic Description | Observability Significance | ML Feature Context |
| :--- | :--- | :--- | :--- | :--- |
| `trajectory_mae_4h` | Float | Mean Absolute Error of the 4-hour ahead forecast. | Critical for Model Predictive Control (MPC) horizon planning. | Long-term planning accuracy metric. |
| `overshoot_predicted` | Boolean | True if the model predicts temperature will exceed target + tolerance. | Early warning signal for the controller to cut heat *before* overheating occurs. | Binary trigger for "brake" logic. |
| `overshoot_prevented_24h` | Integer | Count of times the system successfully intervened to stop a predicted overshoot. | Quantifies comfort improvement (avoiding "too hot" complaints). | Success metric for preemptive control. |
| `convergence_time_avg_min` | Float | Average time for the prediction to converge to the target temperature. | Measures system agility; lower is generally better (faster response). | System response time characteristic. |
| `forecast_integration_quality` | Float | Score (0-1) indicating how well weather/PV forecasts are aiding prediction. | Low scores suggest external forecast data is inaccurate or irrelevant. | Feature selection metric for exogenous variables. |

---

## Measurement: `ml_heating_shadow_benchmark`

Used for "Shadow Mode" analysis, running the ML model in parallel with the legacy Heat Curve controller to prove value without taking control.

| Field Key | Data Type | Semantic Description | Observability Significance | ML Feature Context |
| :--- | :--- | :--- | :--- | :--- |
| `ml_outlet_prediction` | Float | The outlet temperature the ML model *would* have requested. | Comparison baseline against actuals. | Counterfactual prediction. |
| `heat_curve_outlet_actual` | Float | The actual outlet temperature requested by the legacy controller. | The "control" in the A/B test. | Ground truth for legacy behavior. |
| `efficiency_advantage` | Float | Difference in degrees (°C) between Heat Curve and ML (Positive = ML is cooler/more efficient). | Direct proxy for potential energy savings. | Value proposition metric. |
| `energy_savings_pct` | Float | Estimated percentage of energy saved by using ML vs. Heat Curve. | The primary business KPI for the project. | Financial impact estimator. |
| `target_temp` | Float | The setpoint temperature at the time of comparison. | Context for the efficiency calculation. | Control setpoint. |

---

## Measurement: `ml_thermodynamics`

Real-time thermodynamic performance metrics of the heat pump and hydraulic circuit.

| Field Key | Data Type | Semantic Description | Observability Significance | ML Feature Context |
| :--- | :--- | :--- | :--- | :--- |
| `cop_realtime` | Float | Coefficient of Performance (Heat Output / Electrical Input). | The "miles per gallon" of the heat pump. Sudden drops indicate faults. | Efficiency target variable. |
| `thermal_power_kw` | Float | Instantaneous heat output in Kilowatts. | Verifies the heat pump is delivering expected capacity. | Load variable. |
| `delta_t` | Float | Temperature difference between flow (outlet) and return (inlet). | Indicates hydraulic flow health; too high = low flow, too low = short cycling. | Hydraulic state feature. |
| `flow_rate` | Float | Rate of water circulation (L/min or m³/h). | Critical for heat transfer; zero flow with pump on = blockage/failure. | Mass transfer variable. |

---

## Measurement: `feature_importance`

**Metric Nature:** Global (Time-Varying) Model Weights.
These values represent the **learned physics coefficients** of the thermal equilibrium equation, normalized to sum to 1.0. They indicate the model's global sensitivity to each factor based on recent training data, rather than the instantaneous contribution of a specific event.

**Aggregation:** These are slowly evolving weights derived from the adaptive learning engine, representing the "system identification" state.
**Normalization:** All fields sum to 1.0 to show relative influence.
**Persistence:** Learned weights are saved to the unified thermal state and persist across system restarts. They do not decay automatically; they remain at their learned levels until new prediction errors drive further adaptation.

| Field Key | Data Type | Semantic Description | Observability Significance | ML Feature Context |
| :--- | :--- | :--- | :--- | :--- |
| `pv_power` | Float | Normalized weight of solar gain (PV) in the heat balance equation. | Tracks how "solar-sensitive" the building is considered to be. | Learned coupling coefficient. |
| `fireplace` | Float | Normalized weight of fireplace heat output. | Indicates the learned thermal impact of the fireplace relative to other sources. | Learned coupling coefficient. |
| `tv_power` | Float | Normalized weight of internal appliance heat gain. | Tracks the learned thermal efficiency of internal electrical loads. | Learned coupling coefficient. |
| `outdoor_temp` | Float | Normalized weight of the heat loss coefficient (insulation factor). | High values indicate the model sees the home as "leaky" or weather-dependent. | Physics parameter (Heat Loss). |
| `target_temp` | Float | Normalized weight of the outlet effectiveness (radiator efficiency). | High values indicate the heating system is the dominant driver of temperature change. | Physics parameter (Effectiveness). |

**Use Case:**
This measurement tracks the evolution of the model's "mental map" of the building physics. For example, if `outdoor_temp` weight increases over time, it suggests the model is learning that the building's insulation is less effective than initially thought (or windows are open).

---

## Tag Keys (Common)

These tags are indexed and should be used for `GROUP BY` operations in Flux queries.

*   `source`: Always `ml_heating` (identifies the application).
*   `version`: Schema version (e.g., `2.0`).
*   `learning_phase`: For `ml_learning_phase` measurement (e.g., `high_confidence`).
*   `prediction_horizon`: For `ml_trajectory_prediction` (e.g., `4h`).
*   `parameter_type`: For `ml_thermal_parameters` (e.g., `current`).
*   `mode`: For benchmarks (e.g., `shadow`).
