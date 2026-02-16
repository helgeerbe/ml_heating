# Notebook Creation Plan

This plan outlines the creation of three new V2 notebooks for the `ml_heating` project, adhering to the "Fresh Start" architecture and using `src` imports.

## 1. General Analysis Notebook
**Path:** `notebooks/analysis/01_general_model_analysis.ipynb`
**Purpose:** General exploration of model behavior, feature correlations, and basic prediction vs. actual analysis.

**Structure:**
1.  **Setup**: Imports (`src.config`, `src.analysis.DataLoader`, `src.analysis.plotting`, `src.thermal_equilibrium_model`).
2.  **Data Loading**: Fetch recent data (last 7 days) using `DataLoader`.
3.  **Feature Analysis**:
    *   Correlation matrix of key features (indoor, outdoor, outlet, target).
    *   Time-series plots of inputs.
4.  **Model Simulation**:
    *   Initialize `ThermalEquilibriumModel`.
    *   Run a prediction loop over the historical data (simulating "what would the model have predicted").
5.  **Visualization**:
    *   Use `plotting.plot_prediction_vs_actual` to compare model predictions with actual indoor temperatures.
    *   Analyze residuals (prediction errors).

## 2. Performance Notebook
**Path:** `notebooks/performance/01_calibration_and_online_metrics.ipynb`
**Purpose:** Deep dive into model accuracy, calibration status, and online learning performance.

**Structure:**
1.  **Setup**: Imports (`src.prediction_metrics`, `src.utils_metrics`, `src.thermal_equilibrium_model`).
2.  **Data Loading**: Fetch data for a longer period (e.g., 30 days) to see trends.
3.  **Metrics Calculation**:
    *   Instantiate `PredictionMetrics`.
    *   Calculate MAE and RMSE for different time windows (1h, 6h, 24h, All-time).
    *   Calculate "Good Control" percentage (errors within tolerance).
4.  **Error Analysis**:
    *   Histogram of prediction errors.
    *   Scatter plot of Error vs. Outdoor Temperature (to check for bias at extremes).
    *   Scatter plot of Error vs. Outlet Temperature.
5.  **Calibration Verification**:
    *   Check if current performance meets the criteria for a "calibrated" model.

## 3. Monitoring Notebook
**Path:** `notebooks/monitoring/01_model_health_and_drift.ipynb`
**Purpose:** Monitor the health of the adaptive learning system, check for parameter drift, and ensure state persistence is working.

**Structure:**
1.  **Setup**: Imports (`src.unified_thermal_state`, `src.thermal_equilibrium_model`).
2.  **State Inspection**:
    *   Load the unified thermal state using `ThermalStateManager`.
    *   Display current baseline parameters and learning adjustments.
3.  **Parameter Drift Analysis**:
    *   Extract `parameter_history` from the state.
    *   Plot the evolution of key parameters (`thermal_time_constant`, `heat_loss_coefficient`, `outlet_effectiveness`) over time.
    *   Check if parameters are approaching their defined bounds (using `ThermalParameterConfig`).
4.  **Learning Health**:
    *   Plot `learning_confidence` over time.
    *   Analyze `prediction_history` stored in the state (real production predictions).
5.  **Alerting**:
    *   Identify any "red flags" (e.g., low confidence, parameters stuck at bounds, rapid oscillations).

## Implementation Steps
1.  Create directories: `notebooks/analysis`, `notebooks/performance`, `notebooks/monitoring`.
2.  Implement `notebooks/analysis/01_general_model_analysis.ipynb`.
3.  Implement `notebooks/performance/01_calibration_and_online_metrics.ipynb`.
4.  Implement `notebooks/monitoring/01_model_health_and_drift.ipynb`.
