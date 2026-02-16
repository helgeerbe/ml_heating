# Home Assistant Sensor Attributes Guide

This document provides a detailed reference for the Home Assistant sensors created by the ML Heating Control system.

**Important Note:** These sensors are designed for monitoring the long-term performance and behavior of the ML Heating Control system. The data they provide will become more meaningful after the system has been running for a significant period (at least 24-48 hours) and has had a chance to learn from multiple heating cycles.

## Core Metrics Sensors

### `sensor.ml_heating_learning`

This sensor provides a high-level overview of the adaptive learning process, including the model's confidence and its learned physical parameters.

-   **State**: `learning_confidence` (float)
    -   A score representing the model's confidence in its current parameters. Higher values indicate a more stable and reliable model.

-   **Note**: The `learning_confidence` and other learned parameters will gradually improve as the model gathers more data. Initially, these values may fluctuate.

-   **Attributes**:
    -   `thermal_time_constant` (float): The learned time constant of the building, in hours. This represents how quickly the building's temperature responds to changes in heating.
    -   `total_conductance` (float): The overall heat loss rate of the building, in W/°C.
    -   `equilibrium_ratio` (float): A learned parameter representing the ratio of heat distribution.
    -   `heat_loss_coefficient` (float): The learned coefficient representing heat loss.
    -   `outlet_effectiveness` (float): The learned effectiveness of the heat pump outlet in transferring heat to the indoor environment.
    -   `cycle_count` (int): The total number of learning cycles completed.
    -   `parameter_updates` (int): The number of times the model's parameters have been updated.
    -   `model_health` (string): A string indicating the overall health of the model (e.g., "OK", "NEEDS_CALIBRATION").
    -   `learning_progress` (float): A percentage indicating the progress of the current learning phase.
    -   `is_improving` (bool): `True` if the model's performance is currently improving.
    -   `improvement_percentage` (float): The percentage of improvement in model performance over a recent window.
    -   `total_predictions` (int): The total number of predictions made by the model.
    -   `last_updated` (string): The UTC timestamp of the last update.

### `sensor.ml_model_mae`

This sensor tracks the Mean Absolute Error (MAE) of the model's predictions, providing insight into its accuracy.

-   **State**: `mae_all_time` (float)
    -   The Mean Absolute Error over all predictions since the last reset.

-   **Note**: The MAE values will be more representative of the model's true accuracy after a sufficient number of predictions have been made.

-   **Attributes**:
    -   `mae_1h` (float): MAE over the last hour.
    -   `mae_6h` (float): MAE over the last 6 hours.
    -   `mae_24h` (float): MAE over the last 24 hours.
    -   `trend_direction` (string): The trend of the MAE ("improving", "degrading", or "stable").
    -   `prediction_count` (int): Total number of predictions.
    -   `last_updated` (string): The UTC timestamp of the last update.

### `sensor.ml_model_rmse`

This sensor tracks the Root Mean Squared Error (RMSE) of the model's predictions, which is more sensitive to large errors than MAE.

-   **State**: `rmse_all_time` (float)
    -   The Root Mean Squared Error over all predictions since the last reset.

-   **Note**: Similar to MAE, the RMSE is most meaningful after the model has been operating for some time.

-   **Attributes**:
    -   `recent_max_error` (float): The maximum prediction error observed in the recent past.
    -   `std_error` (float): The standard deviation of the prediction errors.
    -   `mean_bias` (float): The average bias of the predictions, indicating systematic over or under-prediction.
    -   `prediction_count` (int): Total number of predictions.
    -   `last_updated` (string): The UTC timestamp of the last update.

### `sensor.ml_prediction_accuracy`

This sensor provides a detailed breakdown of the model's prediction accuracy over different time windows.

-   **State**: `good_control_pct` (float)
    -   The percentage of predictions in the last 24 hours that were within ±0.2°C of the actual temperature.

-   **Note**: The accuracy percentages will become more stable and representative after at least 24 hours of continuous operation.

-   **Attributes**:
    -   `perfect_accuracy_pct` (float): Percentage of predictions in the last 24 hours within ±0.1°C.
    -   `tolerable_accuracy_pct` (float): Percentage of predictions in the last 24 hours within ±0.2°C.
    -   `poor_accuracy_pct` (float): Percentage of predictions in the last 24 hours with an error greater than 0.2°C.
    -   `prediction_count_24h` (int): Number of predictions in the last 24 hours.
    -   `excellent_all_time_pct` (float): All-time percentage of predictions within ±0.1°C.
    -   `good_all_time_pct` (float): All-time percentage of predictions within ±0.2°C.
    -   `last_updated` (string): The UTC timestamp of the last update.
