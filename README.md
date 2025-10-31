# ML Heating Control for Home Assistant

> **Warning**
> This project is an initial test and proof of concept. It is not recommended for unattended production use. Please use it at your own risk and monitor its behavior carefully.

This project implements a machine learning-based heating control system that integrates with Home Assistant. It uses an online learning model to predict the optimal water outlet temperature for a heat pump to efficiently maintain a target indoor temperature.

## Goal

The primary goal of this project is to improve upon traditional heating curves by creating a self-adapting system that learns the unique thermal properties of a house. By predicting the impact of the water outlet temperature on the indoor temperature, it aims to:

-   **Increase Efficiency:** Avoid over- and under-shooting the target temperature.
-   **Improve Comfort:** Maintain a more stable and accurate indoor temperature.
-   **Adapt to Changes:** Automatically adjust to changing seasons, weather patterns, and household conditions.

## Feature Set

-   **Online Learning:** The system continuously learns and adapts. It uses a `river` machine learning model (Adaptive Random Forest Regressor) that updates itself after every cycle based on real-world results.
-   **Home Assistant Integration:** Seamlessly fetches sensor data (temperatures, PV power, etc.) from and sends control commands (target outlet temperature) back to Home Assistant.
-   **InfluxDB for Historical Data:** Uses an InfluxDB database to fetch historical data for initial model training.
-   **Dynamic Boost:** Applies a final correction based on the current error between the target and actual indoor temperatures to react quickly to changes.
-   **Prediction Smoothing:** Smooths the model's predictions over time to prevent rapid, inefficient fluctuations in the heating system.
-   **Smart Rounding:** Intelligently chooses between rounding up or down by predicting which integer temperature will result in an indoor temperature closer to the target.
-   **Confidence-Based Fallback:** The model assesses its own confidence in a prediction. If the confidence is low, it safely falls back to a traditional, reliable heating curve calculation.
-   **Intelligent Temperature Ramping (Gradual Control):** Prevents abrupt changes to the heat pump's setpoint. After blocking events (DHW, defrost), the system can drop to a low temperature. Instead of making a large, inefficient jump back to the target, the controller uses the current *actual* outlet temperature as a baseline and ramps up gradually, improving efficiency and stability.
-   **Grace Period after Blocking:** After a blocking event like DHW or defrosting ends, the water in the system is often much hotter than the normal heating range. To prevent the model from overreacting to this temporary state, the system enters a "grace period." It restores the last known target temperature and waits for the actual outlet temperature to cool down and fall below this target before resuming ML control. This ensures the model makes decisions based on stable, representative data.
-   **Fireplace Mode:** When a fireplace or other significant secondary heat source is active, the model can be configured to use a different temperature sensor (e.g., the average of other rooms) for learning and prediction. This prevents the model from incorrectly learning that the main heating system is more powerful than it is.
-   **Feature Importance:** Logs which factors (features) are most influential in the model's decisions, providing insight into what the model is learning.
-   **Systemd Service:** Can be run as a background service for continuous, unattended operation.

### Features

The model uses a rich set of features engineered from various data sources to understand the thermal dynamics of the house. These include:

-   **Core Features:**
    -   `outdoor_temp`: The current outdoor temperature.
    -   `outlet_temp`: The current water outlet temperature of the heat pump. This is the primary variable the model uses for its search.
    -   `temp_diff_indoor_outdoor`: The difference between the indoor and outdoor temperatures.

-   **Time-based Features:**
    -   Cyclical features for time of day, month of the year, and day of the week (e.g., `hour_sin`, `hour_cos`) to capture daily and seasonal patterns.
    -   `is_weekend`: A flag indicating if it is currently the weekend.

-   **Historical & Lag Features:**
    -   Past values of indoor and outlet temperatures at different intervals (e.g., `indoor_temp_lag_10m`, `outlet_temp_lag_60m`).

-   **Trend & Delta Features:**
    -   The rate of change (gradient) of indoor and outlet temperatures over the last hour.
    -   The difference (delta) between the current temperature and historical values (e.g., `indoor_temp_delta_30m`).

-   **Forecast Features:**
    -   `pv_forecast_*h`: The predicted solar power generation for the next 4 hours.
    -   `temp_forecast_*h`: The predicted outdoor temperature for the next 4 hours.

-   **Polynomial & Interaction Features:**
    -   `outlet_temp_sq`, `outlet_temp_cub`: The square and cube of the outlet temperature to help the model learn non-linear relationships.
    -   `outdoor_temp_x_outlet_temp`: An interaction term to capture how the effect of the outlet temperature might change depending on the outdoor temperature.

-   **Aggregated History Features:**
    -   Statistical summaries of the recent history of indoor and outlet temperatures, including `mean`, `std` (standard deviation), `min`, `max`, `trend`, and quartiles (`q25`, `q75`).

-   **Binary & Status Features:**
    -   `tv_on`: A binary flag indicating if the TV is currently on. This serves as an example of a feature that represents an additional, unmeasurable heat source. The model can learn the correlation between the TV being on and a slight rise in indoor temperature.
    -   `dhw_heating`, `defrosting`, etc.: Binary flags for the status of various heat pump operations that might block or affect heating.

## Model and Approach

### Online Training

This project uses an **online machine learning** approach, which means the model learns incrementally from a stream of live data.

1.  **Initial Training (Optional):** When first run with the `--initial-train` flag, the model is "warmed up" using historical data from InfluxDB. The duration of this lookback period is configurable via the `TRAINING_LOOKBACK_HOURS` environment variable (default is 168 hours, or 7 days). This gives the model a solid baseline understanding of the home's thermal dynamics.
2.  **Live Learning Cycle:**
    -   The system sets a target outlet temperature.
    -   It waits for a cycle (e.g., 5 minutes).
    -   It measures the *actual* change in indoor temperature.
    -   It compares this to what it *predicted* would happen.
    -   It calls the `learn_one` function to update the model with this new data point.

This continuous feedback loop allows the model to adapt to changing seasons, variations in home occupancy, and other dynamic factors without needing to be manually retrained. The model used is a `river.forest.ARFRegressor`, an Adaptive Random Forest that is well-suited for this kind of streaming data.

### Why Adaptive Random Forest Regressor?

The `ARFRegressor` from the `river` library was specifically chosen for several key reasons:

1.  **Designed for Online Learning:** Unlike traditional batch models (e.g., from scikit-learn) that require periodic retraining on large datasets, `river` models are designed to learn from a continuous stream of data, one sample at a time. This is ideal for a system that needs to run and adapt 24/7.
2.  **Adaptability to Concept Drift:** The "A" in ARF stands for "Adaptive." The model has built-in drift detection mechanisms (`PageHinkley` and `ADWIN`). This allows it to detect and adapt to "concept drift"—fundamental changes in the data's underlying patterns, such as the transition from winter to summer.
3.  **Robustness of Ensembles:** As a random forest, it is an ensemble of many decision trees. This makes it more robust and less prone to overfitting than a single model.
4.  **Built-in Uncertainty Measure:** The ensemble nature allows us to easily measure the model's uncertainty. By calculating the standard deviation of the predictions from all the individual trees, we get a reliable indicator of the model's "confidence." This is the basis for the crucial fallback mechanism.

### The Prediction Pipeline: A Step-by-Step Filter

The final target temperature is not a single prediction but the result of a multi-stage filtering and adjustment pipeline. This process is designed to ensure safety, stability, and accuracy, with each step refining the output of the previous one.

1.  **Step 1: Confidence Check & Fallback:** This is the first and most critical gate. The model calculates its confidence by measuring the agreement among its internal decision trees. If the confidence score is below the `CONFIDENCE_THRESHOLD`, the entire ML pipeline is bypassed, and the system safely uses a pre-calculated `baseline_outlet_temp` from a traditional heating curve.

2.  **Step 2: ML-Powered Search:** If confidence is sufficient, the model searches a wide range of possible outlet temperatures (e.g., 25°C to 45°C). For each candidate, it predicts the resulting indoor temperature and identifies the floating-point temperature that is predicted to get closest to the user's target.

3.  **Step 3: Smoothing (EMA Filter):** The raw result from the search can be volatile. To prevent rapid fluctuations, it is smoothed using an Exponential Moving Average (EMA) that considers previous predictions. This ensures the setpoint changes gracefully over time.

4.  **Step 4: Dynamic Boost & Clamping:** To react more quickly to immediate needs, a "boost" is applied. If the room is too cold, the temperature is nudged higher; if it's too warm, it's nudged lower. To prevent extreme or unsafe values, this boosted temperature is then clamped within a safe percentage of the original baseline temperature.

5.  **Step 5: Smart Rounding:** Heat pumps often require whole-number setpoints. Instead of a simple mathematical round, the system performs "smart rounding." It takes the floating-point value from the previous step (e.g., 35.7°C) and runs a final prediction for both the floor (35°C) and the ceiling (36°C). It then chooses the integer that is predicted to result in an indoor temperature closer to the target.

6.  **Step 6: Gradual Temperature Control:** This is the final safety filter, designed to protect the heat pump from inefficient, large temperature jumps, especially after a blocking event like DHW or defrosting. The system compares the proposed integer setpoint from Step 5 to the **current actual outlet temperature**. If the change exceeds the `MAX_TEMP_CHANGE_PER_CYCLE` value (e.g., 2°C), it is capped. For example, if the actual outlet temperature is 30°C and the new pipeline suggests 38°C, the final output will be capped at 32°C. This ensures the heat pump always ramps up or down gently from its current state.

### Metrics: Confidence, MAE, and RMSE

-   **Confidence:** This is not a statistical confidence but rather a measure of agreement among the trees in the random forest model. A lower standard deviation means the trees agree, and the prediction is considered more reliable.
-   **MAE (Mean Absolute Error):** The average absolute difference between the predicted temperature change and the actual temperature change. A lower MAE is better.
-   **RMSE (Root Mean Square Error):** Similar to MAE but gives a higher weight to large errors. A lower RMSE is better.

These metrics are continuously updated and sent to Home Assistant, providing a real-time view of the model's performance.

### ML State Sensor

The application publishes a numeric state sensor `sensor.ml_heating_state` that summarizes the controller's status each cycle and provides attributes with diagnostics. Use the numeric state for automations and the attributes for human-readable context.

State codes (numeric)
- 0 = OK — Prediction done
- 1 = LOW_CONFIDENCE — Confidence too low; fallback used
- 2 = BLOCKED — Blocking activity detected (DHW, defrosting, disinfection, boost) — skipping this cycle
- 3 = NETWORK_ERROR — Failed to fetch Home Assistant states or network calls
- 4 = NO_DATA — Missing critical sensors or insufficient history for features
- 5 = TRAINING — Running initial training / warm-up
- 6 = FALLBACK_BASELINE — Forced baseline/safety fallback (distinct from LOW_CONFIDENCE)
- 7 = MODEL_ERROR — Exception occurred during prediction/learning

Typical attributes
- `state_description`: short human string (e.g., "Confidence - Too Low")
- `confidence`: normalized 0..1 float (1.0 = perfect agreement)
- `sigma`: raw per-tree stddev in °C
- `mae`, `rmse`: current metric floats
- `suggested_temp`, `final_temp`, `predicted_indoor`: recent numeric values
- `fallback_used`: bool
- `blocking_reasons`: list of active blockers (when applicable)
- `missing_sensors`: list of missing critical entities (when applicable)
- `last_prediction_time` / `last_updated`: ISO timestamps
- `last_error`: short error message (for MODEL_ERROR / NETWORK_ERROR)

Example automation use-cases
- Alert if state == 3 (NETWORK_ERROR) or repeated MODEL_ERROR.
- Tally LOW_CONFIDENCE occurrences to gauge model reliability.
- Use `blocking_reasons` to explain skipped cycles in logs or dashboards.

This sensor makes it easy to monitor model health and reason about fallback behaviour directly from Home Assistant.

## Data Flow

1.  **Get Data from Home Assistant:** The script fetches current sensor values (indoor/outdoor temps, etc.) and blocking statuses (DHW, defrost) from Home Assistant.
2.  **Get History from InfluxDB:** The `feature_builder` uses the InfluxDB service to get historical values for features like past outlet temperatures.
3.  **Set Data back to Home Assistant:** The script pushes the following data back to Home Assistant entities:
    -   The final calculated target outlet temperature.
    -   The predicted indoor temperature for the next cycle.
    -   Model performance metrics (Confidence, MAE, RMSE).
    -   Feature importances.

## Feature importance exported to InfluxDB

Feature importance snapshots are exported to InfluxDB for visualization and inspection.

- Bucket: `ml_heating_features`  
- Measurement: `feature_importance`  
- Schema:
  - Each feature is stored as a separate field (float), for example `temp_forecast_3h`, `indoor_hist_mean`, `outlet_temp_lag_60m`, etc.
  - `exported` — integer UNIX epoch seconds marking when the snapshot was written (provenance / run id).
  - `_time` — the snapshot timestamp (when the point was written).
- Configuration:
  - Set the feature export bucket in your `.env`:
    ```ini
    INFLUX_FEATURE_BUCKET=ml_heating_features
    ```
- Visualization tips:
  - Table views may show locale-specific decimals (e.g. `0,03`); charts may apply unit suffixes — set the Y-axis unit to `number`/`none` to display raw floats.
  - Use Flux to scale or format on-read (example: multiply by 100 to show percent) rather than rewriting stored data.

Quick Flux examples
- List measurements in the feature bucket:
```flux
import "influxdata/influxdb/schema"
schema.measurements(bucket: "ml_heating_features")
```

- Inspect recent feature rows:
```flux
from(bucket:"ml_heating_features")
  |> range(start: -7d)
  |> filter(fn: (r) => r._measurement == "feature_importance")
  |> keep(columns: ["_time","_field","_value","exported"])
  |> sort(columns: ["_time"], desc: true)
  |> limit(n: 50)
```

- Convert `exported` to human readable time in Python:
```python
import datetime
ts = 1760904513
print(datetime.datetime.utcfromtimestamp(ts).isoformat() + "Z")
```

## Installation and Setup

### 1. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy the `.env_sample` file to `.env` and fill in the required values.

```bash
cp .env_sample .env
nano .env
```

#### `.env` File Explanation

-   **`HASS_URL`**: The URL of your Home Assistant instance (e.g., `http://homeassistant.local:8123`).
-   **`HASS_TOKEN`**: A Long-Lived Access Token for the Home Assistant API.
-   **`INFLUX_URL`**: The URL of your InfluxDB instance.
-   **`INFLUX_TOKEN`**: An InfluxDB API token with read access to your Home Assistant bucket.
-   **`INFLUX_ORG`**: Your InfluxDB organization name.
-   **`INFLUX_BUCKET`**: The InfluxDB bucket where Home Assistant data is stored.
-   **`MODEL_FILE` / `STATE_FILE`**: Paths to store the trained model and application state. The defaults are usually fine.
-   **`TRAINING_LOOKBACK_HOURS`**: The number of hours of historical data to use for the initial training. Defaults to 168 (7 days).
-   **`CYCLE_INTERVAL_MINUTES`**: The time in minutes between each full cycle of learning and prediction. A longer interval (e.g., 10-15 mins) provides a clearer learning signal, while a shorter one is more responsive. Defaults to 10.
-   **`MAX_TEMP_CHANGE_PER_CYCLE`**: The maximum allowable integer change (in degrees) for the outlet temperature setpoint in a single cycle. This prevents abrupt changes that can cause the heat pump to start and stop frequently. For example, a value of `1` with a `CYCLE_INTERVAL_MINUTES` of `10` limits the maximum change to 6 degrees per hour. Defaults to 2.
-   **Entity IDs**: The script is pre-configured with many entity IDs. You **must** review and update these to match the `entity_id`s in your Home Assistant setup.
-   **`CONFIDENCE_THRESHOLD`**: The model uses a *normalized* confidence in the range (0..1], where `1.0` means perfect agreement between trees (σ = 0 °C). The code maps the per-tree standard deviation σ (in °C) to confidence using:
    ```python
    confidence = 1.0 / (1.0 + sigma)
    ```
    To pick an appropriate threshold, decide the maximum tolerated σ (°C) and convert:
    ```
    threshold = 1.0 / (1.0 + sigma_max)
    ```
    Examples:
    - Tolerate σ_max = 1.0°C → `CONFIDENCE_THRESHOLD = 0.5`
    - Tolerate σ_max = 0.5°C → `CONFIDENCE_THRESHOLD ≈ 0.667`
    The sample files use `0.5` as a reasonable starting point.

### 4. Initial Training (Recommended)

Run the script with the `--initial-train` flag. This will train the model on your recent historical data from InfluxDB.

```bash
python3 -m src.main --initial-train
```

You can use the `--train-only` flag to exit after training is complete.

### 5. Running as a systemd Service

To run the script continuously in the background, create a systemd service file.

1.  Create the service file:

    ```bash
    sudo nano /etc/systemd/system/ml_heating.service
    ```

2.  Paste the following content into the file, making sure to **update the paths** to match your installation directory.

    ```ini
    [Unit]
    Description=ML Heating Control Service
    After=network.target

    [Service]
    Type=simple
    User=your_user
    WorkingDirectory=/path/to/your/ml_heating
    ExecStart=/path/to/your/ml_heating/.venv/bin/python3 -m src.main
    Restart=on-failure
    RestartSec=5min

    [Install]
    WantedBy=multi-user.target
    ```

3.  Reload systemd, enable, and start the service:

    ```bash
    sudo systemctl daemon-reload
    sudo systemctl enable ml_heating.service
    sudo systemctl start ml_heating.service
    ```

4.  You can check the status and logs of the service using:

    ```bash
    sudo systemctl status ml_heating.service
    sudo journalctl -u ml_heating.service -f
    ```

## Script Parameters

-   `--initial-train`: Runs the initial training process on historical data from InfluxDB before starting the main loop.
-   `--train-only`: Runs the initial training process and then exits. Useful for warming up the model without starting the control loop.
-   `--debug`: Enables verbose debug logging. Note that this may cause very noisy output from underlying libraries like `urllib3`. The application attempts to suppress this, but some messages may still appear.
