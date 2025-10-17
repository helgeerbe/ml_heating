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

1.  **Initial Training (Optional):** When first run with the `--initial-train` flag, the model is "warmed up" using up to 168 hours of historical data from InfluxDB. This gives it a solid baseline understanding of the home's thermal dynamics.
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

### Prediction Mechanism

1.  **Find Best Temperature:** The core `find_best_outlet_temp` function searches a range of possible outlet temperatures (e.g., 25°C to 45°C). For each candidate temperature, it runs a prediction to estimate the resulting indoor temperature. It selects the temperature that is predicted to get closest to the user's target.
2.  **Prediction Smoothing:** To avoid rapid changes, the chosen temperature is smoothed using an exponential moving average with the previous prediction.
3.  **Smart Rounding:** The system predicts the outcome for both the `floor` and `ceil` of the smoothed temperature and chooses the integer value that yields a better result.
4.  **Dynamic Boost:** A final boost is added based on the current temperature error (`target_indoor_temp - actual_indoor_temp`) to make the system more responsive.
5.  **Fallback:** Before making a prediction, the model calculates the standard deviation of the predictions from its internal decision trees. If this value (a measure of uncertainty) is above `CONFIDENCE_THRESHOLD`, the system discards the ML prediction and falls back to the `baseline_outlet_temp` calculated from a traditional heating curve.

### Metrics: Confidence, MAE, and RMSE

-   **Confidence:** This is not a statistical confidence but rather a measure of agreement among the trees in the random forest model. A lower standard deviation means the trees agree, and the prediction is considered more reliable.
-   **MAE (Mean Absolute Error):** The average absolute difference between the predicted temperature change and the actual temperature change. A lower MAE is better.
-   **RMSE (Root Mean Square Error):** Similar to MAE but gives a higher weight to large errors. A lower RMSE is better.

These metrics are continuously updated and sent to Home Assistant, providing a real-time view of the model's performance.

## Data Flow

1.  **Get Data from Home Assistant:** The script fetches current sensor values (indoor/outdoor temps, etc.) and blocking statuses (DHW, defrost) from Home Assistant.
2.  **Get History from InfluxDB:** The `feature_builder` uses the InfluxDB service to get historical values for features like past outlet temperatures.
3.  **Set Data back to Home Assistant:** The script pushes the following data back to Home Assistant entities:
    -   The final calculated target outlet temperature.
    -   The predicted indoor temperature for the next cycle.
    -   Model performance metrics (Confidence, MAE, RMSE).
    -   Feature importances.

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
-   **Entity IDs**: The script is pre-configured with many entity IDs. You **must** review and update these to match the `entity_id`s in your Home Assistant setup.
-   **`CONFIDENCE_THRESHOLD`**: A key tuning parameter. A higher value makes the model more "adventurous" but less reliable. A lower value makes it more conservative and more likely to use the fallback heating curve. The default is `0.2`.

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
-   `--debug`: Enables verbose debug logging.
