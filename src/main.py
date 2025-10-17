"""
This module is the central entry point and main control loop for the application.

It orchestrates the entire process of data collection, online learning,
prediction, and action. The script operates in a continuous loop,
performing the following key steps in each iteration:
1.  **Initialization**: Loads the ML model and application state.
2.  **Data Fetching**: Gathers the latest sensor data from Home Assistant.
3.  **Online Learning**: Updates the model with the outcome of the previous
    cycle.
4.  **Feature Engineering**: Builds a feature set from current and
    historical data.
5.  **Prediction**: Uses the model to find the optimal heating temperature.
6.  **Action**: Sets the new target temperature in Home Assistant.
7.  **State Persistence**: Saves the current state for the next cycle.

The script also supports an initial training mode (`--initial-train`) to
bootstrap the model using historical data from InfluxDB.
"""
import argparse
import logging
import time

from dotenv import load_dotenv

from . import config
from .feature_builder import build_features
from .ha_client import create_ha_client, get_sensor_attributes
from .influx_service import create_influx_service
from .model_wrapper import (
    find_best_outlet_temp,
    get_feature_importances,
    initial_train,
    load_model,
    save_model,
)
from .state_manager import load_state, save_state
from .thermal import calculate_baseline_outlet_temp, calculate_dynamic_boost


def main(args):
    """
    The main function that orchestrates the heating control logic.

    This function initializes the system, enters a continuous loop to monitor
    and control the heating, and handles command-line arguments for modes like
    initial training.

    Args:
        args: Command-line arguments parsed by argparse.
    """
    # Load environment variables and configure logging.
    load_dotenv()
    log_level = logging.DEBUG if args.debug or config.DEBUG else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # --- Initialization ---
    # Load the persisted model, metrics, and application state from files.
    # If they don't exist, create new instances.
    model, mae, rmse = load_model()
    state = load_state()
    influx_service = create_influx_service()

    # --- Initial Training ---
    # If the --initial-train flag is set, train the model on historical data
    # from InfluxDB to give it a warm start.
    if args.initial_train:
        initial_train(model, mae, rmse, influx_service)
        save_model(model, mae, rmse)
        # Log feature importances to understand what the model learned
        # initially.
        importances = get_feature_importances(model)
        if importances:
            logging.info("Initial Feature Importances:")
            for feature, importance in sorted(
                importances.items(), key=lambda item: item[1], reverse=True
            ):
                logging.info(f"  - {feature}: {importance:.4f}")
    if args.train_only:
        logging.warning(
            "Training complete. Exiting as requested by --train-only flag."
        )
        return

    # --- Main Control Loop ---
    # This loop runs indefinitely, performing one full cycle of learning and
    # prediction every 5 minutes.
    while True:
        try:
            # Create a new Home Assistant client for each cycle.
            ha_client = create_ha_client()
            # Fetch all states from Home Assistant at once to minimize API calls.
            all_states = ha_client.get_all_states()

            if not all_states:
                logging.warning("Could not fetch states from HA, skipping cycle.")
                time.sleep(300)
                continue

            # --- Check for blocking modes (DHW, Defrost) ---
            # Skip the control logic if the heat pump is busy with other tasks
            # like heating domestic hot water (DHW) or defrosting.
            blocking_entities = [
                config.DHW_STATUS_ENTITY_ID,
                config.DEFROST_STATUS_ENTITY_ID,
                config.DISINFECTION_STATUS_ENTITY_ID,
                config.DHW_BOOST_HEATER_STATUS_ENTITY_ID,
            ]
            is_blocking = any(
                ha_client.get_state(entity, all_states, is_binary=True)
                for entity in blocking_entities
            )
            if is_blocking:
                logging.info(
                    "Blocking process active (DHW/Defrost), skipping."
                )
                time.sleep(300)
                continue

            # --- Get current sensor values ---
            target_indoor_temp = ha_client.get_state(
                config.TARGET_INDOOR_TEMP_ENTITY_ID,
                all_states,
            )
            actual_indoor = ha_client.get_state(
                config.INDOOR_TEMP_ENTITY_ID, all_states
            )
            outdoor_temp = ha_client.get_state(
                config.OUTDOOR_TEMP_ENTITY_ID, all_states
            )
            owm_temp = ha_client.get_state(
                config.OPENWEATHERMAP_TEMP_ENTITY_ID, all_states
            )

            critical_sensors = [
                target_indoor_temp,
                actual_indoor,
                outdoor_temp,
                owm_temp,
            ]
            if any(v is None for v in critical_sensors):
                logging.warning(
                    "One or more critical sensors unavailable, skipping."
                )
                time.sleep(300)
                continue

            # --- Step 1: Online Learning ---
            # This is the heart of the "online" or "incremental" learning
            # process. The script retrieves the features and indoor
            # temperature from the *previous* cycle. It then calculates the
            # *actual* change in indoor temperature that occurred during that
            # last cycle. This actual outcome is used to update (retrain) the
            # model, allowing it to continuously adapt to changing conditions.
            last_run_features = state.get("last_run_features")
            last_indoor_temp = state.get("last_indoor_temp")
            prediction_history = state.get("prediction_history", [])
            if last_run_features is not None and last_indoor_temp is not None:
                # Calculate the true temperature change since the last run.
                actual_delta = actual_indoor - last_indoor_temp
                logging.info(
                    "Learning from last cycle's delta_t: %.3f", actual_delta
                )
                # Get the features and prediction from the last cycle.
                x = last_run_features.to_dict(orient="records")[0]
                y_pred = model.predict_one(x)
                # Update the model with the actual outcome.
                model.learn_one(x, actual_delta)
                # Update the rolling performance metrics.
                mae.update(actual_delta, y_pred)
                rmse.update(actual_delta, y_pred)
                # Persist the updated model and metrics.
                save_model(model, mae, rmse)

            # --- Step 2: Feature Building ---
            # Gathers all the necessary data points (current sensor values,
            # historical data from InfluxDB, etc.) and transforms them into a
            # feature vector. This vector is the input the model will use to
            # make its next prediction.
            features, outlet_history = build_features(
                ha_client, influx_service, all_states
            )
            if features is None:
                logging.warning("Feature building failed, skipping cycle.")
                time.sleep(300)
                continue

            # --- Step 3: Prediction ---
            # This step determines the best outlet temperature for the
            # upcoming cycle. It involves several sub-steps:
            # 1. Calculate a baseline temperature using a traditional heating
            #    curve. This serves as a safe fallback.
            # 2. Use the `find_best_outlet_temp` function to simulate
            #    different outlet temperatures and let the model predict the
            #    outcome for each. The one that gets closest to the target
            #    indoor temperature is chosen.
            # 3. The model's confidence in its prediction is also returned. If
            #    it's too low, the system will revert to the baseline
            #    temperature.
            baseline_outlet_temp = calculate_baseline_outlet_temp(
                outdoor_temp,
                owm_temp,
                ha_client.get_hourly_forecast(),
            )
            # Find the best outlet temperature by simulating different values.
            (
                suggested_temp,
                confidence,
                prediction_history,
            ) = find_best_outlet_temp(
                model,
                features,
                actual_indoor,
                target_indoor_temp,
                baseline_outlet_temp,
                prediction_history,
                outlet_history,
            )
            # --- Step 4: Post-processing and Final Calculation ---
            # The raw suggestion from the model is refined. A "dynamic boost"
            # is applied, which is a corrective factor based on the current
            # error between the target and actual indoor temperatures. This
            # helps the system react more quickly to immediate needs.
            error_target_vs_actual = target_indoor_temp - actual_indoor
            final_temp = calculate_dynamic_boost(
                suggested_temp,
                error_target_vs_actual,
                outdoor_temp,
                baseline_outlet_temp,
            )

            # To provide a "predicted indoor temp" sensor in Home Assistant,
            # we run a final prediction using the chosen `final_temp`. This
            # shows what the model expects the indoor temperature to be at the
            # end of the next cycle, given the chosen outlet temperature.
            final_features = features.copy()
            final_features["outlet_temp"] = final_temp
            final_features["outlet_temp_sq"] = final_temp**2
            final_features["outlet_temp_cub"] = final_temp**3
            final_features["outlet_temp_change_from_last"] = (
                final_temp - outlet_history[-1]
            )
            final_features["outlet_indoor_diff"] = final_temp - actual_indoor
            predicted_delta = model.predict_one(
                final_features.to_dict(orient="records")[0]
            )
            predicted_indoor = actual_indoor + predicted_delta

            # --- Step 5: Update Home Assistant and Log ---
            # The calculated `final_temp` is sent to Home Assistant to
            # control the boiler. Other metrics like model confidence, MAE,
            # and feature importances are also published to HA for monitoring.
            logging.debug("Setting target outlet temp")
            ha_client.set_state(
                config.TARGET_OUTLET_TEMP_ENTITY_ID,
                final_temp,
                get_sensor_attributes(config.TARGET_OUTLET_TEMP_ENTITY_ID),
            )
            ha_client.set_state(
                config.PREDICTED_INDOOR_TEMP_ENTITY_ID,
                predicted_indoor,
                get_sensor_attributes(config.PREDICTED_INDOOR_TEMP_ENTITY_ID),
            )

            # --- Log Metrics ---
            logging.debug("Logging model metrics")
            ha_client.log_model_metrics(confidence, mae.get(), rmse.get())
            importances = get_feature_importances(model)
            if importances:
                logging.info("Feature Importances:")
                for feature, importance in sorted(
                    importances.items(), key=lambda item: item[1], reverse=True
                ):
                    logging.info(f"  - {feature}: {importance:.4f}")
            ha_client.log_feature_importance(importances)

            log_message = (
                "Target: %.1f°C | Suggested: %.1f°C | Final: %.1f°C | "
                "Actual Indoor: %.2f°C | Predicted Indoor: %.2f°C | "
                "Confidence: %.3f | MAE: %.3f | RMSE: %.3f"
            )
            logging.info(
                log_message,
                target_indoor_temp,
                suggested_temp,
                final_temp,
                actual_indoor,
                predicted_indoor,
                confidence,
                mae.get(),
                rmse.get(),
            )

            # --- Step 6: Save State for Next Run ---
            # The features and indoor temperature from the *current* run are
            # saved to a file. This data will be loaded at the start of the
            # next loop iteration to be used in the "Online Learning" step.
            save_state(features, actual_indoor, prediction_history)

        except Exception as e:
            logging.error("Error in main loop: %s", e, exc_info=True)

        time.sleep(300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Heating Controller")
    parser.add_argument(
        "--initial-train",
        action="store_true",
        help="Run the initial training on historical data.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Run the initial training and then exit.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    parsed_args = parser.parse_args()
    main(parsed_args)
