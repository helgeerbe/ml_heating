"""This module contains the main logic for the ml_heating application."""
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
    """Main function to run the heating control logic."""
    load_dotenv()
    log_level = logging.DEBUG if args.debug or config.DEBUG else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # --- Initialization ---
    model, mae, rmse = load_model()
    state = load_state()
    influx_service = create_influx_service()

    # --- Initial Training ---
    if args.initial_train:
        initial_train(model, mae, rmse, influx_service)
        save_model(model, mae, rmse)
        # Also log feature importances right after initial training
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

    # --- Main Loop ---
    while True:
        try:
            ha_client = create_ha_client()
            all_states = ha_client.get_all_states()

            if not all_states:
                logging.warning("Could not fetch states from HA, skipping cycle.")
                time.sleep(300)
                continue

            # --- Check for blocking modes (DHW, Defrost) ---
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
                    "DHW, Defrost, or other blocking process active, skipping."
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

            # --- Online Learning Step ---
            last_run_features = state.get("last_run_features")
            last_indoor_temp = state.get("last_indoor_temp")
            prediction_history = state.get("prediction_history", [])
            if last_run_features is not None and last_indoor_temp is not None:
                actual_delta = actual_indoor - last_indoor_temp
                logging.info(
                    "Learning from last cycle's delta_t: %.3f", actual_delta
                )
                x = last_run_features.to_dict(orient="records")[0]
                y_pred = model.predict_one(x)
                model.learn_one(x, actual_delta)
                mae.update(actual_delta, y_pred)
                rmse.update(actual_delta, y_pred)
                save_model(model, mae, rmse)

            # --- Feature Building ---
            features, outlet_history = build_features(
                ha_client, influx_service, all_states
            )
            if features is None:
                logging.warning("Feature building failed, skipping cycle.")
                time.sleep(300)
                continue

            # --- Prediction ---
            baseline_outlet_temp = calculate_baseline_outlet_temp(
                outdoor_temp,
                owm_temp,
                ha_client.get_hourly_forecast(),
            )
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
            # --- Post-processing and Final Calculation ---
            error_target_vs_actual = target_indoor_temp - actual_indoor
            final_temp = calculate_dynamic_boost(
                suggested_temp,
                error_target_vs_actual,
                outdoor_temp,
                baseline_outlet_temp,
            )

            # This is a bit of a hack, but we need to get the predicted indoor temp
            # for the final chosen outlet temp.
            final_features = features.copy()
            final_features["outlet_temp"] = final_temp
            final_features["outlet_temp_sq"] = final_temp**2
            final_features["outlet_temp_cub"] = final_temp**3
            final_features[
                "outlet_temp_change_from_last"
            ] = final_temp - outlet_history[-1]
            final_features["outlet_indoor_diff"] = final_temp - actual_indoor
            predicted_delta = model.predict_one(
                final_features.to_dict(orient="records")[0]
            )
            predicted_indoor = actual_indoor + predicted_delta

            # --- Update Home Assistant and Log ---
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

            logging.info(
                (
                    "Target: %.1f°C | Suggested: %.1f°C | Final: %.1f°C | "
                    "Actual Indoor: %.2f°C | Predicted Indoor: %.2f°C | "
                    "Confidence: %.3f | MAE: %.3f | RMSE: %.3f"
                ),
                target_indoor_temp,
                suggested_temp,
                final_temp,
                actual_indoor,
                predicted_indoor,
                confidence,
                mae.get(),
                rmse.get(),
            )

            # --- Save State for Next Run ---
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
