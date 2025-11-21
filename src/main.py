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
from datetime import datetime, timezone

import numpy as np
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


def poll_for_blocking(ha_client, state, blocking_entities):
    """
    Poll for blocking events during the idle period so defrost starts/ends
    are detected quickly. This function encapsulates the idle polling logic
    so it can be unit tested.
    """
    end_time = time.time() + config.CYCLE_INTERVAL_MINUTES * 60
    while time.time() < end_time:
        try:
            all_states_poll = ha_client.get_all_states()
        except Exception:
            logging.debug("Failed to poll HA during idle; will retry.", exc_info=True)
            time.sleep(config.BLOCKING_POLL_INTERVAL_SECONDS)
            continue

        blocking_now = any(
            ha_client.get_state(e, all_states_poll, is_binary=True)
            for e in blocking_entities
        )

        # Blocking started during idle -> persist and handle immediately.
        if blocking_now and not state.get("last_is_blocking", False):
            try:
                blocking_reasons_now = [
                    e
                    for e in blocking_entities
                    if ha_client.get_state(e, all_states_poll, is_binary=True)
                ]
                save_state(
                    last_is_blocking=True,
                    last_final_temp=state.get("last_final_temp"),
                    last_blocking_reasons=blocking_reasons_now,
                    last_blocking_end_time=None,
                )
                logging.info("Blocking detected during idle poll; handling immediately.")
            except Exception:
                logging.debug(
                    "Failed to persist blocking start detected during idle poll.", exc_info=True
                )
            return

        # Blocking ended during idle -> persist end time so grace will run.
        if state.get("last_is_blocking", False) and not blocking_now:
            try:
                save_state(
                    last_is_blocking=True,
                    last_blocking_end_time=time.time(),
                    last_blocking_reasons=[],
                )
                logging.info("Blocking ended during idle poll; will run grace on next loop.")
            except Exception:
                logging.debug(
                    "Failed to persist blocking end during idle poll.", exc_info=True
                )
            return

        time.sleep(config.BLOCKING_POLL_INTERVAL_SECONDS)


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

    # Suppress verbose logging from underlying libraries.
    logging.getLogger("requests").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.INFO)

    # --- Initialization ---
    # Load the persisted model, metrics, and application state from files.
    # If they don't exist, create new instances.
    model, mae, rmse = load_model()
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
            influx_service.write_feature_importances(
                importances, bucket=config.INFLUX_FEATURES_BUCKET
            )
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
            # Load the application state at the beginning of each cycle.
            state = load_state()
            # Create a new Home Assistant client for each cycle.
            ha_client = create_ha_client()
            # Fetch all states from Home Assistant at once to minimize API calls.
            all_states = ha_client.get_all_states()

            if not all_states:
                logging.warning("Could not fetch states from HA, skipping cycle.")
                # Emit NETWORK_ERROR state to Home Assistant
                try:
                    ha_client = create_ha_client()
                    attributes_state = get_sensor_attributes(
                        "sensor.ml_heating_state"
                    )
                    attributes_state.update(
                        {
                            "state_description": (
                                "Network Error - could not fetch HA states"
                            ),
                            "last_updated": datetime.now(
                                timezone.utc
                            ).isoformat(),
                        }
                    )
                    ha_client.set_state(
                        "sensor.ml_heating_state",
                        3,
                        attributes_state,
                        round_digits=None,
                    )
                except Exception:
                    logging.debug(
                        "Failed to write NETWORK_ERROR state to HA.",
                        exc_info=True,
                    )
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
            # Build a list of active blocking reasons so we can distinguish
            # DHW-like (long) blockers from short ones like defrost.
            blocking_reasons = [
                e
                for e in blocking_entities
                if ha_client.get_state(e, all_states, is_binary=True)
            ]
            is_blocking = bool(blocking_reasons)

            # --- Grace Period after Blocking ---
            # If a blocking event recently ended, enter a grace period to allow
            # the system to stabilize before resuming ML control. We persist the
            # blocking end time so the grace period can expire if the event is
            # too far in the past. Also abort the grace immediately if blocking
            # reappears while waiting.
            last_is_blocking = state.get("last_is_blocking", False)
            last_blocking_end_time = state.get("last_blocking_end_time")
            if last_is_blocking and not is_blocking:
                # Mark the blocking end time if not already set (this happens when
                # we detect the transition from blocking->not-blocking).
                if last_blocking_end_time is None:
                    last_blocking_end_time = time.time()
                    try:
                        save_state(last_blocking_end_time=last_blocking_end_time)
                    except Exception:
                        logging.debug("Failed to persist last_blocking_end_time.", exc_info=True)

                # If the blocking end is older than GRACE_PERIOD_MAX_MINUTES, skip grace.
                age = time.time() - last_blocking_end_time
                if age > config.GRACE_PERIOD_MAX_MINUTES * 60:
                    logging.info(
                        "Grace period expired (ended %.1f min ago); skipping restore/wait.",
                        age / 60.0,
                    )
                else:
                    logging.info("--- Grace Period Started ---")
                    logging.info(
                        "Blocking event ended %.1f min ago. Entering grace period to allow system to stabilize.",
                        age / 60.0,
                    )
                    last_final_temp = state.get("last_final_temp")

                    if last_final_temp is not None:
                        logging.info(
                            "Restoring last known target temperature of %.1f°C.",
                            last_final_temp,
                        )
                        ha_client.set_state(
                            config.TARGET_OUTLET_TEMP_ENTITY_ID,
                            last_final_temp,
                            get_sensor_attributes(
                                config.TARGET_OUTLET_TEMP_ENTITY_ID
                            ),
                            round_digits=0,
                        )

                        # Determine whether the outlet is currently hotter or colder than
                        # the last target. For DHW-like events the outlet will be hotter;
                        # for defrost it can be colder due to reversed flow. Choose the
                        # waiting direction accordingly and enforce a timeout.
                        start_time = time.time()
                        max_seconds = config.GRACE_PERIOD_MAX_MINUTES * 60

                        actual_outlet_temp_start = ha_client.get_state(
                            config.ACTUAL_OUTLET_TEMP_ENTITY_ID,
                            ha_client.get_all_states(),
                        )
                        if actual_outlet_temp_start is None:
                            logging.warning(
                                "Cannot read actual_outlet_temp at grace start; skipping wait."
                            )
                        else:
                            delta0 = actual_outlet_temp_start - last_final_temp
                            if delta0 == 0:
                                logging.info(
                                    "Actual outlet equals the restored target (%.1f°C); no wait needed.",
                                    last_final_temp,
                                )
                            else:
                                # Choose wait condition based on initial direction.
                                wait_for_cooling = delta0 > 0
                                logging.info(
                                    "Grace period: initial outlet delta %.1f°C -> waiting for %s (timeout %d min).",
                                    delta0,
                                    "actual <= target" if wait_for_cooling else "actual >= target",
                                    config.GRACE_PERIOD_MAX_MINUTES,
                                )

                                while True:
                                    # During the grace wait we also poll for blocking
                                    # reappearance. If blocking appears we abort the
                                    # grace and reset the blocking timer.
                                    all_states_poll = ha_client.get_all_states()
                                    blocking_now_poll = any(
                                        ha_client.get_state(e, all_states_poll, is_binary=True)
                                        for e in blocking_entities
                                    )
                                    if blocking_now_poll:
                                        logging.info(
                                            "Blocking reappeared during grace; aborting wait and preserving blocking state."
                                        )
                                        try:
                                            blocking_reasons_now = [
                                                e
                                                for e in blocking_entities
                                                if ha_client.get_state(e, all_states_poll, is_binary=True)
                                            ]
                                            save_state(
                                                last_is_blocking=True,
                                                last_final_temp=state.get("last_final_temp"),
                                                last_blocking_reasons=blocking_reasons_now,
                                                last_blocking_end_time=None,
                                            )
                                        except Exception:
                                            logging.debug("Failed to persist blocking restart.", exc_info=True)
                                        break

                                    actual_outlet_temp = ha_client.get_state(
                                        config.ACTUAL_OUTLET_TEMP_ENTITY_ID,
                                        all_states_poll,
                                    )
                                    if actual_outlet_temp is None:
                                        logging.warning(
                                            "Cannot read actual_outlet_temp, exiting grace period."
                                        )
                                        break
                                    # If outlet started hotter than target, wait until it cools
                                    # to <= target. If it started colder (defrost), wait until
                                    # it warms to >= target.
                                    if wait_for_cooling and actual_outlet_temp <= last_final_temp:
                                        logging.info(
                                            "Actual outlet temp (%.1f°C) has cooled to or below"
                                            " target (%.1f°C). Resuming control.",
                                            actual_outlet_temp,
                                            last_final_temp,
                                        )
                                        break
                                    if (not wait_for_cooling) and actual_outlet_temp >= last_final_temp:
                                        logging.info(
                                            "Actual outlet temp (%.1f°C) has warmed to or above"
                                            " target (%.1f°C). Resuming control.",
                                            actual_outlet_temp,
                                            last_final_temp,
                                        )
                                        break
                                    elapsed = time.time() - start_time
                                    if elapsed > max_seconds:
                                        logging.warning(
                                            "Grace period timed out after %d minutes; proceeding.",
                                            config.GRACE_PERIOD_MAX_MINUTES,
                                        )
                                        break
                                    logging.info(
                                        "Waiting for outlet to %s target (current: %.1f°C, target: %.1f°C). Elapsed: %d/%d min",
                                        "cool to" if wait_for_cooling else "warm to",
                                        actual_outlet_temp,
                                        last_final_temp,
                                        int(elapsed / 60),
                                        config.GRACE_PERIOD_MAX_MINUTES,
                                    )
                                    time.sleep(config.BLOCKING_POLL_INTERVAL_SECONDS)

                    else:
                        logging.info(
                            "No last_final_temp found in persisted state; skipping restore/wait."
                        )

                    logging.info("--- Grace Period Ended ---")
                    try:
                        save_state(last_is_blocking=False, last_blocking_end_time=None)
                    except Exception:
                        logging.debug("Failed to persist cleared blocking state.", exc_info=True)
                    # skip the rest of this cycle so restored last_final_temp stays in HA while sensors settle
                    continue
                    # Refresh HA states after the grace period so subsequent sensor reads
                    # (used for prediction and clamping) reflect the current system state.
                    try:
                        all_states = ha_client.get_all_states()
                    except Exception:
                        logging.debug(
                            "Failed to refresh HA states after grace period.", exc_info=True
                        )

            if is_blocking:
                logging.info(
                    "Blocking process active (DHW/Defrost), skipping."
                )
                try:
                    blocking_reasons = [
                        e
                        for e in blocking_entities
                        if ha_client.get_state(
                            e, all_states, is_binary=True
                        )
                    ]
                    attributes_state = get_sensor_attributes(
                        "sensor.ml_heating_state"
                    )
                    attributes_state.update(
                        {
                            "state_description": "Blocking activity - Skipping",
                            "blocking_reasons": blocking_reasons,
                            "last_updated": datetime.now(
                                timezone.utc
                            ).isoformat(),
                        }
                    )
                    ha_client.set_state(
                        "sensor.ml_heating_state",
                        2,
                        attributes_state,
                        round_digits=None,
                    )
                except Exception:
                    logging.debug(
                        "Failed to write BLOCKED state to HA.", exc_info=True
                    )
                # Save the blocking state for the next cycle (preserve last_final_temp
                # and record which entities caused the blocking so we can avoid
                # learning from DHW-like cycles).
                save_state(
                    last_is_blocking=True,
                    last_final_temp=state.get("last_final_temp"),
                    last_blocking_reasons=blocking_reasons,
                    last_blocking_end_time=None,
                )
                time.sleep(300)
                continue

            # --- Get current sensor values ---
            target_indoor_temp = ha_client.get_state(
                config.TARGET_INDOOR_TEMP_ENTITY_ID, all_states
            )
            actual_indoor = ha_client.get_state(
                config.INDOOR_TEMP_ENTITY_ID, all_states
            )
            actual_outlet_temp = ha_client.get_state(
                config.ACTUAL_OUTLET_TEMP_ENTITY_ID, all_states
            )
            avg_other_rooms_temp = ha_client.get_state(
                config.AVG_OTHER_ROOMS_TEMP_ENTITY_ID, all_states
            )
            fireplace_on = ha_client.get_state(
                config.FIREPLACE_STATUS_ENTITY_ID, all_states, is_binary=True
            )
            outdoor_temp = ha_client.get_state(
                config.OUTDOOR_TEMP_ENTITY_ID, all_states
            )
            owm_temp = ha_client.get_state(
                config.OPENWEATHERMAP_TEMP_ENTITY_ID, all_states
            )

            critical_sensors = {
                config.TARGET_INDOOR_TEMP_ENTITY_ID: target_indoor_temp,
                config.INDOOR_TEMP_ENTITY_ID: actual_indoor,
                config.OUTDOOR_TEMP_ENTITY_ID: outdoor_temp,
                config.OPENWEATHERMAP_TEMP_ENTITY_ID: owm_temp,
                config.AVG_OTHER_ROOMS_TEMP_ENTITY_ID: avg_other_rooms_temp,
                config.ACTUAL_OUTLET_TEMP_ENTITY_ID: actual_outlet_temp,
            }
            missing_sensors = [
                name for name, value in critical_sensors.items() if value is None
            ]

            if missing_sensors:
                logging.warning(
                    "Critical sensors unavailable: %s. Skipping.",
                    ", ".join(missing_sensors),
                )
                try:
                    attributes_state = get_sensor_attributes(
                        "sensor.ml_heating_state"
                    )
                    attributes_state.update(
                        {
                            "state_description": (
                                "No data - missing critical sensors"
                            ),
                            "missing_sensors": missing_sensors,
                            "last_updated": datetime.now(
                                timezone.utc
                            ).isoformat(),
                        }
                    )
                    ha_client.set_state(
                        "sensor.ml_heating_state",
                        4,
                        attributes_state,
                        round_digits=None,
                    )
                except Exception:
                    logging.debug(
                        "Failed to write NO_DATA state to HA.", exc_info=True
                    )
                time.sleep(300)
                continue

            # --- Step 1: Online Learning ---
            # This is the heart of the "online" or "incremental" learning
            # process. The script retrieves the features and indoor temperature
            # from the *previous* cycle. It then calculates the *actual* change
            # in indoor temperature that occurred during that last cycle. This
            # actual outcome is used to update (retrain) the model, allowing it
            # to continuously adapt to changing conditions.
            last_run_features = state.get("last_run_features")
            last_indoor_temp = state.get("last_indoor_temp")
            last_avg_other_rooms_temp = state.get("last_avg_other_rooms_temp")
            last_fireplace_on = state.get("last_fireplace_on", False)

            prediction_history = state.get("prediction_history", [])
            if last_run_features is not None and last_indoor_temp is not None:
                # Decide which temperature to use for learning based on the fireplace state of the last run
                if last_fireplace_on and last_avg_other_rooms_temp is not None:
                    learning_temp_current = avg_other_rooms_temp
                    learning_temp_last = last_avg_other_rooms_temp
                    logging.info(
                        "Fireplace was on. Learning based on avg other rooms temp."
                    )
                else:
                    learning_temp_current = actual_indoor
                    learning_temp_last = last_indoor_temp
                    logging.info("Fireplace was off. Learning based on main indoor temp.")

                # Calculate the true temperature change since the last run.
                actual_delta = learning_temp_current - learning_temp_last
                logging.info(
                    "Learning from last cycle's delta_t: %.3f", actual_delta
                )
                # Get the features and prediction from the last cycle.
                x = last_run_features.to_dict(orient="records")[0]

                # Determine if the previous run was a long hot-water blocking event.
                last_blocking_reasons = state.get("last_blocking_reasons", []) or []
                dhw_like_blockers = {
                    config.DHW_STATUS_ENTITY_ID,
                    config.DISINFECTION_STATUS_ENTITY_ID,
                    config.DHW_BOOST_HEATER_STATUS_ENTITY_ID,
                }
                if any(b in dhw_like_blockers for b in last_blocking_reasons):
                    logging.info(
                        "Previous cycle had DHW-like blocking (%s); skipping learning for that cycle.",
                        last_blocking_reasons,
                    )
                else:
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
            # feature vector. This vector is the input the model will use to make
            # its next prediction.
            if fireplace_on:
                prediction_indoor_temp = avg_other_rooms_temp
                logging.info(
                    "Fireplace is ON. Using average temperature of other rooms for prediction."
                )
            else:
                prediction_indoor_temp = actual_indoor
                logging.info(
                    "Fireplace is OFF. Using main indoor temp for prediction."
                )

            features, outlet_history = build_features(
                ha_client, influx_service, all_states, target_indoor_temp
            )
            if features is None:
                logging.warning("Feature building failed, skipping cycle.")
                time.sleep(300)
                continue

            # --- Step 3: Prediction ---
            # This step determines the best outlet temperature for the
            # upcoming cycle. It involves several sub-steps:
            # 1. Use the `find_best_outlet_temp` function to simulate different
            #    outlet temperatures and let the model predict the outcome for
            #    each. The one that gets closest to the target indoor
            #    temperature is chosen. Baseline-based logic has been removed;
            #    the function always returns the ML proposal (state still records
            #    confidence).
            # Find the best outlet temperature by simulating different values.
            error_target_vs_actual = target_indoor_temp - prediction_indoor_temp
            (
                suggested_temp,
                confidence,
                prediction_history,
                sigma,
            ) = find_best_outlet_temp(
                model,
                features,
                prediction_indoor_temp,
                target_indoor_temp,
                prediction_history,
                outlet_history,
                error_target_vs_actual,
                outdoor_temp,
            )
            final_temp = suggested_temp

            # --- Gradual Temperature Control ---
            # Final safety check to prevent abrupt setpoint jumps. Baseline
            # selection rules:
            #  - Default baseline: the persisted previous target (`last_final_temp`)
            #    when available. This ensures we clamp relative to the last
            #    intended setpoint rather than a transient measured outlet.
            #  - Exception (soft-start): if the last blocking reason matches a
            #    DHW-like blocker (DHW, disinfection, DHW boost), use the current
            #    measured `actual_outlet_temp` as baseline to enable a gentle ramp.
            #  - Fallback: if `last_final_temp` is not available, use the
            #    instantaneous measured outlet temp.
            if actual_outlet_temp is not None:
                max_change = config.MAX_TEMP_CHANGE_PER_CYCLE
                original_temp = final_temp  # Keep a copy for logging

                last_blocking_reasons = state.get("last_blocking_reasons", []) or []
                last_final_temp = state.get("last_final_temp")

                # DHW-like blockers that should keep the soft-start behavior
                dhw_like_blockers = {
                    config.DHW_STATUS_ENTITY_ID,
                    config.DISINFECTION_STATUS_ENTITY_ID,
                    config.DHW_BOOST_HEATER_STATUS_ENTITY_ID,
                }

                # Default baseline is the persisted last_final_temp if present.
                # Override to measured actual_outlet_temp when last blocking reasons
                # include any DHW-like blocker (soft start).
                if last_final_temp is not None:
                    baseline = last_final_temp
                    if any(b in dhw_like_blockers for b in last_blocking_reasons):
                        baseline = actual_outlet_temp
                else:
                    baseline = actual_outlet_temp

                # Calculate the difference from the chosen baseline
                delta = final_temp - baseline
                # Clamp the delta to the maximum allowed change
                if abs(delta) > max_change:
                    final_temp = baseline + np.clip(delta, -max_change, max_change)
                    logging.info("--- Gradual Temperature Control ---")
                    logging.info(
                        "Change from baseline %.1f°C to suggested %.1f°C exceeds"
                        " max change of %.1f°C. Capping at %.1f°C.",
                        baseline,
                        original_temp,
                        max_change,
                        final_temp,
                    )

            # To provide a "predicted indoor temp" sensor in Home Assistant,
            # we run a final prediction using the chosen `final_temp`. This shows
            # what the model expects the indoor temperature to be at the end of
            # the next cycle, given the chosen outlet temperature.
            final_features = features.copy()
            final_features["outlet_temp"] = final_temp
            final_features["outlet_temp_sq"] = final_temp**2
            final_features["outlet_temp_cub"] = final_temp**3
            final_features["outlet_temp_change_from_last"] = (
                final_temp - outlet_history[-1]
            )
            final_features["outlet_indoor_diff"] = (
                final_temp - prediction_indoor_temp
            )
            # Update interaction features so the final prediction uses the same
            # feature construction as the candidate evaluations in the search.
            final_features["outdoor_temp_x_outlet_temp"] = final_features.get(
                "outdoor_temp", outdoor_temp
            ) * final_temp
            predicted_delta = model.predict_one(
                final_features.to_dict(orient="records")[0]
            )
            predicted_indoor = prediction_indoor_temp + predicted_delta

            # --- Step 4: Update Home Assistant and Log ---
            # The calculated `final_temp` is sent to Home Assistant to control
            # the boiler. Other metrics like model confidence, MAE, and
            # feature importances are also published to HA for monitoring.
            logging.debug("Setting target outlet temp")
            ha_client.set_state(
                config.TARGET_OUTLET_TEMP_ENTITY_ID,
                final_temp,
                get_sensor_attributes(config.TARGET_OUTLET_TEMP_ENTITY_ID),
                round_digits=0,
            )
            ha_client.set_state(
                config.PREDICTED_INDOOR_TEMP_ENTITY_ID,
                predicted_indoor,
                get_sensor_attributes(
                    config.PREDICTED_INDOOR_TEMP_ENTITY_ID
                ),
            )

            # --- Log Metrics ---
            logging.debug("Logging model metrics")
            ha_client.log_model_metrics(confidence, mae.get(), rmse.get())
            importances = get_feature_importances(model)
            if importances:
                logging.info("Feature Importances:")
                try:
                    fvals = features.to_dict(orient="records")[0]
                except Exception:
                    fvals = {}
                for feature, importance in sorted(
                    importances.items(),
                    key=lambda item: item[1],
                    reverse=True,
                ):
                    logging.info(
                        "  - %s: %.4f (value: %s)",
                        feature,
                        importance,
                        fvals.get(feature),
                    )
            ha_client.log_feature_importance(importances)
            influx_service.write_feature_importances(
                importances, bucket=config.INFLUX_FEATURES_BUCKET
            )

            # --- Update ML State sensor ---
            try:
                attributes_state = get_sensor_attributes(
                    "sensor.ml_heating_state"
                )
                attributes_state.update(
                    {
                        "state_description": "Confidence - Too Low"
                        if confidence < config.CONFIDENCE_THRESHOLD
                        else "OK - Prediction done",
                        "confidence": round(confidence, 4),
                        "sigma": round(sigma, 4),
                        "mae": round(mae.get(), 4),
                        "rmse": round(rmse.get(), 4),
                        "suggested_temp": round(suggested_temp, 2),
                        "final_temp": round(final_temp, 2),
                        "predicted_indoor": round(predicted_indoor, 2),
                        "last_prediction_time": (
                            datetime.now(timezone.utc).isoformat()
                        ),
                    }
                )
                ha_client.set_state(
                    "sensor.ml_heating_state",
                    1 if confidence < config.CONFIDENCE_THRESHOLD else 0,
                    attributes_state,
                    round_digits=None,
                )
            except Exception:
                logging.debug(
                    "Failed to write ML state to HA.", exc_info=True
                )

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
            # The features and indoor temperature from the *current* run are saved
            # to a file. This data will be loaded at the start of the next loop
            # iteration to be used in the "Online Learning" step.
            state_to_save = {
                "last_run_features": features,
                "last_indoor_temp": actual_indoor,
                "last_avg_other_rooms_temp": avg_other_rooms_temp,
                "last_fireplace_on": fireplace_on,
                "prediction_history": prediction_history,
                "last_final_temp": final_temp,
                "last_is_blocking": is_blocking,
                "last_blocking_reasons": blocking_reasons if is_blocking else [],
            }
            save_state(**state_to_save)
            # Update in-memory state so the idle poll uses fresh data
            state.update(state_to_save)

        except Exception as e:
            logging.error("Error in main loop: %s", e, exc_info=True)
            try:
                ha_client = create_ha_client()
                attributes_state = get_sensor_attributes(
                    "sensor.ml_heating_state"
                )
                attributes_state.update(
                    {
                        "state_description": "Model error",
                        "last_error": str(e),
                        "last_updated": datetime.now(
                            timezone.utc
                        ).isoformat(),
                    }
                )
                ha_client.set_state(
                    "sensor.ml_heating_state",
                    7,
                    attributes_state,
                    round_digits=None,
                )
            except Exception:
                logging.debug(
                    "Failed to write MODEL_ERROR state to HA.", exc_info=True
                )

        # Poll for blocking events during the idle period so defrost starts/ends
        # are detected quickly. This call will block until the next cycle is
        # due, or until a blocking event starts or ends.
        poll_for_blocking(ha_client, state, blocking_entities)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ML Heating Controller"
    )
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
