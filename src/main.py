"""
This module is the central entry point and main control loop for the application.

It orchestrates the entire process of data collection, prediction, and action
using the enhanced physics-based heating model. The script operates in a
continuous loop, performing the following key steps in each iteration:
1.  **Initialization**: Loads the physics model and application state.
2.  **Data Fetching**: Gathers the latest sensor data from Home Assistant.
3.  **Feature Engineering**: Builds a feature set from current and historical data.
4.  **Prediction**: Uses the physics model to find the optimal heating temperature.
5.  **Action**: Sets the new target temperature in Home Assistant.
6.  **State Persistence**: Saves the current state for the next cycle.
"""
import argparse
import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from . import config
from .physics_features import build_physics_features
from .ha_client import create_ha_client, get_sensor_attributes
from .influx_service import create_influx_service
from .model_wrapper import (
    simplified_outlet_prediction,
)
from .physics_calibration import (
    train_thermal_equilibrium_model,
    validate_thermal_model,
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
    
    # Configure logging to ensure output goes to stdout for systemd capture
    import sys
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,  # Explicitly output to stdout for systemd
        force=True,  # Force reconfigure if already configured
    )

    # Suppress verbose logging from underlying libraries.
    logging.getLogger("requests").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.INFO)

    # --- Initialization ---
    # ThermalEquilibriumModel is now loaded directly in model_wrapper.py
    # Shadow mode comparison metrics (no longer tracking MAE/RMSE)
    shadow_ml_error_sum = 0.0
    shadow_hc_error_sum = 0.0
    shadow_comparison_count = 0
    
    influx_service = create_influx_service()

    # --- Shadow Mode Status ---
    if config.SHADOW_MODE:
        logging.info("🔍 SHADOW MODE ENABLED: ML will observe and learn without affecting heating control")
        logging.info("   - ML predictions calculated but not sent to HA")
        logging.info("   - No HA sensor updates (confidence, MAE, RMSE, state)")
        logging.info("   - Learning from heat curve's actual control decisions")
        logging.info("   - Performance comparison logging active")
    else:
        logging.info("🎯 ACTIVE MODE: ML actively controls heating system")

    # --- Thermal Model Calibration ---
    if args.calibrate_physics:
        try:
            from .physics_calibration import backup_existing_calibration
            
            logging.info("=== CALIBRATING THERMAL EQUILIBRIUM MODEL ===")
            
            # Create backup before calibration
            logging.info("Step 0: Creating backup before calibration...")
            backup_path = backup_existing_calibration()
            if backup_path:
                import os
                logging.info("✅ Previous thermal state backed up: %s", os.path.basename(backup_path))
            else:
                logging.info("ℹ️ No existing thermal state found to backup")
            
            result = train_thermal_equilibrium_model()
            if result:
                logging.info("✅ Thermal model calibrated successfully!")
                logging.info("🔄 Restart ml_heating service to use trained thermal model")
            else:
                logging.error("❌ Thermal model calibration failed")
        except Exception as e:
            logging.error("Thermal model calibration error: %s", e, exc_info=True)
        return
    
    # --- Thermal Model Validation ---
    if args.validate_physics:
        try:
            result = validate_thermal_model()
            if result:
                logging.info("✅ Thermal model validation passed!")
            else:
                logging.error("❌ Thermal model validation failed!")
        except Exception as e:
            logging.error("Thermal model validation error: %s", e, exc_info=True)
        return

    # --- Main Control Loop ---
    # This loop runs indefinitely, performing one full cycle of learning and
    # prediction every 5 minutes.
    
    # Define blocking_entities outside try block so it's available in exception handler
    blocking_entities = [
        config.DHW_STATUS_ENTITY_ID,
        config.DEFROST_STATUS_ENTITY_ID,
        config.DISINFECTION_STATUS_ENTITY_ID,
        config.DHW_BOOST_HEATER_STATUS_ENTITY_ID,
    ]
    
    # Cycle timing debug variables
    cycle_number = 0
    last_cycle_end_time = None
    
    while True:
        try:
            # CYCLE START DEBUG LOGGING
            cycle_number += 1
            cycle_start_time = time.time()
            cycle_start_datetime = datetime.now()
            
            # Calculate interval since last cycle
            if last_cycle_end_time is not None:
                interval_since_last = cycle_start_time - last_cycle_end_time
                logging.info(f"🔄 CYCLE {cycle_number} START: {cycle_start_datetime.strftime('%H:%M:%S')} "
                           f"(interval: {interval_since_last/60:.1f}min since last cycle)")
            else:
                logging.info(f"🔄 CYCLE {cycle_number} START: {cycle_start_datetime.strftime('%H:%M:%S')} "
                           f"(first cycle)")
            # Initialize shadow mode tracking for this cycle
            shadow_mode_active = (
                config.TARGET_OUTLET_TEMP_ENTITY_ID !=
                config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID
            )
            
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

            # --- Step 1: Online Learning from Previous Cycle ---
            # Learn from the results of the previous cycle. This allows
            # the model to continuously adapt to the actual house behavior,
            # whether running in active mode (model controls heating) or
            # shadow mode (heat curve controls heating).
            last_run_features = state.get("last_run_features")
            last_indoor_temp = state.get("last_indoor_temp")
            last_final_temp_stored = state.get("last_final_temp")

            if (last_run_features is not None and
                    last_indoor_temp is not None and
                    last_final_temp_stored is not None):

                # Read the actual target outlet temp that was applied.
                # This reads what temperature was actually set by either:
                # - The model in active mode
                # - The heat curve in shadow mode
                # By reading it now (start of next cycle), we give it time
                # to update after the previous cycle's set_state call.
                actual_applied_temp = ha_client.get_state(
                    config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID, all_states
                )

                if actual_applied_temp is None:
                    logging.debug(
                        "Could not read actual applied temp, using "
                        "last_final_temp as fallback"
                    )
                    actual_applied_temp = last_final_temp_stored

                # Get current indoor temperature to calculate actual change
                current_indoor = ha_client.get_state(
                    config.INDOOR_TEMP_ENTITY_ID, all_states
                )

                if current_indoor is not None:
                    actual_indoor_change = current_indoor - last_indoor_temp

                    # Create learning features with the actual outlet temp
                    # that was applied
                    # Fix: Handle case where last_run_features might be stored as string
                    if isinstance(last_run_features, str):
                        logging.error("CRITICAL: last_run_features corrupted as string - attempting to recover")
                        try:
                            # Try to parse as JSON if it's a string representation
                            import json
                            last_run_features = json.loads(last_run_features)
                            logging.info("✅ Successfully recovered features from JSON string")
                        except (json.JSONDecodeError, TypeError):
                            logging.error("❌ Cannot recover features from string, using empty dict")
                            last_run_features = {}
                    
                    if isinstance(last_run_features, pd.DataFrame):
                        learning_features = last_run_features.copy().to_dict(
                            orient="records"
                        )[0]
                    elif isinstance(last_run_features, dict):
                        learning_features = last_run_features.copy()
                    else:
                        learning_features = last_run_features.copy() if last_run_features else {}

                    learning_features["outlet_temp"] = actual_applied_temp
                    learning_features["outlet_temp_sq"] = (
                        actual_applied_temp ** 2
                    )
                    learning_features["outlet_temp_cub"] = (
                        actual_applied_temp ** 3
                    )

                    # Online learning is now handled by ThermalEquilibriumModel in model_wrapper
                    try:
                        # Import and create model wrapper for learning
                        from .model_wrapper import get_enhanced_model_wrapper
                        
                        # Create wrapper instance
                        wrapper = get_enhanced_model_wrapper()
                        
                        # Prepare prediction context for learning
                        prediction_context = {
                            'outlet_temp': actual_applied_temp,
                            'outdoor_temp': learning_features.get('outdoor_temp', 10.0),
                            'pv_power': learning_features.get('pv_now', 0.0),
                            'fireplace_on': learning_features.get('fireplace_on', 0.0),
                            'tv_on': learning_features.get('tv_on', 0.0)
                        }
                        
                        # Calculate what the model predicted vs actual result
                        predicted_change = 0.0  # Model's prediction for indoor temp change
                        actual_change = actual_indoor_change
                        
                        # Call the learning feedback method that was missing!
                        wrapper.learn_from_prediction_feedback(
                            predicted_temp=last_indoor_temp + predicted_change,
                            actual_temp=current_indoor,
                            prediction_context=prediction_context,
                            timestamp=datetime.now().isoformat()
                        )
                        
                        logging.info(
                            "✅ Online learning: applied_temp=%.1f°C, "
                            "actual_change=%.3f°C, cycle=%d",
                            actual_applied_temp,
                            actual_indoor_change,
                            wrapper.cycle_count
                        )
                    except Exception as e:
                        logging.warning(
                            "Online learning failed: %s", e, exc_info=True
                        )
                    
                    # Shadow mode error tracking
                    # Track what error ML and heat curve made
                    shadow_mode_active = (
                        config.TARGET_OUTLET_TEMP_ENTITY_ID !=
                        config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID
                    )
                    
                    # Shadow mode error tracking removed - handled by ThermalEquilibriumModel
                    if shadow_mode_active and actual_applied_temp != last_final_temp_stored:
                        logging.debug(
                            "Shadow mode: ML would have set %.1f°C, HC set %.1f°C",
                            last_final_temp_stored, actual_applied_temp
                        )
                else:
                    logging.debug(
                        "Skipping online learning: "
                        "current indoor temp unavailable"
                    )
            else:
                logging.debug(
                    "Skipping online learning: no data from previous cycle"
                )

            # --- Check for blocking modes (DHW, Defrost) ---
            # Skip the control logic if the heat pump is busy with other tasks
            # like heating domestic hot water (DHW) or defrosting.
            # blocking_entities already defined outside try block
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
                        # Determine whether the outlet is currently hotter or colder than
                        # the last target. For DHW-like events the outlet will be hotter;
                        # for defrost it can be colder due to reversed flow.
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
                                if wait_for_cooling:
                                    # Outlet hot (DHW): use aggressive cool-down target
                                    # to speed up recovery without overshooting
                                    grace_target = (
                                        last_final_temp +
                                        config.MAX_TEMP_CHANGE_PER_CYCLE
                                    )
                                else:
                                    # Outlet cold (defrost): restore exact pre-defrost
                                    # target and wait for full recovery to prevent
                                    # sawtooth patterns
                                    grace_target = last_final_temp
                                
                                logging.info(
                                    "Restoring outlet target: %.1f°C "
                                    "(last=%.1f°C, actual=%.1f°C, %s)",
                                    grace_target,
                                    last_final_temp,
                                    actual_outlet_temp_start,
                                    (
                                        "cool-down"
                                        if wait_for_cooling
                                        else "warm-up"
                                    ),
                                )
                                ha_client.set_state(
                                    config.TARGET_OUTLET_TEMP_ENTITY_ID,
                                    grace_target,
                                    get_sensor_attributes(
                                        config.TARGET_OUTLET_TEMP_ENTITY_ID
                                    ),
                                    round_digits=0,
                                )

                                start_time = time.time()
                                max_seconds = config.GRACE_PERIOD_MAX_MINUTES * 60
                                logging.info(
                                    "Grace period: initial outlet delta %.1f°C "
                                    "-> waiting for %s (timeout %d min).",
                                    delta0,
                                    (
                                        "actual <= target"
                                        if wait_for_cooling
                                        else "actual >= target"
                                    ),
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
                                    # to <= grace_target. If it started colder (defrost), wait until
                                    # it warms to >= grace_target.
                                    if wait_for_cooling and actual_outlet_temp <= grace_target:
                                        logging.info(
                                            "Actual outlet temp (%.1f°C) has cooled to or below"
                                            " grace target (%.1f°C). Resuming control.",
                                            actual_outlet_temp,
                                            grace_target,
                                        )
                                        break
                                    if (not wait_for_cooling) and actual_outlet_temp >= grace_target:
                                        logging.info(
                                            "Actual outlet temp (%.1f°C) has warmed to or above"
                                            " grace target (%.1f°C). Resuming control.",
                                            actual_outlet_temp,
                                            grace_target,
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
                                        "Waiting for outlet to %s grace target "
                                        "(current: %.1f°C, target: %.1f°C). "
                                        "Elapsed: %d/%d min",
                                        (
                                            "cool to"
                                            if wait_for_cooling
                                            else "warm to"
                                        ),
                                        actual_outlet_temp,
                                        grace_target,
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

            # --- Check if heating system is active ---
            # Skip if the climate entity is not in 'heat' mode
            heating_state = ha_client.get_state(
                config.HEATING_STATUS_ENTITY_ID, all_states
            )
            if heating_state not in ("heat", "auto"):
                logging.info(
                    "Heating system not active (state: %s), skipping cycle.",
                    heating_state
                )
                try:
                    attributes_state = get_sensor_attributes(
                        "sensor.ml_heating_state"
                    )
                    attributes_state.update(
                        {
                            "state_description": f"Heating off ({heating_state})",
                            "heating_state": heating_state,
                            "last_updated": datetime.now(
                                timezone.utc
                            ).isoformat(),
                        }
                    )
                    ha_client.set_state(
                        "sensor.ml_heating_state",
                        6,
                        attributes_state,
                        round_digits=None,
                    )
                except Exception:
                    logging.debug(
                        "Failed to write HEATING_OFF state to HA.",
                        exc_info=True
                    )
                time.sleep(300)
                continue

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

            # --- Step 1: State Retrieval ---
            # Heat balance controller doesn't use prediction history anymore.
            # Removed prediction_history retrieval.

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

            features, outlet_history = build_physics_features(
                ha_client, influx_service
            )
            if features is None:
                logging.warning("Feature building failed, skipping cycle.")
                time.sleep(300)
                continue

            # --- Step 3: Prediction ---
            # Use the Enhanced Model Wrapper for simplified outlet temperature prediction.
            # This replaces the complex Heat Balance Controller with a single prediction call.
            error_target_vs_actual = target_indoor_temp - prediction_indoor_temp
            
            suggested_temp, confidence, metadata = simplified_outlet_prediction(
                features,
                prediction_indoor_temp,
                target_indoor_temp
            )
            final_temp = suggested_temp
            
            # Log simplified prediction info
            logging.info(
                "Enhanced Model Wrapper: temp=%.1f°C, error=%.3f°C, confidence=%.3f",
                suggested_temp,
                abs(error_target_vs_actual),
                confidence
            )

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

            # Final prediction is now handled by ThermalEquilibriumModel in model_wrapper
            # Use confidence metadata for predicted indoor temp if available
            predicted_indoor = metadata.get("predicted_indoor", prediction_indoor_temp)

            # --- Step 4: Update Home Assistant and Log ---
            # The calculated `final_temp` is sent to Home Assistant to control
            # the boiler. Other metrics like model confidence, MAE, and
            # feature importances are also published to HA for monitoring.
            # In shadow mode, skip all HA sensor updates to avoid interference.
            if config.SHADOW_MODE:
                logging.info("🔍 SHADOW MODE: ML prediction calculated but not applied to heating system")
                logging.info("   Final temp: %.1f°C (calculated but not sent to HA)", final_temp)
            else:
                # Apply smart rounding: test floor vs ceiling to see which gets closer to target
                floor_temp = np.floor(final_temp)
                ceiling_temp = np.ceil(final_temp)
                
                if floor_temp == ceiling_temp:
                    # Already an integer
                    smart_rounded_temp = int(final_temp)
                    logging.debug(f"Smart rounding: {final_temp:.2f}°C is already integer")
                else:
                    # Test both options using the thermal model to see which gets closer to target
                    try:
                        from .model_wrapper import get_enhanced_model_wrapper
                        wrapper = get_enhanced_model_wrapper()
                        
                        # Create test contexts for floor and ceiling temperatures
                        test_context_floor = {
                            'outlet_temp': floor_temp,
                            'outdoor_temp': outdoor_temp,
                            'pv_power': features.get('pv_now', 0.0) if hasattr(features, 'get') else 0.0,
                            'fireplace_on': fireplace_on,
                            'tv_on': features.get('tv_on', 0.0) if hasattr(features, 'get') else 0.0
                        }
                        
                        test_context_ceiling = test_context_floor.copy()
                        test_context_ceiling['outlet_temp'] = ceiling_temp
                        
                        # Get predictions for both temperatures
                        floor_predicted = wrapper.predict_indoor_temp(
                            outlet_temp=floor_temp,
                            outdoor_temp=outdoor_temp,
                            pv_power=test_context_floor['pv_power'],
                            fireplace_on=test_context_floor['fireplace_on'],
                            tv_on=test_context_floor['tv_on']
                        )
                        ceiling_predicted = wrapper.predict_indoor_temp(
                            outlet_temp=ceiling_temp,
                            outdoor_temp=outdoor_temp,
                            pv_power=test_context_ceiling['pv_power'],
                            fireplace_on=test_context_ceiling['fireplace_on'],
                            tv_on=test_context_ceiling['tv_on']
                        )
                        
                        # PHASE 5 FIX: Handle None returns from predict_indoor_temp
                        if floor_predicted is None or ceiling_predicted is None:
                            logging.warning("Smart rounding: predict_indoor_temp returned None, using fallback")
                            smart_rounded_temp = round(final_temp)
                            logging.info(f"Smart rounding fallback: {final_temp:.2f}°C → {smart_rounded_temp}°C")
                        else:
                            # Calculate errors from target
                            floor_error = abs(floor_predicted - target_indoor_temp)
                            ceiling_error = abs(ceiling_predicted - target_indoor_temp)
                            
                            if floor_error <= ceiling_error:
                                smart_rounded_temp = int(floor_temp)
                                chosen = "floor"
                            else:
                                smart_rounded_temp = int(ceiling_temp)
                                chosen = "ceiling"
                            
                            logging.info(
                                f"Smart rounding: {final_temp:.2f}°C → {smart_rounded_temp}°C "
                                f"(chose {chosen}: floor→{floor_predicted:.2f}°C [err={floor_error:.3f}], "
                                f"ceiling→{ceiling_predicted:.2f}°C [err={ceiling_error:.3f}], "
                                f"target={target_indoor_temp:.1f}°C)"
                            )
                    except Exception as e:
                        # Fallback to regular rounding if smart rounding fails
                        smart_rounded_temp = round(final_temp)
                        logging.warning(f"Smart rounding failed ({e}), using regular rounding: {final_temp:.2f}°C → {smart_rounded_temp}°C")
                
                logging.debug("Setting target outlet temp")
                ha_client.set_state(
                    config.TARGET_OUTLET_TEMP_ENTITY_ID,
                    smart_rounded_temp,
                    get_sensor_attributes(config.TARGET_OUTLET_TEMP_ENTITY_ID),
                    round_digits=None,  # No additional rounding needed - already handled by smart rounding
                )

            # --- Log Metrics ---
            # Metrics logging now handled by ThermalEquilibriumModel in model_wrapper
            if not config.SHADOW_MODE:
                logging.debug("Logging thermal model metrics")
                # Confidence is logged via simplified_outlet_prediction metadata
                # Feature importances are handled by ThermalEquilibriumModel
                # Learning metrics are exported by ThermalEquilibriumModel

            # --- Update ML State sensor ---
            # Skip ML state sensor updates in shadow mode
            if not config.SHADOW_MODE:
                try:
                    # Get thermal model trust metrics from ThermalEquilibriumModel
                    thermal_trust_metrics = metadata.get("thermal_trust_metrics", {})
                    
                    attributes_state = get_sensor_attributes(
                        "sensor.ml_heating_state"
                    )
                    attributes_state.update(
                        {
                            "state_description": "Confidence - Too Low"
                            if confidence < config.CONFIDENCE_THRESHOLD
                            else "OK - Prediction done",
                            "confidence": round(confidence, 4),
                            "suggested_temp": round(suggested_temp, 2),
                            "final_temp": round(final_temp, 2),
                            "predicted_indoor": round(predicted_indoor, 2),
                            "last_prediction_time": (
                                datetime.now(timezone.utc).isoformat()
                            ),
                            "temperature_error": round(
                                abs(error_target_vs_actual), 3
                            ),
                            # Note: ThermalEquilibriumModel trust metrics moved to 
                            # sensor.ml_heating_learning to eliminate redundancy
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
            else:
                logging.debug("🔍 SHADOW MODE: Skipping ML state sensor updates")

            # --- Shadow Mode Control via HA Boolean ---
            # Read input_boolean.ml_heating to determine control mode
            ml_heating_enabled = ha_client.get_state(
                config.ML_HEATING_CONTROL_ENTITY_ID, 
                all_states, 
                is_binary=True
            )
            
            # Shadow mode is active when:
            # - Config SHADOW_MODE=true (override), OR
            # - ML heating boolean is OFF/unavailable
            if ml_heating_enabled is None:
                logging.warning(
                    "Cannot read %s, defaulting to shadow mode for safety",
                    config.ML_HEATING_CONTROL_ENTITY_ID
                )
                ml_heating_enabled = False
            
            effective_shadow_mode = config.SHADOW_MODE or not ml_heating_enabled
            
            if config.SHADOW_MODE:
                logging.info("🔍 SHADOW MODE: Enabled via config (SHADOW_MODE=true)")
            elif not ml_heating_enabled:
                logging.info(
                    "🔍 SHADOW MODE: ML control disabled via %s", 
                    config.ML_HEATING_CONTROL_ENTITY_ID
                )
            else:
                logging.info(
                    "✅ ACTIVE MODE: ML controlling heating via %s", 
                    config.ML_HEATING_CONTROL_ENTITY_ID
                )
            
            # --- Shadow Mode Comparison Logging ---
            if effective_shadow_mode:
                # Read what the heat curve actually set
                heat_curve_temp = ha_client.get_state(
                    config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID, all_states
                )
                
                if heat_curve_temp is not None and heat_curve_temp != final_temp:
                    # Simple comparison without model prediction
                    logging.info(
                        "SHADOW MODE: ML would set %.1f°C, HC set %.1f°C | Target: %.1f°C",
                        final_temp,
                        heat_curve_temp,
                        target_indoor_temp,
                    )

            # Shadow metrics now handled by ThermalEquilibriumModel
            
            # Use the actual rounded temperature that was applied to HA
            applied_temp = smart_rounded_temp if not config.SHADOW_MODE else final_temp
            
            log_message = (
                "Target: %.1f°C | Suggested: %.1f°C | Applied: %.1f°C | "
                "Actual Indoor: %.2f°C | Predicted Indoor: %.2f°C | "
                "Confidence: %.3f"
            )
            logging.info(
                log_message,
                target_indoor_temp,
                suggested_temp,
                applied_temp,
                actual_indoor,
                predicted_indoor,
                confidence,
            )

            # --- Step 6: State Persistence for Next Run ---
            # Model saving is now handled by ThermalEquilibriumModel in model_wrapper

            # The features and indoor temperature from the *current* run are
            # saved to a file. This data will be loaded at the start of the
            # next loop iteration to be used in the "Online Learning" step.
            # Note: We save final_temp here, but will read the actual applied
            # temp from ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID at the start of
            # the next cycle (after it has had time to update).
            state_to_save = {
                "last_run_features": features,
                "last_indoor_temp": actual_indoor,
                "last_avg_other_rooms_temp": avg_other_rooms_temp,
                "last_fireplace_on": fireplace_on,
                "last_final_temp": final_temp,
                "last_is_blocking": is_blocking,
                "last_blocking_reasons": (
                    blocking_reasons if is_blocking else []
                ),
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

        # CYCLE END DEBUG LOGGING
        cycle_end_time = time.time()
        cycle_duration = cycle_end_time - cycle_start_time
        cycle_end_datetime = datetime.now()
        
        logging.info(f"✅ CYCLE {cycle_number} END: {cycle_end_datetime.strftime('%H:%M:%S')} "
                    f"(duration: {cycle_duration:.1f}s)")
        
        last_cycle_end_time = cycle_end_time

        # Poll for blocking events during the idle period so defrost starts/ends
        # are detected quickly. This call will block until the next cycle is
        # due, or until a blocking event starts or ends.
        logging.info(f"💤 POLLING START: Waiting {config.CYCLE_INTERVAL_MINUTES}min "
                    f"until next cycle...")
        poll_for_blocking(ha_client, state, blocking_entities)
        
        poll_end_time = time.time()
        poll_duration = poll_end_time - cycle_end_time
        logging.info(f"⏰ POLLING END: Waited {poll_duration/60:.1f}min, starting next cycle...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Physics-Based Heating Controller"
    )
    parser.add_argument(
        "--calibrate-physics",
        action="store_true",
        help="Calibrate the physics model using historical temperature data.",
    )
    parser.add_argument(
        "--validate-physics",
        action="store_true",
        help="Test physics model behavior across temperature ranges and exit.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    parsed_args = parser.parse_args()
    main(parsed_args)
