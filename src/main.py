"""
This module is the central entry point and main control loop for the
application.

It orchestrates the entire process of data collection, prediction, and action
using the enhanced physics-based heating model. The script operates in a
continuous loop, performing the following key steps in each iteration:

1.  **Initialization**: Loads the physics model and application state.
2.  **Data Fetching**: Gathers the latest sensor data from Home Assistant.
3.  **Feature Engineering**: Builds a feature set from current and historical
    data.
4.  **Prediction**: Uses the physics model to find the optimal heating
    temperature.
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
from .thermal_constants import PhysicsConstants
from .physics_features import build_physics_features
from .ha_client import create_ha_client, get_sensor_attributes
from .influx_service import create_influx_service
from .model_wrapper import simplified_outlet_prediction
from .physics_calibration import (
    train_thermal_equilibrium_model,
    validate_thermal_model,
)
from .physics_features import calculate_thermodynamic_metrics
from .state_manager import load_state, save_state
from .heating_controller import (
    BlockingStateManager,
    SensorDataManager,
    HeatingSystemStateChecker,
)
from .sensor_buffer import SensorBuffer


def main():
    """
    The main function that orchestrates the heating control logic.

    This function initializes the system, enters a continuous loop to
    monitor and control the heating, and handles command-line arguments
    for modes like initial training.
    """
    parser = argparse.ArgumentParser(description="Heating Controller")
    parser.add_argument(
        "--calibrate-physics",
        action="store_true",
        help="Calibrate the physics model.",
    )
    parser.add_argument(
        "--validate-physics",
        action="store_true",
        help="Test model behavior and exit.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging."
    )
    parser.add_argument(
        "--list-backups", action="store_true", help="List available backups."
    )
    parser.add_argument(
        "--restore-backup", type=str, help="Restore from a backup file."
    )
    args = parser.parse_args()
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

    # --- Sensor Buffer Initialization ---
    # Initialize the circular buffer for sensor smoothing
    sensor_buffer = SensorBuffer(max_age_minutes=120)

    # Hydrate buffer from InfluxDB (Startup only)
    try:
        logging.info("💧 Hydrating sensor buffer from InfluxDB...")
        # Define sensors to hydrate
        hydration_sensors = [
            config.INDOOR_TEMP_ENTITY_ID,
            config.ACTUAL_OUTLET_TEMP_ENTITY_ID,
            config.TARGET_OUTLET_TEMP_ENTITY_ID,
            config.OUTDOOR_TEMP_ENTITY_ID,
            config.INLET_TEMP_ENTITY_ID,
            config.FLOW_RATE_ENTITY_ID,
        ]

        # Fetch raw history
        history_data = influx_service.fetch_recent_history(
            hydration_sensors,
            lookback_minutes=120
        )

        # Hydrate the buffer
        sensor_buffer.hydrate(history_data)
        logging.info("✅ Sensor buffer hydrated successfully")

    except Exception as e:
        logging.warning(
            f"⚠️ Buffer hydration failed: {e}. "
            "Starting with empty buffer (Cold Start Mode)."
        )

    # --- Shadow Mode Status ---
    if config.SHADOW_MODE:
        logging.info(
            "🔍 SHADOW MODE ENABLED: ML will observe and learn without "
            "affecting heating control"
        )
        logging.info("   - ML predictions calculated but not sent to HA")
        logging.info(
            "   - No HA sensor updates (confidence, MAE, RMSE, state)"
        )
        logging.info(
            "   - Learning from heat curve's actual control decisions"
        )
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

                logging.info(
                    "✅ Previous thermal state backed up: %s",
                    os.path.basename(backup_path),
                )
            else:
                logging.info("ℹ️ No existing thermal state found to backup")

            result = train_thermal_equilibrium_model()
            if result:
                logging.info("✅ Thermal model calibrated successfully!")
                logging.info(
                    "🔄 Restart ml_heating to use trained thermal model"
                )
            else:
                logging.error("❌ Thermal model calibration failed")
        except Exception as e:
            logging.error(
                "Thermal model calibration error: %s", e, exc_info=True
            )
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
            logging.error(
                "Thermal model validation error: %s", e, exc_info=True
            )
        return

    if args.list_backups:
        from .unified_thermal_state import get_thermal_state_manager
        import json
        state_manager = get_thermal_state_manager()
        backups = state_manager.list_backups()
        if backups:
            print("Available backups:")
            # print backups in a json format so it is easy to parse
            print(json.dumps(backups, indent=2, default=str))
        else:
            print("No backups found.")
        return

    if args.restore_backup:
        from .unified_thermal_state import get_thermal_state_manager
        state_manager = get_thermal_state_manager()
        success, message = state_manager.restore_from_backup(
            args.restore_backup
        )
        if success:
            print(f"Successfully restored from backup: {args.restore_backup}")
            print(message)
        else:
            print(f"Failed to restore from backup: {args.restore_backup}")
            print(message)
        return

    # --- Main Control Loop ---
    # This loop runs indefinitely, performing one full cycle of learning and
    # prediction every 5 minutes.

    # Initialize the model and export initial metrics to HA
    from .model_wrapper import get_enhanced_model_wrapper

    wrapper = get_enhanced_model_wrapper()
    if not config.SHADOW_MODE:
        try:
            wrapper.export_metrics_to_ha()
            logging.info("✅ Initial metrics exported to HA successfully.")
        except Exception as e:
            logging.error(
                f"❌ FAILED to export initial metrics to HA: {e}", exc_info=True
            )
    # Define blocking_entities outside try block so it's available in
    # exception handler
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
                logging.debug(
                    f"🔄 CYCLE {cycle_number} START: "
                    f"{cycle_start_datetime.strftime('%H:%M:%S')} "
                    f"(interval: {interval_since_last/60:.1f}min since "
                    f"last cycle)"
                )
            else:
                logging.debug(
                    f"🔄 CYCLE {cycle_number} START: "
                    f"{cycle_start_datetime.strftime('%H:%M:%S')} "
                    f"(first cycle)"
                )
            # Initialize shadow mode tracking for this cycle
            shadow_mode_active = (
                config.TARGET_OUTLET_TEMP_ENTITY_ID
                != config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID
            )

            # Load the application state at the beginning of each cycle.
            state = load_state()
            # Create a new Home Assistant client for each cycle.
            ha_client = create_ha_client()
            # Fetch all states from Home Assistant at once to minimize
            # API calls.
            all_states = ha_client.get_all_states()

            # --- Update Sensor Buffer ---
            if all_states:
                current_time = datetime.now(timezone.utc)
                
                # Helper to safely get float state
                def get_float_state(entity_id):
                    try:
                        val = ha_client.get_state(entity_id, all_states)
                        return float(val) if val is not None else None
                    except (ValueError, TypeError):
                        return None

                # Push new readings to buffer
                buffer_updates = {
                    config.INDOOR_TEMP_ENTITY_ID: get_float_state(
                        config.INDOOR_TEMP_ENTITY_ID
                    ),
                    config.ACTUAL_OUTLET_TEMP_ENTITY_ID: get_float_state(
                        config.ACTUAL_OUTLET_TEMP_ENTITY_ID
                    ),
                    config.TARGET_OUTLET_TEMP_ENTITY_ID: get_float_state(
                        config.TARGET_OUTLET_TEMP_ENTITY_ID
                    ),
                    config.OUTDOOR_TEMP_ENTITY_ID: get_float_state(
                        config.OUTDOOR_TEMP_ENTITY_ID
                    ),
                    config.INLET_TEMP_ENTITY_ID: get_float_state(
                        config.INLET_TEMP_ENTITY_ID
                    ),
                    config.FLOW_RATE_ENTITY_ID: get_float_state(
                        config.FLOW_RATE_ENTITY_ID
                    ),
                }

                for entity_id, value in buffer_updates.items():
                    if value is not None:
                        sensor_buffer.add_reading(
                            entity_id, value, current_time
                        )

                # --- Real-time Thermodynamic Sensors ---
                # Calculate and export COP and Thermal Power immediately so
                # they are available even if the cycle is skipped (e.g.
                # blocking/idle).
                try:
                    # Get power consumption which isn't in the buffer updates
                    power_consumption = get_float_state(
                        config.POWER_CONSUMPTION_ENTITY_ID
                    )

                    # Use values we just fetched for the buffer
                    current_outlet = buffer_updates.get(
                        config.ACTUAL_OUTLET_TEMP_ENTITY_ID
                    )
                    current_inlet = buffer_updates.get(
                        config.INLET_TEMP_ENTITY_ID
                    )
                    current_flow = buffer_updates.get(
                        config.FLOW_RATE_ENTITY_ID
                    )

                    # Only calculate if we have the minimum required data
                    # (outlet temp)
                    if current_outlet is not None:
                        thermo_metrics = calculate_thermodynamic_metrics(
                            outlet_temp=current_outlet,
                            inlet_temp=current_inlet,
                            flow_rate=current_flow,
                            power_consumption=power_consumption
                        )

                        # Export to Home Assistant
                        # 1. COP
                        ha_client.set_state(
                            "sensor.ml_heating_cop_realtime",
                            thermo_metrics["cop_realtime"],
                            {
                                "friendly_name": "ML Heating COP (Realtime)",
                                "unit_of_measurement": "COP",
                                "icon": "mdi:heat-pump",
                                "device_class": "power_factor",
                                "state_class": "measurement"
                            },
                            round_digits=2
                        )

                        # 2. Thermal Power
                        ha_client.set_state(
                            "sensor.ml_heating_thermal_power",
                            thermo_metrics["thermal_power_kw"],
                            {
                                "friendly_name": "ML Heating Thermal Power",
                                "unit_of_measurement": "kW",
                                "icon": "mdi:flash",
                                "device_class": "power",
                                "state_class": "measurement"
                            },
                            round_digits=3
                        )

                        logging.debug(
                            "Thermodynamic sensors updated: COP=%.2f, "
                            "Power=%.3fkW",
                            thermo_metrics["cop_realtime"],
                            thermo_metrics["thermal_power_kw"]
                        )
                except Exception as e:
                    logging.warning(
                        "Failed to update thermodynamic sensors: %s", e
                    )

            # --- Determine Shadow Mode Early (needed for grace period) ---
            # Read input_boolean.ml_heating to determine control mode
            ml_heating_enabled = None
            if all_states:
                ml_heating_enabled = ha_client.get_state(
                    config.ML_HEATING_CONTROL_ENTITY_ID,
                    all_states,
                    is_binary=True
                )

            # Shadow mode is active when:
            # - Config SHADOW_MODE=true (override), OR
            # - ML heating boolean is OFF/unavailable
            if ml_heating_enabled is None:
                if all_states:  # Only warn if we could fetch states
                    logging.warning(
                        "Cannot read %s, defaulting to shadow mode",
                        config.ML_HEATING_CONTROL_ENTITY_ID,
                    )
                ml_heating_enabled = False

            effective_shadow_mode = (
                config.SHADOW_MODE or not ml_heating_enabled
            )

            if not all_states:
                logging.warning(
                    "Could not fetch states from HA, skipping cycle."
                )
                # Emit NETWORK_ERROR state to Home Assistant
                try:
                    ha_client = create_ha_client()
                    attributes_state = get_sensor_attributes(
                        "sensor.ml_heating_state"
                    )
                    attributes_state.update(
                        {
                            "state_description": "Network Error",
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
                time.sleep(PhysicsConstants.RETRY_DELAY_SECONDS)
                continue

            # --- Check for blocking modes (DHW, Defrost) ---
            # Skip the control logic if the heat pump is busy with other
            # tasks like heating domestic hot water (DHW) or defrosting.
            # blocking_entities already defined outside try block. Build a
            # list of active blocking reasons so we can distinguish
            # DHW-like (long) blockers from short ones like defrost.
            blocking_manager = BlockingStateManager()
            is_blocking, blocking_reasons = (
                blocking_manager.check_blocking_state(
                    ha_client, all_states
                )
            )

            # --- Step 1: Online Learning from Previous Cycle ---
            # Learn from the results of the previous cycle. This allows the
            # model to continuously adapt to the actual house behavior,
            # whether running in active mode (model controls heating) or
            # shadow mode (heat curve controls heating).
            last_run_features = state.get("last_run_features")
            last_indoor_temp = state.get("last_indoor_temp")
            last_final_temp_stored = state.get("last_final_temp")

            if (
                last_run_features is not None
                and last_indoor_temp is not None
                and last_final_temp_stored is not None
            ):

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
                    actual_applied_temp = last_final_temp_stored

                # Get current indoor temperature to calculate actual change
                current_indoor = ha_client.get_state(
                    config.INDOOR_TEMP_ENTITY_ID, all_states
                )

                if current_indoor is not None:
                    actual_indoor_change = current_indoor - last_indoor_temp

                    # Create learning features with the actual outlet temp
                    # that was applied
                    # Handle case where last_run_features might be stored as
                    # string
                    if isinstance(last_run_features, str):
                        logging.error(
                            "ERROR: last_run_features corrupted as string - "
                            "attempting to recover"
                        )
                        try:
                            # Try to parse as JSON if it's a string
                            # representation
                            import json

                            last_run_features = json.loads(last_run_features)
                            logging.info(
                                "✅ Recovered features from JSON string"
                            )
                        except (json.JSONDecodeError, TypeError):
                            logging.error(
                                "❌ Cannot recover features from string, "
                                "using empty dict"
                            )
                            last_run_features = {}

                    if isinstance(last_run_features, pd.DataFrame):
                        learning_features = last_run_features.copy().to_dict(
                            orient="records"
                        )[0]
                    elif isinstance(last_run_features, dict):
                        learning_features = last_run_features.copy()
                    else:
                        learning_features = (
                            last_run_features.copy()
                            if last_run_features
                            else {}
                        )

                    learning_features["outlet_temp"] = actual_applied_temp
                    learning_features["outlet_temp_sq"] = (
                        actual_applied_temp**2
                    )
                    learning_features["outlet_temp_cub"] = (
                        actual_applied_temp**3
                    )

                    # Online learning is now handled by
                    # ThermalEquilibriumModel in model_wrapper
                    try:
                        # Import and create model wrapper for learning
                        from .model_wrapper import get_enhanced_model_wrapper

                        # Create wrapper instance
                        wrapper = get_enhanced_model_wrapper()

                        # Prepare prediction context for learning
                        prediction_context = {
                            "outlet_temp": actual_applied_temp,
                            "outdoor_temp": learning_features.get(
                                "outdoor_temp", 10.0
                            ),
                            "pv_power": learning_features.get("pv_now", 0.0),
                            "fireplace_on": learning_features.get(
                                "fireplace_on", 0.0
                            ),
                            "tv_on": learning_features.get("tv_on", 0.0),
                            "current_indoor": last_indoor_temp,
                            "thermal_power": learning_features.get(
                                "thermal_power_kw", None
                            ),
                            # Pass auxiliary heat if available in features
                            "auxiliary_heat": learning_features.get(
                                "total_auxiliary_heat_kw", 0.0
                            ),
                        }

                        # FIXED SHADOW MODE LEARNING: Only learn from shadow
                        # mode when actually in shadow mode
                        # In ACTIVE MODE: Learn from ML's own decisions
                        # (even with smart rounding)
                        # In SHADOW MODE: Learn from heat curve decisions
                        was_shadow_mode_cycle = effective_shadow_mode

                        try:
                            # UNIFIED LEARNING: Always use trajectory
                            # prediction for learning to ensure consistency
                            # with the control loop's prediction method.
                            was_shadow_mode_cycle = effective_shadow_mode

                            if was_shadow_mode_cycle:
                                learning_mode = "shadow_mode_hc_trajectory"
                                log_msg = (
                                    "🔍 SHADOW MODE LEARNING (trajectory): "
                                    "Predicting indoor temp from heat "
                                    f"curve's {actual_applied_temp}°C "
                                    "outlet setting"
                                )
                            else:
                                learning_mode = "active_mode_ml_trajectory"
                                log_msg = (
                                    "🎯 ACTIVE MODE LEARNING (trajectory): "
                                    "Verifying ML prediction accuracy for "
                                    f"{actual_applied_temp}°C outlet setting"
                                )
                            logging.debug(log_msg)

                            trajectory = (
                                wrapper.thermal_model
                                .predict_thermal_trajectory(
                                    current_indoor=last_indoor_temp,
                                    target_indoor=last_indoor_temp,
                                    outlet_temp=actual_applied_temp,
                                    outdoor_temp=prediction_context.get(
                                        "outdoor_temp", 10.0
                                    ),
                                    time_horizon_hours=(
                                        config.CYCLE_INTERVAL_MINUTES / 60.0
                                    ),
                                    pv_power=prediction_context.get(
                                        "pv_power", 0.0
                                    ),
                                    fireplace_on=prediction_context.get(
                                        "fireplace_on", 0.0
                                    ),
                                    tv_on=prediction_context.get("tv_on", 0.0),
                                    thermal_power=prediction_context.get(
                                        "thermal_power"
                                    ),
                                )
                            )

                            predicted_indoor_temp = (
                                trajectory["trajectory"][0]
                                if trajectory and trajectory.get("trajectory")
                                else last_indoor_temp
                            )

                            if predicted_indoor_temp is None:
                                logging.warning(
                                    "Skipping online learning (%s): "
                                    "prediction returned None",
                                    learning_mode
                                )
                                continue

                            model_predicted_temp = predicted_indoor_temp

                        except Exception as e:
                            logging.warning(
                                "Skipping online learning: thermal model "
                                f"prediction error: {e}"
                            )
                            continue

                        # Enhanced prediction context with learning mode info
                        enhanced_prediction_context = prediction_context.copy()
                        enhanced_prediction_context[
                            "learning_mode"
                        ] = learning_mode
                        enhanced_prediction_context[
                            "was_shadow_mode_cycle"
                        ] = was_shadow_mode_cycle
                        enhanced_prediction_context[
                            "ml_calculated_temp"
                        ] = last_final_temp_stored
                        enhanced_prediction_context[
                            "hc_applied_temp"
                        ] = actual_applied_temp

                        # Call the learning feedback method with the correct
                        # prediction context
                        wrapper.learn_from_prediction_feedback(
                            predicted_temp=model_predicted_temp,
                            actual_temp=current_indoor,
                            prediction_context=enhanced_prediction_context,
                            timestamp=datetime.now().isoformat(),
                            is_blocking_active=is_blocking,
                        )

                        logging.debug(
                            "✅ Online learning: applied_temp=%.1f°C, "
                            "actual_change=%.3f°C, cycle=%d",
                            actual_applied_temp,
                            actual_indoor_change,
                            wrapper.cycle_count,
                        )
                    except Exception as e:
                        logging.warning(
                            "Online learning failed: %s", e, exc_info=True
                        )

                    # Shadow mode error tracking
                    # Track what error ML and heat curve made
                    shadow_mode_active = (
                        config.TARGET_OUTLET_TEMP_ENTITY_ID
                        != config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID
                    )

                    # Shadow mode error tracking removed - handled by
                    # ThermalEquilibriumModel Only log shadow mode
                    # comparison when actually in shadow mode (not active)
                    effective_shadow_mode = (
                        config.SHADOW_MODE
                        or not ha_client.get_state(
                            config.ML_HEATING_CONTROL_ENTITY_ID,
                            all_states,
                            is_binary=True,
                        )
                    )
                if (
                    effective_shadow_mode
                    and actual_applied_temp != last_final_temp_stored
                ):
                    logging.debug(
                        "Shadow mode: ML would set %.1f°C, HC set %.1f°C",
                        last_final_temp_stored,
                        actual_applied_temp,
                    )
                else:
                    logging.debug(
                        "Skipping online learning: current indoor temp "
                        "unavailable"
                    )
            else:
                logging.debug(
                    "Skipping online learning: no data from previous cycle"
                )

            # --- Grace Period after Blocking ---
            # Use modular blocking state manager for cleaner code
            # organization
            blocking_manager = BlockingStateManager()
            if blocking_manager.handle_grace_period(
                ha_client, state, shadow_mode=effective_shadow_mode
            ):
                continue  # Skip cycle due to grace period

            # --- Check if heating system is active ---
            heating_checker = HeatingSystemStateChecker()
            if not heating_checker.check_heating_active(ha_client, all_states):
                time.sleep(PhysicsConstants.RETRY_DELAY_SECONDS)
                continue

            if is_blocking:
                logging.info(
                    "Blocking process active (DHW/Defrost), skipping."
                )
                try:
                    blocking_reasons = [
                        e
                        for e in blocking_entities
                        if ha_client.get_state(e, all_states, is_binary=True)
                    ]
                    attributes_state = get_sensor_attributes(
                        "sensor.ml_heating_state"
                    )
                    attributes_state.update(
                        {
                            "state_description": (
                                "Blocking activity - Skipping"
                            ),
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
                # Save the blocking state for the next cycle (preserve
                # last_final_temp and record which entities caused the
                # blocking so we can avoid learning from DHW-like cycles).
                save_state(
                    last_is_blocking=True,
                    last_final_temp=state.get("last_final_temp"),
                    last_blocking_reasons=blocking_reasons,
                    last_blocking_end_time=None,
                )
                time.sleep(PhysicsConstants.RETRY_DELAY_SECONDS)
                continue

            # --- Get current sensor values ---
            sensor_manager = SensorDataManager()
            sensor_data, missing_sensors = sensor_manager.get_sensor_data(
                ha_client, cycle_number
            )

            if missing_sensors:
                sensor_manager.handle_missing_sensors(
                    ha_client, missing_sensors
                )
                time.sleep(PhysicsConstants.RETRY_DELAY_SECONDS)
                continue

            # Unpack sensor data
            target_indoor_temp = sensor_data["target_indoor_temp"]
            actual_indoor = sensor_data["actual_indoor"]
            actual_outlet_temp = sensor_data["actual_outlet_temp"]
            avg_other_rooms_temp = sensor_data["avg_other_rooms_temp"]
            fireplace_on = sensor_data["fireplace_on"]
            outdoor_temp = sensor_data["outdoor_temp"]
            owm_temp = sensor_data["owm_temp"]

            # --- Step 1: State Retrieval ---
            # Heat balance controller doesn't use prediction history anymore.
            # Removed prediction_history retrieval.

            # --- Step 2: Feature Building ---
            # Gathers all necessary data points (current sensor values,
            # historical data from InfluxDB, etc.) and transforms them into a
            # feature vector. This vector is the input the model will use to
            # make its next prediction.
            if fireplace_on:
                prediction_indoor_temp = avg_other_rooms_temp
                logging.debug(
                    "Fireplace ON. Using avg other rooms temp for prediction."
                )
            else:
                prediction_indoor_temp = actual_indoor
                logging.debug(
                    "Fireplace is OFF. Using main indoor temp for prediction."
                )

            features, outlet_history = build_physics_features(
                ha_client, influx_service, sensor_buffer
            )
            # Handle both DataFrame and dict features properly
            if isinstance(features, pd.DataFrame):
                # Convert DataFrame to dict for safe access
                features_dict = (
                    features.iloc[0].to_dict() if not features.empty else {}
                )
            else:
                features_dict = features if isinstance(features, dict) else {}
            if features is None:
                logging.warning("Feature building failed, skipping cycle.")
                time.sleep(PhysicsConstants.RETRY_DELAY_SECONDS)
                continue

            # --- Step 3: Prediction ---
            # Use the Enhanced Model Wrapper for simplified outlet
            # temperature prediction. This replaces the complex Heat Balance
            # Controller with a single prediction call.
            error_target_vs_actual = (
                target_indoor_temp - prediction_indoor_temp
            )

            suggested_temp, confidence, metadata = (
                simplified_outlet_prediction(
                    features, prediction_indoor_temp, target_indoor_temp
                )
            )
            final_temp = suggested_temp

            # Log simplified prediction info
            logging.info(
                "Model Wrapper: temp=%.1f°C, error=%.3f°C, confidence=%.3f",
                suggested_temp,
                abs(error_target_vs_actual),
                confidence,
            )

            # --- Gradual Temperature Control (DISABLED) ---
            # if actual_outlet_temp is not None:
            #     max_change = config.MAX_TEMP_CHANGE_PER_CYCLE
            #     original_temp = final_temp  # Keep a copy for logging

            #     last_blocking_reasons =
            #     state.get("last_blocking_reasons", []) or []
            #     last_final_temp = state.get("last_final_temp")

            #     # DHW-like blockers that should keep the soft-start behavior
            #     dhw_like_blockers = {
            #         config.DHW_STATUS_ENTITY_ID,
            #         config.DISINFECTION_STATUS_ENTITY_ID,
            #         config.DHW_BOOST_HEATER_STATUS_ENTITY_ID,
            #     }

            #     # SHADOW MODE FIX: In shadow mode, the baseline for gradual
            #     # control should be the actual heat curve temperature from
            #     # the last cycle, not the ML's (potentially wrong)
            #     # prediction.
            #     if effective_shadow_mode:
            #         baseline = actual_outlet_temp
            #         logging.info(
            #             "Gradual control baseline in shadow mode set to "
            #             "actual_outlet_temp: %.1f°C", baseline
            #         )
            #     elif last_final_temp is not None:
            #         baseline = last_final_temp
            #         if any(b in dhw_like_blockers for b in
            #                   last_blocking_reasons):
            #             baseline = actual_outlet_temp
            #     else:
            #         baseline = actual_outlet_temp

            #     # Calculate the difference from the chosen baseline
            #     delta = final_temp - baseline
            #     # Clamp the delta to the maximum allowed change
            #     if abs(delta) > max_change:
            #         final_temp = baseline + np.clip(delta, -max_change,
            #                                           max_change)
            #         logging.info("--- Gradual Temperature Control ---")
            #         logging.info(
            #             "Change from baseline %.1f°C to suggested %.1f°C "
            #             "exceeds max change of %.1f°C. Capping at %.1f°C.",
            #             baseline,
            #             original_temp,
            #             max_change,
            #             final_temp,
            #         )

            # Final prediction is now handled by ThermalEquilibriumModel in
            # model_wrapper
            # Use confidence metadata for predicted indoor temp if available
            predicted_indoor = metadata.get(
                "predicted_indoor", prediction_indoor_temp
            )

            # --- Step 4: Update Home Assistant and Log ---
            # The calculated `final_temp` is sent to Home Assistant to
            # control the boiler. Other metrics like model confidence, MAE,
            # and feature importances are also published to HA for
            # monitoring. In shadow mode, skip all HA sensor updates to
            # avoid interference.

            if config.SHADOW_MODE:
                logging.info(
                    "🔍 SHADOW MODE: ML prediction calculated but not "
                    "applied to heating system"
                )
                logging.info(
                    "   Final temp: %.1f°C (calculated but not sent to HA)",
                    final_temp,
                )
            else:
                # Apply smart rounding: test floor vs ceiling to see which
                # gets closer to target
                floor_temp = np.floor(final_temp)
                ceiling_temp = np.ceil(final_temp)

                if floor_temp == ceiling_temp:
                    # Already an integer
                    smart_rounded_temp = int(final_temp)
                    logging.debug(
                        f"Smart rounding: {final_temp:.2f}°C is already "
                        "integer"
                    )
                else:
                    # Test both options using the thermal model to see which
                    # gets closer to target
                    try:
                        from .model_wrapper import get_enhanced_model_wrapper

                        wrapper = get_enhanced_model_wrapper()

                        # Create test contexts for floor and ceiling
                        # temperatures
                        test_context_floor = {
                            "outlet_temp": floor_temp,
                            "outdoor_temp": outdoor_temp,
                            "pv_power": (
                                features.get("pv_now", 0.0)
                                if hasattr(features, "get")
                                else 0.0
                            ),
                            "fireplace_on": fireplace_on,
                            "tv_on": (
                                features.get("tv_on", 0.0)
                                if hasattr(features, "get")
                                else 0.0
                            ),
                        }

                        test_context_ceiling = test_context_floor.copy()
                        test_context_ceiling["outlet_temp"] = ceiling_temp

                        # UNIFIED CONTEXT: Use same forecast-based
                        # conditions as binary search
                        from .prediction_context import (
                            prediction_context_manager
                        )

                        # Set up unified prediction context (same as binary
                        # search uses)

                        thermal_features = {
                            "pv_power": features_dict.get("pv_now", 0.0),
                            "fireplace_on": (
                                float(fireplace_on)
                                if fireplace_on is not None
                                else 0.0
                            ),
                            "tv_on": features_dict.get("tv_on", 0.0),
                        }

                        prediction_context_manager.set_features(features_dict)
                        unified_context = (
                            prediction_context_manager.create_context(
                                outdoor_temp=outdoor_temp,
                                pv_power=thermal_features["pv_power"],
                                thermal_features=thermal_features,
                            )
                        )

                        thermal_params = (
                            prediction_context_manager
                            .get_thermal_model_params()
                        )

                        # Get predictions using UNIFIED forecast-based
                        # parameters
                        floor_predicted = wrapper.predict_indoor_temp(
                            outlet_temp=floor_temp,
                            outdoor_temp=thermal_params["outdoor_temp"],
                            current_indoor=prediction_indoor_temp,
                            pv_power=thermal_params["pv_power"],
                            fireplace_on=thermal_params["fireplace_on"],
                            tv_on=thermal_params["tv_on"],
                        )
                        ceiling_predicted = wrapper.predict_indoor_temp(
                            outlet_temp=ceiling_temp,
                            outdoor_temp=thermal_params["outdoor_temp"],
                            current_indoor=prediction_indoor_temp,
                            pv_power=thermal_params["pv_power"],
                            fireplace_on=thermal_params["fireplace_on"],
                            tv_on=thermal_params["tv_on"],
                        )

                        # Handle None returns from predict_indoor_temp
                        if (
                            floor_predicted is None
                            or ceiling_predicted is None
                        ):
                            logging.warning(
                                "Smart rounding: predict_indoor_temp "
                                "returned None, using fallback"
                            )
                            smart_rounded_temp = round(final_temp)
                            logging.debug(
                                f"Smart rounding fallback: {final_temp:.2f}°C "
                                f"→ {smart_rounded_temp}°C"
                            )
                        else:
                            # Calculate errors from target
                            floor_error = abs(
                                floor_predicted - target_indoor_temp
                            )
                            ceiling_error = abs(
                                ceiling_predicted - target_indoor_temp
                            )

                            if floor_error <= ceiling_error:
                                smart_rounded_temp = int(floor_temp)
                                chosen = "floor"
                            else:
                                smart_rounded_temp = int(ceiling_temp)
                                chosen = "ceiling"

                            logging.info(
                                f"Smart rounding: {final_temp:.2f}°C → "
                                f"{smart_rounded_temp}°C (chose {chosen}: "
                                f"floor→{floor_predicted:.2f}°C "
                                f"[err={floor_error:.3f}], "
                                f"ceiling→{ceiling_predicted:.2f}°C "
                                f"[err={ceiling_error:.3f}], "
                                f"target={target_indoor_temp:.1f}°C)"
                            )
                    except Exception as e:
                        # Fallback to regular rounding if smart rounding
                        # fails
                        smart_rounded_temp = round(final_temp)
                        logging.warning(
                            f"Smart rounding failed ({e}), using regular "
                            f"rounding: {final_temp:.2f}°C → "
                            f"{smart_rounded_temp}°C"
                        )

                logging.debug("Setting target outlet temp")
                ha_client.set_state(
                    config.TARGET_OUTLET_TEMP_ENTITY_ID,
                    smart_rounded_temp,
                    get_sensor_attributes(config.TARGET_OUTLET_TEMP_ENTITY_ID),
                    round_digits=None,  # No additional rounding needed
                )

            # --- Log Metrics ---
            # Metrics logging now handled by ThermalEquilibriumModel in
            # model_wrapper
            if not config.SHADOW_MODE:
                logging.debug("Logging thermal model metrics")
                # Confidence is logged via simplified_outlet_prediction
                # metadata
                # Feature importances are handled by ThermalEquilibriumModel
                # Learning metrics are exported by ThermalEquilibriumModel

            # --- Log Thermodynamic Metrics (Feature/Sensor-Update) ---
            # Log COP, Power, and Delta T to InfluxDB for efficiency tracking
            try:
                thermodynamic_metrics = {
                    "cop_realtime": features_dict.get("cop_realtime", 0.0),
                    "thermal_power_kw": features_dict.get(
                        "thermal_power_kw", 0.0
                    ),
                    "delta_t": features_dict.get("delta_t", 0.0),
                    "flow_rate": features_dict.get("flow_rate", 0.0),
                    "inlet_temp": features_dict.get("inlet_temp", 0.0),
                }
                influx_service.write_thermodynamic_metrics(
                    thermodynamic_metrics
                )
            except Exception as e:
                logging.warning("Failed to log thermodynamic metrics: %s", e)

            # --- Update ML State sensor ---
            # Skip ML state sensor updates in shadow mode
            if not config.SHADOW_MODE:
                try:
                    # Get thermal model trust metrics from
                    # ThermalEquilibriumModel
                    thermal_trust_metrics = metadata.get(
                        "thermal_trust_metrics", {}
                    )

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
                            # Note: ThermalEquilibriumModel trust metrics
                            # moved to sensor.ml_heating_learning to
                            # eliminate redundancy
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
                logging.debug(
                    "🔍 SHADOW MODE: Skipping ML state sensor updates"
                )

            # --- Shadow Mode Status Logging ---
            if config.SHADOW_MODE:
                logging.debug(
                    "🔍 SHADOW MODE: Enabled via config (SHADOW_MODE=true)"
                )
            elif not ml_heating_enabled:
                logging.debug(
                    "🔍 SHADOW MODE: ML control disabled via %s",
                    config.ML_HEATING_CONTROL_ENTITY_ID,
                )
            else:
                logging.debug(
                    "✅ ACTIVE MODE: ML controlling heating via %s",
                    config.ML_HEATING_CONTROL_ENTITY_ID,
                )

            # --- Shadow Mode Comparison Logging ---
            if effective_shadow_mode:
                # Read what the heat curve actually set
                heat_curve_temp = ha_client.get_state(
                    config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID, all_states
                )

                if (
                    heat_curve_temp is not None
                    and heat_curve_temp != final_temp
                ):
                    # Simple comparison without model prediction
                    logging.debug(
                        "SHADOW MODE: ML would set %.1f°C, HC set %.1f°C | "
                        "Target: %.1f°C",
                        final_temp,
                        heat_curve_temp,
                        target_indoor_temp,
                    )

            # Shadow metrics now handled by ThermalEquilibriumModel

            # Use the actual rounded temperature that was applied to HA
            applied_temp = (
                smart_rounded_temp if not config.SHADOW_MODE else final_temp
            )

            thermal_params = {}
            # Calculate what the applied temperature will actually predict
            try:
                if not config.SHADOW_MODE and "wrapper" in locals():
                    # Get prediction for the applied smart-rounded temperature
                    applied_prediction = wrapper.predict_indoor_temp(
                        outlet_temp=applied_temp,
                        outdoor_temp=thermal_params.get(
                            "outdoor_temp", outdoor_temp
                        ),
                        current_indoor=prediction_indoor_temp,
                        pv_power=thermal_params.get(
                            "pv_power", features_dict.get("pv_now", 0.0)
                        ),
                        fireplace_on=thermal_params.get(
                            "fireplace_on", fireplace_on
                        ),
                        tv_on=thermal_params.get(
                            "tv_on", features_dict.get("tv_on", 0.0)
                        ),
                    )
                    if applied_prediction is None:
                        applied_prediction = predicted_indoor  # Fallback
                else:
                    # Shadow mode or wrapper not available
                    applied_prediction = predicted_indoor
            except Exception as e:
                logging.warning(f"Failed to get applied temp prediction: {e}")
                applied_prediction = predicted_indoor

            log_message = (
                "Target: %.1f°C | Suggested: %.1f°C | Applied: %.1f°C | "
                "Actual Indoor: %.2f°C | Predicted Indoor: %.2f°C | "
                "Confidence: %.3f"
            )
            logging.debug(
                log_message,
                target_indoor_temp,
                suggested_temp,
                applied_temp,
                actual_indoor,
                applied_prediction,  # Now shows prediction for applied temp
                confidence,
            )

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
                applied_prediction,  # Now shows prediction for applied temp
                confidence,
            )

            # --- Step 6: State Persistence for Next Run ---
            # Model saving is now handled by ThermalEquilibriumModel in
            # model_wrapper

            # The features and indoor temperature from the *current* run are
            # saved to a file. This data will be loaded at the start of the
            # next loop iteration to be used in the "Online Learning" step.
            # Note: We save final_temp here, but will read the actual
            # applied temp from ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID at the
            # start of the next cycle (after it has had time to update).
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
                        "last_updated": datetime.now(timezone.utc).isoformat(),
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

        logging.debug(
            f"✅ CYCLE {cycle_number} END: "
            f"{cycle_end_datetime.strftime('%H:%M:%S')} "
            f"(duration: {cycle_duration:.1f}s)"
        )

        last_cycle_end_time = cycle_end_time

        # Poll for blocking events during the idle period so defrost
        # starts/ends are detected quickly. This call will block until the
        # next cycle is due, or until a blocking event starts or ends.
        logging.debug(
            f"💤 POLLING START: Waiting "
            f"{PhysicsConstants.CYCLE_INTERVAL_MINUTES}min "
            "until next cycle..."
        )
        blocking_manager.poll_for_blocking(ha_client, state, sensor_buffer)

        poll_end_time = time.time()
        poll_duration = poll_end_time - cycle_end_time
        logging.debug(
            f"⏰ POLLING END: Waited {poll_duration/60:.1f}min, starting "
            "next cycle..."
        )


if __name__ == "__main__":
    main()
