"""
Heating Controller Module

This module contains the main heating control logic extracted from main.py
to improve code organization and maintainability.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

from . import config
from .ha_client import HAClient, get_sensor_attributes
from .state_manager import save_state


class BlockingStateManager:
    """Manages blocking states (DHW, Defrost, etc.) and grace periods"""
    
    def __init__(self):
        self.blocking_entities = [
            config.DHW_STATUS_ENTITY_ID,
            config.DEFROST_STATUS_ENTITY_ID,
            config.DISINFECTION_STATUS_ENTITY_ID,
            config.DHW_BOOST_HEATER_STATUS_ENTITY_ID,
        ]
        
    def check_blocking_state(self, ha_client: HAClient, all_states: Dict) -> Tuple[bool, List[str]]:
        """
        Check if any blocking processes are active
        
        Returns:
            Tuple of (is_blocking, blocking_reasons)
        """
        blocking_reasons = [
            e for e in self.blocking_entities
            if ha_client.get_state(e, all_states, is_binary=True)
        ]
        is_blocking = bool(blocking_reasons)
        
        return is_blocking, blocking_reasons
    
    def handle_blocking_state(self, ha_client: HAClient, is_blocking: bool, 
                            blocking_reasons: List[str], state: Dict) -> bool:
        """
        Handle blocking state persistence and HA sensor updates
        
        Returns:
            True if cycle should be skipped
        """
        if is_blocking:
            logging.info("Blocking process active (DHW/Defrost), skipping.")
            
            try:
                attributes_state = get_sensor_attributes("sensor.ml_heating_state")
                attributes_state.update({
                    "state_description": "Blocking activity - Skipping",
                    "blocking_reasons": blocking_reasons,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                })
                ha_client.set_state(
                    "sensor.ml_heating_state", 2, attributes_state, round_digits=None
                )
            except Exception:
                logging.debug("Failed to write BLOCKED state to HA.", exc_info=True)
            
            # Save the blocking state for the next cycle
            save_state(
                last_is_blocking=True,
                last_final_temp=state.get("last_final_temp"),
                last_blocking_reasons=blocking_reasons,
                last_blocking_end_time=None,
            )
            return True  # Skip cycle
        
        return False  # Continue with cycle
    
    def handle_grace_period(self, ha_client: HAClient, state: Dict, shadow_mode: bool = None) -> bool:
        """
        Handle grace period after blocking events end
        
        Args:
            ha_client: Home Assistant client
            state: Current system state
            shadow_mode: Dynamic shadow mode state (overrides config.SHADOW_MODE if provided)
        
        Returns:
            True if in grace period (skip cycle)
        """
        # SHADOW MODE OPTIMIZATION: Skip grace period entirely in shadow mode
        # since ML heating is only observing and not controlling equipment.
        # Use dynamic shadow_mode if provided, otherwise fall back to config
        is_shadow_mode = shadow_mode if shadow_mode is not None else config.SHADOW_MODE
        
        if is_shadow_mode:
            logging.info("⏭️ SHADOW MODE: Skipping grace period (observation mode only)")
            return False  # No grace period needed in shadow mode
            
        last_is_blocking = state.get("last_is_blocking", False)
        last_blocking_end_time = state.get("last_blocking_end_time")
        
        if not last_is_blocking:
            return False  # No grace period needed
            
        # Check if blocking just ended
        is_blocking, _ = self.check_blocking_state(ha_client, ha_client.get_all_states())
        
        if is_blocking:
            return False  # Still blocking, no grace period
            
        # Mark the blocking end time if not already set
        if last_blocking_end_time is None:
            last_blocking_end_time = time.time()
            try:
                save_state(last_blocking_end_time=last_blocking_end_time)
            except Exception:
                logging.debug("Failed to persist last_blocking_end_time.", exc_info=True)

        # Check if grace period has expired
        age = time.time() - last_blocking_end_time
        if age > config.GRACE_PERIOD_MAX_MINUTES * 60:
            logging.info(
                "Grace period expired (ended %.1f min ago); skipping restore/wait.",
                age / 60.0,
            )
            return False
            
        # Execute grace period logic
        self._execute_grace_period(ha_client, state, age)
        
        # Clear blocking state after grace period
        try:
            save_state(last_is_blocking=False, last_blocking_end_time=None)
        except Exception:
            logging.debug("Failed to persist cleared blocking state.", exc_info=True)
            
        return True  # Skip this cycle
    
    def _execute_grace_period(self, ha_client: HAClient, state: Dict, age: float):
        """Execute the grace period temperature restoration logic"""
        logging.info("--- Grace Period Started ---")
        logging.info(
            "Blocking event ended %.1f min ago. Entering grace period to allow system to stabilize.",
            age / 60.0,
        )
        
        last_final_temp = state.get("last_final_temp")
        if last_final_temp is None:
            logging.info("No last_final_temp found in persisted state; skipping restore/wait.")
            return
            
        # Get current outlet temperature
        actual_outlet_temp_start = ha_client.get_state(
            config.ACTUAL_OUTLET_TEMP_ENTITY_ID, ha_client.get_all_states()
        )
        
        if actual_outlet_temp_start is None:
            logging.warning("Cannot read actual_outlet_temp at grace start; skipping wait.")
            return
            
        delta0 = actual_outlet_temp_start - last_final_temp
        if delta0 == 0:
            logging.info(
                "Actual outlet equals the restored target (%.1f°C); no wait needed.",
                last_final_temp,
            )
            return
            
        # Determine grace target and wait condition
        wait_for_cooling = delta0 > 0
        if wait_for_cooling:
            grace_target = last_final_temp + config.MAX_TEMP_CHANGE_PER_CYCLE
        else:
            grace_target = last_final_temp
            
        logging.info(
            "Restoring outlet target: %.1f°C (last=%.1f°C, actual=%.1f°C, %s)",
            grace_target, last_final_temp, actual_outlet_temp_start,
            "cool-down" if wait_for_cooling else "warm-up",
        )
        
        # Set the grace target temperature
        ha_client.set_state(
            config.TARGET_OUTLET_TEMP_ENTITY_ID,
            grace_target,
            get_sensor_attributes(config.TARGET_OUTLET_TEMP_ENTITY_ID),
            round_digits=0,
        )
        
        # Wait for temperature to reach target
        self._wait_for_grace_target(ha_client, grace_target, wait_for_cooling)
        
        logging.info("--- Grace Period Ended ---")
    
    def _wait_for_grace_target(self, ha_client: HAClient, grace_target: float, 
                              wait_for_cooling: bool):
        """Wait for outlet temperature to reach grace target"""
        start_time = time.time()
        max_seconds = config.GRACE_PERIOD_MAX_MINUTES * 60
        
        logging.info(
            "Grace period: waiting for %s (timeout %d min).",
            "actual <= target" if wait_for_cooling else "actual >= target",
            config.GRACE_PERIOD_MAX_MINUTES,
        )
        
        while True:
            # Check for blocking reappearance
            all_states_poll = ha_client.get_all_states()
            is_blocking, blocking_reasons = self.check_blocking_state(ha_client, all_states_poll)
            
            if is_blocking:
                logging.info("Blocking reappeared during grace; aborting wait.")
                try:
                    save_state(
                        last_is_blocking=True,
                        last_blocking_reasons=blocking_reasons,
                        last_blocking_end_time=None,
                    )
                except Exception:
                    logging.debug("Failed to persist blocking restart.", exc_info=True)
                break
                
            # Check outlet temperature
            actual_outlet_temp = ha_client.get_state(
                config.ACTUAL_OUTLET_TEMP_ENTITY_ID, all_states_poll
            )
            
            if actual_outlet_temp is None:
                logging.warning("Cannot read actual_outlet_temp, exiting grace period.")
                break
                
            # Check if target reached
            target_reached = False
            if wait_for_cooling and actual_outlet_temp <= grace_target:
                target_reached = True
            elif not wait_for_cooling and actual_outlet_temp >= grace_target:
                target_reached = True
                
            if target_reached:
                logging.info(
                    "Actual outlet temp (%.1f°C) has reached grace target (%.1f°C). Resuming control.",
                    actual_outlet_temp, grace_target,
                )
                break
                
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > max_seconds:
                logging.warning(
                    "Grace period timed out after %d minutes; proceeding.",
                    config.GRACE_PERIOD_MAX_MINUTES,
                )
                break
                
            logging.info(
                "Waiting for outlet to %s grace target (current: %.1f°C, target: %.1f°C). Elapsed: %d/%d min",
                "cool to" if wait_for_cooling else "warm to",
                actual_outlet_temp, grace_target,
                int(elapsed / 60), config.GRACE_PERIOD_MAX_MINUTES,
            )
            
            time.sleep(config.BLOCKING_POLL_INTERVAL_SECONDS)


class SensorDataManager:
    """Manages sensor data retrieval and validation"""
    
    def get_critical_sensors(self, ha_client: HAClient, all_states: Dict) -> Tuple[Optional[Dict], List[str]]:
        """
        Retrieve and validate critical sensor data
        
        Returns:
            Tuple of (sensor_data_dict, missing_sensors_list)
        """
        sensor_data = {
            'target_indoor_temp': ha_client.get_state(config.TARGET_INDOOR_TEMP_ENTITY_ID, all_states),
            'actual_indoor': ha_client.get_state(config.INDOOR_TEMP_ENTITY_ID, all_states),
            'actual_outlet_temp': ha_client.get_state(config.ACTUAL_OUTLET_TEMP_ENTITY_ID, all_states),
            'avg_other_rooms_temp': ha_client.get_state(config.AVG_OTHER_ROOMS_TEMP_ENTITY_ID, all_states),
            'fireplace_on': ha_client.get_state(config.FIREPLACE_STATUS_ENTITY_ID, all_states, is_binary=True),
            'outdoor_temp': ha_client.get_state(config.OUTDOOR_TEMP_ENTITY_ID, all_states),
            'owm_temp': ha_client.get_state(config.OPENWEATHERMAP_TEMP_ENTITY_ID, all_states),
        }
        
        critical_sensors = {
            config.TARGET_INDOOR_TEMP_ENTITY_ID: sensor_data['target_indoor_temp'],
            config.INDOOR_TEMP_ENTITY_ID: sensor_data['actual_indoor'],
            config.OUTDOOR_TEMP_ENTITY_ID: sensor_data['outdoor_temp'],
            config.OPENWEATHERMAP_TEMP_ENTITY_ID: sensor_data['owm_temp'],
            config.AVG_OTHER_ROOMS_TEMP_ENTITY_ID: sensor_data['avg_other_rooms_temp'],
            config.ACTUAL_OUTLET_TEMP_ENTITY_ID: sensor_data['actual_outlet_temp'],
        }
        
        missing_sensors = [
            name for name, value in critical_sensors.items() if value is None
        ]
        
        if missing_sensors:
            return None, missing_sensors
            
        return sensor_data, []
    
    def handle_missing_sensors(self, ha_client: HAClient, missing_sensors: List[str]) -> bool:
        """
        Handle missing sensor data with appropriate HA state updates
        
        Returns:
            True if cycle should be skipped
        """
        logging.warning("Critical sensors unavailable: %s. Skipping.", ", ".join(missing_sensors))
        
        try:
            attributes_state = get_sensor_attributes("sensor.ml_heating_state")
            attributes_state.update({
                "state_description": "No data - missing critical sensors",
                "missing_sensors": missing_sensors,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            })
            ha_client.set_state(
                "sensor.ml_heating_state", 4, attributes_state, round_digits=None
            )
        except Exception:
            logging.debug("Failed to write NO_DATA state to HA.", exc_info=True)
            
        return True  # Skip cycle


class HeatingSystemStateChecker:
    """Checks heating system operational state"""
    
    def check_heating_active(self, ha_client: HAClient, all_states: Dict) -> bool:
        """
        Check if heating system is active
        
        Returns:
            True if heating is active, False if cycle should be skipped
        """
        heating_state = ha_client.get_state(config.HEATING_STATUS_ENTITY_ID, all_states)
        
        if heating_state not in ("heat", "auto"):
            logging.info("Heating system not active (state: %s), skipping cycle.", heating_state)
            
            try:
                attributes_state = get_sensor_attributes("sensor.ml_heating_state")
                attributes_state.update({
                    "state_description": f"Heating off ({heating_state})",
                    "heating_state": heating_state,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                })
                ha_client.set_state(
                    "sensor.ml_heating_state", 6, attributes_state, round_digits=None
                )
            except Exception:
                logging.debug("Failed to write HEATING_OFF state to HA.", exc_info=True)
                
            return False  # Skip cycle
            
        return True  # Continue with cycle
