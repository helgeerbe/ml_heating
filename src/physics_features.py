"""
Streamlined feature builder for RealisticPhysicsModel.

This module builds only the features needed by the physics-based model,
reducing complexity and computation compared to the full ML feature set.
"""
import logging
from typing import Optional, Tuple

import pandas as pd

# Support both package-relative and direct import
try:
    from . import config
    from .ha_client import HAClient
    from .influx_service import InfluxService
except ImportError:
    import config
    from ha_client import HAClient
    from influx_service import InfluxService


def build_physics_features(
    ha_client: HAClient,
    influx_service: InfluxService,
) -> Tuple[Optional[pd.DataFrame], list[float]]:
    """
    Build features for RealisticPhysicsModel.
    
    Only creates the 19 features actually used by the physics model:
    - Core temperatures: outlet, indoor_lag_30m, target, outdoor
    - System states: dhw_heating, dhw_disinfection, dhw_boost_heater, defrosting
    - External sources: pv_now, fireplace_on, tv_on
    - Forecasts: temp_forecast_1h-4h, pv_forecast_1h-4h
    
    Args:
        ha_client: Home Assistant client
        influx_service: InfluxDB service
        outlet_history: List of recent outlet temperatures
        
    Returns:
        DataFrame with single row containing all features, or None if data missing
    """
    # Fetch current sensor values
    all_states = ha_client.get_all_states()
    
    actual_indoor = ha_client.get_state(
        config.INDOOR_TEMP_ENTITY_ID, all_states
    )
    outdoor_temp = ha_client.get_state(
        config.OUTDOOR_TEMP_ENTITY_ID, all_states
    )
    outlet_temp = ha_client.get_state(
        config.ACTUAL_OUTLET_TEMP_ENTITY_ID, all_states
    )
    target_indoor_temp = ha_client.get_state(
        config.TARGET_INDOOR_TEMP_ENTITY_ID, all_states
    )
    
    # Check for missing critical data
    if None in [actual_indoor, outdoor_temp, outlet_temp, target_indoor_temp]:
        logging.error("Missing critical sensor data. Cannot build features.")
        return None, []
    
    # Get history for temperature features
    outlet_history = influx_service.fetch_outlet_history(config.HISTORY_STEPS)
    indoor_history = influx_service.fetch_indoor_history(config.HISTORY_STEPS)
    
    if len(indoor_history) < 3 or len(outlet_history) < 1:
        logging.error("Insufficient history. Cannot build features.")
        return None, []
    
    # System states (binary) - default to False if unavailable
    dhw_heating = ha_client.get_state(
        config.DHW_STATUS_ENTITY_ID, all_states, is_binary=True
    ) or False
    dhw_disinfection = ha_client.get_state(
        config.DISINFECTION_STATUS_ENTITY_ID, all_states, is_binary=True
    ) or False
    dhw_boost_heater = ha_client.get_state(
        config.DHW_BOOST_HEATER_STATUS_ENTITY_ID, all_states, is_binary=True
    ) or False
    defrosting = ha_client.get_state(
        config.DEFROST_STATUS_ENTITY_ID, all_states, is_binary=True
    ) or False
    
    # External heat sources
    # Sum all PV power sources
    pv_now = ha_client.get_state(config.PV_POWER_ENTITY_ID, all_states) or 0.0
    pv_now = float(pv_now)
    
    fireplace_on = ha_client.get_state(
        config.FIREPLACE_STATUS_ENTITY_ID, all_states, is_binary=True
    ) or False
    tv_on = ha_client.get_state(
        config.TV_STATUS_ENTITY_ID, all_states, is_binary=True
    ) or False
    
    # Forecasts
    pv_forecasts = [0.0, 0.0, 0.0, 0.0]
    if config.PV_FORECAST_ENTITY_ID:
        try:
            # Get attributes directly from state cache
            from datetime import datetime, timezone
            
            pv_state = all_states.get(config.PV_FORECAST_ENTITY_ID)
            pv_forecast_data = pv_state.get("attributes") if pv_state else None
            if pv_forecast_data and "forecast" in pv_forecast_data:
                now = datetime.now(timezone.utc)
                fc = pv_forecast_data["forecast"]
                s = pd.Series(
                    {pd.Timestamp(e["period_start"]): e["pv_estimate"] 
                     for e in fc if "period_start" in e and "pv_estimate" in e},
                    dtype=float,
                ).sort_index()
                
                # Average over each hour
                anchors = [
                    pd.Timestamp(now) + pd.Timedelta(hours=i) 
                    for i in range(1, 5)
                ]
                hourly = []
                for a in anchors:
                    start = a
                    end = a + pd.Timedelta("1h")
                    slice_vals = s[(s.index >= start) & (s.index < end)]
                    hourly.append(
                        float(slice_vals.mean()) if not slice_vals.empty else 0.0
                    )
                pv_forecasts = hourly
        except Exception as e:
            logging.debug(f"Could not fetch PV forecast: {e}")
            pv_forecasts = [0.0, 0.0, 0.0, 0.0]
    
    temp_forecasts = ha_client.get_hourly_forecast()
    
    # Build feature dictionary
    features = {
        # Core temperatures
        'outlet_temp': float(outlet_temp),
        'indoor_temp_lag_30m': float(indoor_history[-3]),  # 30 min ago
        'target_temp': float(target_indoor_temp),
        'outdoor_temp': float(outdoor_temp),
        
        # System states
        'dhw_heating': float(dhw_heating),
        'dhw_disinfection': float(dhw_disinfection),
        'dhw_boost_heater': float(dhw_boost_heater),
        'defrosting': float(defrosting),
        
        # External heat sources
        'pv_now': float(pv_now),
        'fireplace_on': float(fireplace_on),
        'tv_on': float(tv_on),
        
        # Weather forecasts (1-4 hours)
        'temp_forecast_1h': float(temp_forecasts[0]),
        'temp_forecast_2h': float(temp_forecasts[1]),
        'temp_forecast_3h': float(temp_forecasts[2]),
        'temp_forecast_4h': float(temp_forecasts[3]),
        
        # PV forecasts (1-4 hours)
        'pv_forecast_1h': float(pv_forecasts[0]),
        'pv_forecast_2h': float(pv_forecasts[1]),
        'pv_forecast_3h': float(pv_forecasts[2]),
        'pv_forecast_4h': float(pv_forecasts[3]),
    }
    
    return pd.DataFrame([features]), outlet_history
