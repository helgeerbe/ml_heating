"""
Enhanced feature builder for RealisticPhysicsModel with thermal momentum
analysis.

This module builds comprehensive physics features including:
- Original 19 features for backward compatibility
- NEW: 15 thermal momentum and extended lag features for enhanced temperature
  stability
- Thermal gradient analysis for overshoot prevention
- Extended lag features (10m, 60m) for thermal mass understanding
- Cyclical time encoding for daily/seasonal patterns
- Outlet effectiveness analysis for heat transfer optimization

Total: 34 sophisticated thermal intelligence features for ±0.1°C control
precision.
"""
import logging
import math
from datetime import datetime
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
    Build enhanced features for RealisticPhysicsModel with thermal momentum.
    
    Creates 34 physics features including original 19 + 15 new thermal features:
    - Core temperatures: outlet, indoor_lag_30m, target, outdoor
    - System states: dhw_heating, dhw_disinfection, dhw_boost_heater, defrosting
    - External sources: pv_now, fireplace_on, tv_on
    - Forecasts: temp_forecast_1h-4h, pv_forecast_1h-4h
    - NEW: Thermal momentum, extended lag, delta analysis, cyclical time
    
    Args:
        ha_client: Home Assistant client
        influx_service: InfluxDB service
        
    Returns:
        DataFrame with single row containing all features, or None if missing
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
    
    # Get extended history for thermal momentum features
    # Need 6 steps for 60-minute lag analysis (6 * 10min = 60min)
    extended_steps = max(6, config.HISTORY_STEPS)
    outlet_history = influx_service.fetch_outlet_history(extended_steps)
    indoor_history = influx_service.fetch_indoor_history(extended_steps)
    
    if len(indoor_history) < 6 or len(outlet_history) < 3:
        logging.error("Insufficient history for enhanced features. "
                     f"Need 6 indoor + 3 outlet, got {len(indoor_history)} "
                     f"+ {len(outlet_history)}.")
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
    
    # PV Forecasts with correct 'watts' attribute parsing
    pv_forecasts = [0.0, 0.0, 0.0, 0.0]
    if config.PV_FORECAST_ENTITY_ID:
        try:
            from datetime import timezone, timedelta
            
            pv_state = all_states.get(config.PV_FORECAST_ENTITY_ID)
            pv_forecast_data = (pv_state.get("attributes") 
                               if pv_state else None)
            
            # CORRECT: Look for 'watts' attribute, not 'forecast'
            if pv_forecast_data and "watts" in pv_forecast_data:
                now = datetime.now(timezone.utc)
                watts_data = pv_forecast_data["watts"]
                
                # Convert timestamp strings to pandas timestamps
                forecast_dict = {}
                for ts_str, watts in watts_data.items():
                    try:
                        # Parse ISO timestamp (handles timezone offsets)
                        ts = datetime.fromisoformat(ts_str)
                        forecast_dict[pd.Timestamp(ts)] = float(watts)
                    except Exception as e_parse:
                        logging.debug(f"Could not parse timestamp {ts_str}: {e_parse}")
                        continue
                        
                if forecast_dict:
                    s = pd.Series(forecast_dict, dtype=float).sort_index()
                    
                    # Calculate hourly averages for next 4 hours
                    hourly = []
                    for hour in range(1, 5):
                        hour_start = now + timedelta(hours=hour)
                        hour_end = hour_start + timedelta(hours=1)
                        
                        # Find all entries in this hour window
                        hour_entries = []
                        for ts, watts in s.items():
                            # Convert to UTC for comparison
                            ts_utc = ts.tz_convert('UTC') if ts.tz else ts.tz_localize('UTC')
                            if hour_start <= ts_utc < hour_end:
                                hour_entries.append(watts)
                        
                        if hour_entries:
                            avg_watts = sum(hour_entries) / len(hour_entries)
                            hourly.append(round(avg_watts, 1))
                        else:
                            hourly.append(0.0)
                    
                    pv_forecasts = hourly
                    logging.debug(f"PV forecast parsed successfully: {pv_forecasts}W")
                else:
                    logging.debug("No valid PV forecast timestamps parsed")
            else:
                logging.debug(f"PV forecast entity missing 'watts' attribute: {list(pv_forecast_data.keys()) if pv_forecast_data else 'no attributes'}")
                
        except Exception as e:
            logging.debug(f"Could not fetch PV forecast: {e}")
            pv_forecasts = [0.0, 0.0, 0.0, 0.0]
    
    # Convert to float for calculations first
    actual_indoor_f = float(actual_indoor)
    outdoor_temp_f = float(outdoor_temp)
    outlet_temp_f = float(outlet_temp)
    target_temp_f = float(target_indoor_temp)
    
    # Get calibrated temperature forecasts using delta correction
    try:
        # Check if delta calibration is enabled and available
        if (hasattr(config, 'ENABLE_DELTA_FORECAST_CALIBRATION') and 
            config.ENABLE_DELTA_FORECAST_CALIBRATION and
            hasattr(ha_client, 'get_calibrated_hourly_forecast')):
            temp_forecasts = ha_client.get_calibrated_hourly_forecast(
                current_outdoor_temp=outdoor_temp_f,
                enable_delta_calibration=True
            )
        else:
            temp_forecasts = ha_client.get_hourly_forecast()
            
        # Ensure we have a valid list of forecasts
        if not isinstance(temp_forecasts, list) or len(temp_forecasts) < 4:
            # Fallback to default values if forecasts are invalid
            temp_forecasts = [outdoor_temp_f] * 4
    except Exception as e:
        logging.debug(f"Could not fetch temperature forecasts: {e}")
        temp_forecasts = [outdoor_temp_f] * 4
    
    # Get current time for cyclical encoding
    now = datetime.now()
    current_hour = now.hour
    current_month = now.month
    
    # Calculate time period for gradient (in hours)
    time_period = config.HISTORY_STEP_MINUTES / 60.0
    
    # Build enhanced feature dictionary with thermal momentum features
    features = {
        # === ORIGINAL 19 FEATURES (for backward compatibility) ===
        
        # Core temperatures
        'outlet_temp': outlet_temp_f,
        'indoor_temp_lag_30m': float(indoor_history[-3]),  # 30 min ago
        'target_temp': target_temp_f,
        'outdoor_temp': outdoor_temp_f,
        
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
        
        # === NEW 15 THERMAL MOMENTUM FEATURES ===
        
        # P0 Priority: Thermal momentum analysis (3 features)
        'temp_diff_indoor_outdoor': actual_indoor_f - outdoor_temp_f,
        'indoor_temp_gradient': ((actual_indoor_f - float(indoor_history[0]))
                                 / time_period),
        'outlet_indoor_diff': outlet_temp_f - actual_indoor_f,
        
        # P0 Priority: Extended lag features (4 features)
        'indoor_temp_lag_10m': float(indoor_history[-1]),  # 10 min ago
        'indoor_temp_lag_60m': float(indoor_history[-6]),  # 60 min ago
        'outlet_temp_lag_30m': float(outlet_history[-3]),  # 30 min ago
        'outlet_temp_change': outlet_temp_f - float(outlet_history[-1]),
        
        # P1 Priority: Delta analysis (3 features)
        'indoor_temp_delta_10m': actual_indoor_f - float(indoor_history[-1]),
        'indoor_temp_delta_30m': actual_indoor_f - float(indoor_history[-3]),
        'indoor_temp_delta_60m': actual_indoor_f - float(indoor_history[-6]),
        
        # P1 Priority: Cyclical time encoding (4 features)
        'hour_sin': math.sin(2 * math.pi * current_hour / 24),
        'hour_cos': math.cos(2 * math.pi * current_hour / 24),
        'month_sin': math.sin(2 * math.pi * (current_month - 1) / 12),
        'month_cos': math.cos(2 * math.pi * (current_month - 1) / 12),
        
        # P2 Priority: Outlet effectiveness analysis (1 feature)
        'outlet_effectiveness_ratio': ((actual_indoor_f - target_temp_f) /
                                       max(0.1, outlet_temp_f - 
                                           actual_indoor_f)),
        
        # === WEEK 4 ENHANCED FORECAST FEATURES ===
        
        # Enhanced forecast analysis (3 new features)
        'temp_trend_forecast': ((temp_forecasts[3] - outdoor_temp_f) / 4.0),  # °C/hour trend
        'heating_demand_forecast': max(0.0, (21.0 - temp_forecasts[3]) * 0.1),  # Simple heating demand
        'combined_forecast_thermal_load': (max(0.0, (21.0 - temp_forecasts[3]) * 0.1) - (pv_forecasts[3] * 0.001)),  # Net thermal load
    }
    
    return pd.DataFrame([features]), outlet_history
