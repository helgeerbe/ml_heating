"""
Forecast Analytics for Week 4 Enhanced Forecast Utilization.

This module provides forecast quality tracking and analysis capabilities
for weather and PV forecasts used in thermal control decisions.
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime


def analyze_forecast_quality(weather_forecasts: List[float], pv_forecasts: List[float]) -> Dict[str, float]:
    """
    Analyze forecast data quality and availability.
    
    Args:
        weather_forecasts: List of 4 hourly temperature forecasts
        pv_forecasts: List of 4 hourly PV power forecasts
        
    Returns:
        Dictionary with quality metrics
    """
    quality_metrics = {
        'weather_availability': 0.0,
        'pv_availability': 0.0,
        'combined_availability': 0.0,
        'weather_confidence': 0.0,
        'pv_confidence': 0.0,
        'overall_confidence': 0.0
    }
    
    # Check weather forecast availability
    valid_weather = [f for f in weather_forecasts if f is not None and not (f == 0.0)]
    weather_availability = len(valid_weather) / max(1, len(weather_forecasts))
    
    # Check PV forecast availability  
    valid_pv = [f for f in pv_forecasts if f is not None and f >= 0.0]
    pv_availability = len(valid_pv) / max(1, len(pv_forecasts))
    
    # Combined availability
    combined_availability = (weather_availability + pv_availability) / 2.0
    
    # Weather confidence based on reasonable temperature ranges (-20°C to 40°C)
    weather_confidence = 0.0
    if valid_weather:
        reasonable_temps = [f for f in valid_weather if -20.0 <= f <= 40.0]
        weather_confidence = len(reasonable_temps) / len(valid_weather)
    
    # PV confidence based on reasonable power values (0-15000W)  
    pv_confidence = 0.0
    if valid_pv:
        reasonable_pv = [f for f in valid_pv if 0.0 <= f <= 15000.0]
        pv_confidence = len(reasonable_pv) / len(valid_pv)
    
    # Overall confidence
    overall_confidence = (weather_confidence + pv_confidence) / 2.0 if (weather_confidence > 0 or pv_confidence > 0) else 0.0
    
    quality_metrics.update({
        'weather_availability': weather_availability,
        'pv_availability': pv_availability, 
        'combined_availability': combined_availability,
        'weather_confidence': weather_confidence,
        'pv_confidence': pv_confidence,
        'overall_confidence': overall_confidence
    })
    
    logging.debug(
        f"Forecast quality: weather_avail={weather_availability:.2f}, "
        f"pv_avail={pv_availability:.2f}, confidence={overall_confidence:.2f}"
    )
    
    return quality_metrics


def calculate_thermal_forecast_impact(
    temp_forecasts: List[float], 
    pv_forecasts: List[float], 
    current_outdoor_temp: float,
    current_pv_power: float = 0.0
) -> Dict[str, float]:
    """
    Calculate combined thermal impact of weather + PV forecasts.
    
    Args:
        temp_forecasts: List of 4 hourly temperature forecasts
        pv_forecasts: List of 4 hourly PV power forecasts  
        current_outdoor_temp: Current outdoor temperature
        current_pv_power: Current PV power
        
    Returns:
        Dictionary with thermal impact metrics
    """
    thermal_impact = {
        'weather_cooling_trend': 0.0,
        'weather_heating_trend': 0.0,
        'pv_warming_trend': 0.0,
        'net_thermal_trend': 0.0,
        'thermal_load_forecast': 0.0
    }
    
    if not temp_forecasts or not pv_forecasts:
        return thermal_impact
    
    # Calculate weather cooling/warming trends
    temp_trend_4h = temp_forecasts[3] - current_outdoor_temp  # 4-hour temperature change
    
    if temp_trend_4h < 0:
        # Cooling trend - increases heating demand
        thermal_impact['weather_cooling_trend'] = abs(temp_trend_4h) * 0.1  # Simple factor
    else:
        # Warming trend - reduces heating demand
        thermal_impact['weather_heating_trend'] = temp_trend_4h * 0.05  # Asymmetric factor
    
    # Calculate PV solar warming building effect
    pv_trend_4h = pv_forecasts[3] - current_pv_power
    if pv_trend_4h > 0:
        # Increasing PV = more solar warming
        thermal_impact['pv_warming_trend'] = pv_trend_4h * 0.0005  # W to °C factor
    
    # Net thermal trend (positive = heating needed, negative = cooling needed)
    net_thermal_trend = (
        thermal_impact['weather_cooling_trend'] -  # Cooling increases heating need
        thermal_impact['weather_heating_trend'] -  # Warming reduces heating need
        thermal_impact['pv_warming_trend']         # PV reduces heating need
    )
    
    thermal_impact['net_thermal_trend'] = net_thermal_trend
    
    # Thermal load forecast (always positive, represents heating demand)
    base_thermal_load = max(0.0, (21.0 - temp_forecasts[3]) * 0.1)  # Base heating demand
    pv_offset = pv_forecasts[3] * 0.001  # PV warming offset
    thermal_load = max(0.0, base_thermal_load - pv_offset)
    
    thermal_impact['thermal_load_forecast'] = thermal_load
    
    logging.debug(
        f"Thermal forecast impact: cooling_trend={thermal_impact['weather_cooling_trend']:.3f}, "
        f"warming_trend={thermal_impact['weather_heating_trend']:.3f}, "
        f"pv_warming={thermal_impact['pv_warming_trend']:.3f}, "
        f"net_trend={net_thermal_trend:.3f}"
    )
    
    return thermal_impact


def get_forecast_fallback_strategy(
    quality_metrics: Dict[str, float], 
    current_conditions: Dict[str, float]
) -> Dict[str, float]:
    """
    Provide fallback strategies when forecasts are unavailable or poor quality.
    
    Args:
        quality_metrics: Forecast quality metrics from analyze_forecast_quality()
        current_conditions: Current sensor values for fallback
        
    Returns:
        Dictionary with fallback forecast values
    """
    fallback_forecasts = {
        'temp_forecast_1h': current_conditions.get('outdoor_temp', 10.0),
        'temp_forecast_2h': current_conditions.get('outdoor_temp', 10.0),
        'temp_forecast_3h': current_conditions.get('outdoor_temp', 10.0),
        'temp_forecast_4h': current_conditions.get('outdoor_temp', 10.0),
        'pv_forecast_1h': 0.0,
        'pv_forecast_2h': 0.0,
        'pv_forecast_3h': 0.0,
        'pv_forecast_4h': 0.0,
        'fallback_reason': 'high_quality'
    }
    
    overall_confidence = quality_metrics.get('overall_confidence', 0.0)
    combined_availability = quality_metrics.get('combined_availability', 0.0)
    
    # Determine fallback strategy
    if overall_confidence < 0.5:
        fallback_forecasts['fallback_reason'] = 'low_confidence'
        # Use conservative temperature trend (assume current temp persists)
        current_temp = current_conditions.get('outdoor_temp', 10.0)
        for i in range(1, 5):
            fallback_forecasts[f'temp_forecast_{i}h'] = current_temp
        
    elif combined_availability < 0.5:
        fallback_forecasts['fallback_reason'] = 'low_availability'  
        # Use simple seasonal trend (gradual cooling at night)
        current_temp = current_conditions.get('outdoor_temp', 10.0)
        current_hour = datetime.now().hour
        
        # Simple night cooling model
        for i in range(1, 5):
            hour = (current_hour + i) % 24
            if 6 <= hour <= 18:  # Daytime
                temp_adjustment = 0.5  # Slight warming
            else:  # Nighttime  
                temp_adjustment = -0.3  # Slight cooling
            fallback_forecasts[f'temp_forecast_{i}h'] = current_temp + temp_adjustment
    
    # PV fallback: assume zero during night, maintain current during day
    current_hour = datetime.now().hour
    current_pv = current_conditions.get('pv_now', 0.0)
    
    for i in range(1, 5):
        hour = (current_hour + i) % 24
        if 6 <= hour <= 18:  # Daytime
            fallback_forecasts[f'pv_forecast_{i}h'] = max(0.0, current_pv * 0.8)  # Gradual reduction
        else:  # Nighttime
            fallback_forecasts[f'pv_forecast_{i}h'] = 0.0
    
    logging.info(
        f"Forecast fallback strategy: {fallback_forecasts['fallback_reason']} "
        f"(confidence={overall_confidence:.2f}, availability={combined_availability:.2f})"
    )
    
    return fallback_forecasts


def calculate_forecast_accuracy_metrics(
    predicted_values: List[float],
    actual_values: List[float],
    forecast_type: str = "temperature"
) -> Dict[str, float]:
    """
    Calculate forecast accuracy metrics for validation.
    
    Args:
        predicted_values: List of predicted forecast values
        actual_values: List of actual measured values
        forecast_type: Type of forecast ("temperature" or "pv")
        
    Returns:
        Dictionary with accuracy metrics
    """
    accuracy_metrics = {
        'mae': 0.0,  # Mean Absolute Error
        'rmse': 0.0,  # Root Mean Square Error
        'accuracy_score': 0.0,  # Overall accuracy (0-1)
        'sample_size': 0
    }
    
    if not predicted_values or not actual_values or len(predicted_values) != len(actual_values):
        return accuracy_metrics
    
    n = len(predicted_values)
    
    # Calculate MAE and RMSE
    absolute_errors = [abs(pred - actual) for pred, actual in zip(predicted_values, actual_values)]
    squared_errors = [(pred - actual) ** 2 for pred, actual in zip(predicted_values, actual_values)]
    
    mae = sum(absolute_errors) / n
    rmse = (sum(squared_errors) / n) ** 0.5
    
    # Calculate accuracy score based on forecast type
    if forecast_type == "temperature":
        # Good temperature forecast: MAE < 2°C
        accuracy_score = max(0.0, 1.0 - (mae / 2.0))
    elif forecast_type == "pv":
        # Good PV forecast: MAE < 500W  
        accuracy_score = max(0.0, 1.0 - (mae / 500.0))
    else:
        # Generic accuracy
        mean_actual = sum(actual_values) / n
        accuracy_score = max(0.0, 1.0 - (mae / max(0.1, abs(mean_actual))))
    
    accuracy_metrics.update({
        'mae': mae,
        'rmse': rmse,
        'accuracy_score': min(1.0, accuracy_score),
        'sample_size': n
    })
    
    logging.debug(
        f"{forecast_type} forecast accuracy: MAE={mae:.2f}, RMSE={rmse:.2f}, "
        f"score={accuracy_score:.3f} (n={n})"
    )
    
    return accuracy_metrics
