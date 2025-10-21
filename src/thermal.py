"""
This module contains all the thermal-related logic and calculations.

It includes the logic for the traditional heating curve, which serves as a
baseline and fallback for the machine learning model. It also contains the
logic for applying a dynamic boost to the model's predictions to help the
system react more quickly to changes in indoor temperature.
"""
import logging
import numpy as np


def calculate_baseline_outlet_temp(
    outdoor_temp: float,
    owm_temp: float,
    forecast_temps: list[float],
) -> float:
    """
    Calculates a baseline target outlet temperature using a traditional heating curve.

    This function serves two purposes:
    1.  **Fallback**: It provides a safe, reliable setpoint when the ML model's
        confidence is too low to be trusted.
    2.  **Reference**: It acts as a center point for the ML model's search,
        keeping its exploration grounded in a physically sensible range.

    The calculation is based on a linear heating curve (a line defined by
    two points) that maps outdoor temperature to the required heating water
    temperature. It also incorporates a short-term weather forecast to be
    more proactive, slightly raising the temperature if it's about to get
    colder, and vice-versa.

    Args:
        outdoor_temp: The current outdoor temperature from a local sensor.
        owm_temp: The current temperature from the weather service.
        forecast_temps: A list of hourly forecasted temperatures.

    Returns:
        The calculated and rounded baseline outlet temperature.
    """
    # --- Part 1: Define the linear heating curve ---
    # This curve maps outdoor temperature to the required water outlet temperature.
    x1, y1 = -15.0, 64.0  # At -15°C outside, water should be 64°C.
    x2, y2 = 18.0, 31.0  # At 18°C outside, water should be 31°C.
    m = (y1 - y2) / (x1 - x2)  # Slope
    b = y2 - (m * x2)  # Intercept

    # --- Part 2: Adjust for weather forecast ---
    # This anticipates temperature changes to pre-heat or cool the system.
    if not all([outdoor_temp, owm_temp, len(forecast_temps) >= 3]):
        return 40.0  # Return a safe default if data is missing.

    # Compare the 2-hour forecast with the current weather service temperature.
    temp_forecast_delta = forecast_temps[2] - owm_temp
    # The input to the curve is a weighted average of current and forecasted
    # temperature.
    target_temp_for_curve = (
        outdoor_temp * 0.6 + (outdoor_temp + temp_forecast_delta) * 0.4
    )

    # --- Part 3: Final Calculation ---
    outlet = m * target_temp_for_curve + b
    # Clamp the result to a reasonable range for the heating system.
    final_outlet = np.clip(outlet, 16.0, 65.0)
    rounded_outlet = round(final_outlet)

    logging.info("--- Baseline Calculation ---")
    logging.info(
        f"  Outdoor Temp: {outdoor_temp:.2f}°C, OWM Temp: {owm_temp:.2f}°C"
    )
    logging.info(
        f"  2h Forecast: {forecast_temps[2]:.2f}°C -> "
        f"Delta: {temp_forecast_delta:.2f}°C"
    )
    logging.info(f"  Target for Curve: {target_temp_for_curve:.2f}°C")
    logging.info(
        (
            f"  Calculated: {outlet:.2f}°C -> Clamped: {final_outlet:.1f}°C -> "
            f"Rounded: {rounded_outlet:.1f}°C"
        )
    )

    return float(rounded_outlet)
