"""This module contains thermal-related calculations."""
import logging
import numpy as np


def calculate_baseline_outlet_temp(
    outdoor_temp: float,
    owm_temp: float,
    forecast_temps: list[float],
) -> float:
    """
    Calculates the target outlet temperature based on the user's
    heating curve formula, matching the original script's logic.
    """
    # --- Part 1: Calculate the base heating curve value ---
    x1, y1 = -15.0, 64.0
    x2, y2 = 18.0, 31.0
    m = (y1 - y2) / (x1 - x2)
    b = y2 - (m * x2)

    # --- Part 2: Calculate the target temperature input for the curve ---
    if not all([outdoor_temp, owm_temp, len(forecast_temps) >= 3]):
        return 40.0  # Safe default

    temp_forecast_delta = forecast_temps[2] - owm_temp
    target_temp_for_curve = (
        outdoor_temp * 0.6 + (outdoor_temp + temp_forecast_delta) * 0.4
    )

    # --- Part 3: Final Calculation ---
    outlet = m * target_temp_for_curve + b
    final_outlet = np.clip(outlet, 16.0, 65.0)
    rounded_outlet = round(final_outlet)

    logging.debug("--- Baseline Calculation ---")
    logging.debug(f"  Outdoor Temp: {outdoor_temp:.2f}°C, OWM Temp: {owm_temp:.2f}°C")
    logging.debug(
        f"  2h Forecast: {forecast_temps[2]:.2f}°C -> Delta: {temp_forecast_delta:.2f}°C"
    )
    logging.debug(f"  Target for Curve: {target_temp_for_curve:.2f}°C")
    logging.debug(
        f"  Calculated: {outlet:.2f}°C -> Clamped: {final_outlet:.1f}°C -> Rounded: {rounded_outlet:.1f}°C"
    )

    return float(rounded_outlet)


def calculate_dynamic_boost(
    suggested_temp: float,
    error_target_vs_actual: float,
    outdoor_temp: float,
    baseline_temp: float,
) -> float:
    """
    Calculates the final temperature with dynamic boost and clamping,
    matching the original script's logic.
    """
    logging.debug("--- Final Temp Calculation ---")
    logging.debug(f"  Model Suggested: {suggested_temp:.1f}°C")

    boost = 0.0
    if error_target_vs_actual > 0.1:
        boost = min(error_target_vs_actual * 2.0, 5.0)
    elif error_target_vs_actual < -0.1:
        boost = max(error_target_vs_actual * 1.5, -5.0)

    # Do not boost if the sun is shining
    if outdoor_temp > 15:
        boost = min(boost, 0)

    logging.debug(f"  Dynamic Boost Applied: {boost:.2f}°C")
    boosted_temp = suggested_temp + boost
    logging.debug(f"  Boosted Temp: {boosted_temp:.1f}°C")

    # Clamp the final temperature to be within a reasonable range of the baseline
    lower_bound = baseline_temp - 10
    upper_bound = baseline_temp + 12
    final_temp = np.clip(boosted_temp, lower_bound, upper_bound)
    logging.debug(
        "  Final Temp (clamped to %.1f-%.1f°C): %.1f°C",
        lower_bound,
        upper_bound,
        final_temp,
    )

    return round(final_temp)
