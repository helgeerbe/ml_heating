"""
This module encapsulates model logic for the ML heating controller.

It handles loading, prediction, and persistence of the RealisticPhysicsModel.
The physics model learns house characteristics and external heat source
effects from historical data, providing accurate predictions without
online ML training.
"""
import logging
import pickle
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Support both package-relative and direct import for notebooks
try:
    from . import config  # Package-relative import
    from .physics_model import RealisticPhysicsModel
except ImportError:
    import config  # Direct import fallback for notebooks
    from physics_model import RealisticPhysicsModel


class MAE:
    """Mean Absolute Error metric"""
    def __init__(self):
        self._sum_abs_errors = 0.0
        self._n = 0

    def get(self):
        if self._n == 0:
            return 0.0
        return self._sum_abs_errors / self._n

    def update(self, y_true, y_pred):
        self._sum_abs_errors += abs(y_true - y_pred)
        self._n += 1


class RMSE:
    """Root Mean Squared Error metric"""
    def __init__(self):
        self._sum_squared_errors = 0.0
        self._n = 0

    def get(self):
        if self._n == 0:
            return 0.0
        return (self._sum_squared_errors / self._n) ** 0.5

    def update(self, y_true, y_pred):
        self._sum_squared_errors += (y_true - y_pred) ** 2
        self._n += 1


def load_model() -> Tuple[RealisticPhysicsModel, MAE, RMSE]:
    """
    Loads the persisted RealisticPhysicsModel and metrics from disk.

    Attempts to load from `config.MODEL_FILE`. If not found or corrupted,
    initializes a fresh RealisticPhysicsModel.

    Returns:
        Tuple containing the physics model, MAE metric tracker, and
        RMSE metric tracker.
    """
    try:
        with open(config.MODEL_FILE, "rb") as f:
            saved_data = pickle.load(f)
            # Dictionary format containing model and metrics
            if isinstance(saved_data, dict):
                model = saved_data["model"]
                mae = saved_data.get("mae", MAE())
                rmse = saved_data.get("rmse", RMSE())

                # Ensure loaded model is RealisticPhysicsModel
                if not isinstance(model, RealisticPhysicsModel):
                    logging.warning(
                        "Loaded model is not RealisticPhysicsModel "
                        "(type: %s), creating new model.",
                        type(model).__name__
                    )
                    model = RealisticPhysicsModel()
                else:
                    logging.info(
                        "Successfully loaded RealisticPhysicsModel from %s",
                        config.MODEL_FILE,
                    )
            else:
                # Old format - create new model
                logging.warning(
                    "Old save format detected, creating new "
                    "RealisticPhysicsModel."
                )
                model = RealisticPhysicsModel()
                mae = MAE()
                rmse = RMSE()

            return model, mae, rmse
    except (FileNotFoundError, pickle.UnpicklingError, EOFError,
            AttributeError) as e:
        logging.warning(
            "Could not load model from %s (error: %s), creating new "
            "RealisticPhysicsModel.",
            config.MODEL_FILE, e
        )

        # Create new RealisticPhysicsModel
        model = RealisticPhysicsModel()
        mae = MAE()
        rmse = RMSE()

        logging.info("🎯 RealisticPhysicsModel initialized:")
        logging.info("   - MAE: %.4f°C", mae.get())
        logging.info("   - RMSE: %.4f°C", rmse.get())
        logging.info(
            "   - Physics-based predictions with learned house "
            "characteristics"
        )
        logging.info(
            "   - Integrated DHW, fireplace, PV, defrost handling"
        )

        return model, mae, rmse


def save_model(
    model: RealisticPhysicsModel, mae: MAE, rmse: RMSE
) -> None:
    """
    Saves the current state of the model and its metrics to a file.

    This is crucial for persistence, allowing the service to restart
    without losing learned knowledge.

    Args:
        model: The RealisticPhysicsModel instance.
        mae: The Mean Absolute Error metric object.
        rmse: The Root Mean Squared Error metric object.
    """
    try:
        with open(config.MODEL_FILE, "wb") as f:
            pickle.dump({
                "model": model,
                "mae": mae,
                "rmse": rmse
            }, f)
            logging.debug(
                "Successfully saved model and metrics to %s",
                config.MODEL_FILE
            )
    except Exception as e:
        logging.error(
            "Failed to save model and metrics to %s: %s",
            config.MODEL_FILE,
            e,
        )


def find_best_outlet_temp(
    model: RealisticPhysicsModel,
    features: pd.DataFrame,
    current_temp: float,
    target_temp: float,
    prediction_history: list,
    outlet_history: list[float],
    error_target_vs_actual: float,
    outdoor_temp: float,
) -> tuple[float, float, list, float]:
    """
    The core optimization function to find the ideal heating outlet
    temperature.

    This function orchestrates a multi-step process to determine the best
    temperature setting:
    1.  **Raw Prediction**: It performs a grid search over a range of
        possible outlet temperatures, recording the raw predicted indoor
        temperature for each using the physics model.
    2.  **Monotonic Enforcement**: It corrects the raw predictions to
        enforce a physically plausible, non-decreasing relationship between
        outlet temp and indoor temp. This is anchored to the most recent
        actual outlet temperature for reliability.
    3.  **Optimal Search**: It searches the corrected, monotonic curve to
        find the most energy-efficient (i.e., lowest) outlet temperature
        that achieves the best possible outcome.
    4.  **Smoothing & Finalization**: The result is smoothed, boosted, and
        rounded to produce the final, stable setpoint.

    Args:
        model: The RealisticPhysicsModel instance.
        features: The input features for the current time step.
        current_temp: The current indoor temperature.
        target_temp: The desired indoor temperature.
        prediction_history: A list of recent smoothed predictions.
        outlet_history: A list of recent actual outlet temperatures.

    Returns:
        A tuple with the final outlet temp, confidence, updated prediction
        history, and raw model uncertainty (sigma).
    """
    logging.info("--- Finding Best Outlet Temp ---")
    logging.info(f"Target indoor temp: {target_temp:.1f}°C")
    x_base = features.to_dict(orient="records")[0]
    last_outlet_temp = outlet_history[-1] if outlet_history else 35.0

    # --- Confidence Monitoring ---
    # Use real-time sigma based on recent prediction accuracy
    sigma = model.get_realtime_sigma()
    confidence = model.get_realtime_confidence()
    logging.info(
        "RealisticPhysicsModel: real-time confidence=%.3f (σ=%.3f°C)",
        confidence, sigma
    )

    if confidence < config.CONFIDENCE_THRESHOLD:
        logging.warning(
            "Model confidence low (σ=%.3f°C, confidence=%.3f < %.3f). "
            "Proceeding, but the result might be less reliable.",
            sigma,
            confidence,
            config.CONFIDENCE_THRESHOLD,
        )

    # --- 1. Raw Prediction Search ---
    min_search_temp = config.CLAMP_MIN_ABS
    max_search_temp = config.CLAMP_MAX_ABS
    step = 0.5
    search_range = np.arange(min_search_temp, max_search_temp + step, step)
    
    logging.info(
        f"Searching outlet range [{min_search_temp:.1f}°C - "
        f"{max_search_temp:.1f}°C] with step={step}°C "
        f"({len(search_range)} candidates)"
    )

    raw_deltas = {}
    raw_predictions = {}
    for temp_candidate in search_range:
        x_candidate = x_base.copy()
        x_candidate.update(
            {
                "outlet_temp": temp_candidate,
                "outlet_temp_sq": temp_candidate**2,
                "outlet_temp_cub": temp_candidate**3,
                "outlet_temp_change_from_last": (
                    temp_candidate - last_outlet_temp
                ),
                "outlet_indoor_diff": temp_candidate - current_temp,
                "outdoor_temp_x_outlet_temp": outdoor_temp * temp_candidate,
            }
        )
        predicted_delta = model.predict_one(x_candidate)
        predicted_indoor = current_temp + predicted_delta
        raw_deltas[temp_candidate] = predicted_delta
        raw_predictions[temp_candidate] = predicted_indoor

    # --- 2. Monotonic Enforcement During Optimization ---
    # Enforce physics constraint: higher outlet temp → higher indoor temp
    # This ensures the optimization search respects thermodynamic principles

    # Find the last outlet temperature in our search range
    last_outlet_rounded = round(last_outlet_temp * 2) / 2
    last_outlet_rounded = np.clip(
        last_outlet_rounded, min_search_temp, max_search_temp
    )

    # Anchor prediction at last outlet temp
    if last_outlet_rounded in raw_predictions:
        anchor_temp = last_outlet_rounded
        anchor_prediction = raw_predictions[anchor_temp]
    else:
        anchor_temp = min_search_temp
        anchor_prediction = raw_predictions[anchor_temp]

    logging.info(
        f"Physics anchor: outlet={anchor_temp:.1f}°C → "
        f"indoor={anchor_prediction:.2f}°C"
    )

    # Force monotonically increasing predictions
    monotonic_predictions = {}
    min_slope = 0.02  # Minimum heating per degree outlet increase

    for temp_candidate in sorted(search_range):
        if temp_candidate < anchor_temp:
            # Below anchor: predictions must be lower
            if temp_candidate in raw_predictions:
                raw_pred = raw_predictions[temp_candidate]
            else:
                raw_pred = anchor_prediction - (
                    anchor_temp - temp_candidate
                ) * min_slope

            max_allowed = anchor_prediction - (
                anchor_temp - temp_candidate
            ) * min_slope
            monotonic_predictions[temp_candidate] = min(
                raw_pred, max_allowed
            )

        elif temp_candidate == anchor_temp:
            monotonic_predictions[temp_candidate] = anchor_prediction

        else:
            # Above anchor: predictions must be higher
            if temp_candidate in raw_predictions:
                raw_pred = raw_predictions[temp_candidate]
            else:
                raw_pred = anchor_prediction + (
                    temp_candidate - anchor_temp
                ) * min_slope

            min_allowed = anchor_prediction + (
                temp_candidate - anchor_temp
            ) * min_slope
            monotonic_predictions[temp_candidate] = max(
                raw_pred, min_allowed
            )
    
    # Debug: log complete search range with corrections
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug("Complete search range with monotonic corrections:")
        for temp_candidate in sorted(search_range):
            raw_pred = raw_predictions.get(temp_candidate, None)
            mono_pred = monotonic_predictions.get(temp_candidate, None)
            delta = raw_deltas.get(temp_candidate, None)
            if raw_pred is not None and mono_pred is not None:
                logging.debug(
                    f"  - Test {temp_candidate:.1f}°C -> "
                    f"Pred ΔT: {delta:.3f}°C, "
                    f"Raw Indoor: {raw_pred:.2f}°C, "
                    f"Corrected: {mono_pred:.2f}°C"
                )

    # --- 3. Find Optimal Outlet Temperature ---
    best_outlet_temp = None
    best_outcome = -999

    for temp_candidate in sorted(search_range):
        predicted_indoor = monotonic_predictions[temp_candidate]

        # Calculate how close we get to target
        overshoot = max(0, predicted_indoor - target_temp)
        undershoot = max(0, target_temp - predicted_indoor)

        # Strong penalty for being below target
        penalty = undershoot * 5.0 + overshoot * 2.0

        # Energy efficiency bonus for lower outlet temps
        efficiency_bonus = (max_search_temp - temp_candidate) * 0.01

        outcome = -penalty + efficiency_bonus

        if outcome > best_outcome:
            best_outcome = outcome
            best_outlet_temp = temp_candidate

    raw_pred = raw_predictions.get(best_outlet_temp, None)
    raw_pred_str = (
        f"{raw_pred:.2f}" if isinstance(raw_pred, (int, float)) else "N/A"
    )
    logging.info(
        f"Optimization result: outlet={best_outlet_temp:.1f}°C → "
        f"predicted_indoor={monotonic_predictions[best_outlet_temp]:.2f}°C "
        f"(target={target_temp:.1f}°C, raw={raw_pred_str}°C)"
    )

    # --- 4. Smoothing & Finalization ---
    prediction_history.append(best_outlet_temp)
    if len(prediction_history) > 3:
        prediction_history.pop(0)

    history_mean = np.mean(prediction_history)
    smoothed_outlet = float(
        best_outlet_temp * config.SMOOTHING_ALPHA +
        history_mean * (1 - config.SMOOTHING_ALPHA)
    )
    
    logging.info(
        f"Smoothing: best={best_outlet_temp:.1f}°C, "
        f"history_mean={history_mean:.1f}°C, "
        f"smoothed={smoothed_outlet:.1f}°C"
    )

    # Apply boost if we're significantly below target
    boost = 0.0
    if error_target_vs_actual < -0.5:
        boost = min(2.0, abs(error_target_vs_actual) * 0.5)
        logging.info(
            f"Boost: +{boost:.1f}°C due to error={error_target_vs_actual:.2f}°C"
        )
    else:
        logging.info(
            f"No boost needed (error={error_target_vs_actual:.2f}°C >= -0.5°C)"
        )

    before_rounding = smoothed_outlet + boost
    
    # Smart Rounding: Test both floor and ceiling to see which gets closer
    # to target indoor temperature
    floor_temp = np.floor(before_rounding)
    ceiling_temp = np.ceil(before_rounding)
    
    if floor_temp == ceiling_temp:
        # Already an integer
        after_rounding = floor_temp
        logging.debug(
            f"Smart rounding: {before_rounding:.2f}°C is already integer"
        )
    else:
        # Test both options and pick the one that gets closer to target
        floor_features = x_base.copy()
        floor_features.update({
            "outlet_temp": floor_temp,
            "outlet_temp_sq": floor_temp ** 2,
            "outlet_temp_cub": floor_temp ** 3,
            "outlet_temp_change_from_last": floor_temp - last_outlet_temp,
            "outlet_indoor_diff": floor_temp - current_temp,
            "outdoor_temp_x_outlet_temp": (
                floor_features.get("outdoor_temp", outdoor_temp) *
                floor_temp
            ),
        })
        
        ceiling_features = x_base.copy()
        ceiling_features.update({
            "outlet_temp": ceiling_temp,
            "outlet_temp_sq": ceiling_temp ** 2,
            "outlet_temp_cub": ceiling_temp ** 3,
            "outlet_temp_change_from_last": ceiling_temp - last_outlet_temp,
            "outlet_indoor_diff": ceiling_temp - current_temp,
            "outdoor_temp_x_outlet_temp": (
                ceiling_features.get("outdoor_temp", outdoor_temp) *
                ceiling_temp
            ),
        })
        
        floor_delta = model.predict_one(floor_features)
        ceiling_delta = model.predict_one(ceiling_features)
        
        floor_predicted_indoor = current_temp + floor_delta
        ceiling_predicted_indoor = current_temp + ceiling_delta
        
        floor_error = abs(floor_predicted_indoor - target_temp)
        ceiling_error = abs(ceiling_predicted_indoor - target_temp)
        
        if floor_error <= ceiling_error:
            after_rounding = floor_temp
            chosen = "floor"
        else:
            after_rounding = ceiling_temp
            chosen = "ceiling"
        
        logging.info(
            f"Smart rounding: {before_rounding:.2f}°C → "
            f"{after_rounding:.0f}°C (chose {chosen}: "
            f"floor→{floor_predicted_indoor:.2f}°C [err={floor_error:.2f}], "
            f"ceiling→{ceiling_predicted_indoor:.2f}°C "
            f"[err={ceiling_error:.2f}], target={target_temp:.1f}°C)"
        )

    # Final clamping
    before_clamp = after_rounding
    final_outlet_temp = np.clip(
        before_clamp, config.CLAMP_MIN_ABS, config.CLAMP_MAX_ABS
    )
    
    if final_outlet_temp != before_clamp:
        logging.info(
            f"Clamping: {before_clamp:.1f}°C → {final_outlet_temp:.1f}°C "
            f"(limits: [{config.CLAMP_MIN_ABS:.1f}, {config.CLAMP_MAX_ABS:.1f}])"
        )

    logging.info(
        f"Final outlet temp: {final_outlet_temp:.1f}°C "
        f"(confidence={confidence:.3f}, σ={sigma:.3f}°C)"
    )

    return final_outlet_temp, confidence, prediction_history, sigma


def get_feature_importances(model: RealisticPhysicsModel) -> Dict[str, float]:
    """
    Get feature importances for RealisticPhysicsModel.

    For the physics model, returns physics-based importances indicating
    which features have the strongest influence on predictions.

    Args:
        model: The RealisticPhysicsModel instance

    Returns:
        Dictionary mapping feature names to importance scores (0.0-1.0)
    """
    # Physics-based feature importances for all 19 features
    feature_importances = {
        # Core temperatures (highest importance)
        'outlet_temp': 0.25,  # Primary physics driver
        'target_temp': 0.15,  # Target temperature influence
        'indoor_temp_lag_30m': 0.15,  # Temperature difference calculation
        'outdoor_temp': 0.12,  # Weather influence
        
        # System states
        'dhw_heating': 0.08,  # DHW heating state
        'defrosting': 0.05,  # Defrost cycle handling
        'dhw_disinfection': 0.02,  # DHW disinfection
        'dhw_boost_heater': 0.02,  # DHW boost heater
        
        # External heat sources
        'fireplace_on': 0.04,  # Fireplace heat
        'pv_now': 0.03,  # Solar warming
        'tv_on': 0.01,  # TV heat contribution
        
        # Temperature forecasts (1-4 hours)
        'temp_forecast_1h': 0.02,
        'temp_forecast_2h': 0.015,
        'temp_forecast_3h': 0.01,
        'temp_forecast_4h': 0.01,
        
        # PV forecasts (1-4 hours)
        'pv_forecast_1h': 0.015,
        'pv_forecast_2h': 0.01,
        'pv_forecast_3h': 0.005,
        'pv_forecast_4h': 0.005,
    }

    logging.debug("Physics-based feature importances: %s", feature_importances)
    return feature_importances
