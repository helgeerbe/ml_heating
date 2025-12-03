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


def predict_thermal_trajectory(
    model: RealisticPhysicsModel,
    features: pd.DataFrame,
    outlet_temp: float,
    steps: int = 4
) -> list[float]:
    """
    Predict 4-hour thermal trajectory for a given outlet temperature.
    
    Uses iterative prediction with weather and PV forecasts to simulate
    how indoor temperature will evolve over the next 4 hours.
    
    Args:
        model: The RealisticPhysicsModel instance
        features: Current feature set
        outlet_temp: Outlet temperature to test
        steps: Number of 1-hour steps to predict (default 4)
        
    Returns:
        List of predicted indoor temperatures for each hour
    """
    trajectory = []
    current_features = features.to_dict(orient="records")[0].copy()
    
    # Get forecast arrays from current features
    temp_forecasts = [
        current_features.get('temp_forecast_1h', 0.0),
        current_features.get('temp_forecast_2h', 0.0), 
        current_features.get('temp_forecast_3h', 0.0),
        current_features.get('temp_forecast_4h', 0.0)
    ]
    
    pv_forecasts = [
        current_features.get('pv_forecast_1h', 0.0),
        current_features.get('pv_forecast_2h', 0.0),
        current_features.get('pv_forecast_3h', 0.0), 
        current_features.get('pv_forecast_4h', 0.0)
    ]
    
    # Current indoor temperature from features
    current_indoor = current_features.get('indoor_temp_lag_30m', 20.0)
    last_outlet_temp = current_features.get('outlet_temp', 35.0)
    outdoor_temp = current_features.get('outdoor_temp', 10.0)
    
    # CRITICAL FIX: Store original outlet temp to restore for calculations
    original_outlet_temp = current_features.get('outlet_temp', 35.0)
    
    for step in range(steps):
        # Set the test outlet temperature and ALL related features
        current_features.update({
            'outlet_temp': outlet_temp,
            'outlet_temp_sq': outlet_temp ** 2,
            'outlet_temp_cub': outlet_temp ** 3,
            'outlet_temp_change_from_last': outlet_temp - last_outlet_temp,
            'outlet_indoor_diff': outlet_temp - current_indoor,
            'outdoor_temp_x_outlet_temp': outdoor_temp * outlet_temp,
        })
        
        # CRITICAL: Ensure other temperature-based features are consistent
        current_features['target_temp'] = current_features.get('target_temp', 21.0)
        
        # DEBUG: Log what we're passing to the model for first two outlets
        if outlet_temp in [14.0, 15.0, 50.0, 51.0] and step == 0:
            logging.info(
                f"TRAJECTORY DEBUG outlet={outlet_temp:.1f}°C step={step}: "
                f"outlet_indoor_diff={outlet_temp - current_indoor:.1f}°C, "
                f"target={current_features['target_temp']:.1f}°C, "
                f"indoor={current_indoor:.1f}°C, outdoor={outdoor_temp:.1f}°C"
            )
        
        # Predict temperature delta for this step
        predicted_delta = model.predict_one(current_features)
        predicted_indoor = current_indoor + predicted_delta
        trajectory.append(predicted_indoor)
        
        # DEBUG: Log prediction result for first two outlets
        if outlet_temp in [14.0, 15.0, 50.0, 51.0] and step == 0:
            logging.info(
                f"TRAJECTORY DEBUG result: delta={predicted_delta:.4f}°C, "
                f"new_indoor={predicted_indoor:.2f}°C"
            )
        
        # Update features for next prediction step
        current_indoor = predicted_indoor
        current_features['indoor_temp_lag_30m'] = predicted_indoor
        last_outlet_temp = outlet_temp  # For next iteration's change calculation
        
        # Update forecasts for next hour
        if step < len(temp_forecasts) - 1:
            current_features['temp_forecast_1h'] = temp_forecasts[step + 1]
        if step < len(pv_forecasts) - 1:
            current_features['pv_forecast_1h'] = pv_forecasts[step + 1]
        else:
            # After midnight or end of forecasts, set PV to 0
            current_features['pv_forecast_1h'] = 0.0
            
    return trajectory


def apply_smart_rounding(
    before_rounding: float,
    model: RealisticPhysicsModel,
    x_base: dict,
    current_temp: float,
    target_temp: float,
    last_outlet_temp: float,
    outdoor_temp: float
) -> float:
    """
    Smart Rounding: Test both floor and ceiling to see which gets closer
    to target indoor temperature.
    
    Tests floor vs ceiling temperatures and chooses the one that results
    in predicted indoor temperature closer to target. This compensates for
    heat pump truncation while optimizing for target achievement.
    
    Args:
        before_rounding: Optimal outlet temperature before rounding
        model: The RealisticPhysicsModel instance
        x_base: Base feature dictionary for predictions
        current_temp: Current indoor temperature
        target_temp: Target indoor temperature
        last_outlet_temp: Previous outlet temperature for change calculation
        outdoor_temp: Current outdoor temperature
        
    Returns:
        Smart rounded outlet temperature (integer)
    """
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
    
    return after_rounding

def evaluate_trajectory_stability(trajectory: list[float], target_temp: float, 
                                current_temp: float = None) -> float:
    """
    Evaluate the stability of a temperature trajectory using hybrid scoring.
    
    Combines deviation from target, oscillation penalty, final destination,
    and direction correctness to score trajectory quality. Lower scores 
    indicate better stability.
    
    Args:
        trajectory: List of predicted indoor temperatures
        target_temp: Desired target temperature
        current_temp: Current indoor temperature (for direction analysis)
        
    Returns:
        Stability score (lower is better)
    """
    if not trajectory:
        return float('inf')
    
    # 1. Total deviation from target
    deviation_score = sum(abs(temp - target_temp) for temp in trajectory)
    
    # 2. Oscillation penalty (detect direction changes)
    oscillation_penalty = 0.0
    for i in range(1, len(trajectory) - 1):
        prev_trend = trajectory[i] - trajectory[i-1]
        next_trend = trajectory[i+1] - trajectory[i]
        if prev_trend * next_trend < 0:  # Direction reversal
            oscillation_penalty += config.OSCILLATION_PENALTY_WEIGHT
    
    # 3. Final destination check
    final_error = abs(trajectory[-1] - target_temp)
    final_penalty = final_error * config.FINAL_DESTINATION_WEIGHT
    
    # 4. Direction correctness penalty (NEW - critical for cooling/heating)
    direction_penalty = 0.0
    if current_temp is not None and len(trajectory) > 0:
        needed_direction = target_temp - current_temp  # Negative = need cooling
        actual_direction = trajectory[-1] - current_temp  # Negative = cooling
        
        # Heavy penalty if trajectory moves away from target
        if needed_direction * actual_direction < 0:
            # Moving in wrong direction - massive penalty
            direction_penalty = abs(needed_direction) * 5.0
        elif abs(actual_direction) < abs(needed_direction) * 0.1:
            # Moving in right direction but too slowly - moderate penalty
            direction_penalty = abs(needed_direction) * 1.0
    
    # Combined score (lower is better)
    total_score = deviation_score + oscillation_penalty + final_penalty + direction_penalty
    
    return total_score


def determine_control_mode(current_temp: float, target_temp: float) -> str:
    """
    Determine which heat balance controller mode to use.
    
    Args:
        current_temp: Current indoor temperature
        target_temp: Target indoor temperature
        
    Returns:
        Control mode: "CHARGING", "BALANCING", or "MAINTENANCE"
    """
    temperature_error = abs(target_temp - current_temp)
    
    if temperature_error > config.CHARGING_MODE_THRESHOLD:
        return "CHARGING"
    elif temperature_error > config.MAINTENANCE_MODE_THRESHOLD:
        return "BALANCING"
    else:
        return "MAINTENANCE"


def find_best_outlet_temp(
    model: RealisticPhysicsModel,
    features: pd.DataFrame,
    current_temp: float,
    target_temp: float,
    outlet_history: list[float],
    error_target_vs_actual: float,
    outdoor_temp: float,
) -> tuple[float, float, str, float, float, list[float], tuple[float, float]]:
    """
    Heat Balance Controller: Find optimal outlet temperature using trajectory prediction.
    
    Replaces smoothing-based approach with intelligent 3-phase control:
    - CHARGING: Aggressive heating when far from target (>0.2°C error)
    - BALANCING: Trajectory stability optimization (0.1-0.2°C error)  
    - MAINTENANCE: Minimal adjustments when at target (<0.1°C error)
    
    Args:
        model: The RealisticPhysicsModel instance
        features: The input features for current time step
        current_temp: Current indoor temperature
        target_temp: Desired indoor temperature
        outlet_history: List of recent actual outlet temperatures
        error_target_vs_actual: Current temperature error
        outdoor_temp: Current outdoor temperature

    Returns:
        Tuple with (final_outlet_temp, confidence, control_mode, sigma,
                   trajectory_stability_score, predicted_trajectory, tested_outlet_range)
    """
    logging.info("--- Heat Balance Controller: Finding Best Outlet Temp ---")
    logging.info(f"Target: {target_temp:.1f}°C, Current: {current_temp:.1f}°C")
    
    # Determine control mode based on temperature error
    control_mode = determine_control_mode(current_temp, target_temp)
    logging.info(f"Control Mode: {control_mode}")
    
    # Get confidence metrics
    sigma = model.get_realtime_sigma()
    confidence = model.get_realtime_confidence()
    
    if confidence < config.CONFIDENCE_THRESHOLD:
        logging.warning(
            "Model confidence low (σ=%.3f°C, confidence=%.3f < %.3f)",
            sigma, confidence, config.CONFIDENCE_THRESHOLD
        )
    
    # Define search parameters
    min_search_temp = config.CLAMP_MIN_ABS
    max_search_temp = config.CLAMP_MAX_ABS
    
    # BALANCING MODE: Use direction-aware smooth adjustment range
    # This respects heat pump stability while ensuring proper cooling/heating search
    if control_mode == "BALANCING":
        current_outlet_temp = outlet_history[-1] if outlet_history else 35.0
        adjustment_range = 8.0  # ±8°C adjustment range
        
        # Direction-aware range: bias search toward needed direction
        if current_temp > target_temp:
            # Need cooling: search more in cooling range (below current)
            cooling_bias = 6.0  # Strong bias toward lower temperatures for cooling
            min_search_temp = max(config.CLAMP_MIN_ABS, 
                                current_outlet_temp - adjustment_range - cooling_bias)
            max_search_temp = min(config.CLAMP_MAX_ABS, 
                                current_outlet_temp + adjustment_range - cooling_bias)
            direction = "cooling"
        else:
            # Need heating: search more in heating range (above current)
            heating_bias = 6.0  # Strong bias toward higher temperatures for heating
            min_search_temp = max(config.CLAMP_MIN_ABS, 
                                current_outlet_temp - adjustment_range + heating_bias)
            max_search_temp = min(config.CLAMP_MAX_ABS, 
                                current_outlet_temp + adjustment_range + heating_bias)
            direction = "heating"
        
        logging.info(
            f"Balancing mode: {direction}-biased range around current outlet "
            f"{current_outlet_temp:.1f}°C → [{min_search_temp:.1f}°C - {max_search_temp:.1f}°C]"
        )
    
    step = 1.0 if control_mode == "CHARGING" else 0.5
    search_range = np.arange(min_search_temp, max_search_temp + step, step)
    
    logging.info(
        f"Searching outlet range [{min_search_temp:.1f}°C - "
        f"{max_search_temp:.1f}°C] with step={step}°C"
    )
    
    best_outlet_temp = None
    best_score = float('inf')
    best_trajectory = None
    
    last_outlet_temp = outlet_history[-1] if outlet_history else 35.0
    
    if control_mode == "CHARGING":
        # Charging mode: Use target-reaching optimization with monotonic enforcement
        logging.info("CHARGING mode: Using target-reaching optimization with monotonic enforcement")
        
        # Step 1: Find anchor point (last outlet temperature)
        last_outlet_rounded = round(last_outlet_temp * 2) / 2
        last_outlet_rounded = np.clip(
            last_outlet_rounded, min_search_temp, max_search_temp
        )
        
        # Get anchor prediction
        anchor_temp = last_outlet_rounded
        anchor_features = features.to_dict(orient="records")[0].copy()
        anchor_features.update({
            "outlet_temp": anchor_temp,
            "outlet_temp_sq": anchor_temp**2,
            "outlet_temp_cub": anchor_temp**3,
            "outlet_temp_change_from_last": 0,
            "outlet_indoor_diff": anchor_temp - current_temp,
            "outdoor_temp_x_outlet_temp": outdoor_temp * anchor_temp,
        })
        anchor_prediction = current_temp + model.predict_one(anchor_features)
        
        logging.debug(
            f"Physics anchor: outlet={anchor_temp:.1f}°C → "
            f"indoor={anchor_prediction:.2f}°C"
        )
        
        # Step 2: Build predictions using raw physics (no monotonic enforcement)
        # Trust the bidirectional physics model for accurate cooling/heating predictions
        raw_predictions = {}
        raw_deltas = {}  # Track predicted deltas for debug logging
        
        for temp_candidate in search_range:
            # Get raw prediction
            x_candidate = features.to_dict(orient="records")[0].copy()
            x_candidate.update({
                "outlet_temp": temp_candidate,
                "outlet_temp_sq": temp_candidate**2,
                "outlet_temp_cub": temp_candidate**3,
                "outlet_temp_change_from_last": temp_candidate - last_outlet_temp,
                "outlet_indoor_diff": temp_candidate - current_temp,
                "outdoor_temp_x_outlet_temp": outdoor_temp * temp_candidate,
            })
            predicted_delta = model.predict_one(x_candidate)
            raw_prediction = current_temp + predicted_delta
            raw_predictions[temp_candidate] = raw_prediction
            raw_deltas[temp_candidate] = predicted_delta  # Store for debug logging
        
        # Debug: log complete search range with raw physics predictions
        if logging.getLogger().isEnabledFor(logging.INFO):
            logging.info("Complete search range with raw physics predictions:")
            for temp_candidate in sorted(search_range):
                raw_pred = raw_predictions.get(temp_candidate, None)
                delta = raw_deltas.get(temp_candidate, None)
                if raw_pred is not None:
                    logging.info(
                        f"  - Test {temp_candidate:.1f}°C -> "
                        f"Pred ΔT: {delta:.3f}°C, "
                        f"Predicted Indoor: {raw_pred:.2f}°C"
                    )
        
        # Step 3: Battery Charger Optimization - Two-stage logic
        temperature_gap = target_temp - current_temp
        PRECISION_THRESHOLD = 0.2  # Switch to fine-tuning when within 0.2°C
        
        best_outcome = -999
        best_prediction = current_temp  # Initialize to avoid undefined variable
        
        if abs(temperature_gap) > PRECISION_THRESHOLD:
            # Stage 1: Maximum Progress Mode (Battery Charger Logic)
            logging.info(f"Stage 1: Maximum Progress Mode (gap={temperature_gap:.2f}°C > {PRECISION_THRESHOLD}°C)")
            
            if temperature_gap > 0:
                # Need heating: Use maximum available heating
                # Find the highest temperature that provides positive heating
                best_heating = -999
                for temp_candidate in sorted(search_range, reverse=True):  # Start from hottest
                    predicted_indoor = raw_predictions[temp_candidate]
                    heating_progress = predicted_indoor - current_temp
                    
                    if heating_progress > best_heating:
                        best_heating = heating_progress
                        best_outlet_temp = temp_candidate
                        best_prediction = predicted_indoor
                        best_outcome = heating_progress  # Maximize progress
                
                logging.info(f"Heating mode: Selected outlet={best_outlet_temp:.1f}°C for maximum progress={best_heating:.3f}°C")
                
            else:
                # Need cooling: Use maximum available cooling
                # Find the lowest temperature that provides positive cooling
                best_cooling = 999
                for temp_candidate in sorted(search_range):  # Start from coolest
                    predicted_indoor = raw_predictions[temp_candidate]
                    cooling_progress = current_temp - predicted_indoor  # Positive = cooling
                    
                    if predicted_indoor < best_cooling:
                        best_cooling = predicted_indoor
                        best_outlet_temp = temp_candidate
                        best_prediction = predicted_indoor
                        best_outcome = cooling_progress  # Maximize cooling
                
                logging.info(f"Cooling mode: Selected outlet={best_outlet_temp:.1f}°C for maximum cooling to {best_cooling:.2f}°C")
        
        else:
            # Stage 2: Fine Balancing Mode (Original Logic for Precision)
            logging.info(f"Stage 2: Fine Balancing Mode (gap={temperature_gap:.2f}°C <= {PRECISION_THRESHOLD}°C)")
            
            for temp_candidate in search_range:
                predicted_indoor = raw_predictions[temp_candidate]
                
                # Calculate outcome score (closer to target = better)
                overshoot = max(0, predicted_indoor - target_temp)
                undershoot = max(0, target_temp - predicted_indoor)
                penalty = undershoot * 5.0 + overshoot * 2.0
                efficiency_bonus = (max_search_temp - temp_candidate) * 0.01
                outcome = -penalty + efficiency_bonus
                
                if outcome > best_outcome:
                    best_outcome = outcome
                    best_outlet_temp = temp_candidate
                    best_prediction = predicted_indoor
            
            logging.info(f"Fine balancing: Selected outlet={best_outlet_temp:.1f}°C for precision targeting")
        
        best_score = -best_outcome
        best_trajectory = [best_prediction]
        
        logging.info(
            f"CHARGING result: outlet={best_outlet_temp:.1f}°C → "
            f"predicted_indoor={best_prediction:.2f}°C (target={target_temp:.1f}°C)"
        )
        
    elif control_mode == "BALANCING":
        # Balancing mode: Use trajectory stability optimization
        logging.info("BALANCING mode: Using trajectory stability optimization")
        
        # Store all candidates and their details for logging
        trajectory_candidates = {}
        
        for temp_candidate in search_range:
            trajectory = predict_thermal_trajectory(
                model, features, temp_candidate, steps=config.TRAJECTORY_STEPS
            )
            stability_score = evaluate_trajectory_stability(
                trajectory, target_temp, current_temp
            )
            
            # Store for logging
            trajectory_candidates[temp_candidate] = {
                'trajectory': trajectory,
                'stability_score': stability_score
            }
            
            if stability_score < best_score:
                best_score = stability_score
                best_outlet_temp = temp_candidate
                best_trajectory = trajectory
        
        # Debug: log complete search range with trajectory analysis
        if logging.getLogger().isEnabledFor(logging.INFO):
            logging.info("Complete trajectory analysis for balancing mode:")
            for temp_candidate in sorted(trajectory_candidates.keys()):
                candidate_data = trajectory_candidates[temp_candidate]
                trajectory = candidate_data['trajectory']
                stability_score = candidate_data['stability_score']
                final_temp = trajectory[-1] if trajectory else 0.0
                logging.info(
                    f"  - Test {temp_candidate:.1f}°C -> "
                    f"Trajectory: {[f'{t:.2f}' for t in trajectory]} -> "
                    f"Final: {final_temp:.2f}°C, "
                    f"Stability: {stability_score:.3f}"
                )
                
    else:  # MAINTENANCE mode
        # Maintenance mode: Minimal adjustments with dead band for perfect temperature
        logging.info("MAINTENANCE mode: Using minimal adjustments with dead band")
        
        # Calculate adjustment with detailed logging
        temperature_error = current_temp - target_temp
        
        # Dead band: if error is essentially zero, keep outlet temp as is
        if abs(temperature_error) < 0.01:  # Within 0.01°C = essentially perfect
            adjustment = 0.0
            new_outlet_temp = last_outlet_temp
            action_reason = "perfect temperature (dead band)"
        elif current_temp < target_temp:
            adjustment = 0.5
            new_outlet_temp = last_outlet_temp + adjustment
            action_reason = "too cold"
        else:
            adjustment = -0.5
            new_outlet_temp = last_outlet_temp + adjustment
            action_reason = "too warm"
        
        # Debug: log maintenance mode decision process
        if logging.getLogger().isEnabledFor(logging.INFO):
            logging.info("Complete maintenance mode analysis:")
            logging.info(f"  - Current indoor: {current_temp:.2f}°C")
            logging.info(f"  - Target: {target_temp:.1f}°C") 
            logging.info(f"  - Error: {temperature_error:.3f}°C "
                        f"({action_reason})")
            logging.info(f"  - Last outlet: {last_outlet_temp:.1f}°C")
            if adjustment == 0.0:
                logging.info(f"  - Action: Keep as is (dead band)")
            else:
                logging.info(f"  - Adjustment: {adjustment:+.1f}°C → "
                             f"New outlet: {new_outlet_temp:.1f}°C")
            logging.info(f"  - Clamping range: "
                         f"[{min_search_temp:.1f}°C - "
                         f"{max_search_temp:.1f}°C]")
        
        best_outlet_temp = np.clip(
            new_outlet_temp,
            min_search_temp,
            max_search_temp
        )
        
        # Log if clamping occurred
        if best_outlet_temp != new_outlet_temp:
            logging.info(f"  - Clamped outlet: {new_outlet_temp:.1f}°C → {best_outlet_temp:.1f}°C")
        
        best_score = abs(current_temp - target_temp)
        best_trajectory = [current_temp]
    
    # Final processing
    if best_outlet_temp is None:
        best_outlet_temp = last_outlet_temp
        logging.warning("No optimal temperature found, using last outlet temp")
    
    # Apply Smart Rounding (before final clamping)
    x_base = features.to_dict(orient="records")[0]
    smart_rounded_temp = apply_smart_rounding(
        best_outlet_temp,
        model,
        x_base,
        current_temp,
        target_temp,
        last_outlet_temp,
        outdoor_temp
    )
    
    # Apply final clamping (safety net)
    final_outlet_temp = np.clip(
        smart_rounded_temp, config.CLAMP_MIN_ABS, config.CLAMP_MAX_ABS
    )
    
    logging.info(
        f"Heat Balance Controller result: outlet={final_outlet_temp:.1f}°C, "
        f"mode={control_mode}, score={best_score:.3f}, "
        f"confidence={confidence:.3f}"
    )
    
    if best_trajectory and len(best_trajectory) > 0:
        logging.info(f"Predicted trajectory: {best_trajectory}")
    
    # Prepare trajectory details for state sensor
    trajectory_stability_score = best_score
    predicted_trajectory = best_trajectory if best_trajectory else []
    tested_outlet_range = (min_search_temp, max_search_temp)
    
    return (
        final_outlet_temp,
        confidence,
        control_mode,
        sigma,
        trajectory_stability_score,
        predicted_trajectory,
        tested_outlet_range,
    )


def get_feature_importances(model: RealisticPhysicsModel) -> Dict[str, float]:
    """
    Get LEARNED feature importances from RealisticPhysicsModel.

    Dynamically calculates importances based on the model's actual learned 
    parameters instead of hardcoded values. This reflects what the model 
    has actually learned about your specific heating system.

    Args:
        model: The RealisticPhysicsModel instance

    Returns:
        Dictionary mapping feature names to importance scores (0.0-1.0)
    """
    # Extract learned coefficients from the model
    raw_importances = {}
    
    # Core physics parameters (normalized by typical ranges)
    # These control the main heating physics
    raw_importances['outlet_temp'] = model.base_heating_rate * 50  # Typical temp range ~50°C
    raw_importances['target_temp'] = model.target_influence * 10   # Typical gap ~10°C
    raw_importances['indoor_temp_lag_30m'] = model.base_heating_rate * 25  # For temp differences
    raw_importances['outdoor_temp'] = model.outdoor_factor * 30   # Typical range ~30°C
    
    # System states (binary, use as-is but scaled)
    raw_importances['dhw_heating'] = 0.08  # Fixed impact when active
    raw_importances['defrosting'] = 0.05   # Fixed impact when active  
    raw_importances['dhw_disinfection'] = 0.02  # Fixed impact when active
    raw_importances['dhw_boost_heater'] = 0.02  # Fixed impact when active
    
    # External heat sources - USE LEARNED VALUES!
    # Multi-lag coefficients (sum all lags for total importance)
    if hasattr(model, 'pv_coeffs') and model.pv_coeffs:
        pv_total_effect = sum(model.pv_coeffs.values())
        raw_importances['pv_now'] = pv_total_effect * 1000  # Scale by typical PV power
    else:
        # Fallback to simple learned coefficient
        raw_importances['pv_now'] = model.pv_warming_coefficient * 1000
    
    if hasattr(model, 'fireplace_coeffs') and model.fireplace_coeffs:
        fireplace_total_effect = sum(model.fireplace_coeffs.values())
        raw_importances['fireplace_on'] = fireplace_total_effect
    else:
        raw_importances['fireplace_on'] = model.fireplace_heating_rate
    
    if hasattr(model, 'tv_coeffs') and model.tv_coeffs:
        tv_total_effect = sum(model.tv_coeffs.values())
        raw_importances['tv_on'] = tv_total_effect
    else:
        raw_importances['tv_on'] = model.tv_heat_contribution
    
    # Forecast coefficients - learned from model
    forecast_weight = getattr(model, 'weather_forecast_coeff', 0.015)
    pv_forecast_weight = getattr(model, 'pv_forecast_coeff', 0.008)
    
    # Decay weights for forecast horizons
    decay_weights = getattr(model, 'forecast_decay', [1.0, 0.8, 0.6, 0.4])
    
    for i in range(4):
        hour = i + 1
        decay = decay_weights[i] if i < len(decay_weights) else 0.2
        
        # Weather forecasts
        raw_importances[f'temp_forecast_{hour}h'] = forecast_weight * decay * 10
        
        # PV forecasts  
        raw_importances[f'pv_forecast_{hour}h'] = pv_forecast_weight * decay * 100
    
    # Normalize to sum to 1.0
    total_raw = sum(raw_importances.values())
    if total_raw > 0:
        feature_importances = {
            feature: importance / total_raw 
            for feature, importance in raw_importances.items()
        }
    else:
        # Fallback if model not trained yet
        logging.warning("Model not trained yet, using default importances")
        feature_importances = _get_default_importances()
    
    # Log actual learned values for debugging
    logging.info("LEARNED Feature Importances (top 5):")
    sorted_features = sorted(feature_importances.items(), 
                           key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_features[:5]:
        logging.info(f"  {feature}: {importance:.4f}")
    
    # Log some key learned parameters that drive these importances
    logging.info("Key Learned Parameters:")
    logging.info(f"  base_heating_rate: {model.base_heating_rate:.6f}")
    logging.info(f"  target_influence: {model.target_influence:.6f}")
    logging.info(f"  pv_warming_coeff: {model.pv_warming_coefficient:.6f}")
    logging.info(f"  fireplace_heating: {model.fireplace_heating_rate:.6f}")
    logging.info(f"  tv_heat_contrib: {model.tv_heat_contribution:.6f}")
    
    if hasattr(model, 'pv_coeffs'):
        pv_total = sum(model.pv_coeffs.values())
        logging.info(f"  pv_multilag_total: {pv_total:.6f}")
    
    return feature_importances

def _get_default_importances() -> Dict[str, float]:
    """
    Fallback feature importances when model is not yet trained.
    
    Returns:
        Default importance dictionary
    """
    return {
        # Core temperatures (reasonable defaults)
        'outlet_temp': 0.25,
        'target_temp': 0.15, 
        'indoor_temp_lag_30m': 0.15,
        'outdoor_temp': 0.12,
        
        # System states
        'dhw_heating': 0.08,
        'defrosting': 0.05,
        'dhw_disinfection': 0.02,
        'dhw_boost_heater': 0.02,
        
        # External heat sources
        'fireplace_on': 0.04,
        'pv_now': 0.03,
        'tv_on': 0.01,
        
        # Temperature forecasts
        'temp_forecast_1h': 0.02,
        'temp_forecast_2h': 0.015,
        'temp_forecast_3h': 0.01,
        'temp_forecast_4h': 0.01,
        
        # PV forecasts
        'pv_forecast_1h': 0.015,
        'pv_forecast_2h': 0.01,
        'pv_forecast_3h': 0.005,
        'pv_forecast_4h': 0.005,
    }
