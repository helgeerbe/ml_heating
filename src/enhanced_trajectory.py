"""
Enhanced trajectory prediction with thermal momentum for Week 4.

This module provides an enhanced version of trajectory prediction that uses
the Week 1 thermal momentum features for better temperature stability control.
"""
import logging
from typing import List
import pandas as pd

# Support both package-relative and direct import
try:
    from . import config
    from .physics_model import RealisticPhysicsModel
except ImportError:
    import config
    from physics_model import RealisticPhysicsModel


def predict_thermal_trajectory_enhanced(
    model: RealisticPhysicsModel,
    features: pd.DataFrame,
    outlet_temp: float,
    steps: int = 4
) -> List[float]:
    """
    Enhanced thermal trajectory prediction using thermal momentum features.
    
    Builds on the original trajectory prediction but incorporates thermal
    momentum analysis from Week 1 features for improved accuracy.
    
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
    
    # Current state from features
    current_indoor = current_features.get('indoor_temp_lag_30m', 20.0)
    last_outlet_temp = current_features.get('outlet_temp', 35.0)
    outdoor_temp = current_features.get('outdoor_temp', 10.0)
    
    # ENHANCED: Extract thermal momentum features for better prediction
    thermal_momentum_features = {
        'indoor_temp_gradient': current_features.get('indoor_temp_gradient', 0.0),
        'temp_diff_indoor_outdoor': current_features.get('temp_diff_indoor_outdoor', 0.0),
        'outlet_indoor_diff': current_features.get('outlet_indoor_diff', 0.0),
        'indoor_temp_delta_10m': current_features.get('indoor_temp_delta_10m', 0.0),
        'indoor_temp_delta_30m': current_features.get('indoor_temp_delta_30m', 0.0),
    }
    
    logging.debug(
        f"Enhanced trajectory prediction: outlet={outlet_temp:.1f}°C, "
        f"momentum_gradient={thermal_momentum_features['indoor_temp_gradient']:.3f}°C/h, "
        f"temp_diff={thermal_momentum_features['temp_diff_indoor_outdoor']:.1f}°C"
    )
    
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
        
        # ENHANCED: Update thermal momentum features for this step
        if step > 0:
            # Update momentum features based on trajectory so far
            prev_indoor = trajectory[-1] if trajectory else current_indoor
            current_features['indoor_temp_gradient'] = (current_indoor - prev_indoor)  # 1-hour gradient
            current_features['temp_diff_indoor_outdoor'] = current_indoor - outdoor_temp
            current_features['outlet_indoor_diff'] = outlet_temp - current_indoor
            
        # Update forecasts for this hour
        if step < len(temp_forecasts):
            current_features['outdoor_temp'] = temp_forecasts[step]
            outdoor_temp = temp_forecasts[step]  # Update for next iteration
            
        if step < len(pv_forecasts):
            current_features['pv_now'] = pv_forecasts[step]
            
        # ENHANCED: Apply thermal momentum correction
        predicted_delta = model.predict_one(current_features)
        
        # Apply momentum-based smoothing for more realistic predictions
        momentum_factor = abs(thermal_momentum_features['indoor_temp_gradient'])
        if momentum_factor > 0.5:  # High momentum - apply gentle damping
            predicted_delta *= 0.85  # Reduce prediction by 15% for stability
            
        predicted_indoor = current_indoor + predicted_delta
        trajectory.append(predicted_indoor)
        
        # Update for next prediction step
        current_indoor = predicted_indoor
        current_features['indoor_temp_lag_30m'] = predicted_indoor
        last_outlet_temp = outlet_temp  # For next iteration's change calculation
        
    return trajectory


def evaluate_trajectory_stability_enhanced(
    trajectory: List[float], 
    target_temp: float, 
    current_temp: float = None,
    thermal_momentum: dict = None
) -> float:
    """
    Enhanced trajectory stability evaluation using thermal momentum.
    
    Combines original stability scoring with thermal momentum analysis
    for better trajectory quality assessment.
    
    Args:
        trajectory: List of predicted indoor temperatures
        target_temp: Desired target temperature
        current_temp: Current indoor temperature (for direction analysis)
        thermal_momentum: Dict of thermal momentum features
        
    Returns:
        Stability score (lower is better)
    """
    if not trajectory:
        return float('inf')
    
    # Original stability scoring
    # 1. Total deviation from target
    deviation_score = sum(abs(temp - target_temp) for temp in trajectory)
    
    # 2. Oscillation penalty (detect direction changes)
    oscillation_penalty = 0.0
    for i in range(1, len(trajectory) - 1):
        prev_trend = trajectory[i] - trajectory[i-1]
        next_trend = trajectory[i+1] - trajectory[i]
        if prev_trend * next_trend < 0:  # Direction reversal
            oscillation_penalty += getattr(config, 'OSCILLATION_PENALTY_WEIGHT', 2.0)
    
    # 3. Final destination check
    final_error = abs(trajectory[-1] - target_temp)
    final_penalty = final_error * getattr(config, 'FINAL_DESTINATION_WEIGHT', 3.0)
    
    # ENHANCED: 4. Thermal momentum consistency check
    momentum_penalty = 0.0
    if thermal_momentum and current_temp is not None:
        current_gradient = thermal_momentum.get('indoor_temp_gradient', 0.0)
        
        # Check if trajectory is consistent with current thermal momentum
        if len(trajectory) >= 2:
            trajectory_gradient = trajectory[1] - trajectory[0]
            momentum_consistency = abs(trajectory_gradient - current_gradient)
            
            # Penalty for inconsistent momentum (sudden changes are unrealistic)
            if momentum_consistency > 0.5:  # More than 0.5°C/h difference
                momentum_penalty = momentum_consistency * 1.5
    
    # 5. Direction correctness penalty
    direction_penalty = 0.0
    if current_temp is not None and len(trajectory) > 0:
        needed_direction = target_temp - current_temp  # Negative = need cooling
        actual_direction = trajectory[-1] - current_temp  # Negative = cooling
        
        # Heavy penalty if trajectory moves away from target
        if needed_direction * actual_direction < 0:
            direction_penalty = abs(needed_direction) * 5.0
        elif abs(actual_direction) < abs(needed_direction) * 0.1:
            direction_penalty = abs(needed_direction) * 1.0
    
    # Combined enhanced score (lower is better)
    total_score = (deviation_score + oscillation_penalty + final_penalty + 
                   momentum_penalty + direction_penalty)
    
    return total_score


def apply_thermal_momentum_correction(
    predicted_delta: float,
    thermal_momentum: dict,
    target_temp: float,
    current_temp: float
) -> float:
    """
    Apply thermal momentum-based correction to temperature predictions.
    
    Uses thermal momentum features to make predictions more realistic
    and prevent sudden temperature changes.
    
    Args:
        predicted_delta: Raw temperature delta prediction
        thermal_momentum: Dict of thermal momentum features
        target_temp: Target temperature
        current_temp: Current temperature
        
    Returns:
        Corrected temperature delta
    """
    if not thermal_momentum:
        return predicted_delta
    
    # Get momentum indicators
    current_gradient = thermal_momentum.get('indoor_temp_gradient', 0.0)
    temp_diff_outdoor = thermal_momentum.get('temp_diff_indoor_outdoor', 0.0)
    
    # Apply momentum-based smoothing
    corrected_delta = predicted_delta
    
    # If current momentum is strong, limit sudden direction changes
    if abs(current_gradient) > 0.3:  # Strong momentum (>0.3°C/h)
        gradient_direction = 1 if current_gradient > 0 else -1
        prediction_direction = 1 if predicted_delta > 0 else -1
        
        # If prediction opposes strong momentum, reduce the correction
        if gradient_direction != prediction_direction:
            corrected_delta *= 0.7  # Reduce opposition by 30%
            
    # Apply thermal mass effects - larger buildings change more slowly
    thermal_mass_factor = min(1.0, abs(temp_diff_outdoor) / 10.0)  # 0.0-1.0
    if thermal_mass_factor > 0.5:  # High thermal mass scenario
        corrected_delta *= (0.8 + thermal_mass_factor * 0.2)  # 80%-100% scaling
    
    logging.debug(
        f"Momentum correction: {predicted_delta:.3f} → {corrected_delta:.3f} "
        f"(gradient={current_gradient:.3f}, mass_factor={thermal_mass_factor:.2f})"
    )
    
    return corrected_delta
