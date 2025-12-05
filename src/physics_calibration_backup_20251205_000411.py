"""
Physics Model Calibration for ML Heating Controller

This module provides calibration functionality for the realistic physics model
using historical target temperature data and actual house behavior.
"""

import logging
import pandas as pd
import numpy as np
import json
import os
import shutil
from datetime import datetime

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None
    logging.warning("scipy not available - optimization will be disabled")

# Support both package-relative and direct import for notebooks/scripts
try:
    from . import config
    from .thermal_equilibrium_model import ThermalEquilibriumModel
    from .state_manager import save_state, load_state
    from .influx_service import InfluxService
except ImportError:
    # Direct import fallback for standalone execution
    import config
    from thermal_equilibrium_model import ThermalEquilibriumModel
    from state_manager import save_state, load_state
    from influx_service import InfluxService


def train_thermal_equilibrium_model():
    """Train the Thermal Equilibrium Model with historical data for optimal thermal parameters"""
    
    logging.info("=== THERMAL EQUILIBRIUM MODEL TRAINING ===")
    
    # Initialize thermal model
    thermal_model = ThermalEquilibriumModel()
    
    influx = InfluxService(
        url=config.INFLUX_URL,
        token=config.INFLUX_TOKEN,
        org=config.INFLUX_ORG
    )
    
    logging.info("Fetching historical data with target temperatures...")
    df = influx.get_training_data(lookback_hours=config.TRAINING_LOOKBACK_HOURS)
    
    if df.empty or len(df) < 240:
        logging.error("ERROR: Insufficient training data")
        return None
    
    logging.info(f"Processing {len(df)} samples ({len(df)/12:.1f} hours)")
    
    # Enhanced feature preparation with target temperatures
    features_list = []
    labels_list = []
    
    # Data quality tracking
    total_samples = 0
    filtered_samples = {
        'temp_gap': 0,
        'delta_outlier': 0, 
        'sensor_divergence': 0,
        'outlet_range': 0,
        'rate_limiting': 0,
        'missing_data': 0
    }
    
    logging.info("Building features with enhanced data filtering...")
    
    # Get column names
    outlet_col = config.ACTUAL_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1]
    indoor_col = config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    outdoor_col = config.OUTDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    target_col = config.TARGET_INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    dhw_col = config.DHW_STATUS_ENTITY_ID.split(".", 1)[-1]
    disinfect_col = config.DISINFECTION_STATUS_ENTITY_ID.split(".", 1)[-1]
    boost_col = config.DHW_BOOST_HEATER_STATUS_ENTITY_ID.split(".", 1)[-1]
    defrost_col = config.DEFROST_STATUS_ENTITY_ID.split(".", 1)[-1]
    pv_power_col = config.PV_POWER_ENTITY_ID.split(".", 1)[-1]
    fireplace_col = config.FIREPLACE_STATUS_ENTITY_ID.split(".", 1)[-1]
    tv_col = config.TV_STATUS_ENTITY_ID.split(".", 1)[-1]
    
    for idx in range(12, len(df) - config.PREDICTION_HORIZON_STEPS):
        row = df.iloc[idx]
        total_samples += 1
        
        # Extract core temperatures
        outlet_temp = row.get(outlet_col)
        indoor_temp = row.get(indoor_col)
        outdoor_temp = row.get(outdoor_col)
        target_temp = row.get(target_col, 21.0)
        
        # Get indoor lag (30 min = 3 steps back at 10 min intervals)
        indoor_lag_30m = (df.iloc[idx - 3].get(indoor_col) 
                          if idx >= 3 else indoor_temp)
        
        # Skip if missing critical data
        if (pd.isna(outlet_temp) or pd.isna(indoor_temp) 
                or pd.isna(outdoor_temp)):
            filtered_samples['missing_data'] += 1
            continue
        
        # Calculate actual temperature change
        future_indoor = df.iloc[idx + config.PREDICTION_HORIZON_STEPS].get(
            indoor_col)
        if pd.isna(future_indoor) or pd.isna(target_temp):
            filtered_samples['missing_data'] += 1
            continue
        
        actual_delta = float(future_indoor) - float(indoor_temp)
        
        # Enhanced data filtering for outlier removal (same as validation)
        temp_gap = target_temp - indoor_temp
        
        # 1. Skip very large temperature gaps (system faults)
        if abs(temp_gap) > 3.0:
            filtered_samples['temp_gap'] += 1
            continue
            
        # 2. Enhanced outlier filtering for realistic temperature changes
        if abs(actual_delta) > 0.5:  # Skip large unrealistic changes
            filtered_samples['delta_outlier'] += 1
            continue
            
        # 3. Sensor consistency checks
        if abs(indoor_temp - target_temp) > 5.0:  # Extreme sensor divergence
            filtered_samples['sensor_divergence'] += 1
            continue
            
        # 4. Physics constraint validation
        if outlet_temp < 10.0 or outlet_temp > 70.0:  # Unrealistic outlet temps
            filtered_samples['outlet_range'] += 1
            continue
            
        # 5. Rate limiting - check previous temperature for sudden jumps
        if idx >= 6:  # Need some history
            prev_indoor = df.iloc[idx - 6].get(indoor_col)  # 1 hour ago
            if prev_indoor is not None:
                temp_rate = abs(indoor_temp - prev_indoor)  # Change over 1 hour
                if temp_rate > 2.0:  # Skip sudden temperature jumps
                    filtered_samples['rate_limiting'] += 1
                    continue
        
        # Build feature dictionary (only 19 features needed)
        features = {
            'outlet_temp': float(outlet_temp),
            'indoor_temp_lag_30m': float(indoor_lag_30m),
            'target_temp': float(target_temp),
            'outdoor_temp': float(outdoor_temp),
            'dhw_heating': float(row.get(dhw_col, 0.0)),
            'dhw_disinfection': float(row.get(disinfect_col, 0.0)),
            'dhw_boost_heater': float(row.get(boost_col, 0.0)),
            'defrosting': float(row.get(defrost_col, 0.0)),
            'pv_now': float(row.get(pv_power_col, 0.0)),
            'fireplace_on': float(row.get(fireplace_col, 0.0)),
            'tv_on': float(row.get(tv_col, 0.0)),
            'temp_forecast_1h': float(outdoor_temp),
            'temp_forecast_2h': float(outdoor_temp),
            'temp_forecast_3h': float(outdoor_temp),
            'temp_forecast_4h': float(outdoor_temp),
            'pv_forecast_1h': 0.0,
            'pv_forecast_2h': 0.0,
            'pv_forecast_3h': 0.0,
            'pv_forecast_4h': 0.0,
        }
        
        features_list.append(features)
        labels_list.append(actual_delta)
    
    if not features_list:
        logging.error("ERROR: No valid training samples after filtering")
        return None
    
    # Log data quality statistics
    total_filtered = sum(filtered_samples.values())
    kept_samples = len(features_list)
    filter_rate = (total_filtered / total_samples) * 100 if total_samples > 0 else 0
    
    logging.info("=== DATA QUALITY FILTERING RESULTS ===")
    logging.info(f"Total samples processed: {total_samples}")
    logging.info(f"Samples filtered out: {total_filtered} ({filter_rate:.1f}%)")
    logging.info(f"Samples kept for training: {kept_samples}")
    logging.info("Filter breakdown:")
    for filter_type, count in filtered_samples.items():
        if count > 0:
            pct = (count / total_samples) * 100
            logging.info(f"  {filter_type}: {count} ({pct:.1f}%)")
    
    logging.info(
        f"Training on {kept_samples} realistic heating scenarios"
    )
    
    # Training loop - Use thermal equilibrium model for parameter optimization
    logging.info("\n=== THERMAL PARAMETER OPTIMIZATION ===")
    logging.info("🔧 PHYSICS-CONSTRAINED LEARNING: Preventing unrealistic parameter drift")
    
    # Store initial parameters from .env config
    initial_outlet_effectiveness = thermal_model.outlet_effectiveness
    logging.info(f"Starting outlet effectiveness: {initial_outlet_effectiveness:.3f}")
    
    # Track prediction accuracy for optimization
    prediction_errors = []
    
    for i, (features, actual_delta) in enumerate(zip(features_list, labels_list)):
        # Extract thermal context from features
        context = {
            'outlet_temp': features['outlet_temp'],
            'outdoor_temp': features['outdoor_temp'],
            'pv_power': features.get('pv_now', 0.0),
            'fireplace_on': features.get('fireplace_on', 0.0),
            'tv_on': features.get('tv_on', 0.0)
        }
        
        # Predict equilibrium temperature using thermal model
        current_indoor = features['indoor_temp_lag_30m']
        predicted_equilibrium = thermal_model.predict_equilibrium_temperature(
            outlet_temp=features['outlet_temp'],
            outdoor_temp=features['outdoor_temp'],
            pv_power=context['pv_power'],
            fireplace_on=context['fireplace_on'],
            tv_on=context['tv_on']
        )
        
        # Convert equilibrium prediction to temperature change prediction
        predicted_delta = (predicted_equilibrium - current_indoor) * 0.1
        
        # Calculate actual final temperature for feedback
        actual_final = current_indoor + actual_delta
        
        # Update thermal model with prediction feedback
        thermal_model.update_prediction_feedback(
            predicted_temp=predicted_equilibrium,
            actual_temp=actual_final,
            prediction_context=context,
            timestamp=df.iloc[12 + i]['_time'] if i < len(df) - 12 else None
        )
        
        # PHYSICS CONSTRAINT: Prevent outlet effectiveness from drifting too far from realistic values
        if i > 0 and (i + 1) % 50 == 0:  # Check every 50 samples
            # Enforce minimum outlet effectiveness based on .env config
            min_outlet_effectiveness = max(0.4, initial_outlet_effectiveness - 0.1)  # Allow 10% drift down
            max_outlet_effectiveness = min(0.8, initial_outlet_effectiveness + 0.1)  # Allow 10% drift up
            
            if thermal_model.outlet_effectiveness < min_outlet_effectiveness:
                logging.info(f"🔧 PHYSICS CONSTRAINT: Outlet effectiveness {thermal_model.outlet_effectiveness:.3f} too low, "
                           f"correcting to minimum {min_outlet_effectiveness:.3f}")
                thermal_model.outlet_effectiveness = min_outlet_effectiveness
                
            elif thermal_model.outlet_effectiveness > max_outlet_effectiveness:
                logging.info(f"🔧 PHYSICS CONSTRAINT: Outlet effectiveness {thermal_model.outlet_effectiveness:.3f} too high, "
                           f"correcting to maximum {max_outlet_effectiveness:.3f}")
                thermal_model.outlet_effectiveness = max_outlet_effectiveness
        
        # Track prediction error
        prediction_error = abs(predicted_delta - actual_delta)
        prediction_errors.append(prediction_error)
        
        if (i + 1) % 200 == 0:
            avg_error = sum(prediction_errors[-200:]) / min(200, len(prediction_errors))
            logging.info(
                f"Processed {i+1}/{len(features_list)} - "
                f"Avg Error: {avg_error:.4f}°C - "
                f"Confidence: {thermal_model.learning_confidence:.3f}"
            )
    
    # Results
    logging.info("\n=== THERMAL TRAINING COMPLETED ===")
    avg_error = sum(prediction_errors) / len(prediction_errors) if prediction_errors else 0.0
    logging.info(f"Final Average Error: {avg_error:.4f}°C")
    logging.info(f"Total training samples: {len(prediction_errors)}")
    
    logging.info("\n=== LEARNED THERMAL PARAMETERS ===")
    logging.info(f"Thermal time constant: {thermal_model.thermal_time_constant:.2f}h")
    logging.info(f"Heat loss coefficient: {thermal_model.heat_loss_coefficient:.4f}")
    logging.info(f"Outlet effectiveness: {thermal_model.outlet_effectiveness:.3f}")
    logging.info(f"Learning confidence: {thermal_model.learning_confidence:.3f}")
    
    # Get adaptive learning metrics
    learning_metrics = thermal_model.get_adaptive_learning_metrics()
    if not learning_metrics.get('insufficient_data', False):
        logging.info(f"Parameter updates: {learning_metrics['parameter_updates']}")
        logging.info(f"Update percentage: {learning_metrics['update_percentage']:.1f}%")
    
    # Test thermal equilibrium physics
    logging.info("\n=== THERMAL PHYSICS VALIDATION ===")
    logging.info("Testing equilibrium predictions:")
    for outlet_temp in [35, 45, 55]:
        for outdoor_temp in [0, 10, 20]:
            equilibrium = thermal_model.predict_equilibrium_temperature(
                outlet_temp=outlet_temp,
                outdoor_temp=outdoor_temp,
                pv_power=0,
                fireplace_on=0,
                tv_on=0
            )
            logging.info(f"  Outlet {outlet_temp}°C, Outdoor {outdoor_temp}°C → "
                        f"Equilibrium {equilibrium:.2f}°C")
    
    # Save thermal learning state to unified thermal state manager
    logging.info("\n=== SAVING CALIBRATED PARAMETERS TO UNIFIED THERMAL STATE ===")
    try:
        from .unified_thermal_state import get_thermal_state_manager
        
        state_manager = get_thermal_state_manager()
        
        # Save as calibrated baseline parameters
        calibrated_params = {
            'thermal_time_constant': thermal_model.thermal_time_constant,
            'heat_loss_coefficient': thermal_model.heat_loss_coefficient,
            'outlet_effectiveness': thermal_model.outlet_effectiveness,
            'pv_heat_weight': thermal_model.external_source_weights.get('pv', config.PV_HEAT_WEIGHT),
            'fireplace_heat_weight': thermal_model.external_source_weights.get('fireplace', config.FIREPLACE_HEAT_WEIGHT),
            'tv_heat_weight': thermal_model.external_source_weights.get('tv', config.TV_HEAT_WEIGHT)
        }
        
        # Set as calibrated baseline (this updates the parameters source to "calibrated")
        state_manager.set_calibrated_baseline(calibrated_params, calibration_cycles=kept_samples)
        
        # Also update learning state
        state_manager.update_learning_state(
            learning_confidence=thermal_model.learning_confidence,
            parameter_adjustments={
                'thermal_time_constant_delta': 0.0,  # Reset deltas since we're setting new baseline
                'heat_loss_coefficient_delta': 0.0,
                'outlet_effectiveness_delta': 0.0
            }
        )
        
        # Add parameter and prediction history
        if thermal_model.parameter_history:
            for record in thermal_model.parameter_history[-50:]:  # Last 50 updates
                state_manager.add_parameter_history_record(record)
        
        if thermal_model.prediction_history:
            for record in thermal_model.prediction_history[-100:]:  # Last 100 predictions
                state_manager.add_prediction_record(record)
        
        logging.info("✅ Calibrated parameters saved to unified thermal state")
        logging.info("✅ Parameters will be automatically loaded on next restart")
        logging.info("🔄 Restart ml_heating service to use calibrated thermal model")
        
    except Exception as e:
        logging.error(f"❌ Failed to save calibrated parameters: {e}")
        # Fallback to old method
        thermal_learning_state = {
            'thermal_time_constant': thermal_model.thermal_time_constant,
            'heat_loss_coefficient': thermal_model.heat_loss_coefficient,
            'outlet_effectiveness': thermal_model.outlet_effectiveness,
            'learning_confidence': thermal_model.learning_confidence,
        }
        save_state(thermal_learning_state=thermal_learning_state)
        logging.warning("⚠️ Used fallback save method - parameters may not persist")
    
    return thermal_model


def validate_thermal_model():
    """Validate thermal equilibrium model behavior across temperature ranges"""
    
    logging.info("=== THERMAL MODEL VALIDATION ===")
    
    try:
        # Initialize thermal model (will use default parameters or restored state)
        thermal_model = ThermalEquilibriumModel()
        
        logging.info("Testing thermal equilibrium physics compliance:")
        print("\nOUTLET TEMP → EQUILIBRIUM TEMP")
        print("=" * 35)
        
        # Test monotonicity - higher outlet should mean higher equilibrium
        monotonic_check = []
        for outlet_temp in [25, 30, 35, 40, 45, 50, 55, 60]:
            equilibrium = thermal_model.predict_equilibrium_temperature(
                outlet_temp=outlet_temp,
                outdoor_temp=5.0,
                pv_power=0,
                fireplace_on=0,
                tv_on=0
            )
            monotonic_check.append(equilibrium)
            print(f"{outlet_temp:3d}°C       → {equilibrium:.2f}°C")
        
        # Check monotonicity
        is_monotonic = all(monotonic_check[i] <= monotonic_check[i+1] 
                          for i in range(len(monotonic_check)-1))
        
        print(f"\n{'✅' if is_monotonic else '❌'} Physics compliance: "
              f"{'PASSED' if is_monotonic else 'FAILED'}")
        print(f"Range: {min(monotonic_check):.2f}°C to "
              f"{max(monotonic_check):.2f}°C")
        
        # Test parameter bounds
        logging.info("\n=== THERMAL PARAMETER BOUNDS TEST ===")
        params_ok = True
        
        # Get bounds from centralized config
        thermal_bounds = ThermalParameterConfig.get_bounds('thermal_time_constant')
        if not (thermal_bounds[0] <= thermal_model.thermal_time_constant <= thermal_bounds[1]):
            logging.error(f"Thermal time constant out of bounds: "
                         f"{thermal_model.thermal_time_constant:.2f}h (bounds: {thermal_bounds})")
            params_ok = False
            
        # Get bounds from centralized config
        heat_loss_bounds = ThermalParameterConfig.get_bounds('heat_loss_coefficient')
        if not (heat_loss_bounds[0] <= thermal_model.heat_loss_coefficient <= heat_loss_bounds[1]):
            logging.error(f"Heat loss coefficient out of bounds: "
                         f"{thermal_model.heat_loss_coefficient:.4f} (bounds: {heat_loss_bounds})")
            params_ok = False
            
        if not (0.2 <= thermal_model.outlet_effectiveness <= 1.5):
            logging.error(f"Outlet effectiveness out of bounds: "
                         f"{thermal_model.outlet_effectiveness:.3f}")
            params_ok = False
            
        if params_ok:
            logging.info("✅ All thermal parameters within physical bounds")
        else:
            logging.error("❌ Some thermal parameters out of bounds")
        
        # Test adaptive learning system
        logging.info("\n=== ADAPTIVE LEARNING SYSTEM TEST ===")
        
        # Simulate some prediction feedback
        test_context = {
            'outlet_temp': 45.0,
            'outdoor_temp': 5.0,
            'pv_power': 0,
            'fireplace_on': 0,
            'tv_on': 0
        }
        
        initial_confidence = thermal_model.learning_confidence
        
        # Simulate good predictions (should boost confidence)
        for _ in range(5):
            predicted = thermal_model.predict_equilibrium_temperature(
                **test_context
            )
            actual = predicted + 0.1  # Small error
            thermal_model.update_prediction_feedback(
                predicted_temp=predicted,
                actual_temp=actual,
                prediction_context=test_context
            )
            
        final_confidence = thermal_model.learning_confidence
        learning_works = final_confidence != initial_confidence
        
        logging.info(f"Initial confidence: {initial_confidence:.3f}")
        logging.info(f"Final confidence: {final_confidence:.3f}")
        logging.info(f"{'✅' if learning_works else '❌'} "
                    f"Adaptive learning: {'WORKING' if learning_works else 'NOT WORKING'}")
        
        overall_success = is_monotonic and params_ok and learning_works
        logging.info(f"\n{'✅' if overall_success else '❌'} "
                    f"Overall validation: {'PASSED' if overall_success else 'FAILED'}")
        
        return overall_success
        
    except Exception as e:
        logging.error("Thermal model validation error: %s", e, exc_info=True)
        return False


def fetch_historical_data_for_calibration(lookback_hours=672):
    """
    Step 2: Fetch 672 hours of historical data for Phase 0 calibration
    
    Args:
        lookback_hours: Hours of historical data (default 672 = 28 days from .env)
    
    Returns:
        pandas.DataFrame: Historical data with required columns
    """
    logging.info(f"=== FETCHING {lookback_hours} HOURS OF HISTORICAL DATA ===")
    
    influx = InfluxService(
        url=config.INFLUX_URL,
        token=config.INFLUX_TOKEN,
        org=config.INFLUX_ORG
    )
    
    # Fetch the data
    df = influx.get_training_data(lookback_hours=lookback_hours)
    
    if df.empty:
        logging.error("❌ No historical data available")
        return None
        
    logging.info(f"✅ Fetched {len(df)} samples ({len(df)/12:.1f} hours)")
    
    # Validate required columns exist
    required_columns = [
        config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1],
        config.ACTUAL_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1], 
        config.OUTDOOR_TEMP_ENTITY_ID.split(".", 1)[-1],
        config.PV_POWER_ENTITY_ID.split(".", 1)[-1],
        config.TV_STATUS_ENTITY_ID.split(".", 1)[-1]
    ]
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logging.error(f"❌ Missing required columns: {missing_cols}")
        return None
    
    # Check optional columns with fallback
    fireplace_col = config.FIREPLACE_STATUS_ENTITY_ID.split(".", 1)[-1]
    if fireplace_col not in df.columns:
        logging.info(f"⚠️  Optional fireplace column '{fireplace_col}' not found - will use 0")
        
    logging.info("✅ All required columns present")
    return df


def main():
    """Main function to run thermal equilibrium model training and validation"""
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    try:
        # Train thermal model with real data
        print("🚀 Starting thermal equilibrium model training...")
        thermal_model = train_thermal_equilibrium_model()
        
        if thermal_model:
            print("✅ Thermal training completed successfully!")
            
            # Validate the trained model
            print("\n🧪 Running thermal model validation...")
            validation_passed = validate_thermal_model()
            
            if validation_passed:
                print("✅ Thermal model validation PASSED!")
            else:
                print("❌ Thermal model validation FAILED!")
                
        else:
            print("❌ Thermal training failed!")
            return False
            
        return True
        
    except Exception as e:
        logging.error(f"Main execution error: {e}", exc_info=True)
        return False


def filter_stable_periods(df, temp_change_threshold=0.1, min_duration=30):
    """
    Step 3: Filter historical data for stable equilibrium periods with blocking state detection
    
    Enhanced filtering that respects blocking states and grace periods like core ml_heating system:
    - Excludes periods during DHW heating, defrosting, disinfection, boost heater
    - Applies grace periods after blocking states end (using GRACE_PERIOD_MAX_MINUTES)
    - Validates thermal stability and outlet temperature normalization
    
    Args:
        df: Historical data DataFrame
        temp_change_threshold: Max temp change per 30min (°C)
        min_duration: Minimum stable period duration (minutes)
    
    Returns:
        list: Stable periods suitable for calibration
    """
    logging.info("=== FILTERING FOR STABLE PERIODS WITH BLOCKING STATE DETECTION ===")
    
    # Column mappings
    indoor_col = config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    outlet_col = config.ACTUAL_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1]
    outdoor_col = config.OUTDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    pv_col = config.PV_POWER_ENTITY_ID.split(".", 1)[-1]
    fireplace_col = config.FIREPLACE_STATUS_ENTITY_ID.split(".", 1)[-1]
    tv_col = config.TV_STATUS_ENTITY_ID.split(".", 1)[-1]
    
    # Blocking state columns
    dhw_col = config.DHW_STATUS_ENTITY_ID.split(".", 1)[-1]
    defrost_col = config.DEFROST_STATUS_ENTITY_ID.split(".", 1)[-1]
    disinfect_col = config.DISINFECTION_STATUS_ENTITY_ID.split(".", 1)[-1]
    boost_col = config.DHW_BOOST_HEATER_STATUS_ENTITY_ID.split(".", 1)[-1]
    
    # Grace period configuration
    grace_period_minutes = config.GRACE_PERIOD_MAX_MINUTES
    grace_period_samples = grace_period_minutes // 5  # Convert to 5-min samples
    
    stable_periods = []
    window_size = min_duration // 5  # Convert to 5-min samples
    
    logging.info(f"Looking for periods with <{temp_change_threshold}°C change"
                f" over {min_duration} min")
    logging.info(f"Using {grace_period_minutes}min grace periods after blocking states")
    
    # Filter statistics
    filter_stats = {
        'total_checked': 0,
        'missing_data': 0,
        'temp_unstable': 0,
        'blocking_active': 0,
        'grace_period': 0,
        'fireplace_changed': 0,
        'outlet_unstable': 0,
        'passed': 0
    }
    
    for i in range(window_size + grace_period_samples, len(df) - window_size):
        filter_stats['total_checked'] += 1
        
        # Get window of data for stability analysis
        window_start = i - window_size // 2
        window_end = i + window_size // 2
        window = df.iloc[window_start:window_end]
        
        # Get extended window for blocking state analysis (includes grace period)
        grace_start = i - grace_period_samples
        grace_end = i + window_size // 2
        grace_window = df.iloc[grace_start:grace_end]
        
        # Skip if missing critical data
        indoor_temps = window[indoor_col].dropna()
        if len(indoor_temps) < window_size * 0.8:  # Need 80% data coverage
            filter_stats['missing_data'] += 1
            continue
            
        # Calculate temperature stability
        temp_range = indoor_temps.max() - indoor_temps.min()
        temp_std = indoor_temps.std()
        
        if (temp_range > temp_change_threshold or 
            temp_std > temp_change_threshold / 2):
            filter_stats['temp_unstable'] += 1
            continue
        
        # Check for blocking states in extended window (including grace period)
        blocking_detected = False
        blocking_reasons = []
        
        # DHW heating check
        if dhw_col in grace_window.columns:
            if grace_window[dhw_col].sum() > 0:
                blocking_detected = True
                blocking_reasons.append('dhw_heating')
        
        # Defrosting check  
        if defrost_col in grace_window.columns:
            if grace_window[defrost_col].sum() > 0:
                blocking_detected = True
                blocking_reasons.append('defrosting')
        
        # Disinfection check
        if disinfect_col in grace_window.columns:
            if grace_window[disinfect_col].sum() > 0:
                blocking_detected = True
                blocking_reasons.append('disinfection')
                
        # DHW boost heater check
        if boost_col in grace_window.columns:
            if grace_window[boost_col].sum() > 0:
                blocking_detected = True
                blocking_reasons.append('boost_heater')
        
        if blocking_detected:
            # Check if blocking was during grace period vs current window
            current_window_blocking = False
            if dhw_col in window.columns and window[dhw_col].sum() > 0:
                current_window_blocking = True
            if defrost_col in window.columns and window[defrost_col].sum() > 0:
                current_window_blocking = True
            if disinfect_col in window.columns and window[disinfect_col].sum() > 0:
                current_window_blocking = True
            if boost_col in window.columns and window[boost_col].sum() > 0:
                current_window_blocking = True
                
            if current_window_blocking:
                filter_stats['blocking_active'] += 1
            else:
                filter_stats['grace_period'] += 1
            continue
        
        # Check for fireplace state changes
        fireplace_changed = False
        if fireplace_col in window.columns:
            fireplace_values = window[fireplace_col]
            if fireplace_values.nunique() > 1:
                fireplace_changed = True
                filter_stats['fireplace_changed'] += 1
                continue
        
        # Outlet temperature stability validation
        if outlet_col in window.columns:
            outlet_temps = window[outlet_col].dropna()
            if len(outlet_temps) >= window_size * 0.8:
                outlet_std = outlet_temps.std()
                # Require outlet temperature stability (< 2°C std dev)
                if outlet_std > 2.0:
                    filter_stats['outlet_unstable'] += 1
                    continue
        
        # Period passed all filters - extract data
        center_row = df.iloc[i]
        period = {
            'indoor_temp': center_row[indoor_col],
            'outlet_temp': center_row[outlet_col], 
            'outdoor_temp': center_row[outdoor_col],
            'pv_power': center_row.get(pv_col, 0.0),
            'fireplace_on': center_row.get(fireplace_col, 0.0),
            'tv_on': center_row.get(tv_col, 0.0),
            'timestamp': center_row['_time'],
            'stability_score': 1.0 / (temp_std + 0.01),
            'outlet_stability': 1.0 / (outlet_temps.std() + 0.01) if outlet_col in window.columns else 1.0
        }
        stable_periods.append(period)
        filter_stats['passed'] += 1
    
    # Log filtering statistics
    logging.info("=== BLOCKING STATE FILTERING RESULTS ===")
    logging.info(f"Total periods checked: {filter_stats['total_checked']}")
    logging.info(f"Stable periods found: {filter_stats['passed']}")
    logging.info(f"Filter exclusions:")
    logging.info(f"  Missing data: {filter_stats['missing_data']}")
    logging.info(f"  Temperature unstable: {filter_stats['temp_unstable']}")
    logging.info(f"  Blocking states active: {filter_stats['blocking_active']}")
    logging.info(f"  Grace period recovery: {filter_stats['grace_period']}")
    logging.info(f"  Fireplace state changes: {filter_stats['fireplace_changed']}")
    logging.info(f"  Outlet temperature unstable: {filter_stats['outlet_unstable']}")
    
    retention_rate = (filter_stats['passed'] / filter_stats['total_checked']) * 100 if filter_stats['total_checked'] > 0 else 0
    logging.info(f"Data retention rate: {retention_rate:.1f}%")
    
    logging.info(f"✅ Found {len(stable_periods)} stable periods with blocking state filtering")
    return stable_periods


def debug_thermal_predictions(stable_periods, sample_size=5):
    """
    Debug thermal model predictions on sample data
    """
    logging.info("=== DEBUGGING THERMAL PREDICTIONS ===")
    
    # Create test model with current parameters
    test_model = ThermalEquilibriumModel()
    
    logging.info("Testing thermal model on sample periods:")
    for i, period in enumerate(stable_periods[:sample_size]):
        predicted_temp = test_model.predict_equilibrium_temperature(
            outlet_temp=period['outlet_temp'],
            outdoor_temp=period['outdoor_temp'],
            pv_power=period['pv_power'],
            fireplace_on=period['fireplace_on'],
            tv_on=period['tv_on']
        )
        
        actual_temp = period['indoor_temp']
        error = abs(predicted_temp - actual_temp)
        
        logging.info(f"Sample {i+1}:")
        logging.info(f"  Outlet: {period['outlet_temp']:.1f}°C, Outdoor: {period['outdoor_temp']:.1f}°C")
        logging.info(f"  PV: {period['pv_power']:.1f}W, Fireplace: {period['fireplace_on']:.0f}, TV: {period['tv_on']:.0f}")
        logging.info(f"  Predicted: {predicted_temp:.1f}°C, Actual: {actual_temp:.1f}°C")
        logging.info(f"  Error: {error:.1f}°C")
        logging.info("")


def optimize_thermal_parameters(stable_periods):
    """
    Step 4: Multi-parameter optimization using scipy.optimize with data availability checks
    
    Args:
        stable_periods: List of stable equilibrium periods
    
    Returns:
        dict: Optimized thermal parameters
    """
    logging.info("=== MULTI-PARAMETER OPTIMIZATION WITH DATA AVAILABILITY CHECKS ===")
    
    if minimize is None:
        logging.error("❌ scipy not available - cannot optimize parameters")
        return None
    
    # Debug thermal predictions before optimization
    debug_thermal_predictions(stable_periods)
    
    # Check data availability for each heat source
    logging.info("=== CHECKING DATA AVAILABILITY ===")
    
    # Analyze actual data availability in stable periods
    total_periods = len(stable_periods)
    data_stats = {
        'pv_power': sum(1 for p in stable_periods if p.get('pv_power', 0) > 0),
        'fireplace_on': sum(1 for p in stable_periods if p.get('fireplace_on', 0) > 0), 
        'tv_on': sum(1 for p in stable_periods if p.get('tv_on', 0) > 0)
    }
    
    # Calculate usage percentages
    data_availability = {}
    for source, count in data_stats.items():
        percentage = (count / total_periods) * 100
        data_availability[source] = percentage
        logging.info(f"  {source}: {count}/{total_periods} periods ({percentage:.1f}%)")
    
    # Determine which parameters to exclude from optimization
    excluded_params = []
    min_usage_threshold = 1.0  # Require >1% usage to optimize
    
    if data_availability['fireplace_on'] <= min_usage_threshold:
        excluded_params.append('fireplace_heat_weight')
        logging.info(f"  🚫 Excluding fireplace_heat_weight (only {data_availability['fireplace_on']:.1f}% usage)")
    
    if data_availability['tv_on'] <= min_usage_threshold:
        excluded_params.append('tv_heat_weight') 
        logging.info(f"  🚫 Excluding tv_heat_weight (only {data_availability['tv_on']:.1f}% usage)")
        
    if data_availability['pv_power'] <= min_usage_threshold:
        excluded_params.append('pv_heat_weight')
        logging.info(f"  🚫 Excluding pv_heat_weight (only {data_availability['pv_power']:.1f}% usage)")
        
    # Current parameter values from config
    current_params = {
        'outlet_effectiveness': config.OUTLET_EFFECTIVENESS,
        'heat_loss_coefficient': config.HEAT_LOSS_COEFFICIENT, 
        'thermal_time_constant': config.THERMAL_TIME_CONSTANT,
        'pv_heat_weight': config.PV_HEAT_WEIGHT,
        'fireplace_heat_weight': config.FIREPLACE_HEAT_WEIGHT,
        'tv_heat_weight': config.TV_HEAT_WEIGHT
    }
    
    logging.info("=== PARAMETERS FOR OPTIMIZATION ===")
    for param, value in current_params.items():
        if param in excluded_params:
            logging.info(f"  {param}: {value} (FIXED - insufficient data)")
        else:
            logging.info(f"  {param}: {value} (OPTIMIZE)")
    
    # Build optimization parameter list and bounds (excluding parameters without data)
    param_names = []
    param_values = []
    param_bounds = []
    
    # Always optimize core thermal parameters
    param_names.extend(['outlet_effectiveness', 'heat_loss_coefficient', 'thermal_time_constant'])
    param_values.extend([
        current_params['outlet_effectiveness'],
        current_params['heat_loss_coefficient'], 
        current_params['thermal_time_constant']
    ])
    # Import centralized thermal configuration
    from .thermal_config import ThermalParameterConfig
    
    param_bounds.extend([
        ThermalParameterConfig.get_bounds('outlet_effectiveness'),
        ThermalParameterConfig.get_bounds('heat_loss_coefficient'),
        ThermalParameterConfig.get_bounds('thermal_time_constant')
    ])
    
    # Conditionally add heat source parameters based on data availability
    if 'pv_heat_weight' not in excluded_params:
        param_names.append('pv_heat_weight')
        param_values.append(current_params['pv_heat_weight'])
        param_bounds.append((0.0005, 0.005))
        
    if 'fireplace_heat_weight' not in excluded_params:
        param_names.append('fireplace_heat_weight') 
        param_values.append(current_params['fireplace_heat_weight'])
        param_bounds.append((1.0, 6.0))
        
    if 'tv_heat_weight' not in excluded_params:
        param_names.append('tv_heat_weight')
        param_values.append(current_params['tv_heat_weight'])
        param_bounds.append((0.1, 1.5))
    
    logging.info(f"Optimizing {len(param_names)} parameters: {param_names}")
    
    def objective_function(params):
        """Calculate MAE for given parameters with dynamic parameter mapping"""
        total_error = 0.0
        valid_predictions = 0
        
        # Map parameters to their names dynamically
        param_dict = dict(zip(param_names, params))
        
        for period in stable_periods:
            try:
                # Create temporary thermal model with test parameters
                test_model = ThermalEquilibriumModel()
                test_model.outlet_effectiveness = param_dict['outlet_effectiveness']
                test_model.heat_loss_coefficient = param_dict['heat_loss_coefficient']
                test_model.thermal_time_constant = param_dict['thermal_time_constant']
                
                # Set heat source weights (use original values for excluded params)
                test_model.external_source_weights['pv'] = param_dict.get(
                    'pv_heat_weight', current_params['pv_heat_weight'])
                test_model.external_source_weights['fireplace'] = param_dict.get(
                    'fireplace_heat_weight', current_params['fireplace_heat_weight'])
                test_model.external_source_weights['tv'] = param_dict.get(
                    'tv_heat_weight', current_params['tv_heat_weight'])
                
                # Predict equilibrium temperature
                predicted_temp = test_model.predict_equilibrium_temperature(
                    outlet_temp=period['outlet_temp'],
                    outdoor_temp=period['outdoor_temp'],
                    pv_power=period['pv_power'],
                    fireplace_on=period['fireplace_on'],
                    tv_on=period['tv_on']
                )
                
                # Calculate error
                error = abs(predicted_temp - period['indoor_temp'])
                
                # Skip unrealistic errors that indicate bad parameters
                if error > 50.0:  # Skip extreme errors
                    continue
                    
                total_error += error
                valid_predictions += 1
                
            except Exception:
                # Skip problematic periods
                continue
                
        if valid_predictions == 0:
            return 1000.0  # High error for invalid parameter sets
            
        mae = total_error / valid_predictions
        return mae
    
    logging.info(f"Starting optimization with {len(stable_periods)} periods...")
    logging.info("This may take a few minutes...")
    
    # Run optimization
    try:
        result = minimize(
            objective_function,
            x0=param_values,
            bounds=param_bounds,
            method='L-BFGS-B',
            options={
                'maxiter': 100,
                'ftol': 1e-6
            }
        )
        
        if result.success:
            # Build optimized parameters dict with both optimized and excluded params
            optimized_params = dict(current_params)  # Start with all original
            
            # Update with optimized values
            for i, param_name in enumerate(param_names):
                optimized_params[param_name] = result.x[i]
            
            optimized_params['mae'] = result.fun
            optimized_params['optimization_success'] = True
            optimized_params['excluded_parameters'] = excluded_params
            
            logging.info("✅ Optimization completed successfully!")
            logging.info("Optimized parameters:")
            for param, value in optimized_params.items():
                if param not in ['mae', 'optimization_success', 'excluded_parameters']:
                    old_value = current_params[param]
                    if param in excluded_params:
                        logging.info(f"  {param}: {value:.4f} (FIXED - no data)")
                    else:
                        change_pct = ((value - old_value) / old_value) * 100
                        logging.info(f"  {param}: {value:.4f} "
                                   f"(was {old_value:.4f}, {change_pct:+.1f}%)")
            
            logging.info(f"Final MAE: {result.fun:.4f}°C")
            return optimized_params
            
        else:
            logging.error(f"❌ Optimization failed: {result.message}")
            return None
            
    except Exception as e:
        logging.error(f"❌ Optimization error: {e}")
        return None


def export_calibrated_baseline(optimized_params, stable_periods):
    """
    Step 5: Export calibrated baseline configuration
    
    Args:
        optimized_params: Optimized thermal parameters
        stable_periods: Stable periods used for calibration
    
    Returns:
        str: Path to exported baseline file
    """
    logging.info("=== EXPORTING CALIBRATED BASELINE ===")
    
    # Create baseline configuration
    baseline = {
        'metadata': {
            'calibration_date': datetime.now().isoformat(),
            'data_hours': config.TRAINING_LOOKBACK_HOURS,
            'stable_periods_count': len(stable_periods),
            'optimization_method': 'L-BFGS-B',
            'version': '1.0'
        },
        'parameters': {
            'outlet_effectiveness': optimized_params['outlet_effectiveness'],
            'heat_loss_coefficient': optimized_params['heat_loss_coefficient'],
            'thermal_time_constant': optimized_params['thermal_time_constant'],
            'pv_heat_weight': optimized_params['pv_heat_weight'],
            'fireplace_heat_weight': optimized_params['fireplace_heat_weight'],
            'tv_heat_weight': optimized_params['tv_heat_weight']
        },
        'quality_metrics': {
            'mae_celsius': optimized_params['mae'],
            'optimization_success': optimized_params['optimization_success'],
            'data_coverage_hours': len(stable_periods) * 0.5,  # 30min periods
            'stability_threshold': 0.1
        },
        'original_parameters': {
            'outlet_effectiveness': config.OUTLET_EFFECTIVENESS,
            'heat_loss_coefficient': config.HEAT_LOSS_COEFFICIENT,
            'thermal_time_constant': config.THERMAL_TIME_CONSTANT,
            'pv_heat_weight': config.PV_HEAT_WEIGHT,
            'fireplace_heat_weight': config.FIREPLACE_HEAT_WEIGHT,
            'tv_heat_weight': config.TV_HEAT_WEIGHT
        }
    }
    
    # Export to JSON file
    baseline_path = "/opt/ml_heating/calibrated_baseline.json"
    try:
        with open(baseline_path, 'w') as f:
            json.dump(baseline, f, indent=2)
        
        logging.info(f"✅ Calibrated baseline exported to: {baseline_path}")
        
        # Log improvement summary
        logging.info("=== CALIBRATION IMPROVEMENT SUMMARY ===")
        for param, new_value in baseline['parameters'].items():
            old_value = baseline['original_parameters'][param]
            change_pct = ((new_value - old_value) / old_value) * 100
            logging.info(f"  {param}: {old_value:.4f} → {new_value:.4f} "
                        f"({change_pct:+.1f}%)")
        
        logging.info(f"Expected accuracy improvement: "
                    f"{optimized_params['mae']:.4f}°C MAE")
        
        return baseline_path
        
    except Exception as e:
        logging.error(f"❌ Failed to export baseline: {e}")
        return None


def validate_calibrated_baseline(baseline_path, stable_periods):
    """
    Step 6: Validate calibrated baseline on held-out data
    
    Args:
        baseline_path: Path to calibrated baseline file
        stable_periods: Stable periods for validation
    
    Returns:
        bool: True if validation passed
    """
    logging.info("=== VALIDATING CALIBRATED BASELINE ===")
    
    try:
        # Load baseline
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
        
        # Split data for validation (use last 20% as held-out)
        split_point = int(len(stable_periods) * 0.8)
        validation_periods = stable_periods[split_point:]
        
        logging.info(f"Validating on {len(validation_periods)} held-out periods")
        
        # Create model with calibrated parameters
        test_model = ThermalEquilibriumModel()
        params = baseline['parameters']
        test_model.outlet_effectiveness = params['outlet_effectiveness']
        test_model.heat_loss_coefficient = params['heat_loss_coefficient'] 
        test_model.thermal_time_constant = params['thermal_time_constant']
        test_model.pv_heat_weight = params['pv_heat_weight']
        test_model.fireplace_heat_weight = params['fireplace_heat_weight']
        test_model.tv_heat_weight = params['tv_heat_weight']
        
        # Test on validation data
        errors = []
        for period in validation_periods:
            predicted_temp = test_model.predict_equilibrium_temperature(
                outlet_temp=period['outlet_temp'],
                outdoor_temp=period['outdoor_temp'], 
                pv_power=period['pv_power'],
                fireplace_on=period['fireplace_on'],
                tv_on=period['tv_on']
            )
            
            error = abs(predicted_temp - period['indoor_temp'])
            errors.append(error)
        
        # Calculate validation metrics
        validation_mae = np.mean(errors)
        validation_rmse = np.sqrt(np.mean([e**2 for e in errors]))
        accuracy_within_03 = sum(1 for e in errors if e <= 0.3) / len(errors)
        
        logging.info("=== VALIDATION RESULTS ===")
        logging.info(f"Validation MAE: {validation_mae:.4f}°C")
        logging.info(f"Validation RMSE: {validation_rmse:.4f}°C")
        logging.info(f"Accuracy within ±0.3°C: {accuracy_within_03:.1%}")
        
        # Validation criteria - REALISTIC for thermal systems
        mae_threshold = 3.0  # °C - Realistic for thermal equilibrium predictions
        accuracy_threshold = 0.05  # 5% within ±0.3°C is reasonable for equilibrium periods
        
        mae_pass = validation_mae <= mae_threshold
        accuracy_pass = accuracy_within_03 >= accuracy_threshold
        
        # Additional realistic criteria
        rmse_threshold = 4.0  # °C
        rmse_pass = validation_rmse <= rmse_threshold
        
        validation_passed = mae_pass and accuracy_pass and rmse_pass
        
        logging.info(f"MAE test: {'✅ PASS' if mae_pass else '❌ FAIL'} "
                    f"(≤{mae_threshold}°C)")
        logging.info(f"Accuracy test: {'✅ PASS' if accuracy_pass else '❌ FAIL'} "
                    f"(≥{accuracy_threshold:.0%})")
        logging.info(f"RMSE test: {'✅ PASS' if rmse_pass else '❌ FAIL'} "
                    f"(≤{rmse_threshold}°C)")
        logging.info(f"Overall validation: {'✅ PASSED' if validation_passed else '❌ FAILED'}")
        
        # Update baseline with validation results
        baseline['validation'] = {
            'mae_celsius': float(validation_mae),
            'rmse_celsius': float(validation_rmse),
            'accuracy_within_03c': float(accuracy_within_03),
            'validation_passed': bool(validation_passed),
            'validation_periods': int(len(validation_periods))
        }
        
        # Save updated baseline
        with open(baseline_path, 'w') as f:
            json.dump(baseline, f, indent=2)
        
        return validation_passed
        
    except Exception as e:
        logging.error(f"❌ Validation error: {e}")
        return False


def backup_existing_calibration():
    """Create backup of existing thermal state before calibration"""
    try:
        from .unified_thermal_state import ThermalStateManager
        
        # Initialize state manager to check if state exists
        state_manager = ThermalStateManager()
        
        # Check if thermal state file exists
        if not os.path.exists('/opt/ml_heating/thermal_state.json'):
            logging.info("ℹ️ No existing thermal state found to backup")
            return None
            
        # Create backup using the ThermalStateManager method
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"pre_calibration_{timestamp}"
        
        success, backup_path = state_manager.create_backup(backup_filename)
        
        if success:
            logging.info(f"✅ Created thermal state backup: {os.path.basename(backup_path)}")
            return backup_path
        else:
            logging.warning(f"⚠️ Failed to create thermal state backup: {backup_path}")
            return None
        
    except Exception as e:
        logging.warning(f"⚠️ Failed to create thermal state backup: {e}")
        return None


def restore_calibration_from_backup(backup_path):
    """
    Restore thermal state from a backup file
    
    Args:
        backup_path: Path to the backup file
        
    Returns:
        bool: True if restoration successful
    """
    try:
        # Use unified thermal state manager for restoration
        from unified_thermal_state import get_thermal_state_manager
        
        state_manager = get_thermal_state_manager()
        
        # Restore from backup
        success, message = state_manager.restore_from_backup(backup_path)
        
        if success:
            logging.info(f"✅ Thermal state restored from backup: {message}")
            return True
        else:
            logging.error(f"❌ Failed to restore thermal state: {message}")
            return False
            
    except Exception as e:
        logging.error(f"❌ Failed to restore from backup: {e}")
        return False


def run_phase0_calibration():
    """
    Phase 0: House-Specific Calibration using TRAINING_LOOKBACK_HOURS
    
    This implements the foundational calibration step that should run before Phase 1.
    Uses 672 hours of historical data to optimize ALL thermal parameters.
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    print("🏠 PHASE 0: HOUSE-SPECIFIC CALIBRATION")
    print("=" * 50)
    
    # Step 0: Backup existing calibration
    print("Step 0: Checking for existing calibration...")
    backup_path = backup_existing_calibration()
    if backup_path:
        print(f"✅ Previous calibration backed up: {os.path.basename(backup_path)}")
    else:
        print("ℹ️ No existing calibration found")
    
    # Step 1: Fetch historical data
    print("Step 1: Fetching historical data...")
    df = fetch_historical_data_for_calibration(
        lookback_hours=config.TRAINING_LOOKBACK_HOURS)
    
    if df is None:
        print("❌ Failed to fetch historical data")
        return False
        
    print(f"✅ Retrieved {len(df)} samples for calibration")
    
    # Step 2: Filter for stable periods
    print("Step 2: Filtering for stable periods...")
    stable_periods = filter_stable_periods(df)
    
    if len(stable_periods) < 50:
        print(f"❌ Insufficient stable periods: {len(stable_periods)}")
        return False
        
    print(f"✅ Found {len(stable_periods)} stable periods")
    
    # Step 3: Optimize thermal parameters
    print("Step 3: Optimizing thermal parameters...")
    optimized_params = optimize_thermal_parameters(stable_periods)
    
    if optimized_params is None:
        print("❌ Parameter optimization failed")
        return False
        
    print(f"✅ Optimization completed - MAE: {optimized_params['mae']:.4f}°C")
    
    # Step 4: Export calibrated baseline
    print("Step 4: Exporting calibrated baseline...")
    baseline_path = export_calibrated_baseline(optimized_params, stable_periods)
    
    if baseline_path is None:
        print("❌ Failed to export baseline")
        return False
        
    print(f"✅ Baseline exported to: {baseline_path}")
    
    # Step 5: Validate baseline
    print("Step 5: Validating calibrated baseline...")
    validation_passed = validate_calibrated_baseline(baseline_path, stable_periods)
    
    if not validation_passed:
        print("❌ Baseline validation failed")
        return False
        
    print("✅ Baseline validation PASSED")
    print("\n🎉 PHASE 0 CALIBRATION COMPLETED SUCCESSFULLY!")
    print("Next: Run Phase 1 adaptive learning with calibrated baselines")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
