"""
Physics Model Calibration for ML Heating Controller

This module provides calibration functionality for the realistic physics model
using historical target temperature data and actual house behavior.
"""

import logging
import pandas as pd

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
    
    # Save thermal learning state
    logging.info("\n=== SAVING THERMAL LEARNING STATE ===")
    try:
        thermal_learning_state = {
            'thermal_time_constant': thermal_model.thermal_time_constant,
            'heat_loss_coefficient': thermal_model.heat_loss_coefficient,
            'outlet_effectiveness': thermal_model.outlet_effectiveness,
            'learning_confidence': thermal_model.learning_confidence,
            'parameter_history': thermal_model.parameter_history[-50:],  # Last 50 updates
            'prediction_history': thermal_model.prediction_history[-100:],  # Last 100 predictions
        }
        
        save_state(thermal_learning_state=thermal_learning_state)
        logging.info("✅ Thermal learning state saved successfully")
        logging.info("🔄 Restart ml_heating service to use trained thermal model")
        
    except Exception as e:
        logging.error(f"❌ Failed to save thermal learning state: {e}")
    
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
        
        if not (4.0 <= thermal_model.thermal_time_constant <= 96.0):
            logging.error(f"Thermal time constant out of bounds: "
                         f"{thermal_model.thermal_time_constant:.2f}h")
            params_ok = False
            
        if not (0.005 <= thermal_model.heat_loss_coefficient <= 0.25):
            logging.error(f"Heat loss coefficient out of bounds: "
                         f"{thermal_model.heat_loss_coefficient:.4f}")
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


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
