"""
Physics Model Calibration for ML Heating Controller

This module provides calibration functionality for the realistic physics model
using historical target temperature data and actual house behavior.
"""

import logging
import pandas as pd
from river import metrics

from . import config
from .physics_model import RealisticPhysicsModel
from .model_wrapper import save_model
from .feature_builder import build_features_for_training
from .influx_service import InfluxService


def train_realistic_physics_model():
    """Train the Realistic Physics Model with target temperature awareness"""
    
    logging.info("=== REALISTIC PHYSICS MODEL TRAINING ===")
    
    # Initialize components
    model = RealisticPhysicsModel()
    mae = metrics.MAE()
    rmse = metrics.RMSE()
    
    influx = InfluxService(
        url=config.INFLUX_URL,
        token=config.INFLUX_TOKEN,
        org=config.INFLUX_ORG
    )
    
    logging.info("Fetching historical data with target temperatures...")
    df = influx.get_training_data(lookback_hours=168)  # 1 week
    
    if df.empty or len(df) < 240:
        logging.error("ERROR: Insufficient training data")
        return None
    
    logging.info(f"Processing {len(df)} samples ({len(df)/12:.1f} hours)")
    
    # Enhanced feature preparation with target temperatures
    features_list = []
    labels_list = []
    
    logging.info("Building features with target temperature context...")
    for idx in range(12, len(df) - config.PREDICTION_HORIZON_STEPS):
        features = build_features_for_training(df, idx)
        if features is None:
            continue
        
        # Add target temperature to features
        target_temp_col = config.TARGET_INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
        target_temp = df.iloc[idx].get(target_temp_col, 21.0)
        features['target_temp'] = float(target_temp)
        
        # Calculate actual temperature change
        indoor_col = config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
        current_indoor = df.iloc[idx].get(indoor_col)
        future_indoor = df.iloc[idx + config.PREDICTION_HORIZON_STEPS].get(indoor_col)
        
        if pd.isna(current_indoor) or pd.isna(future_indoor) or pd.isna(target_temp):
            continue
        
        actual_delta = float(future_indoor) - float(current_indoor)
        
        # Filter for realistic heating scenarios
        temp_gap = target_temp - current_indoor
        if abs(temp_gap) > 2.0:  # Only learn from active heating/cooling needs
            continue
            
        features_list.append(features)
        labels_list.append(actual_delta)
    
    if not features_list:
        logging.error("ERROR: No valid training samples after filtering")
        return None
        
    logging.info(f"Training on {len(features_list)} realistic heating scenarios")
    
    # Training loop
    logging.info("\n=== TRAINING WITH TARGET AWARENESS ===")
    for i, (features, target) in enumerate(zip(features_list, labels_list)):
        pred = model.predict_one(features)
        model.learn_one(features, target)
        
        # Track metrics after warm-up
        if i >= 50:
            mae.update(target, pred)
            rmse.update(target, pred)
        
        if (i + 1) % 200 == 0:
            logging.info(f"Processed {i+1}/{len(features_list)} - MAE: {mae.get():.4f}°C")
    
    # Results
    logging.info(f"\n=== TRAINING COMPLETED ===")
    logging.info(f"Final MAE: {mae.get():.4f}°C")
    logging.info(f"Final RMSE: {rmse.get():.4f}°C")
    
    logging.info(f"\n=== LEARNED REALISTIC PHYSICS ===")
    logging.info(f"Base heating rate: {model.base_heating_rate:.6f}")
    logging.info(f"Target influence: {model.target_influence:.6f}")
    logging.info(f"PV warming coeff: {model.pv_warming_coefficient:.6f} per 100W")
    logging.info(f"Fireplace heating: {model.fireplace_heating_rate:.6f}")
    logging.info(f"TV heat contrib: {model.tv_heat_contribution:.6f}")
    
    # Test realistic physics
    logging.info(f"\n=== PHYSICS VALIDATION ===")
    test_features = {
        'outlet_temp': 45.0,
        'indoor_temp_lag_30m': 21.0,
        'target_temp': 22.0,  # Target higher than current
        'outdoor_temp': 5.0,
        'dhw_heating': 0.0,
        'defrosting': 0.0,
        'dhw_disinfection': 0.0,
        'dhw_boost_heater': 0.0,
        'fireplace_on': 0.0,
        'pv_now': 0.0,
        'tv_on': 0.0,
        'temp_forecast_1h': 5.0,
        'temp_forecast_2h': 5.0, 
        'temp_forecast_3h': 5.0,
        'temp_forecast_4h': 5.0,
        'pv_forecast_1h': 0.0,
        'pv_forecast_2h': 0.0,
        'pv_forecast_3h': 0.0,
        'pv_forecast_4h': 0.0,
    }
    
    logging.info("Testing with target=22°C, current=21°C (heating needed):")
    for temp in [25, 35, 45, 55, 65]:
        test_features['outlet_temp'] = temp
        pred = model.predict_one(test_features)
        logging.info(f"  Outlet {temp}°C → {pred:+.4f}°C change")
    
    logging.info("\nTesting with target=21°C, current=21°C (no heating needed):")
    test_features['target_temp'] = 21.0
    for temp in [25, 35, 45, 55, 65]:
        test_features['outlet_temp'] = temp
        pred = model.predict_one(test_features)
        logging.info(f"  Outlet {temp}°C → {pred:+.4f}°C change")
    
    # Save the trained model
    logging.info(f"\n=== SAVING REALISTIC PHYSICS MODEL ===")
    try:
        save_model(model, mae, rmse)
        logging.info(f"✅ Realistic Physics Model saved to {config.MODEL_FILE}")
        logging.info("🔄 Restart ml_heating service to use trained model:")
        logging.info("   systemctl restart ml_heating")
    except Exception as e:
        logging.error(f"❌ Failed to save model: {e}")
    
    return model, mae, rmse


def validate_physics_model():
    """Validate physics model behavior across temperature ranges"""
    
    logging.info("=== PHYSICS MODEL VALIDATION ===")
    
    try:
        physics_model = RealisticPhysicsModel()
        
        test_features = {
            'outlet_temp': 45.0,
            'indoor_temp_lag_30m': 21.0,
            'target_temp': 22.0,
            'outdoor_temp': 5.0,
            'dhw_heating': 0.0,
            'defrosting': 0.0,
            'dhw_disinfection': 0.0,
            'dhw_boost_heater': 0.0,
            'fireplace_on': 0.0,
            'pv_now': 0.0,
            'tv_on': 0.0,
            'temp_forecast_1h': 5.0,
            'temp_forecast_2h': 5.0, 
            'temp_forecast_3h': 5.0,
            'temp_forecast_4h': 5.0,
            'pv_forecast_1h': 0.0,
            'pv_forecast_2h': 0.0,
            'pv_forecast_3h': 0.0,
            'pv_forecast_4h': 0.0,
        }
        
        logging.info("Testing physics compliance across temperature range:")
        print("\nOUTLET TEMP → PREDICTED CHANGE")
        print("=" * 35)
        
        monotonic_check = []
        for temp in [25, 30, 35, 40, 45, 50, 55, 60]:
            test_features['outlet_temp'] = temp
            pred = physics_model.predict_one(test_features)
            monotonic_check.append(pred)
            print(f"{temp:3d}°C       → {pred:+.6f}°C")
        
        # Check monotonicity
        is_monotonic = all(monotonic_check[i] <= monotonic_check[i+1] 
                         for i in range(len(monotonic_check)-1))
        
        print(f"\n{'✅' if is_monotonic else '❌'} Physics compliance: {'PASSED' if is_monotonic else 'FAILED'}")
        print(f"Range: {min(monotonic_check):.6f} to {max(monotonic_check):.6f}")
        
        return is_monotonic
        
    except Exception as e:
        logging.error("Physics validation error: %s", e, exc_info=True)
        return False


def deploy_physics_only_model():
    """Deploy pure physics model without ML training"""
    
    logging.info("=== PHYSICS-ONLY MODE ===")
    logging.info("Using only realistic physics model (no ML training)")
    
    # Create physics model directly
    from .model_wrapper import MockMetric
    model = RealisticPhysicsModel()
    mae = MockMetric(0.15)  # Preset reasonable performance
    rmse = MockMetric(0.20)
    
    try:
        save_model(model, mae, rmse)
        logging.info("✅ Physics-only model deployed!")
        return True
    except Exception as e:
        logging.error(f"❌ Failed to deploy physics model: {e}")
