"""
Physics Model Calibration for ML Heating Controller

This module provides calibration functionality for the realistic physics model
using historical target temperature data and actual house behavior.
"""

import logging
import pandas as pd

from . import config
from .physics_model import RealisticPhysicsModel
from .model_wrapper import save_model, MAE, RMSE
from .influx_service import InfluxService


def train_realistic_physics_model():
    """Train the Realistic Physics Model with target temperature awareness"""
    
    logging.info("=== REALISTIC PHYSICS MODEL TRAINING ===")
    
    # Initialize components
    model = RealisticPhysicsModel()
    mae = MAE()
    rmse = RMSE()
    
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
            continue
        
        # Calculate actual temperature change
        future_indoor = df.iloc[idx + config.PREDICTION_HORIZON_STEPS].get(
            indoor_col)
        if pd.isna(future_indoor) or pd.isna(target_temp):
            continue
        
        actual_delta = float(future_indoor) - float(indoor_temp)
        
        # Filter for realistic heating scenarios
        temp_gap = target_temp - indoor_temp
        if abs(temp_gap) > 2.0:
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
        
    logging.info(
        f"Training on {len(features_list)} realistic heating scenarios"
    )
    
    # Training loop
    logging.info("\n=== TRAINING WITH TARGET AWARENESS ===")
    for i, (features, target) in enumerate(zip(features_list, labels_list)):
        # Get the original timestamp of the data point being learned from
        # We use the original DataFrame df and the current index idx to get
        # the correct historical timestamp.
        historical_timestamp = df.iloc[idx]['_time']

        pred = model.predict_one(features)
        model.learn_one(features, target)
        
        # Track metrics after warm-up
        if i >= 50:
            mae.update(target, pred)
            rmse.update(target, pred)
        
        if (i + 1) % 200 == 0:
            logging.info(
                f"Processed {i+1}/{len(features_list)} - MAE: {mae.get():.4f}°C"
            )
        
        # Export learning metrics to InfluxDB with the historical timestamp
        learning_metrics = model.export_learning_metrics()
        influx.write_feature_importances(
            learning_metrics, 
            bucket=config.INFLUX_FEATURES_BUCKET,
            measurement="learning_parameters_calibration", # Use a distinct measurement name
            timestamp=historical_timestamp
        )
    
    # Results
    logging.info("\n=== TRAINING COMPLETED ===")
    logging.info(f"Final MAE: {mae.get():.4f}°C")
    logging.info(f"Final RMSE: {rmse.get():.4f}°C")
    
    logging.info("\n=== LEARNED REALISTIC PHYSICS ===")
    logging.info(f"Base heating rate: {model.base_heating_rate:.6f}")
    logging.info(f"Target influence: {model.target_influence:.6f}")
    logging.info(
        f"PV warming coeff: {model.pv_warming_coefficient:.6f} per 100W"
    )
    logging.info(f"Fireplace heating: {model.fireplace_heating_rate:.6f}")
    logging.info(f"TV heat contrib: {model.tv_heat_contribution:.6f}")
    
    # Test realistic physics
    logging.info("\n=== PHYSICS VALIDATION ===")
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
    logging.info("\n=== SAVING REALISTIC PHYSICS MODEL ===")
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
        
        print(
            f"\n{'✅' if is_monotonic else '❌'} Physics compliance: "
            f"{'PASSED' if is_monotonic else 'FAILED'}"
        )
        print(f"Range: {min(monotonic_check):.6f} to {max(monotonic_check):.6f}")
        
        return is_monotonic
        
    except Exception as e:
        logging.error("Physics validation error: %s", e, exc_info=True)
        return False
