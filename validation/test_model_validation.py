#!/usr/bin/env python3
"""
Physics Model Validation Script with Train/Test Split

This script creates a separate model file by:
1. Loading historical data from InfluxDB
2. Splitting dataset into 2/3 training, 1/3 testing
3. Training a new physics model on training data
4. Evaluating performance on unseen test data
5. Comparing with production model performance
6. Saving the validated model to a separate file

Usage:
    python test_model_validation.py
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from typing import Dict, Tuple

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import config  # noqa: E402
from physics_model import RealisticPhysicsModel  # noqa: E402
from model_wrapper import MAE, RMSE, load_model  # noqa: E402
from influx_service import InfluxService  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)



class ModelValidation:
    """Model validation with train/test split and performance comparison"""
    
    def __init__(self):
        """Initialize validation components"""
        self.influx = InfluxService(
            url=config.INFLUX_URL,
            token=config.INFLUX_TOKEN,
            org=config.INFLUX_ORG
        )
        self.validation_model_file = "physics_model_validation.pkl"
        
    def load_and_prepare_data(self, lookback_hours: int = 336) -> pd.DataFrame:
        """
        Load historical data and prepare features
        
        Args:
            lookback_hours: Hours of data to load (default: 2 weeks)
            
        Returns:
            DataFrame with prepared features and labels
        """
        logging.info("=== LOADING HISTORICAL DATA ===")
        logging.info(f"Fetching {lookback_hours} hours of historical data...")
        
        # Load raw data from InfluxDB
        df = self.influx.get_training_data(lookback_hours=lookback_hours)
        
        if df.empty or len(df) < 500:
            raise ValueError(
                f"Insufficient data: got {len(df)} samples, need at least 500"
            )
        
        logging.info(f"Loaded {len(df)} raw samples ({len(df)/6:.1f} hours)")
        
        # Prepare feature matrix and labels
        features_list = []
        labels_list = []
        timestamps_list = []
        
        logging.info("Preparing features with target temperature context...")
        
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
            indoor_lag_30m = (
                df.iloc[idx - 3].get(indoor_col) 
                if idx >= 3 else indoor_temp
            )
            
            # Skip if missing critical data
            if (pd.isna(outlet_temp) or pd.isna(indoor_temp) or 
                    pd.isna(outdoor_temp)):
                continue
            
            # Calculate actual temperature change
            future_idx = idx + config.PREDICTION_HORIZON_STEPS
            future_indoor = df.iloc[future_idx].get(indoor_col)
            if pd.isna(future_indoor) or pd.isna(target_temp):
                continue
            
            actual_delta = float(future_indoor) - float(indoor_temp)
            
            # Enhanced data filtering for outlier removal
            temp_gap = target_temp - indoor_temp
            
            # 1. Skip very large temperature gaps (system faults)
            if abs(temp_gap) > 3.0:
                continue
                
            # 2. Enhanced outlier filtering for realistic temperature changes
            if abs(actual_delta) > 0.5:  # TIGHTENED: Skip large unrealistic changes
                continue
                
            # 3. Sensor consistency checks
            if abs(indoor_temp - target_temp) > 5.0:  # Extreme sensor divergence
                continue
                
            # 4. Physics constraint validation
            if outlet_temp < 10.0 or outlet_temp > 70.0:  # Unrealistic outlet temps
                continue
                
            # 5. Rate limiting - check previous temperature for sudden jumps
            if idx >= 6:  # Need some history
                prev_indoor = df.iloc[idx - 6].get(indoor_col)  # 1 hour ago
                if prev_indoor is not None:
                    temp_rate = abs(indoor_temp - prev_indoor)  # Change over 1 hour
                    if temp_rate > 2.0:  # Skip sudden temperature jumps (sensor fault)
                        continue
            
            # Build feature dictionary
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
            timestamps_list.append(row['_time'])
        
        if not features_list:
            raise ValueError("No valid training samples after filtering")
            
        logging.info(f"Prepared {len(features_list)} training samples")
        
        # Create DataFrame with features, labels, and timestamps
        features_df = pd.DataFrame(features_list)
        features_df['actual_delta'] = labels_list
        features_df['timestamp'] = timestamps_list
        
        return features_df
    
    def split_data(self, data: pd.DataFrame, 
                   train_ratio: float = 2/3) -> Tuple[pd.DataFrame, 
                                                      pd.DataFrame]:
        """
        Split data into training and testing sets
        
        Args:
            data: Prepared dataset
            train_ratio: Fraction for training (default: 2/3)
            
        Returns:
            Tuple of (train_data, test_data)
        """
        logging.info(
            f"=== SPLITTING DATA ({train_ratio:.1%} train, "
            f"{1-train_ratio:.1%} test) ==="
        )
        
        # Sort by timestamp to ensure proper time-based split
        data_sorted = data.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate split point
        split_idx = int(len(data_sorted) * train_ratio)
        
        train_data = data_sorted.iloc[:split_idx].copy()
        test_data = data_sorted.iloc[split_idx:].copy()
        
        logging.info(f"Training set: {len(train_data)} samples")
        logging.info(f"Test set: {len(test_data)} samples")
        train_start = train_data['timestamp'].min()
        train_end = train_data['timestamp'].max()
        test_start = test_data['timestamp'].min() 
        test_end = test_data['timestamp'].max()
        
        logging.info(f"Train period: {train_start} to {train_end}")
        logging.info(f"Test period: {test_start} to {test_end}")
        
        return train_data, test_data
    
    def train_model(self, train_data: pd.DataFrame) -> Tuple[
            RealisticPhysicsModel, MAE, RMSE]:
        """
        Train a new physics model on training data
        
        Args:
            train_data: Training dataset
            
        Returns:
            Tuple of (trained_model, mae_metric, rmse_metric)
        """
        logging.info("=== TRAINING NEW MODEL ===")
        
        # Initialize fresh model and metrics
        model = RealisticPhysicsModel()
        mae = MAE()
        rmse = RMSE()
        
        # Extract features and labels
        feature_columns = [
            col for col in train_data.columns 
            if col not in ['actual_delta', 'timestamp']
        ]
        
        logging.info("Training model with online learning...")
        
        # Training loop
        for i, row in train_data.iterrows():
            features = row[feature_columns].to_dict()
            target = row['actual_delta']
            
            # Make prediction before learning
            pred = model.predict_one(features)
            
            # Learn from this sample
            model.learn_one(features, target)
            
            # Track metrics after warm-up period
            if i >= 50:
                mae.update(target, pred)
                rmse.update(target, pred)
            
            # Progress logging
            if (i + 1) % 500 == 0:
                progress_msg = (
                    f"Trained on {i+1}/{len(train_data)} samples - "
                    f"MAE: {mae.get():.4f}°C"
                )
                logging.info(progress_msg)
        
        completion_msg = (
            f"Training completed - Final MAE: {mae.get():.4f}°C, "
            f"RMSE: {rmse.get():.4f}°C"
        )
        logging.info(completion_msg)
        
        return model, mae, rmse
    
    def test_model(self, model: RealisticPhysicsModel, 
                   test_data: pd.DataFrame) -> Tuple[float, float, Dict]:
        """
        Test model performance on unseen data
        
        Args:
            model: Trained model
            test_data: Test dataset
            
        Returns:
            Tuple of (test_mae, test_rmse, detailed_results)
        """
        logging.info("=== TESTING MODEL PERFORMANCE ===")
        
        # Initialize test metrics
        test_mae = MAE()
        test_rmse = MAE()  # Using MAE structure for RMSE calculation
        
        # Extract features and labels
        feature_columns = [
            col for col in test_data.columns 
            if col not in ['actual_delta', 'timestamp']
        ]
        
        predictions = []
        actuals = []
        errors = []
        
        # Test on each sample (without learning)
        for i, row in test_data.iterrows():
            features = row[feature_columns].to_dict()
            actual = row['actual_delta']
            
            # Predict without learning
            pred = model.predict_one(features)
            
            # Track metrics
            test_mae.update(actual, pred)
            error = abs(actual - pred)
            
            # Store for detailed analysis
            predictions.append(pred)
            actuals.append(actual)
            errors.append(error)
            
            # Manual RMSE calculation
            test_rmse._sum_abs_errors += (actual - pred) ** 2
            test_rmse._n += 1
        
        # Calculate final test RMSE
        final_test_rmse = (test_rmse._sum_abs_errors / test_rmse._n) ** 0.5
        
        # Detailed results
        results = {
            'test_mae': test_mae.get(),
            'test_rmse': final_test_rmse,
            'predictions': predictions,
            'actuals': actuals,
            'errors': errors,
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(errors),
            'min_error': np.min(errors),
            'samples_tested': len(test_data)
        }
        
        logging.info(f"Test Results:")
        logging.info(f"  MAE: {results['test_mae']:.4f}°C")
        logging.info(f"  RMSE: {results['test_rmse']:.4f}°C")
        logging.info(f"  Mean Error: {results['mean_error']:.4f}°C")
        logging.info(f"  Std Error: {results['std_error']:.4f}°C")
        error_range = (
            f"  Error Range: {results['min_error']:.4f}°C to "
            f"{results['max_error']:.4f}°C"
        )
        logging.info(error_range)
        logging.info(f"  Samples Tested: {results['samples_tested']}")
        
        # Add detailed prediction analysis
        self._analyze_prediction_quality(results, test_data)
        
        return results['test_mae'], results['test_rmse'], results
    
    def _analyze_prediction_quality(self, results: Dict, test_data: pd.DataFrame):
        """
        Analyze prediction quality with detailed examples
        
        Args:
            results: Test results dictionary
            test_data: Original test dataset
        """
        logging.info("\n=== DETAILED PREDICTION ANALYSIS ===")
        
        predictions = results['predictions']
        actuals = results['actuals']
        errors = results['errors']
        
        # Show sample predictions vs actuals
        logging.info("Sample predictions vs actual temperature changes:")
        logging.info("Format: Predicted ΔT | Actual ΔT | Error | Accuracy")
        logging.info("-" * 55)
        
        # Show first 10 predictions
        for i in range(min(10, len(predictions))):
            pred = predictions[i]
            actual = actuals[i]
            error = errors[i]
            accuracy = (1 - min(error / max(abs(actual), 0.1), 1)) * 100
            
            logging.info(f"{pred:+6.3f}°C   | {actual:+6.3f}°C   | {error:5.3f}°C | {accuracy:4.1f}%")
        
        # Accuracy distribution
        accuracy_ranges = [0.1, 0.05, 0.02, 0.01]  # Error thresholds
        logging.info(f"\nPrediction Accuracy Distribution:")
        logging.info(f"Total test samples: {len(errors)}")
        
        for threshold in accuracy_ranges:
            within_threshold = sum(1 for e in errors if e <= threshold)
            percentage = (within_threshold / len(errors)) * 100
            logging.info(f"  Within ±{threshold:.3f}°C: {within_threshold:4d} samples ({percentage:5.1f}%)")
        
        # Error quartiles
        errors_sorted = sorted(errors)
        q1 = errors_sorted[len(errors_sorted) // 4]
        q2 = errors_sorted[len(errors_sorted) // 2]  # median
        q3 = errors_sorted[3 * len(errors_sorted) // 4]
        
        logging.info(f"\nError Distribution Quartiles:")
        logging.info(f"  Q1 (25th percentile): {q1:.4f}°C")
        logging.info(f"  Q2 (median):          {q2:.4f}°C")
        logging.info(f"  Q3 (75th percentile): {q3:.4f}°C")
        
        # Prediction vs actual correlation
        import numpy as np
        correlation = np.corrcoef(predictions, actuals)[0, 1]
        logging.info(f"\nPrediction-Actual Correlation: {correlation:.4f}")
        
        # Best and worst predictions
        best_idx = errors.index(min(errors))
        worst_idx = errors.index(max(errors))
        
        logging.info(f"\nBest Prediction:")
        logging.info(f"  Predicted: {predictions[best_idx]:+.4f}°C, Actual: {actuals[best_idx]:+.4f}°C, Error: {errors[best_idx]:.4f}°C")
        
        logging.info(f"Worst Prediction:")
        logging.info(f"  Predicted: {predictions[worst_idx]:+.4f}°C, Actual: {actuals[worst_idx]:+.4f}°C, Error: {errors[worst_idx]:.4f}°C")
        
        # Prediction bias analysis
        bias = np.mean(np.array(predictions) - np.array(actuals))
        abs_bias = abs(bias)
        logging.info(f"\nPrediction Bias: {bias:+.4f}°C")
        if abs_bias < 0.01:
            logging.info("  → Excellent: Very low systematic bias")
        elif abs_bias < 0.05:
            logging.info("  → Good: Low systematic bias")
        else:
            logging.info(f"  → Moderate bias detected ({'over' if bias > 0 else 'under'}prediction)")
        
        # Temperature range analysis
        heating_predictions = [(p, a, e) for p, a, e in zip(predictions, actuals, errors) if a > 0.01]
        cooling_predictions = [(p, a, e) for p, a, e in zip(predictions, actuals, errors) if a < -0.01]
        neutral_predictions = [(p, a, e) for p, a, e in zip(predictions, actuals, errors) if abs(a) <= 0.01]
        
        if heating_predictions:
            heating_mae = np.mean([e for _, _, e in heating_predictions])
            logging.info(f"\nHeating Scenarios ({len(heating_predictions)} samples):")
            logging.info(f"  MAE: {heating_mae:.4f}°C")
        
        if cooling_predictions:
            cooling_mae = np.mean([e for _, _, e in cooling_predictions])
            logging.info(f"Cooling Scenarios ({len(cooling_predictions)} samples):")
            logging.info(f"  MAE: {cooling_mae:.4f}°C")
        
        if neutral_predictions:
            neutral_mae = np.mean([e for _, _, e in neutral_predictions])
            logging.info(f"Neutral Scenarios ({len(neutral_predictions)} samples):")
            logging.info(f"  MAE: {neutral_mae:.4f}°C")
    
    def compare_with_production(self, validation_results: Dict):
        """
        Compare validation model with production model
        
        Args:
            validation_results: Results from validation model testing
        """
        logging.info("=== COMPARING WITH PRODUCTION MODEL ===")
        
        try:
            # Load production model
            prod_model, prod_mae, prod_rmse = load_model()
            
            logging.info("Production model metrics:")
            logging.info(f"  MAE: {prod_mae.get():.4f}°C")
            logging.info(f"  RMSE: {prod_rmse.get():.4f}°C")
            
            logging.info("Validation model test metrics:")
            logging.info(f"  MAE: {validation_results['test_mae']:.4f}°C")
            logging.info(f"  RMSE: {validation_results['test_rmse']:.4f}°C")
            
            # Calculate improvement/degradation
            mae_diff = validation_results['test_mae'] - prod_mae.get()
            rmse_diff = validation_results['test_rmse'] - prod_rmse.get()
            
            logging.info("Performance comparison:")
            mae_pct = mae_diff/prod_mae.get()*100
            rmse_pct = rmse_diff/prod_rmse.get()*100
            logging.info(f"  MAE difference: {mae_diff:+.4f}°C ({mae_pct:+.1f}%)")
            logging.info(f"  RMSE difference: {rmse_diff:+.4f}°C ({rmse_pct:+.1f}%)")
            
            if mae_diff < 0 and rmse_diff < 0:
                logging.info(
                    "✅ Validation model shows IMPROVEMENT over production model"
                )
            elif mae_diff > 0 or rmse_diff > 0:
                logging.info(
                    "⚠️  Validation model shows mixed/degraded performance"
                )
            else:
                logging.info("➡️  Validation model shows similar performance")
                
        except Exception as e:
            logging.warning(f"Could not load production model for comparison: {e}")
    
    def test_physics_behavior(self, model: RealisticPhysicsModel):
        """
        Test physics model behavior across different scenarios
        
        Args:
            model: Model to test
        """
        logging.info("=== TESTING PHYSICS BEHAVIOR ===")
        
        # Base test features
        base_features = {
            'indoor_temp_lag_30m': 21.0,
            'target_temp': 21.0,
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
        
        logging.info(
            "Testing outlet temperature response (target=21°C, current=21°C):"
        )
        outlet_temps = [20, 25, 30, 35, 40, 45, 50, 55, 60]
        predictions = []
        
        for outlet_temp in outlet_temps:
            test_features = base_features.copy()
            test_features['outlet_temp'] = outlet_temp
            pred = model.predict_one(test_features)
            predictions.append(pred)
            logging.info(f"  Outlet {outlet_temp:2d}°C → {pred:+.4f}°C change")
        
        # Check monotonicity
        is_monotonic = all(
            predictions[i] <= predictions[i+1] 
            for i in range(len(predictions)-1)
        )
        
        compliance_status = '✅ PASSED' if is_monotonic else '❌ FAILED'
        logging.info(f"Physics compliance: {compliance_status}")
        
        # Test cooling scenario
        logging.info(
            "\nTesting cooling scenario (target=21°C, current=22°C):"
        )
        base_features['indoor_temp_lag_30m'] = 22.0
        
        for outlet_temp in [15, 20, 25, 30]:
            test_features = base_features.copy()
            test_features['outlet_temp'] = outlet_temp
            pred = model.predict_one(test_features)
            logging.info(f"  Outlet {outlet_temp:2d}°C → {pred:+.4f}°C change")
        
        # Test heating scenario
        logging.info(
            "\nTesting heating scenario (target=21°C, current=20°C):"
        )
        base_features['indoor_temp_lag_30m'] = 20.0
        
        for outlet_temp in [30, 35, 40, 45]:
            test_features = base_features.copy()
            test_features['outlet_temp'] = outlet_temp
            pred = model.predict_one(test_features)
            logging.info(f"  Outlet {outlet_temp:2d}°C → {pred:+.4f}°C change")
    
    def save_validation_model(self, model: RealisticPhysicsModel, 
                              mae: MAE, rmse: RMSE):
        """
        Save validation model to separate file
        
        Args:
            model: Trained model
            mae: MAE metric
            rmse: RMSE metric
        """
        logging.info("=== SAVING VALIDATION MODEL ===")
        
        try:
            with open(self.validation_model_file, "wb") as f:
                pickle.dump({
                    "model": model,
                    "mae": mae,
                    "rmse": rmse,
                    "validation_timestamp": datetime.now(),
                    "model_type": "RealisticPhysicsModel_Validation"
                }, f)
                
            logging.info(f"✅ Validation model saved to {self.validation_model_file}")
            logging.info("   This model can be compared with production model")
            logging.info(f"   To use this model, copy it to {config.MODEL_FILE}")
            
        except Exception as e:
            logging.error(f"❌ Failed to save validation model: {e}")
    
    def run_full_validation(self, lookback_hours: int = None):
        """
        Run complete model validation pipeline
        
        Args:
            lookback_hours: Hours of data to use (default: from config)
        """
        # Use config parameter if not specified
        if lookback_hours is None:
            lookback_hours = getattr(config, 'CALIBRATION_LOOKBACK_HOURS', 336)
        
        logging.info("🚀 STARTING PHYSICS MODEL VALIDATION WITH REAL INFLUX DATA")
        days = lookback_hours/24
        logging.info(f"Using {lookback_hours} hours ({days:.1f} days) from InfluxDB")
        logging.info(f"InfluxDB: {config.INFLUX_URL}")
        logging.info(f"Bucket: {config.INFLUX_BUCKET}")
        
        try:
            # Step 1: Load and prepare data
            data = self.load_and_prepare_data(lookback_hours)
            
            # Step 2: Split into train/test
            train_data, test_data = self.split_data(data)
            
            # Step 3: Train new model
            model, train_mae, train_rmse = self.train_model(train_data)
            
            # Step 4: Test model performance
            test_mae, test_rmse, detailed_results = self.test_model(model, test_data)
            
            # Step 5: Test physics behavior
            self.test_physics_behavior(model)
            
            # Step 6: Compare with production
            self.compare_with_production(detailed_results)
            
            # Step 7: Save validation model
            self.save_validation_model(model, train_mae, train_rmse)
            
            # Summary
            logging.info("🎯 VALIDATION SUMMARY")
            logging.info(f"  Training MAE: {train_mae.get():.4f}°C")
            logging.info(f"  Training RMSE: {train_rmse.get():.4f}°C")
            logging.info(f"  Test MAE: {test_mae:.4f}°C")
            logging.info(f"  Test RMSE: {test_rmse:.4f}°C")
            logging.info(f"  Validation model saved to: {self.validation_model_file}")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Validation failed: {e}", exc_info=True)
            return False



def main():
    """Main validation script"""
    logging.info("Physics Model Validation with Train/Test Split")
    logging.info("=" * 60)
    
    # Initialize validator
    validator = ModelValidation()
    
    # Get lookback hours from config or use default
    lookback_hours = getattr(config, 'CALIBRATION_LOOKBACK_HOURS', 336)
    
    logging.info(f"Configuration:")
    logging.info(f"  Lookback hours: {lookback_hours}")
    logging.info(f"  InfluxDB URL: {config.INFLUX_URL}")
    logging.info(f"  InfluxDB Bucket: {config.INFLUX_BUCKET}")
    logging.info(f"  Prediction horizon: {config.PREDICTION_HORIZON_STEPS} steps")
    
    # Run validation with real InfluxDB data
    success = validator.run_full_validation(lookback_hours=lookback_hours)
    
    if success:
        logging.info("✅ Model validation completed successfully!")
        logging.info("📊 Check the logs above for detailed performance metrics")
        logging.info("📁 Validation model saved for comparison")
    else:
        logging.error("❌ Model validation failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
