"""
Import helper for Jupyter notebooks to handle the ml_heating module imports.
This module resolves the relative import issues when running notebooks.
"""

import sys
import os

# Add the parent directory to Python path
notebook_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(notebook_dir)

# Add both the parent directory and src directory to path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import modules with a proper approach
print("Loading ml_heating modules...")

try:
    # Import config from src module
    from src import config
    print("  ✓ config")
    
    # Create a simple feature builder mock for notebooks
    def get_feature_names():
        """Return the list of feature names"""
        return [
            'outlet_temp', 'outlet_temp_sq', 'outlet_temp_cub',
            'indoor_temp_lag_10m', 'indoor_temp_lag_30m', 'indoor_temp_lag_60m',
            'outdoor_temp', 'temp_diff_indoor_outdoor',
            'outlet_indoor_diff', 'outdoor_temp_x_outlet_temp',
            'outlet_temp_lag_10m', 'outlet_temp_lag_30m', 'outlet_temp_lag_60m',
            'outlet_temp_delta_10m', 'outlet_temp_delta_30m', 'outlet_temp_delta_60m',
            'outlet_temp_change_from_last', 'outlet_temp_gradient',
            'indoor_temp_delta_10m', 'indoor_temp_delta_30m', 'indoor_temp_delta_60m',
            'indoor_temp_gradient', 'outlet_hist_mean', 'outlet_hist_std',
            'outlet_hist_min', 'outlet_hist_max', 'outlet_hist_q25', 'outlet_hist_q75',
            'outlet_hist_trend', 'indoor_hist_mean', 'indoor_hist_std',
            'indoor_hist_min', 'indoor_hist_max', 'indoor_hist_q25', 'indoor_hist_q75',
            'indoor_hist_trend', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            'day_of_week_sin', 'day_of_week_cos', 'is_weekend',
            'pv_now', 'pv_forecast_1h', 'pv_forecast_2h', 'pv_forecast_3h', 'pv_forecast_4h',
            'temp_forecast_1h', 'temp_forecast_2h', 'temp_forecast_3h', 'temp_forecast_4h',
            'dhw_heating', 'defrosting', 'dhw_disinfection', 'dhw_boost_heater',
            'fireplace_on', 'tv_on', 'defrost_count', 'defrost_recent', 'defrost_age_min'
        ]
    print("  ✓ get_feature_names")
    
    # Import model functions directly
    try:
        import pickle
        import numpy as np
        from river import metrics
        
        # Define PhysicsCompliantWrapper for notebooks
        class PhysicsCompliantWrapper:
            """
            Model wrapper that enforces monotonic physics at predict_one() level.
            """
            
            def __init__(self, base_model):
                self.base_model = base_model
                self._prediction_cache = {}
                
            def predict_one(self, features):
                """Make physics-compliant prediction that respects monotonicity"""
                outlet_temp = features.get('outlet_temp', 0.0)
                
                # Create cache key from relevant features
                cache_key = self._create_cache_key(features)
                
                # Check if we have cached monotonic predictions for scenario
                if cache_key not in self._prediction_cache:
                    self._generate_monotonic_cache(features, cache_key)
                
                # Interpolate from cached monotonic curve
                cache_data = self._prediction_cache[cache_key]
                temp_points = cache_data['temps']
                pred_points = cache_data['predictions']
                
                return float(np.interp(outlet_temp, temp_points, pred_points))
            
            def _create_cache_key(self, features):
                """Create cache key from non-outlet-temp features"""
                key_features = [
                    'indoor_temp_lag_30m', 'outdoor_temp', 
                    'temp_diff_indoor_outdoor', 'pv_now', 'defrost_count'
                ]
                key_values = []
                for feat in key_features:
                    val = features.get(feat, 0.0)
                    key_values.append(round(val, 2))
                return tuple(key_values)
            
            def _generate_monotonic_cache(self, features, cache_key):
                """Generate monotonic prediction curve for this scenario"""
                # Define outlet temperature range for interpolation
                temp_range = np.arange(20, 61, 2)
                raw_predictions = []
                
                # Get raw predictions across temperature range
                for temp in temp_range:
                    temp_features = features.copy()
                    temp_features.update({
                        'outlet_temp': temp,
                        'outlet_temp_sq': temp ** 2,
                        'outlet_temp_cub': temp ** 3,
                        'outlet_indoor_diff': temp - features.get(
                            'indoor_temp_lag_30m', 21.0),
                        'outdoor_temp_x_outlet_temp': features.get(
                            'outdoor_temp', 0.0) * temp,
                    })
                    
                    raw_pred = self.base_model.predict_one(temp_features)
                    raw_predictions.append(raw_pred)
                
                # Enforce strict monotonicity
                monotonic_predictions = [raw_predictions[0]]
                for i in range(1, len(raw_predictions)):
                    # Ensure each prediction >= previous + small increment
                    min_allowed = monotonic_predictions[i-1] + 0.001
                    monotonic_predictions.append(
                        max(raw_predictions[i], min_allowed))
                
                # Cache the monotonic curve
                self._prediction_cache[cache_key] = {
                    'temps': list(temp_range),
                    'predictions': monotonic_predictions
                }
                
                # Limit cache size to prevent memory issues
                if len(self._prediction_cache) > 100:
                    # Remove oldest entry
                    oldest_key = next(iter(self._prediction_cache))
                    del self._prediction_cache[oldest_key]
            
            def learn_one(self, features, target):
                """Learn from training data - invalidate relevant cache"""
                cache_key = self._create_cache_key(features)
                if cache_key in self._prediction_cache:
                    del self._prediction_cache[cache_key]
                return self.base_model.learn_one(features, target)
            
            @property 
            def steps(self):
                """Access to underlying pipeline steps"""
                return self.base_model.steps
        
        def load_model():
            """Load model from config file with PhysicsCompliantWrapper"""
            try:
                # Load the base model first
                with open(config.MODEL_FILE, "rb") as f:
                    saved_data = pickle.load(f)
                    if isinstance(saved_data, dict):
                        base_model = saved_data["model"]
                        mae = saved_data.get("mae", metrics.MAE())
                        rmse = saved_data.get("rmse", metrics.RMSE())
                    else:
                        base_model = saved_data
                        mae = metrics.MAE()
                        rmse = metrics.RMSE()
                
                # Ensure metrics are not None
                if mae is None:
                    mae = metrics.MAE()
                    # Set a default value for demo
                    mae._sum_abs_errors = 1.5
                    mae._n = 10
                
                if rmse is None:
                    rmse = metrics.RMSE()
                    # Set a default value for demo
                    rmse._sum_squared_errors = 2.25
                    rmse._n = 10
                
                # Apply PhysicsCompliantWrapper manually
                print("  Applying PhysicsCompliantWrapper...")
                model = PhysicsCompliantWrapper(base_model)
                return model, mae, rmse
                
            except Exception as e:
                print(f"Error loading model file: {e}")
                print("Creating demo model for notebook analysis...")
                
                # Create a demo base model for notebook testing
                from river import ensemble, preprocessing, compose, tree
                
                demo_base_model = compose.Pipeline(
                    preprocessing.StandardScaler(),
                    ensemble.BaggingRegressor(
                        model=tree.HoeffdingTreeRegressor(),
                        n_models=10
                    )
                )
                
                # Create demo metrics
                mae = metrics.MAE()
                mae._sum_abs_errors = 1.5
                mae._n = 10
                rmse = metrics.RMSE() 
                rmse._sum_squared_errors = 2.25
                rmse._n = 10
                
                # Apply PhysicsCompliantWrapper 
                print("  Applying PhysicsCompliantWrapper to demo model...")
                model = PhysicsCompliantWrapper(demo_base_model)
                return model, mae, rmse
        print("  ✓ load_model")
        
        def get_feature_importances(model):
            """Get feature importances from model"""
            from collections import defaultdict
            
            feature_names = get_feature_names()
            feature_importances = {name: 0.0 for name in feature_names}
            
            try:
                regressor = model.steps.get("learn")
                if not regressor:
                    return feature_importances
                
                total_importances = defaultdict(int)
                
                for tree_model in regressor:
                    if hasattr(tree_model, "_root"):
                        def traverse(node):
                            if node is None:
                                return
                            if hasattr(node, "feature") and node.feature is not None:
                                total_importances[node.feature] += 1
                            if hasattr(node, "children"):
                                for child in node.children:
                                    traverse(child)
                        traverse(tree_model._root)
                
                total_splits = sum(total_importances.values())
                if total_splits > 0:
                    for feature, count in total_importances.items():
                        if feature in feature_importances:
                            feature_importances[feature] = count / total_splits
                            
                return feature_importances
                
            except Exception as e:
                print(f"Error getting feature importances: {e}")
                return feature_importances
        print("  ✓ get_feature_importances")
        
    except Exception as e:
        print(f"Error setting up model functions: {e}")
        load_model = get_feature_importances = None
    
    # Create mock functions for other imports if needed
    def create_influx_service():
        """Mock influx service - not needed for basic model analysis"""
        print("Note: InfluxDB service not available in notebook mode")
        return None
    
    class HAClient:
        """Mock HA client - not needed for basic model analysis"""
        def __init__(self):
            print("Note: HA client not available in notebook mode")
    
    print("✅ Successfully loaded ml_heating modules for notebooks")
    
except Exception as e:
    print(f"❌ Import error: {e}")
    raise
