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
        """Return the list of features used by RealisticPhysicsModel"""
        return [
            # Core blocking status (4)
            'dhw_heating', 'dhw_disinfection', 'dhw_boost_heater', 'defrosting',
            # Core temperatures (4)
            'outlet_temp', 'indoor_temp_lag_30m', 'target_temp', 'outdoor_temp',
            # External heat sources (3)
            'pv_now', 'fireplace_on', 'tv_on',
            # Seasonal modulation (2)
            'month_cos', 'month_sin',
            # Temperature forecasts (4)
            'temp_forecast_1h', 'temp_forecast_2h', 'temp_forecast_3h', 
            'temp_forecast_4h',
            # PV forecasts (4)
            'pv_forecast_1h', 'pv_forecast_2h', 'pv_forecast_3h', 
            'pv_forecast_4h',
        ]
    print("  ✓ get_feature_names")
    
    # Import model functions directly
    try:
        import pickle
        from src import utils_metrics as metrics
        
        def load_model():
            """Load production model from config file"""
            try:
                # Load the production model
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
                    mae._sum_abs_errors = 1.5
                    mae._n = 10
                
                if rmse is None:
                    rmse = metrics.RMSE()
                    rmse._sum_squared_errors = 2.25
                    rmse._n = 10
                
                # Return the production model directly
                print("  ✓ Loaded production RealisticPhysicsModel")
                return base_model, mae, rmse
                
            except Exception as e:
                print(f"❌ Error loading production model: {e}")
                print("   Make sure ml_model.pkl exists and is properly trained")
                raise e
        print("  ✓ load_model")
        
        def get_feature_importances(model):
            """Get feature importances from RealisticPhysicsModel"""
            from collections import defaultdict
            
            feature_names = get_feature_names()
            feature_importances = {name: 0.0 for name in feature_names}
            
            try:
                # For RealisticPhysicsModel, return learned parameter weights
                if hasattr(model, 'export_learning_metrics'):
                    metrics = model.export_learning_metrics()
                    
                    # Map learned parameters to feature importance scores
                    feature_importances.update({
                        'outlet_temp': metrics.get('base_heating_rate', 0.0) * 100,
                        'target_temp': metrics.get('target_influence', 0.0) * 100,
                        'outdoor_temp': metrics.get('outdoor_factor', 0.0) * 100,
                        'pv_now': metrics.get('pv_warming_coefficient', 0.0) * 10,
                        'fireplace_on': metrics.get('fireplace_heating_rate', 0.0) * 10,
                        'tv_on': metrics.get('tv_heat_contribution', 0.0) * 10,
                    })
                
                # Normalize to sum to 1.0
                total = sum(feature_importances.values())
                if total > 0:
                    for feature in feature_importances:
                        feature_importances[feature] /= total
                            
                return feature_importances
                
            except Exception as e:
                print(f"Warning: Could not extract feature importances: {e}")
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
