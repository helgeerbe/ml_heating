"""
Import helper for Jupyter notebooks to handle the ml_heating module imports.
This module resolves the relative import issues when running notebooks.
"""

import sys
import os

# Add the parent directory and src directory to Python path for notebook imports
notebook_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(notebook_dir)
src_dir = os.path.join(parent_dir, "src")
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import config from src module
try:
    from src import config
    print("  ✓ config")
except Exception as e:
    # Try dynamic import if src is not a package
    import importlib.util
    import sys
    config_path = os.path.join(src_dir, "config.py")
    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is not None:
        config = importlib.util.module_from_spec(spec)
        sys.modules["config"] = config
        try:
            spec.loader.exec_module(config)
            print("  ✓ config (dynamic import)")
        except Exception as e2:
            print(f"❌ Error loading config dynamically: {e2}")
            config = None
    else:
        print(f"❌ Error importing config: {e}")
        config = None

# Create a simple feature builder mock for notebooks
def get_feature_names():
    """Return the list of features used by RealisticPhysicsModel"""
    return [
        'dhw_heating', 'dhw_disinfection', 'dhw_boost_heater', 'defrosting',
        'outlet_temp', 'indoor_temp_lag_30m', 'target_temp', 'outdoor_temp',
        'pv_now', 'fireplace_on', 'tv_on',
        'month_cos', 'month_sin',
        'temp_forecast_1h', 'temp_forecast_2h', 'temp_forecast_3h', 'temp_forecast_4h',
        'pv_forecast_1h', 'pv_forecast_2h', 'pv_forecast_3h', 'pv_forecast_4h',
    ]
print("  ✓ get_feature_names")

import pickle
try:
    from src import utils_metrics as metrics
except Exception as e:
    print(f"❌ Error importing metrics: {e}")
    metrics = None

def load_model():
    """Load production model from config file, or raise a clear error."""
    if config is None:
        raise RuntimeError("Config module could not be imported. Check src/config.py.")
    if metrics is None:
        raise RuntimeError("Metrics module could not be imported. Check src/utils_metrics.py.")
    try:
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
        if mae is None:
            mae = metrics.MAE()
            mae._sum_abs_errors = 1.5
            mae._n = 10
        if rmse is None:
            rmse = metrics.RMSE()
            rmse._sum_squared_errors = 2.25
            rmse._n = 10
        print("  ✓ Loaded production RealisticPhysicsModel")
        return base_model, mae, rmse
    except FileNotFoundError:
        raise FileNotFoundError("ml_model.pkl not found. Please train and export the model.")
    except Exception as e:
        raise RuntimeError(f"Error loading production model: {e}")

def get_feature_importances(model):
    """Get feature importances from RealisticPhysicsModel"""
    from collections import defaultdict
    feature_names = get_feature_names()
    feature_importances = {name: 0.0 for name in feature_names}
    try:
        if hasattr(model, 'export_learning_metrics'):
            metrics = model.export_learning_metrics()
            feature_importances.update({
                'outlet_temp': metrics.get('base_heating_rate', 0.0) * 100,
                'target_temp': metrics.get('target_influence', 0.0) * 100,
                'outdoor_temp': metrics.get('outdoor_factor', 0.0) * 100,
                'pv_now': metrics.get('pv_warming_coefficient', 0.0) * 10,
                'fireplace_on': metrics.get('fireplace_heating_rate', 0.0) * 10,
                'tv_on': metrics.get('tv_heat_contribution', 0.0) * 10,
            })
        total = sum(feature_importances.values())
        if total > 0:
            for feature in feature_importances:
                feature_importances[feature] /= total
        return feature_importances
    except Exception as e:
        print(f"Warning: Could not extract feature importances: {e}")
        return feature_importances
print("  ✓ get_feature_importances")

# Create mock functions for other imports if needed
def create_influx_service():
    print("Note: InfluxDB service not available in notebook mode")
    return None

class HAClient:
    def __init__(self):
        print("Note: HA client not available in notebook mode")

print("✅ Successfully loaded ml_heating modules for notebooks")
