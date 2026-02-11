"""
Import helper for Jupyter notebooks to handle the ml_heating module imports.
This module resolves the relative import issues when running notebooks.
"""

import sys
import os
from datetime import datetime

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
        # Load unified thermal state
        from src.unified_thermal_state import ThermalStateManager
        state_manager = ThermalStateManager()
        
        # Create model from state
        from src.thermal_equilibrium_model import ThermalEquilibriumModel
        base_model = ThermalEquilibriumModel()
        
        # Load parameters from state
        params = state_manager.get_parameters()
        base_model.update_parameters(params)
        
        # Get metrics from state
        learning_state = state_manager.state.get('learning_state', {})
        
        mae = metrics.MAE()
        mae._sum_abs_errors = learning_state.get('mae', 1.5) * 10
        mae._n = 10
        
        rmse = metrics.RMSE()
        rmse._sum_squared_errors = learning_state.get('rmse', 2.25) * 10
        rmse._n = 10
        
        print("  ✓ Loaded production ThermalEquilibriumModel from unified state")
        return base_model, mae, rmse
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

# Import real InfluxDB service for notebooks
try:
    from src import influx_service
    
    # Create wrapper class to maintain compatibility with notebook expectations
    class NotebookInfluxService(influx_service.InfluxService):
        """Wrapper to maintain compatibility with notebook fetch_history calls."""
        
        def fetch_history(self, entity_id, steps, default_value):
            """
            Wrapper method for notebook compatibility with the correct signature.
            
            This matches the expected call pattern from notebooks:
            fetch_history(entity_id, steps, default_value)
            """
            # Call the base class method with correct parameters
            return super().fetch_history(entity_id, steps, default_value)
    
    # Expose the wrapped InfluxDB service
    def create_influx_service():
        """Create wrapped InfluxDB service with notebook compatibility."""
        # Use config values directly instead of extracting from client
        wrapped_service = NotebookInfluxService(
            url=config.INFLUX_URL,
            token=config.INFLUX_TOKEN,
            org=config.INFLUX_ORG
        )
        return wrapped_service
        
    InfluxService = NotebookInfluxService
    print("  ✓ influx_service (with notebook compatibility wrapper)")
    
except Exception as e:
    print(f"⚠️ Error importing influx_service: {e}")
    # Fallback mock function if import fails
    def create_influx_service():
        print("Note: InfluxDB service not available - import failed")
        return None
    
    class InfluxService:
        def __init__(self, *args, **kwargs):
            print("Note: InfluxDB service not available - using mock")

# Create mock HA client for notebooks (not needed for monitoring)
class HAClient:
    def __init__(self):
        print("Note: HA client not available in notebook mode")

def strip_entity_domain(entity_id):
    """Strip domain prefix from entity ID like the core system does.
    
    Args:
        entity_id (str): Full entity ID like 'sensor.thermometer_wohnzimmer_kompensiert'
        
    Returns:
        str: Stripped entity ID like 'thermometer_wohnzimmer_kompensiert'
    """
    return entity_id.split(".", 1)[-1]

print("  ✓ strip_entity_domain utility function")
print("✅ Successfully loaded ml_heating modules for notebooks")
