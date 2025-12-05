"""
Unified Thermal State Management - Single JSON File Architecture

This module provides a clean, unified approach to managing all thermal system state
in a single JSON file, eliminating the need for pickle files and multiple JSON files.

Key features:
- Single source of truth for all thermal parameters
- Human-readable JSON format
- Version control friendly
- Calibrated baseline + online learning state
- Robust error handling and validation
"""

import json
import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np

# Import centralized thermal configuration
try:
    from .thermal_config import ThermalParameterConfig
except ImportError:
    from thermal_config import ThermalParameterConfig


class ThermalStateManager:
    """
    Unified thermal state manager using single JSON file.
    
    Replaces:
    - ml_state.pkl (pickle format)
    - calibrated_baseline.json (separate file)  
    - prediction_metrics.json (separate file)
    
    With single thermal_state.json containing everything.
    """
    
    def __init__(self, state_file: str = "/opt/ml_heating/thermal_state.json"):
        self.state_file = state_file
        self.state = self._get_default_state()
        
        # Load existing state if available
        if os.path.exists(state_file):
            self.load_state()
        else:
            logging.info("🆕 Creating new unified thermal state file")
    
    def _get_default_state(self) -> Dict[str, Any]:
        """Get default thermal state structure."""
        return {
            "metadata": {
                "version": "1.0",
                "format": "unified_thermal_state",
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            },
            "baseline_parameters": {
                "thermal_time_constant": ThermalParameterConfig.get_default('thermal_time_constant'),
                "heat_loss_coefficient": ThermalParameterConfig.get_default('heat_loss_coefficient'),
                "outlet_effectiveness": ThermalParameterConfig.get_default('outlet_effectiveness'),
                "pv_heat_weight": ThermalParameterConfig.get_default('pv_heat_weight'),
                "fireplace_heat_weight": ThermalParameterConfig.get_default('fireplace_heat_weight'),
                "tv_heat_weight": ThermalParameterConfig.get_default('tv_heat_weight'),
                "source": "config_defaults",
                "calibration_date": None,
                "calibration_cycles": 0
            },
            "learning_state": {
                "cycle_count": 0,
                "learning_confidence": 3.0,
                "learning_enabled": True,
                "parameter_adjustments": {
                    "thermal_time_constant_delta": 0.0,
                    "heat_loss_coefficient_delta": 0.0,
                    "outlet_effectiveness_delta": 0.0
                },
                "parameter_bounds": {
                    "thermal_time_constant": list(ThermalParameterConfig.get_bounds('thermal_time_constant')),
                    "heat_loss_coefficient": list(ThermalParameterConfig.get_bounds('heat_loss_coefficient')),
                    "outlet_effectiveness": list(ThermalParameterConfig.get_bounds('outlet_effectiveness'))
                },
                "prediction_history": [],
                "parameter_history": []
            },
            "prediction_metrics": {
                "total_predictions": 0,
                "accuracy_stats": {
                    "mae_1h": 0.0,
                    "mae_6h": 0.0,
                    "mae_24h": 0.0,
                    "mae_all_time": 0.0,
                    "rmse_all_time": 0.0
                },
                "recent_performance": {
                    "last_10_mae": 0.0,
                    "last_10_max_error": 0.0
                }
            },
            "operational_state": {
                "last_indoor_temp": None,
                "last_outdoor_temp": None,
                "last_outlet_temp": None,
                "last_prediction": None,
                "last_run_time": None,
                "is_calibrating": False,
                # Added missing fields required by main.py learning logic
                "last_run_features": None,
                "last_final_temp": None,
                "last_avg_other_rooms_temp": None,
                "last_fireplace_on": False,
                "last_is_blocking": False,
                "last_blocking_reasons": [],
                "last_blocking_end_time": None
            }
        }
    
    def load_state(self) -> bool:
        """Load thermal state from JSON file."""
        try:
            with open(self.state_file, 'r') as f:
                loaded_state = json.load(f)
            
            # Validate and merge with default structure
            self.state = self._merge_with_defaults(loaded_state)
            self.state["metadata"]["last_updated"] = datetime.now().isoformat()
            
            logging.info(f"✅ Loaded unified thermal state from {self.state_file}")
            return True
            
        except FileNotFoundError:
            logging.info(f"📋 No existing thermal state file found at {self.state_file}")
            return False
        except Exception as e:
            logging.error(f"❌ Failed to load thermal state: {e}")
            self.state = self._get_default_state()
            return False
    
    def save_state(self) -> bool:
        """Save thermal state to JSON file."""
        try:
            # Update metadata
            self.state["metadata"]["last_updated"] = datetime.now().isoformat()
            
            # Convert numpy types to native Python for JSON serialization
            serializable_state = self._convert_numpy_types(self.state)
            
            with open(self.state_file, 'w') as f:
                json.dump(serializable_state, f, indent=2, default=str)
            
            logging.debug(f"💾 Saved unified thermal state to {self.state_file}")
            return True
            
        except Exception as e:
            logging.error(f"❌ Failed to save thermal state: {e}")
            return False
    
    def _merge_with_defaults(self, loaded_state: Dict) -> Dict:
        """Merge loaded state with default structure to handle schema changes."""
        merged = self._get_default_state()
        
        # Recursively merge loaded data
        def merge_dict(default: Dict, loaded: Dict) -> Dict:
            for key, value in loaded.items():
                if key in default:
                    if isinstance(default[key], dict) and isinstance(value, dict):
                        default[key] = merge_dict(default[key], value)
                    else:
                        default[key] = value
                else:
                    # Handle new keys from loaded state
                    default[key] = value
            return default
        
        return merge_dict(merged, loaded_state)
    
    def _convert_numpy_types(self, obj) -> Any:
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # === BASELINE PARAMETERS MANAGEMENT ===
    
    def set_calibrated_baseline(self, parameters: Dict[str, float], 
                               calibration_cycles: int = 0) -> None:
        """Set calibrated baseline parameters from calibration process."""
        baseline = self.state["baseline_parameters"]
        
        # Update thermal parameters
        if "thermal_time_constant" in parameters:
            baseline["thermal_time_constant"] = parameters["thermal_time_constant"]
        if "heat_loss_coefficient" in parameters:
            baseline["heat_loss_coefficient"] = parameters["heat_loss_coefficient"]
        if "outlet_effectiveness" in parameters:
            baseline["outlet_effectiveness"] = parameters["outlet_effectiveness"]
        
        # Update heat source weights if provided
        if "pv_heat_weight" in parameters:
            baseline["pv_heat_weight"] = parameters["pv_heat_weight"]
        if "fireplace_heat_weight" in parameters:
            baseline["fireplace_heat_weight"] = parameters["fireplace_heat_weight"]
        if "tv_heat_weight" in parameters:
            baseline["tv_heat_weight"] = parameters["tv_heat_weight"]
        
        # Update metadata
        baseline["source"] = "calibrated"
        baseline["calibration_date"] = datetime.now().isoformat()
        baseline["calibration_cycles"] = calibration_cycles
        
        # Reset learning adjustments when new baseline is set
        adjustments = self.state["learning_state"]["parameter_adjustments"]
        for key in adjustments:
            adjustments[key] = 0.0
        
        logging.info(f"🎯 Set calibrated baseline parameters (cycles: {calibration_cycles})")
        self.save_state()
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """Get complete thermal state structure including baseline_parameters section."""
        # Return the complete state structure that thermal_equilibrium_model expects
        return self.state.copy()
    
    def get_computed_parameters(self) -> Dict[str, float]:
        """Get current thermal parameters (baseline + learning adjustments)."""
        baseline = self.state["baseline_parameters"]
        adjustments = self.state["learning_state"]["parameter_adjustments"]
        
        return {
            "thermal_time_constant": baseline["thermal_time_constant"] + adjustments["thermal_time_constant_delta"],
            "heat_loss_coefficient": baseline["heat_loss_coefficient"] + adjustments["heat_loss_coefficient_delta"],
            "outlet_effectiveness": baseline["outlet_effectiveness"] + adjustments["outlet_effectiveness_delta"],
            "pv_heat_weight": baseline["pv_heat_weight"],
            "fireplace_heat_weight": baseline["fireplace_heat_weight"],
            "tv_heat_weight": baseline["tv_heat_weight"]
        }
    
    # === LEARNING STATE MANAGEMENT ===
    
    def update_learning_state(self, cycle_count: int = None, 
                            learning_confidence: float = None,
                            parameter_adjustments: Dict[str, float] = None) -> None:
        """Update learning state parameters."""
        learning_state = self.state["learning_state"]
        
        if cycle_count is not None:
            learning_state["cycle_count"] = cycle_count
        if learning_confidence is not None:
            learning_state["learning_confidence"] = learning_confidence
        if parameter_adjustments is not None:
            for param, delta in parameter_adjustments.items():
                if param in learning_state["parameter_adjustments"]:
                    learning_state["parameter_adjustments"][param] = delta
        
        self.save_state()
    
    def add_prediction_record(self, prediction_record: Dict) -> None:
        """Add a prediction record to history."""
        history = self.state["learning_state"]["prediction_history"]
        history.append(prediction_record)
        
        # Keep manageable history size
        if len(history) > 200:
            self.state["learning_state"]["prediction_history"] = history[-100:]
        
        # Update prediction count
        self.state["prediction_metrics"]["total_predictions"] += 1
    
    def add_parameter_history_record(self, parameter_record: Dict) -> None:
        """Add a parameter history record."""
        history = self.state["learning_state"]["parameter_history"]
        history.append(parameter_record)
        
        # Keep manageable history size
        if len(history) > 500:
            self.state["learning_state"]["parameter_history"] = history[-250:]
    
    # === OPERATIONAL STATE MANAGEMENT ===
    
    def update_operational_state(self, **kwargs) -> None:
        """Update operational state parameters."""
        operational = self.state["operational_state"]
        
        for key, value in kwargs.items():
            if key in operational:
                operational[key] = value
        
        operational["last_run_time"] = datetime.now().isoformat()
    
    def get_operational_state(self) -> Dict[str, Any]:
        """Get current operational state."""
        return self.state["operational_state"].copy()
    
    def set_calibration_mode(self, is_calibrating: bool) -> None:
        """Set calibration mode flag."""
        self.state["operational_state"]["is_calibrating"] = is_calibrating
        self.save_state()
    
    # === METRICS AND REPORTING ===
    
    def get_learning_metrics(self) -> Dict[str, Any]:
        """Get comprehensive learning metrics for monitoring."""
        learning_state = self.state["learning_state"]
        baseline = self.state["baseline_parameters"]
        metrics = self.state["prediction_metrics"]
        
        return {
            "baseline_source": baseline["source"],
            "calibration_date": baseline["calibration_date"],
            "calibration_cycles": baseline["calibration_cycles"],
            "current_cycle_count": learning_state["cycle_count"],
            "learning_confidence": learning_state["learning_confidence"],
            "total_predictions": metrics["total_predictions"],
            "current_parameters": self.get_current_parameters(),
            "parameter_adjustments": learning_state["parameter_adjustments"].copy(),
            "accuracy_stats": metrics["accuracy_stats"].copy(),
            "learning_enabled": learning_state["learning_enabled"]
        }
    
    def reset_learning_state(self, keep_baseline: bool = True) -> None:
        """Reset learning state while optionally preserving baseline."""
        if keep_baseline:
            # Reset only learning components
            self.state["learning_state"]["cycle_count"] = 0
            self.state["learning_state"]["learning_confidence"] = 3.0
            self.state["learning_state"]["prediction_history"] = []
            self.state["learning_state"]["parameter_history"] = []
            
            # Reset parameter adjustments
            adjustments = self.state["learning_state"]["parameter_adjustments"]
            for key in adjustments:
                adjustments[key] = 0.0
        else:
            # Reset everything including baseline
            self.state = self._get_default_state()
        
        logging.info(f"🔄 Reset learning state (baseline preserved: {keep_baseline})")
        self.save_state()
    
    # === BACKUP AND RESTORE FUNCTIONALITY ===
    
    def create_backup(self, backup_name: str = None) -> tuple[bool, str]:
        """Create a backup of the current thermal state."""
        try:
            if not backup_name:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_name = f'thermal_state_backup_{timestamp}'
            
            # Ensure backup has .json extension
            if not backup_name.endswith('.json'):
                backup_name += '.json'
            
            backup_path = os.path.dirname(self.state_file)
            backup_file = os.path.join(backup_path, backup_name)
            
            # Copy current state to backup file
            import shutil
            shutil.copy2(self.state_file, backup_file)
            
            logging.info(f"🔄 Created thermal state backup: {backup_name}")
            return True, backup_file
            
        except Exception as e:
            logging.error(f"❌ Failed to create thermal state backup: {e}")
            return False, str(e)
    
    def restore_from_backup(self, backup_file: str) -> tuple[bool, str]:
        """Restore thermal state from a backup file."""
        try:
            if not os.path.exists(backup_file):
                return False, f"Backup file not found: {backup_file}"
            
            # Create safety backup before restore
            safety_backup_name = f'pre_restore_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            self.create_backup(safety_backup_name)
            
            # Copy backup to current state file
            import shutil
            shutil.copy2(backup_file, self.state_file)
            
            # Reload state from restored file
            self.load_state()
            
            logging.info(f"🔄 Restored thermal state from backup: {os.path.basename(backup_file)}")
            return True, f"Thermal state restored successfully. Safety backup created as {safety_backup_name}.json"
            
        except Exception as e:
            logging.error(f"❌ Failed to restore from backup: {e}")
            return False, str(e)
    
    def list_backups(self) -> list[dict]:
        """List all available thermal state backups."""
        try:
            backup_path = os.path.dirname(self.state_file)
            backups = []
            
            # Look for backup files in same directory as state file
            for file in os.listdir(backup_path):
                if file.startswith('thermal_state_backup_') and file.endswith('.json'):
                    file_path = os.path.join(backup_path, file)
                    stat = os.stat(file_path)
                    
                    backups.append({
                        'name': file,
                        'path': file_path,
                        'size': stat.st_size,
                        'created': datetime.fromtimestamp(stat.st_mtime),
                        'type': 'thermal_state_backup'
                    })
            
            return sorted(backups, key=lambda x: x['created'], reverse=True)
            
        except Exception as e:
            logging.error(f"❌ Failed to list backups: {e}")
            return []


# Global instance for easy access
_thermal_state_manager = None

def get_thermal_state_manager() -> ThermalStateManager:
    """Get global thermal state manager instance."""
    global _thermal_state_manager
    if _thermal_state_manager is None:
        _thermal_state_manager = ThermalStateManager()
    return _thermal_state_manager


def save_thermal_state(**kwargs) -> None:
    """Convenience function to save thermal state."""
    manager = get_thermal_state_manager()
    manager.update_operational_state(**kwargs)
    manager.save_state()


def load_thermal_state() -> Dict[str, Any]:
    """Convenience function to load thermal state."""
    manager = get_thermal_state_manager()
    manager.load_state()
    return manager.state
