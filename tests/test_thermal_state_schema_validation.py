"""
Schema validation tests for thermal state JSON file.

This module provides comprehensive schema validation for the unified thermal state
JSON format to prevent silent failures during reading and writing operations.
"""

import json
import pytest
from typing import Dict, Any
from jsonschema import validate, ValidationError
import tempfile
import os
from pathlib import Path


class ThermalStateSchema:
    """Schema definitions for thermal state JSON validation."""
    
    @staticmethod
    def get_unified_thermal_state_schema() -> Dict[str, Any]:
        """
        Get the JSON schema for unified thermal state format.
        
        This schema validates the structure created by the calibration system
        and expected by the thermal model parameter loading.
        """
        return {
            "type": "object",
            "required": ["metadata", "baseline_parameters", "learning_state", 
                        "prediction_metrics", "operational_state"],
            "properties": {
                "metadata": {
                    "type": "object",
                    "required": ["version", "format", "created", "last_updated"],
                    "properties": {
                        "version": {"type": "string"},
                        "format": {"type": "string", "enum": ["unified_thermal_state"]},
                        "created": {"type": "string", "format": "date-time"},
                        "last_updated": {"type": "string", "format": "date-time"}
                    },
                    "additionalProperties": False
                },
                "baseline_parameters": {
                    "type": "object",
                    "required": ["thermal_time_constant", "heat_loss_coefficient", 
                               "outlet_effectiveness", "pv_heat_weight", 
                               "fireplace_heat_weight", "tv_heat_weight", 
                               "source"],
                    "properties": {
                        "thermal_time_constant": {"type": "number", "minimum": 0.1, "maximum": 24.0},
                        "heat_loss_coefficient": {"type": "number", "minimum": 0.001, "maximum": 1.0},
                        "outlet_effectiveness": {"type": "number", "minimum": 0.001, "maximum": 1.0},
                        "pv_heat_weight": {"type": "number", "minimum": 0.0, "maximum": 0.1},
                        "fireplace_heat_weight": {"type": "number", "minimum": 0.0, "maximum": 50.0},
                        "tv_heat_weight": {"type": "number", "minimum": 0.0, "maximum": 5.0},
                        "source": {"type": "string", "enum": ["config", "calibrated", "adaptive"]},
                        "calibration_date": {"type": "string", "format": "date-time"},
                        "calibration_cycles": {"type": "integer", "minimum": 0}
                    },
                    "additionalProperties": False
                },
                "learning_state": {
                    "type": "object",
                    "required": ["cycle_count", "learning_confidence", "learning_enabled",
                               "parameter_adjustments", "parameter_bounds"],
                    "properties": {
                        "cycle_count": {"type": "integer", "minimum": 0},
                        "learning_confidence": {"type": "number", "minimum": 0.1, "maximum": 10.0},
                        "learning_enabled": {"type": "boolean"},
                        "parameter_adjustments": {
                            "type": "object",
                            "required": ["thermal_time_constant_delta", "heat_loss_coefficient_delta", 
                                       "outlet_effectiveness_delta"],
                            "properties": {
                                "thermal_time_constant_delta": {"type": "number"},
                                "heat_loss_coefficient_delta": {"type": "number"},
                                "outlet_effectiveness_delta": {"type": "number"}
                            },
                            "additionalProperties": False
                        },
                        "parameter_bounds": {
                            "type": "object",
                            "required": ["thermal_time_constant", "heat_loss_coefficient", 
                                       "outlet_effectiveness"],
                            "properties": {
                                "thermal_time_constant": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 2,
                                    "maxItems": 2
                                },
                                "heat_loss_coefficient": {
                                    "type": "array", 
                                    "items": {"type": "number"},
                                    "minItems": 2,
                                    "maxItems": 2
                                },
                                "outlet_effectiveness": {
                                    "type": "array",
                                    "items": {"type": "number"}, 
                                    "minItems": 2,
                                    "maxItems": 2
                                }
                            },
                            "additionalProperties": False
                        },
                        "prediction_history": {"type": "array"},
                        "parameter_history": {"type": "array"}
                    },
                    "additionalProperties": False
                },
                "prediction_metrics": {
                    "type": "object",
                    "required": ["total_predictions", "accuracy_stats", "recent_performance"],
                    "properties": {
                        "total_predictions": {"type": "integer", "minimum": 0},
                        "accuracy_stats": {
                            "type": "object",
                            "required": ["mae_1h", "mae_6h", "mae_24h", "mae_all_time", "rmse_all_time"],
                            "properties": {
                                "mae_1h": {"type": "number", "minimum": 0.0},
                                "mae_6h": {"type": "number", "minimum": 0.0},
                                "mae_24h": {"type": "number", "minimum": 0.0},
                                "mae_all_time": {"type": "number", "minimum": 0.0},
                                "rmse_all_time": {"type": "number", "minimum": 0.0}
                            },
                            "additionalProperties": False
                        },
                        "recent_performance": {
                            "type": "object",
                            "required": ["last_10_mae", "last_10_max_error"],
                            "properties": {
                                "last_10_mae": {"type": "number", "minimum": 0.0},
                                "last_10_max_error": {"type": "number", "minimum": 0.0}
                            },
                            "additionalProperties": False
                        }
                    },
                    "additionalProperties": False
                },
                "operational_state": {
                    "type": "object",
                    "required": ["last_run_time", "is_calibrating"],
                    "properties": {
                        "last_indoor_temp": {"type": ["number", "null"]},
                        "last_outdoor_temp": {"type": ["number", "null"]},
                        "last_outlet_temp": {"type": ["number", "null"]},
                        "last_prediction": {"type": ["number", "null"]},
                        "last_run_time": {"type": "string", "format": "date-time"},
                        "is_calibrating": {"type": "boolean"},
                        "last_run_features": {"type": ["string", "null"]},
                        "last_final_temp": {"type": ["number", "null"]},
                        "last_avg_other_rooms_temp": {"type": ["number", "null"]},
                        "last_fireplace_on": {"type": ["number", "boolean", "null"]},
                        "last_is_blocking": {"type": "boolean"},
                        "last_blocking_reasons": {"type": "array"},
                        "last_blocking_end_time": {"type": ["string", "null"]}
                    },
                    "additionalProperties": False
                }
            },
            "additionalProperties": False
        }


class TestThermalStateSchemaValidation:
    """Test cases for thermal state JSON schema validation."""

    def test_valid_thermal_state_passes_validation(self):
        """Test that a valid thermal state JSON passes schema validation."""
        valid_thermal_state = {
            "metadata": {
                "version": "1.0",
                "format": "unified_thermal_state", 
                "created": "2025-12-05T13:09:17.041092",
                "last_updated": "2025-12-05T13:09:42.259879"
            },
            "baseline_parameters": {
                "thermal_time_constant": 4.0,
                "heat_loss_coefficient": 0.12860044414752056,
                "outlet_effectiveness": 0.08528937343235657,
                "pv_heat_weight": 0.0005,
                "fireplace_heat_weight": 5.0,
                "tv_heat_weight": 0.18166179600343998,
                "source": "calibrated",
                "calibration_date": "2025-12-05T13:09:17.375559",
                "calibration_cycles": 3704
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
                    "thermal_time_constant": [3.0, 8.0],
                    "heat_loss_coefficient": [0.002, 0.25],
                    "outlet_effectiveness": [0.01, 0.5]
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
                "last_indoor_temp": 20.6,
                "last_outdoor_temp": None,
                "last_outlet_temp": None,
                "last_prediction": None,
                "last_run_time": "2025-12-05T13:09:42.259874",
                "is_calibrating": False,
                "last_run_features": "   outlet_temp  ...  combined_forecast_thermal_load\n0         46.0  ...                            1.58\n\n[1 rows x 37 columns]",
                "last_final_temp": 41.0,
                "last_avg_other_rooms_temp": 20.4,
                "last_fireplace_on": None,
                "last_is_blocking": False,
                "last_blocking_reasons": [],
                "last_blocking_end_time": None
            }
        }
        
        schema = ThermalStateSchema.get_unified_thermal_state_schema()
        
        # Should not raise any exception
        validate(instance=valid_thermal_state, schema=schema)

    def test_missing_required_section_fails_validation(self):
        """Test that missing required sections fail validation."""
        invalid_thermal_state = {
            "metadata": {
                "version": "1.0",
                "format": "unified_thermal_state",
                "created": "2025-12-05T13:09:17.041092", 
                "last_updated": "2025-12-05T13:09:42.259879"
            }
            # Missing baseline_parameters, learning_state, etc.
        }
        
        schema = ThermalStateSchema.get_unified_thermal_state_schema()
        
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=invalid_thermal_state, schema=schema)
        
        assert "baseline_parameters" in str(exc_info.value)

    def test_invalid_parameter_source_fails_validation(self):
        """Test that invalid parameter source values fail validation."""
        invalid_thermal_state = {
            "metadata": {
                "version": "1.0", 
                "format": "unified_thermal_state",
                "created": "2025-12-05T13:09:17.041092",
                "last_updated": "2025-12-05T13:09:42.259879"
            },
            "baseline_parameters": {
                "thermal_time_constant": 4.0,
                "heat_loss_coefficient": 0.1286,
                "outlet_effectiveness": 0.0853,
                "pv_heat_weight": 0.0005,
                "fireplace_heat_weight": 5.0,
                "tv_heat_weight": 0.18,
                "source": "invalid_source"  # Invalid enum value
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
                    "thermal_time_constant": [3.0, 8.0],
                    "heat_loss_coefficient": [0.002, 0.25],
                    "outlet_effectiveness": [0.01, 0.5]
                }
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
                "last_run_time": "2025-12-05T13:09:42.259874",
                "is_calibrating": False
            }
        }
        
        schema = ThermalStateSchema.get_unified_thermal_state_schema()
        
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=invalid_thermal_state, schema=schema)
        
        assert "invalid_source" in str(exc_info.value)

    def test_out_of_range_parameters_fail_validation(self):
        """Test that parameters outside valid ranges fail validation."""
        invalid_thermal_state = {
            "metadata": {
                "version": "1.0",
                "format": "unified_thermal_state", 
                "created": "2025-12-05T13:09:17.041092",
                "last_updated": "2025-12-05T13:09:42.259879"
            },
            "baseline_parameters": {
                "thermal_time_constant": -1.0,  # Invalid: negative
                "heat_loss_coefficient": 2.0,   # Invalid: > 1.0
                "outlet_effectiveness": 0.0853,
                "pv_heat_weight": 0.0005,
                "fireplace_heat_weight": 5.0,
                "tv_heat_weight": 0.18,
                "source": "calibrated"
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
                    "thermal_time_constant": [3.0, 8.0],
                    "heat_loss_coefficient": [0.002, 0.25],
                    "outlet_effectiveness": [0.01, 0.5]
                }
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
                "last_run_time": "2025-12-05T13:09:42.259874",
                "is_calibrating": False
            }
        }
        
        schema = ThermalStateSchema.get_unified_thermal_state_schema()
        
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=invalid_thermal_state, schema=schema)
        
        # Should fail on negative thermal_time_constant
        assert "-1.0" in str(exc_info.value) or "minimum" in str(exc_info.value)

    def test_file_read_write_with_validation(self):
        """Test reading and writing thermal state files with schema validation."""
        valid_thermal_state = {
            "metadata": {
                "version": "1.0",
                "format": "unified_thermal_state",
                "created": "2025-12-05T13:09:17.041092",
                "last_updated": "2025-12-05T13:09:42.259879"
            },
            "baseline_parameters": {
                "thermal_time_constant": 4.0,
                "heat_loss_coefficient": 0.1286,
                "outlet_effectiveness": 0.0853,
                "pv_heat_weight": 0.0005,
                "fireplace_heat_weight": 5.0,
                "tv_heat_weight": 0.18,
                "source": "calibrated"
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
                    "thermal_time_constant": [3.0, 8.0],
                    "heat_loss_coefficient": [0.002, 0.25],
                    "outlet_effectiveness": [0.01, 0.5]
                }
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
                "last_run_time": "2025-12-05T13:09:42.259874",
                "is_calibrating": False
            }
        }
        
        schema = ThermalStateSchema.get_unified_thermal_state_schema()
        
        # Test writing to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(valid_thermal_state, f, indent=2)
            temp_file_path = f.name
        
        try:
            # Test reading from file and validating
            with open(temp_file_path, 'r') as f:
                loaded_data = json.load(f)
            
            # Should not raise any exception
            validate(instance=loaded_data, schema=schema)
            
            # Verify data integrity
            assert loaded_data["baseline_parameters"]["source"] == "calibrated"
            assert loaded_data["baseline_parameters"]["heat_loss_coefficient"] == 0.1286
            
        finally:
            # Clean up
            os.unlink(temp_file_path)

    def test_corrupted_json_fails_gracefully(self):
        """Test that corrupted JSON files fail with clear error messages."""
        corrupted_json = '{"metadata": {"version": "1.0", "format": "unified_thermal_state"'  # Missing closing braces
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(corrupted_json)
            temp_file_path = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                with open(temp_file_path, 'r') as f:
                    json.load(f)
        finally:
            os.unlink(temp_file_path)

    def test_calibrated_vs_config_source_detection(self):
        """Test that the schema correctly distinguishes between calibrated and config sources."""
        calibrated_state = {
            "metadata": {
                "version": "1.0",
                "format": "unified_thermal_state",
                "created": "2025-12-05T13:09:17.041092",
                "last_updated": "2025-12-05T13:09:42.259879"
            },
            "baseline_parameters": {
                "thermal_time_constant": 4.0,
                "heat_loss_coefficient": 0.1286,
                "outlet_effectiveness": 0.0853,
                "pv_heat_weight": 0.0005,
                "fireplace_heat_weight": 5.0,
                "tv_heat_weight": 0.18,
                "source": "calibrated"  # This should pass validation
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
                    "thermal_time_constant": [3.0, 8.0],
                    "heat_loss_coefficient": [0.002, 0.25],
                    "outlet_effectiveness": [0.01, 0.5]
                }
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
                "last_run_time": "2025-12-05T13:09:42.259874",
                "is_calibrating": False
            }
        }
        
        schema = ThermalStateSchema.get_unified_thermal_state_schema()
        
        # Should pass validation for calibrated source
        validate(instance=calibrated_state, schema=schema)
        
        # Test config source
        calibrated_state["baseline_parameters"]["source"] = "config"
        validate(instance=calibrated_state, schema=schema)
        
        # Test adaptive source  
        calibrated_state["baseline_parameters"]["source"] = "adaptive"
        validate(instance=calibrated_state, schema=schema)


def validate_thermal_state_file(file_path: str) -> bool:
    """
    Utility function to validate a thermal state JSON file.
    
    Args:
        file_path: Path to the JSON file to validate
        
    Returns:
        True if valid, raises ValidationError if invalid
        
    Raises:
        FileNotFoundError: If file doesn't exist
        JSONDecodeError: If file contains invalid JSON
        ValidationError: If JSON doesn't match schema
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Thermal state file not found: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in thermal state file {file_path}: {e}")
    
    schema = ThermalStateSchema.get_unified_thermal_state_schema()
    validate(instance=data, schema=schema)
    
    return True


if __name__ == "__main__":
    # Example usage for testing current thermal_state.json
    try:
        validate_thermal_state_file("/opt/ml_heating/thermal_state.json")
        print("✅ thermal_state.json is valid!")
    except Exception as e:
        print(f"❌ thermal_state.json validation failed: {e}")
