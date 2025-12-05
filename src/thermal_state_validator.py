"""
Thermal state JSON schema validation utility.

This module provides runtime validation for thermal state JSON files
to prevent silent failures during parameter loading.
"""

import json
import logging
from typing import Dict, Any, Optional

try:
    from jsonschema import validate, ValidationError
    SCHEMA_VALIDATION_AVAILABLE = True
except ImportError:
    logging.warning("jsonschema not installed - schema validation disabled")
    SCHEMA_VALIDATION_AVAILABLE = False


class ThermalStateValidationError(Exception):
    """Custom exception for thermal state validation errors."""
    pass


class ThermalStateValidator:
    """Runtime validator for thermal state JSON files."""
    
    @staticmethod
    def validate_thermal_state_data(data: Dict[str, Any], 
                                   strict: bool = False) -> bool:
        """
        Validate thermal state data against schema.
        
        Args:
            data: Thermal state dictionary to validate
            strict: If True, use full schema validation (requires jsonschema)
                   If False, use basic validation only
                   
        Returns:
            True if valid
            
        Raises:
            ThermalStateValidationError: If validation fails
        """
        
        # Basic validation - check required top-level sections
        required_sections = ["metadata", "baseline_parameters", 
                           "learning_state", "prediction_metrics", 
                           "operational_state"]
        
        for section in required_sections:
            if section not in data:
                raise ThermalStateValidationError(
                    f"Missing required section: {section}"
                )
        
        # Validate baseline_parameters
        baseline = data.get("baseline_parameters", {})
        required_params = ["thermal_time_constant", "heat_loss_coefficient",
                         "outlet_effectiveness", "pv_heat_weight", 
                         "fireplace_heat_weight", "tv_heat_weight", 
                         "source"]
        
        for param in required_params:
            if param not in baseline:
                raise ThermalStateValidationError(
                    f"Missing required parameter: baseline_parameters.{param}"
                )
        
        # Validate parameter ranges
        param_ranges = {
            "thermal_time_constant": (0.1, 24.0),
            "heat_loss_coefficient": (0.001, 1.0),
            "outlet_effectiveness": (0.001, 1.0),
            "pv_heat_weight": (0.0, 0.1),
            "fireplace_heat_weight": (0.0, 50.0),
            "tv_heat_weight": (0.0, 5.0)
        }
        
        for param, (min_val, max_val) in param_ranges.items():
            if param in baseline:
                value = baseline[param]
                if not isinstance(value, (int, float)):
                    raise ThermalStateValidationError(
                        f"Parameter {param} must be numeric, got {type(value)}"
                    )
                if not (min_val <= value <= max_val):
                    raise ThermalStateValidationError(
                        f"Parameter {param}={value} out of range "
                        f"[{min_val}, {max_val}]"
                    )
        
        # Validate source enum
        valid_sources = ["config", "calibrated", "adaptive"]
        source = baseline.get("source")
        if source not in valid_sources:
            raise ThermalStateValidationError(
                f"Invalid source '{source}', must be one of {valid_sources}"
            )
        
        # Strict validation with full schema (if available)
        if strict and SCHEMA_VALIDATION_AVAILABLE:
            from tests.test_thermal_state_schema_validation import ThermalStateSchema
            schema = ThermalStateSchema.get_unified_thermal_state_schema()
            try:
                validate(instance=data, schema=schema)
            except ValidationError as e:
                raise ThermalStateValidationError(f"Schema validation failed: {e}")
        
        logging.info("✅ Thermal state validation passed")
        return True
    
    @staticmethod
    def validate_file(file_path: str, strict: bool = False) -> bool:
        """
        Validate a thermal state JSON file.
        
        Args:
            file_path: Path to the JSON file to validate
            strict: If True, use full schema validation
                   
        Returns:
            True if valid
            
        Raises:
            ThermalStateValidationError: If validation fails
            FileNotFoundError: If file doesn't exist
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise ThermalStateValidationError(
                f"Thermal state file not found: {file_path}"
            )
        except json.JSONDecodeError as e:
            raise ThermalStateValidationError(
                f"Invalid JSON in {file_path}: {e}"
            )
        
        return ThermalStateValidator.validate_thermal_state_data(data, strict)


def validate_thermal_state_safely(data: Dict[str, Any]) -> bool:
    """
    Safe wrapper for thermal state validation that doesn't raise exceptions.
    
    Args:
        data: Thermal state data to validate
        
    Returns:
        True if valid, False if invalid (logs errors)
    """
    try:
        return ThermalStateValidator.validate_thermal_state_data(data)
    except ThermalStateValidationError as e:
        logging.error(f"❌ Thermal state validation failed: {e}")
        return False
    except Exception as e:
        logging.error(f"❌ Unexpected validation error: {e}")
        return False


if __name__ == "__main__":
    # Test with current thermal_state.json
    import sys
    import os
    
    thermal_state_path = "/opt/ml_heating/thermal_state.json"
    if os.path.exists(thermal_state_path):
        try:
            ThermalStateValidator.validate_file(thermal_state_path, strict=False)
            print("✅ thermal_state.json validation passed!")
        except ThermalStateValidationError as e:
            print(f"❌ Validation failed: {e}")
    else:
        print("❌ thermal_state.json not found")
