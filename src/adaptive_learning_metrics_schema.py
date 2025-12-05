"""
Adaptive Learning Metrics Schema Definition

This module defines the InfluxDB schema for exporting comprehensive
adaptive learning metrics from the ML Heating System.
"""

from typing import Dict, List, Optional
from datetime import datetime, timezone
import logging

# Schema definitions for InfluxDB measurements
ADAPTIVE_LEARNING_SCHEMAS = {
    # Core prediction accuracy metrics
    "ml_prediction_metrics": {
        "measurement": "ml_prediction_metrics",
        "tags": {
            "source": "ml_heating",
            "version": "2.0"
        },
        "fields": {
            # MAE metrics for different time windows
            "mae_1h": "float",
            "mae_6h": "float", 
            "mae_24h": "float",
            
            # RMSE metrics for different time windows
            "rmse_1h": "float",
            "rmse_6h": "float",
            "rmse_24h": "float",
            
            # Prediction accuracy percentages
            "accuracy_excellent_pct": "float",  # Within ±0.1°C
            "accuracy_very_good_pct": "float",  # Within ±0.2°C
            "accuracy_good_pct": "float",       # Within ±0.5°C
            "accuracy_acceptable_pct": "float", # Within ±1.0°C
            
            # Trend analysis
            "mae_improvement_pct": "float",
            "is_improving": "boolean",
            
            # Counts
            "total_predictions": "int",
            "predictions_24h": "int"
        }
    },
    
    # Learning phase classification and adaptive learning status
    "ml_learning_phase": {
        "measurement": "ml_learning_phase", 
        "tags": {
            "source": "ml_heating",
            "learning_phase": "string"  # high_confidence, low_confidence, skip
        },
        "fields": {
            # Current learning state
            "current_learning_phase": "string",
            "stability_score": "float",
            "learning_weight_applied": "float",
            "stable_period_duration_min": "int",
            
            # Learning distribution (24h counts)
            "high_confidence_updates_24h": "int",
            "low_confidence_updates_24h": "int", 
            "skipped_updates_24h": "int",
            
            # Learning effectiveness
            "learning_efficiency_pct": "float",
            "correction_stability": "float",
            "false_learning_prevention_pct": "float"
        }
    },
    
    # Thermal model parameters and learning progress
    "ml_thermal_parameters": {
        "measurement": "ml_thermal_parameters",
        "tags": {
            "source": "ml_heating",
            "parameter_type": "string"  # baseline, current, correction
        },
        "fields": {
            # Core thermal parameters
            "outlet_effectiveness": "float",
            "heat_loss_coefficient": "float", 
            "thermal_time_constant": "float",
            
            # Parameter corrections (online learning)
            "outlet_effectiveness_correction_pct": "float",
            "heat_loss_correction_pct": "float",
            "time_constant_correction_pct": "float",
            
            # Learning metadata
            "learning_confidence": "float",
            "current_learning_rate": "float",
            "parameter_updates_total": "int",
            "parameter_updates_24h": "int"
        }
    },
    
    # Trajectory prediction accuracy and overshoot prevention
    "ml_trajectory_prediction": {
        "measurement": "ml_trajectory_prediction",
        "tags": {
            "source": "ml_heating",
            "prediction_horizon": "string"  # 1h, 2h, 4h
        },
        "fields": {
            # Trajectory accuracy by horizon
            "trajectory_mae_1h": "float",
            "trajectory_mae_2h": "float", 
            "trajectory_mae_4h": "float",
            
            # Overshoot prevention metrics
            "overshoot_predicted": "boolean",
            "overshoot_prevented_24h": "int",
            "undershoot_prevented_24h": "int",
            
            # Convergence analysis
            "convergence_time_avg_min": "float",
            "convergence_accuracy_pct": "float",
            
            # Forecast integration quality
            "weather_forecast_available": "boolean",
            "pv_forecast_available": "boolean",
            "forecast_integration_quality": "float"
        }
    }
}

def get_schema_for_measurement(measurement_name: str) -> Optional[Dict]:
    """
    Get the schema definition for a specific measurement.
    
    Args:
        measurement_name: Name of the InfluxDB measurement
        
    Returns:
        Schema dictionary or None if not found
    """
    return ADAPTIVE_LEARNING_SCHEMAS.get(measurement_name)

def validate_metrics_data(measurement_name: str, data: Dict) -> bool:
    """
    Validate data against the schema for a measurement.
    
    Args:
        measurement_name: Name of the measurement
        data: Data dictionary to validate
        
    Returns:
        True if data is valid, False otherwise
    """
    schema = get_schema_for_measurement(measurement_name)
    if not schema:
        logging.warning(f"No schema found for measurement: {measurement_name}")
        return False
    
    expected_fields = schema.get("fields", {})
    
    # Check that all required numeric fields are present and valid
    for field_name, field_type in expected_fields.items():
        if field_name not in data:
            continue  # Optional field
            
        value = data[field_name]
        
        # Type validation
        if field_type == "float":
            try:
                float(value)
            except (ValueError, TypeError):
                logging.warning(f"Invalid float value for {field_name}: {value}")
                return False
                
        elif field_type == "int":
            try:
                int(value)
            except (ValueError, TypeError):
                logging.warning(f"Invalid int value for {field_name}: {value}")
                return False
                
        elif field_type == "boolean":
            if not isinstance(value, bool):
                logging.warning(f"Invalid boolean value for {field_name}: {value}")
                return False
    
    return True

def get_all_measurement_names() -> List[str]:
    """Get list of all available measurement names."""
    return list(ADAPTIVE_LEARNING_SCHEMAS.keys())

def get_schema_summary() -> str:
    """Get a human-readable summary of all schemas."""
    summary = "Adaptive Learning Metrics Schema Summary:\n\n"
    
    for measurement_name, schema in ADAPTIVE_LEARNING_SCHEMAS.items():
        field_count = len(schema.get("fields", {}))
        tag_count = len(schema.get("tags", {}))
        
        summary += f"📊 {measurement_name}:\n"
        summary += f"   - Fields: {field_count}\n"
        summary += f"   - Tags: {tag_count}\n"
        summary += f"   - Description: {schema.get('description', 'Core adaptive learning metrics')}\n\n"
    
    return summary


# Example usage and testing
if __name__ == "__main__":
    print("🗄️  Testing Adaptive Learning Metrics Schema")
    
    # Test schema retrieval
    schema = get_schema_for_measurement("ml_prediction_metrics")
    print(f"\n📋 ml_prediction_metrics schema:")
    print(f"   Fields: {len(schema['fields'])}")
    print(f"   Sample fields: {list(schema['fields'].keys())[:5]}")
    
    # Test data validation
    test_data = {
        "mae_1h": 0.25,
        "mae_24h": 0.31,
        "rmse_1h": 0.33,
        "accuracy_excellent_pct": 78.5,
        "total_predictions": 150,
        "is_improving": True
    }
    
    is_valid = validate_metrics_data("ml_prediction_metrics", test_data)
    print(f"\n✅ Test data validation: {'PASS' if is_valid else 'FAIL'}")
    
    # Test invalid data
    invalid_data = {
        "mae_1h": "invalid_float",
        "total_predictions": "not_an_int"
    }
    
    is_invalid = validate_metrics_data("ml_prediction_metrics", invalid_data)
    print(f"❌ Invalid data validation: {'PASS' if not is_invalid else 'FAIL'}")
    
    # Print schema summary
    print(f"\n{get_schema_summary()}")
    
    print("✅ Schema definition ready for Phase 2 Task 2.4!")
