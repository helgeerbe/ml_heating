#!/usr/bin/env python3
"""
Test InfluxDB Export Functionality

Simple test script to verify that the adaptive learning metrics export
to InfluxDB is working correctly for Phase 2 Task 2.4.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.adaptive_learning_metrics_schema import (
    validate_metrics_data, 
    get_all_measurement_names,
    get_schema_summary
)
from src.influx_service import create_influx_service
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_schema_validation():
    """Test the metrics schema validation."""
    print("🔍 Testing adaptive learning metrics schema validation...")
    
    # Test valid prediction metrics data
    valid_prediction_data = {
        "mae_1h": 0.25,
        "mae_6h": 0.31,
        "mae_24h": 0.28,
        "rmse_1h": 0.33,
        "rmse_6h": 0.39,
        "rmse_24h": 0.35,
        "accuracy_excellent_pct": 78.5,
        "accuracy_very_good_pct": 85.2,
        "accuracy_good_pct": 92.1,
        "accuracy_acceptable_pct": 96.8,
        "mae_improvement_pct": 12.5,
        "is_improving": True,
        "total_predictions": 150,
        "predictions_24h": 288
    }
    
    is_valid = validate_metrics_data("ml_prediction_metrics", valid_prediction_data)
    print(f"   ✅ Prediction metrics validation: {'PASS' if is_valid else 'FAIL'}")
    
    # Test invalid data
    invalid_data = {
        "mae_1h": "not_a_float",
        "total_predictions": "not_an_int"
    }
    
    is_invalid = validate_metrics_data("ml_prediction_metrics", invalid_data)
    print(f"   ✅ Invalid data rejection: {'PASS' if not is_invalid else 'FAIL'}")
    
    return is_valid and not is_invalid

def test_influxdb_connection():
    """Test InfluxDB connection."""
    print("🔗 Testing InfluxDB connection...")
    
    try:
        influx_service = create_influx_service()
        print("   ✅ InfluxDB service created successfully")
        return True
    except Exception as e:
        print(f"   ❌ InfluxDB connection failed: {e}")
        return False

def test_prediction_metrics_export():
    """Test prediction metrics export."""
    print("📊 Testing prediction metrics export...")
    
    try:
        influx_service = create_influx_service()
        
        # Create test prediction metrics
        test_metrics = {
            "1h": {"mae": 0.22, "rmse": 0.28},
            "6h": {"mae": 0.29, "rmse": 0.35},
            "24h": {"mae": 0.31, "rmse": 0.37},
            "all": {"mae": 0.33, "rmse": 0.39, "count": 125},
            "accuracy_breakdown": {
                "excellent": {"percentage": 78.5},
                "very_good": {"percentage": 85.2},
                "good": {"percentage": 92.1},
                "acceptable": {"percentage": 96.8}
            },
            "trends": {
                "mae_improvement_percentage": 12.5,
                "is_improving": True,
                "insufficient_data": False
            }
        }
        
        # Export to InfluxDB
        influx_service.write_prediction_metrics(test_metrics)
        print("   ✅ Prediction metrics exported successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Prediction metrics export failed: {e}")
        return False

def test_thermal_learning_metrics_export():
    """Test thermal learning metrics export with mock thermal model."""
    print("🌡️  Testing thermal learning metrics export...")
    
    try:
        influx_service = create_influx_service()
        
        # Create mock thermal model
        class MockThermalModel:
            def __init__(self):
                self.outlet_effectiveness = 0.55
                self.heat_loss_coefficient = 0.045
                self.thermal_time_constant = 26.5
                self.learning_confidence = 3.8
            
            def get_adaptive_learning_metrics(self):
                return {
                    'current_parameters': {
                        'outlet_effectiveness': self.outlet_effectiveness,
                        'heat_loss_coefficient': self.heat_loss_coefficient,
                        'thermal_time_constant': self.thermal_time_constant
                    },
                    'learning_confidence': self.learning_confidence,
                    'current_learning_rate': 0.01,
                    'parameter_updates': 45,
                    'thermal_time_constant_stability': 0.89,
                    'heat_loss_coefficient_stability': 0.92,
                    'outlet_effectiveness_stability': 0.87,
                    'insufficient_data': False
                }
        
        mock_model = MockThermalModel()
        
        # Export to InfluxDB
        influx_service.write_thermal_learning_metrics(mock_model)
        print("   ✅ Thermal learning metrics exported successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Thermal learning metrics export failed: {e}")
        return False

def test_learning_phase_metrics_export():
    """Test learning phase metrics export."""
    print("📈 Testing learning phase metrics export...")
    
    try:
        influx_service = create_influx_service()
        
        # Create test learning phase data
        test_phase_data = {
            'current_learning_phase': 'high_confidence',
            'stability_score': 0.92,
            'learning_weight_applied': 1.0,
            'stable_period_duration_min': 45,
            'learning_updates_24h': {
                'high_confidence': 78,
                'low_confidence': 12,
                'skipped': 23
            },
            'learning_efficiency_pct': 87.5,
            'correction_stability': 0.89,
            'false_learning_prevention_pct': 94.2
        }
        
        # Export to InfluxDB
        influx_service.write_learning_phase_metrics(test_phase_data)
        print("   ✅ Learning phase metrics exported successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Learning phase metrics export failed: {e}")
        return False

def test_trajectory_prediction_metrics_export():
    """Test trajectory prediction metrics export."""
    print("🎯 Testing trajectory prediction metrics export...")
    
    try:
        influx_service = create_influx_service()
        
        # Create test trajectory data
        test_trajectory_data = {
            'prediction_horizon': '4h',
            'trajectory_accuracy': {
                'mae_1h': 0.25,
                'mae_2h': 0.32,
                'mae_4h': 0.48
            },
            'overshoot_prevention': {
                'overshoot_predicted': False,
                'prevented_24h': 3,
                'undershoot_prevented_24h': 1
            },
            'convergence': {
                'avg_time_minutes': 42.5,
                'accuracy_percentage': 89.2
            },
            'forecast_integration': {
                'weather_available': True,
                'pv_available': True,
                'quality_score': 0.87
            }
        }
        
        # Export to InfluxDB
        influx_service.write_trajectory_prediction_metrics(test_trajectory_data)
        print("   ✅ Trajectory prediction metrics exported successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Trajectory prediction metrics export failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing InfluxDB Export Functionality for Phase 2 Task 2.4")
    print("=" * 70)
    
    # Print schema summary
    print("\n📋 Adaptive Learning Metrics Schema:")
    print(get_schema_summary())
    
    # Run tests
    tests = [
        ("Schema Validation", test_schema_validation),
        ("InfluxDB Connection", test_influxdb_connection),
        ("Prediction Metrics Export", test_prediction_metrics_export),
        ("Thermal Learning Metrics Export", test_thermal_learning_metrics_export),
        ("Learning Phase Metrics Export", test_learning_phase_metrics_export),
        ("Trajectory Prediction Metrics Export", test_trajectory_prediction_metrics_export)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   💥 Test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! InfluxDB export functionality is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
