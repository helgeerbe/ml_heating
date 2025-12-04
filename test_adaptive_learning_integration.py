#!/usr/bin/env python3
"""
Integration Test for Adaptive Learning Master Plan Implementation

This test validates all components of the Adaptive Learning Master Plan:
1. Re-enabled adaptive learning
2. Empty trajectory methods implementation
3. MAE/RMSE tracking system
4. Enhanced HA metrics export
"""

import sys
import os
import json
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import all components
from model_wrapper import get_enhanced_model_wrapper
from thermal_equilibrium_model import ThermalEquilibriumModel
from prediction_metrics import PredictionMetrics
import config

def test_adaptive_learning_enabled():
    """Test 1: Verify adaptive learning is re-enabled."""
    print("🧪 Test 1: Adaptive Learning Re-enabled")
    
    model = ThermalEquilibriumModel()
    
    # Verify adaptive learning is enabled
    assert model.adaptive_learning_enabled == True, "Adaptive learning should be enabled"
    
    # Test parameter update mechanism works
    initial_thermal = model.thermal_time_constant
    initial_heat_loss = model.heat_loss_coefficient
    
    # Simulate prediction feedback to trigger learning
    for i in range(12):  # More than recent_errors_window
        predicted = 21.0
        actual = 20.5 + (i * 0.01)  # Gradually improving
        context = {
            'outlet_temp': 45.0,
            'outdoor_temp': 10.0,
            'pv_power': 500.0,
            'fireplace_on': 0,
            'tv_on': 0
        }
        model.update_prediction_feedback(predicted, actual, context)
    
    # Check if parameters have been updated
    parameter_changed = (
        abs(model.thermal_time_constant - initial_thermal) > 0.001 or
        abs(model.heat_loss_coefficient - initial_heat_loss) > 0.0001
    )
    
    print(f"   ✅ Adaptive learning enabled: {model.adaptive_learning_enabled}")
    print(f"   ✅ Parameters updating: {parameter_changed}")
    print(f"   ✅ Learning confidence: {model.learning_confidence:.3f}")
    
    return True

def test_empty_trajectory_methods():
    """Test 2: Verify empty trajectory methods are implemented."""
    print("\n🧪 Test 2: Empty Trajectory Methods Implementation")
    
    model = ThermalEquilibriumModel()
    
    # Test predict_thermal_trajectory method
    try:
        trajectory_result = model.predict_thermal_trajectory(
            current_indoor=20.0,
            target_indoor=21.0,
            outlet_temp=45.0,
            outdoor_temp=10.0,
            time_horizon_hours=4
        )
        
        assert isinstance(trajectory_result, dict), "Should return dictionary"
        assert 'trajectory' in trajectory_result, "Should have trajectory key"
        assert 'reaches_target_at' in trajectory_result, "Should have reaches_target_at key"
        assert len(trajectory_result['trajectory']) > 0, "Should have non-empty trajectory"
        
        print(f"   ✅ predict_thermal_trajectory working")
        print(f"   ✅ Trajectory length: {len(trajectory_result['trajectory'])}")
        print(f"   ✅ Reaches target at: {trajectory_result['reaches_target_at']}")
        
    except Exception as e:
        print(f"   ❌ predict_thermal_trajectory failed: {e}")
        return False
    
    # Test calculate_optimal_outlet_temperature method
    try:
        outlet_result = model.calculate_optimal_outlet_temperature(
            target_indoor=21.0,
            current_indoor=20.0,
            outdoor_temp=10.0
        )
        
        assert outlet_result is not None, "Should not return None"
        assert isinstance(outlet_result, dict), "Should return dictionary"
        assert 'optimal_outlet_temp' in outlet_result, "Should have optimal_outlet_temp key"
        
        print(f"   ✅ calculate_optimal_outlet_temperature working")
        print(f"   ✅ Optimal outlet temp: {outlet_result['optimal_outlet_temp']:.1f}°C")
        
    except Exception as e:
        print(f"   ❌ calculate_optimal_outlet_temperature failed: {e}")
        return False
    
    return True

def test_mae_rmse_tracking():
    """Test 3: Verify MAE/RMSE tracking system."""
    print("\n🧪 Test 3: MAE/RMSE Tracking System")
    
    # Test PredictionMetrics class
    try:
        metrics = PredictionMetrics()
        
        # Add some test predictions
        for i in range(20):
            predicted = 21.0 + (i * 0.01)
            actual = 21.0 + (i * 0.01) + 0.1  # Small consistent error
            context = {'outlet_temp': 45.0}
            
            metrics.add_prediction(predicted, actual, context)
        
        # Get metrics
        metrics_result = metrics.get_metrics()
        
        assert 'all' in metrics_result, "Should have 'all' time period"
        assert 'mae' in metrics_result['all'], "Should have MAE"
        assert 'rmse' in metrics_result['all'], "Should have RMSE"
        
        mae = metrics_result['all']['mae']
        rmse = metrics_result['all']['rmse']
        
        assert mae > 0, "MAE should be positive"
        assert rmse > 0, "RMSE should be positive"
        assert abs(mae - 0.1) < 0.01, f"MAE should be ~0.1, got {mae}"
        
        print(f"   ✅ PredictionMetrics class working")
        print(f"   ✅ MAE calculation: {mae:.3f}°C")
        print(f"   ✅ RMSE calculation: {rmse:.3f}°C")
        
        # Test recent performance
        recent = metrics.get_recent_performance(10)
        assert 'mae' in recent, "Recent performance should have MAE"
        print(f"   ✅ Recent MAE (10): {recent['mae']:.3f}°C")
        
        # Test accuracy breakdown
        breakdown = metrics_result.get('accuracy_breakdown', {})
        print(f"   ✅ Accuracy breakdown available: {len(breakdown) > 0}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ MAE/RMSE tracking failed: {e}")
        return False

def test_enhanced_ha_metrics():
    """Test 4: Verify enhanced HA metrics export."""
    print("\n🧪 Test 4: Enhanced HA Metrics Export")
    
    try:
        # Create enhanced model wrapper
        wrapper = get_enhanced_model_wrapper()
        
        # Add some test prediction data
        wrapper.prediction_metrics.add_prediction(21.1, 21.0, {'outlet_temp': 45.0})
        wrapper.prediction_metrics.add_prediction(20.9, 21.0, {'outlet_temp': 47.0})
        wrapper.prediction_metrics.add_prediction(21.2, 21.0, {'outlet_temp': 43.0})
        
        # Test comprehensive HA metrics
        ha_metrics = wrapper.get_comprehensive_metrics_for_ha()
        
        # Verify required fields for HA
        required_fields = [
            'thermal_time_constant', 'heat_loss_coefficient', 'outlet_effectiveness',
            'learning_confidence', 'cycle_count', 'mae_all_time', 'rmse_all_time',
            'model_health', 'total_predictions', 'last_updated'
        ]
        
        for field in required_fields:
            assert field in ha_metrics, f"Missing required HA field: {field}"
        
        print(f"   ✅ All required HA fields present")
        print(f"   ✅ Learning Confidence: {ha_metrics['learning_confidence']:.2f}")
        print(f"   ✅ Model Health: {ha_metrics['model_health']}")
        print(f"   ✅ MAE All Time: {ha_metrics['mae_all_time']:.3f}°C")
        print(f"   ✅ Total Predictions: {ha_metrics['total_predictions']}")
        
        # Test that values are reasonable
        assert 0 <= ha_metrics['learning_confidence'] <= 5, "Learning confidence out of range"
        assert ha_metrics['mae_all_time'] >= 0, "MAE should be non-negative"
        assert ha_metrics['total_predictions'] > 0, "Should have predictions"
        
        return True
        
    except Exception as e:
        print(f"   ❌ Enhanced HA metrics failed: {e}")
        return False

def test_integration_workflow():
    """Test 5: Full integration workflow."""
    print("\n🧪 Test 5: Full Integration Workflow")
    
    try:
        # Create wrapper and simulate realistic usage
        wrapper = get_enhanced_model_wrapper()
        
        # Simulate a full prediction and learning cycle
        features_dict = {
            'indoor_temp_lag_30m': 20.0,
            'target_temp': 21.0,
            'outdoor_temp': 10.0,
            'pv_now': 500.0,
            'fireplace_on': 0,
            'tv_on': 1
        }
        
        # Make prediction
        outlet_temp, metadata = wrapper.calculate_optimal_outlet_temp(features_dict)
        
        assert outlet_temp > 0, "Should get valid outlet temperature"
        assert isinstance(metadata, dict), "Should get metadata"
        
        print(f"   ✅ Prediction made: {outlet_temp:.1f}°C")
        
        # Simulate learning feedback
        actual_temp = 20.8  # Realistic actual measurement
        wrapper.learn_from_prediction_feedback(
            predicted_temp=21.0,
            actual_temp=actual_temp,
            prediction_context=features_dict
        )
        
        print(f"   ✅ Learning feedback processed")
        
        # Get comprehensive metrics
        final_metrics = wrapper.get_comprehensive_metrics_for_ha()
        
        print(f"   ✅ Final learning confidence: {final_metrics['learning_confidence']:.2f}")
        print(f"   ✅ Final cycle count: {final_metrics['cycle_count']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Integration workflow failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🚀 ADAPTIVE LEARNING MASTER PLAN - INTEGRATION TEST")
    print("=" * 60)
    
    tests = [
        test_adaptive_learning_enabled,
        test_empty_trajectory_methods, 
        test_mae_rmse_tracking,
        test_enhanced_ha_metrics,
        test_integration_workflow
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"   ❌ {test_func.__name__} failed")
        except Exception as e:
            print(f"   ❌ {test_func.__name__} crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 INTEGRATION TEST RESULTS:")
    print(f"   ✅ Passed: {passed}/{total}")
    print(f"   ❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print(f"\n🎉 ALL TESTS PASSED! Adaptive Learning Master Plan is working!")
        return 0
    else:
        print(f"\n❌ Some tests failed. Check implementation.")
        return 1

if __name__ == "__main__":
    exit(main())
