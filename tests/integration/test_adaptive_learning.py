#!/usr/bin/env python3
"""
Integration Test for Adaptive Learning Master Plan Implementation

This test validates all components of the Adaptive Learning Master Plan:
1. Re-enabled adaptive learning
2. Empty trajectory methods implementation
3. MAE/RMSE tracking system
4. Enhanced HA metrics export
"""

from src.model_wrapper import get_enhanced_model_wrapper
from src.thermal_equilibrium_model import ThermalEquilibriumModel
from src.prediction_metrics import PredictionMetrics


def test_adaptive_learning_with_small_errors():
    """Test adaptive learning with small, consistent errors."""
    print("\n\U0001f9ea Test: Adaptive Learning with Small Errors")
    model = ThermalEquilibriumModel()
    initial_confidence = model.learning_confidence
    
    for _ in range(10):
        model.update_prediction_feedback(
            21.0, 21.1, {'outlet_temp': 45.0, 'current_indoor': 21.05}
        )

    assert model.learning_confidence > initial_confidence
    print(f"   \u2705 Confidence increased: {model.learning_confidence:.3f}")


def test_adaptive_learning_with_large_errors():
    """Test adaptive learning with large, corrective errors."""
    print("\n\U0001f9ea Test: Adaptive Learning with Large Errors")
    model = ThermalEquilibriumModel()
    initial_thermal = model.thermal_time_constant
    initial_heat_loss = model.heat_loss_coefficient
    initial_confidence = model.learning_confidence

    for i in range(10):  # Corrective learning
        model.update_prediction_feedback(
            22.0,
            20.0,
            {
                'outlet_temp': 50.0 - i,
                'outdoor_temp': 5.0,
                'current_indoor': 20.0,
                'target_temp': 22.0,
            },
        )

    parameter_changed = (
        abs(model.thermal_time_constant - initial_thermal) > 0.001
        or abs(model.heat_loss_coefficient - initial_heat_loss) > 0.0001
    )
    
    assert parameter_changed, "Parameters should update with large errors"
    assert model.learning_confidence < initial_confidence
    print("   \u2705 Parameters updated with large errors")
    print(f"   \u2705 Confidence decreased: {model.learning_confidence:.3f}")


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
        
        print("   ✅ PredictionMetrics class working")
        print(f"   ✅ MAE calculation: {mae:.3f}°C")
        print(f"   ✅ RMSE calculation: {rmse:.3f}°C")
        
        # Test recent performance
        recent = metrics.get_recent_performance(10)
        assert 'mae' in recent, "Recent performance should have MAE"
        print(f"   ✅ Recent MAE (10): {recent['mae']:.3f}°C")
        
        # Test accuracy breakdown
        breakdown = metrics_result.get('accuracy_breakdown', {})
        print(f"   ✅ Accuracy breakdown available: {bool(breakdown)}")

    except Exception as e:
        print(f"   ❌ MAE/RMSE tracking failed: {e}")
        assert False, f"MAE/RMSE tracking failed: {e}"


def test_enhanced_ha_metrics():
    """Test 4: Verify enhanced HA metrics export."""
    print("\n🧪 Test 4: Enhanced HA Metrics Export")
    
    try:
        # Create enhanced model wrapper
        wrapper = get_enhanced_model_wrapper()
        
        # Add some test prediction data
        wrapper.prediction_metrics.add_prediction(
            21.1, 21.0, {'outlet_temp': 45.0}
        )
        wrapper.prediction_metrics.add_prediction(
            20.9, 21.0, {'outlet_temp': 47.0}
        )
        wrapper.prediction_metrics.add_prediction(
            21.2, 21.0, {'outlet_temp': 43.0}
        )
        
        # Test comprehensive HA metrics
        ha_metrics = wrapper.get_comprehensive_metrics_for_ha()
        
        required_fields = [
            'thermal_time_constant',
            'heat_loss_coefficient',
            'outlet_effectiveness',
            'learning_confidence',
            'cycle_count',
            'mae_all_time',
            'rmse_all_time',
            'model_health',
            'total_predictions',
            'last_updated',
        ]
        
        for field in required_fields:
            assert field in ha_metrics, f"Missing required HA field: {field}"
        
        print("   ✅ All required HA fields present")
        learning_confidence = ha_metrics['learning_confidence']
        model_health = ha_metrics['model_health']
        mae_all_time = ha_metrics['mae_all_time']
        total_predictions = ha_metrics['total_predictions']
        print(f"   ✅ Learning Confidence: {learning_confidence:.2f}")
        print(f"   ✅ Model Health: {model_health}")
        print(f"   ✅ MAE (all): {mae_all_time:.3f}°C")
        print(f"   ✅ Total Predictions: {total_predictions}")
        
        # Test that values are reasonable
        assert 0 <= ha_metrics['learning_confidence'] <= 5, "Confidence range"
        assert ha_metrics['mae_all_time'] >= 0, "MAE non-negative"
        assert ha_metrics['total_predictions'] > 0, "Should have predictions"

    except Exception as e:
        print(f"   ❌ Enhanced HA metrics failed: {e}")
        assert False, f"Enhanced HA metrics failed: {e}"


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
        outlet_temp, metadata = wrapper.calculate_optimal_outlet_temp(
            features_dict
        )
        
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
        
        print("   ✅ Learning feedback processed")
        
        # Get comprehensive metrics
        final_metrics = wrapper.get_comprehensive_metrics_for_ha()
        
        final_confidence = final_metrics['learning_confidence']
        final_cycle_count = final_metrics['cycle_count']
        print(f"   ✅ Final confidence: {final_confidence:.2f}")
        print(f"   ✅ Final cycle count: {final_cycle_count}")

    except Exception as e:
        print(f"   ❌ Integration workflow failed: {e}")
        assert False, f"Integration failed: {e}"
