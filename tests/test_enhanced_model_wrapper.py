#!/usr/bin/env python3
"""
Quick test for the Enhanced Model Wrapper to validate integration.
"""
import sys
import os

# Add src to path for testing
sys.path.insert(0, 'src')

from model_wrapper import EnhancedModelWrapper

def test_enhanced_wrapper():
    """Test basic functionality of the enhanced model wrapper."""
    
    print("🧪 Testing Enhanced Model Wrapper...")
    
    # Test initialization
    try:
        wrapper = EnhancedModelWrapper()
        print("✅ Initialization successful")
        
        # Test basic prediction with minimal features
        test_features = {
            'indoor_temp_lag_30m': 20.5,
            'target_temp': 21.0,
            'outdoor_temp': 5.0,
            'pv_now': 2500.0,
            'fireplace_on': 0,
            'tv_on': 1,
            'hour_sin': 0.5,
            'hour_cos': 0.866,
            'month_sin': -0.5,
            'month_cos': 0.866,
            'temp_diff_indoor_outdoor': 15.5,
            'indoor_temp_gradient': 0.02
        }
        
        optimal_temp, metadata = wrapper.calculate_optimal_outlet_temp(test_features)
        
        print(f"✅ Prediction successful:")
        print(f"   - Optimal outlet temp: {optimal_temp:.1f}°C")
        print(f"   - Confidence: {metadata['learning_confidence']:.3f}")
        print(f"   - Method: {metadata['prediction_method']}")
        
        # Test learning feedback
        wrapper.learn_from_prediction_feedback(
            predicted_temp=35.0,
            actual_temp=34.2,
            prediction_context={'indoor_temp': 20.5, 'outdoor_temp': 5.0}
        )
        print("✅ Learning feedback successful")
        
        # Test metrics
        metrics = wrapper.get_learning_metrics()
        print(f"✅ Learning metrics: {len(metrics)} metrics available")
        
        print("\n🎉 All tests passed! Enhanced Model Wrapper is ready.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_wrapper()
    sys.exit(0 if success else 1)
