#!/usr/bin/env python3
"""
Test the simplified model wrapper to ensure it works correctly.
"""
import sys
import os
import pandas as pd

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_wrapper import (
    simplified_outlet_prediction, 
    get_enhanced_model_wrapper
)


def test_simplified_wrapper():
    """Test the simplified model wrapper functionality."""
    
    print("🧪 Testing Simplified Model Wrapper...")
    
    try:
        # Test enhanced wrapper creation
        wrapper = get_enhanced_model_wrapper()
        print("✅ Enhanced wrapper creation successful")
        
        # Test simplified prediction with basic features
        test_features = pd.DataFrame([{
            'indoor_temp_lag_30m': 20.5,
            'target_temp': 21.0,
            'outdoor_temp': 5.0,
            'pv_now': 1500.0,
            'fireplace_on': 0,
            'tv_on': 1
        }])
        
        outlet_temp, confidence, metadata = simplified_outlet_prediction(
            test_features, 20.5, 21.0
        )
        
        print("✅ Simplified prediction successful:")
        print(f"   - Outlet temp: {outlet_temp:.1f}°C")
        print(f"   - Confidence: {confidence:.3f}")
        print(f"   - Method: {metadata.get('prediction_method', 'unknown')}")
        
        # Test enhanced wrapper directly
        features_dict = test_features.to_dict(orient="records")[0]
        features_dict['indoor_temp_lag_30m'] = 20.5
        features_dict['target_temp'] = 21.0
        
        outlet_temp_2, metadata_2 = wrapper.calculate_optimal_outlet_temp(features_dict)
        
        print("✅ Enhanced wrapper direct test successful:")
        print(f"   - Outlet temp: {outlet_temp_2:.1f}°C")
        print(f"   - Learning confidence: {metadata_2.get('learning_confidence', 0):.3f}")
        
        print("\n🎉 All simplified wrapper tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_simplified_wrapper()
    sys.exit(0 if success else 1)
