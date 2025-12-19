#!/usr/bin/env python3
"""
Quick test for the Enhanced Model Wrapper to validate integration.
"""
import sys
import os
import unittest.mock
import logging

# Add src to path for testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from model_wrapper import EnhancedModelWrapper

def test_enhanced_wrapper():
    """Test basic functionality of the enhanced model wrapper."""
    
    print("🧪 Testing Enhanced Model Wrapper...")
    
    # Test initialization
    try:
        # Ensure a clean state for this test
        if os.path.exists('thermal_state.json'):
            os.remove('thermal_state.json')

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
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"Test failed: {e}"

def get_thermal_params(model):
    """Helper to get a dictionary of thermal model parameters."""
    return {
        "thermal_time_constant": model.thermal_time_constant,
        "heat_loss_coefficient": model.heat_loss_coefficient,
        "outlet_effectiveness": model.outlet_effectiveness,
    }

def test_first_cycle_learning_skip():
    """Verify that online learning is skipped on the first cycle after a restart."""
    print("\n🧪 Testing first-cycle learning skip...")

    try:
        # Ensure a clean state for this test
        if os.path.exists('thermal_state.json'):
            os.remove('thermal_state.json')

        # Use a new instance of the wrapper to simulate a restart
        wrapper = EnhancedModelWrapper()
        
        # On a fresh start, cycle_count from a new state file is 1
        assert wrapper.cycle_count == 1, f"Initial cycle count should be 1, but is {wrapper.cycle_count}"
        print(f"   - Initial cycle count is {wrapper.cycle_count}.")

        # Get initial thermal parameters
        initial_params = get_thermal_params(wrapper.thermal_model)
        
        with unittest.mock.patch('logging.info') as mock_log_info:
            # First call: should be skipped due to the guard at `cycle_count <= 1`
            print("   - First learning call (should be skipped)")
            wrapper.learn_from_prediction_feedback(
                predicted_temp=22.0,
                actual_temp=21.0,
                prediction_context={'outdoor_temp': 10.0, 'outlet_temp': 40.0, 'current_indoor': 20.5}
            )

            # 1. Check that learning was skipped: parameters should be unchanged
            params_after_first_call = get_thermal_params(wrapper.thermal_model)
            assert initial_params == params_after_first_call, "Thermal parameters should not change on first cycle"
            print("   ✅ Thermal parameters are unchanged.")
            
            # 2. Check that the special log message was emitted
            mock_log_info.assert_any_call("Skipping online learning on the first cycle to ensure stability.")
            print("   ✅ 'Skipping' message was logged.")

            # 3. Check that cycle count was incremented
            assert wrapper.cycle_count == 2, f"Cycle count should have been incremented to 2, but is {wrapper.cycle_count}"
            print(f"   ✅ Cycle count incremented to {wrapper.cycle_count}.")

            # Second call: should execute learning.
            # We need to call it enough times to fill the recent_errors_window.
            print(f"\n   - Filling error window ({wrapper.thermal_model.recent_errors_window} calls)...")
            for i in range(wrapper.thermal_model.recent_errors_window):
                wrapper.learn_from_prediction_feedback(
                    predicted_temp=22.0,
                    actual_temp=21.0, # Consistent error of -1.0
                    prediction_context={'outdoor_temp': 10.0, 'outlet_temp': 40.0, 'current_indoor': 20.5}
                )

            # 4. Check that parameters have now changed
            params_after_learning = get_thermal_params(wrapper.thermal_model)
            assert initial_params != params_after_learning, "Thermal parameters should change after filling the error window"
            print("   ✅ Thermal parameters updated successfully.")

            # 5. Check cycle count incremented again
            expected_cycle_count = 2 + wrapper.thermal_model.recent_errors_window
            assert wrapper.cycle_count == expected_cycle_count, f"Cycle count should be {expected_cycle_count}, but is {wrapper.cycle_count}"
            print(f"   ✅ Cycle count incremented to {wrapper.cycle_count}.")

        print("\n🎉 First-cycle learning skip test passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"Test failed: {e}"

if __name__ == "__main__":
    test_enhanced_wrapper()
    test_first_cycle_learning_skip()
