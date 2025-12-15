"""
Test to identify environmental parameter mismatch between binary search and smart rounding.

This test specifically looks for the physically impossible behavior where:
- Binary search: 42.7°C → 21.06°C 
- Smart rounding: 42.0°C → 21.55°C (IMPOSSIBLE - lower outlet, higher prediction)

This suggests they're using different environmental conditions.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.model_wrapper import get_enhanced_model_wrapper
from src.temperature_control import SmartRounding
from src.prediction_context import prediction_context_manager


def test_environmental_conditions_capture():
    """
    Capture the EXACT environmental conditions used by binary search vs smart rounding.
    """
    
    print("\n" + "="*80)
    print("ENVIRONMENTAL CONDITIONS MISMATCH TEST")
    print("="*80)
    
    # Create test scenario matching the logs
    test_features = {
        'outdoor_temp': 6.8,
        'pv_now': 316.0,  # Current PV from logs
        'temp_forecast_1h': 6.8,
        'temp_forecast_2h': 6.8, 
        'temp_forecast_3h': 6.8,
        'temp_forecast_4h': 6.8,
        'pv_forecast_1h': 5.0,  # Forecast PV from logs
        'pv_forecast_2h': 5.0,
        'pv_forecast_3h': 5.0,
        'pv_forecast_4h': 5.0,
        'fireplace_on': 0.0,
        'tv_on': 0.0,
        'indoor_temp_lag_30m': 21.1,
        'target_temp': 21.0
    }
    
    # Get wrapper instance
    wrapper = get_enhanced_model_wrapper()
    thermal_model = wrapper.thermal_model
    
    # Capture parameters used in each call
    captured_params = []
    original_predict = thermal_model.predict_equilibrium_temperature
    
    def capture_predict_equilibrium_temperature(*args, **kwargs):
        # Capture the parameters
        captured_params.append({
            'method': 'predict_equilibrium_temperature',
            'outlet_temp': kwargs.get('outlet_temp', args[0] if args else None),
            'outdoor_temp': kwargs.get('outdoor_temp', args[1] if len(args) > 1 else None),
            'current_indoor': kwargs.get('current_indoor', args[2] if len(args) > 2 else None),
            'pv_power': kwargs.get('pv_power', args[3] if len(args) > 3 else None),
            'fireplace_on': kwargs.get('fireplace_on', kwargs.get('fireplace_on', 0.0)),
            'tv_on': kwargs.get('tv_on', kwargs.get('tv_on', 0.0)),
            'full_kwargs': kwargs.copy()
        })
        return original_predict(*args, **kwargs)
    
    # Patch the thermal model to capture calls
    thermal_model.predict_equilibrium_temperature = capture_predict_equilibrium_temperature
    
    print("Testing Binary Search Path...")
    print("-" * 40)
    
    # Test 1: Binary search path (simulate what happens in _calculate_required_outlet_temp)
    wrapper._current_features = test_features
    
    # This simulates binary search execution
    thermal_features = {
        'pv_power': test_features['pv_now'],
        'fireplace_on': 0.0,
        'tv_on': 0.0
    }
    
    # Get forecast conditions (what binary search uses)
    avg_outdoor, avg_pv, outdoor_forecast, pv_forecast = wrapper._get_forecast_conditions(
        test_features['outdoor_temp'], thermal_features['pv_power'], thermal_features
    )
    
    print(f"Binary search forecast conditions: outdoor={avg_outdoor:.1f}°C, pv={avg_pv:.0f}W")
    
    # Simulate binary search call
    binary_result = thermal_model.predict_equilibrium_temperature(
        outlet_temp=42.7,
        outdoor_temp=avg_outdoor,
        current_indoor=21.1,
        pv_power=avg_pv,
        fireplace_on=0.0,
        tv_on=0.0,
        _suppress_logging=True
    )
    
    print(f"Binary search result: 42.7°C → {binary_result:.2f}°C")
    binary_params = captured_params[-1].copy()
    
    print("\nTesting Smart Rounding Path...")
    print("-" * 40)
    
    # Test 2: Smart rounding path
    smart_rounding = SmartRounding()
    
    # This simulates what happens in smart rounding
    prediction_context_manager.set_features(test_features)
    unified_context = prediction_context_manager.create_context(
        outdoor_temp=test_features['outdoor_temp'],
        pv_power=thermal_features['pv_power'],
        thermal_features=thermal_features
    )
    
    thermal_params = prediction_context_manager.get_thermal_model_params()
    print(f"Smart rounding unified params: outdoor={thermal_params['outdoor_temp']:.1f}°C, "
          f"pv={thermal_params['pv_power']:.0f}W")
    
    # Simulate smart rounding call via predict_indoor_temp
    smart_result = wrapper.predict_indoor_temp(
        outlet_temp=42.0,
        outdoor_temp=thermal_params['outdoor_temp'],
        pv_power=thermal_params['pv_power'],
        fireplace_on=thermal_params['fireplace_on'],
        tv_on=thermal_params['tv_on'],
        current_indoor=21.1
    )
    
    print(f"Smart rounding result: 42.0°C → {smart_result:.2f}°C")
    smart_params = captured_params[-1].copy()
    
    print("\n" + "="*80)
    print("PARAMETER COMPARISON")
    print("="*80)
    
    print("Binary Search Parameters:")
    for key, value in binary_params.items():
        if key != 'full_kwargs':
            print(f"  {key}: {value}")
    
    print("\nSmart Rounding Parameters:")
    for key, value in smart_params.items():
        if key != 'full_kwargs':
            print(f"  {key}: {value}")
    
    print("\nParameter Differences:")
    differences_found = False
    for key in ['outlet_temp', 'outdoor_temp', 'current_indoor', 'pv_power', 'fireplace_on', 'tv_on']:
        if binary_params.get(key) != smart_params.get(key):
            print(f"  ❌ {key}: Binary={binary_params.get(key)} vs Smart={smart_params.get(key)}")
            differences_found = True
    
    if not differences_found:
        print("  ✅ All parameters identical!")
    
    print("\n" + "="*80)
    print("PHYSICS CHECK")
    print("="*80)
    
    # Test with IDENTICAL parameters to see if physics is consistent
    test_outdoor = avg_outdoor  # Use binary search conditions
    test_pv = avg_pv
    
    result_42_0 = original_predict(
        outlet_temp=42.0,
        outdoor_temp=test_outdoor,
        current_indoor=21.1,
        pv_power=test_pv,
        fireplace_on=0.0,
        tv_on=0.0,
        _suppress_logging=True
    )
    
    result_42_7 = original_predict(
        outlet_temp=42.7,
        outdoor_temp=test_outdoor,
        current_indoor=21.1,
        pv_power=test_pv,
        fireplace_on=0.0,
        tv_on=0.0,
        _suppress_logging=True
    )
    
    print(f"With IDENTICAL conditions (outdoor={test_outdoor:.1f}°C, pv={test_pv:.0f}W):")
    print(f"  42.0°C → {result_42_0:.2f}°C")
    print(f"  42.7°C → {result_42_7:.2f}°C")
    print(f"  Difference: {result_42_7 - result_42_0:.3f}°C (should be positive)")
    
    if result_42_7 > result_42_0:
        print("  ✅ PHYSICS CORRECT: Higher outlet temp → Higher indoor temp")
    else:
        print("  ❌ PHYSICS VIOLATION: Higher outlet temp → Lower indoor temp!")
    
    print("\n" + "="*80)
    print("ROOT CAUSE ANALYSIS") 
    print("="*80)
    
    # Check if the issue is environmental parameter differences
    if differences_found:
        print("🔍 ISSUE: Different environmental parameters used by binary search vs smart rounding")
        print("   - This explains the physically impossible behavior")
        print("   - Unified prediction context is NOT being used consistently")
    elif abs(binary_result - 21.06) < 0.1 and abs(smart_result - 21.55) < 0.1:
        print("🔍 ISSUE: Environmental conditions change between binary search and smart rounding execution")
        print("   - Parameters appear identical in test but differ in actual execution")
        print("   - Possible timing/state change issue")
    else:
        print("✅ Test does not reproduce the log discrepancy")
        print("   - Issue may be timing-dependent or require different test conditions")
    
    print("="*80)
    
    # Restore original method
    thermal_model.predict_equilibrium_temperature = original_predict


if __name__ == "__main__":
    test_environmental_conditions_capture()
