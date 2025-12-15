"""
Test to debug prediction consistency between binary search and smart rounding.

This test investigates why binary search and smart rounding give different predictions
for nearly identical outlet temperatures, despite using unified prediction context.
"""

import pytest
from src.model_wrapper import get_enhanced_model_wrapper
from src.temperature_control import SmartRounding
from src.prediction_context import prediction_context_manager


def test_prediction_consistency_debug():
    """
    Debug test to identify why binary search and smart rounding give different predictions.
    
    Based on logs:
    - Binary search: 42.7°C → 21.06°C 
    - Smart rounding: 42°C → 21.55°C (significant difference for similar input)
    """
    
    print("\n" + "="*80)
    print("DEBUGGING PREDICTION CONSISTENCY ISSUE")
    print("="*80)
    
    # Create test conditions matching the log scenario
    outlet_temp_binary = 42.7  # Binary search result
    outlet_temp_smart = 42.0   # Smart rounding floor test
    outdoor_temp = 6.8
    target_indoor = 21.0
    
    # Mock features with forecast data
    test_features = {
        'outdoor_temp': outdoor_temp,
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
        'tv_on': 0.0
    }
    
    thermal_features = {
        'pv_power': test_features['pv_now'],
        'fireplace_on': 0.0,
        'tv_on': 0.0
    }
    
    # Get model wrapper instance
    wrapper = get_enhanced_model_wrapper()
    thermal_model = wrapper.thermal_model
    
    print(f"\nThermal Model Parameters:")
    print(f"  - Thermal time constant: {thermal_model.thermal_time_constant:.3f}h")
    print(f"  - Heat loss coefficient: {thermal_model.heat_loss_coefficient:.6f}")
    print(f"  - Outlet effectiveness: {thermal_model.outlet_effectiveness:.6f}")
    print(f"  - Learning confidence: {thermal_model.learning_confidence:.3f}")
    
    # Test 1: Direct thermal model calls (what binary search uses)
    print(f"\n" + "-"*50)
    print("TEST 1: Direct thermal model calls (Binary Search Method)")
    print("-"*50)
    
    # Simulate binary search forecast conditions
    wrapper._current_features = test_features
    avg_outdoor, avg_pv, outdoor_forecast, pv_forecast = wrapper._get_forecast_conditions(
        outdoor_temp, thermal_features['pv_power'], thermal_features
    )
    
    print(f"Forecast conditions: outdoor={avg_outdoor:.1f}°C, pv={avg_pv:.0f}W")
    
    # Direct call to thermal model (binary search method)
    binary_prediction = thermal_model.predict_equilibrium_temperature(
        outlet_temp=outlet_temp_binary,
        outdoor_temp=avg_outdoor,
        current_indoor=21.1,  # Approximate from logs
        pv_power=avg_pv,
        fireplace_on=0.0,
        tv_on=0.0,
        _suppress_logging=True
    )
    
    print(f"Binary search method: {outlet_temp_binary:.1f}°C → {binary_prediction:.2f}°C")
    
    # Test 2: Smart rounding method
    print(f"\n" + "-"*50)
    print("TEST 2: Smart rounding method")
    print("-"*50)
    
    # Set up prediction context manager
    prediction_context_manager.set_features(test_features)
    unified_context = prediction_context_manager.create_context(
        outdoor_temp=outdoor_temp,
        pv_power=thermal_features['pv_power'], 
        thermal_features=thermal_features
    )
    
    thermal_params = prediction_context_manager.get_thermal_model_params()
    print(f"Unified context params: outdoor={thermal_params['outdoor_temp']:.1f}°C, "
          f"pv={thermal_params['pv_power']:.0f}W")
    
    # Smart rounding method using unified context
    smart_prediction = wrapper.predict_indoor_temp(
        outlet_temp=outlet_temp_smart,
        outdoor_temp=thermal_params['outdoor_temp'],
        pv_power=thermal_params['pv_power'],
        fireplace_on=thermal_params['fireplace_on'],
        tv_on=thermal_params['tv_on'],
        current_indoor=21.1
    )
    
    print(f"Smart rounding method: {outlet_temp_smart:.1f}°C → {smart_prediction:.2f}°C")
    
    # Test 3: Compare with identical parameters
    print(f"\n" + "-"*50) 
    print("TEST 3: Direct comparison with identical parameters")
    print("-"*50)
    
    # Use the SAME parameters for both methods
    test_outdoor = avg_outdoor
    test_pv = avg_pv
    test_current_indoor = 21.1
    
    # Method 1: Direct thermal model call
    direct_42_7 = thermal_model.predict_equilibrium_temperature(
        outlet_temp=42.7,
        outdoor_temp=test_outdoor,
        current_indoor=test_current_indoor,
        pv_power=test_pv,
        fireplace_on=0.0,
        tv_on=0.0,
        _suppress_logging=True
    )
    
    # Method 2: Through wrapper predict_indoor_temp 
    wrapper_42_7 = wrapper.predict_indoor_temp(
        outlet_temp=42.7,
        outdoor_temp=test_outdoor,
        pv_power=test_pv,
        fireplace_on=0.0,
        tv_on=0.0,
        current_indoor=test_current_indoor
    )
    
    print(f"Direct model call (42.7°C): {direct_42_7:.3f}°C")
    print(f"Wrapper call (42.7°C): {wrapper_42_7:.3f}°C")
    print(f"Difference: {abs(direct_42_7 - wrapper_42_7):.6f}°C")
    
    # Test with 42.0°C
    direct_42_0 = thermal_model.predict_equilibrium_temperature(
        outlet_temp=42.0,
        outdoor_temp=test_outdoor,
        current_indoor=test_current_indoor,
        pv_power=test_pv,
        fireplace_on=0.0,
        tv_on=0.0,
        _suppress_logging=True
    )
    
    wrapper_42_0 = wrapper.predict_indoor_temp(
        outlet_temp=42.0,
        outdoor_temp=test_outdoor,
        pv_power=test_pv,
        fireplace_on=0.0,
        tv_on=0.0,
        current_indoor=test_current_indoor
    )
    
    print(f"Direct model call (42.0°C): {direct_42_0:.3f}°C")
    print(f"Wrapper call (42.0°C): {wrapper_42_0:.3f}°C")
    print(f"Difference: {abs(direct_42_0 - wrapper_42_0):.6f}°C")
    
    # Test 4: Parameter sensitivity analysis
    print(f"\n" + "-"*50)
    print("TEST 4: Parameter sensitivity analysis")
    print("-"*50)
    
    # Test small parameter variations
    param_tests = [
        ("outdoor_temp", [6.8, 6.7, 6.9]),
        ("pv_power", [5.0, 4.0, 6.0]),
        ("current_indoor", [21.1, 21.0, 21.2])
    ]
    
    for param_name, values in param_tests:
        print(f"\n{param_name} sensitivity (42°C outlet):")
        base_params = {
            'outlet_temp': 42.0,
            'outdoor_temp': test_outdoor,
            'current_indoor': test_current_indoor,
            'pv_power': test_pv,
            'fireplace_on': 0.0,
            'tv_on': 0.0
        }
        
        for value in values:
            test_params = base_params.copy()
            test_params[param_name] = value
            
            result = thermal_model.predict_equilibrium_temperature(
                **{k: v for k, v in test_params.items() if k != 'outlet_temp'},
                outlet_temp=test_params['outlet_temp'],
                _suppress_logging=True
            )
            print(f"  {param_name}={value:.1f} → {result:.3f}°C")
    
    print(f"\n" + "="*80)
    print("ANALYSIS SUMMARY:")
    print("="*80)
    print(f"Expected from logs:")
    print(f"  - Binary search: 42.7°C → 21.06°C")
    print(f"  - Smart rounding: 42.0°C → 21.55°C")
    print(f"\nActual test results:")
    print(f"  - Direct model (42.7°C): {direct_42_7:.2f}°C")
    print(f"  - Wrapper model (42.0°C): {wrapper_42_0:.2f}°C")
    
    # Check if we can reproduce the discrepancy
    if abs(direct_42_7 - 21.06) < 0.1 and abs(wrapper_42_0 - 21.55) < 0.1:
        print(f"\n✅ DISCREPANCY REPRODUCED: Different prediction methods give different results")
        print(f"   This confirms the issue is in the prediction method differences")
    elif abs(direct_42_7 - wrapper_42_7) > 0.01:
        print(f"\n⚠️  WRAPPER DIFFERENCE DETECTED: predict_indoor_temp differs from direct calls")
        print(f"   This suggests parameter conversion issues in the wrapper")
    else:
        print(f"\n✅ PREDICTIONS CONSISTENT: Issue may be timing/context related")
        print(f"   Both methods give same results with identical parameters")
    
    print("="*80)

if __name__ == "__main__":
    test_prediction_consistency_debug()
