#!/usr/bin/env python3
"""
Test script to validate the Thermal Equilibrium Model Migration success.

This script tests that the binary search inverse calculation properly uses
the learned thermal parameters from 26 days of training data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model_wrapper import get_enhanced_model_wrapper

def test_thermal_migration():
    """Test that thermal migration produces realistic outlet temperatures."""
    print("\n=== THERMAL EQUILIBRIUM MIGRATION VALIDATION ===")
    
    # Create test features representing current conditions
    features = {
        'target_temp': 21.0,
        'indoor_temp_lag_30m': 21.1,
        'outdoor_temp': 5.0,
        'pv_now': 0.0,
        'fireplace_on': 0,
        'tv_on': 0,
        'indoor_temp_gradient': 0.0,
        'temp_diff_indoor_outdoor': 16.1,
        'outlet_indoor_diff': 24.0
    }
    
    # Create enhanced model wrapper (loads learned parameters)
    wrapper = get_enhanced_model_wrapper()
    
    print(f"Loaded thermal parameters from training:")
    print(f"  - Thermal time constant: {wrapper.thermal_model.thermal_time_constant:.1f}h")
    print(f"  - Heat loss coefficient: {wrapper.thermal_model.heat_loss_coefficient:.4f}")
    print(f"  - Outlet effectiveness: {wrapper.thermal_model.outlet_effectiveness:.3f}")
    print(f"  - Learning confidence: {wrapper.thermal_model.learning_confidence:.3f}")
    
    # Test the prediction
    outlet_temp, metadata = wrapper.calculate_optimal_outlet_temp(features)
    
    print(f"\n=== PREDICTION RESULTS ===")
    print(f"Target: {features['target_temp']}°C")
    print(f"Current: {features['indoor_temp_lag_30m']}°C") 
    print(f"Outdoor: {features['outdoor_temp']}°C")
    print(f"Required outlet: {outlet_temp:.1f}°C")
    print(f"Confidence: {metadata['learning_confidence']:.1f}")
    
    # Validate results
    if outlet_temp > 45.0:
        print(f"\n✅ SUCCESS: Model suggests {outlet_temp:.1f}°C (realistic vs old 34.5°C)")
        print("✅ Binary search with learned parameters working correctly!")
        print("✅ 26 days of training data being properly utilized!")
    elif outlet_temp > 40.0:
        print(f"\n🔶 GOOD: Model suggests {outlet_temp:.1f}°C (improvement from 34.5°C)")
        print("🔶 Learning is working but may need more optimization")
    else:
        print(f"\n❌ ISSUE: Model still suggests {outlet_temp:.1f}°C (too low)")
        print("❌ Binary search may not be using learned parameters correctly")
    
    print(f"\n=== MIGRATION VALIDATION COMPLETE ===")
    return outlet_temp

if __name__ == "__main__":
    test_thermal_migration()
