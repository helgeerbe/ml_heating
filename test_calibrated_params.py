#!/usr/bin/env python3
"""
Quick test to verify calibrated parameters are being loaded correctly.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_calibrated_parameter_loading():
    """Test that ThermalEquilibriumModel loads calibrated parameters."""
    print("🧪 Testing Calibrated Parameter Loading")
    print("=" * 50)
    
    # Import the model
    from thermal_equilibrium_model import ThermalEquilibriumModel
    
    # Check if calibrated_baseline.json exists
    calibrated_file = "/opt/ml_heating/calibrated_baseline.json"
    if os.path.exists(calibrated_file):
        print(f"✅ Found calibrated_baseline.json")
        
        # Load the expected calibrated values
        import json
        with open(calibrated_file, 'r') as f:
            baseline = json.load(f)
        
        expected_params = baseline.get('parameters', {})
        print(f"📋 Expected calibrated values:")
        print(f"   - thermal_time_constant: {expected_params.get('thermal_time_constant', 'N/A')}")
        print(f"   - heat_loss_coefficient: {expected_params.get('heat_loss_coefficient', 'N/A')}")
        print(f"   - outlet_effectiveness: {expected_params.get('outlet_effectiveness', 'N/A')}")
        print()
        
        # Create model instance
        print("🔧 Creating ThermalEquilibriumModel...")
        model = ThermalEquilibriumModel()
        
        # Check actual loaded values
        print(f"🎯 Actual loaded values:")
        print(f"   - thermal_time_constant: {model.thermal_time_constant}")
        print(f"   - heat_loss_coefficient: {model.heat_loss_coefficient}")
        print(f"   - outlet_effectiveness: {model.outlet_effectiveness}")
        print()
        
        # Compare values
        print("🔍 Verification:")
        
        expected_thermal = expected_params.get('thermal_time_constant')
        if expected_thermal and abs(model.thermal_time_constant - expected_thermal) < 0.01:
            print(f"✅ thermal_time_constant MATCHES calibrated value")
        else:
            print(f"❌ thermal_time_constant MISMATCH: expected {expected_thermal}, got {model.thermal_time_constant}")
        
        expected_heat_loss = expected_params.get('heat_loss_coefficient')
        if expected_heat_loss and abs(model.heat_loss_coefficient - expected_heat_loss) < 0.0001:
            print(f"✅ heat_loss_coefficient MATCHES calibrated value")
        else:
            print(f"❌ heat_loss_coefficient MISMATCH: expected {expected_heat_loss}, got {model.heat_loss_coefficient}")
        
        expected_effectiveness = expected_params.get('outlet_effectiveness')
        if expected_effectiveness and abs(model.outlet_effectiveness - expected_effectiveness) < 0.001:
            print(f"✅ outlet_effectiveness MATCHES calibrated value")
        else:
            print(f"❌ outlet_effectiveness MISMATCH: expected {expected_effectiveness}, got {model.outlet_effectiveness}")
        
        # Test external source weights
        print()
        print(f"🔧 Heat source weights:")
        print(f"   - PV: {model.external_source_weights['pv']}")
        print(f"   - Fireplace: {model.external_source_weights['fireplace']}")
        print(f"   - TV: {model.external_source_weights['tv']}")
        
    else:
        print(f"❌ No calibrated_baseline.json found at {calibrated_file}")
        return False
    
    print("\n" + "=" * 50)
    print("✅ Test completed!")
    return True

if __name__ == "__main__":
    test_calibrated_parameter_loading()
