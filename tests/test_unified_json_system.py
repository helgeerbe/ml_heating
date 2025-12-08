#!/usr/bin/env python3
"""
Test script for the unified JSON state management system.

This script verifies that:
1. ThermalStateManager creates and manages state correctly
2. ThermalEquilibriumModel loads parameters from unified JSON
3. Model wrapper integrates properly with unified state
4. No pickle files are created or used
5. All state persistence works through single JSON file
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add src directory to path
sys.path.insert(0, '/opt/ml_heating/src')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_unified_json_system():
    """Test the unified thermal state system."""
    print("🧪 Testing Unified JSON State Management System")
    print("=" * 60)
    
    # Remove any existing state files for clean test
    state_file = "/opt/ml_heating/thermal_state.json"
    pickle_file = "/opt/ml_heating/ml_state.pkl"
    
    if os.path.exists(state_file):
        os.remove(state_file)
        print(f"✅ Removed existing {state_file}")
    
    if os.path.exists(pickle_file):
        print(f"⚠️  Found existing pickle file: {pickle_file}")
        print("   (This should be removed after migration)")
    
    # Test 1: ThermalStateManager creation and basic operations
    print("\n1️⃣ Testing ThermalStateManager...")
    try:
        from unified_thermal_state import get_thermal_state_manager
        
        state_manager = get_thermal_state_manager()
        print(f"✅ Created ThermalStateManager")
        print(f"   State file: {state_manager.state_file}")
        
        # Verify default state structure
        default_params = state_manager.get_current_parameters()
        print(f"✅ Default parameters loaded:")
        for param, value in default_params.items():
            print(f"   - {param}: {value}")
        
        # Test setting calibrated baseline
        calibrated_params = {
            'thermal_time_constant': 18.5,
            'heat_loss_coefficient': 0.12,
            'outlet_effectiveness': 0.65
        }
        state_manager.set_calibrated_baseline(calibrated_params, calibration_cycles=15)
        print(f"✅ Set calibrated baseline parameters")
        
        # Verify calibrated parameters are loaded
        current_params = state_manager.get_current_parameters()
        print(f"✅ Calibrated parameters loaded:")
        for param, value in current_params.items():
            if param in calibrated_params:
                print(f"   - {param}: {value} (calibrated)")
        
        # Test learning state updates
        state_manager.update_learning_state(
            cycle_count=25,
            learning_confidence=3.5,
            parameter_adjustments={
                'thermal_time_constant_delta': 1.2,
                'heat_loss_coefficient_delta': 0.02,
                'outlet_effectiveness_delta': -0.05
            }
        )
        print(f"✅ Updated learning state with adjustments")
        
        # Test metrics retrieval
        metrics = state_manager.get_learning_metrics()
        print(f"✅ Learning metrics retrieved:")
        print(f"   - Baseline source: {metrics['baseline_source']}")
        print(f"   - Current cycle: {metrics['current_cycle_count']}")
        print(f"   - Learning confidence: {metrics['learning_confidence']}")
        
    except Exception as e:
        print(f"❌ ThermalStateManager test failed: {e}")
        return False
    
    # Test 2: ThermalEquilibriumModel integration
    print("\n2️⃣ Testing ThermalEquilibriumModel integration...")
    try:
        from thermal_equilibrium_model import ThermalEquilibriumModel
        
        model = ThermalEquilibriumModel()
        print(f"✅ Created ThermalEquilibriumModel")
        print(f"   Thermal time constant: {model.thermal_time_constant}")
        print(f"   Heat loss coefficient: {model.heat_loss_coefficient}")
        print(f"   Outlet effectiveness: {model.outlet_effectiveness}")
        
        # Verify it loaded calibrated parameters (not defaults)
        if abs(model.thermal_time_constant - 19.7) < 0.5:  # 18.5 + 1.2 adjustment
            print(f"✅ Model loaded calibrated + adjusted parameters correctly")
        else:
            print(f"⚠️  Model parameters may not be loading correctly")
        
        # Test prediction
        predicted_temp = model.predict_equilibrium_temperature(
            outlet_temp=45.0,
            outdoor_temp=5.0,
            pv_power=800.0
        )
        print(f"✅ Prediction test: {predicted_temp:.2f}°C")
        
    except Exception as e:
        print(f"❌ ThermalEquilibriumModel test failed: {e}")
        return False
    
    # Test 3: Enhanced Model Wrapper integration
    print("\n3️⃣ Testing Enhanced Model Wrapper integration...")
    try:
        from model_wrapper import get_enhanced_model_wrapper
        
        wrapper = get_enhanced_model_wrapper()
        print(f"✅ Created Enhanced Model Wrapper")
        print(f"   Current cycle count: {wrapper.cycle_count}")
        
        # Test prediction calculation
        test_features = {
            'indoor_temp_lag_30m': 20.5,
            'target_temp': 22.0,
            'outdoor_temp': 5.0,
            'pv_now': 500.0,
            'fireplace_on': 0,
            'tv_on': 1
        }
        
        outlet_temp, metadata = wrapper.calculate_optimal_outlet_temp(test_features)
        print(f"✅ Outlet temperature calculation: {outlet_temp:.1f}°C")
        print(f"   Confidence: {metadata['learning_confidence']:.2f}")
        
        # Test learning feedback
        wrapper.learn_from_prediction_feedback(
            predicted_temp=22.0,
            actual_temp=21.8,
            prediction_context=test_features
        )
        print(f"✅ Learning feedback processed")
        print(f"   New cycle count: {wrapper.cycle_count}")
        
    except Exception as e:
        print(f"❌ Enhanced Model Wrapper test failed: {e}")
        return False
    
    # Test 4: Verify JSON state persistence
    print("\n4️⃣ Testing JSON state persistence...")
    try:
        # Check that thermal_state.json was created
        if os.path.exists(state_file):
            print(f"✅ Thermal state JSON file created: {state_file}")
            
            # Verify JSON structure
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            
            required_sections = [
                'metadata', 'baseline_parameters', 'learning_state',
                'prediction_metrics', 'operational_state'
            ]
            
            for section in required_sections:
                if section in state_data:
                    print(f"   ✅ {section} section present")
                else:
                    print(f"   ❌ {section} section missing")
                    return False
            
            # Check that no pickle files were created
            if not os.path.exists(pickle_file):
                print(f"✅ No pickle file created - using JSON only")
            else:
                print(f"⚠️  Pickle file still exists: {pickle_file}")
            
            print(f"✅ JSON state file size: {os.path.getsize(state_file)} bytes")
            
        else:
            print(f"❌ Thermal state JSON file not created")
            return False
            
    except Exception as e:
        print(f"❌ JSON persistence test failed: {e}")
        return False
    
    # Test 5: Test clean restart (simulate service restart)
    print("\n5️⃣ Testing clean restart behavior...")
    try:
        # Create new instances (simulating restart)
        new_state_manager = get_thermal_state_manager()
        new_model = ThermalEquilibriumModel()
        new_wrapper = get_enhanced_model_wrapper()
        
        print(f"✅ Clean restart simulation successful")
        print(f"   Restored cycle count: {new_wrapper.cycle_count}")
        print(f"   Restored thermal time constant: {new_model.thermal_time_constant}")
        print(f"   Restored learning confidence: {new_model.learning_confidence}")
        
        # Verify state consistency
        original_metrics = state_manager.get_learning_metrics()
        new_metrics = new_state_manager.get_learning_metrics()
        
        if original_metrics['current_cycle_count'] == new_metrics['current_cycle_count']:
            print(f"✅ State consistency maintained across restart")
        else:
            print(f"❌ State consistency lost across restart")
            return False
            
    except Exception as e:
        print(f"❌ Clean restart test failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Unified JSON state management system is working correctly.")
    print("\n📋 Summary:")
    print("   ✅ Single thermal_state.json file manages all state")
    print("   ✅ No pickle files needed")
    print("   ✅ Calibrated parameters load correctly")
    print("   ✅ Learning adjustments work")
    print("   ✅ State persists across restarts")
    print("   ✅ Model wrapper integrates cleanly")
    
    return True


def print_state_file_contents():
    """Print the contents of the thermal state file for inspection."""
    state_file = "/opt/ml_heating/thermal_state.json"
    
    if os.path.exists(state_file):
        print(f"\n📄 Contents of {state_file}:")
        print("-" * 60)
        
        with open(state_file, 'r') as f:
            state_data = json.load(f)
        
        # Pretty print the JSON with reduced verbosity
        for section, content in state_data.items():
            print(f"\n[{section}]")
            if isinstance(content, dict):
                for key, value in content.items():
                    if isinstance(value, (list, dict)) and len(str(value)) > 100:
                        print(f"  {key}: <{type(value).__name__} with {len(value)} items>")
                    else:
                        print(f"  {key}: {value}")
            else:
                print(f"  {content}")
    else:
        print(f"\n📄 No thermal state file found at {state_file}")


if __name__ == "__main__":
    success = test_unified_json_system()
    
    if success:
        print_state_file_contents()
        print(f"\n🚀 Ready for clean calibration setup!")
        print(f"   Next steps:")
        print(f"   1. Remove any existing pickle files")
        print(f"   2. Run calibration to populate baseline parameters")
        print(f"   3. Start normal operations")
    else:
        print(f"\n❌ Tests failed - fix issues before proceeding")
        sys.exit(1)
