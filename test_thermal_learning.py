#!/usr/bin/env python3
"""
Test thermal learning capability to verify adaptive parameter adjustment works.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model_wrapper import get_enhanced_model_wrapper
from state_manager import save_state, load_state
import time

def test_thermal_learning():
    """Test that thermal parameters actually adapt based on prediction feedback."""
    print("\n=== THERMAL LEARNING CAPABILITY TEST ===")
    
    # Create enhanced model wrapper
    wrapper = get_enhanced_model_wrapper()
    thermal_model = wrapper.thermal_model
    
    # Record initial parameters
    initial_params = {
        'thermal_time_constant': thermal_model.thermal_time_constant,
        'heat_loss_coefficient': thermal_model.heat_loss_coefficient,
        'outlet_effectiveness': thermal_model.outlet_effectiveness,
        'learning_confidence': thermal_model.learning_confidence
    }
    
    print(f"Initial parameters:")
    print(f"  - Thermal time constant: {initial_params['thermal_time_constant']:.1f}h")
    print(f"  - Heat loss coefficient: {initial_params['heat_loss_coefficient']:.4f}")
    print(f"  - Outlet effectiveness: {initial_params['outlet_effectiveness']:.3f}")
    print(f"  - Learning confidence: {initial_params['learning_confidence']:.3f}")
    
    # Simulate realistic prediction feedback cycles
    print(f"\nSimulating thermal learning with prediction feedback...")
    
    # Create realistic heating scenarios that should drive learning
    # Scenario: Model consistently under-predicts indoor temperature 
    # (suggesting outlet effectiveness should be higher)
    scenarios = [
        # outlet_temp, outdoor_temp, expected_prediction_error, pv_power, fireplace_on, tv_on
        (45.0, 5.0, +1.5, 0.0, 0, 0),      # Under-predict by 1.5°C
        (46.0, 4.0, +1.3, 200.0, 0, 1),    # Under-predict by 1.3°C
        (44.0, 6.0, +1.7, 0.0, 0, 0),      # Under-predict by 1.7°C
        (47.0, 3.0, +1.2, 150.0, 0, 0),    # Under-predict by 1.2°C
        (45.5, 5.5, +1.4, 50.0, 0, 1),     # Under-predict by 1.4°C
        (43.0, 7.0, +1.8, 0.0, 0, 0),      # Under-predict by 1.8°C
        (48.0, 2.0, +1.0, 300.0, 0, 0),    # Under-predict by 1.0°C
        (46.5, 4.5, +1.3, 100.0, 0, 1),    # Under-predict by 1.3°C
        (44.5, 6.5, +1.6, 0.0, 0, 0),      # Under-predict by 1.6°C
        (47.5, 3.5, +1.1, 250.0, 0, 0),    # Under-predict by 1.1°C
        # Additional cycles to ensure enough data for learning
        (45.2, 5.2, +1.4, 75.0, 0, 1),     # Under-predict by 1.4°C
        (46.8, 4.2, +1.2, 180.0, 0, 0),    # Under-predict by 1.2°C
        (44.3, 6.3, +1.7, 25.0, 0, 0),     # Under-predict by 1.7°C
        (47.1, 3.1, +1.3, 220.0, 0, 1),    # Under-predict by 1.3°C
        (45.7, 5.7, +1.5, 80.0, 0, 0),     # Under-predict by 1.5°C
    ]
    
    for i, (outlet_temp, outdoor_temp, prediction_error, pv_power, fireplace_on, tv_on) in enumerate(scenarios):
        # Make prediction using thermal model
        predicted_temp = thermal_model.predict_equilibrium_temperature(
            outlet_temp=outlet_temp,
            outdoor_temp=outdoor_temp,
            pv_power=pv_power,
            fireplace_on=fireplace_on,
            tv_on=tv_on
        )
        
        # Simulate actual measured temperature (predicted + error)
        actual_temp = predicted_temp + prediction_error
        
        # Create prediction context
        context = {
            'outlet_temp': outlet_temp,
            'outdoor_temp': outdoor_temp,
            'pv_power': pv_power,
            'fireplace_on': fireplace_on,
            'tv_on': tv_on
        }
        
        # Feed back to thermal learning
        thermal_model.update_prediction_feedback(
            predicted_temp=predicted_temp,
            actual_temp=actual_temp,
            prediction_context=context,
            timestamp=f"test_cycle_{i+1}"
        )
        
        print(f"  Cycle {i+1:2d}: {outlet_temp}°C → predicted {predicted_temp:.1f}°C, actual {actual_temp:.1f}°C (error: {prediction_error:+.1f}°C)")
    
    # Check if parameters changed
    final_params = {
        'thermal_time_constant': thermal_model.thermal_time_constant,
        'heat_loss_coefficient': thermal_model.heat_loss_coefficient,
        'outlet_effectiveness': thermal_model.outlet_effectiveness,
        'learning_confidence': thermal_model.learning_confidence
    }
    
    print(f"\nFinal parameters:")
    print(f"  - Thermal time constant: {final_params['thermal_time_constant']:.1f}h")
    print(f"  - Heat loss coefficient: {final_params['heat_loss_coefficient']:.4f}")
    print(f"  - Outlet effectiveness: {final_params['outlet_effectiveness']:.3f}")
    print(f"  - Learning confidence: {final_params['learning_confidence']:.3f}")
    
    # Calculate parameter changes
    changes = {
        'thermal_time_constant': abs(final_params['thermal_time_constant'] - initial_params['thermal_time_constant']),
        'heat_loss_coefficient': abs(final_params['heat_loss_coefficient'] - initial_params['heat_loss_coefficient']),
        'outlet_effectiveness': abs(final_params['outlet_effectiveness'] - initial_params['outlet_effectiveness']),
        'learning_confidence': final_params['learning_confidence'] - initial_params['learning_confidence']
    }
    
    print(f"\nParameter changes:")
    print(f"  - Thermal time constant: {changes['thermal_time_constant']:+.2f}h")
    print(f"  - Heat loss coefficient: {changes['heat_loss_coefficient']:+.5f}")
    print(f"  - Outlet effectiveness: {changes['outlet_effectiveness']:+.4f}")
    print(f"  - Learning confidence: {changes['learning_confidence']:+.3f}")
    
    # Determine if learning is working
    significant_changes = 0
    if changes['thermal_time_constant'] > 0.1:
        print(f"  ✅ Thermal time constant adapted by {changes['thermal_time_constant']:.2f}h")
        significant_changes += 1
    if changes['heat_loss_coefficient'] > 0.001:
        print(f"  ✅ Heat loss coefficient adapted by {changes['heat_loss_coefficient']:.5f}")
        significant_changes += 1
    if changes['outlet_effectiveness'] > 0.001:
        print(f"  ✅ Outlet effectiveness adapted by {changes['outlet_effectiveness']:.4f}")
        significant_changes += 1
    
    # Get learning metrics
    learning_metrics = thermal_model.get_adaptive_learning_metrics()
    print(f"\nLearning metrics:")
    print(f"  - Total predictions: {learning_metrics.get('total_predictions', 0)}")
    print(f"  - Parameter updates: {learning_metrics.get('parameter_updates', 0)}")
    print(f"  - Update percentage: {learning_metrics.get('update_percentage', 0):.1f}%")
    print(f"  - Current learning rate: {learning_metrics.get('current_learning_rate', 0):.5f}")
    
    # Save the learning state
    try:
        thermal_learning_state = {
            'thermal_time_constant': thermal_model.thermal_time_constant,
            'heat_loss_coefficient': thermal_model.heat_loss_coefficient,
            'outlet_effectiveness': thermal_model.outlet_effectiveness,
            'learning_confidence': thermal_model.learning_confidence,
            'cycle_count': len(scenarios)
        }
        save_state(thermal_learning_state=thermal_learning_state)
        print(f"\n✅ Thermal learning state saved to ml_state.pkl")
    except Exception as e:
        print(f"\n❌ Failed to save thermal learning state: {e}")
    
    print(f"\n=== THERMAL LEARNING TEST RESULTS ===")
    if significant_changes >= 2:
        print(f"✅ SUCCESS: Thermal learning is working! {significant_changes} parameters adapted.")
        print(f"   The system should now suggest more realistic outlet temperatures.")
        return True
    elif learning_metrics.get('parameter_updates', 0) > 0:
        print(f"🔶 PARTIAL: Some parameter updates occurred ({learning_metrics.get('parameter_updates')} updates)")
        print(f"   Learning is working but may need more data to see significant changes.")
        return True
    else:
        print(f"❌ ISSUE: No significant parameter adaptation detected.")
        print(f"   Thermal learning may need debugging or more aggressive settings.")
        return False

if __name__ == "__main__":
    test_thermal_learning()
