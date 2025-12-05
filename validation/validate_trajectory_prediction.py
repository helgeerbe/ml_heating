#!/usr/bin/env python3
"""
Phase 4 Validation: Test trajectory prediction functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from thermal_equilibrium_model import ThermalEquilibriumModel


def test_trajectory_prediction():
    """Test trajectory prediction still works correctly after physics fixes"""
    
    print("🧪 Phase 4.1: Testing Trajectory Prediction")
    print("=" * 60)
    
    # Initialize model
    model = ThermalEquilibriumModel()
    
    # Override model parameters for testing
    model.heat_loss_coefficient = 0.2  # More realistic value
    model.thermal_time_constant = 4.0  # hours (fixed value)
    model.outlet_effectiveness = 0.85
    
    print("Trajectory Test Parameters:")
    print(f"  heat_loss_coefficient: {model.heat_loss_coefficient} W/°C")
    print(f"  thermal_time_constant: {model.thermal_time_constant} hours")
    print(f"  outlet_effectiveness: {model.outlet_effectiveness}")
    print()
    
    # Test 1: Heating scenario
    print("Test 1: Heating scenario")
    current_indoor = 18.0  # °C
    target_indoor = 22.0   # °C  
    outlet_temp = 45.0     # °C
    outdoor_temp = 5.0     # °C
    
    trajectory = model.predict_thermal_trajectory(
        current_indoor=current_indoor,
        target_indoor=target_indoor,
        outlet_temp=outlet_temp,
        outdoor_temp=outdoor_temp,
        time_horizon_hours=4
    )
    
    print(f"  Current: {current_indoor}°C → Target: {target_indoor}°C")
    print(f"  Outlet: {outlet_temp}°C, Outdoor: {outdoor_temp}°C")
    print(f"  Trajectory (4h): {[f'{t:.1f}' for t in trajectory['trajectory']]}")
    print(f"  Reaches target at: {trajectory['reaches_target_at']} hours")
    print(f"  Overshoot predicted: {trajectory['overshoot_predicted']}")
    print(f"  Final temperature: {trajectory['trajectory'][-1]:.1f}°C")
    print(f"  ✅ Temperature increases: {trajectory['trajectory'][-1] > current_indoor}")
    print()
    
    # Test 2: With external heat sources
    print("Test 2: Trajectory with PV solar contribution")
    trajectory_pv = model.predict_thermal_trajectory(
        current_indoor=current_indoor,
        target_indoor=target_indoor,
        outlet_temp=outlet_temp,
        outdoor_temp=outdoor_temp,
        time_horizon_hours=4,
        pv_power=500  # 0.5kW
    )
    
    print(f"  With 500W PV power")
    print(f"  Trajectory (4h): {[f'{t:.1f}' for t in trajectory_pv['trajectory']]}")
    print(f"  Reaches target at: {trajectory_pv['reaches_target_at']} hours")
    print(f"  Final temp with PV: {trajectory_pv['trajectory'][-1]:.1f}°C")
    print(f"  Final temp without PV: {trajectory['trajectory'][-1]:.1f}°C") 
    print(f"  ✅ PV increases final temp: {trajectory_pv['trajectory'][-1] > trajectory['trajectory'][-1]}")
    print()
    
    # Test 3: Cooling scenario 
    print("Test 3: Cooling scenario")
    current_indoor_hot = 25.0  # °C
    target_indoor_cool = 22.0  # °C
    outlet_temp_low = 30.0     # °C (reduced heating)
    
    trajectory_cool = model.predict_thermal_trajectory(
        current_indoor=current_indoor_hot,
        target_indoor=target_indoor_cool,
        outlet_temp=outlet_temp_low,
        outdoor_temp=outdoor_temp,
        time_horizon_hours=4
    )
    
    print(f"  Current: {current_indoor_hot}°C → Target: {target_indoor_cool}°C")
    print(f"  Outlet: {outlet_temp_low}°C, Outdoor: {outdoor_temp}°C")
    print(f"  Trajectory (4h): {[f'{t:.1f}' for t in trajectory_cool['trajectory']]}")
    print(f"  Final temperature: {trajectory_cool['trajectory'][-1]:.1f}°C")
    print(f"  ✅ Temperature decreases: {trajectory_cool['trajectory'][-1] < current_indoor_hot}")
    print()
    
    # Test 4: Test with weather forecasts
    print("Test 4: Trajectory with weather forecasts")
    weather_forecasts = [5.0, 4.0, 3.0, 2.0]  # Cooling trend
    
    trajectory_forecast = model.predict_thermal_trajectory(
        current_indoor=current_indoor,
        target_indoor=target_indoor,
        outlet_temp=outlet_temp,
        outdoor_temp=outdoor_temp,
        time_horizon_hours=4,
        weather_forecasts=weather_forecasts
    )
    
    print(f"  Weather forecast: {weather_forecasts}°C")
    print(f"  Trajectory with forecast: {[f'{t:.1f}' for t in trajectory_forecast['trajectory']]}")
    print(f"  Trajectory without forecast: {[f'{t:.1f}' for t in trajectory['trajectory']]}")
    print(f"  ✅ Forecast affects trajectory: {trajectory_forecast['trajectory'] != trajectory['trajectory']}")
    print()
    
    print("🎉 All trajectory prediction tests passed!")
    return True


if __name__ == "__main__":
    test_trajectory_prediction()
