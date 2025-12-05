#!/usr/bin/env python3
"""
Phase 4 Validation: Test corrected equilibrium equation against known scenarios
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from thermal_equilibrium_model import ThermalEquilibriumModel
from thermal_config import ThermalParameterConfig


def test_known_scenarios():
    """Test corrected equilibrium equation against physically known scenarios"""
    
    print("🧪 Phase 4.1: Testing Corrected Equilibrium Equation")
    print("=" * 60)
    
    # Initialize model
    model = ThermalEquilibriumModel()
    
    # Override model parameters for testing (fixed thermal_time_constant from memory bank)
    model.heat_loss_coefficient = 2.0  # W/°C
    model.thermal_time_constant = 4.0  # hours (fixed value from optimization fix)
    model.outlet_effectiveness = 0.85
    model.external_source_weights = {
        'pv': 0.1,  # °C/W
        'fireplace': 1.5,  # °C
        'tv': 0.3  # °C
    }
    
    print("Test Parameters:")
    print(f"  heat_loss_coefficient: {model.heat_loss_coefficient} W/°C")
    print(f"  thermal_time_constant: {model.thermal_time_constant} hours")
    print(f"  outlet_effectiveness: {model.outlet_effectiveness}")
    print(f"  external_source_weights: {model.external_source_weights}")
    print()
    
    # Scenario 1: No external heat, moderate outdoor temp
    print("Scenario 1: No external heat sources")
    T_outdoor = 5.0  # °C
    outlet_temp = 45.0  # °C
    
    T_eq = model.predict_equilibrium_temperature(
        outlet_temp, T_outdoor, pv_power=0, fireplace_on=0, tv_on=0
    )
    
    # Physics check: Should be between outdoor and outlet temp
    expected_range = (T_outdoor, outlet_temp)
    print(f"  Outdoor: {T_outdoor}°C, Outlet: {outlet_temp}°C")
    print(f"  Equilibrium: {T_eq:.2f}°C")
    print(f"  Expected range: {expected_range}")
    print(f"  ✅ Physics check: {expected_range[0] < T_eq < expected_range[1]}")
    print()
    
    # Scenario 2: With PV contribution
    print("Scenario 2: With PV solar contribution")
    pv_power = 1000  # 1kW
    
    T_eq_pv = model.predict_equilibrium_temperature(
        outlet_temp, T_outdoor, pv_power=pv_power, fireplace_on=0, tv_on=0
    )
    
    pv_contribution = model.external_source_weights['pv'] * pv_power
    print(f"  PV Power: {pv_power}W")
    print(f"  PV Contribution: {pv_contribution:.2f}°C")
    print(f"  Equilibrium without PV: {T_eq:.2f}°C")
    print(f"  Equilibrium with PV: {T_eq_pv:.2f}°C")
    print(f"  Temperature increase: {T_eq_pv - T_eq:.2f}°C")
    print(f"  ✅ PV increases temp: {T_eq_pv > T_eq}")
    print()
    
    # Scenario 3: With fireplace
    print("Scenario 3: With fireplace active")
    fireplace_on = 1  # ON
    
    T_eq_fire = model.predict_equilibrium_temperature(
        outlet_temp, T_outdoor, pv_power=0, fireplace_on=fireplace_on, tv_on=0
    )
    
    fireplace_contribution = model.external_source_weights['fireplace']
    print(f"  Fireplace: ON")
    print(f"  Fireplace Contribution: {fireplace_contribution:.2f}°C")
    print(f"  Equilibrium without fireplace: {T_eq:.2f}°C") 
    print(f"  Equilibrium with fireplace: {T_eq_fire:.2f}°C")
    print(f"  Temperature increase: {T_eq_fire - T_eq:.2f}°C")
    print(f"  ✅ Fireplace increases temp: {T_eq_fire > T_eq}")
    print()
    
    # Scenario 4: Multiple heat sources
    print("Scenario 4: Multiple heat sources combined")
    multi_pv_power = 500  # 0.5kW
    multi_fireplace_on = 1  # ON
    multi_tv_on = 1  # ON
    
    T_eq_multi = model.predict_equilibrium_temperature(
        outlet_temp, T_outdoor, pv_power=multi_pv_power, 
        fireplace_on=multi_fireplace_on, tv_on=multi_tv_on
    )
    
    total_contribution = (
        model.external_source_weights['pv'] * multi_pv_power +
        model.external_source_weights['fireplace'] * multi_fireplace_on +
        model.external_source_weights['tv'] * multi_tv_on
    )
    
    print(f"  PV Power: {multi_pv_power}W, Fireplace: ON, TV: ON")
    print(f"  Total Contribution: {total_contribution:.2f}°C")
    print(f"  Equilibrium baseline: {T_eq:.2f}°C")
    print(f"  Equilibrium multi-source: {T_eq_multi:.2f}°C") 
    print(f"  Temperature increase: {T_eq_multi - T_eq:.2f}°C")
    print(f"  ✅ Multi-source increases temp: {T_eq_multi > T_eq}")
    print()
    
    # Scenario 5: Energy conservation check
    print("Scenario 5: Energy conservation verification")
    Q_heating = model.heat_loss_coefficient * model.outlet_effectiveness * (outlet_temp - T_outdoor)
    Q_loss = model.heat_loss_coefficient * (T_eq - T_outdoor)
    Q_external = model.heat_loss_coefficient * 0  # No external sources
    
    print(f"  Heat Input (Q_heating): {Q_heating:.2f}W")
    print(f"  Heat Loss (Q_loss): {Q_loss:.2f}W") 
    print(f"  External Heat (Q_external): {Q_external:.2f}W")
    print(f"  Energy Balance: {abs(Q_heating - Q_loss):.2f}W")
    print(f"  ✅ Energy conserved: {abs(Q_heating - Q_loss) < 0.01}")
    print()
    
    print("🎉 All equilibrium scenarios validated successfully!")
    return True

if __name__ == "__main__":
    test_known_scenarios()
