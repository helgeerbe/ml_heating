#!/usr/bin/env python3
"""
Test the Battery Charger Logic with User's Original Problem Scenario

This tests the exact scenario from the user's logs:
- Target: 21.0°C, Current: 20.4°C (0.6°C gap)
- Previous selection: 14°C (WRONG!)
- Expected new selection: 65°C (CORRECT!)
"""

import sys
import logging
import pandas as pd

# Add src to path for imports
sys.path.insert(0, 'src')

from model_wrapper import find_best_outlet_temp
from physics_model import RealisticPhysicsModel

# Configure logging to see the optimization logic
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

def create_user_scenario_features():
    """Create features matching user's log scenario"""
    return pd.DataFrame([{
        'outlet_temp': 19.0,  # Previous was selecting 19°C
        'indoor_temp_lag_30m': 20.4,  # Current indoor temp
        'target_temp': 21.0,  # Target temp
        'outdoor_temp': 5.0,   # Winter outdoor temp
        'dhw_heating': 0.0,
        'dhw_disinfection': 0.0,
        'dhw_boost_heater': 0.0,
        'defrosting': 0.0,
        'pv_now': 0.0,  # Night time
        'fireplace_on': 0.0,
        'tv_on': 0.0,
        'temp_forecast_1h': 5.0,
        'temp_forecast_2h': 5.0,
        'temp_forecast_3h': 5.0,
        'temp_forecast_4h': 5.0,
        'pv_forecast_1h': 0.0,
        'pv_forecast_2h': 0.0,
        'pv_forecast_3h': 0.0,
        'pv_forecast_4h': 0.0,
    }])

def test_user_scenario():
    """Test the exact user scenario"""
    
    print("🏠 USER SCENARIO TEST: Dec 02 07:04:05 Log Reproduction")
    print("=" * 65)
    print("Original Problem:")
    print("  - Target: 21.0°C, Current: 20.4°C")
    print("  - Old controller selected: 14°C (WRONG!)")
    print("  - Predicted result: 20.38°C (still 0.62°C short)")
    print("  - User's heat curve needs: 61°C for proper heating")
    print("")
    print("Expected Fix:")
    print("  - New controller should select: 65°C (maximum heating)")
    print("  - Battery charger logic: Maximum progress for 0.6°C gap")
    print("-" * 65)
    
    # Initialize model and scenario
    model = RealisticPhysicsModel()
    features = create_user_scenario_features()
    outlet_history = [19.0]  # Previous selection
    outdoor_temp = 5.0
    
    # Test the Heat Balance Controller
    current_temp = 20.4
    target_temp = 21.0
    gap = target_temp - current_temp
    
    print(f"\n🧪 TESTING BATTERY CHARGER LOGIC:")
    print(f"   Current: {current_temp:.1f}°C")
    print(f"   Target:  {target_temp:.1f}°C")
    print(f"   Gap:     {gap:+.1f}°C")
    print(f"   Stage:   {'1 (Maximum Progress)' if abs(gap) > 0.2 else '2 (Fine Balancing)'}")
    print("")
    
    result = find_best_outlet_temp(
        model=model,
        features=features,
        current_temp=current_temp,
        target_temp=target_temp,
        outlet_history=outlet_history,
        error_target_vs_actual=gap,
        outdoor_temp=outdoor_temp
    )
    
    selected_outlet, confidence, control_mode, sigma, score, trajectory, outlet_range = result
    
    # Analyze the fix
    print(f"\n🎯 RESULT ANALYSIS:")
    print(f"   Selected outlet: {selected_outlet:.1f}°C")
    print(f"   Control mode:    {control_mode}")
    print(f"   Confidence:      {confidence:.3f}")
    
    # Verify the fix
    old_selection = 14.0
    improvement = selected_outlet - old_selection
    
    print(f"\n📊 BEFORE vs AFTER:")
    print(f"   OLD: {old_selection:.1f}°C → Insufficient heating")
    print(f"   NEW: {selected_outlet:.1f}°C → Maximum heating")
    print(f"   IMPROVEMENT: +{improvement:.1f}°C outlet temperature")
    
    # Success criteria
    success_criteria = [
        ("Stage 1 Detection", abs(gap) > 0.2),
        ("Maximum Heating", selected_outlet >= 60.0),
        ("CHARGING Mode", control_mode == "CHARGING"),
        ("Significant Improvement", improvement >= 40.0)
    ]
    
    print(f"\n✅ SUCCESS CRITERIA:")
    all_passed = True
    for criterion, passed in success_criteria:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status} {criterion}")
        all_passed &= passed
    
    print(f"\n{'🏆 MISSION ACCOMPLISHED!' if all_passed else '❌ SOME ISSUES DETECTED'}")
    
    if all_passed:
        print("The Heat Balance Controller now uses proper battery charger logic:")
        print("  ✅ Detects large temperature gaps (0.6°C > 0.2°C threshold)")
        print("  ✅ Selects maximum heating (65°C) for rapid progress")
        print("  ✅ No more inefficient 14-19°C selections")
        print("  ✅ Will reach target faster with proper outlet temperature")
        print("")
        print("🔥 The days of selecting 19°C for heating are OVER!")
        print("🔥 Maximum progress mode will give you the 61°C heating you need!")
    
    return all_passed

if __name__ == "__main__":
    success = test_user_scenario()
    sys.exit(0 if success else 1)
