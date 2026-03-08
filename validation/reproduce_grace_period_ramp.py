import sys
import os
import logging
import time
from unittest.mock import MagicMock, patch

# Add root to path to allow importing from src
sys.path.append(os.getcwd())

from src.heating_controller import BlockingStateManager
from src.state_manager import SystemState
import src.config as config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def reproduce_grace_period_ramp():
    print("\n--- Reproducing Grace Period Ramp ---")

    # 1. Setup Mocks
    ha_client = MagicMock()

    # Scenario: 
    # - Blocking just ended.
    # - Model thinks we need 65°C (e.g. bug or extreme cold).
    # - Current outlet is 35°C.
    # - Previous setpoint was 40°C.
    # - Grace period loop runs every 60s (simulated).
    
    current_indoor = 20.0
    target_indoor = 21.0
    outdoor_temp = 5.0
    actual_outlet = 35.0
    
    # We need to simulate the loop in _wait_for_grace_target
    # We can't easily mock the loop itself without rewriting the test to call the body of the loop.
    # Instead, we will instantiate BlockingStateManager and call _wait_for_grace_target
    # but we need to mock time.sleep to advance time and avoid waiting,
    # and mock ha_client.get_state to return values that might change or stay same.
    
    # We want to see if ha_client.set_state is called with increasing values.
    
    bsm = BlockingStateManager()
    
    # Mock wrapper
    mock_wrapper = MagicMock()
    mock_wrapper._calculate_required_outlet_temp.return_value = 65.0 # Always demands 65
    mock_wrapper._extract_thermal_features.return_value = {"dummy": 1}
    
    thermal_features = {"dummy": 1}
    
    # Mock HA Client behavior
    # It needs to return states for the loop
    def get_state_side_effect(entity_id, all_states=None, is_binary=False):
        if entity_id == config.ACTUAL_OUTLET_TEMP_ENTITY_ID:
            return actual_outlet # Constant for now, or we could simulate it rising slowly
        if entity_id == config.INDOOR_TEMP_ENTITY_ID:
            return current_indoor
        if entity_id == config.TARGET_INDOOR_TEMP_ENTITY_ID:
            return target_indoor
        if entity_id == config.OUTDOOR_TEMP_ENTITY_ID:
            return outdoor_temp
        return None

    ha_client.get_state.side_effect = get_state_side_effect
    ha_client.get_all_states.return_value = {}
    
    # Mock check_blocking_state to always return False (no blocking)
    bsm.check_blocking_state = MagicMock(return_value=(False, []))
    
    # Mock time.time to advance
    start_time = 1000.0
    current_time = start_time
    
    def mock_time():
        return current_time
    
    # Mock time.sleep to advance our fake time
    def mock_sleep(seconds):
        nonlocal current_time
        current_time += seconds
        print(f"  [Simulated Sleep] Advanced {seconds}s. Time: {current_time - start_time}s")

    # Patch time
    with patch('time.time', side_effect=mock_time), \
         patch('time.sleep', side_effect=mock_sleep), \
         patch('src.config.GRACE_PERIOD_MAX_MINUTES', 5), \
         patch('src.config.BLOCKING_POLL_INTERVAL_SECONDS', 60), \
         patch('src.config.MAX_TEMP_CHANGE_PER_CYCLE', 2.0):
        
        print(f"Starting Grace Period Wait. Max Minutes: 5. Poll Interval: 60s.")
        print(f"Model Demand: 65.0°C. Initial Target: 42.0°C (assumed clamped from 40+2)")
        
        # Initial grace target passed to the function
        # Assuming previous was 40, max change 2 -> 42.
        initial_grace_target = 42.0 
        
        # We need to capture set_state calls
        set_state_calls = []
        # Accept any kwargs to handle round_digits
        ha_client.set_state.side_effect = lambda e, v, a, **kwargs: set_state_calls.append(v)
        
        # Run the wait function
        # We set wait_for_cooling=False (heating up)
        bsm._wait_for_grace_target(
            ha_client,
            initial_grace_target,
            wait_for_cooling=False,
            wrapper=mock_wrapper,
            thermal_features=thermal_features
        )
        
        print("\n--- Results ---")
        print(f"Initial Target: {initial_grace_target}°C")
        print(f"Set State Calls (Targets set during loop):")
        for i, val in enumerate(set_state_calls):
            print(f"  Update {i+1}: {val}°C")
            
        if len(set_state_calls) > 0:
            final_val = set_state_calls[-1]
            if final_val > initial_grace_target + 1.0:
                print(f"\n[FAIL] Target ramped up significantly! {initial_grace_target} -> {final_val}")
            else:
                print(f"\n[PASS] Target stayed stable.")
        else:
            print("\n[INFO] No updates to target (might be stable).")

if __name__ == "__main__":
    reproduce_grace_period_ramp()
