import unittest
from unittest.mock import MagicMock, patch
import logging
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from model_wrapper import EnhancedModelWrapper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestSmartRoundingBug(unittest.TestCase):
    def setUp(self):
        self.wrapper = EnhancedModelWrapper()
        # Mock the internal physics model
        self.wrapper.thermal_model = MagicMock()
        self.wrapper.thermal_model.predict_equilibrium_temperature.return_value = 20.0
        
    def test_smart_rounding_overrides_pv_history(self):
        """
        Test that Smart Rounding currently overrides the passed PV history 
        with the scalar forecast 'avg_pv', bypassing solar lag.
        """
        # Setup a cycle-aligned forecast with high PV (e.g. sunrise forecast)
        # The code uses 'pv_power' key from cycle_aligned_forecast
        self.wrapper.cycle_aligned_forecast = {
            'outdoor_temp': 5.0,
            'pv_power': 500.0,  # Forecast says 500W coming
            'fireplace_on': 0.0,
            'tv_on': 0.0
        }
        
        # Caller passes PV history (lagged, currently low)
        # e.g. [0, 0, 0, 10, 20] -> mean is low
        pv_history = [0.0] * 10 + [50.0] 
        
        # Call predict_indoor_temp (which performs smart rounding logic)
        self.wrapper.predict_indoor_temp(
            outlet_temp=40.0,
            outdoor_temp=5.0,
            pv_power=pv_history
        )
        
        # Check what was passed to the internal model
        # We expect the BUG to be that it passed 500.0 (scalar) instead of pv_history
        args, _ = self.wrapper.thermal_model.predict_equilibrium_temperature.call_args
        passed_pv = args[2] # signature: (outlet_temp, outdoor_temp, pv_power, ...)
        
        print(f"\nPassed PV to model: {passed_pv}")
        
        # In the buggy state, it passes the scalar forecast 500.0
        # If fixed, it should pass the history list or a value closer to current reality
        
        if passed_pv == 500.0:
            print("❌ BUG REPRODUCED: Smart Rounding used forecast PV (500W) instead of history.")
            # We want this test to FAIL when the bug is present if we were doing TDD, 
            # but here we want to confirm reproduction.
            # So asserting True confirms we found the bug.
            self.assertTrue(True)
        else:
            print(f"✅ BUG NOT REPRODUCED: Passed {passed_pv}")
            self.fail("Expected bug to be present (usage of 500.0)")

if __name__ == '__main__':
    unittest.main()
