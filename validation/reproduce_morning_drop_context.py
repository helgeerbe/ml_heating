import unittest
from unittest.mock import patch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.prediction_context import UnifiedPredictionContext
import src.config as config

class TestMorningDropContext(unittest.TestCase):
    """
    Reproduction script for the 'morning drop' issue where the system
    over-anticipates rising outdoor temperatures due to incorrect forecast blending.
    """

    def setUp(self):
        self.features = {
            'temp_forecast_1h': 10.0,  # Rising temperature (Sunrise)
            'temp_forecast_2h': 12.0,
            'temp_forecast_3h': 14.0,
            'temp_forecast_4h': 15.0,
            'pv_forecast_1h': 100,
            'pv_forecast_2h': 200,
            'pv_forecast_3h': 300,
            'pv_forecast_4h': 400,
        }
        self.thermal_features = {'fireplace_on': 0, 'tv_on': 0}
        self.current_outdoor = 6.0  # Cold morning

    def test_forecast_blending_weight(self):
        """
        Verify that for a short control cycle (e.g. 30 mins), the forecast weight
        targets the cycle midpoint (15 mins) rather than the full hour.
        
        Previous behavior (Bug):
        Weight = 0.5 (hardcoded for < 1h)
        Effective Temp = 6.0 * 0.5 + 10.0 * 0.5 = 8.0°C
        
        Correct behavior (Fix):
        Cycle = 30 mins = 0.5 hours
        Midpoint = 15 mins = 0.25 hours
        Weight = 0.25 / 1.0 = 0.25
        Effective Temp = 6.0 * 0.75 + 10.0 * 0.25 = 4.5 + 2.5 = 7.0°C
        """
        
        # Test with 30 minute cycle
        with patch.object(config, 'CYCLE_INTERVAL_MINUTES', 30):
            context = UnifiedPredictionContext.create_prediction_context(
                self.features, 
                self.current_outdoor, 
                0, 
                self.thermal_features
            )
            
            print(f"\n--- 30 Minute Cycle Analysis ---")
            print(f"Current Outdoor: {self.current_outdoor}°C")
            print(f"1h Forecast: {self.features['temp_forecast_1h']}°C")
            print(f"Calculated Effective Outdoor: {context['avg_outdoor']}°C")
            
            # We expect the effective temperature to be 7.0°C
            # If it's 8.0°C, the bug is present (weight 0.5)
            
            expected_temp = 7.0
            self.assertAlmostEqual(context['avg_outdoor'], expected_temp, places=2,
                msg=f"Expected effective temp {expected_temp}°C, got {context['avg_outdoor']}°C")
            
            print("SUCCESS: Forecast blending correctly targets cycle midpoint.")

    def test_forecast_blending_weight_10min_cycle(self):
        """
        Verify behavior for a very short cycle (10 mins).
        
        Cycle = 10 mins = 0.166 hours
        Midpoint = 5 mins = 0.0833 hours
        Weight = 0.0833
        Effective Temp = 6.0 * (1-0.0833) + 10.0 * 0.0833
                       = 6.0 * 0.9167 + 0.833
                       = 5.5 + 0.833 = 6.33°C
        """
        with patch.object(config, 'CYCLE_INTERVAL_MINUTES', 10):
            context = UnifiedPredictionContext.create_prediction_context(
                self.features, 
                self.current_outdoor, 
                0, 
                self.thermal_features
            )
            
            cycle_hours = 10/60.0
            weight = cycle_hours / 2.0
            expected_temp = self.current_outdoor * (1 - weight) + self.features['temp_forecast_1h'] * weight
            
            print(f"\n--- 10 Minute Cycle Analysis ---")
            print(f"Calculated Effective Outdoor: {context['avg_outdoor']:.4f}°C")
            print(f"Expected Effective Outdoor: {expected_temp:.4f}°C")
            
            self.assertAlmostEqual(context['avg_outdoor'], expected_temp, places=4)

if __name__ == '__main__':
    unittest.main()
