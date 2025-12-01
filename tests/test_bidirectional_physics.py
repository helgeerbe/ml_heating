#!/usr/bin/env python3
"""
Unit tests for bidirectional physics model functionality.

Tests that the physics model can predict both heating and cooling effects
and that charging mode works correctly for both scenarios.
"""

import sys
import os
import unittest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from physics_model import RealisticPhysicsModel
from model_wrapper import find_best_outlet_temp


class TestBidirectionalPhysics(unittest.TestCase):
    """Unit tests for bidirectional physics functionality."""

    def setUp(self):
        """Set up test model for each test."""
        self.model = RealisticPhysicsModel()

    def create_test_features(self, indoor_temp=21.0, outlet_temp=35.0, 
                           target_temp=21.0, outdoor_temp=10.0):
        """Create test feature set with specified values"""
        features = {
            'indoor_temp_lag_30m': indoor_temp,
            'outlet_temp': outlet_temp,
            'target_temp': target_temp,
            'outdoor_temp': outdoor_temp,
            'dhw_heating': 0.0,
            'dhw_disinfection': 0.0,
            'dhw_boost_heater': 0.0,
            'defrosting': 0.0,
            'pv_now': 0.0,
            'fireplace_on': 0.0,
            'tv_on': 0.0,
            'month_cos': 0.0,
            'month_sin': 0.0,
            'temp_forecast_1h': outdoor_temp,
            'temp_forecast_2h': outdoor_temp,
            'temp_forecast_3h': outdoor_temp,
            'temp_forecast_4h': outdoor_temp,
            'pv_forecast_1h': 0.0,
            'pv_forecast_2h': 0.0,
            'pv_forecast_3h': 0.0,
            'pv_forecast_4h': 0.0,
        }
        return features

    def test_cooling_prediction_low_outlet_temp(self):
        """Test that low outlet temperatures predict cooling (negative deltas)"""
        features = self.create_test_features(
            indoor_temp=21.0,
            outlet_temp=14.0,
            target_temp=21.0,
            outdoor_temp=10.0
        )
        
        prediction = self.model.predict_one(features)
        
        # Should predict cooling (negative delta) when outlet much lower than indoor
        self.assertLess(prediction, 0.0, 
                       "Low outlet temp should predict cooling (negative delta)")
        self.assertGreaterEqual(prediction, -0.15,
                               "Cooling prediction should be within bounds")

    def test_neutral_prediction_equal_temps(self):
        """Test that equal outlet/indoor temperatures predict minimal effect"""
        features = self.create_test_features(
            indoor_temp=21.0,
            outlet_temp=21.0,
            target_temp=21.0,
            outdoor_temp=10.0
        )
        
        prediction = self.model.predict_one(features)
        
        # Should predict minimal effect when outlet equals indoor
        self.assertGreater(prediction, -0.02,
                          "Equal temps should predict minimal effect")
        self.assertLess(prediction, 0.02,
                       "Equal temps should predict minimal effect")

    def test_heating_prediction_high_outlet_temp(self):
        """Test that high outlet temperatures predict heating (positive deltas)"""
        features = self.create_test_features(
            indoor_temp=21.0,
            outlet_temp=65.0,
            target_temp=21.0,
            outdoor_temp=10.0
        )
        
        prediction = self.model.predict_one(features)
        
        # Should predict heating (positive delta) when outlet much higher than indoor
        self.assertGreater(prediction, 0.0,
                          "High outlet temp should predict heating (positive delta)")
        self.assertLessEqual(prediction, 0.25,
                            "Heating prediction should be within bounds")

    def test_physics_bounds_respected(self):
        """Test that predictions stay within physics bounds"""
        test_cases = [
            (10.0, 21.0),  # Very low outlet
            (21.0, 21.0),  # Equal temps
            (70.0, 21.0),  # Very high outlet
        ]
        
        for outlet_temp, indoor_temp in test_cases:
            with self.subTest(outlet=outlet_temp, indoor=indoor_temp):
                features = self.create_test_features(
                    indoor_temp=indoor_temp,
                    outlet_temp=outlet_temp,
                    target_temp=21.0,
                    outdoor_temp=10.0
                )
                
                prediction = self.model.predict_one(features)
                
                # Check bounds
                self.assertGreaterEqual(prediction, -0.15,
                                       "Prediction should be >= min_prediction")
                self.assertLessEqual(prediction, 0.25,
                                    "Prediction should be <= max_prediction")

    def test_gradient_behavior(self):
        """Test that predictions form a reasonable gradient from cooling to heating"""
        indoor_temp = 21.0
        outlet_temps = [14.0, 16.0, 18.0, 20.0, 21.0, 22.0, 24.0, 26.0, 28.0, 30.0]
        predictions = []
        
        for outlet_temp in outlet_temps:
            features = self.create_test_features(
                indoor_temp=indoor_temp,
                outlet_temp=outlet_temp,
                target_temp=21.0,
                outdoor_temp=10.0
            )
            prediction = self.model.predict_one(features)
            predictions.append(prediction)
        
        # Check that predictions generally increase with outlet temperature
        # (more negative = more cooling, more positive = more heating)
        for i in range(1, len(predictions)):
            self.assertGreaterEqual(
                predictions[i], predictions[i-1] - 0.01,
                f"Prediction should increase or stay similar as outlet temp rises "
                f"(outlet {outlet_temps[i-1]}°C → {outlet_temps[i]}°C: "
                f"prediction {predictions[i-1]:.4f} → {predictions[i]:.4f})"
            )


class TestBalancingModeDirectionAware(unittest.TestCase):
    """Test direction-aware search ranges in balancing mode"""
    
    def setUp(self):
        self.model = RealisticPhysicsModel()
        
        # Base feature set for testing
        self.base_features = {
            'outdoor_temp': 0.0,
            'outlet_temp': 35.0,
            'outlet_temp_sq': 35.0 ** 2,
            'outlet_temp_cub': 35.0 ** 3,
            'outlet_temp_change_from_last': 0.0,
            'outlet_indoor_diff': 35.0 - 21.0,
            'outdoor_temp_x_outlet_temp': 0.0 * 35.0,
            'pv_now': 0.0,
            'dhw_heating': 0.0,
            'defrosting': 0.0,
            'dhw_disinfection': 0.0,
            'dhw_boost_heater': 0.0,
            'fireplace_on': 0.0,
            'tv_on': 0.0,
            'temp_forecast_1h': 1.0,
            'temp_forecast_2h': 1.0,
            'temp_forecast_3h': 1.0,
            'temp_forecast_4h': 1.0,
            'pv_forecast_1h': 0.0,
            'pv_forecast_2h': 0.0,
            'pv_forecast_3h': 0.0,
            'pv_forecast_4h': 0.0,
        }
    
    def test_balancing_cooling_scenario(self):
        """Test that balancing mode biases search range toward cooling when indoor > target"""
        features = self.base_features.copy()
        features.update({
            'indoor_temp_lag_30m': 21.15,  # Slightly too warm
            'target_temp': 21.0,
            'outlet_temp': 38.0,
            'outlet_temp_sq': 38.0 ** 2,
            'outlet_temp_cub': 38.0 ** 3,
            'outlet_indoor_diff': 38.0 - 21.15,
        })
        
        features_df = pd.DataFrame([features])
        outlet_history = [38.0]
        
        result = find_best_outlet_temp(
            model=self.model,
            features=features_df,
            current_temp=21.15,  # Too warm
            target_temp=21.0,
            outlet_history=outlet_history,
            error_target_vs_actual=0.15,  # Balancing range
            outdoor_temp=0.0
        )
        
        outlet_temp, confidence, control_mode, sigma, stability_score, trajectory, range_tested = result
        
        # Should be in balancing mode
        self.assertEqual(control_mode, "BALANCING")
        
        # Search range should be biased toward cooling (lower temperatures)
        min_range, max_range = range_tested
        current_outlet = 38.0
        
        # For cooling: range should be shifted down from normal ±8°C around current
        # Normal range would be [30.0, 46.0], cooling-biased should be lower
        self.assertLess(min_range, 28.0, "Cooling range should be biased toward lower temperatures")
        self.assertLess(max_range, 44.0, "Cooling range should be biased toward lower temperatures")
    
    def test_balancing_heating_scenario(self):
        """Test that balancing mode biases search range toward heating when indoor < target"""
        features = self.base_features.copy()
        features.update({
            'indoor_temp_lag_30m': 20.85,  # Slightly too cool
            'target_temp': 21.0,
            'outlet_temp': 35.0,
            'outlet_temp_sq': 35.0 ** 2,
            'outlet_temp_cub': 35.0 ** 3,
            'outlet_indoor_diff': 35.0 - 20.85,
        })
        
        features_df = pd.DataFrame([features])
        outlet_history = [35.0]
        
        result = find_best_outlet_temp(
            model=self.model,
            features=features_df,
            current_temp=20.85,  # Too cool
            target_temp=21.0,
            outlet_history=outlet_history,
            error_target_vs_actual=0.15,  # Balancing range
            outdoor_temp=0.0
        )
        
        outlet_temp, confidence, control_mode, sigma, stability_score, trajectory, range_tested = result
        
        # Should be in balancing mode
        self.assertEqual(control_mode, "BALANCING")
        
        # Search range should be biased toward heating (higher temperatures)
        min_range, max_range = range_tested
        current_outlet = 35.0
        
        # For heating: range should be shifted up from normal ±8°C around current
        # Normal range would be [27.0, 43.0], heating-biased should be higher
        self.assertGreater(min_range, 29.0, "Heating range should be biased toward higher temperatures")
        self.assertGreater(max_range, 41.0, "Heating range should be biased toward higher temperatures")
    
    def test_balancing_outlet_temperature_selection(self):
        """Test that balancing mode selects appropriate outlet temperatures for cooling vs heating"""
        # Test cooling scenario
        cooling_features = self.base_features.copy()
        cooling_features.update({
            'indoor_temp_lag_30m': 21.15,  # Too warm
            'target_temp': 21.0,
            'outlet_temp': 38.0,
            'outlet_temp_sq': 38.0 ** 2,
            'outlet_temp_cub': 38.0 ** 3,
            'outlet_indoor_diff': 38.0 - 21.15,
        })
        
        cooling_df = pd.DataFrame([cooling_features])
        cooling_result = find_best_outlet_temp(
            model=self.model,
            features=cooling_df,
            current_temp=21.15,
            target_temp=21.0,
            outlet_history=[38.0],
            error_target_vs_actual=0.15,
            outdoor_temp=0.0
        )
        
        # Test heating scenario
        heating_features = self.base_features.copy()
        heating_features.update({
            'indoor_temp_lag_30m': 20.85,  # Too cool
            'target_temp': 21.0,
            'outlet_temp': 35.0,
            'outlet_temp_sq': 35.0 ** 2,
            'outlet_temp_cub': 35.0 ** 3,
            'outlet_indoor_diff': 35.0 - 20.85,
        })
        
        heating_df = pd.DataFrame([heating_features])
        heating_result = find_best_outlet_temp(
            model=self.model,
            features=heating_df,
            current_temp=20.85,
            target_temp=21.0,
            outlet_history=[35.0],
            error_target_vs_actual=0.15,
            outdoor_temp=0.0
        )
        
        cooling_outlet = cooling_result[0]
        heating_outlet = heating_result[0]
        
        # Cooling scenario should choose lower outlet temperature than heating scenario
        self.assertLess(cooling_outlet, heating_outlet, 
                       "Cooling scenario should choose lower outlet temperature than heating")
        
        # Sanity check: temperatures should be within reasonable ranges
        self.assertGreaterEqual(cooling_outlet, 14.0, "Outlet temperature should be above minimum")
        self.assertLessEqual(heating_outlet, 65.0, "Outlet temperature should be below maximum")


class TestChargingModeBidirectional(unittest.TestCase):
    """Unit tests for charging mode bidirectional functionality."""

    def create_test_features_df(self, indoor_temp=21.3, target_temp=21.0, 
                               current_outlet=38.3, outdoor_temp=0.0):
        """Create test feature DataFrame matching real scenario"""
        features = {
            'indoor_temp_lag_30m': indoor_temp,
            'outdoor_temp': outdoor_temp,
            'target_temp': target_temp,
            'outlet_temp': current_outlet,
            'outlet_temp_sq': current_outlet ** 2,
            'outlet_temp_cub': current_outlet ** 3,
            'outlet_temp_change_from_last': 0.0,
            'outlet_indoor_diff': current_outlet - indoor_temp,
            'outdoor_temp_x_outlet_temp': outdoor_temp * current_outlet,
            'pv_now': 0.0,
            'dhw_heating': 0.0,
            'defrosting': 0.0,
            'dhw_disinfection': 0.0,
            'dhw_boost_heater': 0.0,
            'fireplace_on': 0.0,
            'tv_on': 1.0,
            'temp_forecast_1h': 1.6,
            'temp_forecast_2h': 1.8,
            'temp_forecast_3h': 2.1,
            'temp_forecast_4h': 2.1,
            'pv_forecast_1h': 0.0,
            'pv_forecast_2h': 0.0,
            'pv_forecast_3h': 0.0,
            'pv_forecast_4h': 0.0,
        }
        return pd.DataFrame([features])

    def test_charging_mode_cooling_scenario(self):
        """Test charging mode chooses cooling temperatures when needed"""
        model = RealisticPhysicsModel()
        
        features = self.create_test_features_df(
            indoor_temp=21.3,    # Current indoor (too warm)
            target_temp=21.0,    # Target (need cooling)
            current_outlet=38.3, # Current outlet from logs
            outdoor_temp=0.0     # Cold outdoor temp from logs
        )
        
        outlet_history = [38.3]  # Current outlet temp from logs
        
        result = find_best_outlet_temp(
            model=model,
            features=features, 
            current_temp=21.3,   # Too warm
            target_temp=21.0,    # Target
            outlet_history=outlet_history,
            error_target_vs_actual=0.3,  # Positive error = too warm
            outdoor_temp=0.0
        )
        
        outlet_temp, confidence, control_mode, sigma, stability_score, trajectory, range_tested = result
        
        # Verify charging mode is used for large error
        self.assertEqual(control_mode, "CHARGING", 
                        "Should use charging mode for 0.3°C error")
        
        # Verify full range is tested
        self.assertEqual(range_tested[0], 14.0, "Should test from minimum temp")
        self.assertEqual(range_tested[1], 65.0, "Should test to maximum temp")
        
        # CRITICAL: Should now choose a cooling temperature (< 21.0°C)
        self.assertLess(outlet_temp, 21.0,
                       "Should choose outlet temp below target for cooling")
        
        # Should choose a temperature that provides actual cooling effect
        self.assertLess(outlet_temp, 20.0,
                       "Should choose significantly lower temp for effective cooling")

    def test_charging_mode_heating_scenario(self):
        """Test charging mode still works for heating scenarios"""
        model = RealisticPhysicsModel()
        
        features = self.create_test_features_df(
            indoor_temp=20.5,    # Current indoor (too cool)
            target_temp=21.0,    # Target (need heating)
            current_outlet=35.0, # Current outlet
            outdoor_temp=5.0     # Moderate outdoor temp
        )
        
        outlet_history = [35.0]
        
        result = find_best_outlet_temp(
            model=model,
            features=features, 
            current_temp=20.5,   # Too cool
            target_temp=21.0,    # Target
            outlet_history=outlet_history,
            error_target_vs_actual=-0.5,  # Negative error = too cool
            outdoor_temp=5.0
        )
        
        outlet_temp, confidence, control_mode, sigma, stability_score, trajectory, range_tested = result
        
        # Verify charging mode
        self.assertEqual(control_mode, "CHARGING", 
                        "Should use charging mode for 0.5°C error")
        
        # Should choose a heating temperature (> current indoor)
        self.assertGreater(outlet_temp, 21.0,
                          "Should choose outlet temp above target for heating")


if __name__ == "__main__":
    unittest.main(verbosity=2)
