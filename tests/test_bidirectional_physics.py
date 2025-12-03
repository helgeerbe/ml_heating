#!/usr/bin/env python3
"""
Unit tests for bidirectional physics model functionality.

Tests that the physics model can predict both heating and cooling effects.
"""

import sys
import os
import unittest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from physics_model import RealisticPhysicsModel
from model_wrapper import simplified_outlet_prediction


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


class TestSimplifiedOutletPrediction(unittest.TestCase):
    """Test the simplified outlet prediction functionality."""
    
    def test_simplified_prediction_basic_scenario(self):
        """Test basic simplified outlet prediction functionality"""
        features = pd.DataFrame([{
            'outdoor_temp': 5.0,
            'dhw_heating': 0.0,
            'defrosting': 0.0,
            'fireplace_on': 0.0,
            'pv_now': 0.0,
            'tv_on': 0.0,
        }])
        
        outlet_temp, confidence, metadata = simplified_outlet_prediction(
            features, 
            current_temp=21.0, 
            target_temp=21.5
        )
        
        # Should return reasonable values
        self.assertGreaterEqual(outlet_temp, 15.0, "Outlet temp should be reasonable")
        self.assertLessEqual(outlet_temp, 65.0, "Outlet temp should be reasonable")
        self.assertGreater(confidence, 0.0, "Confidence should be positive")
        self.assertIsInstance(metadata, dict, "Metadata should be a dictionary")

    def test_simplified_prediction_empty_features(self):
        """Test simplified prediction with empty features DataFrame"""
        features = pd.DataFrame([{}])  # Empty features
        
        outlet_temp, confidence, metadata = simplified_outlet_prediction(
            features, 
            current_temp=20.5, 
            target_temp=21.0
        )
        
        # Should handle empty features gracefully
        self.assertGreaterEqual(outlet_temp, 15.0, "Outlet temp should be reasonable")
        self.assertLessEqual(outlet_temp, 65.0, "Outlet temp should be reasonable")
        self.assertGreater(confidence, 0.0, "Confidence should be positive")

    def test_simplified_prediction_cooling_scenario(self):
        """Test simplified prediction for cooling scenario"""
        features = pd.DataFrame([{
            'outdoor_temp': 25.0,  # Hot outdoor temperature
            'dhw_heating': 0.0,
            'defrosting': 0.0,
            'fireplace_on': 0.0,
            'pv_now': 0.0,
        }])
        
        outlet_temp, confidence, metadata = simplified_outlet_prediction(
            features, 
            current_temp=22.0,    # Current temp too high
            target_temp=21.0      # Want to cool down
        )
        
        # For cooling scenario, outlet temp should be reasonable
        self.assertGreaterEqual(outlet_temp, 15.0, "Outlet temp should be reasonable")
        self.assertLessEqual(outlet_temp, 65.0, "Outlet temp should be reasonable")
        self.assertIsInstance(metadata, dict, "Should return metadata")

    def test_simplified_prediction_heating_scenario(self):
        """Test simplified prediction for heating scenario"""
        features = pd.DataFrame([{
            'outdoor_temp': -5.0,  # Cold outdoor temperature
            'dhw_heating': 0.0,
            'defrosting': 0.0,
            'fireplace_on': 0.0,
            'pv_now': 0.0,
        }])
        
        outlet_temp, confidence, metadata = simplified_outlet_prediction(
            features, 
            current_temp=20.0,    # Current temp too low
            target_temp=21.0      # Want to heat up
        )
        
        # For heating scenario, outlet temp should be reasonable
        self.assertGreaterEqual(outlet_temp, 15.0, "Outlet temp should be reasonable")
        self.assertLessEqual(outlet_temp, 65.0, "Outlet temp should be reasonable")
        self.assertIsInstance(metadata, dict, "Should return metadata")


if __name__ == "__main__":
    unittest.main(verbosity=2)
