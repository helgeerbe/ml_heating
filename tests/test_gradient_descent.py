
import unittest
import numpy as np
from src.thermal_equilibrium_model import ThermalEquilibriumModel

class TestGradientDescent(unittest.TestCase):
    def test_thermal_time_constant_gradient_descent(self):
        model = ThermalEquilibriumModel()
        model.reset_adaptive_learning()

        # Mock prediction history with a positive error
        model.prediction_history = [
            {
                "error": 0.5,
                "context": {
                    "outlet_temp": 40.0,
                    "outdoor_temp": 10.0,
                    "current_indoor": 20.0,
                    "pv_power": 0,
                    "fireplace_on": 0,
                    "tv_on": 0,
                },
                "timestamp": "2023-01-01T00:00:00Z",
            }
        ] * model.recent_errors_window

        initial_thermal_time_constant = model.thermal_time_constant
        
        # Adapt parameters
        model._adapt_parameters_from_recent_errors()
        
        # Ensure thermal_time_constant decreases (gradient descent)
        self.assertLessEqual(model.thermal_time_constant, initial_thermal_time_constant)

if __name__ == "__main__":
    unittest.main()
