
import logging
from typing import Dict, Any

# Mocking the necessary classes/functions for standalone testing
class ThermalParameterConfig:
    DEFAULTS = {
        'heat_loss_coefficient': 0.4,
        'outlet_effectiveness': 0.5,
        'thermal_time_constant': 4.0,
        'pv_heat_weight': 0.002,
        'fireplace_heat_weight': 5.0,
        'tv_heat_weight': 0.2,
    }
    BOUNDS = {
        'heat_loss_coefficient': (0.01, 1.5), # Updated bound
        'outlet_effectiveness': (0.3, 1.0),
        'thermal_time_constant': (3.0, 100.0),
        'pv_heat_weight': (0.0001, 0.005),
        'tv_heat_weight': (0.05, 1.5),
    }

    @classmethod
    def get_default(cls, param):
        return cls.DEFAULTS.get(param)

    @classmethod
    def get_bounds(cls, param):
        return cls.BOUNDS.get(param)

class ThermalEquilibriumModel:
    def __init__(self):
        self.heat_loss_coefficient = 0.4
        self.outlet_effectiveness = 0.5
        self.thermal_time_constant = 4.0
        self.external_source_weights = {}
        self.learning_confidence = 3.0
        self.prediction_history = []
        self.parameter_history = []

    def _load_config_defaults(self):
        print("⚙️ Loading config defaults...")
        self.heat_loss_coefficient = ThermalParameterConfig.get_default('heat_loss_coefficient')
        self.outlet_effectiveness = ThermalParameterConfig.get_default('outlet_effectiveness')
        self.thermal_time_constant = ThermalParameterConfig.get_default('thermal_time_constant')

    def _initialize_learning_attributes(self):
        pass

    def _detect_parameter_corruption(self):
        # Updated logic
        hcl_bounds = ThermalParameterConfig.get_bounds("heat_loss_coefficient")
        if not (hcl_bounds[0] <= self.heat_loss_coefficient <= hcl_bounds[1]):
            print(f"HLC {self.heat_loss_coefficient} out of bounds {hcl_bounds}")
            return True

        oe_bounds = ThermalParameterConfig.get_bounds("outlet_effectiveness")
        if not (oe_bounds[0] <= self.outlet_effectiveness <= oe_bounds[1]):
            print(f"OE {self.outlet_effectiveness} out of bounds {oe_bounds}")
            return True

        # Check for physically impossible combinations (drift detection)
        if (
            self.heat_loss_coefficient > 1.2
            and self.outlet_effectiveness < 0.4
        ):
            print(f"Drift detected: HLC={self.heat_loss_coefficient}, OE={self.outlet_effectiveness}")
            return True
            
        return False

    def _load_thermal_parameters(self, mock_state):
        # Simulate loading from state
        baseline_params = mock_state.get("baseline_parameters", {})
        learning_state = mock_state.get("learning_state", {})
        adjustments = learning_state.get("parameter_adjustments", {})

        self.heat_loss_coefficient = (
            baseline_params["heat_loss_coefficient"]
            + adjustments.get("heat_loss_coefficient_delta", 0.0)
        )
        self.outlet_effectiveness = (
            baseline_params["outlet_effectiveness"]
            + adjustments.get("outlet_effectiveness_delta", 0.0)
        )
        
        print(f"Loaded: HLC={self.heat_loss_coefficient}, OE={self.outlet_effectiveness}")

        if self._detect_parameter_corruption():
            print("🗑️ Detected corrupted parameters on load. Resetting.")
            self._load_config_defaults()
            return

# Test Case
mock_state = {
    "baseline_parameters": {
        "heat_loss_coefficient": 0.4,
        "outlet_effectiveness": 0.5,
        "thermal_time_constant": 4.0,
        "pv_heat_weight": 0.002,
        "fireplace_heat_weight": 5.0,
        "tv_heat_weight": 0.2,
        "source": "calibrated"
    },
    "learning_state": {
        "parameter_adjustments": {
            "heat_loss_coefficient_delta": 1.4697, # Resulting in 1.8697
            "outlet_effectiveness_delta": -0.1907 # Resulting in 0.3093
        }
    }
}

print("--- Testing Parameter Load with Drift ---")
model = ThermalEquilibriumModel()
model._load_thermal_parameters(mock_state)

print(f"Final Parameters: HLC={model.heat_loss_coefficient}, OE={model.outlet_effectiveness}")
