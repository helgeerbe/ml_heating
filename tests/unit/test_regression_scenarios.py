import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path if needed (though usually handled by test runner)
sys.path.append(os.path.join(os.getcwd(), "src"))

from src.model_wrapper import EnhancedModelWrapper  # noqa: E402


class TestRegressionScenarios(unittest.TestCase):
    """
    Regression tests for reported issues:
    1. Startup Overshoot (fake gradient due to missing history)
    2. Parameter Drift (corrupted thermal parameters)
    """

    def setUp(self):
        # Reset singleton if it exists to ensure fresh state
        import src.model_wrapper
        src.model_wrapper._enhanced_model_wrapper_instance = None

        # Mock config to ensure consistent test environment
        self.config_patcher = patch('src.config')
        self.mock_config = self.config_patcher.start()

        # Set standard config values
        self.mock_config.CYCLE_INTERVAL_MINUTES = 30
        self.mock_config.HISTORY_STEPS = 18
        self.mock_config.CLAMP_MIN_ABS = 20.0
        self.mock_config.CLAMP_MAX_ABS = 75.0
        self.mock_config.TRAJECTORY_PREDICTION_ENABLED = True
        self.mock_config.TRAJECTORY_STEPS = 4.0
        self.mock_config.SHADOW_MODE = False

        # Setup common test data
        self.current_indoor = 19.0
        self.target_indoor = 21.0
        self.outdoor_temp = 5.0

    def tearDown(self):
        self.config_patcher.stop()

    @patch('src.model_wrapper.get_thermal_state_manager')
    def test_startup_overshoot_prevention(self, mock_get_state_manager):
        """
        Test Case 1: Startup Overshoot Prevention
        Simulates "missing history" scenario where InfluxDB returns defaults
        (21.0), but actual house is colder (19.0), creating a fake massive
        temperature drop. Verifies that prediction remains safe (< 60°C) for
        a 2°C gap.
        """
        # Setup State Manager Mock
        mock_state_manager = MagicMock()
        mock_state_manager.get_learning_metrics.return_value = {
            "current_cycle_count": 10
        }
        # Simulate no saved state
        mock_state_manager.load_state.return_value = False
        mock_get_state_manager.return_value = mock_state_manager

        # Initialize Wrapper
        wrapper = EnhancedModelWrapper()

        # Force standard parameters (not corrupted yet)
        wrapper.thermal_model.heat_loss_coefficient = 0.6
        wrapper.thermal_model.outlet_effectiveness = 0.8
        wrapper.thermal_model.thermal_time_constant = 15.0

        # Construct features simulating the "Startup" scenario
        # History was 21.0 (default), Current is 19.0 -> Gradient looks like
        # -4.0°C/hr (huge drop)
        features = {
            # The "default" history value, but we override for current
            # context below
            "indoor_temp_lag_30m": 19.0,  # Current temp
            "indoor_temp_gradient": -4.0,  # Fake gradient due to mismatch
            "outdoor_temp": self.outdoor_temp,
            "target_temp": self.target_indoor,
            "pv_now": 0.0,
            "fireplace_on": 0,
            "tv_on": 0,
            "thermal_power_kw": 0.0,
            "temp_diff_indoor_outdoor": (
                self.current_indoor - self.outdoor_temp
            ),
            "outlet_indoor_diff": 10.0,  # Arbitrary previous state
            # Forecasts (flat for simplicity)
            "temp_forecast_1h": self.outdoor_temp,
            "temp_forecast_2h": self.outdoor_temp,
            "temp_forecast_3h": self.outdoor_temp,
            "temp_forecast_4h": self.outdoor_temp,
            "pv_forecast_1h": 0.0,
            "pv_forecast_2h": 0.0,
            "pv_forecast_3h": 0.0,
            "pv_forecast_4h": 0.0,
        }

        # Run prediction
        outlet_temp, metadata = wrapper.calculate_optimal_outlet_temp(features)

        # Assertions
        print(f"Test 1 Result: Outlet Temp = {outlet_temp:.2f}°C")

        # Should be high because of the gap + gradient, but clamped to safe
        # limit. The user mentioned "does not spike to extreme values
        # (e.g., 65°C)"
        self.assertLess(
            outlet_temp,
            60.0,
            "Outlet temperature spiked above 60°C for 2°C gap"
        )
        self.assertGreater(
            outlet_temp,
            35.0,
            "Outlet temperature too low for heating scenario"
        )

    @patch('src.model_wrapper.get_thermal_state_manager')
    def test_parameter_drift_detection(self, mock_get_state_manager):
        """
        Test Case 2: Parameter Drift Detection
        Manually sets thermal model parameters to known "corrupted" values
        (High Heat Loss + Low Effectiveness).
        Verifies that _detect_parameter_corruption() identifies this state.
        """
        # Setup State Manager Mock
        mock_state_manager = MagicMock()
        mock_state_manager.get_learning_metrics.return_value = {
            "current_cycle_count": 10
        }
        mock_state_manager.load_state.return_value = False
        mock_get_state_manager.return_value = mock_state_manager

        # Initialize Wrapper
        wrapper = EnhancedModelWrapper()
        model = wrapper.thermal_model

        # Case A: High Heat Loss + Low Effectiveness (The "Corrupted" State)
        model.heat_loss_coefficient = 0.8
        model.outlet_effectiveness = 0.019

        is_corrupt = model._detect_parameter_corruption()
        self.assertTrue(
            is_corrupt,
            "Failed to detect corruption: High HLC + Low Eff"
        )

        # Case B: Extreme Heat Loss (Runaway)
        model.heat_loss_coefficient = 2.0
        model.outlet_effectiveness = 0.5  # Normal eff

        is_corrupt = model._detect_parameter_corruption()
        self.assertTrue(is_corrupt, "Failed to detect corruption: Extreme HLC")

        # Case C: Normal Parameters (Should not be corrupt)
        model.heat_loss_coefficient = 0.6
        model.outlet_effectiveness = 0.8

        is_corrupt = model._detect_parameter_corruption()
        self.assertFalse(
            is_corrupt,
            "False positive: Normal parameters flagged as corrupt"
        )

        # Case D: Borderline Bad (Moderate HLC, Very Low Eff)
        # As per code: HLC > 0.6 and Eff < 0.35
        model.heat_loss_coefficient = 0.65
        model.outlet_effectiveness = 0.30

        is_corrupt = model._detect_parameter_corruption()
        self.assertTrue(
            is_corrupt,
            "Failed to detect corruption: Borderline Bad"
        )


if __name__ == '__main__':
    unittest.main()
