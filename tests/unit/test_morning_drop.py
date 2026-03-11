import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from src.model_wrapper import EnhancedModelWrapper
from src.thermal_equilibrium_model import ThermalEquilibriumModel
from src import config

class TestMorningDropPrevention:
    """
    Regression tests for the "morning drop" issue where the system would
    prematurely reduce heating in the morning due to anticipated solar gain.
    """

    @pytest.fixture
    def wrapper(self):
        """Create a model wrapper with mocked dependencies."""
        wrapper = EnhancedModelWrapper()
        # Mock the thermal model to return predictable results
        wrapper.thermal_model = MagicMock(spec=ThermalEquilibriumModel)
        wrapper.thermal_model.thermal_time_constant = 20.0
        wrapper.thermal_model.heat_loss_coefficient = 1.0
        wrapper.thermal_model.outlet_effectiveness = 1.0
        wrapper.thermal_model.pv_heat_weight = 0.001
        wrapper.thermal_model.solar_lag_minutes = 45.0
        
        # Mock prediction methods
        wrapper.thermal_model.predict_equilibrium_temperature.return_value = 40.0
        wrapper.thermal_model.predict_thermal_trajectory.return_value = {
            "trajectory": [20.0, 20.5, 21.0],
            "times": [0.5, 1.0, 1.5],
            "reaches_target_at": 1.5
        }
        
        # Mock learning_confidence property
        wrapper.thermal_model.learning_confidence = 3.0
        
        return wrapper

    def test_pv_history_initialization_uses_current_pv(self, wrapper):
        """
        Test that when PV history is missing, it initializes with CURRENT PV,
        not the blended average (which includes forecast).
        """
        # Setup scenario:
        # Current PV = 0 (Morning, sun just rising)
        # Forecast PV = 2000 (Bright day ahead)
        # Avg PV (blended) = 500
        
        current_pv = 0.0
        forecast_pv = 2000.0
        
        features = {
            "outdoor_temp": 5.0,
            "pv_power": current_pv,
            "target_temp": 21.0,
            "indoor_temp_lag_30m": 20.0,
            "temp_forecast_1h": 10.0,
            "pv_forecast_1h": forecast_pv,
            # Crucially: No history
            "pv_power_history": None
        }
        
        # Mock _get_forecast_conditions to return the blended average
        # This simulates what happens in the real code
        wrapper._get_forecast_conditions = MagicMock(return_value=(
            7.5,   # avg_outdoor
            500.0, # avg_pv (blended)
            [10.0] * 4, # outdoor_forecast
            [forecast_pv] * 4 # pv_forecast
        ))
        
        # Run calculation
        wrapper.calculate_optimal_outlet_temp(features)
        
        # Verify that predict_equilibrium_temperature was called with current_pv (0.0)
        # as the pv_power input (which is used for history initialization),
        # NOT the blended avg_pv (500.0).
        
        # Check the calls to predict_equilibrium_temperature
        # We expect it to be called during the binary search
        calls = wrapper.thermal_model.predict_equilibrium_temperature.call_args_list
        
        # Filter for calls that are part of the binary search (not pre-checks)
        # In the fix, we pass `pv_input` which should be `current_pv`
        
        found_correct_call = False
        for call in calls:
            kwargs = call.kwargs
            if "pv_power" in kwargs:
                pv_arg = kwargs["pv_power"]
                # It should be 0.0 (current), not 500.0 (avg)
                if pv_arg == current_pv:
                    found_correct_call = True
                    break
                if pv_arg == 500.0:
                    pytest.fail("Found call using blended avg PV (500.0) instead of current PV (0.0)")
        
        assert found_correct_call, "Did not find any call using current PV for history initialization"

    def test_dynamic_horizon_shortens_when_cold(self, wrapper):
        """
        Test that the optimization horizon shortens to 1.0h when the house is cold
        to prioritize immediate recovery.
        """
        # Setup cold scenario
        current_indoor = 19.0
        target_temp = 21.0 # Gap = 2.0°C (Cold)
        
        features = {
            "outdoor_temp": 5.0,
            "pv_power": 0.0,
            "target_temp": target_temp,
            "indoor_temp_lag_30m": current_indoor,
            "pv_power_history": [0.0] * 10
        }
        
        # Mock _get_forecast_conditions
        wrapper._get_forecast_conditions = MagicMock(return_value=(
            5.0, 0.0, [5.0]*4, [0.0]*4
        ))
        
        # Ensure pre-check doesn't trigger early exit by making min_prediction low
        # and max_prediction high enough to encompass target
        # We need enough side effects for:
        # 1. min_prediction (pre-check)
        # 2. max_prediction (pre-check)
        # 3. binary search iterations (up to 20)
        # 4. final prediction (if needed)
        wrapper.thermal_model.predict_equilibrium_temperature.side_effect = [
            15.0, # min_prediction (at outlet_min) -> below target 21.0
            30.0, # max_prediction (at outlet_max) -> above target 21.0
        ] + [21.0] * 25 # binary search iterations...

        # Run calculation
        wrapper.calculate_optimal_outlet_temp(features)
        
        # Verify predict_thermal_trajectory was called with time_horizon_hours=1.0
        # (Aggressive Recovery)
        calls = wrapper.thermal_model.predict_thermal_trajectory.call_args_list
        assert len(calls) > 0
        
        # Check the last call (which is usually the one that matters in the loop)
        # NOTE: The binary search loop calls predict_thermal_trajectory multiple times.
        # The dynamic horizon logic is inside the loop.
        # We need to check that AT LEAST ONE call used the correct horizon.
        # Or better, check the call corresponding to the final iteration or a specific iteration.
        
        # Let's check all calls made during the binary search
        found_correct_horizon = False
        for call in calls:
            if call.kwargs.get("time_horizon_hours") == 1.0:
                found_correct_horizon = True
                break
        
        assert found_correct_horizon, "Did not find any call with time_horizon_hours=1.0"

    def test_dynamic_horizon_lengthens_when_stable(self, wrapper):
        """
        Test that the optimization horizon is 4.0h when the house is at or above target.
        """
        # Setup stable scenario
        current_indoor = 21.0
        target_temp = 21.0 # Gap = 0.0°C (Stable)
        
        features = {
            "outdoor_temp": 5.0,
            "pv_power": 0.0,
            "target_temp": target_temp,
            "indoor_temp_lag_30m": current_indoor,
            "pv_power_history": [0.0] * 10
        }
        
        # Mock _get_forecast_conditions
        wrapper._get_forecast_conditions = MagicMock(return_value=(
            5.0, 0.0, [5.0]*4, [0.0]*4
        ))
        
        # Ensure pre-check doesn't trigger early exit
        # We need enough side effects for:
        # 1. min_prediction (pre-check)
        # 2. max_prediction (pre-check)
        # 3. binary search iterations (up to 20)
        # 4. final prediction (if needed)
        wrapper.thermal_model.predict_equilibrium_temperature.side_effect = [
            15.0, # min_prediction -> below target
            30.0, # max_prediction -> above target
        ] + [21.0] * 25 # binary search iterations...

        # Run calculation
        wrapper.calculate_optimal_outlet_temp(features)
        
        # Verify predict_thermal_trajectory was called with time_horizon_hours=4.0
        # (Stability)
        calls = wrapper.thermal_model.predict_thermal_trajectory.call_args_list
        assert len(calls) > 0
        
        # Check all calls made during the binary search
        found_correct_horizon = False
        for call in calls:
            if call.kwargs.get("time_horizon_hours") == 4.0:
                found_correct_horizon = True
                break
        
        assert found_correct_horizon, "Did not find any call with time_horizon_hours=4.0"

    def test_dynamic_horizon_moderate_when_cool(self, wrapper):
        """
        Test that the optimization horizon is 2.0h when the house is slightly cool
        (gap > 0.0 but <= 0.3).
        """
        # Setup cool scenario
        current_indoor = 20.9
        target_temp = 21.0 # Gap = 0.1°C (Cool)
        
        features = {
            "outdoor_temp": 5.0,
            "pv_power": 0.0,
            "target_temp": target_temp,
            "indoor_temp_lag_30m": current_indoor,
            "pv_power_history": [0.0] * 10
        }
        
        # Mock _get_forecast_conditions
        wrapper._get_forecast_conditions = MagicMock(return_value=(
            5.0, 0.0, [5.0]*4, [0.0]*4
        ))
        
        # Ensure pre-check doesn't trigger early exit
        wrapper.thermal_model.predict_equilibrium_temperature.side_effect = [
            15.0, # min_prediction
            30.0, # max_prediction
        ] + [21.0] * 25

        # Run calculation
        wrapper.calculate_optimal_outlet_temp(features)
        
        # Verify predict_thermal_trajectory was called with time_horizon_hours=2.0
        calls = wrapper.thermal_model.predict_thermal_trajectory.call_args_list
        assert len(calls) > 0
        
        found_correct_horizon = False
        for call in calls:
            if call.kwargs.get("time_horizon_hours") == 2.0:
                found_correct_horizon = True
                break
        
        assert found_correct_horizon, "Did not find any call with time_horizon_hours=2.0"
