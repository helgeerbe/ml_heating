import pytest
import numpy as np
from src.thermal_equilibrium_model import ThermalEquilibriumModel
from src.thermal_config import ThermalParameterConfig

class TestSolarThermalLag:
    @pytest.fixture
    def model(self):
        model = ThermalEquilibriumModel()
        # Set known parameters for testing
        model.solar_lag_minutes = 30.0
        model.external_source_weights["pv"] = 0.002
        return model

    def test_calculate_effective_solar_scalar(self, model):
        """Test that scalar input returns the value directly (instantaneous)."""
        assert model._calculate_effective_solar(1000.0) == 1000.0

    def test_calculate_effective_solar_short_lag(self, model):
        """Test that very short lag returns instantaneous value."""
        model.solar_lag_minutes = 1.0
        history = [0.0, 500.0, 1000.0]
        assert model._calculate_effective_solar(history) == 1000.0

    def test_calculate_effective_solar_rolling_average(self, model):
        """Test rolling average calculation."""
        # 10-minute steps
        # History: [0, 0, 0, 1000, 1000, 1000]
        # Lag: 30 minutes -> should average last 3 values (1000, 1000, 1000) -> 1000
        history = [0.0, 0.0, 0.0, 1000.0, 1000.0, 1000.0]
        model.solar_lag_minutes = 30.0
        assert model._calculate_effective_solar(history) == 1000.0

        # Lag: 60 minutes -> average last 6 values -> 500
        model.solar_lag_minutes = 60.0
        assert model._calculate_effective_solar(history) == 500.0

    def test_predict_equilibrium_temperature_with_lag(self, model):
        """Test that equilibrium temperature uses lagged PV."""
        # Setup
        outlet = 30.0
        outdoor = 10.0
        indoor = 20.0
        
        # Case 1: Instantaneous PV (scalar)
        # Heat gain = 1000 * 0.002 = 2.0 deg
        teq_instant = model.predict_equilibrium_temperature(
            outlet, outdoor, indoor, pv_power=1000.0
        )
        
        # Case 2: Lagged PV (history)
        # History: [0, 0, 0, 1000, 1000, 1000]
        # Lag 60 mins -> Effective PV = 500
        # Heat gain = 500 * 0.002 = 1.0 deg
        history = [0.0, 0.0, 0.0, 1000.0, 1000.0, 1000.0]
        model.solar_lag_minutes = 60.0
        
        teq_lagged = model.predict_equilibrium_temperature(
            outlet, outdoor, indoor, pv_power=history
        )
        
        assert teq_lagged < teq_instant
        
        # Verify exact calculation logic if possible, or just relative direction
        # We expect teq_lagged to be lower because effective PV is lower

    def test_trajectory_prediction_updates_buffer(self, model):
        """Test that trajectory prediction correctly updates the PV buffer."""
        # Setup
        current_indoor = 20.0
        target_indoor = 21.0
        outlet_temp = 40.0
        outdoor_temp = 10.0
        
        # Initial history: 0 PV
        pv_history = [0.0] * 18
        
        # Forecast: PV spikes to 1000 immediately
        pv_forecasts = [1000.0] * 4 # 4 hours of 1000W
        
        model.solar_lag_minutes = 60.0 # 1 hour lag
        
        # Predict trajectory
        result = model.predict_thermal_trajectory(
            current_indoor=current_indoor,
            target_indoor=target_indoor,
            outlet_temp=outlet_temp,
            outdoor_temp=outdoor_temp,
            time_horizon_hours=2.0,
            time_step_minutes=10, # 10 min steps
            pv_power=pv_history,
            pv_forecasts=pv_forecasts
        )
        
        trajectory = result["trajectory"]
        
        # With 60 min lag, the effect of 1000W PV should ramp up slowly
        # Step 1: Buffer = [0...0, 1000]. Avg(last 6) = 1000/6 = 166
        # Step 6: Buffer = [0...0, 1000, 1000, 1000, 1000, 1000, 1000]. Avg = 1000
        
        # We can't easily check internal buffer state, but we can check the trajectory shape
        # It should be convex (accelerating) or linear-ish, not jumping immediately
        
        # Compare with 0 lag
        model.solar_lag_minutes = 0.0
        result_no_lag = model.predict_thermal_trajectory(
            current_indoor=current_indoor,
            target_indoor=target_indoor,
            outlet_temp=outlet_temp,
            outdoor_temp=outdoor_temp,
            time_horizon_hours=2.0,
            time_step_minutes=10,
            pv_power=pv_history,
            pv_forecasts=pv_forecasts
        )
        trajectory_no_lag = result_no_lag["trajectory"]
        
        # The no-lag trajectory should be hotter initially
        assert trajectory_no_lag[0] > trajectory[0]
