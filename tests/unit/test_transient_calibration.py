import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.physics_calibration import filter_transient_periods, calibrate_transient_parameters, calculate_cooling_time_constant
from src.thermal_equilibrium_model import ThermalEquilibriumModel

@pytest.fixture
def mock_config():
    with patch('src.physics_calibration.config') as mock:
        mock.INDOOR_TEMP_ENTITY_ID = "sensor.indoor_temp"
        mock.ACTUAL_OUTLET_TEMP_ENTITY_ID = "sensor.outlet_temp"
        mock.OUTDOOR_TEMP_ENTITY_ID = "sensor.outdoor_temp"
        mock.PV_POWER_ENTITY_ID = "sensor.pv_power"
        mock.FIREPLACE_STATUS_ENTITY_ID = "binary_sensor.fireplace"
        mock.TV_STATUS_ENTITY_ID = "binary_sensor.tv"
        mock.DHW_STATUS_ENTITY_ID = "sensor.dhw_heating"
        yield mock

@pytest.fixture
def transient_df():
    # Create a DataFrame with a clear heating transient
    # Temperature rises from 20.0 to 21.0 over 1 hour
    timestamps = pd.date_range(start="2024-01-01 12:00", periods=13, freq="5min")
    
    data = {
        "_time": timestamps,
        "indoor_temp": np.linspace(20.0, 21.0, 13),
        "outlet_temp": [40.0] * 13,  # High outlet temp (heating)
        "outdoor_temp": [5.0] * 13,
        "pv_power": [0.0] * 13,
        "fireplace": [0.0] * 13,
        "tv": [0.0] * 13
    }
    return pd.DataFrame(data)

def test_filter_transient_periods(mock_config, transient_df):
    """Test that transient periods are correctly identified."""
    samples = filter_transient_periods(transient_df)
    
    # We expect samples because:
    # 1. Outlet (40) > Indoor (20-21) + 5
    # 2. Temp is changing
    assert len(samples) > 0
    
    # Check sample structure
    sample = samples[0]
    assert 'current_indoor' in sample
    assert 'next_indoor' in sample
    assert 'outlet_temp' in sample
    assert 'time_step_hours' in sample
    
    # Check values
    assert sample['outlet_temp'] == 40.0
    assert abs(sample['time_step_hours'] - 5/60) < 0.001

def test_calibrate_transient_parameters(mock_config):
    """Test that calibration finds a reasonable time constant."""
    
    # Mock model
    model = MagicMock(spec=ThermalEquilibriumModel)
    model.thermal_time_constant = 4.0
    
    # Setup:
    # T_start = 20.0
    # T_eq = 40.0 (High equilibrium due to heating)
    # dt = 1.0 hour
    # T_end = 25.0
    #
    # Physics: T_end = T_start + (T_eq - T_start) * (1 - exp(-dt/tau))
    # 25 = 20 + (40 - 20) * (1 - exp(-1/tau))
    # 5 = 20 * (1 - exp(-1/tau))
    # 0.25 = 1 - exp(-1/tau)
    # exp(-1/tau) = 0.75
    # -1/tau = ln(0.75) ≈ -0.2877
    # tau = 1 / 0.2877 ≈ 3.47 hours
    
    # We need the model to predict T_eq = 40.0
    model.predict_equilibrium_temperature.return_value = 40.0
    
    samples = [{
        'current_indoor': 20.0,
        'next_indoor': 25.0,
        'outlet_temp': 50.0, # Arbitrary, model mock returns 40.0 eq
        'outdoor_temp': 0.0,
        'pv_power': 0,
        'fireplace_on': 0,
        'tv_on': 0,
        'time_step_hours': 1.0
    }]
    
    # Run calibration
    # We need to patch minimize to use the real scipy minimize or a mock that behaves like it
    # But here we want to test the logic, so let's use the real one if available, or skip
    try:
        from scipy.optimize import minimize
    except ImportError:
        pytest.skip("scipy not available")
        
    best_tau = calibrate_transient_parameters(model, samples)
    
    assert best_tau is not None
    # Should be close to 3.47
    assert 3.0 < best_tau < 4.0


def test_calculate_cooling_time_constant(mock_config):
    """Test cooling time constant calculation."""
    
    # Create synthetic cooling data
    # T(t) = T_out + (T_start - T_out) * exp(-t/tau)
    # Let tau = 10h, T_out = 5, T_start = 25
    tau_true = 10.0
    t_out = 5.0
    t_start = 25.0
    
    timestamps = pd.date_range(start="2024-01-01 12:00", periods=24, freq="10min")
    times_h = np.arange(24) * 10 / 60.0
    
    temps = t_out + (t_start - t_out) * np.exp(-times_h / tau_true)
    
    data = {
        "_time": timestamps,
        "indoor_temp": temps,
        "outlet_temp": [20.0] * 24, # Low outlet (pump off)
        "outdoor_temp": [t_out] * 24,
        "dhw_heating": [1] * 24, # DHW active (heating paused)
        "defrosting": [0] * 24,
        "disinfection": [0] * 24,
        "boost_heater": [0] * 24
    }
    df = pd.DataFrame(data)
    
    tau, r2 = calculate_cooling_time_constant(df)
    
    assert tau is not None
    assert 9.0 < tau < 11.0
    assert r2 > 0.95
