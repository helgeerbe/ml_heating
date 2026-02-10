
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock, MagicMock

from src import physics_calibration
from src.thermal_equilibrium_model import ThermalEquilibriumModel


@pytest.fixture(autouse=True)
def mock_config():
    with patch('src.physics_calibration.config') as mock_config:
        mock_config.INDOOR_TEMP_ENTITY_ID = "sensor.indoor_temp"
        mock_config.ACTUAL_OUTLET_TEMP_ENTITY_ID = "sensor.outlet_temp"
        mock_config.OUTDOOR_TEMP_ENTITY_ID = "sensor.outdoor_temp"
        mock_config.PV_POWER_ENTITY_ID = "sensor.pv_power"
        mock_config.TV_STATUS_ENTITY_ID = "sensor.tv_on"
        mock_config.FIREPLACE_STATUS_ENTITY_ID = "sensor.fireplace_on"
        mock_config.DHW_STATUS_ENTITY_ID = "sensor.dhw_heating"
        mock_config.DEFROST_STATUS_ENTITY_ID = "sensor.defrosting"
        mock_config.DISINFECTION_STATUS_ENTITY_ID = "sensor.disinfection"
        mock_config.DHW_BOOST_HEATER_STATUS_ENTITY_ID = "sensor.boost_heater"
        mock_config.GRACE_PERIOD_MAX_MINUTES = 30
        mock_config.TRAINING_LOOKBACK_HOURS = 24
        yield mock_config


def test_check_blocking_states():
    df = pd.DataFrame({
        'dhw_heating': [0, 1, 0],
        'defrosting': [0, 0, 0],
        'disinfection': [0, 0, 1],
        'boost_heater': [0, 0, 0]
    })
    blocking, reasons = physics_calibration.check_blocking_states(df, 'dhw_heating', 'defrosting', 'disinfection', 'boost_heater')
    assert blocking
    assert 'dhw_heating' in reasons
    assert 'disinfection' in reasons

    df_no_blocking = pd.DataFrame({
        'dhw_heating': [0, 0, 0],
    })
    blocking, reasons = physics_calibration.check_blocking_states(df_no_blocking, 'dhw_heating', 'defrosting', 'disinfection', 'boost_heater')
    assert not blocking
    assert not reasons


@pytest.fixture
def sample_df():
    base_time = pd.to_datetime('2023-01-01')
    time_series = pd.to_datetime([base_time + pd.Timedelta(minutes=i*5) for i in range(100)])
    data = {
        '_time': time_series,
        'indoor_temp': 21.0,
        'outlet_temp': 45.0,
        'outdoor_temp': 5.0,
        'pv_power': 0.0,
        'fireplace_on': 0.0,
        'tv_on': 0.0,
        'dhw_heating': 0,
        'defrosting': 0,
        'disinfection': 0,
        'boost_heater': 0,
    }
    df = pd.DataFrame(data)
    return df


def test_filter_stable_periods_stable_data(sample_df):
    with patch('src.physics_calibration.log_filtering_stats'), \
         patch('src.physics_calibration.json'), \
         patch('builtins.open'):
        stable_periods = physics_calibration.filter_stable_periods(sample_df)
        assert len(stable_periods) > 0
        for period in stable_periods:
            assert 'stability_score' in period


def test_filter_stable_periods_unstable_temp(sample_df):
    sample_df['indoor_temp'] = np.sin(np.arange(100)) * 5 + 20
    with patch('src.physics_calibration.log_filtering_stats'), \
         patch('src.physics_calibration.json'), \
         patch('builtins.open'):
        stable_periods = physics_calibration.filter_stable_periods(sample_df)
        assert len(stable_periods) == 0


def test_filter_stable_periods_with_blocking(sample_df):
    sample_df.loc[40:60, 'dhw_heating'] = 1
    with patch('src.physics_calibration.log_filtering_stats'), \
         patch('src.physics_calibration.json'), \
         patch('builtins.open'):
        stable_periods = physics_calibration.filter_stable_periods(sample_df)
        assert len(stable_periods) > 0
        assert len(stable_periods) < 83


@pytest.fixture
def stable_periods_fixture():
    return [
        {
            'indoor_temp': 21.0, 'outlet_temp': 40.0, 'outdoor_temp': 5.0,
            'pv_power': 0.0, 'fireplace_on': 0.0, 'tv_on': 0.0
        },
        {
            'indoor_temp': 22.0, 'outlet_temp': 45.0, 'outdoor_temp': 10.0,
            'pv_power': 500.0, 'fireplace_on': 0.0, 'tv_on': 1.0
        }
    ]


def test_calculate_mae_for_params(stable_periods_fixture):
    param_names = ['thermal_time_constant', 'heat_loss_coefficient', 'outlet_effectiveness']
    params = [20.0, 0.5, 0.8]
    current_params = {
        'thermal_time_constant': 20.0, 'heat_loss_coefficient': 0.5, 'outlet_effectiveness': 0.8,
        'pv_heat_weight': 0.001, 'fireplace_heat_weight': 2.0, 'tv_heat_weight': 0.2
    }

    with patch('src.physics_calibration.ThermalEquilibriumModel') as mock_model_class:
        mock_model_instance = mock_model_class.return_value
        mock_model_instance.predict_equilibrium_temperature.return_value = 21.5

        mae = physics_calibration.calculate_mae_for_params(
            params, param_names, stable_periods_fixture, current_params
        )
        assert mae == pytest.approx(0.5)
        assert mock_model_instance.predict_equilibrium_temperature.call_count == 1


def test_build_optimization_params():
    current_params = {
        'thermal_time_constant': 10, 'heat_loss_coefficient': 0.1, 'outlet_effectiveness': 0.5,
        'pv_heat_weight': 0.001, 'fireplace_heat_weight': 2.0, 'tv_heat_weight': 0.5
    }
    excluded_params = ['fireplace_heat_weight']

    with patch('src.physics_calibration.ThermalParameterConfig') as mock_thermal_config:
        mock_thermal_config.get_bounds.return_value = (0, 1)

        names, values, bounds = physics_calibration.build_optimization_params(current_params, excluded_params)

        assert 'fireplace_heat_weight' not in names
        assert 'pv_heat_weight' in names
        assert len(names) == 5
        assert len(values) == 5
        assert len(bounds) == 5
        assert names[3] == 'pv_heat_weight'
        assert values[3] == 0.001


@pytest.fixture
def mock_scipy_minimize():
    with patch('src.physics_calibration.minimize') as mock_minimize:
        yield mock_minimize


@patch('src.physics_calibration.ThermalParameterConfig')
def test_optimize_thermal_parameters(mock_thermal_config, stable_periods_fixture, mock_scipy_minimize):
    mock_scipy_minimize.return_value = MagicMock(success=True, x=[1, 2, 3, 4, 5], fun=0.123)
    mock_thermal_config.get_default.return_value = 0.5
    mock_thermal_config.get_bounds.return_value = (0, 1)

    with patch('src.physics_calibration.debug_thermal_predictions'):
        result = physics_calibration.optimize_thermal_parameters(stable_periods_fixture)

        mock_scipy_minimize.assert_called_once()
        assert result is not None
        assert result['optimization_success'] is True
        assert result['mae'] == 0.123


def test_optimize_thermal_parameters_scipy_disabled(stable_periods_fixture):
     with patch('src.physics_calibration.minimize', None):
         result = physics_calibration.optimize_thermal_parameters(stable_periods_fixture)
         assert result is None

