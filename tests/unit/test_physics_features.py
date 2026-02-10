
import pytest
import pandas as pd
from unittest.mock import MagicMock
from src.physics_features import build_physics_features


@pytest.fixture
def mock_ha_client():
    client = MagicMock()
    client.get_all_states.return_value = {}
    client.get_state.side_effect = [20.0, 5.0, 40.0, 21.0, True, False, False, True, 500.0, True, False]
    client.get_calibrated_hourly_forecast.return_value = [6.0, 7.0, 8.0, 9.0]
    return client


@pytest.fixture
def mock_influx_service():
    service = MagicMock()
    service.fetch_outlet_history.return_value = [35.0, 36.0, 37.0, 38.0, 39.0, 40.0]
    service.fetch_indoor_history.return_value = [19.0, 19.2, 19.4, 19.6, 19.8, 20.0]
    return service


def test_build_physics_features_success(mock_ha_client, mock_influx_service):
    """Test successful build of physics features."""
    features_df, _ = build_physics_features(mock_ha_client, mock_influx_service)
    assert isinstance(features_df, pd.DataFrame)
    assert len(features_df.columns) == 37
    assert features_df['indoor_temp_lag_30m'][0] == 19.6


def test_build_physics_features_missing_data(mock_ha_client, mock_influx_service):
    """Test feature building with missing critical data."""
    mock_ha_client.get_state.side_effect = [None, 5.0, 40.0, 21.0]
    features_df, _ = build_physics_features(mock_ha_client, mock_influx_service)
    assert features_df is None


def test_build_physics_features_insufficient_history(mock_ha_client, mock_influx_service):
    """Test feature building with insufficient history."""
    mock_influx_service.fetch_indoor_history.return_value = [19.8, 20.0]
    features_df, _ = build_physics_features(mock_ha_client, mock_influx_service)
    assert features_df is None

