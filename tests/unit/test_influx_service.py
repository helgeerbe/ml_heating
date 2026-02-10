
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime, timezone

from src.influx_service import (
    InfluxService,
    get_influx_service,
    reset_influx_service,
)


@pytest.fixture
def mock_config():
    """Fixture to mock the config module."""
    with patch('src.influx_service.config') as mock_config:
        mock_config.INFLUX_URL = "http://localhost:8086"
        mock_config.INFLUX_TOKEN = "test-token"
        mock_config.INFLUX_ORG = "test-org"
        mock_config.INFLUX_BUCKET = "test-bucket"
        mock_config.HISTORY_STEP_MINUTES = 5
        mock_config.ACTUAL_OUTLET_TEMP_ENTITY_ID = "sensor.outlet_temp"
        mock_config.INDOOR_TEMP_ENTITY_ID = "sensor.indoor_temp"
        yield mock_config


@pytest.fixture
def influx_service(mock_config):
    """Returns an initialized InfluxService instance."""
    return InfluxService(
        url="http://localhost:8086", token="test-token", org="test-org"
    )


def test_get_pv_forecast_success(influx_service):
    """Test successful retrieval of PV forecast."""
    mock_df = pd.DataFrame({
        "_time": pd.to_datetime(
            ["2023-01-01T13:00:00Z", "2023-01-01T14:00:00Z"]
        ),
        "total": [100.0, 200.0]
    })
    influx_service.query_api.query_data_frame = MagicMock(return_value=mock_df)

    mock_now = datetime(2023, 1, 1, 12, 1, 0, tzinfo=timezone.utc)
    with patch('src.influx_service.datetime') as mock_datetime:
        mock_datetime.now.return_value = mock_now
        result = influx_service.get_pv_forecast()
        assert len(result) == 4
        assert result[0] == 100.0
        assert result[1] == 200.0


def test_fetch_history_success(influx_service):
    """Test successful fetching of historical data."""
    mock_df = pd.DataFrame({"value": [20.0, 21.0, 22.0]})
    influx_service.query_api.query_data_frame = MagicMock(return_value=mock_df)
    result = influx_service.fetch_history("sensor.test", 5, 19.0)
    assert len(result) == 5
    assert result == [20.0, 21.0, 22.0, 22.0, 22.0]


def test_write_metrics(influx_service):
    """Test writing various metrics to InfluxDB."""
    influx_service.write_api.write = MagicMock()

    # Test write_feature_importances
    influx_service.write_feature_importances({"feat1": 0.5})
    influx_service.write_api.write.assert_called()

    # Test write_prediction_metrics
    influx_service.write_prediction_metrics({"1h": {"mae": 0.1}})
    influx_service.write_api.write.assert_called()

    # Test write_thermal_learning_metrics
    mock_model = MagicMock()
    mock_model.get_adaptive_learning_metrics.return_value = {
        "current_parameters": {}
    }
    influx_service.write_thermal_learning_metrics(mock_model)
    influx_service.write_api.write.assert_called()


def test_singleton_logic(mock_config):
    """Test the singleton creation and reset logic."""
    reset_influx_service()
    instance1 = get_influx_service()
    instance2 = get_influx_service()
    assert instance1 is instance2
    reset_influx_service()
    instance3 = get_influx_service()
    assert instance1 is not instance3
