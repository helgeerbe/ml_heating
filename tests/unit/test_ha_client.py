
import pytest
from unittest.mock import patch, MagicMock
from src.ha_client import get_sensor_attributes, create_ha_client


@pytest.fixture
def mock_config():
    """Fixture to mock the config module."""
    with patch('src.ha_client.config') as mock_config:
        mock_config.HASS_URL = "http://localhost:8123"
        mock_config.HASS_TOKEN = "test-token"
        mock_config.MAE_ENTITY_ID = "sensor.ml_model_mae"
        mock_config.RMSE_ENTITY_ID = "sensor.ml_model_rmse"
        yield mock_config


@pytest.fixture
def ha_client(mock_config):
    """Returns an initialized HAClient instance."""
    return create_ha_client()


@patch('src.ha_client.requests')
def test_get_all_states_success(mock_requests, ha_client):
    """Test successful retrieval of all states."""
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {"entity_id": "sensor.temp1", "state": "20"},
        {"entity_id": "sensor.temp2", "state": "30"},
    ]
    mock_requests.get.return_value = mock_response

    states = ha_client.get_all_states()
    assert len(states) == 2
    assert states["sensor.temp1"]["state"] == "20"


@patch('src.ha_client.requests')
def test_get_state_from_cache(mock_requests, ha_client):
    """Test retrieving state from cache."""
    states_cache = {"sensor.temp1": {"state": "25"}}
    state = ha_client.get_state("sensor.temp1", states_cache)
    assert state == 25.0


@patch('src.ha_client.requests')
def test_set_state_success(mock_requests, ha_client):
    """Test successfully setting a state."""
    ha_client.set_state("sensor.test", 22.5)
    mock_requests.post.assert_called_once()


def test_get_calibrated_hourly_forecast(ha_client):
    """Test forecast calibration logic."""
    with patch.object(
        ha_client, 'get_hourly_forecast', return_value=[10, 12, 14, 11]
    ):
        calibrated = ha_client.get_calibrated_hourly_forecast(11.5)
        assert calibrated == [11.5, 13.5, 15.5, 12.5]


@patch('src.ha_client.requests')
def test_log_feature_importance(mock_requests, ha_client):
    """Test logging feature importance."""
    ha_client.log_feature_importance({"feat1": 0.8, "feat2": 0.2})
    mock_requests.post.assert_called()


@patch('src.ha_client.requests')
def test_log_model_metrics(mock_requests, ha_client):
    """Test logging MAE and RMSE metrics."""
    ha_client.log_model_metrics(0.1, 0.2)
    assert mock_requests.post.call_count == 2


@patch('src.ha_client.HAClient.set_state')
def test_log_adaptive_learning_metrics(mock_set_state, ha_client):
    """Test logging of adaptive learning metrics."""
    metrics = {
        "learning_confidence": 0.95,
        "mae_all_time": 0.1,
        "rmse_all_time": 0.2,
    }
    ha_client.log_adaptive_learning_metrics(metrics)
    assert mock_set_state.call_count == 4


def test_get_sensor_attributes():
    """Test the retrieval of sensor attributes."""
    attrs = get_sensor_attributes("sensor.ml_model_mae")
    assert attrs["friendly_name"] == "ML Model MAE"
