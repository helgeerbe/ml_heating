import pytest
from unittest.mock import MagicMock
import pandas as pd
from src.physics_features import build_physics_features
from src import config
from src.sensor_buffer import SensorBuffer

@pytest.fixture
def mock_ha_client():
    client = MagicMock()
    client.get_all_states.return_value = {}
    # Default return values for get_state
    # We will override side_effect in specific tests if needed, 
    # but here we provide a callable or a default to avoid errors.
    
    def get_state_side_effect(entity_id, all_states, is_binary=False):
        if entity_id == config.INDOOR_TEMP_ENTITY_ID:
            return 20.0 # Raw value
        if entity_id == config.OUTDOOR_TEMP_ENTITY_ID:
            return 5.0 # Raw value
        if entity_id == config.ACTUAL_OUTLET_TEMP_ENTITY_ID:
            return 40.0 # Raw value
        if entity_id == config.TARGET_INDOOR_TEMP_ENTITY_ID:
            return 21.0
        if entity_id == config.INLET_TEMP_ENTITY_ID:
            return 35.0
        if entity_id == config.FLOW_RATE_ENTITY_ID:
            return 1000.0
        if entity_id == config.POWER_CONSUMPTION_ENTITY_ID:
            return 1500.0
        if is_binary:
            return False
        return 0.0

    client.get_state.side_effect = get_state_side_effect
    client.get_calibrated_hourly_forecast.return_value = [6.0, 7.0, 8.0, 9.0]
    return client

@pytest.fixture
def mock_influx_service():
    service = MagicMock()
    service.fetch_outlet_history.return_value = [35.0, 36.0, 37.0, 38.0, 39.0, 40.0]
    service.fetch_indoor_history.return_value = [19.0, 19.2, 19.4, 19.6, 19.8, 20.0]
    return service

@pytest.fixture
def mock_sensor_buffer():
    buffer = MagicMock(spec=SensorBuffer)
    return buffer

def test_smoothing_override(mock_ha_client, mock_influx_service, mock_sensor_buffer):
    """Test that sensor buffer averages override raw HA values."""
    
    # Setup buffer to return different values than HA
    def get_average_side_effect(entity_id, window_minutes):
        if entity_id == config.INDOOR_TEMP_ENTITY_ID:
            return 20.5 # Smoothed value (Raw is 20.0)
        if entity_id == config.OUTDOOR_TEMP_ENTITY_ID:
            return 5.5 # Smoothed value (Raw is 5.0)
        if entity_id == config.ACTUAL_OUTLET_TEMP_ENTITY_ID:
            return 40.5 # Smoothed value (Raw is 40.0)
        if entity_id == config.INLET_TEMP_ENTITY_ID:
            return 35.5 # Smoothed value (Raw is 35.0)
        if entity_id == config.FLOW_RATE_ENTITY_ID:
            return 1050.0 # Smoothed value (Raw is 1000.0)
        return None

    mock_sensor_buffer.get_average.side_effect = get_average_side_effect

    features_df, _ = build_physics_features(
        mock_ha_client, 
        mock_influx_service, 
        sensor_buffer=mock_sensor_buffer
    )

    assert features_df is not None
    
    # Check if smoothed values were used in features
    # Note: features_df keys might be slightly different from entity IDs, check mapping in build_physics_features
    
    # 'temp_diff_indoor_outdoor': actual_indoor_f - outdoor_temp_f
    # 20.5 - 5.5 = 15.0
    assert features_df['temp_diff_indoor_outdoor'][0] == 15.0
    
    # 'outlet_indoor_diff': outlet_temp_f - actual_indoor_f
    # 40.5 - 20.5 = 20.0
    assert features_df['outlet_indoor_diff'][0] == 20.0
    
    # Check thermodynamic features directly
    assert features_df['inlet_temp'][0] == 35.5
    assert features_df['flow_rate'][0] == 1050.0
    
    # Verify buffer was called with correct windows
    mock_sensor_buffer.get_average.assert_any_call(config.INDOOR_TEMP_ENTITY_ID, 15)
    mock_sensor_buffer.get_average.assert_any_call(config.OUTDOOR_TEMP_ENTITY_ID, 30)
    mock_sensor_buffer.get_average.assert_any_call(config.ACTUAL_OUTLET_TEMP_ENTITY_ID, 5)

def test_smoothing_fallback(mock_ha_client, mock_influx_service, mock_sensor_buffer):
    """Test that we fall back to raw values if buffer returns None."""
    
    # Buffer returns None for everything
    mock_sensor_buffer.get_average.return_value = None

    features_df, _ = build_physics_features(
        mock_ha_client, 
        mock_influx_service, 
        sensor_buffer=mock_sensor_buffer
    )

    assert features_df is not None
    
    # Should use raw values from mock_ha_client
    # Indoor: 20.0, Outdoor: 5.0
    # Diff: 15.0
    assert features_df['temp_diff_indoor_outdoor'][0] == 15.0
    
    # Outlet: 40.0, Indoor: 20.0
    # Diff: 20.0
    assert features_df['outlet_indoor_diff'][0] == 20.0
    
    assert features_df['inlet_temp'][0] == 35.0
    assert features_df['flow_rate'][0] == 1000.0

def test_no_buffer_provided(mock_ha_client, mock_influx_service):
    """Test that it works fine without a buffer (None)."""
    
    features_df, _ = build_physics_features(
        mock_ha_client, 
        mock_influx_service, 
        sensor_buffer=None
    )

    assert features_df is not None
    
    # Should use raw values
    assert features_df['temp_diff_indoor_outdoor'][0] == 15.0
