
import pytest
from unittest.mock import MagicMock, patch
from src.heating_controller import (
    SensorDataManager,
    BlockingStateManager,
    HeatingSystemStateChecker
)
from src.state_manager import SystemState


class TestHeatingComponentsSociable:
    """
    Sociable unit tests for Heating Controller components.

    These tests use real instances of helper classes
    mocking only the external boundaries (HAClient, SystemState).
    """

    @pytest.fixture
    def mock_ha_client(self):
        return MagicMock()

    @pytest.fixture
    def mock_system_state(self):
        return MagicMock(spec=SystemState)

    @pytest.fixture
    def sensor_manager(self):
        return SensorDataManager()

    @pytest.fixture
    def blocking_manager(self):
        return BlockingStateManager()

    @pytest.fixture
    def state_checker(self):
        return HeatingSystemStateChecker()

    def test_sensor_manager_retrieval(self, sensor_manager, mock_ha_client):
        """Test SensorDataManager retrieves and structures data correctly."""
        # Patch config to use predictable entity IDs
        with patch("src.heating_controller.config") as mock_config:
            mock_config.TARGET_INDOOR_TEMP_ENTITY_ID = "sensor.target_indoor_temp"
            mock_config.INDOOR_TEMP_ENTITY_ID = "sensor.indoor_temp"
            mock_config.OUTDOOR_TEMP_ENTITY_ID = "sensor.outdoor_temp"
            mock_config.ACTUAL_OUTLET_TEMP_ENTITY_ID = "sensor.outlet_temp"
            mock_config.AVG_OTHER_ROOMS_TEMP_ENTITY_ID = "sensor.avg_other_rooms"
            mock_config.OPENWEATHERMAP_TEMP_ENTITY_ID = "sensor.openweathermap"
            mock_config.FIREPLACE_STATUS_ENTITY_ID = "binary_sensor.fireplace"

            def get_state_side_effect(entity_id, states, is_binary=False):
                if entity_id == "sensor.indoor_temp":
                    return 20.5
                if entity_id == "sensor.target_indoor_temp":
                    return 21.0
                if entity_id == "sensor.outdoor_temp":
                    return 5.0
                if entity_id == "sensor.outlet_temp":
                    return 35.0
                if entity_id == "sensor.avg_other_rooms":
                    return 20.0
                if entity_id == "sensor.openweathermap":
                    return 4.0
                if entity_id == "binary_sensor.fireplace":
                    return False
                return None

            mock_ha_client.get_state.side_effect = get_state_side_effect
            mock_ha_client.get_all_states.return_value = {}

            data, missing = sensor_manager.get_critical_sensors(
                mock_ha_client, {}
            )

            assert missing == []
            assert data["actual_indoor"] == 20.5
            assert data["outdoor_temp"] == 5.0

    def test_blocking_manager_detection(
        self, blocking_manager, mock_ha_client
    ):
        """Test BlockingStateManager detects blocking entities."""
        # Mock get_state to return True for a blocking entity
        def get_state_side_effect(entity_id, states, is_binary=False):
            if "defrost" in entity_id:
                return True
            return False

        mock_ha_client.get_state.side_effect = get_state_side_effect
        mock_ha_client.get_all_states.return_value = {}

        is_blocking, reasons = blocking_manager.check_blocking_state(
            mock_ha_client, {}
        )

        assert is_blocking is True
        assert len(reasons) > 0

    def test_state_checker_active(self, state_checker, mock_ha_client):
        """Test HeatingSystemStateChecker validates heating state."""
        mock_ha_client.get_state.return_value = "heat"

        is_active = state_checker.check_heating_active(mock_ha_client, {})
        assert is_active is True

    def test_state_checker_inactive(self, state_checker, mock_ha_client):
        """Test HeatingSystemStateChecker detects inactive state."""
        mock_ha_client.get_state.return_value = "off"

        is_active = state_checker.check_heating_active(mock_ha_client, {})
        assert is_active is False
