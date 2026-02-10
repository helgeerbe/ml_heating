
import pytest
from unittest.mock import Mock, patch
from src.heating_controller import BlockingStateManager


@pytest.fixture
def mock_ha_client():
    return Mock()


class TestBlockingStateManager:
    @pytest.fixture
    def blocking_manager(self):
        return BlockingStateManager()

    def test_grace_period_skipped_in_shadow_mode(
        self, blocking_manager, mock_ha_client
    ):
        """Test that the grace period is skipped when in shadow mode."""
        # Arrange
        state = {"last_is_blocking": True, "last_blocking_end_time": None}

        # Act
        in_grace_period = blocking_manager.handle_grace_period(
            mock_ha_client, state, shadow_mode=True
        )

        # Assert
        assert not in_grace_period

    @patch("src.heating_controller.time.time")
    def test_grace_period_handled_when_blocking_ends(
        self, mock_time, blocking_manager, mock_ha_client
    ):
        """Test that the grace period is handled when blocking ends."""
        # Arrange
        mock_time.return_value = 1000  # Mock current time
        state = {"last_is_blocking": True, "last_blocking_end_time": None}
        mock_ha_client.get_state.return_value = False  # No longer blocking

        with patch.object(
            blocking_manager, "_execute_grace_period"
        ) as mock_execute_grace_period, patch(
            "src.heating_controller.save_state"
        ) as mock_save_state:
            # Act
            in_grace_period = blocking_manager.handle_grace_period(
                mock_ha_client, state, shadow_mode=False
            )

            # Assert
            assert in_grace_period
            mock_execute_grace_period.assert_called_once()
            mock_save_state.assert_called_with(
                last_is_blocking=False, last_blocking_end_time=None
            )

    def test_no_grace_period_if_not_blocking_before(
        self, blocking_manager, mock_ha_client
    ):
        """Test no grace period if not blocking previously."""
        # Arrange
        state = {"last_is_blocking": False}

        # Act
        in_grace_period = blocking_manager.handle_grace_period(
            mock_ha_client, state, shadow_mode=False
        )

        # Assert
        assert not in_grace_period

    @patch("src.heating_controller.config")
    @patch("src.heating_controller.logging")
    def test_execute_grace_period_no_last_final_temp(
        self, mock_logging, mock_config, blocking_manager, mock_ha_client
    ):
        """Test grace period with no last_final_temp in state."""
        # Arrange
        state = {}  # No last_final_temp
        age = 120.0
        mock_config.INDOOR_TEMP_ENTITY_ID = "sensor.indoor_temp"
        mock_ha_client.get_state.return_value = None  # Missing sensors
        mock_ha_client.get_all_states.return_value = {}

        with patch.object(
            blocking_manager, "_wait_for_grace_target"
        ) as mock_wait_for_target:
            # Act
            blocking_manager._execute_grace_period(mock_ha_client, state, age)

            # Assert
            mock_logging.info.assert_any_call(
                "No last_final_temp in persisted state; skipping restore/wait."
            )
            mock_ha_client.set_state.assert_not_called()
            mock_wait_for_target.assert_not_called()

    @patch("src.heating_controller.config")
    @patch("src.heating_controller.logging")
    def test_execute_grace_period_no_actual_outlet_temp(
        self, mock_logging, mock_config, blocking_manager, mock_ha_client
    ):
        """Test grace period when outlet_temp is not available."""
        # Arrange
        state = {"last_final_temp": 38.0}
        age = 120.0
        # All critical sensors are None
        mock_ha_client.get_state.return_value = None
        mock_ha_client.get_all_states.return_value = {}

        # Act
        blocking_manager._execute_grace_period(mock_ha_client, state, age)

        # Assert
        mock_logging.warning.assert_any_call(
            "Cannot read actual_outlet_temp at grace start; skipping wait."
        )
        mock_ha_client.set_state.assert_not_called()
