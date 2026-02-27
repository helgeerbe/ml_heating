import pytest
from unittest.mock import Mock, patch
from src.heating_controller import (
    BlockingStateManager, SensorDataManager, HeatingSystemStateChecker
)
from src.state_manager import SystemState


# Mock HAClient for testing
@pytest.fixture
def mock_ha_client():
    return Mock()


class TestBlockingStateManager:
    @pytest.fixture
    def blocking_manager(self):
        return BlockingStateManager()

    def test_check_blocking_state_no_blocking(
            self, blocking_manager, mock_ha_client):
        mock_ha_client.get_state.return_value = False
        is_blocking, reasons = blocking_manager.check_blocking_state(
            mock_ha_client, {})
        assert is_blocking is False
        assert len(reasons) == 0

    def test_check_blocking_state_with_blocking(
            self, blocking_manager, mock_ha_client):
        mock_ha_client.get_state.return_value = True
        is_blocking, reasons = blocking_manager.check_blocking_state(
            mock_ha_client, {})
        assert is_blocking is True
        assert len(reasons) > 0

    def test_handle_blocking_state(self, blocking_manager, mock_ha_client):
        with patch('src.heating_controller.save_state') as mock_save_state:
            state = SystemState()
            skip_cycle = blocking_manager.handle_blocking_state(
                mock_ha_client, True, ["reason"], state)
            assert skip_cycle is True
            mock_save_state.assert_called_once()

    def test_handle_blocking_state_ha_error(
        self, blocking_manager, mock_ha_client
    ):
        """Test handle_blocking_state handles Home Assistant errors."""
        mock_ha_client.set_state.side_effect = Exception("HA API Error")
        with patch(
            "src.heating_controller.save_state"
        ) as mock_save_state, patch(
            "src.heating_controller.logging"
        ) as mock_logging:
            state = SystemState()
            skip_cycle = blocking_manager.handle_blocking_state(
                mock_ha_client, True, ["reason"], state
            )
            assert skip_cycle is True
            mock_save_state.assert_called_once()
            mock_logging.debug.assert_called_with(
                "Failed to write BLOCKED state to HA.", exc_info=True
            )

    def test_handle_grace_period_shadow_mode(
        self, blocking_manager, mock_ha_client
    ):
        """Test that grace period is skipped in shadow mode."""
        with patch("src.heating_controller.config.SHADOW_MODE", True):
            state = SystemState()
            skip_cycle = blocking_manager.handle_grace_period(
                mock_ha_client, state
            )
            assert skip_cycle is False

    def test_handle_grace_period_not_blocking_before(
        self, blocking_manager, mock_ha_client
    ):
        """Test grace period is skipped if not blocking before."""
        state = SystemState(last_is_blocking=False)
        skip_cycle = blocking_manager.handle_grace_period(
            mock_ha_client, state
        )
        assert skip_cycle is False

    @patch("src.heating_controller.time.time")
    def test_handle_grace_period_expired(
        self, mock_time, blocking_manager, mock_ha_client
    ):
        """Test grace period is skipped if it has expired."""
        mock_time.return_value = 10000
        state = SystemState(last_is_blocking=True, last_blocking_end_time=100)
        blocking_manager.check_blocking_state = Mock(return_value=(False, []))

        with patch(
            "src.heating_controller.config.GRACE_PERIOD_MAX_MINUTES", 5
        ):
            skip_cycle = blocking_manager.handle_grace_period(
                mock_ha_client, state
            )
        assert skip_cycle is False

    @patch("src.heating_controller.BlockingStateManager._execute_grace_period")
    @patch("src.heating_controller.time.time")
    def test_handle_grace_period_starts(
        self, mock_time, mock_execute, blocking_manager, mock_ha_client
    ):
        """Test that a grace period is started."""
        mock_time.return_value = 100
        state = SystemState(last_is_blocking=True)
        blocking_manager.check_blocking_state = Mock(return_value=(False, []))

        with patch("src.heating_controller.save_state") as mock_save:
            skip = blocking_manager.handle_grace_period(mock_ha_client, state)
            assert skip is True
            mock_save.assert_any_call(last_blocking_end_time=100)
            mock_execute.assert_called_once()

    @patch("src.heating_controller.time.time", return_value=100)
    @patch("src.heating_controller.get_sensor_attributes", return_value={})
    @patch(
        "src.heating_controller.BlockingStateManager._wait_for_grace_target"
    )
    def test_grace_period_fallback(
        self,
        mock_wait,
        mock_attrs,
        mock_time,
        blocking_manager,
        mock_ha_client,
    ):
        """Test grace period fallback logic when sensors are missing."""
        state = SystemState(
            last_is_blocking=True,
            last_blocking_end_time=90,
            last_final_temp=42.0
        )
        blocking_manager.check_blocking_state = Mock(return_value=(False, []))
        mock_ha_client.get_state.side_effect = [None, None, None, 40.0]

        with patch("src.heating_controller.save_state"):
            blocking_manager.handle_grace_period(mock_ha_client, state)
            mock_ha_client.set_state.assert_called_with(
                'sensor.ml_vorlauftemperatur', 42.0, {}, round_digits=0
            )
            mock_wait.assert_called_once()

    @patch("src.heating_controller.time.time", return_value=100)
    @patch("src.model_wrapper.get_enhanced_model_wrapper")
    @patch("src.physics_features.build_physics_features")
    @patch("src.influx_service.create_influx_service")
    def test_grace_period_intelligent_recovery_no_wait(
        self,
        mock_influx,
        mock_build,
        mock_wrapper_getter,
        mock_time,
        blocking_manager,
        mock_ha_client,
    ):
        """Test intelligent recovery when outlet temp is close to target."""
        state = SystemState(last_is_blocking=True, last_blocking_end_time=90)
        blocking_manager.check_blocking_state = Mock(return_value=(False, []))
        mock_ha_client.get_state.side_effect = [21.0, 22.0, 5.0, 44.8]

        mock_wrapper = Mock()
        mock_wrapper._calculate_required_outlet_temp.return_value = 45.0
        mock_wrapper_getter.return_value = mock_wrapper
        
        mock_df = Mock()
        mock_df.empty = False
        # Mock iloc[0].to_dict()
        mock_row = Mock()
        mock_row.to_dict.return_value = {"some": "features"}
        # Configure iloc to return mock_row when accessed with index 0
        mock_df.iloc = [mock_row]
        
        mock_build.return_value = (mock_df, {})
    
        with patch("src.heating_controller.save_state"):
            blocking_manager.handle_grace_period(mock_ha_client, state)
            mock_ha_client.set_state.assert_not_called()

    @patch("src.heating_controller.time.sleep")
    @patch("src.heating_controller.time.time")
    def test_wait_for_grace_target_timeout(
        self, mock_time, mock_sleep, blocking_manager, mock_ha_client
    ):
        """Test that the grace period wait times out."""
        mock_time.side_effect = [1000, 1001, 1000 + 301]
        blocking_manager.check_blocking_state = Mock(return_value=(False, []))
        mock_ha_client.get_state.return_value = 50.0

        with patch(
            "src.heating_controller.config.GRACE_PERIOD_MAX_MINUTES", 5
        ):
            blocking_manager._wait_for_grace_target(
                mock_ha_client, 40.0, True
            )
        assert mock_sleep.called

    @patch("src.heating_controller.time.sleep")
    @patch("src.heating_controller.time.time", return_value=1000)
    def test_wait_for_grace_target_reaches_target(
        self, mock_time, mock_sleep, blocking_manager, mock_ha_client
    ):
        """Test wait ends when target is reached."""
        blocking_manager.check_blocking_state = Mock(return_value=(False, []))
        mock_ha_client.get_state.side_effect = [50.0, 40.0]

        blocking_manager._wait_for_grace_target(mock_ha_client, 40.0, True)
        mock_sleep.assert_called_once()

    @patch("src.heating_controller.time.sleep")
    @patch("src.heating_controller.time.time", return_value=1000)
    def test_wait_for_grace_target_blocking_reappears(
        self, mock_time, mock_sleep, blocking_manager, mock_ha_client
    ):
        """Test wait ends when blocking reappears."""
        blocking_manager.check_blocking_state = Mock(
            side_effect=[(False, []), (True, ["reason"])]
        )
        mock_ha_client.get_state.return_value = 50.0

        with patch("src.heating_controller.save_state") as mock_save:
            blocking_manager._wait_for_grace_target(mock_ha_client, 40.0, True)
            mock_sleep.assert_called_once()
            mock_save.assert_called_once()

    @patch("src.heating_controller.get_sensor_attributes", return_value={})
    @patch("src.heating_controller.time.sleep")
    @patch("src.heating_controller.time.time")
    def test_wait_for_grace_target_dynamic_update(
        self,
        mock_time,
        mock_sleep,
        mock_get_attrs,
        blocking_manager,
        mock_ha_client,
    ):
        """Test that grace target is dynamically updated during wait."""
        from src.heating_controller import config

        # Setup time
        start_time = 1000
        mock_time.side_effect = [
            start_time,  # Start
            start_time + 10,  # Loop 1
            start_time + 20,  # Loop 2
            start_time + 30,  # Loop 3
            start_time + 40,  # Loop 4
            start_time + 50,  # Loop 5
        ]

        # Mock blocking check to avoid complexity with blocking entities
        blocking_manager.check_blocking_state = Mock(return_value=(False, []))

        # Setup get_state to look up in the dict passed as 2nd arg
        def get_state_side_effect(
            entity_id, all_states, is_binary=False, round_digits=None
        ):
            if isinstance(all_states, dict):
                return all_states.get(entity_id)
            return None

        mock_ha_client.get_state.side_effect = get_state_side_effect

        # Base state
        base_state = {
            config.TARGET_INDOOR_TEMP_ENTITY_ID: 21.0,
            config.OUTDOOR_TEMP_ENTITY_ID: 5.0,
        }

        # Loop 1: Initial. Target 40.0. Actual 30.0.
        state_1 = base_state.copy()
        state_1.update({
            config.ACTUAL_OUTLET_TEMP_ENTITY_ID: 30.0,
            config.INDOOR_TEMP_ENTITY_ID: 20.0,
        })

        # Loop 2: Small change. Indoor 20.1.
        state_2 = base_state.copy()
        state_2.update({
            config.ACTUAL_OUTLET_TEMP_ENTITY_ID: 30.0,
            config.INDOOR_TEMP_ENTITY_ID: 20.1,
        })

        # Loop 3: Big change. Indoor 19.0.
        state_3 = base_state.copy()
        state_3.update({
            config.ACTUAL_OUTLET_TEMP_ENTITY_ID: 30.0,
            config.INDOOR_TEMP_ENTITY_ID: 19.0,
        })

        # Loop 4: Target reached. Actual 41.0.
        state_4 = base_state.copy()
        state_4.update({
            config.ACTUAL_OUTLET_TEMP_ENTITY_ID: 41.0,
            config.INDOOR_TEMP_ENTITY_ID: 19.0,
        })

        mock_ha_client.get_all_states.side_effect = [
            state_1,
            state_2,
            state_3,
            state_4,
        ]

        # Mock wrapper
        mock_wrapper = Mock()
        mock_features = {"some": "features"}

        # calculate_required_outlet_temp side effects
        mock_wrapper._calculate_required_outlet_temp.side_effect = [
            40.0,  # Loop 1: Matches initial (40.0). No update.
            40.2,  # Loop 2: Diff 0.2 < 0.5. No update.
            41.0,  # Loop 3: Diff 0.8 >= 0.5. Update to 41.0.
            41.0,  # Loop 4: Same as current (41.0). No update.
        ]

        # Run
        blocking_manager._wait_for_grace_target(
            mock_ha_client,
            initial_grace_target=40.0,
            wait_for_cooling=False,
            wrapper=mock_wrapper,
            thermal_features=mock_features,
        )

        # Verify
        # Should have updated target exactly once (in Loop 3)
        mock_ha_client.set_state.assert_called_once()
        args, _ = mock_ha_client.set_state.call_args
        assert args[0] == config.TARGET_OUTLET_TEMP_ENTITY_ID
        assert args[1] == 41.0


class TestSensorDataManager:
    @pytest.fixture
    def sensor_manager(self):
        return SensorDataManager()

    def test_get_critical_sensors_all_present(
            self, sensor_manager, mock_ha_client):
        mock_ha_client.get_state.return_value = 1.0
        sensor_data, missing = sensor_manager.get_critical_sensors(
            mock_ha_client, {})
        assert sensor_data is not None
        assert len(missing) == 0

    def test_get_critical_sensors_some_missing(
            self, sensor_manager, mock_ha_client):
        mock_ha_client.get_state.side_effect = [
            1.0, None, 1.0, 1.0, True, 1.0, 1.0]
        sensor_data, missing = sensor_manager.get_critical_sensors(
            mock_ha_client, {})
        assert sensor_data is None
        assert len(missing) > 0

    def test_handle_missing_sensors_ha_error(
        self, sensor_manager, mock_ha_client
    ):
        """Test handle_missing_sensors handles Home Assistant errors."""
        mock_ha_client.set_state.side_effect = Exception("HA API Error")
        with patch("src.heating_controller.logging") as mock_logging:
            skip_cycle = sensor_manager.handle_missing_sensors(
                mock_ha_client, ["sensor.missing"]
            )
            assert skip_cycle is True
            mock_logging.debug.assert_called_with(
                "Failed to write NO_DATA state to HA.", exc_info=True
            )


class TestHeatingSystemStateChecker:
    @pytest.fixture
    def state_checker(self):
        return HeatingSystemStateChecker()

    def test_check_heating_active(self, state_checker, mock_ha_client):
        mock_ha_client.get_state.return_value = "heat"
        assert state_checker.check_heating_active(mock_ha_client, {}) is True

    def test_check_heating_inactive(self, state_checker, mock_ha_client):
        mock_ha_client.get_state.return_value = "off"
        assert state_checker.check_heating_active(mock_ha_client, {}) is False

    def test_check_heating_inactive_ha_error(
        self, state_checker, mock_ha_client
    ):
        """Test check_heating_active handles HA errors when inactive."""
        mock_ha_client.get_state.return_value = "off"
        mock_ha_client.set_state.side_effect = Exception("HA API Error")
        with patch("src.heating_controller.logging") as mock_logging:
            should_continue = state_checker.check_heating_active(
                mock_ha_client, {}
            )
            assert should_continue is False
            mock_logging.debug.assert_called_with(
                "Failed to write HEATING_OFF state to HA.", exc_info=True
            )
