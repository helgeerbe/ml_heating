
import argparse
import pytest
from unittest.mock import patch, MagicMock, ANY
from src import main, config


@patch("src.main.time.sleep")
@patch("src.main.time")
@patch("src.main.save_state")
def test_poll_for_blocking_starts_blocking(mock_save_state, mock_time, mock_sleep):
    """Test poll_for_blocking detects blocking starts."""
    # Arrange
    mock_ha_client = MagicMock()
    state = {"last_is_blocking": False}
    blocking_entities = ["sensor.blocking_entity"]
    config.CYCLE_INTERVAL_MINUTES = 30
    config.BLOCKING_POLL_INTERVAL_SECONDS = 10

    # Simulate time to run the poll loop twice, with blocking starting on
    # the second poll
    mock_time.time.side_effect = [
        0,  # Initial time for end_time calculation
        10,  # First poll check
        20,  # Second poll check
    ]

    # First poll returns not blocking, second poll returns blocking
    mock_ha_client.get_state.side_effect = [
        False,  # First check for blocking_now
        True,  # Second check for blocking_now
        True,  # Check for blocking_reasons_now
    ]

    # Act
    main.poll_for_blocking(mock_ha_client, state, blocking_entities)

    # Assert
    assert mock_ha_client.get_state.call_count == 3
    mock_save_state.assert_called_once_with(
        last_is_blocking=True,
        last_final_temp=None,
        last_blocking_reasons=["sensor.blocking_entity"],
        last_blocking_end_time=None,
    )


@patch("src.main.time.sleep")
@patch("src.main.time")
@patch("src.main.save_state")
def test_poll_for_blocking_ends_blocking(mock_save_state, mock_time, mock_sleep):
    """Test poll_for_blocking detects blocking ends."""
    # Arrange
    mock_ha_client = MagicMock()
    state = {"last_is_blocking": True}  # Blocking is currently on
    blocking_entities = ["sensor.blocking_entity"]
    config.CYCLE_INTERVAL_MINUTES = 30
    config.BLOCKING_POLL_INTERVAL_SECONDS = 10

    # Simulate time to run the poll loop twice, with blocking ending on the
    # second poll
    mock_time.time.side_effect = [
        0,  # Initial time for end_time calculation
        10,  # First poll check
        20,  # Second poll check
        20,  # Third poll check for save_state
    ]

    # First poll returns blocking, second poll returns not blocking
    mock_ha_client.get_state.side_effect = [
        True,  # First check for blocking_now
        False,  # Second check for blocking_now
    ]

    # Act
    main.poll_for_blocking(mock_ha_client, state, blocking_entities)

    # Assert
    assert mock_ha_client.get_state.call_count == 2
    mock_save_state.assert_called_once_with(
        last_is_blocking=True,
        last_blocking_end_time=20,  # The time of the second check
        last_blocking_reasons=[],
    )


@patch("src.main.time.sleep")
@patch("src.main.time")
@patch("src.main.save_state")
def test_poll_for_blocking_no_change(mock_save_state, mock_time, mock_sleep):
    """Test poll_for_blocking with no blocking state change."""
    # Arrange
    mock_ha_client = MagicMock()
    state = {"last_is_blocking": False}
    blocking_entities = ["sensor.blocking_entity"]
    config.CYCLE_INTERVAL_MINUTES = 30
    config.BLOCKING_POLL_INTERVAL_SECONDS = 10

    # Simulate time so the loop runs a few times then ends
    mock_time.time.side_effect = [
        0,      # Initial time
        10,     # First poll
        20,     # Second poll
        30,     # Third poll
        config.CYCLE_INTERVAL_MINUTES * 60 + 1,  # End loop
    ]

    # Blocking is always off
    mock_ha_client.get_state.return_value = False

    # Act
    main.poll_for_blocking(mock_ha_client, state, blocking_entities)

    # Assert
    mock_save_state.assert_not_called()


@patch("src.main.train_thermal_equilibrium_model")
@patch("src.physics_calibration.backup_existing_calibration")
@patch("src.main.load_dotenv")
@patch("src.main.logging")
@patch("src.main.create_influx_service")
def test_main_calibrate_physics(
    mock_create_influx, mock_logging, mock_load_dotenv, mock_backup, mock_train
):
    """Test main function with --calibrate-physics argument."""
    # Arrange
    mock_backup.return_value = "/fake/path"
    mock_train.return_value = True

    # Act
    with patch("sys.argv", ["main.py", "--calibrate-physics"]):
        main.main()

    # Assert
    mock_backup.assert_called_once()
    mock_train.assert_called_once()


@patch("src.main.validate_thermal_model")
@patch("src.main.load_dotenv")
@patch("src.main.logging")
@patch("src.main.create_influx_service")
def test_main_validate_physics(
    mock_create_influx, mock_logging, mock_load_dotenv, mock_validate
):
    """Test main function with --validate-physics argument."""
    # Arrange
    mock_validate.return_value = True

    # Act
    with patch("sys.argv", ["main.py", "--validate-physics"]):
        main.main()

    # Assert
    mock_validate.assert_called_once()


@patch("src.main.poll_for_blocking")
@patch("src.main.time.sleep")
@patch("src.main.save_state")
@patch("src.main.simplified_outlet_prediction")
@patch("src.main.build_physics_features")
@patch("src.main.create_ha_client")
@patch("src.main.load_state")
@patch("src.main.logging")
@patch("src.main.load_dotenv")
@patch("src.main.create_influx_service")
def test_main_loop_heating_off(
    mock_create_influx,
    mock_load_dotenv,
    mock_logging,
    mock_load_state,
    mock_create_ha_client,
    mock_build_features,
    mock_prediction,
    mock_save_state,
    mock_sleep,
    mock_poll,
):
    """Test main loop skips when heating is off and loop breaks."""
    # Arrange
    mock_ha_client = MagicMock()
    # On the second run of the main loop, create_ha_client will raise an
    # exception. This will be caught, and then poll_for_blocking will be
    # called, which will raise a second exception to stop the test.
    mock_create_ha_client.side_effect = [mock_ha_client, Exception("Stop loop")]

    # Loop 1: heating off, so continue is called.
    mock_ha_client.get_state.side_effect = [
        True,  # ML-Heating enabled
        False,  # No blocking
        False,
        False,
        False,
        "off",  # Heating is off
    ]
    mock_load_state.return_value = {}
    mock_poll.side_effect = StopIteration("Stop test")

    # Act
    with patch("src.main.get_sensor_attributes", return_value={}):
        with pytest.raises(StopIteration, match="Stop test"):
            with patch("sys.argv", ["main.py"]):
                main.main()

    # Assert that the main logic was skipped
    mock_build_features.assert_not_called()
    mock_prediction.assert_not_called()
    mock_save_state.assert_not_called()

    # Assert that the state was set to "Heating off"
    mock_ha_client.set_state.assert_called_once_with(
        "sensor.ml_heating_state", 6, ANY, round_digits=None
    )
    # Assert that the system is idle
    mock_sleep.assert_called_once_with(300)
    # Assert that poll was called once before breaking
    mock_poll.assert_called_once()
