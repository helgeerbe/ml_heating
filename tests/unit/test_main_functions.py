
import argparse
import pytest
from unittest.mock import patch, MagicMock, ANY
from src import main, config




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


@patch("src.main.BlockingStateManager")
@patch("src.main.HeatingSystemStateChecker")
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
    MockHeatingSystemStateChecker,
    MockBlockingStateManager,
):
    """Test main loop skips ML prediction when heating is off.

    The ML prediction logic (build_physics_features + simplified_outlet_prediction)
    is genuinely skipped on a heating-off cycle. However, the passive-learning
    change now *saves* passive state (fresh last_indoor_temp +
    last_run_features with heating_active=False) on the heating-off path so the
    next cycle's learning block uses the passive equilibrium model. So
    save_state IS expected to be called once, but with heating_active=False.
    """
    # Arrange
    mock_ha_client = MagicMock()
    # On the second run of the main loop, create_ha_client will raise an
    # exception. This will be caught, and then poll_for_blocking will be
    # called, which will raise a second exception to stop the test.
    mock_create_ha_client.side_effect = [mock_ha_client, Exception("Stop loop")]

    # Mock BlockingStateManager
    mock_blocking_manager = MockBlockingStateManager.return_value
    mock_blocking_manager.check_blocking_state.return_value = (False, [])
    mock_blocking_manager.handle_grace_period.return_value = False
    mock_blocking_manager.poll_for_blocking.side_effect = StopIteration("Stop test")

    # Mock HeatingSystemStateChecker
    mock_heating_checker = MockHeatingSystemStateChecker.return_value
    mock_heating_checker.check_heating_active.return_value = False

    mock_load_state.return_value = {}

    # Act
    with patch("src.main.get_sensor_attributes", return_value={}):
        with pytest.raises(StopIteration, match="Stop test"):
            with patch("sys.argv", ["main.py"]):
                main.main()

    # Assert that the main ML logic was skipped
    mock_build_features.assert_not_called()
    mock_prediction.assert_not_called()

    # The passive-learning change now saves state on the heating-off path
    # (fresh last_indoor_temp + last_run_features with heating_active=False)
    # so the next cycle's learning block uses the passive equilibrium model.
    mock_save_state.assert_called_once()
    saved_kwargs = mock_save_state.call_args.kwargs
    assert saved_kwargs["last_run_features"]["heating_active"] is False, (
        "Heating-off save_state must mark heating_active=False so the next "
        "cycle uses the passive equilibrium model (no outlet/OE term)."
    )

    # Assert that the system is idle
    # mock_sleep.assert_called_once_with(300)
    # Assert that poll was called once before breaking
    mock_blocking_manager.poll_for_blocking.assert_called_once()
