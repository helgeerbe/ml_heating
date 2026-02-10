import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import numpy as np
from src.adaptive_fireplace_learning import (
    AdaptiveFireplaceLearning,
    FireplaceLearningState,
)


@pytest.fixture
def afl_instance(tmp_path):
    """Create an instance of AdaptiveFireplaceLearning with a temp state file."""
    state_file = tmp_path / "fireplace_learning_state.json"
    return AdaptiveFireplaceLearning(state_file=str(state_file))


def test_initialization(afl_instance):
    """Test that the AdaptiveFireplaceLearning class initializes correctly."""
    assert isinstance(afl_instance.learning_state, FireplaceLearningState)
    assert afl_instance.learning_state.learned_coefficients[
        "base_heat_output_kw"
    ] == pytest.approx(6.0)


def test_load_non_existent_state(tmp_path):
    """Test that a new state is created when the state file does not exist."""
    state_file = tmp_path / "non_existent_state.json"
    afl = AdaptiveFireplaceLearning(state_file=str(state_file))
    assert isinstance(afl.learning_state, FireplaceLearningState)


@patch("src.adaptive_fireplace_learning.datetime")
def test_observe_fireplace_session_start(mock_dt, afl_instance):
    """Test that a new fireplace session is started correctly."""
    now = datetime.now()
    mock_dt.now.return_value = now
    result = afl_instance.observe_fireplace_state(22.0, 20.0, 5.0, True)
    assert result["session_update"]["status"] == "session_started"
    assert afl_instance.current_session is not None
    assert afl_instance.current_session["start_time"] == now


@patch("src.adaptive_fireplace_learning.datetime")
def test_observe_fireplace_session_end(mock_dt, afl_instance):
    """Test that a fireplace session ends and an observation is created."""
    start_time = datetime.now() - timedelta(minutes=12)
    end_time = datetime.now()

    afl_instance.current_session = {
        "start_time": start_time,
        "start_differential": 2.0,
        "outdoor_temp": 5.0,
        "peak_differential": 2.5,
    }
    afl_instance.session_start_time = start_time
    mock_dt.now.return_value = end_time

    result = afl_instance.observe_fireplace_state(21.0, 20.0, 5.0, False)
    assert result["session_update"]["status"] == "session_ended"
    assert result["session_update"]["learned"] is True
    assert len(afl_instance.learning_state.observations) == 1
    assert afl_instance.current_session is None
    assert result["session_update"]["duration_minutes"] == pytest.approx(12.0)


def test_save_and_load_state(afl_instance, tmp_path):
    """Test that the state is saved and loaded correctly."""
    # Create an observation
    with patch("src.adaptive_fireplace_learning.datetime") as mock_dt:
        start_time = datetime.now() - timedelta(minutes=15)
        end_time = datetime.now()
        mock_dt.now.side_effect = [start_time, end_time, end_time]
        afl_instance.observe_fireplace_state(22.0, 20.0, 5.0, True)  # Start
        afl_instance.observe_fireplace_state(21.0, 20.0, 5.0, False)  # End

    # Now, create a new instance and see if it loads the state
    state_file = tmp_path / "fireplace_learning_state.json"
    new_afl = AdaptiveFireplaceLearning(state_file=str(state_file))
    assert len(new_afl.learning_state.observations) == 1
    assert new_afl.learning_state.observations[0].duration_minutes == pytest.approx(
        15.0, abs=1e-1
    )


@patch("src.adaptive_fireplace_learning.datetime")
@patch("src.adaptive_fireplace_learning.np.corrcoef")
@patch("src.adaptive_fireplace_learning.np.mean")
def test_update_learning_coefficients(
    mock_mean, mock_corrcoef, mock_dt, afl_instance
):
    """Test that the learning coefficients are updated correctly."""
    mock_mean.return_value = 2.0
    mock_corrcoef.return_value = np.array([[1.0, 0.5], [0.5, 1.0]])
    now = datetime.now()
    mock_dt.now.return_value = now

    observations = []
    for i in range(6):
        obs = MagicMock()
        obs.duration_minutes = 20
        obs.peak_differential = 2.0
        obs.outdoor_temp = 5.0 + i * 2
        obs.timestamp = now - timedelta(days=i)
        observations.append(obs)
    afl_instance.learning_state.observations = observations
    afl_instance._save_state = MagicMock()

    result = afl_instance._update_learning_coefficients()

    assert result["status"] == "coefficients_updated"
    assert (
        afl_instance.learning_state.learned_coefficients[
            "differential_to_heat_ratio"
        ]
        != 2.5
    )
    assert (
        afl_instance.learning_state.learned_coefficients[
            "outdoor_temp_correlation"
        ]
        != 0.1
    )
    afl_instance._save_state.assert_called_once()
