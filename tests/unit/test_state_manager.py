
import pytest
from unittest.mock import patch, MagicMock, call
import logging

from src import state_manager

@pytest.fixture
def mock_thermal_state_manager():
    """Fixture to mock the UnifiedThermalStateManager."""
    with patch('src.state_manager.get_thermal_state_manager') as mock_get_manager:
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager
        yield mock_manager

def test_load_state_success(mock_thermal_state_manager):
    """Test successful loading of state."""
    expected_state = {"key": "value"}
    mock_thermal_state_manager.get_operational_state.return_value = expected_state
    
    result = state_manager.load_state()
    
    assert result == expected_state
    mock_thermal_state_manager.get_operational_state.assert_called_once()

def test_load_state_failure(mock_thermal_state_manager, caplog):
    """Test that a fresh state is returned if loading fails."""
    mock_thermal_state_manager.get_operational_state.side_effect = Exception("Load error")
    
    with caplog.at_level(logging.WARNING):
        result = state_manager.load_state()
    
    default_state = {
        "last_run_features": None,
        "last_indoor_temp": None,
        "last_avg_other_rooms_temp": None,
        "last_fireplace_on": False,
        "last_final_temp": None,
        "last_is_blocking": False,
        "last_blocking_end_time": None,
    }
    
    assert result == default_state
    assert "Could not load operational state" in caplog.text
    assert "Load error" in caplog.text

def test_save_state_success(mock_thermal_state_manager):
    """Test successful saving of state."""
    state_to_save = {"new_key": "new_value", "other_key": 123}
    
    state_manager.save_state(**state_to_save)
    
    mock_thermal_state_manager.update_operational_state.assert_called_once_with(**state_to_save)
    mock_thermal_state_manager.save_state.assert_called_once()

def test_save_state_failure(mock_thermal_state_manager, caplog):
    """Test that an error is logged if saving fails."""
    mock_thermal_state_manager.save_state.side_effect = Exception("Save error")
    state_to_save = {"key": "value"}
    
    with caplog.at_level(logging.ERROR):
        state_manager.save_state(**state_to_save)
        
    assert "Failed to save operational state" in caplog.text
    assert "Save error" in caplog.text
