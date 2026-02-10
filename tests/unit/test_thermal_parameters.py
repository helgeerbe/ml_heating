
import pytest
from unittest.mock import patch
import os

from src.thermal_parameters import (
    ThermalParameterManager,
    ParameterInfo,
    get_thermal_parameter,
    validate_thermal_parameter,
)


@pytest.fixture
def mock_thermal_config():
    """Fixture to mock the ThermalParameterConfig."""
    with patch("src.thermal_parameters.ThermalParameterConfig") as mock_config:
        mock_config.get_all_parameter_info.return_value = {
            "param1": {
                "default": 1.0,
                "bounds": (0.0, 10.0),
                "description": "",
                "unit": "",
            },
            "param2": {
                "default": 5.0,
                "bounds": (0.0, 10.0),
                "description": "",
                "unit": "",
            },
        }
        yield mock_config


@pytest.fixture
def manager(mock_thermal_config):
    """Returns an initialized ThermalParameterManager instance."""
    return ThermalParameterManager()


def test_initialization(manager):
    """Test that the manager initializes correctly."""
    assert "param1" in manager._PARAMETERS
    assert manager._PARAMETERS["param1"].default == 1.0
    assert manager.get("param1") == 1.0


def test_get_unknown_parameter(manager):
    """Test that getting an unknown parameter raises a KeyError."""
    with pytest.raises(KeyError):
        manager.get("unknown")


def test_set_parameter(manager):
    """Test setting a parameter value."""
    assert manager.set("param1", 2.0)
    assert manager.get("param1") == 2.0


def test_set_parameter_out_of_bounds(manager):
    """Test that setting a parameter out of bounds fails."""
    assert not manager.set("param1", 11.0)
    assert manager.get("param1") == 1.0  # Should remain default


def test_load_from_environment(mock_thermal_config):
    """Test loading parameters from environment variables."""
    with patch.dict(os.environ, {"PARAM1": "3.0", "PARAM2": "invalid", "UNKNOWN": "1.0"}):
        manager = ThermalParameterManager()
        assert manager.get("param1") == 3.0
        assert manager.get("param2") == 5.0  # Should be default


def test_get_all_parameters(manager):
    """Test getting all parameter values."""
    manager.set("param1", 2.5)
    all_params = manager.get_all_parameters()
    assert all_params == {"param1": 2.5, "param2": 5.0}


def test_get_all_defaults(manager):
    """Test getting all default parameter values."""
    defaults = manager.get_all_defaults()
    assert defaults == {"param1": 1.0, "param2": 5.0}


def test_validate_all(manager):
    """Test validating all parameters."""
    manager.set("param1", 11.0)  # This set will fail
    validation = manager.validate_all()
    # The value of param1 was not updated, so it is still the valid default
    assert validation == {"param1": True, "param2": True}


def test_reload_from_environment(mock_thermal_config):
    """Test reloading parameters from environment."""
    manager = ThermalParameterManager()
    manager.set("param1", 9.0)

    with patch.dict(os.environ, {"PARAM1": "4.0"}):
        manager.reload_from_environment()
        assert manager.get("param1") == 4.0


def test_legacy_functions(manager):
    """Test the legacy compatibility functions."""
    with patch("src.thermal_parameters.thermal_params", manager):
        assert get_thermal_parameter("param1") == 1.0
        assert validate_thermal_parameter("param1", 5.0)
        assert not validate_thermal_parameter("param1", 15.0)


def test_get_info(manager):
    """Test getting parameter info."""
    info = manager.get_info("param1")
    assert isinstance(info, ParameterInfo)
    assert info.name == "param1"


def test_has_single_source_of_truth(manager):
    """Test the single source of truth check."""
    assert manager.has_single_source_of_truth()
