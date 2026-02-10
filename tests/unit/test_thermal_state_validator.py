
import pytest
import json
from unittest.mock import patch, mock_open, MagicMock
from src.thermal_state_validator import (
    ThermalStateValidator,
    ThermalStateValidationError,
    validate_thermal_state_safely,
)


@pytest.fixture
def valid_thermal_state_data():
    """Provides a valid thermal state data dictionary."""
    return {
        "metadata": {"version": "1.0"},
        "baseline_parameters": {
            "thermal_time_constant": 10.0,
            "heat_loss_coefficient": 0.5,
            "outlet_effectiveness": 0.8,
            "pv_heat_weight": 0.05,
            "fireplace_heat_weight": 20.0,
            "tv_heat_weight": 1.0,
            "source": "calibrated",
        },
        "learning_state": {},
        "prediction_metrics": {},
        "operational_state": {},
    }


def test_validate_thermal_state_data_valid(valid_thermal_state_data):
    """Test that valid data passes validation."""
    assert (
        ThermalStateValidator.validate_thermal_state_data(valid_thermal_state_data)
        is True
    )


def test_validate_thermal_state_data_missing_section(valid_thermal_state_data):
    """Test that a missing required section fails validation."""
    del valid_thermal_state_data["baseline_parameters"]
    with pytest.raises(
        ThermalStateValidationError, match="Missing required section: baseline_parameters"
    ):
        ThermalStateValidator.validate_thermal_state_data(valid_thermal_state_data)


def test_validate_thermal_state_data_missing_parameter(valid_thermal_state_data):
    """Test that a missing parameter fails validation."""
    del valid_thermal_state_data["baseline_parameters"]["thermal_time_constant"]
    with pytest.raises(
        ThermalStateValidationError,
        match="Missing required parameter: baseline_parameters.thermal_time_constant",
    ):
        ThermalStateValidator.validate_thermal_state_data(valid_thermal_state_data)


def test_validate_thermal_state_data_parameter_out_of_range(valid_thermal_state_data):
    """Test that an out-of-range parameter fails validation."""
    valid_thermal_state_data["baseline_parameters"]["thermal_time_constant"] = 100.0
    with pytest.raises(ThermalStateValidationError, match="out of range"):
        ThermalStateValidator.validate_thermal_state_data(valid_thermal_state_data)


def test_validate_thermal_state_data_invalid_parameter_type(
    valid_thermal_state_data,
):
    """Test that a parameter with an invalid type fails validation."""
    valid_thermal_state_data["baseline_parameters"]["thermal_time_constant"] = "invalid"
    with pytest.raises(ThermalStateValidationError, match="must be numeric"):
        ThermalStateValidator.validate_thermal_state_data(valid_thermal_state_data)


def test_validate_thermal_state_data_invalid_source(valid_thermal_state_data):
    """Test that an invalid source fails validation."""
    valid_thermal_state_data["baseline_parameters"]["source"] = "invalid_source"
    with pytest.raises(ThermalStateValidationError, match="Invalid source"):
        ThermalStateValidator.validate_thermal_state_data(valid_thermal_state_data)


@patch("src.thermal_state_validator.SCHEMA_VALIDATION_AVAILABLE", False)
def test_strict_validation_unavailable(valid_thermal_state_data, caplog):
    """Test that strict validation is skipped if jsonschema is not available."""
    assert (
        ThermalStateValidator.validate_thermal_state_data(
            valid_thermal_state_data, strict=True
        )
        is True
    )


@patch("src.thermal_state_validator.SCHEMA_VALIDATION_AVAILABLE", True)
@patch("src.thermal_state_validator.validate")
def test_strict_validation_passing(mock_validate, valid_thermal_state_data):
    """Test that strict validation passes with a valid schema."""
    mock_schema_class = MagicMock()
    mock_schema_class.get_unified_thermal_state_schema.return_value = {
        "type": "object"
    }
    mock_schema_module = MagicMock()
    mock_schema_module.ThermalStateSchema = mock_schema_class

    with patch.dict(
        "sys.modules",
        {"tests.test_thermal_state_schema_validation": mock_schema_module},
    ):
        mock_validate.return_value = None
        assert (
            ThermalStateValidator.validate_thermal_state_data(
                valid_thermal_state_data, strict=True
            )
            is True
        )
        mock_validate.assert_called_once_with(
            instance=valid_thermal_state_data, schema={"type": "object"}
        )


def test_validate_file_valid(valid_thermal_state_data):
    """Test validation of a valid file."""
    read_data = json.dumps(valid_thermal_state_data)
    with patch("builtins.open", mock_open(read_data=read_data)) as m:
        assert ThermalStateValidator.validate_file("dummy_path.json") is True
        m.assert_called_once_with("dummy_path.json", "r")


def test_validate_file_not_found():
    """Test validation with a non-existent file."""
    with patch("builtins.open", mock_open()) as m:
        m.side_effect = FileNotFoundError
        with pytest.raises(
            ThermalStateValidationError, match="Thermal state file not found"
        ):
            ThermalStateValidator.validate_file("non_existent.json")


def test_validate_file_invalid_json():
    """Test validation of a file with invalid JSON."""
    with patch("builtins.open", mock_open(read_data="{invalid_json")):
        with pytest.raises(ThermalStateValidationError, match="Invalid JSON"):
            ThermalStateValidator.validate_file("invalid.json")


def test_validate_thermal_state_safely_valid(valid_thermal_state_data):
    """Test safe validation with valid data."""
    assert validate_thermal_state_safely(valid_thermal_state_data) is True


def test_validate_thermal_state_safely_invalid(
    valid_thermal_state_data, caplog
):
    """Test safe validation with invalid data."""
    del valid_thermal_state_data["metadata"]
    assert validate_thermal_state_safely(valid_thermal_state_data) is False
    assert "Thermal state validation failed" in caplog.text

