
import pytest
from src.adaptive_learning_metrics_schema import (
    get_schema_for_measurement,
    validate_metrics_data,
    get_all_measurement_names,
    get_schema_summary,
    ADAPTIVE_LEARNING_SCHEMAS,
)


def test_get_schema_for_measurement_valid():
    """Test that a valid measurement name returns the correct schema."""
    schema = get_schema_for_measurement("ml_prediction_metrics")
    assert schema is not None
    assert schema["measurement"] == "ml_prediction_metrics"


def test_get_schema_for_measurement_invalid():
    """Test that an invalid measurement name returns None."""
    schema = get_schema_for_measurement("invalid_measurement")
    assert schema is None


@pytest.mark.parametrize(
    "measurement_name, data, expected",
    [
        ("ml_prediction_metrics", {"mae_1h": 0.5, "is_improving": True}, True),
        ("ml_prediction_metrics", {"mae_1h": "invalid_float"}, False),
        ("ml_prediction_metrics", {"total_predictions": "not_an_int"}, False),
        ("ml_prediction_metrics", {"is_improving": "not_a_bool"}, False),
        (
            "ml_learning_phase",
            {
                "current_learning_phase": "high_confidence",
                "stability_score": 0.9,
            },
            True,
        ),
        ("ml_learning_phase", {"stability_score": "invalid_float"}, False),
    ],
)
def test_validate_metrics_data(measurement_name, data, expected):
    """Test the validation of metrics data against a schema."""
    assert validate_metrics_data(measurement_name, data) is expected


def test_validate_metrics_data_no_schema():
    """Test validation with a measurement name that has no schema."""
    assert validate_metrics_data("invalid_measurement", {}) is False


def test_get_all_measurement_names():
    """Test that all measurement names are returned."""
    expected_names = list(ADAPTIVE_LEARNING_SCHEMAS.keys())
    assert get_all_measurement_names() == expected_names


def test_get_schema_summary():
    """Test the generation of the schema summary."""
    summary = get_schema_summary()
    assert "Adaptive Learning Metrics Schema Summary" in summary
    for measurement_name in ADAPTIVE_LEARNING_SCHEMAS:
        assert measurement_name in summary

