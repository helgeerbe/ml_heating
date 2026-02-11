"""
Pytest Configuration for TDD Enforcement.

This file defines a pytest fixture that enforces a consistent, TDD-compliant 
environment for all test runs. It achieves this by programmatically overriding 
the environment variables related to the thermal model parameters before any 
tests are executed.

The `enforce_tdd_thermal_parameters` fixture ensures that all tests, regardless 
of the local `.env` configuration, run against the same physically realistic 
and validated baseline. This is critical for maintaining the integrity and 
reliability of the test suite.

By centralizing the TDD enforcement here, we prevent configuration drift and 
ensure that our tests are always aligned with the documented thermal model
specification.
"""
import pytest


@pytest.fixture(autouse=True)
def enforce_tdd_thermal_parameters(monkeypatch):
    """
    Enforces TDD-compliant thermal parameters for all tests.

    This fixture is automatically applied to every test function
    (`autouse=True`). It uses `monkeypatch` to set environment variables
    to the exact values defined in the TDD specification, ensuring a
    consistent and controlled testing environment.
    """
    tdd_params = {
        # Core Thermal Properties
        "THERMAL_TIME_CONSTANT": "4.0",
        "HEAT_LOSS_COEFFICIENT": "0.2",
        "OUTLET_EFFECTIVENESS": "0.04",
        "OUTDOOR_COUPLING": "0.3",
        "THERMAL_BRIDGE_FACTOR": "0.1",

        # External Heat Source Weights
        "PV_HEAT_WEIGHT": "0.002",
        "FIREPLACE_HEAT_WEIGHT": "5.0",
        "TV_HEAT_WEIGHT": "0.2",

        # Adaptive Learning Parameters
        "ADAPTIVE_LEARNING_RATE": "0.05",
        "MIN_LEARNING_RATE": "0.01",
        "MAX_LEARNING_RATE": "0.2",
        "LEARNING_CONFIDENCE": "3.0",
        "RECENT_ERRORS_WINDOW": "10"
    }
    
    for key, value in tdd_params.items():
        monkeypatch.setenv(key, value)


@pytest.fixture(autouse=True)
def cleanup_influx_service():
    """Ensure InfluxService is cleaned up after every test."""
    yield
    from src.influx_service import reset_influx_service
    reset_influx_service()
