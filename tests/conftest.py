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
        "RECENT_ERRORS_WINDOW": "10",
        "SOLAR_LAG_MINUTES": "45.0"
    }
    
    for key, value in tdd_params.items():
        monkeypatch.setenv(key, value)


@pytest.fixture(autouse=True)
def cleanup_influx_service():
    """Ensure InfluxService is cleaned up after every test."""
    yield
    from src.influx_service import reset_influx_service
    reset_influx_service()


@pytest.fixture(autouse=True)
def isolated_thermal_state(tmp_path):
    """Isolate the global thermal-state singleton from the LIVE state file.

    ``ThermalEquilibriumModel()`` (and ``EnhancedModelWrapper()``, which
    constructs one internally) loads parameters + learning history via
    ``get_thermal_state_manager()`` -- a module-level singleton in
    ``src/unified_thermal_state`` backed by ``config.UNIFIED_STATE_FILE``, i.e.
    the live ``/opt/ml_heating/unified_thermal_state.json``.

    Without isolation the test suite:
      1. Inherits the live calibrated parameters and a real prediction_history
         (so tests are non-deterministic and depend on machine-local state).
      2. *Writes back* to the live state file: any test that triggers parameter
         adaptation calls ``update_learning_state()`` -> ``save_state()`` on the
         singleton, persisting accumulated learning deltas into the production
         state file -- genuine, harmful contamination.

    This fixture installs a throwaway ``ThermalStateManager`` pointing at a
    per-test temp file (``tmp_path``) for the duration of every test and
    restores the prior singleton on teardown. The temp manager holds the
    default state (``baseline_parameters.source == "config"``), so models load
    config/TDD defaults (the regime the rest of conftest.py pins via env vars)
    with empty learning history instead of live calibrated values.

    Tests that manage the singleton explicitly
    (``test_model_wrapper.clean_state``, ``test_unified_thermal_state``,
    ``test_regression_scenarios``, ``test_physics_constraints`` ...) override
    ``_thermal_state_manager`` / patch ``get_thermal_state_manager`` in their
    own fixtures, so this autouse default is a safe baseline that only binds
    for tests relying on the factory's default behaviour.
    """
    import src.unified_thermal_state as uts

    saved = uts._thermal_state_manager
    manager = uts.ThermalStateManager(
        state_file=str(tmp_path / "isolated_thermal_state.json")
    )
    # Materialise the temp file so the model's load_state() takes the happy
    # path (returns True) rather than the one-shot "failed to load" warning
    # branch. The on-disk content is the default state (source="config").
    manager.save_state()
    uts._thermal_state_manager = manager
    try:
        yield
    finally:
        uts._thermal_state_manager = saved
