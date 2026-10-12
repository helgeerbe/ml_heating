"""
Regression tests for the untrained-model outlet-pegging incident (P0 T0.1-T0.5).

These tests guard against the incident where an untrained/degraded thermal
model silently pegs the commanded outlet temperature to the ceiling:
  - T0.2: an uncalibrated model (default baseline, few predictions) must be
    capped to a conservative, heat-curve-anchored ceiling.
  - T0.3: physically implausible outlet_effectiveness / HLC:OE ratios must
    fall back to a conservative outlet instead of the exploding inversion.
  - T0.1: when the raw inversion is clamped to the model soft-max, a flag
    and warning must be emitted.
  - T0.4: the model uses a single source of truth for its max outlet.
"""
import os
import sys
from unittest.mock import MagicMock

import pytest

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.model_wrapper import EnhancedModelWrapper
from src.thermal_equilibrium_model import ThermalEquilibriumModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def wrapper():
    """Fresh EnhancedModelWrapper with a mocked state manager so calibration
    metrics are deterministic and no real state file is required.

    ``MIN_PREDICTIONS_FOR_TRUST`` is env-configurable in production (lowered in
    ``.env`` to re-engage faster after the parameter reset). These regression
    tests encode the *original* safety intent of the untrained-model guard, so
    they pin the threshold to a deterministic value independent of the
    deployment environment.
    """
    w = EnhancedModelWrapper()
    # Pin to the historical default so the gate semantics are deterministic
    # regardless of any MIN_PREDICTIONS_FOR_TRUST override in .env.
    w.MIN_PREDICTIONS_FOR_TRUST = 50
    w.state_manager = MagicMock()
    # Default: uncalibrated (default baseline, no predictions)
    w.state_manager.get_learning_metrics.return_value = {
        "baseline_source": "config",
        "total_predictions": 0,
    }
    # Avoid real prediction calls during the gate test
    w._calculate_required_outlet_temp = MagicMock(return_value=60.0)
    w.predict_indoor_temp = MagicMock(return_value=21.0)
    return w


@pytest.fixture
def model():
    """Clean ThermalEquilibriumModel with realistic parameters."""
    m = ThermalEquilibriumModel()
    m.thermal_time_constant = 4.0
    m.heat_loss_coefficient = 0.4
    m.outlet_effectiveness = 0.5
    m.external_source_weights = {"pv": 0.002, "fireplace": 5.0, "tv": 0.2}
    m.solar_lag_minutes = 45.0
    return m


# ---------------------------------------------------------------------------
# T0.2 — calibration gate
# ---------------------------------------------------------------------------

class TestCalibrationGate:
    def test_uncalibrated_when_default_baseline(self, wrapper):
        assert wrapper._is_model_calibrated() is False

    def test_calibrated_when_calibrated_and_enough_predictions(self, wrapper):
        wrapper.state_manager.get_learning_metrics.return_value = {
            "baseline_source": "calibrated",
            "total_predictions": 100,
        }
        assert wrapper._is_model_calibrated() is True

    def test_not_calibrated_when_few_predictions(self, wrapper):
        wrapper.state_manager.get_learning_metrics.return_value = {
            "baseline_source": "calibrated",
            "total_predictions": 10,
        }
        assert wrapper._is_model_calibrated() is False

    def test_uncalibrated_model_caps_outlet_to_conservative_ceiling(self, wrapper):
        """An uncalibrated model must NOT command an outlet above the
        conservative ceiling, even when the binary search returned a high
        value (60°C here)."""
        features = {
            "target_temp": 21.0,
            "indoor_temp_lag_30m": 18.0,
            "outdoor_temp": 0.0,
        }
        outlet, meta = wrapper.calculate_optimal_outlet_temp(features)
        # No heat-curve reading -> conservative fallback (35°C), no +3 margin.
        assert outlet == pytest.approx(35.0, abs=0.01)
        assert outlet <= 45.0
        assert meta["model_calibrated"] is False
        assert meta["uncalibrated_capped"] is True
        assert meta["outlet_source"] == "heat_curve_passthrough"

    def test_passthrough_rejects_garbage_heat_curve(self, wrapper):
        """A stale/zero heat-curve reading is replaced with the conservative
        fallback rather than commanded blindly."""
        features = {
            "target_temp": 21.0,
            "indoor_temp_lag_30m": 18.0,
            "outdoor_temp": 0.0,
            "heat_curve_temp": 0.0,  # garbage / sensor offline
        }
        outlet, meta = wrapper.calculate_optimal_outlet_temp(features)
        assert outlet == pytest.approx(35.0, abs=0.01)
        assert meta["outlet_source"] == "heat_curve_passthrough"

    def test_uncalibrated_model_anchors_on_heat_curve(self, wrapper):
        """T0.2b: in passthrough the outlet equals the heat-curve reading
        exactly — NO +3 margin (the +3 was the original incident risk). The ML
        value is still computed and kept in metadata for delta logging."""
        features = {
            "target_temp": 21.0,
            "indoor_temp_lag_30m": 18.0,
            "outdoor_temp": 0.0,
            "heat_curve_temp": 30.0,
        }
        outlet, meta = wrapper.calculate_optimal_outlet_temp(features)
        # Pure passthrough: heat_curve 30, no +3.
        assert outlet == pytest.approx(30.0, abs=0.01)
        assert meta["uncalibrated_capped"] is True
        assert meta["outlet_source"] == "heat_curve_passthrough"
        assert meta["conservative_ceiling"] == pytest.approx(30.0, abs=0.01)
        # The ML value must still be surfaced for shadow-mode delta logging.
        assert meta["ml_outlet_temp"] == pytest.approx(60.0, abs=0.01)

    def test_calibrated_model_not_capped(self, wrapper):
        """A calibrated model is allowed to command the full inversion."""
        wrapper.state_manager.get_learning_metrics.return_value = {
            "baseline_source": "calibrated",
            "total_predictions": 100,
        }
        features = {
            "target_temp": 21.0,
            "indoor_temp_lag_30m": 18.0,
            "outdoor_temp": 0.0,
        }
        outlet, meta = wrapper.calculate_optimal_outlet_temp(features)
        # Not capped: the mocked 60°C passes through
        assert outlet == pytest.approx(60.0, abs=0.01)
        assert meta["model_calibrated"] is True
        assert meta["uncalibrated_capped"] is False


# ---------------------------------------------------------------------------
# T0.3 — OE / HLC:OE sanity guard
# ---------------------------------------------------------------------------

class TestOutletEffectivenessGuard:
    def test_tiny_outlet_effectiveness_uses_conservative_outlet(self, model):
        model.outlet_effectiveness = 0.01  # below OE_SAFE_MIN (0.05)
        model.heat_loss_coefficient = 0.4
        outlet = model._calculate_equilibrium_outlet_temperature(21.0, 0.0)
        # Must not explode to the ceiling; conservative and below model max.
        assert outlet <= 45.0
        assert outlet < 65.0
        assert model.outlet_clamped_to_max is False

    def test_extreme_hlc_oe_ratio_uses_conservative_outlet(self, model):
        # OE = 0.1, HLC = 1.2 -> ratio 12 > 5.0
        model.outlet_effectiveness = 0.1
        model.heat_loss_coefficient = 1.2
        outlet = model._calculate_equilibrium_outlet_temperature(21.0, -10.0)
        assert outlet <= 45.0
        assert model.outlet_clamped_to_max is False

    def test_zero_outlet_effectiveness_returns_safe_value(self, model):
        model.outlet_effectiveness = 0.0
        outlet = model._calculate_equilibrium_outlet_temperature(21.0, 0.0)
        assert outlet == 35.0


# ---------------------------------------------------------------------------
# T0.1 — clamp warning/flag
# ---------------------------------------------------------------------------

class TestClampFlag:
    def test_equilibrium_clamp_sets_flag(self, model):
        # Valid params but extreme demand -> inversion clamped to model max.
        model.outlet_effectiveness = 0.5
        model.heat_loss_coefficient = 0.4
        # target 30, outdoor -40 -> equilibrium outlet explodes
        outlet = model._calculate_equilibrium_outlet_temperature(30.0, -40.0)
        assert outlet == pytest.approx(65.0, abs=0.01)
        assert model.outlet_clamped_to_max is True

    def test_no_clamp_clears_flag(self, model):
        model.outlet_effectiveness = 0.5
        model.heat_loss_coefficient = 0.4
        outlet = model._calculate_equilibrium_outlet_temperature(21.0, 5.0)
        assert model.outlet_clamped_to_max is False
        # Reasonable, unclamped outlet
        assert 25.0 <= outlet <= 65.0

    def test_direct_calc_clamps_to_model_max_not_70(self, model):
        """T0.4: the direct-calculation path must use the unified model max
        (65), not the old hardcoded 70."""
        result = model.calculate_optimal_outlet_temperature(
            target_indoor=30.0,
            current_indoor=20.0,
            outdoor_temp=-40.0,
        )
        # The raw inversion would be ~86; must be clamped to model max (65).
        assert result["optimal_outlet_temp"] <= 65.0
        assert model.outlet_clamped_to_max is True
        assert result["bounded"] is True


# ---------------------------------------------------------------------------
# T0.4 — single source of truth for model max
# ---------------------------------------------------------------------------

class TestUnifiedMaxOutlet:
    def test_model_max_constant_is_65(self, model):
        assert model._MAX_OUTLET_TEMP == 65.0

    def test_both_paths_use_same_max(self, model):
        """Equilibrium fallback and direct calc share one max source."""
        # Force equilibrium clamp
        eq_outlet = model._calculate_equilibrium_outlet_temperature(30.0, -40.0)
        assert eq_outlet == 65.0
        # Force direct calc clamp
        direct = model.calculate_optimal_outlet_temperature(
            target_indoor=30.0,
            current_indoor=20.0,
            outdoor_temp=-40.0,
        )
        assert direct["optimal_outlet_temp"] <= 65.0

