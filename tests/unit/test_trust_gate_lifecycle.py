"""T0.7: Tests for the self-calibration trust-gate lifecycle.

The self-calibrating trust gate replaces the manual ML_ACCURACY_GATE_ENABLED
flag. It auto-promotes a model once it meets all five criteria, auto-demotes
when accuracy degrades, and re-promotes on recovery -- with the parameter
stability guard as the explicit anti-inversion backstop.
"""

import pytest

from src import config
from src.model_wrapper import EnhancedModelWrapper
from src.unified_thermal_state import ThermalStateManager
from unittest.mock import MagicMock


@pytest.fixture
def wrapper(tmp_path, monkeypatch):
    """Fresh wrapper with an isolated state file and a stubbed thermal model."""
    state_file = str(tmp_path / "thermal_state.json")
    monkeypatch.setattr(config, "UNIFIED_STATE_FILE", state_file)

    state_mgr = ThermalStateManager(state_file=state_file)
    thermal = MagicMock()
    thermal.thermal_time_constant = 4.0
    thermal.heat_loss_coefficient = 0.5
    thermal.outlet_effectiveness = 0.6
    thermal.external_source_weights = {"pv": 0.0, "fireplace": 0.0, "tv": 0.0}
    thermal._MAX_OUTLET_TEMP = 50.0
    thermal.parameter_history = []
    # learning_confidence is on the 0.1-5.0 scale; default above the
    # configured CONFIDENCE_THRESHOLD so existing "active"-stage tests pass.
    # Tests that exercise the low-confidence path lower it explicitly.
    thermal.learning_confidence = 3.0

    pred_metrics = MagicMock()
    pred_metrics.get_metrics.return_value = {
        "24h": {"mae": 0.0, "mean_error": 0.0}
    }
    pred_metrics.get_24h_good_control_percentage.return_value = 100.0

    w = EnhancedModelWrapper.__new__(EnhancedModelWrapper)
    w.state_manager = state_mgr
    w.thermal_model = thermal
    w.prediction_metrics = pred_metrics
    w.cycle_count = 0
    w._demotion_streak = 0
    return w


class TestCalibrationGate:
    def test_gate_not_met_with_few_predictions(self, wrapper):
        gate = wrapper._evaluate_calibration_gate()
        assert gate["all_met"] is False
        assert gate["criteria"]["sufficient_predictions"] is False

    def test_gate_not_met_with_high_mae(self, wrapper):
        wrapper.state_manager.state["prediction_metrics"]["total_predictions"] = 100
        wrapper.prediction_metrics.get_metrics.return_value = {
            "24h": {"mae": 10.0, "mean_error": 0.0}
        }
        gate = wrapper._evaluate_calibration_gate()
        assert gate["all_met"] is False
        assert gate["criteria"]["mae_ok"] is False
        assert gate["promotion_progress"] < 1.0

    def test_gate_not_met_with_high_bias(self, wrapper):
        wrapper.state_manager.state["prediction_metrics"]["total_predictions"] = 100
        wrapper.prediction_metrics.get_metrics.return_value = {
            "24h": {"mae": 0.1, "mean_error": 10.0}
        }
        gate = wrapper._evaluate_calibration_gate()
        assert gate["all_met"] is False
        assert gate["criteria"]["bias_ok"] is False

    def test_gate_not_met_with_unstable_params(self, wrapper):
        wrapper.state_manager.state["prediction_metrics"]["total_predictions"] = 250
        wrapper.prediction_metrics.get_metrics.return_value = {
            "24h": {"mae": 0.1, "mean_error": 0.0}
        }
        wrapper.prediction_metrics.get_24h_good_control_percentage.return_value = 100.0
        wrapper.thermal_model.parameter_history = [
            {"heat_loss_coefficient": 0.1 + 0.2 * i, "outlet_effectiveness": 0.6}
            for i in range(20)
        ]
        gate = wrapper._evaluate_calibration_gate()
        assert gate["all_met"] is False
        assert gate["criteria"]["params_stable"] is False

    def test_gate_met_promotes_and_sets_trust(self, wrapper):
        wrapper.state_manager.state["prediction_metrics"]["total_predictions"] = 250
        wrapper.prediction_metrics.get_metrics.return_value = {
            "24h": {"mae": 0.1, "mean_error": 0.0}
        }
        wrapper.prediction_metrics.get_24h_good_control_percentage.return_value = 100.0
        wrapper.thermal_model.parameter_history = [
            {"heat_loss_coefficient": 0.5, "outlet_effectiveness": 0.6}
            for _ in range(20)
        ]
        gate = wrapper._evaluate_calibration_gate()
        assert gate["all_met"] is True
        promoted = wrapper._check_auto_promotion()
        assert promoted is True
        trust = wrapper.get_trust_state()
        assert trust["is_trusted"] is True
        assert trust["stage"] == "active"
        assert trust["trust_reason"] == "active"


class TestDemotionRePromotion:
    def _calibrated_wrapper(self, wrapper):
        wrapper.state_manager.state["prediction_metrics"]["total_predictions"] = 100
        wrapper.state_manager.state["baseline_parameters"]["source"] = "calibrated"
        wrapper.state_manager.state["baseline_parameters"]["accuracy_state"] = "accurate"
        wrapper.state_manager.state["baseline_parameters"]["calibration_cycles"] = 100

    def test_demotion_on_degraded_accuracy(self, wrapper):
        self._calibrated_wrapper(wrapper)
        wrapper.prediction_metrics.get_metrics.return_value = {
            "24h": {"mae": 5.0, "mean_error": 0.0}
        }
        wrapper.prediction_metrics.get_24h_good_control_percentage.return_value = 50.0
        for _ in range(config.DEMOTION_SUSTAINED_CYCLES):
            wrapper._check_accuracy_demotion()
        m = wrapper.state_manager.get_learning_metrics()
        assert m["accuracy_state"] == "inaccurate"
        trust = wrapper.get_trust_state()
        assert trust["is_trusted"] is False
        assert trust["stage"] == "inaccurate"

    def test_re_promotion_on_recovery(self, wrapper):
        self._calibrated_wrapper(wrapper)
        wrapper.state_manager.state["baseline_parameters"]["accuracy_state"] = "inaccurate"
        wrapper.prediction_metrics.get_metrics.return_value = {
            "24h": {"mae": 0.1, "mean_error": 0.0}
        }
        wrapper.prediction_metrics.get_24h_good_control_percentage.return_value = 100.0
        for _ in range(config.DEMOTION_SUSTAINED_CYCLES):
            wrapper._check_accuracy_demotion()
        m = wrapper.state_manager.get_learning_metrics()
        assert m["accuracy_state"] == "accurate"
        trust = wrapper.get_trust_state()
        assert trust["is_trusted"] is True

    def test_demotion_requires_sustained_degradation(self, wrapper):
        self._calibrated_wrapper(wrapper)
        wrapper.prediction_metrics.get_metrics.return_value = {
            "24h": {"mae": 5.0, "mean_error": 0.0}
        }
        wrapper.prediction_metrics.get_24h_good_control_percentage.return_value = 50.0
        for _ in range(config.DEMOTION_SUSTAINED_CYCLES - 1):
            wrapper._check_accuracy_demotion()
        m = wrapper.state_manager.get_learning_metrics()
        assert m["accuracy_state"] == "accurate"


class TestTrustState:
    def test_passthrough_when_uncalibrated(self, wrapper):
        wrapper.state_manager.state["prediction_metrics"]["total_predictions"] = 0
        trust = wrapper.get_trust_state()
        assert trust["is_trusted"] is False
        assert trust["stage"] == "learning"
        assert trust["outlet_source"] == "heat_curve_passthrough"

    def test_passthrough_when_insufficient_predictions(self, wrapper):
        wrapper.state_manager.state["baseline_parameters"]["source"] = "calibrated"
        wrapper.state_manager.state["prediction_metrics"]["total_predictions"] = 1
        trust = wrapper.get_trust_state()
        assert trust["is_trusted"] is False
        assert trust["stage"] == "learning"

    def test_safe_fallback_on_error(self, wrapper):
        wrapper.state_manager = MagicMock()
        wrapper.state_manager.get_learning_metrics.side_effect = RuntimeError("boom")
        trust = wrapper.get_trust_state()
        assert trust["is_trusted"] is False
        assert trust["outlet_source"] == "heat_curve_passthrough"
    def test_safe_fallback_on_error(self, wrapper):
        wrapper.state_manager = MagicMock()
        wrapper.state_manager.get_learning_metrics.side_effect = RuntimeError("boom")
        trust = wrapper.get_trust_state()
        assert trust["is_trusted"] is False
        assert trust["outlet_source"] == "heat_curve_passthrough"


class TestConfidenceGate:
    """CONFIDENCE_THRESHOLD (configured in .env) must gate control authority.

    A calibrated, accurate model whose learning_confidence has collapsed below
    the threshold must fall back to heat-curve passthrough rather than
    commanding the ML-predicted outlet. This reproduces the production incident
    where a model with confidence=0.11 (threshold=2.0) was still driving
    heating because the threshold was never wired into the trust gate.
    """

    def _calibrated_accurate(self, wrapper):
        wrapper.state_manager.state["prediction_metrics"]["total_predictions"] = 100
        wrapper.state_manager.state["baseline_parameters"]["source"] = "calibrated"
        wrapper.state_manager.state["baseline_parameters"]["accuracy_state"] = "accurate"
        wrapper.state_manager.state["baseline_parameters"]["calibration_cycles"] = 100

    def test_low_confidence_demotes_to_passthrough(self, wrapper):
        self._calibrated_accurate(wrapper)
        # Below the configured CONFIDENCE_THRESHOLD (default 2.0): this is the
        # exact production scenario (confidence 0.11).
        wrapper.thermal_model.learning_confidence = 0.11
        trust = wrapper.get_trust_state()
        assert trust["is_trusted"] is False
        assert trust["is_calibrated"] is True
        assert trust["is_accurate"] is True
        assert trust["is_confident"] is False
        assert trust["stage"] == "low_confidence"
        assert trust["trust_reason"] == "low_confidence"
        assert trust["outlet_source"] == "heat_curve_passthrough"

    def test_confidence_recovery_re_promotes_to_active(self, wrapper):
        self._calibrated_accurate(wrapper)
        wrapper.thermal_model.learning_confidence = 0.11
        assert wrapper.get_trust_state()["is_trusted"] is False
        # Confidence recovers above the threshold -> back to active.
        wrapper.thermal_model.learning_confidence = config.CONFIDENCE_THRESHOLD
        trust = wrapper.get_trust_state()
        assert trust["is_trusted"] is True
        assert trust["stage"] == "active"
        assert trust["is_confident"] is True

    def test_inaccurate_takes_priority_over_low_confidence(self, wrapper):
        self._calibrated_accurate(wrapper)
        wrapper.state_manager.state["baseline_parameters"]["accuracy_state"] = "inaccurate"
        wrapper.thermal_model.learning_confidence = 0.11
        trust = wrapper.get_trust_state()
        # Both conditions fail, but the inaccurate (demotion) reason wins.
        assert trust["is_trusted"] is False
        assert trust["stage"] == "inaccurate"
        assert trust["trust_reason"] == "inaccurate"

    def test_confidence_reported_in_trust_state(self, wrapper):
        self._calibrated_accurate(wrapper)
        wrapper.thermal_model.learning_confidence = 1.5
        trust = wrapper.get_trust_state()
        assert trust["confidence"] == 1.5
        assert trust["is_confident"] is False


class TestUnknownStateDemotion:
    """A baseline promoted with accuracy_state='unknown' (e.g. a config-based
    promotion) must still be demotable when accuracy degrades. Previously
    'unknown' was a dead zone: the demotion logic only ran for 'accurate',
    so a poor model could remain permanently trusted.
    """

    def _calibrated_unknown(self, wrapper):
        wrapper.state_manager.state["prediction_metrics"]["total_predictions"] = 100
        wrapper.state_manager.state["baseline_parameters"]["source"] = "calibrated"
        wrapper.state_manager.state["baseline_parameters"]["accuracy_state"] = "unknown"
        wrapper.state_manager.state["baseline_parameters"]["calibration_cycles"] = 100

    def test_unknown_state_can_be_demoted(self, wrapper):
        self._calibrated_unknown(wrapper)
        wrapper.prediction_metrics.get_metrics.return_value = {
            "24h": {"mae": 5.0, "mean_error": 0.0}
        }
        wrapper.prediction_metrics.get_24h_good_control_percentage.return_value = 50.0
        for _ in range(config.DEMOTION_SUSTAINED_CYCLES):
            wrapper._check_accuracy_demotion()
        m = wrapper.state_manager.get_learning_metrics()
        assert m["accuracy_state"] == "inaccurate"

    def test_unknown_state_not_demoted_by_transient_spike(self, wrapper):
        self._calibrated_unknown(wrapper)
        wrapper.prediction_metrics.get_metrics.return_value = {
            "24h": {"mae": 5.0, "mean_error": 0.0}
        }
        wrapper.prediction_metrics.get_24h_good_control_percentage.return_value = 50.0
        for _ in range(config.DEMOTION_SUSTAINED_CYCLES - 1):
            wrapper._check_accuracy_demotion()
        m = wrapper.state_manager.get_learning_metrics()
        # Still unknown (not yet sustained long enough to demote).
        assert m["accuracy_state"] == "unknown"
