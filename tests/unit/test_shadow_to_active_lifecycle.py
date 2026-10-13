"""Shadow -> active lifecycle: an uncalibrated model must run in passthrough
(shadow) while it accumulates evidence, then auto-promote to "active" the
moment the self-calibration gate is met, after which get_trust_state() reports
is_trusted=True / outlet_source="ml_predicted".

The sibling test_trust_gate_lifecycle.py validates _check_auto_promotion() and
get_trust_state() in isolation by calling them directly with a mocked thermal
model and pre-seeded state. This file exercises the *composition* -- it drives
the real learn_from_prediction_feedback() loop (which itself calls
_check_auto_promotion / _check_accuracy_demotion every cycle against a *real*
ThermalStateManager) and asserts the end-to-end passthrough -> active
transition. This is the wiring that, if broken, would let the model either
never take control or take control before it has earned trust.
"""

import pytest

from src import config
from src.model_wrapper import EnhancedModelWrapper
from src.unified_thermal_state import ThermalStateManager
from unittest.mock import MagicMock


@pytest.fixture
def lifecycle_wrapper(tmp_path, monkeypatch):
    """A wrapper wired to a *real* state manager but with a stubbed thermal
    model + prediction metrics so the gate signals are deterministic.

    Only the I/O side-effects of the learning loop (Influx export, HA export)
    are stubbed; the promotion/demotion/trust-state logic runs for real.
    """
    state_file = str(tmp_path / "thermal_state.json")
    monkeypatch.setattr(config, "UNIFIED_STATE_FILE", state_file)
    state_mgr = ThermalStateManager(state_file=state_file)

    thermal = MagicMock()
    thermal.thermal_time_constant = 4.0
    thermal.heat_loss_coefficient = 0.5
    thermal.outlet_effectiveness = 0.6
    thermal.external_source_weights = {"pv": 0.0, "fireplace": 0.0, "tv": 0.0}
    thermal._MAX_OUTLET_TEMP = 50.0
    # Pre-populate a long, perfectly-stable parameter history so the
    # _evaluate_parameter_stability() guard (needs >= window identical entries)
    # passes from the very first cycle.
    thermal.parameter_history = [
        {"heat_loss_coefficient": 0.5, "outlet_effectiveness": 0.6}
        for _ in range(25)
    ]
    thermal.learning_confidence = 3.0  # well above CONFIDENCE_THRESHOLD (2.0)
    thermal.update_prediction_feedback.return_value = 0.5

    pred_metrics = MagicMock()
    pred_metrics.get_metrics.return_value = {
        "24h": {"mae": 0.1, "mean_error": 0.0}
    }
    pred_metrics.get_24h_good_control_percentage.return_value = 100.0
    pred_metrics.get_24h_prediction_count.return_value = 100
    pred_metrics.add_prediction = MagicMock()  # no-op; state count is real

    # Lower the prediction-count thresholds so only a handful of real cycles
    # are required. AUTO_CAL_MIN_PREDICTIONS is read from `config` inside the
    # gate; MIN_PREDICTIONS_FOR_TRUST is a class attribute read off `self`.
    monkeypatch.setattr(config, "AUTO_CAL_MIN_PREDICTIONS", 3)
    monkeypatch.setattr(EnhancedModelWrapper, "MIN_PREDICTIONS_FOR_TRUST", 3)

    w = EnhancedModelWrapper.__new__(EnhancedModelWrapper)
    w.state_manager = state_mgr
    w.thermal_model = thermal
    w.prediction_metrics = pred_metrics
    w.adaptive_fireplace = MagicMock()
    w.cycle_count = 0
    w._demotion_streak = 0
    w.learning_enabled = True
    # Stub the I/O exports the loop performs every cycle.
    w._export_metrics_to_influxdb = lambda: None
    w.export_metrics_to_ha = lambda: None
    return w


def _active_context():
    """An active (heating-on) prediction context with no fireplace, so the
    adaptive-fireplace branch is never entered."""
    return {
        "outlet_temp": 45.0,
        "outdoor_temp": 5.0,
        "current_indoor": 20.0,
        "heating_active": True,
        "fireplace_on": 0,
        "tv_on": 0,
        "pv_power": 0,
    }


class TestShadowToActiveLifecycle:
    def test_starts_in_passthrough_while_uncalibrated(self, lifecycle_wrapper):
        """Before any learning, an uncalibrated model must defer to the heat
        pump's curve (passthrough), never commanding the ML outlet."""
        trust = lifecycle_wrapper.get_trust_state()
        assert trust["is_trusted"] is False
        assert trust["stage"] == "learning"
        assert trust["outlet_source"] == "heat_curve_passthrough"

    def test_auto_promotes_to_active_through_learning_loop(
        self, lifecycle_wrapper
    ):
        """Driving learn_from_prediction_feedback() through enough active
        cycles must flip baseline_source to "calibrated" and get_trust_state()
        to "active" / ml_predicted -- end to end through the real state
        manager."""
        w = lifecycle_wrapper

        # Two cycles are consumed by the first-cycle restart guard
        # (cycle_count <= 1); subsequent cycles accumulate real predictions in
        # the state manager until the gate is met (>= 3 predictions here).
        for _ in range(7):
            w.learn_from_prediction_feedback(
                predicted_temp=20.0,
                actual_temp=20.0,
                prediction_context=_active_context(),
            )

        metrics = w.state_manager.get_learning_metrics()
        assert metrics["baseline_source"] == "calibrated", (
            "Model failed to auto-promote after sufficient active cycles"
        )

        trust = w.get_trust_state()
        assert trust["is_trusted"] is True
        assert trust["stage"] == "active"
        assert trust["trust_reason"] == "active"
        assert trust["outlet_source"] == "ml_predicted"

    def test_stays_passthrough_before_gate_threshold(self, lifecycle_wrapper):
        """The model must remain in passthrough until the minimum-prediction
        criterion is satisfied -- promotion cannot fire early just because the
        other four criteria pass. After the 2-cycle restart guard only ~1
        prediction is recorded, which is below the AUTO_CAL_MIN_PREDICTIONS=3
        floor, so the gate stays closed."""
        w = lifecycle_wrapper
        # Cycle 0 -> restart guard (no prediction recorded).
        w.learn_from_prediction_feedback(
            predicted_temp=20.0, actual_temp=20.0,
            prediction_context=_active_context(),
        )
        # Cycle 1 -> restart guard again (still cycle_count <= 1).
        w.learn_from_prediction_feedback(
            predicted_temp=20.0, actual_temp=20.0,
            prediction_context=_active_context(),
        )
        # Cycle 2 -> first real learning cycle: 1 prediction recorded.
        w.learn_from_prediction_feedback(
            predicted_temp=20.0, actual_temp=20.0,
            prediction_context=_active_context(),
        )
        assert w.state_manager.get_learning_metrics()["total_predictions"] == 1
        trust = w.get_trust_state()
        assert trust["is_trusted"] is False
        assert trust["stage"] == "learning"
        assert trust["outlet_source"] == "heat_curve_passthrough"
