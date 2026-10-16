"""OE delta mean-reversion safety-net tests (drift recurrence fix).

``outlet_effectiveness_delta`` (persisted in unified_thermal_state.json) is a
pure accumulator: every adaptation cycle adds the new OE adjustment with no
reversion path. The restoring force is the equilibrium-based OE gradient, but
it is gated out of active-heating cycles by the 0.15 C/cycle stability
threshold, while the biased trajectory gradient still fires on those cycles ->
one-sided permanent drift across a season (the 2026-10-12 manual reset fixed
the value, not the mechanism).

These tests pin the mean-reversion safety net:
  * revert-to-baseline on no-signal (fully-passive window AND active-but-unstable
    window) -- the persisted delta decays toward 0.0 and the in-memory OE is
    frozen.
  * freeze / byte-identical accumulation on a clean active signal (>=1 stable
    active record) -- the delta accumulates the new adjustment exactly as the
    legacy accumulator did.
  * flag-disabled fallback -- with OE_DELTA_MEAN_REVERSION_ENABLED=false the
    behaviour reverts to the pure accumulator even on no-signal windows.
"""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src import thermal_equilibrium_model
from src import config


class FakeStateManager:
    """Minimal stand-in for the unified thermal state manager.

    Only implements the surface used by _save_learning_to_thermal_state:
    a state dict with the parameter_adjustments structure and an
    update_learning_state() that actually persists the deltas so the test can
    read them back.
    """

    def __init__(self, oe_delta=0.0):
        self.state = {
            "baseline_parameters": {
                "source": "calibrated",
                "thermal_time_constant": 4.0,
                "heat_loss_coefficient": 0.4,
                "outlet_effectiveness": 0.5,
                "pv_heat_weight": 0.001,
                "fireplace_heat_weight": 1.5,
                "tv_heat_weight": 0.1,
            },
            "learning_state": {
                "learning_confidence": 3.0,
                "parameter_adjustments": {
                    "thermal_time_constant_delta": 0.0,
                    "heat_loss_coefficient_delta": 0.0,
                    "outlet_effectiveness_delta": oe_delta,
                    "pv_heat_weight_delta": 0.0,
                    "tv_heat_weight_delta": 0.0,
                },
            },
        }

    def update_learning_state(self, cycle_count=None, learning_confidence=None,
                               parameter_adjustments=None):
        ls = self.state["learning_state"]
        if cycle_count is not None:
            ls["cycle_count"] = cycle_count
        if learning_confidence is not None:
            ls["learning_confidence"] = learning_confidence
        if parameter_adjustments is not None:
            for k, v in parameter_adjustments.items():
                if k in ls["parameter_adjustments"]:
                    ls["parameter_adjustments"][k] = v
def _make_prediction_record(predicted, actual, outlet_temp, outdoor_temp,
                            current_indoor, heating_active=True, pv_power=0,
                            fireplace_on=0, tv_on=0, indoor_gradient=0.0):
    return {
        "predicted": predicted,
        "actual": actual,
        "error": actual - predicted,
        "timestamp": "now",
        "context": {
            "outlet_temp": outlet_temp,
            "outdoor_temp": outdoor_temp,
            "current_indoor": current_indoor,
            "pv_power": pv_power,
            "fireplace_on": fireplace_on,
            "tv_on": tv_on,
            "heating_active": heating_active,
            "indoor_temp_gradient": indoor_gradient,
        },
    }


@pytest.fixture(scope="function")
def model():
    """Fresh model with safe, in-bounds parameters and deterministic LR."""
    m = thermal_equilibrium_model.ThermalEquilibriumModel()
    m.thermal_time_constant = 4.0
    m.heat_loss_coefficient = 0.8
    m.outlet_effectiveness = 1.0
    m.pv_heat_weight = 0.001
    m.tv_heat_weight = 0.1
    m.solar_lag_minutes = 45.0
    # Deterministic learning rate: base = max(0.05, 0.001)*1.0 = 0.05, clipped
    # to max_learning_rate 0.01 -> 0.01. No oscillation / error boost.
    m.learning_confidence = 1.0
    m.learning_rate = 0.05
    m.min_learning_rate = 0.001
    m.max_learning_rate = 0.01
    m.parameter_history = []
    return m


def _run_adapt(model, records, oe_delta, oe_traj_grad=0.0, oe_eq_grad=0.0,
               reversion_enabled=True):
    """Drive one adaptation cycle against a FakeStateManager and return
    ``(fake, old_oe, old_delta)`` so the caller can assert on the result."""
    model.prediction_history = list(records)
    model.recent_errors_window = len(records)
    old_oe = model.outlet_effectiveness
    fake = FakeStateManager(oe_delta=oe_delta)
    old_delta = fake.state["learning_state"]["parameter_adjustments"][
        "outlet_effectiveness_delta"
    ]

    with patch("src.unified_thermal_state.get_thermal_state_manager",
               return_value=fake), \
         patch.object(config, "OE_DELTA_MEAN_REVERSION_ENABLED",
                       reversion_enabled), \
         patch.object(model, "_calculate_outlet_effectiveness_gradient",
                       return_value=oe_traj_grad), \
         patch.object(model, "_calculate_equilibrium_based_oe_gradient",
                       return_value=oe_eq_grad), \
         patch.object(model, "_calculate_thermal_time_constant_gradient",
                       return_value=0.0), \
         patch.object(model, "_calculate_heat_loss_coefficient_gradient",
                       return_value=0.0), \
         patch.object(model, "_calculate_pv_heat_weight_gradient",
                       return_value=0.0), \
         patch.object(model, "_calculate_tv_heat_weight_gradient",
                       return_value=0.0), \
         patch.object(model, "_calculate_solar_lag_gradient",
                       return_value=0.0):
        model._adapt_parameters_from_recent_errors()

    return fake, old_oe, old_delta


def _oe_delta(fake):
    return fake.state["learning_state"]["parameter_adjustments"][
        "outlet_effectiveness_delta"
    ]


# --- Tests: revert-to-baseline on no clean signal ------------------------

class TestRevertToBaselineOnNoSignal:
    def test_decays_when_all_passive(self, model):
        """Fully-passive window: persisted OE delta decays toward 0, in-memory
        OE frozen (no clean signal to learn OE from)."""
        recs = [
            _make_prediction_record(
                predicted=20.0, actual=19.5, outlet_temp=0.0,
                outdoor_temp=15.0, current_indoor=20.0, heating_active=False,
            )
            for _ in range(5)
        ]
        fake, old_oe, old_delta = _run_adapt(
            model, recs, oe_delta=-0.252, oe_traj_grad=0.0, oe_eq_grad=0.0,
        )
        new_delta = _oe_delta(fake)
        rate = config.OE_DELTA_MEAN_REVERSION_RATE
        assert new_delta == pytest.approx(old_delta * (1.0 - rate), rel=1e-9)
        # Magnitude strictly decreased (reverting toward baseline 0.0).
        assert abs(new_delta) < abs(old_delta)
        # In-memory OE frozen -- no clean signal, no biased update.
        assert model.outlet_effectiveness == pytest.approx(old_oe)

    def test_decays_and_freezes_when_active_but_unstable(self, model):
        """Active-but-unstable window (n_stable_active_records==0): the biased
        trajectory gradient is suppressed (in-memory OE frozen) and the
        persisted delta decays. This is the actual drift scenario the fix
        targets."""
        recs = [
            _make_prediction_record(
                predicted=21.0, actual=22.0, outlet_temp=45.0,
                outdoor_temp=15.0, current_indoor=21.0, heating_active=True,
                indoor_gradient=0.5,  # >= 0.15 stability threshold -> unstable
            )
            for _ in range(5)
        ]
        # Biased trajectory gradient would normally move OE; equilibrium
        # gradient is 0 (unstable records are skipped by the real method).
        fake, old_oe, old_delta = _run_adapt(
            model, recs, oe_delta=-0.252, oe_traj_grad=0.5, oe_eq_grad=0.0,
        )
        new_delta = _oe_delta(fake)
        rate = config.OE_DELTA_MEAN_REVERSION_RATE
        assert new_delta == pytest.approx(old_delta * (1.0 - rate), rel=1e-9)
        assert abs(new_delta) < abs(old_delta)
        # In-memory OE frozen despite a nonzero biased trajectory gradient.
        assert model.outlet_effectiveness == pytest.approx(old_oe)

    def test_decay_accumulates_over_multiple_no_signal_cycles(self, model):
        """Repeated no-signal cycles pull the delta monotonically toward 0."""
        recs = [
            _make_prediction_record(
                predicted=20.0, actual=19.5, outlet_temp=0.0,
                outdoor_temp=15.0, current_indoor=20.0, heating_active=False,
            )
            for _ in range(5)
        ]
        rate = config.OE_DELTA_MEAN_REVERSION_RATE
        delta = -0.252
        for _ in range(3):
            fake, _old_oe, _old_delta = _run_adapt(
                model, recs, oe_delta=delta, oe_traj_grad=0.0, oe_eq_grad=0.0,
            )
            delta = _oe_delta(fake)
        assert delta == pytest.approx(-0.252 * ((1.0 - rate) ** 3), rel=1e-9)


# --- Tests: byte-identical accumulation on clean active signal -----------

class TestFreezeOnCleanActiveSignal:
    def test_accumulates_normally_on_stable_active_signal(self, model):
        """Clean OE signal (>=1 stable active record): the persisted delta
        accumulates the new adjustment exactly as the legacy accumulator --
        no decay applied. Byte-identical to pre-fix behaviour."""
        recs = [
            _make_prediction_record(
                predicted=21.0, actual=22.0, outlet_temp=45.0,
                outdoor_temp=15.0, current_indoor=21.0, heating_active=True,
                indoor_gradient=0.0,  # stable -> equilibrium signal fires
            )
            for _ in range(5)
        ]
        # Equilibrium gradient (clean signal) drives the update.
        fake, old_oe, old_delta = _run_adapt(
            model, recs, oe_delta=-0.2, oe_traj_grad=0.0, oe_eq_grad=0.5,
        )
        new_delta = _oe_delta(fake)
        oe_change = model.outlet_effectiveness - old_oe
        # OE actually moved (the clean signal was applied).
        assert oe_change != 0.0
        # Legacy accumulation: new = old + change (NOT the decayed product).
        assert new_delta == pytest.approx(old_delta + oe_change, rel=1e-9)
        # And specifically NOT the mean-reverted value.
        rate = config.OE_DELTA_MEAN_REVERSION_RATE
        assert new_delta != pytest.approx(old_delta * (1.0 - rate), rel=1e-3)

    def test_byte_identical_with_flag_on_vs_off_on_clean_signal(self, model):
        """On a clean active signal the resulting persisted delta is the same
        whether the flag is on or off -- the flag only affects no-signal
        cycles."""
        recs = [
            _make_prediction_record(
                predicted=21.0, actual=22.0, outlet_temp=45.0,
                outdoor_temp=15.0, current_indoor=21.0, heating_active=True,
                indoor_gradient=0.0,
            )
            for _ in range(5)
        ]
        fake_on, _oe_on, _ = _run_adapt(
            model, recs, oe_delta=-0.2, oe_traj_grad=0.0, oe_eq_grad=0.5,
            reversion_enabled=True,
        )
        delta_on = _oe_delta(fake_on)
        # Rebuild a fresh model for the flag-off run.
        m2 = thermal_equilibrium_model.ThermalEquilibriumModel()
        m2.thermal_time_constant = 4.0
        m2.heat_loss_coefficient = 0.8
        m2.outlet_effectiveness = 1.0
        m2.pv_heat_weight = 0.001
        m2.tv_heat_weight = 0.1
        m2.solar_lag_minutes = 45.0
        m2.learning_confidence = 1.0
        m2.learning_rate = 0.05
        m2.min_learning_rate = 0.001
        m2.max_learning_rate = 0.01
        m2.parameter_history = []
        fake_off, _oe_off, _ = _run_adapt(
            m2, recs, oe_delta=-0.2, oe_traj_grad=0.0, oe_eq_grad=0.5,
            reversion_enabled=False,
        )
        delta_off = _oe_delta(fake_off)
        assert delta_on == pytest.approx(delta_off, rel=1e-9)


# --- Tests: flag-disabled fallback --------------------------------------

class TestFlagDisabledFallback:
    def test_disabled_pure_accumulator_on_no_signal(self, model):
        """With the flag off, an active-but-unstable (no-signal) window behaves
        exactly as the legacy pure accumulator: the biased trajectory gradient
        moves the in-memory OE and the persisted delta accumulates it (no
        decay, no suppression)."""
        recs = [
            _make_prediction_record(
                predicted=21.0, actual=22.0, outlet_temp=45.0,
                outdoor_temp=15.0, current_indoor=21.0, heating_active=True,
                indoor_gradient=0.5,
            )
            for _ in range(5)
        ]
        fake, old_oe, old_delta = _run_adapt(
            model, recs, oe_delta=-0.2, oe_traj_grad=0.5, oe_eq_grad=0.0,
            reversion_enabled=False,
        )
        new_delta = _oe_delta(fake)
        oe_change = model.outlet_effectiveness - old_oe
        # Flag off -> biased update is NOT suppressed.
        assert oe_change != 0.0
        # Flag off -> pure accumulation, no decay.
        assert new_delta == pytest.approx(old_delta + oe_change, rel=1e-9)


# --- Tests: InfluxDB persistence of OE delta + telemetry -----------------

class TestOEDeltaInfluxPersistence:
    """``get_adaptive_learning_metrics`` surfaces the persisted
    ``outlet_effectiveness_delta`` and the per-cycle OE telemetry stashed by
    the optimizer, so the InfluxDB export can chart drift-vs-heal."""

    def test_metrics_expose_oe_delta_and_telemetry_after_adapt(self, model):
        """After a clean-signal adaptation cycle the metrics dict carries the
        persisted OE delta (read back from the state manager) and the exact
        telemetry the optimizer stashed on the model."""
        recs = [
            _make_prediction_record(
                predicted=21.0, actual=22.0, outlet_temp=45.0,
                outdoor_temp=15.0, current_indoor=21.0, heating_active=True,
                indoor_gradient=0.0,
            )
            for _ in range(5)
        ]
        fake, _old_oe, _old_delta = _run_adapt(
            model, recs, oe_delta=-0.2, oe_traj_grad=0.0, oe_eq_grad=0.5,
        )
        expected_delta = _oe_delta(fake)

        # get_adaptive_learning_metrics reads the state manager, so re-establish
        # the same patch the _run_adapt context manager has already exited.
        with patch("src.unified_thermal_state.get_thermal_state_manager",
                   return_value=fake):
            metrics = model.get_adaptive_learning_metrics()

        assert "outlet_effectiveness_delta" in metrics
        assert metrics["outlet_effectiveness_delta"] == pytest.approx(
            expected_delta, rel=1e-9
        )
        tel = metrics["oe_telemetry"]
        assert isinstance(tel, dict)
        assert tel.get("trajectory_oe_gradient") == pytest.approx(0.0, abs=1e-9)
        assert tel.get("equilibrium_oe_gradient") == pytest.approx(0.5, abs=1e-9)
        assert tel.get("n_stable_active_records") == 5
        assert tel.get("n_active") == 5
        assert tel.get("n_passive") == 0
        # clean signal -> mean_revert must be False
        assert tel.get("oe_mean_revert") is False

    def test_metrics_default_when_no_adaptation_ran(self, model):
        """A fresh model that has never adapted exposes an empty telemetry dict
        and a zero OE delta (read from a clean state manager)."""
        model.prediction_history = [
            _make_prediction_record(
                predicted=20.0, actual=20.1, outlet_temp=45.0,
                outdoor_temp=15.0, current_indoor=20.0, heating_active=True,
            )
        ]
        fake = FakeStateManager(oe_delta=0.0)
        with patch("src.unified_thermal_state.get_thermal_state_manager",
                   return_value=fake):
            metrics = model.get_adaptive_learning_metrics()
        assert metrics["outlet_effectiveness_delta"] == pytest.approx(0.0)
        assert metrics["oe_telemetry"] == {}


