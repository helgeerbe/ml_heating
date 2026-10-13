"""End-to-end integration tests for ``src/main.py``'s ``main()`` control loop.

The existing ``tests/integration/test_main.py`` drives one cycle but uses an
empty state (so the online-learning block is skipped) and an integer
``final_temp`` (so model-based smart rounding is skipped) and a ``MagicMock``
trust gate (so the rich trust-attribute block errors out).  These tests
close those gaps by driving the real control-flow code with deterministic,
fully-mocked externals so the *entire* heating-active path executes:

* online learning from a previous cycle (passive vs active vs shadow),
* feature build + prediction,
* model-based smart rounding (floor vs ceiling via ``predict_indoor_temp``),
* the self-calibration trust-gate sensor update (rich attributes + fallback),
* shadow-mode observation + comparison logging,
* grace-period passthrough state save.

Only ``main()``'s own control flow is exercised; every external dependency
(Home Assistant, InfluxDB, the model wrapper, sensor/blocking managers) is
mocked, so these are pure, fast, and deterministic.
"""
import contextlib
from unittest.mock import MagicMock, patch

import pytest

from src import config, main
from src.state_manager import SystemState


def _trust_state(stage="active", is_trusted=True):
    """A realistic ``get_trust_state`` return value (see model_wrapper.py)."""
    return {
        "is_trusted": is_trusted,
        "is_calibrated": is_trusted,
        "is_accurate": True,
        "is_confident": True,
        "confidence": 3.0,
        "stage": stage,
        "trust_reason": "active" if is_trusted else "passthrough",
        "promotion_progress": 1.0 if is_trusted else 0.3,
        "ml_predicted_outlet": 43.5,
        "heat_curve_outlet": 43.0,
        "outlet_source": "ml_predicted" if is_trusted else "heat_curve_passthrough",
        "calibration_method": "calibrated",
        "total_predictions": 500,
    }


def _all_states():
    """A realistic HA state snapshot used by the mocked ``get_state``."""
    return {
        config.ML_HEATING_CONTROL_ENTITY_ID: {"state": "on"},
        config.INDOOR_TEMP_ENTITY_ID: {"state": "21.0"},
        config.ACTUAL_OUTLET_TEMP_ENTITY_ID: {"state": "35.0"},
        config.TARGET_OUTLET_TEMP_ENTITY_ID: {"state": "43.0"},
        config.OUTDOOR_TEMP_ENTITY_ID: {"state": "10.0"},
        config.INLET_TEMP_ENTITY_ID: {"state": "25.0"},
        config.FLOW_RATE_ENTITY_ID: {"state": "5.0"},
        config.POWER_CONSUMPTION_ENTITY_ID: {"state": "1.5"},
        config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID: {"state": "43.0"},
        config.DHW_STATUS_ENTITY_ID: {"state": "off"},
        config.DEFROST_STATUS_ENTITY_ID: {"state": "off"},
        config.DISINFECTION_STATUS_ENTITY_ID: {"state": "off"},
        config.DHW_BOOST_HEATER_STATUS_ENTITY_ID: {"state": "off"},
    }


def _sensor_data():
    """What the mocked ``SensorDataManager.get_sensor_data`` returns."""
    return {
        "target_indoor_temp": 21.0,
        "actual_indoor": 21.0,
        "actual_outlet_temp": 43.0,
        "avg_other_rooms_temp": 21.0,
        "fireplace_on": False,
        "outdoor_temp": 10.0,
        "owm_temp": 10.0,
    }, []


def _features():
    """A dict feature set (the simplest branch in main's feature handling)."""
    return {
        "outdoor_temp": 10.0,
        "pv_now": 0.0,
        "fireplace_on": 0,
        "tv_on": 0,
        "heat_curve_temp": 43.0,
        "cop_realtime": 4.0,
        "thermal_power_kw": 2.0,
        "delta_t": 18.0,
        "flow_rate": 5.0,
        "inlet_temp": 25.0,
    }


def _prior_state(heating_active=True):
    """A populated previous-cycle state so the online-learning block fires."""
    return SystemState(
        last_run_features={
            "outdoor_temp": 10.0,
            "pv_now": 0.0,
            "fireplace_on": 0,
            "tv_on": 0,
            "heating_active": heating_active,
            "thermal_power_kw": 1.0,
        },
        last_indoor_temp=20.5,
        last_final_temp=43.0,
    )
def _run_main(
    *,
    shadow_mode=False,
    prior_state=None,
    trust_state=None,
    trust_raises=False,
    heating_active=True,
    final_temp=43.5,
    grace=False,
):
    """Run exactly one ``main()`` cycle with every external mocked.

    ``grace=True`` makes the cycle hit the grace-period ``continue`` path; the
    loop is then broken on the second cycle via ``KeyboardInterrupt`` (a
    ``BaseException`` that escapes ``main()``'s ``except Exception``).
    Otherwise the single cycle runs to completion and is broken by
    ``poll_for_blocking`` raising ``StopIteration`` (outside the ``try`` block).

    Returns a dict of the mock objects (call records survive patch exit).
    """
    mocks = {}
    breaker = StopIteration("end-of-cycle")

    with contextlib.ExitStack() as stack:
        mocks["load_dotenv"] = stack.enter_context(patch("src.main.load_dotenv"))
        mocks["logging"] = stack.enter_context(patch("src.main.logging"))
        mocks["create_influx"] = stack.enter_context(patch("src.main.create_influx_service"))
        mocks["sensor_buffer"] = stack.enter_context(patch("src.main.SensorBuffer"))
        stack.enter_context(patch("src.main.config.SHADOW_MODE", shadow_mode))
        # Pin the confidence threshold so the trust-gate fallback assertion is
        # deterministic regardless of the live .env override (code default 0.2,
        # .env override 2.0). confidence 0.9 >= 0.5 -> "OK - Prediction done".
        stack.enter_context(patch("src.main.config.CONFIDENCE_THRESHOLD", 0.5))
        mocks["load_state"] = stack.enter_context(patch("src.main.load_state"))
        mocks["save_state"] = stack.enter_context(patch("src.main.save_state"))
        mocks["create_ha"] = stack.enter_context(patch("src.main.create_ha_client"))
        mocks["get_sensor_attrs"] = stack.enter_context(patch("src.main.get_sensor_attributes"))
        mocks["calc_thermo"] = stack.enter_context(patch("src.main.calculate_thermodynamic_metrics"))
        mocks["build_features"] = stack.enter_context(patch("src.main.build_physics_features"))
        mocks["predict"] = stack.enter_context(patch("src.main.simplified_outlet_prediction"))
        mocks["get_wrapper"] = stack.enter_context(patch("src.model_wrapper.get_enhanced_model_wrapper"))
        mocks["pred_ctx"] = stack.enter_context(patch("src.prediction_context.prediction_context_manager"))
        stack.enter_context(patch("src.main.time.sleep"))
        MockBSM = stack.enter_context(patch("src.main.BlockingStateManager"))
        MockHSC = stack.enter_context(patch("src.main.HeatingSystemStateChecker"))
        MockSDM = stack.enter_context(patch("src.main.SensorDataManager"))

        ha_client = MagicMock()
        all_states = _all_states()
        ha_client.get_all_states.return_value = all_states

        def _get_state(entity_id, states_dict, is_binary=False):
            info = states_dict.get(entity_id)
            if not info:
                return None
            state = info.get("state")
            if is_binary:
                return state == "on"
            try:
                return float(state)
            except (ValueError, TypeError):
                return state

        ha_client.get_state.side_effect = _get_state
        mocks["load_state"].return_value = (
            prior_state if prior_state is not None else SystemState()
        )
        mocks["create_influx"].return_value = MagicMock()
        # Fresh dict per call so the trust-gate block can't mutate the dict
        # recorded against the earlier target-outlet set_state call.
        mocks["get_sensor_attrs"].side_effect = lambda *args, **kwargs: {}
        mocks["calc_thermo"].return_value = {"cop_realtime": 4.0, "thermal_power_kw": 2.0}
        mocks["build_features"].return_value = (_features(), [])
        mocks["predict"].return_value = (
            final_temp, 0.9,
            {"ml_outlet_temp": final_temp, "predicted_indoor": 21.0,
             "uncalibrated_capped": False, "model_calibrated": True},
        )
        wrapper = MagicMock()
        wrapper.export_metrics_to_ha.return_value = None
        wrapper.predict_indoor_temp.return_value = 21.0
        wrapper.thermal_model.predict_thermal_trajectory.return_value = {"trajectory": [21.0]}
        wrapper.learn_from_prediction_feedback.return_value = None
        wrapper.prediction_metrics.get_metrics.return_value = {}
        if trust_raises:
            wrapper.get_trust_state.side_effect = RuntimeError("trust gate down")
        else:
            wrapper.get_trust_state.return_value = (
                trust_state if trust_state is not None else _trust_state()
            )
        mocks["get_wrapper"].return_value = wrapper
        mocks["pred_ctx"].get_thermal_model_params.return_value = {
            "outdoor_temp": 10.0, "pv_power": 0.0, "fireplace_on": 0.0, "tv_on": 0.0}
        mocks["pred_ctx"].create_context.return_value = MagicMock()
        bsm = MockBSM.return_value
        bsm.check_blocking_state.return_value = (False, [])
        bsm.handle_grace_period.return_value = grace
        MockHSC.return_value.check_heating_active.return_value = heating_active
        MockSDM.return_value.get_sensor_data.return_value = _sensor_data()
        if grace:
            mocks["create_ha"].side_effect = [ha_client, KeyboardInterrupt("end")]
            breaker = KeyboardInterrupt("end")
        else:
            mocks["create_ha"].return_value = ha_client
            bsm.poll_for_blocking.side_effect = StopIteration("end-of-cycle")
        mocks["ha_client"] = ha_client
        mocks["wrapper"] = wrapper
        mocks["bsm"] = bsm
        with patch("sys.argv", ["main.py"]):
            with pytest.raises(type(breaker), match="end"):
                main.main()

    return mocks



def test_main_loop_active_full_cycle_with_online_learning():
    """Active + trusted: full control path with online learning from a prior
    heating-active cycle, model-based smart rounding, and the rich trust-gate
    sensor update."""
    mocks = _run_main(
        shadow_mode=False,
        prior_state=_prior_state(heating_active=True),
        trust_state=_trust_state(stage="active", is_trusted=True),
        final_temp=43.5,
    )

    # Online learning fired against the previous cycle.
    mocks["wrapper"].thermal_model.predict_thermal_trajectory.assert_called_once()
    mocks["wrapper"].learn_from_prediction_feedback.assert_called_once()

    # Prediction + smart rounding (43.5 -> floor 43 since both predict 21.0).
    mocks["predict"].assert_called_once()
    mocks["wrapper"].predict_indoor_temp.assert_called()  # floor + ceiling + applied

    # The rounded target was written to HA.
    mocks["ha_client"].set_state.assert_any_call(
        config.TARGET_OUTLET_TEMP_ENTITY_ID, 43, {}, round_digits=None
    )

    # The ML state sensor shows ML is driving (state 1, "active").
    state_calls = [
        c for c in mocks["ha_client"].set_state.call_args_list
        if c.args and c.args[0] == "sensor.ml_heating_state"
    ]
    assert state_calls, "sensor.ml_heating_state must be updated every cycle"
    assert state_calls[-1].args[1] == 1  # active + trusted -> 1

    # State persisted with the heating-active flag for next cycle's learning.
    mocks["save_state"].assert_called()
    saved = mocks["save_state"].call_args.kwargs
    assert saved["last_final_temp"] == 43.5
    assert saved["last_indoor_temp"] == 21.0
    assert saved["last_run_features"]["heating_active"] is True


def test_main_loop_active_trust_gate_fallback():
    """Active mode but ``get_trust_state`` raises -> confidence-only fallback
    attributes (covers the ``trust is None`` branch of the sensor update)."""
    mocks = _run_main(
        shadow_mode=False,
        prior_state=None,  # no prior data -> online learning skipped
        trust_raises=True,
        final_temp=43.0,
    )

    # No prior-cycle data, so online learning must NOT fire.
    mocks["wrapper"].learn_from_prediction_feedback.assert_not_called()

    # Fallback path still updates the sensor (confidence >= threshold -> 1).
    state_calls = [
        c for c in mocks["ha_client"].set_state.call_args_list
        if c.args and c.args[0] == "sensor.ml_heating_state"
    ]
    assert state_calls
    attrs = state_calls[-1].args[2]
    assert attrs["ml_control_mode"] == "unknown"
    # confidence 0.9 >= CONFIDENCE_THRESHOLD(0.2) -> "OK - Prediction done"
    assert attrs["state_description"] == "OK - Prediction done"


def test_main_loop_shadow_mode_cycle():
    """Shadow mode: ML prediction is calculated but NOT applied; the heat-curve
    comparison is logged and the sensor reports state 0 ("shadow")."""
    mocks = _run_main(
        shadow_mode=True,
        prior_state=_prior_state(heating_active=True),
        trust_state=_trust_state(stage="active", is_trusted=True),
        final_temp=43.5,  # != heat-curve 43.0 so comparison logs
    )

    # Shadow learning context was used.
    mocks["wrapper"].learn_from_prediction_feedback.assert_called_once()

    # The target outlet must NOT be written in shadow mode.
    target_writes = [
        c for c in mocks["ha_client"].set_state.call_args_list
        if c.args and c.args[0] == config.TARGET_OUTLET_TEMP_ENTITY_ID
    ]
    assert not target_writes, "Shadow mode must not write the target outlet"

    # Sensor still updated (T0.7: all modes), value 0 because shadow.
    state_calls = [
        c for c in mocks["ha_client"].set_state.call_args_list
        if c.args and c.args[0] == "sensor.ml_heating_state"
    ]
    assert state_calls[-1].args[1] == 0
    assert state_calls[-1].args[2]["ml_control_mode"] == "shadow"

    # State saved with the (un-rounded) final_temp.
    saved = mocks["save_state"].call_args.kwargs
    assert saved["last_final_temp"] == 43.5


def test_main_loop_grace_period_passthrough():
    """Grace period after blocking: control is skipped, the previous valid
    target is preserved (no state poisoning), and a grace-period passthrough
    state is saved for the next cycle's learning block."""
    mocks = _run_main(
        shadow_mode=False,
        prior_state=None,  # last_final_temp None -> exercises the fallback
        grace=True,
    )

    # Control logic was skipped entirely.
    mocks["predict"].assert_not_called()
    mocks["build_features"].assert_not_called()

    # Grace-period passthrough state saved with a preserved target.
    mocks["save_state"].assert_called()
    saved = mocks["save_state"].call_args.kwargs
    assert saved["last_is_blocking"] is False
    assert saved["last_run_features"] == {"learning_mode": "grace_period_passthrough"}
    # last_final_temp was None so the fallback read ACTUAL_OUTLET (35.0).
    assert saved["last_final_temp"] == 35.0

