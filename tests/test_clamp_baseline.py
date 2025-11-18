import numpy as np
import pytest

from src import config


def compute_baseline_and_clamped(final_temp, actual_outlet_temp, state):
    """
    Re-implement the baseline selection and clamping logic from src.main
    so we can unit test the policy in isolation.

    Returns (baseline, clamped_final_temp)
    """
    max_change = config.MAX_TEMP_CHANGE_PER_CYCLE

    last_blocking_reasons = state.get("last_blocking_reasons", []) or []
    last_final_temp = state.get("last_final_temp")

    dhw_like_blockers = {
        config.DHW_STATUS_ENTITY_ID,
        config.DISINFECTION_STATUS_ENTITY_ID,
        config.DHW_BOOST_HEATER_STATUS_ENTITY_ID,
    }

    # Default baseline is persisted last_final_temp when present.
    # Override to measured actual_outlet_temp when last blocking reasons
    # include any DHW-like blocker (soft start).
    if last_final_temp is not None:
        baseline = last_final_temp
        if any(b in dhw_like_blockers for b in last_blocking_reasons):
            baseline = actual_outlet_temp
    else:
        baseline = actual_outlet_temp

    delta = final_temp - baseline
    if abs(delta) > max_change:
        clamped = baseline + np.clip(delta, -max_change, max_change)
    else:
        clamped = final_temp

    return baseline, clamped


def test_default_baseline_uses_last_final_temp():
    state = {"last_final_temp": 40.1, "last_blocking_reasons": []}
    baseline, clamped = compute_baseline_and_clamped(
        final_temp=45.0, actual_outlet_temp=41.0, state=state
    )
    assert baseline == pytest.approx(40.1)
    # delta = 4.9, max_change default is an integer; ensure clamping occurred
    expected = 40.1 + min(
        config.MAX_TEMP_CHANGE_PER_CYCLE, 45.0 - 40.1
    )
    assert clamped == pytest.approx(expected)


def test_dhw_like_uses_actual_outlet_temp_for_soft_start():
    # If last blocking reasons include a DHW-like blocker, baseline must be actual outlet temp
    state = {
        "last_final_temp": 40.1,
        "last_blocking_reasons": [config.DHW_STATUS_ENTITY_ID],
    }
    baseline, clamped = compute_baseline_and_clamped(
        final_temp=45.0, actual_outlet_temp=41.0, state=state
    )
    assert baseline == pytest.approx(41.0)
    # delta = 4.0 -> clamped relative to 41.0
    delta = 45.0 - 41.0
    expected = 41.0 + min(
        config.MAX_TEMP_CHANGE_PER_CYCLE,
        delta,
    )
    assert clamped == pytest.approx(expected)


def test_fallback_to_actual_when_no_last_final_temp():
    state = {"last_final_temp": None, "last_blocking_reasons": []}
    baseline, clamped = compute_baseline_and_clamped(
        final_temp=39.0, actual_outlet_temp=38.5, state=state
    )
    assert baseline == pytest.approx(38.5)
    # delta = 0.5 -> no clamping expected if max_change >= 1
    assert clamped == pytest.approx(39.0)
