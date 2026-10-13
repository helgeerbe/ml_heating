"""
Dedicated unit tests for src/prediction_metrics.py.

These tests exercise the PredictionMetrics tracker directly (state-manager
backed and in-memory), covering rolling-window MAE/RMSE calculations, trend
analysis, accuracy breakdowns, file persistence, the 24h-window helpers used by
the trust gate, the module-level convenience functions and the __main__ demo
block.

A lightweight FakeStateManager stands in for ThermalStateManager so the tests
never touch a live thermal-state file (avoiding the fixture contamination
noted for the broader suite).
"""
import json
import runpy
from collections import deque
from datetime import datetime, timedelta, timezone

import pytest

from src.prediction_metrics import (
    PredictionMetrics,
    prediction_tracker,
    track_prediction,
    get_current_metrics,
    get_metrics_summary,
)


class FakeStateManager:
    """Minimal stand-in for ThermalStateManager used by PredictionMetrics."""

    def __init__(self, history=None):
        records = list(history or [])
        self.state = {
            "learning_state": {"prediction_history": records},
            "prediction_metrics": {"total_predictions": len(records)},
        }
        self.save_calls = 0

    def save_state(self):
        self.save_calls += 1
        return True


class ExplodingStateManager:
    """state access raises, to exercise _load_from_state's except branch."""

    @property
    def state(self):
        raise RuntimeError("state unavailable")

    def save_state(self):
        return True


class SaveRaisingStateManager(FakeStateManager):
    """save_state() raises, to exercise _save_to_state's except branch."""

    def save_state(self):
        raise RuntimeError("save failed")


def _record(predicted, actual, timestamp=None):
    """Build a unified-state prediction record (with 'error' only)."""
    return {
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "predicted": predicted,
        "actual": actual,
        "error": actual - predicted,
        "context": {},
    }


# --- __init__ / _load_from_state -------------------------------------------


def test_init_no_state_manager_uses_in_memory_deque():
    metrics = PredictionMetrics()
    assert isinstance(metrics.predictions, deque)
    assert metrics.max_history_size == 1000
    assert len(metrics.predictions) == 0
    assert metrics.state_manager is None


def test_init_with_state_manager_loads_history_and_derives_fields():
    history = [_record(20.0, 20.5), _record(21.0, 20.0)]
    sm = FakeStateManager(history=history)
    metrics = PredictionMetrics(state_manager=sm)

    assert len(metrics.predictions) == 2
    rec = list(metrics.predictions)[0]
    assert rec["abs_error"] == pytest.approx(abs(20.5 - 20.0))
    assert rec["squared_error"] == pytest.approx((20.5 - 20.0) ** 2)


def test_load_from_state_exception_falls_back_to_in_memory_deque():
    metrics = PredictionMetrics(state_manager=ExplodingStateManager())
    assert isinstance(metrics.predictions, deque)
    assert len(metrics.predictions) == 0


def test_load_from_state_records_already_having_abs_error_are_kept():
    history = [
        {"timestamp": datetime.now(timezone.utc).isoformat(),
         "predicted": 20.0, "actual": 20.3,
         "error": 0.3, "abs_error": 0.3, "squared_error": 0.09, "context": {}}
    ]
    metrics = PredictionMetrics(state_manager=FakeStateManager(history=history))
    rec = list(metrics.predictions)[0]
    assert rec["abs_error"] == 0.3
    assert rec["squared_error"] == 0.09


# --- _save_to_state --------------------------------------------------------


def test_save_to_state_skipped_without_state_manager():
    metrics = PredictionMetrics()
    metrics.predictions.append(_record(20.0, 20.0))
    metrics._save_to_state()  # no-op early return (line 91)
    assert len(metrics.predictions) == 1


def test_save_to_state_trims_to_200_and_persists():
    sm = FakeStateManager()
    metrics = PredictionMetrics(state_manager=sm)
    for i in range(205):
        metrics.predictions.append(_record(20.0, 20.0 + i * 0.01))

    metrics._save_to_state()

    persisted = sm.state["learning_state"]["prediction_history"]
    assert len(persisted) == 200  # trimmed (line 99)
    assert sm.save_calls == 1
    assert sm.state["prediction_metrics"]["total_predictions"] == 200


def test_save_to_state_exception_is_swallowed():
    sm = SaveRaisingStateManager()
    metrics = PredictionMetrics(state_manager=sm)
    metrics.predictions.append(_record(20.0, 20.0))
    metrics._save_to_state()  # must not raise (lines 112-113)


# --- add_prediction --------------------------------------------------------


def test_add_prediction_creates_record_and_invalidates_cache():
    sm = FakeStateManager()
    metrics = PredictionMetrics(state_manager=sm)
    metrics._cache_timestamp = datetime.now(timezone.utc)

    metrics.add_prediction(20.0, 20.5, context={"outlet_temp": 45.0})

    assert metrics._cache_timestamp is None
    rec = list(metrics.predictions)[0]
    assert rec["predicted"] == 20.0
    assert rec["actual"] == 20.5
    assert rec["error"] == pytest.approx(0.5)
    assert rec["abs_error"] == pytest.approx(0.5)
    assert rec["squared_error"] == pytest.approx(0.25)
    assert rec["context"] == {"outlet_temp": 45.0}
    assert sm.save_calls == 1


def test_add_prediction_trims_beyond_200_records():
    sm = FakeStateManager()
    metrics = PredictionMetrics(state_manager=sm)
    for i in range(205):
        metrics.add_prediction(20.0, 20.0 + i * 0.01)
    assert len(metrics.predictions) == 200  # lines 146-147


def test_add_prediction_explicit_timestamp_is_used():
    metrics = PredictionMetrics()
    metrics.add_prediction(20.0, 20.0, timestamp="2024-01-01T00:00:00")
    assert list(metrics.predictions)[0]["timestamp"] == "2024-01-01T00:00:00"


# --- get_metrics / _calculate_window_metrics / caching --------------------


def test_get_metrics_empty_tracker():
    m = PredictionMetrics().get_metrics()
    for window in ("1h", "6h", "24h", "all"):
        assert m[window]["count"] == 0
        assert m[window]["mae"] == 0.0
        assert m[window]["rmse"] == 0.0
    assert m["accuracy_breakdown"] == {}
    assert m["trends"].get("insufficient_data") is True


def test_get_metrics_with_data_has_all_windows():
    metrics = PredictionMetrics()
    for i in range(20):
        metrics.add_prediction(20.0, 20.0 + (i % 3) * 0.1)
    m = metrics.get_metrics()
    assert m["all"]["count"] == 20
    assert m["1h"]["count"] <= 12
    assert m["24h"]["count"] == 20
    assert m["all"]["mae"] > 0
    assert "rmse" in m["all"]
    assert set(m["accuracy_breakdown"]) == {
        "excellent", "very_good", "good", "acceptable", "poor"
    }


def test_get_metrics_uses_cache_within_window():
    metrics = PredictionMetrics()
    metrics.add_prediction(20.0, 20.1)
    first = metrics.get_metrics()
    second = metrics.get_metrics()  # within 60s -> cached
    assert first is second


def test_get_metrics_refresh_cache_recomputes():
    metrics = PredictionMetrics()
    metrics.add_prediction(20.0, 20.1)
    first = metrics.get_metrics()
    second = metrics.get_metrics(refresh_cache=True)
    assert first is not second


def test_calculate_window_metrics_empty_window_is_zeroed():
    result = PredictionMetrics()._calculate_window_metrics(12)
    assert result["count"] == 0
    assert result["mae"] == 0.0


# --- _calculate_trends -----------------------------------------------------


def test_trends_insufficient_data_with_single_prediction():
    metrics = PredictionMetrics()
    metrics.add_prediction(20.0, 20.1)
    assert metrics._calculate_trends().get("insufficient_data") is True


def test_trends_improving():
    # _calculate_trends requires >= 10 predictions (line 248 guard).
    metrics = PredictionMetrics()
    for _ in range(6):  # first half: large errors
        metrics.add_prediction(20.0, 21.0)
    for _ in range(6):  # second half: small errors
        metrics.add_prediction(20.0, 20.1)
    trends = metrics._calculate_trends()
    assert bool(trends["is_improving"]) is True
    assert trends["mae_improvement"] > 0
    assert trends["first_half_mae"] > trends["second_half_mae"]


def test_trends_degrading_and_percentage_clamped():
    metrics = PredictionMetrics()
    for _ in range(6):  # first half: tiny error
        metrics.add_prediction(20.0, 20.001)
    for _ in range(6):  # second half: huge error
        metrics.add_prediction(20.0, 40.0)
    trends = metrics._calculate_trends()
    assert bool(trends["is_improving"]) is False
    assert trends["mae_improvement"] < 0
    assert -100.0 <= trends["mae_improvement_percentage"] <= 100.0


def test_trends_zero_first_half_mae_yields_zero_percentage():
    metrics = PredictionMetrics()
    for _ in range(6):  # first half: zero error
        metrics.add_prediction(20.0, 20.0)
    for _ in range(6):  # second half: non-zero error
        metrics.add_prediction(20.0, 21.0)
    trends = metrics._calculate_trends()
    assert trends["first_half_mae"] == 0.0
    assert trends["mae_improvement_percentage"] == 0  # line 277


# --- get_recent_performance ------------------------------------------------


def test_get_recent_performance_no_data():
    assert PredictionMetrics().get_recent_performance() == {"no_data": True}


def test_get_recent_performance_with_data():
    metrics = PredictionMetrics()
    for i in range(5):
        metrics.add_prediction(20.0, 20.0 + i * 0.1)
    perf = metrics.get_recent_performance(last_n=3)
    assert perf["count"] == 3
    assert perf["mae"] > 0
    assert perf["max_error"] >= perf["min_error"]


def test_get_recent_performance_caps_to_available():
    metrics = PredictionMetrics()
    metrics.add_prediction(20.0, 20.5)
    assert metrics.get_recent_performance(last_n=10)["count"] == 1


# --- save_state / load_state (file persistence) ----------------------------


def test_save_state_writes_json_file(tmp_path):
    metrics = PredictionMetrics()
    metrics.add_prediction(20.0, 20.5)
    filepath = str(tmp_path / "metrics.json")
    metrics.save_state(filepath)
    assert metrics.state_file == filepath
    with open(filepath) as f:
        data = json.load(f)
    assert len(data["predictions"]) == 1
    assert data["max_history_size"] == 1000
    assert "saved_at" in data


def test_load_state_restores_predictions(tmp_path):
    metrics = PredictionMetrics()
    for i in range(3):
        metrics.add_prediction(20.0, 20.0 + i * 0.1)
    filepath = str(tmp_path / "metrics.json")
    metrics.save_state(filepath)
    other = PredictionMetrics()
    assert other.load_state(filepath) is True
    assert len(other.predictions) == 3
    assert other.state_file == filepath
    assert other._cache_timestamp is None


def test_load_state_missing_file_returns_false(tmp_path):
    assert PredictionMetrics().load_state(str(tmp_path / "nope.json")) is False


def test_load_state_corrupt_file_returns_false(tmp_path):
    filepath = str(tmp_path / "corrupt.json")
    with open(filepath, "w") as f:
        f.write("{not valid json")
    assert PredictionMetrics().load_state(filepath) is False


def test_save_state_unwritable_path_is_swallowed(tmp_path):
    # Writing into a non-existent directory raises -> except branch (359-360).
    metrics = PredictionMetrics()
    metrics.add_prediction(20.0, 20.5)
    metrics.save_state(str(tmp_path / "no_such_dir" / "metrics.json"))
    assert not hasattr(metrics, "state_file") or metrics.state_file != str(
        tmp_path / "no_such_dir" / "metrics.json"
    )


# --- get_summary -----------------------------------------------------------


def test_get_summary_empty():
    assert PredictionMetrics().get_summary() == "No prediction data available"


def test_get_summary_improving_trend():
    metrics = PredictionMetrics()
    for _ in range(6):
        metrics.add_prediction(20.0, 21.0)
    for _ in range(6):
        metrics.add_prediction(20.0, 20.1)
    summary = metrics.get_summary()
    assert "Prediction Accuracy Summary" in summary
    assert "improving" in summary


def test_get_summary_degrading_trend():
    metrics = PredictionMetrics()
    for _ in range(6):
        metrics.add_prediction(20.0, 20.1)
    for _ in range(6):
        metrics.add_prediction(20.0, 21.0)
    summary = metrics.get_summary()
    assert "degrading" in summary


def test_get_summary_insufficient_trend_has_no_trend_str():
    metrics = PredictionMetrics()
    metrics.add_prediction(20.0, 20.5)
    summary = metrics.get_summary()
    assert "Prediction Accuracy Summary" in summary
    assert "improving" not in summary
    assert "degrading" not in summary


# --- get_simplified_accuracy_breakdown / get_good_control_percentage -------


def test_simplified_breakdown_empty():
    bd = PredictionMetrics().get_simplified_accuracy_breakdown()
    assert bd == {
        "perfect": {"count": 0, "percentage": 0.0},
        "tolerable": {"count": 0, "percentage": 0.0},
        "poor": {"count": 0, "percentage": 0.0},
    }


def test_simplified_breakdown_with_data():
    metrics = PredictionMetrics()
    metrics.add_prediction(20.0, 20.0)      # perfect (0.0)
    metrics.add_prediction(20.0, 20.05)      # tolerable (<0.2)
    metrics.add_prediction(20.0, 20.5)       # poor (>=0.2)
    bd = metrics.get_simplified_accuracy_breakdown()
    assert bd["perfect"]["count"] == 1
    assert bd["tolerable"]["count"] == 1
    assert bd["poor"]["count"] == 1
    total = bd["perfect"]["percentage"] + bd["tolerable"]["percentage"] + bd["poor"]["percentage"]
    assert total == pytest.approx(100.0)


def test_good_control_percentage():
    metrics = PredictionMetrics()
    metrics.add_prediction(20.0, 20.0)   # perfect
    metrics.add_prediction(20.0, 20.05)  # tolerable
    metrics.add_prediction(20.0, 20.5)   # poor
    assert metrics.get_good_control_percentage() == pytest.approx((2 / 3) * 100)


# --- 24h-window helpers ----------------------------------------------------


def _add_with_offset(metrics, hours_ago, error):
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    metrics.add_prediction(20.0, 20.0 + error, timestamp=ts)


def test_get_predictions_in_24h_window_filters_by_time():
    metrics = PredictionMetrics()
    _add_with_offset(metrics, 1, 0.1)    # within 24h
    _add_with_offset(metrics, 25, 0.2)  # outside 24h
    window = metrics._get_predictions_in_24h_window()
    assert len(window) == 1
    assert window[0]["abs_error"] == pytest.approx(0.1)


def test_get_predictions_in_24h_window_skips_none_timestamp():
    metrics = PredictionMetrics()
    # add_prediction replaces a None timestamp with now(), so to exercise the
    # None-timestamp skip branch (line 507) we append a raw record directly.
    metrics.predictions.append(
        {"timestamp": None, "abs_error": 0.1, "error": 0.1,
         "predicted": 20.0, "actual": 20.1, "context": {}}
    )
    _add_with_offset(metrics, 1, 0.2)
    assert len(metrics._get_predictions_in_24h_window()) == 1


def test_get_predictions_in_24h_window_skips_invalid_timestamp():
    metrics = PredictionMetrics()
    metrics.add_prediction(20.0, 20.1, timestamp="not-a-real-date")  # lines 515-517
    _add_with_offset(metrics, 1, 0.2)
    assert len(metrics._get_predictions_in_24h_window()) == 1


def test_get_24h_accuracy_breakdown_empty():
    bd = PredictionMetrics().get_24h_accuracy_breakdown()
    assert bd == {
        "perfect": {"count": 0, "percentage": 0.0},
        "tolerable": {"count": 0, "percentage": 0.0},
        "poor": {"count": 0, "percentage": 0.0},
    }


def test_get_24h_accuracy_breakdown_with_data():
    metrics = PredictionMetrics()
    _add_with_offset(metrics, 1, 0.0)    # perfect
    _add_with_offset(metrics, 2, 0.05)   # tolerable
    _add_with_offset(metrics, 3, 0.5)    # poor
    _add_with_offset(metrics, 30, 0.5)   # outside window, ignored
    bd = metrics.get_24h_accuracy_breakdown()
    assert bd["perfect"]["count"] == 1
    assert bd["tolerable"]["count"] == 1
    assert bd["poor"]["count"] == 1


def test_get_24h_good_control_percentage():
    metrics = PredictionMetrics()
    _add_with_offset(metrics, 1, 0.0)
    _add_with_offset(metrics, 2, 0.05)
    _add_with_offset(metrics, 3, 0.5)
    assert metrics.get_24h_good_control_percentage() == pytest.approx((2 / 3) * 100)


def test_get_24h_prediction_count():
    metrics = PredictionMetrics()
    _add_with_offset(metrics, 1, 0.1)
    _add_with_offset(metrics, 5, 0.1)
    _add_with_offset(metrics, 30, 0.1)  # outside
    assert metrics.get_24h_prediction_count() == 2  # line 585


# --- module-level convenience functions & __main__ demo block -------------


def test_convenience_functions_use_global_tracker():
    prediction_tracker.predictions.clear()
    try:
        track_prediction(20.0, 20.1, context={"step": 0})
        track_prediction(20.0, 20.2, context={"step": 1})
        metrics = get_current_metrics()
        assert metrics["all"]["count"] == 2
        assert metrics["all"]["mae"] > 0
        summary = get_metrics_summary()
        assert "Prediction Accuracy Summary" in summary
    finally:
        prediction_tracker.predictions.clear()


def test_main_demo_block_runs_without_error(capsys):
    # Executes the `if __name__ == "__main__"` demo block (lines 618-665).
    import warnings
    with warnings.catch_warnings():
        # runpy emits a benign RuntimeWarning because the module is already
        # imported; the demo block still executes correctly.
        warnings.simplefilter("ignore", RuntimeWarning)
        runpy.run_module("src.prediction_metrics", run_name="__main__")
    out = capsys.readouterr().out
    assert "Prediction metrics system ready" in out
