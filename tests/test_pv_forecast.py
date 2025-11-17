import pytest
from datetime import datetime, timezone

from src import feature_builder, config


class DummyHA:
    def __init__(self, state_map, hourly_forecast=None):
        self.state_map = state_map
        self.hourly_forecast = hourly_forecast or [0.0, 0.0, 0.0, 0.0]

    def get_state(self, entity_id, states_cache=None, is_binary=False):
        """
        Return a value compatible with HAClient.get_state.

        Priority:
        1. If a `states_cache` dict is provided and contains the entity_id,
           use that (tests populate `all_states` with plain values or dicts).
        2. Fall back to the internal state_map.
        """
        # Prefer the provided cached states (feature_builder passes `all_states`)
        if states_cache and entity_id in states_cache:
            data = states_cache[entity_id]
            # If the cached value is a full HA state dict, mimic HAClient conversion.
            if isinstance(data, dict):
                state = data.get("state")
                if state in (None, "unknown", "unavailable"):
                    return None
                if is_binary:
                    return state == "on"
                try:
                    return float(state)
                except (TypeError, ValueError):
                    return state
            # Otherwise tests often place raw numeric values directly.
            return data
        # Fallback to internal state_map provided when creating DummyHA.
        return self.state_map.get(entity_id)

    def get_hourly_forecast(self):
        return self.hourly_forecast


class DummyInflux:
    def __init__(self, outlet_history, indoor_history):
        self._outlet = outlet_history
        self._indoor = indoor_history

    def fetch_outlet_history(self, steps):
        return self._outlet

    def fetch_indoor_history(self, steps):
        return self._indoor


def _set_fake_now(monkeypatch, year, month, day, hour, minute=0):
    class FakeDatetime:
        @classmethod
        def now(cls, tz=None):
            return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)

    monkeypatch.setattr(feature_builder, "datetime", FakeDatetime)


def test_pv_forecast_daytime_means(monkeypatch):
    """
    Normal daytime: watts has multiple 15-min entries per hour -> mean used.
    Anchors computed from next full hour.
    """
    # Freeze "now" to 2025-11-17 07:05 UTC -> first anchor 08:00 UTC
    _set_fake_now(monkeypatch, 2025, 11, 17, 7, 5)

    # Build watts_map: samples for hours 08,09,10; hour 11 empty -> expect 0.0
    watts_map = {
        "2025-11-17T08:15:00Z": 100.0,
        "2025-11-17T08:30:00Z": 200.0,
        "2025-11-17T09:15:00Z": 300.0,
        "2025-11-17T09:45:00Z": 500.0,
        "2025-11-17T10:30:00Z": 600.0,
    }

    pv_entity = config.PV_FORECAST_ENTITY_ID
    all_states = {
        pv_entity: {"attributes": {"watts": watts_map}},
        config.INDOOR_TEMP_ENTITY_ID: {"state": "20.0"},
        config.OUTDOOR_TEMP_ENTITY_ID: {"state": "5.0"},
        config.ACTUAL_OUTLET_TEMP_ENTITY_ID: {"state": "35.0"},
    }

    # Provide minimal histories
    outlet_history = [30.0] * config.HISTORY_STEPS
    indoor_history = [20.0] * config.HISTORY_STEPS

    ha = DummyHA({})
    influx = DummyInflux(outlet_history, indoor_history)

    df, outlet_hist = feature_builder.build_features(
        ha, influx, all_states, target_indoor_temp=21.0
    )

    assert df is not None
    # Expect hourly means: hour 08 -> mean(100,200)=150; 09 -> mean(300,500)=400; 10 -> 600; 11 -> 0
    assert df["pv_forecast_1h"].iloc[0] == pytest.approx(150.0)
    assert df["pv_forecast_2h"].iloc[0] == pytest.approx(400.0)
    assert df["pv_forecast_3h"].iloc[0] == pytest.approx(600.0)
    assert df["pv_forecast_4h"].iloc[0] == pytest.approx(0.0)


def test_pv_forecast_midnight_day_change(monkeypatch):
    """
    Day-change edge: when now is just before midnight, anchors start at midnight
    of next day. Ensure timestamps across day boundary are handled.
    """
    # Freeze "now" to 2025-11-17 23:50 UTC -> first anchor 2025-11-18 00:00 UTC
    _set_fake_now(monkeypatch, 2025, 11, 17, 23, 50)

    # Samples on 2025-11-18 00:15, 01:15, 02:15
    watts_map = {
        "2025-11-18T00:15:00Z": 10.0,
        "2025-11-18T01:15:00Z": 20.0,
        "2025-11-18T02:15:00Z": 30.0,
    }

    pv_entity = config.PV_FORECAST_ENTITY_ID
    all_states = {
        pv_entity: {"attributes": {"watts": watts_map}},
        config.INDOOR_TEMP_ENTITY_ID: {"state": "20.0"},
        config.OUTDOOR_TEMP_ENTITY_ID: {"state": "6.0"},
        config.ACTUAL_OUTLET_TEMP_ENTITY_ID: {"state": "35.0"},
    }

    outlet_history = [30.0] * config.HISTORY_STEPS
    indoor_history = [20.0] * config.HISTORY_STEPS

    ha = DummyHA({})
    influx = DummyInflux(outlet_history, indoor_history)

    df, _ = feature_builder.build_features(
        ha, influx, all_states, target_indoor_temp=21.0
    )

    assert df is not None
    assert df["pv_forecast_1h"].iloc[0] == pytest.approx(10.0)
    assert df["pv_forecast_2h"].iloc[0] == pytest.approx(20.0)
    assert df["pv_forecast_3h"].iloc[0] == pytest.approx(30.0)
    # no sample for 4th hour
    assert df["pv_forecast_4h"].iloc[0] == pytest.approx(0.0)


def test_pv_forecast_missing_attribute(monkeypatch):
    """
    Malformed/missing attributes -> should produce zeros for forecasts.
    """
    _set_fake_now(monkeypatch, 2025, 11, 17, 7, 5)

    pv_entity = config.PV_FORECAST_ENTITY_ID
    # Attribute present but empty dict
    all_states = {
        pv_entity: {"attributes": {"watts": {}}},
        config.INDOOR_TEMP_ENTITY_ID: {"state": "20.0"},
        config.OUTDOOR_TEMP_ENTITY_ID: {"state": "5.0"},
        config.ACTUAL_OUTLET_TEMP_ENTITY_ID: {"state": "35.0"},
    }

    outlet_history = [30.0] * config.HISTORY_STEPS
    indoor_history = [20.0] * config.HISTORY_STEPS

    ha = DummyHA({})
    influx = DummyInflux(outlet_history, indoor_history)

    df, _ = feature_builder.build_features(
        ha, influx, all_states, target_indoor_temp=21.0
    )

    assert df is not None
    assert all(df[f"pv_forecast_{i+1}h"].iloc[0] == 0.0 for i in range(4))
