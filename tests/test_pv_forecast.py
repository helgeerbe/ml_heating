import pytest
from datetime import datetime, timezone

from src import physics_features, config


class DummyHA:
    def __init__(self, state_map, hourly_forecast=None):
        self.state_map = state_map
        self.hourly_forecast = hourly_forecast or [0.0, 0.0, 0.0, 0.0]

    def get_all_states(self):
        """Return all states for physics_features to use."""
        return self.state_map

    def get_state(self, entity_id, states_cache=None, is_binary=False):
        """
        Return a value compatible with HAClient.get_state.

        Priority:
        1. If a `states_cache` dict is provided and contains the entity_id,
           use that (tests populate states with plain values or dicts).
        2. Fall back to the internal state_map.
        """
        # Prefer the provided cached states (physics_features passes all_states)
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
    """Set up datetime mocking."""
    fake_dt = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
    
    class FakeDatetime:
        @classmethod
        def now(cls, tz=None):
            return fake_dt
        
        # Preserve other datetime functionality
        @staticmethod
        def strptime(*args, **kwargs):
            return datetime.strptime(*args, **kwargs)
            
        @staticmethod
        def fromtimestamp(*args, **kwargs):
            return datetime.fromtimestamp(*args, **kwargs)
            
        # Forward all other attributes to real datetime
        def __getattr__(self, name):
            return getattr(datetime, name)

    # Patch datetime in the physics_features module directly
    from src import physics_features
    monkeypatch.setattr(physics_features, "datetime", FakeDatetime)


def test_pv_forecast_daytime_means(monkeypatch):
    """
    Normal daytime: watts has multiple 15-min entries per hour -> mean used.
    Anchors computed from current time + 1h, 2h, 3h, 4h.
    """
    # Freeze "now" to 2025-11-17 07:05 UTC -> first anchor 08:05 UTC
    _set_fake_now(monkeypatch, 2025, 11, 17, 7, 5)

    # Build forecast data: samples for hours starting from 08:05
    forecast_data = [
        # Hour 1: 08:05-09:05
        {"period_start": "2025-11-17T08:15:00Z", "pv_estimate": 100.0},
        {"period_start": "2025-11-17T08:30:00Z", "pv_estimate": 200.0},
        # Hour 2: 09:05-10:05  
        {"period_start": "2025-11-17T09:15:00Z", "pv_estimate": 300.0},
        {"period_start": "2025-11-17T09:45:00Z", "pv_estimate": 500.0},
        # Hour 3: 10:05-11:05
        {"period_start": "2025-11-17T10:30:00Z", "pv_estimate": 600.0},
    ]

    pv_entity = config.PV_FORECAST_ENTITY_ID
    all_states = {
        pv_entity: {"attributes": {"forecast": forecast_data}},
        config.INDOOR_TEMP_ENTITY_ID: {"state": "20.0"},
        config.OUTDOOR_TEMP_ENTITY_ID: {"state": "5.0"},
        config.ACTUAL_OUTLET_TEMP_ENTITY_ID: {"state": "35.0"},
        config.TARGET_INDOOR_TEMP_ENTITY_ID: {"state": "21.0"},
    }

    # Provide minimal histories
    outlet_history = [30.0] * config.HISTORY_STEPS
    indoor_history = [20.0] * config.HISTORY_STEPS

    ha = DummyHA(all_states)
    influx = DummyInflux(outlet_history, indoor_history)

    df, outlet_hist = physics_features.build_physics_features(
        ha, influx
    )

    assert df is not None
    # Expect hourly means: hour 1 -> mean(100,200)=150; hour 2 -> mean(300,500)=400; hour 3 -> 600; hour 4 -> 0
    assert df["pv_forecast_1h"].iloc[0] == pytest.approx(150.0)
    assert df["pv_forecast_2h"].iloc[0] == pytest.approx(400.0)
    assert df["pv_forecast_3h"].iloc[0] == pytest.approx(600.0)
    assert df["pv_forecast_4h"].iloc[0] == pytest.approx(0.0)


def test_pv_forecast_midnight_day_change(monkeypatch):
    """
    Day-change edge: when now is just before midnight, anchors are
    current time + 1h, 2h, 3h, 4h (23:50 + 1h = 00:50 next day).
    """
    # Freeze "now" to 2025-11-17 23:50 UTC -> first anchor 2025-11-18 00:50 UTC
    _set_fake_now(monkeypatch, 2025, 11, 17, 23, 50)

    # Samples for hours starting from 00:50 next day
    forecast_data = [
        # Hour 1: 00:50-01:50 next day
        {"period_start": "2025-11-18T01:15:00Z", "pv_estimate": 10.0},
        # Hour 2: 01:50-02:50 next day  
        {"period_start": "2025-11-18T02:15:00Z", "pv_estimate": 20.0},
        # Hour 3: 02:50-03:50 next day
        {"period_start": "2025-11-18T03:15:00Z", "pv_estimate": 30.0},
    ]

    pv_entity = config.PV_FORECAST_ENTITY_ID
    all_states = {
        pv_entity: {"attributes": {"forecast": forecast_data}},
        config.INDOOR_TEMP_ENTITY_ID: {"state": "20.0"},
        config.OUTDOOR_TEMP_ENTITY_ID: {"state": "6.0"},
        config.ACTUAL_OUTLET_TEMP_ENTITY_ID: {"state": "35.0"},
        config.TARGET_INDOOR_TEMP_ENTITY_ID: {"state": "21.0"},
    }

    outlet_history = [30.0] * config.HISTORY_STEPS
    indoor_history = [20.0] * config.HISTORY_STEPS

    ha = DummyHA(all_states)
    influx = DummyInflux(outlet_history, indoor_history)

    df, _ = physics_features.build_physics_features(
        ha, influx
    )

    assert df is not None
    # Algorithm uses now + 1h, 2h, 3h, 4h for anchors
    assert df["pv_forecast_1h"].iloc[0] == pytest.approx(10.0)
    assert df["pv_forecast_2h"].iloc[0] == pytest.approx(20.0)
    assert df["pv_forecast_3h"].iloc[0] == pytest.approx(30.0)
    assert df["pv_forecast_4h"].iloc[0] == pytest.approx(0.0)  # No data for 4th hour


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
        config.TARGET_INDOOR_TEMP_ENTITY_ID: {"state": "21.0"},
    }

    outlet_history = [30.0] * config.HISTORY_STEPS
    indoor_history = [20.0] * config.HISTORY_STEPS

    ha = DummyHA(all_states)
    influx = DummyInflux(outlet_history, indoor_history)

    df, _ = physics_features.build_physics_features(
        ha, influx
    )

    assert df is not None
    assert all(df[f"pv_forecast_{i+1}h"].iloc[0] == 0.0 for i in range(4))
