import time

from src import config
import src.main as main_module

def make_ha_with_sequence(sequence, blocking_entities):
    """
    sequence: iterable of booleans indicating whether blocking is present
      for each successive poll.
    blocking_entities: list of entity ids to include in the returned states.
    """
    class DummyHA:
        def __init__(self, seq):
            self.seq = list(seq)
            self.calls = 0

        def get_all_states(self):
            # Build a states cache mapping entity_id -> {"entity_id": ..., "state": "on"/"off"}
            present = self.seq[min(self.calls, len(self.seq) - 1)]
            self.calls += 1
            state_val = "on" if present else "off"
            return {e: {"entity_id": e, "state": state_val} for e in blocking_entities}

        def get_state(self, entity_id, states_cache=None, is_binary=False):
            if states_cache is None:
                return None
            data = states_cache.get(entity_id)
            if not data:
                return None
            state = data.get("state")
            if state in ("unknown", "unavailable"):
                return None
            if is_binary:
                return state == "on"
            try:
                return float(state)
            except Exception:
                return state

    return DummyHA(sequence)

def test_blocking_start_during_idle(monkeypatch):
    blocking_entities = [
        config.DHW_STATUS_ENTITY_ID,
        config.DEFROST_STATUS_ENTITY_ID,
        config.DISINFECTION_STATUS_ENTITY_ID,
        config.DHW_BOOST_HEATER_STATUS_ENTITY_ID,
    ]
    # Sequence: first poll -> blocking present
    ha = make_ha_with_sequence([True], blocking_entities)
    state = {"last_is_blocking": False, "last_final_temp": 42.0}

    saved = {}

    def fake_save_state(**kwargs):
        saved.update(kwargs)

    monkeypatch.setattr(main_module, "save_state", fake_save_state)

    main_module.poll_for_blocking(ha, state, blocking_entities)

    assert saved.get("last_is_blocking") is True
    assert "last_blocking_reasons" in saved
    assert isinstance(saved["last_blocking_reasons"], list)
    # last_blocking_end_time should be explicitly set to None on start
    assert saved.get("last_blocking_end_time") is None

def test_blocking_end_during_idle(monkeypatch):
    blocking_entities = [
        config.DHW_STATUS_ENTITY_ID,
        config.DEFROST_STATUS_ENTITY_ID,
        config.DISINFECTION_STATUS_ENTITY_ID,
        config.DHW_BOOST_HEATER_STATUS_ENTITY_ID,
    ]
    # Sequence: first poll -> no blocking (simulate end)
    ha = make_ha_with_sequence([False], blocking_entities)
    state = {"last_is_blocking": True, "last_final_temp": 42.0, "last_blocking_end_time": None}

    saved = {}

    def fake_save_state(**kwargs):
        saved.update(kwargs)

    monkeypatch.setattr(main_module, "save_state", fake_save_state)

    main_module.poll_for_blocking(ha, state, blocking_entities)

    # Expect that we persisted a last_blocking_end_time (timestamp float)
    assert "last_blocking_end_time" in saved
    assert isinstance(saved["last_blocking_end_time"], float)
