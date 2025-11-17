import os
import sys
import pickle

# Ensure project root is on sys.path so `src` package is importable during tests.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import state_manager, config

def test_save_state_merges(tmp_path):
    """
    Verify that calling save_state with a partial payload (e.g.
    last_is_blocking=True) preserves other fields like last_final_temp.
    """
    tmp_state = tmp_path / "state.pkl"

    initial = {
        "last_run_features": None,
        "last_indoor_temp": 20.0,
        "prediction_history": [],
        "last_avg_other_rooms_temp": None,
        "last_fireplace_on": False,
        "last_final_temp": 45.0,
        "last_is_blocking": False,
    }

    # Write initial state to the temporary file.
    with open(tmp_state, "wb") as f:
        pickle.dump(initial, f)

    # Temporarily point the config.STATE_FILE to our temp file.
    old_state_file = config.STATE_FILE
    config.STATE_FILE = str(tmp_state)
    try:
        # Perform a partial save that would previously wipe other fields.
        state_manager.save_state(last_is_blocking=True)

        # Reload the file and assert values were merged, not overwritten.
        with open(tmp_state, "rb") as f:
            new_state = pickle.load(f)

        assert new_state["last_final_temp"] == 45.0
        assert new_state["last_is_blocking"] is True
    finally:
        # Restore original config
        config.STATE_FILE = old_state_file
