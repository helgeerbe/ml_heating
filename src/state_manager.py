"""
This module handles the persistence of the application's state.

The state includes data from the previous run, such as the features used for
prediction and the resulting indoor temperature. This information is crucial for
the online learning process, allowing the model to learn from its past
performance in the next cycle.
"""
import logging
import pickle
import os
import tempfile
from typing import Any, Dict

from . import config


def load_state() -> Dict[str, Any]:
    """
    Loads the application state from a pickle file.

    If the file doesn't exist or is corrupted, it returns a fresh,
    empty state dictionary. This ensures the application can always start,
    even without a previous state.
    """
    try:
        with open(config.STATE_FILE, "rb") as f:
            state = pickle.load(f)
            logging.info("Successfully loaded state from %s", config.STATE_FILE)
            return state
    except (
        FileNotFoundError,
        pickle.UnpicklingError,
        EOFError,
        TypeError,
    ) as e:
        logging.warning(
            "Could not load state from %s, starting fresh. Reason: %s",
            config.STATE_FILE,
            e,
        )
        # Return a default state structure if loading fails.
        return {
            "last_run_features": None,
            "last_indoor_temp": None,
            "prediction_history": [],
            "last_avg_other_rooms_temp": None,
            "last_fireplace_on": False,
            "last_final_temp": None,
            "last_is_blocking": False,
        }


def save_state(**kwargs: Any) -> None:
    """
    Saves the application's current state to a pickle file.

    This function merges provided keys into the existing persisted state
    instead of overwriting the entire file. This prevents accidental loss of
    unrelated fields when doing partial updates (e.g., saving only
    `last_is_blocking`).
    """
    # Load the existing state (returns defaults on failure).
    try:
        existing = load_state() or {
            "last_run_features": None,
            "last_indoor_temp": None,
            "prediction_history": [],
            "last_avg_other_rooms_temp": None,
            "last_fireplace_on": False,
            "last_final_temp": None,
            "last_is_blocking": False,
        }
    except Exception:
        existing = {
            "last_run_features": None,
            "last_indoor_temp": None,
            "prediction_history": [],
            "last_avg_other_rooms_temp": None,
            "last_fireplace_on": False,
            "last_final_temp": None,
            "last_is_blocking": False,
        }

    # Update only the provided keys.
    existing.update(kwargs)
    # Log which keys were updated for easier debugging of partial saves.
    try:
        logging.debug("Merged state saved; updated keys: %s", list(kwargs.keys()))
    except Exception:
        # Ensure logging failures don't prevent state persistence.
        pass

    # Atomically write the updated state to avoid corruption.
    try:
        dirpath = os.path.dirname(config.STATE_FILE) or "."
        fd, tmp_path = tempfile.mkstemp(dir=dirpath)
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(existing, f)
            os.replace(tmp_path, config.STATE_FILE)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        logging.debug("Successfully saved merged state to %s", config.STATE_FILE)
    except Exception as e:
        logging.error("Failed to save state to %s: %s", config.STATE_FILE, e)
