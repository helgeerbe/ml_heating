"""
This module handles the persistence of the application's state.

The state includes data from the previous run, such as the features used for
prediction and the resulting indoor temperature. This information is crucial for
the online learning process, allowing the model to learn from its past
performance in the next cycle.
"""
import logging
import pickle
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
        }


def save_state(**kwargs: Any) -> None:
    """
    Saves the application's current state to a pickle file.

    This includes the feature set from the current run and the measured indoor
    temperature, which will be used in the next cycle for online learning.
    """
    state = {
        "last_run_features": kwargs.get("last_run_features"),
        "last_indoor_temp": kwargs.get("last_indoor_temp"),
        "prediction_history": kwargs.get("prediction_history", []),
        "last_avg_other_rooms_temp": kwargs.get("last_avg_other_rooms_temp"),
        "last_fireplace_on": kwargs.get("last_fireplace_on", False),
    }
    try:
        with open(config.STATE_FILE, "wb") as f:
            pickle.dump(state, f)
            logging.debug("Successfully saved state to %s", config.STATE_FILE)
    except Exception as e:
        logging.error("Failed to save state to %s: %s", config.STATE_FILE, e)
