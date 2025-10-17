"""This module handles loading and saving the application state."""
import logging
import pickle
from typing import Any, Dict

from . import config


def load_state() -> Dict[str, Any]:
    """
    Load state from a file or create a new one, matching the original script's logic.
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
        return {
            "last_run_features": None,
            "last_indoor_temp": None,
            "prediction_history": [],
        }


def save_state(
    last_run_features: Any, last_indoor_temp: float, prediction_history: list
) -> None:
    """Save the application state to a file."""
    state = {
        "last_run_features": last_run_features,
        "last_indoor_temp": last_indoor_temp,
        "prediction_history": prediction_history,
    }
    try:
        with open(config.STATE_FILE, "wb") as f:
            pickle.dump(state, f)
            logging.debug("Successfully saved state to %s", config.STATE_FILE)
    except Exception as e:
        logging.error("Failed to save state to %s: %s", config.STATE_FILE, e)
