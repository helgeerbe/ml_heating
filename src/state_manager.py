"""
This module handles the persistence of the application's operational state.

UPDATED: Now uses unified JSON state management instead of pickle files.
The operational state includes data from the previous run, such as the features 
used for prediction and the resulting indoor temperature.
"""
import logging
from typing import Any, Dict

# Support both package-relative and direct import for notebooks
try:
    from .unified_thermal_state import get_thermal_state_manager
except ImportError:
    from unified_thermal_state import get_thermal_state_manager


def load_state() -> Dict[str, Any]:
    """
    Loads the application operational state from unified JSON.

    If the state doesn't exist, it returns a fresh, empty state dictionary.
    This ensures the application can always start, even without previous state.
    """
    try:
        state_manager = get_thermal_state_manager()
        operational_state = state_manager.get_operational_state()
        
        logging.info("Successfully loaded operational state from unified JSON")
        return operational_state
        
    except Exception as e:
        logging.warning(
            "Could not load operational state from unified JSON, starting fresh. Reason: %s",
            e,
        )
        # Return a default state structure if loading fails.
        return {
            "last_run_features": None,
            "last_indoor_temp": None,
            "last_avg_other_rooms_temp": None,
            "last_fireplace_on": False,
            "last_final_temp": None,
            "last_is_blocking": False,
            "last_blocking_end_time": None,
        }


def save_state(**kwargs: Any) -> None:
    """
    Saves the application's current operational state to unified JSON.

    This function merges provided keys into the existing persisted state
    instead of overwriting the entire state. This prevents accidental loss of
    unrelated fields when doing partial updates.
    """
    try:
        state_manager = get_thermal_state_manager()
        
        # Update operational state with provided keys (use **kwargs for unpacking)
        state_manager.update_operational_state(**kwargs)
        
        # Save the state after updating
        state_manager.save_state()
        
        # Log which keys were updated for easier debugging
        logging.debug("Operational state saved; updated keys: %s", list(kwargs.keys()))
        
    except Exception as e:
        logging.error("Failed to save operational state to unified JSON: %s", e)
