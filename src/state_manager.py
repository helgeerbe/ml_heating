"""
This module handles the persistence of the application's operational state.

UPDATED: Now uses unified JSON state management and Type-Safe SystemState
dataclass. The operational state includes data from the previous run, such as
the features used for prediction and the resulting indoor temperature.
"""
import logging
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

# Support both package-relative and direct import for notebooks
try:
    from .unified_thermal_state import get_thermal_state_manager
except ImportError:
    from unified_thermal_state import (  # type: ignore
        get_thermal_state_manager,
    )


@dataclass
class SystemState:
    """
    Type-safe representation of the system's operational state.
    """

    last_run_features: Optional[Dict[str, Any]] = None
    last_indoor_temp: Optional[float] = None
    last_avg_other_rooms_temp: Optional[float] = None
    last_fireplace_on: bool = False
    last_final_temp: Optional[float] = None
    last_is_blocking: bool = False
    last_blocking_reasons: List[str] = field(default_factory=list)
    last_blocking_end_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return asdict(self)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Mimic dict.get() for backward compatibility during refactoring.
        """
        return getattr(self, key, default)

    def update(self, other: Dict[str, Any]) -> None:
        """
        Update state from a dictionary, similar to dict.update().
        """
        for key, value in other.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style assignment."""
        if hasattr(self, key):
            setattr(self, key, value)


def load_state() -> SystemState:
    """
    Loads the application operational state from unified JSON into a
    SystemState object.

    If the state doesn't exist, it returns a fresh, empty SystemState.
    """
    try:
        state_manager = get_thermal_state_manager()
        operational_state_dict = state_manager.get_operational_state()

        # Filter dict to only include known fields to avoid TypeError
        known_fields = SystemState.__dataclass_fields__.keys()
        filtered_state = {
            k: v
            for k, v in operational_state_dict.items()
            if k in known_fields
        }

        logging.info("Successfully loaded operational state from unified JSON")
        return SystemState(**filtered_state)

    except Exception as e:
        logging.warning(
            "Could not load operational state from unified JSON, "
            "starting fresh. Reason: %s",
            e,
        )
        return SystemState()


def save_state(**kwargs: Any) -> None:
    """
    Saves the application's current operational state to unified JSON.

    This function merges provided keys into the existing persisted state.
    """
    try:
        state_manager = get_thermal_state_manager()

        # Update operational state with provided keys
        state_manager.update_operational_state(**kwargs)

        # Save the state after updating
        state_manager.save_state()

        # Log which keys were updated for easier debugging
        logging.debug(
            "Operational state saved; updated keys: %s", list(kwargs.keys())
        )

    except Exception as e:
        logging.error(
            "Failed to save operational state to unified JSON: %s", e
        )
