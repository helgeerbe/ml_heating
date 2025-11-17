"""
Helpers for grace-period decision logic.

This module provides a small, testable function used to decide how the
controller should wait after a blocking event ends. It intentionally does
not depend on Home Assistant clients so it can be unit tested easily.
"""
from typing import Optional


def choose_wait_direction(
    actual_outlet_temp_start: Optional[float], last_final_temp: float
) -> Optional[str]:
    """
    Determine wait direction for the grace period.

    Args:
        actual_outlet_temp_start: measured outlet temperature at the moment
            the blocking event ended (or None if unavailable).
        last_final_temp: the last target outlet temperature restored.

    Returns:
        - "cooling" if outlet is hotter than target (wait for actual <= target)
        - "warming" if outlet is colder than target (wait for actual >= target)
        - "none" if equal (no wait)
        - None if actual_outlet_temp_start is None (cannot decide)
    """
    if actual_outlet_temp_start is None:
        return None
    delta = actual_outlet_temp_start - last_final_temp
    if delta == 0:
        return "none"
    return "cooling" if delta > 0 else "warming"
