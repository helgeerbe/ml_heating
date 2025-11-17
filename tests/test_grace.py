import pytest

from src.grace import choose_wait_direction

def test_choose_wait_direction_none_when_actual_missing():
    assert choose_wait_direction(None, 40.0) is None

def test_choose_wait_direction_none_when_equal():
    assert choose_wait_direction(45.0, 45.0) == "none"

def test_choose_wait_direction_cooling_when_actual_hotter():
    # actual is hotter than target -> wait for cooling
    assert choose_wait_direction(58.6, 43.0) == "cooling"

def test_choose_wait_direction_warming_when_actual_colder():
    # actual is colder than target -> wait for warming
    assert choose_wait_direction(30.0, 35.0) == "warming"
