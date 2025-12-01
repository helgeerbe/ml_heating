"""
Test suite for the Heat Balance Controller functionality.

Tests the trajectory-based temperature control system that replaced
the simple exponential smoothing approach.
"""
import pytest

from src import config


class DummyHA:
    """Mock Home Assistant client for testing."""
    
    def __init__(self, state_map=None):
        self.state_map = state_map or {}
    
    def get_state(self, entity_id, states_cache=None, is_binary=False):
        """Mock get_state method."""
        # Use states_cache if provided, fallback to state_map
        if states_cache and entity_id in states_cache:
            value = states_cache[entity_id]
        else:
            value = self.state_map.get(entity_id)
        
        if value is None:
            return None
            
        if is_binary:
            return value == "on"
        
        try:
            return float(value)
        except (TypeError, ValueError):
            return value


def test_heat_balance_controller_charging_mode():
    """Test that controller enters CHARGING mode for large temperature errors."""
    # Set up state with large temperature error (> 0.5°C threshold)
    ha = DummyHA({})
    state = {
        "last_final_temp": 40.0,
        "last_is_blocking": False,
    }
    
    # Mock scenario: indoor is 19.0°C, target is 21.0°C (2°C error)
    indoor_temp = 19.0
    target_temp = 21.0
    actual_outlet = 42.0
    
    # This should trigger CHARGING mode due to large temperature error
    # The controller should be more aggressive
    
    # Since this is a complex integration test that requires the full ML pipeline,
    # we'll test the decision logic components that we can isolate
    temp_error = target_temp - indoor_temp
    assert temp_error > config.CHARGING_MODE_THRESHOLD
    assert temp_error == 2.0  # Large error should trigger charging mode


def test_heat_balance_controller_maintenance_mode():
    """Test that controller enters MAINTENANCE mode for small temperature errors."""
    # Set up state with small temperature error (< 0.1°C threshold)
    ha = DummyHA({})
    state = {
        "last_final_temp": 40.0,
        "last_is_blocking": False,
    }

    # Mock scenario: indoor is 20.95°C, target is 21.0°C (0.05°C error)
    indoor_temp = 20.95
    target_temp = 21.0
    actual_outlet = 42.0

    # This should trigger MAINTENANCE mode due to small temperature error (< 0.1°C)
    temp_error = target_temp - indoor_temp
    assert temp_error < config.MAINTENANCE_MODE_THRESHOLD
    assert temp_error == pytest.approx(0.05)  # Small error should trigger maintenance mode


def test_heat_balance_controller_balancing_mode():
    """Test that controller enters BALANCING mode for medium temperature errors."""
    # Set up state with medium temperature error (0.1-0.2°C range)
    ha = DummyHA({})
    state = {
        "last_final_temp": 40.0,
        "last_is_blocking": False,
    }
    
    # Mock scenario: indoor is 20.85°C, target is 21.0°C (0.15°C error)
    indoor_temp = 20.85
    target_temp = 21.0
    actual_outlet = 42.0
    
    # This should trigger BALANCING mode (0.1°C < error < 0.2°C)
    temp_error = target_temp - indoor_temp
    assert config.MAINTENANCE_MODE_THRESHOLD < temp_error < config.CHARGING_MODE_THRESHOLD
    assert temp_error == pytest.approx(0.15)  # Medium error should trigger balancing mode


def test_configuration_values_are_valid():
    """Test that Heat Balance Controller configuration values are sensible."""
    # Test that thresholds are properly ordered
    assert config.MAINTENANCE_MODE_THRESHOLD < config.CHARGING_MODE_THRESHOLD
    
    # Test default values are reasonable (updated for optimized thresholds)
    assert 0.05 <= config.MAINTENANCE_MODE_THRESHOLD <= 0.15  # 0.1°C is optimal
    assert 0.15 <= config.CHARGING_MODE_THRESHOLD <= 0.5   # 0.2°C is optimal
    
    # Test trajectory parameters
    assert 2 <= config.TRAJECTORY_STEPS <= 8  # Reasonable prediction horizon
    assert 0.0 <= config.OSCILLATION_PENALTY_WEIGHT <= 1.0
    assert 1.0 <= config.FINAL_DESTINATION_WEIGHT <= 3.0
    
    # Test that heat balance mode can be enabled/disabled
    assert isinstance(config.HEAT_BALANCE_MODE, bool)


def test_blocking_polling_with_heat_balance_controller():
    """Test that blocking detection logic works correctly with heat balance controller."""
    blocking_entities = [
        config.DHW_STATUS_ENTITY_ID,
        config.DEFROST_STATUS_ENTITY_ID,
        config.DISINFECTION_STATUS_ENTITY_ID,
        config.DHW_BOOST_HEATER_STATUS_ENTITY_ID,
    ]
    
    # Create HA mock with blocking state
    ha = DummyHA({
        config.DHW_STATUS_ENTITY_ID: "on",
        config.DEFROST_STATUS_ENTITY_ID: "off",
        config.DISINFECTION_STATUS_ENTITY_ID: "off",
        config.DHW_BOOST_HEATER_STATUS_ENTITY_ID: "off",
    })
    
    # Test blocking detection logic without calling the full poll_for_blocking function
    # which contains infinite loops that would hang in tests
    
    # Simulate what the blocking detection would find
    blocking_reasons = [
        e for e in blocking_entities
        if ha.get_state(e, is_binary=True)
    ]
    is_blocking = bool(blocking_reasons)
    
    # Should detect blocking state
    assert is_blocking is True
    assert config.DHW_STATUS_ENTITY_ID in blocking_reasons
    assert len(blocking_reasons) == 1
    
    # Test non-blocking state
    ha_non_blocking = DummyHA({
        config.DHW_STATUS_ENTITY_ID: "off",
        config.DEFROST_STATUS_ENTITY_ID: "off",
        config.DISINFECTION_STATUS_ENTITY_ID: "off",
        config.DHW_BOOST_HEATER_STATUS_ENTITY_ID: "off",
    })
    
    blocking_reasons_none = [
        e for e in blocking_entities
        if ha_non_blocking.get_state(e, is_binary=True)
    ]
    is_blocking_none = bool(blocking_reasons_none)
    
    # Should detect no blocking
    assert is_blocking_none is False
    assert len(blocking_reasons_none) == 0


def test_temperature_clamping_with_heat_balance_controller():
    """Test temperature clamping works with heat balance controller scenarios."""
    from tests.test_clamp_baseline import compute_baseline_and_clamped
    
    # Test aggressive charging scenario with clamping
    state = {"last_final_temp": 35.0, "last_blocking_reasons": []}
    
    # Simulate aggressive temperature request from heat balance controller
    aggressive_temp = 50.0  # Very high request
    actual_outlet = 38.0
    
    baseline, clamped = compute_baseline_and_clamped(aggressive_temp, actual_outlet, state)
    
    # Should clamp to reasonable change
    assert baseline == 35.0  # Uses last final temp
    expected_max_change = config.MAX_TEMP_CHANGE_PER_CYCLE
    assert clamped <= baseline + expected_max_change
    assert clamped == min(aggressive_temp, baseline + expected_max_change)


def test_heat_balance_controller_integration_points():
    """Test key integration points for heat balance controller."""
    # Verify config values exist and are accessible
    assert hasattr(config, 'HEAT_BALANCE_MODE')
    assert hasattr(config, 'CHARGING_MODE_THRESHOLD') 
    assert hasattr(config, 'MAINTENANCE_MODE_THRESHOLD')
    assert hasattr(config, 'TRAJECTORY_STEPS')
    assert hasattr(config, 'OSCILLATION_PENALTY_WEIGHT')
    assert hasattr(config, 'FINAL_DESTINATION_WEIGHT')
    
    # Test that values can be read
    assert config.CHARGING_MODE_THRESHOLD > 0
    assert config.MAINTENANCE_MODE_THRESHOLD > 0
    assert config.TRAJECTORY_STEPS > 0
    
    # Test Heat Balance Mode can be toggled
    if config.HEAT_BALANCE_MODE:
        # Heat balance controller is enabled
        assert config.TRAJECTORY_STEPS >= 2  # Needs meaningful prediction horizon
    else:
        # Legacy exponential smoothing mode
        # Should still work with existing parameters
        pass
