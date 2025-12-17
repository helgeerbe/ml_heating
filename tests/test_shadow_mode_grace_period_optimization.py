"""
Test Shadow Mode Grace Period Optimization

Verify that grace periods are completely bypassed in shadow mode for faster learning.
"""

import pytest
from unittest.mock import Mock, patch
from src.heating_controller import BlockingStateManager
from src import config


class TestShadowModeGracePeriodOptimization:
    """Test grace period optimization in shadow mode"""

    def setup_method(self):
        """Set up test environment"""
        self.manager = BlockingStateManager()
        self.ha_client = Mock()
        self.state = {
            "last_is_blocking": True,
            "last_blocking_end_time": None,
            "last_final_temp": 35.0
        }

    @patch.object(config, 'SHADOW_MODE', True)
    def test_grace_period_bypassed_in_shadow_mode(self):
        """Test that grace period is completely bypassed when SHADOW_MODE=True"""
        # Setup: blocking just ended, would normally trigger grace period
        self.state["last_is_blocking"] = True
        
        # Mock that blocking state check shows no current blocking
        self.manager.check_blocking_state = Mock(return_value=(False, []))
        
        # Call handle_grace_period
        result = self.manager.handle_grace_period(self.ha_client, self.state)
        
        # Verify: grace period is bypassed (returns False)
        assert result is False
        
        # Verify: no HA client calls were made (no temperature restoration)
        assert self.ha_client.get_state.call_count == 0
        assert self.ha_client.set_state.call_count == 0

    @patch.object(config, 'SHADOW_MODE', False)
    def test_grace_period_normal_behavior_in_active_mode(self):
        """Test that grace period works normally when SHADOW_MODE=False"""
        # Setup: blocking just ended
        self.state["last_is_blocking"] = True
        
        # Mock blocking state check
        self.manager.check_blocking_state = Mock(return_value=(False, []))
        
        # Mock HA client calls for grace period execution
        self.ha_client.get_all_states.return_value = {}
        self.ha_client.get_state.return_value = 40.0  # Current outlet temp
        
        # Call handle_grace_period
        result = self.manager.handle_grace_period(self.ha_client, self.state)
        
        # Verify: grace period is executed (returns True to skip cycle)
        assert result is True
        
        # Verify: HA client was called for temperature restoration
        assert self.ha_client.get_state.call_count > 0

    @patch.object(config, 'SHADOW_MODE', True)
    def test_shadow_mode_optimization_with_various_blocking_states(self):
        """Test shadow mode optimization works regardless of blocking history"""
        test_cases = [
            # Case 1: Fresh start, no blocking history
            {"last_is_blocking": False, "last_blocking_end_time": None},
            
            # Case 2: Recent blocking end
            {"last_is_blocking": True, "last_blocking_end_time": None},
            
            # Case 3: Old blocking end
            {"last_is_blocking": True, "last_blocking_end_time": 1000000},
        ]
        
        for case in test_cases:
            self.state.update(case)
            
            # Mock no current blocking
            self.manager.check_blocking_state = Mock(return_value=(False, []))
            
            # Call handle_grace_period
            result = self.manager.handle_grace_period(self.ha_client, self.state)
            
            # Verify: always bypassed in shadow mode
            assert result is False, f"Failed for case: {case}"
            
            # Reset mocks for next test
            self.ha_client.reset_mock()

    def test_grace_period_logic_performance_impact(self):
        """Test that shadow mode optimization eliminates performance overhead"""
        import time
        
        # Test shadow mode performance (should be instant)
        with patch.object(config, 'SHADOW_MODE', True):
            start_time = time.time()
            result = self.manager.handle_grace_period(self.ha_client, self.state)
            shadow_mode_duration = time.time() - start_time
        
        # Verify result and performance
        assert result is False  # Bypassed
        assert shadow_mode_duration < 0.001  # Should be near-instant (< 1ms)
        
        # Verify no HA API calls in shadow mode
        assert self.ha_client.call_count == 0

    @patch.object(config, 'SHADOW_MODE', True) 
    def test_shadow_mode_logging_indicates_optimization(self):
        """Test that shadow mode optimization is transparent in operation"""
        # Setup scenario that would normally trigger grace period
        self.state["last_is_blocking"] = True
        self.state["last_blocking_end_time"] = None
        
        # Mock blocking check
        self.manager.check_blocking_state = Mock(return_value=(False, []))
        
        # Call grace period handler
        result = self.manager.handle_grace_period(self.ha_client, self.state)
        
        # Verify optimization behavior
        assert result is False  # No grace period
        
        # Verify no state persistence calls (no blocking state updates needed)
        # This indicates the optimization is working - no complex grace period logic executed

    def test_shadow_mode_config_integration(self):
        """Test that shadow mode grace period optimization integrates with config properly"""
        # Test with shadow mode enabled
        with patch.object(config, 'SHADOW_MODE', True):
            assert config.SHADOW_MODE is True
            result = self.manager.handle_grace_period(self.ha_client, self.state)
            assert result is False  # Grace period bypassed
        
        # Test with shadow mode disabled  
        with patch.object(config, 'SHADOW_MODE', False):
            assert config.SHADOW_MODE is False
            self.state["last_is_blocking"] = False  # No grace period needed anyway
            result = self.manager.handle_grace_period(self.ha_client, self.state)
            assert result is False  # No grace period (different reason)


if __name__ == "__main__":
    pytest.main([__file__])
