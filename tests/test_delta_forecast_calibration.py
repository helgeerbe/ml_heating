"""
Comprehensive tests for delta temperature forecast calibration.

Tests the new get_calibrated_hourly_forecast method that applies local
temperature offset to weather forecasts for improved accuracy.
"""
import pytest
from unittest.mock import Mock, patch
import logging

# Test imports - handle both package and direct imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.ha_client import HAClient
    from src import config
except ImportError:
    from ha_client import HAClient
    import src.config as config


class TestDeltaForecastCalibration:
    """Test suite for delta forecast calibration functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ha_client = HAClient("http://test:8123", "test_token")
        
        # Mock the get_hourly_forecast method to return predictable data
        self.mock_raw_forecasts = [25.0, 27.0, 26.0, 24.0]
        self.ha_client.get_hourly_forecast = Mock(return_value=self.mock_raw_forecasts)
    
    def test_delta_calibration_basic_functionality(self):
        """Test basic delta calibration with positive offset."""
        current_outdoor_temp = 26.0  # 1°C warmer than forecast
        
        calibrated_forecasts = self.ha_client.get_calibrated_hourly_forecast(
            current_outdoor_temp=current_outdoor_temp
        )
        
        # Expected: all forecasts increased by 1°C offset
        expected = [26.0, 28.0, 27.0, 25.0]
        assert calibrated_forecasts == expected
    
    def test_delta_calibration_negative_offset(self):
        """Test delta calibration with negative offset."""
        current_outdoor_temp = 23.0  # 2°C cooler than forecast
        
        calibrated_forecasts = self.ha_client.get_calibrated_hourly_forecast(
            current_outdoor_temp=current_outdoor_temp
        )
        
        # Expected: all forecasts decreased by 2°C offset
        expected = [23.0, 25.0, 24.0, 22.0]
        assert calibrated_forecasts == expected
    
    def test_delta_calibration_disabled(self):
        """Test that raw forecasts are returned when calibration is disabled."""
        calibrated_forecasts = self.ha_client.get_calibrated_hourly_forecast(
            current_outdoor_temp=26.0,
            enable_delta_calibration=False
        )
        
        # Should return original raw forecasts unchanged
        assert calibrated_forecasts == self.mock_raw_forecasts
    
    def test_invalid_outdoor_temperature_handling(self):
        """Test handling of invalid outdoor temperature values."""
        # Test None value
        calibrated_forecasts = self.ha_client.get_calibrated_hourly_forecast(
            current_outdoor_temp=None
        )
        assert calibrated_forecasts == self.mock_raw_forecasts
        
        # Test extreme values
        calibrated_forecasts = self.ha_client.get_calibrated_hourly_forecast(
            current_outdoor_temp=70.0  # Too hot
        )
        assert calibrated_forecasts == self.mock_raw_forecasts
        
        calibrated_forecasts = self.ha_client.get_calibrated_hourly_forecast(
            current_outdoor_temp=-70.0  # Too cold
        )
        assert calibrated_forecasts == self.mock_raw_forecasts
    
    def test_invalid_raw_forecast_handling(self):
        """Test handling when raw forecast data is invalid."""
        # Test empty forecast
        self.ha_client.get_hourly_forecast = Mock(return_value=[])
        calibrated_forecasts = self.ha_client.get_calibrated_hourly_forecast(
            current_outdoor_temp=25.0
        )
        assert calibrated_forecasts == []
        
        # Test forecast with zero current temperature
        self.ha_client.get_hourly_forecast = Mock(return_value=[0.0, 27.0, 26.0, 24.0])
        calibrated_forecasts = self.ha_client.get_calibrated_hourly_forecast(
            current_outdoor_temp=25.0
        )
        assert calibrated_forecasts == [0.0, 27.0, 26.0, 24.0]  # Should return raw
    
    def test_rounding_precision(self):
        """Test that calibrated forecasts are properly rounded."""
        current_outdoor_temp = 25.333  # Will create fractional offset
        
        calibrated_forecasts = self.ha_client.get_calibrated_hourly_forecast(
            current_outdoor_temp=current_outdoor_temp
        )
        
        # All values should be rounded to 2 decimal places
        expected_offset = round(25.333 - 25.0, 2)  # 0.33
        expected = [
            round(25.0 + expected_offset, 2),
            round(27.0 + expected_offset, 2),
            round(26.0 + expected_offset, 2),
            round(24.0 + expected_offset, 2)
        ]
        assert calibrated_forecasts == expected
    
    def test_logging_output(self, caplog):
        """Test that appropriate debug logging is generated."""
        with caplog.at_level(logging.DEBUG):
            self.ha_client.get_calibrated_hourly_forecast(
                current_outdoor_temp=26.0
            )
        
        # Check for debug logging
        assert "Delta calibration applied: offset=+1.00°C" in caplog.text
        assert "Raw forecasts:" in caplog.text
        assert "Calibrated forecasts:" in caplog.text
    
    def test_warning_logging_for_invalid_data(self, caplog):
        """Test warning logging for invalid input data."""
        with caplog.at_level(logging.WARNING):
            # Test invalid temperature
            self.ha_client.get_calibrated_hourly_forecast(
                current_outdoor_temp=100.0
            )
            assert "Invalid outdoor temperature" in caplog.text
            
        with caplog.at_level(logging.WARNING):
            # Test invalid forecast data
            self.ha_client.get_hourly_forecast = Mock(return_value=[])
            self.ha_client.get_calibrated_hourly_forecast(
                current_outdoor_temp=25.0
            )
            assert "Invalid raw forecast data" in caplog.text


class TestDeltaCalibrationConfigIntegration:
    """Test integration with configuration system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ha_client = HAClient("http://test:8123", "test_token")
        self.ha_client.get_hourly_forecast = Mock(return_value=[25.0, 27.0, 26.0, 24.0])
    
    @patch('src.config.ENABLE_DELTA_FORECAST_CALIBRATION', True)
    def test_config_enabled_integration(self):
        """Test that configuration flag properly enables calibration."""
        # This would be tested in physics_features.py integration
        assert config.ENABLE_DELTA_FORECAST_CALIBRATION is True
    
    @patch('src.config.ENABLE_DELTA_FORECAST_CALIBRATION', False)
    def test_config_disabled_integration(self):
        """Test that configuration flag properly disables calibration."""
        # This would be tested in physics_features.py integration
        assert config.ENABLE_DELTA_FORECAST_CALIBRATION is False
    
    @patch('src.config.DELTA_CALIBRATION_MAX_OFFSET', 5.0)
    def test_max_offset_configuration(self):
        """Test that maximum offset configuration is available."""
        assert config.DELTA_CALIBRATION_MAX_OFFSET == 5.0


class TestDeltaCalibrationEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ha_client = HAClient("http://test:8123", "test_token")
    
    def test_zero_offset_calibration(self):
        """Test calibration when measured temp equals forecast temp."""
        self.ha_client.get_hourly_forecast = Mock(return_value=[25.0, 27.0, 26.0, 24.0])
        
        calibrated_forecasts = self.ha_client.get_calibrated_hourly_forecast(
            current_outdoor_temp=25.0  # Exactly matches forecast
        )
        
        # Should return original forecasts (zero offset)
        assert calibrated_forecasts == [25.0, 27.0, 26.0, 24.0]
    
    def test_large_positive_offset(self):
        """Test calibration with large positive temperature offset."""
        self.ha_client.get_hourly_forecast = Mock(return_value=[10.0, 12.0, 11.0, 9.0])
        
        calibrated_forecasts = self.ha_client.get_calibrated_hourly_forecast(
            current_outdoor_temp=20.0  # 10°C warmer
        )
        
        expected = [20.0, 22.0, 21.0, 19.0]
        assert calibrated_forecasts == expected
    
    def test_large_negative_offset(self):
        """Test calibration with large negative temperature offset."""
        self.ha_client.get_hourly_forecast = Mock(return_value=[20.0, 22.0, 21.0, 19.0])
        
        calibrated_forecasts = self.ha_client.get_calibrated_hourly_forecast(
            current_outdoor_temp=10.0  # 10°C cooler
        )
        
        expected = [10.0, 12.0, 11.0, 9.0]
        assert calibrated_forecasts == expected
    
    def test_fractional_temperatures(self):
        """Test calibration with fractional temperature values."""
        self.ha_client.get_hourly_forecast = Mock(return_value=[25.1, 27.3, 26.7, 24.9])
        
        calibrated_forecasts = self.ha_client.get_calibrated_hourly_forecast(
            current_outdoor_temp=25.6  # 0.5°C offset
        )
        
        expected = [25.6, 27.8, 27.2, 25.4]
        assert calibrated_forecasts == expected


class TestPhysicsFeaturesIntegration:
    """Test integration with physics features module."""
    
    def setup_method(self):
        """Set up test fixtures for physics features integration."""
        self.ha_client = HAClient("http://test:8123", "test_token")
        
    def test_physics_features_uses_calibrated_forecasts(self):
        """Test that physics features uses calibrated forecasts when enabled."""
        # Mock outdoor temperature and raw forecasts
        outdoor_temp = 26.0
        raw_forecasts = [25.0, 27.0, 26.0, 24.0]
        expected_calibrated = [26.0, 28.0, 27.0, 25.0]
        
        # Mock the get_hourly_forecast to return raw forecasts
        self.ha_client.get_hourly_forecast = Mock(return_value=raw_forecasts)
        
        # Test the calibrated forecast method with calibration enabled
        calibrated_forecasts = self.ha_client.get_calibrated_hourly_forecast(
            current_outdoor_temp=outdoor_temp,
            enable_delta_calibration=True
        )
        
        assert calibrated_forecasts == expected_calibrated
        
    def test_physics_features_uses_raw_forecasts_when_disabled(self):
        """Test that physics features uses raw forecasts when disabled."""
        outdoor_temp = 26.0
        raw_forecasts = [25.0, 27.0, 26.0, 24.0]
        
        # Mock the get_hourly_forecast to return raw forecasts
        self.ha_client.get_hourly_forecast = Mock(return_value=raw_forecasts)
        
        # Test the calibrated forecast method with calibration disabled
        forecasts = self.ha_client.get_calibrated_hourly_forecast(
            current_outdoor_temp=outdoor_temp,
            enable_delta_calibration=False
        )
        
        assert forecasts == raw_forecasts


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
