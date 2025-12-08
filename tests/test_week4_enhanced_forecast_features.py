"""
Test Week 4 enhanced forecast features.

Tests the 3 new forecast analysis features added to physics_features.py:
- temp_trend_forecast
- heating_demand_forecast 
- combined_forecast_thermal_load
"""
import unittest
from unittest.mock import Mock, patch
import pandas as pd
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from physics_features import build_physics_features
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class TestWeek4EnhancedForecastFeatures(unittest.TestCase):
    """Test Week 4 enhanced forecast features functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock HA client
        self.mock_ha_client = Mock()
        self.mock_ha_client.get_all_states.return_value = {}
        
        # Mock core sensor states
        self.mock_ha_client.get_state.side_effect = self._mock_get_state
        
        # Mock forecasts
        self.mock_ha_client.get_hourly_forecast.return_value = [10.0, 8.0, 6.0, 4.0]  # Cooling trend
        
        # Mock InfluxDB service
        self.mock_influx_service = Mock()
        self.mock_influx_service.fetch_outlet_history.return_value = [35.0, 35.5, 36.0, 36.5, 37.0, 37.5]
        self.mock_influx_service.fetch_indoor_history.return_value = [20.8, 20.9, 21.0, 21.1, 21.2, 21.3]

    def _mock_get_state(self, entity_id, all_states=None, is_binary=False):
        """Mock get_state method with predictable values."""
        state_map = {
            'sensor.indoor_temp': 21.0,
            'sensor.outdoor_temp': 12.0,
            'sensor.outlet_temp': 35.0,
            'sensor.target_temp': 21.5,
        }
        
        if is_binary:
            return False  # Default binary states to False
            
        return state_map.get(entity_id, 0.0)

    @patch('physics_features.config')
    def test_temp_trend_forecast_feature(self, mock_config):
        """Test temp_trend_forecast feature calculation."""
        # Setup mock config
        mock_config.INDOOR_TEMP_ENTITY_ID = 'sensor.indoor_temp'
        mock_config.OUTDOOR_TEMP_ENTITY_ID = 'sensor.outdoor_temp'
        mock_config.ACTUAL_OUTLET_TEMP_ENTITY_ID = 'sensor.outlet_temp'
        mock_config.TARGET_INDOOR_TEMP_ENTITY_ID = 'sensor.target_temp'
        mock_config.HISTORY_STEPS = 6
        mock_config.HISTORY_STEP_MINUTES = 10
        mock_config.PV_FORECAST_ENTITY_ID = None
        
        # Disable delta calibration for this test to use original forecast method
        mock_config.ENABLE_DELTA_FORECAST_CALIBRATION = False
        
        # Mock other entity IDs to avoid errors
        for attr in ['DHW_STATUS_ENTITY_ID', 'DISINFECTION_STATUS_ENTITY_ID', 
                     'DHW_BOOST_HEATER_STATUS_ENTITY_ID', 'DEFROST_STATUS_ENTITY_ID',
                     'PV_POWER_ENTITY_ID', 'FIREPLACE_STATUS_ENTITY_ID', 'TV_STATUS_ENTITY_ID']:
            setattr(mock_config, attr, f'sensor.{attr.lower()}')

        # Build features
        features_df, _ = build_physics_features(self.mock_ha_client, self.mock_influx_service)
        
        # Verify temp_trend_forecast exists and is calculated
        self.assertIsNotNone(features_df)
        self.assertIn('temp_trend_forecast', features_df.columns)
        
        # Calculate expected value: (temp_forecast_4h - outdoor_temp) / 4.0
        # (4.0 - 12.0) / 4.0 = -2.0 °C/hour (cooling trend)
        expected_temp_trend = (4.0 - 12.0) / 4.0
        actual_temp_trend = features_df['temp_trend_forecast'].iloc[0]
        self.assertAlmostEqual(actual_temp_trend, expected_temp_trend, places=2)

    @patch('physics_features.config')
    def test_heating_demand_forecast_feature(self, mock_config):
        """Test heating_demand_forecast feature calculation."""
        # Setup mock config
        mock_config.INDOOR_TEMP_ENTITY_ID = 'sensor.indoor_temp'
        mock_config.OUTDOOR_TEMP_ENTITY_ID = 'sensor.outdoor_temp'
        mock_config.ACTUAL_OUTLET_TEMP_ENTITY_ID = 'sensor.outlet_temp'
        mock_config.TARGET_INDOOR_TEMP_ENTITY_ID = 'sensor.target_temp'
        mock_config.HISTORY_STEPS = 6
        mock_config.HISTORY_STEP_MINUTES = 10
        mock_config.PV_FORECAST_ENTITY_ID = None
        
        # Disable delta calibration for this test to use original forecast method
        mock_config.ENABLE_DELTA_FORECAST_CALIBRATION = False
        
        # Mock other entity IDs
        for attr in ['DHW_STATUS_ENTITY_ID', 'DISINFECTION_STATUS_ENTITY_ID', 
                     'DHW_BOOST_HEATER_STATUS_ENTITY_ID', 'DEFROST_STATUS_ENTITY_ID',
                     'PV_POWER_ENTITY_ID', 'FIREPLACE_STATUS_ENTITY_ID', 'TV_STATUS_ENTITY_ID']:
            setattr(mock_config, attr, f'sensor.{attr.lower()}')

        # Build features
        features_df, _ = build_physics_features(self.mock_ha_client, self.mock_influx_service)
        
        # Verify heating_demand_forecast exists and is calculated
        self.assertIsNotNone(features_df)
        self.assertIn('heating_demand_forecast', features_df.columns)
        
        # Calculate expected value: max(0.0, (21.0 - temp_forecast_4h) * 0.1)
        # max(0.0, (21.0 - 4.0) * 0.1) = 1.7 (heating needed)
        expected_heating_demand = max(0.0, (21.0 - 4.0) * 0.1)
        actual_heating_demand = features_df['heating_demand_forecast'].iloc[0]
        self.assertAlmostEqual(actual_heating_demand, expected_heating_demand, places=2)

    @patch('physics_features.config')
    def test_combined_forecast_thermal_load_feature(self, mock_config):
        """Test combined_forecast_thermal_load feature calculation."""
        # Setup mock config
        mock_config.INDOOR_TEMP_ENTITY_ID = 'sensor.indoor_temp'
        mock_config.OUTDOOR_TEMP_ENTITY_ID = 'sensor.outdoor_temp'
        mock_config.ACTUAL_OUTLET_TEMP_ENTITY_ID = 'sensor.outlet_temp'
        mock_config.TARGET_INDOOR_TEMP_ENTITY_ID = 'sensor.target_temp'
        mock_config.HISTORY_STEPS = 6
        mock_config.HISTORY_STEP_MINUTES = 10
        mock_config.PV_FORECAST_ENTITY_ID = None
        
        # Disable delta calibration for this test to use original forecast method
        mock_config.ENABLE_DELTA_FORECAST_CALIBRATION = False
        
        # Mock other entity IDs
        for attr in ['DHW_STATUS_ENTITY_ID', 'DISINFECTION_STATUS_ENTITY_ID', 
                     'DHW_BOOST_HEATER_STATUS_ENTITY_ID', 'DEFROST_STATUS_ENTITY_ID',
                     'PV_POWER_ENTITY_ID', 'FIREPLACE_STATUS_ENTITY_ID', 'TV_STATUS_ENTITY_ID']:
            setattr(mock_config, attr, f'sensor.{attr.lower()}')

        # Build features
        features_df, _ = build_physics_features(self.mock_ha_client, self.mock_influx_service)
        
        # Verify combined_forecast_thermal_load exists and is calculated
        self.assertIsNotNone(features_df)
        self.assertIn('combined_forecast_thermal_load', features_df.columns)
        
        # Calculate expected value: heating_demand - (pv_forecast_4h * 0.001)
        # (max(0.0, (21.0 - 4.0) * 0.1) - (0.0 * 0.001)) = 1.7
        heating_demand = max(0.0, (21.0 - 4.0) * 0.1)
        pv_offset = 0.0 * 0.001  # No PV forecasts in this test
        expected_thermal_load = heating_demand - pv_offset
        actual_thermal_load = features_df['combined_forecast_thermal_load'].iloc[0]
        self.assertAlmostEqual(actual_thermal_load, expected_thermal_load, places=2)

    @patch('physics_features.config')
    def test_week4_features_count(self, mock_config):
        """Test that we now have 37 total features (34 original + 3 new)."""
        # Setup mock config
        mock_config.INDOOR_TEMP_ENTITY_ID = 'sensor.indoor_temp'
        mock_config.OUTDOOR_TEMP_ENTITY_ID = 'sensor.outdoor_temp'
        mock_config.ACTUAL_OUTLET_TEMP_ENTITY_ID = 'sensor.outlet_temp'
        mock_config.TARGET_INDOOR_TEMP_ENTITY_ID = 'sensor.target_temp'
        mock_config.HISTORY_STEPS = 6
        mock_config.HISTORY_STEP_MINUTES = 10
        mock_config.PV_FORECAST_ENTITY_ID = None
        
        # Mock other entity IDs
        for attr in ['DHW_STATUS_ENTITY_ID', 'DISINFECTION_STATUS_ENTITY_ID', 
                     'DHW_BOOST_HEATER_STATUS_ENTITY_ID', 'DEFROST_STATUS_ENTITY_ID',
                     'PV_POWER_ENTITY_ID', 'FIREPLACE_STATUS_ENTITY_ID', 'TV_STATUS_ENTITY_ID']:
            setattr(mock_config, attr, f'sensor.{attr.lower()}')

        # Build features
        features_df, _ = build_physics_features(self.mock_ha_client, self.mock_influx_service)
        
        # Should now have 37 features (34 original + 3 new Week 4 features)
        self.assertIsNotNone(features_df)
        self.assertEqual(len(features_df.columns), 37)
        
        # Verify all 3 new features exist
        week4_features = ['temp_trend_forecast', 'heating_demand_forecast', 'combined_forecast_thermal_load']
        for feature in week4_features:
            self.assertIn(feature, features_df.columns, f"Missing Week 4 feature: {feature}")


if __name__ == '__main__':
    unittest.main()
