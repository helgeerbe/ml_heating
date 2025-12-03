"""
Unit tests for enhanced physics features with thermal momentum analysis.

Tests the new 15 thermal momentum features added to physics_features.py:
- Thermal momentum analysis (3 features)
- Extended lag features (4 features) 
- Delta analysis (3 features)
- Cyclical time encoding (4 features)
- Outlet effectiveness analysis (1 feature)

Following project's mandatory unit test policy for all code changes.
"""
import unittest
from unittest.mock import Mock, patch
import math
from datetime import datetime
import pandas as pd

# Support both package-relative and direct import
try:
    from src.physics_features import build_physics_features
    from src import config
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.physics_features import build_physics_features
    from src import config


class TestEnhancedPhysicsFeatures(unittest.TestCase):
    """Test suite for enhanced physics features with thermal momentum."""
    
    def setUp(self):
        """Set up test fixtures with mock data."""
        self.mock_ha_client = Mock()
        self.mock_influx_service = Mock()
        
        # Mock config entity IDs for testing
        config.INDOOR_TEMP_ENTITY_ID = 'sensor.indoor_temp'
        config.OUTDOOR_TEMP_ENTITY_ID = 'sensor.outdoor_temp' 
        config.ACTUAL_OUTLET_TEMP_ENTITY_ID = 'sensor.outlet_temp'
        config.TARGET_INDOOR_TEMP_ENTITY_ID = 'sensor.target_temp'
        config.DHW_STATUS_ENTITY_ID = 'sensor.dhw_heating'
        config.DISINFECTION_STATUS_ENTITY_ID = 'sensor.dhw_disinfection'
        config.DHW_BOOST_HEATER_STATUS_ENTITY_ID = 'sensor.dhw_boost_heater'
        config.DEFROST_STATUS_ENTITY_ID = 'sensor.defrosting'
        config.PV_POWER_ENTITY_ID = 'sensor.pv_power'
        config.FIREPLACE_STATUS_ENTITY_ID = 'sensor.fireplace'
        config.TV_STATUS_ENTITY_ID = 'sensor.tv'
        config.PV_FORECAST_ENTITY_ID = None  # Disable PV forecast for testing
        
        # Mock sensor states
        self.mock_states = {
            'sensor.indoor_temp': {'state': '21.5'},
            'sensor.outdoor_temp': {'state': '5.0'},
            'sensor.outlet_temp': {'state': '45.0'},
            'sensor.target_temp': {'state': '21.0'},
            'sensor.dhw_heating': {'state': 'off'},
            'sensor.dhw_disinfection': {'state': 'off'},
            'sensor.dhw_boost_heater': {'state': 'off'},
            'sensor.defrosting': {'state': 'off'},
            'sensor.pv_power': {'state': '1500'},
            'sensor.fireplace': {'state': 'off'},
            'sensor.tv': {'state': 'on'}
        }
        
        # Mock history data for thermal momentum features
        # 6 indoor values (60 min history): [oldest -> newest]
        self.mock_indoor_history = [21.0, 21.1, 21.2, 21.3, 21.4, 21.5]
        # 3 outlet values (30 min history): [oldest -> newest]  
        self.mock_outlet_history = [42.0, 43.0, 44.0]
        
        # Mock temperature forecasts
        self.mock_temp_forecasts = [4.5, 4.0, 3.5, 3.0]
        
        # Setup standard mocks
        self.mock_ha_client.get_all_states.return_value = self.mock_states
        self.mock_ha_client.get_state.side_effect = self._mock_get_state
        self.mock_ha_client.get_hourly_forecast.return_value = (
            self.mock_temp_forecasts
        )
        
        self.mock_influx_service.fetch_indoor_history.return_value = (
            self.mock_indoor_history
        )
        self.mock_influx_service.fetch_outlet_history.return_value = (
            self.mock_outlet_history
        )
        
        # Mock config values
        config.HISTORY_STEPS = 6
        config.HISTORY_STEP_MINUTES = 10

    def _mock_get_state(self, entity_id, states, is_binary=False):
        """Helper to mock ha_client.get_state behavior."""
        if entity_id in states:
            state_value = states[entity_id]['state']
            if is_binary:
                return state_value == 'on'
            else:
                try:
                    return float(state_value)
                except ValueError:
                    return state_value
        return None

    def test_enhanced_feature_count(self):
        """Test that 37 total features are generated (19 original + 15 enhanced + 3 Week 4 forecast)."""
        features_df, _ = build_physics_features(
            self.mock_ha_client, self.mock_influx_service
        )
        
        self.assertIsNotNone(features_df)
        self.assertEqual(len(features_df.columns), 37)
        self.assertEqual(len(features_df), 1)  # Single row
        
    def test_backward_compatibility_original_features(self):
        """Test that all 19 original features are still present."""
        features_df, _ = build_physics_features(
            self.mock_ha_client, self.mock_influx_service
        )
        
        # Original 19 feature names
        original_features = [
            'outlet_temp', 'indoor_temp_lag_30m', 'target_temp', 'outdoor_temp',
            'dhw_heating', 'dhw_disinfection', 'dhw_boost_heater', 'defrosting',
            'pv_now', 'fireplace_on', 'tv_on',
            'temp_forecast_1h', 'temp_forecast_2h', 
            'temp_forecast_3h', 'temp_forecast_4h',
            'pv_forecast_1h', 'pv_forecast_2h', 
            'pv_forecast_3h', 'pv_forecast_4h'
        ]
        
        for feature in original_features:
            self.assertIn(feature, features_df.columns)
        
        # Test original feature values are correct
        features = features_df.iloc[0]
        self.assertEqual(features['outlet_temp'], 45.0)
        self.assertEqual(features['indoor_temp_lag_30m'], 21.3)  # -3 index
        self.assertEqual(features['target_temp'], 21.0)
        self.assertEqual(features['outdoor_temp'], 5.0)

    def test_thermal_momentum_features(self):
        """Test P0 Priority: Thermal momentum analysis features."""
        features_df, _ = build_physics_features(
            self.mock_ha_client, self.mock_influx_service
        )
        
        features = features_df.iloc[0]
        
        # Test temp_diff_indoor_outdoor
        expected_diff = 21.5 - 5.0  # indoor - outdoor
        self.assertEqual(features['temp_diff_indoor_outdoor'], expected_diff)
        
        # Test indoor_temp_gradient  
        # (current - oldest) / time_period
        # (21.5 - 21.0) / (10/60) = 0.5 / 0.167 = ~3.0
        time_period = config.HISTORY_STEP_MINUTES / 60.0
        expected_gradient = (21.5 - 21.0) / time_period
        self.assertAlmostEqual(
            features['indoor_temp_gradient'], 
            expected_gradient, places=2
        )
        
        # Test outlet_indoor_diff
        expected_outlet_diff = 45.0 - 21.5  # outlet - indoor
        self.assertEqual(features['outlet_indoor_diff'], expected_outlet_diff)

    def test_extended_lag_features(self):
        """Test P0 Priority: Extended lag features."""
        features_df, _ = build_physics_features(
            self.mock_ha_client, self.mock_influx_service
        )
        
        features = features_df.iloc[0]
        
        # Test indoor_temp_lag_10m (index -1, most recent)
        self.assertEqual(features['indoor_temp_lag_10m'], 21.5)
        
        # Test indoor_temp_lag_60m (index -6, oldest)
        self.assertEqual(features['indoor_temp_lag_60m'], 21.0)
        
        # Test outlet_temp_lag_30m (index -3)
        self.assertEqual(features['outlet_temp_lag_30m'], 42.0)
        
        # Test outlet_temp_change
        expected_change = 45.0 - 44.0  # current - previous
        self.assertEqual(features['outlet_temp_change'], expected_change)

    def test_delta_analysis_features(self):
        """Test P1 Priority: Delta analysis features."""
        features_df, _ = build_physics_features(
            self.mock_ha_client, self.mock_influx_service
        )
        
        features = features_df.iloc[0]
        
        # Test indoor_temp_delta_10m (current - 10min ago)
        expected_10m = 21.5 - 21.5  # current - index[-1]
        self.assertEqual(features['indoor_temp_delta_10m'], expected_10m)
        
        # Test indoor_temp_delta_30m (current - 30min ago)
        expected_30m = 21.5 - 21.3  # current - index[-3]
        self.assertEqual(features['indoor_temp_delta_30m'], expected_30m)
        
        # Test indoor_temp_delta_60m (current - 60min ago)
        expected_60m = 21.5 - 21.0  # current - index[-6]
        self.assertEqual(features['indoor_temp_delta_60m'], expected_60m)

    @patch('src.physics_features.datetime')
    def test_cyclical_time_encoding(self, mock_datetime):
        """Test P1 Priority: Cyclical time encoding features."""
        # Mock specific time: 15:30 on March 15th
        mock_now = Mock()
        mock_now.hour = 15
        mock_now.month = 3
        mock_datetime.now.return_value = mock_now
        
        features_df, _ = build_physics_features(
            self.mock_ha_client, self.mock_influx_service
        )
        
        features = features_df.iloc[0]
        
        # Test hour encoding (15:30 -> 15th hour)
        expected_hour_sin = math.sin(2 * math.pi * 15 / 24)
        expected_hour_cos = math.cos(2 * math.pi * 15 / 24)
        
        self.assertAlmostEqual(
            features['hour_sin'], expected_hour_sin, places=5
        )
        self.assertAlmostEqual(
            features['hour_cos'], expected_hour_cos, places=5
        )
        
        # Test month encoding (March -> month 3)
        expected_month_sin = math.sin(2 * math.pi * (3 - 1) / 12)
        expected_month_cos = math.cos(2 * math.pi * (3 - 1) / 12)
        
        self.assertAlmostEqual(
            features['month_sin'], expected_month_sin, places=5
        )
        self.assertAlmostEqual(
            features['month_cos'], expected_month_cos, places=5
        )

    def test_outlet_effectiveness_analysis(self):
        """Test P2 Priority: Outlet effectiveness analysis."""
        features_df, _ = build_physics_features(
            self.mock_ha_client, self.mock_influx_service
        )
        
        features = features_df.iloc[0]
        
        # Test outlet_effectiveness_ratio
        # (indoor - target) / max(0.1, outlet - indoor)
        # (21.5 - 21.0) / max(0.1, 45.0 - 21.5)
        # 0.5 / 23.5 ≈ 0.021
        expected_ratio = (21.5 - 21.0) / max(0.1, 45.0 - 21.5)
        self.assertAlmostEqual(
            features['outlet_effectiveness_ratio'], 
            expected_ratio, places=3
        )

    def test_insufficient_history_error(self):
        """Test edge case: Insufficient history data."""
        # Mock insufficient indoor history (less than 6 required)
        self.mock_influx_service.fetch_indoor_history.return_value = [21.0, 21.1]
        
        features_df, _ = build_physics_features(
            self.mock_ha_client, self.mock_influx_service
        )
        
        self.assertIsNone(features_df)

    def test_insufficient_outlet_history_error(self):
        """Test edge case: Insufficient outlet history data."""
        # Mock insufficient outlet history (less than 3 required)
        self.mock_influx_service.fetch_outlet_history.return_value = [42.0]
        
        features_df, _ = build_physics_features(
            self.mock_ha_client, self.mock_influx_service
        )
        
        self.assertIsNone(features_df)

    def test_missing_sensor_data_error(self):
        """Test edge case: Missing critical sensor data."""
        # Mock missing indoor temperature sensor
        self.mock_ha_client.get_state.side_effect = (
            lambda entity_id, states, is_binary=False: None
        )
        
        features_df, _ = build_physics_features(
            self.mock_ha_client, self.mock_influx_service
        )
        
        self.assertIsNone(features_df)

    def test_outlet_effectiveness_division_by_zero_protection(self):
        """Test edge case: Division by zero protection in effectiveness ratio."""
        # Mock scenario where outlet temp equals indoor temp
        self.mock_states['sensor.outlet_temp']['state'] = '21.5'
        
        features_df, _ = build_physics_features(
            self.mock_ha_client, self.mock_influx_service
        )
        
        features = features_df.iloc[0]
        
        # Should use max(0.1, 0) = 0.1 to prevent division by zero
        expected_ratio = (21.5 - 21.0) / 0.1
        self.assertAlmostEqual(
            features['outlet_effectiveness_ratio'], 
            expected_ratio, places=3
        )

    def test_extended_steps_configuration(self):
        """Test that extended steps are properly calculated."""
        # Test with smaller HISTORY_STEPS
        config.HISTORY_STEPS = 3
        
        # Should still request 6 steps for thermal momentum features
        build_physics_features(
            self.mock_ha_client, self.mock_influx_service
        )
        
        # Verify that max(6, 3) = 6 steps were requested
        self.mock_influx_service.fetch_indoor_history.assert_called_with(6)
        self.mock_influx_service.fetch_outlet_history.assert_called_with(6)

    def test_all_new_features_present(self):
        """Test that all 15 new thermal momentum features are present."""
        features_df, _ = build_physics_features(
            self.mock_ha_client, self.mock_influx_service
        )
        
        new_features = [
            # Thermal momentum (3)
            'temp_diff_indoor_outdoor', 'indoor_temp_gradient', 
            'outlet_indoor_diff',
            # Extended lag (4) 
            'indoor_temp_lag_10m', 'indoor_temp_lag_60m',
            'outlet_temp_lag_30m', 'outlet_temp_change',
            # Delta analysis (3)
            'indoor_temp_delta_10m', 'indoor_temp_delta_30m',
            'indoor_temp_delta_60m',
            # Cyclical time (4)
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            # Outlet effectiveness (1)
            'outlet_effectiveness_ratio'
        ]
        
        for feature in new_features:
            self.assertIn(feature, features_df.columns)
        
        # Verify exactly 15 new features
        self.assertEqual(len(new_features), 15)


if __name__ == '__main__':
    unittest.main()
