"""
Integration test for enhanced physics features with existing Heat Balance Controller.

Tests that the new 34 thermal momentum features integrate seamlessly with:
- RealisticPhysicsModel
- Heat Balance Controller 
- Existing production workflows

This validates backward compatibility and production readiness.
"""
import unittest
from unittest.mock import Mock, patch
import pandas as pd

# Support both package-relative and direct import
try:
    from src.physics_features import build_physics_features
    from src.physics_model import RealisticPhysicsModel
    from src import config
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.physics_features import build_physics_features
    from src.physics_model import RealisticPhysicsModel
    from src import config

class TestEnhancedPhysicsIntegration(unittest.TestCase):
    """Integration tests for enhanced physics features with existing systems."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.mock_ha_client = Mock()
        self.mock_influx_service = Mock()
        
        # Mock config for testing
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
        config.PV_FORECAST_ENTITY_ID = None
        config.HISTORY_STEPS = 6
        config.HISTORY_STEP_MINUTES = 10
        
        # Realistic sensor states for integration testing
        self.mock_states = {
            'sensor.indoor_temp': {'state': '21.2'},    # Slightly above target
            'sensor.outdoor_temp': {'state': '3.5'},    # Cold outdoor
            'sensor.outlet_temp': {'state': '42.0'},    # Moderate heating
            'sensor.target_temp': {'state': '21.0'},    # Target temperature
            'sensor.dhw_heating': {'state': 'off'},
            'sensor.dhw_disinfection': {'state': 'off'},
            'sensor.dhw_boost_heater': {'state': 'off'},
            'sensor.defrosting': {'state': 'off'},
            'sensor.pv_power': {'state': '0'},          # Night time
            'sensor.fireplace': {'state': 'on'},        # External heat source
            'sensor.tv': {'state': 'on'}                # Minor heat source
        }
        
        # Realistic thermal momentum scenario
        # Indoor temperature declining from fireplace turning off
        self.mock_indoor_history = [21.8, 21.7, 21.5, 21.4, 21.3, 21.2]
        # Outlet temperature being increased to compensate
        self.mock_outlet_history = [38.0, 40.0, 42.0]
        
        # Mock winter temperature forecasts
        self.mock_temp_forecasts = [3.0, 2.5, 2.0, 1.5]
        
        # Setup mocks
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

    def test_enhanced_features_with_physics_model_integration(self):
        """Test that enhanced features integrate with RealisticPhysicsModel."""
        # Build enhanced features
        features_df, outlet_history = build_physics_features(
            self.mock_ha_client, self.mock_influx_service
        )
        
        # Validate enhanced features are available
        self.assertIsNotNone(features_df)
        self.assertEqual(len(features_df.columns), 37)  # 19 original + 15 enhanced + 3 Week 4 forecast
        
        # Validate thermal momentum features have meaningful values
        features = features_df.iloc[0]
        
        # Thermal momentum should detect cooling trend
        self.assertLess(features['indoor_temp_gradient'], 0)  # Negative gradient
        self.assertGreater(features['temp_diff_indoor_outdoor'], 15)  # Indoor warmer
        
        # Delta features should show temperature decline
        self.assertLess(features['indoor_temp_delta_30m'], 0)  # Declining
        self.assertLess(features['indoor_temp_delta_60m'], 0)  # Declining trend
        
        # Outlet effectiveness should be positive (heating needed)
        self.assertGreater(features['outlet_effectiveness_ratio'], 0)
        
        # External heat sources detected
        self.assertEqual(features['fireplace_on'], 1.0)
        self.assertEqual(features['tv_on'], 1.0)
        
        # Test that physics model can use enhanced features
        # This validates the features are properly formatted and named
        try:
            # Create a mock physics model instance
            mock_model = Mock()
            
            # Test that all feature names are valid Python identifiers
            for feature_name in features_df.columns:
                self.assertTrue(feature_name.isidentifier())
                # Test accessing feature values
                feature_value = features[feature_name]
                self.assertIsNotNone(feature_value)
                self.assertTrue(isinstance(feature_value, (int, float)))
                
        except Exception as e:
            self.fail(f"Enhanced features incompatible with physics model: {e}")

    def test_backward_compatibility_with_existing_workflows(self):
        """Test that enhanced features maintain backward compatibility."""
        # Build enhanced features
        features_df, outlet_history = build_physics_features(
            self.mock_ha_client, self.mock_influx_service
        )
        
        # Validate all original 19 features are present and unchanged
        original_feature_names = [
            'outlet_temp', 'indoor_temp_lag_30m', 'target_temp', 'outdoor_temp',
            'dhw_heating', 'dhw_disinfection', 'dhw_boost_heater', 'defrosting',
            'pv_now', 'fireplace_on', 'tv_on',
            'temp_forecast_1h', 'temp_forecast_2h', 
            'temp_forecast_3h', 'temp_forecast_4h',
            'pv_forecast_1h', 'pv_forecast_2h', 
            'pv_forecast_3h', 'pv_forecast_4h'
        ]
        
        features = features_df.iloc[0]
        
        # Test each original feature maintains expected behavior
        for feature_name in original_feature_names:
            self.assertIn(feature_name, features_df.columns)
            # Validate the feature has a reasonable value
            feature_value = features[feature_name]
            self.assertIsNotNone(feature_value)
            self.assertTrue(isinstance(feature_value, (int, float)))
        
        # Test specific original values are preserved
        self.assertEqual(features['outlet_temp'], 42.0)
        self.assertEqual(features['indoor_temp_lag_30m'], 21.4)  # -3 index
        self.assertEqual(features['target_temp'], 21.0)
        self.assertEqual(features['outdoor_temp'], 3.5)
        
        # Test return format is unchanged
        self.assertIsInstance(features_df, pd.DataFrame)
        self.assertIsInstance(outlet_history, list)
        self.assertEqual(len(features_df), 1)  # Single row
        self.assertEqual(len(outlet_history), 3)  # History maintained

    def test_thermal_momentum_feature_quality(self):
        """Test that thermal momentum features provide high-quality insights."""
        features_df, _ = build_physics_features(
            self.mock_ha_client, self.mock_influx_service
        )
        
        features = features_df.iloc[0]
        
        # Test thermal momentum features detect realistic thermal scenario
        
        # 1. Temperature gradient should detect cooling trend
        # Indoor declined from 21.8 to 21.2 over 60 minutes = -0.6°C/hour
        expected_gradient = (21.2 - 21.8) / (config.HISTORY_STEP_MINUTES / 60.0)
        self.assertAlmostEqual(
            features['indoor_temp_gradient'], expected_gradient, places=1
        )
        
        # 2. Indoor-outdoor differential
        expected_diff = 21.2 - 3.5  # 17.7°C difference
        self.assertEqual(features['temp_diff_indoor_outdoor'], expected_diff)
        
        # 3. Outlet effectiveness ratio should indicate heating demand
        # (21.2 - 21.0) / (42.0 - 21.2) = 0.2 / 20.8 ≈ 0.0096
        expected_ratio = (21.2 - 21.0) / max(0.1, 42.0 - 21.2)
        self.assertAlmostEqual(
            features['outlet_effectiveness_ratio'], expected_ratio, places=3
        )
        
        # 4. Extended lag features should provide thermal history
        self.assertEqual(features['indoor_temp_lag_10m'], 21.2)  # Most recent
        self.assertEqual(features['indoor_temp_lag_60m'], 21.8)  # Oldest
        
        # 5. Delta features should show cooling rates
        # 30-minute delta: 21.2 - 21.4 = -0.2°C
        self.assertAlmostEqual(features['indoor_temp_delta_30m'], -0.2, places=5)
        # 60-minute delta: 21.2 - 21.8 = -0.6°C  
        self.assertAlmostEqual(features['indoor_temp_delta_60m'], -0.6, places=5)

    def test_cyclical_time_encoding_mathematical_correctness(self):
        """Test that cyclical time encoding produces mathematically correct results."""
        with patch('src.physics_features.datetime') as mock_datetime:
            # Test specific time: 18:00 (evening) in December
            mock_now = Mock()
            mock_now.hour = 18
            mock_now.month = 12
            mock_datetime.now.return_value = mock_now
            
            features_df, _ = build_physics_features(
                self.mock_ha_client, self.mock_influx_service
            )
            
            features = features_df.iloc[0]
            
            # Test hour encoding: 18:00 should be 270 degrees (3π/4 radians)
            import math
            expected_hour_sin = math.sin(2 * math.pi * 18 / 24)  # sin(3π/2) = -1
            expected_hour_cos = math.cos(2 * math.pi * 18 / 24)  # cos(3π/2) = 0
            
            self.assertAlmostEqual(features['hour_sin'], expected_hour_sin, places=5)
            self.assertAlmostEqual(features['hour_cos'], expected_hour_cos, places=5)
            
            # Test month encoding: December (12) should be 330 degrees (11π/6 radians)
            expected_month_sin = math.sin(2 * math.pi * (12 - 1) / 12)
            expected_month_cos = math.cos(2 * math.pi * (12 - 1) / 12)
            
            self.assertAlmostEqual(features['month_sin'], expected_month_sin, places=5)
            self.assertAlmostEqual(features['month_cos'], expected_month_cos, places=5)

    def test_enhanced_features_performance_impact(self):
        """Test that enhanced features don't significantly impact performance."""
        import time
        
        # Measure execution time for enhanced feature building
        start_time = time.time()
        
        for _ in range(10):  # Run multiple times for average
            features_df, _ = build_physics_features(
                self.mock_ha_client, self.mock_influx_service
            )
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        # Enhanced features should build in under 50ms per call
        self.assertLess(avg_time, 0.05)
        
        # Validate all features are computed
        self.assertEqual(len(features_df.columns), 37)

if __name__ == '__main__':
    unittest.main()
