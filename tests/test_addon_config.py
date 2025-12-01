#!/usr/bin/env python3
"""
Unit tests for ML Heating Add-on Configuration.

This test suite validates that the addon configuration works correctly
and provides regression testing for future changes.
"""
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import json


class TestAddonConfiguration(unittest.TestCase):
    """Test cases for addon configuration functionality."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.original_env = os.environ.copy()
        # Clear any existing environment variables that might interfere
        env_vars_to_clear = [
            'HASS_URL', 'HASS_TOKEN', 'SUPERVISOR_TOKEN', 
            'MODEL_FILE_PATH', 'STATE_FILE_PATH',
            'TARGET_INDOOR_TEMP_ENTITY_ID', 'INDOOR_TEMP_ENTITY_ID',
            'MODEL_FILE', 'STATE_FILE'
        ]
        for var in env_vars_to_clear:
            os.environ.pop(var, None)
        
        # Also clear any modules to prevent .env from being cached
        modules_to_clear = ['src.config', 'config']
        for module in modules_to_clear:
            if module in sys.modules:
                del sys.modules[module]
    
    def tearDown(self):
        """Clean up test environment after each test."""
        # Clear modules to prevent state leakage
        modules_to_clear = ['src.config', 'config']
        for module in modules_to_clear:
            if module in sys.modules:
                del sys.modules[module]
        
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_addon_default_hass_url(self):
        """Test that HASS_URL defaults to supervisor API in addon mode."""
        # Ensure no .env variables interfere
        os.environ.pop('HASS_URL', None)
        
        # Import config after clearing environment
        from src import config
        
        # Should use supervisor API by default
        self.assertEqual(config.HASS_URL, "http://supervisor/core")
    
    def test_addon_supervisor_token(self):
        """Test that SUPERVISOR_TOKEN is correctly used for authentication."""
        test_token = "test_supervisor_token_12345"
        os.environ['SUPERVISOR_TOKEN'] = test_token
        
        # Reload config to pick up environment change
        import importlib
        if 'src.config' in sys.modules:
            importlib.reload(sys.modules['src.config'])
        from src import config
        
        self.assertEqual(config.HASS_TOKEN, test_token)
        self.assertIn(f"Bearer {test_token}", config.HASS_HEADERS['Authorization'])
    
    def test_addon_model_file_paths(self):
        """Test that model files default to /data/ directory structure."""
        import importlib
        if 'src.config' in sys.modules:
            importlib.reload(sys.modules['src.config'])
        from src import config
        
        # Should use addon-specific paths by default
        self.assertEqual(config.MODEL_FILE, "/data/models/ml_model.pkl")
        self.assertEqual(config.STATE_FILE, "/data/models/ml_state.pkl")
    
    def test_addon_environment_variable_override(self):
        """Test that environment variables from config_adapter work correctly."""
        # Set environment variables as config_adapter would
        os.environ['MODEL_FILE_PATH'] = '/data/models/custom_model.pkl'
        os.environ['STATE_FILE_PATH'] = '/data/models/custom_state.pkl'
        os.environ['SUPERVISOR_TOKEN'] = 'custom_token_67890'
        os.environ['HASS_URL'] = 'http://supervisor/core'
        
        # Reload config to pick up environment changes
        import importlib
        if 'src.config' in sys.modules:
            importlib.reload(sys.modules['src.config'])
        from src import config
        
        self.assertEqual(config.MODEL_FILE, '/data/models/custom_model.pkl')
        self.assertEqual(config.STATE_FILE, '/data/models/custom_state.pkl')
        self.assertEqual(config.HASS_TOKEN, 'custom_token_67890')
        self.assertEqual(config.HASS_URL, 'http://supervisor/core')
    
    def test_entity_id_configuration(self):
        """Test that entity IDs are properly configured from environment."""
        # Set test entity IDs
        test_entities = {
            'TARGET_INDOOR_TEMP_ENTITY_ID': 'input_number.test_target',
            'INDOOR_TEMP_ENTITY_ID': 'sensor.test_indoor_temp',
            'OUTDOOR_TEMP_ENTITY_ID': 'sensor.test_outdoor_temp'
        }
        
        for key, value in test_entities.items():
            os.environ[key] = value
        
        # Reload config
        import importlib
        if 'src.config' in sys.modules:
            importlib.reload(sys.modules['src.config'])
        from src import config
        
        self.assertEqual(config.TARGET_INDOOR_TEMP_ENTITY_ID, 'input_number.test_target')
        self.assertEqual(config.INDOOR_TEMP_ENTITY_ID, 'sensor.test_indoor_temp')
        self.assertEqual(config.OUTDOOR_TEMP_ENTITY_ID, 'sensor.test_outdoor_temp')


class TestModelWrapperWithAddonPaths(unittest.TestCase):
    """Test model wrapper functionality with addon paths."""
    
    def setUp(self):
        """Set up test environment with addon paths."""
        self.original_env = os.environ.copy()
        
        # Set up addon-style environment
        os.environ['MODEL_FILE_PATH'] = '/tmp/test_models/ml_model.pkl'
        os.environ['STATE_FILE_PATH'] = '/tmp/test_models/ml_state.pkl'
        os.environ['SUPERVISOR_TOKEN'] = 'test_token'
        
        # Create test directories
        os.makedirs('/tmp/test_models', exist_ok=True)
        
        # Reload config with new environment
        import importlib
        if 'src.config' in sys.modules:
            importlib.reload(sys.modules['src.config'])
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree('/tmp/test_models', ignore_errors=True)
        
        os.environ.clear()
        os.environ.update(self.original_env)
        
        # Reload config to restore original state
        import importlib
        if 'src.config' in sys.modules:
            importlib.reload(sys.modules['src.config'])
    
    def test_model_loading_with_addon_paths(self):
        """Test that model wrapper works with addon directory structure."""
        from src import model_wrapper
        from src import config
        
        # Verify paths are set correctly
        self.assertEqual(config.MODEL_FILE, '/tmp/test_models/ml_model.pkl')
        self.assertEqual(config.STATE_FILE, '/tmp/test_models/ml_state.pkl')
        
        # Test model loading (should create new model since none exists)
        model, mae, rmse = model_wrapper.load_model()
        
        self.assertIsNotNone(model)
        self.assertEqual(mae.get(), 0.0)  # New MAE should start at 0
        self.assertEqual(rmse.get(), 0.0)  # New RMSE should start at 0
    
    def test_model_saving_with_addon_paths(self):
        """Test that model saving works with addon directory structure."""
        from src import model_wrapper
        from src.utils_metrics import MAE, RMSE
        
        # Load model
        model, mae, rmse = model_wrapper.load_model()
        
        # Update metrics to have some data
        mae.update(1.0, 1.2)  # Actual vs predicted
        rmse.update(1.0, 1.2)
        
        # Save model
        model_wrapper.save_model(model, mae, rmse)
        
        # Verify file was created
        from src import config
        self.assertTrue(os.path.exists(config.MODEL_FILE))
        
        # Test reloading
        model2, mae2, rmse2 = model_wrapper.load_model()
        self.assertAlmostEqual(mae2.get(), mae.get(), places=4)
        self.assertAlmostEqual(rmse2.get(), rmse.get(), places=4)


class TestStateManagerWithAddonPaths(unittest.TestCase):
    """Test state manager functionality with addon paths."""
    
    def setUp(self):
        """Set up test environment with addon paths."""
        self.original_env = os.environ.copy()
        
        # Set up addon-style environment
        os.environ['STATE_FILE_PATH'] = '/tmp/test_state/ml_state.pkl'
        
        # Create test directories
        os.makedirs('/tmp/test_state', exist_ok=True)
        
        # Reload config with new environment
        import importlib
        if 'src.config' in sys.modules:
            importlib.reload(sys.modules['src.config'])
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree('/tmp/test_state', ignore_errors=True)
        
        os.environ.clear()
        os.environ.update(self.original_env)
        
        # Reload config to restore original state
        import importlib
        if 'src.config' in sys.modules:
            importlib.reload(sys.modules['src.config'])
    
    def test_state_loading_with_addon_paths(self):
        """Test that state manager works with addon directory structure."""
        from src import state_manager
        from src import config
        
        # Verify path is set correctly
        self.assertEqual(config.STATE_FILE, '/tmp/test_state/ml_state.pkl')
        
        # Test loading state (should return empty dict since none exists)
        state = state_manager.load_state()
        # State manager returns a default dict with None values when no file exists
        self.assertIsInstance(state, dict)
        # Verify that all expected keys exist but are None/False
        expected_keys = ['last_run_features', 'last_indoor_temp', 'last_final_temp',
                        'last_avg_other_rooms_temp', 'last_fireplace_on', 
                        'last_is_blocking', 'last_blocking_end_time']
        for key in expected_keys:
            self.assertIn(key, state)
    
    def test_state_saving_with_addon_paths(self):
        """Test that state saving works with addon directory structure."""
        from src import state_manager
        from src import config
        
        # Save test state
        test_data = {
            'last_indoor_temp': 21.5,
            'last_final_temp': 45.0,
            'last_is_blocking': False
        }
        state_manager.save_state(**test_data)
        
        # Verify file was created
        self.assertTrue(os.path.exists(config.STATE_FILE))
        
        # Test reloading
        loaded_state = state_manager.load_state()
        self.assertEqual(loaded_state['last_indoor_temp'], 21.5)
        self.assertEqual(loaded_state['last_final_temp'], 45.0)
        self.assertEqual(loaded_state['last_is_blocking'], False)


class TestAddonIntegration(unittest.TestCase):
    """Integration tests for full addon configuration flow."""
    
    def test_config_adapter_environment_setup(self):
        """Test that config_adapter environment variables work correctly."""
        # Simulate what config_adapter.py sets
        addon_env = {
            'HASS_URL': 'http://supervisor/core',
            'SUPERVISOR_TOKEN': 'addon_supervisor_token_123',
            'TARGET_INDOOR_TEMP_ENTITY_ID': 'input_number.hp_auto_correct_target',
            'INDOOR_TEMP_ENTITY_ID': 'sensor.thermometer_wohnzimmer_kompensiert',
            'OUTDOOR_TEMP_ENTITY_ID': 'sensor.thermometer_waermepume_kompensiert',
            'MODEL_FILE_PATH': '/data/models/ml_model.pkl',
            'STATE_FILE_PATH': '/data/models/ml_state.pkl',
            'CYCLE_INTERVAL_MINUTES': '30',
            'MAX_TEMP_CHANGE_PER_CYCLE': '2',
            'CLAMP_MIN_ABS': '14.0',
            'CLAMP_MAX_ABS': '65.0'
        }
        
        # Clear and set environment
        original_env = os.environ.copy()
        try:
            os.environ.clear()
            os.environ.update(addon_env)
            
            # Reload config
            import importlib
            if 'src.config' in sys.modules:
                importlib.reload(sys.modules['src.config'])
            from src import config
            
            # Verify all values are loaded correctly
            self.assertEqual(config.HASS_URL, 'http://supervisor/core')
            self.assertEqual(config.HASS_TOKEN, 'addon_supervisor_token_123')
            self.assertEqual(config.TARGET_INDOOR_TEMP_ENTITY_ID, 'input_number.hp_auto_correct_target')
            self.assertEqual(config.INDOOR_TEMP_ENTITY_ID, 'sensor.thermometer_wohnzimmer_kompensiert')
            self.assertEqual(config.MODEL_FILE, '/data/models/ml_model.pkl')
            self.assertEqual(config.STATE_FILE, '/data/models/ml_state.pkl')
            self.assertEqual(config.CYCLE_INTERVAL_MINUTES, 30)
            self.assertEqual(config.MAX_TEMP_CHANGE_PER_CYCLE, 2)
            self.assertEqual(config.CLAMP_MIN_ABS, 14.0)
            self.assertEqual(config.CLAMP_MAX_ABS, 65.0)
            
        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)
            
            # Reload config to restore original state
            import importlib
            if 'src.config' in sys.modules:
                importlib.reload(sys.modules['src.config'])


def suite():
    """Create test suite for addon configuration."""
    test_suite = unittest.TestSuite()
    
    # Add all test cases
    test_suite.addTest(unittest.makeSuite(TestAddonConfiguration))
    test_suite.addTest(unittest.makeSuite(TestModelWrapperWithAddonPaths))
    test_suite.addTest(unittest.makeSuite(TestStateManagerWithAddonPaths))
    test_suite.addTest(unittest.makeSuite(TestAddonIntegration))
    
    return test_suite


if __name__ == '__main__':
    # Run the test suite
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite())
    
    # Print summary
    print("\n" + "="*60)
    print("ADDON CONFIGURATION TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split(chr(10))[-2]}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split(chr(10))[-2]}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n🎉 ALL TESTS PASSED!")
        print("Addon configuration is working correctly and protected against regressions.")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Check the error details above and fix the issues.")
    
    sys.exit(0 if success else 1)
