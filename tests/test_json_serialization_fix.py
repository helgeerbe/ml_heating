"""
Test JSON serialization fix for numpy boolean types.

This test ensures that numpy boolean types (np.bool_, np.bool8) are properly
converted to Python booleans for JSON serialization.
"""

import unittest
import numpy as np
import tempfile
import os
from src.unified_thermal_state import ThermalStateManager


class TestJSONSerializationFix(unittest.TestCase):
    """Test JSON serialization fix for numpy boolean types."""

    def setUp(self):
        """Set up test with temporary thermal state file."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        self.manager = ThermalStateManager(state_file=self.temp_file.name)

    def tearDown(self):
        """Clean up temporary file."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_numpy_boolean_conversion(self):
        """Test that numpy boolean types are properly converted."""
        # Test numpy boolean types
        test_data = {
            'np_bool_': np.bool_(True),
            'np_bool_false': np.bool_(False),
            'regular_bool': True,
            'nested_data': {
                'np_bool_nested': np.bool_(True),
                'list_with_np_bool': [np.bool_(False), True, np.bool_(True)]
            }
        }

        # Test the conversion function directly
        converted = self.manager._convert_numpy_types(test_data)
        
        # All values should be Python booleans or native types
        self.assertIsInstance(converted['np_bool_'], bool)
        self.assertIsInstance(converted['np_bool_false'], bool)
        self.assertIsInstance(converted['regular_bool'], bool)
        self.assertIsInstance(converted['nested_data']['np_bool_nested'], bool)
        
        # Check list conversion
        for item in converted['nested_data']['list_with_np_bool']:
            self.assertIsInstance(item, bool)

        # Values should be preserved correctly
        self.assertEqual(converted['np_bool_'], True)
        self.assertEqual(converted['np_bool_false'], False)
        self.assertEqual(converted['regular_bool'], True)
        self.assertEqual(converted['nested_data']['np_bool_nested'], True)
        self.assertEqual(converted['nested_data']['list_with_np_bool'], [False, True, True])

    def test_insufficient_data_serialization(self):
        """Test that insufficient_data responses can be serialized."""
        # Simulate the insufficient_data response that was causing issues
        insufficient_data_response = {"insufficient_data": True}
        
        # This should not raise a JSON serialization error
        converted = self.manager._convert_numpy_types(insufficient_data_response)
        self.assertEqual(converted, {"insufficient_data": True})
        
        # Test with numpy boolean (the likely culprit)
        insufficient_data_np = {"insufficient_data": np.bool_(True)}
        converted_np = self.manager._convert_numpy_types(insufficient_data_np)
        self.assertEqual(converted_np, {"insufficient_data": True})
        self.assertIsInstance(converted_np["insufficient_data"], bool)

    def test_thermal_state_save_with_booleans(self):
        """Test saving thermal state with boolean values."""
        # Add some boolean values to the operational state
        self.manager.update_operational_state(
            last_fireplace_on=np.bool_(True),
            last_is_blocking=np.bool_(False),
            is_calibrating=True
        )
        
        # This should not raise a JSON serialization error
        success = self.manager.save_state()
        self.assertTrue(success)
        
        # Reload and verify values are preserved correctly
        new_manager = ThermalStateManager(state_file=self.temp_file.name)
        operational_state = new_manager.get_operational_state()
        
        self.assertEqual(operational_state['last_fireplace_on'], True)
        self.assertEqual(operational_state['last_is_blocking'], False)
        self.assertEqual(operational_state['is_calibrating'], True)


if __name__ == '__main__':
    unittest.main()
