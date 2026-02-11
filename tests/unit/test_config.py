
import unittest
from unittest.mock import patch
import sys
import os

class TestConfig(unittest.TestCase):

    def setUp(self):
        # This ensures that src.config is reloaded for each test
        if 'src.config' in sys.modules:
            del sys.modules['src.config']

    @patch('dotenv.load_dotenv')
    def test_addon_environment_config(self, mock_load_dotenv):
        """Test that config uses addon paths when SUPERVISOR_TOKEN is set."""
        with patch.dict(os.environ, {'SUPERVISOR_TOKEN': 'dummy-token'}, clear=True):
            from src import config
            self.assertEqual(config.UNIFIED_STATE_FILE, "/opt/ml_heating/unified_thermal_state.json")

    @patch('dotenv.load_dotenv')
    def test_standalone_environment_config(self, mock_load_dotenv):
        """Test that config uses standalone paths when SUPERVISOR_TOKEN is not set."""
        with patch.dict(os.environ, {}, clear=True):
            from src import config
            self.assertEqual(config.UNIFIED_STATE_FILE, "/opt/ml_heating/unified_thermal_state.json")


if __name__ == '__main__':
    unittest.main()
