"""
Unit tests for the refactored smart rounding functionality.
"""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch

from src.temperature_control import SmartRounding


class TestUnifiedSmartRounding(unittest.TestCase):
    """Test cases for the refactored SmartRounding class."""

    @patch('src.temperature_control.get_enhanced_model_wrapper')
    def test_smart_rounding_logic(self, mock_get_wrapper):
        """Test that smart rounding correctly chooses floor or ceiling."""
        
        mock_wrapper = MagicMock()
        mock_get_wrapper.return_value = mock_wrapper

        # Define the unified forecast conditions
        mock_wrapper.cycle_aligned_forecast = {
            'outdoor_temp': 5.0,
            'pv_power': 100.0,
            'fireplace_on': 0.0,
            'tv_on': 1.0,
        }

        rounding_logic = SmartRounding()

        test_cases = [
            (37.9, 22.0, 38, "Round up"),
            (37.1, 22.0, 37, "Round down"),
            (42.5, 20.0, 43, "Round up for cooling"),
        ]

        for final_temp, target_temp, expected, desc in test_cases:
            with self.subTest(msg=desc):
                floor_pred = target_temp + (0.5 if expected == np.ceil(final_temp) else 0.2)
                ceil_pred = target_temp + (0.2 if expected == np.ceil(final_temp) else 0.5)
                
                mock_wrapper.predict_indoor_temp.side_effect = [floor_pred, ceil_pred]

                result = rounding_logic.apply_smart_rounding(final_temp, target_temp)
                self.assertEqual(result, expected)

    @patch('src.temperature_control.get_enhanced_model_wrapper')
    def test_smart_rounding_no_rounding_needed(self, mock_get_wrapper):
        """Test that integer temperatures are not rounded."""
        rounding_logic = SmartRounding()
        result = rounding_logic.apply_smart_rounding(38.0, 22.0)
        self.assertEqual(result, 38)

    @patch('src.temperature_control.get_enhanced_model_wrapper')
    def test_smart_rounding_fallback_on_none(self, mock_get_wrapper):
        """Test fallback to standard rounding if prediction fails."""
        mock_wrapper = MagicMock()
        mock_get_wrapper.return_value = mock_wrapper
        mock_wrapper.predict_indoor_temp.return_value = None

        rounding_logic = SmartRounding()
        result = rounding_logic.apply_smart_rounding(37.9, 22.0)
        self.assertEqual(result, 38)

if __name__ == '__main__':
    unittest.main()
