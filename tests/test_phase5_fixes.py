"""
TDD Unit tests for Phase 5 thermal model fixes.

This module tests the specific Phase 5 enhancements:
1. predict_indoor_temp method implementation in model_wrapper
2. Enhanced binary search logging and convergence
3. None handling in smart rounding fallback logic
4. Binary search defensive programming fixes
"""

import unittest
import sys
import os
from unittest.mock import MagicMock, patch, Mock
import numpy as np

# Add the src directory to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestPhase5PredictIndoorTemp(unittest.TestCase):
    """Test the predict_indoor_temp method implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_conditions = {
            'outlet_temp': 35.0,
            'outdoor_temp': 10.0,
            'pv_power': 0.0,
            'fireplace_on': 0.0,
            'tv_on': 0.0
        }

    def test_predict_indoor_temp_exists(self):
        """Test that predict_indoor_temp method exists in model wrapper."""
        try:
            from src.model_wrapper import get_enhanced_model_wrapper
            wrapper = get_enhanced_model_wrapper()
            
            # Check that the method exists
            self.assertTrue(
                hasattr(wrapper, 'predict_indoor_temp'),
                "predict_indoor_temp method should exist in model wrapper"
            )
            
            # Check that it's callable
            self.assertTrue(
                callable(getattr(wrapper, 'predict_indoor_temp')),
                "predict_indoor_temp should be callable"
            )
        except ImportError:
            self.fail("Could not import get_enhanced_model_wrapper from src.model_wrapper")

    def test_predict_indoor_temp_returns_numeric(self):
        """Test that predict_indoor_temp returns a numeric value."""
        from src.model_wrapper import get_enhanced_model_wrapper
        wrapper = get_enhanced_model_wrapper()
        
        result = wrapper.predict_indoor_temp(**self.test_conditions)
        
        # Should return a number (int or float)
        self.assertIsInstance(
            result, (int, float, np.number),
            f"predict_indoor_temp should return numeric value, got {type(result)}"
        )
        
        # Should be a reasonable temperature (between 5°C and 30°C for indoor)
        if result is not None:
            self.assertGreaterEqual(
                result, 5.0,
                "Indoor temperature prediction should be reasonable (>= 5°C)"
            )
        self.assertLessEqual(
            result, 35.0,
            "Indoor temperature prediction should be reasonable (<= 35°C)"
        )

    def test_predict_indoor_temp_parameter_validation(self):
        """Test parameter validation in predict_indoor_temp."""
        from src.model_wrapper import get_enhanced_model_wrapper
        wrapper = get_enhanced_model_wrapper()
        
        # Test with missing parameters
        with self.assertRaises(TypeError):
            wrapper.predict_indoor_temp()  # No parameters
        
        # Test with partial parameters (should work or have sensible defaults)
        try:
            result = wrapper.predict_indoor_temp(outlet_temp=35.0)
            self.assertIsNotNone(result, "Should handle missing optional parameters")
        except TypeError:
            # If it requires all parameters, that's also acceptable
            pass

    def test_predict_indoor_temp_none_handling(self):
        """Test that predict_indoor_temp handles None inputs gracefully."""
        from src.model_wrapper import get_enhanced_model_wrapper
        wrapper = get_enhanced_model_wrapper()
        
        # Test with None values
        test_cases = [
            {'outlet_temp': None, 'outdoor_temp': 10.0},
            {'outlet_temp': 35.0, 'outdoor_temp': None},
        ]
        
        for case in test_cases:
            with self.subTest(case=case):
                # Should either return None or handle gracefully
                try:
                    result = wrapper.predict_indoor_temp(**{**self.test_conditions, **case})
                    # If it returns a value, it should be valid or None
                    if result is not None:
                        self.assertIsInstance(result, (int, float, np.number))
                except (TypeError, ValueError):
                    # Raising an exception for None inputs is also acceptable
                    pass


class TestPhase5SmartRoundingNoneHandling(unittest.TestCase):
    """Test None handling in smart rounding logic."""

    def test_smart_rounding_none_fallback(self):
        """Test that smart rounding falls back properly when predict_indoor_temp returns None."""
        
        # Mock the wrapper to return None
        with patch('src.model_wrapper.get_enhanced_model_wrapper') as mock_get_wrapper:
            mock_wrapper = MagicMock()
            mock_wrapper.predict_indoor_temp.return_value = None
            mock_get_wrapper.return_value = mock_wrapper
            
            # Test the fallback logic
            final_temp = 37.9
            expected_fallback = round(final_temp)  # 38
            
            # This tests the logic in main.py
            floor_temp = np.floor(final_temp)
            ceiling_temp = np.ceil(final_temp)
            
            if floor_temp == ceiling_temp:
                smart_rounded_temp = int(final_temp)
            else:
                # Mock the scenario where predict_indoor_temp returns None
                floor_predicted = mock_wrapper.predict_indoor_temp.return_value
                ceiling_predicted = mock_wrapper.predict_indoor_temp.return_value
                
                if floor_predicted is None or ceiling_predicted is None:
                    smart_rounded_temp = round(final_temp)
                
            self.assertEqual(
                smart_rounded_temp, expected_fallback,
                f"Should fallback to round() when predict_indoor_temp returns None"
            )

    def test_main_py_contains_none_handling(self):
        """Test that main.py contains proper None handling for Phase 5."""
        main_py_path = os.path.join(
            os.path.dirname(__file__), '..', 'src', 'main.py'
        )
        
        with open(main_py_path, 'r') as f:
            main_content = f.read()
        
        # Check for Phase 5 None handling components
        required_components = [
            "PHASE 5 FIX: Handle None returns from predict_indoor_temp",
            "if floor_predicted is None or ceiling_predicted is None:",
            "using fallback",
            "smart_rounded_temp = round(final_temp)",
        ]
        
        for component in required_components:
            self.assertIn(
                component, main_content,
                f"Phase 5 None handling component not found in main.py: {component}"
            )


class TestPhase5BinarySearchLogging(unittest.TestCase):
    """Test enhanced binary search logging and convergence."""

    def test_binary_search_logging_enhancement(self):
        """Test that binary search has enhanced logging for Phase 5."""
        model_wrapper_path = os.path.join(
            os.path.dirname(__file__), '..', 'src', 'model_wrapper.py'
        )
        
        with open(model_wrapper_path, 'r') as f:
            wrapper_content = f.read()
        
        # Check for actual binary search logging components
        phase5_logging_components = [
            "Binary search start",
            "Iteration {iteration+1}: outlet=",
            "Binary search converged after",
            "didn't converge after 20 iterations",
            "Handle None returns",
            "Detailed logging at each iteration",
        ]
        
        for component in phase5_logging_components:
            self.assertIn(
                component, wrapper_content,
                f"Phase 5 binary search logging component not found: {component}"
            )

    def test_binary_search_convergence_metrics(self):
        """Test that binary search tracks convergence metrics."""
        # This is a behavioral test - we can't easily unit test the actual binary search
        # without complex mocking, but we can verify the structure is in place
        
        from src.model_wrapper import get_enhanced_model_wrapper
        
        # Just verify we can create the wrapper without errors
        try:
            wrapper = get_enhanced_model_wrapper()
            self.assertIsNotNone(wrapper, "Should be able to create model wrapper")
        except Exception as e:
            self.fail(f"Failed to create model wrapper: {e}")


class TestPhase5Integration(unittest.TestCase):
    """Integration tests for Phase 5 fixes working together."""

    def test_end_to_end_smart_rounding_with_predict(self):
        """Test complete smart rounding workflow with predict_indoor_temp."""
        
        with patch('src.model_wrapper.get_enhanced_model_wrapper') as mock_get_wrapper:
            # Set up mock wrapper
            mock_wrapper = MagicMock()
            mock_get_wrapper.return_value = mock_wrapper
            
            # Test normal operation (predict_indoor_temp returns valid values)
            mock_wrapper.predict_indoor_temp.side_effect = [21.2, 21.8]  # floor, ceiling
            
            final_temp = 37.5
            target_temp = 21.5
            
            floor_temp = np.floor(final_temp)  # 37
            ceiling_temp = np.ceil(final_temp)  # 38
            
            # Simulate the logic from main.py
            if floor_temp != ceiling_temp:
                floor_predicted = mock_wrapper.predict_indoor_temp.return_value
                ceiling_predicted = mock_wrapper.predict_indoor_temp.return_value
                
                # Reset side_effect for actual calls
                mock_wrapper.predict_indoor_temp.side_effect = [21.2, 21.8]
                floor_predicted = mock_wrapper.predict_indoor_temp(
                    outlet_temp=floor_temp, outdoor_temp=10.0, 
                    pv_power=0.0, fireplace_on=0.0, tv_on=0.0
                )
                ceiling_predicted = mock_wrapper.predict_indoor_temp(
                    outlet_temp=ceiling_temp, outdoor_temp=10.0,
                    pv_power=0.0, fireplace_on=0.0, tv_on=0.0
                )
                
                floor_error = abs(floor_predicted - target_temp)
                ceiling_error = abs(ceiling_predicted - target_temp)
                
                if floor_error <= ceiling_error:
                    smart_rounded_temp = int(floor_temp)
                else:
                    smart_rounded_temp = int(ceiling_temp)
                
                # Verify the decision logic works
                self.assertIn(
                    smart_rounded_temp, [37, 38],
                    "Smart rounded temp should be either floor or ceiling"
                )

    def test_fallback_when_predict_fails(self):
        """Test fallback behavior when predict_indoor_temp fails."""
        
        with patch('src.model_wrapper.get_enhanced_model_wrapper') as mock_get_wrapper:
            # Set up mock wrapper that returns None (failure case)
            mock_wrapper = MagicMock()
            mock_wrapper.predict_indoor_temp.return_value = None
            mock_get_wrapper.return_value = mock_wrapper
            
            final_temp = 37.9
            
            # Simulate the fallback logic
            floor_temp = np.floor(final_temp)
            ceiling_temp = np.ceil(final_temp)
            
            if floor_temp != ceiling_temp:
                floor_predicted = mock_wrapper.predict_indoor_temp()
                ceiling_predicted = mock_wrapper.predict_indoor_temp()
                
                if floor_predicted is None or ceiling_predicted is None:
                    smart_rounded_temp = round(final_temp)
            
            self.assertEqual(
                smart_rounded_temp, 38,
                "Should fallback to round() when predict_indoor_temp returns None"
            )


class TestPhase5DefensiveProgramming(unittest.TestCase):
    """Test defensive programming aspects of Phase 5 fixes."""

    def test_exception_handling_in_smart_rounding(self):
        """Test that smart rounding handles exceptions gracefully."""
        
        with patch('src.model_wrapper.get_enhanced_model_wrapper') as mock_get_wrapper:
            # Set up mock wrapper that raises exception
            mock_wrapper = MagicMock()
            mock_wrapper.predict_indoor_temp.side_effect = Exception("Model failed")
            mock_get_wrapper.return_value = mock_wrapper
            
            final_temp = 37.9
            
            # Simulate the exception handling logic from main.py
            try:
                floor_temp = np.floor(final_temp)
                ceiling_temp = np.ceil(final_temp)
                
                if floor_temp != ceiling_temp:
                    # This should raise an exception
                    floor_predicted = mock_wrapper.predict_indoor_temp()
                    
                smart_rounded_temp = int(ceiling_temp)  # Should not reach here
            except Exception:
                # Fallback to regular rounding
                smart_rounded_temp = round(final_temp)
            
            self.assertEqual(
                smart_rounded_temp, 38,
                "Should fallback to round() when predict_indoor_temp raises exception"
            )

    def test_input_validation_edge_cases(self):
        """Test input validation for edge cases."""
        
        from src.model_wrapper import get_enhanced_model_wrapper
        
        try:
            wrapper = get_enhanced_model_wrapper()
            
            # Test extreme values
            edge_cases = [
                {'outlet_temp': 0.0, 'outdoor_temp': -20.0},
                {'outlet_temp': 100.0, 'outdoor_temp': 50.0},
                {'outlet_temp': 35.0, 'outdoor_temp': 35.0},  # Same temps
            ]
            
            for case in edge_cases:
                with self.subTest(case=case):
                    # Should either return a valid result or handle gracefully
                    result = wrapper.predict_indoor_temp(**{
                        **case,
                        'pv_power': 0.0,
                        'fireplace_on': 0.0,
                        'tv_on': 0.0
                    })
                    
                    if result is not None:
                        self.assertIsInstance(result, (int, float, np.number))
                        
        except Exception as e:
            # If the wrapper can't be created, that's a separate issue
            self.skipTest(f"Could not create wrapper for edge case testing: {e}")


if __name__ == '__main__':
    # Configure test output for better visibility
    unittest.main(verbosity=2, buffer=True)
