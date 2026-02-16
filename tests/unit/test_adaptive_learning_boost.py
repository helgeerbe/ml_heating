import pytest
from unittest.mock import patch
from src.thermal_equilibrium_model import ThermalEquilibriumModel


class TestAdaptiveLearningBoost:

    @pytest.fixture
    def model(self):
        # Initialize model
        # We might need to mock dependencies if __init__ does heavy lifting, 
        # but looking at the code it seems safe to instantiate for unit tests 
        # if we don't call methods that use external services.
        model = ThermalEquilibriumModel()
        # Reset history to ensure clean state
        model.prediction_history = []
        model.parameter_history = []
        return model

    def test_learning_rate_boost_medium_error(self, model):
        """
        Verify that learning rate is boosted by 1.5x when average error is
        > 0.2°C but <= 1.0°C.
        """
        # Ensure SHADOW_MODE is False so we hit the production logic
        with patch('src.thermal_equilibrium_model.config.SHADOW_MODE', False):
            # Set up prediction history with errors averaging 0.3
            # We need at least 5 entries for the logic to trigger
            errors = [0.3, 0.3, 0.3, 0.3, 0.3]
            model.prediction_history = [{"error": e} for e in errors]
            
            # Set known learning parameters
            model.learning_rate = 0.01
            model.min_learning_rate = 0.001
            model.learning_confidence = 1.0
            model.max_learning_rate = 1.0
            
            # Expected calculation:
            # base_rate = 0.01 * 1.0 = 0.01
            # avg_error = 0.3
            # Since 0.2 < 0.3 <= 1.0, boost factor is 1.5
            # expected_rate = 0.01 * 1.5 = 0.015
            
            calculated_rate = model._calculate_adaptive_learning_rate()
            
            assert calculated_rate == pytest.approx(0.015)

    def test_learning_rate_boost_high_error(self, model):
        """
        Verify that learning rate is boosted by 2.0x when average error is
        > 1.0°C.
        """
        with patch('src.thermal_equilibrium_model.config.SHADOW_MODE', False):
            errors = [1.2, 1.2, 1.2, 1.2, 1.2]
            model.prediction_history = [{"error": e} for e in errors]
            
            model.learning_rate = 0.01
            model.min_learning_rate = 0.001
            model.learning_confidence = 1.0
            model.max_learning_rate = 1.0
            
            # Expected calculation:
            # avg_error = 1.2
            # Since 1.0 < 1.2 <= 2.0, boost factor is 2.0
            # expected_rate = 0.01 * 2.0 = 0.02
            
            calculated_rate = model._calculate_adaptive_learning_rate()
            
            assert calculated_rate == pytest.approx(0.02)

    def test_learning_rate_no_boost_small_error(self, model):
        """
        Verify that learning rate is NOT boosted when average error is
        <= 0.2°C.
        """
        with patch('src.thermal_equilibrium_model.config.SHADOW_MODE', False):
            errors = [0.15, 0.15, 0.15, 0.15, 0.15]
            model.prediction_history = [{"error": e} for e in errors]
            
            model.learning_rate = 0.01
            model.min_learning_rate = 0.001
            model.learning_confidence = 1.0
            
            # Expected calculation:
            # avg_error = 0.15
            # No boost applied
            # expected_rate = 0.01
            
            calculated_rate = model._calculate_adaptive_learning_rate()
            
            assert calculated_rate == pytest.approx(0.01)
