import pytest
import numpy as np
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.thermal_equilibrium_model import ThermalEquilibriumModel
from src.thermal_constants import PhysicsConstants

@pytest.fixture
def clean_model():
    """Fixture to ensure a clean model instance for each test function."""
    model = ThermalEquilibriumModel()
    # Set realistic test parameters
    model.thermal_time_constant = 4.0
    model.heat_loss_coefficient = 1.2
    model.outlet_effectiveness = 0.75
    model.external_source_weights = {
        "pv": 0.0001,
        "fireplace": 1.5,
        "tv": 0.1
    }
    return model

class TestConfidenceBoosting:
    def test_confidence_boosting_logic(self, clean_model):
        """Verify confidence boosting logic with new thresholds."""
        model = clean_model
        model.learning_confidence = 1.0
        model.confidence_boost_rate = 1.1
        model.confidence_decay_rate = 0.9
        model.recent_errors_window = 5
        
        # Mock prediction history with small errors (< 0.2)
        # This should trigger confidence boost
        # We need enough history to trigger the check (>= recent_errors_window)
        model.prediction_history = [
            {"error": 0.1, "timestamp": "2023-01-01T00:00:00", "context": {}, "abs_error": 0.1},
            {"error": 0.1, "timestamp": "2023-01-01T00:00:00", "context": {}, "abs_error": 0.1},
            {"error": 0.1, "timestamp": "2023-01-01T00:00:00", "context": {}, "abs_error": 0.1},
            {"error": 0.1, "timestamp": "2023-01-01T00:00:00", "context": {}, "abs_error": 0.1},
            {"error": 0.1, "timestamp": "2023-01-01T00:00:00", "context": {}, "abs_error": 0.1}
        ]
        
        # Trigger update
        prediction_context = {
            "outlet_temp": 40.0,
            "current_indoor": 20.0,
            "outdoor_temp": 5.0,
            "pv_power": 0,
            "fireplace_on": 0,
            "tv_on": 0
        }
        
        # We need to mock _adapt_parameters_from_recent_errors to avoid side effects
        # and focus only on confidence update logic in update_prediction_feedback
        original_adapt = model._adapt_parameters_from_recent_errors
        model._adapt_parameters_from_recent_errors = lambda: None
        
        try:
            # Add a new prediction with small error
            # The error will be calculated as actual - predicted = 20.1 - 20.0 = 0.1
            # 0.1 < ERROR_THRESHOLD_CONFIDENCE (0.2) -> Boost
            model.update_prediction_feedback(
                predicted_temp=20.0,
                actual_temp=20.1,
                prediction_context=prediction_context
            )
            
            # Confidence should have increased
            # 1.0 * 1.1 = 1.1
            assert model.learning_confidence > 1.0
            assert np.isclose(model.learning_confidence, 1.1)
            
            # Now test decay with large error (> ERROR_THRESHOLD_HIGH = 1.0)
            model.learning_confidence = 1.0
            # Fill history with dummy data to ensure length check passes
            model.prediction_history = [
                {"error": 0.1, "timestamp": "2023-01-01T00:00:00", "context": {}, "abs_error": 0.1}
            ] * 10
            
            # The error will be calculated as actual - predicted = 21.5 - 20.0 = 1.5
            # 1.5 > ERROR_THRESHOLD_HIGH (1.0) -> Decay
            model.update_prediction_feedback(
                predicted_temp=20.0,
                actual_temp=21.5,
                prediction_context=prediction_context
            )
            
            # Confidence should have decreased
            # 1.0 * 0.9 = 0.9
            assert model.learning_confidence < 1.0
            assert np.isclose(model.learning_confidence, 0.9)
            
        finally:
            # Restore method
            model._adapt_parameters_from_recent_errors = original_adapt
