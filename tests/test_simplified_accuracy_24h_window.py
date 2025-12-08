"""
Unit tests for simplified 3-category accuracy model with 24-hour moving window.

TDD approach: These tests define the specification before implementation.
Tests should FAIL initially with current 5-category all-time implementation.
"""

import pytest
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from prediction_metrics import PredictionMetrics


class TestSimplified3CategoryAccuracy:
    """Test the simplified 3-category accuracy model: Perfect/Tolerable/Poor."""
    
    def test_perfect_accuracy_zero_error(self):
        """Test perfect accuracy with exactly 0.0°C error."""
        metrics = PredictionMetrics()
        
        # Add perfect predictions
        metrics.add_prediction(21.0, 21.0)  # 0.0°C error
        metrics.add_prediction(20.5, 20.5)  # 0.0°C error
        
        breakdown = metrics.get_simplified_accuracy_breakdown()
        
        assert breakdown['perfect']['percentage'] == 100.0
        assert breakdown['perfect']['count'] == 2
        assert breakdown['tolerable']['percentage'] == 0.0
        assert breakdown['poor']['percentage'] == 0.0
    
    def test_tolerable_accuracy_within_point_one(self):
        """Test tolerable accuracy with ≤0.1°C error."""
        metrics = PredictionMetrics()
        
        # Add tolerable predictions
        metrics.add_prediction(21.0, 21.1)  # 0.1°C error - exactly at boundary
        metrics.add_prediction(21.0, 20.9)  # 0.1°C error - exactly at boundary
        metrics.add_prediction(21.0, 21.05) # 0.05°C error - within tolerance
        
        breakdown = metrics.get_simplified_accuracy_breakdown()
        
        assert breakdown['tolerable']['percentage'] == 100.0
        assert breakdown['tolerable']['count'] == 3
        assert breakdown['perfect']['percentage'] == 0.0
        assert breakdown['poor']['percentage'] == 0.0
    
    def test_poor_accuracy_above_point_two(self):
        """Test poor accuracy with >0.2°C error."""
        metrics = PredictionMetrics()
        
        # Add poor predictions
        metrics.add_prediction(21.0, 21.3)  # 0.3°C error
        metrics.add_prediction(21.0, 20.7)  # 0.3°C error
        metrics.add_prediction(21.0, 21.5)  # 0.5°C error
        
        breakdown = metrics.get_simplified_accuracy_breakdown()
        
        assert breakdown['poor']['percentage'] == 100.0
        assert breakdown['poor']['count'] == 3
        assert breakdown['perfect']['percentage'] == 0.0
        assert breakdown['tolerable']['percentage'] == 0.0
    
    def test_mixed_accuracy_distribution(self):
        """Test mixed predictions across all 3 categories."""
        metrics = PredictionMetrics()
        
        # Add mixed predictions (10 total)
        # Perfect: 2 predictions (20%)
        metrics.add_prediction(21.0, 21.0)  # 0.0°C
        metrics.add_prediction(20.5, 20.5)  # 0.0°C
        
        # Tolerable: 3 predictions (30%) 
        metrics.add_prediction(21.0, 21.1)  # 0.1°C
        metrics.add_prediction(21.0, 20.9)  # 0.1°C
        metrics.add_prediction(21.0, 21.05) # 0.05°C
        
        # Poor: 5 predictions (50%)
        metrics.add_prediction(21.0, 21.3)  # 0.3°C
        metrics.add_prediction(21.0, 20.7)  # 0.3°C
        metrics.add_prediction(21.0, 21.5)  # 0.5°C
        metrics.add_prediction(21.0, 20.5)  # 0.5°C
        metrics.add_prediction(21.0, 22.0)  # 1.0°C
        
        breakdown = metrics.get_simplified_accuracy_breakdown()
        
        assert breakdown['perfect']['percentage'] == 20.0
        assert breakdown['perfect']['count'] == 2
        assert breakdown['tolerable']['percentage'] == 30.0
        assert breakdown['tolerable']['count'] == 3
        assert breakdown['poor']['percentage'] == 50.0
        assert breakdown['poor']['count'] == 5
    
    def test_edge_case_exactly_point_one_error(self):
        """Test edge case: exactly 0.1°C error should be tolerable."""
        metrics = PredictionMetrics()
        
        metrics.add_prediction(21.0, 21.1)  # Exactly 0.1°C error
        
        breakdown = metrics.get_simplified_accuracy_breakdown()
        
        assert breakdown['tolerable']['count'] == 1
        assert breakdown['perfect']['count'] == 0
        assert breakdown['poor']['count'] == 0
    
    def test_edge_case_exactly_point_two_error(self):
        """Test edge case: exactly 0.2°C error should be poor."""
        metrics = PredictionMetrics()
        
        metrics.add_prediction(21.0, 21.2)  # Exactly 0.2°C error
        
        breakdown = metrics.get_simplified_accuracy_breakdown()
        
        assert breakdown['poor']['count'] == 1
        assert breakdown['perfect']['count'] == 0
        assert breakdown['tolerable']['count'] == 0
    
    def test_good_control_percentage_calculation(self):
        """Test 'good control' percentage = perfect + tolerable."""
        metrics = PredictionMetrics()
        
        # Add test data: 2 perfect, 3 tolerable, 5 poor = 10 total
        # Good control = (2 + 3) / 10 = 50%
        for _ in range(2):
            metrics.add_prediction(21.0, 21.0)  # Perfect
        for _ in range(3):
            metrics.add_prediction(21.0, 21.1)  # Tolerable
        for _ in range(5):
            metrics.add_prediction(21.0, 21.3)  # Poor
        
        good_control_pct = metrics.get_good_control_percentage()
        
        assert good_control_pct == 50.0


class Test24HourMovingWindow:
    """Test 24-hour moving window functionality."""
    
    def test_24h_window_with_full_data(self):
        """Test 24h window with exactly 288 predictions (24h worth)."""
        metrics = PredictionMetrics()
        
        # Add 288 predictions (24h × 12 predictions/hour)
        base_time = datetime.now()
        for i in range(288):
            timestamp = (base_time - timedelta(minutes=i*5)).isoformat()
            # Make older predictions worse, recent ones better
            if i < 144:  # Last 12 hours - good
                metrics.add_prediction(21.0, 21.0, timestamp=timestamp)  # Perfect
            else:  # Older 12 hours - poor
                metrics.add_prediction(21.0, 21.5, timestamp=timestamp)  # Poor
        
        # Get 24h window accuracy (should only include recent good predictions)
        accuracy_24h = metrics.get_24h_accuracy_breakdown()
        
        # Should have 144 perfect and 144 poor predictions within 24h window
        assert accuracy_24h['perfect']['percentage'] == 50.0
        assert accuracy_24h['perfect']['count'] == 144
        assert accuracy_24h['tolerable']['percentage'] == 0.0
        assert accuracy_24h['poor']['percentage'] == 50.0
        assert accuracy_24h['poor']['count'] == 144
    
    def test_24h_window_with_partial_data(self):
        """Test 24h window with less than 24h of data."""
        metrics = PredictionMetrics()
        
        # Add only 50 predictions (about 4 hours worth)
        for i in range(50):
            metrics.add_prediction(21.0, 21.0)  # All perfect
        
        accuracy_24h = metrics.get_24h_accuracy_breakdown()
        
        # Should use all available predictions
        assert accuracy_24h['perfect']['count'] == 50
        assert accuracy_24h['perfect']['percentage'] == 100.0
    
    def test_24h_window_excludes_old_data(self):
        """Test that 24h window excludes predictions older than 24h."""
        metrics = PredictionMetrics()
        
        # Add old predictions (25 hours ago) - these should be excluded
        old_time = datetime.now() - timedelta(hours=25)
        for i in range(10):
            timestamp = (old_time + timedelta(minutes=i*5)).isoformat()
            metrics.add_prediction(21.0, 21.5, timestamp=timestamp)  # Poor
        
        # Add recent predictions (last hour) - these should be included
        recent_time = datetime.now() - timedelta(minutes=30)
        for i in range(10):
            timestamp = (recent_time + timedelta(minutes=i*3)).isoformat()
            metrics.add_prediction(21.0, 21.0, timestamp=timestamp)  # Perfect
        
        accuracy_24h = metrics.get_24h_accuracy_breakdown()
        
        # Should only include the 10 recent perfect predictions
        assert accuracy_24h['perfect']['count'] == 10
        assert accuracy_24h['poor']['count'] == 0
        assert accuracy_24h['perfect']['percentage'] == 100.0
    
    def test_24h_good_control_percentage(self):
        """Test 24h good control percentage calculation."""
        metrics = PredictionMetrics()
        
        # Add test data within 24h window
        recent_time = datetime.now() - timedelta(minutes=30)
        
        # 20 perfect predictions
        for i in range(20):
            timestamp = (recent_time + timedelta(minutes=i)).isoformat()
            metrics.add_prediction(21.0, 21.0, timestamp=timestamp)  # Perfect
        
        # 30 tolerable predictions  
        for i in range(30):
            timestamp = (recent_time + timedelta(minutes=20+i)).isoformat()
            metrics.add_prediction(21.0, 21.1, timestamp=timestamp)  # Tolerable
        
        # 50 poor predictions
        for i in range(50):
            timestamp = (recent_time + timedelta(minutes=50+i)).isoformat()
            metrics.add_prediction(21.0, 21.3, timestamp=timestamp)  # Poor
        
        # Good control = (20 + 30) / 100 = 50%
        good_control_24h = metrics.get_24h_good_control_percentage()
        
        assert good_control_24h == 50.0


class TestModelWrapperIntegration:
    """Test integration with model wrapper for HA sensor export."""
    
    def test_ha_metrics_use_24h_window(self):
        """Test that HA metrics use 24h window instead of all-time."""
        # Mock the model wrapper to avoid complex dependencies
        from model_wrapper import EnhancedModelWrapper
        
        # This test should verify that get_comprehensive_metrics_for_ha()
        # returns 24h window accuracy instead of all-time accuracy
        # Will need to be implemented after the main functionality
        pass
    
    def test_simplified_accuracy_in_ha_export(self):
        """Test that HA sensor exports simplified 3-category accuracy."""
        # This test should verify the HA sensor shows:
        # - perfect_accuracy_pct 
        # - tolerable_accuracy_pct
        # - poor_accuracy_pct
        # - good_control_pct (perfect + tolerable)
        pass


class TestBackwardsCompatibility:
    """Test that existing functionality still works."""
    
    def test_old_get_metrics_still_works(self):
        """Test that existing get_metrics() method still works."""
        metrics = PredictionMetrics()
        
        # Add some test data
        metrics.add_prediction(21.0, 21.0)
        metrics.add_prediction(21.0, 21.1)
        
        # Should not break existing functionality
        old_metrics = metrics.get_metrics()
        
        # Should still have the old keys
        assert 'accuracy_breakdown' in old_metrics
        assert '1h' in old_metrics
        assert '24h' in old_metrics
        assert 'all' in old_metrics
    
    def test_mae_rmse_calculations_unchanged(self):
        """Test that MAE/RMSE calculations are not affected."""
        metrics = PredictionMetrics()
        
        # Add test data with known errors
        metrics.add_prediction(21.0, 21.1)  # 0.1°C error
        metrics.add_prediction(21.0, 20.9)  # 0.1°C error
        
        results = metrics.get_metrics()
        
        # MAE should be 0.1°C
        assert abs(results['all']['mae'] - 0.1) < 0.001
        
        # RMSE should be 0.1°C (since all errors are the same)
        assert abs(results['all']['rmse'] - 0.1) < 0.001


# Test helper functions
def create_test_predictions_with_errors(errors, base_predicted=21.0):
    """Helper function to create test predictions with specific errors."""
    predictions = []
    for error in errors:
        actual = base_predicted + error
        predictions.append((base_predicted, actual))
    return predictions


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
