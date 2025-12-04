"""
Prediction Metrics Tracking System

This module provides comprehensive MAE/RMSE tracking for the adaptive learning system
with rolling window calculations and state persistence.
"""

import numpy as np
import logging
import json
from collections import deque
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class PredictionMetrics:
    """
    Rolling window prediction accuracy metrics tracker.
    
    Tracks MAE (Mean Absolute Error) and RMSE (Root Mean Square Error)
    over multiple time windows for comprehensive accuracy analysis.
    """
    
    def __init__(self, max_history_size: int = 1000):
        """
        Initialize prediction metrics tracker.
        
        Args:
            max_history_size: Maximum number of predictions to keep in memory
        """
        self.max_history_size = max_history_size
        
        # Prediction history storage
        self.predictions = deque(maxlen=max_history_size)
        
        # Time window configurations (in number of predictions)
        self.windows = {
            '1h': 12,    # 12 predictions = 1 hour (5min intervals)
            '6h': 72,    # 72 predictions = 6 hours  
            '24h': 288,  # 288 predictions = 24 hours
            'all': None  # All available data
        }
        
        # Cached metrics for efficiency
        self._cached_metrics = {}
        self._cache_timestamp = None
        self._cache_valid_duration = 60  # Cache valid for 60 seconds
        
        # Persistent state for reloading
        self.state_file = None
        
    def add_prediction(self, predicted: float, actual: float, 
                      context: Dict = None, timestamp: str = None):
        """
        Add a new prediction result for tracking.
        
        Args:
            predicted: Predicted temperature value
            actual: Actual measured temperature value
            context: Context information (outlet_temp, outdoor_temp, etc.)
            timestamp: Prediction timestamp (ISO format)
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
            
        prediction_record = {
            'timestamp': timestamp,
            'predicted': float(predicted),
            'actual': float(actual),
            'error': float(actual - predicted),
            'abs_error': abs(float(actual - predicted)),
            'squared_error': (float(actual - predicted)) ** 2,
            'context': context or {}
        }
        
        self.predictions.append(prediction_record)
        
        # Invalidate cache
        self._cache_timestamp = None
        
        logging.debug(f"Added prediction: pred={predicted:.2f}, "
                     f"actual={actual:.2f}, error={prediction_record['error']:.3f}")
    
    def get_metrics(self, refresh_cache: bool = False) -> Dict:
        """
        Get comprehensive prediction metrics for all time windows.
        
        Args:
            refresh_cache: Force refresh of cached metrics
            
        Returns:
            Dict with MAE, RMSE, and other metrics for each time window
        """
        # Check cache validity
        now = datetime.now()
        if (self._cache_timestamp and 
            not refresh_cache and
            (now - self._cache_timestamp).total_seconds() < self._cache_valid_duration):
            return self._cached_metrics
        
        metrics = {}
        
        for window_name, window_size in self.windows.items():
            window_metrics = self._calculate_window_metrics(window_size)
            metrics[window_name] = window_metrics
        
        # Add trend analysis
        metrics['trends'] = self._calculate_trends()
        
        # Add accuracy categories
        metrics['accuracy_breakdown'] = self._calculate_accuracy_breakdown()
        
        # Cache results
        self._cached_metrics = metrics
        self._cache_timestamp = now
        
        return metrics
    
    def _calculate_window_metrics(self, window_size: Optional[int]) -> Dict:
        """Calculate metrics for a specific time window."""
        if not self.predictions:
            return {
                'mae': 0.0,
                'rmse': 0.0,
                'count': 0,
                'mean_error': 0.0,
                'std_error': 0.0,
                'min_error': 0.0,
                'max_error': 0.0
            }
        
        # Get predictions for this window
        if window_size is None:
            window_predictions = list(self.predictions)
        else:
            window_predictions = list(self.predictions)[-window_size:]
        
        if not window_predictions:
            return {
                'mae': 0.0,
                'rmse': 0.0,
                'count': 0,
                'mean_error': 0.0,
                'std_error': 0.0,
                'min_error': 0.0,
                'max_error': 0.0
            }
        
        # Extract errors
        errors = [p['error'] for p in window_predictions]
        abs_errors = [p['abs_error'] for p in window_predictions]
        squared_errors = [p['squared_error'] for p in window_predictions]
        
        # Calculate metrics
        mae = np.mean(abs_errors)
        rmse = np.sqrt(np.mean(squared_errors))
        mean_error = np.mean(errors)  # Bias
        std_error = np.std(errors)
        min_error = np.min(errors)
        max_error = np.max(errors)
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'count': len(window_predictions),
            'mean_error': float(mean_error),  # Bias
            'std_error': float(std_error),
            'min_error': float(min_error),
            'max_error': float(max_error)
        }
    
    def _calculate_trends(self) -> Dict:
        """Calculate accuracy trends over time."""
        if len(self.predictions) < 20:
            return {'insufficient_data': True}
        
        # Split into two halves for trend analysis
        half_point = len(self.predictions) // 2
        first_half = list(self.predictions)[:half_point]
        second_half = list(self.predictions)[half_point:]
        
        # Safely calculate MAE with empty slice protection
        if not first_half or not second_half:
            return {'insufficient_data': True}
            
        first_half_errors = [p['abs_error'] for p in first_half]
        second_half_errors = [p['abs_error'] for p in second_half]
        
        if not first_half_errors or not second_half_errors:
            return {'insufficient_data': True}
            
        first_half_mae = np.mean(first_half_errors)
        second_half_mae = np.mean(second_half_errors)
        
        improvement = first_half_mae - second_half_mae
        improvement_percentage = (improvement / first_half_mae) * 100 if first_half_mae > 0 else 0
        
        return {
            'mae_improvement': float(improvement),
            'mae_improvement_percentage': float(improvement_percentage),
            'first_half_mae': float(first_half_mae),
            'second_half_mae': float(second_half_mae),
            'is_improving': improvement > 0
        }
    
    def _calculate_accuracy_breakdown(self) -> Dict:
        """Calculate accuracy breakdown by error ranges."""
        if not self.predictions:
            return {}
        
        abs_errors = [p['abs_error'] for p in self.predictions]
        total_predictions = len(abs_errors)
        
        # Define accuracy categories
        categories = {
            'excellent': 0.1,   # ±0.1°C
            'very_good': 0.2,   # ±0.2°C
            'good': 0.5,        # ±0.5°C
            'acceptable': 1.0,  # ±1.0°C
            'poor': float('inf')
        }
        
        breakdown = {}
        remaining_count = total_predictions
        
        for category, threshold in categories.items():
            if category == 'poor':
                count = remaining_count
            else:
                count = sum(1 for error in abs_errors if error <= threshold)
                abs_errors = [e for e in abs_errors if e > threshold]
                remaining_count -= count
            
            percentage = (count / total_predictions) * 100 if total_predictions > 0 else 0
            breakdown[category] = {
                'count': count,
                'percentage': float(percentage)
            }
        
        return breakdown
    
    def get_recent_performance(self, last_n: int = 10) -> Dict:
        """Get performance metrics for the most recent N predictions."""
        if len(self.predictions) < last_n:
            last_n = len(self.predictions)
        
        recent_predictions = list(self.predictions)[-last_n:]
        
        if not recent_predictions:
            return {'no_data': True}
        
        recent_errors = [p['abs_error'] for p in recent_predictions]
        
        return {
            'count': len(recent_predictions),
            'mae': float(np.mean(recent_errors)),
            'max_error': float(np.max(recent_errors)),
            'min_error': float(np.min(recent_errors)),
            'std_error': float(np.std(recent_errors)) if len(recent_errors) > 1 else 0.0
        }
    
    def save_state(self, filepath: str):
        """Save prediction history to file for persistence."""
        try:
            # Convert deque to list for JSON serialization
            state_data = {
                'predictions': list(self.predictions),
                'max_history_size': self.max_history_size,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            self.state_file = filepath
            logging.info(f"Prediction metrics saved to {filepath}")
            
        except Exception as e:
            logging.error(f"Failed to save prediction metrics: {e}")
    
    def load_state(self, filepath: str) -> bool:
        """Load prediction history from file."""
        try:
            with open(filepath, 'r') as f:
                state_data = json.load(f)
            
            # Restore predictions
            self.predictions = deque(
                state_data['predictions'], 
                maxlen=state_data.get('max_history_size', self.max_history_size)
            )
            
            self.state_file = filepath
            logging.info(f"Loaded {len(self.predictions)} predictions from {filepath}")
            
            # Invalidate cache
            self._cache_timestamp = None
            
            return True
            
        except FileNotFoundError:
            logging.info("ℹ️  No existing prediction metrics file found - starting fresh")
            return False
        except Exception as e:
            logging.warning(f"Failed to load prediction metrics: {e}")
            return False
    
    def get_summary(self) -> str:
        """Get a human-readable summary of current metrics."""
        metrics = self.get_metrics()
        
        if not self.predictions:
            return "No prediction data available"
        
        recent_mae = metrics['1h']['mae']
        all_time_mae = metrics['all']['mae']
        total_predictions = len(self.predictions)
        
        # Get accuracy breakdown
        breakdown = metrics['accuracy_breakdown']
        excellent_pct = breakdown.get('excellent', {}).get('percentage', 0)
        good_pct = (breakdown.get('excellent', {}).get('percentage', 0) + 
                   breakdown.get('very_good', {}).get('percentage', 0) +
                   breakdown.get('good', {}).get('percentage', 0))
        
        # Get trend
        trends = metrics['trends']
        trend_str = ""
        if not trends.get('insufficient_data', False):
            if trends['is_improving']:
                trend_str = f" (improving by {trends['mae_improvement_percentage']:.1f}%)"
            else:
                trend_str = f" (degrading by {abs(trends['mae_improvement_percentage']):.1f}%)"
        
        summary = (
            f"Prediction Accuracy Summary:\n"
            f"  Total predictions: {total_predictions}\n"
            f"  Recent MAE (1h): {recent_mae:.3f}°C\n"
            f"  All-time MAE: {all_time_mae:.3f}°C{trend_str}\n"
            f"  Excellent accuracy (±0.1°C): {excellent_pct:.1f}%\n"
            f"  Good+ accuracy (±0.5°C): {good_pct:.1f}%"
        )
        
        return summary


# Global instance for easy access
prediction_tracker = PredictionMetrics()


def track_prediction(predicted: float, actual: float, 
                    context: Dict = None, timestamp: str = None):
    """
    Convenience function to track a prediction globally.
    
    Args:
        predicted: Predicted temperature value
        actual: Actual measured temperature value  
        context: Context information
        timestamp: Prediction timestamp
    """
    prediction_tracker.add_prediction(predicted, actual, context, timestamp)


def get_current_metrics() -> Dict:
    """Get current prediction metrics from global tracker."""
    return prediction_tracker.get_metrics()


def get_metrics_summary() -> str:
    """Get human-readable metrics summary from global tracker."""
    return prediction_tracker.get_summary()


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Prediction Metrics System")
    
    # Create test tracker
    metrics = PredictionMetrics()
    
    # Add some test predictions
    import random
    np.random.seed(42)
    
    for i in range(50):
        # Simulate predictions with improving accuracy over time
        base_error = 1.0 - (i / 100)  # Error decreases over time
        predicted = 20.5
        actual = predicted + np.random.normal(0, base_error)
        
        context = {
            'outlet_temp': 45.0,
            'outdoor_temp': 5.0,
            'step': i
        }
        
        metrics.add_prediction(predicted, actual, context)
    
    # Test metrics calculation
    results = metrics.get_metrics()
    print(f"\n📊 Test Results:")
    print(f"  1h MAE: {results['1h']['mae']:.3f}°C")
    print(f"  24h MAE: {results['24h']['mae']:.3f}°C")
    print(f"  All-time MAE: {results['all']['mae']:.3f}°C")
    
    # Test accuracy breakdown
    breakdown = results['accuracy_breakdown']
    print(f"\n📈 Accuracy Breakdown:")
    for category, data in breakdown.items():
        print(f"  {category}: {data['percentage']:.1f}% ({data['count']} predictions)")
    
    # Test trends
    trends = results['trends']
    if not trends.get('insufficient_data'):
        print(f"\n📉 Trend Analysis:")
        print(f"  Improving: {trends['is_improving']}")
        print(f"  MAE change: {trends['mae_improvement']:+.3f}°C")
        print(f"  Improvement: {trends['mae_improvement_percentage']:+.1f}%")
    
    print(f"\n📋 Summary:")
    print(metrics.get_summary())
    
    print(f"\n✅ Prediction metrics system ready!")
