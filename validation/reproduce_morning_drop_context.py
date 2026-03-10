import logging
import sys
import os

# Add the project root to sys.path so we can import from src
sys.path.append(os.getcwd())

from src.prediction_context import UnifiedPredictionContext
from src import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def test_morning_drop_context():
    # Override config for testing
    config.CYCLE_INTERVAL_MINUTES = 30
    
    print(f"Testing Morning Drop Context with Cycle Interval: {config.CYCLE_INTERVAL_MINUTES} minutes")
    
    # Simulate morning conditions:
    # Current time: 7:00 AM
    # Current Outdoor: 0.0°C
    # Forecast 1h (8:00 AM): 4.0°C (Sun rising)
    # PV: 0W -> 100W
    
    current_outdoor = 0.0
    current_pv = 0.0
    
    features = {
        'temp_forecast_1h': 4.0,
        'temp_forecast_2h': 6.0,
        'temp_forecast_3h': 8.0,
        'temp_forecast_4h': 10.0,
        'pv_forecast_1h': 100.0,
        'pv_forecast_2h': 500.0,
        'pv_forecast_3h': 1000.0,
        'pv_forecast_4h': 1500.0
    }
    
    thermal_features = {
        'fireplace_on': 0.0,
        'tv_on': 0.0
    }
    
    # Create context
    context = UnifiedPredictionContext.create_prediction_context(
        features=features,
        outdoor_temp=current_outdoor,
        pv_power=current_pv,
        thermal_features=thermal_features
    )
    
    print("\n--- Context Result ---")
    print(f"Current Outdoor: {context['current_outdoor']}°C")
    print(f"Forecast 1h: {features['temp_forecast_1h']}°C")
    print(f"Effective Outdoor (avg_outdoor): {context['avg_outdoor']}°C")
    
    # Calculate what we expect for a 30-minute cycle
    # We want the average temperature over the next 30 minutes.
    # If temp rises linearly from 0 to 4 over 60 mins:
    # T(t) = 0 + (4/60)*t
    # Average over [0, 30] is T(15) = 0 + (4/60)*15 = 1.0°C
    
    expected_avg = 1.0
    actual_avg = context['avg_outdoor']
    
    print(f"\nExpected Average (Linear Interpolation @ 15min): {expected_avg}°C")
    print(f"Actual Average: {actual_avg}°C")
    
    if abs(actual_avg - expected_avg) > 0.1:
        print("\n❌ FAIL: Significant discrepancy detected!")
        print("The system is over-estimating the outdoor temperature rise.")
        print(f"Over-estimation: {actual_avg - expected_avg:.2f}°C")
    else:
        print("\n✅ PASS: Calculation matches expectations.")

if __name__ == "__main__":
    test_morning_drop_context()
