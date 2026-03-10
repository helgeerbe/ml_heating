
import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from src.prediction_context import UnifiedPredictionContext
from src.model_wrapper import EnhancedModelWrapper
from src import config

def reproduce_morning_drop_v2():
    print("=== Reproducing Morning Drop Scenario V2 ===")
    
    # 1. Setup Scenario
    # Morning transition: Cold but rising outdoor temp, Sun coming up
    current_outdoor = 5.0
    forecast_1h_outdoor = 10.0 # Rapid rise expected
    
    current_pv = 0.0
    forecast_1h_pv = 2000.0 # Strong Sun expected
    
    target_temp = 21.0
    current_indoor = 20.5 # Below target, needs heating
    
    # Mock features
    features = {
        "temp_forecast_1h": forecast_1h_outdoor,
        "temp_forecast_2h": forecast_1h_outdoor + 2,
        "temp_forecast_3h": forecast_1h_outdoor + 3,
        "temp_forecast_4h": forecast_1h_outdoor + 4,
        "pv_forecast_1h": forecast_1h_pv,
        "pv_forecast_2h": forecast_1h_pv,
        "pv_forecast_3h": forecast_1h_pv,
        "pv_forecast_4h": forecast_1h_pv,
        "indoor_temp_lag_30m": current_indoor,
        "target_temp": target_temp,
        "outdoor_temp": current_outdoor,
        "pv_now": current_pv,
        "fireplace_on": 0,
        "tv_on": 0,
        "thermal_power_kw": 0,
        "indoor_temp_gradient": 0,
        "temp_diff_indoor_outdoor": target_temp - current_outdoor,
        "outlet_indoor_diff": 0,
        # Crucially: NO PV HISTORY (simulating morning wakeup or restart)
        "pv_power_history": None 
    }
    
    thermal_features = {
        "fireplace_on": 0.0,
        "tv_on": 0.0,
        "pv_power": current_pv,
        "pv_power_history": None
    }

    # 2. Test Prediction Context Weighting
    print(f"\n--- Checking Prediction Context Weighting (Cycle: {config.CYCLE_INTERVAL_MINUTES} min) ---")
    
    context = UnifiedPredictionContext.create_prediction_context(
        features=features,
        outdoor_temp=current_outdoor,
        pv_power=current_pv,
        thermal_features=thermal_features,
        target_temp=target_temp,
        current_temp=current_indoor
    )
    
    print(f"Current PV: {current_pv}W")
    print(f"Forecast 1h PV: {forecast_1h_pv}W")
    print(f"Calculated Avg PV: {context['avg_pv']:.4f}W")
    
    # 3. Test Model Wrapper Output
    print(f"\n--- Checking Model Wrapper Output ---")
    wrapper = EnhancedModelWrapper()
    
    # Force parameters to make system sensitive to PV
    wrapper.thermal_model.heat_loss_coefficient = 1.0
    wrapper.thermal_model.outlet_effectiveness = 0.5
    wrapper.thermal_model.pv_heat_weight = 0.005 # Stronger PV effect
    wrapper.thermal_model.solar_lag_minutes = 60 # Significant lag
    
    print(f"PV Heat Weight: {wrapper.thermal_model.pv_heat_weight}")
    
    # Calculate optimal outlet temp
    outlet_temp, meta = wrapper.calculate_optimal_outlet_temp(features)
    
    print(f"Optimal Outlet Temp: {outlet_temp:.2f}°C")
    
    # Compare with what it would be if we used current conditions only
    features_current_only = features.copy()
    features_current_only["temp_forecast_1h"] = current_outdoor
    features_current_only["pv_forecast_1h"] = current_pv
    features_current_only["pv_forecast_2h"] = current_pv
    features_current_only["pv_forecast_3h"] = current_pv
    features_current_only["pv_forecast_4h"] = current_pv
    
    outlet_temp_current, _ = wrapper.calculate_optimal_outlet_temp(features_current_only)
    print(f"Outlet Temp (Current Conditions Only): {outlet_temp_current:.2f}°C")
    
    drop = outlet_temp_current - outlet_temp
    print(f"Morning Drop Magnitude: {drop:.2f}°C")
    
    # We expect a significant drop because the model (incorrectly) thinks 
    # the "Avg PV" (500W) has been active for the last 3 hours (history init),
    # so it thinks the house is warmer/gaining heat than it actually is.
    
    if drop > 2.0:
        print("⚠️  FAILURE REPRODUCED: Significant morning drop detected!")
        print("Explanation: Model is using blended Avg PV as history initialization, bypassing solar lag.")
    else:
        print("✅ Morning drop is contained (or not reproduced).")

if __name__ == "__main__":
    reproduce_morning_drop_v2()
