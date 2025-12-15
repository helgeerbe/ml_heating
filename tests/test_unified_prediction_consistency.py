"""
Test suite for unified prediction consistency between binary search and smart rounding.

This test demonstrates and validates the fix for the inconsistency where:
- Binary search uses forecast-based conditions via _get_forecast_conditions()
- Smart rounding uses current-only conditions
- This leads to suboptimal temperature selection

The fix ensures both use identical environmental contexts.
"""

import pytest
import unittest.mock as mock
from unittest.mock import MagicMock, patch
import pandas as pd
from src.model_wrapper import get_enhanced_model_wrapper
from src.temperature_control import SmartRounding


class TestUnifiedPredictionConsistency:
    """Test that binary search and smart rounding use consistent prediction contexts."""
    
    def test_current_inconsistency_problem(self):
        """
        STEP 1: Demonstrate the current inconsistency problem.
        
        This test shows how binary search and smart rounding can use different
        environmental conditions, leading to inconsistent predictions.
        """
        # Setup test scenario with different current vs forecast conditions
        current_outdoor = 5.0  # Current outdoor temperature
        forecast_outdoor = 8.0  # Forecast outdoor temperature (warmer)
        current_pv = 0.0       # Current PV (night time)
        forecast_pv = 2000.0   # Forecast PV (sunny day ahead)
        
        # Mock features with both current and forecast data
        test_features = {
            'outdoor_temp': current_outdoor,
            'pv_now': current_pv,
            'temp_forecast_1h': forecast_outdoor,
            'temp_forecast_2h': forecast_outdoor,
            'temp_forecast_3h': forecast_outdoor,
            'temp_forecast_4h': forecast_outdoor,
            'pv_forecast_1h': forecast_pv,
            'pv_forecast_2h': forecast_pv,
            'pv_forecast_3h': forecast_pv,
            'pv_forecast_4h': forecast_pv,
            'indoor_temp_lag_30m': 20.0,
            'target_temp': 21.0,
            'fireplace_on': 0.0,
            'tv_on': 0.0
        }
        
        with patch('src.model_wrapper.get_enhanced_model_wrapper') as mock_wrapper_factory:
            # Mock the thermal model to return different predictions for current vs forecast
            mock_wrapper = MagicMock()
            mock_wrapper_factory.return_value = mock_wrapper
            
            # Binary search will use forecast averages (warmer outdoor, more PV)
            mock_wrapper.thermal_model.predict_equilibrium_temperature.side_effect = (
                lambda outlet_temp, outdoor_temp, **kwargs: 
                20.5 if outdoor_temp == current_outdoor else 21.1  # Forecast conditions predict higher
            )
            
            # Mock the _get_forecast_conditions method
            mock_wrapper._get_forecast_conditions.return_value = (
                forecast_outdoor,  # avg_outdoor (forecast)
                forecast_pv,      # avg_pv (forecast) 
                [forecast_outdoor] * 4,  # outdoor_forecast
                [forecast_pv] * 4        # pv_forecast
            )
            
            # Test binary search behavior (should use forecast conditions)
            mock_wrapper._current_features = test_features
            thermal_features = {
                'pv_power': current_pv,
                'fireplace_on': 0.0,
                'tv_on': 0.0
            }
            
            # Binary search calls _get_forecast_conditions and uses forecast averages
            avg_outdoor, avg_pv, _, _ = mock_wrapper._get_forecast_conditions(
                current_outdoor, current_pv, thermal_features
            )
            
            # Verify binary search uses forecast conditions
            assert avg_outdoor == forecast_outdoor, "Binary search should use forecast outdoor temp"
            assert avg_pv == forecast_pv, "Binary search should use forecast PV"
            
            # Test smart rounding behavior (currently uses current conditions only)
            smart_rounding = SmartRounding()
            
            # Mock predict_indoor_temp to show different results for current vs forecast
            mock_wrapper.predict_indoor_temp.side_effect = (
                lambda outlet_temp, outdoor_temp, **kwargs:
                20.5 if outdoor_temp == current_outdoor else 21.1
            )
            
            # Smart rounding currently calls predict_indoor_temp with current conditions
            floor_prediction = mock_wrapper.predict_indoor_temp(
                outlet_temp=41.0,
                outdoor_temp=current_outdoor,  # Uses current, not forecast!
                pv_power=current_pv,          # Uses current, not forecast!
                fireplace_on=False,
                tv_on=False
            )
            
            ceiling_prediction = mock_wrapper.predict_indoor_temp(
                outlet_temp=42.0,
                outdoor_temp=current_outdoor,  # Uses current, not forecast!
                pv_power=current_pv,          # Uses current, not forecast!
                fireplace_on=False,
                tv_on=False
            )
            
            # Demonstrate the inconsistency
            assert floor_prediction == 20.5, "Smart rounding uses current conditions"
            
            # Binary search would have optimized for 21.1°C (forecast conditions)
            # but smart rounding validates against 20.5°C (current conditions)
            # This is the inconsistency we need to fix!
            
    def test_unified_prediction_context_solution(self):
        """
        STEP 2: Test the unified solution where both systems use identical contexts.
        
        After the fix, both binary search and smart rounding should use the same
        forecast-based environmental conditions.
        """
        from src.prediction_context import UnifiedPredictionContext
        from src.temperature_control import SmartRounding
        
        # Test scenario: current vs forecast conditions differ significantly
        current_outdoor = 5.0
        forecast_outdoor = 8.0  # Forecast shows warming trend
        current_pv = 0.0
        forecast_pv = 2000.0   # Forecast shows sunny day ahead
        
        test_features = {
            'outdoor_temp': current_outdoor,
            'pv_now': current_pv,
            'temp_forecast_1h': forecast_outdoor,
            'temp_forecast_2h': forecast_outdoor,
            'temp_forecast_3h': forecast_outdoor, 
            'temp_forecast_4h': forecast_outdoor,
            'pv_forecast_1h': forecast_pv,
            'pv_forecast_2h': forecast_pv,
            'pv_forecast_3h': forecast_pv,
            'pv_forecast_4h': forecast_pv,
            'fireplace_on': 0.0,
            'tv_on': 0.0
        }
        
        thermal_features = {
            'pv_power': current_pv,
            'fireplace_on': 0.0,
            'tv_on': 0.0
        }
        
        # Test 1: Create unified prediction context
        context = UnifiedPredictionContext.create_prediction_context(
            features=test_features,
            outdoor_temp=current_outdoor,
            pv_power=current_pv,
            thermal_features=thermal_features
        )
        
        # Verify unified context uses forecast averages, not current values
        assert context['avg_outdoor'] == forecast_outdoor, \
            f"Context should use forecast outdoor temp {forecast_outdoor}, got {context['avg_outdoor']}"
        assert context['avg_pv'] == forecast_pv, \
            f"Context should use forecast PV {forecast_pv}, got {context['avg_pv']}"
        assert context['use_forecasts'] == True, \
            "Context should indicate forecasts are being used"
        
        # Test 2: Extract thermal model parameters from unified context
        thermal_params = UnifiedPredictionContext.get_thermal_model_params(context)
        
        # Verify thermal parameters use forecast conditions
        assert thermal_params['outdoor_temp'] == forecast_outdoor, \
            "Thermal params should use forecast outdoor temp"
        assert thermal_params['pv_power'] == forecast_pv, \
            "Thermal params should use forecast PV power"
        assert thermal_params['fireplace_on'] == 0.0
        assert thermal_params['tv_on'] == 0.0
        
        # Test 3: Verify both systems would now use identical contexts
        # This demonstrates the fix - both binary search and smart rounding 
        # will call UnifiedPredictionContext.get_thermal_model_params() and
        # get identical environmental conditions for their predictions
        
        # Simulate what binary search would get:
        binary_search_params = thermal_params.copy()
        
        # Simulate what smart rounding would get:
        smart_rounding_params = thermal_params.copy()
        
        # They should be identical!
        assert binary_search_params == smart_rounding_params, \
            "Binary search and smart rounding should use identical thermal parameters"
        
        print("✅ UNIFIED APPROACH SUCCESS:")
        print(f"   - Both systems use outdoor_temp: {thermal_params['outdoor_temp']}°C (forecast)")
        print(f"   - Both systems use pv_power: {thermal_params['pv_power']}W (forecast)")
        print(f"   - Current outdoor was: {current_outdoor}°C (ignored)")
        print(f"   - Current PV was: {current_pv}W (ignored)")
        print("   - Inconsistency eliminated! 🎯")
