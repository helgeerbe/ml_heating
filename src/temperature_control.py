"""
Temperature Control Module

This module handles temperature prediction, control logic, and smart rounding
extracted from main.py for better code organization.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional
import numpy as np

from . import config
from .ha_client import HAClient, get_sensor_attributes
from .physics_features import build_physics_features
from .model_wrapper import simplified_outlet_prediction, get_enhanced_model_wrapper
from .state_manager import save_state
from .prediction_context import prediction_context_manager


class TemperaturePredictor:
    """Handles temperature prediction using the enhanced model wrapper"""
    
    def predict_optimal_temperature(self, features: Dict, prediction_indoor_temp: float, 
                                  target_indoor_temp: float) -> Tuple[float, float, Dict]:
        """
        Predict optimal outlet temperature using the enhanced model wrapper
        
        Returns:
            Tuple of (suggested_temp, confidence, metadata)
        """
        error_target_vs_actual = target_indoor_temp - prediction_indoor_temp
        
        suggested_temp, confidence, metadata = simplified_outlet_prediction(
            features, prediction_indoor_temp, target_indoor_temp
        )
        
        # Log simplified prediction info
        logging.info(
            "Model Wrapper: temp=%.1f°C, error=%.3f°C, confidence=%.3f",
            suggested_temp, abs(error_target_vs_actual), confidence
        )
        
        return suggested_temp, confidence, metadata


class GradualTemperatureControl:
    """Handles gradual temperature changes to prevent abrupt setpoint jumps"""
    
    def apply_gradual_control(self, final_temp: float, actual_outlet_temp: Optional[float], 
                            state: Dict) -> float:
        """
        Apply gradual temperature control to prevent abrupt changes
        
        Returns:
            Clamped final temperature
        """
        if actual_outlet_temp is None:
            return final_temp
            
        max_change = config.MAX_TEMP_CHANGE_PER_CYCLE
        original_temp = final_temp
        
        last_blocking_reasons = state.get("last_blocking_reasons", []) or []
        last_final_temp = state.get("last_final_temp")
        
        # DHW-like blockers that should keep the soft-start behavior
        dhw_like_blockers = {
            config.DHW_STATUS_ENTITY_ID,
            config.DISINFECTION_STATUS_ENTITY_ID,
            config.DHW_BOOST_HEATER_STATUS_ENTITY_ID,
        }
        
        # Determine baseline temperature
        if last_final_temp is not None:
            baseline = last_final_temp
            if any(b in dhw_like_blockers for b in last_blocking_reasons):
                baseline = actual_outlet_temp
        else:
            baseline = actual_outlet_temp
            
        # Apply gradual control
        delta = final_temp - baseline
        if abs(delta) > max_change:
            final_temp = baseline + np.clip(delta, -max_change, max_change)
            logging.info("--- Gradual Temperature Control ---")
            logging.info(
                "Change from baseline %.1f°C to suggested %.1f°C exceeds"
                " max change of %.1f°C. Capping at %.1f°C.",
                baseline, original_temp, max_change, final_temp,
            )
            
        return final_temp


class SmartRounding:
    """Handles smart temperature rounding using thermal model predictions"""
    
    def apply_smart_rounding(self, final_temp: float, outdoor_temp: float, 
                           features: Dict, fireplace_on: bool, 
                           target_indoor_temp: float) -> int:
        """
        Apply smart rounding by testing floor vs ceiling temperatures.
        
        UNIFIED APPROACH: Uses the same forecast-based prediction context
        as binary search to ensure consistency.
        
        Returns:
            Smart rounded temperature as integer
        """
        floor_temp = np.floor(final_temp)
        ceiling_temp = np.ceil(final_temp)
        
        if floor_temp == ceiling_temp:
            # Already an integer
            logging.debug(f"Smart rounding: {final_temp:.2f}°C is already integer")
            return int(final_temp)
            
        try:
            wrapper = get_enhanced_model_wrapper()
            
            # UNIFIED: Create prediction context using the same method as binary search
            prediction_context_manager.set_features(features)
            
            # Extract thermal features
            thermal_features = {
                'pv_power': features.get('pv_now', 0.0) if hasattr(features, 'get') else 0.0,
                'fireplace_on': float(fireplace_on),
                'tv_on': features.get('tv_on', 0.0) if hasattr(features, 'get') else 0.0
            }
            
            # Create unified context (same as binary search uses)
            unified_context = prediction_context_manager.create_context(
                outdoor_temp=outdoor_temp,
                pv_power=thermal_features['pv_power'],
                thermal_features=thermal_features
            )
            
            # Get thermal model parameters from unified context
            thermal_params = prediction_context_manager.get_thermal_model_params()
            
            # Test floor temperature using UNIFIED context
            floor_predicted = wrapper.predict_indoor_temp(
                outlet_temp=floor_temp,
                outdoor_temp=thermal_params['outdoor_temp'],  # Uses forecast average
                pv_power=thermal_params['pv_power'],         # Uses forecast average
                fireplace_on=thermal_params['fireplace_on'],
                tv_on=thermal_params['tv_on']
            )
            
            # Test ceiling temperature using UNIFIED context
            ceiling_predicted = wrapper.predict_indoor_temp(
                outlet_temp=ceiling_temp,
                outdoor_temp=thermal_params['outdoor_temp'],  # Uses forecast average
                pv_power=thermal_params['pv_power'],         # Uses forecast average
                fireplace_on=thermal_params['fireplace_on'],
                tv_on=thermal_params['tv_on']
            )
            
            # Handle None returns from predict_indoor_temp
            if floor_predicted is None or ceiling_predicted is None:
                logging.warning("Smart rounding: predict_indoor_temp returned None, using fallback")
                return round(final_temp)
                
            # Calculate errors from target
            floor_error = abs(floor_predicted - target_indoor_temp)
            ceiling_error = abs(ceiling_predicted - target_indoor_temp)
            
            if floor_error <= ceiling_error:
                smart_rounded_temp = int(floor_temp)
                chosen = "floor"
            else:
                smart_rounded_temp = int(ceiling_temp)
                chosen = "ceiling"
                
            logging.info(
                f"Smart rounding: {final_temp:.2f}°C → {smart_rounded_temp}°C "
                f"(chose {chosen}: floor→{floor_predicted:.2f}°C [err={floor_error:.3f}], "
                f"ceiling→{ceiling_predicted:.2f}°C [err={ceiling_error:.3f}], "
                f"target={target_indoor_temp:.1f}°C)"
            )
            
            # Log final prediction with the applied smart-rounded temperature
            applied_prediction = floor_predicted if chosen == "floor" else ceiling_predicted
            logging.info(
                f"🎯 FINAL: Applied outlet {smart_rounded_temp}°C → Predicted indoor {applied_prediction:.2f}°C "
                f"(target: {target_indoor_temp:.1f}°C, error: {abs(applied_prediction - target_indoor_temp):.3f}°C)"
            )
            
            return smart_rounded_temp
            
        except Exception as e:
            # Fallback to regular rounding if smart rounding fails
            smart_rounded_temp = round(final_temp)
            logging.warning(
                f"Smart rounding failed ({e}), using regular rounding: "
                f"{final_temp:.2f}°C → {smart_rounded_temp}°C"
            )
            return smart_rounded_temp


class OnlineLearning:
    """Handles online learning from previous cycle results"""
    
    def learn_from_previous_cycle(self, state: Dict, ha_client: HAClient, 
                                all_states: Dict) -> None:
        """
        Learn from the results of the previous cycle
        """
        last_run_features = state.get("last_run_features")
        last_indoor_temp = state.get("last_indoor_temp")
        last_final_temp_stored = state.get("last_final_temp")
        
        if not all([last_run_features, last_indoor_temp, last_final_temp_stored]):
            logging.debug("Skipping online learning: no data from previous cycle")
            return
            
        # Read the actual target outlet temp that was applied
        actual_applied_temp = ha_client.get_state(
            config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID, all_states
        )
        
        if actual_applied_temp is None:
            logging.debug("Could not read actual applied temp, using last_final_temp as fallback")
            actual_applied_temp = last_final_temp_stored
            
        # Get current indoor temperature to calculate actual change
        current_indoor = ha_client.get_state(config.INDOOR_TEMP_ENTITY_ID, all_states)
        
        if current_indoor is None:
            logging.debug("Skipping online learning: current indoor temp unavailable")
            return
            
        actual_indoor_change = current_indoor - last_indoor_temp
        
        # Prepare learning features
        learning_features = self._prepare_learning_features(
            last_run_features, actual_applied_temp
        )
        
        # Perform online learning
        self._perform_online_learning(
            learning_features, actual_applied_temp, actual_indoor_change, current_indoor
        )
        
        # Log shadow mode comparison if applicable
        self._log_shadow_mode_comparison(
            actual_applied_temp, last_final_temp_stored
        )
    
    def _prepare_learning_features(self, last_run_features: Any, 
                                 actual_applied_temp: float) -> Dict:
        """Prepare features for online learning"""
        # Handle case where last_run_features might be stored as string
        if isinstance(last_run_features, str):
            logging.error("ERROR: last_run_features corrupted as string - attempting to recover")
            try:
                import json
                last_run_features = json.loads(last_run_features)
                logging.info("✅ Successfully recovered features from JSON string")
            except (json.JSONDecodeError, TypeError):
                logging.error("❌ Cannot recover features from string, using empty dict")
                last_run_features = {}
        
        # Convert to dict format
        if hasattr(last_run_features, 'to_dict'):
            learning_features = last_run_features.to_dict(orient="records")[0]
        elif isinstance(last_run_features, dict):
            learning_features = last_run_features.copy()
        else:
            learning_features = last_run_features.copy() if last_run_features else {}
            
        # Add outlet temperature features
        learning_features["outlet_temp"] = actual_applied_temp
        learning_features["outlet_temp_sq"] = actual_applied_temp ** 2
        learning_features["outlet_temp_cub"] = actual_applied_temp ** 3
        
        return learning_features
    
    def _perform_online_learning(self, learning_features: Dict, actual_applied_temp: float,
                               actual_indoor_change: float, current_indoor: float) -> None:
        """Perform the actual online learning"""
        try:
            wrapper = get_enhanced_model_wrapper()
            
            # Shadow mode vs Active mode learning context
            if config.SHADOW_MODE:
                # Shadow mode: Learn pure physics (heat curve outlet → actual indoor)
                prediction_context = {
                    'outlet_temp': actual_applied_temp,  # Heat curve's actual setting
                    'outdoor_temp': learning_features.get('outdoor_temp', 10.0),
                    'pv_power': learning_features.get('pv_now', 0.0),
                    'fireplace_on': learning_features.get('fireplace_on', 0.0),
                    'tv_on': learning_features.get('tv_on', 0.0)
                    # NO target temperature in shadow mode learning context!
                }
                
                # Previous indoor temperature (what we're learning to predict)
                previous_indoor = current_indoor - actual_indoor_change
                
                # Learn: heat curve outlet → actual indoor (pure physics)
                wrapper.learn_from_prediction_feedback(
                    predicted_temp=previous_indoor,  # Starting indoor temp
                    actual_temp=current_indoor,      # Actual result indoor temp
                    prediction_context=prediction_context,
                    timestamp=datetime.now().isoformat()
                )
                
                logging.info(
                    "🔬 Shadow mode physics learning: heat_curve_outlet=%.1f°C → "
                    "indoor_change=%.3f°C (%.1f→%.1f°C), cycle=%d",
                    actual_applied_temp, actual_indoor_change, 
                    previous_indoor, current_indoor, wrapper.cycle_count
                )
                
            else:
                # Active mode: Normal learning from ML's own predictions
                prediction_context = {
                    'outlet_temp': actual_applied_temp,  # ML's own prediction that was applied
                    'outdoor_temp': learning_features.get('outdoor_temp', 10.0),
                    'pv_power': learning_features.get('pv_now', 0.0),
                    'fireplace_on': learning_features.get('fireplace_on', 0.0),
                    'tv_on': learning_features.get('tv_on', 0.0)
                }
                
                # Calculate what the model predicted vs actual result
                predicted_change = 0.0  # Model's prediction for indoor temp change
                
                # Call the learning feedback method
                wrapper.learn_from_prediction_feedback(
                    predicted_temp=current_indoor - actual_indoor_change + predicted_change,
                    actual_temp=current_indoor,
                    prediction_context=prediction_context,
                    timestamp=datetime.now().isoformat()
                )
                
                logging.info(
                    "✅ Active mode learning: ml_outlet=%.1f°C, actual_change=%.3f°C, cycle=%d",
                    actual_applied_temp, actual_indoor_change, wrapper.cycle_count
                )
            
        except Exception as e:
            logging.warning("Online learning failed: %s", e, exc_info=True)
    
    def _log_shadow_mode_comparison(self, actual_applied_temp: float, 
                                  last_final_temp_stored: float) -> None:
        """Log shadow mode comparison and benchmarking if applicable"""
        # Only log comparison when actually in shadow mode (not active mode)
        # In active mode, ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID reads what ML itself set
        effective_shadow_mode = (
            config.SHADOW_MODE or 
            config.TARGET_OUTLET_TEMP_ENTITY_ID != config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID
        )
        
        if effective_shadow_mode and actual_applied_temp != last_final_temp_stored:
            # Calculate efficiency advantage (lower outlet = more efficient)
            efficiency_advantage = actual_applied_temp - last_final_temp_stored
            
            # Enhanced benchmarking logging
            logging.info(
                "🎯 Shadow Benchmark: ML would predict %.1f°C, "
                "Heat Curve set %.1f°C (difference: %+.1f°C)",
                last_final_temp_stored, actual_applied_temp, efficiency_advantage
            )
            
            # Export benchmark data to InfluxDB if available
            self._export_shadow_benchmark_data(
                ml_outlet_prediction=last_final_temp_stored,
                heat_curve_outlet_actual=actual_applied_temp,
                efficiency_advantage=efficiency_advantage
            )

    def calculate_ml_benchmark_prediction(self, target_indoor_temp: float, 
                                        current_indoor_temp: float, 
                                        context: Dict) -> float:
        """
        Calculate what ML would predict for current target temperature.
        
        This implements Task 2.1 from the TODO - provides the missing ML 
        prediction calculation for proper benchmarking against heat curve.
        
        Args:
            target_indoor_temp: Target temperature to achieve
            current_indoor_temp: Current indoor temperature
            context: Environmental context (outdoor_temp, pv_power, etc.)
            
        Returns:
            ML's predicted optimal outlet temperature for the target
        """
        try:
            wrapper = get_enhanced_model_wrapper()
            
            # Extract environmental context
            outdoor_temp = context.get('outdoor_temp', 10.0)
            pv_power = context.get('pv_power', 0.0)
            fireplace_on = context.get('fireplace_on', 0.0)
            tv_on = context.get('tv_on', 0.0)
            
            # Get ML's prediction for achieving target temperature
            ml_optimal_result = wrapper.calculate_optimal_outlet_temperature(
                target_indoor=target_indoor_temp,
                current_indoor=current_indoor_temp,
                outdoor_temp=outdoor_temp,
                pv_power=pv_power,
                fireplace_on=fireplace_on,
                tv_on=tv_on
            )
            
            if ml_optimal_result is None:
                logging.warning("ML optimal calculation failed, using fallback prediction")
                return 40.0  # Fallback
                
            ml_predicted_outlet = ml_optimal_result.get('optimal_outlet_temp', 40.0)
            
            logging.debug(
                "ML benchmark prediction: target=%.1f°C, current=%.1f°C, "
                "outdoor=%.1f°C → ML predicts %.1f°C outlet",
                target_indoor_temp, current_indoor_temp, outdoor_temp, ml_predicted_outlet
            )
            
            return ml_predicted_outlet
            
        except Exception as e:
            logging.warning("ML benchmark prediction failed: %s, using fallback", e)
            return 40.0  # Safe fallback
    
    def _export_shadow_benchmark_data(self, ml_outlet_prediction: float,
                                    heat_curve_outlet_actual: float,
                                    efficiency_advantage: float) -> None:
        """Export shadow mode benchmark data to InfluxDB"""
        try:
            # Import here to avoid circular imports
            from .influx_service import get_influx_service
            
            influx_service = get_influx_service()
            if influx_service is None:
                logging.debug("No InfluxDB service available for benchmark export")
                return
                
            benchmark_data = {
                'ml_outlet_prediction': ml_outlet_prediction,
                'heat_curve_outlet_actual': heat_curve_outlet_actual,
                'efficiency_advantage': efficiency_advantage,
                'timestamp': datetime.now().isoformat()
            }
            
            # Write benchmark data point
            influx_service.write_shadow_mode_benchmarks(benchmark_data)
            
            logging.debug(
                "📊 Exported shadow benchmark: ML=%.1f°C, HC=%.1f°C, "
                "efficiency_advantage=%+.1f°C",
                ml_outlet_prediction, heat_curve_outlet_actual, efficiency_advantage
            )
            
        except Exception as e:
            logging.warning("Failed to export shadow mode benchmark data: %s", e)


class TemperatureControlManager:
    """Main temperature control manager that orchestrates all temperature-related operations"""
    
    def __init__(self):
        self.predictor = TemperaturePredictor()
        self.gradual_control = GradualTemperatureControl()
        self.smart_rounding = SmartRounding()
        self.online_learning = OnlineLearning()
    
    def determine_prediction_indoor_temp(self, fireplace_on: bool, actual_indoor: float,
                                       avg_other_rooms_temp: float) -> float:
        """Determine which indoor temperature to use for prediction"""
        if fireplace_on:
            logging.info("Fireplace is ON. Using average temperature of other rooms for prediction.")
            return avg_other_rooms_temp
        else:
            logging.info("Fireplace is OFF. Using main indoor temp for prediction.")
            return actual_indoor
    
    def build_features(self, ha_client: HAClient, influx_service) -> Tuple[Optional[Dict], Optional[Any]]:
        """Build physics features for prediction"""
        features, outlet_history = build_physics_features(ha_client, influx_service)
        
        if features is None:
            logging.warning("Feature building failed, skipping cycle.")
            return None, None
            
        return features, outlet_history
    
    def execute_temperature_control_cycle(self, features: Dict, prediction_indoor_temp: float,
                                        target_indoor_temp: float, actual_outlet_temp: Optional[float],
                                        outdoor_temp: float, fireplace_on: bool, state: Dict) -> Tuple[float, float, Dict, int]:
        """
        Execute complete temperature control cycle
        
        Returns:
            Tuple of (final_temp, confidence, metadata, smart_rounded_temp)
        """
        # Step 1: Predict optimal temperature
        suggested_temp, confidence, metadata = self.predictor.predict_optimal_temperature(
            features, prediction_indoor_temp, target_indoor_temp
        )
        
        # Step 2: Apply gradual temperature control
        final_temp = self.gradual_control.apply_gradual_control(
            suggested_temp, actual_outlet_temp, state
        )
        
        # Step 3: Apply smart rounding
        smart_rounded_temp = self.smart_rounding.apply_smart_rounding(
            final_temp, outdoor_temp, features, fireplace_on, target_indoor_temp
        )
        
        return final_temp, confidence, metadata, smart_rounded_temp
