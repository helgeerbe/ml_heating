"""
ThermalEquilibriumModel-based Model Wrapper.

This module provides a clean interface for thermal physics-based heating control
using only the ThermalEquilibriumModel. All legacy ML model code has been removed
as part of the thermal equilibrium model migration.

Key features:
- Single ThermalEquilibriumModel-based prediction pathway
- Persistent thermal learning state across service restarts
- Simplified outlet temperature prediction interface
- Adaptive thermal parameter learning
"""
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime

import pandas as pd

# Support both package-relative and direct import for notebooks
try:
    from .thermal_equilibrium_model import ThermalEquilibriumModel
    from .state_manager import load_state, save_state
    from .prediction_metrics import PredictionMetrics, track_prediction
except ImportError:
    from thermal_equilibrium_model import ThermalEquilibriumModel
    from state_manager import load_state, save_state
    from prediction_metrics import PredictionMetrics, track_prediction


class EnhancedModelWrapper:
    """
    Simplified model wrapper using ThermalEquilibriumModel for persistent learning.
    
    Replaces the complex Heat Balance Controller with a single prediction path
    that continuously adapts thermal parameters and survives service restarts.
    """
    
    def __init__(self):
        self.thermal_model = ThermalEquilibriumModel()
        self.learning_enabled = True
        self.cycle_count = 0
        
        # Initialize prediction metrics tracker
        self.prediction_metrics = PredictionMetrics()
        
        # Load any existing thermal learning state
        self._restore_learning_state()
        
        # Load prediction metrics if available
        try:
            self.prediction_metrics.load_state('/opt/ml_heating/prediction_metrics.json')
            logging.info("✅ Loaded existing prediction metrics")
        except FileNotFoundError:
            logging.info("ℹ️  No existing prediction metrics found - starting fresh")
        except Exception as e:
            logging.warning(f"Failed to load prediction metrics: {e}")
        
        logging.info("🎯 Enhanced Model Wrapper initialized with ThermalEquilibriumModel")
        logging.info(f"   - Thermal time constant: {self.thermal_model.thermal_time_constant:.1f}h")
        logging.info(f"   - Heat loss coefficient: {self.thermal_model.heat_loss_coefficient:.4f}")
        logging.info(f"   - Outlet effectiveness: {self.thermal_model.outlet_effectiveness:.3f}")
        logging.info(f"   - Learning confidence: {self.thermal_model.learning_confidence:.2f}")
    
    def calculate_optimal_outlet_temp(self, features: Dict) -> Tuple[float, Dict]:
        """Calculate optimal outlet temperature using direct thermal physics prediction."""
        try:
            # Extract core thermal parameters
            current_indoor = features.get('indoor_temp_lag_30m', 21.0)
            target_indoor = features.get('target_temp', 21.0) 
            outdoor_temp = features.get('outdoor_temp', 10.0)
            
            # Extract enhanced thermal intelligence features
            thermal_features = self._extract_thermal_features(features)
            
            # Calculate required outlet temperature using iterative approach
            optimal_outlet_temp = self._calculate_required_outlet_temp(
                current_indoor, target_indoor, outdoor_temp, thermal_features
            )
            
            # Get prediction metadata
            confidence = self.thermal_model.learning_confidence
            prediction_metadata = {
                'thermal_time_constant': self.thermal_model.thermal_time_constant,
                'heat_loss_coefficient': self.thermal_model.heat_loss_coefficient,
                'outlet_effectiveness': self.thermal_model.outlet_effectiveness,
                'learning_confidence': confidence,
                'prediction_method': 'thermal_equilibrium_single_prediction',
                'cycle_count': self.cycle_count
            }
            
            if optimal_outlet_temp is not None:
                logging.info(
                    f"Enhanced prediction: {current_indoor:.2f}°C → {target_indoor:.1f}°C "
                    f"requires {optimal_outlet_temp:.1f}°C (confidence: {confidence:.3f})"
                )
            else:
                logging.warning("Failed to calculate optimal outlet temperature")
                optimal_outlet_temp = 35.0  # Safe fallback
            
            return optimal_outlet_temp, prediction_metadata
            
        except Exception as e:
            logging.error(f"Enhanced prediction failed: {e}", exc_info=True)
            # Fallback to safe temperature
            fallback_temp = 35.0
            fallback_metadata = {
                'prediction_method': 'fallback_safe_temperature',
                'error': str(e)
            }
            return fallback_temp, fallback_metadata
    
    def _extract_thermal_features(self, features: Dict) -> Dict:
        """Extract thermal intelligence features for the thermal model."""
        thermal_features = {}
        
        # Multi-heat source features
        thermal_features['pv_power'] = features.get('pv_now', 0.0)
        thermal_features['fireplace_on'] = float(features.get('fireplace_on', 0))
        thermal_features['tv_on'] = float(features.get('tv_on', 0))
        
        # Enhanced thermal intelligence features
        thermal_features['indoor_temp_gradient'] = features.get('indoor_temp_gradient', 0.0)
        thermal_features['temp_diff_indoor_outdoor'] = features.get('temp_diff_indoor_outdoor', 0.0)
        thermal_features['outlet_indoor_diff'] = features.get('outlet_indoor_diff', 0.0)
        
        # Note: Removed occupancy and cooking features as they don't have corresponding sensors
        
        return thermal_features
    
    def _calculate_required_outlet_temp(self, current_indoor: float, target_indoor: float, 
                                      outdoor_temp: float, thermal_features: Dict) -> float:
        """Calculate the outlet temperature required to reach target indoor temperature using learned thermal model."""
        # If we're already at target, use moderate heating
        if abs(current_indoor - target_indoor) < 0.1:
            return 35.0
            
        # Use the calibrated thermal model to find required outlet temperature
        # This leverages the 26 days of learned parameters instead of simple heuristics
        pv_power = thermal_features.get('pv_power', 0.0)
        fireplace_on = thermal_features.get('fireplace_on', 0.0)
        tv_on = thermal_features.get('tv_on', 0.0)
        
        # Iterative search to find outlet temperature that produces target indoor temp
        # This uses the learned thermal physics parameters from calibration
        tolerance = 0.1  # °C
        outlet_min, outlet_max = 25.0, 65.0
        
        # Binary search for optimal outlet temperature
        for iteration in range(20):  # Max 20 iterations for efficiency
            outlet_mid = (outlet_min + outlet_max) / 2.0
            
            # Predict indoor temperature with this outlet temperature
            predicted_indoor = self.thermal_model.predict_equilibrium_temperature(
                outlet_temp=outlet_mid,
                outdoor_temp=outdoor_temp,
                pv_power=pv_power,
                fireplace_on=fireplace_on,
                tv_on=tv_on
            )
            
            # Check if we're close enough
            error = predicted_indoor - target_indoor
            if abs(error) < tolerance:
                logging.debug(f"Converged after {iteration+1} iterations: {outlet_mid:.1f}°C → {predicted_indoor:.1f}°C (target: {target_indoor:.1f}°C)")
                return outlet_mid
            
            # Adjust search range based on error
            if predicted_indoor < target_indoor:
                # Need higher outlet temperature
                outlet_min = outlet_mid
            else:
                # Need lower outlet temperature
                outlet_max = outlet_mid
        
        # Return best guess if didn't converge
        final_outlet = (outlet_min + outlet_max) / 2.0
        final_predicted = self.thermal_model.predict_equilibrium_temperature(
            outlet_temp=final_outlet,
            outdoor_temp=outdoor_temp,
            pv_power=pv_power,
            fireplace_on=fireplace_on,
            tv_on=tv_on
        )
        logging.debug(f"Binary search completed: {final_outlet:.1f}°C → {final_predicted:.1f}°C (target: {target_indoor:.1f}°C)")
        
        return final_outlet
    
    def _restore_learning_state(self):
        """Restore thermal learning state from persistent storage."""
        try:
            state = load_state()
            thermal_learning_state = state.get('thermal_learning_state', {})
            
            if thermal_learning_state:
                self.thermal_model.thermal_time_constant = thermal_learning_state.get(
                    'thermal_time_constant', 24.0
                )
                self.thermal_model.heat_loss_coefficient = thermal_learning_state.get(
                    'heat_loss_coefficient', 0.05
                )
                self.thermal_model.outlet_effectiveness = thermal_learning_state.get(
                    'outlet_effectiveness', 0.8
                )
                self.thermal_model.learning_confidence = thermal_learning_state.get(
                    'learning_confidence', 3.0
                )
                self.cycle_count = thermal_learning_state.get('cycle_count', 0)
                
                logging.info(f"🔄 Warm start: Restored thermal learning state from cycle {self.cycle_count}")
            else:
                logging.info("🆕 Cold start: No existing thermal learning state found")
                
        except Exception as e:
            logging.warning(f"Failed to restore learning state: {e}")
    
    def learn_from_prediction_feedback(self, predicted_temp: float, actual_temp: float, 
                                     prediction_context: Dict, timestamp: Optional[str] = None):
        """Learn from prediction feedback using the thermal model's adaptive learning."""
        if not self.learning_enabled:
            return
            
        try:
            # Update thermal model with prediction feedback
            self.thermal_model.update_prediction_feedback(
                predicted_temp=predicted_temp,
                actual_temp=actual_temp,
                prediction_context=prediction_context,
                timestamp=timestamp or datetime.now().isoformat()
            )
            
            # Track prediction in metrics system
            self.prediction_metrics.add_prediction(
                predicted=predicted_temp,
                actual=actual_temp,
                context=prediction_context,
                timestamp=timestamp
            )
            
            # Track learning cycles
            self.cycle_count += 1
            
            # Auto-save learning state every cycle
            self._save_learning_state()
            
            # Save prediction metrics periodically
            if self.cycle_count % 10 == 0:
                try:
                    self.prediction_metrics.save_state('/opt/ml_heating/prediction_metrics.json')
                except Exception as e:
                    logging.warning(f"Failed to save prediction metrics: {e}")
            
            prediction_error = abs(predicted_temp - actual_temp)
            logging.debug(
                f"Learning cycle {self.cycle_count}: error={prediction_error:.3f}°C, "
                f"confidence={self.thermal_model.learning_confidence:.3f}"
            )
            
        except Exception as e:
            logging.error(f"Learning from feedback failed: {e}", exc_info=True)
    
    def get_prediction_confidence(self) -> float:
        """Get current prediction confidence from thermal model."""
        return self.thermal_model.learning_confidence
    
    def get_learning_metrics(self) -> Dict:
        """Get comprehensive learning metrics for monitoring."""
        try:
            return self.thermal_model.get_adaptive_learning_metrics()
        except AttributeError:
            # Fallback if method doesn't exist
            return {
                'thermal_time_constant': self.thermal_model.thermal_time_constant,
                'heat_loss_coefficient': self.thermal_model.heat_loss_coefficient,
                'outlet_effectiveness': self.thermal_model.outlet_effectiveness,
                'learning_confidence': self.thermal_model.learning_confidence,
                'cycle_count': self.cycle_count
            }
    
    def get_comprehensive_metrics_for_ha(self) -> Dict:
        """Get comprehensive metrics for Home Assistant sensor export."""
        try:
            # Get thermal learning metrics
            thermal_metrics = self.get_learning_metrics()
            
            # Get prediction accuracy metrics
            prediction_metrics = self.prediction_metrics.get_metrics()
            
            # Get recent performance
            recent_performance = self.prediction_metrics.get_recent_performance(10)
            
            # Combine into comprehensive HA-friendly format
            ha_metrics = {
                # Core thermal parameters (learned)
                'thermal_time_constant': thermal_metrics.get('thermal_time_constant', 24.0),
                'heat_loss_coefficient': thermal_metrics.get('heat_loss_coefficient', 0.05),
                'outlet_effectiveness': thermal_metrics.get('outlet_effectiveness', 0.8),
                'learning_confidence': thermal_metrics.get('learning_confidence', 3.0),
                
                # Learning progress
                'cycle_count': self.cycle_count,
                'parameter_updates': thermal_metrics.get('parameter_updates', 0),
                'update_percentage': thermal_metrics.get('update_percentage', 0),
                
                # Prediction accuracy (MAE/RMSE)
                'mae_1h': prediction_metrics.get('1h', {}).get('mae', 0.0),
                'mae_6h': prediction_metrics.get('6h', {}).get('mae', 0.0), 
                'mae_24h': prediction_metrics.get('24h', {}).get('mae', 0.0),
                'mae_all_time': prediction_metrics.get('all', {}).get('mae', 0.0),
                'rmse_all_time': prediction_metrics.get('all', {}).get('rmse', 0.0),
                
                # Recent performance
                'recent_mae_10': recent_performance.get('mae', 0.0),
                'recent_max_error': recent_performance.get('max_error', 0.0),
                
                # Accuracy breakdown
                'excellent_accuracy_pct': prediction_metrics.get('accuracy_breakdown', {}).get('excellent', {}).get('percentage', 0.0),
                'good_accuracy_pct': (
                    prediction_metrics.get('accuracy_breakdown', {}).get('excellent', {}).get('percentage', 0.0) +
                    prediction_metrics.get('accuracy_breakdown', {}).get('very_good', {}).get('percentage', 0.0) +
                    prediction_metrics.get('accuracy_breakdown', {}).get('good', {}).get('percentage', 0.0)
                ),
                
                # Trend analysis
                'is_improving': prediction_metrics.get('trends', {}).get('is_improving', False),
                'improvement_percentage': prediction_metrics.get('trends', {}).get('mae_improvement_percentage', 0.0),
                
                # Model health summary
                'model_health': 'excellent' if thermal_metrics.get('learning_confidence', 0) >= 4.0 else
                               'good' if thermal_metrics.get('learning_confidence', 0) >= 3.0 else
                               'fair' if thermal_metrics.get('learning_confidence', 0) >= 2.0 else 'poor',
                
                # Total predictions tracked
                'total_predictions': len(self.prediction_metrics.predictions),
                
                # Timestamp
                'last_updated': datetime.now().isoformat()
            }
            
            return ha_metrics
            
        except Exception as e:
            logging.error(f"Failed to get comprehensive metrics: {e}")
            return {
                'error': str(e),
                'cycle_count': self.cycle_count,
                'last_updated': datetime.now().isoformat()
            }
    
    def _save_learning_state(self):
        """Save current thermal learning state to persistent storage."""
        try:
            thermal_learning_state = {
                'thermal_time_constant': self.thermal_model.thermal_time_constant,
                'heat_loss_coefficient': self.thermal_model.heat_loss_coefficient,
                'outlet_effectiveness': self.thermal_model.outlet_effectiveness,
                'learning_confidence': self.thermal_model.learning_confidence,
                'cycle_count': self.cycle_count,
                'last_updated': datetime.now().isoformat()
            }
            
            # Save enhanced state with thermal learning
            save_state(thermal_learning_state=thermal_learning_state)
            
        except Exception as e:
            logging.error(f"Failed to save learning state: {e}")


# Legacy functions removed - ThermalEquilibriumModel handles persistence internally


def get_enhanced_model_wrapper() -> EnhancedModelWrapper:
    """
    Create and return an enhanced model wrapper for simplified control.
    
    This replaces the complex Heat Balance Controller with the 
    ThermalEquilibriumModel-based approach.
    """
    return EnhancedModelWrapper()


def simplified_outlet_prediction(
    features: pd.DataFrame,
    current_temp: float,
    target_temp: float
) -> Tuple[float, float, Dict]:
    """
    SIMPLIFIED outlet temperature prediction using Enhanced Model Wrapper.
    
    This replaces the complex find_best_outlet_temp() function with a single
    call to the Enhanced Model Wrapper, dramatically simplifying the codebase.
    
    Args:
        features: Input features DataFrame
        current_temp: Current indoor temperature
        target_temp: Target indoor temperature
        
    Returns:
        Tuple of (outlet_temp, confidence, metadata)
    """
    try:
        # Create enhanced model wrapper
        wrapper = get_enhanced_model_wrapper()
        
        # Convert features to dict format - handle empty DataFrame
        records = features.to_dict(orient="records")
        if len(records) == 0:
            features_dict = {}
        else:
            features_dict = records[0]
            
        features_dict['indoor_temp_lag_30m'] = current_temp
        features_dict['target_temp'] = target_temp
        
        # Get simplified prediction
        outlet_temp, metadata = wrapper.calculate_optimal_outlet_temp(features_dict)
        confidence = metadata.get('learning_confidence', 3.0)
        
        # Calculate thermal trust metrics for HA sensor display
        thermal_trust_metrics = _calculate_thermal_trust_metrics(wrapper, outlet_temp, current_temp, target_temp)
        metadata['thermal_trust_metrics'] = thermal_trust_metrics
        
        logging.info(
            f"✨ Simplified prediction: {current_temp:.2f}°C → {target_temp:.1f}°C "
            f"requires {outlet_temp:.1f}°C (confidence: {confidence:.3f})"
        )
        
        return outlet_temp, confidence, metadata
        
    except Exception as e:
        logging.error(f"Simplified prediction failed: {e}", exc_info=True)
        # Safe fallback
        return 35.0, 2.0, {'error': str(e), 'method': 'fallback'}


def _calculate_thermal_trust_metrics(
    wrapper: EnhancedModelWrapper, 
    outlet_temp: float, 
    current_temp: float, 
    target_temp: float
) -> Dict:
    """
    Calculate thermal trust metrics for HA sensor display.
    
    These metrics replace legacy MAE/RMSE with physics-based trust indicators
    that show how well the thermal model is performing.
    """
    try:
        # Get thermal model parameters
        thermal_model = wrapper.thermal_model
        
        # Calculate thermal stability (how stable are the thermal parameters)
        time_constant_stability = min(1.0, thermal_model.thermal_time_constant / 48.0)
        heat_loss_stability = min(1.0, thermal_model.heat_loss_coefficient * 20.0)
        effectiveness_stability = thermal_model.outlet_effectiveness
        thermal_stability = (time_constant_stability + heat_loss_stability + effectiveness_stability) / 3.0
        
        # Calculate prediction consistency (how reasonable is this prediction)
        temp_diff = abs(target_temp - current_temp)
        outlet_indoor_diff = abs(outlet_temp - current_temp)
        
        # Reasonable outlet temps should be 5-40°C above indoor temp for heating
        if temp_diff > 0.1:  # Need heating
            reasonable_range = outlet_indoor_diff >= 5.0 and outlet_indoor_diff <= 40.0
        else:  # At target
            reasonable_range = outlet_indoor_diff >= 0.0 and outlet_indoor_diff <= 20.0
            
        prediction_consistency = 1.0 if reasonable_range else 0.5
        
        # Calculate physics alignment (how well does prediction align with physics)
        # Higher outlet temps should be needed for larger temperature differences
        if temp_diff > 0.1:
            expected_outlet_range = current_temp + (temp_diff * 8.0)  # Rough physics heuristic
            physics_error = abs(outlet_temp - expected_outlet_range)
            physics_alignment = max(0.0, 1.0 - (physics_error / 20.0))
        else:
            physics_alignment = 1.0
            
        # Model health assessment
        confidence = thermal_model.learning_confidence
        if confidence >= 4.0:
            model_health = "excellent"
        elif confidence >= 3.0:
            model_health = "good"
        elif confidence >= 2.0:
            model_health = "fair"
        else:
            model_health = "poor"
            
        # Learning progress (how much has the model learned)
        cycle_count = wrapper.cycle_count
        learning_progress = min(1.0, cycle_count / 100.0)  # Fully learned after 100 cycles
        
        return {
            'thermal_stability': thermal_stability,
            'prediction_consistency': prediction_consistency,
            'physics_alignment': physics_alignment,
            'model_health': model_health,
            'learning_progress': learning_progress
        }
        
    except Exception as e:
        logging.error(f"Failed to calculate thermal trust metrics: {e}")
        return {
            'thermal_stability': 0.0,
            'prediction_consistency': 0.0,
            'physics_alignment': 0.0,
            'model_health': 'error',
            'learning_progress': 0.0
        }


# Legacy functions completely removed - ThermalEquilibriumModel provides all needed functionality


# No backward compatibility functions needed - clean slate approach
