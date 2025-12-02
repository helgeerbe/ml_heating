"""
Realistic Physics Model for ML Heating Controller

This module contains the physics-based heating model that learns actual
house characteristics and external heat source effects from historical data.
"""

import numpy as np
import logging


class RealisticPhysicsModel:
    """
    Physics model that learns from target temperatures and actual effects.
    
    Key improvements:
    - Uses target temperature history for realistic calibration
    - Learns actual PV, fireplace, TV effects from data
    - Ensures positive heating predictions for proper outlet temperatures
    """
    
    def __init__(self):
        # Core heating physics - will be learned from data
        self.base_heating_rate = 0.002  # Start higher for proper physics
        self.target_influence = 0.01    # How target temp affects heating
        self.outdoor_factor = 0.003     # Outdoor temperature influence

        # External heat source coefficients (learned from data)
        self.pv_warming_coefficient = 0.001 # Initial value for PV warming
        self.fireplace_heating_rate = 0.01  # Initial value for fireplace
        self.tv_heat_contribution = 0.005   # Initial value for TV
        
        # Live performance tracking
        self.prediction_errors = []  # Rolling window of recent errors
        self.max_error_history = 50  # Keep last 50 predictions for performance
        
        # SEASONAL: Learned seasonal modulation (cos/sin components)
        # Applied to core parameters for automatic seasonal adaptation
        self.seasonal_heating_cos = 0.0
        self.seasonal_heating_sin = 0.0
        self.seasonal_target_cos = 0.0
        self.seasonal_target_sin = 0.0
        self.seasonal_outdoor_cos = 0.0
        self.seasonal_outdoor_sin = 0.0
        
        # MULTI-LAG: External heat source coefficients with time delays
        # PV Solar (longest lags - thermal mass delay)
        self.pv_coeffs = {
            'lag_1': 0.0,    # t-30min
            'lag_2': 0.0,    # t-60min (expected peak)
            'lag_3': 0.0,    # t-90min
            'lag_4': 0.0,    # t-120min
        }
        self.pv_seasonal_cos = 0.0  # PV varies with season (windows open/closed)
        self.pv_seasonal_sin = 0.0
        
        # Fireplace (medium lags - radiant + convective, winter-only, no seasonal)
        self.fireplace_coeffs = {
            'immediate': 0.0,  # t=0 (radiant)
            'lag_1': 0.0,      # t-30min (peak convective)
            'lag_2': 0.0,      # t-60min (sustained)
            'lag_3': 0.0,      # t-90min (declining)
        }
        
        # TV/Electronics (short lags - quick equilibrium)
        self.tv_coeffs = {
            'immediate': 0.0,  # t=0 (direct radiant)
            'lag_1': 0.0,      # t-30min (steady state)
        }
        self.tv_seasonal_cos = 0.0  # TV varies with season (room usage)
        self.tv_seasonal_sin = 0.0
        
        # MULTI-LAG: History buffers (ring buffers for efficiency)
        self.pv_history = []           # Last 5 cycles (0-120min)
        self.fireplace_history = []    # Last 4 cycles (0-90min)
        self.tv_history = []           # Last 2 cycles (0-30min)
        
        # Forecast parameters
        self.weather_forecast_coeff = 0.015
        self.pv_forecast_coeff = 0.008
        self.forecast_decay = [1.0, 0.8, 0.6, 0.4]
        
        # Physics bounds - allow bidirectional heat transfer
        self.min_prediction = -0.15  # Allow cooling effects when outlet < indoor
        self.max_prediction = 0.25   # Reasonable heating limit
        
        # Learning tracking
        self.training_count = 0
        self.adaptation_frequency = 50  # Learn every 50 samples
        
        # MULTI-LAG: Detailed effect tracking with historical context
        self.pv_effects_detailed = []
        self.fireplace_effects_detailed = []
        self.tv_effects_detailed = []
        
        # SEASONAL: Track performance by season for adaptation
        self.seasonal_performance = []  # (month_cos, month_sin, error)
        
        # SUMMER: Track external sources when HVAC off
        self.hvac_off_tracking = []  # Learn from summer/off periods
        
        # Legacy simple tracking for backward compatibility
        self.pv_effects = []
        self.fireplace_effects = []
        self.tv_effects = []
    
    def _init_multilag_attributes(self):
        """Initialize multi-lag attributes for backward compatibility"""
        if not hasattr(self, 'pv_history'):
            self.pv_history = []
        if not hasattr(self, 'fireplace_history'):
            self.fireplace_history = []
        if not hasattr(self, 'tv_history'):
            self.tv_history = []
        if not hasattr(self, 'pv_coeffs'):
            self.pv_coeffs = {
                'lag_1': 0.0, 'lag_2': 0.0, 'lag_3': 0.0, 'lag_4': 0.0
            }
        if not hasattr(self, 'fireplace_coeffs'):
            self.fireplace_coeffs = {
                'immediate': 0.0, 'lag_1': 0.0, 'lag_2': 0.0, 'lag_3': 0.0
            }
        if not hasattr(self, 'tv_coeffs'):
            self.tv_coeffs = {
                'immediate': 0.0, 'lag_1': 0.0
            }
        if not hasattr(self, 'pv_seasonal_cos'):
            self.pv_seasonal_cos = 0.0
            self.pv_seasonal_sin = 0.0
        if not hasattr(self, 'tv_seasonal_cos'):
            self.tv_seasonal_cos = 0.0
            self.tv_seasonal_sin = 0.0
        if not hasattr(self, 'seasonal_heating_cos'):
            self.seasonal_heating_cos = 0.0
            self.seasonal_heating_sin = 0.0
            self.seasonal_target_cos = 0.0
            self.seasonal_target_sin = 0.0
            self.seasonal_outdoor_cos = 0.0
            self.seasonal_outdoor_sin = 0.0
        if not hasattr(self, 'pv_effects_detailed'):
            self.pv_effects_detailed = []
            self.fireplace_effects_detailed = []
            self.tv_effects_detailed = []
            self.seasonal_performance = []
            self.hvac_off_tracking = []
    
    def predict_one(self, features):
        """Enhanced physics-based prediction with physics compliance validation"""
        
        # System availability check
        dhw_heating = features.get('dhw_heating', 0.0)
        dhw_disinfection = features.get('dhw_disinfection', 0.0)  
        dhw_boost_heater = features.get('dhw_boost_heater', 0.0)
        defrosting = features.get('defrosting', 0.0)
        
        if dhw_heating or dhw_disinfection or dhw_boost_heater:
            return 0.001  # Heat pump busy with DHW
            
        # Core temperatures
        outlet_temp = features.get('outlet_temp', 35.0)
        indoor_temp = features.get('indoor_temp_lag_30m', 21.0)
        target_temp = features.get('target_temp', 21.0)  # KEY: Use target!
        outdoor_temp = features.get('outdoor_temp', 5.0)
        
        # Defrost effect on outlet temperature
        if defrosting:
            effective_outlet = outlet_temp - 15.0  # Defrost penalty
        else:
            effective_outlet = outlet_temp
            
        # Core heating physics with target awareness
        temp_gap = target_temp - indoor_temp  # How much heating needed
        outlet_effect = effective_outlet - indoor_temp  # Heating potential
        outdoor_penalty = max(0, 10 - outdoor_temp) / 15  # Cold weather factor
        
        # Main heating calculation
        base_heating = outlet_effect * self.base_heating_rate
        target_boost = temp_gap * self.target_influence  # More heating when target is higher
        weather_adjustment = base_heating * outdoor_penalty * self.outdoor_factor
        
        # External heat sources (learned from data)
        pv_now = features.get('pv_now', 0.0)
        fireplace_on = features.get('fireplace_on', 0.0)
        tv_on = features.get('tv_on', 0.0)
        
        # Seasonal context for modulation
        month_cos = features.get('month_cos', 0.0)
        month_sin = features.get('month_sin', 0.0)
        
        # Update history buffers for multi-lag
        self._update_histories(pv_now, fireplace_on, tv_on)
        
        # PHYSICS-CONSTRAINED EXTERNAL SOURCES
        pv_contribution = self._calculate_pv_lagged_constrained(month_cos, month_sin, features)
        fireplace_contribution = self._calculate_fireplace_lagged_constrained()
        tv_contribution = self._calculate_tv_lagged_constrained(month_cos, month_sin)
        
        # Fallback to simple if not enough history (with constraints)
        if pv_contribution == 0.0 and len(self.pv_history) < 5:
            pv_contribution = self._calculate_pv_simple_constrained(pv_now, features)
        if fireplace_contribution == 0.0 and len(self.fireplace_history) < 4:
            fireplace_contribution = min(fireplace_on * self.fireplace_heating_rate, 0.02)  # Max 0.02°C
        if tv_contribution == 0.0 and len(self.tv_history) < 2:
            tv_contribution = min(tv_on * self.tv_heat_contribution, 0.01)  # Max 0.01°C
        
        # Forecast adjustments
        forecast_effect = self._calculate_forecast_adjustment(features)
        
        # PHYSICS COMPLIANCE CHECK
        basic_physics = base_heating + target_boost + weather_adjustment
        total_external = pv_contribution + fireplace_contribution + tv_contribution
        
        # CRITICAL: When outlet temp < indoor temp, prediction MUST be negative or very small
        if outlet_effect < -2.0:  # Cold outlet (more than 2°C below indoor)
            # External sources cannot make cold outlet predict significant heating
            max_allowed_external = max(0.01, abs(basic_physics) * 0.5)  # Max 50% of physics magnitude or 0.01°C
            
            if total_external > max_allowed_external:
                scale_factor = max_allowed_external / total_external
                pv_contribution *= scale_factor
                fireplace_contribution *= scale_factor
                tv_contribution *= scale_factor
                total_external = pv_contribution + fireplace_contribution + tv_contribution
                
                logging.debug(
                    f"Cold outlet physics compliance: scaled external sources by {scale_factor:.3f} "
                    f"(outlet_effect={outlet_effect:.1f}°C, basic_physics={basic_physics:.6f}°C)"
                )
        
        # General external source scaling (less aggressive for normal cases)
        elif abs(total_external) > abs(basic_physics) * 2.0 and abs(basic_physics) > 0.005:
            # Scale down external sources to respect physics
            scale_factor = abs(basic_physics) * 2.0 / abs(total_external)
            pv_contribution *= scale_factor
            fireplace_contribution *= scale_factor
            tv_contribution *= scale_factor
            total_external = pv_contribution + fireplace_contribution + tv_contribution
            
            logging.debug(
                f"General physics compliance: scaled external sources by {scale_factor:.3f} "
                f"(basic_physics={basic_physics:.6f}°C, external={total_external:.6f}°C)"
            )
        
        # Total prediction
        total_effect = (basic_physics + total_external + forecast_effect)
        
        # FINAL SANITY CHECK: Cold outlet cannot predict significant heating
        if outlet_effect < -2.0 and total_effect > 0.02:
            # Force prediction to be cooling or minimal heating
            total_effect = min(total_effect, 0.01)  # Max 0.01°C heating from cold outlet
            logging.debug(
                f"Final sanity check: limited prediction to {total_effect:.6f}°C "
                f"(outlet {outlet_effect:.1f}°C below indoor)"
            )
        
        # Apply physics bounds
        return np.clip(total_effect, self.min_prediction, self.max_prediction)
    
    def _calculate_forecast_adjustment(self, features):
        """Calculate forecast-based heating adjustments"""
        current_outdoor = features.get('outdoor_temp', 5.0)
        current_pv = features.get('pv_now', 0.0)
        
        # Weather forecasting - REDUCED IMPACT to prevent overwhelming physics
        weather_adjustment = 0.0
        for i in range(4):
            forecast_temp = features.get(f'temp_forecast_{i+1}h', current_outdoor)
            temp_change = forecast_temp - current_outdoor
            decay = self.forecast_decay[i]
            
            if temp_change > 1.5:  # Significant warming
                weather_adjustment -= temp_change * self.weather_forecast_coeff * decay * 0.1  # REDUCED
            elif temp_change < -1.5:  # Significant cooling  
                weather_adjustment -= temp_change * self.weather_forecast_coeff * decay * 0.06  # REDUCED
        
        # PV forecasting
        pv_adjustment = 0.0
        for i in range(4):
            forecast_pv = features.get(f'pv_forecast_{i+1}h', 0.0)
            pv_increase = max(0, forecast_pv - current_pv)
            decay = self.forecast_decay[i]
            
            if pv_increase > 200:  # Significant solar expected
                pv_adjustment -= pv_increase * self.pv_forecast_coeff * decay * 0.001
        
        return weather_adjustment + pv_adjustment
    
    def _update_histories(self, pv_power, fireplace, tv):
        """Update ring buffers for multi-lag tracking"""
        # Backward compatibility: initialize if not present (loaded old model)
        if not hasattr(self, 'pv_history'):
            if hasattr(self, '_init_multilag_attributes'):
                self._init_multilag_attributes()
            else:
                # Direct initialization for very old pickled models
                self.pv_history = []
                self.fireplace_history = []
                self.tv_history = []
                self.pv_coeffs = {'lag_1': 0.0, 'lag_2': 0.0,
                                 'lag_3': 0.0, 'lag_4': 0.0}
                self.fireplace_coeffs = {'immediate': 0.0, 'lag_1': 0.0,
                                        'lag_2': 0.0, 'lag_3': 0.0}
                self.tv_coeffs = {'immediate': 0.0, 'lag_1': 0.0}
                self.pv_seasonal_cos = 0.0
                self.pv_seasonal_sin = 0.0
                self.tv_seasonal_cos = 0.0
                self.tv_seasonal_sin = 0.0
                self.seasonal_heating_cos = 0.0
                self.seasonal_heating_sin = 0.0
                self.seasonal_target_cos = 0.0
                self.seasonal_target_sin = 0.0
                self.seasonal_outdoor_cos = 0.0
                self.seasonal_outdoor_sin = 0.0
                self.pv_effects_detailed = []
                self.fireplace_effects_detailed = []
                self.tv_effects_detailed = []
                self.seasonal_performance = []
                self.hvac_off_tracking = []
        
        self.pv_history.append(pv_power)
        if len(self.pv_history) > 5:
            self.pv_history.pop(0)
        
        self.fireplace_history.append(fireplace)
        if len(self.fireplace_history) > 4:
            self.fireplace_history.pop(0)
        
        self.tv_history.append(tv)
        if len(self.tv_history) > 2:
            self.tv_history.pop(0)
    
    def _calculate_pv_lagged(self, month_cos, month_sin):
        """Calculate PV warming with time delays and seasonal"""
        if len(self.pv_history) < 5:
            return 0.0
        
        pv_seasonal = 1.0 + (
            self.pv_seasonal_cos * month_cos +
            self.pv_seasonal_sin * month_sin
        )
        pv_seasonal = max(0.5, min(1.5, pv_seasonal))
        
        effect = 0.0
        effect += (self.pv_history[-2] * 0.01 *
                  self.pv_coeffs['lag_1'])
        effect += (self.pv_history[-3] * 0.01 *
                  self.pv_coeffs['lag_2'])
        if len(self.pv_history) >= 4:
            effect += (self.pv_history[-4] * 0.01 *
                      self.pv_coeffs['lag_3'])
        if len(self.pv_history) >= 5:
            effect += (self.pv_history[-5] * 0.01 *
                      self.pv_coeffs['lag_4'])
        
        return effect * pv_seasonal
    
    def _calculate_fireplace_lagged(self):
        """Calculate fireplace heating with time delays"""
        if len(self.fireplace_history) < 4:
            return 0.0
        
        effect = 0.0
        effect += (self.fireplace_history[-1] *
                  self.fireplace_coeffs['immediate'])
        effect += (self.fireplace_history[-2] *
                  self.fireplace_coeffs['lag_1'])
        effect += (self.fireplace_history[-3] *
                  self.fireplace_coeffs['lag_2'])
        effect += (self.fireplace_history[-4] *
                  self.fireplace_coeffs['lag_3'])
        
        return effect
    
    def _calculate_tv_lagged(self, month_cos, month_sin):
        """Calculate TV heat with time delay and seasonal"""
        if len(self.tv_history) < 2:
            return 0.0
        
        tv_seasonal = 1.0 + (
            self.tv_seasonal_cos * month_cos +
            self.tv_seasonal_sin * month_sin
        )
        tv_seasonal = max(0.7, min(1.3, tv_seasonal))
        
        effect = 0.0
        effect += self.tv_history[-1] * self.tv_coeffs['immediate']
        effect += self.tv_history[-2] * self.tv_coeffs['lag_1']
        
        return effect * tv_seasonal
    
    def _calculate_pv_lagged_constrained(self, month_cos, month_sin, features):
        """Calculate PV warming with natural thermal physics - no artificial constraints"""
        if len(self.pv_history) < 5:
            return 0.0
        
        # NATURAL PHYSICS: Only apply thermal effects where PV energy actually existed
        # This creates natural sunrise ramp-up and sunset decay without time checks
        
        pv_seasonal = 1.0 + (
            self.pv_seasonal_cos * month_cos +
            self.pv_seasonal_sin * month_sin
        )
        pv_seasonal = max(0.5, min(1.5, pv_seasonal))
        
        effect = 0.0
        
        # Only apply lag coefficients to non-zero PV values (natural physics)
        if len(self.pv_history) >= 2 and self.pv_history[-2] > 0:  # lag_1 (30min ago)
            effect += (self.pv_history[-2] * 0.01 * min(self.pv_coeffs['lag_1'], 0.005))
            
        if len(self.pv_history) >= 3 and self.pv_history[-3] > 0:  # lag_2 (60min ago)
            effect += (self.pv_history[-3] * 0.01 * min(self.pv_coeffs['lag_2'], 0.005))
            
        if len(self.pv_history) >= 4 and self.pv_history[-4] > 0:  # lag_3 (90min ago)
            effect += (self.pv_history[-4] * 0.01 * min(self.pv_coeffs['lag_3'], 0.003))
            
        if len(self.pv_history) >= 5 and self.pv_history[-5] > 0:  # lag_4 (120min ago)
            effect += (self.pv_history[-5] * 0.01 * min(self.pv_coeffs['lag_4'], 0.001))
        
        # Apply seasonal modulation and reasonable maximum
        constrained_effect = effect * pv_seasonal
        max_pv_effect = 0.03  # Maximum 0.03°C heating from PV
        
        return min(constrained_effect, max_pv_effect)
    
    def _calculate_fireplace_lagged_constrained(self):
        """Calculate fireplace heating with time delays and magnitude constraints"""
        if len(self.fireplace_history) < 4:
            return 0.0
        
        effect = 0.0
        # Apply magnitude limits to prevent unrealistic heating
        effect += (self.fireplace_history[-1] * min(self.fireplace_coeffs['immediate'], 0.05))
        effect += (self.fireplace_history[-2] * min(self.fireplace_coeffs['lag_1'], 0.04))
        effect += (self.fireplace_history[-3] * min(self.fireplace_coeffs['lag_2'], 0.03))
        effect += (self.fireplace_history[-4] * min(self.fireplace_coeffs['lag_3'], 0.02))
        
        # Maximum fireplace effect
        max_fireplace_effect = 0.08  # Maximum 0.08°C heating from fireplace
        
        return min(effect, max_fireplace_effect)
    
    def _calculate_tv_lagged_constrained(self, month_cos, month_sin):
        """Calculate TV heat with time delay, seasonal, and magnitude constraints"""
        if len(self.tv_history) < 2:
            return 0.0
        
        tv_seasonal = 1.0 + (
            self.tv_seasonal_cos * month_cos +
            self.tv_seasonal_sin * month_sin
        )
        tv_seasonal = max(0.7, min(1.3, tv_seasonal))
        
        effect = 0.0
        # Apply magnitude limits to prevent unrealistic TV heating
        effect += self.tv_history[-1] * min(self.tv_coeffs['immediate'], 0.01)
        effect += self.tv_history[-2] * min(self.tv_coeffs['lag_1'], 0.005)
        
        constrained_effect = effect * tv_seasonal
        
        # Maximum TV effect - TVs don't generate significant heat
        max_tv_effect = 0.015  # Maximum 0.015°C heating from TV
        
        return min(constrained_effect, max_tv_effect)
    
    def _calculate_pv_simple_constrained(self, pv_now, features):
        """Calculate simple PV contribution with natural physics - no artificial constraints"""
        # NATURAL PHYSICS: Only apply thermal effects when there's actual PV energy
        if pv_now <= 0:
            return 0.0  # No energy = no thermal effect (natural!)
        
        # Constrained PV coefficient
        constrained_coeff = min(self.pv_warming_coefficient, 0.003)  # Max 0.3% per 100W
        simple_effect = pv_now * constrained_coeff * 0.01
        
        # Maximum simple PV effect
        max_simple_pv = 0.02  # Maximum 0.02°C heating from current PV
        
        return min(simple_effect, max_simple_pv)
    
    # NOTE: _estimate_hour_from_pv and _is_nighttime methods removed
    # Natural physics approach eliminates need for artificial time-of-day checks
    # PV thermal effects now naturally follow energy presence/absence
    
    def learn_one(self, features, target):
        """Learn from training data and adapt parameters"""
        self.training_count += 1
        
        # Get prediction before learning
        prediction = self.predict_one(features)
        error = target - prediction
        
        # Track external effects for learning
        self._track_external_effects(features, target)
        
        # Adapt parameters periodically
        if self.training_count % self.adaptation_frequency == 0:
            self._adapt_parameters(error)
            self._learn_external_effects()
            
            # MULTI-LAG & SEASONAL: Enhanced learning every 200 samples
            if self.training_count % 200 == 0:
                self._learn_pv_lags()
                self._learn_fireplace_lags()
                self._learn_tv_lags()
                self._learn_seasonal_variations()
    
    def _track_external_effects(self, features, actual_change):
        """Track external effects with historical context for multi-lag"""
        pv_now = features.get('pv_now', 0.0)
        fireplace_on = features.get('fireplace_on', 0.0)
        tv_on = features.get('tv_on', 0.0)
        heating_active = features.get('dhw_heating', 0.0) == 0
        month_cos = features.get('month_cos', 0.0)
        month_sin = features.get('month_sin', 0.0)
        
        # MULTI-LAG: Store detailed context with history
        if pv_now > 100 and len(self.pv_history) >= 5:
            self.pv_effects_detailed.append({
                'power_history': list(self.pv_history[-5:]),
                'actual_change': actual_change,
                'heating_active': heating_active,
                'month_cos': month_cos,
                'month_sin': month_sin
            })
        
        if fireplace_on > 0 and len(self.fireplace_history) >= 4:
            self.fireplace_effects_detailed.append({
                'on_history': list(self.fireplace_history[-4:]),
                'actual_change': actual_change,
                'heating_active': heating_active
            })
        
        if tv_on > 0 and len(self.tv_history) >= 2:
            self.tv_effects_detailed.append({
                'on_history': list(self.tv_history[-2:]),
                'actual_change': actual_change,
                'heating_active': heating_active,
                'month_cos': month_cos,
                'month_sin': month_sin
            })
        
        # SUMMER: Track when HVAC off for clean signal
        if not heating_active:
            self.hvac_off_tracking.append({
                'pv': pv_now,
                'fireplace': fireplace_on,
                'tv': tv_on,
                'actual_change': actual_change,
                'month_cos': month_cos,
                'month_sin': month_sin
            })
        
        # Keep tracking manageable
        max_track = 300
        if len(self.pv_effects_detailed) > max_track:
            self.pv_effects_detailed = self.pv_effects_detailed[-150:]
        if len(self.fireplace_effects_detailed) > max_track:
            self.fireplace_effects_detailed = (
                self.fireplace_effects_detailed[-150:])
        if len(self.tv_effects_detailed) > max_track:
            self.tv_effects_detailed = self.tv_effects_detailed[-150:]
        if len(self.hvac_off_tracking) > max_track:
            self.hvac_off_tracking = self.hvac_off_tracking[-150:]
        
        # Legacy simple tracking
        if pv_now > 100:
            self.pv_effects.append((pv_now, actual_change))
        if fireplace_on > 0:
            self.fireplace_effects.append((fireplace_on, actual_change))
        if tv_on > 0:
            self.tv_effects.append((tv_on, actual_change))
        
        max_tracking = 200
        if len(self.pv_effects) > max_tracking:
            self.pv_effects = self.pv_effects[-100:]
        if len(self.fireplace_effects) > max_tracking:
            self.fireplace_effects = self.fireplace_effects[-100:]
        if len(self.tv_effects) > max_tracking:
            self.tv_effects = self.tv_effects[-100:]
    
    def _adapt_parameters(self, error):
        """Adapt core heating parameters based on error"""
        learning_rate = 0.001
        
        if abs(error) > 0.03:  # Significant error
            self.base_heating_rate += error * learning_rate
            self.base_heating_rate = np.clip(self.base_heating_rate, 0.0005, 0.01)
            
        if abs(error) > 0.05:  # Large error
            self.target_influence += error * learning_rate * 0.5
            self.target_influence = np.clip(self.target_influence, 0.005, 0.03)
            
        if self.training_count % 500 == 0:
            logging.info(f"Adapted: heating_rate={self.base_heating_rate:.6f}, "
                        f"target_influence={self.target_influence:.6f}")
    
    def _learn_external_effects(self):
        """Learn actual effects of PV, fireplace, TV from tracked data"""
        
        # Learn PV effect
        if len(self.pv_effects) > 20:
            pv_powers, pv_changes = zip(*self.pv_effects[-50:])
            # Simple correlation learning
            avg_power = np.mean(pv_powers)
            avg_change = np.mean(pv_changes)
            if avg_power > 0 and avg_change > 0:
                learned_pv_coeff = (avg_change / avg_power) * 100  # Per 100W
                self.pv_warming_coefficient = 0.8 * self.pv_warming_coefficient + 0.2 * learned_pv_coeff
                self.pv_warming_coefficient = np.clip(self.pv_warming_coefficient, 0, 0.05)
        
        # Learn fireplace effect
        if len(self.fireplace_effects) > 10:
            _, fireplace_changes = zip(*self.fireplace_effects[-20:])
            avg_fireplace_effect = np.mean(fireplace_changes)
            if avg_fireplace_effect > 0:
                self.fireplace_heating_rate = 0.7 * self.fireplace_heating_rate + 0.3 * avg_fireplace_effect
                self.fireplace_heating_rate = np.clip(self.fireplace_heating_rate, 0, 0.1)
        
        # Learn TV effect
        if len(self.tv_effects) > 10:
            _, tv_changes = zip(*self.tv_effects[-20:])
            avg_tv_effect = np.mean(tv_changes)
            if avg_tv_effect > 0:
                self.tv_heat_contribution = 0.8 * self.tv_heat_contribution + 0.2 * avg_tv_effect
                self.tv_heat_contribution = np.clip(self.tv_heat_contribution, 0, 0.02)
        
        if self.training_count % 500 == 0:
            logging.info(f"Learned effects: PV={self.pv_warming_coefficient:.6f}, "
                        f"fireplace={self.fireplace_heating_rate:.6f}, "
                        f"TV={self.tv_heat_contribution:.6f}")
    
    def _learn_pv_lags(self):
        """Learn PV time-delayed effects via correlation"""
        if len(self.pv_effects_detailed) < 50:
            return
        
        recent = self.pv_effects_detailed[-100:]
        changes = np.array([e['actual_change'] for e in recent])
        
        # Calculate correlations for each lag
        correlations = []
        std_changes = np.std(changes)
        if std_changes == 0: # Avoid division by zero if all changes are the same
            return # No meaningful correlation to calculate

        for lag_idx in range(1, 5):  # lags 1-4 (t-30 to t-120)
            lag_powers = np.array([e['power_history'][-1-lag_idx]
                                  for e in recent])
            
            std_lag_powers = np.std(lag_powers)
            if std_lag_powers == 0: # If no variation in lag_powers, correlation is 0
                corr = 0.0
            else:
                # Calculate correlation, handling potential NaN/inf with nan_to_num
                corr = np.nan_to_num(np.corrcoef(lag_powers, changes)[0, 1])
            correlations.append(max(0, corr))
        
        # Distribute total effect across lags by correlation
        if sum(correlations) > 0:
            weights = np.array(correlations) / sum(correlations)
            total_effect = np.mean([abs(e['actual_change'])
                                   for e in recent])
            
            lag_keys = ['lag_1', 'lag_2', 'lag_3', 'lag_4']
            for i, key in enumerate(lag_keys):
                learned = weights[i] * total_effect * 0.01
                self.pv_coeffs[key] = (0.85 * self.pv_coeffs[key] +
                                      0.15 * learned)
                self.pv_coeffs[key] = np.clip(self.pv_coeffs[key],
                                             0, 0.1)
    
    def _learn_fireplace_lags(self):
        """Learn fireplace time-delayed effects"""
        if len(self.fireplace_effects_detailed) < 20:
            return
        
        recent = self.fireplace_effects_detailed[-50:]
        changes = np.array([e['actual_change'] for e in recent])

        std_changes = np.std(changes)
        if std_changes == 0:  # Avoid division by zero if all changes are the same
            return  # No meaningful correlation to calculate
        
        correlations = []
        for lag_idx in range(4):  # immediate, lag1, lag2, lag3
            lag_on = np.array([e['on_history'][-1-lag_idx]
                              for e in recent])
            
            std_lag_on = np.std(lag_on)
            if std_lag_on == 0: # If no variation in lag_on, correlation is 0
                corr = 0.0
            else:
                # Calculate correlation, handling potential NaN/inf with nan_to_num
                corr = np.nan_to_num(np.corrcoef(lag_on, changes)[0, 1])
            correlations.append(max(0, corr))
        
        if sum(correlations) > 0:
            weights = np.array(correlations) / sum(correlations)
            total_effect = np.mean([abs(e['actual_change'])
                                   for e in recent])
            
            lag_keys = ['immediate', 'lag_1', 'lag_2', 'lag_3']
            for i, key in enumerate(lag_keys):
                learned = weights[i] * total_effect
                self.fireplace_coeffs[key] = (0.85 *
                                             self.fireplace_coeffs[key] +
                                             0.15 * learned)
                self.fireplace_coeffs[key] = np.clip(
                    self.fireplace_coeffs[key], 0, 0.15)
    
    def _learn_tv_lags(self):
        """Learn TV time-delayed effects"""
        if len(self.tv_effects_detailed) < 20:
            return
        
        recent = self.tv_effects_detailed[-50:]
        changes = np.array([e['actual_change'] for e in recent])

        std_changes = np.std(changes)
        if std_changes == 0:  # Avoid division by zero if all changes are the same
            return  # No meaningful correlation to calculate
        
        correlations = []
        for lag_idx in range(2):  # immediate, lag1
            lag_on = np.array([e['on_history'][-1-lag_idx]
                              for e in recent])
            
            std_lag_on = np.std(lag_on)
            if std_lag_on == 0: # If no variation in lag_on, correlation is 0
                corr = 0.0
            else:
                # Calculate correlation, handling potential NaN/inf with nan_to_num
                corr = np.nan_to_num(np.corrcoef(lag_on, changes)[0, 1])
            correlations.append(max(0, corr))
        
        if sum(correlations) > 0:
            weights = np.array(correlations) / sum(correlations)
            total_effect = np.mean([abs(e['actual_change'])
                                   for e in recent])
            
            lag_keys = ['immediate', 'lag_1']
            for i, key in enumerate(lag_keys):
                learned = weights[i] * total_effect
                self.tv_coeffs[key] = (0.85 * self.tv_coeffs[key] +
                                      0.15 * learned)
                self.tv_coeffs[key] = np.clip(self.tv_coeffs[key],
                                             0, 0.03)
    
    def _learn_seasonal_variations(self):
        """Learn seasonal modulation of external sources"""
        if len(self.hvac_off_tracking) < 100:
            return
        
        recent = self.hvac_off_tracking[-200:]
        
        # Learn PV seasonal (±50%)
        pv_samples = [e for e in recent if e['pv'] > 100]
        if len(pv_samples) > 30:
            changes = np.array([s['actual_change'] for s in pv_samples])
            cos_vals = np.array([s['month_cos'] for s in pv_samples])
            sin_vals = np.array([s['month_sin'] for s in pv_samples])
            
            # Simple linear regression for cos/sin components
            std_changes = np.std(changes)
            if std_changes > 0.01:
                std_cos_vals = np.std(cos_vals)
                std_sin_vals = np.std(sin_vals)

                cos_corr = 0.0
                if std_cos_vals > 0:
                    cos_corr = np.nan_to_num(np.corrcoef(cos_vals, changes)[0, 1])

                sin_corr = 0.0
                if std_sin_vals > 0:
                    sin_corr = np.nan_to_num(np.corrcoef(sin_vals, changes)[0, 1])
                
                cos_coeff = cos_corr * 0.3
                sin_coeff = sin_corr * 0.3
                
                self.pv_seasonal_cos = (0.9 * self.pv_seasonal_cos +
                                        0.1 * cos_coeff)
                self.pv_seasonal_sin = (0.9 * self.pv_seasonal_sin +
                                        0.1 * sin_coeff)
                self.pv_seasonal_cos = np.clip(self.pv_seasonal_cos,
                                                -0.5, 0.5)
                self.pv_seasonal_sin = np.clip(self.pv_seasonal_sin,
                                                -0.5, 0.5)
        
        # Learn TV seasonal (±30%)
        tv_samples = [e for e in recent if e['tv'] > 0]
        if len(tv_samples) > 30:
            changes = np.array([s['actual_change'] for s in tv_samples])
            cos_vals = np.array([s['month_cos'] for s in tv_samples])
            sin_vals = np.array([s['month_sin'] for s in tv_samples])
            
            std_changes = np.std(changes)
            if std_changes > 0.01:
                std_cos_vals = np.std(cos_vals)
                std_sin_vals = np.std(sin_vals)
                
                cos_corr = 0.0
                if std_cos_vals > 0:
                    cos_corr = np.nan_to_num(np.corrcoef(cos_vals, changes)[0, 1])

                sin_corr = 0.0
                if std_sin_vals > 0:
                    sin_corr = np.nan_to_num(np.corrcoef(sin_vals, changes)[0, 1])
                
                cos_coeff = cos_corr * 0.2
                sin_coeff = sin_corr * 0.2
                
                self.tv_seasonal_cos = (0.9 * self.tv_seasonal_cos +
                                        0.1 * cos_coeff)
                self.tv_seasonal_sin = (0.9 * self.tv_seasonal_sin +
                                        0.1 * sin_coeff)
                self.tv_seasonal_cos = np.clip(self.tv_seasonal_cos,
                                                -0.3, 0.3)
                self.tv_seasonal_sin = np.clip(self.tv_seasonal_sin,
                                                -0.3, 0.3)
    
    def track_prediction_error(self, predicted_change, actual_change):
        """Track prediction error for real-time performance monitoring"""
        # Initialize prediction_errors if missing (backward compatibility)
        if not hasattr(self, 'prediction_errors'):
            self.prediction_errors = []
            self.max_error_history = 50
        
        error = abs(predicted_change - actual_change)
        self.prediction_errors.append(error)
        
        # Maintain rolling window
        if len(self.prediction_errors) > self.max_error_history:
            self.prediction_errors.pop(0)
    
    def get_realtime_sigma(self):
        """Calculate real-time uncertainty (sigma) based on recent prediction errors"""
        # Initialize prediction_errors if missing (backward compatibility)
        if not hasattr(self, 'prediction_errors'):
            self.prediction_errors = []
            self.max_error_history = 50
        
        if len(self.prediction_errors) < 5:
            # Not enough data yet, use conservative estimate
            return 0.15  # Higher uncertainty when starting
        
        # Calculate standard deviation of recent errors
        sigma = np.std(self.prediction_errors)
        
        # Add minimum uncertainty to prevent overconfidence
        min_sigma = 0.02
        sigma = max(sigma, min_sigma)
        
        # Cap maximum uncertainty to prevent total lack of confidence
        max_sigma = 0.5
        sigma = min(sigma, max_sigma)
        
        return sigma
    
    def get_realtime_confidence(self):
        """Calculate real-time confidence based on recent prediction accuracy"""
        sigma = self.get_realtime_sigma()
        confidence = 1.0 / (1.0 + sigma)
        return confidence
    
    def export_learning_metrics(self):
        """Export current learning parameters for InfluxDB monitoring"""
        # Initialize prediction_errors if missing (backward compatibility)
        if not hasattr(self, 'prediction_errors'):
            self.prediction_errors = []
            self.max_error_history = 50
        
        metrics = {
            # Core parameters
            'base_heating_rate': self.base_heating_rate,
            'target_influence': self.target_influence,
            'outdoor_factor': self.outdoor_factor,
            
            # Simple external source coefficients (legacy)
            'pv_warming_coefficient': self.pv_warming_coefficient,
            'fireplace_heating_rate': self.fireplace_heating_rate,
            'tv_heat_contribution': self.tv_heat_contribution,
            
            # Multi-lag PV coefficients
            'pv_lag_1_30min': self.pv_coeffs['lag_1'],
            'pv_lag_2_60min': self.pv_coeffs['lag_2'],
            'pv_lag_3_90min': self.pv_coeffs['lag_3'],
            'pv_lag_4_120min': self.pv_coeffs['lag_4'],
            'pv_total_effect': sum(self.pv_coeffs.values()),
            
            # Multi-lag Fireplace coefficients
            'fireplace_immediate': self.fireplace_coeffs['immediate'],
            'fireplace_lag_1_30min': self.fireplace_coeffs['lag_1'],
            'fireplace_lag_2_60min': self.fireplace_coeffs['lag_2'],
            'fireplace_lag_3_90min': self.fireplace_coeffs['lag_3'],
            'fireplace_total_effect': sum(self.fireplace_coeffs.values()),
            
            # Multi-lag TV coefficients
            'tv_immediate': self.tv_coeffs['immediate'],
            'tv_lag_1_30min': self.tv_coeffs['lag_1'],
            'tv_total_effect': sum(self.tv_coeffs.values()),
            
            # Seasonal modulation
            'pv_seasonal_cos': self.pv_seasonal_cos,
            'pv_seasonal_sin': self.pv_seasonal_sin,
            'tv_seasonal_cos': self.tv_seasonal_cos,
            'tv_seasonal_sin': self.tv_seasonal_sin,
            
            # Live performance metrics
            'realtime_sigma': self.get_realtime_sigma(),
            'realtime_confidence': self.get_realtime_confidence(),
            'prediction_error_count': len(self.prediction_errors),
            'avg_prediction_error': np.mean(self.prediction_errors) if self.prediction_errors else 0.0,
            
            # Sample counts
            'training_count': self.training_count,
            'pv_samples': len(self.pv_effects_detailed),
            'fireplace_samples': len(self.fireplace_effects_detailed),
            'tv_samples': len(self.tv_effects_detailed),
            'hvac_off_samples': len(self.hvac_off_tracking),
        }
        
        return metrics
    
    @property
    def steps(self):
        """Compatibility property"""
        return {'features': None, 'learn': self}
