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
        
        # System state effects - learned from data
        self.pv_warming_coefficient = 0.0  # Per 100W, learned
        self.fireplace_heating_rate = 0.0  # Per hour active, learned  
        self.tv_heat_contribution = 0.0    # Electronics heat, learned
        
        # Forecast parameters
        self.weather_forecast_coeff = 0.015
        self.pv_forecast_coeff = 0.008
        self.forecast_decay = [1.0, 0.8, 0.6, 0.4]
        
        # Physics bounds
        self.min_prediction = -0.05  # Allow some cooling
        self.max_prediction = 0.25   # Reasonable heating limit
        
        # Learning tracking
        self.training_count = 0
        self.adaptation_frequency = 50  # Learn every 50 samples
        
        # Effect tracking for learning
        self.pv_effects = []
        self.fireplace_effects = []
        self.tv_effects = []
        
    def predict_one(self, features):
        """Enhanced physics-based prediction using target temperature context"""
        
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
        
        pv_contribution = pv_now * self.pv_warming_coefficient * 0.01
        fireplace_contribution = fireplace_on * self.fireplace_heating_rate
        tv_contribution = tv_on * self.tv_heat_contribution
        
        # Forecast adjustments
        forecast_effect = self._calculate_forecast_adjustment(features)
        
        # Total prediction
        total_effect = (base_heating + target_boost + weather_adjustment + 
                       pv_contribution + fireplace_contribution + 
                       tv_contribution + forecast_effect)
        
        # Apply physics bounds
        return np.clip(total_effect, self.min_prediction, self.max_prediction)
    
    def _calculate_forecast_adjustment(self, features):
        """Calculate forecast-based heating adjustments"""
        current_outdoor = features.get('outdoor_temp', 5.0)
        current_pv = features.get('pv_now', 0.0)
        
        # Weather forecasting
        weather_adjustment = 0.0
        for i in range(4):
            forecast_temp = features.get(f'temp_forecast_{i+1}h', current_outdoor)
            temp_change = forecast_temp - current_outdoor
            decay = self.forecast_decay[i]
            
            if temp_change > 1.5:  # Significant warming
                weather_adjustment -= temp_change * self.weather_forecast_coeff * decay
            elif temp_change < -1.5:  # Significant cooling  
                weather_adjustment -= temp_change * self.weather_forecast_coeff * decay * 0.6
        
        # PV forecasting
        pv_adjustment = 0.0
        for i in range(4):
            forecast_pv = features.get(f'pv_forecast_{i+1}h', 0.0)
            pv_increase = max(0, forecast_pv - current_pv)
            decay = self.forecast_decay[i]
            
            if pv_increase > 200:  # Significant solar expected
                pv_adjustment -= pv_increase * self.pv_forecast_coeff * decay * 0.001
        
        return weather_adjustment + pv_adjustment
    
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
    
    def _track_external_effects(self, features, actual_change):
        """Track external heat source effects for learning"""
        pv_now = features.get('pv_now', 0.0)
        fireplace_on = features.get('fireplace_on', 0.0)
        tv_on = features.get('tv_on', 0.0)
        
        # Store effects when sources are active
        if pv_now > 100:  # Significant PV
            self.pv_effects.append((pv_now, actual_change))
        
        if fireplace_on > 0:  # Fireplace active
            self.fireplace_effects.append((fireplace_on, actual_change))
            
        if tv_on > 0:  # TV active
            self.tv_effects.append((tv_on, actual_change))
        
        # Keep effect tracking manageable
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
    
    @property
    def steps(self):
        """Compatibility property"""
        return {'features': None, 'learn': self}
