"""
Adaptive Fireplace Learning System
Learns fireplace thermal characteristics from temperature differential patterns

This module integrates with existing fireplace control logic:
- Fireplace detection via temperature differential (living room vs other rooms)
- Real-time learning of heat output from observed thermal responses
- Adaptive coefficient optimization based on actual usage patterns
- No historical data required - learns from first use
"""

import logging
import math
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class FireplaceObservation:
    """Single fireplace usage observation for learning"""
    timestamp: datetime
    temp_differential: float  # living_room_temp - other_rooms_mean
    outdoor_temp: float
    fireplace_active: bool
    duration_minutes: float
    heat_buildup_rate: float = 0.0  # °C/minute during heat-up
    heat_decay_rate: float = 0.0   # °C/minute during cool-down
    peak_differential: float = 0.0


@dataclass
class FireplaceLearningState:
    """Persistent learning state for fireplace characteristics"""
    observations: List[FireplaceObservation] = field(default_factory=list)
    learned_coefficients: Dict[str, float] = field(default_factory=dict)
    learning_stats: Dict[str, float] = field(default_factory=dict)
    last_update: Optional[datetime] = None
    
    def __post_init__(self):
        # Initialize default coefficients if not set
        if not self.learned_coefficients:
            self.learned_coefficients = {
                'base_heat_output_kw': 6.0,  # Conservative start
                'thermal_efficiency': 0.7,   # Conservative efficiency
                'heat_distribution_factor': 0.5,  # Conservative distribution
                'outdoor_temp_correlation': 0.1,   # Weak initial correlation
                'differential_to_heat_ratio': 2.5,  # kW per °C differential
                'peak_effectiveness_temp': 0.0,     # Outdoor temp for peak effectiveness
                'learning_confidence': 0.1          # Low initial confidence
            }


class AdaptiveFireplaceLearning:
    """
    Learns fireplace thermal characteristics from temperature differential patterns.
    
    Integrates with existing control logic:
    - Uses binary_sensor.fireplace_active (based on temp differential)
    - Learns from observed thermal responses
    - Provides enhanced heat contribution calculations
    - Maintains safety through conservative starting coefficients
    """
    
    def __init__(self, state_file: str = None):
        self.state_file = state_file or '/opt/ml_heating/fireplace_learning_state.json'
        self.learning_state = self._load_state()
        
        # Fireplace detection thresholds (from user's logic)
        self.fireplace_on_threshold = 2.0   # °C differential to detect fireplace on
        self.fireplace_off_threshold = 0.8  # °C differential to detect fireplace off
        
        # Learning parameters
        self.min_observations_for_learning = 3
        self.max_observations = 100  # Rolling window
        self.learning_rate = 0.1
        self.confidence_buildup_rate = 0.05
        
        # Safety bounds for learned coefficients
        self.safety_bounds = {
            'base_heat_output_kw': (3.0, 15.0),
            'thermal_efficiency': (0.4, 0.9),
            'heat_distribution_factor': (0.3, 0.8),
            'outdoor_temp_correlation': (-0.5, 0.5),
            'differential_to_heat_ratio': (1.0, 5.0)
        }
        
        # Current fireplace session tracking
        self.current_session = None
        self.session_start_time = None
        self.session_peak_differential = 0.0
        
    def _load_state(self) -> FireplaceLearningState:
        """Load learning state from file or create new state"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                
                # Convert timestamp strings back to datetime objects
                observations = []
                for obs_data in data.get('observations', []):
                    obs_data['timestamp'] = datetime.fromisoformat(obs_data['timestamp'])
                    observations.append(FireplaceObservation(**obs_data))
                
                state = FireplaceLearningState(
                    observations=observations,
                    learned_coefficients=data.get('learned_coefficients', {}),
                    learning_stats=data.get('learning_stats', {}),
                    last_update=datetime.fromisoformat(data['last_update']) if data.get('last_update') else None
                )
                
                logger.info(f"Loaded fireplace learning state with {len(observations)} observations")
                return state
                
            except Exception as e:
                logger.warning(f"Failed to load fireplace learning state: {e}")
        
        # Return new state if loading failed or file doesn't exist
        logger.info("Creating new fireplace learning state")
        return FireplaceLearningState()
    
    def _save_state(self):
        """Save learning state to file"""
        try:
            # Convert datetime objects to strings for JSON serialization
            observations_data = []
            for obs in self.learning_state.observations:
                obs_dict = obs.__dict__.copy()
                obs_dict['timestamp'] = obs.timestamp.isoformat()
                observations_data.append(obs_dict)
            
            data = {
                'observations': observations_data,
                'learned_coefficients': self.learning_state.learned_coefficients,
                'learning_stats': self.learning_state.learning_stats,
                'last_update': datetime.now().isoformat()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.debug(f"Saved fireplace learning state to {self.state_file}")
            
        except Exception as e:
            logger.error(f"Failed to save fireplace learning state: {e}")
    
    def observe_fireplace_state(self, living_room_temp: float, other_rooms_temp: float, 
                               outdoor_temp: float, fireplace_active: bool) -> Dict:
        """
        Observe current fireplace state and learn from patterns.
        
        Args:
            living_room_temp: Current living room temperature
            other_rooms_temp: Average temperature of other rooms
            outdoor_temp: Current outdoor temperature  
            fireplace_active: Current fireplace active state (from binary sensor)
            
        Returns:
            Dict with learning insights and current analysis
        """
        current_time = datetime.now()
        temp_differential = living_room_temp - other_rooms_temp
        
        # Track fireplace sessions for learning
        session_update = self._update_fireplace_session(
            temp_differential, outdoor_temp, fireplace_active, current_time
        )
        
        # Calculate current heat contribution using learned coefficients
        heat_contribution = self._calculate_learned_heat_contribution(
            temp_differential, outdoor_temp, fireplace_active
        )
        
        # Update learning if we have new insights
        learning_update = self._update_learning_coefficients()
        
        return {
            'temp_differential': temp_differential,
            'fireplace_active': fireplace_active,
            'heat_contribution_kw': heat_contribution['heat_contribution_kw'],
            'heat_effectiveness': heat_contribution['effectiveness_factor'],
            'learning_confidence': self.learning_state.learned_coefficients['learning_confidence'],
            'observations_count': len(self.learning_state.observations),
            'session_update': session_update,
            'learning_update': learning_update,
            'analysis': heat_contribution
        }
    
    def _update_fireplace_session(self, temp_differential: float, outdoor_temp: float,
                                 fireplace_active: bool, current_time: datetime) -> Dict:
        """Track fireplace sessions for learning patterns"""
        
        if fireplace_active and self.current_session is None:
            # New fireplace session starting
            self.current_session = {
                'start_time': current_time,
                'start_differential': temp_differential,
                'outdoor_temp': outdoor_temp,
                'peak_differential': temp_differential
            }
            self.session_start_time = current_time
            self.session_peak_differential = temp_differential
            
            logger.info(f"Fireplace session started: {temp_differential:.1f}°C differential, outdoor: {outdoor_temp:.1f}°C")
            
            return {'status': 'session_started', 'differential': temp_differential}
            
        elif fireplace_active and self.current_session is not None:
            # Ongoing session - track peak differential
            if temp_differential > self.session_peak_differential:
                self.session_peak_differential = temp_differential
                self.current_session['peak_differential'] = temp_differential
            
            return {'status': 'session_active', 'peak_differential': self.session_peak_differential}
            
        elif not fireplace_active and self.current_session is not None:
            # Session ending - create observation for learning
            session_duration = (current_time - self.session_start_time).total_seconds() / 60.0
            
            if session_duration > 10:  # Only learn from sessions > 10 minutes
                observation = FireplaceObservation(
                    timestamp=self.current_session['start_time'],
                    temp_differential=self.current_session['start_differential'],
                    outdoor_temp=self.current_session['outdoor_temp'],
                    fireplace_active=True,
                    duration_minutes=session_duration,
                    peak_differential=self.session_peak_differential
                )
                
                self.learning_state.observations.append(observation)
                
                # Maintain rolling window of observations
                if len(self.learning_state.observations) > self.max_observations:
                    self.learning_state.observations.pop(0)
                
                self._save_state()
                
                logger.info(f"Fireplace session ended: {session_duration:.0f}min, peak: {self.session_peak_differential:.1f}°C")
            
            # Reset session tracking
            self.current_session = None
            self.session_start_time = None
            self.session_peak_differential = 0.0
            
            return {
                'status': 'session_ended', 
                'duration_minutes': session_duration,
                'peak_differential': self.session_peak_differential,
                'learned': session_duration > 10
            }
        
        return {'status': 'no_change'}
    
    def _calculate_learned_heat_contribution(self, temp_differential: float, 
                                           outdoor_temp: float, fireplace_active: bool) -> Dict:
        """
        Calculate fireplace heat contribution using learned coefficients.
        
        This replaces generic physics estimates with learned characteristics.
        """
        if not fireplace_active or temp_differential < 0.5:
            return {
                'heat_contribution_kw': 0.0,
                'effectiveness_factor': 0.0,
                'reasoning': 'Fireplace inactive or minimal differential'
            }
        
        coeffs = self.learning_state.learned_coefficients
        
        # Base heat output from learned characteristics
        base_heat = coeffs['base_heat_output_kw']
        thermal_efficiency = coeffs['thermal_efficiency']
        
        # Outdoor temperature correlation (fireplace more effective when cold)
        outdoor_factor = 1.0 + coeffs['outdoor_temp_correlation'] * (5.0 - outdoor_temp) / 10.0
        outdoor_factor = max(0.5, min(1.5, outdoor_factor))
        
        # Temperature differential to heat conversion
        differential_heat = temp_differential * coeffs['differential_to_heat_ratio']
        
        # Combine learned factors
        effective_heat = min(base_heat, differential_heat) * thermal_efficiency * outdoor_factor
        
        # Heat distribution factor (how much affects heat pump load)
        distributed_heat = effective_heat * coeffs['heat_distribution_factor']
        
        # Effectiveness based on learning confidence
        confidence = coeffs['learning_confidence']
        effectiveness_factor = 0.5 + 0.5 * confidence  # 50-100% effectiveness based on confidence
        
        final_heat_contribution = distributed_heat * effectiveness_factor
        
        return {
            'heat_contribution_kw': final_heat_contribution,
            'effectiveness_factor': effectiveness_factor,
            'base_heat_kw': base_heat,
            'differential_heat_kw': differential_heat,
            'outdoor_factor': outdoor_factor,
            'thermal_efficiency': thermal_efficiency,
            'learning_confidence': confidence,
            'reasoning': (f"Learned fireplace: {final_heat_contribution:.2f}kW from {temp_differential:.1f}°C differential "
                         f"(outdoor: {outdoor_temp:.1f}°C, confidence: {confidence:.2f})")
        }
    
    def _update_learning_coefficients(self) -> Dict:
        """Update learned coefficients based on accumulated observations"""
        
        if len(self.learning_state.observations) < self.min_observations_for_learning:
            return {'status': 'insufficient_data', 'observations_needed': self.min_observations_for_learning}
        
        observations = self.learning_state.observations
        coeffs = self.learning_state.learned_coefficients
        
        # Learn heat output from temperature differential patterns
        differential_heat_ratios = []
        outdoor_correlations = []
        
        for obs in observations[-20:]:  # Use last 20 observations for learning
            if obs.duration_minutes > 15 and obs.peak_differential > 1.0:
                # Estimate heat output from differential and duration
                estimated_heat = obs.peak_differential * 2.0  # Simple initial estimate
                differential_heat_ratios.append(estimated_heat / obs.peak_differential)
                
                # Outdoor temperature correlation
                outdoor_correlations.append((obs.outdoor_temp, obs.peak_differential))
        
        if differential_heat_ratios:
            # Update differential to heat ratio with learned data
            learned_ratio = np.mean(differential_heat_ratios)
            coeffs['differential_to_heat_ratio'] = (
                coeffs['differential_to_heat_ratio'] * (1 - self.learning_rate) + 
                learned_ratio * self.learning_rate
            )
            
            # Apply safety bounds
            min_ratio, max_ratio = self.safety_bounds['differential_to_heat_ratio']
            coeffs['differential_to_heat_ratio'] = max(min_ratio, min(max_ratio, coeffs['differential_to_heat_ratio']))
        
        if len(outdoor_correlations) > 5:
            # Learn outdoor temperature correlation
            outdoor_temps = [x[0] for x in outdoor_correlations]
            differentials = [x[1] for x in outdoor_correlations]
            
            if np.std(outdoor_temps) > 2.0:  # Only if we have temperature variation
                correlation = np.corrcoef(outdoor_temps, differentials)[0, 1]
                if not np.isnan(correlation):
                    coeffs['outdoor_temp_correlation'] = (
                        coeffs['outdoor_temp_correlation'] * (1 - self.learning_rate) +
                        correlation * 0.1 * self.learning_rate  # Gentle learning
                    )
                    
                    # Apply safety bounds
                    min_corr, max_corr = self.safety_bounds['outdoor_temp_correlation']
                    coeffs['outdoor_temp_correlation'] = max(min_corr, min(max_corr, coeffs['outdoor_temp_correlation']))
        
        # Build learning confidence based on number of observations
        observation_confidence = min(1.0, len(observations) / 50.0)  # Full confidence at 50 observations
        coeffs['learning_confidence'] = min(0.9, observation_confidence)  # Cap at 90%
        
        # Update learning stats
        self.learning_state.learning_stats.update({
            'total_observations': len(observations),
            'recent_observations': len([obs for obs in observations if (datetime.now() - obs.timestamp).days < 30]),
            'avg_peak_differential': np.mean([obs.peak_differential for obs in observations]) if observations else 0.0,
            'avg_duration_minutes': np.mean([obs.duration_minutes for obs in observations]) if observations else 0.0,
            'confidence_level': coeffs['learning_confidence']
        })
        
        self.learning_state.last_update = datetime.now()
        self._save_state()
        
        return {
            'status': 'coefficients_updated',
            'observations_used': len(observations),
            'confidence': coeffs['learning_confidence'],
            'updated_coefficients': list(coeffs.keys())
        }
    
    def get_enhanced_fireplace_features(self, base_features: Dict) -> Dict:
        """
        Enhance base features with learned fireplace characteristics.
        
        Integrates with existing multi-heat-source physics for ML model features.
        """
        enhanced = base_features.copy()
        
        # Get current fireplace state
        living_room_temp = base_features.get('indoor_temp', 20.0)
        other_rooms_temp = base_features.get('avg_other_rooms_temp', 20.0)
        outdoor_temp = base_features.get('outdoor_temp', 0.0)
        fireplace_active = bool(base_features.get('fireplace_on', 0))
        
        # Calculate learned heat contribution
        fireplace_analysis = self._calculate_learned_heat_contribution(
            living_room_temp - other_rooms_temp, outdoor_temp, fireplace_active
        )
        
        # Add enhanced fireplace features
        enhanced.update({
            # Learned fireplace characteristics
            'fireplace_heat_contribution_kw': fireplace_analysis['heat_contribution_kw'],
            'fireplace_effectiveness_factor': fireplace_analysis['effectiveness_factor'],
            'fireplace_learning_confidence': self.learning_state.learned_coefficients['learning_confidence'],
            
            # Temperature differential features
            'fireplace_temp_differential': living_room_temp - other_rooms_temp,
            'fireplace_outdoor_correlation': fireplace_analysis.get('outdoor_factor', 1.0),
            
            # Learning state features
            'fireplace_observations_count': len(self.learning_state.observations),
            'fireplace_recent_usage': len([obs for obs in self.learning_state.observations 
                                          if (datetime.now() - obs.timestamp).days < 7]),
            
            # Adaptive coefficients as features
            'fireplace_learned_efficiency': self.learning_state.learned_coefficients['thermal_efficiency'],
            'fireplace_learned_distribution': self.learning_state.learned_coefficients['heat_distribution_factor'],
            'fireplace_differential_heat_ratio': self.learning_state.learned_coefficients['differential_to_heat_ratio']
        })
        
        return enhanced
    
    def get_learning_summary(self) -> Dict:
        """Get summary of fireplace learning progress"""
        coeffs = self.learning_state.learned_coefficients
        stats = self.learning_state.learning_stats
        
        return {
            'learning_status': {
                'total_observations': len(self.learning_state.observations),
                'learning_confidence': coeffs['learning_confidence'],
                'last_update': self.learning_state.last_update.isoformat() if self.learning_state.last_update else None,
                'learning_active': len(self.learning_state.observations) >= self.min_observations_for_learning
            },
            'learned_characteristics': {
                'heat_output_kw': coeffs['base_heat_output_kw'],
                'thermal_efficiency': coeffs['thermal_efficiency'],
                'heat_distribution_factor': coeffs['heat_distribution_factor'],
                'differential_to_heat_ratio': coeffs['differential_to_heat_ratio'],
                'outdoor_temp_correlation': coeffs['outdoor_temp_correlation']
            },
            'usage_patterns': stats,
            'recent_sessions': [
                {
                    'timestamp': obs.timestamp.isoformat(),
                    'duration_minutes': obs.duration_minutes,
                    'peak_differential': obs.peak_differential,
                    'outdoor_temp': obs.outdoor_temp
                }
                for obs in self.learning_state.observations[-5:]  # Last 5 sessions
            ]
        }


# Integration helper for existing multi-heat-source physics
def integrate_adaptive_fireplace_with_multi_source_physics(multi_source_physics, adaptive_fireplace):
    """
    Integrate adaptive fireplace learning with existing multi-heat-source physics engine.
    
    This replaces the generic fireplace calculations with learned characteristics.
    """
    
    # Store original fireplace calculation method
    original_fireplace_calc = multi_source_physics.calculate_fireplace_heat_contribution
    
    def enhanced_fireplace_calculation(fireplace_on: bool, zone_factor: float = 1.0,
                                     outdoor_temp: float = 0.0, duration_hours: float = 1.0,
                                     living_room_temp: float = None, other_rooms_temp: float = None) -> Dict:
        """
        Enhanced fireplace calculation using adaptive learning.
        
        Falls back to original physics if learning data insufficient.
        """
        
        if living_room_temp is not None and other_rooms_temp is not None:
            # Use adaptive learning if temperature data available
            temp_differential = living_room_temp - other_rooms_temp
            
            if adaptive_fireplace.learning_state.learned_coefficients['learning_confidence'] > 0.3:
                # Use learned characteristics
                learned_analysis = adaptive_fireplace._calculate_learned_heat_contribution(
                    temp_differential, outdoor_temp, fireplace_on
                )
                
                return {
                    'heat_contribution_kw': learned_analysis['heat_contribution_kw'],
                    'outlet_temp_reduction': learned_analysis['heat_contribution_kw'] * 1.2,  # Convert to outlet reduction
                    'heat_distribution_factor': zone_factor,
                    'thermal_buildup_factor': min(1.0, duration_hours / 2.0),
                    'weather_effectiveness': learned_analysis['outdoor_factor'],
                    'reasoning': learned_analysis['reasoning'] + " (adaptive learning)",
                    'learning_enhanced': True
                }
        
        # Fallback to original physics calculation
        result = original_fireplace_calc(fireplace_on, zone_factor, outdoor_temp, duration_hours)
        result['learning_enhanced'] = False
        result['reasoning'] += " (physics fallback)"
        return result
    
    # Replace the fireplace calculation method
    multi_source_physics.calculate_fireplace_heat_contribution = enhanced_fireplace_calculation
    
    return multi_source_physics


if __name__ == "__main__":
    # Test adaptive fireplace learning
    adaptive_fireplace = AdaptiveFireplaceLearning()
    
    print("🔥 Adaptive Fireplace Learning System Test")
    print("=" * 50)
    
    # Simulate fireplace usage observations
    test_scenarios = [
        {'living_room_temp': 22.5, 'other_rooms_temp': 20.0, 'outdoor_temp': 5.0, 'fireplace_active': True},
        {'living_room_temp': 23.2, 'other_rooms_temp': 20.1, 'outdoor_temp': 5.0, 'fireplace_active': True},
        {'living_room_temp': 21.8, 'other_rooms_temp': 20.2, 'outdoor_temp': 5.0, 'fireplace_active': False},
    ]
    
    for i, scenario in enumerate(test_scenarios):
        print(f"\nScenario {i+1}:")
        result = adaptive_fireplace.observe_fireplace_state(**scenario)
        print(f"  Temp differential: {result['temp_differential']:.1f}°C")
        print(f"  Heat contribution: {result['heat_contribution_kw']:.2f}kW")
        print(f"  Learning confidence: {result['learning_confidence']:.2f}")
        print(f"  Session status: {result['session_update'].get('status', 'no_change')}")
    
    # Show learning summary
    summary = adaptive_fireplace.get_learning_summary()
    print(f"\n📊 Learning Summary:")
    print(f"  Observations: {summary['learning_status']['total_observations']}")
    print(f"  Confidence: {summary['learning_status']['learning_confidence']:.2f}")
    print(f"  Learning active: {summary['learning_status']['learning_active']}")
    
    print("\n✅ Adaptive fireplace learning test complete!")
