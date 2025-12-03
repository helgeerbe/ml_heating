"""
Multi-Heat-Source Physics Integration for Week 2 Enhancement

This module provides intelligent heat source coordination algorithms that replace
binary heat source flags with sophisticated heat contribution calculations.

Key Features:
- PV solar warming calculation based on building thermal characteristics
- Fireplace heat equivalent estimation with zone-based distribution
- TV/electronics heat including occupancy warming effects
- Combined multi-source decision engine for optimal outlet temperature
- Physics-based heat contribution modeling

This transforms the ML Heating System from single-variable control (outdoor temp)
to comprehensive multi-variable optimization for ±0.1°C temperature stability.
"""

import logging
import math
from typing import Dict, Optional, Tuple
from datetime import datetime

# Support both package-relative and direct import
try:
    from . import config
except ImportError:
    import config


class MultiHeatSourcePhysics:
    """
    Advanced physics calculations for multi-heat-source integration.
    
    Replaces simple binary flags (fireplace_on, tv_on) with sophisticated
    heat contribution analysis that enables intelligent heat pump coordination.
    """
    
    def __init__(self):
        # Building thermal characteristics (configurable per installation)
        self.building_thermal_mass = 3500.0  # kWh/°C - building thermal capacity
        self.heat_pump_cop = 3.5  # Coefficient of Performance estimate
        self.zone_heat_distribution = 0.7  # Primary zone receives 70% of localized heat
        
        # PV solar warming parameters (calibrated from real data)
        self.pv_building_heating_factor = 0.25  # 25% of PV becomes building heat
        self.pv_thermal_efficiency_base = 0.8  # Base thermal conversion efficiency
        self.pv_thermal_decay_factor = 0.05  # Efficiency loss per temperature differential
        
        # Fireplace heat characteristics (physics-based estimates)
        self.fireplace_heat_output_kw = 8.0  # 8kW peak heat output
        self.fireplace_thermal_efficiency = 0.75  # 75% heat transfer to room air
        self.fireplace_heat_distribution_factor = 0.6  # 60% distributed beyond immediate area
        
        # Electronics and occupancy heat (calibrated estimates)
        self.tv_electronics_base_heat = 250  # 250W from modern TV + electronics
        self.tv_occupancy_indicator_factor = 1.2  # TV on suggests active occupancy
        self.human_body_heat_per_person = 100  # 100W per person body heat
        self.occupancy_activity_multiplier = 1.5  # Active vs passive occupancy
        
        # System state heat impacts
        self.dhw_heat_reduction_factor = 0.15  # 15% less heating available during DHW
        self.defrost_heat_reduction_factor = 0.35  # 35% less heating during defrost
        self.boost_heater_heat_addition = 2.0  # 2kW additional from boost heater
        
        # Thermal momentum and lag factors
        self.heat_source_thermal_lag_minutes = {
            'pv': 30,  # PV solar warming has 30min thermal lag
            'fireplace': 15,  # Fireplace has immediate but distributed effect
            'tv': 45,  # Electronics have slower thermal impact
            'dhw': 5,   # DHW impact is immediate
            'defrost': 2  # Defrost impact is immediate
        }
        
        # Weather-dependent adjustment factors
        self.outdoor_temp_heat_effectiveness = {
            'very_cold': {'threshold': -10, 'factor': 1.3},  # More heat retention in extreme cold
            'cold': {'threshold': 0, 'factor': 1.1},         # Slightly more heat retention
            'mild': {'threshold': 10, 'factor': 1.0},        # Baseline effectiveness
            'warm': {'threshold': 20, 'factor': 0.8}         # Less heat retention in warm weather
        }
        
    def calculate_pv_heat_contribution(self, pv_power: float, indoor_temp: float, 
                                     outdoor_temp: float, time_of_day: Optional[int] = None) -> Dict:
        """
        Calculate equivalent heating contribution from PV solar warming effect.
        
        PV panels warm the building through:
        1. Direct solar heating of roof/walls where panels are mounted
        2. Reduced heat loss through building envelope thermal bridging
        3. Infrared radiation from warm panels heating adjacent surfaces
        
        Args:
            pv_power: Current total PV power generation (W)
            indoor_temp: Current indoor temperature (°C)
            outdoor_temp: Current outdoor temperature (°C)
            time_of_day: Current hour (0-23) for thermal efficiency calculation
            
        Returns:
            Dict with heat contribution analysis
        """
        if pv_power <= 50:  # Minimal PV generation
            return {
                'heat_contribution_kw': 0.0,
                'outlet_temp_reduction': 0.0,
                'thermal_effectiveness': 0.0,
                'reasoning': f"Minimal PV ({pv_power:.0f}W) - no thermal contribution"
            }
        
        # Base heat contribution (percentage of PV power becomes building heat)
        base_heat_contribution = pv_power * self.pv_building_heating_factor / 1000.0  # Convert to kW
        
        # Thermal efficiency depends on temperature differential
        # Larger temp difference = more effective heat transfer
        temp_differential = max(0.1, indoor_temp - outdoor_temp)
        thermal_effectiveness = self.pv_thermal_efficiency_base * (
            1 - self.pv_thermal_decay_factor * max(0, 20 - temp_differential)
        )
        thermal_effectiveness = max(0.1, min(1.0, thermal_effectiveness))
        
        # Time-of-day thermal efficiency (peak solar heating midday)
        if time_of_day is not None:
            # Peak effectiveness at solar noon (12:00), minimum at night
            hour_effectiveness = 0.3 + 0.7 * max(0, math.cos(2 * math.pi * (time_of_day - 12) / 24))
        else:
            hour_effectiveness = 0.8  # Assume good solar conditions
        
        # Weather-dependent effectiveness
        weather_effectiveness = self._get_weather_heat_effectiveness(outdoor_temp)
        
        # Combined heat contribution
        effective_heat_contribution = (base_heat_contribution * thermal_effectiveness * 
                                     hour_effectiveness * weather_effectiveness)
        
        # Convert to outlet temperature reduction (heat pump doesn't need to work as hard)
        # Heat pump COP means 1kW heat contribution = outlet reduction / COP
        outlet_temp_reduction = (effective_heat_contribution * self.heat_pump_cop * 
                               1000 / self.building_thermal_mass)
        
        return {
            'heat_contribution_kw': effective_heat_contribution,
            'outlet_temp_reduction': outlet_temp_reduction,
            'thermal_effectiveness': thermal_effectiveness,
            'hour_effectiveness': hour_effectiveness,
            'weather_effectiveness': weather_effectiveness,
            'reasoning': (f"PV thermal: {effective_heat_contribution:.2f}kW from {pv_power:.0f}W "
                         f"(eff: {thermal_effectiveness:.1%}, time: {hour_effectiveness:.1%}, "
                         f"weather: {weather_effectiveness:.1%})")
        }
    
    def calculate_fireplace_heat_contribution(self, fireplace_on: bool, zone_factor: float = 1.0,
                                            outdoor_temp: float = 0.0, 
                                            duration_hours: float = 1.0) -> Dict:
        """
        Calculate equivalent heating contribution from fireplace operation.
        
        Fireplace provides significant uncontrollable heating that must be compensated
        by reducing heat pump demand to prevent overheating.
        
        Args:
            fireplace_on: Whether fireplace is currently operating
            zone_factor: Heat distribution to controlled zone (0.0-1.0)
            outdoor_temp: Outdoor temperature for effectiveness calculation
            duration_hours: How long fireplace has been on (for thermal buildup)
            
        Returns:
            Dict with fireplace heat contribution analysis
        """
        if not fireplace_on:
            return {
                'heat_contribution_kw': 0.0,
                'outlet_temp_reduction': 0.0,
                'heat_distribution_factor': 0.0,
                'reasoning': "Fireplace off - no thermal contribution"
            }
        
        # Base fireplace heat output
        base_heat_output = self.fireplace_heat_output_kw * self.fireplace_thermal_efficiency
        
        # Zone-based heat distribution (fireplace mainly heats one area)
        distributed_heat = base_heat_output * zone_factor * self.fireplace_heat_distribution_factor
        
        # Duration-based thermal buildup (fireplace takes time to heat thermal mass)
        thermal_buildup_factor = min(1.0, 0.3 + 0.7 * (duration_hours / 2.0))  # Full effect after 2 hours
        
        # Weather effectiveness (fireplace more effective in cold weather)
        weather_effectiveness = self._get_weather_heat_effectiveness(outdoor_temp)
        
        # Combined fireplace heat contribution
        effective_heat_contribution = (distributed_heat * thermal_buildup_factor * 
                                     weather_effectiveness)
        
        # Convert to heat pump outlet reduction
        outlet_temp_reduction = (effective_heat_contribution * self.heat_pump_cop * 
                               1000 / self.building_thermal_mass)
        
        return {
            'heat_contribution_kw': effective_heat_contribution,
            'outlet_temp_reduction': outlet_temp_reduction,
            'heat_distribution_factor': zone_factor,
            'thermal_buildup_factor': thermal_buildup_factor,
            'weather_effectiveness': weather_effectiveness,
            'reasoning': (f"Fireplace thermal: {effective_heat_contribution:.2f}kW "
                         f"(zone: {zone_factor:.1f}, buildup: {thermal_buildup_factor:.1%}, "
                         f"weather: {weather_effectiveness:.1%})")
        }
    
    def calculate_electronics_occupancy_heat(self, tv_on: bool, estimated_occupancy: int = None,
                                           activity_level: str = 'normal') -> Dict:
        """
        Calculate heating contribution from TV/electronics and occupancy.
        
        TV operation indicates both:
        1. Direct heat from electronics (TV, sound system, gaming, etc.)
        2. Human occupancy with body heat and activity
        
        Args:
            tv_on: Whether TV/entertainment system is on
            estimated_occupancy: Number of people estimated (None = infer from TV)
            activity_level: 'low', 'normal', 'high' activity
            
        Returns:
            Dict with electronics and occupancy heat analysis
        """
        if not tv_on:
            return {
                'heat_contribution_kw': 0.0,
                'outlet_temp_reduction': 0.0,
                'electronics_heat': 0.0,
                'occupancy_heat': 0.0,
                'reasoning': "TV off - no electronics/occupancy thermal contribution"
            }
        
        # Direct electronics heat
        electronics_heat = self.tv_electronics_base_heat / 1000.0  # Convert to kW
        
        # Infer occupancy from TV usage if not provided
        if estimated_occupancy is None:
            # TV on suggests 1-2 people actively using living areas
            estimated_occupancy = 2 if activity_level in ['normal', 'high'] else 1
        
        # Human body heat calculation
        activity_multipliers = {'low': 0.8, 'normal': 1.0, 'high': 1.3}
        activity_factor = activity_multipliers.get(activity_level, 1.0)
        
        occupancy_heat = (estimated_occupancy * self.human_body_heat_per_person * 
                         activity_factor * self.occupancy_activity_multiplier) / 1000.0  # Convert to kW
        
        # Total heat contribution
        total_heat_contribution = electronics_heat + occupancy_heat
        
        # Convert to outlet temperature reduction
        outlet_temp_reduction = (total_heat_contribution * self.heat_pump_cop * 
                               1000 / self.building_thermal_mass)
        
        return {
            'heat_contribution_kw': total_heat_contribution,
            'outlet_temp_reduction': outlet_temp_reduction,
            'electronics_heat': electronics_heat,
            'occupancy_heat': occupancy_heat,
            'estimated_occupancy': estimated_occupancy,
            'activity_factor': activity_factor,
            'reasoning': (f"Electronics+occupancy: {total_heat_contribution:.3f}kW "
                         f"(TV: {electronics_heat:.3f}kW + {estimated_occupancy}p × "
                         f"{activity_factor:.1f}: {occupancy_heat:.3f}kW)")
        }
    
    def calculate_system_state_impacts(self, dhw_heating: bool = False, dhw_disinfection: bool = False,
                                     dhw_boost_heater: bool = False, defrosting: bool = False) -> Dict:
        """
        Calculate heat pump system state impacts on available heating capacity.
        
        System states affect heat pump's ability to provide space heating by
        redirecting capacity to other functions (DHW, defrost) or adding auxiliary heat.
        
        Args:
            dhw_heating: Domestic hot water heating active
            dhw_disinfection: DHW disinfection cycle active
            dhw_boost_heater: DHW boost heater active
            defrosting: Heat pump defrost cycle active
            
        Returns:
            Dict with system state impact analysis
        """
        total_capacity_reduction = 0.0
        total_auxiliary_heat = 0.0
        active_states = []
        
        # DHW heating reduces space heating capacity
        if dhw_heating or dhw_disinfection:
            dhw_reduction = self.dhw_heat_reduction_factor
            total_capacity_reduction += dhw_reduction
            active_states.append(f"DHW ({dhw_reduction:.0%} reduction)")
        
        # Defrost significantly reduces heating capacity
        if defrosting:
            defrost_reduction = self.defrost_heat_reduction_factor
            total_capacity_reduction += defrost_reduction
            active_states.append(f"Defrost ({defrost_reduction:.0%} reduction)")
        
        # Boost heater adds auxiliary heating
        if dhw_boost_heater:
            boost_heat = self.boost_heater_heat_addition
            total_auxiliary_heat += boost_heat
            active_states.append(f"Boost heater (+{boost_heat:.1f}kW)")
        
        # Cap total reduction at 50% (system must maintain some heating)
        total_capacity_reduction = min(0.5, total_capacity_reduction)
        
        # Convert to outlet temperature adjustments
        # Reduced capacity = higher outlet temp needed for same heat output
        capacity_outlet_adjustment = total_capacity_reduction * 15.0  # 15°C per 100% reduction
        
        # Auxiliary heat = lower outlet temp needed
        auxiliary_outlet_reduction = (total_auxiliary_heat * self.heat_pump_cop * 
                                    1000 / self.building_thermal_mass)
        
        # Net system impact
        net_outlet_adjustment = capacity_outlet_adjustment - auxiliary_outlet_reduction
        
        return {
            'capacity_reduction_percent': total_capacity_reduction * 100,
            'auxiliary_heat_kw': total_auxiliary_heat,
            'net_outlet_adjustment': net_outlet_adjustment,
            'active_states': active_states,
            'reasoning': (f"System states: {', '.join(active_states) if active_states else 'Normal operation'} "
                         f"(net adjustment: {net_outlet_adjustment:+.1f}°C)")
        }
    
    def _get_weather_heat_effectiveness(self, outdoor_temp: float) -> float:
        """
        Calculate weather-dependent heat effectiveness factor.
        
        Args:
            outdoor_temp: Current outdoor temperature (°C)
            
        Returns:
            Heat effectiveness multiplier (0.5-1.5)
        """
        if outdoor_temp <= self.outdoor_temp_heat_effectiveness['very_cold']['threshold']:
            return self.outdoor_temp_heat_effectiveness['very_cold']['factor']
        elif outdoor_temp <= self.outdoor_temp_heat_effectiveness['cold']['threshold']:
            return self.outdoor_temp_heat_effectiveness['cold']['factor']
        elif outdoor_temp <= self.outdoor_temp_heat_effectiveness['mild']['threshold']:
            return self.outdoor_temp_heat_effectiveness['mild']['factor']
        else:
            return self.outdoor_temp_heat_effectiveness['warm']['factor']
    
    def calculate_combined_heat_sources(self, pv_power: float = 0, fireplace_on: bool = False,
                                      tv_on: bool = False, indoor_temp: float = 20.0,
                                      outdoor_temp: float = 0.0, dhw_heating: bool = False,
                                      dhw_disinfection: bool = False, dhw_boost_heater: bool = False,
                                      defrosting: bool = False, zone_factor: float = 1.0,
                                      estimated_occupancy: int = None, 
                                      activity_level: str = 'normal',
                                      living_room_temp: float = None,
                                      other_rooms_temp: float = None) -> Dict:
        """
        Calculate combined thermal contribution from all heat sources.
        
        This is the core multi-heat-source integration function that replaces
        simple binary flags with sophisticated thermal analysis.
        
        Args:
            pv_power: Current PV power generation (W)
            fireplace_on: Fireplace operating status
            tv_on: TV/electronics operating status
            indoor_temp: Current indoor temperature (°C)
            outdoor_temp: Current outdoor temperature (°C)
            dhw_heating: DHW heating active
            dhw_disinfection: DHW disinfection active
            dhw_boost_heater: DHW boost heater active
            defrosting: Heat pump defrost active
            zone_factor: Heat distribution factor for localized sources
            estimated_occupancy: Number of occupants (None = infer from TV)
            activity_level: Occupancy activity level
            
        Returns:
            Dict with comprehensive multi-source heat analysis
        """
        # Get current time for PV effectiveness calculation
        current_time = datetime.now()
        current_hour = current_time.hour
        
        # Calculate individual heat source contributions
        pv_contribution = self.calculate_pv_heat_contribution(
            pv_power, indoor_temp, outdoor_temp, current_hour
        )
        
        fireplace_contribution = self.calculate_fireplace_heat_contribution(
            fireplace_on, zone_factor, outdoor_temp
        )
        
        electronics_contribution = self.calculate_electronics_occupancy_heat(
            tv_on, estimated_occupancy, activity_level
        )
        
        system_impacts = self.calculate_system_state_impacts(
            dhw_heating, dhw_disinfection, dhw_boost_heater, defrosting
        )
        
        # Combine all thermal contributions
        total_heat_contribution = (
            pv_contribution['heat_contribution_kw'] +
            fireplace_contribution['heat_contribution_kw'] +
            electronics_contribution['heat_contribution_kw'] +
            system_impacts['auxiliary_heat_kw']
        )
        
        # Combine all outlet temperature adjustments
        total_outlet_reduction = (
            pv_contribution['outlet_temp_reduction'] +
            fireplace_contribution['outlet_temp_reduction'] +
            electronics_contribution['outlet_temp_reduction'] -
            system_impacts['net_outlet_adjustment']  # System impacts can increase outlet need
        )
        
        # Heat source diversity factor (more sources = more stable heating)
        active_sources = sum([
            1 if pv_power > 100 else 0,
            1 if fireplace_on else 0,
            1 if tv_on else 0,
            1 if any([dhw_boost_heater]) else 0  # Only boost heater adds heat
        ])
        
        diversity_factor = min(1.3, 1.0 + 0.1 * active_sources)  # Max 30% boost for diversity
        
        # Apply diversity factor to effectiveness
        effective_outlet_reduction = total_outlet_reduction * diversity_factor
        
        # Thermal coordination analysis
        coordination_analysis = self._analyze_heat_source_coordination(
            pv_contribution, fireplace_contribution, electronics_contribution, system_impacts
        )
        
        return {
            'total_heat_contribution_kw': total_heat_contribution,
            'total_outlet_temp_reduction': effective_outlet_reduction,
            'heat_source_diversity': active_sources,
            'diversity_factor': diversity_factor,
            'pv_contribution': pv_contribution,
            'fireplace_contribution': fireplace_contribution,
            'electronics_contribution': electronics_contribution,
            'system_impacts': system_impacts,
            'coordination_analysis': coordination_analysis,
            'multi_source_reasoning': (
                f"Multi-source thermal: {total_heat_contribution:.2f}kW total, "
                f"{effective_outlet_reduction:.1f}°C outlet reduction "
                f"({active_sources} active sources, diversity: {diversity_factor:.1f}x)"
            )
        }
    
    def _analyze_heat_source_coordination(self, pv_contrib: Dict, fireplace_contrib: Dict,
                                        electronics_contrib: Dict, system_impacts: Dict) -> Dict:
        """
        Analyze coordination between different heat sources for optimization insights.
        
        Args:
            pv_contrib: PV heat contribution analysis
            fireplace_contrib: Fireplace heat contribution analysis
            electronics_contrib: Electronics/occupancy heat contribution analysis
            system_impacts: System state impacts analysis
            
        Returns:
            Dict with coordination analysis
        """
        # Identify dominant heat source
        heat_sources = {
            'PV': pv_contrib['heat_contribution_kw'],
            'Fireplace': fireplace_contrib['heat_contribution_kw'],
            'Electronics': electronics_contrib['heat_contribution_kw'],
            'System': system_impacts['auxiliary_heat_kw']
        }
        
        dominant_source = max(heat_sources, key=heat_sources.get)
        dominant_heat = heat_sources[dominant_source]
        total_heat = sum(heat_sources.values())
        
        # Calculate heat source balance
        if total_heat > 0:
            pv_percentage = (heat_sources['PV'] / total_heat) * 100
            fireplace_percentage = (heat_sources['Fireplace'] / total_heat) * 100
            electronics_percentage = (heat_sources['Electronics'] / total_heat) * 100
            system_percentage = (heat_sources['System'] / total_heat) * 100
        else:
            pv_percentage = fireplace_percentage = electronics_percentage = system_percentage = 0
        
        # Coordination opportunities
        coordination_opportunities = []
        
        # PV + Fireplace coordination (both provide significant heat)
        if pv_contrib['heat_contribution_kw'] > 0.5 and fireplace_contrib['heat_contribution_kw'] > 1.0:
            coordination_opportunities.append("PV+Fireplace: Major heat sources active - significant outlet reduction possible")
        
        # High occupancy + electronics during PV (thermal overload risk)
        if (electronics_contrib.get('estimated_occupancy', 0) > 2 and 
            pv_contrib['heat_contribution_kw'] > 1.0 and
            electronics_contrib['heat_contribution_kw'] > 0.3):
            coordination_opportunities.append("High occupancy+PV: Overheating risk - aggressive outlet reduction recommended")
        
        # System conflicts (DHW + defrost during high external heat)
        if (system_impacts['capacity_reduction_percent'] > 20 and total_heat > 2.0):
            coordination_opportunities.append("System conflicts: Reduced capacity during high external heat - careful coordination needed")
        
        return {
            'dominant_source': dominant_source,
            'dominant_heat_kw': dominant_heat,
            'heat_distribution': {
                'PV': pv_percentage,
                'Fireplace': fireplace_percentage,
                'Electronics': electronics_percentage,
                'System': system_percentage
            },
            'coordination_opportunities': coordination_opportunities,
            'thermal_balance': 'balanced' if dominant_heat < total_heat * 0.6 else 'dominated',
            'analysis_summary': (f"Heat sources: {dominant_source} dominant ({dominant_heat:.1f}kW), "
                               f"PV: {pv_percentage:.0f}%, Fireplace: {fireplace_percentage:.0f}%, "
                               f"Electronics: {electronics_percentage:.0f}%, System: {system_percentage:.0f}%")
        }
    
    def calculate_optimized_outlet_temperature(self, base_outlet_temp: float, 
                                             heat_source_analysis: Dict,
                                             safety_margin: float = 2.0,
                                             min_outlet_temp: float = 16.0,
                                             max_outlet_temp: float = 65.0) -> Dict:
        """
        Calculate optimized outlet temperature accounting for all heat sources.
        
        This is the core multi-source optimization that replaces single-variable
        heat curves with comprehensive thermal intelligence.
        
        Args:
            base_outlet_temp: Base outlet temperature from physics model (°C)
            heat_source_analysis: Result from calculate_combined_heat_sources()
            safety_margin: Safety margin for outlet temperature bounds (°C)
            min_outlet_temp: Minimum allowable outlet temperature (°C)
            max_outlet_temp: Maximum allowable outlet temperature (°C)
            
        Returns:
            Dict with optimized outlet temperature and reasoning
        """
        # Start with base physics prediction
        optimized_outlet = base_outlet_temp
        
        # Apply total heat source reduction
        total_reduction = heat_source_analysis['total_outlet_temp_reduction']
        optimized_outlet -= total_reduction
        
        # Apply safety margins and bounds
        safe_outlet = max(min_outlet_temp + safety_margin, 
                         min(optimized_outlet, max_outlet_temp - safety_margin))
        
        # Calculate optimization effectiveness
        optimization_amount = base_outlet_temp - safe_outlet
        optimization_percentage = (optimization_amount / base_outlet_temp * 100) if base_outlet_temp > 0 else 0
        
        # Generate optimization reasoning
        reasoning_components = []
        
        # Significant contributors to reasoning
        if heat_source_analysis['pv_contribution']['heat_contribution_kw'] > 0.2:
            reasoning_components.append(
                f"PV: -{heat_source_analysis['pv_contribution']['outlet_temp_reduction']:.1f}°C"
            )
        
        if heat_source_analysis['fireplace_contribution']['heat_contribution_kw'] > 0.5:
            reasoning_components.append(
                f"Fireplace: -{heat_source_analysis['fireplace_contribution']['outlet_temp_reduction']:.1f}°C"
            )
        
        if heat_source_analysis['electronics_contribution']['heat_contribution_kw'] > 0.1:
            reasoning_components.append(
                f"Electronics: -{heat_source_analysis['electronics_contribution']['outlet_temp_reduction']:.1f}°C"
            )
        
        if abs(heat_source_analysis['system_impacts']['net_outlet_adjustment']) > 0.5:
            adj = heat_source_analysis['system_impacts']['net_outlet_adjustment']
            reasoning_components.append(f"System: {adj:+.1f}°C")
        
        diversity_note = f"Diversity: {heat_source_analysis['diversity_factor']:.1f}x" if heat_source_analysis['diversity_factor'] > 1.0 else ""
        
        optimization_reasoning = (
            f"Multi-source optimization: {base_outlet_temp:.1f}°C → {safe_outlet:.1f}°C "
            f"({optimization_amount:+.1f}°C, {optimization_percentage:+.0f}%) "
            f"[{', '.join(reasoning_components)}] {diversity_note}".strip()
        )
        
        return {
            'optimized_outlet_temp': safe_outlet,
            'base_outlet_temp': base_outlet_temp,
            'optimization_amount': optimization_amount,
            'optimization_percentage': optimization_percentage,
            'total_heat_contribution': heat_source_analysis['total_heat_contribution_kw'],
            'active_heat_sources': heat_source_analysis['heat_source_diversity'],
            'optimization_reasoning': optimization_reasoning,
            'heat_source_breakdown': {
                'pv_reduction': heat_source_analysis['pv_contribution']['outlet_temp_reduction'],
                'fireplace_reduction': heat_source_analysis['fireplace_contribution']['outlet_temp_reduction'],
                'electronics_reduction': heat_source_analysis['electronics_contribution']['outlet_temp_reduction'],
                'system_adjustment': heat_source_analysis['system_impacts']['net_outlet_adjustment'],
                'total_reduction': total_reduction
            }
        }


# Integration helper functions for existing physics_features.py

def enhance_physics_features_with_heat_sources(existing_features: Dict, 
                                             multi_source_physics: MultiHeatSourcePhysics) -> Dict:
    """
    Enhance existing physics features with multi-heat-source analysis.
    
    This function replaces simple binary heat source flags with sophisticated
    heat contribution calculations while maintaining backward compatibility.
    
    Args:
        existing_features: Current feature dict from physics_features.py
        multi_source_physics: Configured MultiHeatSourcePhysics instance
        
    Returns:
        Enhanced feature dict with heat source analysis
    """
    # Extract heat source states from existing features
    pv_power = existing_features.get('pv_now', 0)
    fireplace_on = bool(existing_features.get('fireplace_on', 0))
    tv_on = bool(existing_features.get('tv_on', 0))
    
    # Extract system states
    dhw_heating = bool(existing_features.get('dhw_heating', 0))
    dhw_disinfection = bool(existing_features.get('dhw_disinfection', 0))
    dhw_boost_heater = bool(existing_features.get('dhw_boost_heater', 0))
    defrosting = bool(existing_features.get('defrosting', 0))
    
    # Get temperature context
    indoor_temp = existing_features.get('indoor_temp_lag_30m', 20.0)  # Use lag for current indoor estimate
    outdoor_temp = existing_features.get('outdoor_temp', 0.0)
    
    # Calculate combined heat source analysis
    heat_source_analysis = multi_source_physics.calculate_combined_heat_sources(
        pv_power=pv_power,
        fireplace_on=fireplace_on,
        tv_on=tv_on,
        indoor_temp=indoor_temp,
        outdoor_temp=outdoor_temp,
        dhw_heating=dhw_heating,
        dhw_disinfection=dhw_disinfection,
        dhw_boost_heater=dhw_boost_heater,
        defrosting=defrosting
    )
    
    # Create enhanced features dict (copy existing features first)
    enhanced_features = existing_features.copy()
    
    # Replace binary heat source flags with sophisticated heat contribution features
    enhanced_features.update({
        # Multi-heat-source integration features (replace simple binary flags)
        'pv_heat_contribution_kw': heat_source_analysis['pv_contribution']['heat_contribution_kw'],
        'fireplace_heat_contribution_kw': heat_source_analysis['fireplace_contribution']['heat_contribution_kw'],
        'electronics_heat_contribution_kw': heat_source_analysis['electronics_contribution']['heat_contribution_kw'],
        'total_auxiliary_heat_kw': heat_source_analysis['total_heat_contribution_kw'],
        
        # Heat source outlet temperature adjustments
        'pv_outlet_reduction': heat_source_analysis['pv_contribution']['outlet_temp_reduction'],
        'fireplace_outlet_reduction': heat_source_analysis['fireplace_contribution']['outlet_temp_reduction'],
        'electronics_outlet_reduction': heat_source_analysis['electronics_contribution']['outlet_temp_reduction'],
        'total_outlet_reduction': heat_source_analysis['total_outlet_temp_reduction'],
        
        # Heat source diversity and coordination features
        'heat_source_diversity': heat_source_analysis['heat_source_diversity'],
        'heat_source_diversity_factor': heat_source_analysis['diversity_factor'],
        
        # System state impacts
        'system_capacity_reduction_percent': heat_source_analysis['system_impacts']['capacity_reduction_percent'],
        'system_auxiliary_heat_kw': heat_source_analysis['system_impacts']['auxiliary_heat_kw'],
        'system_outlet_adjustment': heat_source_analysis['system_impacts']['net_outlet_adjustment'],
        
        # Coordination analysis features
        'dominant_heat_source': _encode_heat_source(heat_source_analysis['coordination_analysis']['dominant_source']),
        'thermal_balance_score': 1.0 if heat_source_analysis['coordination_analysis']['thermal_balance'] == 'balanced' else 0.0,
        
        # Weather and effectiveness features
        'pv_thermal_effectiveness': heat_source_analysis['pv_contribution'].get('thermal_effectiveness', 0.0),
        'fireplace_thermal_buildup': heat_source_analysis['fireplace_contribution'].get('thermal_buildup_factor', 0.0),
        'electronics_occupancy_factor': heat_source_analysis['electronics_contribution'].get('activity_factor', 1.0),
    })
    
    return enhanced_features


def _encode_heat_source(dominant_source: str) -> float:
    """
    Encode dominant heat source as numeric feature for ML model.
    
    Args:
        dominant_source: Dominant heat source name
        
    Returns:
        Encoded numeric value
    """
    encoding_map = {
        'PV': 1.0,
        'Fireplace': 2.0,
        'Electronics': 3.0,
        'System': 4.0
    }
    return encoding_map.get(dominant_source, 0.0)


# Example integration pattern for existing Heat Balance Controller

def create_multi_source_heat_balance_controller_integration():
    """
    Example integration pattern for enhancing Heat Balance Controller
    with multi-heat-source intelligence.
    
    Returns:
        Integration example code as string
    """
    return """
# Enhanced Heat Balance Controller with Multi-Heat-Source Integration

class MultiSourceHeatBalanceController:
    def __init__(self):
        self.multi_source_physics = MultiHeatSourcePhysics()
        self.thermal_equilibrium_model = ThermalEquilibriumModel()
        
        # Original Heat Balance Controller settings as fallback
        self.original_charging_threshold = 0.5
        self.original_balancing_threshold = 0.2
        
    def determine_optimal_outlet_temperature(self, features_df, target_temp):
        '''Calculate optimal outlet temperature with multi-source optimization'''
        
        # Extract current features
        features = features_df.iloc[0].to_dict()
        
        # Step 1: Calculate base physics prediction (existing thermal equilibrium model)
        base_outlet = self.thermal_equilibrium_model.calculate_optimal_outlet_temperature(
            features['indoor_temp_lag_30m'],
            target_temp,
            features['outdoor_temp'],
            **features  # Pass all features for comprehensive analysis
        )['optimal_outlet_temp']
        
        # Step 2: Calculate multi-heat-source analysis
        heat_source_analysis = self.multi_source_physics.calculate_combined_heat_sources(
            pv_power=features['pv_now'],
            fireplace_on=bool(features['fireplace_on']),
            tv_on=bool(features['tv_on']),
            indoor_temp=features['indoor_temp_lag_30m'],
            outdoor_temp=features['outdoor_temp'],
            dhw_heating=bool(features['dhw_heating']),
            dhw_disinfection=bool(features['dhw_disinfection']),
            dhw_boost_heater=bool(features['dhw_boost_heater']),
            defrosting=bool(features['defrosting'])
        )
        
        # Step 3: Optimize outlet temperature with multi-source intelligence
        optimization_result = self.multi_source_physics.calculate_optimized_outlet_temperature(
            base_outlet, heat_source_analysis
        )
        
        # Step 4: Return comprehensive result
        return {
            'recommended_outlet_temp': optimization_result['optimized_outlet_temp'],
            'base_outlet_temp': base_outlet,
            'multi_source_optimization': optimization_result,
            'heat_source_analysis': heat_source_analysis,
            'reasoning': optimization_result['optimization_reasoning']
        }
        
    def get_enhanced_features(self, existing_features):
        '''Get enhanced features with multi-heat-source analysis'''
        return enhance_physics_features_with_heat_sources(
            existing_features, self.multi_source_physics
        )

# Configuration example
config = {
    'multi_source_integration_enabled': True,
    'pv_building_heating_factor': 0.25,
    'fireplace_heat_output_kw': 8.0,
    'tv_electronics_base_heat': 250,
    'building_thermal_mass': 3500.0,
    'heat_pump_cop': 3.5
}
"""


if __name__ == "__main__":
    # Test multi-heat-source physics integration
    physics = MultiHeatSourcePhysics()
    
    print("🔥 Multi-Heat-Source Physics Integration Test")
    
    # Test scenario: High PV + Fireplace + TV on cold day
    test_scenario = {
        'pv_power': 2500,  # 2.5kW solar
        'fireplace_on': True,
        'tv_on': True,
        'indoor_temp': 21.0,
        'outdoor_temp': 5.0,
        'dhw_heating': False,
        'defrosting': False
    }
    
    print(f"Test scenario: PV {test_scenario['pv_power']}W, "
          f"Fireplace: {test_scenario['fireplace_on']}, "
          f"TV: {test_scenario['tv_on']}")
    print(f"Indoor: {test_scenario['indoor_temp']}°C, "
          f"Outdoor: {test_scenario['outdoor_temp']}°C")
    
    # Calculate combined heat sources
    analysis = physics.calculate_combined_heat_sources(**test_scenario)
    
    print(f"\n🎯 Multi-Source Analysis:")
    print(f"Total heat contribution: {analysis['total_heat_contribution_kw']:.2f}kW")
    print(f"Total outlet reduction: {analysis['total_outlet_temp_reduction']:.1f}°C")
    print(f"Active heat sources: {analysis['heat_source_diversity']}")
    print(f"Diversity factor: {analysis['diversity_factor']:.1f}x")
    print(f"Reasoning: {analysis['multi_source_reasoning']}")
    
    # Test outlet optimization
    base_outlet = 45.0  # Example base outlet temperature
    optimization = physics.calculate_optimized_outlet_temperature(base_outlet, analysis)
    
    print(f"\n🚀 Outlet Optimization:")
    print(f"Base outlet: {base_outlet:.1f}°C")
    print(f"Optimized outlet: {optimization['optimized_outlet_temp']:.1f}°C")
    print(f"Optimization: {optimization['optimization_amount']:+.1f}°C "
          f"({optimization['optimization_percentage']:+.0f}%)")
    print(f"Reasoning: {optimization['optimization_reasoning']}")
    
    print("\n✅ Multi-Heat-Source Physics Integration test complete!")
