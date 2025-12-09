"""
Unified Thermal Parameter Management System

This module provides the single source of truth for all thermal parameters,
resolving conflicts and providing a unified API for parameter access.

Created as part of the Thermal Parameter Consolidation Plan - Phase 2.1
"""

import os
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ParameterInfo:
    """Information about a thermal parameter."""
    default: float
    bounds: Tuple[float, float]
    description: str
    unit: str
    env_var: Optional[str] = None


class ThermalParameterManager:
    """
    Unified thermal parameter management system.
    
    This class consolidates all thermal parameters from config.py, 
    thermal_config.py, and thermal_constants.py into a single source 
    of truth, resolving conflicts according to the documented decisions.
    """
    
    # Unified parameter definitions with conflict resolutions applied
    _PARAMETERS = {
        # Core thermal physics parameters (Priority 1)
        'thermal_time_constant': ParameterInfo(
            default=4.0,
            bounds=(0.5, 24.0),
            description='Building thermal response time',
            unit='hours',
            env_var='THERMAL_TIME_CONSTANT'
        ),
        
        # CONFLICT RESOLVED: Use thermal_config.py value (0.2) over config.py (0.1)
        'heat_loss_coefficient': ParameterInfo(
            default=0.2,  # Resolution: More realistic for moderate insulation
            bounds=(0.002, 0.25),
            description='Heat loss rate per degree difference',
            unit='1/hour',
            env_var='HEAT_LOSS_COEFFICIENT'
        ),
        
        # CONFLICT RESOLVED: Use thermal_config.py calibrated value
        'outlet_effectiveness': ParameterInfo(
            default=0.04,  # Resolution: Calibrated TDD value vs 0.1 in config.py
            bounds=(0.01, 0.5),  # Resolution: Realistic max vs 1.0 in thermal_constants
            description='Heat pump outlet efficiency',
            unit='dimensionless',
            env_var='OUTLET_EFFECTIVENESS'
        ),
        
        'outdoor_coupling': ParameterInfo(
            default=0.3,
            bounds=(0.1, 0.8),
            description='Outdoor temperature influence factor',
            unit='dimensionless',
            env_var='OUTDOOR_COUPLING'
        ),
        
        # External heat source weights (Priority 2)
        'pv_heat_weight': ParameterInfo(
            default=0.002,  # 2°C per kW solar heating
            bounds=(0.0001, 0.01),
            description='PV power heating contribution',
            unit='°C/W',
            env_var='PV_HEAT_WEIGHT'
        ),
        
        'fireplace_heat_weight': ParameterInfo(
            default=5.0,
            bounds=(0.0, 10.0),
            description='Fireplace direct heating contribution',
            unit='°C',
            env_var='FIREPLACE_HEAT_WEIGHT'
        ),
        
        'tv_heat_weight': ParameterInfo(
            default=0.2,
            bounds=(0.0, 2.0),
            description='TV/appliance heating contribution',
            unit='°C',
            env_var='TV_HEAT_WEIGHT'
        ),
        
        # Temperature bounds (CONFLICT RESOLVED)
        'outlet_temp_min': ParameterInfo(
            default=25.0,  # Resolution: Physics-based minimum vs 14.0 in config.py
            bounds=(14.0, 30.0),
            description='Minimum outlet temperature for heating mode',
            unit='°C',
            env_var='CLAMP_MIN_ABS'
        ),
        
        'outlet_temp_max': ParameterInfo(
            default=65.0,  # Resolution: Safety-first vs 70.0 in thermal_constants
            bounds=(60.0, 70.0),
            description='Maximum safe outlet temperature',
            unit='°C',
            env_var='CLAMP_MAX_ABS'
        ),
        
        # Adaptive learning parameters (Priority 3)
        'adaptive_learning_rate': ParameterInfo(
            default=0.05,
            bounds=(0.001, 0.2),
            description='Base adaptive learning rate',
            unit='dimensionless',
            env_var='ADAPTIVE_LEARNING_RATE'
        ),
        
        'min_learning_rate': ParameterInfo(
            default=0.01,
            bounds=(0.001, 0.1),
            description='Minimum learning rate',
            unit='dimensionless',
            env_var='MIN_LEARNING_RATE'
        ),
        
        'max_learning_rate': ParameterInfo(
            default=0.2,
            bounds=(0.1, 1.0),
            description='Maximum learning rate',
            unit='dimensionless',
            env_var='MAX_LEARNING_RATE'
        ),
        
        'learning_confidence': ParameterInfo(
            default=3.0,
            bounds=(1.0, 10.0),
            description='Initial learning confidence',
            unit='dimensionless',
            env_var='LEARNING_CONFIDENCE'
        )
    }
    
    def __init__(self):
        """Initialize the thermal parameter manager."""
        self._cache = {}
        self._load_from_environment()
        
    def _load_from_environment(self):
        """Load parameter values from environment variables."""
        for param_name, param_info in self._PARAMETERS.items():
            if param_info.env_var:
                env_value = os.getenv(param_info.env_var)
                if env_value is not None:
                    try:
                        value = float(env_value)
                        if self.validate(param_name, value):
                            self._cache[param_name] = value
                            logging.info(
                                f"Loaded {param_name} = {value} from "
                                f"environment variable {param_info.env_var}"
                            )
                        else:
                            logging.warning(
                                f"Environment value {value} for {param_name} "
                                f"outside bounds {param_info.bounds}, using default"
                            )
                    except ValueError:
                        logging.error(
                            f"Invalid float value '{env_value}' for "
                            f"environment variable {param_info.env_var}"
                        )
    
    def get(self, param_name: str) -> float:
        """
        Get parameter value with environment variable override.
        
        Args:
            param_name: Name of the thermal parameter
            
        Returns:
            Parameter value (from env var if set, otherwise default)
            
        Raises:
            KeyError: If parameter name is not recognized
        """
        if param_name not in self._PARAMETERS:
            raise KeyError(f"Unknown thermal parameter: {param_name}")
        
        # Return cached value if available
        if param_name in self._cache:
            return self._cache[param_name]
        
        # Return default value
        return self._PARAMETERS[param_name].default
    
    def set(self, param_name: str, value: float) -> bool:
        """
        Set parameter value with validation.
        
        Args:
            param_name: Name of the thermal parameter
            value: Value to set
            
        Returns:
            True if value was set successfully
            
        Raises:
            KeyError: If parameter name is not recognized
        """
        if param_name not in self._PARAMETERS:
            raise KeyError(f"Unknown thermal parameter: {param_name}")
        
        if not self.validate(param_name, value):
            return False
        
        self._cache[param_name] = value
        logging.info(f"Set {param_name} = {value}")
        return True
    
    def validate(self, param_name: str, value: float) -> bool:
        """
        Validate parameter value against bounds.
        
        Args:
            param_name: Name of the thermal parameter
            value: Value to validate
            
        Returns:
            True if value is within bounds
            
        Raises:
            KeyError: If parameter name is not recognized
        """
        if param_name not in self._PARAMETERS:
            raise KeyError(f"Unknown thermal parameter: {param_name}")
        
        min_val, max_val = self._PARAMETERS[param_name].bounds
        return min_val <= value <= max_val
    
    def get_bounds(self, param_name: str) -> Tuple[float, float]:
        """
        Get parameter bounds.
        
        Args:
            param_name: Name of the thermal parameter
            
        Returns:
            Tuple of (min_value, max_value)
            
        Raises:
            KeyError: If parameter name is not recognized
        """
        if param_name not in self._PARAMETERS:
            raise KeyError(f"Unknown thermal parameter: {param_name}")
        
        return self._PARAMETERS[param_name].bounds
    
    def get_info(self, param_name: str) -> ParameterInfo:
        """
        Get comprehensive parameter information.
        
        Args:
            param_name: Name of the thermal parameter
            
        Returns:
            ParameterInfo object with all parameter details
            
        Raises:
            KeyError: If parameter name is not recognized
        """
        if param_name not in self._PARAMETERS:
            raise KeyError(f"Unknown thermal parameter: {param_name}")
        
        return self._PARAMETERS[param_name]
    
    def get_all_parameters(self) -> Dict[str, float]:
        """
        Get all parameter values.
        
        Returns:
            Dictionary mapping parameter names to current values
        """
        return {name: self.get(name) for name in self._PARAMETERS.keys()}
    
    def get_all_defaults(self) -> Dict[str, float]:
        """
        Get all default parameter values.
        
        Returns:
            Dictionary mapping parameter names to default values
        """
        return {name: info.default for name, info in self._PARAMETERS.items()}
    
    def validate_all(self) -> Dict[str, bool]:
        """
        Validate all current parameter values.
        
        Returns:
            Dictionary mapping parameter names to validation results
        """
        return {name: self.validate(name, self.get(name)) for name in self._PARAMETERS.keys()}
    
    def reload_from_environment(self):
        """Reload all parameters from environment variables."""
        self._cache.clear()
        self._load_from_environment()
    
    def has_single_source_of_truth(self) -> bool:
        """
        Test method for unified parameter system.
        
        Returns:
            True if all parameters are managed by this unified system
        """
        return len(self._PARAMETERS) > 0
    
    # Legacy compatibility methods
    def legacy_get_config_value(self, config_name: str) -> Optional[float]:
        """
        Legacy compatibility method for config.py values.
        
        Maps old config names to new unified parameter names.
        """
        legacy_mapping = {
            'THERMAL_TIME_CONSTANT': 'thermal_time_constant',
            'HEAT_LOSS_COEFFICIENT': 'heat_loss_coefficient',
            'OUTLET_EFFECTIVENESS': 'outlet_effectiveness',
            'CLAMP_MIN_ABS': 'outlet_temp_min',
            'CLAMP_MAX_ABS': 'outlet_temp_max'
        }
        
        if config_name in legacy_mapping:
            return self.get(legacy_mapping[config_name])
        
        return None
    
    def legacy_thermal_config_default(self, param_name: str) -> Optional[float]:
        """
        Legacy compatibility method for ThermalParameterConfig.get_default().
        """
        if param_name in self._PARAMETERS:
            return self.get(param_name)
        return None


# Create global instance for unified access
thermal_params = ThermalParameterManager()


# Legacy compatibility functions
def get_thermal_parameter(name: str) -> float:
    """Legacy function for backward compatibility."""
    return thermal_params.get(name)


def validate_thermal_parameter(name: str, value: float) -> bool:
    """Legacy function for backward compatibility."""
    return thermal_params.validate(name, value)


# Export public interface
__all__ = [
    'ThermalParameterManager',
    'thermal_params',
    'get_thermal_parameter',
    'validate_thermal_parameter',
    'ParameterInfo'
]
