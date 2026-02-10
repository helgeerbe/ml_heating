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

# Import the single source of truth for thermal parameters
try:
    from .thermal_config import ThermalParameterConfig
except ImportError:
    from thermal_config import ThermalParameterConfig


@dataclass
class ParameterInfo:
    """Information about a thermal parameter."""
    name: str
    default: float
    bounds: Tuple[float, float]
    description: str
    unit: str
    env_var: Optional[str] = None


class ThermalParameterManager:
    """
    Unified thermal parameter management system.
    
    This class centralizes access to all thermal parameters defined in 
    ThermalParameterConfig, providing a single, consistent API for the application.
    """
    
    def __init__(self):
        self._PARAMETERS: Dict[str, ParameterInfo] = {}
        """Initialize the thermal parameter manager."""
        self._cache = {}
        self._initialize_parameters()
        self._load_from_environment()

    def _initialize_parameters(self):
        """Dynamically build the parameter list from ThermalParameterConfig."""
        all_info = ThermalParameterConfig.get_all_parameter_info()
        for name, info in all_info.items():
            # Assume a convention for environment variables for now
            env_var = name.upper()
            self._PARAMETERS[name] = ParameterInfo(
                name=name,
                default=info['default'],
                bounds=info['bounds'],
                description=info['description'],
                unit=info['unit'],
                env_var=env_var
            )
        
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
                                f"Environment value {value} for {param_name} outside "
                                f"bounds {param_info.bounds}, using default"
                            )
                    except ValueError:
                        logging.error(
                            f"Invalid float value '{env_value}' for environment "
                            f"variable {param_info.env_var}"
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
    
    # The new design makes these legacy methods obsolete.
    # They are removed to enforce the single source of truth.


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
