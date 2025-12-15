"""
Central Thermal Parameter Configuration - Single Source of Truth

This module provides a centralized configuration for all thermal parameters,
eliminating the architectural anti-pattern of having duplicate parameter
definitions scattered across multiple files.

Key benefits:
- Single source of truth for all thermal parameters
- Consistent bounds across calibration and runtime systems
- Built-in validation and type safety
- Clear documentation for each parameter
- Easy maintenance and updates
"""

from typing import Dict, Tuple


class ThermalParameterConfig:
    """
    Centralized thermal parameter configuration for the ML heating system.
    
    This class provides defaults, bounds, and validation for all thermal
    parameters used throughout the system, ensuring consistency between
    calibration, runtime operation, and validation systems.
    """
    
    # Default parameter values optimized for moderate insulation houses
    # UPDATED FOR TDD COMPLIANCE - physically reasonable ranges with realistic heat balance
    DEFAULTS = {
        'thermal_time_constant': 4.0,      # hours - realistic for moderate insulation 
        'heat_loss_coefficient': 0.2,      # 1/hour - increased for realistic heat balance (was 0.05)
        'outlet_effectiveness': 0.04,      # dimensionless - adjusted for realistic outlets
        'pv_heat_weight': 0.002,           # °C/W - 2°C per kW (TDD compliant: 0.001-0.01 range)
        'fireplace_heat_weight': 5.0,      # °C - direct 5°C contribution (TDD compliant: 2-10°C range)
        'tv_heat_weight': 0.2              # °C - 0.2°C for realistic TV contribution (1°C with 0.2 coefficient)
    }
    
    # Parameter bounds (min, max) for optimization and validation
    # These bounds are designed to allow realistic parameter exploration
    # while preventing physically impossible values
    BOUNDS = {
        'thermal_time_constant': (3.0, 8.0),     # Hours - realistic building range
        'heat_loss_coefficient': (0.002, 0.25),  # 1/hour - very low to high heat loss
        'outlet_effectiveness': (0.01, 0.5),     # dimensionless - adjusted for new default
        'pv_heat_weight': (0.0001, 0.005),       # W/°C - minimal to moderate PV effect
        'fireplace_heat_weight': (0.01, 6.0),    # 1/°C - small to very large fireplace effect
        'tv_heat_weight': (0.05, 1.5)            # W/°C - small to moderate appliance effect
    }
    
    # Parameter descriptions for documentation and debugging
    DESCRIPTIONS = {
        'thermal_time_constant': 'Time constant for thermal equilibrium (hours)',
        'heat_loss_coefficient': 'Heat loss coefficient (1/hour)',
        'outlet_effectiveness': 'Heat pump outlet effectiveness (dimensionless)',
        'pv_heat_weight': 'PV power heating contribution (W/°C)',
        'fireplace_heat_weight': 'Fireplace heating contribution (1/°C)',
        'tv_heat_weight': 'TV/appliance heating contribution (W/°C)'
    }
    
    # Parameter units for display and logging
    UNITS = {
        'thermal_time_constant': 'hours',
        'heat_loss_coefficient': '1/hour',
        'outlet_effectiveness': 'dimensionless',
        'pv_heat_weight': 'W/°C',
        'fireplace_heat_weight': '1/°C',
        'tv_heat_weight': 'W/°C'
    }
    
    @classmethod
    def get_default(cls, param_name: str) -> float:
        """
        Get the default value for a thermal parameter.
        
        Args:
            param_name: Name of the thermal parameter
            
        Returns:
            Default value for the parameter
            
        Raises:
            KeyError: If parameter name is not recognized
        """
        if param_name not in cls.DEFAULTS:
            raise KeyError(f"Unknown thermal parameter: {param_name}")
        return cls.DEFAULTS[param_name]
    
    @classmethod  
    def get_bounds(cls, param_name: str) -> Tuple[float, float]:
        """
        Get the bounds (min, max) for a thermal parameter.
        
        Args:
            param_name: Name of the thermal parameter
            
        Returns:
            Tuple of (min_value, max_value)
            
        Raises:
            KeyError: If parameter name is not recognized
        """
        if param_name not in cls.BOUNDS:
            raise KeyError(f"Unknown thermal parameter: {param_name}")
        return cls.BOUNDS[param_name]
    
    @classmethod
    def validate_parameter(cls, param_name: str, value: float) -> bool:
        """
        Validate that a parameter value is within acceptable bounds.
        
        Args:
            param_name: Name of the thermal parameter
            value: Value to validate
            
        Returns:
            True if value is within bounds, False otherwise
            
        Raises:
            KeyError: If parameter name is not recognized
        """
        min_val, max_val = cls.get_bounds(param_name)
        return min_val <= value <= max_val
    
    @classmethod
    def clamp_parameter(cls, param_name: str, value: float) -> float:
        """
        Clamp a parameter value to be within acceptable bounds.
        
        Args:
            param_name: Name of the thermal parameter
            value: Value to clamp
            
        Returns:
            Value clamped to be within bounds
            
        Raises:
            KeyError: If parameter name is not recognized
        """
        min_val, max_val = cls.get_bounds(param_name)
        return max(min_val, min(value, max_val))
    
    @classmethod
    def get_description(cls, param_name: str) -> str:
        """
        Get a human-readable description of a thermal parameter.
        
        Args:
            param_name: Name of the thermal parameter
            
        Returns:
            Description string
            
        Raises:
            KeyError: If parameter name is not recognized
        """
        if param_name not in cls.DESCRIPTIONS:
            raise KeyError(f"Unknown thermal parameter: {param_name}")
        return cls.DESCRIPTIONS[param_name]
    
    @classmethod
    def get_unit(cls, param_name: str) -> str:
        """
        Get the unit string for a thermal parameter.
        
        Args:
            param_name: Name of the thermal parameter
            
        Returns:
            Unit string
            
        Raises:
            KeyError: If parameter name is not recognized
        """
        if param_name not in cls.UNITS:
            raise KeyError(f"Unknown thermal parameter: {param_name}")
        return cls.UNITS[param_name]
    
    @classmethod
    def get_all_defaults(cls) -> Dict[str, float]:
        """
        Get all default parameter values.
        
        Returns:
            Dictionary mapping parameter names to default values
        """
        return cls.DEFAULTS.copy()
    
    @classmethod
    def get_all_bounds(cls) -> Dict[str, Tuple[float, float]]:
        """
        Get all parameter bounds.
        
        Returns:
            Dictionary mapping parameter names to (min, max) tuples
        """
        return cls.BOUNDS.copy()
    
    @classmethod
    def get_parameter_info(cls, param_name: str) -> Dict:
        """
        Get comprehensive information about a thermal parameter.
        
        Args:
            param_name: Name of the thermal parameter
            
        Returns:
            Dictionary with default, bounds, description, and unit
            
        Raises:
            KeyError: If parameter name is not recognized
        """
        return {
            'default': cls.get_default(param_name),
            'bounds': cls.get_bounds(param_name),
            'description': cls.get_description(param_name),
            'unit': cls.get_unit(param_name)
        }
    
    @classmethod
    def get_all_parameter_info(cls) -> Dict[str, Dict]:
        """
        Get comprehensive information about all thermal parameters.
        
        Returns:
            Dictionary mapping parameter names to their info dictionaries
        """
        return {
            param_name: cls.get_parameter_info(param_name)
            for param_name in cls.DEFAULTS.keys()
        }


# Convenience functions for backward compatibility
def get_thermal_default(param_name: str) -> float:
    """Convenience function to get thermal parameter default."""
    return ThermalParameterConfig.get_default(param_name)


def get_thermal_bounds(param_name: str) -> Tuple[float, float]:
    """Convenience function to get thermal parameter bounds."""
    return ThermalParameterConfig.get_bounds(param_name)


def validate_thermal_parameter(param_name: str, value: float) -> bool:
    """Convenience function to validate thermal parameter."""
    return ThermalParameterConfig.validate_parameter(param_name, value)
