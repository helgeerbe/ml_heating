"""
Thermal Constants and Units System.

This module defines all thermal physics constants and provides a standardized
units system for the thermal equilibrium model. Created as part of Phase 2:
Implementation Quality Fixes.

Physical Constants and Units:
- Temperature: °C (Celsius)
- Heat Input: Dimensionless thermal units  
- Heat Loss Coefficient: °C per thermal unit
- PV Power: W (Watts)
- PV Heat Weight: °C/W (temperature rise per watt)
- Fireplace/TV Heat: °C (direct temperature contribution)
- Time: hours
"""

from typing import Dict, Any, Tuple
import logging


class PhysicsConstants:
    """Physical constants for thermal modeling."""
    
    # Temperature bounds (°C)
    MIN_BUILDING_TEMP = -20.0  # Minimum realistic building temperature
    MAX_BUILDING_TEMP = 50.0   # Maximum realistic building temperature
    MIN_OUTDOOR_TEMP = -40.0   # Extreme cold limit
    MAX_OUTDOOR_TEMP = 50.0    # Extreme heat limit
    
    # Heat pump bounds (°C)
    MIN_OUTLET_TEMP = 25.0     # Minimum heat pump outlet temperature
    MAX_OUTLET_TEMP = 70.0     # Maximum safe heat pump outlet temperature
    
    # Physics bounds
    ABSOLUTE_ZERO_CELSIUS = -273.15  # Absolute zero in Celsius
    
    # Time constants (hours)
    MIN_TIME_CONSTANT = 0.5    # Minimum thermal time constant (30 minutes)
    MAX_TIME_CONSTANT = 24.0   # Maximum thermal time constant (24 hours)
    
    # Heat loss bounds
    MIN_HEAT_LOSS_COEFF = 0.1  # Minimum heat loss coefficient
    MAX_HEAT_LOSS_COEFF = 10.0  # Maximum heat loss coefficient

    # Specific heat capacity of water (kJ/kg·K)
    SPECIFIC_HEAT_WATER = 4.186
    # Minimum valid flow rate (L/h)
    MIN_FLOW_RATE = 0.0
    # Maximum valid flow rate (L/h) - sanity check
    MAX_FLOW_RATE = 5000.0
    
    # Effectiveness bounds
    MIN_EFFECTIVENESS = 0.01   # Minimum outlet effectiveness (1%)
    MAX_EFFECTIVENESS = 1.0    # Maximum outlet effectiveness (100%)
    
    # Gradient calculation epsilon values (Phase 3.2 addition)
    # Hours - step size for thermal time constant gradients
    THERMAL_TIME_CONSTANT_EPSILON = 2.0
    # Step size for heat loss coefficient gradients
    HEAT_LOSS_COEFFICIENT_EPSILON = 0.005
    # Step size for outlet effectiveness gradients
    OUTLET_EFFECTIVENESS_EPSILON = 0.05
    # Step size for PV heat weight gradients
    PV_HEAT_WEIGHT_EPSILON = 0.0005
    # Step size for TV heat weight gradients
    TV_HEAT_WEIGHT_EPSILON = 0.05

    # Learning rate bounds and factors (Phase 3.2 addition)
    # Learning confidence decay per cycle
    CONFIDENCE_DECAY_RATE = 0.99
    # Learning confidence boost when improving
    CONFIDENCE_BOOST_RATE = 1.1
    # Learning rate reduction for stable parameters
    STABILITY_REDUCTION_FACTOR = 0.8
    # Learning rate boost for very large errors (>2°C)
    ERROR_BOOST_FACTOR_HIGH = 3.0
    # Learning rate boost for large errors (>1°C)
    ERROR_BOOST_FACTOR_MEDIUM = 2.0
    # Learning rate boost for medium errors (>0.5°C)
    ERROR_BOOST_FACTOR_LOW = 1.5

    # Error thresholds for learning rate scaling (Phase 3.2 addition)
    # °C - threshold for very large errors
    ERROR_THRESHOLD_VERY_HIGH = 2.0
    # °C - threshold for large errors
    ERROR_THRESHOLD_HIGH = 1.0
    # °C - threshold for medium errors
    ERROR_THRESHOLD_MEDIUM = 0.5

    # Parameter stability thresholds (Phase 3.2 addition)
    # Thermal time constant stability (hours)
    THERMAL_STABILITY_THRESHOLD = 0.05
    # Heat loss coefficient stability
    HEAT_LOSS_STABILITY_THRESHOLD = 0.0005
    # Outlet effectiveness stability
    EFFECTIVENESS_STABILITY_THRESHOLD = 0.005

    # Default safety and operational values (Phase 3.2 addition)
    # °C - default safety margin for predictions
    DEFAULT_SAFETY_MARGIN = 0.2
    # Hours - default prediction time horizon
    DEFAULT_PREDICTION_HORIZON = 4.0
    # Thermal momentum decay rate
    MOMENTUM_DECAY_RATE = 0.1
    # Maximum momentum reduction (20%)
    MOMENTUM_REDUCTION_FACTOR = 0.2

    # History management constants (Phase 3.2 addition)
    # Maximum stored prediction records
    MAX_PREDICTION_HISTORY = 200
    # Trim history to this size when max reached
    TRIM_PREDICTION_HISTORY = 100
    # Maximum stored parameter change records
    MAX_PARAMETER_HISTORY = 500
    # Trim history to this size when max reached
    TRIM_PARAMETER_HISTORY = 250

    # Parameter bounds validation constants (Phase 3.2 addition)
    # °C - minimum outlet temp above outdoor
    MINIMUM_OUTLET_ABOVE_OUTDOOR = 5.0
    # °C - maximum safe heat pump outlet
    MAXIMUM_SAFE_OUTLET_TEMP = 70.0
    # °C - fallback outlet temperature
    DEFAULT_FALLBACK_OUTLET = 35.0
    # °C - minimum equilibrium outlet temp
    EQUILIBRIUM_OUTLET_MIN = 25.0
    # °C - maximum equilibrium outlet temp
    EQUILIBRIUM_OUTLET_MAX = 65.0

    # Significant change thresholds for logging (Phase 3.2 addition)
    # Hours - log thermal time constant changes
    SIGNIFICANT_THERMAL_CHANGE = 0.01
    # Log heat loss coefficient changes
    SIGNIFICANT_HEAT_LOSS_CHANGE = 0.0001
    # Log outlet effectiveness changes
    SIGNIFICANT_EFFECTIVENESS_CHANGE = 0.001

    # Operational constants (Phase 3.2 addition)
    RETRY_DELAY_SECONDS = 300  # Wait time before retrying after error
    BLOCKING_POLL_INTERVAL_SECONDS = 60  # Poll interval for blocking events
    CYCLE_INTERVAL_MINUTES = 30  # Main control loop interval

    # Learning update bounds (Phase 3.2 addition)
    MAX_HEAT_LOSS_COEFFICIENT_CHANGE = 0.02
    MAX_OUTLET_EFFECTIVENESS_CHANGE = 0.02
    MAX_THERMAL_TIME_CONSTANT_CHANGE = 0.5
    ERROR_IMPROVEMENT_THRESHOLD = 0.05

    # Learning rate defaults (Phase 3.2 addition)
    DEFAULT_LEARNING_RATE = 0.01
    MIN_LEARNING_RATE = 0.001
    MAX_LEARNING_RATE = 0.1
    INITIAL_LEARNING_CONFIDENCE = 3.0


class ThermalUnits:
    """
    Standardized units system for thermal parameters.
    
    This class defines the units for all thermal parameters and provides
    validation and conversion utilities.
    """
    
    # Unit definitions
    UNITS = {
        # Core thermal parameters
        'thermal_time_constant': 'hours',
        'heat_loss_coefficient': '°C/thermal_unit',
        'outlet_effectiveness': 'dimensionless',
        
        # External heat source weights
        'pv_heat_weight': '°C/W',
        'fireplace_heat_weight': '°C',
        'tv_heat_weight': '°C',
        
        # Temperature measurements
        'indoor_temperature': '°C',
        'outdoor_temperature': '°C',
        'outlet_temperature': '°C',
        'target_temperature': '°C',
        
        # Power measurements
        'pv_power': 'W',
        
        # Binary states
        'fireplace_on': 'boolean',
        'tv_on': 'boolean',
        
        # Learning parameters
        'learning_rate': 'dimensionless',
        'learning_confidence': 'dimensionless',
        
        # Bounds and tolerances
        'safety_margin': '°C',
        'prediction_horizon': 'hours'
    }
    
    # Expected value ranges for validation
    RANGES = {
        'thermal_time_constant': (
            PhysicsConstants.MIN_TIME_CONSTANT,
            PhysicsConstants.MAX_TIME_CONSTANT
        ),
        'heat_loss_coefficient': (
            PhysicsConstants.MIN_HEAT_LOSS_COEFF,
            PhysicsConstants.MAX_HEAT_LOSS_COEFF
        ),
        'outlet_effectiveness': (
            PhysicsConstants.MIN_EFFECTIVENESS,
            PhysicsConstants.MAX_EFFECTIVENESS
        ),
        'indoor_temperature': (
            PhysicsConstants.MIN_BUILDING_TEMP,
            PhysicsConstants.MAX_BUILDING_TEMP
        ),
        'outdoor_temperature': (
            PhysicsConstants.MIN_OUTDOOR_TEMP,
            PhysicsConstants.MAX_OUTDOOR_TEMP
        ),
        'outlet_temperature': (
            PhysicsConstants.MIN_OUTLET_TEMP,
            PhysicsConstants.MAX_OUTLET_TEMP
        ),
        'pv_power': (0.0, 20000.0),  # 0 to 20kW
        'pv_heat_weight': (0.0, 0.01),  # 0 to 0.01 °C/W
        'fireplace_heat_weight': (0.0, 10.0),  # 0 to 10°C
        'tv_heat_weight': (0.0, 2.0),  # 0 to 2°C
        'learning_rate': (0.001, 1.0),  # 0.1% to 100%
        'learning_confidence': (0.1, 10.0),  # 10% to 1000%
        'safety_margin': (0.1, 5.0),  # 0.1°C to 5°C
        'prediction_horizon': (1.0, 24.0)  # 1 to 24 hours
    }
    
    @classmethod
    def get_unit(cls, parameter_name: str) -> str:
        """Get the unit for a given parameter."""
        return cls.UNITS.get(parameter_name, 'unknown')
    
    @classmethod
    def get_range(cls, parameter_name: str) -> Tuple[float, float]:
        """Get the expected range for a given parameter."""
        return cls.RANGES.get(parameter_name, (-float('inf'), float('inf')))
    
    @classmethod
    def validate_parameter(cls, parameter_name: str, value: float) -> bool:
        """
        Validate that a parameter value is within expected range.
        
        Args:
            parameter_name: Name of the parameter
            value: Value to validate
            
        Returns:
            True if value is valid, False otherwise
        """
        if parameter_name not in cls.RANGES:
            logging.warning(
                f"No validation range defined for {parameter_name}"
            )
            return True
        
        min_val, max_val = cls.RANGES[parameter_name]
        is_valid = min_val <= value <= max_val
        
        if not is_valid:
            unit = cls.get_unit(parameter_name)
            logging.error(
                f"Parameter {parameter_name} value {value:.3f} {unit} "
                f"outside valid range [{min_val:.3f}, {max_val:.3f}] {unit}"
            )
        
        return is_valid
    
    @classmethod
    def validate_parameters(
        cls, parameters: Dict[str, Any]
    ) -> Dict[str, bool]:
        """
        Validate multiple parameters at once.
        
        Args:
            parameters: Dictionary of parameter_name: value pairs
            
        Returns:
            Dictionary of parameter_name: is_valid pairs
        """
        results = {}
        for name, value in parameters.items():
            if isinstance(value, (int, float)):
                results[name] = cls.validate_parameter(name, float(value))
            else:
                results[name] = True  # Non-numeric values pass validation
        
        return results
    
    @classmethod
    def format_parameter(cls, parameter_name: str, value: float) -> str:
        """
        Format a parameter value with its units for display.
        
        Args:
            parameter_name: Name of the parameter
            value: Value to format
            
        Returns:
            Formatted string with value and units
        """
        unit = cls.get_unit(parameter_name)
        
        # Special formatting for different parameter types
        if 'temperature' in parameter_name:
            return f"{value:.1f} {unit}"
        elif 'coefficient' in parameter_name:
            return f"{value:.3f} {unit}"
        elif 'effectiveness' in parameter_name:
            return f"{value:.2f} {unit}"
        elif 'power' in parameter_name:
            return f"{value:.0f} {unit}"
        else:
            return f"{value:.3f} {unit}"


class ThermalParameterValidator:
    """
    Validates thermal parameters for physics correctness.
    
    This class provides comprehensive validation of thermal parameters
    to ensure they follow physical laws and realistic bounds.
    """
    
    def __init__(self):
        self.validation_errors = []
        self.validation_warnings = []
    
    def validate_heat_balance_parameters(
        self,
        heat_loss_coeff: float,
        outlet_effectiveness: float,
        external_weights: Dict[str, float]
    ) -> bool:
        """
        Validate parameters used in heat balance calculations.
        
        Args:
            heat_loss_coeff: Heat loss coefficient
            outlet_effectiveness: Outlet heat transfer effectiveness
            external_weights: External heat source weights
            
        Returns:
            True if all parameters are valid
        """
        self.validation_errors.clear()
        self.validation_warnings.clear()
        
        # Validate core parameters
        if not ThermalUnits.validate_parameter(
            'heat_loss_coefficient', heat_loss_coeff
        ):
            self.validation_errors.append("Heat loss coefficient out of range")

        if not ThermalUnits.validate_parameter(
            'outlet_effectiveness', outlet_effectiveness
        ):
            self.validation_errors.append("Outlet effectiveness out of range")

        # Validate external heat source weights
        for source, weight in external_weights.items():
            param_name = f"{source}_heat_weight"
            if not ThermalUnits.validate_parameter(param_name, weight):
                self.validation_errors.append(
                    f"{source} heat weight out of range"
                )

        # Physics consistency checks
        if heat_loss_coeff <= 0:
            self.validation_errors.append(
                "Heat loss coefficient must be positive"
            )

        if outlet_effectiveness <= 0:
            self.validation_errors.append(
                "Outlet effectiveness must be positive"
            )
        
        # Warning for unusual but valid values
        if heat_loss_coeff > 5.0:
            self.validation_warnings.append(
                "Very high heat loss coefficient - building poorly insulated?"
            )
        
        if outlet_effectiveness < 0.3:
            self.validation_warnings.append(
                "Low outlet effectiveness - heat pump efficiency issues?"
            )
        
        return len(self.validation_errors) == 0
    
    def validate_temperature_inputs(
        self,
        indoor: float,
        outdoor: float,
        outlet: float
    ) -> bool:
        """
        Validate temperature inputs for physical realism.
        
        Args:
            indoor: Indoor temperature
            outdoor: Outdoor temperature  
            outlet: Heat pump outlet temperature
            
        Returns:
            True if temperatures are physically realistic
        """
        self.validation_errors.clear()
        self.validation_warnings.clear()
        
        # Range validation
        valid_ranges = [
            ThermalUnits.validate_parameter('indoor_temperature', indoor),
            ThermalUnits.validate_parameter('outdoor_temperature', outdoor),
            ThermalUnits.validate_parameter('outlet_temperature', outlet)
        ]
        
        if not all(valid_ranges):
            self.validation_errors.append("Temperature values out of range")
        
        # Physics consistency checks
        if outlet <= outdoor and indoor > outdoor:
            self.validation_errors.append(
                "Outlet temperature cannot be below outdoor when heating"
            )
        
        if indoor < outdoor - 20:
            self.validation_warnings.append(
                "Indoor much colder than outdoor - system not working?"
            )
        
        if outlet > 60 and indoor < 25:
            self.validation_warnings.append(
                "High outlet temperature but low indoor - heat loss issues?"
            )
        
        return len(self.validation_errors) == 0
    
    def get_validation_report(self) -> str:
        """Generate a human-readable validation report."""
        report = []
        
        if self.validation_errors:
            report.append("VALIDATION ERRORS:")
            for error in self.validation_errors:
                report.append(f"  ❌ {error}")
        
        if self.validation_warnings:
            report.append("VALIDATION WARNINGS:")
            for warning in self.validation_warnings:
                report.append(f"  ⚠️  {warning}")
        
        if not self.validation_errors and not self.validation_warnings:
            report.append("✅ All parameters valid")
        
        return "\n".join(report)


# Convenience functions for common operations
def validate_thermal_parameters(parameters: Dict[str, Any]) -> bool:
    """
    Quick validation of thermal parameters.
    
    Args:
        parameters: Dictionary of parameter names and values
        
    Returns:
        True if all parameters are valid
    """
    results = ThermalUnits.validate_parameters(parameters)
    return all(results.values())


def format_thermal_state(parameters: Dict[str, Any]) -> str:
    """
    Format thermal parameters for logging or display.
    
    Args:
        parameters: Dictionary of parameter names and values
        
    Returns:
        Formatted string with all parameters and units
    """
    lines = []
    for name, value in parameters.items():
        if isinstance(value, (int, float)):
            formatted = ThermalUnits.format_parameter(name, float(value))
            lines.append(f"  {name}: {formatted}")
        else:
            lines.append(f"  {name}: {value}")
    
    return "\n".join(lines)


# Export public interface
__all__ = [
    'PhysicsConstants',
    'ThermalUnits', 
    'ThermalParameterValidator',
    'validate_thermal_parameters',
    'format_thermal_state'
]
