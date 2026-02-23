"""
Unified Prediction Context Service

This module provides a centralized service for creating consistent
environmental contexts across all prediction systems (binary search,
smart rounding, trajectory).

Eliminates the inconsistency where:
- Binary search used forecast-based conditions
- Smart rounding used current-only conditions  
- Trajectory prediction used forecast-based conditions

Now all three systems use identical environmental data through this service.
"""

import logging
from typing import Dict, Tuple, List, Optional

from . import config


class UnifiedPredictionContext:
    """
    Centralized service for creating consistent prediction contexts.
    
    Ensures binary search, smart rounding, and trajectory prediction all use
    identical environmental conditions for their thermal model predictions.
    """
    
    @staticmethod
    def create_prediction_context(
        features: Optional[Dict],
        outdoor_temp: float,
        pv_power: float,
        thermal_features: Dict
    ) -> Dict:
        """
        Create a unified prediction context that all systems should use.

        This replaces the individual context creation in each system with
        a single source of truth for environmental conditions.

        Args:
            features: Full feature dictionary with forecast data
            outdoor_temp: Current outdoor temperature
            pv_power: Current PV power
            thermal_features: Thermal features dict

        Returns:
            Dict with standardized prediction context including:
            - avg_outdoor: Cycle-aligned outdoor temp (forecast if available)
            - avg_pv: Cycle-aligned PV power (forecast if available)
            - outdoor_forecast: 4-hour outdoor temperature forecast array
            - pv_forecast: 4-hour PV power forecast array
            - fireplace_on: Fireplace status
            - tv_on: TV status
            - use_forecasts: Boolean indicating if forecasts were used
        """
        context = {}

        # Get cycle time from config
        cycle_minutes = config.CYCLE_INTERVAL_MINUTES
        cycle_hours = cycle_minutes / 60.0

        # Validate cycle time against reasonable limits
        max_reasonable_cycle = 180
        if cycle_minutes > max_reasonable_cycle:
            cycle_hours = max_reasonable_cycle / 60.0

        # Extract forecast data if available
        outdoor_forecast = []
        pv_forecast = []

        if features:
            # Extract 4-hour forecasts
            forecast_1h_outdoor = features.get(
                'temp_forecast_1h', outdoor_temp
            )
            forecast_2h_outdoor = features.get(
                'temp_forecast_2h', outdoor_temp
            )
            forecast_3h_outdoor = features.get(
                'temp_forecast_3h', outdoor_temp
            )
            forecast_4h_outdoor = features.get(
                'temp_forecast_4h', outdoor_temp
            )

            forecast_1h_pv = features.get('pv_forecast_1h', pv_power)
            forecast_2h_pv = features.get('pv_forecast_2h', pv_power)
            forecast_3h_pv = features.get('pv_forecast_3h', pv_power)
            forecast_4h_pv = features.get('pv_forecast_4h', pv_power)

            outdoor_forecast = [
                forecast_1h_outdoor,
                forecast_2h_outdoor,
                forecast_3h_outdoor,
                forecast_4h_outdoor
            ]

            pv_forecast = [
                forecast_1h_pv,
                forecast_2h_pv,
                forecast_3h_pv,
                forecast_4h_pv
            ]

            # Calculate cycle-aligned forecast using appropriate interpolation
            if cycle_hours <= 0.5:
                # 0-30min cycles: use average over the cycle
                # Assuming linear interpolation between current and 1h forecast
                # For 30 min cycle, we want the average condition during that
                # 30 mins. This is roughly the value at t=15min (0.25h)
                # UPDATE: Using 0.5 (midpoint/average of current+forecast)
                # to align with trajectory optimizer and avoid drops during
                # rapid solar changes
                weight = 0.5  # Increased weight to capture rapid solar rise

                avg_outdoor = (
                    outdoor_temp * (1 - weight) + forecast_1h_outdoor * weight
                )
                avg_pv = pv_power * (1 - weight) + forecast_1h_pv * weight

            elif cycle_hours <= 1.0:
                # 30-60min cycles: use average over the cycle
                # Assuming linear interpolation between current and 1h forecast
                weight = cycle_hours / 2.0

                avg_outdoor = (
                    outdoor_temp * (1 - weight) + forecast_1h_outdoor * weight
                )
                avg_pv = pv_power * (1 - weight) + forecast_1h_pv * weight

            elif cycle_hours <= 1.51:  # 60-90min cycles: use 1h forecast
                avg_outdoor = forecast_1h_outdoor
                avg_pv = forecast_1h_pv

            elif cycle_hours <= 2.5:  # 90-150min cycles: use 2h forecast
                avg_outdoor = forecast_2h_outdoor
                avg_pv = forecast_2h_pv
            elif cycle_hours <= 3.5:  # 150-210min cycles: use 3h forecast
                avg_outdoor = forecast_3h_outdoor
                avg_pv = forecast_3h_pv
            else:  # >180min cycles: cap at 4h forecast
                avg_outdoor = forecast_4h_outdoor
                avg_pv = forecast_4h_pv

            use_forecasts = True

            logging.info(
                f"🌡️ Using cycle-aligned forecast ({cycle_minutes}min): "
                f"outdoor={avg_outdoor:.1f}°C "
                f"(vs current {outdoor_temp:.1f}°C), "
                f"PV={avg_pv:.0f}W "
                f"(vs current {pv_power:.0f}W)"
            )
        else:
            # No forecast data available, use current values
            avg_outdoor = outdoor_temp
            avg_pv = pv_power
            outdoor_forecast = [outdoor_temp] * 4
            pv_forecast = [pv_power] * 4
            use_forecasts = False

            logging.debug(
                f"🌡️ Using current conditions (no forecasts): "
                f"outdoor={outdoor_temp:.1f}°C, PV={pv_power:.0f}W"
            )
        
        # Build unified context
        context = {
            'avg_outdoor': avg_outdoor,
            'avg_pv': avg_pv,
            'outdoor_forecast': outdoor_forecast,
            'pv_forecast': pv_forecast,
            'fireplace_on': thermal_features.get('fireplace_on', 0.0),
            'tv_on': thermal_features.get('tv_on', 0.0),
            'use_forecasts': use_forecasts,
            'current_outdoor': outdoor_temp,
            'current_pv': pv_power
        }
        
        return context
    
    @staticmethod
    def get_thermal_model_params(context: Dict) -> Dict:
        """
        Extract thermal model parameters from unified context.
        
        This ensures all thermal model calls use identical parameters
        regardless of which system (binary search, smart rounding, trajectory)
        is making the call.
        
        Args:
            context: Unified prediction context from
                create_prediction_context()

        Returns:
            Dict with thermal model parameters
        """
        return {
            'outdoor_temp': context['avg_outdoor'],  # Use forecast average
            'pv_power': context['avg_pv'],          # Use forecast average  
            'fireplace_on': context['fireplace_on'],
            'tv_on': context['tv_on']
        }


class PredictionContextManager:
    """
    Manager class that maintains prediction context state and provides
    convenient access methods for different prediction systems.
    """
    
    def __init__(self):
        self._current_context: Optional[Dict] = None
        self._features: Optional[Dict] = None
    
    def set_features(self, features: Dict) -> None:
        """Store features for context creation."""
        self._features = features
        
    def create_context(self, outdoor_temp: float, pv_power: float,
                       thermal_features: Dict) -> Dict:
        """
        Create and store unified prediction context.

        This context will be used by all prediction systems to ensure
        consistency.
        """
        self._current_context = (
            UnifiedPredictionContext.create_prediction_context(
                features=self._features,
                outdoor_temp=outdoor_temp,
                pv_power=pv_power,
                thermal_features=thermal_features
            )
        )
        return self._current_context
    
    def get_context(self) -> Optional[Dict]:
        """Get the current unified context."""
        return self._current_context
    
    def get_thermal_model_params(self) -> Dict:
        """Get thermal model parameters from current context."""
        if self._current_context is None:
            raise ValueError(
                "No prediction context available. Call create_context() first."
            )

        return UnifiedPredictionContext.get_thermal_model_params(
            self._current_context
        )

    def get_forecast_arrays(self) -> Tuple[List[float], List[float]]:
        """Get forecast arrays for trajectory prediction."""
        if self._current_context is None:
            raise ValueError(
                "No prediction context available. Call create_context() first."
            )

        return (
            self._current_context['outdoor_forecast'],
            self._current_context['pv_forecast']
        )
    
    def uses_forecasts(self) -> bool:
        """Check if the current context uses forecast data."""
        if self._current_context is None:
            return False
        return self._current_context.get('use_forecasts', False)


# Global instance for easy access across modules
prediction_context_manager = PredictionContextManager()
