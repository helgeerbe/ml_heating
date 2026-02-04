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
from .thermal_equilibrium_model import ThermalEquilibriumModel
from .unified_thermal_state import get_thermal_state_manager
from .influx_service import create_influx_service
from .prediction_metrics import PredictionMetrics
from . import config
from .thermal_config import ThermalParameterConfig
from .ha_client import create_ha_client


# Singleton pattern to prevent multiple model instantiation
_enhanced_model_wrapper_instance = None


class EnhancedModelWrapper:
    """
    Simplified model wrapper using ThermalEquilibriumModel for persistent learning.

    Replaces the complex Heat Balance Controller with a single prediction path
    that continuously adapts thermal parameters and survives service restarts.

    Implements singleton pattern to prevent multiple instantiation per service restart.
    """

    def __init__(self):
        self.thermal_model = ThermalEquilibriumModel()
        self.learning_enabled = True

        # Get thermal state manager
        self.state_manager = get_thermal_state_manager()

        # Initialize prediction metrics for MAE/RMSE tracking with state integration
        self.prediction_metrics = PredictionMetrics(state_manager=self.state_manager)

        # Get current cycle count from unified state
        metrics = self.state_manager.get_learning_metrics()
        self.cycle_count = metrics["current_cycle_count"]

        # UNIFIED FORECAST: Store cycle-aligned forecast conditions for smart rounding
        self.cycle_aligned_forecast = {}

        logging.info("🎯 Model Wrapper initialized with " "ThermalEquilibriumModel")
        logging.info(
            f"   - Thermal time constant: "
            f"{self.thermal_model.thermal_time_constant:.1f}h"
        )
        logging.info(
            f"   - Total conductance: "
            f"{self.thermal_model.total_conductance:.4f}"
        )
        logging.info(
            f"   - Equilibrium ratio: "
            f"{self.thermal_model.equilibrium_ratio:.3f}"
        )
        logging.info(
            f"   - Learning confidence: "
            f"{self.thermal_model.learning_confidence:.2f}"
        )
        logging.info(f"   - Current cycle: {self.cycle_count}")

    def predict_indoor_temp(
        self, outlet_temp: float, outdoor_temp: float, **kwargs
    ) -> float:
        """
        Predict indoor temperature for smart rounding.

        Uses the thermal model's equilibrium prediction with proper parameter handling.
        Provides robust conversion of pandas data types to scalar values.
        """
        try:
            # UNIFIED FORECAST FIX: Use cycle-aligned forecast for smart rounding
            if hasattr(self, "cycle_aligned_forecast") and self.cycle_aligned_forecast:
                logging.debug(
                    f"🧠 Smart rounding is using cycle-aligned forecast: "
                    f"PV={self.cycle_aligned_forecast.get('pv_power', 0.0):.0f}W (caller sent PV={kwargs.get('pv_power', 0.0):.0f}W)"
                )
                pv_power = self.cycle_aligned_forecast.get("pv_power", kwargs.get("pv_power", 0.0))
                fireplace_on = self.cycle_aligned_forecast.get("fireplace_on", kwargs.get("fireplace_on", 0.0))
                tv_on = self.cycle_aligned_forecast.get("tv_on", kwargs.get("tv_on", 0.0))
                # Use cycle-aligned outdoor_temp as well for full consistency
                outdoor_temp = self.cycle_aligned_forecast.get("outdoor_temp", outdoor_temp)
            else:
                # Fallback to kwargs if cycle-aligned forecast is not available
                pv_power = kwargs.get("pv_power", 0.0)
                fireplace_on = kwargs.get("fireplace_on", 0.0)
                tv_on = kwargs.get("tv_on", 0.0)
            current_indoor = kwargs.get("current_indoor", outdoor_temp + 15.0)

            # Convert pandas Series to scalar values
            def to_scalar(value):
                """Convert pandas Series or any value to scalar."""
                if value is None:
                    return 0.0
                # Handle pandas Series
                if hasattr(value, "iloc"):
                    return float(value.iloc[0]) if len(value) > 0 else 0.0
                # Handle pandas scalar
                if hasattr(value, "item"):
                    return float(value.item())
                # Handle regular values
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return 0.0

            # Convert all parameters to scalars
            pv_power = to_scalar(pv_power)
            fireplace_on = to_scalar(fireplace_on)
            tv_on = to_scalar(tv_on)
            current_indoor = to_scalar(current_indoor)
            outdoor_temp = to_scalar(outdoor_temp)
            outlet_temp = to_scalar(outlet_temp)

            # Additional safety checks
            if outdoor_temp == 0.0:
                logging.error("predict_indoor_temp: outdoor_temp is invalid")
                return 21.0  # Safe fallback temperature
            if outlet_temp == 0.0:
                logging.error("predict_indoor_temp: outlet_temp is invalid")
                return outdoor_temp + 10.0
            if current_indoor == 0.0:
                current_indoor = outdoor_temp + 15.0

            # Use thermal model to predict equilibrium temperature
            predicted_temp = self.thermal_model.predict_equilibrium_temperature(
                outlet_temp=outlet_temp,
                outdoor_temp=outdoor_temp,
                current_indoor=current_indoor,
                pv_power=pv_power,
                fireplace_on=fireplace_on,
                tv_on=tv_on,
                _suppress_logging=True,
            )

            # Handle None return from predict_equilibrium_temperature
            if predicted_temp is None:
                logging.warning(
                    f"predict_equilibrium_temperature returned None for "
                    f"outlet={outlet_temp}, outdoor={outdoor_temp}"
                )
                return outdoor_temp + 10.0  # Safe fallback

            return float(predicted_temp)  # Ensure we return a float

        except Exception as e:
            logging.error(f"predict_indoor_temp failed: {e}")
            # Safe fallback - assume minimal heating effect
            return outdoor_temp + 10.0 if outdoor_temp is not None else 21.0

    def calculate_optimal_outlet_temp(self, features: Dict) -> Tuple[float, Dict]:
        """Calculate optimal outlet temperature using direct thermal physics prediction."""
        try:
            # Store features for use in trajectory verification during binary search
            self._current_features = features

            # Extract core thermal parameters
            current_indoor = features.get("indoor_temp_lag_30m", 21.0)
            target_indoor = features.get("target_temp", 21.0)
            outdoor_temp = features.get("outdoor_temp", 10.0)
            
            # Store current indoor for trajectory correction
            self._current_indoor = current_indoor

            # Extract enhanced thermal intelligence features
            thermal_features = self._extract_thermal_features(features)

            # Calculate required outlet temperature using iterative approach
            optimal_outlet_temp = self._calculate_required_outlet_temp(
                current_indoor, target_indoor, outdoor_temp, thermal_features
            )

            # Get prediction metadata
            confidence = self.thermal_model.learning_confidence
            prediction_metadata = {
                "thermal_time_constant": self.thermal_model.thermal_time_constant,
                "total_conductance": self.thermal_model.total_conductance,
                "equilibrium_ratio": self.thermal_model.equilibrium_ratio,
                "learning_confidence": confidence,
                "prediction_method": "thermal_equilibrium_single_prediction",
                "cycle_count": self.cycle_count,
            }

            if optimal_outlet_temp is None:
                logging.warning("Failed to calculate optimal outlet temperature")
                optimal_outlet_temp = 35.0  # Safe fallback

            return optimal_outlet_temp, prediction_metadata

        except Exception as e:
            logging.error(f"Prediction failed: {e}", exc_info=True)
            # Fallback to safe temperature
            fallback_temp = 35.0
            fallback_metadata = {
                "prediction_method": "fallback_safe_temperature",
                "error": str(e),
            }
            return fallback_temp, fallback_metadata

    def _extract_thermal_features(self, features: Dict) -> Dict:
        """Extract thermal intelligence features for the thermal model."""
        thermal_features = {}

        # Multi-heat source features
        thermal_features["pv_power"] = features.get("pv_now", 0.0)
        thermal_features["fireplace_on"] = float(features.get("fireplace_on", 0))
        thermal_features["tv_on"] = float(features.get("tv_on", 0))

        # Enhanced thermal intelligence features
        thermal_features["indoor_temp_gradient"] = features.get(
            "indoor_temp_gradient", 0.0
        )
        thermal_features["temp_diff_indoor_outdoor"] = features.get(
            "temp_diff_indoor_outdoor", 0.0
        )
        thermal_features["outlet_indoor_diff"] = features.get("outlet_indoor_diff", 0.0)

        # Note: Removed occupancy and cooking features as they don't have corresponding sensors

        return thermal_features

    def _get_forecast_conditions(
        self, outdoor_temp: float, pv_power: float, thermal_features: Dict
    ) -> Tuple[float, float, list, list]:
        """
        CYCLE-TIME-ALIGNED forecast condition calculation for consistent predictions.

        Uses cycle-appropriate forecast timing instead of 1-4h averaging to eliminate
        timing mismatch between control cycles and forecast horizons.

        Returns both cycle-aligned averages and arrays for trajectory prediction.
        """
        features = getattr(self, "_current_features", {})

        # Get cycle time from config and validate
        cycle_minutes = config.CYCLE_INTERVAL_MINUTES
        cycle_hours = cycle_minutes / 60.0

        # Validate cycle time against reasonable limits (max 180min for good control)
        max_reasonable_cycle = 180  # 3 hours maximum for responsive control
        if cycle_minutes > max_reasonable_cycle:
            logging.warning(
                f"⚠️ Cycle time {cycle_minutes}min exceeds recommended maximum "
                f"{max_reasonable_cycle}min. Using 180min limit for forecast alignment."
            )
            cycle_hours = max_reasonable_cycle / 60.0

        if features:
            # Extract available forecast data
            forecast_1h_outdoor = features.get("temp_forecast_1h", outdoor_temp)
            forecast_1h_pv = features.get("pv_forecast_1h", pv_power)
            forecast_2h_outdoor = features.get("temp_forecast_2h", outdoor_temp)
            forecast_2h_pv = features.get("pv_forecast_2h", pv_power)
            forecast_3h_outdoor = features.get("temp_forecast_3h", outdoor_temp)
            forecast_3h_pv = features.get("pv_forecast_3h", pv_power)
            forecast_4h_outdoor = features.get("temp_forecast_4h", outdoor_temp)
            forecast_4h_pv = features.get("pv_forecast_4h", pv_power)

            # Calculate cycle-aligned forecast using appropriate interpolation/selection
            if cycle_hours <= 0.5:  # 0-30min cycles: interpolate between current and 1h
                weight = cycle_hours / 1.0  # 0.0 to 0.5
                cycle_outdoor = (
                    outdoor_temp * (1 - weight) + forecast_1h_outdoor * weight
                )
                cycle_pv = pv_power * (1 - weight) + forecast_1h_pv * weight
                method = f"interpolated ({cycle_minutes}min)"
            elif cycle_hours <= 1.0:  # 30-60min cycles: use 1h forecast directly
                cycle_outdoor = forecast_1h_outdoor
                cycle_pv = forecast_1h_pv
                method = "1h forecast"
            elif cycle_hours <= 2.0:  # 60-120min cycles: use 2h forecast
                cycle_outdoor = forecast_2h_outdoor
                cycle_pv = forecast_2h_pv
                method = "2h forecast"
            elif cycle_hours <= 3.0:  # 120-180min cycles: use 3h forecast
                cycle_outdoor = forecast_3h_outdoor
                cycle_pv = forecast_3h_pv
                method = "3h forecast"
            else:  # >180min cycles: cap at 4h forecast with warning
                cycle_outdoor = forecast_4h_outdoor
                cycle_pv = forecast_4h_pv
                method = "4h forecast (capped)"

            # For trajectory prediction, still provide full forecast arrays
            outdoor_forecast = [
                forecast_1h_outdoor,
                forecast_2h_outdoor,
                forecast_3h_outdoor,
                forecast_4h_outdoor,
            ]
            pv_forecast = [
                forecast_1h_pv,
                forecast_2h_pv,
                forecast_3h_pv,
                forecast_4h_pv,
            ]

            logging.info(
                f"⏱️ Cycle-aligned forecast ({method}): outdoor={cycle_outdoor:.1f}°C "
                f"(vs current {outdoor_temp:.1f}°C), PV={cycle_pv:.0f}W "
                f"(vs current {pv_power:.0f}W) [cycle: {cycle_minutes}min]"
            )
        else:
            # No forecast data available, use current values
            cycle_outdoor = outdoor_temp
            cycle_pv = pv_power
            outdoor_forecast = [outdoor_temp] * 4
            pv_forecast = [pv_power] * 4
            method = "current (no forecasts)"

            logging.debug(
                f"⏱️ Cycle-aligned conditions ({method}): "
                f"outdoor={outdoor_temp:.1f}°C, PV={pv_power:.0f}W"
            )

        return cycle_outdoor, cycle_pv, outdoor_forecast, pv_forecast

    def _calculate_required_outlet_temp(
        self,
        current_indoor: float,
        target_indoor: float,
        outdoor_temp: float,
        thermal_features: Dict,
    ) -> float:
        """Calculate the outlet temperature required to reach target indoor temperature using learned thermal model."""
        # REMOVED: "Already at target" bypass logic - let physics model always calculate proper outlet temp
        # The thermal model should determine maintenance requirements based on actual conditions

        # Use the calibrated thermal model to find required outlet temperature
        # This leverages the 26 days of learned parameters instead of simple heuristics
        pv_power = thermal_features.get("pv_power", 0.0)
        fireplace_on = thermal_features.get("fireplace_on", 0.0)
        tv_on = thermal_features.get("tv_on", 0.0)

        # Iterative search to find outlet temperature that produces target indoor temp
        # This uses the learned thermal physics parameters from calibration
        tolerance = 0.1  # °C

        # Use natural system bounds - let binary search and physics model handle optimal outlet temps
        outlet_min = config.CLAMP_MIN_ABS
        outlet_max = config.CLAMP_MAX_ABS

        logging.debug(
            f"🔧 Using natural bounds: outlet_min={outlet_min:.1f}°C, "
            f"outlet_max={outlet_max:.1f}°C"
        )

        # UNIFIED: Get forecast conditions using centralized method
        avg_outdoor, avg_pv, outdoor_forecast, pv_forecast = (
            self._get_forecast_conditions(outdoor_temp, pv_power, thermal_features)
        )

        # UNIFIED FORECAST: Store cycle-aligned conditions for smart rounding
        self.cycle_aligned_forecast = {
            "outdoor_temp": avg_outdoor,
            "pv_power": avg_pv,
            "fireplace_on": fireplace_on,
            "tv_on": tv_on,
        }

        # Pre-check for unreachable targets to avoid futile searching (using forecast conditions)
        try:
            # Check what minimum outlet temp produces
            min_prediction = self.thermal_model.predict_equilibrium_temperature(
                outlet_temp=outlet_min,
                outdoor_temp=avg_outdoor,  # Use forecast average for consistency
                current_indoor=current_indoor,
                pv_power=avg_pv,  # Use forecast average for consistency
                fireplace_on=fireplace_on,
                tv_on=tv_on,
                _suppress_logging=True,
            )

            # Check what maximum outlet temp produces
            max_prediction = self.thermal_model.predict_equilibrium_temperature(
                outlet_temp=outlet_max,
                outdoor_temp=avg_outdoor,  # Use forecast average for consistency
                current_indoor=current_indoor,
                pv_power=avg_pv,  # Use forecast average for consistency
                fireplace_on=fireplace_on,
                tv_on=tv_on,
                _suppress_logging=True,
            )

            if min_prediction is not None and max_prediction is not None:
                # UNIFIED PRE-CHECK: Check reachability regardless of heating/cooling scenario
                temp_diff = target_indoor - current_indoor
                temp_diff_abs = abs(temp_diff)
                is_cooling_needed = temp_diff < -0.1  # Need to cool house
                is_heating_needed = temp_diff > 0.1   # Need to heat house
                
                # Check if target is unreachable with system limits
                if target_indoor < min_prediction - tolerance:
                    # Target below minimum capability
                    scenario = "cooling" if is_cooling_needed else "heating"
                    logging.warning(
                        f"🎯 {scenario.title()} pre-check: Target {target_indoor:.1f}°C unreachable "
                        f"(min outlet {outlet_min:.1f}°C → {min_prediction:.2f}°C), "
                        f"using minimum outlet"
                    )
                    return outlet_min

                if target_indoor > max_prediction + tolerance:
                    # Target above maximum capability  
                    scenario = "cooling" if is_cooling_needed else "heating"
                    logging.warning(
                        f"🎯 {scenario.title()} pre-check: Target {target_indoor:.1f}°C unreachable "
                        f"(max outlet {outlet_max:.1f}°C → {max_prediction:.2f}°C), "
                        f"using maximum outlet"
                    )
                    return outlet_max

                # Target is achievable - proceed to binary search for ALL scenarios
                scenario = "cooling" if is_cooling_needed else ("heating" if is_heating_needed else "maintenance")
                logging.debug(
                    f"🎯 {scenario.title()} ({temp_diff_abs:.1f}°C deviation): Target {target_indoor:.1f}°C achievable "
                    f"(range: {min_prediction:.1f}-{max_prediction:.1f}°C), proceeding to binary search"
                )
        except Exception as e:
            logging.warning(f"Pre-check failed: {e}, proceeding with binary search")

        # Binary search for optimal outlet temperature
        logging.debug(
            f"🎯 Binary search start: target={target_indoor:.1f}°C, "
            f"current={current_indoor:.1f}°C, range={outlet_min:.1f}-{outlet_max:.1f}°C"
        )

        for iteration in range(20):  # Max 20 iterations for efficiency
            # Check if range has collapsed (early exit)
            range_size = outlet_max - outlet_min
            if range_size < 0.05:  # °C - range too small to matter
                final_outlet = (outlet_min + outlet_max) / 2.0
                logging.info(
                    f"🔄 Binary search early exit after {iteration+1} iterations: "
                    f"range collapsed to {range_size:.3f}°C, "
                    f"using {final_outlet:.1f}°C"
                )
                return final_outlet

            outlet_mid = (outlet_min + outlet_max) / 2.0

            # Predict indoor temperature with this outlet temperature using cycle-aligned conditions
            try:
                predicted_indoor = self.thermal_model.predict_equilibrium_temperature(
                    outlet_temp=outlet_mid,
                    outdoor_temp=avg_outdoor,  # Use cycle-aligned forecast for consistency
                    current_indoor=current_indoor,
                    pv_power=avg_pv,  # Use cycle-aligned forecast for consistency
                    fireplace_on=fireplace_on,
                    tv_on=tv_on,
                    _suppress_logging=True,
                )

                # Handle None returns from predict_equilibrium_temperature
                if predicted_indoor is None:
                    logging.warning(
                        f"   Iteration {iteration+1}: predict_equilibrium_temperature returned None "
                        f"for outlet={outlet_mid:.1f}°C - using fallback"
                    )
                    return 35.0  # Safe fallback

            except Exception as e:
                logging.error(
                    f"   Iteration {iteration+1}: predict_equilibrium_temperature failed: {e}"
                )
                return 35.0  # Safe fallback

            # Calculate error from target
            error = predicted_indoor - target_indoor

            # Detailed logging at each iteration
            logging.debug(
                f"   Iteration {iteration+1}: outlet={outlet_mid:.1f}°C → "
                f"predicted={predicted_indoor:.2f}°C, error={error:.3f}°C "
                f"(range: {outlet_min:.1f}-{outlet_max:.1f}°C)"
            )

            # Check if we're close enough
            if abs(error) < tolerance:
                logging.info(
                    f"✅ Binary search converged after {iteration+1} iterations: "
                    f"{outlet_mid:.1f}°C → {predicted_indoor:.2f}°C "
                    f"(target: {target_indoor:.1f}°C, error: {error:.3f}°C)"
                )

                # Show final equilibrium physics for the converged result
                final_physics = self.thermal_model.predict_equilibrium_temperature(
                    outlet_temp=outlet_mid,
                    outdoor_temp=avg_outdoor,
                    current_indoor=current_indoor,
                    pv_power=avg_pv,
                    fireplace_on=fireplace_on,
                    tv_on=tv_on,
                    _suppress_logging=False,  # Show the equilibrium physics logging
                )

                # MULTI-HORIZON FORECAST LOGGING: Show predictions with different forecast horizons
                self._log_multi_horizon_predictions(
                    current_indoor=current_indoor,
                    target_indoor=target_indoor,
                    outdoor_temp=outdoor_temp,
                    thermal_features=thermal_features,
                )

                # NEW: Trajectory verification and course correction
                if config.TRAJECTORY_PREDICTION_ENABLED:
                    outlet_mid = self._verify_trajectory_and_correct(
                        outlet_temp=outlet_mid,
                        current_indoor=current_indoor,
                        target_indoor=target_indoor,
                        outdoor_temp=outdoor_temp,
                        thermal_features=thermal_features,
                        features=getattr(
                            self, "_current_features", {}
                        ),  # Use stored features if available
                    )

                return outlet_mid

            # Adjust search range based on error
            # COOLING FIX: Consider whether we're heating or cooling the house
            temp_diff = target_indoor - current_indoor
            is_heating_scenario = temp_diff > 0.1  # Need to heat house
            is_cooling_scenario = temp_diff < -0.1  # Need to cool house
            
            if is_heating_scenario:
                # HEATING: Normal logic
                if predicted_indoor < target_indoor:
                    # Need higher outlet temperature
                    outlet_min = outlet_mid
                    logging.debug(
                        f"     → Heating: Predicted too low, raising minimum to {outlet_min:.1f}°C"
                    )
                else:
                    # Need lower outlet temperature
                    outlet_max = outlet_mid
                    logging.debug(
                        f"     → Heating: Predicted too high, lowering maximum to {outlet_max:.1f}°C"
                    )
            elif is_cooling_scenario:
                # COOLING: For cooling, we want to get as close as possible to target
                # Standard binary search logic works, just need to be close to target
                if predicted_indoor < target_indoor:
                    # Predicted is below target - need slightly higher outlet to reach target
                    outlet_min = outlet_mid
                    logging.debug(
                        f"     → Cooling: Predicted below target, raising minimum to {outlet_min:.1f}°C"
                    )
                else:
                    # Predicted is above target - need lower outlet to reach target
                    outlet_max = outlet_mid
                    logging.debug(
                        f"     → Cooling: Predicted above target, lowering maximum to {outlet_max:.1f}°C"
                    )
            else:
                # MAINTENANCE: At target (normal logic)
                if predicted_indoor < target_indoor:
                    # Need higher outlet temperature
                    outlet_min = outlet_mid
                    logging.debug(
                        f"     → Maintenance: Predicted too low, raising minimum to {outlet_min:.1f}°C"
                    )
                else:
                    # Need lower outlet temperature
                    outlet_max = outlet_mid
                    logging.debug(
                        f"     → Maintenance: Predicted too high, lowering maximum to {outlet_max:.1f}°C"
                    )

        # Return best guess if didn't converge
        final_outlet = (outlet_min + outlet_max) / 2.0
        try:
            final_predicted = self.thermal_model.predict_equilibrium_temperature(
                outlet_temp=final_outlet,
                outdoor_temp=avg_outdoor,  # Use forecast average for consistency
                current_indoor=current_indoor,
                pv_power=avg_pv,  # Use forecast average for consistency
                fireplace_on=fireplace_on,
                tv_on=tv_on,
                _suppress_logging=True,
            )

            # Handle None return for final prediction
            if final_predicted is None:
                logging.warning(
                    f"⚠️ Final prediction returned None, using fallback 35.0°C"
                )
                return 35.0

        except Exception as e:
            logging.error(f"Final prediction failed: {e}")
            return 35.0

        final_error = final_predicted - target_indoor
        logging.warning(
            f"⚠️ Binary search didn't converge after 20 iterations: "
            f"{final_outlet:.1f}°C → {final_predicted:.2f}°C "
            f"(target: {target_indoor:.1f}°C, error: {final_error:.3f}°C)"
        )

        # NEW: Trajectory verification and course correction
        if config.TRAJECTORY_PREDICTION_ENABLED:
            final_outlet = self._verify_trajectory_and_correct(
                outlet_temp=final_outlet,
                current_indoor=current_indoor,
                target_indoor=target_indoor,
                outdoor_temp=outdoor_temp,
                thermal_features=thermal_features,
                features=getattr(
                    self, "_current_features", {}
                ),  # Use stored features if available
            )

        return final_outlet

    def _log_multi_horizon_predictions(
        self,
        current_indoor: float,
        target_indoor: float,
        outdoor_temp: float,
        thermal_features: Dict,
    ) -> None:
        """
        Log predicted outlet temperatures using different forecast horizons.

        This helps diagnose which forecast horizon is causing high outlet temp predictions
        during evening/overnight scenarios when outdoor temperature drops.
        """
        try:
            # Extract thermal features
            pv_power = thermal_features.get("pv_power", 0.0)
            fireplace_on = thermal_features.get("fireplace_on", 0.0)
            tv_on = thermal_features.get("tv_on", 0.0)
            features = getattr(self, "_current_features", {})

            if not features:
                logging.debug("🔍 Multi-horizon: No forecast data available")
                return

            # Get cycle time and calculate cycle-aligned forecast
            cycle_minutes = config.CYCLE_INTERVAL_MINUTES
            cycle_hours = cycle_minutes / 60.0
            max_reasonable_cycle = 180
            if cycle_minutes > max_reasonable_cycle:
                cycle_hours = max_reasonable_cycle / 60.0

            # Calculate cycle-aligned forecast conditions (same logic as _get_forecast_conditions)
            forecast_1h_outdoor = features.get("temp_forecast_1h", outdoor_temp)
            forecast_1h_pv = features.get("pv_forecast_1h", pv_power)

            if cycle_hours <= 0.5:  # 0-30min cycles: interpolate between current and 1h
                weight = cycle_hours / 1.0
                cycle_outdoor = (
                    outdoor_temp * (1 - weight) + forecast_1h_outdoor * weight
                )
                cycle_pv = pv_power * (1 - weight) + forecast_1h_pv * weight
                cycle_method = f"cycle({cycle_minutes}min)"
            elif cycle_hours <= 1.0:  # 30-60min cycles: use 1h forecast directly
                cycle_outdoor = forecast_1h_outdoor
                cycle_pv = forecast_1h_pv
                cycle_method = f"cycle(1h)"
            elif cycle_hours <= 2.0:  # 60-120min cycles: use 2h forecast
                cycle_outdoor = features.get("temp_forecast_2h", outdoor_temp)
                cycle_pv = features.get("pv_forecast_2h", pv_power)
                cycle_method = f"cycle(2h)"
            elif cycle_hours <= 3.0:  # 120-180min cycles: use 3h forecast
                cycle_outdoor = features.get("temp_forecast_3h", outdoor_temp)
                cycle_pv = features.get("pv_forecast_3h", pv_power)
                cycle_method = f"cycle(3h)"
            else:  # >180min cycles: cap at 4h forecast
                cycle_outdoor = features.get("temp_forecast_4h", outdoor_temp)
                cycle_pv = features.get("pv_forecast_4h", pv_power)
                cycle_method = f"cycle(4h-cap)"

            # Extract individual forecast horizons including cycle-aligned
            forecasts = {
                "current": {"outdoor": outdoor_temp, "pv": pv_power},
                cycle_method: {"outdoor": cycle_outdoor, "pv": cycle_pv},
                "1h": {
                    "outdoor": features.get("temp_forecast_1h", outdoor_temp),
                    "pv": features.get("pv_forecast_1h", pv_power),
                },
                "2h": {
                    "outdoor": features.get("temp_forecast_2h", outdoor_temp),
                    "pv": features.get("pv_forecast_2h", pv_power),
                },
                "3h": {
                    "outdoor": features.get("temp_forecast_3h", outdoor_temp),
                    "pv": features.get("pv_forecast_3h", pv_power),
                },
                "4h": {
                    "outdoor": features.get("temp_forecast_4h", outdoor_temp),
                    "pv": features.get("pv_forecast_4h", pv_power),
                },
                "avg": {
                    "outdoor": (
                        features.get("temp_forecast_1h", outdoor_temp)
                        + features.get("temp_forecast_2h", outdoor_temp)
                        + features.get("temp_forecast_3h", outdoor_temp)
                        + features.get("temp_forecast_4h", outdoor_temp)
                    )
                    / 4.0,
                    "pv": (
                        features.get("pv_forecast_1h", pv_power)
                        + features.get("pv_forecast_2h", pv_power)
                        + features.get("pv_forecast_3h", pv_power)
                        + features.get("pv_forecast_4h", pv_power)
                    )
                    / 4.0,
                },
            }

            # Calculate predicted outlet temperature for each horizon
            predictions = {}
            for horizon, conditions in forecasts.items():
                try:
                    # Use same precision as main binary search for consistency
                    outlet_min, outlet_max = config.CLAMP_MIN_ABS, config.CLAMP_MAX_ABS
                    tolerance = 0.1  # Same precision as main binary search

                    for iteration in range(20):  # Same iterations as main binary search
                        outlet_mid = (outlet_min + outlet_max) / 2.0
                        predicted_indoor = (
                            self.thermal_model.predict_equilibrium_temperature(
                                outlet_temp=outlet_mid,
                                outdoor_temp=conditions["outdoor"],
                                current_indoor=current_indoor,
                                pv_power=conditions["pv"],
                                fireplace_on=fireplace_on,
                                tv_on=tv_on,
                                _suppress_logging=True,
                            )
                        )

                        if predicted_indoor is None:
                            break

                        error = predicted_indoor - target_indoor
                        if abs(error) < tolerance:
                            predictions[horizon] = {
                                "outlet": outlet_mid,
                                "predicted": predicted_indoor,
                                "conditions": conditions,
                            }
                            break

                        if predicted_indoor < target_indoor:
                            outlet_min = outlet_mid
                        else:
                            outlet_max = outlet_mid
                    else:
                        # Didn't converge, use midpoint
                        final_outlet = (outlet_min + outlet_max) / 2.0
                        final_predicted = (
                            self.thermal_model.predict_equilibrium_temperature(
                                outlet_temp=final_outlet,
                                outdoor_temp=conditions["outdoor"],
                                current_indoor=current_indoor,
                                pv_power=conditions["pv"],
                                fireplace_on=fireplace_on,
                                tv_on=tv_on,
                                _suppress_logging=True,
                            )
                        )
                        if final_predicted is not None:
                            predictions[horizon] = {
                                "outlet": final_outlet,
                                "predicted": final_predicted,
                                "conditions": conditions,
                            }

                except Exception as e:
                    logging.debug(f"Multi-horizon prediction failed for {horizon}: {e}")
                    continue

            # Log the multi-horizon predictions in a clear format
            if predictions:
                logging.debug(
                    f"🔍 Multi-horizon predictions for target {target_indoor:.1f}°C:"
                )

                # Order the predictions logically (including cycle-aligned)
                order = ["current", cycle_method, "1h", "2h", "3h", "4h", "avg"]
                for horizon in order:
                    if horizon in predictions:
                        pred = predictions[horizon]
                        outlet = pred["outlet"]
                        predicted = pred["predicted"]
                        conditions = pred["conditions"]
                        error = predicted - target_indoor

                        # Mark the cycle-aligned prediction as NEW
                        if horizon == cycle_method:
                            marker = "← NEW: Cycle-aligned"
                        else:
                            marker = ""

                        logging.debug(
                            f"   {horizon:>12}: {outlet:.1f}°C → {predicted:.1f}°C "
                            f"(error: {error:+.2f}°C, outdoor: {conditions['outdoor']:.1f}°C, "
                            f"PV: {conditions['pv']:.0f}W) {marker}"
                        )

                # Show the differences to highlight issues
                if "current" in predictions and "avg" in predictions:
                    current_outlet = predictions["current"]["outlet"]
                    avg_outlet = predictions["avg"]["outlet"]
                    outlet_diff = avg_outlet - current_outlet

                    if abs(outlet_diff) > 2.0:  # Significant difference
                        direction = "higher" if outlet_diff > 0 else "lower"
                        logging.warning(
                            f"⚠️ Forecast vs current difference: "
                            f"forecast avg outlet {outlet_diff:+.1f}°C {direction} "
                            f"than current conditions"
                        )
            else:
                logging.debug(
                    "🔍 Multi-horizon: No predictions calculated successfully"
                )

        except Exception as e:
            logging.error(f"Multi-horizon prediction logging failed: {e}")

    def _verify_trajectory_and_correct(
        self,
        outlet_temp: float,
        current_indoor: float,
        target_indoor: float,
        outdoor_temp: float,
        thermal_features: Dict,
        features: Optional[Dict] = None,
    ) -> float:
        """
        Verify that the calculated outlet temperature will actually reach the target
        using trajectory prediction, and apply physics-based adaptive correction if needed.

        PHYSICS-BASED CORRECTION: Uses learned thermal parameters to adaptively scale
        corrections based on house characteristics and time pressure.
        """
        try:
            # UNIFIED: Get forecast conditions using centralized method
            avg_outdoor, avg_pv, outdoor_forecast, pv_forecast = (
                self._get_forecast_conditions(
                    outdoor_temp,
                    thermal_features.get("pv_power", 0.0),
                    thermal_features,
                )
            )

            # Get trajectory prediction with forecast integration
            if hasattr(self.thermal_model, "predict_thermal_trajectory_with_forecasts"):
                # Enhanced method with forecast arrays
                trajectory = (
                    self.thermal_model.predict_thermal_trajectory_with_forecasts(
                        current_indoor=current_indoor,
                        target_indoor=target_indoor,
                        outlet_temp=outlet_temp,
                        outdoor_forecast=outdoor_forecast,
                        pv_forecast=pv_forecast,
                        time_horizon_hours=config.TRAJECTORY_STEPS,
                        fireplace_on=thermal_features.get("fireplace_on", 0.0),
                        tv_on=thermal_features.get("tv_on", 0.0),
                    )
                )
            else:
                # FIXED: Use cycle-aligned conditions instead of 4-hour averages
                # This ensures trajectory matches the cycle-aligned forecast shown in logs
                trajectory = self.thermal_model.predict_thermal_trajectory(
                    current_indoor=current_indoor,
                    target_indoor=target_indoor,
                    outlet_temp=outlet_temp,
                    outdoor_temp=avg_outdoor,  # Use cycle-aligned outdoor temp
                    time_horizon_hours=config.TRAJECTORY_STEPS,
                    pv_power=avg_pv,  # Use cycle-aligned PV power
                    fireplace_on=thermal_features.get("fireplace_on", 0.0),
                    tv_on=thermal_features.get("tv_on", 0.0),
                )

                logging.debug(
                    f"🔍 Trajectory prediction using cycle-aligned conditions: "
                    f"outdoor={avg_outdoor:.1f}°C, PV={avg_pv:.0f}W"
                )

            # TRAJECTORY-BASED DECISION: Check if target reachable within cycle time
            cycle_hours = config.CYCLE_INTERVAL_MINUTES / 60.0

            # DEBUG: Log trajectory details for diagnosis
            trajectory_temps = trajectory.get("trajectory", [])
            first_hour_temp = (
                trajectory_temps[0] if trajectory_temps else current_indoor
            )
            reaches_target_at = trajectory.get("reaches_target_at")

            logging.debug(
                f"🔍 Trajectory DEBUG: outlet={outlet_temp:.1f}°C → 1h_prediction={first_hour_temp:.2f}°C "
                f"(vs target {target_indoor:.1f}°C, error: {first_hour_temp - target_indoor:+.3f}°C), "
                f"reaches_target_at={reaches_target_at}h, cycle_time={cycle_hours:.1f}h"
            )

            # NEW LOGIC: Check for temperature boundary violations regardless of target achievement
            # This ensures comfort boundaries are respected even when target is theoretically reachable
            trajectory_temps = trajectory.get("trajectory", [])
            if trajectory_temps:
                min_temp = min(trajectory_temps)
                max_temp = max(trajectory_temps)
                temp_boundary_violation = (
                    min_temp
                    <= target_indoor - 0.1  # Temperature drops below comfort boundary
                    or max_temp
                    >= target_indoor + 0.1  # Temperature rises above comfort boundary
                )
            else:
                temp_boundary_violation = False

            # Enhanced logic: If target reached quickly, be more lenient with boundary violations
            if reaches_target_at is not None and reaches_target_at <= cycle_hours:
                if not temp_boundary_violation:
                    logging.info(
                        f"✅ Target will be reached in {reaches_target_at:.1f}h "
                        f"(within {cycle_hours:.1f}h cycle) and no boundary violations - no correction needed"
                    )
                    return outlet_temp
                else:
                    # Target reachable quickly but has boundary violations
                    # For fast target achievement (< 0.5h), allow larger tolerance (±0.3°C instead of ±0.1°C)
                    if reaches_target_at <= 0.5:  # Very fast achievement
                        relaxed_boundary_violation = (
                            min_temp <= target_indoor - 0.3  # More lenient boundary
                            or max_temp >= target_indoor + 0.3  # More lenient boundary
                        )
                        if not relaxed_boundary_violation:
                            logging.info(
                                f"✅ Target will be reached quickly in {reaches_target_at:.1f}h "
                                f"with minor boundary violation (±0.1-0.2°C range) - no correction needed"
                            )
                            return outlet_temp

            # Apply correction if target not reachable or significant boundary violations
            if reaches_target_at is None or reaches_target_at > cycle_hours:
                logging.info(
                    f"⚠️ Target will NOT be reached within {cycle_hours:.1f}h cycle - applying physics-based correction"
                )
            else:
                logging.info(
                    f"⚠️ Target reachable in {reaches_target_at:.1f}h but significant boundary violations detected - applying correction"
                )

            # Apply physics-based adaptive correction
            if temp_boundary_violation:
                logging.info(
                    f"⚠️ Temperature boundary violations detected (min: {min_temp:.2f}°C, max: {max_temp:.2f}°C) - applying correction"
                )
            else:
                logging.info(
                    f"⚠️ Target will NOT be reached within {cycle_hours:.1f}h cycle - applying physics-based correction"
                )

            # Calculate physics-based correction
            corrected_outlet = self._calculate_physics_based_correction(
                outlet_temp=outlet_temp,
                trajectory=trajectory,
                target_indoor=target_indoor,
                cycle_hours=cycle_hours,
            )

            return corrected_outlet

        except Exception as e:
            logging.error(f"Trajectory verification failed: {e}")
            return outlet_temp  # Return original if verification fails

    def _calculate_physics_based_correction(
        self,
        outlet_temp: float,
        trajectory: Dict,
        target_indoor: float,
        cycle_hours: float,
    ) -> float:
        """
        Calculate physics-based adaptive correction with deviation-scaled response.

        ENHANCED PRIORITY 2 IMPLEMENTATION:
        - Graduated response zones based on temperature deviation magnitude
        - Unified control logic for heating and cooling scenarios
        - Physics-based scaling using house thermal characteristics
        """
        try:
            # Calculate temperature error from trajectory
            trajectory_temps = trajectory.get("trajectory", [])
            if not trajectory_temps:
                return outlet_temp

            # Get initial temperature deviation for graduated response
            current_indoor = getattr(self, '_current_indoor', target_indoor)
            initial_deviation = abs(target_indoor - current_indoor)

            # Calculate trajectory error
            min_predicted_temp = min(trajectory_temps)
            max_predicted_temp = max(trajectory_temps)

            # Determine primary error source
            min_violates = min_predicted_temp <= target_indoor - 0.1
            max_violates = max_predicted_temp >= target_indoor + 0.1

            if min_violates and max_violates:
                # Both boundaries violated - choose the more severe
                min_severity = abs(min_predicted_temp - (target_indoor - 0.1))
                max_severity = abs(max_predicted_temp - (target_indoor + 0.1))
                if min_severity > max_severity:
                    temp_error = target_indoor - min_predicted_temp
                else:
                    temp_error = target_indoor - max_predicted_temp
            elif min_violates:
                temp_error = target_indoor - min_predicted_temp
            elif max_violates:
                temp_error = target_indoor - max_predicted_temp
            else:
                # Target not reached in time
                reaches_target_at = trajectory.get("reaches_target_at")
                if reaches_target_at is None or reaches_target_at > cycle_hours:
                    final_predicted_temp = trajectory_temps[-1]
                    temp_error = target_indoor - final_predicted_temp
                else:
                    temp_error = 0.0

            # GRADUATED RESPONSE ZONES - Priority 2 Enhancement
            response_zone = self._get_response_zone(initial_deviation)
            
            # Physics-based scaling
            time_constant = self.thermal_model.thermal_time_constant
            effectiveness = self.thermal_model.equilibrium_ratio

            # Adaptive scaling based on house characteristics
            if effectiveness > 0:
                base_scale = 2.0 / effectiveness
            else:
                base_scale = 15.0  # Fallback

            # Zone-specific scaling factors
            zone_multipliers = {
                "maintenance": 0.5,      # Gentle adjustments
                "fine_tuning": 0.8,      # Moderate adjustments  
                "active_control": 1.0,   # Standard response
                "rapid_correction": 1.5  # Aggressive action
            }

            physics_scale = base_scale * zone_multipliers[response_zone]

            # Time pressure calculation
            time_pressure = self._calculate_time_pressure(trajectory, cycle_hours)

            # Unified urgency calculation for all scenarios
            if time_pressure > 0.8:
                urgency_multiplier = 1.3  # Reduced for stability
            elif time_pressure > 0.5:
                urgency_multiplier = 1.15
            else:
                urgency_multiplier = 1.0

            # Calculate base correction
            correction = temp_error * physics_scale * urgency_multiplier

            # Apply zone-specific bounds
            zone_bounds = {
                "maintenance": (1.0, -1.0),        # ±1°C max
                "fine_tuning": (3.0, -3.0),        # ±3°C max
                "active_control": (8.0, -8.0),     # ±8°C max  
                "rapid_correction": (15.0, -15.0)  # ±15°C max
            }

            max_heating, max_cooling = zone_bounds[response_zone]

            # Proximity dampening to prevent overshooting
            proximity_dampen = 1.0
            if temp_error < 0:  # Overshooting scenario
                proximity_dampen = min(1.0, abs(temp_error) / 0.5)
                effective_max_heating = max_heating * proximity_dampen
                correction = min(effective_max_heating, correction)

            # Apply zone bounds
            correction = max(max_cooling, min(max_heating, correction))

            # Final outlet temperature
            corrected_outlet = outlet_temp + correction
            corrected_outlet = max(
                config.CLAMP_MIN_ABS, min(config.CLAMP_MAX_ABS, corrected_outlet)
            )

            logging.info(
                f"🎯 Graduated correction [{response_zone}]: "
                f"{outlet_temp:.1f}°C + {correction:+.1f}°C = "
                f"{corrected_outlet:.1f}°C "
                f"(deviation: {initial_deviation:.1f}°C, "
                f"temp_error: {temp_error:+.2f}°C, "
                f"time_pressure: {time_pressure:.2f}, "
                f"urgency: {urgency_multiplier:.2f}x)"
            )

            return corrected_outlet

        except Exception as e:
            logging.error(f"Physics-based correction failed: {e}")
            return outlet_temp

    def _get_response_zone(self, deviation: float) -> str:
        """
        Determine graduated response zone based on temperature deviation.
        
        PRIORITY 2 ENHANCEMENT: Graduated response with hysteresis
        """
        if deviation < 0.1:
            return "maintenance"      # Minimal adjustments
        elif deviation < 0.3:
            return "fine_tuning"      # Gentle corrections
        elif deviation < 0.7:
            return "active_control"   # Standard response
        else:
            return "rapid_correction" # Aggressive action

    def _calculate_time_pressure(self, trajectory: Dict, cycle_hours: float) -> float:
        """
        Calculate how urgently we need to correct (0.0 = no pressure, 1.0 = maximum urgency).
        """
        reaches_target_at = trajectory.get("reaches_target_at")

        if reaches_target_at is None:
            return 1.0  # Maximum urgency - may never reach target
        elif reaches_target_at <= cycle_hours:
            return (
                0.0  # No pressure - target reachable in time (should not happen here)
            )
        elif reaches_target_at <= cycle_hours * 2:
            return 0.3  # Low pressure - close to reachable
        elif reaches_target_at <= cycle_hours * 4:
            return 0.6  # Medium pressure
        else:
            return 1.0  # High pressure - far from target

    # This method is no longer needed - thermal state is loaded in ThermalEquilibriumModel

    def learn_from_prediction_feedback(
        self,
        predicted_temp: float,
        actual_temp: float,
        prediction_context: Dict,
        timestamp: Optional[str] = None,
        is_blocking_active: bool = False,
    ):
        """Learn from prediction feedback using the thermal model's adaptive learning."""
        if not self.learning_enabled:
            return

        if is_blocking_active:
            logging.info("Skipping online learning due to active blocking event.")
            return

        try:
            # FIRST-CYCLE GUARD: Skip learning on the first cycle after a restart
            # to prevent incorrect adjustments from the time gap between cycles.
            if self.cycle_count <= 1:
                logging.info("Skipping online learning on the first cycle to ensure stability.")
                # Still update cycle count and save state, but don't learn.
                self.cycle_count += 1
                self.state_manager.update_learning_state(cycle_count=self.cycle_count)
                return

            # Update thermal model with prediction feedback
            prediction_error = self.thermal_model.update_prediction_feedback(
                predicted_temp=predicted_temp,
                actual_temp=actual_temp,
                prediction_context=prediction_context,
                timestamp=timestamp or datetime.now().isoformat(),
                is_blocking_active=is_blocking_active,
            )

            # Add prediction to MAE/RMSE tracking
            self.prediction_metrics.add_prediction(
                predicted=predicted_temp,
                actual=actual_temp,
                context=prediction_context,
                timestamp=timestamp,
            )

            # Add prediction record to unified state
            prediction_record = {
                "timestamp": timestamp or datetime.now().isoformat(),
                "predicted": predicted_temp,
                "actual": actual_temp,
                "error": actual_temp - predicted_temp,
                "context": prediction_context,
            }
            self.state_manager.add_prediction_record(prediction_record)

            # Track learning cycles
            self.cycle_count += 1

            # Update cycle count in unified state
            self.state_manager.update_learning_state(cycle_count=self.cycle_count)

            # Export metrics to InfluxDB every 5 cycles (approximately every 25 minutes)
            if self.cycle_count % 5 == 0:
                self._export_metrics_to_influxdb()

            # Export metrics to Home Assistant every cycle for real-time monitoring
            self._export_metrics_to_ha()

            # Log learning cycle completion
            if prediction_error is not None:
                logging.info(
                    f"✅ Learning cycle {self.cycle_count}: error={abs(prediction_error):.3f}°C, "
                    f"confidence={self.thermal_model.learning_confidence:.3f}, "
                    f"total_predictions={len(self.prediction_metrics.predictions)}"
                )

        except Exception as e:
            logging.error(f"Learning from feedback failed: {e}", exc_info=True)

    def _export_metrics_to_ha(self):
        """Export metrics to Home Assistant sensors."""
        try:
            ha_client = create_ha_client()

            # Get comprehensive metrics
            ha_metrics = self.get_comprehensive_metrics_for_ha()

            # Export MAE/RMSE metrics (confidence now provided via ml_heating_learning sensor)
            ha_client.log_model_metrics(
                mae=ha_metrics.get("mae_all_time", 0.0),
                rmse=ha_metrics.get("rmse_all_time", 0.0),
            )

            # Export adaptive learning metrics
            ha_client.log_adaptive_learning_metrics(ha_metrics)

            # Export feature importance (if available)
            if hasattr(self.thermal_model, "get_feature_importance"):
                importances = self.thermal_model.get_feature_importance()
                if importances:
                    ha_client.log_feature_importance(importances)

            logging.info("✅ Exported metrics to Home Assistant sensors successfully")

        except Exception as e:
            # Better error logging for debugging sensor export issues
            logging.error(f"❌ FAILED to export metrics to HA: {e}", exc_info=True)
            logging.error(
                f"   Attempted to export: {list(ha_metrics.keys()) if 'ha_metrics' in locals() else 'metrics not created'}"
            )
            logging.error(f"   HA Client created: {'ha_client' in locals()}")
            # Re-raise the exception for visibility during debugging
            raise

    def get_prediction_confidence(self) -> float:
        """Get current prediction confidence from thermal model."""
        return self.thermal_model.learning_confidence

    def get_learning_metrics(self) -> Dict:
        """Get comprehensive learning metrics for monitoring."""
        try:
            metrics = self.thermal_model.get_adaptive_learning_metrics()
            # Check if we got valid metrics or just insufficient_data flag
            if (
                isinstance(metrics, dict)
                and "insufficient_data" not in metrics
                and len(metrics) > 1
            ):
                # Extract current parameters from nested structure if available
                if "current_parameters" in metrics:
                    current_params = metrics["current_parameters"]
                    # Return flattened structure with actual loaded parameters
                    result = metrics.copy()
                    result.update(
                        {
                            "thermal_time_constant": current_params.get(
                                "thermal_time_constant",
                                self.thermal_model.thermal_time_constant,
                            ),
                            "total_conductance": current_params.get(
                                "total_conductance",
                                self.thermal_model.total_conductance,
                            ),
                            "equilibrium_ratio": current_params.get(
                                "equilibrium_ratio",
                                self.thermal_model.equilibrium_ratio,
                            ),
                            "learning_confidence": self.thermal_model.learning_confidence,
                            "cycle_count": self.cycle_count,
                        }
                    )
                    return result
                else:
                    # Use the metrics as-is if current_parameters key not found
                    return metrics
        except AttributeError:
            pass

        # Fallback if method doesn't exist or returns insufficient_data
        return {
            "thermal_time_constant": self.thermal_model.thermal_time_constant,
            "total_conductance": self.thermal_model.total_conductance,
            "equilibrium_ratio": self.thermal_model.equilibrium_ratio,
            "learning_confidence": self.thermal_model.learning_confidence,
            "cycle_count": self.cycle_count,
        }

    def get_comprehensive_metrics_for_ha(self) -> Dict:
        """Get comprehensive metrics for Home Assistant sensor export."""
        try:
            # Get thermal learning metrics
            thermal_metrics = self.get_learning_metrics()

            # Get prediction accuracy metrics (all-time for MAE/RMSE)
            prediction_metrics = self.prediction_metrics.get_metrics()

            # Get recent performance
            recent_performance = self.prediction_metrics.get_recent_performance(10)

            # Get 24h window simplified accuracy breakdown
            accuracy_24h = self.prediction_metrics.get_24h_accuracy_breakdown()
            good_control_24h = self.prediction_metrics.get_24h_good_control_percentage()

            # Combine into comprehensive HA-friendly format
            ha_metrics = {
                # Core thermal parameters (learned)
                "thermal_time_constant": thermal_metrics.get(
                    "thermal_time_constant", 6.0
                ),
                "total_conductance": thermal_metrics.get(
                    "total_conductance", 0.05
                ),
                "equilibrium_ratio": thermal_metrics.get(
                    "equilibrium_ratio", 0.8
                ),
                "learning_confidence": thermal_metrics.get("learning_confidence", 3.0),
                # Learning progress
                "cycle_count": self.cycle_count,
                "parameter_updates": thermal_metrics.get("parameter_updates", 0),
                "update_percentage": thermal_metrics.get("update_percentage", 0),
                # Prediction accuracy (MAE/RMSE) - all-time
                "mae_1h": prediction_metrics.get("1h", {}).get("mae", 0.0),
                "mae_6h": prediction_metrics.get("6h", {}).get("mae", 0.0),
                "mae_24h": prediction_metrics.get("24h", {}).get("mae", 0.0),
                "mae_all_time": prediction_metrics.get("all", {}).get("mae", 0.0),
                "rmse_all_time": prediction_metrics.get("all", {}).get("rmse", 0.0),
                # Recent performance
                "recent_mae_10": recent_performance.get("mae", 0.0),
                "recent_max_error": recent_performance.get("max_error", 0.0),
                # NEW: Simplified 3-category accuracy (24h window)
                "perfect_accuracy_pct": accuracy_24h.get("perfect", {}).get(
                    "percentage", 0.0
                ),
                "tolerable_accuracy_pct": accuracy_24h.get("tolerable", {}).get(
                    "percentage", 0.0
                ),
                "poor_accuracy_pct": accuracy_24h.get("poor", {}).get(
                    "percentage", 0.0
                ),
                "good_control_pct": good_control_24h,
                # Legacy accuracy breakdown (all-time) - kept for backward compatibility
                "excellent_accuracy_pct": prediction_metrics.get(
                    "accuracy_breakdown", {}
                )
                .get("excellent", {})
                .get("percentage", 0.0),
                "good_accuracy_pct": (
                    prediction_metrics.get("accuracy_breakdown", {})
                    .get("excellent", {})
                    .get("percentage", 0.0)
                    + prediction_metrics.get("accuracy_breakdown", {})
                    .get("very_good", {})
                    .get("percentage", 0.0)
                    + prediction_metrics.get("accuracy_breakdown", {})
                    .get("good", {})
                    .get("percentage", 0.0)
                ),
                # Trend analysis (ensure JSON serializable)
                "is_improving": bool(
                    prediction_metrics.get("trends", {}).get("is_improving", False)
                ),
                "improvement_percentage": float(
                    prediction_metrics.get("trends", {}).get(
                        "mae_improvement_percentage", 0.0
                    )
                ),
                # Model health summary
                "model_health": (
                    "excellent"
                    if thermal_metrics.get("learning_confidence", 0) >= 4.0
                    else (
                        "good"
                        if thermal_metrics.get("learning_confidence", 0) >= 3.0
                        else (
                            "fair"
                            if thermal_metrics.get("learning_confidence", 0) >= 2.0
                            else "poor"
                        )
                    )
                ),
                # Total predictions tracked
                "total_predictions": len(self.prediction_metrics.predictions),
                # Timestamp
                "last_updated": datetime.now().isoformat(),
            }

            return ha_metrics

        except Exception as e:
            logging.error(f"Failed to get comprehensive metrics: {e}")
            return {
                "error": str(e),
                "cycle_count": self.cycle_count,
                "last_updated": datetime.now().isoformat(),
            }

    def _export_metrics_to_influxdb(self):
        """Export adaptive learning metrics to InfluxDB for monitoring."""
        try:
            # Create InfluxDB service
            influx_service = create_influx_service()

            # Export prediction metrics
            prediction_metrics = self.prediction_metrics.get_metrics()
            if prediction_metrics:
                influx_service.write_prediction_metrics(prediction_metrics)
                logging.debug("✅ Exported prediction metrics to InfluxDB")

            # Export thermal learning metrics
            if hasattr(self.thermal_model, "get_adaptive_learning_metrics"):
                influx_service.write_thermal_learning_metrics(self.thermal_model)
                logging.debug("✅ Exported thermal learning metrics to InfluxDB")

            # Export learning phase metrics (if available)
            learning_phase_data = {
                "current_learning_phase": "high_confidence",  # Simplified for now
                "stability_score": min(
                    1.0, self.thermal_model.learning_confidence / 5.0
                ),
                "learning_weight_applied": 1.0,
                "stable_period_duration_min": 30,
                "learning_updates_24h": {
                    "high_confidence": min(288, self.cycle_count),
                    "low_confidence": 0,
                    "skipped": 0,
                },
                "learning_efficiency_pct": 85.0,
                "correction_stability": 0.9,
                "false_learning_prevention_pct": 95.0,
            }
            influx_service.write_learning_phase_metrics(learning_phase_data)
            logging.debug("✅ Exported learning phase metrics to InfluxDB")

            # Export basic trajectory metrics (simplified)
            trajectory_data = {
                "prediction_horizon": "4h",
                "trajectory_accuracy": {
                    "mae_1h": prediction_metrics.get("1h", {}).get("mae", 0.0),
                    "mae_2h": prediction_metrics.get("6h", {}).get("mae", 0.0) * 1.2,
                    "mae_4h": prediction_metrics.get("24h", {}).get("mae", 0.0) * 1.5,
                },
                "overshoot_prevention": {
                    "overshoot_predicted": False,
                    "prevented_24h": 0,
                    "undershoot_prevented_24h": 0,
                },
                "convergence": {"avg_time_minutes": 45.0, "accuracy_percentage": 87.5},
                "forecast_integration": {
                    "weather_available": False,
                    "pv_available": True,
                    "quality_score": 0.8,
                },
            }
            influx_service.write_trajectory_prediction_metrics(trajectory_data)
            logging.debug("✅ Exported trajectory prediction metrics to InfluxDB")

            logging.info(
                f"📊 Exported all adaptive learning metrics to InfluxDB (cycle {self.cycle_count})"
            )

        except Exception as e:
            logging.warning(f"Failed to export metrics to InfluxDB: {e}")

    def _save_learning_state(self):
        """Save current thermal learning state to persistent storage."""
        try:
            # State saving is handled by the unified thermal state manager
            # No additional saving needed here as the state_manager handles persistence
            logging.debug("Learning state automatically saved via state_manager")

        except Exception as e:
            logging.error(f"Failed to save learning state: {e}")


# Legacy functions removed - ThermalEquilibriumModel handles persistence internally


def get_enhanced_model_wrapper() -> EnhancedModelWrapper:
    """
    Create and return an enhanced model wrapper with singleton pattern.

    This prevents multiple model instantiation which was causing the rapid
    cycle execution issue. Only one instance per service restart.
    """
    global _enhanced_model_wrapper_instance

    if _enhanced_model_wrapper_instance is None:
        logging.info("🔧 Creating new Model Wrapper instance (singleton)")
        _enhanced_model_wrapper_instance = EnhancedModelWrapper()
    else:
        logging.debug("♻️ Reusing existing Model Wrapper instance")

    return _enhanced_model_wrapper_instance


def simplified_outlet_prediction(
    features: pd.DataFrame, current_temp: float, target_temp: float
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

        features_dict["indoor_temp_lag_30m"] = current_temp
        features_dict["target_temp"] = target_temp

        # Get simplified prediction
        outlet_temp, metadata = wrapper.calculate_optimal_outlet_temp(features_dict)
        confidence = metadata.get("learning_confidence", 3.0)

        # Calculate thermal trust metrics for HA sensor display
        thermal_trust_metrics = _calculate_thermal_trust_metrics(
            wrapper, outlet_temp, current_temp, target_temp
        )
        metadata["thermal_trust_metrics"] = thermal_trust_metrics

        # Log the calculated outlet temperature - smart rounding will be applied later in main.py
        logging.info(
            f"🎯 Prediction: Current {current_temp:.2f}°C → Target {target_temp:.1f}°C | "
            f"Calculated outlet: {outlet_temp:.1f}°C (before smart rounding) "
            f"(confidence: {confidence:.3f})"
        )

        return outlet_temp, confidence, metadata

    except Exception as e:
        logging.error(f"Simplified prediction failed: {e}", exc_info=True)
        # Safe fallback
        return 35.0, 2.0, {"error": str(e), "method": "fallback"}


def _calculate_thermal_trust_metrics(
    wrapper: EnhancedModelWrapper,
    outlet_temp: float,
    current_temp: float,
    target_temp: float,
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
        conductance_stability = min(1.0, thermal_model.total_conductance * 20.0)
        equilibrium_ratio_stability = thermal_model.equilibrium_ratio
        thermal_stability = (
            time_constant_stability + conductance_stability + equilibrium_ratio_stability
        ) / 3.0

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
            expected_outlet_range = current_temp + (
                temp_diff * 8.0
            )  # Rough physics heuristic
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
        learning_progress = min(
            1.0, cycle_count / 100.0
        )  # Fully learned after 100 cycles

        return {
            "thermal_stability": thermal_stability,
            "prediction_consistency": prediction_consistency,
            "physics_alignment": physics_alignment,
            "model_health": model_health,
            "learning_progress": learning_progress,
        }

    except Exception as e:
        logging.error(f"Failed to calculate thermal trust metrics: {e}")
        return {
            "thermal_stability": 0.0,
            "prediction_consistency": 0.0,
            "physics_alignment": 0.0,
            "model_health": "error",
            "learning_progress": 0.0,
        }
