"""
Thermal Equilibrium Model with Adaptive Learning.

This module defines the core physics-based model for predicting thermal
equilibrium and adapting its parameters in real-time based on prediction
accuracy. It combines a heat balance equation with a gradient-based
learning mechanism to continuously improve its accuracy.
"""

import numpy as np
import logging
from typing import Dict, List, Optional

# MIGRATION: Use unified thermal parameter system
try:
    from .thermal_parameters import thermal_params  # type: ignore
    from .thermal_constants import PhysicsConstants  # type: ignore
    from . import config  # type: ignore
except ImportError:
    # Direct import fallback for notebooks and standalone tests
    from thermal_parameters import thermal_params  # type: ignore
    from thermal_constants import PhysicsConstants  # type: ignore
    import config  # type: ignore

class ThermalEquilibriumModel:
    """
    A physics-based thermal model that predicts indoor temperature equilibrium
    and adapts its parameters based on real-world feedback.
    """

    def __init__(self):
        # Load calibrated parameters first, fallback to config defaults
        self._load_thermal_parameters()

        # self.outdoor_coupling = config.OUTDOOR_COUPLING
        # thermal_bridge_factor removed in Phase 2: was not used in
        # calculations

    def _load_thermal_parameters(self):
        """
        Load thermal parameters with proper baseline + adjustments.
        This ensures trained parameters persist across restarts.
        """
        try:
            # Try to load calibrated parameters from unified thermal state
            from .unified_thermal_state import get_thermal_state_manager

            state_manager = get_thermal_state_manager()
            thermal_state = state_manager.get_current_parameters()

            # Check for calibrated parameters in baseline_parameters section
            baseline_params = thermal_state.get("baseline_parameters", {})
            if baseline_params.get("source") == "calibrated":
                # Load baseline + adjustments for trained parameters
                learning_state = thermal_state.get("learning_state", {})
                adjustments = learning_state.get("parameter_adjustments", {})

                # Apply learning adjustments to baseline
                self.thermal_time_constant = (
                    baseline_params["thermal_time_constant"]
                    + adjustments.get("thermal_time_constant_delta", 0.0)
                )
                self.heat_loss_coefficient = (
                    baseline_params["heat_loss_coefficient"]
                    + adjustments.get("heat_loss_coefficient_delta", 0.0)
                )
                self.outlet_effectiveness = (
                    baseline_params["outlet_effectiveness"]
                    + adjustments.get("outlet_effectiveness_delta", 0.0)
                )

                self.external_source_weights = {
                    "pv": baseline_params.get(
                        "pv_heat_weight", config.PV_HEAT_WEIGHT
                    ),
                    "fireplace": baseline_params.get(
                        "fireplace_heat_weight", config.FIREPLACE_HEAT_WEIGHT,
                    ),
                    "tv": baseline_params.get(
                        "tv_heat_weight", config.TV_HEAT_WEIGHT
                    ),
                }

                logging.info(
                    "🎯 Loading CALIBRATED thermal parameters "
                    "(baseline + learning adjustments):"
                )
                logging.info(
                    "   heat_loss_coefficient: %.4f + %.5f = %.4f",
                    baseline_params["heat_loss_coefficient"],
                    adjustments.get("heat_loss_coefficient_delta", 0.0),
                    self.heat_loss_coefficient,
                )
                logging.info(
                    "   outlet_effectiveness: %.3f + %.3f = %.3f",
                    baseline_params["outlet_effectiveness"],
                    adjustments.get("outlet_effectiveness_delta", 0.0),
                    self.outlet_effectiveness,
                )
                logging.info(
                    "   pv_heat_weight: %.4f", self.external_source_weights["pv"]
                )

                # Validate parameters using schema validator
                try:
                    from thermal_state_validator import (
                        validate_thermal_state_safely,
                    )

                    if not validate_thermal_state_safely(thermal_state):
                        logging.warning(
                            "⚠️ Thermal state validation failed, "
                            "using config defaults"
                        )
                        self._load_config_defaults()
                        return
                except ImportError:
                    logging.debug("Schema validation not available")

                # Initialize learning attributes
                self._initialize_learning_attributes()

                # Restore learning history from saved state
                self.learning_confidence = max(
                    learning_state.get("learning_confidence", 3.0), 0.1
                )
                self.prediction_history = learning_state.get(
                    "prediction_history", []
                )
                self.parameter_history = learning_state.get(
                    "parameter_history", []
                )

                logging.info(
                    "   - Restored learning confidence: %.3f",
                    self.learning_confidence,
                )
                logging.info(
                    "   - Restored prediction history: %d records",
                    len(self.prediction_history),
                )
                logging.info(
                    "   - Restored parameter history: %d records",
                    len(self.parameter_history),
                )

                # STABILITY FIX: Detect and reset corrupted parameters on load
                if self._detect_parameter_corruption():
                    logging.warning(
                        "🗑️ Detected corrupted parameters on load. "
                        "Resetting to defaults."
                    )
                    self._load_config_defaults()
                    # CRITICAL: Return to avoid using corrupted state
                    return

            else:
                # Use config defaults
                self._load_config_defaults()
                logging.info(
                    "⚙️ Loading DEFAULT config parameters "
                    "(no calibration found)"
                )

        except Exception as e:
            # Fallback to config defaults if thermal state unavailable
            logging.warning(f"⚠️ Failed to load calibrated parameters: {e}")
            self._load_config_defaults()
            logging.info("⚙️ Using config defaults as fallback")

    def _load_config_defaults(self):
        """MIGRATED: Load thermal parameters from unified parameter system."""
        self.thermal_time_constant = thermal_params.get("thermal_time_constant")
        self.heat_loss_coefficient = thermal_params.get("heat_loss_coefficient")
        self.outlet_effectiveness = thermal_params.get("outlet_effectiveness")

        self.external_source_weights = {
            "pv": thermal_params.get("pv_heat_weight"),
            "fireplace": thermal_params.get("fireplace_heat_weight"),
            "tv": thermal_params.get("tv_heat_weight"),
        }

        # Initialize remaining attributes
        self._initialize_learning_attributes()

    def _initialize_learning_attributes(self):
        """Initialize adaptive learning and other attributes."""
        self.adaptive_learning_enabled = True
        self.safety_margin = PhysicsConstants.DEFAULT_SAFETY_MARGIN
        self.prediction_horizon_hours = (
            PhysicsConstants.DEFAULT_PREDICTION_HORIZON
        )
        self.momentum_decay_rate = PhysicsConstants.MOMENTUM_DECAY_RATE

        self.learning_rate = (
            thermal_params.get("adaptive_learning_rate")
            or PhysicsConstants.DEFAULT_LEARNING_RATE
        )
        self.equilibrium_samples = []
        self.trajectory_samples = []
        self.overshoot_events = []

        self.prediction_errors = []
        self.mode_switch_history = []
        self.overshoot_prevention_count = 0

        self.prediction_history: List[Dict] = []
        self.parameter_history: List[Dict] = []
        self.learning_confidence = PhysicsConstants.INITIAL_LEARNING_CONFIDENCE
        self.min_learning_rate = (
            thermal_params.get("min_learning_rate")
            or PhysicsConstants.MIN_LEARNING_RATE
        )
        self.max_learning_rate = (
            thermal_params.get("max_learning_rate")
            or PhysicsConstants.MAX_LEARNING_RATE
        )
        self.confidence_decay_rate = PhysicsConstants.CONFIDENCE_DECAY_RATE
        self.confidence_boost_rate = PhysicsConstants.CONFIDENCE_BOOST_RATE
        self.recent_errors_window = config.RECENT_ERRORS_WINDOW

        try:
            from .thermal_config import ThermalParameterConfig
        except ImportError:
            from thermal_config import ThermalParameterConfig

        self.thermal_time_constant_bounds = ThermalParameterConfig.get_bounds(
            "thermal_time_constant"
        )
        self.heat_loss_coefficient_bounds = ThermalParameterConfig.get_bounds(
            "heat_loss_coefficient"
        )
        self.outlet_effectiveness_bounds = ThermalParameterConfig.get_bounds(
            "outlet_effectiveness"
        )

        self.parameter_stability_threshold = (
            PhysicsConstants.THERMAL_STABILITY_THRESHOLD
        )
        self.error_improvement_threshold = (
            PhysicsConstants.ERROR_IMPROVEMENT_THRESHOLD
        )

    def predict_equilibrium_temperature(
        self,
        outlet_temp: float,
        outdoor_temp: float,
        current_indoor: float,
        pv_power: float = 0,
        fireplace_on: float = 0,
        tv_on: float = 0,
        _suppress_logging: bool = False,
    ) -> float:
        """
        Predict equilibrium temperature using standard heat balance physics.
        """
        heat_from_pv = pv_power * self.external_source_weights.get("pv", 0.0)
        heat_from_fireplace = fireplace_on * self.external_source_weights.get(
            "fireplace", 0.0
        )
        heat_from_tv = tv_on * self.external_source_weights.get("tv", 0.0)

        external_thermal_power = (
            heat_from_pv + heat_from_fireplace + heat_from_tv
        )

        total_conductance = self.heat_loss_coefficient + self.outlet_effectiveness
        if total_conductance > 0:
            equilibrium_temp = (
                self.outlet_effectiveness * outlet_temp
                + self.heat_loss_coefficient * outdoor_temp
                + external_thermal_power
            ) / total_conductance
        else:
            equilibrium_temp = outdoor_temp

        if outlet_temp > outdoor_temp:
            equilibrium_temp = max(outdoor_temp, equilibrium_temp)
        elif outlet_temp < outdoor_temp:
            equilibrium_temp = min(outdoor_temp, equilibrium_temp)
        else:
            if total_conductance > 0:
                equilibrium_temp = (
                    outdoor_temp + external_thermal_power / total_conductance
                )
            else:
                equilibrium_temp = outdoor_temp

        if not _suppress_logging:
            logging.debug(
                "🔬 Equilibrium physics: outlet=%.1f°C, outdoor=%.1f°C, "
                "heat_loss_coeff=%.3f, outlet_eff=%.3f, "
                "equilibrium=%.2f°C",
                outlet_temp,
                outdoor_temp,
                self.heat_loss_coefficient,
                self.outlet_effectiveness,
                equilibrium_temp,
            )

        return equilibrium_temp

    def update_prediction_feedback(
        self,
        predicted_temp: float,
        actual_temp: float,
        prediction_context: Dict,
        timestamp: Optional[str] = None,
        is_blocking_active: bool = False,
    ):
        """
        Update the model with real-world feedback to enable adaptive learning.
        """
        if not self.adaptive_learning_enabled:
            return

        if is_blocking_active:
            logging.debug(
                "⏸️ Skipping learning during blocking event (DHW/Defrost)"
            )
            return

        if self._detect_parameter_corruption():
            logging.warning("🛑 Parameter corruption detected - learning DISABLED")
            return

        outlet_temp = prediction_context.get("outlet_temp")
        if outlet_temp is None:
            logging.warning(
                "No outlet_temp in prediction context, skipping learning"
            )
            return

        current_indoor = prediction_context.get("current_indoor")
        if current_indoor is None:
            logging.warning(
                "No current_indoor in prediction context, skipping learning"
            )
            return

        prediction_error = actual_temp - predicted_temp

        predicted_trajectory = self.predict_thermal_trajectory(
            current_indoor=current_indoor,
            target_indoor=current_indoor,
            outlet_temp=outlet_temp,
            outdoor_temp=prediction_context.get("outdoor_temp", 10.0),
            time_horizon_hours=config.CYCLE_INTERVAL_MINUTES / 60.0,
            time_step_minutes=config.CYCLE_INTERVAL_MINUTES,
            pv_power=prediction_context.get("pv_power", 0),
            fireplace_on=prediction_context.get("fireplace_on", 0),
            tv_on=prediction_context.get("tv_on", 0),
        )
        predicted_temp_at_cycle_end = predicted_trajectory["trajectory"][0]

        system_state = (
            "shadow_mode_physics" if config.SHADOW_MODE else "active_mode"
        )

        prediction_record = {
            "timestamp": timestamp,
            "predicted": predicted_temp,
            "actual": actual_temp,
            "error": prediction_error,
            "context": prediction_context.copy(),
            "model_internal_prediction": predicted_temp_at_cycle_end,
            "parameters_at_prediction": {
                "thermal_time_constant": self.thermal_time_constant,
                "heat_loss_coefficient": self.heat_loss_coefficient,
                "outlet_effectiveness": self.outlet_effectiveness,
            },
            "shadow_mode": config.SHADOW_MODE,
            "system_state": system_state,
            "learning_quality": self._assess_learning_quality(
                prediction_context, prediction_error
            ),
        }

        self.prediction_history.append(prediction_record)

        if len(self.prediction_history) > 200:
            self.prediction_history = self.prediction_history[-100:]
            
        if len(self.prediction_history) >= self.recent_errors_window:
            error_magnitude = abs(prediction_error)
            if error_magnitude < 0.25:  # Good prediction
                self.learning_confidence *= self.confidence_boost_rate
            elif error_magnitude > 1.0:  # Bad prediction
                self.learning_confidence *= self.confidence_decay_rate

            self.learning_confidence = np.clip(self.learning_confidence, 0.1, 5.0)
            self._adapt_parameters_from_recent_errors()

        logging.debug(
            "Prediction feedback: error=%.3f°C, confidence=%.3f",
            prediction_error,
            self.learning_confidence,
        )
        return prediction_error

    def _assess_learning_quality(
        self, prediction_context: Dict, prediction_error: float
    ) -> str:
        """Assess the quality of this learning opportunity."""
        try:
            temp_gradient = abs(
                prediction_context.get("indoor_temp_gradient", 0.0)
            )
            is_stable = temp_gradient < 0.1
            error_magnitude = abs(prediction_error)

            if error_magnitude < 0.1 and is_stable:
                return "excellent"
            elif error_magnitude < 0.5 and is_stable:
                return "good"
            elif error_magnitude < 0.1:
                return "fair"
            elif is_stable:
                return "fair"
            else:
                return "poor"
        except Exception:
            return "unknown"

    def _adapt_parameters_from_recent_errors(self):
        """Adapt model parameters with corrected gradient calculations."""
        recent_predictions = self.prediction_history[
            -self.recent_errors_window:
        ]

        if len(recent_predictions) < self.recent_errors_window:
            return

        if self._detect_parameter_corruption():
            logging.warning(
                "🛑 Parameter corruption detected in adaptation - "
                "learning DISABLED"
            )
            return

        recent_errors = [abs(p["error"]) for p in recent_predictions]
        has_catastrophic_error = any(error >= 5.0 for error in recent_errors)

        if has_catastrophic_error:
            max_error = max(recent_errors)
            logging.warning(
                "🛑 Blocking parameter updates due to catastrophic error "
                "(%.1f°C)",
                max_error,
            )
            return

        thermal_gradient = self._calculate_thermal_time_constant_gradient(
            recent_predictions
        )
        heat_loss_coefficient_gradient = (
            self._calculate_heat_loss_coefficient_gradient(recent_predictions)
        )
        outlet_effectiveness_gradient = (
            self._calculate_outlet_effectiveness_gradient(recent_predictions)
        )

        current_learning_rate = self._calculate_adaptive_learning_rate()

        old_thermal_time_constant = self.thermal_time_constant
        old_heat_loss_coefficient = self.heat_loss_coefficient
        old_outlet_effectiveness = self.outlet_effectiveness

        thermal_update = current_learning_rate * thermal_gradient
        heat_loss_coefficient_update = (
            current_learning_rate * heat_loss_coefficient_gradient
        )
        outlet_effectiveness_update = (
            current_learning_rate * outlet_effectiveness_gradient
        )

        max_heat_loss_coefficient_change = (
            PhysicsConstants.MAX_HEAT_LOSS_COEFFICIENT_CHANGE
        )
        heat_loss_coefficient_update = np.clip(
            heat_loss_coefficient_update,
            -max_heat_loss_coefficient_change,
            max_heat_loss_coefficient_change,
        )
        max_outlet_effectiveness_change = (
            PhysicsConstants.MAX_OUTLET_EFFECTIVENESS_CHANGE
        )
        outlet_effectiveness_update = np.clip(
            outlet_effectiveness_update,
            -max_outlet_effectiveness_change,
            max_outlet_effectiveness_change,
        )
        max_thermal_time_constant_change = (
            PhysicsConstants.MAX_THERMAL_TIME_CONSTANT_CHANGE
        )
        thermal_update = np.clip(
            thermal_update,
            -max_thermal_time_constant_change,
            max_thermal_time_constant_change,
        )

        self.thermal_time_constant = float(
            np.clip(
                self.thermal_time_constant - thermal_update,
                self.thermal_time_constant_bounds[0],
                self.thermal_time_constant_bounds[1],
            )
        )

        self.heat_loss_coefficient = float(
            np.clip(
                self.heat_loss_coefficient - heat_loss_coefficient_update,
                self.heat_loss_coefficient_bounds[0],
                self.heat_loss_coefficient_bounds[1],
            )
        )

        self.outlet_effectiveness = float(
            np.clip(
                self.outlet_effectiveness - outlet_effectiveness_update,
                self.outlet_effectiveness_bounds[0],
                self.outlet_effectiveness_bounds[1],
            )
        )

        thermal_change = abs(
            self.thermal_time_constant - old_thermal_time_constant
        )
        heat_loss_coefficient_change = abs(
            self.heat_loss_coefficient - old_heat_loss_coefficient
        )
        outlet_effectiveness_change = abs(
            self.outlet_effectiveness - old_outlet_effectiveness
        )

        self.parameter_history.append(
            {
                "timestamp": recent_predictions[-1]["timestamp"],
                "thermal_time_constant": self.thermal_time_constant,
                "heat_loss_coefficient": self.heat_loss_coefficient,
                "outlet_effectiveness": self.outlet_effectiveness,
                "learning_rate": current_learning_rate,
                "learning_confidence": self.learning_confidence,
                "avg_recent_error": np.mean(
                    [abs(p["error"]) for p in recent_predictions]
                ),
                "gradients": {
                    "thermal": thermal_gradient,
                    "heat_loss_coefficient": heat_loss_coefficient_gradient,
                    "outlet_effectiveness": outlet_effectiveness_gradient,
                },
                "changes": {
                    "thermal": thermal_change,
                    "heat_loss_coefficient": heat_loss_coefficient_change,
                    "outlet_effectiveness": outlet_effectiveness_change,
                },
            }
        )

        if len(self.parameter_history) > 500:
            self.parameter_history = self.parameter_history[-250:]

        if (
            thermal_change > 0.001
            or heat_loss_coefficient_change > 0.0001
            or outlet_effectiveness_change > 0.0001
        ):
            logging.info(
                "Adaptive learning update: "
                "thermal: %.2f→%.2f (Δ%+.3f), "
                "heat_loss_coeff: %.4f→%.4f (Δ%+.5f), "
                "outlet_eff: %.3f→%.3f (Δ%+.3f)",
                old_thermal_time_constant,
                self.thermal_time_constant,
                thermal_change,
                old_heat_loss_coefficient,
                self.heat_loss_coefficient,
                heat_loss_coefficient_change,
                old_outlet_effectiveness,
                self.outlet_effectiveness,
                outlet_effectiveness_change,
            )

            self._save_learning_to_thermal_state(
                self.thermal_time_constant - old_thermal_time_constant,
                self.heat_loss_coefficient - old_heat_loss_coefficient,
                self.outlet_effectiveness - old_outlet_effectiveness,
            )
        else:
            logging.debug(
                "Micro learning update: thermal_Δ=%+.5f, "
                "heat_loss_coeff_Δ=%+.7f, outlet_eff_Δ=%+.5f",
                thermal_change,
                heat_loss_coefficient_change,
                outlet_effectiveness_change,
            )

    def _calculate_parameter_gradient(
        self,
        parameter_name: str,
        epsilon: float,
        recent_predictions: List[Dict],
    ) -> float:
        """
        Generic finite-difference gradient calculation for any parameter.
        """
        gradient_sum = 0.0
        count = 0

        original_value = getattr(self, parameter_name)

        for pred in recent_predictions:
            context = pred["context"]

            if not all(
                key in context
                for key in ["outlet_temp", "outdoor_temp", "current_indoor"]
            ):
                continue

            setattr(self, parameter_name, original_value + epsilon)
            pred_plus_trajectory = self.predict_thermal_trajectory(
                current_indoor=context["current_indoor"],
                target_indoor=context["current_indoor"],
                outlet_temp=context["outlet_temp"],
                outdoor_temp=context["outdoor_temp"],
                time_horizon_hours=config.CYCLE_INTERVAL_MINUTES / 60.0,
                time_step_minutes=config.CYCLE_INTERVAL_MINUTES,
                pv_power=context.get("pv_power", 0),
                fireplace_on=context.get("fireplace_on", 0),
                tv_on=context.get("tv_on", 0),
            )
            pred_plus = pred_plus_trajectory["trajectory"][0]

            setattr(self, parameter_name, original_value - epsilon)
            pred_minus_trajectory = self.predict_thermal_trajectory(
                current_indoor=context["current_indoor"],
                target_indoor=context["current_indoor"],
                outlet_temp=context["outlet_temp"],
                outdoor_temp=context["outdoor_temp"],
                time_horizon_hours=config.CYCLE_INTERVAL_MINUTES / 60.0,
                time_step_minutes=config.CYCLE_INTERVAL_MINUTES,
                pv_power=context.get("pv_power", 0),
                fireplace_on=context.get("fireplace_on", 0),
                tv_on=context.get("tv_on", 0),
            )
            pred_minus = pred_minus_trajectory["trajectory"][0]

            setattr(self, parameter_name, original_value)

            finite_diff = (pred_plus - pred_minus) / (2 * epsilon)
            error = np.clip(pred["error"], -2.0, 2.0)
            gradient = -finite_diff * error
            gradient_sum += gradient
            count += 1

        return gradient_sum / count if count > 0 else 0.0

    def _calculate_thermal_time_constant_gradient(
        self, recent_predictions: List[Dict]
    ) -> float:
        """
        Calculate thermal time constant gradient.
        """
        return self._calculate_parameter_gradient(
            "thermal_time_constant",
            PhysicsConstants.THERMAL_TIME_CONSTANT_EPSILON,
            recent_predictions,
        )

    def _calculate_heat_loss_coefficient_gradient(
        self, recent_predictions: List[Dict]
    ) -> float:
        """
        Calculate heat loss coefficient gradient.
        """
        return self._calculate_parameter_gradient(
            "heat_loss_coefficient",
            PhysicsConstants.HEAT_LOSS_COEFFICIENT_EPSILON,
            recent_predictions,
        )

    def _calculate_outlet_effectiveness_gradient(
        self, recent_predictions: List[Dict]
    ) -> float:
        """
        Calculate outlet effectiveness gradient.
        """
        return self._calculate_parameter_gradient(
            "outlet_effectiveness",
            PhysicsConstants.OUTLET_EFFECTIVENESS_EPSILON,
            recent_predictions,
        )

    def _calculate_adaptive_learning_rate(self) -> float:
        """
        Calculate adaptive learning rate based on model performance.
        """
        base_rate = (
            max(self.learning_rate, self.min_learning_rate)
            * self.learning_confidence
        )

        heat_loss_coefficient_std = 0.0
        thermal_time_constant_std = 0.0
        outlet_effectiveness_std = 0.0
        if len(self.parameter_history) >= 3:
            recent_params = self.parameter_history[-3:]
            heat_loss_coefficient_std = np.std(
                [p["heat_loss_coefficient"] for p in recent_params]
            )
            thermal_time_constant_std = np.std(
                [p["thermal_time_constant"] for p in recent_params]
            )
            outlet_effectiveness_std = np.std(
                [p["outlet_effectiveness"] for p in recent_params]
            )

        if (
            heat_loss_coefficient_std > 0.02
            or outlet_effectiveness_std > 0.05
            or thermal_time_constant_std > 0.5
        ):
            base_rate *= 0.01
            logging.debug(
                (
                    "⚠️ Parameter oscillation detected (heat_loss_coeff=%.3f, "
                    "outlet_eff=%.3f, thermal=%.3f), "
                    "reducing learning rate by 99%%"
                ),
                heat_loss_coefficient_std,
                outlet_effectiveness_std,
                thermal_time_constant_std,
            )
        elif (
            heat_loss_coefficient_std > 0.01
            or outlet_effectiveness_std > 0.02
            or thermal_time_constant_std > 0.2
        ):
            base_rate *= 0.1

        if self.prediction_history:
            recent_errors = [
                abs(p["error"]) for p in self.prediction_history[-5:]
            ]
            has_catastrophic_error = any(
                error >= 5.0 for error in recent_errors
            )

            if has_catastrophic_error:
                base_rate = 0.0
                max_error = max(recent_errors)
                logging.warning(
                    "🛑 Catastrophic error (%.1f°C) - learning DISABLED",
                    max_error,
                )

            elif len(self.prediction_history) >= 5:
                avg_error = np.mean(recent_errors)
                if config.SHADOW_MODE:
                    # In shadow mode, we want to learn faster, even from large
                    # errors, now that the physics are corrected. The confidence
                    # mechanism will naturally temper the learning rate.
                    if avg_error > 2.0:
                        base_rate /= 1.5  # Less aggressive reduction
                else:
                    if avg_error > 3.0:
                        base_rate /= 10.0
                    elif avg_error > 2.0:
                        base_rate /= 5.0
                    elif avg_error > 1.0:
                        base_rate /= 2.5
                    elif avg_error > 0.5:
                        base_rate /= 1.5

        return np.clip(base_rate, 0.0, self.max_learning_rate)

    def predict_thermal_trajectory(
        self,
        current_indoor,
        target_indoor,
        outlet_temp,
        outdoor_temp,
        time_horizon_hours=None,
        time_step_minutes=60,
        weather_forecasts=None,
        pv_forecasts=None,
        **external_sources,
    ):
        """
        Predict temperature trajectory over time horizon.
        """
        if time_horizon_hours is None:
            time_horizon_hours = int(self.prediction_horizon_hours)

        trajectory = []
        current_temp = current_indoor

        pv_power = external_sources.get("pv_power", 0)
        fireplace_on = external_sources.get("fireplace_on", 0)
        tv_on = external_sources.get("tv_on", 0)

        time_step_hours = time_step_minutes / 60.0
        num_steps = int(time_horizon_hours * 60 / time_step_minutes)

        if weather_forecasts and time_step_minutes == 60:
            outdoor_forecasts = weather_forecasts
        else:
            outdoor_forecasts = [outdoor_temp] * num_steps
        if pv_forecasts and time_step_minutes == 60:
            pv_power_forecasts = pv_forecasts
        else:
            pv_power_forecasts = [pv_power] * num_steps

        for step in range(num_steps):
            future_outdoor = (
                outdoor_forecasts[step]
                if step < len(outdoor_forecasts)
                else outdoor_temp
            )
            future_pv = (
                pv_power_forecasts[step]
                if step < len(pv_power_forecasts)
                else pv_power
            )

            equilibrium_temp = self.predict_equilibrium_temperature(
                outlet_temp=outlet_temp,
                outdoor_temp=future_outdoor,
                current_indoor=current_temp,
                pv_power=future_pv,
                fireplace_on=fireplace_on,
                tv_on=tv_on,
                _suppress_logging=True,
            )

            time_constant_hours = self.thermal_time_constant
            approach_factor = 1 - np.exp(-time_step_hours / time_constant_hours)
            temp_change = (equilibrium_temp - current_temp) * approach_factor

            if step > 0:
                momentum_factor = np.exp(
                    -step * time_step_hours * self.momentum_decay_rate
                )
                temp_change *= 1.0 - momentum_factor * 0.2

            current_temp = current_temp + temp_change
            trajectory.append(current_temp)

        reaches_target_at = None
        sensor_precision_tolerance = 0.1

        for i, temp in enumerate(trajectory):
            if abs(temp - target_indoor) < sensor_precision_tolerance:
                reaches_target_at = (i + 1) * time_step_hours
                break

        if (
            trajectory
            and abs(trajectory[0] - target_indoor) < sensor_precision_tolerance
        ):
            cycle_hours = config.CYCLE_INTERVAL_MINUTES / 60.0
            if reaches_target_at is not None:
                reaches_target_at = min(reaches_target_at, cycle_hours)
            else:
                reaches_target_at = cycle_hours

        overshoot_predicted = False
        max_predicted = max(trajectory) if trajectory else current_indoor

        if target_indoor > current_indoor:
            overshoot_predicted = max_predicted > (
                target_indoor + self.safety_margin
            )
        else:
            min_predicted = min(trajectory) if trajectory else current_indoor
            overshoot_predicted = min_predicted < (
                target_indoor - self.safety_margin
            )

        return {
            "trajectory": trajectory,
            "times": [
                (step + 1) * time_step_hours for step in range(num_steps)
            ],
            "reaches_target_at": reaches_target_at,
            "overshoot_predicted": overshoot_predicted,
            "max_predicted": max(trajectory) if trajectory else current_indoor,
            "min_predicted": min(trajectory) if trajectory else current_indoor,
            "equilibrium_temp": trajectory[-1] if trajectory else current_indoor,
            "final_error": (
                abs(trajectory[-1] - target_indoor)
                if trajectory
                else abs(current_indoor - target_indoor)
            ),
        }

    def calculate_optimal_outlet_temperature(
        self,
        target_indoor,
        current_indoor,
        outdoor_temp,
        time_available_hours=1.0,
        config_override=None,
        **external_sources,
    ):
        """
        Calculate optimal outlet temperature to reach target indoor temperature.
        """
        pv_power = external_sources.get(
            "pv_power", external_sources.get("pv_now", 0)
        )
        fireplace_on = external_sources.get("fireplace_on", 0)
        tv_on = external_sources.get("tv_on", 0)

        temp_change_required = target_indoor - current_indoor

        if abs(temp_change_required) < 0.1:
            outlet_temp = self._calculate_equilibrium_outlet_temperature(
                target_indoor, outdoor_temp, pv_power, fireplace_on, tv_on
            )
            return {
                "optimal_outlet_temp": outlet_temp,
                "method": "equilibrium_maintenance",
                "temp_change_required": temp_change_required,
                "time_available": time_available_hours,
            }

        method = "direct_calculation"
        heat_loss_coefficient = self.heat_loss_coefficient
        outlet_effectiveness = self.outlet_effectiveness
        if config_override:
            heat_loss_coefficient = config_override.get(
                "heat_loss_coefficient", heat_loss_coefficient
            )
            outlet_effectiveness = config_override.get(
                "outlet_effectiveness", outlet_effectiveness
            )
        total_conductance = heat_loss_coefficient + outlet_effectiveness

        external_heating = (
            pv_power * self.external_source_weights.get("pv", 0.0)
            + fireplace_on * self.external_source_weights.get("fireplace", 0.0)
            + tv_on * self.external_source_weights.get("tv", 0.0)
        )

        if outlet_effectiveness <= 0:
            return None

        optimal_outlet = (
            target_indoor * total_conductance
            - heat_loss_coefficient * outdoor_temp
            - external_heating
        ) / outlet_effectiveness
        required_equilibrium = target_indoor

        min_outlet = max(outdoor_temp + 5, 25.0)
        max_outlet = 70.0

        optimal_outlet_bounded = max(
            min_outlet, min(optimal_outlet, max_outlet)
        )

        if optimal_outlet_bounded < outdoor_temp:
            fallback_outlet = self._calculate_equilibrium_outlet_temperature(
                target_indoor, outdoor_temp, pv_power, fireplace_on, tv_on
            )
            return {
                "optimal_outlet_temp": fallback_outlet,
                "method": "fallback_equilibrium",
                "reason": "unrealistic_outlet_temp",
                "original_calculation": optimal_outlet,
                "temp_change_required": temp_change_required,
                "time_available": time_available_hours,
            }

        return {
            "optimal_outlet_temp": optimal_outlet_bounded,
            "method": method,
            "required_equilibrium": required_equilibrium,
            "temp_change_required": temp_change_required,
            "time_available": time_available_hours,
            "external_heating": external_heating,
            "bounded": optimal_outlet != optimal_outlet_bounded,
            "original_calculation": optimal_outlet,
        }

    def _calculate_equilibrium_outlet_temperature(
        self, target_temp, outdoor_temp, pv_power=0, fireplace_on=0, tv_on=0
    ):
        """
        Calculate outlet temperature needed for equilibrium at target temp.
        """
        external_heating = (
            pv_power * self.external_source_weights["pv"]
            + fireplace_on * self.external_source_weights["fireplace"]
            + tv_on * self.external_source_weights["tv"]
        )

        if self.outlet_effectiveness <= 0:
            return 35.0

        total_conductance = self.heat_loss_coefficient + self.outlet_effectiveness
        equilibrium_outlet = (
            target_temp * total_conductance
            - self.heat_loss_coefficient * outdoor_temp
            - external_heating
        ) / self.outlet_effectiveness

        min_outlet = max(outdoor_temp + 5, 25.0)
        max_outlet = 65.0

        return max(min_outlet, min(equilibrium_outlet, max_outlet))

    def calculate_physics_aware_thresholds(self, *args, **kwargs):
        """Keep original threshold calculation method unchanged"""
        pass

    def get_adaptive_learning_metrics(self) -> Dict:
        """
        ENHANCED: Get metrics with additional debugging info.
        """
        if len(self.prediction_history) < 5:
            return {"insufficient_data": True}

        recent_errors = [
            abs(p["error"]) for p in self.prediction_history[-20:]
        ]
        all_errors = [abs(p["error"]) for p in self.prediction_history]

        if len(recent_errors) >= 10:
            first_half_errors = recent_errors[: len(recent_errors) // 2]
            second_half_errors = recent_errors[len(recent_errors) // 2:]
            error_improvement = np.mean(first_half_errors) - np.mean(
                second_half_errors
            )
        else:
            error_improvement = 0.0

        if len(self.parameter_history) >= 5:
            recent_params = self.parameter_history[-5:]
            thermal_stability = np.std(
                [p["thermal_time_constant"] for p in recent_params]
            )
            heat_loss_coefficient_stability = np.std(
                [p["heat_loss_coefficient"] for p in recent_params]
            )
            outlet_effectiveness_stability = np.std(
                [p["outlet_effectiveness"] for p in recent_params]
            )
            recent_gradients = recent_params[-1].get("gradients", {})
        else:
            thermal_stability = 0.0
            heat_loss_coefficient_stability = 0.0
            outlet_effectiveness_stability = 0.0
            recent_gradients = {}

        return {
            "total_predictions": len(self.prediction_history),
            "parameter_updates": len(self.parameter_history),
            "update_percentage": (
                len(self.parameter_history)
                / len(self.prediction_history)
                * 100
                if self.prediction_history
                else 0
            ),
            "avg_recent_error": np.mean(recent_errors),
            "avg_all_time_error": np.mean(all_errors),
            "error_improvement_trend": error_improvement,
            "learning_confidence": self.learning_confidence,
            "current_learning_rate": self._calculate_adaptive_learning_rate(),
            "thermal_time_constant_stability": thermal_stability,
            "heat_loss_coefficient_stability": heat_loss_coefficient_stability,
            "outlet_effectiveness_stability": outlet_effectiveness_stability,
            "recent_gradients": recent_gradients,
            "current_parameters": {
                "thermal_time_constant": self.thermal_time_constant,
                "heat_loss_coefficient": self.heat_loss_coefficient,
                "outlet_effectiveness": self.outlet_effectiveness,
            },
            "fixes_applied": "VERSION_WITH_CORRECTED_GRADIENTS",
        }

    def _save_learning_to_thermal_state(
        self,
        new_thermal_adjustment,
        new_heat_loss_coefficient_adjustment,
        new_outlet_effectiveness_adjustment,
    ):
        """
        Save learned parameter adjustments to unified thermal state.
        """
        try:
            try:
                from .unified_thermal_state import get_thermal_state_manager
            except ImportError:
                from unified_thermal_state import get_thermal_state_manager

            state_manager = get_thermal_state_manager()
            learning_state = state_manager.state.get("learning_state", {})

            current_deltas = learning_state.get("parameter_adjustments", {})
            current_thermal_delta = current_deltas.get(
                "thermal_time_constant_delta", 0.0
            )
            current_heat_loss_coefficient_delta = current_deltas.get(
                "heat_loss_coefficient_delta", 0.0
            )
            current_outlet_effectiveness_delta = current_deltas.get(
                "outlet_effectiveness_delta", 0.0
            )

            updated_thermal_delta = (
                current_thermal_delta + new_thermal_adjustment
            )
            updated_heat_loss_coefficient_delta = (
                current_heat_loss_coefficient_delta
                + new_heat_loss_coefficient_adjustment
            )
            updated_outlet_effectiveness_delta = (
                current_outlet_effectiveness_delta
                + new_outlet_effectiveness_adjustment
            )

            if (
                abs(new_thermal_adjustment) > 0.001
                or abs(new_heat_loss_coefficient_adjustment) > 0.0001
                or abs(new_outlet_effectiveness_adjustment) > 0.0001
            ):
                state_manager.update_learning_state(
                    learning_confidence=self.learning_confidence,
                    parameter_adjustments={
                        "thermal_time_constant_delta": updated_thermal_delta,
                        "heat_loss_coefficient_delta": (
                            updated_heat_loss_coefficient_delta
                        ),
                        "outlet_effectiveness_delta": (
                            updated_outlet_effectiveness_delta
                        ),
                    },
                )
                logging.debug(
                    (
                        "💾 Accumulated learning deltas: "
                        "thermal_Δ=%+.3f (+%+.3f), "
                        "heat_loss_coeff_Δ=%+.5f (+%+.5f), "
                        "outlet_eff_Δ=%+.3f (+%+.3f)"
                    ),
                    updated_thermal_delta,
                    new_thermal_adjustment,
                    updated_heat_loss_coefficient_delta,
                    new_heat_loss_coefficient_adjustment,
                    updated_outlet_effectiveness_delta,
                    new_outlet_effectiveness_adjustment,
                )
            else:
                state_manager.update_learning_state(
                    learning_confidence=self.learning_confidence
                )
                logging.debug(
                    f"💾 Updated learning confidence: "
                    f"{self.learning_confidence:.3f} "
                    f"(no significant parameter changes)"
                )
        except Exception as e:
            logging.error(
                f"❌ Failed to save learning to thermal state: {e}"
            )

    def _detect_parameter_corruption(self) -> bool:
        """
        Detect if parameters are in a corrupted state.
        """
        try:
            from .thermal_config import ThermalParameterConfig
        except ImportError:
            from thermal_config import ThermalParameterConfig

        # Check heat_loss_coefficient bounds
        hcl_bounds = ThermalParameterConfig.get_bounds("heat_loss_coefficient")
        if not (hcl_bounds[0] <= self.heat_loss_coefficient <= hcl_bounds[1]):
            logging.warning(
                "heat_loss_coefficient %s is outside of bounds %s",
                self.heat_loss_coefficient,
                hcl_bounds,
            )
            return True

        # Check outlet_effectiveness bounds
        oe_bounds = ThermalParameterConfig.get_bounds("outlet_effectiveness")
        if not (oe_bounds[0] <= self.outlet_effectiveness <= oe_bounds[1]):
            logging.warning(
                "outlet_effectiveness %s is outside of bounds %s",
                self.outlet_effectiveness,
                oe_bounds,
            )
            return True

        # Check thermal_time_constant bounds
        ttc_bounds = ThermalParameterConfig.get_bounds("thermal_time_constant")
        if not (ttc_bounds[0] <= self.thermal_time_constant <= ttc_bounds[1]):
            logging.warning(
                "thermal_time_constant %s is outside of bounds %s",
                self.thermal_time_constant,
                ttc_bounds,
            )
            return True

        return False

    def reset_adaptive_learning(self):
        """Reset adaptive learning state with aggressive initial settings."""
        self._load_config_defaults()
        self.prediction_history = []
        self.parameter_history = []
        self.learning_confidence = 3.0  # Start with high confidence
        logging.info(
            "Adaptive learning state reset with aggressive settings"
        )
