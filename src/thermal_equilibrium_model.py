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
    
    _load_warning_logged = False

    def __init__(self):
        # Initialize default attributes to ensure existence
        self.solar_lag_minutes = config.SOLAR_LAG_MINUTES

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
            try:
                from .unified_thermal_state import get_thermal_state_manager
            except ImportError:
                from unified_thermal_state import get_thermal_state_manager

            state_manager = get_thermal_state_manager()
            # Only log the warning if the file actually exists but failed
            # to load or if we haven't logged it recently to avoid spamming
            if not state_manager.load_state():
                if not ThermalEquilibriumModel._load_warning_logged:
                    logging.warning(
                        "⚠️ Failed to load thermal state. System is "
                        "running with DEFAULT parameters (HLC=%.2f). "
                        "Check logs for load errors.",
                        state_manager.state["baseline_parameters"][
                            "heat_loss_coefficient"
                        ]
                    )
                    ThermalEquilibriumModel._load_warning_logged = True

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
                self.solar_lag_minutes = (
                    baseline_params.get(
                        "solar_lag_minutes", config.SOLAR_LAG_MINUTES
                    )
                    + adjustments.get("solar_lag_minutes_delta", 0.0)
                )

                self.external_source_weights = {
                    "pv": baseline_params.get(
                        "pv_heat_weight", config.PV_HEAT_WEIGHT
                    ) + adjustments.get("pv_heat_weight_delta", 0.0),
                    "fireplace": baseline_params.get(
                        "fireplace_heat_weight", config.FIREPLACE_HEAT_WEIGHT,
                    ) + adjustments.get("fireplace_heat_weight_delta", 0.0),
                    "tv": baseline_params.get(
                        "tv_heat_weight", config.TV_HEAT_WEIGHT
                    ) + adjustments.get("tv_heat_weight_delta", 0.0),
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
                    "   pv_heat_weight: %.4f",
                    self.external_source_weights["pv"]
                )

                # Validate parameters using schema validator
                try:
                    from thermal_state_validator import (
                        validate_thermal_state_safely,
                    )

                    if not validate_thermal_state_safely(thermal_state):
                        logging.warning(
                            "⚠️ Thermal state validation failed, but core "
                            "parameters were loaded. Retaining calibrated "
                            "parameters and relying on safety clamping."
                        )
                        # We do NOT revert to defaults here because we
                        # successfully loaded the baseline parameters above.
                        # Reverting would lose calibration due to minor schema
                        # mismatches (e.g. missing optional fields).
                        # The code below will clamp parameters to safe bounds.
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

                # EXTRA SAFETY: Clamp parameters to current bounds
                # This handles cases where bounds were tightened in config
                # but old parameters are still within the old (wider) bounds
                # but outside the new (tighter) bounds.
                try:
                    try:
                        from .thermal_config import ThermalParameterConfig
                    except ImportError:
                        from thermal_config import ThermalParameterConfig

                    # Clamp heat_loss_coefficient
                    hcl_bounds = ThermalParameterConfig.get_bounds(
                        "heat_loss_coefficient"
                    )
                    if not (
                        hcl_bounds[0]
                        <= self.heat_loss_coefficient
                        <= hcl_bounds[1]
                    ):
                        logging.warning(
                            "⚠️ Clamping heat_loss_coefficient %.4f to "
                            "bounds %s",
                            self.heat_loss_coefficient,
                            hcl_bounds,
                        )
                        self.heat_loss_coefficient = max(
                            hcl_bounds[0],
                            min(self.heat_loss_coefficient, hcl_bounds[1]),
                        )

                    # Clamp thermal_time_constant
                    ttc_bounds = ThermalParameterConfig.get_bounds(
                        "thermal_time_constant"
                    )
                    if not (
                        ttc_bounds[0]
                        <= self.thermal_time_constant
                        <= ttc_bounds[1]
                    ):
                        logging.warning(
                            "⚠️ Clamping thermal_time_constant %.2f to "
                            "bounds %s",
                            self.thermal_time_constant,
                            ttc_bounds,
                        )
                        self.thermal_time_constant = max(
                            ttc_bounds[0],
                            min(self.thermal_time_constant, ttc_bounds[1]),
                        )

                    # Clamp outlet_effectiveness
                    oe_bounds = ThermalParameterConfig.get_bounds(
                        "outlet_effectiveness"
                    )
                    if not (
                        oe_bounds[0]
                        <= self.outlet_effectiveness
                        <= oe_bounds[1]
                    ):
                        logging.warning(
                            "⚠️ Clamping outlet_effectiveness %.4f to "
                            "bounds %s",
                            self.outlet_effectiveness,
                            oe_bounds,
                        )
                        self.outlet_effectiveness = max(
                            oe_bounds[0],
                            min(self.outlet_effectiveness, oe_bounds[1]),
                        )

                except Exception as e:
                    logging.error(f"Error during parameter clamping: {e}")

                # STABILITY FIX: Detect and reset corrupted parameters on load
                if self._detect_parameter_corruption():
                    logging.warning(
                        "🗑️ Detected corrupted parameters on load. "
                        "Resetting to defaults and clearing learning state."
                    )
                    logging.warning(
                        "   Corrupted values - Heat Loss: %.4f, "
                        "Effectiveness: %.4f",
                        self.heat_loss_coefficient,
                        self.outlet_effectiveness
                    )

                    # CRITICAL: Reset persistent state to prevent reloading
                    # bad deltas
                    
                    # Check if baseline itself is corrupted
                    # If baseline is bad, we MUST wipe it, otherwise it
                    # persists
                    baseline_hl = baseline_params.get(
                        "heat_loss_coefficient", 0
                    )
                    baseline_oe = baseline_params.get(
                        "outlet_effectiveness", 1
                    )

                    is_baseline_corrupted = (
                        baseline_hl > 1.2
                        or baseline_oe < 0.4
                        or (
                            baseline_hl > 0.8 and baseline_oe < 0.35
                        )  # Specific bad combo
                    )

                    if is_baseline_corrupted:
                        logging.error(
                            "🚨 BASELINE IS CORRUPTED! (HL=%.2f, Eff=%.2f). "
                            "Wiping entire thermal state including baseline.",
                            baseline_hl,
                            baseline_oe,
                        )
                        state_manager.reset_learning_state(keep_baseline=False)
                    else:
                        logging.warning(
                            "⚠️ Baseline seems okay, only resetting "
                            "learning state."
                        )
                        state_manager.reset_learning_state(keep_baseline=True)

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
        self.thermal_time_constant = thermal_params.get(
            "thermal_time_constant"
        )
        self.heat_loss_coefficient = thermal_params.get(
            "heat_loss_coefficient"
        )
        self.outlet_effectiveness = thermal_params.get("outlet_effectiveness")
        self.solar_lag_minutes = thermal_params.get("solar_lag_minutes")

        self.external_source_weights = {
            "pv": thermal_params.get("pv_heat_weight"),
            "fireplace": thermal_params.get("fireplace_heat_weight"),
            "tv": thermal_params.get("tv_heat_weight"),
        }

        # Initialize remaining attributes
        self._initialize_learning_attributes()

    @property
    def pv_heat_weight(self) -> float:
        """Get PV heat weight."""
        return self.external_source_weights.get("pv", 0.0)

    @pv_heat_weight.setter
    def pv_heat_weight(self, value: float):
        """Set PV heat weight."""
        self.external_source_weights["pv"] = value

    @property
    def tv_heat_weight(self) -> float:
        """Get TV heat weight."""
        return self.external_source_weights.get("tv", 0.0)

    @tv_heat_weight.setter
    def tv_heat_weight(self, value: float):
        """Set TV heat weight."""
        self.external_source_weights["tv"] = value

    @property
    def fireplace_heat_weight(self) -> float:
        """Get Fireplace heat weight."""
        return self.external_source_weights.get("fireplace", 0.0)

    @fireplace_heat_weight.setter
    def fireplace_heat_weight(self, value: float):
        """Set Fireplace heat weight."""
        self.external_source_weights["fireplace"] = value

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
        self.pv_heat_weight_bounds = ThermalParameterConfig.get_bounds(
            "pv_heat_weight"
        )
        self.tv_heat_weight_bounds = ThermalParameterConfig.get_bounds(
            "tv_heat_weight"
        )
        self.solar_lag_minutes_bounds = ThermalParameterConfig.get_bounds(
            "solar_lag_minutes"
        )

        self.parameter_stability_threshold = (
            PhysicsConstants.THERMAL_STABILITY_THRESHOLD
        )
        self.error_improvement_threshold = (
            PhysicsConstants.ERROR_IMPROVEMENT_THRESHOLD
        )

    def _calculate_effective_solar(
        self, pv_input: float | List[float], lag_minutes: float = None
    ) -> float:
        """
        Calculate effective solar power using a rolling average window.
        Uses fractional steps to ensure differentiability for optimization.

        Args:
            pv_input: Current PV power (float) or history list (List[float])
            lag_minutes: Window size in minutes (defaults to self.solar_lag)

        Returns:
            Effective PV power (float)
        """
        if lag_minutes is None:
            lag_minutes = self.solar_lag_minutes

        # Handle scalar input or empty list
        if isinstance(pv_input, (int, float)):
            return float(pv_input)
        if not pv_input:
            return 0.0

        # Avoid division by zero for very small lags
        if lag_minutes < 1e-3:
            return float(pv_input[-1])

        # Calculate number of steps to average (fractional)
        step_minutes = config.HISTORY_STEP_MINUTES
        float_steps = lag_minutes / step_minutes

        # Split into full steps and fractional remainder
        num_full_steps = int(float_steps)
        fraction = float_steps - num_full_steps

        total_val = 0.0

        # Add full steps (most recent N steps)
        if num_full_steps > 0:
            full_window = pv_input[-num_full_steps:]
            total_val += sum(full_window)

        # Add fractional step (the one just before the full window)
        if fraction > 0:
            # Index for the partial step is -(num_full_steps + 1)
            partial_idx = -(num_full_steps + 1)
            if abs(partial_idx) <= len(pv_input):
                total_val += pv_input[partial_idx] * fraction
            # If history isn't long enough, we assume 0 for the missing
            # past data

        return float(total_val / float_steps)

    def predict_equilibrium_temperature(
        self,
        outlet_temp: float,
        outdoor_temp: float,
        current_indoor: float,
        pv_power: float | List[float] = 0,
        fireplace_on: float = 0,
        tv_on: float = 0,
        thermal_power: float = None,
        auxiliary_heat: float = 0.0,
        _suppress_logging: bool = False,
        fireplace_power_kw: float = None,
    ) -> float:
        """
        Predict equilibrium temperature using standard heat balance physics.
        Supports both temperature-based approximation and energy-based
        modeling.
        """
        # Calculate effective solar power with lag
        effective_pv = self._calculate_effective_solar(pv_power)
        heat_from_pv = effective_pv * self.external_source_weights.get(
            "pv", 0.0
        )

        if fireplace_power_kw is not None:
            heat_from_fireplace = fireplace_power_kw
        else:
            heat_from_fireplace = (
                fireplace_on
                * self.external_source_weights.get("fireplace", 0.0)
            )

        heat_from_tv = tv_on * self.external_source_weights.get("tv", 0.0)

        external_thermal_power = (
            heat_from_pv + heat_from_fireplace + heat_from_tv + auxiliary_heat
        )

        # Energy-based physical modeling (Preferred if thermal_power is
        # available)
        if thermal_power is not None:
            # Teq = Tout + (P_total / U_loss)
            # P_total = P_thermal + P_external
            total_power = thermal_power + external_thermal_power

            if self.heat_loss_coefficient > 0:
                equilibrium_temp = outdoor_temp + (
                    total_power / self.heat_loss_coefficient
                )
            else:
                equilibrium_temp = outdoor_temp

            if not _suppress_logging:
                logging.debug(
                    "⚡ Energy physics: thermal_power=%.3f, ext_power=%.3f, "
                    "U_loss=%.3f, equilibrium=%.2f°C",
                    thermal_power,
                    external_thermal_power,
                    self.heat_loss_coefficient,
                    equilibrium_temp,
                )
            return equilibrium_temp

        # Fallback: Temperature-based approximation
        
        # RESTORED: Differential-based effectiveness scaling
        # Radiators are more effective at higher temperatures (non-linear
        # output). We scale the base effectiveness based on how hot the outlet
        # is relative to the room.
        # Base scaling factor: 1.0 at 45°C (typical), increases for hotter
        # water.
        #
        # UPDATE (2026-03-05): Disabled scaling (factor 0.0) as it was causing
        # significant over-prediction of indoor temperature at high outlet
        # temps, leading to under-heating.
        if outlet_temp > current_indoor:
            temp_diff = max(1.0, outlet_temp - current_indoor)
            # Scale factor increases with temperature difference
            # 0.5 at 25°C diff, 1.0 at 45°C diff, 1.5 at 65°C diff
            # REDUCED TO 0.0 to disable scaling temporarily
            scaling_factor = 1.0 + (temp_diff - 25.0) * 0.0
            effective_outlet_effectiveness = (
                self.outlet_effectiveness * max(0.8, scaling_factor)
            )
            if not _suppress_logging and scaling_factor > 1.1:
                logging.debug(
                    "🔥 High-Diff Scaling: diff=%.1f, factor=%.3f, "
                    "base_eff=%.3f, effective_eff=%.3f",
                    temp_diff, scaling_factor, self.outlet_effectiveness,
                    effective_outlet_effectiveness
                )
        else:
            effective_outlet_effectiveness = self.outlet_effectiveness

        total_conductance = (
            self.heat_loss_coefficient + effective_outlet_effectiveness
        )
        if total_conductance > 0:
            equilibrium_temp = (
                effective_outlet_effectiveness * outlet_temp
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
            # Check if this is a safety floor event (outlet temp clamped to
            # min). If so, we should still learn because the model might be
            # over-optimistic
            outlet_temp = prediction_context.get("outlet_temp")
            is_safety_floor = False
            # Assuming 25 is safety floor
            if outlet_temp is not None and outlet_temp <= 25.0:
                is_safety_floor = True

            if not is_safety_floor:
                logging.debug(
                    "⏸️ Skipping learning during blocking event (DHW/Defrost)"
                )
                return
            else:
                logging.info("⚠️ Learning enabled during safety floor event")

        if self._detect_parameter_corruption():
            logging.warning(
                "🛑 Parameter corruption detected - learning DISABLED"
            )
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

        # Pass PV history if available for trajectory calculation
        pv_input = prediction_context.get("pv_power_history")
        if pv_input is None:
            pv_input = prediction_context.get("pv_power", 0)

        predicted_trajectory = self.predict_thermal_trajectory(
            current_indoor=current_indoor,
            target_indoor=current_indoor,
            outlet_temp=outlet_temp,
            outdoor_temp=prediction_context.get("outdoor_temp", 10.0),
            time_horizon_hours=config.CYCLE_INTERVAL_MINUTES / 60.0,
            time_step_minutes=config.CYCLE_INTERVAL_MINUTES,
            pv_power=pv_input,
            fireplace_on=prediction_context.get("fireplace_on", 0),
            tv_on=prediction_context.get("tv_on", 0),
            thermal_power=prediction_context.get("thermal_power"),
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
                "solar_lag_minutes": self.solar_lag_minutes,
            },
            "shadow_mode": config.SHADOW_MODE,
            "system_state": system_state,
            "learning_quality": self._assess_learning_quality(
                prediction_context, prediction_error
            ),
        }

        self.prediction_history.append(prediction_record)

        if len(self.prediction_history) > 200:
            self.prediction_history = self.prediction_history[-200:]

        if len(self.prediction_history) >= self.recent_errors_window:
            error_magnitude = abs(prediction_error)
            if error_magnitude < PhysicsConstants.ERROR_THRESHOLD_CONFIDENCE:
                # Good prediction
                self.learning_confidence *= self.confidence_boost_rate
            elif error_magnitude > PhysicsConstants.ERROR_THRESHOLD_HIGH:
                # Bad prediction
                self.learning_confidence *= self.confidence_decay_rate

            self.learning_confidence = float(
                np.clip(self.learning_confidence, 0.1, 5.0)
            )
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

            if (
                error_magnitude < PhysicsConstants.ERROR_THRESHOLD_LOW
                and is_stable
            ):
                return "excellent"
            elif (
                error_magnitude < PhysicsConstants.ERROR_THRESHOLD_MEDIUM
                and is_stable
            ):
                return "good"
            elif error_magnitude < PhysicsConstants.ERROR_THRESHOLD_LOW:
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
            # Check if this is due to PV over-estimation (predicted >> actual)
            # In this case, we WANT to learn (reduce PV weight)
            raw_errors = [p["error"] for p in recent_predictions]
            has_large_negative_error = any(e <= -5.0 for e in raw_errors)

            if has_large_negative_error:
                logging.warning(
                    "⚠️ Large negative error (over-prediction) detected - "
                    "Allowing learning to correct model."
                )
            else:
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
        pv_heat_weight_gradient = (
            self._calculate_pv_heat_weight_gradient(recent_predictions)
        )
        tv_heat_weight_gradient = (
            self._calculate_tv_heat_weight_gradient(recent_predictions)
        )
        solar_lag_gradient = (
            self._calculate_solar_lag_gradient(recent_predictions)
        )

        current_learning_rate = self._calculate_adaptive_learning_rate()

        old_thermal_time_constant = self.thermal_time_constant
        old_heat_loss_coefficient = self.heat_loss_coefficient
        old_outlet_effectiveness = self.outlet_effectiveness
        old_pv_heat_weight = self.pv_heat_weight
        old_tv_heat_weight = self.tv_heat_weight
        old_solar_lag = self.solar_lag_minutes

        thermal_update = current_learning_rate * thermal_gradient
        heat_loss_coefficient_update = (
            current_learning_rate * heat_loss_coefficient_gradient
        )

        outlet_effectiveness_update = (
            current_learning_rate * outlet_effectiveness_gradient
        )

        # COLD WEATHER PROTECTION:
        # If outdoor temperature is dropping rapidly (e.g. night), the house
        # thermal mass (inertia) keeps it warm. The model might misinterpret
        # this "staying warm" as "better insulation" (lower heat loss coeff)
        # or "better radiators" (higher outlet effectiveness).
        # We must prevent parameters from drifting incorrectly during these
        # transient events.

        # Calculate average outdoor temp from recent history
        recent_outdoor = [
            p["context"].get("outdoor_temp", 0) for p in recent_predictions
        ]
        avg_outdoor = np.mean(recent_outdoor)

        # If it's cold outside, be skeptical of "better performance"
        if avg_outdoor < PhysicsConstants.COLD_WEATHER_PROTECTION_THRESHOLD:
            damping_factor = PhysicsConstants.COLD_WEATHER_DAMPING_FACTOR
            if (
                avg_outdoor
                < PhysicsConstants.EXTREME_COLD_PROTECTION_THRESHOLD
            ):
                damping_factor = PhysicsConstants.EXTREME_COLD_DAMPING_FACTOR

            # 1. Protect HLC (prevent reduction)
            # update > 0 means we subtract, so HLC goes down
            if heat_loss_coefficient_update > 0:
                heat_loss_coefficient_update *= damping_factor
                logging.debug(
                    "❄️ Cold weather protection: Dampening HLC reduction by "
                    "%.2f (outdoor=%.1f°C)",
                    damping_factor,
                    avg_outdoor,
                )

            # 2. Protect OE (prevent increase)
            # update < 0 means we subtract a negative, so OE goes up
            if outlet_effectiveness_update < 0:
                outlet_effectiveness_update *= damping_factor
                logging.debug(
                    "❄️ Cold weather protection: Dampening OE increase by "
                    "%.2f (outdoor=%.1f°C)",
                    damping_factor,
                    avg_outdoor,
                )
        pv_heat_weight_update = (
            current_learning_rate * pv_heat_weight_gradient
        )
        tv_heat_weight_update = (
            current_learning_rate * tv_heat_weight_gradient
        )
        solar_lag_update = (
            current_learning_rate * solar_lag_gradient * 100.0
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

        # Clip PV/TV updates (limit max change per step)
        # Increased from 0.0005 to 0.005 to allow faster adaptation for PV
        # The parameter range is typically 0.0001-0.005, so 0.0005 was too slow
        max_weight_change = 0.005
        pv_heat_weight_update = np.clip(
            pv_heat_weight_update, -max_weight_change, max_weight_change
        )
        tv_heat_weight_update = np.clip(
            tv_heat_weight_update, -max_weight_change, max_weight_change
        )
        
        # Limit lag change to +/- 5 minutes per update
        solar_lag_update = np.clip(solar_lag_update, -5.0, 5.0)

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

        self.pv_heat_weight = float(
            np.clip(
                self.pv_heat_weight - pv_heat_weight_update,
                self.pv_heat_weight_bounds[0],
                self.pv_heat_weight_bounds[1],
            )
        )

        self.tv_heat_weight = float(
            np.clip(
                self.tv_heat_weight - tv_heat_weight_update,
                self.tv_heat_weight_bounds[0],
                self.tv_heat_weight_bounds[1],
            )
        )
        
        self.solar_lag_minutes = float(
            np.clip(
                self.solar_lag_minutes - solar_lag_update,
                self.solar_lag_minutes_bounds[0],
                self.solar_lag_minutes_bounds[1],
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
        pv_heat_weight_change = abs(self.pv_heat_weight - old_pv_heat_weight)
        tv_heat_weight_change = abs(self.tv_heat_weight - old_tv_heat_weight)
        solar_lag_change = abs(self.solar_lag_minutes - old_solar_lag)

        self.parameter_history.append(
            {
                "timestamp": recent_predictions[-1]["timestamp"],
                "thermal_time_constant": self.thermal_time_constant,
                "heat_loss_coefficient": self.heat_loss_coefficient,
                "outlet_effectiveness": self.outlet_effectiveness,
                "pv_heat_weight": self.pv_heat_weight,
                "tv_heat_weight": self.tv_heat_weight,
                "solar_lag_minutes": self.solar_lag_minutes,
                "learning_rate": current_learning_rate,
                "learning_confidence": self.learning_confidence,
                "avg_recent_error": np.mean(
                    [abs(p["error"]) for p in recent_predictions]
                ),
                "gradients": {
                    "thermal": thermal_gradient,
                    "heat_loss_coefficient": heat_loss_coefficient_gradient,
                    "outlet_effectiveness": outlet_effectiveness_gradient,
                    "pv_heat_weight": pv_heat_weight_gradient,
                    "tv_heat_weight": tv_heat_weight_gradient,
                    "solar_lag": solar_lag_gradient,
                },
                "changes": {
                    "thermal": thermal_change,
                    "heat_loss_coefficient": heat_loss_coefficient_change,
                    "outlet_effectiveness": outlet_effectiveness_change,
                    "pv_heat_weight": pv_heat_weight_change,
                    "tv_heat_weight": tv_heat_weight_change,
                    "solar_lag": solar_lag_change,
                },
            }
        )

        if len(self.parameter_history) > 500:
            self.parameter_history = self.parameter_history[-500:]

        if (
            thermal_change > 0.001
            or heat_loss_coefficient_change > 0.0001
            or outlet_effectiveness_change > 0.0001
            or pv_heat_weight_change > 0.0001
            or tv_heat_weight_change > 0.0001
            or solar_lag_change > 0.1
        ):
            logging.info(
                "Adaptive learning update: "
                "thermal: %.2f→%.2f (Δ%+.3f), "
                "heat_loss_coeff: %.4f→%.4f (Δ%+.5f), "
                "outlet_eff: %.3f→%.3f (Δ%+.3f), "
                "pv_weight: %.3f→%.3f (Δ%+.3f), "
                "tv_weight: %.3f→%.3f (Δ%+.3f), "
                "solar_lag: %.1f→%.1f (Δ%+.1f)",
                old_thermal_time_constant,
                self.thermal_time_constant,
                thermal_change,
                old_heat_loss_coefficient,
                self.heat_loss_coefficient,
                heat_loss_coefficient_change,
                old_outlet_effectiveness,
                self.outlet_effectiveness,
                outlet_effectiveness_change,
                old_pv_heat_weight,
                self.pv_heat_weight,
                pv_heat_weight_change,
                old_tv_heat_weight,
                self.tv_heat_weight,
                tv_heat_weight_change,
                old_solar_lag,
                self.solar_lag_minutes,
                solar_lag_change,
            )

            self._save_learning_to_thermal_state(
                self.thermal_time_constant - old_thermal_time_constant,
                self.heat_loss_coefficient - old_heat_loss_coefficient,
                self.outlet_effectiveness - old_outlet_effectiveness,
                self.pv_heat_weight - old_pv_heat_weight,
                self.tv_heat_weight - old_tv_heat_weight,
            )
        else:
            logging.debug(
                "Micro learning update: thermal_Δ=%+.5f, "
                "heat_loss_coeff_Δ=%+.7f, outlet_eff_Δ=%+.5f, "
                "pv_Δ=%+.5f, tv_Δ=%+.5f, lag_Δ=%+.1f",
                thermal_change,
                heat_loss_coefficient_change,
                outlet_effectiveness_change,
                pv_heat_weight_change,
                tv_heat_weight_change,
                solar_lag_change,
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
                thermal_power=context.get("thermal_power"),
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
                thermal_power=context.get("thermal_power"),
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

    def _calculate_pv_heat_weight_gradient(
        self, recent_predictions: List[Dict]
    ) -> float:
        """
        Calculate PV heat weight gradient.
        """
        return self._calculate_parameter_gradient(
            "pv_heat_weight",
            PhysicsConstants.PV_HEAT_WEIGHT_EPSILON,
            recent_predictions,
        )

    def _calculate_tv_heat_weight_gradient(
        self, recent_predictions: List[Dict]
    ) -> float:
        """
        Calculate TV heat weight gradient.
        """
        return self._calculate_parameter_gradient(
            "tv_heat_weight",
            PhysicsConstants.TV_HEAT_WEIGHT_EPSILON,
            recent_predictions,
        )

    def _calculate_solar_lag_gradient(
        self, recent_predictions: List[Dict]
    ) -> float:
        """
        Calculate solar lag gradient.
        """
        # Use a larger epsilon for lag (e.g., 5 minutes) to see an effect
        return self._calculate_parameter_gradient(
            "solar_lag_minutes",
            5.0,
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
                # Check if this is due to PV over-estimation (predicted >>
                # actual) In this case, we WANT to learn (reduce PV weight)
                # We need to check the sign of the error.
                # recent_errors contains absolute errors, so we need to check
                # the raw errors.
                raw_errors = [p["error"] for p in self.prediction_history[-5:]]
                # Error = Actual - Predicted.
                # If Predicted >> Actual, Error is large negative.
                # So we check if we have large negative errors.

                has_large_negative_error = any(e <= -5.0 for e in raw_errors)

                if has_large_negative_error:
                    logging.warning(
                        "⚠️ Large negative error (over-prediction) detected - "
                        "Allowing learning to correct model."
                    )
                    # Allow learning, maybe even boost it?
                    base_rate *= 2.0
                else:
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
                    # errors, now that the physics are corrected. The
                    # confidence mechanism will naturally temper the learning
                    # rate.
                    if avg_error > 2.0:
                        base_rate /= 1.5  # Less aggressive reduction
                else:
                    # Boost learning rate for significant errors to adapt
                    # faster
                    if avg_error > 3.0:
                        base_rate *= 5.0
                    elif (
                        avg_error > PhysicsConstants.ERROR_THRESHOLD_VERY_HIGH
                    ):
                        base_rate *= PhysicsConstants.ERROR_BOOST_FACTOR_HIGH
                    elif avg_error > PhysicsConstants.ERROR_THRESHOLD_HIGH:
                        base_rate *= PhysicsConstants.ERROR_BOOST_FACTOR_MEDIUM
                    elif avg_error > PhysicsConstants.ERROR_THRESHOLD_MEDIUM:
                        base_rate *= PhysicsConstants.ERROR_BOOST_FACTOR_LOW

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
        thermal_time_constant=None,
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
        thermal_power = external_sources.get("thermal_power", None)
        auxiliary_heat = external_sources.get("auxiliary_heat", 0.0)
        fireplace_power_kw = external_sources.get("fireplace_power_kw", None)

        time_step_hours = time_step_minutes / 60.0
        num_steps = int(time_horizon_hours * 60 / time_step_minutes)
        if num_steps == 0:
            num_steps = 1

        # Handle outdoor temperature forecast (array or scalar)
        if isinstance(outdoor_temp, (list, np.ndarray)):
            # Interpolate array input to simulation steps
            source_len = len(outdoor_temp)
            # ASSUMPTION: Input arrays are hourly forecasts starting at t=0
            # We use time-based interpolation instead of index-based to handle
            # horizons shorter than the forecast array (e.g. 1h horizon
            # with 4h forecast)
            source_times = np.arange(source_len) * 1.0
            target_times = np.linspace(0, time_horizon_hours, num_steps)

            outdoor_forecasts = np.interp(
                target_times, source_times, outdoor_temp
            ).tolist()
        elif weather_forecasts:
            # Legacy support for separate argument
            source_len = len(weather_forecasts)
            # Assume weather_forecasts are hourly if not specified, map to
            # steps.
            # FIX: Use actual time steps (0, 1, 2...) instead of scaling to
            # horizon. This prevents time compression when horizon < forecast
            # length.
            source_times = np.arange(source_len) * 1.0
            target_times = np.linspace(0, time_horizon_hours, num_steps)
            outdoor_forecasts = np.interp(
                target_times, source_times, weather_forecasts
            ).tolist()
        else:
            outdoor_forecasts = [outdoor_temp] * num_steps

        # Handle PV power forecast (array or scalar)
        # CRITICAL: For trajectory, we need to maintain the rolling buffer
        # Initialize buffer with history if available, else current value
        pv_buffer = []
        if isinstance(pv_power, list):
            pv_buffer = list(pv_power)
        else:
            # If scalar, assume constant history for initialization
            # (not ideal but safe fallback)
            pv_buffer = [float(pv_power)] * 18  # 3 hours of history

        # Prepare forecast source
        if pv_forecasts:
            source_len = len(pv_forecasts)
            # ASSUMPTION: Input arrays are hourly forecasts starting at t=0
            source_times = np.arange(source_len) * 1.0
            target_times = np.linspace(0, time_horizon_hours, num_steps)

            # Use step interpolation (zero-order hold) for PV to be
            # conservative. This prevents "morning drop" where linear
            # interpolation would start ramping up solar gain immediately
            # from t=0, causing the model to reduce heating before the sun
            # actually provides warmth.
            # NOTE: side='left' ensures that at exact hour boundaries
            # (e.g. t=1.0), we still use the previous value, delaying the new
            # forecast until we are strictly past the hour.
            indices = np.searchsorted(
                source_times, target_times, side='left'
            ) - 1
            indices = np.clip(indices, 0, source_len - 1)
            pv_future_values = np.array(pv_forecasts)[indices].tolist()
        else:
            # If no forecast, assume persistence of last known value
            last_val = pv_buffer[-1] if pv_buffer else 0.0
            pv_future_values = [last_val] * num_steps

        for step in range(num_steps):
            future_outdoor = outdoor_forecasts[step]
            
            # Update PV buffer: append forecast, remove oldest
            next_pv_val = pv_future_values[step]
            pv_buffer.append(next_pv_val)
            if len(pv_buffer) > 18:  # Keep buffer size manageable
                pv_buffer.pop(0)

            # Pass the updated buffer to predict_equilibrium_temperature
            # It will calculate the effective solar based on solar_lag_minutes
            equilibrium_temp = self.predict_equilibrium_temperature(
                outlet_temp=outlet_temp,
                outdoor_temp=future_outdoor,
                current_indoor=current_temp,
                pv_power=pv_buffer,
                fireplace_on=fireplace_on,
                tv_on=tv_on,
                thermal_power=thermal_power,
                auxiliary_heat=auxiliary_heat,
                _suppress_logging=True,
                fireplace_power_kw=fireplace_power_kw,
            )

            time_constant_hours = (
                thermal_time_constant
                if thermal_time_constant is not None
                else self.thermal_time_constant
            )
            approach_factor = 1 - np.exp(
                -time_step_hours / time_constant_hours
            )
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
            "equilibrium_temp": (
                trajectory[-1] if trajectory else current_indoor
            ),
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
        Calculate optimal outlet temperature to reach target indoor
        temperature.
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

        total_conductance = (
            self.heat_loss_coefficient + self.outlet_effectiveness
        )
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

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get relative importance of different heat sources/parameters.
        Acts as a compatibility layer for the legacy feature importance metric.
        """
        # Normalize weights to sum to 1.0 for relative importance
        weights = {
            "pv_power": self.external_source_weights.get("pv", 0.0),
            "fireplace": self.external_source_weights.get("fireplace", 0.0),
            "tv_power": self.external_source_weights.get("tv", 0.0),
            # Proxy for outdoor influence
            "outdoor_temp": self.heat_loss_coefficient,
            # Proxy for target influence
            "target_temp": self.outlet_effectiveness,
        }

        total = sum(abs(v) for v in weights.values())
        if total > 0:
            return {k: abs(v) / total for k, v in weights.items()}
        return weights

    def get_adaptive_learning_metrics(self) -> Dict:
        """
        ENHANCED: Get metrics with additional debugging info.
        """
        # Allow export even with minimal history for debugging
        if len(self.prediction_history) < 1:
            return {
                "insufficient_data": True,
                "history_len": len(self.prediction_history),
                "required_len": 1
            }

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
        new_pv_heat_weight_adjustment=0.0,
        new_tv_heat_weight_adjustment=0.0,
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
            current_pv_heat_weight_delta = current_deltas.get(
                "pv_heat_weight_delta", 0.0
            )
            current_tv_heat_weight_delta = current_deltas.get(
                "tv_heat_weight_delta", 0.0
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
            updated_pv_heat_weight_delta = (
                current_pv_heat_weight_delta + new_pv_heat_weight_adjustment
            )
            updated_tv_heat_weight_delta = (
                current_tv_heat_weight_delta + new_tv_heat_weight_adjustment
            )

            if (
                abs(new_thermal_adjustment) > 0.001
                or abs(new_heat_loss_coefficient_adjustment) > 0.0001
                or abs(new_outlet_effectiveness_adjustment) > 0.0001
                or abs(new_pv_heat_weight_adjustment) > 0.0001
                or abs(new_tv_heat_weight_adjustment) > 0.0001
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
                        "pv_heat_weight_delta": updated_pv_heat_weight_delta,
                        "tv_heat_weight_delta": updated_tv_heat_weight_delta,
                    },
                )
                logging.debug(
                    (
                        "💾 Accumulated learning deltas: "
                        "thermal_Δ=%+.3f (+%+.3f), "
                        "heat_loss_coeff_Δ=%+.5f (+%+.5f), "
                        "outlet_eff_Δ=%+.3f (+%+.3f), "
                        "pv_weight_Δ=%+.3f (+%+.3f), "
                        "tv_weight_Δ=%+.3f (+%+.3f)"
                    ),
                    updated_thermal_delta,
                    new_thermal_adjustment,
                    updated_heat_loss_coefficient_delta,
                    new_heat_loss_coefficient_adjustment,
                    updated_outlet_effectiveness_delta,
                    new_outlet_effectiveness_adjustment,
                    updated_pv_heat_weight_delta,
                    new_pv_heat_weight_adjustment,
                    updated_tv_heat_weight_delta,
                    new_tv_heat_weight_adjustment,
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

        # Check for physically impossible combinations (drift detection)
        # High heat loss + low effectiveness = broken physics model
        # Also catch extreme heat loss regardless of effectiveness
        if self.heat_loss_coefficient > 1.8:
            logging.warning(
                "⚠️ Extreme heat loss detected (%.2f). Resetting to prevent "
                "runaway heating.",
                self.heat_loss_coefficient,
            )
            return True

        if (
            self.heat_loss_coefficient > 1.2
            and self.outlet_effectiveness < 0.4
        ):
            logging.warning(
                "⚠️ Parameter drift detected: High heat loss (%.2f) "
                "with low effectiveness (%.2f) indicates broken learning "
                "state",
                self.heat_loss_coefficient,
                self.outlet_effectiveness,
            )
            return True

        # Enhanced check for "borderline bad" combination that causes 65C
        # Lowered threshold to catch more cases (0.8 -> 0.6)
        if (
            self.heat_loss_coefficient > 0.6
            and self.outlet_effectiveness < 0.35
        ):
            logging.warning(
                "⚠️ Parameter drift detected: Moderate heat loss (%.2f) "
                "with very low effectiveness (%.2f) causes extreme "
                "predictions",
                self.heat_loss_coefficient,
                self.outlet_effectiveness,
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
