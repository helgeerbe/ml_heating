"""
Parameter Optimizer - gradient-based adaptive learning for the thermal model.

Extracted from ``ThermalEquilibriumModel`` (Pass C1 refactor) to decouple the
online parameter-adaptation logic from the physics-prediction god-class.

Design notes
------------
* The optimizer holds **no state of its own**. All learned parameters and
  history (``prediction_history``, ``parameter_history``, ``learning_confidence``
  bounds, learning rates, ...) remain attributes of the owning
  ``ThermalEquilibriumModel`` instance, accessed via ``self.model``.  This keeps
  the public/test contract intact: tests poke ``model.<attr>`` directly and
  call/patch ``model._<method>`` directly.
* ``ThermalEquilibriumModel`` keeps thin delegating methods (same names) so all
  existing call sites and ``unittest.mock.patch.object(model, ...)`` patches
  keep working unchanged.
* Every original ``self.`` reference was translated to ``self.model.`` so that
  cross-method calls route *through the model* - this guarantees monkey-patches
  on the model (e.g. patching a gradient calculator in the physics-constraint
  tests) still take effect, exactly as before the refactor.  The two bare-``self``
  code sites (``getattr(self, ...)`` / ``setattr(self, ...)`` in
  ``_calculate_parameter_gradient``) were translated the same way.
"""

import logging
from typing import Dict, List

import numpy as np

# Same dual-import pattern as the rest of the package. ``config`` is a singleton
# in ``sys.modules``, so patching ``src.thermal_equilibrium_model.config.<ATTR>``
# (e.g. ``SHADOW_MODE`` in the adaptive-learning-rate tests) also affects this
# module's ``config.<ATTR>`` reference - both names point at the same module
# object.
try:
    from .thermal_constants import PhysicsConstants  # type: ignore
    from . import config  # type: ignore
except ImportError:  # pragma: no cover - direct import fallback for notebooks
    from thermal_constants import PhysicsConstants  # type: ignore
    import config  # type: ignore


class ParameterOptimizer:
    """Gradient-descent parameter adapter for ``ThermalEquilibriumModel``.

    Owns the entire online-learning adaptation loop: per-parameter finite-
    difference gradient calculation, the recent-error adaptation step (with
    cold-weather protection, passive-cycle suppression and catastrophic-error
    gating), the adaptive learning-rate schedule, learning-quality scoring,
    corruption detection, and persistence of learning deltas to the unified
    thermal state.

    The optimizer is constructed once by the model and bound to it via
    ``self.model``; it never caches parameter values, so it always operates on
    the model's current (possibly monkey-patched) state.
    """

    def __init__(self, model):
        self.model = model

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
        recent_predictions = self.model.prediction_history[
            -self.model.recent_errors_window:
        ]

        if len(recent_predictions) < self.model.recent_errors_window:
            return

        if self.model._detect_parameter_corruption():
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

        thermal_gradient = self.model._calculate_thermal_time_constant_gradient(
            recent_predictions
        )
        heat_loss_coefficient_gradient = (
            self.model._calculate_heat_loss_coefficient_gradient(recent_predictions)
        )
        outlet_effectiveness_gradient = (
            self.model._calculate_outlet_effectiveness_gradient(recent_predictions)
        )
        # Equilibrium-based OE gradient supplement (T0.5 fix).
        # The trajectory gradient is too weak in stable rooms (approach factor
        # ~0.017 at 30 min with τ≈29 h). This supplement uses the direct
        # equilibrium error (actual vs predicted equilibrium), which is ~50×
        # stronger and lets the model self-correct OE in hours instead of
        # weeks. Only fires when the room is stable; returns 0.0 otherwise.
        equilibrium_oe_gradient = (
            self.model._calculate_equilibrium_based_oe_gradient(recent_predictions)
        )
        outlet_effectiveness_gradient += equilibrium_oe_gradient

        # Passive-learning protection: if ALL recent predictions are from
        # heating-off cycles, force OE gradient to zero. Passive cycles
        # carry no outlet signal, so any OE gradient would be spurious and
        # would restart the OE ratchet that corrupted the model before.
        # (The per-record skip in _calculate_equilibrium_based_oe_gradient
        # and the passive trajectory path in _calculate_parameter_gradient
        # already reduce this, but we add a belt-and-suspenders guard here
        # for the fully-passive-window case.)
        all_passive = all(
            not p.get("context", {}).get("heating_active", True)
            for p in recent_predictions
        )
        if all_passive:
            outlet_effectiveness_gradient = 0.0
            logging.debug(
                "🔇 All recent predictions are passive (heating off) — "
                "OE gradient suppressed to prevent corruption."
            )
        pv_heat_weight_gradient = (
            self.model._calculate_pv_heat_weight_gradient(recent_predictions)
        )
        tv_heat_weight_gradient = (
            self.model._calculate_tv_heat_weight_gradient(recent_predictions)
        )
        solar_lag_gradient = (
            self.model._calculate_solar_lag_gradient(recent_predictions)
        )

        current_learning_rate = self.model._calculate_adaptive_learning_rate()

        old_thermal_time_constant = self.model.thermal_time_constant
        old_heat_loss_coefficient = self.model.heat_loss_coefficient
        old_outlet_effectiveness = self.model.outlet_effectiveness
        old_pv_heat_weight = self.model.pv_heat_weight
        old_tv_heat_weight = self.model.tv_heat_weight
        old_solar_lag = self.model.solar_lag_minutes

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

            # 2. Protect OE — SYMMETRIC damping (T0.5 fix).
            # Previously only OE *increases* were damped here, creating an
            # asymmetric one-way ratchet: OE decreases (over-prediction
            # episodes, common when thermal mass keeps the house warm) hit
            # OE at full strength, while OE increases (under-prediction
            # episodes) were crushed to 1-10%. Over thousands of cycles this
            # ratcheted outlet_effectiveness from 1.1 → 0.36, corrupting the
            # equilibrium calculation so badly that the model commanded 53°C
            # outlets where 32°C sufficed.
            # Fix: damp BOTH directions equally during cold weather. This
            # preserves the original intent (be cautious during transient
            # cold-weather events) while preventing the one-way drift.
            if outlet_effectiveness_update != 0:
                outlet_effectiveness_update *= damping_factor
                logging.debug(
                    "❄️ Cold weather protection: Dampening OE update by "
                    "%.2f (outdoor=%.1f°C, update=%.5f)",
                    damping_factor,
                    avg_outdoor,
                    outlet_effectiveness_update,
                )
        # Scale down PV update because its gradient is massive
        # (due to PV power being in Watts, 0-5000)
        # A typical finite diff for PV is ~1000. For HLC it's ~10.
        pv_heat_weight_update = (
            current_learning_rate * pv_heat_weight_gradient * 0.001
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

        self.model.thermal_time_constant = float(
            np.clip(
                self.model.thermal_time_constant - thermal_update,
                self.model.thermal_time_constant_bounds[0],
                self.model.thermal_time_constant_bounds[1],
            )
        )

        self.model.heat_loss_coefficient = float(
            np.clip(
                self.model.heat_loss_coefficient - heat_loss_coefficient_update,
                self.model.heat_loss_coefficient_bounds[0],
                self.model.heat_loss_coefficient_bounds[1],
            )
        )

        self.model.outlet_effectiveness = float(
            np.clip(
                self.model.outlet_effectiveness - outlet_effectiveness_update,
                self.model.outlet_effectiveness_bounds[0],
                self.model.outlet_effectiveness_bounds[1],
            )
        )

        self.model.pv_heat_weight = float(
            np.clip(
                self.model.pv_heat_weight - pv_heat_weight_update,
                self.model.pv_heat_weight_bounds[0],
                self.model.pv_heat_weight_bounds[1],
            )
        )

        self.model.tv_heat_weight = float(
            np.clip(
                self.model.tv_heat_weight - tv_heat_weight_update,
                self.model.tv_heat_weight_bounds[0],
                self.model.tv_heat_weight_bounds[1],
            )
        )
        
        self.model.solar_lag_minutes = float(
            np.clip(
                self.model.solar_lag_minutes - solar_lag_update,
                self.model.solar_lag_minutes_bounds[0],
                self.model.solar_lag_minutes_bounds[1],
            )
        )

        thermal_change = abs(
            self.model.thermal_time_constant - old_thermal_time_constant
        )
        heat_loss_coefficient_change = abs(
            self.model.heat_loss_coefficient - old_heat_loss_coefficient
        )
        outlet_effectiveness_change = abs(
            self.model.outlet_effectiveness - old_outlet_effectiveness
        )
        pv_heat_weight_change = abs(self.model.pv_heat_weight - old_pv_heat_weight)
        tv_heat_weight_change = abs(self.model.tv_heat_weight - old_tv_heat_weight)
        solar_lag_change = abs(self.model.solar_lag_minutes - old_solar_lag)

        self.model.parameter_history.append(
            {
                "timestamp": recent_predictions[-1]["timestamp"],
                "thermal_time_constant": self.model.thermal_time_constant,
                "heat_loss_coefficient": self.model.heat_loss_coefficient,
                "outlet_effectiveness": self.model.outlet_effectiveness,
                "pv_heat_weight": self.model.pv_heat_weight,
                "tv_heat_weight": self.model.tv_heat_weight,
                "solar_lag_minutes": self.model.solar_lag_minutes,
                "learning_rate": current_learning_rate,
                "learning_confidence": self.model.learning_confidence,
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

        if len(self.model.parameter_history) > 500:
            self.model.parameter_history = self.model.parameter_history[-500:]

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
                self.model.thermal_time_constant,
                thermal_change,
                old_heat_loss_coefficient,
                self.model.heat_loss_coefficient,
                heat_loss_coefficient_change,
                old_outlet_effectiveness,
                self.model.outlet_effectiveness,
                outlet_effectiveness_change,
                old_pv_heat_weight,
                self.model.pv_heat_weight,
                pv_heat_weight_change,
                old_tv_heat_weight,
                self.model.tv_heat_weight,
                tv_heat_weight_change,
                old_solar_lag,
                self.model.solar_lag_minutes,
                solar_lag_change,
            )

            self.model._save_learning_to_thermal_state(
                self.model.thermal_time_constant - old_thermal_time_constant,
                self.model.heat_loss_coefficient - old_heat_loss_coefficient,
                self.model.outlet_effectiveness - old_outlet_effectiveness,
                self.model.pv_heat_weight - old_pv_heat_weight,
                self.model.tv_heat_weight - old_tv_heat_weight,
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

        Each prediction record carries a ``heating_active`` flag in its
        context. When ``False`` (heat pump was off), the trajectory is
        predicted with the passive model (no outlet/OE term) so that
        gradients for HLC, PV weight, etc. are computed correctly from
        passive dynamics.
        """
        gradient_sum = 0.0
        count = 0

        original_value = getattr(self.model, parameter_name)

        for pred in recent_predictions:
            context = pred["context"]

            if not all(
                key in context
                for key in ["outlet_temp", "outdoor_temp", "current_indoor"]
            ):
                continue

            heating_active = context.get("heating_active", True)

            setattr(self.model, parameter_name, original_value + epsilon)
            pred_plus_trajectory = self.model.predict_thermal_trajectory(
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
                heating_active=heating_active,
            )
            pred_plus = pred_plus_trajectory["trajectory"][0]

            setattr(self.model, parameter_name, original_value - epsilon)
            pred_minus_trajectory = self.model.predict_thermal_trajectory(
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
                heating_active=heating_active,
            )
            pred_minus = pred_minus_trajectory["trajectory"][0]

            setattr(self.model, parameter_name, original_value)

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
        return self.model._calculate_parameter_gradient(
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
        return self.model._calculate_parameter_gradient(
            "heat_loss_coefficient",
            PhysicsConstants.HEAT_LOSS_COEFFICIENT_EPSILON,
            recent_predictions,
        )

    def _calculate_outlet_effectiveness_gradient(
        self, recent_predictions: List[Dict]
    ) -> float:
        """
        Calculate outlet effectiveness gradient using trajectory prediction.

        Combined with the equilibrium-based supplement below (see
        ``_calculate_equilibrium_based_oe_gradient``) to provide a strong
        calibration signal that the trajectory-only gradient is too weak to
        deliver on its own.
        """
        return self.model._calculate_parameter_gradient(
            "outlet_effectiveness",
            PhysicsConstants.OUTLET_EFFECTIVENESS_EPSILON,
            recent_predictions,
        )

    def _calculate_equilibrium_based_oe_gradient(
        self, recent_predictions: List[Dict]
    ) -> float:
        """
        Calculate OE gradient from direct equilibrium error (T0.5 fix).

        PROBLEM: The trajectory-based gradient uses a 30-minute prediction
        horizon. With a thermal time constant of ~29 hours, the approach
        factor at 30 min is only ``1 - exp(-0.5/29) ≈ 0.017``. A 4 °C
        equilibrium error therefore produces only a ~0.07 °C trajectory
        error, yielding a microscopic gradient. The model cannot "see" the
        equilibrium error through the short-horizon trajectory, so OE never
        self-corrects (estimated recovery time: 40+ days).

        SOLUTION: When the room is thermally stable (indoor temp barely
        changing), the actual indoor temperature *is* the equilibrium. We
        compute the predicted equilibrium directly and use the difference
        between actual and predicted equilibrium as the error signal. This
        gives a strong, direct gradient (~50× stronger than the trajectory
        signal) that drives OE to its correct value in hours rather than
        weeks.

        Safety: the equilibrium error is clipped to ±5 °C and the per-cycle
        OE change is clipped to ``MAX_OUTLET_EFFECTIVENESS_CHANGE`` (0.02)
        downstream, so even a large equilibrium error cannot cause a
        destabilising jump.
        """
        gradient_sum = 0.0
        count = 0
        epsilon = PhysicsConstants.OUTLET_EFFECTIVENESS_EPSILON
        original_oe = self.model.outlet_effectiveness
        # Room must be stable: |indoor gradient| below this → actual ≈ equilibrium
        stability_threshold = 0.15  # °C per cycle

        for pred in recent_predictions:
            context = pred.get("context", {})
            actual_temp = pred.get("actual")

            if actual_temp is None:
                continue

            if not all(
                key in context
                for key in ["outlet_temp", "outdoor_temp", "current_indoor"]
            ):
                continue

            # Skip passive (heating-off) records — they carry no outlet/OE
            # signal and would produce a meaningless OE gradient.
            if not context.get("heating_active", True):
                continue

            # Only use equilibrium signal when room is thermally stable
            indoor_gradient = abs(context.get("indoor_temp_gradient", 999))
            if indoor_gradient >= stability_threshold:
                continue

            outlet_temp = context["outlet_temp"]
            outdoor_temp = context["outdoor_temp"]
            current_indoor = context["current_indoor"]
            pv_power = context.get("pv_power", 0)
            fireplace_on = context.get("fireplace_on", 0)
            tv_on = context.get("tv_on", 0)
            thermal_power = context.get("thermal_power")

            eq_kwargs = dict(
                outlet_temp=outlet_temp,
                outdoor_temp=outdoor_temp,
                current_indoor=current_indoor,
                pv_power=pv_power,
                fireplace_on=fireplace_on,
                tv_on=tv_on,
                thermal_power=thermal_power,
                _suppress_logging=True,
            )

            # Predicted equilibrium at the current (original) OE
            eq_base = self.model.predict_equilibrium_temperature(**eq_kwargs)

            # Perturb OE up
            self.model.outlet_effectiveness = original_oe + epsilon
            eq_plus = self.model.predict_equilibrium_temperature(**eq_kwargs)

            # Perturb OE down
            self.model.outlet_effectiveness = original_oe - epsilon
            eq_minus = self.model.predict_equilibrium_temperature(**eq_kwargs)

            # Restore original OE
            self.model.outlet_effectiveness = original_oe

            # Finite difference: d(equilibrium) / d(OE)
            finite_diff = (eq_plus - eq_minus) / (2 * epsilon)

            # Equilibrium error (same sign convention as prediction_error:
            # actual - predicted; positive = under-prediction)
            equilibrium_error = float(np.clip(actual_temp - eq_base, -5.0, 5.0))

            gradient = -finite_diff * equilibrium_error
            gradient_sum += gradient
            count += 1

        if count == 0:
            return 0.0

        avg_gradient = gradient_sum / count
        logging.debug(
            "🔬 Equilibrium-based OE gradient: %.5f (from %d stable samples, "
            "error=%.3f°C)",
            avg_gradient,
            count,
            equilibrium_error,
        )
        return avg_gradient

    def _calculate_pv_heat_weight_gradient(
        self, recent_predictions: List[Dict]
    ) -> float:
        """
        Calculate PV heat weight gradient.
        """
        return self.model._calculate_parameter_gradient(
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
        return self.model._calculate_parameter_gradient(
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
        return self.model._calculate_parameter_gradient(
            "solar_lag_minutes",
            5.0,
            recent_predictions,
        )

    def _calculate_adaptive_learning_rate(self) -> float:
        """
        Calculate adaptive learning rate based on model performance.
        """
        base_rate = (
            max(self.model.learning_rate, self.model.min_learning_rate)
            * self.model.learning_confidence
        )

        heat_loss_coefficient_std = 0.0
        thermal_time_constant_std = 0.0
        outlet_effectiveness_std = 0.0
        if len(self.model.parameter_history) >= 3:
            recent_params = self.model.parameter_history[-3:]
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

        if self.model.prediction_history:
            recent_errors = [
                abs(p["error"]) for p in self.model.prediction_history[-5:]
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
                raw_errors = [p["error"] for p in self.model.prediction_history[-5:]]
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

            elif len(self.model.prediction_history) >= 5:
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

        return np.clip(base_rate, 0.0, self.model.max_learning_rate)

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
                    learning_confidence=self.model.learning_confidence,
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
                    learning_confidence=self.model.learning_confidence
                )
                logging.debug(
                    f"💾 Updated learning confidence: "
                    f"{self.model.learning_confidence:.3f} "
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
        if not (hcl_bounds[0] <= self.model.heat_loss_coefficient <= hcl_bounds[1]):
            logging.warning(
                "heat_loss_coefficient %s is outside of bounds %s",
                self.model.heat_loss_coefficient,
                hcl_bounds,
            )
            return True

        # Check outlet_effectiveness bounds
        oe_bounds = ThermalParameterConfig.get_bounds("outlet_effectiveness")
        if not (oe_bounds[0] <= self.model.outlet_effectiveness <= oe_bounds[1]):
            logging.warning(
                "outlet_effectiveness %s is outside of bounds %s",
                self.model.outlet_effectiveness,
                oe_bounds,
            )
            return True

        # Check for physically impossible combinations (drift detection)
        # High heat loss + low effectiveness = broken physics model
        # Also catch extreme heat loss regardless of effectiveness
        if self.model.heat_loss_coefficient > 1.8:
            logging.warning(
                "⚠️ Extreme heat loss detected (%.2f). Resetting to prevent "
                "runaway heating.",
                self.model.heat_loss_coefficient,
            )
            return True

        if (
            self.model.heat_loss_coefficient > 1.2
            and self.model.outlet_effectiveness < 0.4
        ):
            logging.warning(
                "⚠️ Parameter drift detected: High heat loss (%.2f) "
                "with low effectiveness (%.2f) indicates broken learning "
                "state",
                self.model.heat_loss_coefficient,
                self.model.outlet_effectiveness,
            )
            return True

        # Enhanced check for "borderline bad" combination that causes 65C
        # Lowered threshold to catch more cases (0.8 -> 0.6)
        if (
            self.model.heat_loss_coefficient > 0.6
            and self.model.outlet_effectiveness < 0.35
        ):
            logging.warning(
                "⚠️ Parameter drift detected: Moderate heat loss (%.2f) "
                "with very low effectiveness (%.2f) causes extreme "
                "predictions",
                self.model.heat_loss_coefficient,
                self.model.outlet_effectiveness,
            )
            return True

        # Check thermal_time_constant bounds
        ttc_bounds = ThermalParameterConfig.get_bounds("thermal_time_constant")
        if not (ttc_bounds[0] <= self.model.thermal_time_constant <= ttc_bounds[1]):
            logging.warning(
                "thermal_time_constant %s is outside of bounds %s",
                self.model.thermal_time_constant,
                ttc_bounds,
            )
            return True

        return False
