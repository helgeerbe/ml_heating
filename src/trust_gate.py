"""
Trust Gate - self-calibration promotion/demotion logic for the model wrapper.

Extracted from ``EnhancedModelWrapper`` (Pass C2 refactor) to decouple the
self-calibrating trust-gate lifecycle (auto-promotion, accuracy demotion /
re-promotion, parameter-stability guard and trust-state reporting) from the
wrapper god-class.

Design notes
------------
* The gate holds **no state of its own**. All collaborators
  (``state_manager``, ``prediction_metrics``, ``thermal_model``), the
  sustained-demotion counter (``_demotion_streak``) and the trust thresholds
  (``MIN_PREDICTIONS_FOR_TRUST`` / ``MIN_PREDICTIONS_FOR_DEMOTION``) remain
  attributes of the owning ``EnhancedModelWrapper`` instance, accessed via
  ``self.wrapper``.  This keeps the public/test contract intact: tests build
  the wrapper via ``__new__`` and poke ``wrapper.<attr>`` directly, then call
  / ``unittest.mock.patch.object`` the gate methods directly on the wrapper.
* ``EnhancedModelWrapper`` keeps thin delegating methods (same names and
  signatures) so all existing call sites and patches keep working unchanged.
  Because three test files construct the wrapper via ``__new__`` (bypassing
  ``__init__``), the wrapper resolves the gate through a **lazy** accessor
  ``_get_trust_gate()`` rather than relying solely on ``__init__``.
* Every original ``self.`` reference was translated to ``self.wrapper.`` so
  that cross-method calls route *through the wrapper* — e.g.
  ``_evaluate_calibration_gate`` invokes
  ``self.wrapper._evaluate_parameter_stability()`` so monkey-patches on the
  wrapper still take effect, exactly as before the refactor.  The
  ``_demotion_streak`` counter is likewise read/written on ``self.wrapper`` so
  the test fixtures that set ``wrapper._demotion_streak`` are unaffected.
* ``config`` is a singleton in ``sys.modules``, so patching
  ``src.model_wrapper.config.<ATTR>`` (e.g. the demotion thresholds in the
  trust-gate tests) also affects this module's ``config.<ATTR>`` reference -
  both names point at the same module object.
"""

import logging
from typing import Dict, Optional

import numpy as np

# Same dual-import pattern as the rest of the package. ``config`` is a
# singleton in ``sys.modules``, so patching ``src.model_wrapper.config.<ATTR>``
# also affects this module's ``config.<ATTR>`` reference.
try:
    from .thermal_config import ThermalParameterConfig  # type: ignore
    from . import config  # type: ignore
except ImportError:  # pragma: no cover - direct import fallback for notebooks
    from thermal_config import ThermalParameterConfig  # type: ignore
    import config  # type: ignore


class TrustGate:
    """Self-calibration trust-gate logic for ``EnhancedModelWrapper``.

    Owns the 5-criteria auto-promotion gate, the sustained accuracy
    demotion / re-promotion hysteresis, the parameter-stability anti-inversion
    guard, the calibrated-model guard and the single-source-of-truth
    ``get_trust_state`` report (Active vs Passthrough).

    The gate is bound to the wrapper via ``self.wrapper`` and never caches
    collaborator references, so it always operates on the wrapper's current
    (possibly monkey-patched) state.
    """

    def __init__(self, wrapper):
        self.wrapper = wrapper

    def _is_model_calibrated(self) -> bool:
        """Return True only when the model has a calibrated baseline AND
        enough prediction history to be trusted for aggressive outlet
        commands.

        Guards against the untrained-model incident: a model with default
        (uncalibrated) parameters pegs the binary-search outlet to the
        ceiling. Until calibration has run and accumulated predictions, the
        caller must cap the outlet to a conservative, heat-curve-anchored
        ceiling.
        """
        try:
            metrics = self.wrapper.state_manager.get_learning_metrics()
            baseline_source = metrics.get("baseline_source", "config")
            total_predictions = metrics.get("total_predictions", 0)
            return (
                baseline_source == "calibrated"
                and total_predictions >= self.wrapper.MIN_PREDICTIONS_FOR_TRUST
            )
        except Exception as e:
            logging.warning(
                "Calibration check failed: %s — treating as uncalibrated", e,
            )
            return False

    # === T0.7: Self-Calibrating Trust Gate ===

    def _evaluate_parameter_stability(self) -> bool:
        """Return True if the key learned parameters have converged (low drift
        relative to their allowed bounds) over the recent history window.

        This is the inversion-safety guard: a model that *looks* accurate but
        whose parameters are still drifting can invert dangerously (the 65°C
        incident). We require stable heat_loss_coefficient and
        outlet_effectiveness — the two parameters that drive the equilibrium
        inversion.
        """
        history = getattr(self.wrapper.thermal_model, "parameter_history", [])
        window = config.AUTO_CAL_PARAM_STABILITY_WINDOW
        if len(history) < window:
            return False
        recent = history[-window:]
        for param in ("heat_loss_coefficient", "outlet_effectiveness"):
            values = [r.get(param) for r in recent
                      if r.get(param) is not None]
            if len(values) < max(5, window // 2):
                return False
            lo, hi = ThermalParameterConfig.get_bounds(param)
            bound_range = hi - lo
            if bound_range <= 0:
                continue
            std = float(np.std(values))
            if std / bound_range > config.AUTO_CAL_PARAM_STABILITY_FRAC:
                return False
        return True

    def _evaluate_calibration_gate(self) -> Dict:
        """Evaluate the 5-criteria auto-promotion gate.

        Returns a dict with each criterion's pass/fail, the raw signals, the
        fraction of criteria met (for `promotion_progress` telemetry) and a
        boolean `all_met`.
        """
        metrics = self.wrapper.state_manager.get_learning_metrics()
        total_predictions = metrics.get("total_predictions", 0)

        pred_metrics = self.wrapper.prediction_metrics.get_metrics(refresh_cache=True)
        window_24h = pred_metrics.get("24h", {})
        mae_24h = float(window_24h.get("mae", 0.0))
        mean_error_24h = float(window_24h.get("mean_error", 0.0))
        good_control_pct = float(
            self.wrapper.prediction_metrics.get_24h_good_control_percentage()
        )
        params_stable = self.wrapper._evaluate_parameter_stability()

        criteria = {
            "sufficient_predictions":
                total_predictions >= config.AUTO_CAL_MIN_PREDICTIONS,
            "mae_ok": mae_24h <= config.AUTO_CAL_MAE_THRESHOLD,
            "bias_ok": abs(mean_error_24h) <= config.AUTO_CAL_BIAS_THRESHOLD,
            "good_control_ok": good_control_pct >= config.AUTO_CAL_GOOD_CONTROL_PCT,
            "params_stable": params_stable,
        }

        all_met = all(criteria.values())
        progress = sum(criteria.values()) / len(criteria)

        return {
            "criteria": criteria,
            "all_met": all_met,
            "promotion_progress": float(progress),
            "signals": {
                "total_predictions": total_predictions,
                "mae_24h": mae_24h,
                "mean_error_24h": mean_error_24h,
                "good_control_pct": good_control_pct,
                "params_stable": params_stable,
            },
        }

    def _check_auto_promotion(self) -> bool:
        """Evaluate the promotion gate and auto-promote if all criteria met.

        Only promotes a not-yet-calibrated model. Returns True if a promotion
        happened this cycle.
        """
        try:
            metrics = self.wrapper.state_manager.get_learning_metrics()
            baseline_source = metrics.get("baseline_source", "config")
            if baseline_source == "calibrated":
                # Already calibrated; promotion is a no-op. Re-promotion of a
                # demoted model is handled by the demotion logic on recovery.
                return False

            gate = self.wrapper._evaluate_calibration_gate()
            if not gate["all_met"]:
                logging.debug(
                    "Self-cal not ready: %d/5 criteria met "
                    "(predictions=%s, mae_24h=%.2f, bias_24h=%.2f, "
                    "good_ctrl=%.1f%%, params_stable=%s).",
                    sum(gate["criteria"].values()),
                    gate["signals"]["total_predictions"],
                    gate["signals"]["mae_24h"],
                    gate["signals"]["mean_error_24h"],
                    gate["signals"]["good_control_pct"],
                    gate["signals"]["params_stable"],
                )
                return False

            # Promote using the current learned parameters.
            params = {
                "thermal_time_constant":
                    self.wrapper.thermal_model.thermal_time_constant,
                "heat_loss_coefficient":
                    self.wrapper.thermal_model.heat_loss_coefficient,
                "outlet_effectiveness":
                    self.wrapper.thermal_model.outlet_effectiveness,
                "pv_heat_weight": self.wrapper.thermal_model.external_source_weights
                                  .get("pv", 0.0),
                "fireplace_heat_weight":
                    self.wrapper.thermal_model.external_source_weights
                    .get("fireplace", 0.0),
                "tv_heat_weight": self.wrapper.thermal_model.external_source_weights
                                  .get("tv", 0.0),
            }
            self.wrapper.state_manager.auto_promote_baseline(
                params, calibration_cycles=self.wrapper.cycle_count
            )
            return True
        except Exception as e:
            logging.error("Auto-promotion check failed: %s", e, exc_info=True)
            return False

    def _check_accuracy_demotion(self) -> None:
        """Evaluate the demotion/re-promotion gate for an already-calibrated
        model.

        Demotes to passthrough when accuracy degrades beyond the promotion
        thresholds scaled by DEMOTION_FACTOR, sustained for
        DEMOTION_SUSTAINED_CYCLES cycles. Re-promotes when accuracy recovers
        below the promotion thresholds.
        """
        try:
            metrics = self.wrapper.state_manager.get_learning_metrics()
            if metrics.get("baseline_source") != "calibrated":
                return  # only applies to a calibrated model

            pred_metrics = self.wrapper.prediction_metrics.get_metrics(
                refresh_cache=True
            )
            window_24h = pred_metrics.get("24h", {})
            mae_24h = float(window_24h.get("mae", 0.0))
            mean_error_24h = float(window_24h.get("mean_error", 0.0))
            good_control_pct = float(
                self.wrapper.prediction_metrics.get_24h_good_control_percentage()
            )

            # Minimum-predictions guard: with fewer than
            # MIN_PREDICTIONS_FOR_DEMOTION time-based predictions in the 24h
            # window the good_control_pct / mae / bias signals are
            # statistically meaningless (e.g. right after a history reset, or
            # during extended heating-off periods where passive learning adds
            # no trust-gate predictions). Skip the entire gate so a sparse
            # window can neither demote a working model nor prematurely
            # re-promote a demoted one. This breaks the restart death-spiral
            # where stale passthrough-era predictions kept re-demoting the
            # model on every service start.
            try:
                pred_count_24h = int(
                    self.wrapper.prediction_metrics.get_24h_prediction_count()
                )
            except Exception:
                pred_count_24h = 0
            if pred_count_24h < self.wrapper.MIN_PREDICTIONS_FOR_DEMOTION:
                logging.debug(
                    "Skipping accuracy demotion check: only %d predictions in "
                    "24h window (minimum %d).",
                    pred_count_24h,
                    self.wrapper.MIN_PREDICTIONS_FOR_DEMOTION,
                )
                self.wrapper._demotion_streak = 0
                return

            demoted = (
                mae_24h > config.AUTO_CAL_MAE_THRESHOLD * config.DEMOTION_FACTOR
                or abs(mean_error_24h)
                > config.AUTO_CAL_BIAS_THRESHOLD * config.DEMOTION_FACTOR
                or good_control_pct < config.AUTO_CAL_GOOD_CONTROL_PCT
                / config.DEMOTION_FACTOR
            )
            # Re-promotion uses the *scaled* good_control threshold
            # (AUTO_CAL_GOOD_CONTROL_PCT / DEMOTION_FACTOR), matching the
            # demotion floor for symmetric hysteresis. The previous unscaled
            # 80% threshold was unreachable once the model was demoted to
            # passthrough (passthrough scoring collapses good_control_pct),
            # trapping the model in a permanent death-spiral. The
            # DEMOTION_SUSTAINED_CYCLES requirement supplies the temporal
            # hysteresis that prevents flapping.
            recovered = (
                mae_24h <= config.AUTO_CAL_MAE_THRESHOLD
                and abs(mean_error_24h) <= config.AUTO_CAL_BIAS_THRESHOLD
                and good_control_pct
                >= config.AUTO_CAL_GOOD_CONTROL_PCT / config.DEMOTION_FACTOR
            )

            accuracy_state = metrics.get("accuracy_state", "unknown")

            # Treat "unknown" (e.g. a config-promoted baseline that was never
            # explicitly marked accurate) the same as "accurate" so it CAN be
            # demoted. Previously "unknown" fell into a dead zone where the
            # demotion/re-promotion logic never ran, leaving a poor model
            # permanently trusted.
            if accuracy_state != "inaccurate":
                if demoted:
                    self.wrapper._demotion_streak += 1
                    if self.wrapper._demotion_streak >= config.DEMOTION_SUSTAINED_CYCLES:
                        self.wrapper.state_manager.mark_inaccurate()
                        self.wrapper._demotion_streak = 0
                else:
                    self.wrapper._demotion_streak = 0
            elif accuracy_state == "inaccurate":
                # Demoted: wait for sustained recovery before re-promoting.
                if recovered:
                    self.wrapper._demotion_streak += 1
                    if self.wrapper._demotion_streak >= config.DEMOTION_SUSTAINED_CYCLES:
                        self.wrapper.state_manager.mark_accurate()
                        self.wrapper._demotion_streak = 0
                else:
                    self.wrapper._demotion_streak = 0
        except Exception as e:
            logging.error("Accuracy demotion check failed: %s", e,
                          exc_info=True)

    def get_trust_state(self, ml_predicted_outlet: Optional[float] = None,
                        heat_curve_outlet: Optional[float] = None) -> Dict:
        """Return the current self-calibration trust state.

        Single source of truth for whether the model may command the
        ML-predicted outlet (Active / state 1) or must defer to the heat-pump's
        own curve (Passthrough / state 0).
        """
        try:
            metrics = self.wrapper.state_manager.get_learning_metrics()
            baseline_source = metrics.get("baseline_source", "config")
            accuracy_state = metrics.get("accuracy_state", "unknown")
            total_predictions = metrics.get("total_predictions", 0)

            is_calibrated = (
                baseline_source == "calibrated"
                and total_predictions >= self.wrapper.MIN_PREDICTIONS_FOR_TRUST
            )
            is_accurate = accuracy_state != "inaccurate"
            # Honor CONFIDENCE_THRESHOLD (configured in .env). Even a
            # calibrated, accurate model must NOT command the outlet when its
            # learning confidence has collapsed below the configured floor.
            # `learning_confidence` is on the same 0.1-5.0 scale as the
            # threshold (see .env: "Model Confidence" / CONFIDENCE_THRESHOLD).
            learning_confidence = float(
                getattr(self.wrapper.thermal_model, "learning_confidence", 0.0)
            )
            is_confident = (
                learning_confidence >= config.CONFIDENCE_THRESHOLD
            )

            if not is_calibrated:
                if baseline_source != "calibrated":
                    reason = "learning_uncalibrated"
                else:
                    reason = "learning_insufficient_predictions"
                stage = "learning"
                progress = self.wrapper._evaluate_calibration_gate()[
                    "promotion_progress"
                ]
            elif not is_accurate:
                reason = "inaccurate"
                stage = "inaccurate"
                progress = 0.0
            elif not is_confident:
                reason = "low_confidence"
                stage = "low_confidence"
                progress = (
                    min(
                        1.0,
                        learning_confidence / config.CONFIDENCE_THRESHOLD,
                    )
                    if config.CONFIDENCE_THRESHOLD > 0
                    else 1.0
                )
            else:
                reason = "active"
                stage = "active"
                progress = 1.0

            is_trusted = is_calibrated and is_accurate and is_confident

            heat_curve_val = None
            if heat_curve_outlet is not None:
                try:
                    heat_curve_val = float(heat_curve_outlet)
                except (TypeError, ValueError):
                    heat_curve_val = None

            return {
                "is_trusted": is_trusted,
                "is_calibrated": is_calibrated,
                "is_accurate": is_accurate,
                "is_confident": is_confident,
                "confidence": learning_confidence,
                "stage": stage,
                "trust_reason": reason,
                "promotion_progress": float(progress),
                "ml_predicted_outlet": ml_predicted_outlet,
                "heat_curve_outlet": heat_curve_val,
                "outlet_source": "ml_predicted" if is_trusted
                                 else "heat_curve_passthrough",
                "calibration_method":
                    metrics.get("calibration_method", "config"),
                "total_predictions": total_predictions,
            }
        except Exception as e:
            logging.warning(
                "get_trust_state failed: %s — returning safe passthrough", e
            )
            return {
                "is_trusted": False,
                "is_calibrated": False,
                "is_accurate": False,
                "is_confident": False,
                "confidence": 0.0,
                "stage": "learning",
                "trust_reason": "evaluation_error",
                "promotion_progress": 0.0,
                "ml_predicted_outlet": ml_predicted_outlet,
                "heat_curve_outlet": None,
                "outlet_source": "heat_curve_passthrough",
                "calibration_method": "config",
                "total_predictions": 0,
            }
