"""
Outlet Calculator - outlet-temperature inversion for the thermal model.

Extracted from ``ThermalEquilibriumModel`` (Pass C2 refactor) to decouple the
optimal/equilibrium outlet-temperature calculation (the direct-calculation
and equilibrium-fallback paths) from the physics-prediction god-class.

Design notes
------------
* The calculator holds **no state of its own**. All learned parameters
  (``heat_loss_coefficient``, ``outlet_effectiveness``,
  ``external_source_weights``) and the model soft-max
  (``_MAX_OUTLET_TEMP``) remain attributes of the owning
  ``ThermalEquilibriumModel`` instance, accessed via ``self.model``.  This
  keeps the public/test contract intact: tests poke ``model.<attr>`` directly
  and call/patch ``model.calculate_optimal_outlet_temperature`` /
  ``model._calculate_equilibrium_outlet_temperature`` directly.
* ``ThermalEquilibriumModel`` keeps thin delegating methods (same names and
  signatures) so all existing call sites and ``unittest.mock.patch.object``
  patches keep working unchanged.
* Every original ``self.`` reference was translated to ``self.model.`` so that
  cross-method calls route *through the model* - the equilibrium fallback is
  invoked via ``self.model._calculate_equilibrium_outlet_temperature(...)`` so
  monkey-patches on the model still take effect, exactly as before the
  refactor.  The clamp flag (``outlet_clamped_to_max``) is likewise written onto
  ``self.model`` so tests reading ``model.outlet_clamped_to_max`` are
  unaffected.
"""

import logging


class OutletCalculator:
    """Optimal/equilibrium outlet-temperature calculator for
    ``ThermalEquilibriumModel``.

    Owns the two outlet-inversion paths: the direct-calculation path
    (``calculate_optimal_outlet_temperature``) and the equilibrium-fallback
    path (``_calculate_equilibrium_outlet_temperature``), including the
    untrained/degraded-parameter guards (OE / HLC:OE-ratio sanity checks and
    the unified model soft-max clamping with ``outlet_clamped_to_max`` flagging).

    The calculator is constructed once by the model and bound to it via
    ``self.model``; it never caches parameter values, so it always operates
    on the model's current (possibly monkey-patched) state.
    """

    def __init__(self, model):
        self.model = model

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
            outlet_temp = self.model._calculate_equilibrium_outlet_temperature(
                target_indoor, outdoor_temp, pv_power, fireplace_on, tv_on
            )
            return {
                "optimal_outlet_temp": outlet_temp,
                "method": "equilibrium_maintenance",
                "temp_change_required": temp_change_required,
                "time_available": time_available_hours,
            }

        method = "direct_calculation"
        heat_loss_coefficient = self.model.heat_loss_coefficient
        outlet_effectiveness = self.model.outlet_effectiveness
        if config_override:
            heat_loss_coefficient = config_override.get(
                "heat_loss_coefficient", heat_loss_coefficient
            )
            outlet_effectiveness = config_override.get(
                "outlet_effectiveness", outlet_effectiveness
            )
        total_conductance = heat_loss_coefficient + outlet_effectiveness

        external_heating = (
            pv_power * self.model.external_source_weights.get("pv", 0.0)
            + fireplace_on * self.model.external_source_weights.get("fireplace", 0.0)
            + tv_on * self.model.external_source_weights.get("tv", 0.0)
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
        max_outlet = self.model._MAX_OUTLET_TEMP

        optimal_outlet_bounded = max(
            min_outlet, min(optimal_outlet, max_outlet)
        )

        # T0.1: surface when the raw inversion demanded an outlet above the
        # model's soft-max. Silent ceiling-pegging is the untrained-model
        # incident signature; flag it so the control layer / HA can alert.
        if optimal_outlet > max_outlet:
            self.model.outlet_clamped_to_max = True
            logging.warning(
                "⚠️ Direct-calc outlet clamped to model max %.1f°C "
                "(raw calc %.1f°C) — possible untrained/degraded parameters.",
                max_outlet, optimal_outlet,
            )
        else:
            self.model.outlet_clamped_to_max = False

        if optimal_outlet_bounded < outdoor_temp:
            fallback_outlet = self.model._calculate_equilibrium_outlet_temperature(
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
            pv_power * self.model.external_source_weights["pv"]
            + fireplace_on * self.model.external_source_weights["fireplace"]
            + tv_on * self.model.external_source_weights["tv"]
        )

        # T0.3: Sanity guard on outlet effectiveness / HLC:OE ratio. An
        # untrained or degraded model can have a tiny outlet_effectiveness
        # (or a huge heat_loss_coefficient), which makes the equilibrium
        # inversion explode to the ceiling - the 65°C incident root cause.
        # When the parameters are physically implausible, fall back to a
        # conservative, heat-curve-anchored outlet instead of trusting the
        # inversion.
        OE_SAFE_MIN = 0.05
        HLC_OE_RATIO_MAX = 5.0
        oe = self.model.outlet_effectiveness
        hlc = self.model.heat_loss_coefficient

        if oe <= 0:
            self.model.outlet_clamped_to_max = False
            return 35.0

        if oe < OE_SAFE_MIN or (hlc / oe) > HLC_OE_RATIO_MAX:
            self.model.outlet_clamped_to_max = False
            # Conservative heat-curve-anchored outlet: the target plus a
            # bounded lift proportional to the indoor/outdoor delta, never
            # extreme and always below the model soft-max.
            lift = max(0.0, target_temp - outdoor_temp) * 0.7
            conservative_outlet = max(
                max(outdoor_temp + 5, 25.0),
                min(self.model._MAX_OUTLET_TEMP - 5.0, target_temp + lift),
            )
            logging.warning(
                "⚠️ Untrustworthy thermal params (OE=%.4f, HLC=%.4f, "
                "HLC/OE=%.1f) — using conservative heat-curve-anchored "
                "outlet %.1f°C instead of equilibrium inversion.",
                oe, hlc, hlc / oe, conservative_outlet,
            )
            return conservative_outlet

        total_conductance = (
            self.model.heat_loss_coefficient + self.model.outlet_effectiveness
        )
        equilibrium_outlet = (
            target_temp * total_conductance
            - self.model.heat_loss_coefficient * outdoor_temp
            - external_heating
        ) / self.model.outlet_effectiveness

        min_outlet = max(outdoor_temp + 5, 25.0)
        max_outlet = self.model._MAX_OUTLET_TEMP

        clamped_outlet = max(min_outlet, min(equilibrium_outlet, max_outlet))

        # T0.1: flag/warn when the equilibrium outlet is clamped to the
        # model soft-max.
        if equilibrium_outlet > max_outlet:
            self.model.outlet_clamped_to_max = True
            logging.warning(
                "⚠️ Equilibrium outlet clamped to model max %.1f°C "
                "(raw %.1f°C) — possible untrained/degraded parameters.",
                max_outlet, equilibrium_outlet,
            )
        else:
            self.model.outlet_clamped_to_max = False

        return clamped_outlet
