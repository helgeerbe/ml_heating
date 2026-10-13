"""Passive learning tests: heating-OFF cycles must learn passive dynamics
without corrupting outlet_effectiveness.

These tests verify the 6-change implementation:
1. predict_equilibrium_temperature excludes outlet/OE when heating_active=False
2. predict_thermal_trajectory propagates heating_active to equilibrium
3. Gradient calculators are per-record heating_active-aware
4. _calculate_equilibrium_based_oe_gradient skips passive records
5. _adapt_parameters_from_recent_errors suppresses OE when all passive
6. model_wrapper.learn_from_prediction_feedback skips trust-gate metrics
   for passive cycles
"""

import pytest
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src import thermal_equilibrium_model


@pytest.fixture(scope="function")
def clean_model():
    """Fresh model with realistic test parameters."""
    model = thermal_equilibrium_model.ThermalEquilibriumModel()
    model.thermal_time_constant = 4.0
    model.heat_loss_coefficient = 1.2
    model.outlet_effectiveness = 0.75
    model.external_source_weights = {
        "pv": 0.001,
        "fireplace": 1.5,
        "tv": 0.1,
    }
    model.pv_heat_weight = 0.001
    model.tv_heat_weight = 0.1
    model.solar_lag_minutes = 45.0
    return model


def _make_prediction_record(
    predicted, actual, outlet_temp, outdoor_temp, current_indoor,
    heating_active=True, pv_power=0, fireplace_on=0, tv_on=0,
    indoor_gradient=0.0,
):
    """Build a prediction record matching update_prediction_feedback structure."""
    return {
        "predicted": predicted,
        "actual": actual,
        "error": actual - predicted,
        "context": {
            "outlet_temp": outlet_temp,
            "outdoor_temp": outdoor_temp,
            "current_indoor": current_indoor,
            "pv_power": pv_power,
            "fireplace_on": fireplace_on,
            "tv_on": tv_on,
            "heating_active": heating_active,
            "indoor_temp_gradient": indoor_gradient,
        },
    }


class TestPassiveEquilibrium:
    def test_passive_equilibrium_excludes_outlet_term(self, clean_model):
        """Passive equilibrium must NOT depend on outlet_temp at all."""
        outdoor = 5.0
        indoor = 20.0
        eq_low = clean_model.predict_equilibrium_temperature(
            outlet_temp=0.0, outdoor_temp=outdoor, current_indoor=indoor,
            heating_active=False,
        )
        eq_high = clean_model.predict_equilibrium_temperature(
            outlet_temp=100.0, outdoor_temp=outdoor, current_indoor=indoor,
            heating_active=False,
        )
        assert eq_low == pytest.approx(eq_high, abs=1e-9)

    def test_passive_equilibrium_equals_outdoor_plus_ext_over_hlc(
        self, clean_model
    ):
        """T_eq = T_outdoor + (PV + fireplace + TV) / HLC — exact formula."""
        import math
        outdoor = 5.0
        pv_power = 2000.0
        fireplace_on = 1.0
        tv_on = 1.0
        raw_pv = pv_power * clean_model.external_source_weights["pv"]
        max_solar = thermal_equilibrium_model.PhysicsConstants.MAX_SOLAR_CONTRIBUTION
        heat_pv = max_solar * math.tanh(raw_pv / max_solar) if raw_pv > 0 else 0.0
        heat_fire = fireplace_on * clean_model.external_source_weights["fireplace"]
        heat_tv = tv_on * clean_model.external_source_weights["tv"]
        ext = heat_pv + heat_fire + heat_tv
        expected = outdoor + ext / clean_model.heat_loss_coefficient

        result = clean_model.predict_equilibrium_temperature(
            outlet_temp=0.0, outdoor_temp=outdoor, current_indoor=20.0,
            pv_power=pv_power, fireplace_on=fireplace_on, tv_on=tv_on,
            heating_active=False,
        )
        assert result == pytest.approx(expected, abs=0.01)

    def test_passive_equilibrium_with_pv_only(self, clean_model):
        """PV gain raises passive equilibrium above outdoor temp."""
        outdoor = 5.0
        eq_no = clean_model.predict_equilibrium_temperature(
            outlet_temp=0.0, outdoor_temp=outdoor, current_indoor=20.0,
            pv_power=0.0, heating_active=False,
        )
        eq_yes = clean_model.predict_equilibrium_temperature(
            outlet_temp=0.0, outdoor_temp=outdoor, current_indoor=20.0,
            pv_power=3000.0, heating_active=False,
        )
        assert eq_yes > eq_no, "PV should raise passive equilibrium"

    def test_passive_equilibrium_with_fireplace(self, clean_model):
        """Fireplace raises passive equilibrium."""
        outdoor = 5.0
        eq_no = clean_model.predict_equilibrium_temperature(
            outlet_temp=0.0, outdoor_temp=outdoor, current_indoor=20.0,
            fireplace_on=0, heating_active=False,
        )
        eq_yes = clean_model.predict_equilibrium_temperature(
            outlet_temp=0.0, outdoor_temp=outdoor, current_indoor=20.0,
            fireplace_on=1, heating_active=False,
        )
        assert eq_yes > eq_no, "Fireplace should raise passive equilibrium"

    def test_passive_equilibrium_with_tv(self, clean_model):
        """TV heat gain raises passive equilibrium."""
        outdoor = 5.0
        eq_no = clean_model.predict_equilibrium_temperature(
            outlet_temp=0.0, outdoor_temp=outdoor, current_indoor=20.0,
            tv_on=0, heating_active=False,
        )
        eq_yes = clean_model.predict_equilibrium_temperature(
            outlet_temp=0.0, outdoor_temp=outdoor, current_indoor=20.0,
            tv_on=1, heating_active=False,
        )
        assert eq_yes > eq_no, "TV should raise passive equilibrium"


class TestPassiveTrajectory:
    def test_passive_trajectory_independent_of_outlet(self, clean_model):
        """Trajectory with heating_active=False must not depend on outlet."""
        outdoor = 5.0
        indoor = 21.0
        traj_low = clean_model.predict_thermal_trajectory(
            current_indoor=indoor, target_indoor=indoor,
            outlet_temp=0.0, outdoor_temp=outdoor,
            time_horizon_hours=0.5, time_step_minutes=30,
            heating_active=False,
        )
        traj_high = clean_model.predict_thermal_trajectory(
            current_indoor=indoor, target_indoor=indoor,
            outlet_temp=80.0, outdoor_temp=outdoor,
            time_horizon_hours=0.5, time_step_minutes=30,
            heating_active=False,
        )
        assert traj_low["trajectory"][0] == pytest.approx(
            traj_high["trajectory"][0], abs=1e-9
        ), "Passive trajectory must not depend on outlet_temp"


class TestPassiveGradients:
    def test_oe_gradient_zero_when_all_passive(self, clean_model):
        """When all recent predictions are passive, OE gradient must be ~0."""
        passive_recs = [
            _make_prediction_record(
                predicted=20.0, actual=19.5, outlet_temp=0.0,
                outdoor_temp=5.0, current_indoor=20.0, heating_active=False,
            )
            for _ in range(clean_model.recent_errors_window)
        ]
        oe_grad = clean_model._calculate_outlet_effectiveness_gradient(
            passive_recs
        )
        eq_oe_grad = clean_model._calculate_equilibrium_based_oe_gradient(
            passive_recs
        )
        assert abs(oe_grad) < 1e-6, (
            f"Trajectory OE gradient non-zero for passive: {oe_grad}"
        )
        assert eq_oe_grad == 0.0, (
            f"Equilibrium OE gradient non-zero for passive: {eq_oe_grad}"
        )

    def test_equilibrium_oe_gradient_skips_passive_records(self, clean_model):
        """Equilibrium-based OE gradient must skip passive records entirely."""
        records = [
            _make_prediction_record(
                predicted=21.0, actual=23.0, outlet_temp=45.0,
                outdoor_temp=5.0, current_indoor=21.0,
                heating_active=True, indoor_gradient=0.0,
            ),
            _make_prediction_record(
                predicted=20.0, actual=19.0, outlet_temp=0.0,
                outdoor_temp=5.0, current_indoor=20.0, heating_active=False,
            ),
        ]
        grad = clean_model._calculate_equilibrium_based_oe_gradient(records)
        assert grad != 0.0, (
            "Active record should produce non-zero OE gradient even when "
            "mixed with passive records"
        )

    def test_hlc_gradient_nonzero_for_passive_records(self, clean_model):
        """HLC gradient must be non-zero for passive records with error.

        HLC only has an effect on passive equilibrium when there are external
        heat sources (T_eq = T_out + ext/HLC). With ext=0, T_eq=T_out
        regardless of HLC, so the gradient is zero. We add PV power to
        create a learnable HLC signal.
        """
        passive_recs = [
            _make_prediction_record(
                predicted=19.0, actual=20.0, outlet_temp=0.0,
                outdoor_temp=5.0, current_indoor=20.0, heating_active=False,
                pv_power=2000.0,
            )
            for _ in range(3)
        ]
        hlc_grad = clean_model._calculate_heat_loss_coefficient_gradient(
            passive_recs
        )
        assert hlc_grad != 0.0, (
            "HLC gradient must be non-zero for passive records with error"
        )


class TestTrustGateIsolation:
    def test_passive_prediction_skips_add_prediction(self):
        """model_wrapper must NOT call prediction_metrics.add_prediction
        for passive (heating_active=False) cycles."""
        from src.model_wrapper import EnhancedModelWrapper
        from src.unified_thermal_state import ThermalStateManager

        state_file = str(os.path.join(
            os.path.dirname(__file__), "..", "..",
            "test_passive_trust_gate.json"
        ))
        if os.path.exists(state_file):
            os.remove(state_file)

        state_mgr = ThermalStateManager(state_file=state_file)

        thermal = MagicMock()
        thermal.thermal_time_constant = 4.0
        thermal.heat_loss_coefficient = 1.2
        thermal.outlet_effectiveness = 0.75
        thermal.external_source_weights = {
            "pv": 0.0, "fireplace": 0.0, "tv": 0.0
        }
        thermal._MAX_OUTLET_TEMP = 50.0
        thermal.parameter_history = []
        thermal.learning_confidence = 3.0
        thermal.prediction_history = []
        thermal.recent_errors_window = 10
        thermal.update_prediction_feedback.return_value = 0.5

        pred_metrics = MagicMock()
        pred_metrics.get_metrics.return_value = {
            "24h": {"mae": 0.0, "mean_error": 0.0}
        }
        pred_metrics.get_24h_good_control_percentage.return_value = 100.0

        wrapper = EnhancedModelWrapper.__new__(EnhancedModelWrapper)
        wrapper.state_manager = state_mgr
        wrapper.thermal_model = thermal
        wrapper.prediction_metrics = pred_metrics
        wrapper.cycle_count = 5
        wrapper._demotion_streak = 0
        wrapper.learning_enabled = True

        # --- Passive cycle: heating_active=False ---
        passive_ctx = {
            "outlet_temp": 0.0,
            "outdoor_temp": 5.0,
            "current_indoor": 20.0,
            "heating_active": False,
        }
        wrapper.learn_from_prediction_feedback(
            predicted_temp=20.0, actual_temp=19.5,
            prediction_context=passive_ctx,
        )
        assert not wrapper.prediction_metrics.add_prediction.called, (
            "add_prediction must NOT be called for passive cycles"
        )

        # --- Active cycle: heating_active=True ---
        active_ctx = {
            "outlet_temp": 45.0,
            "outdoor_temp": 5.0,
            "current_indoor": 20.0,
            "heating_active": True,
        }
        wrapper.prediction_metrics.add_prediction.reset_mock()
        wrapper.learn_from_prediction_feedback(
            predicted_temp=20.0, actual_temp=20.5,
            prediction_context=active_ctx,
        )
        assert wrapper.prediction_metrics.add_prediction.called, (
            "add_prediction MUST be called for active (heating-on) cycles"
        )

        if os.path.exists(state_file):
            os.remove(state_file)
