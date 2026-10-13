"""
Tests for the ``sensor.ml_heating_state`` value semantic.

The state value answers: *is the ML model actually commanding the heat pump
this cycle?*

* ``1`` -> active (not shadow mode AND trusted by the self-calibration gate)
* ``0`` -> shadow mode OR active-but-passthrough (learning / recovering)

The ``ml_control_mode`` attribute disambiguates the cases:
``"active"`` | ``"passthrough"`` | ``"shadow"`` | ``"unknown"``.

These tests target the pure decision helper ``determine_ml_heating_state``
that the main control loop calls when writing the sensor. The sensor is now
updated in ALL modes (the previous "skip in shadow" behaviour was removed).
"""
from src.main import determine_ml_heating_state


class TestDetermineMlHeatingState:
    # --- Trust gate available (the normal path) ---------------------------

    def test_active_and_trusted_is_state_1_active(self):
        """ML is driving the heat pump -> state 1, mode 'active'."""
        state, mode = determine_ml_heating_state(
            effective_shadow_mode=False, is_trusted=True
        )
        assert state == 1
        assert mode == "active"

    def test_active_but_not_trusted_is_state_0_passthrough(self):
        """Active mode but trust gate says 'passthrough' -> state 0."""
        state, mode = determine_ml_heating_state(
            effective_shadow_mode=False, is_trusted=False
        )
        assert state == 0
        assert mode == "passthrough"

    def test_shadow_mode_trusted_is_state_0_shadow(self):
        """Even when trusted, shadow mode means ML is NOT driving -> state 0."""
        state, mode = determine_ml_heating_state(
            effective_shadow_mode=True, is_trusted=True
        )
        assert state == 0
        assert mode == "shadow"

    def test_shadow_mode_not_trusted_is_state_0_shadow(self):
        state, mode = determine_ml_heating_state(
            effective_shadow_mode=True, is_trusted=False
        )
        assert state == 0
        assert mode == "shadow"

    # --- Trust gate unavailable (fallback path) ---------------------------

    def test_fallback_shadow_is_state_0_shadow(self):
        """Wrapper missing in shadow mode -> sensor stays live, state 0."""
        state, mode = determine_ml_heating_state(
            effective_shadow_mode=True,
            is_trusted=None,
            confidence=4.0,
            confidence_threshold=2.0,
        )
        assert state == 0
        assert mode == "shadow"

    def test_fallback_active_ok_confidence_is_state_1_unknown(self):
        """Active fallback, confidence above threshold -> ML drives (state 1).

        Higher confidence means the model is trustworthy enough to command
        the outlet, so the sensor reports active (state 1). Values are on the
        production 0.1-5.0 learning_confidence scale.
        """
        state, mode = determine_ml_heating_state(
            effective_shadow_mode=False,
            is_trusted=None,
            confidence=4.0,
            confidence_threshold=2.0,
        )
        assert state == 1
        assert mode == "unknown"

    def test_fallback_active_low_confidence_is_state_0_unknown(self):
        """Active fallback, confidence below threshold -> passthrough (state 0).

        Low confidence means the model is NOT trustworthy enough to command
        the outlet, so the sensor reports passthrough (state 0). Values are on
        the production 0.1-5.0 learning_confidence scale.
        """
        state, mode = determine_ml_heating_state(
            effective_shadow_mode=False,
            is_trusted=None,
            confidence=1.0,
            confidence_threshold=2.0,
        )
        assert state == 0
        assert mode == "unknown"
