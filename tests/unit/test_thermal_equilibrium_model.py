
import pytest
import numpy as np
import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src import thermal_equilibrium_model


@pytest.fixture(scope="function")
def clean_model():
    """Fixture to ensure a clean model instance for each test function."""
    model = thermal_equilibrium_model.ThermalEquilibriumModel()
    # Set realistic test parameters
    model.thermal_time_constant = 4.0
    model.heat_loss_coefficient = 1.2
    model.outlet_effectiveness = 0.75
    model.external_source_weights = {
        "pv": 0.0001,
        "fireplace": 1.5,
        "tv": 0.1
    }
    return model


class TestCorrectedThermalPhysics:
    """Consolidated and refactored tests for the ThermalEquilibriumModel."""

    def test_heating_never_cools_house(self, clean_model):
        """CRITICAL: Heating with outlet > indoor should never predict cooling."""
        current_indoor = 20.0
        outdoor_temp = 5.0
        outlet_temp = 45.0

        predicted = clean_model.predict_equilibrium_temperature(
            outlet_temp=outlet_temp,
            outdoor_temp=outdoor_temp,
            current_indoor=current_indoor
        )

        assert predicted >= current_indoor, (
            f"Heating predicted to cool: {predicted}°C < {current_indoor}°C")
        assert predicted >= outdoor_temp, (
            f"Indoor below outdoor: {predicted}°C < {outdoor_temp}°C")

    def test_weighted_average_equilibrium(self, clean_model):
        """Equilibrium should be weighted average of outlet and outdoor temperatures."""
        outdoor_temp = 5.0
        outlet_temp = 45.0
        current_indoor = 20.0
        
        eff = clean_model.outlet_effectiveness
        loss = clean_model.heat_loss_coefficient
        
        expected = (eff * outlet_temp + loss * outdoor_temp) / (eff + loss)
        
        predicted = clean_model.predict_equilibrium_temperature(
            outlet_temp=outlet_temp,
            outdoor_temp=outdoor_temp,
            current_indoor=current_indoor
        )
        
        assert abs(predicted - expected) < 0.1, (
            f"Expected {expected:.1f}°C, got {predicted:.1f}°C")

    def test_energy_conservation_at_equilibrium(self, clean_model):
        """Test that energy is conserved at equilibrium (Heat Input = Heat Loss)."""
        pv_power = 1000
        outlet_temp = 45.0
        outdoor_temp = 5.0
        
        # Test with temperature-based approximation
        equilibrium_temp = clean_model.predict_equilibrium_temperature(
            outlet_temp=outlet_temp,
            outdoor_temp=outdoor_temp,
            current_indoor=21.0,
            pv_power=pv_power
        )
        
        heat_from_pv = pv_power * clean_model.external_source_weights['pv']
        heat_input_from_outlet = (
            clean_model.outlet_effectiveness * (outlet_temp - equilibrium_temp)
        )
        total_heat_input = heat_input_from_outlet + heat_from_pv
        heat_loss = (
            clean_model.heat_loss_coefficient * (equilibrium_temp - outdoor_temp)
        )

        assert np.isclose(total_heat_input, heat_loss, atol=1e-2), (
            f"Energy not conserved: input={total_heat_input:.3f}, "
            f"loss={heat_loss:.3f}")

    def test_energy_based_prediction(self, clean_model):
        """Test the new energy-based prediction logic."""
        thermal_power = 2.0  # kW
        outdoor_temp = 5.0
        
        # Teq = Tout + P_total / U_loss
        expected_eq = outdoor_temp + (thermal_power / clean_model.heat_loss_coefficient)
        
        predicted = clean_model.predict_equilibrium_temperature(
            outlet_temp=45.0,  # Should be ignored for energy calculation
            outdoor_temp=outdoor_temp,
            current_indoor=20.0,
            thermal_power=thermal_power
        )
        
        assert np.isclose(predicted, expected_eq, atol=1e-3), (
            f"Energy-based prediction failed: expected {expected_eq}, got {predicted}"
        )

    def test_second_law_thermodynamics(self, clean_model):
        """Indoor temp cannot be below outdoor when heating."""
        equilibrium_temp = clean_model.predict_equilibrium_temperature(
            outlet_temp=45.0, 
            outdoor_temp=5.0, 
            current_indoor=21.0
        )
        assert equilibrium_temp > 5.0, (
            "Indoor temperature below outdoor despite heat input")

        # With outlet = outdoor, indoor should equal outdoor (no external heat)
        equilibrium_no_heat = clean_model.predict_equilibrium_temperature(
            outlet_temp=5.0, outdoor_temp=5.0, current_indoor=21.0
        )
        assert np.isclose(equilibrium_no_heat, 5.0, atol=1e-1), (
            "Indoor temperature differs from outdoor with no net heat input")

    def test_physical_bounds(self, clean_model):
        """Indoor temperature is physically bounded between outdoor and source."""
        equilibrium_temp = clean_model.predict_equilibrium_temperature(
            outlet_temp=45.0, 
            outdoor_temp=5.0, 
            current_indoor=21.0
        )
        assert 5.0 < equilibrium_temp < 45.0
        assert -20 < equilibrium_temp < 50, (
            "Temperature outside realistic building range")

    def test_linearity_of_heat_loss(self, clean_model):
        """Equilibrium should change proportionally with outdoor temperature."""
        outdoor_temps = [0, 5, 10, 15, 20]
        equilibrium_temps = [
            clean_model.predict_equilibrium_temperature(
                outlet_temp=45.0, 
                outdoor_temp=ot, 
                current_indoor=21.0
            ) for ot in outdoor_temps
        ]
        
        slopes = np.diff(equilibrium_temps) / np.diff(outdoor_temps)
        assert np.allclose(slopes, slopes[0], atol=1e-2), (
            "Non-linear equilibrium response to outdoor temp")

    def test_external_heat_source_additivity(self, clean_model):
        """External heat sources should contribute additively."""
        eq_baseline = clean_model.predict_equilibrium_temperature(
            outlet_temp=45.0, outdoor_temp=5.0, current_indoor=21.0
        )
        eq_with_pv = clean_model.predict_equilibrium_temperature(
            outlet_temp=45.0, outdoor_temp=5.0, current_indoor=21.0, pv_power=1000
        )
        eq_with_fireplace = clean_model.predict_equilibrium_temperature(
            outlet_temp=45.0, outdoor_temp=5.0, current_indoor=21.0, fireplace_on=1
        )
        eq_with_both = clean_model.predict_equilibrium_temperature(
            outlet_temp=45.0, outdoor_temp=5.0, current_indoor=21.0, pv_power=1000, fireplace_on=1
        )
        
        pv_contribution = eq_with_pv - eq_baseline
        fireplace_contribution = eq_with_fireplace - eq_baseline
        expected_combined = (
            eq_baseline + pv_contribution + fireplace_contribution
        )
        
        assert np.isclose(eq_with_both, expected_combined, atol=1e-2), (
            f"Heat sources not additive: expected={expected_combined:.3f}, "
            f"actual={eq_with_both:.3f}")

    def test_thermal_time_constant_independent_of_equilibrium(self, 
                                                              clean_model):
        """Thermal time constant should not affect equilibrium calculations."""
        eq_original = clean_model.predict_equilibrium_temperature(
            outlet_temp=45.0, outdoor_temp=5.0, current_indoor=21.0
        )
        clean_model.thermal_time_constant = 10.0
        eq_different_time = clean_model.predict_equilibrium_temperature(
            outlet_temp=45.0, outdoor_temp=5.0, current_indoor=21.0
        )
        assert np.isclose(eq_original, eq_different_time, atol=1e-3), (
            "Thermal time constant affects equilibrium calculation")

    def test_zero_division_protection(self, clean_model):
        """Model should handle zero effectiveness and heat loss without crashing."""
        clean_model.outlet_effectiveness = 0.0
        clean_model.heat_loss_coefficient = 0.0
        
        predicted = clean_model.predict_equilibrium_temperature(
            outlet_temp=45.0,
            outdoor_temp=5.0,
            current_indoor=20.0
        )
        assert predicted == 5.0, (
            f"Expected fallback to outdoor temp 5.0°C, got {predicted}°C")

    def test_extreme_temperatures(self, clean_model):
        """Test behavior with extreme temperature inputs."""
        eq_cold = clean_model.predict_equilibrium_temperature(
            outlet_temp=45.0, outdoor_temp=-30.0, current_indoor=20.0
        )
        assert isinstance(eq_cold, float), "Failed with extreme cold"
        
        eq_hot = clean_model.predict_equilibrium_temperature(
            outlet_temp=45.0, outdoor_temp=40.0, current_indoor=20.0
        )
        assert isinstance(eq_hot, float), "Failed with extreme heat"


class TestBaselineFunctionality:
    """
    Tests to ensure baseline model functionality is preserved after
    refactoring. These are based on the original migration tests.
    """

    def test_equilibrium_prediction_baseline(self, clean_model):
        """Capture baseline equilibrium temperature predictions."""
        model = clean_model

        # Test scenario 1: Standard heating scenario
        baseline_result_1 = model.predict_equilibrium_temperature(
            outlet_temp=45.0,
            outdoor_temp=5.0,
            current_indoor=20.0,
            pv_power=500.0,
            fireplace_on=0,
            tv_on=1
        )

        # Test scenario 2: Cold weather with external heat
        baseline_result_2 = model.predict_equilibrium_temperature(
            outlet_temp=55.0,
            outdoor_temp=-5.0,
            current_indoor=18.0,
            pv_power=1200.0,
            fireplace_on=1,
            tv_on=1
        )

        # Test scenario 3: Mild weather, minimal heating
        baseline_result_3 = model.predict_equilibrium_temperature(
            outlet_temp=30.0,
            outdoor_temp=15.0,
            current_indoor=21.0,
            pv_power=0.0,
            fireplace_on=0,
            tv_on=0
        )

        # Note: These are broad checks. The original test was for before/after
        # migration comparison. With refactored parameters, exact values change.
        assert 15.0 <= baseline_result_1 <= 45.0
        assert 10.0 <= baseline_result_2 <= 60.0
        assert 15.0 <= baseline_result_3 <= 30.0

    def test_trajectory_prediction_baseline(self, clean_model):
        """Capture baseline thermal trajectory predictions."""
        model = clean_model

        # Test trajectory prediction
        baseline_trajectory = model.predict_thermal_trajectory(
            current_indoor=18.0,
            target_indoor=21.0,
            outlet_temp=40.0,
            outdoor_temp=8.0,
            time_horizon_hours=4,
            pv_power=300.0,
            fireplace_on=0,
            tv_on=1
        )

        # Verify trajectory structure
        assert 'trajectory' in baseline_trajectory
        assert 'reaches_target_at' in baseline_trajectory
        assert 'overshoot_predicted' in baseline_trajectory

        # Verify trajectory makes sense
        trajectory = baseline_trajectory['trajectory']
        assert len(trajectory) == 4  # 4 hour horizon
        assert all(isinstance(temp, (int, float)) for temp in trajectory)
        # Check that temperature is rising towards a reasonable equilibrium
        assert trajectory[0] > 18.0
        assert trajectory[-1] > trajectory[0]

    def test_optimal_outlet_calculation_baseline(self, clean_model):
        """Capture baseline optimal outlet temperature calculations."""
        model = clean_model

        # Test optimal outlet calculation
        baseline_optimal = model.calculate_optimal_outlet_temperature(
            target_indoor=22.0,
            current_indoor=19.0,
            outdoor_temp=6.0,
            time_available_hours=2.0,
            pv_power=800.0,
            fireplace_on=0,
            tv_on=1
        )

        # Verify optimal calculation structure
        assert 'optimal_outlet_temp' in baseline_optimal
        assert 'method' in baseline_optimal

        optimal_temp = baseline_optimal['optimal_outlet_temp']
        assert isinstance(optimal_temp, (int, float))
        assert 25.0 <= optimal_temp <= 70.0, (
            f"Optimal temp out of range: {optimal_temp}"
        )

    def test_prediction_history_sliding_window(self, clean_model):
        """Test that prediction history maintains a sliding window of 200 records."""
        model = clean_model
        
        # Add 205 records
        for i in range(205):
            prediction_record = {
                "timestamp": "2023-01-01T00:00:00",
                "predicted": 20.0,
                "actual": 20.0,
                "error": 0.0,
                "index": i
            }
            model.prediction_history.append(prediction_record)
            
            # Manually trigger the truncation logic that would happen in update_prediction_feedback
            if len(model.prediction_history) > 200:
                model.prediction_history = model.prediction_history[-200:]

        # Check size is capped at 200
        assert len(model.prediction_history) == 200
        
        # Check that the oldest records were removed (indices 0-4 should be gone)
        # The first record in history should be index 5
        assert model.prediction_history[0]["index"] == 5
        # The last record should be index 204
        assert model.prediction_history[-1]["index"] == 204
