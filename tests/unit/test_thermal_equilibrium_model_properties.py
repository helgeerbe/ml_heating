
import pytest
from hypothesis import given, strategies as st
from src.thermal_equilibrium_model import ThermalEquilibriumModel


class TestThermalEquilibriumModelProperties:

    @pytest.fixture(scope="class")
    def model(self):
        return ThermalEquilibriumModel()

    @given(
        outdoor_temp=st.floats(min_value=-30.0, max_value=40.0),
        outlet_temp=st.floats(min_value=20.0, max_value=80.0),
        indoor_temp=st.floats(min_value=10.0, max_value=35.0)
    )
    def test_equilibrium_temperature_bounds(
        self, model, outdoor_temp, outlet_temp, indoor_temp
    ):
        """
        Property: Predicted equilibrium temperature should be reasonable.

        It should generally be between the outdoor temperature and the outlet
        temperature (assuming heating mode and no extreme internal gains).
        """
        # We mock the internal state or ensure it's initialized
        # The model might need parameters loaded.
        # Assuming default parameters are reasonable.

        equilibrium = model.predict_equilibrium_temperature(
            outdoor_temp=outdoor_temp,
            outlet_temp=outlet_temp,
            current_indoor=indoor_temp
        )

        # Basic physical sanity check
        # Equilibrium should be roughly bounded by outdoor and outlet temps
        # We allow a margin for internal gains (making it hotter) or losses
        min_bound = min(outdoor_temp, outlet_temp) - 5.0
        # +10 for internal gains
        max_bound = max(outdoor_temp, outlet_temp) + 10.0
    
        assert equilibrium > min_bound
        assert equilibrium < max_bound

        # It shouldn't be infinitely hot
        assert equilibrium < 100.0

    @given(
        outdoor_temp=st.floats(min_value=-20.0, max_value=15.0),
        target_temp=st.floats(min_value=18.0, max_value=25.0)
    )
    def test_optimal_outlet_monotonicity(
        self, model, outdoor_temp, target_temp
    ):
        """
        Property: Outdoor temp decreases -> optimal outlet temp increases.

        (or stay same if at max) to maintain the same target.
        """
        t1 = model.calculate_optimal_outlet_temperature(
            target_indoor=target_temp,
            current_indoor=target_temp - 1.0,
            outdoor_temp=outdoor_temp
        )["optimal_outlet_temp"]
    
        t2 = model.calculate_optimal_outlet_temperature(
            target_indoor=target_temp,
            current_indoor=target_temp - 1.0,
            outdoor_temp=outdoor_temp - 1.0  # Colder
        )["optimal_outlet_temp"]
    
        assert t2 >= t1

    @given(
        outdoor_temp=st.floats(min_value=-10.0, max_value=10.0),
        target_temp=st.floats(min_value=18.0, max_value=25.0)
    )
    def test_optimal_outlet_target_monotonicity(
        self, model, outdoor_temp, target_temp
    ):
        """
        Property: Target indoor increases -> optimal outlet increases.
        """
        t1 = model.calculate_optimal_outlet_temperature(
            target_indoor=target_temp,
            current_indoor=target_temp - 2.0,
            outdoor_temp=outdoor_temp
        )["optimal_outlet_temp"]
    
        t2 = model.calculate_optimal_outlet_temperature(
            target_indoor=target_temp + 1.0,  # Warmer target
            current_indoor=target_temp - 2.0,
            outdoor_temp=outdoor_temp
        )["optimal_outlet_temp"]
    
        assert t2 >= t1

    @given(
        outdoor_temp=st.floats(min_value=-30.0, max_value=40.0),
        outlet_temp=st.floats(min_value=20.0, max_value=80.0)
    )
    def test_prediction_consistency(self, model, outdoor_temp, outlet_temp):
        """
        Property: Predicting equilibrium twice yields same result.
        """
        e1 = model.predict_equilibrium_temperature(
            outdoor_temp=outdoor_temp,
            outlet_temp=outlet_temp,
            current_indoor=20.0
        )
        e2 = model.predict_equilibrium_temperature(
            outdoor_temp=outdoor_temp,
            outlet_temp=outlet_temp,
            current_indoor=20.0
        )
        assert e1 == e2
