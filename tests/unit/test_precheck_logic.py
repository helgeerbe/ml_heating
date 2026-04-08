import pytest
from unittest.mock import MagicMock
from src.model_wrapper import EnhancedModelWrapper

@pytest.fixture
def wrapper_instance():
    wrapper = EnhancedModelWrapper()
    wrapper.thermal_model = MagicMock()
    return wrapper

def test_precheck_ignores_unreachable_target_if_cooling_needed(wrapper_instance):
    # Setup: Room is 22.4C, Target is 21.2C. We need to cool.
    # But equilibrium model says even with max heat (65C), we only reach 21.08C.
    # So target (21.2C) > max_prediction (21.08C).
    # The old logic would short-circuit and return 65C.
    # The new logic should realize we need to cool, so it shouldn't apply max heat.
    
    wrapper_instance.thermal_model.predict_equilibrium_temperature.side_effect = [
        15.0,  # min_prediction
        21.08  # max_prediction
    ]
    
    # Mock binary search to just return a dummy value so we know it reached it
    wrapper_instance.thermal_model.predict_thermal_trajectory.return_value = {
        "final_temp": 21.2,
        "trajectory": []
    }
    
    # Call the method
    result = wrapper_instance._calculate_required_outlet_temp(
        target_indoor=21.2,
        current_indoor=22.4,
        outdoor_temp=0.0,
        thermal_features={"pv_power": 0.0}
    )
    
    # It should NOT return 65.0 (outlet_max). It should proceed to binary search.
    # Since we mocked binary search to return a dummy trajectory, it will return the midpoint of the last iteration.
    assert result != 65.0
    assert wrapper_instance.thermal_model.predict_thermal_trajectory.called

def test_precheck_ignores_unreachable_target_if_heating_needed(wrapper_instance):
    # Setup: Room is 18.0C, Target is 21.0C. We need to heat.
    # But equilibrium model says even with min heat (25C), we reach 22.0C.
    # So target (21.0C) < min_prediction (22.0C).
    # The old logic would short-circuit and return 25C.
    # The new logic should realize we need to heat, so it shouldn't apply min heat.
    
    wrapper_instance.thermal_model.predict_equilibrium_temperature.side_effect = [
        22.0,  # min_prediction
        30.0   # max_prediction
    ]
    
    wrapper_instance.thermal_model.predict_thermal_trajectory.return_value = {
        "final_temp": 21.0,
        "trajectory": []
    }
    
    result = wrapper_instance._calculate_required_outlet_temp(
        target_indoor=21.0,
        current_indoor=18.0,
        outdoor_temp=0.0,
        thermal_features={"pv_power": 0.0}
    )
    
    assert result != 25.0
    assert wrapper_instance.thermal_model.predict_thermal_trajectory.called
