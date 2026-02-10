
import pytest
from src.multi_heat_source_physics import MultiHeatSourcePhysics


@pytest.fixture
def physics():
    return MultiHeatSourcePhysics()


def test_calculate_pv_heat_contribution(physics):
    """Test PV heat contribution calculation."""
    result = physics.calculate_pv_heat_contribution(1000, 20, 10)
    assert result["heat_contribution_kw"] > 0
    assert result["outlet_temp_reduction"] > 0


def test_calculate_fireplace_heat_contribution(physics):
    """Test fireplace heat contribution calculation."""
    result = physics.calculate_fireplace_heat_contribution(True)
    assert result["heat_contribution_kw"] > 0
    assert result["outlet_temp_reduction"] > 0


def test_calculate_electronics_occupancy_heat(physics):
    """Test electronics and occupancy heat calculation."""
    result = physics.calculate_electronics_occupancy_heat(True)
    assert result["heat_contribution_kw"] > 0
    assert result["outlet_temp_reduction"] > 0


def test_calculate_system_state_impacts(physics):
    """Test system state impacts calculation."""
    result = physics.calculate_system_state_impacts(dhw_heating=True, defrosting=True)
    assert result["capacity_reduction_percent"] > 0
    assert result["net_outlet_adjustment"] > 0


def test_calculate_combined_heat_sources(physics):
    """Test combined heat sources calculation."""
    result = physics.calculate_combined_heat_sources(pv_power=1000, fireplace_on=True, tv_on=True)
    assert result["total_heat_contribution_kw"] > 0
    assert result["total_outlet_temp_reduction"] > 0


def test_calculate_optimized_outlet_temperature(physics):
    """Test optimized outlet temperature calculation."""
    heat_source_analysis = physics.calculate_combined_heat_sources(pv_power=1000)
    result = physics.calculate_optimized_outlet_temperature(45.0, heat_source_analysis)
    assert result["optimized_outlet_temp"] < 45.0

