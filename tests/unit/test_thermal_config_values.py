from src.thermal_config import ThermalParameterConfig


class TestThermalConfigValues:
    """Test the actual configuration values in ThermalParameterConfig."""

    def test_heat_loss_coefficient_bounds(self):
        """Verify heat_loss_coefficient bounds are tightened to (0.01, 1.5)."""
        bounds = ThermalParameterConfig.get_bounds('heat_loss_coefficient')
        assert bounds == (0.01, 1.5), \
            "Heat loss coefficient bounds should be (0.01, 1.5)"

    def test_all_parameters_have_bounds(self):
        """Ensure all default parameters have corresponding bounds."""
        defaults = ThermalParameterConfig.get_all_defaults()
        bounds = ThermalParameterConfig.get_all_bounds()
        assert set(defaults.keys()) == set(bounds.keys()), \
            "Mismatch between parameters with defaults and bounds"

    def test_defaults_within_bounds(self):
        """Ensure all default values are within their defined bounds."""
        defaults = ThermalParameterConfig.get_all_defaults()
        for param, value in defaults.items():
            min_val, max_val = ThermalParameterConfig.get_bounds(param)
            assert min_val <= value <= max_val, \
                f"{param} default {value} not in bounds {min_val, max_val}"

    def test_critical_parameter_defaults(self):
        """Verify critical parameter defaults are set to expected values."""
        # These values are critical for the baseline physics model
        assert ThermalParameterConfig.get_default(
            'thermal_time_constant') == 4.0
        assert ThermalParameterConfig.get_default(
            'heat_loss_coefficient') == 0.4
        assert ThermalParameterConfig.get_default('equilibrium_ratio') == 0.17
