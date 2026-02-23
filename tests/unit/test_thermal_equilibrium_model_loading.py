import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src import thermal_equilibrium_model  # noqa: E402
from src.thermal_config import ThermalParameterConfig  # noqa: E402


class TestThermalEquilibriumModelLoading:
    """Tests for loading thermal parameters and handling validation failures."""

    @pytest.fixture
    def mock_state_manager(self):
        with patch('src.thermal_equilibrium_model.get_thermal_state_manager') as mock:
            yield mock

    @pytest.fixture
    def mock_validator(self):
        # We need to patch where it is imported in the module under test
        # Since it's imported inside the method, we might need to patch the module itself
        # or use patch.dict on sys.modules if it was already imported.
        # However, since we are in a test, we can patch the function in the validator module
        # and ensure it's used.
        with patch('src.thermal_state_validator.validate_thermal_state_safely') as mock:
            yield mock

    def test_load_parameters_validation_failure_recovery(self):
        """
        Test that the model retains calibrated parameters even if schema validation fails,
        provided the core parameters are present and valid.
        """
        # 1. Setup Mock State
        calibrated_hlc = 0.55
        calibrated_state = {
            "baseline_parameters": {
                "thermal_time_constant": 20.0,
                "heat_loss_coefficient": calibrated_hlc,
                "outlet_effectiveness": 0.4,
                "source": "calibrated",
                "pv_heat_weight": 0.0,
                "fireplace_heat_weight": 0.0,
                "tv_heat_weight": 0.0
            },
            "learning_state": {}
        }

        # 2. Setup Mocks
        # We need to mock get_thermal_state_manager to return our state
        mock_manager_instance = MagicMock()
        mock_manager_instance.get_current_parameters.return_value = calibrated_state

        # We use patch.dict to ensure we catch the import inside the function if needed,
        # or just patch the target module.
        # Since thermal_equilibrium_model does 'from .unified_thermal_state import ...',
        # we should patch src.unified_thermal_state.get_thermal_state_manager if running as package.

        with patch('src.unified_thermal_state.get_thermal_state_manager',
                   return_value=mock_manager_instance), \
             patch('src.thermal_state_validator.validate_thermal_state_safely',
                   return_value=False):

            # 3. Instantiate Model
            model = thermal_equilibrium_model.ThermalEquilibriumModel()

            # 4. Assertions
            # The model should have loaded the calibrated HLC
            assert model.heat_loss_coefficient == calibrated_hlc, \
                f"Model should retain calibrated HLC {calibrated_hlc}, got {model.heat_loss_coefficient}"

            # Verify it didn't fall back to default
            default_hlc = ThermalParameterConfig.get_default("heat_loss_coefficient")
            assert model.heat_loss_coefficient != default_hlc, \
                "Model should not have fallen back to default parameters"

    def test_load_parameters_missing_keys_fallback(self):
        """
        Test that the model DOES fall back to defaults if core parameters are missing,
        triggering a KeyError during loading.
        """
        # 1. Setup Mock State with MISSING heat_loss_coefficient
        broken_state = {
            "baseline_parameters": {
                "thermal_time_constant": 20.0,
                # "heat_loss_coefficient": MISSING
                "outlet_effectiveness": 0.4,
                "source": "calibrated"
            },
            "learning_state": {}
        }

        mock_manager_instance = MagicMock()
        mock_manager_instance.get_current_parameters.return_value = broken_state

        with patch('src.unified_thermal_state.get_thermal_state_manager',
                   return_value=mock_manager_instance), \
             patch('src.thermal_state_validator.validate_thermal_state_safely',
                   return_value=False):

            # 3. Instantiate Model
            model = thermal_equilibrium_model.ThermalEquilibriumModel()

            # 4. Assertions
            # Should have fallen back to defaults because of KeyError
            default_hlc = ThermalParameterConfig.get_default("heat_loss_coefficient")
            assert model.heat_loss_coefficient == default_hlc, \
                "Model should have fallen back to defaults due to missing key"
