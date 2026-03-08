import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from src.thermal_equilibrium_model import ThermalEquilibriumModel
from src.thermal_config import ThermalParameterConfig
from src.thermal_constants import PhysicsConstants

class TestPhysicsConstraints:
    """
    Regression tests for physics constraints:
    1. Heat loss coefficient (U) < 1.2
    2. Thermal time constant up to 100h
    3. Solar lag behavior
    """

    @pytest.fixture
    def model(self):
        """Create a fresh model instance for each test."""
        # Patching _load_thermal_parameters to avoid side effects during init
        with patch.object(ThermalEquilibriumModel, '_load_thermal_parameters'):
            model = ThermalEquilibriumModel()
            # Manually initialize defaults usually done in _load_thermal_parameters
            model.thermal_time_constant = 4.0
            model.heat_loss_coefficient = 0.4
            model.outlet_effectiveness = 0.5
            model.solar_lag_minutes = 45.0
            model.external_source_weights = {
                'pv': 0.002,
                'fireplace': 5.0,
                'tv': 0.2
            }
            model._initialize_learning_attributes()
            return model

    def test_heat_loss_coefficient_clamping_on_load(self):
        """
        Test that heat_loss_coefficient is clamped to the maximum bound (1.2)
        when loading parameters, even if the saved state has a higher value.
        """
        # Create a real model but mock the state manager to return a high U value
        with patch('src.unified_thermal_state.get_thermal_state_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_get_manager.return_value = mock_manager
            
            # Setup a state with U = 1.5 (way above 1.2 limit)
            mock_manager.load_state.return_value = True
            mock_manager.get_current_parameters.return_value = {
                "baseline_parameters": {
                    "source": "calibrated",
                    "thermal_time_constant": 20.0,
                    "heat_loss_coefficient": 1.5,  # Too high!
                    "outlet_effectiveness": 0.5,
                    "solar_lag_minutes": 45.0,
                    "pv_heat_weight": 0.002,
                    "fireplace_heat_weight": 5.0,
                    "tv_heat_weight": 0.2
                },
                "learning_state": {
                    "parameter_adjustments": {},
                    "learning_confidence": 3.0
                }
            }
            
            # Initialize model - this calls _load_thermal_parameters
            model = ThermalEquilibriumModel()
            
            # Check bounds from config to be sure what we expect
            bounds = ThermalParameterConfig.get_bounds("heat_loss_coefficient")
            max_u = bounds[1]
            
            # Assert U was clamped
            assert model.heat_loss_coefficient == max_u
            assert model.heat_loss_coefficient < 1.5
            assert model.heat_loss_coefficient == 1.2  # Explicit check for the requirement

    def test_thermal_time_constant_extended_range(self):
        """
        Test that thermal_time_constant allows values up to 100h.
        """
        # Check config bounds first
        bounds = ThermalParameterConfig.get_bounds("thermal_time_constant")
        assert bounds[1] >= 100.0
        
        # Create model
        with patch('src.unified_thermal_state.get_thermal_state_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_get_manager.return_value = mock_manager
            
            # Case 1: Valid high value (95h)
            mock_manager.get_current_parameters.return_value = {
                "baseline_parameters": {
                    "source": "calibrated",
                    "thermal_time_constant": 95.0,  # High but valid
                    "heat_loss_coefficient": 0.4,
                    "outlet_effectiveness": 0.5
                },
                "learning_state": {}
            }
            
            model = ThermalEquilibriumModel()
            assert model.thermal_time_constant == 95.0
            
            # Case 2: Invalid high value (150h)
            mock_manager.get_current_parameters.return_value = {
                "baseline_parameters": {
                    "source": "calibrated",
                    "thermal_time_constant": 150.0,  # Too high
                    "heat_loss_coefficient": 0.4,
                    "outlet_effectiveness": 0.5
                },
                "learning_state": {}
            }
            
            model = ThermalEquilibriumModel()
            assert model.thermal_time_constant == bounds[1]  # Should be clamped to max (100.0)

    def test_solar_lag_smoothing_logic(self, model):
        """
        Test that _calculate_effective_solar correctly smooths PV input
        based on solar_lag_minutes.
        """
        # Setup: 60 minute lag
        model.solar_lag_minutes = 60.0
        
        # Case 1: Instantaneous scalar input (fallback)
        assert model._calculate_effective_solar(1000.0) == 1000.0
        
        # Case 2: History buffer
        # Create a buffer representing 60 minutes of data.
        # Assuming HISTORY_STEP_MINUTES is 10 (standard config)
        # 60 mins = 6 steps.
        
        # Scenario A: Constant full sun
        pv_history_full = [1000.0] * 10
        # Average of last 60 mins (6 steps) should be 1000
        effective_solar = model._calculate_effective_solar(pv_history_full)
        assert effective_solar == 1000.0
        
        # Scenario B: Sun just started (Step function)
        # History: [0, 0, 0, 0, 0, 1000] (Most recent is last)
        # If lag is 60 mins (6 steps), we average the last 6 values.
        # If we only have 6 values in history:
        pv_history_step = [0.0] * 5 + [1000.0]
        effective_solar = model._calculate_effective_solar(pv_history_step)
        
        # Expected: (0*5 + 1000*1) / 6 = 166.66...
        expected = 1000.0 / 6.0
        assert np.isclose(effective_solar, expected, rtol=0.01)
        
        # Scenario C: Sun turned off 30 mins ago
        # History: [1000, 1000, 1000, 0, 0, 0]
        # Average of last 6 steps: (1000*3 + 0*3) / 6 = 500
        pv_history_drop = [1000.0] * 3 + [0.0] * 3
        effective_solar = model._calculate_effective_solar(pv_history_drop)
        assert np.isclose(effective_solar, 500.0, rtol=0.01)

    def test_solar_lag_fractional_steps(self, model):
        """
        Test solar lag with fractional steps (e.g. 45 mins with 10 min steps).
        """
        # 45 mins lag, 10 min steps => 4.5 steps
        model.solar_lag_minutes = 45.0
        
        # History: [1000, 1000, 1000, 1000, 1000] (5 steps of 1000)
        # Should be 1000
        pv_history = [1000.0] * 10
        assert model._calculate_effective_solar(pv_history) == 1000.0
        
        # History: [0, 1000, 1000, 1000, 1000] (Last 4 are 1000, 5th back is 0)
        # Calculation:
        # Full steps: 4 (most recent) -> 1000, 1000, 1000, 1000
        # Fractional step: 0.5 * (value at index -5)
        # Value at index -5 is 0.
        # Total = 4000 + 0.5*0 = 4000
        # Divisor = 4.5
        # Result = 4000 / 4.5 = 888.88
        
        pv_history_fractional = [0.0] + [1000.0] * 4
        effective_solar = model._calculate_effective_solar(pv_history_fractional)
        expected = 4000.0 / 4.5
        assert np.isclose(effective_solar, expected, rtol=0.01)

    def test_heat_loss_coefficient_learning_constraints(self, model):
        """
        Test that adaptive learning respects the 1.2 limit for heat_loss_coefficient.
        """
        # Set U to near the limit
        model.heat_loss_coefficient = 1.19
        
        # Mock prediction history to suggest U should increase (positive error)
        # If actual > predicted, it means we are losing less heat than thought (or gaining more).
        # Wait, if actual > predicted, the house is warmer than expected.
        # This implies U might be too high (losing too much in model) -> U should decrease.
        # Or effective gain is higher.
        
        # Let's look at the gradient calculation in ThermalEquilibriumModel._calculate_parameter_gradient
        # gradient = -finite_diff * error
        # If we want U to increase, we need a negative gradient (since param = param - lr * grad).
        # So we need grad > 0 for U to decrease, grad < 0 for U to increase.
        
        # Let's just manually call _adapt_parameters_from_recent_errors with mocked gradients
        # to ensure the clipping logic works.
        
        model.prediction_history = [{"error": 1.0, "context": {}, "timestamp": "now"}] * 10
        model.recent_errors_window = 5
        
        # Mock the gradient calculation methods
        with patch.object(model, '_calculate_heat_loss_coefficient_gradient') as mock_grad:
            # We want U to increase.
            # update = lr * grad
            # new_val = old_val - update
            # To increase new_val, update must be negative.
            # So grad must be negative (since lr is positive).
            mock_grad.return_value = -10.0 # Strong negative gradient
            
            # Mock other gradients to 0
            with patch.object(model, '_calculate_thermal_time_constant_gradient', return_value=0), \
                 patch.object(model, '_calculate_outlet_effectiveness_gradient', return_value=0), \
                 patch.object(model, '_calculate_pv_heat_weight_gradient', return_value=0), \
                 patch.object(model, '_calculate_tv_heat_weight_gradient', return_value=0), \
                 patch.object(model, '_calculate_solar_lag_gradient', return_value=0):
                
                model._adapt_parameters_from_recent_errors()
                
                # It should have tried to increase U, but clamped at 1.2
                assert model.heat_loss_coefficient <= 1.2
                # It should have increased from 1.19, likely hitting 1.2
                # (Assuming learning rate allows it to reach 1.2)
                assert model.heat_loss_coefficient >= 1.19
