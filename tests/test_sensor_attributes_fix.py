"""
Test for updated sensor attributes including all thermal parameters.

This test verifies that the sensor.ml_heating_learning sensor now exports
all thermal parameters currently used by the system, including the previously
missing total_conductance and equilibrium_ratio parameters.
"""
from unittest.mock import Mock, ANY
from src.ha_client import HAClient


class TestSensorAttributesFix:
    """Test updated sensor attributes include all thermal parameters."""
    
    def test_ml_heating_learning_includes_all_thermal_parameters(self):
        """Test that sensor.ml_heating_learning includes all 5 thermal parameters."""
        # Create mock HA client
        ha_client = HAClient("http://test:8123", "fake_token")
        ha_client.set_state = Mock()
        
        # Mock learning metrics with all thermal parameters
        learning_metrics = {
            "thermal_time_constant": 4.0,
            "heat_loss_coefficient": 0.1,
            "outlet_effectiveness": 0.8,
            "total_conductance": 0.1947,  # Previously missing
            "equilibrium_ratio": 0.406,   # Previously missing
            "learning_confidence": 3.0,
            "cycle_count": 5,
            "parameter_updates": 2,
            "model_health": "learning",
            "is_improving": True,
            "improvement_percentage": 5.2,
            "total_predictions": 150
        }
        
        # Call the function
        ha_client.log_adaptive_learning_metrics(learning_metrics)
        
        # Verify set_state was called for ml_heating_learning sensor
        ha_client.set_state.assert_any_call(
            "sensor.ml_heating_learning",
            3.0,  # learning_confidence
            ANY,  # attributes
            round_digits=3
        )
        
        # Extract the attributes that were passed
        call_args = [call for call in ha_client.set_state.call_args_list 
                    if call[0][0] == "sensor.ml_heating_learning"][0]
        attributes = call_args[0][2]
        
        # Verify all thermal parameters are included
        assert "thermal_time_constant" in attributes
        assert attributes["thermal_time_constant"] == 4.0
        
        assert "heat_loss_coefficient" in attributes  
        assert attributes["heat_loss_coefficient"] == 0.1
        
        assert "outlet_effectiveness" in attributes
        assert attributes["outlet_effectiveness"] == 0.8
        
        # Verify previously missing parameters are now included
        assert "total_conductance" in attributes
        assert attributes["total_conductance"] == 0.1947
        
        assert "equilibrium_ratio" in attributes  
        assert attributes["equilibrium_ratio"] == 0.406
        
        # Verify learning progress indicators
        assert "cycle_count" in attributes
        assert attributes["cycle_count"] == 5
        
        assert "parameter_updates" in attributes
        assert attributes["parameter_updates"] == 2
        
        assert "model_health" in attributes
        assert attributes["model_health"] == "learning"
        
        assert "is_improving" in attributes
        assert attributes["is_improving"] is True
        
        assert "improvement_percentage" in attributes
        assert attributes["improvement_percentage"] == 5.2
        
        assert "total_predictions" in attributes
        assert attributes["total_predictions"] == 150
    
    def test_thermal_parameters_use_defaults_when_missing(self):
        """Test that thermal parameters use appropriate defaults when not in metrics."""
        ha_client = HAClient("http://test:8123", "fake_token")
        ha_client.set_state = Mock()
        
        # Minimal learning metrics (missing thermal parameters)
        learning_metrics = {
            "learning_confidence": 2.5,
            "cycle_count": 0
        }
        
        # Call the function
        ha_client.log_adaptive_learning_metrics(learning_metrics)
        
        # Extract attributes
        call_args = [call for call in ha_client.set_state.call_args_list 
                    if call[0][0] == "sensor.ml_heating_learning"][0]
        attributes = call_args[0][2]
        
        # Verify default values are used
        assert attributes["thermal_time_constant"] == 6.0  # default
        assert attributes["heat_loss_coefficient"] == 0.05  # default
        assert attributes["outlet_effectiveness"] == 0.8   # default
        assert attributes["total_conductance"] == 0.3      # default
        assert attributes["equilibrium_ratio"] == 0.4      # default
        
        # Verify state uses the provided learning confidence
        assert ha_client.set_state.call_args_list[0][0][1] == 2.5
    
    def test_all_sensor_types_created(self):
        """Test that all four main sensor types are created properly."""
        ha_client = HAClient("http://test:8123", "fake_token")
        ha_client.set_state = Mock()
        
        # Complete learning metrics
        learning_metrics = {
            "learning_confidence": 4.5,
            "thermal_time_constant": 3.5,
            "total_conductance": 0.25,
            "equilibrium_ratio": 0.45,
            "mae_all_time": 0.8,
            "mae_1h": 0.6,
            "mae_6h": 0.7,
            "mae_24h": 0.75,
            "rmse_all_time": 1.2,
            "recent_max_error": 2.1,
            "good_control_pct": 78.5,
            "perfect_accuracy_pct": 45.2,
            "tolerable_accuracy_pct": 65.8,
            "poor_accuracy_pct": 8.3,
            "prediction_count_24h": 144,
            "total_predictions": 500,
            "improvement_percentage": 12.5
        }
        
        # Call the function
        ha_client.log_adaptive_learning_metrics(learning_metrics)
        
        # Verify all four sensor types were created
        sensor_calls = [call[0][0] for call in ha_client.set_state.call_args_list]
        
        assert "sensor.ml_heating_learning" in sensor_calls
        assert "sensor.ml_model_mae" in sensor_calls
        assert "sensor.ml_model_rmse" in sensor_calls
        assert "sensor.ml_prediction_accuracy" in sensor_calls
        
        # Verify correct states were set
        for call in ha_client.set_state.call_args_list:
            args, kwargs = call
            sensor_name = args[0]
            state = args[1]
            attributes = args[2]
            round_digits = kwargs.get('round_digits')
            
            if sensor_name == "sensor.ml_heating_learning":
                assert state == 4.5  # learning_confidence
                assert round_digits == 3
                
            elif sensor_name == "sensor.ml_model_mae":
                assert state == 0.8  # mae_all_time
                assert round_digits == 4
                
            elif sensor_name == "sensor.ml_model_rmse":
                assert state == 1.2  # rmse_all_time
                assert round_digits == 4
                
            elif sensor_name == "sensor.ml_prediction_accuracy":
                assert state == 78.5  # good_control_pct
                assert round_digits == 1
