import unittest
from unittest.mock import MagicMock, patch
from src.main import main


class TestSensorIntegration(unittest.TestCase):
    def setUp(self):
        self.mock_ha_client = MagicMock()
        self.mock_influx_service = MagicMock()
        self.mock_sensor_buffer = MagicMock()

    @patch('src.main.create_ha_client')
    @patch('src.main.create_influx_service')
    @patch('src.main.SensorBuffer')
    @patch('src.main.BlockingStateManager')
    @patch('src.main.HeatingSystemStateChecker')
    @patch('src.main.SensorDataManager')
    @patch('src.main.build_physics_features')
    @patch('src.main.simplified_outlet_prediction')
    @patch('src.main.save_state')
    @patch('src.main.load_state')
    @patch('src.main.calculate_thermodynamic_metrics')
    @patch('src.main.get_sensor_attributes')
    @patch('argparse.ArgumentParser.parse_args')
    def test_thermodynamic_sensor_export(
        self,
        mock_parse_args,
        mock_get_attributes,
        mock_calc_metrics,
        mock_load_state,
        mock_save_state,
        mock_prediction,
        mock_build_features,
        mock_sensor_manager,
        mock_heating_checker,
        mock_blocking_manager,
        mock_sensor_buffer_cls,
        mock_get_influx,
        mock_create_ha
    ):
        # Setup mocks
        mock_parse_args.return_value = MagicMock(
            calibrate_physics=False,
            validate_physics=False,
            debug=False,
            list_backups=False,
            restore_backup=None
        )
        mock_create_ha.return_value = self.mock_ha_client
        mock_get_influx.return_value = self.mock_influx_service
        mock_sensor_buffer_cls.return_value = self.mock_sensor_buffer

        # Mock state loading
        mock_load_state.return_value = {}

        # Mock blocking manager
        mock_blocking_instance = mock_blocking_manager.return_value
        mock_blocking_instance.check_blocking_state.return_value = (False, [])
        mock_blocking_instance.handle_grace_period.return_value = False

        # Mock heating checker
        mock_heating_checker.return_value.check_heating_active.return_value = (
            True
        )

        # Mock sensor data
        mock_sensor_manager.return_value.get_sensor_data.return_value = ({
            "target_indoor_temp": 21.0,
            "actual_indoor": 20.5,
            "actual_outlet_temp": 35.0,
            "avg_other_rooms_temp": 20.0,
            "fireplace_on": False,
            "outdoor_temp": 5.0,
            "owm_temp": 5.0
        }, [])

        # Mock features with thermodynamic data
        features_dict = {
            "cop_realtime": 3.5,
            "thermal_power_kw": 4.2,
            "delta_t": 5.0,
            "flow_rate": 800.0,
            "inlet_temp": 30.0,
            "pv_now": 0.0,
            "tv_on": 0.0
        }
        mock_build_features.return_value = (features_dict, [])

        # Mock calculate_thermodynamic_metrics for early export
        mock_calc_metrics.return_value = {
            "cop_realtime": 3.5,
            "thermal_power_kw": 4.2,
            "delta_t": 5.0,
            "flow_rate": 800.0,
            "inlet_temp": 30.0,
            "power_consumption": 1.2
        }

        # Mock HA client get_state to return values for early export
        self.mock_ha_client.get_state.return_value = "10.0"

        # Mock prediction
        mock_prediction.return_value = (38.0, 5.0, {})

        # Mock attributes
        mock_get_attributes.return_value = {}

        # Run main for one cycle
        # We need to break the infinite loop in main.
        # Since main() is an infinite loop, we can't easily test it directly
        # without refactoring or raising an exception.
        # However, we can verify the logic by inspecting the code or extracting
        # the logic.
        # For this test, let's assume we can't easily run main() directly
        # because of the loop.
        # Instead, let's verify the logic by mocking the loop condition or
        # using a side effect to break it.

        # A common pattern to test infinite loops is to have a side effect that
        # raises an exception after one iteration.
        mock_blocking_instance.poll_for_blocking.side_effect = StopIteration(
            "Break loop"
        )

        try:
            main()
        except StopIteration:
            pass

        # Verify InfluxDB write
        (
            self.mock_influx_service.write_thermodynamic_metrics
            .assert_called_once()
        )
        call_args = (
            self.mock_influx_service.write_thermodynamic_metrics
            .call_args[0][0]
        )
        self.assertEqual(call_args['cop_realtime'], 3.5)
        self.assertEqual(call_args['thermal_power_kw'], 4.2)

        # Verify HA export (This is what we expect to implement)
        # We expect set_state to be called for both new sensors

        # Check for Thermal Power
        self.mock_ha_client.set_state.assert_any_call(
            "sensor.ml_heating_thermal_power",
            4.2,
            {
                "friendly_name": "ML Heating Thermal Power",
                "unit_of_measurement": "kW",
                "icon": "mdi:flash",
                "device_class": "power",
                "state_class": "measurement"
            },
            round_digits=3
        )

        # Check for COP
        self.mock_ha_client.set_state.assert_any_call(
            "sensor.ml_heating_cop_realtime",
            3.5,
            {
                "friendly_name": "ML Heating COP (Realtime)",
                "unit_of_measurement": "COP",
                "icon": "mdi:heat-pump",
                "device_class": "power_factor",
                "state_class": "measurement"
            },
            round_digits=2
        )


if __name__ == '__main__':
    unittest.main()
