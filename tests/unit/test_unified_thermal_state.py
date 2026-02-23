
import pytest
import os

from src.unified_thermal_state import (
    ThermalStateManager, get_thermal_state_manager
)


# Fixture to create a temporary state file for testing
@pytest.fixture
def temp_state_file(tmp_path):
    return tmp_path / "test_thermal_state.json"


# Fixture to create a clean ThermalStateManager instance for each test
@pytest.fixture
def state_manager(temp_state_file):
    # Ensure the global instance is reset
    import src.unified_thermal_state
    src.unified_thermal_state._thermal_state_manager = None

    manager = ThermalStateManager(state_file=str(temp_state_file))
    return manager


class TestThermalStateManager:
    def test_initialization_creates_new_state_file(
            self, state_manager, temp_state_file):
        """Test that a new state file is created on init if it doesn't exist."""
        assert os.path.exists(temp_state_file) is False
        state_manager.save_state()
        assert os.path.exists(temp_state_file) is True

    def test_default_state_structure(self, state_manager):
        """Test the structure of the default thermal state."""
        default_state = state_manager._get_default_state()
        assert "metadata" in default_state
        assert "baseline_parameters" in default_state
        assert "learning_state" in default_state
        assert "prediction_metrics" in default_state
        assert "operational_state" in default_state

    def test_save_and_load_state(self, state_manager, temp_state_file):
        """Test saving the state to a file and loading it back."""
        state_manager.state["operational_state"]["last_indoor_temp"] = 21.5
        state_manager.save_state()

        # Create a new manager instance to load the state from the file
        new_manager = ThermalStateManager(state_file=str(temp_state_file))
        assert new_manager.state["operational_state"]["last_indoor_temp"] == 21.5

    def test_set_and_get_calibrated_baseline(self, state_manager):
        """Test setting and getting calibrated baseline parameters."""
        params = {
            "heat_loss_coefficient": 0.1,
            "outlet_effectiveness": 0.2,
            "thermal_time_constant": 25.5,
        }
        state_manager.set_calibrated_baseline(params, calibration_cycles=100)

        current_params = state_manager.get_current_parameters()
        assert current_params["baseline_parameters"]["source"] == "calibrated"
        assert current_params["baseline_parameters"][
            "heat_loss_coefficient"] == 0.1
        assert current_params["baseline_parameters"][
            "thermal_time_constant"] == 25.5
        assert current_params["baseline_parameters"][
            "calibration_cycles"] == 100

    def test_update_and_get_learning_state(self, state_manager):
        """Test updating and getting the learning state."""
        state_manager.update_learning_state(
            cycle_count=10, learning_confidence=4.5)

        learning_state = state_manager.get_learning_metrics()
        assert learning_state["current_cycle_count"] == 10
        assert learning_state["learning_confidence"] == 4.5

    def test_add_prediction_record(self, state_manager):
        """Test adding a prediction record."""
        record = {"predicted": 22.0, "actual": 21.8}
        state_manager.add_prediction_record(record)

        history = state_manager.state["learning_state"]["prediction_history"]
        assert len(history) == 1
        assert history[0]["predicted"] == 22.0

    def test_prediction_history_sliding_window(self, state_manager):
        """Test that prediction history maintains a sliding window of 200 records."""
        # Add 205 records
        for i in range(205):
            record = {
                "index": i,
                "predicted": 20.0,
                "actual": 20.0
            }
            state_manager.add_prediction_record(record)

        history = state_manager.state["learning_state"]["prediction_history"]
        
        # Check size is capped at 200
        assert len(history) == 200
        
        # Check that the oldest records were removed (indices 0-4 should be gone)
        # The first record in history should be index 5
        assert history[0]["index"] == 5
        # The last record should be index 204
        assert history[-1]["index"] == 204

    def test_parameter_history_sliding_window(self, state_manager):
        """Test that parameter history maintains a sliding window of 500 records."""
        # Add 505 records
        for i in range(505):
            record = {
                "index": i,
                "param_value": 0.5
            }
            state_manager.add_parameter_history_record(record)

        history = state_manager.state["learning_state"]["parameter_history"]
        
        # Check size is capped at 500
        assert len(history) == 500
        
        # Check that the oldest records were removed (indices 0-4 should be gone)
        # The first record in history should be index 5
        assert history[0]["index"] == 5
        # The last record should be index 504
        assert history[-1]["index"] == 504

    def test_update_and_get_operational_state(self, state_manager):
        """Test updating and getting the operational state."""
        state_manager.update_operational_state(last_prediction=25.0)

        op_state = state_manager.get_operational_state()
        assert op_state["last_prediction"] == 25.0

    def test_reset_learning_state(self, state_manager):
        """Test resetting the learning state."""
        state_manager.update_learning_state(learning_confidence=1.0)
        state_manager.reset_learning_state()

        learning_state = state_manager.get_learning_metrics()
        assert learning_state["learning_confidence"] == 3.0

    def test_backup_and_restore(self, state_manager, temp_state_file):
        """Test creating a backup and restoring from it."""
        state_manager.update_operational_state(last_indoor_temp=22.0)
        state_manager.save_state()

        success, backup_file = state_manager.create_backup("test_backup")
        assert success
        assert os.path.exists(backup_file)

        # Modify the current state
        state_manager.update_operational_state(last_indoor_temp=23.0)
        state_manager.save_state()

        # Restore from backup
        state_manager.restore_from_backup(backup_file)
        op_state = state_manager.get_operational_state()
        assert op_state["last_indoor_temp"] == 22.0

    def test_get_thermal_state_manager_singleton(self, temp_state_file):
        """Test that get_thermal_state_manager returns a singleton."""
        import src.unified_thermal_state
        src.unified_thermal_state._thermal_state_manager = None

        manager1 = get_thermal_state_manager()
        manager2 = get_thermal_state_manager()
        assert manager1 is manager2

