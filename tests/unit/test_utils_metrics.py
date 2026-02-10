
import pytest
import numpy as np
from src.utils_metrics import (
    mae,
    rmse,
    rolling_sigma,
    confidence_from_sigma,
    MAE,
    RMSE,
)

# Test data
Y_TRUE = [1, 2, 3, 4, 5]
Y_PRED = [1.5, 2.5, 3.5, 4.5, 5.5]
EMPTY_LIST = []


def test_mae_basic():
    """Test MAE with basic lists."""
    assert mae(Y_TRUE, Y_PRED) == pytest.approx(0.5)


def test_mae_numpy():
    """Test MAE with NumPy arrays."""
    assert mae(np.array(Y_TRUE), np.array(Y_PRED)) == pytest.approx(0.5)


def test_mae_empty():
    """Test MAE with empty lists."""
    assert mae(EMPTY_LIST, EMPTY_LIST) == 0.0


def test_rmse_basic():
    """Test RMSE with basic lists."""
    assert rmse(Y_TRUE, Y_PRED) == pytest.approx(0.5)


def test_rmse_numpy():
    """Test RMSE with NumPy arrays."""
    assert rmse(np.array(Y_TRUE), np.array(Y_PRED)) == pytest.approx(0.5)


def test_rmse_empty():
    """Test RMSE with empty lists."""
    assert rmse(EMPTY_LIST, EMPTY_LIST) == 0.0


def test_rolling_sigma_basic():
    """Test rolling_sigma with enough data."""
    errors = [0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 0.2, 0.3]
    sigma = rolling_sigma(errors, window=5)
    assert sigma > 0


def test_rolling_sigma_not_enough_data():
    """Test rolling_sigma with insufficient data."""
    errors = [0.1, 0.2]
    assert rolling_sigma(errors) == 0.15


def test_rolling_sigma_min_sigma():
    """Test that rolling_sigma respects min_sigma."""
    errors = [0.001] * 10
    assert rolling_sigma(errors, min_sigma=0.05) == 0.05


def test_rolling_sigma_max_value():
    """Test that rolling_sigma is capped at 0.5."""
    errors = [1.0] * 10
    assert rolling_sigma(errors) <= 0.5


def test_confidence_from_sigma():
    """Test confidence_from_sigma conversion."""
    assert confidence_from_sigma(0.1) == pytest.approx(1.0 / 1.1)
    assert confidence_from_sigma(0.5) == pytest.approx(1.0 / 1.5)
    assert confidence_from_sigma(0.0) == 1.0


def test_confidence_from_sigma_invalid_input():
    """Test confidence_from_sigma with invalid input."""
    assert confidence_from_sigma("invalid") == pytest.approx(1.0 / 1.5)


class TestIncrementalMetrics:
    def test_mae_incremental(self):
        """Test incremental MAE with scalar updates."""
        m = MAE()
        m.update(1, 1.5)
        m.update(2, 2.5)
        assert m.get() == pytest.approx(0.5)

    def test_mae_vectorized(self):
        """Test incremental MAE with vectorized updates."""
        m = MAE()
        m.update(Y_TRUE, Y_PRED)
        assert m.get() == pytest.approx(0.5)

    def test_mae_mixed_updates(self):
        """Test incremental MAE with mixed scalar and vector updates."""
        m = MAE()
        m.update(1, 1.5)
        m.update([2, 3], [2.5, 3.5])
        assert m.get() == pytest.approx(0.5)

    def test_rmse_incremental(self):
        """Test incremental RMSE with scalar updates."""
        m = RMSE()
        m.update(1, 1.5)
        m.update(2, 2.5)
        assert m.get() == pytest.approx(0.5)

    def test_rmse_vectorized(self):
        """Test incremental RMSE with vectorized updates."""
        m = RMSE()
        m.update(Y_TRUE, Y_PRED)
        assert m.get() == pytest.approx(0.5)

    def test_rmse_mixed_updates(self):
        """Test incremental RMSE with mixed scalar and vector updates."""
        m = RMSE()
        m.update(1, 1.5)
        m.update([2, 3], [2.5, 3.5])
        assert m.get() == pytest.approx(0.5)

    def test_empty_metrics(self):
        """Test that get() returns 0 for no updates."""
        mae_metric = MAE()
        rmse_metric = RMSE()
        assert mae_metric.get() == 0.0
        assert rmse_metric.get() == 0.0

