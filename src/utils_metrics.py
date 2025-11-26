"""
Lightweight metrics helpers — pure Python / NumPy implementations.

Provides:
- mae(y_true, y_pred)
- rmse(y_true, y_pred)
- rolling_sigma(errors, window=50, min_sigma=0.02)
- confidence_from_sigma(sigma)

Designed to match the behaviour used in the project:
same definitions as sklearn's mean_absolute_error and
sqrt(mean_squared_error), and the physics model's confidence
formula.
"""
from typing import Sequence
import numpy as np

def _to_np(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.array(x, dtype=float)

def mae(y_true: Sequence, y_pred: Sequence) -> float:
    """Mean Absolute Error (batch). Returns 0.0 for empty inputs."""
    y_t = _to_np(y_true)
    y_p = _to_np(y_pred)
    if y_t.size == 0:
        return 0.0
    return float(np.mean(np.abs(y_t - y_p)))

def rmse(y_true: Sequence, y_pred: Sequence) -> float:
    """Root Mean Squared Error (batch). Returns 0.0 for empty inputs."""
    y_t = _to_np(y_true)
    y_p = _to_np(y_pred)
    if y_t.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((y_t - y_p) ** 2)))

def rolling_sigma(errors: Sequence, window: int = 50, min_sigma: float = 0.02) -> float:
    """
    Compute a rolling standard deviation over the most recent `window`
    absolute errors.

    Behaviour:
    - If fewer than 5 samples are available, returns a conservative default.
    - Enforces a minimum `min_sigma` and caps at 0.5 to match physics model.
    """
    errs = _to_np(errors)
    if errs.size < 5:
        return 0.15
    if window <= 0:
        window = len(errs)
    relevant = errs[-window:] if errs.size > window else errs
    sigma = float(np.std(relevant))
    sigma = max(sigma, float(min_sigma))
    sigma = min(sigma, 0.5)
    return sigma

def confidence_from_sigma(sigma: float) -> float:
    """Convert sigma to a 0..1 confidence score using project formula."""
    try:
        s = float(sigma)
    except Exception:
        s = 0.5
    return 1.0 / (1.0 + s)
