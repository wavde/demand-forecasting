"""Forecast error metrics.

All functions take 1-D array-likes of equal length and return a float. They are
deliberately dependency-light (numpy only) and pure so they are easy to unit test.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["mae", "rmse", "mape", "smape", "mase", "mase_scale"]


def _to_arrays(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    if yt.size == 0:
        raise ValueError("empty input")
    return yt, yp


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error."""
    yt, yp = _to_arrays(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error."""
    yt, yp = _to_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute percentage error (%), ignoring zero-actual points.

    Zero actuals make MAPE undefined; those points are dropped. Use sMAPE or
    MASE when zeros are common.
    """
    yt, yp = _to_arrays(y_true, y_pred)
    mask = yt != 0
    if not mask.any():
        raise ValueError("MAPE undefined: all actuals are zero")
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100.0)


def smape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Symmetric MAPE (%), in [0, 200]. Zero-denominator points are dropped."""
    yt, yp = _to_arrays(y_true, y_pred)
    denom = np.abs(yt) + np.abs(yp)
    mask = denom != 0
    if not mask.any():
        return 0.0
    return float(np.mean(2.0 * np.abs(yp[mask] - yt[mask]) / denom[mask]) * 100.0)


def mase_scale(y_train: ArrayLike, seasonal_period: int = 1) -> float:
    """In-sample MAE of the (seasonal) naive forecast on ``y_train``: the MASE denominator."""
    tr = np.asarray(y_train, dtype=float)
    m = int(seasonal_period)
    if m < 1:
        raise ValueError("seasonal_period must be >= 1")
    if tr.size <= m:
        raise ValueError("y_train too short for the requested seasonal_period")
    scale = float(np.mean(np.abs(tr[m:] - tr[:-m])))
    if scale == 0:
        raise ValueError("naive scaling factor is zero (constant training series)")
    return scale


def mase(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_train: ArrayLike,
    seasonal_period: int = 1,
) -> float:
    """Mean absolute scaled error (Hyndman & Koehler, 2006).

    The forecast MAE is scaled by the in-sample MAE of a (seasonal) naive
    forecast on ``y_train``. MASE < 1 means the forecast beats that naive
    benchmark on the training history.

    Args:
        y_true: Out-of-sample actuals.
        y_pred: Out-of-sample forecasts.
        y_train: In-sample training actuals used to compute the scaling.
        seasonal_period: 1 for a random-walk naive scaler, m for seasonal naive.
    """
    yt, yp = _to_arrays(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)) / mase_scale(y_train, seasonal_period))
