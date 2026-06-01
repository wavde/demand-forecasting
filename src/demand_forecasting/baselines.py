"""Reference (baseline) forecasts.

These are the benchmarks every real model must beat. ``seasonal_naive`` with a
weekly period is the headline baseline used to scale MASE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["naive", "seasonal_naive", "moving_average"]


def naive(history: pd.Series, horizon: int) -> np.ndarray:
    """Random-walk forecast: repeat the last observed value."""
    return np.repeat(float(history.iloc[-1]), horizon)


def seasonal_naive(history: pd.Series, horizon: int, period: int = 7) -> np.ndarray:
    """Seasonal naive: forecast equals the value one season ago.

    Step h maps back to the same weekday in the most recent observed week.
    """
    if len(history) < period:
        return naive(history, horizon)
    last_season = history.iloc[-period:].to_numpy(dtype=float)
    return np.array([last_season[h % period] for h in range(horizon)])


def moving_average(history: pd.Series, horizon: int, window: int = 7) -> np.ndarray:
    """Flat forecast at the mean of the last ``window`` observations."""
    w = min(window, len(history))
    return np.repeat(float(history.iloc[-w:].mean()), horizon)
