"""Post-processing that injects domain knowledge into raw model forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from . import config

__all__ = ["apply_closed_days"]


def apply_closed_days(
    values: ArrayLike,
    index: pd.DatetimeIndex,
    closed_dow: tuple[int, ...] = config.CLOSED_DOW,
) -> np.ndarray:
    """Force forecasts to zero on known closed days (e.g. Saturdays).

    The retailer does not trade on Saturdays, so any positive forecast there is
    structurally wrong and only inflates error. The seasonal-naive baseline gets
    this for free from the prior week; applying the same rule to every model keeps
    the comparison honest.
    """
    out = np.asarray(values, dtype=float).copy()
    if len(out) != len(index):
        raise ValueError(f"length mismatch: {len(out)} values vs {len(index)} dates")
    mask = np.isin(index.dayofweek, closed_dow)
    out[mask] = 0.0
    return out
