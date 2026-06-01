"""Feature engineering for the ML forecaster.

All features are computed from the target's own past plus the calendar, so a row
dated ``t`` never sees information from ``t`` or later. Lags and rolling windows
are shifted by at least one day to guarantee no leakage.
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

__all__ = ["add_calendar_features", "add_lag_features", "make_features", "FEATURE_COLUMNS"]

DEFAULT_LAGS: tuple[int, ...] = (1, 2, 3, 7, 14, 21, 28)
DEFAULT_WINDOWS: tuple[int, ...] = (7, 14, 28)


def add_calendar_features(df: pd.DataFrame, country: str = "GB") -> pd.DataFrame:
    """Add day-of-week, month, period flags, and a holiday indicator.

    Day-of-week and month are encoded both as integer codes (for tree models) and
    as cyclical sin/cos pairs so distance between e.g. Dec and Jan is small.
    """
    out = df.copy()
    idx = out.index
    out["dow"] = idx.dayofweek
    out["month"] = idx.month
    out["day"] = idx.day
    out["weekofyear"] = idx.isocalendar().week.astype(int).to_numpy()
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["is_month_start"] = idx.is_month_start.astype(int)
    out["is_month_end"] = idx.is_month_end.astype(int)

    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    years = range(idx.year.min(), idx.year.max() + 1)
    hol = holidays.country_holidays(country, years=list(years))
    out["is_holiday"] = pd.Series(idx.date, index=idx).map(lambda d: int(d in hol)).to_numpy()
    return out


def add_lag_features(
    df: pd.DataFrame,
    target: str = "units",
    lags: tuple[int, ...] = DEFAULT_LAGS,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Add lagged values and rolling stats of the target, all leakage-safe.

    Rolling statistics are computed on the lag-1 series, so the window ending at
    ``t`` only uses values up to ``t-1``.
    """
    out = df.copy()
    for lag in lags:
        out[f"lag_{lag}"] = out[target].shift(lag)
    base = out[target].shift(1)
    for w in windows:
        out[f"roll_mean_{w}"] = base.rolling(w).mean()
        out[f"roll_std_{w}"] = base.rolling(w).std()
        out[f"roll_min_{w}"] = base.rolling(w).min()
        out[f"roll_max_{w}"] = base.rolling(w).max()
    return out


# Columns the model consumes; the target and any helper columns are excluded.
FEATURE_COLUMNS: list[str] = (
    ["dow", "month", "day", "weekofyear", "is_weekend", "is_month_start", "is_month_end",
     "dow_sin", "dow_cos", "month_sin", "month_cos", "is_holiday"]
    + [f"lag_{lag}" for lag in DEFAULT_LAGS]
    + [f"roll_{stat}_{w}" for w in DEFAULT_WINDOWS for stat in ("mean", "std", "min", "max")]
)


def make_features(
    df: pd.DataFrame,
    target: str = "units",
    country: str = "GB",
    dropna: bool = True,
) -> pd.DataFrame:
    """Full feature pipeline: calendar + lag/rolling features.

    Args:
        dropna: Drop the warm-up rows that have undefined lags. Set False when
            building a single future row for recursive forecasting.
    """
    out = add_calendar_features(df, country=country)
    out = add_lag_features(out, target=target)
    if dropna:
        out = out.dropna(subset=[c for c in FEATURE_COLUMNS if c.startswith(("lag_", "roll_"))])
    return out
