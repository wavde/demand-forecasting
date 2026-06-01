"""Rolling-origin (time-series) cross-validation.

The series is split into ``n_folds`` consecutive test windows of length
``horizon``. Fold k trains on everything before its test window and forecasts the
window. This mimics production: always predict the future from the past, never the
reverse. No shuffling, no leakage.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import baselines, config
from .metrics import mase, rmse, smape
from .postprocess import apply_closed_days


@dataclass
class Fold:
    index: int
    train: pd.Series
    test: pd.Series


def rolling_origin_splits(
    series: pd.Series,
    horizon: int = config.HORIZON,
    n_folds: int = config.N_FOLDS,
    min_train: int | None = None,
) -> Iterator[Fold]:
    """Yield expanding-window train/test folds from oldest to newest."""
    n = len(series)
    needed = (min_train or 0) + horizon * n_folds
    if n < needed:
        raise ValueError(f"series too short: have {n}, need >= {needed}")
    first_test_start = n - horizon * n_folds
    for k in range(n_folds):
        test_start = first_test_start + k * horizon
        test_end = test_start + horizon
        yield Fold(
            index=k,
            train=series.iloc[:test_start],
            test=series.iloc[test_start:test_end],
        )


def backtest(
    series: pd.Series,
    models: dict,
    horizon: int = config.HORIZON,
    n_folds: int = config.N_FOLDS,
    seasonal_period: int = config.SEASONAL_PERIOD,
    closed_dow: tuple[int, ...] = config.CLOSED_DOW,
) -> pd.DataFrame:
    """Backtest each model over rolling-origin folds.

    ``models`` maps a display name to a callable ``f(train_series, horizon) ->
    np.ndarray``. A seasonal-naive baseline is always included automatically. The
    closed-day business rule is applied to every forecast so each model gets the
    same structural benefit the seasonal-naive baseline gets for free. The returned
    long-format frame has one row per (model, fold) with sMAPE, RMSE, and MASE
    (scaled by the in-sample seasonal-naive error of that fold's train split).
    """
    all_models = {
        "seasonal_naive": lambda tr, h: baselines.seasonal_naive(tr, h, seasonal_period),
        **models,
    }

    rows = []
    folds = list(rolling_origin_splits(series, horizon, n_folds))
    for name, fn in all_models.items():
        for fold in folds:
            y_true = fold.test.to_numpy(dtype=float)
            y_pred = np.asarray(fn(fold.train, horizon), dtype=float)
            y_pred = apply_closed_days(y_pred, fold.test.index, closed_dow)
            rows.append(
                {
                    "model": name,
                    "fold": fold.index,
                    "smape": smape(y_true, y_pred),
                    "rmse": rmse(y_true, y_pred),
                    "mase": mase(y_true, y_pred, fold.train.to_numpy(float), seasonal_period),
                }
            )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Average metrics across folds and add improvement vs seasonal-naive."""
    agg = (
        results.groupby("model")[["smape", "rmse", "mase"]]
        .mean()
        .sort_values("mase")
    )
    base_smape = agg.loc["seasonal_naive", "smape"]
    base_mase = agg.loc["seasonal_naive", "mase"]
    agg["smape_improve_pct"] = (1 - agg["smape"] / base_smape) * 100
    agg["mase_improve_pct"] = (1 - agg["mase"] / base_mase) * 100
    return agg
