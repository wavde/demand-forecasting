"""Rolling-origin (time-series) cross-validation.

The series is split into ``n_folds`` consecutive test windows of length
``horizon``. Fold k trains on everything before its test window and forecasts the
window. This mimics production: always predict the future from the past, never the
reverse. No shuffling, no leakage.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import baselines, config
from .metrics import mase_scale, rmse, smape
from .postprocess import apply_closed_days

ForecastFn = Callable[[pd.Series, int], np.ndarray]


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


def with_history_limit(fn: ForecastFn, max_history: int | None) -> ForecastFn:
    """Wrap a forecaster so it only sees the last ``max_history`` points of each train split.

    The fold's full train split is still what the backtest uses for MASE scaling, so
    errors stay comparable across history lengths. ``None`` means no limit.
    """
    if max_history is None:
        return fn
    if max_history < 1:
        raise ValueError("max_history must be >= 1")

    def limited(train: pd.Series, horizon: int) -> np.ndarray:
        return fn(train.iloc[-max_history:], horizon)

    return limited


def backtest_forecasts(
    series: pd.Series,
    models: dict[str, ForecastFn],
    horizon: int = config.HORIZON,
    n_folds: int = config.N_FOLDS,
    seasonal_period: int = config.SEASONAL_PERIOD,
    closed_dow: tuple[int, ...] = config.CLOSED_DOW,
) -> pd.DataFrame:
    """Run every model over the rolling-origin folds and keep each point forecast.

    Same contract as :func:`backtest` (seasonal-naive always included, closed-day rule
    applied to every model), but returns one row per (model, fold, step) with the
    actual, the forecast, and ``scale``: the fold's in-sample seasonal-naive MAE used
    as the MASE denominator. Use :func:`score_forecasts` or :func:`horizon_profile`
    to turn it into metrics.
    """
    all_models = {
        "seasonal_naive": lambda tr, h: baselines.seasonal_naive(tr, h, seasonal_period),
        **models,
    }

    frames = []
    folds = list(rolling_origin_splits(series, horizon, n_folds))
    for name, fn in all_models.items():
        for fold in folds:
            y_true = fold.test.to_numpy(dtype=float)
            y_pred = np.asarray(fn(fold.train, horizon), dtype=float)
            if y_pred.shape != y_true.shape:
                raise ValueError(
                    f"{name} returned shape {y_pred.shape} on fold {fold.index}, "
                    f"expected {y_true.shape}"
                )
            y_pred = apply_closed_days(y_pred, fold.test.index, closed_dow)
            frames.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "fold": fold.index,
                        "step": np.arange(1, horizon + 1),
                        "date": fold.test.index,
                        "y_true": y_true,
                        "y_pred": y_pred,
                        "scale": mase_scale(fold.train.to_numpy(float), seasonal_period),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def score_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Per-(model, fold) sMAPE, RMSE and MASE from a :func:`backtest_forecasts` frame."""
    rows = []
    for (name, fold), g in forecasts.groupby(["model", "fold"], sort=False):
        y_true = g["y_true"].to_numpy(dtype=float)
        y_pred = g["y_pred"].to_numpy(dtype=float)
        rows.append(
            {
                "model": name,
                "fold": fold,
                "smape": smape(y_true, y_pred),
                "rmse": rmse(y_true, y_pred),
                "mase": float(np.mean(np.abs(y_true - y_pred)) / g["scale"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


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
    forecasts = backtest_forecasts(series, models, horizon, n_folds, seasonal_period, closed_dow)
    return score_forecasts(forecasts)


def horizon_profile(forecasts: pd.DataFrame, bucket_size: int = 7) -> pd.DataFrame:
    """MASE by lead-time bucket (e.g. days 1-7, 8-14, ...), averaged across folds.

    Within each fold the bucket's MAE is divided by that fold's MASE scale, so the
    buckets average back to the overall MASE when they are of equal size.
    Returns a frame indexed by model with one column per bucket label.
    """
    if bucket_size < 1:
        raise ValueError("bucket_size must be >= 1")
    f = forecasts.copy()
    start = ((f["step"] - 1) // bucket_size) * bucket_size + 1
    end = np.minimum(start + bucket_size - 1, f["step"].max())
    f["bucket"] = start.astype(str) + "-" + end.astype(str)
    f["abs_err"] = (f["y_true"] - f["y_pred"]).abs()
    per_fold = (
        f.groupby(["model", "fold", "bucket"], sort=False)
        .agg(mae=("abs_err", "mean"), scale=("scale", "first"), first_step=("step", "min"))
        .reset_index()
    )
    per_fold["mase"] = per_fold["mae"] / per_fold["scale"]
    order = per_fold.sort_values("first_step")["bucket"].drop_duplicates().tolist()
    out = per_fold.groupby(["model", "bucket"])["mase"].mean().unstack("bucket")
    return out[order]


def head_to_head(scores: pd.DataFrame, challenger: str, incumbent: str) -> pd.DataFrame:
    """Fold-by-fold MASE of two models from a per-(model, fold) scores frame.

    Columns: ``fold``, ``incumbent``, ``challenger``, ``diff`` (challenger minus
    incumbent; negative means the challenger is better) and ``challenger_wins``.
    """
    wide = scores.pivot(index="fold", columns="model", values="mase")
    missing = {challenger, incumbent} - set(wide.columns)
    if missing:
        raise KeyError(f"models not in scores: {sorted(missing)}")
    out = pd.DataFrame(
        {
            "fold": wide.index.to_numpy(),
            "incumbent": wide[incumbent].to_numpy(),
            "challenger": wide[challenger].to_numpy(),
        }
    )
    out["diff"] = out["challenger"] - out["incumbent"]
    out["challenger_wins"] = out["diff"] < 0
    return out


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
