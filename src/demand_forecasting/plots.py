"""Matplotlib figures for EDA and forecast diagnostics.

Every function saves a PNG into docs/ and returns the path. Styling is kept simple
so the figures read cleanly in a README or on the site.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 10,
    }
)
_BLUE = "#2f6df6"
_ORANGE = "#f59e0b"
_GREY = "#8b95a7"


def _save(fig: plt.Figure, name: str) -> Path:
    config.ensure_dirs()
    path = config.DOCS_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_eda(series: pd.Series) -> Path:
    """Series overview: full history, weekly seasonality, and the rolling mean."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    axes[0, 0].plot(series.index, series.values, color=_BLUE, lw=0.8)
    axes[0, 0].plot(
        series.index, series.rolling(28).mean(), color=_ORANGE, lw=1.6, label="28-day mean"
    )
    axes[0, 0].set_title("Daily units sold (UK)")
    axes[0, 0].legend()

    by_dow = series.groupby(series.index.dayofweek).mean()
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    axes[0, 1].bar(labels, by_dow.to_numpy(), color=_BLUE)
    axes[0, 1].set_title("Mean demand by day of week")

    by_month = series.groupby(series.index.month).mean()
    axes[1, 0].plot(by_month.index, by_month.to_numpy(), marker="o", color=_BLUE)
    axes[1, 0].set_xticks(range(1, 13))
    axes[1, 0].set_title("Mean demand by month (Christmas ramp)")

    axes[1, 1].hist(series.values, bins=40, color=_GREY)
    axes[1, 1].set_title("Distribution of daily units")

    fig.suptitle("Online Retail II — exploratory view", fontweight="bold")
    return _save(fig, "eda.png")


def plot_forecast(
    history: pd.Series,
    actual: pd.Series,
    mean: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    tail: int = 120,
    name: str = "forecast.png",
) -> Path:
    """Forecast vs actual on the held-out test window with a prediction band."""
    fig, ax = plt.subplots(figsize=(12, 5))
    hist_tail = history.iloc[-tail:]
    ax.plot(hist_tail.index, hist_tail.values, color=_GREY, lw=1.0, label="history")
    ax.plot(actual.index, actual.values, color=_BLUE, lw=1.8, label="actual")
    ax.plot(actual.index, mean, color=_ORANGE, lw=1.8, label="forecast")
    if lower is not None and upper is not None:
        ax.fill_between(actual.index, lower, upper, color=_ORANGE, alpha=0.2, label="80% PI")
    ax.set_title("Forecast vs actual — held-out test window")
    ax.legend()
    return _save(fig, name)


def plot_residuals(actual: pd.Series, mean: np.ndarray, name: str = "residuals.png") -> Path:
    """Residual diagnostics: residuals over time and vs day of week."""
    resid = actual.to_numpy(dtype=float) - np.asarray(mean, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].axhline(0, color=_GREY, lw=1)
    axes[0].plot(actual.index, resid, marker="o", ms=3, color=_BLUE, lw=0.8)
    axes[0].set_title("Residuals over the test window")

    dow = pd.Series(resid, index=actual.index).groupby(actual.index.dayofweek).mean()
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    axes[1].bar([labels[i] for i in dow.index], dow.to_numpy(), color=_BLUE)
    axes[1].axhline(0, color=_GREY, lw=1)
    axes[1].set_title("Mean residual by day of week")

    fig.suptitle("Error analysis", fontweight="bold")
    return _save(fig, name)


def plot_backtest(summary: pd.DataFrame, name: str = "backtest.png") -> Path:
    """Bar chart of mean MASE by model with the MASE = 1 baseline line."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = summary.sort_values("mase")
    colors = [_ORANGE if m == "seasonal_naive" else _BLUE for m in order.index]
    ax.barh(order.index, order["mase"], color=colors)
    ax.axvline(1.0, color=_GREY, ls="--", lw=1, label="MASE = 1 (naive)")
    ax.set_xlabel("MASE (lower is better)")
    ax.set_title("Backtested MASE by model")
    ax.legend()
    return _save(fig, name)
