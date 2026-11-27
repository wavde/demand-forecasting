"""Chronos-Bolt wrapper tests. No torch or chronos needed: the predictor is injected."""

import numpy as np
import pandas as pd
import pytest

from demand_forecasting import baselines
from demand_forecasting.backtest import backtest
from demand_forecasting.pretrained import QUANTILE_LEVELS, ChronosBoltModel


def _series(n=200):
    idx = pd.date_range("2010-01-04", periods=n, freq="D")
    # A small trend keeps the seasonal-naive MASE scale away from zero.
    vals = 100 + 30 * np.sin(2 * np.pi * np.arange(n) / 7) + 0.2 * np.arange(n)
    return pd.Series(vals, index=idx, name="units")


class _Recorder:
    """Fake predictor: quantiles = last value + (-50, 0, +50); records the context."""

    def __init__(self):
        self.contexts = []

    def __call__(self, context, horizon, quantile_levels):
        self.contexts.append(np.asarray(context))
        last = float(context[-1])
        return np.column_stack([np.full(horizon, last + d) for d in (-50.0, 0.0, 50.0)])


def test_forecast_is_the_median_quantile():
    rec = _Recorder()
    s = _series()
    out = ChronosBoltModel(predict_fn=rec).forecast(s, 28)
    assert out.shape == (28,)
    assert np.allclose(out, s.iloc[-1])
    assert QUANTILE_LEVELS[1] == 0.5


def test_context_is_exactly_the_history_no_future_values():
    rec = _Recorder()
    s = _series()
    ChronosBoltModel(predict_fn=rec).forecast(s.iloc[:150], 28)
    assert np.array_equal(rec.contexts[0], s.iloc[:150].to_numpy())


def test_quantiles_are_clipped_at_zero():
    def negative(context, horizon, levels):
        return np.full((horizon, len(levels)), -5.0)

    q = ChronosBoltModel(predict_fn=negative).forecast_quantiles(_series(), 7)
    assert (q == 0).all()


def test_wrong_predictor_shape_raises():
    def bad(context, horizon, levels):
        return np.zeros((horizon - 1, len(levels)))

    with pytest.raises(ValueError):
        ChronosBoltModel(predict_fn=bad).forecast(_series(), 28)


def test_empty_history_raises():
    with pytest.raises(ValueError):
        ChronosBoltModel(predict_fn=_Recorder()).forecast(_series().iloc[:0], 7)


def test_plugs_into_the_shared_backtest():
    # A predictor that reproduces seasonal-naive must score exactly like the baseline.
    def seasonal(context, horizon, levels):
        point = baselines.seasonal_naive(pd.Series(context), horizon, 7)
        return np.column_stack([point] * len(levels))

    model = ChronosBoltModel(predict_fn=seasonal)
    res = backtest(_series(), {"Chronos-Bolt": model.forecast}, horizon=28, n_folds=4)
    by = res.pivot(index="fold", columns="model", values="mase")
    assert np.allclose(by["Chronos-Bolt"], by["seasonal_naive"])
