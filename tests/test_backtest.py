import numpy as np
import pandas as pd
import pytest

from demand_forecasting import baselines
from demand_forecasting.backtest import backtest, rolling_origin_splits, summarize


@pytest.fixture
def seasonal_series():
    idx = pd.date_range("2010-01-01", periods=280, freq="D")
    rng = np.random.default_rng(0)
    base = 100 + 20 * np.sin(2 * np.pi * np.arange(len(idx)) / 7)
    vals = base + rng.normal(0, 3, size=len(idx))
    return pd.Series(vals, index=idx, name="units")


def test_rolling_origin_splits_are_chronological_and_nonoverlapping(seasonal_series):
    folds = list(rolling_origin_splits(seasonal_series, horizon=28, n_folds=6))
    assert len(folds) == 6
    for fold in folds:
        # Train always precedes test in time.
        assert fold.train.index.max() < fold.test.index.min()
        assert len(fold.test) == 28
    # Test windows advance forward.
    starts = [f.test.index.min() for f in folds]
    assert starts == sorted(starts)


def test_rolling_origin_raises_when_series_too_short():
    s = pd.Series(range(10), index=pd.date_range("2010-01-01", periods=10, freq="D"))
    with pytest.raises(ValueError):
        list(rolling_origin_splits(s, horizon=28, n_folds=6))


def test_backtest_includes_seasonal_naive_and_summarizes(seasonal_series):
    models = {"ma7": lambda tr, h: baselines.moving_average(tr, h, 7)}
    results = backtest(seasonal_series, models, horizon=28, n_folds=4)
    assert set(results["model"]) == {"seasonal_naive", "ma7"}
    summary = summarize(results)
    assert summary.loc["seasonal_naive", "mase_improve_pct"] == pytest.approx(0.0)
    assert "mase" in summary.columns
