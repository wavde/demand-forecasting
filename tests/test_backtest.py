import numpy as np
import pandas as pd
import pytest

from demand_forecasting import baselines, metrics
from demand_forecasting.backtest import (
    backtest,
    backtest_forecasts,
    head_to_head,
    horizon_profile,
    rolling_origin_splits,
    score_forecasts,
    summarize,
    with_history_limit,
)


@pytest.fixture
def seasonal_series():
    idx = pd.date_range("2010-01-01", periods=280, freq="D")
    rng = np.random.default_rng(0)
    base = 100 + 20 * np.sin(2 * np.pi * np.arange(len(idx)) / 7)
    vals = base + rng.normal(0, 3, size=len(idx))
    return pd.Series(vals, index=idx, name="units")


def _ma7(tr, h):
    return baselines.moving_average(tr, h, 7)


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


def test_backtest_forecasts_has_one_row_per_model_fold_step(seasonal_series):
    fc = backtest_forecasts(seasonal_series, {"ma7": _ma7}, horizon=28, n_folds=4)
    assert len(fc) == 2 * 4 * 28
    assert set(fc.columns) >= {"model", "fold", "step", "date", "y_true", "y_pred", "scale"}
    one = fc[(fc["model"] == "ma7") & (fc["fold"] == 0)]
    assert one["step"].tolist() == list(range(1, 29))
    assert one["scale"].nunique() == 1


def test_score_forecasts_matches_direct_metric_calls(seasonal_series):
    fc = backtest_forecasts(seasonal_series, {"ma7": _ma7}, horizon=28, n_folds=4)
    scores = score_forecasts(fc).set_index(["model", "fold"])
    for fold in rolling_origin_splits(seasonal_series, horizon=28, n_folds=4):
        pred = _ma7(fold.train, 28)
        # Saturdays are forced to zero by the closed-day rule inside the backtest.
        pred[fold.test.index.day_of_week == 5] = 0.0
        expected = metrics.mase(fold.test.to_numpy(), pred, fold.train.to_numpy(), 7)
        assert scores.loc[("ma7", fold.index), "mase"] == pytest.approx(expected)


def test_backtest_forecasts_rejects_wrong_length_forecast(seasonal_series):
    with pytest.raises(ValueError):
        backtest_forecasts(
            seasonal_series, {"bad": lambda tr, h: np.zeros(h - 1)}, horizon=28, n_folds=2
        )


def test_with_history_limit_passes_only_the_tail():
    seen = []

    def spy(train, horizon):
        seen.append(train)
        return np.zeros(horizon)

    s = pd.Series(np.arange(100.0), index=pd.date_range("2010-01-01", periods=100, freq="D"))
    with_history_limit(spy, 10)(s, 5)
    assert len(seen[0]) == 10
    assert seen[0].index[-1] == s.index[-1]
    assert with_history_limit(spy, None) is spy
    with pytest.raises(ValueError):
        with_history_limit(spy, 0)


def test_history_limit_does_not_change_mase_scale(seasonal_series):
    full = backtest_forecasts(seasonal_series, {"ma7": _ma7}, horizon=28, n_folds=3)
    short = backtest_forecasts(
        seasonal_series, {"ma7": with_history_limit(_ma7, 14)}, horizon=28, n_folds=3
    )
    assert np.allclose(full["scale"], short["scale"])


def test_horizon_profile_buckets_average_to_overall_mase(seasonal_series):
    fc = backtest_forecasts(seasonal_series, {"ma7": _ma7}, horizon=28, n_folds=4)
    prof = horizon_profile(fc, bucket_size=7)
    assert list(prof.columns) == ["1-7", "8-14", "15-21", "22-28"]
    overall = summarize(score_forecasts(fc))["mase"]
    for model in ("ma7", "seasonal_naive"):
        assert prof.loc[model].mean() == pytest.approx(overall[model])


def test_head_to_head_counts_challenger_wins():
    scores = pd.DataFrame(
        {
            "model": ["a", "a", "b", "b"],
            "fold": [0, 1, 0, 1],
            "mase": [1.0, 0.5, 0.8, 0.9],
        }
    )
    h2h = head_to_head(scores, challenger="b", incumbent="a")
    assert h2h["challenger_wins"].tolist() == [True, False]
    assert h2h["diff"].tolist() == pytest.approx([-0.2, 0.4])
    with pytest.raises(KeyError):
        head_to_head(scores, challenger="c", incumbent="a")
