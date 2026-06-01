import numpy as np
import pandas as pd
import pytest

from demand_forecasting import baselines, features
from demand_forecasting.features import FEATURE_COLUMNS


@pytest.fixture
def daily_series():
    idx = pd.date_range("2010-01-01", periods=120, freq="D")
    # Deterministic weekly pattern plus mild trend.
    vals = 100 + 10 * (idx.dayofweek) + np.arange(len(idx)) * 0.5
    return pd.Series(vals, index=idx, name="units")


def test_lag_features_are_not_leaky(daily_series):
    frame = daily_series.to_frame("units")
    feat = features.add_lag_features(frame)
    # lag_1 at row t must equal the target at t-1.
    assert feat["lag_1"].iloc[5] == pytest.approx(daily_series.iloc[4])
    assert feat["lag_7"].iloc[10] == pytest.approx(daily_series.iloc[3])


def test_rolling_features_exclude_current_day(daily_series):
    frame = daily_series.to_frame("units")
    feat = features.add_lag_features(frame, windows=(7,))
    # roll_mean_7 at t is the mean of t-7..t-1, never including t.
    expected = daily_series.iloc[3:10].mean()
    assert feat["roll_mean_7"].iloc[10] == pytest.approx(expected)


def test_calendar_features_present(daily_series):
    feat = features.add_calendar_features(daily_series.to_frame("units"))
    for col in ["dow", "month", "is_weekend", "is_holiday", "dow_sin", "dow_cos"]:
        assert col in feat.columns
    assert set(feat["dow"].unique()).issubset(set(range(7)))
    assert set(feat["is_weekend"].unique()).issubset({0, 1})


def test_make_features_drops_warmup_rows_and_has_all_columns(daily_series):
    feat = features.make_features(daily_series.to_frame("units"))
    assert not feat[FEATURE_COLUMNS].isna().any().any()
    for col in FEATURE_COLUMNS:
        assert col in feat.columns


def test_seasonal_naive_repeats_last_week(daily_series):
    fc = baselines.seasonal_naive(daily_series, horizon=7, period=7)
    np.testing.assert_allclose(fc, daily_series.iloc[-7:].to_numpy())


def test_seasonal_naive_wraps_for_long_horizon(daily_series):
    fc = baselines.seasonal_naive(daily_series, horizon=14, period=7)
    np.testing.assert_allclose(fc[:7], fc[7:])


def test_naive_is_flat(daily_series):
    fc = baselines.naive(daily_series, horizon=5)
    assert np.all(fc == daily_series.iloc[-1])


def test_moving_average_matches_window_mean(daily_series):
    fc = baselines.moving_average(daily_series, horizon=3, window=7)
    assert fc[0] == pytest.approx(daily_series.iloc[-7:].mean())
