import numpy as np
import pytest

from demand_forecasting import metrics


def test_perfect_forecast_is_zero_error():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    assert metrics.mae(y, y) == 0.0
    assert metrics.rmse(y, y) == 0.0
    assert metrics.mape(y, y) == 0.0
    assert metrics.smape(y, y) == 0.0


def test_rmse_penalizes_large_errors_more_than_mae():
    y_true = np.array([0.0, 0.0, 0.0, 0.0])
    y_pred = np.array([0.0, 0.0, 0.0, 4.0])
    assert metrics.rmse(y_true, y_pred) > metrics.mae(y_true, y_pred)


def test_mape_known_value():
    # 10% error on every point.
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 220.0])
    assert metrics.mape(y_true, y_pred) == pytest.approx(10.0)


def test_mape_drops_zero_actuals():
    y_true = np.array([0.0, 100.0])
    y_pred = np.array([5.0, 110.0])
    # Only the second point counts: 10% error.
    assert metrics.mape(y_true, y_pred) == pytest.approx(10.0)


def test_smape_is_symmetric():
    a, b = np.array([100.0]), np.array([110.0])
    assert metrics.smape(a, b) == pytest.approx(metrics.smape(b, a))


def test_smape_bounded_by_200():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([-100.0, -200.0, -300.0])
    assert 0.0 <= metrics.smape(y_true, y_pred) <= 200.0


def test_mase_naive_forecast_is_one_on_random_walk():
    # On a pure random walk, a naive forecast should score ~1 by construction.
    rng = np.random.default_rng(0)
    train = np.cumsum(rng.normal(size=500)) + 100
    # Out-of-sample naive: predict previous value.
    y_true = train[1:50]
    y_pred = train[:49]
    score = metrics.mase(y_true, y_pred, train, seasonal_period=1)
    assert score == pytest.approx(1.0, abs=0.25)


def test_mase_better_than_one_when_beating_naive():
    weekly = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
    # Add a per-week trend so the seasonal scaling factor is non-zero.
    train = np.concatenate([weekly + 5 * w for w in range(10)])
    y_true = weekly + 5 * 10
    score = metrics.mase(y_true, y_true, train, seasonal_period=7)
    assert score == 0.0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        metrics.mae([1, 2, 3], [1, 2])


def test_empty_input_raises():
    with pytest.raises(ValueError):
        metrics.rmse([], [])


def test_mase_requires_enough_training_history():
    with pytest.raises(ValueError):
        metrics.mase([1.0], [1.0], [1.0, 2.0], seasonal_period=7)
