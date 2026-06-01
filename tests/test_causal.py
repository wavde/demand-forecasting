import numpy as np
import pandas as pd
import pytest

from demand_forecasting import causal
from demand_forecasting.postprocess import apply_closed_days


def test_apply_closed_days_zeros_saturdays():
    idx = pd.date_range("2010-01-01", periods=10, freq="D")  # 2010-01-02 is a Saturday
    vals = np.full(10, 100.0)
    out = apply_closed_days(vals, idx, closed_dow=(5,))
    sat = idx.dayofweek == 5
    assert np.all(out[sat] == 0.0)
    assert np.all(out[~sat] == 100.0)


def test_apply_closed_days_length_mismatch_raises():
    idx = pd.date_range("2010-01-01", periods=3, freq="D")
    with pytest.raises(ValueError):
        apply_closed_days(np.ones(4), idx)


def test_did_recovers_known_uplift():
    idx = pd.date_range("2018-01-01", periods=300, freq="D")
    rng = np.random.default_rng(0)
    series = pd.Series(1000 + 200 * np.sin(2 * np.pi * np.arange(300) / 7) + rng.normal(0, 20, 300),
                       index=idx, name="units")
    res = causal.run_promo_study(series, uplift=0.15)
    assert res.recovered
    assert res.estimated_uplift == pytest.approx(0.15, abs=0.03)
    assert res.p_value < 0.01


def test_simulate_panel_structure():
    idx = pd.date_range("2018-01-01", periods=200, freq="D")
    series = pd.Series(np.linspace(500, 1500, 200), index=idx, name="units")
    panel = causal.simulate_panel(series, uplift=0.2, n_stores=10, window=120)
    assert panel["store"].nunique() == 10
    assert panel["date"].nunique() == 120
    # did flag only on treated stores in the post period.
    assert ((panel["did"] == 1) == ((panel["treated"] == 1) & (panel["post"] == 1))).all()
