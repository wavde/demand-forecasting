import numpy as np
import pandas as pd

from demand_forecasting.models import AutoSarimaModel


def _weekly(n=140):
    idx = pd.date_range("2010-01-04", periods=n, freq="D")
    rng = np.random.default_rng(1)
    vals = 200 + 40 * np.sin(2 * np.pi * np.arange(n) / 7) + 0.3 * np.arange(n)
    return pd.Series(vals + rng.normal(0, 4, n), index=idx)


def test_auto_sarima_picks_a_candidate_and_records_it():
    model = AutoSarimaModel(p_values=(0, 1), q_values=(0,), seasonal_p_values=(0,))
    s = _weekly()
    pred = model.forecast(s, 14)
    assert pred.shape == (14,)
    assert (pred >= 0).all()
    assert len(model.candidates) == 4
    sel = model.selected[0]
    assert sel["train_end"] == str(s.index[-1].date())
    assert sel["n_train"] == len(s)
    assert sel["order"][1] == 1 and sel["seasonal_order"][1] == 1
    assert (sel["order"], (*sel["seasonal_order"], 7)) in model.candidates


def test_auto_sarima_is_deterministic_across_instances():
    s = _weekly()
    a = AutoSarimaModel(p_values=(0, 1), q_values=(0,), seasonal_p_values=(0,))
    b = AutoSarimaModel(p_values=(0, 1), q_values=(0,), seasonal_p_values=(0,))
    np.testing.assert_allclose(a.forecast(s, 7), b.forecast(s, 7))
    assert a.selected == b.selected
