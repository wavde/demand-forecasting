"""Smoke harness tests: synthetic data has the UCI schema and flows through the real
cleaning and aggregation code; the Chronos stub matches the predictor contract."""

import numpy as np

from demand_forecasting import config, data, smoke
from demand_forecasting.pretrained import QUANTILE_LEVELS, ChronosBoltModel


def test_synthetic_transactions_have_uci_schema_and_cancellations():
    df = smoke.synthetic_transactions(n_days=60)
    assert tuple(df.columns) == smoke.UCI_COLUMNS
    cancelled = df["Invoice"].str.startswith("C")
    assert cancelled.any()
    assert (df.loc[cancelled, "Quantity"] < 0).all()
    assert config.PRIMARY_COUNTRY in set(df["Country"])


def test_synthetic_transactions_flow_through_real_cleaning():
    raw = smoke.synthetic_transactions(n_days=60)
    daily = data.aggregate_daily(data._clean(raw))
    uk = raw[(raw["Country"] == config.PRIMARY_COUNTRY) & ~raw["Invoice"].str.startswith("C")]
    assert daily["units"].sum() == uk["Quantity"].sum()
    assert (daily.index.to_series().diff().dropna() == np.timedelta64(1, "D")).all()
    closed = daily[daily.index.dayofweek.isin(config.CLOSED_DOW)]
    assert (closed["units"] == 0).all()


def test_stub_predictor_matches_chronos_contract():
    s = data.aggregate_daily(data._clean(smoke.synthetic_transactions(n_days=120)))["units"]
    model = ChronosBoltModel(predict_fn=smoke.stub_chronos_predictor)
    q = model.forecast_quantiles(s, 28)
    assert q.shape == (28, len(QUANTILE_LEVELS))
    assert np.all(np.diff(q, axis=1) >= 0)
    # The median is seasonal naive: the last week repeated.
    assert np.allclose(model.forecast(s, 7), s.iloc[-7:].clip(lower=0))
