"""Daily aggregation tests on a tiny in-memory frame (no download)."""

import pandas as pd
import pytest

from demand_forecasting import config
from demand_forecasting.data import aggregate_daily


@pytest.fixture
def clean():
    return pd.DataFrame(
        {
            "invoice": ["1", "2", "3", "4", "5"],
            "quantity": [10, 5, 7, 3, 4],
            "invoice_date": pd.to_datetime(
                [
                    "2010-01-01 09:00",
                    "2010-01-01 15:00",
                    "2010-01-03 10:00",
                    "2010-01-01 11:00",
                    "2010-01-02 12:00",
                ]
            ),
            "price": [1.0] * 5,
            "country": [config.PRIMARY_COUNTRY] * 3 + ["France", "Germany"],
        }
    )


def test_primary_country_daily_totals_with_gap_filled(clean):
    out = aggregate_daily(clean)
    assert out["units"].tolist() == [15.0, 0.0, 7.0]
    assert out.index.name == "date"
    assert list(out.index) == list(pd.date_range("2010-01-01", periods=3, freq="D"))


def test_rest_of_world_excludes_primary_country(clean):
    out = aggregate_daily(clean, config.REST_OF_WORLD)
    assert out["units"].tolist() == [3.0, 4.0]


def test_unknown_country_raises(clean):
    with pytest.raises(ValueError):
        aggregate_daily(clean, "Atlantis")
