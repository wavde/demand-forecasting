"""Smoke-run harness for CI: run the evidence pipelines without the network.

``activate(tmp_dir)`` makes ``run.py`` and ``run_pretrained.py`` run end to end on
synthetic data, writing every output under ``tmp_dir``:

- **Data.** ``data.load_transactions`` is replaced by ``synthetic_transactions``,
  which returns transaction lines with the UCI Online Retail II columns (including
  cancellations, a closed Saturday and several countries). The real cleaning,
  country filter and daily aggregation code still runs on them. Nothing is
  downloaded.
- **Paths.** Data, docs and reports directories point into ``tmp_dir``, so a smoke
  run never touches the committed results.

``stub_chronos_predictor`` stands in for Chronos-Bolt. It is a seasonal-naive
quantile forecaster with the same call signature, so the pretrained pipeline runs
without torch, transformers or model weights. Smoke outputs are not results.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, data

UCI_COLUMNS = (
    "Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price",
    "Customer ID", "Country",
)
SMOKE_DAYS = 420


def synthetic_transactions(n_days: int = SMOKE_DAYS, seed: int = config.SEED) -> pd.DataFrame:
    """Transaction lines with the UCI Online Retail II schema.

    Daily UK volume has a weekly pattern and a mild trend; there is no trading on
    Saturdays. About 5% of lines are cancellations (invoice prefix ``C``, negative
    quantity), which the real cleaning step must drop.
    """
    rng = np.random.default_rng(seed)
    days = pd.date_range("2010-01-04", periods=n_days, freq="D")
    rows = []
    invoice = 500_000
    for i, day in enumerate(days):
        if day.dayofweek in config.CLOSED_DOW:
            continue
        weekly = 1.0 + 0.3 * np.sin(2 * np.pi * day.dayofweek / 7)
        for country, scale in ((config.PRIMARY_COUNTRY, 40), ("France", 4), ("Germany", 4)):
            n_lines = rng.poisson(scale * weekly * (1 + i / (2 * n_days)))
            for _ in range(n_lines):
                invoice += 1
                cancelled = rng.random() < 0.05
                qty = int(rng.integers(1, 24))
                rows.append((
                    f"C{invoice}" if cancelled else str(invoice),
                    "85123A",
                    "SYNTHETIC ITEM",
                    -qty if cancelled else qty,
                    day + pd.Timedelta(hours=int(rng.integers(8, 18))),
                    round(float(rng.uniform(0.5, 10.0)), 2),
                    float(rng.integers(12_000, 18_000)),
                    country,
                ))
    return pd.DataFrame(rows, columns=list(UCI_COLUMNS))


def stub_chronos_predictor(
    context: np.ndarray, horizon: int, quantile_levels: Sequence[float]
) -> np.ndarray:
    """Seasonal-naive quantiles in Chronos-Bolt's output shape: ``(horizon, n_quantiles)``."""
    m = config.SEASONAL_PERIOD
    last = np.asarray(context, dtype=float)[-m:]
    median = np.resize(last, horizon)
    spread = np.std(context) if len(context) > 1 else 0.0
    z = {0.1: -1.2816, 0.5: 0.0, 0.9: 1.2816}
    return np.column_stack([median + z.get(q, 0.0) * spread for q in quantile_levels])


def activate(tmp_dir: Path) -> None:
    """Point every data and output path into ``tmp_dir`` and serve synthetic data."""
    tmp_dir = Path(tmp_dir)
    config.DATA_DIR = tmp_dir / "data"
    config.RAW_DIR = config.DATA_DIR / "raw"
    config.PROCESSED_DIR = config.DATA_DIR / "processed"
    config.DAILY_CSV = config.PROCESSED_DIR / "daily_demand.csv"
    config.DOCS_DIR = tmp_dir / "docs"
    config.REPORTS_DIR = tmp_dir / "reports"
    data.load_transactions = synthetic_transactions
    config.ensure_dirs()
