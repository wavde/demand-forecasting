"""Project paths and shared constants."""

from __future__ import annotations

from pathlib import Path

# Repo root is two levels up from this file (src/demand_forecasting/config.py).
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DOCS_DIR = ROOT / "docs"
REPORTS_DIR = ROOT / "reports"

# UCI Online Retail II. Public, no authentication required.
DATA_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
RAW_ZIP = RAW_DIR / "online_retail_ii.zip"
RAW_XLSX = RAW_DIR / "online_retail_II.xlsx"
DAILY_CSV = PROCESSED_DIR / "daily_demand.csv"

# Reproducibility.
SEED = 7

# Forecasting setup.
SEASONAL_PERIOD = 7  # weekly seasonality for a daily series
HORIZON = 28  # forecast 4 weeks ahead
N_FOLDS = 6  # rolling-origin backtest folds

# Business rule: the retailer does not process orders on Saturdays, so Saturday
# demand is ~always zero in the data. dayofweek 5 == Saturday (Mon=0). We force
# model forecasts to zero on closed days; the seasonal-naive baseline already
# inherits this from the prior week, so the comparison stays fair.
CLOSED_DOW: tuple[int, ...] = (5,)

# The series is a single online retailer; the country with the vast majority of
# volume is the United Kingdom. We forecast total UK daily units.
PRIMARY_COUNTRY = "United Kingdom"


def ensure_dirs() -> None:
    """Create the local data/report directories if they do not exist."""
    for d in (RAW_DIR, PROCESSED_DIR, DOCS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
