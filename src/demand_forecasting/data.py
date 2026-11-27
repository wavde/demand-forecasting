"""Download UCI Online Retail II and build a daily demand series.

The raw workbook has two sheets ("Year 2009-2010" and "Year 2010-2011") of
transaction lines. We concatenate them, clean returns/cancellations, restrict to
the primary country, and aggregate to a continuous daily total-units series.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

from . import config

__all__ = [
    "download",
    "load_transactions",
    "aggregate_daily",
    "build_daily_demand",
    "get_daily_demand",
]


def download(force: bool = False) -> None:
    """Fetch the raw zip from UCI and extract the xlsx into data/raw."""
    config.ensure_dirs()
    if config.RAW_XLSX.exists() and not force:
        return
    if not config.RAW_ZIP.exists() or force:
        resp = requests.get(config.DATA_URL, timeout=120)
        resp.raise_for_status()
        config.RAW_ZIP.write_bytes(resp.content)
    with zipfile.ZipFile(config.RAW_ZIP) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".xlsx")]
        if not names:
            raise RuntimeError("no .xlsx found inside the UCI zip")
        with zf.open(names[0]) as src:
            config.RAW_XLSX.write_bytes(src.read())


def load_transactions() -> pd.DataFrame:
    """Load and concatenate both sheets of the raw workbook."""
    download()
    sheets = pd.read_excel(config.RAW_XLSX, sheet_name=None, engine="openpyxl")
    df = pd.concat(sheets.values(), ignore_index=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(
        columns={
            "Invoice": "invoice",
            "InvoiceNo": "invoice",
            "Quantity": "quantity",
            "InvoiceDate": "invoice_date",
            "Price": "price",
            "UnitPrice": "price",
            "Country": "country",
        }
    )
    df = df[["invoice", "quantity", "invoice_date", "price", "country"]].copy()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df = df.dropna(subset=["invoice_date", "quantity"])
    # Cancellations have invoice numbers starting with 'C' and negative quantity.
    df["invoice"] = df["invoice"].astype(str)
    df = df[~df["invoice"].str.startswith("C")]
    # Keep genuine sales: positive quantity and non-negative price.
    df = df[(df["quantity"] > 0) & (df["price"] >= 0)]
    return df


def aggregate_daily(clean: pd.DataFrame, country: str | None = None) -> pd.DataFrame:
    """Aggregate cleaned transaction lines to a continuous daily ``units`` series.

    ``country`` selects one country; ``config.REST_OF_WORLD`` selects every country
    except ``config.PRIMARY_COUNTRY``. Gaps inside the observed trading range are
    filled with 0.
    """
    country = country or config.PRIMARY_COUNTRY
    if country == config.REST_OF_WORLD:
        df = clean[clean["country"] != config.PRIMARY_COUNTRY].copy()
    else:
        df = clean[clean["country"] == country].copy()
    if df.empty:
        raise ValueError(f"no transactions for country={country!r}")
    df["date"] = df["invoice_date"].dt.normalize()
    daily = df.groupby("date", as_index=True)["quantity"].sum().rename("units")

    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_idx, fill_value=0.0)
    daily.index.name = "date"
    out = daily.to_frame()
    out["units"] = out["units"].astype(float)
    return out


def build_daily_demand(country: str | None = None) -> pd.DataFrame:
    """Aggregate cleaned transactions to a continuous daily units series.

    Returns a frame with a complete ``date`` DatetimeIndex (gaps filled with 0)
    and a single ``units`` column. The trailing/leading all-zero padding from the
    reindex is trimmed to the observed trading range.
    """
    return aggregate_daily(_clean(load_transactions()), country)


def _cache_path(country: str | None) -> Path:
    if country is None or country == config.PRIMARY_COUNTRY:
        return config.DAILY_CSV
    slug = re.sub(r"[^a-z0-9]+", "_", country.lower()).strip("_")
    return config.PROCESSED_DIR / f"daily_demand_{slug}.csv"


def get_daily_demand(country: str | None = None, rebuild: bool = False) -> pd.DataFrame:
    """Return the daily demand series, caching the result to CSV (one file per country)."""
    config.ensure_dirs()
    path = _cache_path(country)
    if path.exists() and not rebuild:
        return pd.read_csv(path, index_col="date", parse_dates=["date"])
    out = build_daily_demand(country)
    out.to_csv(path)
    return out
