# demand-forecasting

> Daily demand forecasting on a public retail dataset, evaluated the way a
> production team would: a seasonal-naive baseline, three models of increasing
> sophistication, and a rolling-origin backtest that reports the lift over the
> baseline. Plus a small causal bonus that recovers a known promo uplift.

![CI](https://github.com/wavde/demand-forecasting/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)

## TL;DR

**SARIMA reaches MASE 0.74 on a 6-fold rolling-origin backtest (28-day horizon) —
18.5% lower error than a seasonal-naive baseline, and lower on every metric
(sMAPE 29.7 vs 37.7, RMSE −14%).** A LightGBM with lag/calendar features is
included and, honestly, does *not* beat the classical seasonal models on this
single rigid-weekly series — a finding worth stating rather than hiding. A
two-way fixed-effects difference-in-differences recovers a known +15% promo
uplift as **14.9% (95% CI [14.2%, 15.6%])**.

```text
model            sMAPE     RMSE    MASE   MASE vs naive
SARIMA           29.67   6981.8   0.741        +18.5%
ETS              30.63   7174.3   0.773        +15.0%
seasonal_naive   37.73   8104.9   0.910          0.0%
LightGBM         36.77   8501.9   0.966         −6.1%
```

![backtest](docs/backtest.png)

## Data

[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii):
~1M transaction lines from a UK-based online gift retailer, Dec 2009 – Dec 2011.
Public, **no Kaggle authentication** — the download script pulls it straight from
UCI so the project reproduces from a single command. Raw data is **not** committed
(`data/` is gitignored); `python run.py` downloads and caches it locally.

Transactions are cleaned (cancellations and non-positive quantities dropped),
restricted to the United Kingdom (the dominant market), and aggregated to a
continuous **daily total-units** series of 739 days.

## What the EDA shows

![eda](docs/eda.png)

- **Hard weekly seasonality.** Saturday is a structural zero — the retailer does
  not process orders that day. The model treats "store closed on Saturday" as a
  business rule and forces those forecasts to zero; the seasonal-naive baseline
  inherits the same zero from the prior week, so the comparison stays fair.
- **A Christmas ramp** into November/December, the retailer's peak.
- **Heavy-tailed daily volume** driven by occasional large wholesale orders, which
  is what makes the daily series genuinely hard (daily CV 0.86 vs weekly 0.44).

## Approach

```
data ─▶ EDA ─▶ baseline (seasonal-naive) ─▶ models ─▶ rolling-origin backtest ─▶ report
                                              │
                              ETS · SARIMA · LightGBM(lags+calendar)
```

1. **Baseline.** Seasonal-naive (m = 7) and a moving average. Everything must beat this.
2. **Classical.** Holt-Winters ETS and SARIMA(1,1,1)(1,1,1)₇.
3. **ML.** LightGBM on engineered features — lags (1,2,3,7,14,21,28), rolling
   mean/std/min/max (7,14,28), and calendar/holiday features — forecast
   recursively, with quantile models for prediction intervals. All lag and
   rolling features are shifted ≥1 day, so a row dated *t* never sees *t* or later.
4. **Evaluation.** Time-based **rolling-origin cross-validation**: 6 consecutive
   28-day test windows, each model trained only on its past. No shuffling, no
   leakage. Metrics: MASE (headline, scaled by the in-sample seasonal-naive
   error), sMAPE, RMSE.
5. **Prediction intervals + error analysis** on the most recent fold.
6. **Causal bonus.** Simulate a known +15% promo on treated stores in a synthetic
   panel and recover it with a two-way fixed-effects DiD.

## Forecast and error analysis

![forecast](docs/forecast.png)

The 80% prediction interval comes from the SARIMA state-space model. Residuals are
roughly centered across the test window; the largest misses are the unpredictable
wholesale spikes, not the weekly structure.

![residuals](docs/residuals.png)

## Causal bonus — recovering a promo uplift

There is no labelled promotion in the data, so we *simulate* one and check the
estimator against ground truth. A synthetic 24-store × 180-day panel is built from
the real demand *shape* (shared seasonality, store-level scaling, noise); a known
**+15%** uplift is applied to treated stores in the post period. A two-way
fixed-effects DiD on log demand — store FE absorbing level, date FE absorbing the
common seasonal movement — recovers **14.9% (95% CI [14.2%, 15.6%], p ≈ 0)**.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py        # downloads data, backtests, writes docs/*.png and reports/*
```

Everything is deterministic (`SEED = 7`), developed on **Python 3.12** and tested
on 3.10–3.12. Outputs land in `docs/` (figures) and `reports/` (`metrics.json`,
`metrics.md`).

```bash
pytest -q        # 26 tests: metrics, feature leakage, baselines, backtest, DiD
ruff check .     # lint
```

## Repo layout

```
src/demand_forecasting/
  config.py       paths, seed, horizon, the closed-Saturday business rule
  data.py         download + clean + aggregate to a daily series
  metrics.py      MAPE / sMAPE / RMSE / MASE
  features.py     leakage-safe lag, rolling, calendar/holiday features
  baselines.py    naive, seasonal-naive, moving average
  models.py       ETS, SARIMA, LightGBM (recursive, with intervals)
  backtest.py     rolling-origin CV + summary
  postprocess.py  closed-day business rule
  causal.py       synthetic promo panel + two-way FE DiD
  plots.py        EDA and diagnostic figures
run.py            single-command end-to-end pipeline
tests/            unit tests for the pure functions
```

## What I'd do next

- **Where ML should win:** move from one aggregate series to a *global* model
  across many SKUs/stores. A single rigid-weekly aggregate is exactly the regime
  where classical seasonal models are hard to beat; cross-series learning and
  hierarchical reconciliation are where gradient boosting and global neural
  forecasters earn their keep.
- **Probabilistic evaluation:** score the prediction intervals directly
  (pinball loss, empirical coverage), not just the point forecast.
- **Exogenous drivers:** promotions, price, and weather as features, which also
  makes the causal and forecasting tracks share one feature store.

## License

MIT. All data is public (UCI) and downloaded at runtime; none is committed.
