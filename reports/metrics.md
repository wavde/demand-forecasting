# Results

_Python 3.12.10, seed 7, 739 days (2009-12-01 to 2011-12-09)._

**Headline:** SARIMA reaches **MASE 0.741** on a 6-fold rolling-origin backtest (28-day horizon) -- **18.5% lower error than the seasonal-naive baseline**, sMAPE 29.67%.

## Backtested metrics (mean across folds)

| model | sMAPE | RMSE | MASE | MASE improve vs naive |
|---|---|---|---|---|
| SARIMA | 29.67 | 6981.8 | 0.741 | 18.5% |
| ETS | 30.63 | 7174.3 | 0.773 | 15.0% |
| seasonal_naive | 37.73 | 8104.9 | 0.910 | 0.0% |
| LightGBM | 36.77 | 8501.9 | 0.966 | -6.1% |

## Causal bonus -- simulated promo (difference-in-differences)

A known **15%** multiplicative uplift was injected on treated stores in the post period of a synthetic 24-store x 180-day panel and recovered with a two-way fixed-effects DiD (store + date FE): estimate **14.9%** (95% CI [14.2%, 15.6%], p=0). Ground truth inside the interval: **True**.
