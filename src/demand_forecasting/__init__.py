"""Reproducible daily demand forecasting on UCI Online Retail II.

Modules:
    config      Paths and project constants.
    data        Download + aggregate transactions to a daily demand series.
    metrics     Forecast error metrics (MAPE, sMAPE, RMSE, MASE).
    features    Lag / rolling / calendar feature engineering.
    baselines   Naive and seasonal-naive reference forecasts.
    models      Classical (ETS/SARIMA) and ML (LightGBM) forecasters.
    backtest    Rolling-origin (time-series) cross-validation.
    causal      Simulated promo intervention with a difference-in-differences read.
    plots       EDA and diagnostic figures.
"""

__version__ = "0.1.0"
