"""Forecasting models with a shared ``forecast(history, horizon)`` interface.

- ``ETSModel``    : exponential smoothing (statsmodels), additive trend + weekly season.
- ``SarimaModel`` : SARIMA(1,1,1)(1,1,1)_7, a classical seasonal benchmark.
- ``AutoSarimaModel``: SARIMA with (p,q)(P,Q) re-selected by AIC on each training split.
- ``LightGBMModel``: gradient-boosted trees on lag/rolling/calendar features, forecast
  recursively, with quantile models for prediction intervals.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config
from .features import FEATURE_COLUMNS, make_features

__all__ = ["ETSModel", "SarimaModel", "AutoSarimaModel", "LightGBMModel", "ForecastResult"]


@dataclass
class ForecastResult:
    """Point forecast plus optional prediction interval."""

    mean: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None


class ETSModel:
    """Holt-Winters exponential smoothing with weekly seasonality."""

    name = "ETS"

    def __init__(self, seasonal_periods: int = config.SEASONAL_PERIOD) -> None:
        self.seasonal_periods = seasonal_periods

    def forecast(self, history: pd.Series, horizon: int) -> np.ndarray:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        y = history.astype(float).clip(lower=0) + 1.0  # keep multiplicative season well-defined
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(
                y,
                trend="add",
                seasonal="add",
                seasonal_periods=self.seasonal_periods,
                initialization_method="estimated",
            ).fit()
            pred = fit.forecast(horizon) - 1.0
        return np.clip(np.asarray(pred, dtype=float), 0, None)


class SarimaModel:
    """SARIMA(1,1,1)(1,1,1)_m seasonal ARIMA."""

    name = "SARIMA"

    def __init__(self, seasonal_periods: int = config.SEASONAL_PERIOD) -> None:
        self.m = seasonal_periods

    def _fit(self, history: pd.Series):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SARIMAX(
                history.astype(float),
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, self.m),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)

    def forecast(self, history: pd.Series, horizon: int) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = self._fit(history).forecast(horizon)
        return np.clip(np.asarray(pred, dtype=float), 0, None)

    def forecast_interval(
        self, history: pd.Series, horizon: int, alpha: float = 0.2
    ) -> ForecastResult:
        """Point forecast plus an analytic (1-alpha) prediction interval."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = self._fit(history).get_forecast(horizon)
            mean = np.clip(np.asarray(res.predicted_mean, dtype=float), 0, None)
            ci = res.conf_int(alpha=alpha)
        ci = np.asarray(ci, dtype=float)
        return ForecastResult(
            mean=mean,
            lower=np.clip(ci[:, 0], 0, None),
            upper=np.clip(ci[:, 1], 0, None),
        )


class AutoSarimaModel:
    """SARIMA(p,1,q)(P,1,Q)_m with the orders picked by AIC on each training split.

    Differencing is fixed (d = D = 1) so AIC is comparable across candidates: every
    candidate is fit to the same differenced series. Selection uses only the history
    passed in, never the test window. ``selected`` records every choice made, in order.
    """

    name = "SARIMA-tuned"

    def __init__(
        self,
        seasonal_periods: int = config.SEASONAL_PERIOD,
        p_values: tuple[int, ...] = (0, 1, 2),
        q_values: tuple[int, ...] = (0, 1, 2),
        seasonal_p_values: tuple[int, ...] = (0, 1),
        seasonal_q_values: tuple[int, ...] = (0, 1),
    ) -> None:
        self.m = seasonal_periods
        self.candidates = [
            ((p, 1, q), (sp, 1, sq, seasonal_periods))
            for p, q, sp, sq in itertools.product(
                p_values, q_values, seasonal_p_values, seasonal_q_values
            )
        ]
        self.selected: list[dict] = []

    def select(self, history: pd.Series):
        """Fit every candidate on ``history`` and return the lowest-AIC fit."""
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        y = history.astype(float)
        best, best_aic, best_spec = None, np.inf, None
        for order, seasonal_order in self.candidates:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    fit = SARIMAX(
                        y,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False)
                except (ValueError, np.linalg.LinAlgError):
                    continue
            if np.isfinite(fit.aic) and fit.aic < best_aic:
                best, best_aic, best_spec = fit, float(fit.aic), (order, seasonal_order)
        if best is None:
            raise RuntimeError("no SARIMA candidate could be fit")
        self.selected.append(
            {
                "train_end": str(history.index[-1].date()),
                "n_train": int(len(history)),
                "order": best_spec[0],
                "seasonal_order": best_spec[1][:3],
                "aic": round(best_aic, 2),
            }
        )
        return best

    def forecast(self, history: pd.Series, horizon: int) -> np.ndarray:
        fit = self.select(history)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = fit.forecast(horizon)
        return np.clip(np.asarray(pred, dtype=float), 0, None)


@dataclass
class LightGBMModel:
    """Recursive LightGBM forecaster with quantile prediction intervals."""

    name: str = "LightGBM"
    country: str = "GB"
    lower_alpha: float = 0.1
    upper_alpha: float = 0.9
    params: dict = field(
        default_factory=lambda: {
            "objective": "regression",
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "random_state": config.SEED,
            "verbose": -1,
        }
    )

    def __post_init__(self) -> None:
        self._mean: lgb.LGBMRegressor | None = None
        self._lower: lgb.LGBMRegressor | None = None
        self._upper: lgb.LGBMRegressor | None = None

    def _fit_on(self, history: pd.Series) -> None:
        frame = history.to_frame("units")
        feat = make_features(frame, country=self.country, dropna=True)
        x, y = feat[FEATURE_COLUMNS], feat["units"]

        self._mean = lgb.LGBMRegressor(**self.params).fit(x, y)
        q_params = {k: v for k, v in self.params.items() if k != "objective"}
        self._lower = lgb.LGBMRegressor(
            objective="quantile", alpha=self.lower_alpha, **q_params
        ).fit(x, y)
        self._upper = lgb.LGBMRegressor(
            objective="quantile", alpha=self.upper_alpha, **q_params
        ).fit(x, y)

    def _recursive_paths(self, history: pd.Series, horizon: int) -> ForecastResult:
        assert self._mean is not None and self._lower is not None and self._upper is not None
        series = history.astype(float).copy()
        freq = pd.infer_freq(series.index) or "D"
        future_idx = pd.date_range(series.index[-1], periods=horizon + 1, freq=freq)[1:]

        mean_out, lo_out, hi_out = [], [], []
        for ts in future_idx:
            series.loc[ts] = np.nan
            feat = make_features(series.to_frame("units"), country=self.country, dropna=False)
            x_row = feat.loc[[ts], FEATURE_COLUMNS]
            yhat = float(self._mean.predict(x_row)[0])
            lo = float(self._lower.predict(x_row)[0])
            hi = float(self._upper.predict(x_row)[0])
            yhat = max(yhat, 0.0)
            series.loc[ts] = yhat  # feed the point forecast back in for the next step
            mean_out.append(yhat)
            lo_out.append(max(min(lo, yhat), 0.0))
            hi_out.append(max(hi, yhat))
        return ForecastResult(
            mean=np.array(mean_out), lower=np.array(lo_out), upper=np.array(hi_out)
        )

    def forecast(self, history: pd.Series, horizon: int) -> np.ndarray:
        self._fit_on(history)
        return self._recursive_paths(history, horizon).mean

    def forecast_interval(self, history: pd.Series, horizon: int) -> ForecastResult:
        """Point forecast plus a prediction interval from quantile models."""
        self._fit_on(history)
        return self._recursive_paths(history, horizon)
