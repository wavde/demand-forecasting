"""A small causal/uplift bonus: recover a promotion's effect with difference-in-differences.

The public dataset has no labelled promotion, so we *simulate* a known multiplicative
uplift on a treated set of stores and then recover it — the same recover-the-truth
discipline used across the other case-study repos.

We build a synthetic multi-store panel from the real demand *shape*: every store shares
the observed daily pattern (so seasonality and trend are common), scaled by a store-level
factor plus noise. A known uplift is applied to treated stores in the post period. A
two-way fixed-effects DiD (store + date dummies) then differences out everything shared
and recovers the uplift. Store FE absorbs level differences; date FE absorbs the common
seasonal/holiday movement, so the parallel-trends assumption holds by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from . import config


@dataclass
class DiDResult:
    true_uplift: float
    estimated_uplift: float
    ci_low: float
    ci_high: float
    p_value: float
    n_stores: int
    window: int

    @property
    def recovered(self) -> bool:
        """True if the known uplift falls inside the 95% confidence interval."""
        return self.ci_low <= self.true_uplift <= self.ci_high


def simulate_panel(
    series: pd.Series,
    uplift: float = 0.15,
    n_stores: int = 24,
    window: int = 180,
    noise: float = 0.05,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Construct a store x day panel with a known promo applied to treated stores.

    Half the stores are treated; the promo turns on at the midpoint of the window.
    Returns long-format columns: store, date, treated, post, did, units.
    """
    rng = np.random.default_rng(seed)
    base = np.clip(series.iloc[-window:].to_numpy(dtype=float), 1.0, None)
    dates = pd.date_range("2025-01-01", periods=window, freq="D")
    cutoff = window // 2
    post = (np.arange(window) >= cutoff).astype(int)
    treated_ids = set(range(n_stores // 2))

    frames = []
    for store in range(n_stores):
        level = rng.uniform(0.5, 1.5)
        eps = rng.normal(1.0, noise, size=window)
        units = base * level * eps
        treated = int(store in treated_ids)
        units = units * (1 + uplift * treated * post)
        frames.append(
            pd.DataFrame(
                {
                    "store": store,
                    "date": dates,
                    "treated": treated,
                    "post": post,
                    "units": units,
                }
            )
        )
    panel = pd.concat(frames, ignore_index=True)
    panel["did"] = panel["treated"] * panel["post"]
    return panel


def estimate_did(panel: pd.DataFrame, true_uplift: float) -> DiDResult:
    """Two-way fixed-effects DiD on log demand.

    Modelling log(units) makes the interaction coefficient a multiplicative uplift
    (exp(beta) - 1), matching how the lift was injected.
    """
    df = panel.copy()
    df["log_units"] = np.log(df["units"])
    model = smf.ols("log_units ~ did + C(store) + C(date)", data=df).fit()

    beta = model.params["did"]
    ci = model.conf_int().loc["did"]
    return DiDResult(
        true_uplift=true_uplift,
        estimated_uplift=float(np.expm1(beta)),
        ci_low=float(np.expm1(ci[0])),
        ci_high=float(np.expm1(ci[1])),
        p_value=float(model.pvalues["did"]),
        n_stores=int(df["store"].nunique()),
        window=int(df["date"].nunique()),
    )


def run_promo_study(series: pd.Series, uplift: float = 0.15) -> DiDResult:
    """End-to-end: simulate a known promo uplift across stores and recover it via DiD."""
    panel = simulate_panel(series, uplift=uplift)
    return estimate_did(panel, true_uplift=uplift)
