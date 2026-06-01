"""End-to-end pipeline: data -> EDA -> backtest -> forecast -> causal -> reports.

Run everything with a single command:

    python run.py

Outputs:
    docs/eda.png, docs/backtest.png, docs/forecast.png, docs/residuals.png
    reports/metrics.json, reports/metrics.md
"""

from __future__ import annotations

import json
import platform
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np  # noqa: E402

from demand_forecasting import causal, config, data, plots, postprocess  # noqa: E402
from demand_forecasting.backtest import backtest, rolling_origin_splits, summarize  # noqa: E402
from demand_forecasting.models import ETSModel, LightGBMModel, SarimaModel  # noqa: E402


def set_seeds(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    set_seeds()
    config.ensure_dirs()

    print("[1/6] Loading daily demand series ...")
    df = data.get_daily_demand()
    series = df["units"]
    print(f"      {len(series)} days, {series.index.min().date()} -> {series.index.max().date()}")

    print("[2/6] EDA plot ...")
    plots.plot_eda(series)

    print("[3/6] Backtesting (rolling-origin) ...")
    models = {
        "ETS": ETSModel().forecast,
        "SARIMA": SarimaModel().forecast,
        "LightGBM": LightGBMModel().forecast,
    }
    results = backtest(series, models)
    summary = summarize(results)
    plots.plot_backtest(summary)
    print(summary.round(3).to_string())

    best = summary.index[0]
    headline = {
        "best_model": best,
        "mase": round(float(summary.loc[best, "mase"]), 3),
        "smape": round(float(summary.loc[best, "smape"]), 2),
        "mase_improve_pct_vs_seasonal_naive": round(
            float(summary.loc[best, "mase_improve_pct"]), 1
        ),
        "smape_improve_pct_vs_seasonal_naive": round(
            float(summary.loc[best, "smape_improve_pct"]), 1
        ),
    }

    print("[4/6] Forecast + intervals on the most recent fold ...")
    last_fold = list(rolling_origin_splits(series))[-1]
    test_idx = last_fold.test.index
    interval_models = {"SARIMA": SarimaModel(), "LightGBM": LightGBMModel()}
    fc_model = interval_models.get(best, SarimaModel())
    fc = fc_model.forecast_interval(last_fold.train, len(last_fold.test))
    mean = postprocess.apply_closed_days(fc.mean, test_idx)
    lower = postprocess.apply_closed_days(fc.lower, test_idx)
    upper = postprocess.apply_closed_days(fc.upper, test_idx)
    plots.plot_forecast(last_fold.train, last_fold.test, mean, lower, upper)
    plots.plot_residuals(last_fold.test, mean)

    print("[5/6] Causal bonus: simulated promo, recovered via DiD ...")
    did = causal.run_promo_study(series, uplift=0.15)
    print(
        f"      true uplift {did.true_uplift:.0%}, "
        f"estimated {did.estimated_uplift:.1%} "
        f"[{did.ci_low:.1%}, {did.ci_high:.1%}], recovered={did.recovered}"
    )

    print("[6/6] Writing reports ...")
    report = {
        "python": platform.python_version(),
        "seed": config.SEED,
        "series": {
            "days": int(len(series)),
            "start": str(series.index.min().date()),
            "end": str(series.index.max().date()),
        },
        "backtest": {
            "horizon": config.HORIZON,
            "n_folds": config.N_FOLDS,
            "seasonal_period": config.SEASONAL_PERIOD,
        },
        "headline": headline,
        "metrics_by_model": summary.round(4).reset_index().to_dict(orient="records"),
        "causal_did": {
            "design": "two-way fixed-effects panel DiD",
            "n_stores": did.n_stores,
            "window_days": did.window,
            "true_uplift": did.true_uplift,
            "estimated_uplift": round(did.estimated_uplift, 4),
            "ci": [round(did.ci_low, 4), round(did.ci_high, 4)],
            "p_value": round(did.p_value, 6),
            "recovered": did.recovered,
        },
    }
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(report, summary)

    print("\n=== HEADLINE ===")
    print(
        f"{headline['best_model']}: MASE {headline['mase']} "
        f"({headline['mase_improve_pct_vs_seasonal_naive']}% lower than seasonal-naive), "
        f"sMAPE {headline['smape']}%"
    )


def _write_markdown(report: dict, summary) -> None:
    h = report["headline"]
    lines = [
        "# Results",
        "",
        f"_Python {report['python']}, seed {report['seed']}, "
        f"{report['series']['days']} days "
        f"({report['series']['start']} to {report['series']['end']})._",
        "",
        f"**Headline:** {h['best_model']} reaches **MASE {h['mase']}** on a "
        f"{report['backtest']['n_folds']}-fold rolling-origin backtest "
        f"({report['backtest']['horizon']}-day horizon) -- "
        f"**{h['mase_improve_pct_vs_seasonal_naive']}% lower error than the "
        f"seasonal-naive baseline**, sMAPE {h['smape']}%.",
        "",
        "## Backtested metrics (mean across folds)",
        "",
        "| model | sMAPE | RMSE | MASE | MASE improve vs naive |",
        "|---|---|---|---|---|",
    ]
    for model, row in summary.iterrows():
        lines.append(
            f"| {model} | {row['smape']:.2f} | {row['rmse']:.1f} | "
            f"{row['mase']:.3f} | {row['mase_improve_pct']:.1f}% |"
        )
    c = report["causal_did"]
    lines += [
        "",
        "## Causal bonus -- simulated promo (difference-in-differences)",
        "",
        f"A known **{c['true_uplift']:.0%}** multiplicative uplift was injected on treated "
        f"stores in the post period of a synthetic {c['n_stores']}-store x {c['window_days']}-day "
        f"panel and recovered with a two-way fixed-effects DiD (store + date FE): estimate "
        f"**{c['estimated_uplift']:.1%}** "
        f"(95% CI [{c['ci'][0]:.1%}, {c['ci'][1]:.1%}], p={c['p_value']:.3g}). "
        f"Ground truth inside the interval: **{c['recovered']}**.",
        "",
    ]
    (config.REPORTS_DIR / "metrics.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
