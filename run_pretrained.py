"""Zero-shot pretrained forecaster (Chronos-Bolt) vs SARIMA on the existing backtest.

One command, after installing the optional extra:

    pip install -r requirements-pretrained.txt
    python run_pretrained.py

Every model, including the pretrained one, is scored on exactly the folds, horizon,
metrics, baselines and closed-day rule used by ``run.py``: 6 rolling-origin folds,
28-day horizon, MASE scaled by the in-sample seasonal-naive error. Chronos-Bolt is
used zero-shot (pretrained weights, no fine-tuning) and only ever sees each fold's
training history.

Beyond the headline table it reports what a team needs to decide when to use it:
a second series (all non-UK countries), fold-by-fold wins, error by lead time, and
error when every model is given only a short training history.

Outputs:
    docs/pretrained_results.md, docs/pretrained_vs_sarima.png
    reports/pretrained_metrics.json
"""

from __future__ import annotations

import json
import platform
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from demand_forecasting import config, data, plots  # noqa: E402
from demand_forecasting.backtest import (  # noqa: E402
    backtest_forecasts,
    head_to_head,
    horizon_profile,
    score_forecasts,
    summarize,
    with_history_limit,
)
from demand_forecasting.models import ETSModel, LightGBMModel, SarimaModel  # noqa: E402
from demand_forecasting.pretrained import (  # noqa: E402
    DEFAULT_MODEL_ID,
    ChronosBoltModel,
    describe_environment,
)

CHALLENGER = "Chronos-Bolt"
INCUMBENT = "SARIMA"
BASELINE = "seasonal_naive"

# Series label -> country selector understood by data.get_daily_demand.
SERIES = {
    "UK": config.PRIMARY_COUNTRY,
    "Rest of world": config.REST_OF_WORLD,
}
PRIMARY = "UK"

# Training history each model is allowed to see in the sweep (None = everything).
# The seasonal-naive baseline only ever uses the last 7 days, so it is unaffected.
HISTORY_LIMITS: tuple[int | None, ...] = (None, 365, 112, 56)

RESULTS_MD = config.DOCS_DIR / "pretrained_results.md"
RESULTS_JSON = config.REPORTS_DIR / "pretrained_metrics.json"
CHART = "pretrained_vs_sarima.png"


def set_seeds(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def _history_label(limit: int | None) -> str:
    return "full" if limit is None else str(limit)


def history_sweep(
    series: pd.Series, chronos: ChronosBoltModel, full_summary: pd.DataFrame
) -> pd.DataFrame:
    """Mean MASE per model when each model sees only the last N days of every train split."""
    rows = {}
    for limit in HISTORY_LIMITS:
        if limit is None:
            mase = full_summary["mase"]
        else:
            models = {
                "ETS": with_history_limit(ETSModel().forecast, limit),
                INCUMBENT: with_history_limit(SarimaModel().forecast, limit),
                CHALLENGER: with_history_limit(chronos.forecast, limit),
            }
            mase = summarize(score_forecasts(backtest_forecasts(series, models)))["mase"]
        rows[_history_label(limit)] = mase
    cols = [BASELINE, INCUMBENT, "ETS", CHALLENGER]
    return pd.DataFrame(rows).T[cols]


def evaluate_series(label: str, country: str, chronos: ChronosBoltModel) -> dict:
    series = data.get_daily_demand(country)["units"]
    start, end = series.index.min().date(), series.index.max().date()
    print(f"  {label}: {len(series)} days, {start} -> {end}")

    models = {
        "ETS": ETSModel().forecast,
        INCUMBENT: SarimaModel().forecast,
        "LightGBM": LightGBMModel().forecast,
        CHALLENGER: chronos.forecast,
    }
    forecasts = backtest_forecasts(series, models)
    scores = score_forecasts(forecasts)
    summary = summarize(scores)
    print(summary.round(3).to_string())

    h2h = head_to_head(scores, CHALLENGER, INCUMBENT)
    windows = forecasts.groupby("fold")["date"].agg(["min", "max"])
    h2h["test_start"] = [str(windows.loc[f, "min"].date()) for f in h2h["fold"]]
    h2h["test_end"] = [str(windows.loc[f, "max"].date()) for f in h2h["fold"]]

    print(f"  {label}: history-length sweep ...")
    history = history_sweep(series, chronos, summary)

    return {
        "label": label,
        "series": series,
        "summary": summary,
        "head_to_head": h2h,
        "horizon": horizon_profile(forecasts),
        "history": history,
    }


def _fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def _conditions(results: list[dict]) -> tuple[list[str], list[str]]:
    """Every (series, slice) cell, split by whether Chronos-Bolt has the lower MASE."""
    wins, losses = [], []
    for r in results:
        summ, hz, hist = r["summary"], r["horizon"], r["history"]
        cells = [
            (
                "full history, all lead times",
                summ.loc[CHALLENGER, "mase"],
                summ.loc[INCUMBENT, "mase"],
            )
        ]
        for bucket in hz.columns:
            cells.append(
                (
                    f"full history, days {bucket} ahead",
                    hz.loc[CHALLENGER, bucket],
                    hz.loc[INCUMBENT, bucket],
                )
            )
        for days in hist.index:
            if days == "full":
                continue
            cells.append(
                (
                    f"last {days} days of history, all lead times",
                    hist.loc[days, CHALLENGER],
                    hist.loc[days, INCUMBENT],
                )
            )
        for desc, c, s in cells:
            line = f"{r['label']}, {desc}: Chronos-Bolt {_fmt(c)} vs SARIMA {_fmt(s)}"
            (wins if c < s else losses).append(line)
    return wins, losses


def write_markdown(results: list[dict], env: dict) -> None:
    lines = [
        "# Zero-shot pretrained forecaster vs SARIMA",
        "",
        "_Generated by `python run_pretrained.py`; do not edit by hand. "
        f"Python {platform.python_version()}, seed {config.SEED}, "
        f"model `{env['model_id']}` (commit `{env['resolved_commit']}`), "
        f"chronos-forecasting {env['chronos-forecasting']}, torch {env['torch']}, "
        f"transformers {env['transformers']}._",
        "",
        "## Setup",
        "",
        f"- **Same backtest as the main pipeline.** {config.N_FOLDS} rolling-origin folds, "
        f"{config.HORIZON}-day horizon, expanding training window, identical folds for "
        "every model. Saturday forecasts are forced to zero for every model (the "
        "retailer does not trade on Saturdays).",
        "- **Metric.** MASE, scaled by each fold's in-sample seasonal-naive (m = 7) "
        "error, averaged across folds. sMAPE and RMSE are reported alongside.",
        "- **Baselines.** Seasonal-naive (m = 7) and SARIMA(1,1,1)(1,1,1)7, the same "
        "fixed specification as the main backtest (not re-tuned per fold). ETS and "
        "LightGBM are included for continuity with the main results.",
        f"- **Pretrained model.** `{env['model_id']}`, zero-shot: pretrained weights, "
        "no fine-tuning, only the fold's training history as context. Point forecast "
        "is the 0.5 quantile (the optimal point forecast under absolute error).",
        "- **Determinism.** Chronos-Bolt predicts quantiles in one forward pass with no "
        "sampling. The run uses CPU, one torch thread and a fixed seed.",
        "- **Series.** `UK` is the daily-units series from the main backtest. "
        "`Rest of world` is every other country combined: smaller and noisier.",
        "",
        "## Headline",
        "",
    ]
    for r in results:
        s = r["summary"]
        c, inc, base = (s.loc[m, "mase"] for m in (CHALLENGER, INCUMBENT, BASELINE))
        h2h = r["head_to_head"]
        rel = (c - inc) / inc
        word = "lower" if c < inc else "higher"
        worst = h2h.loc[h2h["diff"].abs().idxmax()]
        lines.append(
            f"- **{r['label']}:** Chronos-Bolt MASE **{_fmt(c)}** vs SARIMA **{_fmt(inc)}** "
            f"({abs(rel):.1%} {word}); Chronos-Bolt has the lower MASE in "
            f"**{int(h2h['challenger_wins'].sum())} of {len(h2h)} folds**. Median "
            f"per-fold difference {h2h['diff'].median():+.3f}; largest "
            f"{worst['diff']:+.3f} (fold {int(worst['fold'])}, {worst['test_start']} to "
            f"{worst['test_end']}). Seasonal-naive: {_fmt(base)}."
        )

    lines += ["", "## Full-history backtest (mean across folds)", ""]
    for r in results:
        lines += [
            f"### {r['label']}",
            "",
            "| model | sMAPE | RMSE | MASE | MASE vs seasonal-naive |",
            "|---|---|---|---|---|",
        ]
        for model, row in r["summary"].iterrows():
            lines.append(
                f"| {model} | {row['smape']:.2f} | {row['rmse']:.1f} | "
                f"{row['mase']:.3f} | {row['mase_improve_pct']:+.1f}% |"
            )
        lines.append("")

    lines += [
        "## Fold by fold: Chronos-Bolt vs SARIMA (MASE)",
        "",
        "| series | fold | test window | SARIMA | Chronos-Bolt | difference | lower |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        for _, row in r["head_to_head"].iterrows():
            lines.append(
                f"| {r['label']} | {int(row['fold'])} | {row['test_start']} to "
                f"{row['test_end']} | {row['incumbent']:.3f} | {row['challenger']:.3f} | "
                f"{row['diff']:+.3f} | {CHALLENGER if row['challenger_wins'] else INCUMBENT} |"
            )
    lines += [
        "",
        f"With {config.N_FOLDS} folds per series, a win count is descriptive, not a "
        "significance test.",
        "",
        "## MASE by lead time (full history)",
        "",
    ]
    for r in results:
        hz = r["horizon"]
        lines += [
            f"### {r['label']}",
            "",
            "| model | " + " | ".join(f"days {b}" for b in hz.columns) + " |",
            "|---|" + "---|" * len(hz.columns),
        ]
        for model, row in hz.loc[r["summary"].index].iterrows():
            lines.append(f"| {model} | " + " | ".join(_fmt(v) for v in row) + " |")
        lines.append("")

    lines += [
        "## MASE when models see only the last N days of history",
        "",
        "Same folds and test windows; only the context given to each model is "
        "truncated. MASE is still scaled by the full training split, so rows are "
        "comparable. Seasonal-naive only uses the last 7 days, so it does not change.",
        "",
    ]
    for r in results:
        hist = r["history"]
        lines += [
            f"### {r['label']}",
            "",
            "| history (days) | " + " | ".join(hist.columns) + " |",
            "|---|" + "---|" * len(hist.columns),
        ]
        for label, row in hist.iterrows():
            lines.append(f"| {label} | " + " | ".join(_fmt(v) for v in row) + " |")
        lines.append("")

    wins, losses = _conditions(results)
    lines += ["## Where Chronos-Bolt beats SARIMA in this backtest", ""]
    lines += [f"- {w}" for w in wins] or ["- Nowhere."]
    lines += ["", "## Where it does not (SARIMA lower or equal)", ""]
    lines += [f"- {w}" for w in losses] or ["- Nowhere."]

    lines += [
        "",
        "## Limitations",
        "",
        "- One retailer, two daily series, six 28-day test windows each. Differences of "
        "a few percent in MASE are within fold-to-fold noise.",
        "- SARIMA uses one fixed order chosen up front, not a per-fold order search. A "
        "per-fold search could move its numbers in either direction.",
        "- The model's public pretraining corpus is not audited here for overlap with "
        "UCI Online Retail II. Overlap would flatter the pretrained model.",
        "- Point forecasts only. The quantile forecasts are not scored for coverage.",
        "- CPU float results can differ in the last digits across machines or library "
        "versions; the versions used are recorded above.",
        "",
        f"![comparison]({CHART})",
        "",
    ]
    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")


def write_json(results: list[dict], env: dict) -> None:
    def _records(df: pd.DataFrame, index_name: str) -> list[dict]:
        out = df.round(4).copy()
        out.index.name = index_name
        return out.reset_index().to_dict(orient="records")

    report = {
        "python": platform.python_version(),
        "seed": config.SEED,
        "environment": env,
        "backtest": {
            "horizon": config.HORIZON,
            "n_folds": config.N_FOLDS,
            "seasonal_period": config.SEASONAL_PERIOD,
            "closed_dow": list(config.CLOSED_DOW),
            "history_limits": [_history_label(h) for h in HISTORY_LIMITS],
        },
        "series": {
            r["label"]: {
                "days": int(len(r["series"])),
                "start": str(r["series"].index.min().date()),
                "end": str(r["series"].index.max().date()),
                "metrics_by_model": _records(r["summary"], "model"),
                "head_to_head": r["head_to_head"].round(4).to_dict(orient="records"),
                "mase_by_lead_time": _records(r["horizon"], "model"),
                "mase_by_history_days": _records(r["history"], "history_days"),
            }
            for r in results
        },
    }
    RESULTS_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def main() -> None:
    set_seeds()
    config.ensure_dirs()

    print(f"[1/3] Loading {DEFAULT_MODEL_ID} (zero-shot, CPU) ...")
    chronos = ChronosBoltModel()
    env = describe_environment(chronos.model_id, chronos.revision)

    print("[2/3] Backtesting on the shared rolling-origin folds ...")
    results = [evaluate_series(label, country, chronos) for label, country in SERIES.items()]

    print("[3/3] Writing results and chart ...")
    primary = next(r for r in results if r["label"] == PRIMARY)
    order = primary["summary"].index
    mase_by_series = pd.DataFrame({r["label"]: r["summary"]["mase"] for r in results}).loc[order]
    plots.plot_pretrained_comparison(
        mase_by_series,
        primary["horizon"].loc[order],
        primary["history"],
        primary_label=PRIMARY,
        name=CHART,
    )
    write_markdown(results, env)
    write_json(results, env)

    print("\n=== HEADLINE ===")
    for r in results:
        s = r["summary"]
        print(
            f"{r['label']}: Chronos-Bolt MASE {s.loc[CHALLENGER, 'mase']:.3f} vs "
            f"SARIMA {s.loc[INCUMBENT, 'mase']:.3f} vs seasonal-naive "
            f"{s.loc[BASELINE, 'mase']:.3f}"
        )
    print(f"Wrote {RESULTS_MD}, {config.DOCS_DIR / CHART}, {RESULTS_JSON}")


if __name__ == "__main__":
    main()
