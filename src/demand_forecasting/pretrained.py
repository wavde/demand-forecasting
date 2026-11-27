"""Zero-shot pretrained forecaster (Chronos-Bolt) behind the shared ``forecast`` interface.

Optional dependency. Install it with ``pip install -r requirements-pretrained.txt``.
The core pipeline (``run.py``) and the test suite never import torch or chronos, so
the base install and CI stay light.

How the model is used:

- **Zero-shot.** The pretrained weights are loaded once and never updated. Each
  fold passes only that fold's training history as context, so the model never
  sees the test window and is not fine-tuned on anything.
- **Point forecast = median.** Chronos-Bolt outputs quantiles directly. MASE is an
  absolute-error metric, and the median is the optimal point forecast under
  absolute error, so we score the 0.5 quantile.
- **Determinism.** Chronos-Bolt has no sampling step: it is one deterministic
  forward pass from context to quantiles. (The original Chronos, unlike Bolt,
  samples trajectories.) We also fix the torch seed and run on CPU with one
  thread, so float reductions happen in a fixed order and reruns match bit for bit
  on the same machine and library versions.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import metadata

import numpy as np
import pandas as pd

from . import config

__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_REVISION",
    "QUANTILE_LEVELS",
    "ChronosBoltModel",
    "load_chronos_predictor",
    "describe_environment",
]

DEFAULT_MODEL_ID = "amazon/chronos-bolt-small"
# Hugging Face commit the published results were produced with; pinned for reproducibility.
DEFAULT_REVISION = "772f3d25d38aec6d914c8949dab4462e2d46f5d8"
QUANTILE_LEVELS: tuple[float, ...] = (0.1, 0.5, 0.9)

# (context, horizon, quantile_levels) -> array of shape (horizon, len(quantile_levels))
PredictFn = Callable[[np.ndarray, int, Sequence[float]], np.ndarray]


def load_chronos_predictor(
    model_id: str = DEFAULT_MODEL_ID,
    revision: str | None = DEFAULT_REVISION,
    seed: int = config.SEED,
    num_threads: int | None = 1,
) -> PredictFn:
    """Load a Chronos-Bolt pipeline on CPU and return a numpy-in, numpy-out predictor."""
    try:
        import torch
        from chronos import BaseChronosPipeline
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Chronos-Bolt needs the optional dependencies: "
            "pip install -r requirements-pretrained.txt"
        ) from exc

    torch.manual_seed(seed)
    if num_threads:
        torch.set_num_threads(num_threads)
    kwargs: dict = {"device_map": "cpu", "torch_dtype": torch.float32}
    if revision:
        kwargs["revision"] = revision
    pipeline = BaseChronosPipeline.from_pretrained(model_id, **kwargs)

    def predict(context: np.ndarray, horizon: int, quantile_levels: Sequence[float]) -> np.ndarray:
        with torch.no_grad():
            quantiles, _mean = pipeline.predict_quantiles(
                torch.tensor(np.asarray(context, dtype=np.float32)),
                prediction_length=horizon,
                quantile_levels=list(quantile_levels),
            )
        # (batch=1, horizon, n_quantiles) -> (horizon, n_quantiles)
        return quantiles[0].detach().cpu().numpy()

    return predict


@dataclass
class ChronosBoltModel:
    """Chronos-Bolt used zero-shot. ``predict_fn`` can be injected for tests."""

    model_id: str = DEFAULT_MODEL_ID
    revision: str | None = DEFAULT_REVISION
    seed: int = config.SEED
    num_threads: int | None = 1
    predict_fn: PredictFn | None = None
    name: str = "Chronos-Bolt"

    def _predictor(self) -> PredictFn:
        if self.predict_fn is None:
            self.predict_fn = load_chronos_predictor(
                self.model_id, self.revision, self.seed, self.num_threads
            )
        return self.predict_fn

    def forecast_quantiles(self, history: pd.Series, horizon: int) -> np.ndarray:
        """Non-negative quantile forecasts, shape ``(horizon, len(QUANTILE_LEVELS))``."""
        context = history.to_numpy(dtype=float)
        if context.size == 0:
            raise ValueError("empty history")
        q = np.asarray(self._predictor()(context, horizon, QUANTILE_LEVELS), dtype=float)
        expected = (horizon, len(QUANTILE_LEVELS))
        if q.shape != expected:
            raise ValueError(f"predictor returned shape {q.shape}, expected {expected}")
        return np.clip(q, 0, None)

    def forecast(self, history: pd.Series, horizon: int) -> np.ndarray:
        """Median (0.5-quantile) point forecast."""
        return self.forecast_quantiles(history, horizon)[:, QUANTILE_LEVELS.index(0.5)]


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def describe_environment(
    model_id: str = DEFAULT_MODEL_ID, revision: str | None = DEFAULT_REVISION
) -> dict:
    """Package versions and the resolved Hugging Face commit, for the results file."""
    info = {
        "model_id": model_id,
        "requested_revision": revision or "main",
        "resolved_commit": "unknown",
        "chronos-forecasting": _version("chronos-forecasting"),
        "torch": _version("torch"),
        "transformers": _version("transformers"),
    }
    try:
        from huggingface_hub import HfApi

        info["resolved_commit"] = HfApi().model_info(model_id, revision=revision).sha or "unknown"
    except Exception:
        # Offline or hub unavailable: the forecast still runs from the local cache.
        pass
    return info
