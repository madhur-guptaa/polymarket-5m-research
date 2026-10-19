"""
polaris/api/cache.py

Per-symbol live estimator cache for the serving layer.

Fitting the KDE on the rolling training window is ~10-70ms, far too slow to do
per request/tick. So we fit once per symbol, cache the fitted model plus the
precomputed heatmap grid, and re-fit lazily when the entry ages past
model.estimator_refresh_sec. The live tick path then just calls predict_proba
(<1ms) against the cached model.

The model is built from config bandwidths (defaults are fine — calibration only
sharpens them; the model serves predictions either way).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import numpy as np
from psycopg_pool import AsyncConnectionPool

from polaris.api.query_context import live_held_query
from polaris.core.config import Settings
from polaris.core.db import load_current_window_obs
from polaris.model.base import FeatureSet, ProbabilityModel, Query
from polaris.model.dataset import load_dataset
from polaris.model.factory import build_model

logger = logging.getLogger(__name__)

# Heatmap grid axes (match the frontend surface): delta in bps, time in seconds.
_DELTA_RANGE = np.linspace(-30.0, 30.0, 61)
_TIME_RANGE = np.arange(0, 301, 5, dtype=np.float64)


@dataclass
class EstimatorState:
    model: ProbabilityModel
    fs: FeatureSet
    grid: dict            # {"x": time, "y": delta, "z": p_up grid}
    fitted_at: float      # monotonic seconds


class EstimatorCache:
    """Lazily-refreshed estimator per symbol (whatever MODEL_KIND builds)."""

    def __init__(self, pool: AsyncConnectionPool, cfg: Settings) -> None:
        self._pool = pool
        self._cfg = cfg
        self._states: dict[str, EstimatorState] = {}

    async def get(self, symbol: str) -> EstimatorState | None:
        """Return a fitted estimator for `symbol`, refitting if stale/missing."""
        st = self._states.get(symbol)
        if st is not None and (time.monotonic() - st.fitted_at) < self._cfg.model.estimator_refresh_sec:
            return st

        # Independent queries -- fetch concurrently rather than back-to-back.
        fs, obs = await asyncio.gather(
            load_dataset(self._pool, self._cfg, symbol),
            load_current_window_obs(self._pool, symbol),
        )
        if fs is None or fs.n == 0:
            return None

        # Through the factory, so MODEL_KIND decides what serves -- this used to
        # construct KDEModel directly and silently ignore any other model.
        model = build_model(self._cfg).fit(fs)
        # Condition the grid on the CURRENT window's momentum/imbalance/vol/tod
        # (shared with live.py's point prediction via query_context.py) rather
        # than leaving them unset -- an unset Query field means predict_grid's
        # `shared` multiplier skips that kernel entirely, i.e. the surface
        # silently averages over all regimes instead of showing the live one.
        # Falls back to the delta/time-only stub when there's no current
        # window yet (e.g. right after a symbol's first-ever window opens).
        held = live_held_query(obs, self._cfg) if obs else Query(delta=0.0, time=0.0)
        z = model.predict_grid(_DELTA_RANGE, _TIME_RANGE, held)
        grid = {
            "x": _TIME_RANGE.tolist(),
            "y": [round(d, 1) for d in _DELTA_RANGE.tolist()],
            "z": z.tolist(),
            "n_observations": fs.n,
            "n_windows": fs.n_windows,
            "bandwidths": model.bw.as_dict() if hasattr(model, "bw") else None,
            "model_kind": self._cfg.model.kind,
            "y_axis": ("price vs TWAP anchor (bps)" if self._cfg.model.kind == "kde_twap"
                       else "δ vs window open (bps)"),
        }
        st = EstimatorState(model=model, fs=fs, grid=grid, fitted_at=time.monotonic())
        self._states[symbol] = st
        logger.info("[%s] estimator (re)fitted: %d obs / %d windows", symbol, fs.n, fs.n_windows)
        return st
