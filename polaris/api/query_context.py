"""
polaris/api/query_context.py

Build a model Query (momentum/imbalance/log-vol/tod-of-day) from the current
window's latest observation. Shared by api/live.py (the point prediction
behind the signal band) and api/cache.py (the heatmap grid) so both condition
on the SAME live features -- the same D6 hazard features.py's docstring
warns about (derivation logic duplicated between live serving and
calibration), one level up: this is serving-layer orchestration (fetch the
latest obs row, run momentum/log_vol/tod through features.py), not raw
feature math, so it doesn't belong in features.py itself, but it must not
be duplicated between live.py and cache.py either -- hence its own module,
importable by both without a live.py<->cache.py import cycle (live.py
already imports EstimatorCache from cache.py).

delta/time are always 0.0 here -- callers that need a specific (delta, time)
point (the live tick's predict_proba call) override those two fields; the
heatmap grid sweeps them as its two axes and only needs this for the other
four dimensions.
"""

from __future__ import annotations

import math

import numpy as np

from polaris.core.config import Settings
from polaris.model import features
from polaris.model.base import Query


def _finite(x) -> float | None:
    return float(x) if x is not None and math.isfinite(x) else None


def live_held_query(obs: list[dict], cfg: Settings) -> Query:
    """Query(delta=0.0, time=0.0, momentum=..., imbalance=..., log_vol=...,
    tod_sin=..., tod_cos=...) from the current window's observations so far.
    `obs` must be non-empty (callers already gate on load_current_window_obs
    returning rows)."""
    deltas = np.array([r["delta_bps"] for r in obs], dtype=np.float64)
    times = np.array([r["time_elapsed_s"] for r in obs], dtype=np.float64)
    wids = np.zeros(len(obs), dtype=np.int64)  # single window
    momenta = features.momentum_over_lookback(deltas, times, wids, cfg.model.momentum_lookback_sec)

    last = obs[-1]
    mom_last = float(momenta[-1])
    cum_imb = last["cum_imbalance"]  # float | None
    vol_last = last["volatility_10s"]
    log_vol = _finite(features.log_volatility(np.array([vol_last if vol_last is not None else np.nan]))[0])
    ts_sin, ts_cos = features.tod_sin_cos(np.array([last["bucket_time"].timestamp()]))
    tod_sin, tod_cos = _finite(ts_sin[0]), _finite(ts_cos[0])
    imb_q = float(features.clamp_imbalance(np.array([cum_imb]))[0]) if cum_imb is not None else None

    # TWAP geometry: the two stored columns only. req/locked depend on `time`,
    # which callers override, so the model derives them itself at predict time.
    twap_sofar = _finite(last.get("twap_sofar_bps"))
    spot_anchor = _finite(last.get("spot_vs_anchor_bps"))

    return Query(delta=0.0, time=0.0, momentum=mom_last, imbalance=imb_q,
                 log_vol=log_vol, tod_sin=tod_sin, tod_cos=tod_cos,
                 twap_sofar=twap_sofar, spot_vs_anchor=spot_anchor)
