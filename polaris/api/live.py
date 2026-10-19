"""
polaris/api/live.py

Assemble the live "tick" for the current window: the delta path, the six
features, and the Bayesian P(up) with credible interval.

Features come from the current window's window_observations (what the aggregator
already computed) and are derived through the SAME model/features.py used in
training — so live and training features cannot drift apart.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import numpy as np

from polaris.api.cache import EstimatorCache
from polaris.api.query_context import live_held_query
from polaris.core.config import Settings
from polaris.core.db import load_current_window_obs, load_latest_trades
from polaris.model import features

logger = logging.getLogger(__name__)


async def compute_tick(pool, cfg: Settings, cache: EstimatorCache, symbol: str) -> dict[str, Any]:
    """Full live state for `symbol`. status: waiting | ok."""
    trades = await load_latest_trades(pool, symbol, limit=12)
    obs = await load_current_window_obs(pool, symbol)
    current_price = trades[0]["price"] if trades else None

    base = {
        "status": "waiting", "symbol": symbol, "x": [], "y": [],
        "current_price": current_price, "prob": None, "lower_95": None,
        "upper_95": None, "n_eff": None, "cum_imbalance": None, "delta_bps": None,
        "momentum_bps": None, "log_volatility": None, "tod_sin": None, "tod_cos": None,
        "flow_history": [], "bandwidths": None, "trades": trades,
        "window_start": None, "model_kind": cfg.model.kind,
        "twap_sofar_bps": None, "spot_vs_anchor_bps": None, "req_bps": None,
    }
    if not obs:
        return base

    deltas = np.array([r["delta_bps"] for r in obs], dtype=np.float64)
    times = np.array([r["time_elapsed_s"] for r in obs], dtype=np.float64)
    last = obs[-1]
    delta_last = float(last["delta_bps"])
    time_last = int(last["time_elapsed_s"])
    cum_imb = last["cum_imbalance"]  # float | None
    # momentum/imbalance/log_vol/tod, shared with cache.py's heatmap grid so
    # both condition on the same live features (see query_context.py).
    held = live_held_query(obs, cfg)

    flow_history = [
        {"t": int(r["time_elapsed_s"]), "imbalance": round(r["cum_imbalance"], 4)}
        for r in obs if r["cum_imbalance"] is not None
    ]

    prob = lo = hi = n_eff = None
    bandwidths = None
    st = await cache.get(symbol)
    if st is not None:
        bandwidths = st.grid["bandwidths"]
        if time_last > 0:
            pred = st.model.predict_proba(dataclasses.replace(held, delta=delta_last, time=float(time_last)))
            prob, lo, hi, n_eff = (round(pred.p_up, 4), round(pred.lo, 4),
                                   round(pred.hi, 4), round(pred.n_eff, 2))

    # The live path is drawn over the model's surface, so it must use the SAME
    # vertical coordinate: the TWAP model's surface is price vs the TWAP
    # anchor, not vs the window's opening VWAP (they correlate ~0.9 but are
    # not the same point, and a path in one over a surface in the other lands
    # the "now" dot in the wrong cell).
    spots = [r.get("spot_vs_anchor_bps") for r in obs]
    if cfg.model.kind == "kde_twap" and all(v is not None for v in spots):
        path_y, y_axis = [round(float(v), 2) for v in spots], "price vs TWAP anchor (bps)"
    else:
        path_y, y_axis = [round(d, 2) for d in deltas], "δ vs window open (bps)"

    # window_start as epoch seconds: bucket_time minus its own elapsed offset.
    # Matches the BIGINT convention of poly_historical_prices / paper_signals,
    # so a signal can be joined to market prices and to the resolved outcome.
    window_start = int(last["bucket_time"].timestamp()) - time_last

    # The contract's own state: where the running average sits vs the anchor
    # and how far the rest of the window has to move to change the answer.
    req_bps = None
    if held.twap_sofar is not None and held.spot_vs_anchor is not None:
        r, _ = features.twap_geometry(np.array([held.twap_sofar]),
                                      np.array([held.spot_vs_anchor]),
                                      np.array([float(time_last)]),
                                      cfg.aggregate.window_sec)
        req_bps = round(float(r[0]), 2) if np.isfinite(r[0]) else None

    return {
        "status": "ok", "symbol": symbol, "window_start": window_start,
        "x": [int(t) for t in times], "y": path_y, "y_axis": y_axis,
        "current_price": current_price,
        "prob": prob, "lower_95": lo, "upper_95": hi, "n_eff": n_eff,
        "cum_imbalance": round(cum_imb, 4) if cum_imb is not None else None,
        "delta_bps": round(delta_last, 2), "momentum_bps": round(held.momentum, 2),
        "log_volatility": round(held.log_vol, 4) if held.log_vol is not None else None,
        "tod_sin": round(held.tod_sin, 4) if held.tod_sin is not None else None,
        "tod_cos": round(held.tod_cos, 4) if held.tod_cos is not None else None,
        "flow_history": flow_history, "bandwidths": bandwidths, "trades": trades,
        "model_kind": cfg.model.kind,
        "twap_sofar_bps": round(held.twap_sofar, 2) if held.twap_sofar is not None else None,
        "spot_vs_anchor_bps": (round(held.spot_vs_anchor, 2)
                               if held.spot_vs_anchor is not None else None),
        "req_bps": req_bps,
    }
