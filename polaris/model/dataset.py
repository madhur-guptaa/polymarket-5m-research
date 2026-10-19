"""
polaris/model/dataset.py

Load window_observations from PostgreSQL into a FeatureSet, computing derived
features through the shared model/features.py pipeline. Replaces the old
dataset_step1.py (its feature logic now lives in features.py, shared with live
inference).

Only labelled observations are loaded — an unlabelled window is still in
progress and can't be trained on.

WHICH LABEL (this is the important part)
    window_observations.outcome is derived from BINANCE trade prices. Polymarket
    settles on the Chainlink TWAP-60s stream, and since the 2026-08-07 switch
    the two disagree on ~12% of windows -- concentrated almost entirely in the
    flat windows that decide everything (27.9% of windows moving <1bp).

    So the Binance column is a PROXY, and training on it means optimising for an
    outcome nobody pays out on. That mistake stood for two months precisely
    because the proxy was the path of least resistance: it lived in the same
    table, and using the real label required remembering to join
    poly_resolutions by hand.

    Hence the default here is label_source="settlement" -- real settlements
    only, windows without one are DROPPED. Not COALESCE: silently backfilling a
    12%-wrong proxy into the gaps is how this class of bug survives. "proxy" and
    "coalesce" remain available but must be asked for explicitly, and every mode
    logs which label it used and how many windows it discarded.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from psycopg_pool import AsyncConnectionPool

from polaris.core.config import Settings
from polaris.model import features
from polaris.model.base import FeatureSet

logger = logging.getLogger(__name__)

LabelSource = Literal["settlement", "proxy", "coalesce"]

# The label expression and the row filter, per mode. `r.outcome_up` is the real
# settlement; `o.outcome` is the Binance-derived proxy.
_LABEL_SQL: dict[str, tuple[str, str]] = {
    "settlement": ("r.outcome_up", "r.outcome_up IS NOT NULL"),
    "proxy": ("o.outcome", "o.outcome IS NOT NULL"),
    "coalesce": ("COALESCE(r.outcome_up, o.outcome)",
                 "COALESCE(r.outcome_up, o.outcome) IS NOT NULL"),
}

_QUERY = """
    SELECT o.window_start, o.bucket_time, o.time_elapsed_s, o.delta_bps,
           o.cum_imbalance, o.volatility_10s,
           {label} AS outcome,
           o.twap_sofar_bps, o.spot_vs_anchor_bps,
           (r.outcome_up IS NOT NULL) AS is_settled
    FROM window_observations o
    LEFT JOIN poly_resolutions r
           ON r.symbol = o.symbol
          AND r.window_start = EXTRACT(EPOCH FROM o.window_start)::BIGINT
    WHERE o.symbol = %s
      AND o.window_start >= now() - make_interval(days => %s)
      AND {filter}
    ORDER BY o.window_start, o.bucket_time
"""


async def load_dataset(
    pool: AsyncConnectionPool, cfg: Settings, symbol: str, days: int | None = None,
    label_source: LabelSource = "settlement",
) -> FeatureSet | None:
    """Fetch `days` of labelled observations for `symbol` -> FeatureSet.

    label_source defaults to real Polymarket settlements; see the module
    docstring for why the Binance-derived column is not the default.
    """
    days = days or cfg.model.training_window_days
    if label_source not in _LABEL_SQL:
        raise ValueError(f"label_source must be one of {sorted(_LABEL_SQL)}")
    label, filt = _LABEL_SQL[label_source]
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(_QUERY.format(label=label, filter=filt), (symbol, days))
        rows = await cur.fetchall()
    if not rows:
        logger.error("[%s] no labelled observations in the last %d days", symbol, days)
        return None

    n = len(rows)
    ws_epoch = np.empty(n, dtype=np.int64)   # window_start, Unix seconds
    bt_epoch = np.empty(n, dtype=np.float64)  # bucket_time, Unix seconds
    deltas = np.empty(n, dtype=np.float64)
    times = np.empty(n, dtype=np.float64)
    imbalances = np.full(n, np.nan, dtype=np.float64)
    vol_raw = np.full(n, np.nan, dtype=np.float64)
    outcomes = np.empty(n, dtype=np.float64)

    settled = np.zeros(n, dtype=bool)
    twap_sofar = np.full(n, np.nan, dtype=np.float64)
    spot_anchor = np.full(n, np.nan, dtype=np.float64)
    for i, (ws, bt, te, db, ci, v10, oc, tw, sa, is_settled) in enumerate(rows):
        settled[i] = bool(is_settled)
        if tw is not None:
            twap_sofar[i] = float(tw)
        if sa is not None:
            spot_anchor[i] = float(sa)
        ws_epoch[i] = int(ws.timestamp())
        bt_epoch[i] = bt.timestamp()
        deltas[i] = float(db)
        times[i] = float(te)
        if ci is not None:
            imbalances[i] = float(ci)
        if v10 is not None:
            vol_raw[i] = float(v10)
        outcomes[i] = float(oc)

    # Group observations into windows (int labels 0..n_windows-1).
    _, window_ids = np.unique(ws_epoch, return_inverse=True)

    momenta = features.momentum_over_lookback(
        deltas, times, window_ids, cfg.model.momentum_lookback_sec)
    log_vols = features.log_volatility(vol_raw)
    imb = features.clamp_imbalance(imbalances)  # NaN stays NaN
    tod_sin, tod_cos = features.tod_sin_cos(bt_epoch)

    req, locked = features.twap_geometry(twap_sofar, spot_anchor, times,
                                         cfg.aggregate.window_sec)
    fs = FeatureSet(
        deltas=deltas, times=times, momenta=momenta, imbalances=imb,
        log_vols=log_vols, tod_sin=tod_sin, tod_cos=tod_cos, outcomes=outcomes,
        window_ids=window_ids.astype(np.int64), window_start_ts=ws_epoch,
        twap_sofar=twap_sofar, spot_vs_anchor=spot_anchor, req=req, locked=locked,
    )
    up = float(np.mean(outcomes))
    logger.info(
        "[%s] loaded %d obs / %d windows | label=%s "
        "(settlement available for %.0f%% of rows) | "
        "up-rate %.1f%% | NaN imb=%.0f%% vol=%.0f%%",
        symbol, fs.n, fs.n_windows, label_source, 100 * settled.mean(), 100 * up,
        100 * np.isnan(imb).mean(), 100 * np.isnan(log_vols).mean(),
    )
    twap_missing = float(np.isnan(twap_sofar).mean())
    if twap_missing > 0.05:
        # A TWAP-geometry model fitted on this would learn from the NaN-neutral
        # kernel alone -- i.e. from almost nothing -- while still returning
        # confident-looking numbers. Loud, because it would not otherwise show.
        logger.warning(
            "[%s] TWAP geometry missing on %.0f%% of rows -- history predates the "
            "column. Run `polaris recompute-obs --days %d` to fill it.",
            symbol, 100 * twap_missing, days)
    if label_source == "coalesce" and bool((~settled).any()):
        logger.warning(
            "[%s] %d rows (%.2f%%) fell back to the BINANCE-DERIVED proxy label, "
            "which disagrees with settlement on ~12%% of windows. Run "
            "`polaris backfill-resolutions` to close the gap.",
            symbol, int((~settled).sum()), 100 * (~settled).mean())
    elif label_source == "proxy":
        logger.warning("[%s] training on the Binance-derived PROXY label; results "
                       "will not correspond to what Polymarket pays out", symbol)
    return fs
