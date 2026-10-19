"""
polaris/paper.py

Forward paper-trade recorder. READ-ONLY research: it records what the live
system said so it can be scored later against realized outcomes. There is no
order placement, no credentials, and no execution path anywhere in this
codebase.

Every result in the project so far is a backtest — model trained on past days,
scored on held-out days, against historical quotes. That controls for lookahead
but not for anything about running in real time: staleness of the aggregation
pipeline, live discovery latency, the market state at the instant we'd actually
act. This service is the forward test that does.

IT DELIBERATELY REUSES THE API'S CODE PATH (api.live.compute_tick +
polymarket.discovery + api.edge.compute_edge) rather than recomputing signals
its own way. What gets logged must be exactly what the live system produced —
a parallel implementation would drift, which is the failure this project has
already hit twice (features duplicated between live and calibration; the
heatmap grid and point prediction disagreeing on which Query fields to set).

Scoring is a join, not a stored field: the aggregator writes the resolved label
to window_observations.outcome when a window completes, so
    paper_signals JOIN window_observations ON (symbol, window_start)
gives realized P&L. And because EVERY evaluation is logged (not just the ones
that passed the gate), the forward test can be re-gated offline at any
min_edge / n_eff / trade window — the same way the backtest re-sweeps cached
candidates without recomputing.
"""

from __future__ import annotations

import asyncio
import logging
import time

from psycopg_pool import AsyncConnectionPool

from polaris.api.cache import EstimatorCache
from polaris.api.edge import compute_edge
from polaris.api.live import compute_tick
from polaris.core.config import Settings
from polaris.polymarket.client import SYMBOL_CONFIG, PolymarketClient
from polaris.polymarket.discovery import LiveDiscovery

logger = logging.getLogger(__name__)

_INSERT = """
    INSERT INTO paper_signals (
        symbol, window_start, elapsed_seconds, p_up, n_eff, lower_95, upper_95,
        delta_bps, yes_price, no_price, poly_spread, edge, side,
        in_window, gates_passed, model_kind
    ) VALUES (%s,%s,%s,%s,%s,%s,%s, %s,%s,%s,%s,%s,%s, %s,%s, %s)
    ON CONFLICT (symbol, window_start, elapsed_seconds) DO NOTHING
"""


async def _record_symbol(pool: AsyncConnectionPool, cfg: Settings,
                         cache: EstimatorCache, disc: LiveDiscovery,
                         symbol: str) -> str:
    """Record one live signal. Returns 'ok', 'nodata', or 'stale'."""
    tick = await compute_tick(pool, cfg, cache, symbol)
    # No window data yet, or the model can't score it (cold cache / t=0) —
    # nothing meaningful to log, and a NULL-everywhere row would just be noise.
    if tick.get("window_start") is None or tick.get("prob") is None:
        return "nodata"

    # STALENESS GUARD -- load-bearing, do not remove.
    #
    # compute_tick reads the LATEST window present in window_observations,
    # which is only "current" while the aggregator is keeping up. If ingest or
    # aggregate is stopped or behind, that latest window can be days old, and
    # the model's reading of it then gets paired with TODAY's live market
    # price. Observed while wiring this up: rows captured 2026-09-10 19:44
    # keyed to a window_start of 2026-08-30 19:50, yielding a confident-looking
    # edge of 0.047 from comparing an 11-day-old model state against a current
    # quote. Silent, plausible, and completely meaningless -- so refuse rather
    # than log it. One window of lag is normal (aggregate's safe_lag means the
    # newest complete second can still belong to the previous window).
    now = int(time.time())
    window_sec = cfg.aggregate.window_sec
    current_ws = now - (now % window_sec)
    if tick["window_start"] < current_ws - window_sec:
        return "stale"

    poly = await disc.prices(symbol)
    xs = tick.get("x") or []
    elapsed = xs[-1] if xs else None
    edge = compute_edge(tick["prob"], tick.get("n_eff"), poly, cfg.trading, elapsed)

    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(_INSERT, (
            symbol, tick["window_start"], elapsed,
            tick["prob"], tick.get("n_eff"), tick.get("lower_95"), tick.get("upper_95"),
            tick.get("delta_bps"),
            (poly or {}).get("yes_price"), (poly or {}).get("no_price"),
            (poly or {}).get("spread"),
            (edge or {}).get("edge"), (edge or {}).get("side"),
            (edge or {}).get("in_window"), (edge or {}).get("trade"),
            tick.get("model_kind"),
        ))
        await conn.commit()
    return True


async def run(pool: AsyncConnectionPool, cfg: Settings, stop: asyncio.Event,
              interval: float = 5.0) -> None:
    """Record live signals for every Polymarket-supported symbol until stopped."""
    symbols = [s for s in cfg.binance.symbols if s in SYMBOL_CONFIG]
    if not symbols:
        logger.error("no Polymarket-supported symbols in %s", cfg.binance.symbols)
        return

    cache = EstimatorCache(pool, cfg)
    written = nodata = stale = errors = 0
    async with PolymarketClient(cfg.polymarket) as pm:
        disc = LiveDiscovery(pm, cfg.aggregate.window_sec)
        logger.info("paper-trade recorder started symbols=%s every %.1fs "
                    "(gate: edge>=%.3f n_eff>=%.0f window %d-%ds)",
                    symbols, interval, cfg.trading.min_edge, cfg.trading.min_n_eff,
                    cfg.trading.trade_window_start, cfg.trading.trade_window_end)
        while not stop.is_set():
            t0 = time.monotonic()
            for sym in symbols:
                try:
                    outcome = await _record_symbol(pool, cfg, cache, disc, sym)
                    if outcome == "ok":
                        written += 1
                    elif outcome == "stale":
                        stale += 1
                        if stale in (1, 12) or stale % 240 == 0:
                            logger.warning(
                                "[%s] aggregation pipeline is STALE -- refusing to log "
                                "(is `aggregate` running and caught up?)", sym)
                    else:
                        nodata += 1
                except Exception as exc:  # noqa: BLE001
                    errors += 1
                    if errors <= 5 or errors % 100 == 0:
                        logger.warning("[%s] record failed (#%d): %s", sym, errors, exc)
            if written and written % 500 < len(symbols):
                logger.info("paper-trade: %d logged, %d nodata, %d stale, %d errors",
                            written, nodata, stale, errors)
            try:  # sleep the remainder so cadence doesn't drift
                await asyncio.wait_for(stop.wait(),
                                       timeout=max(0.0, interval - (time.monotonic() - t0)))
            except asyncio.TimeoutError:
                pass
    logger.info("paper-trade stopped: %d logged, %d nodata, %d stale, %d errors",
                written, nodata, stale, errors)
