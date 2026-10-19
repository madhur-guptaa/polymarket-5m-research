"""
polaris/polymarket/resolutions.py

Backfill the label that actually matters: how Polymarket settled each window.

WHY THIS EXISTS
    window_observations.outcome is computed from Binance trade prices. On
    2026-09-12 it was measured against Polymarket's own settlement and found to
    disagree on 14.6% of windows -- and the disagreement is concentrated almost
    entirely in near-flat windows (45.7% of windows that moved <1bp), which are
    exactly the hard ones. No Binance-derived rule does better, so this is a
    different price feed, not a convention mismatch. Anything trained or scored
    on the Binance label is answering a question the market does not pay out on.

    See core/schema.sql's poly_resolutions comment for the full measurement.

DESIGN
    Idempotent and resumable: only windows missing from poly_resolutions are
    fetched, so re-running after a rate-limit stop costs nothing. Unresolved and
    non-existent markets are simply not written -- never guessed -- so a window
    absent from this table means "we do not know", which is different from and
    much safer than a wrong label.
"""

from __future__ import annotations

import asyncio
import logging

from psycopg_pool import AsyncConnectionPool

from polaris.core.config import Settings
from polaris.polymarket.client import PolymarketClient

logger = logging.getLogger(__name__)

_MISSING_SQL = """
    SELECT gs.ws
    FROM generate_series(%(start)s::BIGINT, %(end)s::BIGINT, %(step)s::BIGINT) AS gs(ws)
    LEFT JOIN poly_resolutions r ON r.symbol = %(sym)s AND r.window_start = gs.ws
    WHERE r.window_start IS NULL
    ORDER BY gs.ws DESC
"""

_INSERT_SQL = """
    INSERT INTO poly_resolutions (symbol, window_start, outcome_up)
    VALUES (%s, %s, %s)
    ON CONFLICT (symbol, window_start) DO NOTHING
"""


async def backfill(pool: AsyncConnectionPool, cfg: Settings, symbol: str,
                   days: int, concurrency: int = 8) -> dict:
    """Fetch and store settlements for every unknown window in the last `days`.

    Returns counts. `unresolved` covers both markets that have not settled yet
    (the most recent windows always will) and slugs Polymarket never created.
    """
    import time as _time

    step = cfg.aggregate.window_sec
    now = int(_time.time())
    end = now - (now % step) - step          # the current window cannot be settled
    start = end - days * 86400

    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(_MISSING_SQL,
                          {"sym": symbol, "start": start, "end": end, "step": step})
        missing = [int(r[0]) for r in await cur.fetchall()]
    if not missing:
        logger.info("[%s] resolutions already complete for %d days", symbol, days)
        return {"requested": 0, "stored": 0, "unresolved": 0}

    logger.info("[%s] fetching %d missing resolutions", symbol, len(missing))
    sem = asyncio.Semaphore(concurrency)
    stored = unresolved = 0
    lock = asyncio.Lock()

    async with PolymarketClient(cfg.polymarket) as pm:
        async def one(ws: int) -> None:
            nonlocal stored, unresolved
            async with sem:
                out = await pm.resolution(symbol, ws)
            if out is None:
                async with lock:
                    unresolved += 1
                return
            async with lock:
                async with pool.connection() as conn, conn.cursor() as cur:
                    await cur.execute(_INSERT_SQL, (symbol, ws, out))
                    await conn.commit()
                stored += 1
                if stored % 500 == 0:
                    logger.info("[%s] stored %d/%d", symbol, stored, len(missing))

        await asyncio.gather(*(one(ws) for ws in missing))

    logger.info("[%s] done: stored=%d unresolved=%d", symbol, stored, unresolved)
    return {"requested": len(missing), "stored": stored, "unresolved": unresolved}


async def agreement_report(pool: AsyncConnectionPool, symbol: str) -> dict:
    """Compare the Binance-derived label to Polymarket's settlement.

    This is the check that should be run before trusting ANY result computed on
    window_observations.outcome. Reported by move size, because the error is not
    uniform -- it concentrates where the window barely moved.
    """
    sql = """
        WITH w AS (
            SELECT EXTRACT(EPOCH FROM o.window_start)::BIGINT AS ws,
                   MAX(o.outcome) AS ours,
                   MAX(ABS(o.delta_bps)) FILTER (
                       WHERE o.time_elapsed_s = (
                           SELECT MAX(i.time_elapsed_s) FROM window_observations i
                           WHERE i.symbol = o.symbol AND i.window_start = o.window_start)
                   ) AS final_move
            FROM window_observations o
            WHERE o.symbol = %(sym)s AND o.outcome IS NOT NULL
            GROUP BY o.symbol, o.window_start
        )
        SELECT count(*)                                        AS n,
               count(*) FILTER (WHERE w.ours <> r.outcome_up)  AS disagree
        FROM w JOIN poly_resolutions r
          ON r.symbol = %(sym)s AND r.window_start = w.ws
    """
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(sql, {"sym": symbol})
        n, dis = await cur.fetchone()
    return {"symbol": symbol, "n": int(n or 0), "disagree": int(dis or 0),
            "disagree_pct": (100.0 * dis / n) if n else 0.0}
