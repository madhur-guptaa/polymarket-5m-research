"""
polaris/chainlink/collect.py

Forward-only collector for the Chainlink TWAP-60s series.

The relay only ever serves the trailing ~58 seconds (see client.py), so this
polls rather than listens. The poll interval must stay comfortably BELOW the
snapshot depth: at 30s against a ~58s window every second is seen about twice,
which is what closes the gaps a single sampling pass would leave. Measured over
three polls: 130 unique seconds across a 131s span.

Writes are ON CONFLICT DO NOTHING, so the deliberate overlap costs nothing and a
restart can never corrupt what is already stored.
"""

from __future__ import annotations

import asyncio
import logging
import time

from psycopg_pool import AsyncConnectionPool

from polaris.chainlink.client import snapshot
from polaris.core.config import Settings

logger = logging.getLogger(__name__)

_INSERT = """
    INSERT INTO chainlink_twap (symbol, ts, window_s, value, full_accuracy_value)
    VALUES (%s, %s, %s, %s, NULLIF(%s, '')::NUMERIC)
    ON CONFLICT (symbol, ts, window_s) DO NOTHING
"""

_COVERAGE = """
    SELECT symbol, count(*), min(ts), max(ts)
    FROM chainlink_twap
    WHERE window_s = %s AND ts >= %s
    GROUP BY symbol ORDER BY symbol
"""


async def run(pool: AsyncConnectionPool, cfg: Settings, symbols: list[str] | None = None,
              window_s: int = 60, interval_sec: float = 30.0) -> None:
    """Poll the relay forever, upserting every per-second TWAP point it returns."""
    syms = symbols or list(cfg.binance.symbols)
    logger.info("chainlink-collect: %s twap-%ds every %.0fs (snapshot depth ~58s)",
                ",".join(syms), window_s, interval_sec)
    if interval_sec >= 55:
        logger.warning("interval %.0fs is at or beyond the ~58s snapshot depth -- "
                       "this WILL leave gaps; 30s is the tested setting", interval_sec)

    last_report = 0.0
    while True:
        started = time.monotonic()
        rows = await snapshot(syms, window_s=window_s)
        stored = 0
        if rows:
            try:
                async with pool.connection() as conn, conn.cursor() as cur:
                    for row in rows:
                        await cur.execute(_INSERT, row)
                        stored += cur.rowcount or 0
                    await conn.commit()
            except Exception:  # noqa: BLE001
                logger.exception("chainlink-collect: write failed; continuing")
        else:
            logger.warning("chainlink-collect: empty snapshot")

        # Hourly coverage line: the only way a silent degradation (relay
        # changing shape, poll drifting too slow) becomes visible.
        if time.monotonic() - last_report > 3600:
            last_report = time.monotonic()
            try:
                since = int(time.time()) - 3600
                async with pool.connection() as conn, conn.cursor() as cur:
                    await cur.execute(_COVERAGE, (window_s, since))
                    for sym, n, lo, hi in await cur.fetchall():
                        span = max((hi - lo) + 1, 1)
                        logger.info("chainlink-collect: [%s] last hour %d/%d seconds "
                                    "(%.1f%% coverage)", sym, n, span, 100.0 * n / span)
            except Exception:  # noqa: BLE001
                logger.exception("chainlink-collect: coverage report failed")

        logger.debug("chainlink-collect: %d points, %d new", len(rows), stored)
        await asyncio.sleep(max(0.0, interval_sec - (time.monotonic() - started)))
