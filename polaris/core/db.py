"""
polaris/core/db.py

PostgreSQL access for the whole application: one async connection pool
(psycopg3) plus schema bootstrapping. Query helpers and the estimator cache
land here in later sessions; for now this is what ingestion needs.

We stay on psycopg3 (not asyncpg) — it already gives us async + the COPY
protocol, so there is no reason to take on a driver migration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from psycopg.rows import dict_row, tuple_row
from psycopg_pool import AsyncConnectionPool

from polaris.core.config import PostgresSettings

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


async def open_pool(
    pg: PostgresSettings,
    *,
    autocommit: bool = False,
    options: str | None = None,
) -> AsyncConnectionPool:
    """Open and return a ready async connection pool.

    autocommit=False is the default because COPY runs inside an explicit
    transaction. Read-only callers can pass autocommit=True. `options` sets
    libpq per-connection options, e.g. "-c synchronous_commit=off" for the
    batch writer.
    """
    kwargs: dict = {"autocommit": autocommit, "row_factory": tuple_row}
    if options:
        kwargs["options"] = options
    pool = AsyncConnectionPool(
        conninfo=pg.dsn,
        min_size=pg.pool_min,
        max_size=pg.pool_max,
        kwargs=kwargs,
        open=False,
    )
    await pool.open(wait=True, timeout=15.0)
    logger.info(
        "PG pool ready host=%s db=%s (min=%d max=%d)",
        pg.host, pg.database, pg.pool_min, pg.pool_max,
    )
    return pool


async def ensure_schema(pool: AsyncConnectionPool) -> None:
    """Apply schema.sql idempotently (every statement is IF NOT EXISTS)."""
    ddl = SCHEMA_PATH.read_text(encoding="utf-8")
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(ddl)
        await conn.commit()
    logger.info("Schema ensured from %s", SCHEMA_PATH.name)


# ── read queries for the serving layer ───────────────────────────────────────

async def _rows(pool: AsyncConnectionPool, sql: str, params: tuple) -> list[dict[str, Any]]:
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(sql, params)
        return await cur.fetchall()


async def load_current_window_obs(pool: AsyncConnectionPool, symbol: str) -> list[dict[str, Any]]:
    """The latest window's per-second observations (current window, outcome may be NULL).

    Reads what the aggregator already computed, so the live features match the
    training features exactly. Ordered by elapsed second.
    """
    return await _rows(pool, """
        WITH latest AS (
            SELECT window_start FROM window_observations
            WHERE symbol = %s ORDER BY window_start DESC LIMIT 1
        )
        SELECT wo.time_elapsed_s, wo.bucket_time, wo.delta_bps,
               wo.cum_imbalance, wo.cum_volume, wo.volatility_10s,
               wo.twap_sofar_bps, wo.spot_vs_anchor_bps
        FROM window_observations wo CROSS JOIN latest
        WHERE wo.symbol = %s AND wo.window_start = latest.window_start
        ORDER BY wo.time_elapsed_s
    """, (symbol, symbol))


async def load_latest_trades(pool: AsyncConnectionPool, symbol: str, limit: int = 20) -> list[dict[str, Any]]:
    """Most recent trades (price/qty/time), newest first — for the order-flow panel."""
    rows = await _rows(pool, """
        SELECT price::float8 AS price, quantity::float8 AS quantity, trade_time
        FROM crypto_trades WHERE symbol = %s ORDER BY trade_time DESC LIMIT %s
    """, (symbol, limit))
    for r in rows:
        r["trade_time"] = r["trade_time"].isoformat()
    return rows


async def load_10s_volume(pool: AsyncConnectionPool, symbol: str) -> list[dict[str, Any]]:
    """10-second volume bars over the last 5 minutes (micro volume profile)."""
    rows = await _rows(pool, """
        SELECT date_bin('10 seconds', bucket_time, '2020-01-01 00:00:00+00'::timestamptz) AS bucket_time,
               SUM(volume)::float8 AS volume
        FROM crypto_1s_bars
        WHERE symbol = %s AND bucket_time >= now() - interval '5 minutes'
        GROUP BY 1 ORDER BY 1
    """, (symbol,))
    for r in rows:
        r["bucket_time"] = r["bucket_time"].isoformat()
    return rows


async def load_1h_trend(pool: AsyncConnectionPool, symbol: str) -> list[dict[str, Any]]:
    """1-minute average price over the last hour (macro trend)."""
    rows = await _rows(pool, """
        SELECT date_bin('1 minute', bucket_time, '2020-01-01 00:00:00+00'::timestamptz) AS minute_time,
               AVG(vwap)::float8 AS avg_price
        FROM crypto_1s_bars
        WHERE symbol = %s AND bucket_time >= now() - interval '1 hour'
        GROUP BY 1 ORDER BY 1
    """, (symbol,))
    for r in rows:
        r["minute_time"] = r["minute_time"].isoformat()
    return rows
