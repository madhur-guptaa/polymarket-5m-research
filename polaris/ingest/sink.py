"""
polaris/ingest/sink.py

The write target for ingested trades, behind a small Protocol.

Today there is one implementation, PgTradeSink, which bulk-writes straight to
PostgreSQL. The Protocol is the seam the refactor promised (decision D1): if we
ever need multiple consumers of the trade stream, a KafkaSink can be dropped in
here without touching the websocket parsing in binance.py.

TradeRow is a plain tuple in COPY column order — deliberately NOT a Pydantic
model. Ingestion is the hot path; per-record validation would tax throughput
for no benefit, and COPY wants tuples anyway.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Protocol, runtime_checkable

import psycopg
import psycopg.errors
from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)

# COPY column order. Any producer of TradeRow MUST match this exactly —
# a reordering is silent data corruption (price stored as quantity), not an error.
COPY_COLUMNS: tuple[str, ...] = (
    "symbol", "trade_id", "price", "quantity", "is_buyer_mm", "trade_time",
)
# id (IDENTITY) and ingested_at (DEFAULT now()) are intentionally omitted.
COPY_SQL = f"COPY crypto_trades ({', '.join(COPY_COLUMNS)}) FROM STDIN"

# (symbol, trade_id, price_str, quantity_str, is_buyer_mm, trade_time)
TradeRow = tuple[str, int, str, str, bool, datetime]


@runtime_checkable
class Sink(Protocol):
    """A destination for batches of trade rows."""

    async def write(self, rows: list[TradeRow]) -> bool:
        """Write a batch. Returns True on success, False on transient failure
        (which the caller treats as backpressure — retry, do not drop)."""
        ...

    async def close(self) -> None: ...


class PgTradeSink:
    """Writes trade batches to PostgreSQL via COPY, idempotently.

    Fast path: direct ``COPY ... FROM STDIN`` — the common case, no duplicates.
    Slow path: if COPY hits the unique (symbol, trade_id) constraint, fall back
    to COPY-into-temp then INSERT ... ON CONFLICT DO NOTHING, which skips only
    the duplicate rows and keeps the rest. Duplicates only occur on restart /
    replay, so the rare-case cost is acceptable.
    """

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def write(self, rows: list[TradeRow]) -> bool:
        if not rows:
            return True
        try:
            async with self._pool.connection() as conn:
                try:
                    async with conn.cursor() as cur, cur.copy(COPY_SQL) as copy:
                        for row in rows:
                            await copy.write_row(row)
                    await conn.commit()
                    return True
                except psycopg.errors.UniqueViolation:
                    await conn.rollback()
                    return await self._merge_via_temp(conn, rows)
        except psycopg.Error as exc:
            # PG unavailable/slow — signal backpressure so the caller retries.
            logger.error("PG flush failed: %s", exc)
            return False

    async def _merge_via_temp(
        self, conn: psycopg.AsyncConnection, rows: list[TradeRow]
    ) -> bool:
        """COPY into an ON COMMIT DROP temp table, then merge with DO NOTHING."""
        cols = ", ".join(COPY_COLUMNS)
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "CREATE TEMP TABLE IF NOT EXISTS _tmp_trades "
                    "(LIKE crypto_trades INCLUDING DEFAULTS) ON COMMIT DROP"
                )
                await cur.execute("TRUNCATE _tmp_trades")
                async with cur.copy(f"COPY _tmp_trades ({cols}) FROM STDIN") as copy:
                    for row in rows:
                        await copy.write_row(row)
                await cur.execute(
                    f"INSERT INTO crypto_trades ({cols}) "
                    f"SELECT {cols} FROM _tmp_trades "
                    "ON CONFLICT (symbol, trade_id) DO NOTHING"
                )
                inserted = cur.rowcount or 0
            await conn.commit()
            skipped = len(rows) - inserted
            if skipped > 0:
                logger.info("Merge: %d inserted, %d duplicates skipped", inserted, skipped)
            return True
        except psycopg.Error as exc:
            logger.error("Temp-table merge failed: %s", exc)
            await conn.rollback()
            return False

    async def close(self) -> None:
        # The pool is owned by the caller (main.py); nothing to release here.
        return None
