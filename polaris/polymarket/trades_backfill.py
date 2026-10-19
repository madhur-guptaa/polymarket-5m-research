"""
polaris/polymarket/trades_backfill.py

Backfills Polymarket trade-level fills into poly_trades.

This exists because the original price source was too coarse to backtest on:
/prices-history serves 1-minute resolution, and forward-filling it to
per-second produced prices a mean 12.9c from the real order book inside the
trade gate. data-api's /trades serves individual fills at second resolution and
reaches back 60+ days, which makes the research period re-testable on prices
someone actually paid.

Per window: resolve the slug -> condition_id (Gamma), then page /trades until a
short page arrives, keep the fills that land inside the window, and upsert.
Idempotent via ON CONFLICT DO NOTHING, so re-running fills gaps rather than
duplicating.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from psycopg_pool import AsyncConnectionPool

from polaris.core.config import Settings
from polaris.polymarket.client import SYMBOL_CONFIG, PolymarketClient

logger = logging.getLogger(__name__)

_UPSERT = """
    INSERT INTO poly_trades (symbol, window_start, elapsed_seconds, ts,
                             outcome, side, price, size, tx_hash)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT DO NOTHING
"""


async def _fetch_window(client: PolymarketClient, symbol: str, window_start: int,
                        window_sec: int, page: int = 1000,
                        max_pages: int = 6) -> list[tuple] | None:
    """All fills inside one window as upsert rows, or None if the fetch FAILED.

    None vs [] is load-bearing: [] means the market genuinely had no trades,
    None means we never found out. Conflating them is what made the first run
    report success while losing two thirds of its windows to rate limits.
    """
    ref = await client.resolve_market(symbol, window_start)
    if ref is None or not ref.condition_id:
        return None
    rows: list[tuple] = []
    for p in range(max_pages):
        batch = await client.trades(ref.condition_id, limit=page, offset=p * page)
        if batch is None:
            return None          # rate-limited / failed: do not claim coverage
        if not batch:
            break
        for t in batch:
            try:
                ts = int(t["timestamp"])
                el = ts - window_start
                if not (0 <= el <= window_sec):
                    continue        # the API returns neighbours; keep this window only
                rows.append((symbol, window_start, el, ts,
                             str(t.get("outcome", ""))[:8], str(t.get("side", ""))[:4],
                             float(t["price"]), float(t.get("size") or 0.0),
                             str(t.get("transactionHash", ""))[:80]))
            except (KeyError, TypeError, ValueError):
                continue
        if len(batch) < page:
            break               # short page => end of history for this market
    return rows


async def _worker(client: PolymarketClient, pool: AsyncConnectionPool,
                  sem: asyncio.Semaphore, symbol: str, window_start: int,
                  window_sec: int) -> tuple[int, bool]:
    """-> (fills_written, failed)."""
    async with sem:
        try:
            rows = await _fetch_window(client, symbol, window_start, window_sec)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] window %d failed: %s", symbol, window_start, exc)
            return 0, True
        if rows is None:
            return 0, True
        if not rows:
            return 0, False
        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.executemany(_UPSERT, rows)
            await conn.commit()
        return len(rows), False


async def backfill(cfg: Settings, pool: AsyncConnectionPool, *,
                   symbols: list[str] | None = None, days: int = 30,
                   concurrency: int | None = None) -> None:
    """Backfill `days` of trade-level fills for the given symbols."""
    window_sec = cfg.aggregate.window_sec
    symbols = [s for s in (symbols or cfg.binance.symbols) if s in SYMBOL_CONFIG]
    if not symbols:
        logger.error("no Polymarket-supported symbols requested")
        return
    # data-api throttles hard; default well below the generic backfill concurrency.
    conc = concurrency or 3

    now = int(datetime.now(timezone.utc).timestamp())
    current = now - (now % window_sec)
    starts = list(range(current - days * 86400, current, window_sec))

    async with PolymarketClient(cfg.polymarket) as client:
        sem = asyncio.Semaphore(conc)
        for symbol in symbols:
            logger.info("[%s] trade backfill: %d windows (%d days, concurrency %d)",
                        symbol, len(starts), days, conc)
            done = written = failed = 0
            # Chunk so progress is visible and memory stays flat.
            for i in range(0, len(starts), 200):
                chunk = starts[i:i + 200]
                res = await asyncio.gather(*[
                    _worker(client, pool, sem, symbol, ws, window_sec) for ws in chunk])
                done += len(chunk)
                written += sum(n for n, _ in res)
                failed += sum(1 for _, f in res if f)
                logger.info("[%s] %d/%d windows, %d fills, %d FAILED",
                            symbol, done, len(starts), written, failed)
            # Report coverage rather than just success -- a backfill that lost
            # windows to rate limits must not look like a completed one.
            logger.info("[%s] trade backfill done: %d fills over %d windows; "
                        "%d windows FAILED (%.1f%%) -- re-run to fill those",
                        symbol, written, len(starts) - failed, failed,
                        100.0 * failed / max(1, len(starts)))
