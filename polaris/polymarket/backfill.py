"""
polaris/polymarket/backfill.py

Historical Polymarket price backfill -> poly_historical_prices.

For each 5-minute window in the requested range, per symbol:
  1. resolve the market slug -> CLOB token ids (Gamma)
  2. fetch YES and NO tick history concurrently (CLOB)
  3. forward-fill sparse ticks into dense per-second prices
  4. upsert (ON CONFLICT DO NOTHING) — idempotent, re-runs fill only gaps

WHY FORWARD-FILL: the CLOB only records a price when a trade occurs; between
trades the price is the last trade. Step 2's backtest needs a price at every
second, so we sample-and-hold. We emit a second only once BOTH sides have a
known price — emitting a NULL would poison the edge computation downstream.

This table is joined to model predictions on (window_start, elapsed_seconds).
window_start here is epoch seconds and %300==0, which lines up exactly with the
aggregator's date_bin origin (2020-01-01, itself %300==0) — that alignment is
what makes the join land.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from psycopg_pool import AsyncConnectionPool

from polaris.core.config import Settings
from polaris.polymarket.client import SYMBOL_CONFIG, MarketRef, PolymarketClient

logger = logging.getLogger(__name__)

_UPSERT_SQL = """
    INSERT INTO poly_historical_prices
        (symbol, window_start, elapsed_seconds, yes_price, no_price)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (symbol, window_start, elapsed_seconds) DO NOTHING
"""


async def _process_window(
    client: PolymarketClient, pool: AsyncConnectionPool, sem: asyncio.Semaphore,
    symbol: str, window_start: int, window_sec: int,
) -> int:
    """Backfill one window. Returns dense rows written (0 if no market/data)."""
    async with sem:
        try:
            ref: MarketRef | None = await client.resolve_market(symbol, window_start)
            if ref is None:
                return 0
            yes_hist, no_hist = await asyncio.gather(
                client.price_history(ref.token_yes, window_start, window_start + window_sec),
                client.price_history(ref.token_no, window_start, window_start + window_sec),
                return_exceptions=True,
            )
            yes_hist = yes_hist if isinstance(yes_hist, list) else []
            no_hist = no_hist if isinstance(no_hist, list) else []

            rows = _dense_rows(symbol, yes_hist, no_hist, window_start, window_sec)
            if not rows:
                return 0
            async with pool.connection() as conn, conn.cursor() as cur:
                await cur.executemany(_UPSERT_SQL, rows)
                await conn.commit()
            return len(rows)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] window %d failed: %s", symbol, window_start, exc)
            return 0


def _dense_rows(
    symbol: str, yes_hist: list[dict], no_hist: list[dict],
    window_start: int, window_sec: int,
) -> list[tuple]:
    """Build (symbol, window_start, elapsed, yes, no) rows via forward-fill."""
    def by_elapsed(hist: list[dict]) -> dict[int, float]:
        out: dict[int, float] = {}
        for p in hist:
            t, v = p.get("t"), p.get("p")
            if t is None or v is None:
                continue
            elapsed = int(t) - window_start
            if 0 <= elapsed <= window_sec:
                out[elapsed] = float(v)
        return out

    ups_yes, ups_no = by_elapsed(yes_hist), by_elapsed(no_hist)
    rows: list[tuple] = []
    cur_yes: float | None = None
    cur_no: float | None = None
    for elapsed in range(window_sec + 1):
        cur_yes = ups_yes.get(elapsed, cur_yes)
        cur_no = ups_no.get(elapsed, cur_no)
        if cur_yes is not None and cur_no is not None:
            rows.append((symbol, window_start, elapsed, cur_yes, cur_no))
    return rows


async def backfill(
    cfg: Settings, pool: AsyncConnectionPool, *,
    symbols: list[str] | None = None, days: int = 14,
) -> None:
    """Backfill `days` of dense Polymarket prices for the given symbols."""
    window_sec = cfg.aggregate.window_sec
    symbols = symbols or cfg.binance.symbols
    supported = [s for s in symbols if s in SYMBOL_CONFIG]
    skipped = set(symbols) - set(supported)
    if skipped:
        logger.warning("Skipping unsupported Polymarket symbols: %s", sorted(skipped))
    if not supported:
        logger.error("No supported Polymarket symbols in %s", symbols)
        return

    now_ts = int(datetime.now(timezone.utc).timestamp())
    current = now_ts - (now_ts % window_sec)
    start = current - days * 86400
    window_starts = list(range(start, current, window_sec))

    async with PolymarketClient(cfg.polymarket) as client:
        sem = asyncio.Semaphore(cfg.polymarket.backfill_concurrency)
        for symbol in supported:
            logger.info("[%s] backfilling %d windows (%d days)", symbol, len(window_starts), days)
            results = await asyncio.gather(*[
                _process_window(client, pool, sem, symbol, ws, window_sec)
                for ws in window_starts
            ])
            logger.info(
                "[%s] done: %d windows had data, %d dense rows total",
                symbol, sum(1 for r in results if r), sum(results),
            )
    logger.info("Polymarket backfill complete.")
