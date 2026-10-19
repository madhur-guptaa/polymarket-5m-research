"""
polaris/polymarket/bookcollect.py

Samples the live Polymarket CLOB order book into poly_book_snapshots.

WHY A SERVICE AND NOT A BACKFILL: Polymarket publishes last-TRADE price
history (/prices-history) but no historical order book, so execution cost
cannot be backfilled — it can only be accumulated forward. Every spread number
in the research so far is a swept assumption; this is what turns it into a
measurement.

WHAT IT RECORDS, per sample, for both the YES and NO token of the currently
active market:
  * best bid / best ask / spread / mid — the touch
  * last_trade_price, captured in the SAME payload as the book. This is the
    one that matters: the backtest fills at the historical last-trade price,
    so the correction it needs is (best_ask - last_trade), the cost of
    crossing from where the tape printed to where you could actually buy.
    NOTE the API trap this works around: last_trade_price is MARKET-level, so
    the YES and NO tokens return the SAME number, denominated in whichever
    side traded last. It is normalized per side here (see _normalize_last).
  * eff_ask_{1,10,100,1000} — size-weighted average fill price walking the ask
    side for a $N order, NULL if the book is too thin to fill it. The touch
    price is only correct for an infinitesimal order and the touch is often
    thin (observed: 5 contracts at the best ask).
  * the top N levels per side, raw, so later analysis isn't restricted to the
    order sizes chosen today.

Cost: 2 requests per symbol per interval (default 5s) — ~0.4 req/s against a
90 req/s CLOB budget, so it runs comfortably alongside everything else.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from psycopg_pool import AsyncConnectionPool

from polaris.core.config import Settings
from polaris.polymarket.client import SYMBOL_CONFIG, PolymarketClient
from polaris.polymarket.discovery import LiveDiscovery

logger = logging.getLogger(__name__)

ORDER_SIZES = (1, 10, 100, 1000)   # dollars

_UPSERT = """
    INSERT INTO poly_book_snapshots (
        symbol, window_start, elapsed_seconds, side,
        best_bid, best_ask, spread, mid, last_trade, last_trade_raw,
        eff_ask_1, eff_ask_10, eff_ask_100, eff_ask_1000,
        bid_depth, ask_depth, levels
    ) VALUES (%s,%s,%s,%s, %s,%s,%s,%s,%s,%s, %s,%s,%s,%s, %s,%s,%s)
    ON CONFLICT (symbol, window_start, elapsed_seconds, side) DO NOTHING
"""


def _levels(raw: list[dict] | None) -> list[tuple[float, float]]:
    """[{price,size},...] -> [(price,size),...] sorted BEST FIRST.

    The API returns both sides worst-to-best; normalising here means callers
    never have to remember which end the touch is on.
    """
    out = []
    for lv in raw or []:
        try:
            out.append((float(lv["price"]), float(lv["size"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def effective_ask(asks_best_first: list[tuple[float, float]], budget: float) -> float | None:
    """Average price paid per contract, walking the asks to spend `budget`.

    None when the visible book cannot absorb the order — a real constraint,
    not a missing value, so it must not be silently treated as zero cost.
    """
    remaining, contracts = budget, 0.0
    for price, size in asks_best_first:
        if price <= 0:
            continue
        level_cost = price * size
        if remaining >= level_cost:
            remaining -= level_cost
            contracts += size
        else:
            contracts += remaining / price
            remaining = 0.0
            break
    if remaining > 1e-9 or contracts <= 0:
        return None
    return budget / contracts


def _normalize_last(raw: float | None, mid: float | None) -> float | None:
    """Express a market-level last_trade_price in THIS token's denomination.

    /book returns one market-level last trade for both tokens, denominated in
    whichever side traded most recently, so the raw value is in the wrong
    frame roughly half the time. The pair is complementary, so the correct
    value is either raw or 1-raw; pick whichever is consistent with this
    token's own book. Near 0.50 the two candidates converge, so an ambiguous
    pick there is also a harmless one.
    """
    if raw is None:
        return None
    if mid is None:
        return raw
    return raw if abs(raw - mid) <= abs((1.0 - raw) - mid) else 1.0 - raw


def summarize(book: dict, keep_levels: int) -> dict | None:
    """Book payload -> the row we store. None if the book has no usable side."""
    bids = sorted(_levels(book.get("bids")), key=lambda x: -x[0])   # best = highest
    asks = sorted(_levels(book.get("asks")), key=lambda x: x[0])    # best = lowest
    if not bids and not asks:
        return None
    best_bid = bids[0][0] if bids else None
    best_ask = asks[0][0] if asks else None
    spread = (best_ask - best_bid) if (best_bid is not None and best_ask is not None) else None
    mid = ((best_ask + best_bid) / 2) if spread is not None else None
    try:
        last_trade = float(book["last_trade_price"])
    except (KeyError, TypeError, ValueError):
        last_trade = None
    return {
        "best_bid": best_bid, "best_ask": best_ask, "spread": spread, "mid": mid,
        "last_trade": _normalize_last(last_trade, mid), "last_trade_raw": last_trade,
        "eff": [effective_ask(asks, float(n)) for n in ORDER_SIZES],
        "bid_depth": sum(s for _, s in bids),
        "ask_depth": sum(s for _, s in asks),
        "levels": json.dumps({"bids": bids[:keep_levels], "asks": asks[:keep_levels]}),
    }


async def _sample_symbol(pm: PolymarketClient, disc: LiveDiscovery,
                         pool: AsyncConnectionPool, symbol: str,
                         window_sec: int, keep_levels: int) -> int:
    ref = await disc.active_market(symbol)
    if ref is None:
        return 0
    now = time.time()
    window_start = int(now) - (int(now) % window_sec)
    elapsed = int(now) - window_start

    books = await asyncio.gather(pm.book(ref.token_yes), pm.book(ref.token_no),
                                 return_exceptions=True)
    rows = []
    for side, bk in zip(("YES", "NO"), books):
        if not isinstance(bk, dict):
            continue
        s = summarize(bk, keep_levels)
        if s is None:
            continue
        rows.append((symbol, window_start, elapsed, side,
                     s["best_bid"], s["best_ask"], s["spread"], s["mid"],
                     s["last_trade"], s["last_trade_raw"],
                     *s["eff"], s["bid_depth"], s["ask_depth"], s["levels"]))
    if not rows:
        return 0
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.executemany(_UPSERT, rows)
        await conn.commit()
    return len(rows)


async def run(pool: AsyncConnectionPool, cfg: Settings, stop: asyncio.Event) -> None:
    """Sample the book for every supported symbol until `stop` is set."""
    symbols = [s for s in cfg.binance.symbols if s in SYMBOL_CONFIG]
    if not symbols:
        logger.error("no Polymarket-supported symbols in %s", cfg.binance.symbols)
        return
    interval = cfg.polymarket.book_interval_sec
    keep = cfg.polymarket.book_keep_levels
    written = errors = 0

    async with PolymarketClient(cfg.polymarket) as pm:
        disc = LiveDiscovery(pm, cfg.aggregate.window_sec)
        logger.info("book-collect started symbols=%s every %.1fs", symbols, interval)
        while not stop.is_set():
            t0 = time.monotonic()
            for sym in symbols:
                try:
                    written += await _sample_symbol(pm, disc, pool, sym,
                                                    cfg.aggregate.window_sec, keep)
                except Exception as exc:  # noqa: BLE001
                    errors += 1
                    if errors <= 5 or errors % 100 == 0:
                        logger.warning("[%s] book sample failed (#%d): %s", sym, errors, exc)
            if written and written % 200 < len(symbols) * 2:
                logger.info("book-collect: %d rows written, %d errors", written, errors)
            # Sleep the remainder of the interval so cadence doesn't drift.
            try:
                await asyncio.wait_for(stop.wait(),
                                       timeout=max(0.0, interval - (time.monotonic() - t0)))
            except asyncio.TimeoutError:
                pass
    logger.info("book-collect stopped: %d rows written, %d errors", written, errors)
