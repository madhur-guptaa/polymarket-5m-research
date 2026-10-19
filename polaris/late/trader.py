"""
polaris/late/trader.py

Late-window PAPER collector. Places no orders and holds no credentials.

Every data-second from LOG_FROM to the end of each window, for each symbol, it
records the Chainlink-based settlement estimate (late/model.py) next to the live
order book at that instant (late/feeds.py). It stores no trade decisions on
purpose: the rule is applied offline against poly_resolutions, so the estimator
can be recalibrated and thresholds replayed without having to re-collect --
this data cannot be re-collected.

"Data-second" means the Chainlink clock (newest spot point), not the wall
clock; cl_age records the gap (~1.5s measured), so latency can be studied too.
"""

from __future__ import annotations

import asyncio
import logging
import time

from psycopg_pool import AsyncConnectionPool

from polaris.core.config import Settings
from polaris.late.feeds import RELAY_SYMBOLS, ChainlinkSpotFeed, ClobBookFeed
from polaris.late.model import estimate

logger = logging.getLogger(__name__)

# Earliest second logged. The whole window, not just the end: the taker study
# centres on t >= 240, but the maker study needs the model and the touch at
# every second a resting order could be filled.
LOG_FROM = 0

_SIG_SQL = """
    INSERT INTO late_signals (symbol, window_start, t, wall_t, cl_age, anchor, mean_final,
        sd_final, spot, sigma, p_up, yes_ask, yes_ask_size, yes_bid, yes_eff10,
        no_ask, no_ask_size, no_bid, no_eff10, book_age_ms, yes_bid_size, no_bid_size)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT (symbol, window_start, t) DO NOTHING
"""
_CROSS_SQL = """
    INSERT INTO cross_signals (symbol, ws15, t15, a5, a15, p5, p15,
        m5_yes_ask, m5_yes_ask_size, m5_yes_bid, m5_no_ask, m5_no_ask_size, m5_no_bid,
        m15_yes_ask, m15_yes_ask_size, m15_yes_bid, m15_no_ask, m15_no_ask_size, m15_no_bid)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT (symbol, ws15, t15) DO NOTHING
"""
_TOUCH_SQL = """
    INSERT INTO poly_ws_touch (symbol, window_start, outcome, ts_ms, best_bid, bid_size,
        best_ask, ask_size) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
"""
_SPOT_SQL = """INSERT INTO chainlink_spot (symbol, ts, value) VALUES (%s,%s,%s)
               ON CONFLICT (symbol, ts) DO NOTHING"""
_TRADE_SQL = """
    INSERT INTO poly_ws_trades (symbol, window_start, asset_id, outcome, side, price, size,
        fee_rate_bps, ts_ms, tx_hash)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT (asset_id, ts_ms, tx_hash) DO NOTHING
"""


async def run(pool: AsyncConnectionPool, cfg: Settings, stop: asyncio.Event) -> None:
    # Every series the free Chainlink relay streams. The thinner ones (SOL..ZEC)
    # are where a late-window edge is most likely to survive -- fewer bots --
    # and least likely to have the depth to use it; logging all eight measures
    # both.
    symbols = list(RELAY_SYMBOLS.values())
    window = cfg.aggregate.window_sec
    spot = ChainlinkSpotFeed()
    # Warm start from what earlier runs stored. Without it every restart -- and
    # the laptop restarts the stack on every wake -- loses its first window,
    # because the anchor needs the minute BEFORE the window opened.
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute("SELECT symbol, ts, value FROM chainlink_spot "
                          "WHERE ts > extract(epoch from now())::bigint - %s", (spot.keep_sec,))
        warm = await cur.fetchall()
    for sym, ts, val in warm:
        if sym in spot.spot:
            spot.spot[sym][int(ts)] = float(val)
    logger.info("late-paper: warm-started %d chainlink points from the DB", len(warm))
    books = ClobBookFeed(cfg.polymarket, symbols, window)
    # ms touch log (poly_ws_touch) for the maker study. OFF since 2026-10-08: it
    # answered its question (RESEARCH_LOG session 32) and costs ~8 GB/day.
    # Set True to collect again.
    books.record_touch = False
    # 15-minute books, for the cross-series consistency study (cross_signals).
    books15 = ClobBookFeed(cfg.polymarket, symbols, 3 * window, series="15m")
    rows: list[tuple] = []
    cross_rows: list[tuple] = []
    cross_logged: set[tuple[str, int, int]] = set()
    logged: set[tuple[str, int, int]] = set()

    async def sampler() -> None:
        while not stop.is_set():
            await asyncio.sleep(0.25)
            ws = books.window_start
            if ws is None:
                continue
            wall = time.time()
            for sym in symbols:
                newest = spot.newest(sym)
                if newest is None or sym not in books.tokens:
                    continue
                t = newest - ws
                if t < LOG_FROM or t >= window or (sym, ws, t) in logged:
                    continue
                est = estimate(spot.spot[sym], ws, t)
                if est is None:
                    continue
                y_id, n_id = books.tokens[sym]
                yb, nb = books.books.get(y_id), books.books.get(n_id)
                if yb is None or nb is None:
                    continue
                ya, yas = yb.best_ask()
                na, nas = nb.best_ask()
                book_age = wall * 1000 - max(yb.updated_ms, nb.updated_ms)
                rows.append((sym, ws, t, wall - ws, wall - newest,
                             est["anchor"], est["mean_final"], est["sd_final"], est["spot"],
                             est["sigma"], est["p_up"],
                             ya, yas, yb.best_bid(), yb.effective_ask(10.0),
                             na, nas, nb.best_bid(), nb.effective_ask(10.0), book_age,
                             yb.best_bid_size(), nb.best_bid_size()))
                logged.add((sym, ws, t))
            if len(logged) > 5000:
                cutoff = (books.window_start or 0) - window
                logged.difference_update({k for k in logged if k[1] < cutoff})

    def _touch(book) -> tuple:
        if book is None:
            return (None, None, None)
        a, sz = book.best_ask()
        return (a, sz, book.best_bid())

    async def cross_sampler() -> None:
        """Last 5 minutes of each 15m window: the 5m and 15m books on one shared F."""
        while not stop.is_set():
            await asyncio.sleep(0.25)
            ws15, ws5 = books15.window_start, books.window_start
            if ws15 is None or ws5 is None or ws5 != ws15 + 2 * window:
                continue
            for sym in symbols:
                newest = spot.newest(sym)
                if newest is None or sym not in books.tokens or sym not in books15.tokens:
                    continue
                t15 = newest - ws15
                if t15 < 2 * window or t15 >= 3 * window or (sym, ws15, t15) in cross_logged:
                    continue
                e5 = estimate(spot.spot[sym], ws5, newest - ws5, window)
                e15 = estimate(spot.spot[sym], ws15, t15, 3 * window)
                if e5 is None or e15 is None:
                    continue
                y5, n5 = books.tokens[sym]
                y15, n15 = books15.tokens[sym]
                cross_rows.append((sym, ws15, t15, e5["anchor"], e15["anchor"],
                                   e5["p_up"], e15["p_up"],
                                   *_touch(books.books.get(y5)), *_touch(books.books.get(n5)),
                                   *_touch(books15.books.get(y15)), *_touch(books15.books.get(n15))))
                cross_logged.add((sym, ws15, t15))
            if len(cross_logged) > 5000:
                cross_logged.clear()

    async def writer() -> None:
        windows_seen = set()
        while not stop.is_set():
            await asyncio.sleep(5)
            batch, rows[:] = list(rows), []
            spots, spot.pending[:] = list(spot.pending), []
            trades, books.pending_trades[:] = list(books.pending_trades), []
            # 15m fills are discarded, not stored: poly_ws_trades has no series
            # column, and a 15m window_start (a multiple of 900) is also a valid
            # 5m window_start -- mixing them would corrupt the 5m maker study.
            books15.pending_trades[:] = []
            cross, cross_rows[:] = list(cross_rows), []
            touch, books.pending_touch[:] = list(books.pending_touch), []
            try:
                async with pool.connection() as conn, conn.cursor() as cur:
                    if batch:
                        await cur.executemany(_SIG_SQL, batch)
                    if spots:
                        await cur.executemany(_SPOT_SQL, spots)
                    if trades:
                        await cur.executemany(_TRADE_SQL, trades)
                    if cross:
                        await cur.executemany(_CROSS_SQL, cross)
                    if touch:
                        await cur.executemany(_TOUCH_SQL, touch)
                    await conn.commit()
            except Exception:  # noqa: BLE001
                logger.exception("late-paper write failed; dropping %d rows", len(batch))
            for r in batch:
                if r[1] not in windows_seen:
                    windows_seen.add(r[1])
                    logger.info("late-paper: logging window %s (cl_age %.1fs)", r[1], r[4])

    await asyncio.gather(spot.run(stop), books.run(stop), books15.run(stop),
                         sampler(), cross_sampler(), writer())
