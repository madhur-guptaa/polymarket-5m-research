"""
polaris/aggregate/aggregate.py

Incremental aggregation: crypto_trades -> crypto_1s_bars -> window_observations.

Per cycle, per symbol, in one transaction:
  1. bars  — aggregate new trades (> cursor) into 1s bars, upsert
  2. obs   — recompute window observations for every window touched by new data
  3. cursor — advance the high-water mark

CORRECTNESS FIX vs. the old bar_aggregator:
    Window features (open_vwap, cum_volume, cum_imbalance, volatility) are
    *cumulative within a 5-minute window*. The old code computed observations
    over only the newly-processed bar range, so any window that spanned two
    cycles got its cumulative features computed on a partial window — wrong,
    and "aggregate more frequently" made it worse.

    Here, step 2 always recomputes from the window's true start: the obs query
    lower bound is floored to the window boundary of the cursor, and it reads
    from the persisted crypto_1s_bars table (not just this cycle's bars). So a
    window in progress is fully recomputed each cycle from second 0. Because
    the obs upsert is idempotent and COALESCEs a set outcome, repeatedly
    recomputing a filling window is correct and self-healing — the numbers are
    right at every intermediate step and finalize when the window completes.

Window alignment uses the same fixed origin as the SQL (date_bin), so the
Python floor and the SQL binning agree exactly.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import psycopg
from psycopg_pool import AsyncConnectionPool

from polaris.core.config import Settings
from polaris.aggregate.sql import (
    AGGREGATE_BARS_SQL,
    EARLIEST_TRADE_SQL,
    GET_CURSOR_SQL,
    SET_CURSOR_SQL,
    UPSERT_BAR_SQL,
    UPSERT_OBS_SQL,
    WINDOW_OBS_SQL,
)

logger = logging.getLogger(__name__)

# MUST match WINDOW_ORIGIN_SQL in sql.py.
WINDOW_ORIGIN = datetime(2020, 1, 1, tzinfo=timezone.utc)

# How far before a window's start the obs query must read. The TWAP anchor is a
# trailing 60s TWAP at the window's first bar (see WINDOW_OBS_SQL), so the
# query needs the whole preceding minute plus the 2s the outcome label's
# prev_close LAG already required. Bars from that lookback feed the window
# functions but are never emitted (min_window_start filters them).
ANCHOR_LOOKBACK_SEC = 62


def _utc(ts: datetime) -> datetime:
    """Normalize to UTC-aware (PG may hand back naive depending on settings)."""
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def align_down(ts: datetime, window_sec: int) -> datetime:
    """Floor `ts` to its window boundary, using the same origin as date_bin."""
    delta = (ts - WINDOW_ORIGIN).total_seconds()
    return WINDOW_ORIGIN + timedelta(seconds=delta - (delta % window_sec))


async def get_cursor(pool: AsyncConnectionPool, symbol: str) -> datetime:
    """Where to resume: stored cursor, else 1s before the first trade, else now-10m."""
    async with pool.connection() as conn:
        row = await (await conn.execute(GET_CURSOR_SQL, {"sym": symbol})).fetchone()
        if row and row[0] is not None:
            return _utc(row[0])
        row = await (await conn.execute(EARLIEST_TRADE_SQL, {"sym": symbol})).fetchone()
        if row and row[0] is not None:
            start = _utc(row[0]) - timedelta(seconds=1)
            logger.info("[%s] no cursor — starting from first trade: %s", symbol, start)
            return start
    fallback = datetime.now(timezone.utc) - timedelta(minutes=10)
    logger.info("[%s] no trades — fallback cursor: %s", symbol, fallback)
    return fallback


async def aggregate_once(
    pool: AsyncConnectionPool, cfg: Settings, symbol: str,
    since: datetime, until: datetime,
) -> datetime:
    """Process one (since, until) chunk in a single transaction. Returns new cursor.

    `until` is floored to a whole second: we only ever aggregate COMPLETE
    seconds. This is what lets the cursor advance cleanly to a second boundary —
    otherwise it would sit on a truncated bucket_time and re-scan that second's
    sub-second trades forever (`trade_time > bucket` is always true for them).
    """
    window_sec = cfg.aggregate.window_sec
    until = until.replace(microsecond=0)
    if until <= since:
        return since  # not even one full new second yet

    async with pool.connection() as conn:
        # ── Phase 1: 1-second bars from new trades ──
        bar_rows = await (await conn.execute(
            AGGREGATE_BARS_SQL, {"sym": symbol, "since": since, "until": until}
        )).fetchall()

        if not bar_rows:
            # Nothing here — advance so we don't re-scan this empty range forever.
            await conn.execute(SET_CURSOR_SQL, {"sym": symbol, "ts": until})
            await conn.commit()
            return until

        bar_params = [{
            "symbol": symbol, "bucket_time": _utc(r[0]),
            "open_price": r[1], "high_price": r[2], "low_price": r[3],
            "close_price": r[4], "volume": r[5], "trade_count": r[6],
            "vwap": r[7], "mm_buy_volume": r[8], "total_taker_volume": r[9],
        } for r in bar_rows]
        async with conn.cursor() as wc, conn.pipeline():
            await wc.executemany(UPSERT_BAR_SQL, bar_params)

        # ── Phase 2: recompute observations from the window floor of `since` ──
        # Reads ANCHOR_LOOKBACK_SEC of bars before the window floor: 60s for the
        # TWAP anchor (a trailing 60s TWAP at the window's first bar) plus 2s
        # for the outcome label's prev_close_price LAG. Those bars belong to the
        # PRECEDING window and must never be emitted -- they would compute as a
        # truncated partition and overwrite that window's already-correct,
        # already-finalized row -- so min_window_start marks where output
        # begins and WINDOW_OBS_SQL uses everything before it as context only.
        min_window_start = align_down(since, window_sec)
        obs_rows = await _write_observations(conn, symbol, min_window_start, until, window_sec)

        # ── Phase 3: advance cursor to the processed second boundary ──
        # `until` is second-aligned and every second below it is fully done,
        # so this is exactly the high-water mark — no re-scan, no split seconds.
        await conn.execute(SET_CURSOR_SQL, {"sym": symbol, "ts": until})
        await conn.commit()

    logger.info(
        "[%s] %d bars, %d obs | %s -> %s",
        symbol, len(bar_params), len(obs_rows), bar_params[0]["bucket_time"], until,
    )
    return until


def _obs_params(r) -> dict:
    return {
        "symbol": r[0], "window_start": _utc(r[1]), "bucket_time": _utc(r[2]),
        "time_elapsed_s": r[3], "delta_bps": r[4], "cum_imbalance": r[5],
        "cum_volume": r[6], "volatility_10s": r[7], "outcome": r[8],
        "twap_sofar_bps": r[9], "spot_vs_anchor_bps": r[10],
    }


async def _write_observations(conn, symbol: str, min_window_start: datetime,
                              until: datetime, window_sec: int) -> list:
    """Recompute and upsert observations for windows starting >= min_window_start.

    Reads crypto_1s_bars only -- never crypto_trades -- which is what lets
    recompute_observations() rebuild history whose raw trades have been pruned.
    """
    obs_since = min_window_start - timedelta(seconds=ANCHOR_LOOKBACK_SEC)
    obs_rows = await (await conn.execute(WINDOW_OBS_SQL, {
        "sym": symbol, "since": obs_since, "until": until, "window_sec": window_sec,
        "min_window_start": min_window_start,
    })).fetchall()
    if obs_rows:
        async with conn.cursor() as wc, conn.pipeline():
            await wc.executemany(UPSERT_OBS_SQL, [_obs_params(r) for r in obs_rows])
    return obs_rows


async def recompute_observations(pool: AsyncConnectionPool, cfg: Settings, symbol: str,
                                 since: datetime, until: datetime,
                                 chunk: timedelta = timedelta(hours=6)) -> int:
    """Rebuild window_observations over [since, until) from the persisted bars.

    For adding a feature column to history. The live loop cannot do it: it only
    recomputes observations for periods where it has just aggregated NEW TRADES,
    and raw trades are pruned after 30 days, so rewinding its cursor silently
    skips everything older. Bars are kept forever, and observations need only
    bars, so this walks history directly -- no cursor involved, and safe to run
    alongside the live aggregator (the upsert is idempotent; stop `until` short
    of the live edge so the two never race on the same window).

    Chunks are whole windows so no window is ever split across two queries.
    """
    window_sec = cfg.aggregate.window_sec
    start = align_down(_utc(since), window_sec)
    end = align_down(_utc(until), window_sec)
    step = timedelta(seconds=max(window_sec,
                                 int(chunk.total_seconds()) // window_sec * window_sec))
    total = 0
    t = start
    while t < end:
        t_next = min(t + step, end)
        async with pool.connection() as conn:
            rows = await _write_observations(conn, symbol, t, t_next, window_sec)
            await conn.commit()
        total += len(rows)
        logger.info("[%s] recomputed %d obs | %s -> %s", symbol, len(rows), t, t_next)
        t = t_next
    return total


async def run(pool: AsyncConnectionPool, cfg: Settings, stop: asyncio.Event) -> None:
    """Continuous aggregation loop for all configured symbols."""
    a = cfg.aggregate
    # Chunk span, rounded to a whole number of windows (bounds catch-up query size).
    chunk_sec = max(a.window_sec, (a.max_chunk_sec // a.window_sec) * a.window_sec)
    symbols = cfg.binance.symbols

    cursors = {s: await get_cursor(pool, s) for s in symbols}
    logger.info("Aggregator started symbols=%s chunk=%ds", symbols, chunk_sec)

    while not stop.is_set():
        try:
            safe_limit = datetime.now(timezone.utc) - timedelta(seconds=a.safe_lag_sec)
            any_work = False
            for symbol in symbols:
                last = cursors[symbol]
                if safe_limit <= last:
                    continue
                until = min(safe_limit, last + timedelta(seconds=chunk_sec))
                new_last = await aggregate_once(pool, cfg, symbol, last, until)
                if new_last != last:
                    cursors[symbol] = new_last
                    any_work = True
            if not any_work:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=a.poll_interval_sec)
                except asyncio.TimeoutError:
                    pass
        except psycopg.Error as exc:
            logger.error("DB error in aggregator: %s", exc, exc_info=True)
            await asyncio.sleep(5)
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error in aggregator: %s", exc, exc_info=True)
            await asyncio.sleep(5)
