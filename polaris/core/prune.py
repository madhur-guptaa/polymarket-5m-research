"""
polaris/core/prune.py

Data retention. Deliberately asymmetric, because these tables are not equally
replaceable — and the difference is the whole point of this module.

  crypto_trades          37 GB, 86% of the database, and fully re-derivable:
                         Binance Vision is a permanent public archive. Pruned.
  crypto_1s_bars         KEPT. Only ~1.2 GB, and it is the substrate for most
                         of the near-term feature work (trade intensity,
                         realized variance, average trade size, range-based
                         volatility all come from bars, not ticks). Pruning it
                         would force re-downloads to save almost nothing.
  window_observations    KEPT. The model's actual input, and the thing that
                         defines effective sample size: ~300 autocorrelated
                         rows per window share one label, so what matters is
                         the WINDOW count. Keeping it forever is what lets
                         feature headroom keep growing.
  poly_historical_prices KEPT. Re-fetchable but slow and rate-limited.
  poly_book_snapshots    NEVER. Forward-only: there is no historical order
  paper_signals          NEVER. book, and no way to re-observe what the live
                         system said in the past. These two are the only
                         irreplaceable data in the system and are tiny.

The NEVER entries are enforced, not documented-and-hoped: prune() refuses to
touch them even if named explicitly, so a careless "prune everything" cannot
destroy the one thing a backfill can't rebuild.

Dry-run is the default. Deletes are batched, because a single DELETE over tens
of millions of rows is a WAL storm and a long lock.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Policy:
    table: str
    time_column: str | None          # None => not time-pruned
    default_days: int | None         # None => keep forever
    reason: str                      # why, shown to the user
    never: bool = False              # refuse even if explicitly requested


POLICIES: tuple[Policy, ...] = (
    Policy("crypto_trades", "trade_time", 30,
           "86% of the DB; fully re-derivable from Binance Vision"),
    Policy("crypto_1s_bars", "bucket_time", None,
           "~1.2 GB; substrate for most new features — keep"),
    Policy("window_observations", "window_start", None,
           "the model's input; defines effective sample size — keep"),
    Policy("poly_historical_prices", None, None,
           "re-fetchable but slow/rate-limited — keep"),
    Policy("poly_book_snapshots", None, None,
           "FORWARD-ONLY: no historical order book exists", never=True),
    Policy("paper_signals", None, None,
           "FORWARD-ONLY: cannot re-observe past live signals", never=True),
    # The free relay serves only the trailing ~60s and Chainlink's own history
    # API is paid ($150/mo, no free tier). A second of this deleted is a second
    # that cannot be bought back at any price -- strictly worse than
    # poly_book_snapshots, which at least has no paid alternative to regret.
    Policy("chainlink_twap", None, None,
           "FORWARD-ONLY: the free relay has no history endpoint", never=True),
    # Re-fetchable from Gamma, but slowly (~40 windows/sec) and only while the
    # markets stay resolvable. This is THE label -- losing it silently reverts
    # every downstream result to the Binance-derived outcome that was wrong on
    # 12% of windows.
    Policy("poly_resolutions", None, None,
           "the real label; re-fetchable from Gamma but slow — keep"),
    Policy("poly_trades", None, None,
           "re-fetchable from the data-api but heavily rate-limited — keep"),
    # Late-window / maker / cross-series research (polaris/late). All streamed
    # live from Polymarket's websockets; none has a history endpoint.
    Policy("chainlink_spot", None, None,
           "FORWARD-ONLY: Chainlink spot from the relay; no history", never=True),
    Policy("late_signals", None, None,
           "FORWARD-ONLY: live estimate + book per second", never=True),
    Policy("poly_ws_trades", None, None,
           "FORWARD-ONLY: live fills with ms timestamps", never=True),
    Policy("poly_ws_touch", None, None,
           "FORWARD-ONLY: ms best-bid/ask changes (~8 GB/day when on)", never=True),
    Policy("cross_signals", None, None,
           "FORWARD-ONLY: 5m and 15m books side by side", never=True),
)
BY_NAME = {p.table: p for p in POLICIES}


async def _size(pool: AsyncConnectionPool, table: str) -> str:
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute("SELECT pg_size_pretty(pg_total_relation_size(%s))", (table,))
        return (await cur.fetchone())[0]


async def _count_older(pool: AsyncConnectionPool, p: Policy, days: int) -> int:
    sql = (f"SELECT count(*) FROM {p.table} "
           f"WHERE {p.time_column} < now() - make_interval(days => %s)")
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(sql, (days,))
        return (await cur.fetchone())[0]


async def _delete_batched(pool: AsyncConnectionPool, p: Policy, days: int,
                          batch: int = 500_000) -> int:
    """Delete in committed batches so we never hold one enormous transaction."""
    # Subquery on the primary key: cheaper than re-scanning by timestamp for
    # every batch, and lets Postgres use the (symbol, trade_time) index once.
    sql = (f"DELETE FROM {p.table} WHERE ctid IN ("
           f"  SELECT ctid FROM {p.table} "
           f"  WHERE {p.time_column} < now() - make_interval(days => %s) "
           f"  LIMIT %s)")
    total = 0
    while True:
        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(sql, (days, batch))
            n = cur.rowcount or 0
            await conn.commit()
        total += n
        if n:
            logger.info("[%s] deleted %d (running total %d)", p.table, n, total)
        if n < batch:
            return total


async def _vacuum(pool: AsyncConnectionPool, table: str) -> None:
    """Reclaim the dead tuples for reuse. Note: plain VACUUM makes the space
    reusable by Postgres but does NOT return it to the OS — that needs
    VACUUM FULL, which takes an exclusive lock and temporarily doubles disk
    use, so it is left as a deliberate manual step."""
    async with pool.connection() as conn:
        await conn.set_autocommit(True)          # VACUUM cannot run in a transaction
        async with conn.cursor() as cur:
            # Serial, not parallel. A parallel VACUUM allocates dynamic shared
            # memory, and inside the postgres container /dev/shm is Docker's
            # 64 MB default: on 2026-10-08, after deleting 155M crypto_trades
            # rows, it failed with "could not resize shared memory segment
            # ... No space left on device" -- the rows were gone but none of
            # the 46 GB was reclaimed. Serial VACUUM needs no DSM.
            await cur.execute("SET max_parallel_maintenance_workers = 0")
            await cur.execute(f"VACUUM ANALYZE {table}")


async def prune(pool: AsyncConnectionPool, *, days: int | None = None,
                tables: list[str] | None = None, apply: bool = False) -> None:
    """Report (default) or apply retention. `apply=False` deletes nothing."""
    requested = tables or [p.table for p in POLICIES]
    print("\n=== retention " + ("(APPLYING)" if apply else "(dry run — nothing deleted)") + " ===")

    for name in requested:
        p = BY_NAME.get(name)
        if p is None:
            print(f"  {name:24} UNKNOWN TABLE — skipped")
            continue
        if p.never:
            # Explicit refusal, not a silent skip: if someone asked for this
            # table by name they should be told why the answer is no.
            print(f"  {name:24} REFUSED — {p.reason}")
            continue
        if p.time_column is None or p.default_days is None:
            print(f"  {name:24} keep  ({p.reason}) [{await _size(pool, name)}]")
            continue

        keep_days = days if days is not None else p.default_days
        n = await _count_older(pool, p, keep_days)
        size = await _size(pool, name)
        if not n:
            print(f"  {name:24} nothing older than {keep_days}d [{size}]")
            continue
        if not apply:
            print(f"  {name:24} would delete {n:,} rows older than {keep_days}d [{size}]")
            continue

        print(f"  {name:24} deleting {n:,} rows older than {keep_days}d [{size}] …")
        deleted = await _delete_batched(pool, p, keep_days)
        await _vacuum(pool, name)
        print(f"  {name:24} deleted {deleted:,}; size now {await _size(pool, name)}")

    if not apply:
        print("\n  re-run with --apply to actually delete.")
    print()
