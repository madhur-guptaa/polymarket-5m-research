"""
polaris/ingest/backfill.py

Historical trade backfill from Binance Vision (the public daily archive CDN),
into the same crypto_trades table the live ingester writes to.

Per day, per symbol:
  1. stream the daily ZIP to a temp file (files can be hundreds of MB)
  2. in a worker thread: unzip, COPY the CSV into an UNLOGGED staging temp
     table, then INSERT ... SELECT ... ON CONFLICT DO NOTHING into production
  3. delete the temp ZIP

The staging + ON CONFLICT pattern is kept deliberately (the v1 plan suggested
dropping it): it is what makes backfill both fast (COPY speed) and idempotent
(re-running a day is a no-op, a partial re-run fills only the gaps). A direct
COPY into the production table aborts the whole day on a single duplicate.

TIMESTAMP GOTCHA (handled here):
    Binance Vision changed the CSV trade-time unit from milliseconds to
    microseconds in early 2025. The old backfill hard-coded microseconds and
    would misdate any millisecond-era file by 1000x. _epoch_to_dt() detects the
    unit by magnitude, so archives on either side of the changeover load
    correctly.

CSV columns (no header): [0] trade_id [1] price [2] qty [3] quote_qty
    [4] timestamp(ms|µs) [5] is_buyer_maker("True"/"False") [6] is_best_match
"""

from __future__ import annotations

import asyncio
import csv
import logging
import os
import zipfile
from datetime import datetime, timedelta, timezone
from io import TextIOWrapper

import httpx
import psycopg

from polaris.core.config import Settings

logger = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data/spot/daily/trades"


def _epoch_to_dt(v: int) -> datetime:
    """Normalize a Binance epoch (µs / ms / s) to a UTC datetime.

    Detects the unit by magnitude so archives from before and after the
    2025 ms->µs switch both load correctly.
        >= 1e15  microseconds   (~ year 2001+ in µs)
        >= 1e12  milliseconds
        else     seconds
    """
    if v >= 1_000_000_000_000_000:
        secs = v / 1_000_000.0
    elif v >= 1_000_000_000_000:
        secs = v / 1_000.0
    else:
        secs = float(v)
    return datetime.fromtimestamp(secs, tz=timezone.utc)


async def _download(client: httpx.AsyncClient, url: str, path: str) -> bool:
    """Stream a URL to disk. Returns False on 404 (date not published) or error."""
    try:
        async with client.stream("GET", url) as resp:
            if resp.status_code == 404:
                logger.warning("Not on Binance Vision (yet): %s", url)
                return False
            resp.raise_for_status()
            with open(path, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=1 << 16):
                    f.write(chunk)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Download failed %s: %s", url, exc)
        return False


def _extract_and_load(symbol: str, zip_path: str, date_str: str, dsn: str) -> None:
    """Unzip and load one day into crypto_trades via staging + ON CONFLICT."""
    try:
        with zipfile.ZipFile(zip_path) as z:
            with z.open(z.namelist()[0]) as raw:
                reader = csv.reader(TextIOWrapper(raw, encoding="utf-8"))
                with psycopg.connect(dsn) as conn, conn.cursor() as cur:
                    cur.execute(
                        "CREATE TEMP TABLE _staging_trades ("
                        " symbol VARCHAR(20), trade_id BIGINT, price NUMERIC(18,8),"
                        " quantity NUMERIC(18,8), trade_time TIMESTAMPTZ, is_buyer_mm BOOLEAN"
                        ") ON COMMIT DROP"
                    )
                    rows = 0
                    with cur.copy(
                        "COPY _staging_trades "
                        "(symbol, trade_id, price, quantity, trade_time, is_buyer_mm) "
                        "FROM STDIN"
                    ) as copy:
                        for row in reader:
                            if not row:
                                continue
                            try:
                                trade_id = int(row[0])
                            except ValueError:
                                continue  # header or malformed line
                            copy.write_row((
                                symbol,
                                trade_id,
                                row[1],                       # price (str -> NUMERIC)
                                row[2],                       # qty   (str -> NUMERIC)
                                _epoch_to_dt(int(row[4])),
                                row[5].strip().upper() == "TRUE",
                            ))
                            rows += 1
                    cur.execute(
                        "INSERT INTO crypto_trades "
                        "(symbol, trade_id, price, quantity, trade_time, is_buyer_mm) "
                        "SELECT symbol, trade_id, price, quantity, trade_time, is_buyer_mm "
                        "FROM _staging_trades ON CONFLICT (symbol, trade_id) DO NOTHING"
                    )
                    inserted = cur.rowcount or 0
                    conn.commit()
        logger.info(
            "[%s] %s: parsed %d, inserted %d (%d dupes skipped)",
            symbol, date_str, rows, inserted, rows - inserted,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("[%s] %s: extract/load failed: %s", symbol, date_str, exc)


async def _one_day(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    symbol: str,
    day: datetime,
    dsn: str,
    tmp_dir: str,
) -> None:
    date_str = day.strftime("%Y-%m-%d")
    fname = f"{symbol}-trades-{date_str}.zip"
    url = f"{BASE_URL}/{symbol}/{fname}"
    zip_path = os.path.join(tmp_dir, fname)
    async with sem:
        if not await _download(client, url, zip_path):
            return
        # Unzip + COPY is blocking/CPU-bound -> run off the event loop.
        await asyncio.to_thread(_extract_and_load, symbol, zip_path, date_str, dsn)
        if os.path.exists(zip_path):
            os.remove(zip_path)


async def backfill(
    cfg: Settings,
    *,
    symbols: list[str] | None = None,
    days: int = 14,
    concurrency: int = 2,
    tmp_dir: str = "temp_binance_data",
) -> None:
    """Backfill `days` of trades ending yesterday (today's archive isn't final).

    Concurrency is bounded because each unzipped CSV can hold 0.5-1 GB in flight.
    """
    symbols = symbols or cfg.binance.symbols
    dsn = cfg.pg.dsn
    os.makedirs(tmp_dir, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    end = datetime.now(timezone.utc) - timedelta(days=1)

    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = [
            _one_day(client, sem, symbol, end - timedelta(days=i), dsn, tmp_dir)
            for symbol in symbols
            for i in range(days)
        ]
        logger.info("Backfilling %d symbol-days (symbols=%s, days=%d)", len(tasks), symbols, days)
        await asyncio.gather(*tasks)
    logger.info("Backfill complete.")
