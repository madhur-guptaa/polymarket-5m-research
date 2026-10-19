"""
polaris/ingest/binance.py

Live ingestion: Binance combined trade websocket -> PostgreSQL, direct.

    Binance WS -> parse -> buffer -> Sink.write (COPY)

No Kafka (decision D1). One async task on the event loop:
  * read frames (with a receive timeout so the flush timer still fires in lulls)
  * parse each trade to a plain tuple
  * flush on batch-size OR time-interval, whichever comes first
  * on flush failure, retry with backoff instead of dropping (bounded by a
    buffer cap so PostgreSQL being down can't exhaust memory)
  * reconnect with jittered backoff on any websocket error
  * final flush on shutdown

Idempotency is the sink's job (unique symbol+trade_id), so a reconnect that
replays a few trades is harmless.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timezone

import orjson
import websockets
from websockets.asyncio.client import connect as ws_connect

from polaris.core.config import Settings
from polaris.ingest.sink import Sink, TradeRow

logger = logging.getLogger(__name__)

_loads = orjson.loads
_from_ts = datetime.fromtimestamp


@dataclass(slots=True)
class IngestMetrics:
    frames: int = 0
    trades: int = 0
    parse_errors: int = 0
    dropped: int = 0
    flushes: int = 0


class TradeIngester:
    """Streams Binance trades into a Sink."""

    def __init__(self, cfg: Settings, sink: Sink) -> None:
        self._sink = sink
        self._url = cfg.binance.combined_stream_url
        self._reconnect_delay = cfg.binance.reconnect_delay_sec
        self._ping_interval = cfg.binance.ping_interval_sec
        self._ping_timeout = cfg.binance.ping_timeout_sec
        self._max_frame = cfg.binance.max_frame_bytes

        self._batch_size = cfg.ingest.batch_size
        self._flush_interval = cfg.ingest.flush_interval_sec
        # Cap unflushed rows so a prolonged PG outage can't exhaust memory.
        self._max_buffer = self._batch_size * 100

        self._buffer: list[TradeRow] = []
        self._metrics = IngestMetrics()

    async def run(self, stop_event: asyncio.Event) -> None:
        """Reconnecting main loop; returns after a final flush on stop."""
        try:
            while not stop_event.is_set():
                try:
                    await self._connect_and_stream(stop_event)
                except asyncio.CancelledError:
                    raise
                except (websockets.ConnectionClosed, OSError) as exc:
                    logger.warning("WS dropped (%s) — reconnecting…", exc)
                except Exception:  # noqa: BLE001
                    logger.exception("WS unexpected error — reconnecting…")

                if stop_event.is_set():
                    break
                delay = self._reconnect_delay + random.uniform(0, 0.3 * self._reconnect_delay)
                await asyncio.sleep(delay)
        finally:
            await self._flush(force=True)
            m = self._metrics
            logger.info(
                "Ingest stopped | frames=%d trades=%d flushes=%d parse_err=%d dropped=%d",
                m.frames, m.trades, m.flushes, m.parse_errors, m.dropped,
            )

    async def _connect_and_stream(self, stop_event: asyncio.Event) -> None:
        async with ws_connect(
            self._url,
            ping_interval=self._ping_interval,
            ping_timeout=self._ping_timeout,
            max_size=self._max_frame,
            compression=None,
        ) as ws:
            logger.info("Connected to Binance: %s", self._url)
            last_flush = asyncio.get_running_loop().time()

            while not stop_event.is_set():
                # Bound the wait so the time-based flush fires even if the
                # stream goes quiet. TimeoutError just means "no frame yet".
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=self._flush_interval)
                    self._metrics.frames += 1
                    row = self._parse(raw)
                    if row is not None:
                        self._buffer.append(row)
                except TimeoutError:
                    pass

                now = asyncio.get_running_loop().time()
                if len(self._buffer) >= self._batch_size or (
                    self._buffer and now - last_flush >= self._flush_interval
                ):
                    await self._flush()
                    last_flush = now

    def _parse(self, raw: bytes | str) -> TradeRow | None:
        """Combined-stream frame -> TradeRow tuple, or None if not a trade.

        Binance trade event keys: s=symbol, t=trade_id, p=price(str),
        q=qty(str), T=trade_time(ms), m=is_buyer_maker.
        """
        try:
            data = _loads(raw)["data"]
            if data.get("e") != "trade":
                return None
            return (
                data["s"],
                data["t"],
                data["p"],
                data["q"],
                data["m"],
                _from_ts(data["T"] / 1000.0, tz=timezone.utc),
            )
        except (KeyError, orjson.JSONDecodeError, TypeError):
            self._metrics.parse_errors += 1
            if self._metrics.parse_errors <= 5 or self._metrics.parse_errors % 1000 == 0:
                logger.error("Parse error #%d", self._metrics.parse_errors)
            return None

    async def _flush(self, *, force: bool = False) -> None:
        """Write the buffer, retrying on transient failure.

        On repeated failure we back off; if the buffer would grow past the cap
        we drop the oldest rows (public market data — the next tick fills the
        gap, and backfill can recover history).
        """
        if not self._buffer:
            return

        backoff = 0.1
        attempts = 0
        while True:
            ok = await self._sink.write(self._buffer)
            if ok:
                self._metrics.trades += len(self._buffer)
                self._metrics.flushes += 1
                self._buffer = []
                return

            attempts += 1
            if not force and len(self._buffer) < self._max_buffer:
                # Keep buffering new frames while PG recovers; caller will retry.
                logger.warning("Flush failed (attempt %d) — buffering, will retry", attempts)
                return
            if len(self._buffer) >= self._max_buffer:
                drop = len(self._buffer) - self._batch_size
                self._metrics.dropped += drop
                self._buffer = self._buffer[drop:]
                logger.error("Buffer cap hit — dropped %d rows", drop)
            await asyncio.sleep(backoff)
            backoff = min(5.0, backoff * 2)
            if force and attempts >= 5:
                logger.error("Giving up final flush of %d rows", len(self._buffer))
                return
