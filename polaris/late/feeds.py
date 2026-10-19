"""
polaris/late/feeds.py

Two streaming feeds, both free and unauthenticated, for the late-window study.

ChainlinkSpotFeed -- Chainlink's per-second spot price via Polymarket's relay
    (wss://ws-live-data.polymarket.com, topic crypto_prices_chainlink). Two
    measured quirks drive the shape of this class:
      * the subscription filter must be EMPTY. With {"symbol": "btc/usd"} the
        relay sends one snapshot of the last minute and then nothing; with ""
        it streams every symbol continuously (~1 msg/s/symbol, ~1.5s old).
      * the relay wants an application-level text "PING" every few seconds.
    Chainlink's TWAP-60 -- what Polymarket settles on -- equals the mean of
    this series over the trailing 60s to within 0.1 bp, so this feed alone
    determines how a window settles.

ClobBookFeed -- the live order book for the current window's YES/NO tokens via
    the CLOB market channel (wss://ws-subscriptions-clob.polymarket.com/ws/market).
    ~22 ms from the matching engine to us, measured. Each 5-minute window is a
    new market with new token ids, so the feed reconnects per window; the
    subscription answers with a full `book` snapshot, then `price_change` deltas.
    Book levels arrive as strings and both sides worst-to-best.

Both reconnect forever with backoff: a research collector that dies on one
dropped socket loses data that cannot be recovered.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

import websockets

from polaris.core.config import PolymarketSettings
from polaris.polymarket.client import PolymarketClient

logger = logging.getLogger(__name__)

RTDS_URL = "wss://ws-live-data.polymarket.com"
CLOB_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
RELAY_SYMBOLS = {"btc/usd": "BTCUSDT", "eth/usd": "ETHUSDT", "sol/usd": "SOLUSDT",
                 "xrp/usd": "XRPUSDT", "doge/usd": "DOGEUSDT", "bnb/usd": "BNBUSDT",
                 "hype/usd": "HYPEUSDT", "zec/usd": "ZECUSDT"}

# Watchdogs. Both sockets can STALL -- stay open, answer nothing, raise nothing:
# the first overnight smoke run lost the Chainlink stream after ~4.5 minutes and
# never reconnected, so the collector logged zero rows while looking healthy.
# Silence past these limits forces a reconnect. The book feed's limit is looser
# because a quiet book is legitimate; Chainlink should tick every second.
SPOT_STALL_SEC = 5.0
BOOK_STALL_SEC = 30.0


class FeedStalled(Exception):
    """No data within the watchdog limit; caught by the reconnect loop."""


class ChainlinkSpotFeed:
    """Per-second Chainlink spot, kept in memory per symbol (last ~15 minutes)."""

    def __init__(self, keep_sec: int = 900) -> None:
        self.spot: dict[str, dict[int, float]] = {s: {} for s in RELAY_SYMBOLS.values()}
        self.last_recv: float = 0.0
        self.keep_sec = keep_sec
        self.pending: list[tuple[str, int, float]] = []   # rows for the DB writer

    async def run(self, stop: asyncio.Event) -> None:
        delay = 1.0
        while not stop.is_set():
            try:
                async with websockets.connect(RTDS_URL, open_timeout=15) as ws:
                    await ws.send(json.dumps({"action": "subscribe", "subscriptions": [
                        {"topic": "crypto_prices_chainlink", "type": "update", "filters": ""}]}))
                    logger.info("chainlink spot feed connected")
                    delay = 1.0
                    next_ping = time.monotonic() + 5
                    last_data = time.monotonic()
                    while not stop.is_set():
                        if time.monotonic() >= next_ping:
                            await ws.send("PING")
                            next_ping = time.monotonic() + 5
                        if time.monotonic() - last_data > SPOT_STALL_SEC:
                            raise FeedStalled(f"no chainlink data for {SPOT_STALL_SEC}s")
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue
                        if self._ingest(raw):
                            last_data = time.monotonic()
            except Exception as exc:  # noqa: BLE001
                logger.warning("chainlink spot feed dropped (%s); reconnecting in %.0fs",
                               type(exc).__name__, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)

    def _ingest(self, raw) -> bool:
        """Store any spot points in `raw`; True if at least one arrived."""
        if not raw or raw in ("PONG", "pong"):
            return False
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return False
        got = False
        payload = (msg or {}).get("payload") or {}
        points = payload.get("data") if isinstance(payload.get("data"), list) else [payload]
        relay_sym = str(payload.get("symbol", "")).lower()
        for p in points:
            sym = RELAY_SYMBOLS.get(str(p.get("symbol", relay_sym)).lower())
            ts, val = p.get("timestamp"), p.get("value")
            if sym is None or ts is None or val is None:
                continue
            sec = int(ts) // 1000
            book = self.spot[sym]
            if sec not in book:
                self.pending.append((sym, sec, float(val)))
            book[sec] = float(val)
            self.last_recv = time.time()
            got = True
            if len(book) > self.keep_sec + 120:
                cutoff = sec - self.keep_sec
                for k in [k for k in book if k < cutoff]:
                    del book[k]
        return got

    def newest(self, symbol: str) -> int | None:
        s = self.spot.get(symbol)
        return max(s) if s else None


@dataclass
class TokenBook:
    """One token's order book: price -> size per side."""
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    updated_ms: float = 0.0

    def best_ask(self) -> tuple[float, float] | tuple[None, None]:
        live = [(p, s) for p, s in self.asks.items() if s > 0]
        return min(live) if live else (None, None)

    def best_bid(self) -> float | None:
        live = [p for p, s in self.bids.items() if s > 0]
        return max(live) if live else None

    def best_bid_size(self) -> float | None:
        b = self.best_bid()
        return self.bids.get(b) if b is not None else None

    def effective_ask(self, budget: float) -> float | None:
        """Average price paid to spend `budget` USDC walking the asks; None if too thin."""
        spent = shares = 0.0
        for p, s in sorted((p, s) for p, s in self.asks.items() if s > 0):
            take = min(s, (budget - spent) / p)
            spent += take * p
            shares += take
            if spent >= budget - 1e-9:
                return spent / shares
        return None


class ClobBookFeed:
    """Books for the CURRENT window's YES/NO tokens, both symbols; one socket per window."""

    def __init__(self, cfg: PolymarketSettings, symbols: list[str], window_sec: int = 300,
                 series: str = "5m") -> None:
        self.cfg = cfg
        self.series = series
        self.symbols = symbols
        self.window_sec = window_sec
        self.window_start: int | None = None
        self.books: dict[str, TokenBook] = {}
        self.token_meta: dict[str, tuple[str, str]] = {}   # asset_id -> (symbol, YES|NO)
        self.tokens: dict[str, tuple[str, str]] = {}       # symbol -> (yes_id, no_id)
        self.pending_trades: list[tuple] = []
        # touch changes (best bid/ask price or size), when record_touch is set
        self.record_touch = False
        self.pending_touch: list[tuple] = []
        self._last_touch: dict[str, tuple] = {}

    async def _resolve(self, pm: PolymarketClient, ws: int) -> dict[str, tuple[str, str]]:
        out = {}
        for sym in self.symbols:
            ref = await pm.resolve_market(sym, ws, self.series)
            if ref is not None:
                out[sym] = (ref.token_yes, ref.token_no)
        return out

    async def run(self, stop: asyncio.Event) -> None:
        async with PolymarketClient(self.cfg) as pm:
            while not stop.is_set():
                now = time.time()
                ws_start = int(now) - int(now) % self.window_sec
                try:
                    tokens = await self._resolve(pm, ws_start)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("market resolve failed: %s", exc)
                    tokens = {}
                if not tokens:
                    await asyncio.sleep(2)
                    continue
                self.window_start, self.tokens = ws_start, tokens
                self.books = {}
                self.token_meta = {}
                for sym, (y, n) in tokens.items():
                    self.books[y], self.books[n] = TokenBook(), TokenBook()
                    self.token_meta[y], self.token_meta[n] = (sym, "YES"), (sym, "NO")
                await self._stream_until(ws_start + self.window_sec, stop)

    async def _stream_until(self, end_epoch: float, stop: asyncio.Event) -> None:
        """Hold one socket for this window; reconnect within it if it drops."""
        while not stop.is_set() and time.time() < end_epoch:
            try:
                async with websockets.connect(CLOB_URL, open_timeout=10) as ws:
                    await ws.send(json.dumps({"type": "market",
                                              "assets_ids": list(self.token_meta)}))
                    next_ping = time.monotonic() + 10
                    last_data = time.monotonic()
                    while not stop.is_set() and time.time() < end_epoch:
                        if time.monotonic() >= next_ping:
                            await ws.send("PING")
                            next_ping = time.monotonic() + 10
                        if time.monotonic() - last_data > BOOK_STALL_SEC:
                            raise FeedStalled(f"no book events for {BOOK_STALL_SEC}s")
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                        except asyncio.TimeoutError:
                            continue
                        if raw and raw != "PONG":
                            last_data = time.monotonic()
                        self._ingest(raw)
            except Exception as exc:  # noqa: BLE001
                logger.warning("clob book feed dropped (%s); reconnecting", type(exc).__name__)
                await asyncio.sleep(0.5)

    def _note_touch(self, asset: str, ts: float) -> None:
        """Queue a touch row if this asset's best bid/ask (price or size) changed."""
        if not self.record_touch:
            return
        meta, b = self.token_meta.get(asset), self.books.get(asset)
        if meta is None or b is None or self.window_start is None:
            return
        bb = b.best_bid()
        ba, bas = b.best_ask()
        touch = (bb, b.bids.get(bb) if bb is not None else None, ba, bas)
        if self._last_touch.get(asset) != touch:
            self._last_touch[asset] = touch
            self.pending_touch.append((meta[0], self.window_start, meta[1], int(ts), *touch))

    def _ingest(self, raw) -> None:
        if not raw or raw == "PONG":
            return
        try:
            msgs = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return
        for m in msgs if isinstance(msgs, list) else [msgs]:
            et = m.get("event_type")
            ts = float(m.get("timestamp") or 0)
            if et == "book":
                b = self.books.get(m.get("asset_id"))
                if b is None:
                    continue
                b.bids = {float(x["price"]): float(x["size"]) for x in m.get("bids", [])}
                b.asks = {float(x["price"]): float(x["size"]) for x in m.get("asks", [])}
                b.updated_ms = ts
                self._note_touch(m.get("asset_id"), ts)
            elif et == "price_change":
                for ch in m.get("price_changes", []):
                    b = self.books.get(ch.get("asset_id"))
                    if b is None:
                        continue
                    side = b.bids if ch.get("side") == "BUY" else b.asks
                    price, size = float(ch["price"]), float(ch["size"])
                    if size <= 0:
                        side.pop(price, None)
                    else:
                        side[price] = size
                    b.updated_ms = ts
                    self._note_touch(ch.get("asset_id"), ts)
            elif et == "last_trade_price":
                meta = self.token_meta.get(m.get("asset_id"))
                if meta is None or self.window_start is None:
                    continue
                fee = m.get("fee_rate_bps")
                self.pending_trades.append((
                    meta[0], self.window_start, m.get("asset_id"), meta[1], m.get("side"),
                    float(m.get("price") or 0), float(m.get("size") or 0),
                    int(fee) if fee not in (None, "") else None, int(ts),
                    m.get("transaction_hash") or ""))
