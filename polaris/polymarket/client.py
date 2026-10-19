"""
polaris/polymarket/client.py

Shared Polymarket API primitives used by both the historical backfill and
(later) the live edge discovery in the api layer. Keeping slug construction,
Gamma resolution, CLOB access, the rate limiter, and the HTTP client in one
place is exactly why polymarket/ is its own package — splitting it across
ingest/ and api/ would duplicate all of this.

Polymarket runs 5-minute binary "up/down" crypto markets on a CLOB. A market
is named by a slug like `btc-updown-5m-<window_start_epoch>`; the Gamma API
maps that slug to CLOB token ids (YES, NO); the CLOB API serves price history
and live midpoint/spread.

Pydantic is used for the response models here (not on any hot path): these are
low-rate control-plane calls, so validating the shape of Polymarket's
occasionally-inconsistent JSON is pure upside (decision D4).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Final

import httpx
from pydantic import BaseModel

from polaris.core.config import PolymarketSettings

logger = logging.getLogger(__name__)

# Internal symbol -> Polymarket slug prefix + CEX tickers (the CEX tickers are
# for the live strike-price proxy, added with discovery.py later).
SYMBOL_CONFIG: Final[dict[str, dict[str, str]]] = {
    "BTCUSDT": {"slug_prefix": "btc-updown-5m", "binance_sym": "BTCUSDT",
                "coinbase_sym": "BTC-USD", "kraken_sym": "XBTUSD"},
    "ETHUSDT": {"slug_prefix": "eth-updown-5m", "binance_sym": "ETHUSDT",
                "coinbase_sym": "ETH-USD", "kraken_sym": "XETHZUSD"},
    "SOLUSDT": {"slug_prefix": "sol-updown-5m", "binance_sym": "SOLUSDT",
                "coinbase_sym": "SOL-USD", "kraken_sym": "SOLUSD"},
    "XRPUSDT": {"slug_prefix": "xrp-updown-5m", "binance_sym": "XRPUSDT",
                "coinbase_sym": "XRP-USD", "kraken_sym": "XXRPZUSD"},
    # The thinner series (added 2026-10-08 for the late-window study). Only the
    # slug prefix is used for them; the CEX fields are kept for shape.
    "DOGEUSDT": {"slug_prefix": "doge-updown-5m", "binance_sym": "DOGEUSDT",
                 "coinbase_sym": "DOGE-USD", "kraken_sym": "XDGUSD"},
    "BNBUSDT": {"slug_prefix": "bnb-updown-5m", "binance_sym": "BNBUSDT",
                "coinbase_sym": "BNB-USD", "kraken_sym": "BNBUSD"},
    "HYPEUSDT": {"slug_prefix": "hype-updown-5m", "binance_sym": "HYPEUSDT",
                 "coinbase_sym": "HYPE-USD", "kraken_sym": "HYPEUSD"},
    "ZECUSDT": {"slug_prefix": "zec-updown-5m", "binance_sym": "ZECUSDT",
                "coinbase_sym": "ZEC-USD", "kraken_sym": "XZECZUSD"},
}


def slug_for(symbol: str, window_start: int, series: str = "5m") -> str | None:
    """Polymarket market slug for a symbol's window, or None if unsupported.

    series picks the recurring market length: "5m" (default), "15m" or "4h".
    All three settle on the same Chainlink TWAP-60 stream (end vs start of their
    own range), which is what makes cross-series consistency checks possible.
    """
    cfg = SYMBOL_CONFIG.get(symbol)
    if not cfg:
        return None
    prefix = cfg["slug_prefix"].replace("-5m", f"-{series}")
    return f"{prefix}-{window_start}"


class MarketRef(BaseModel):
    """Resolved market identity — enough to fetch prices."""
    symbol: str
    window_start: int
    condition_id: str | None = None
    token_yes: str
    token_no: str
    title: str = ""


class AsyncRateLimiter:
    """Sliding-window limiter that spaces requests to avoid 429 bans.

    Polymarket's limiter behaves like a sliding window and flags bursts even
    under the nominal cap, so we match it rather than use a token bucket.
    """

    def __init__(self, max_requests: int, window_sec: float) -> None:
        self._max = max_requests
        self._window = window_sec
        self._stamps: list[float] = []
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            self._stamps = [t for t in self._stamps if now - t <= self._window]
            if len(self._stamps) >= self._max:
                sleep = self._window - (now - self._stamps[0])
                if sleep > 0:
                    await asyncio.sleep(sleep)
                now = time.monotonic()
                self._stamps = [t for t in self._stamps if now - t <= self._window]
            self._stamps.append(now)


class PolymarketClient:
    """Async Polymarket API client with connection pooling + rate limiting.

    Use as an async context manager:
        async with PolymarketClient(cfg.polymarket) as pm:
            ref = await pm.resolve_market("BTCUSDT", 1719849600)
    """

    def __init__(self, cfg: PolymarketSettings) -> None:
        self._cfg = cfg
        self._client = httpx.AsyncClient(
            timeout=cfg.timeout_sec,
            limits=httpx.Limits(max_connections=cfg.backfill_concurrency * 3,
                                max_keepalive_connections=20),
            follow_redirects=True,
        )
        self._gamma = AsyncRateLimiter(cfg.gamma_max_req, cfg.gamma_window_sec)
        self._clob = AsyncRateLimiter(cfg.clob_max_req, cfg.clob_window_sec)
        self._data = AsyncRateLimiter(cfg.data_max_req, cfg.data_window_sec)

    async def __aenter__(self) -> "PolymarketClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if not self._client.is_closed:
            await self._client.aclose()

    @staticmethod
    def _first_market(events: list[dict]) -> dict | None:
        """Pull the first market dict out of a Gamma events response."""
        if not events:
            return None
        return (events[0].get("markets") or [None])[0]

    @staticmethod
    def _token_ids(market: dict) -> tuple[str, str] | None:
        """Extract (yes, no) CLOB token ids; Gamma returns them as a JSON string."""
        ids = market.get("clobTokenIds")
        if isinstance(ids, str):
            try:
                ids = json.loads(ids)
            except json.JSONDecodeError:
                return None
        if not ids or len(ids) < 2:
            return None
        return ids[0], ids[1]

    async def resolve_market(self, symbol: str, window_start: int,
                             series: str = "5m") -> MarketRef | None:
        """Resolve a symbol's window slug to its CLOB token ids via Gamma.

        Exact slug (no offset probing): backfill knows the historical
        window_start precisely. Returns None if the market doesn't exist.
        """
        slug = slug_for(symbol, window_start, series)
        if slug is None:
            logger.warning("Unsupported Polymarket symbol: %s", symbol)
            return None
        await self._gamma.wait()
        try:
            r = await self._client.get(self._cfg.gamma_api_url, params={"slug": slug})
            if r.status_code == 404:
                return None
            r.raise_for_status()
            market = self._first_market(r.json())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gamma error for %s: %s", slug, exc)
            return None
        if not market:
            return None
        tokens = self._token_ids(market)
        if tokens is None:
            return None
        return MarketRef(
            symbol=symbol, window_start=window_start,
            condition_id=market.get("conditionId"),
            token_yes=tokens[0], token_no=tokens[1],
            title=market.get("question", "") or market.get("groupItemTitle", ""),
        )

    async def resolution(self, symbol: str, window_start: int,
                         attempts: int = 4, series: str = "5m") -> int | None:
        """How Polymarket ACTUALLY resolved this window: 1 = Up, 0 = Down.

        Returns None when the market does not exist or has not resolved yet --
        never a guess. Gamma reports the settled payout in `outcomePrices` as
        ["1","0"] / ["0","1"] once resolution has happened; anything else (a
        live market still quoting "0.42") means not-yet-resolved.

        This is the ONLY trustworthy label. See the poly_resolutions comment in
        core/schema.sql for why the Binance-derived label is not.
        `series` selects the market length ("5m", "15m", "4h").
        """
        slug = slug_for(symbol, window_start, series)
        if slug is None:
            return None
        delay = 1.0
        for attempt in range(attempts):
            await self._gamma.wait()
            try:
                r = await self._client.get(self._cfg.gamma_api_url, params={"slug": slug})
                if r.status_code == 429:
                    if attempt < attempts - 1:
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, 16.0)
                        continue
                    return None
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                market = self._first_market(r.json())
            except Exception:  # noqa: BLE001
                if attempt < attempts - 1:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 16.0)
                    continue
                return None
            if not market:
                return None
            prices = market.get("outcomePrices")
            if isinstance(prices, str):
                try:
                    prices = json.loads(prices)
                except json.JSONDecodeError:
                    return None
            # Settled markets pay exactly 1/0. A live market quotes fractions.
            if not prices or prices[0] not in ("0", "1"):
                return None
            # outcomes is ['Up','Down'] on these markets, so index 0 is Up.
            return int(prices[0] == "1")
        return None

    async def price_history(self, token_id: str, start_ts: int, end_ts: int) -> list[dict]:
        """Raw (sparse) CLOB price ticks for a token over [start_ts, end_ts]."""
        await self._clob.wait()
        r = await self._client.get(
            f"{self._cfg.clob_api_url}/prices-history",
            params={"market": token_id, "startTs": start_ts, "endTs": end_ts},
        )
        r.raise_for_status()
        return r.json().get("history", [])

    async def book(self, token_id: str, attempts: int = 3) -> dict | None:
        """Full CLOB order book for a token.

        Returns the raw payload: bids/asks as [{price,size}, ...] plus
        last_trade_price, tick_size and min_order_size. NOTE the level
        ordering -- both sides come back worst-to-best, so the touch is
        bids[-1] / asks[-1], not [0].

        RETRIES ON 429, which matters more than it looks. /book turns out to
        have a far tighter per-endpoint limit than the documented CLOB budget:
        measured 180 429s while sending only ~0.8 req/s against a nominal 90/s
        allowance. Without a retry each 429 silently became a dropped sample --
        and since the order book cannot be backfilled, a dropped sample is gone
        permanently. Transient network errors (DNS failures on network change,
        e.g. a laptop moving between networks) get the same treatment.
        """
        delay = 0.5
        for attempt in range(attempts):
            await self._clob.wait()
            try:
                r = await self._client.get(f"{self._cfg.clob_api_url}/book",
                                           params={"token_id": token_id})
                if r.status_code == 429:
                    if attempt < attempts - 1:
                        await asyncio.sleep(delay)
                        delay *= 2
                        continue
                    logger.warning("CLOB book rate-limited after %d attempts", attempts)
                    return None
                r.raise_for_status()
                return r.json()
            except Exception as exc:  # noqa: BLE001
                if attempt < attempts - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                logger.warning("CLOB book failed after %d attempts: %s", attempts, exc)
                return None
        return None

    async def trades(self, condition_id: str, limit: int = 1000, offset: int = 0,
                     attempts: int = 5) -> list[dict] | None:
        """One page of trade-level fills for a market (data-api).

        Unlike /prices-history (1-minute resolution) this returns individual
        fills with second-level timestamps -- ~1,000 per 5-minute window,
        covering 50-170 distinct seconds. Caller pages with `offset` until a
        short page comes back.

        Returns None on FAILURE and [] on "genuinely no trades". The
        distinction matters: an earlier version returned [] for both, so a
        rate-limited backfill looked like a completed one and silently lost
        two thirds of its windows.
        """
        delay = 1.0
        for attempt in range(attempts):
            await self._data.wait()
            try:
                r = await self._client.get(
                    f"{self._cfg.data_api_url}/trades",
                    params={"market": condition_id, "limit": limit, "offset": offset})
                if r.status_code == 429:
                    if attempt < attempts - 1:
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, 16.0)
                        continue
                    return None
                r.raise_for_status()
                j = r.json()
                return j if isinstance(j, list) else []
            except Exception as exc:  # noqa: BLE001
                if attempt < attempts - 1:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 16.0)
                    continue
                logger.warning("data-api trades failed (offset=%d): %s", offset, exc)
                return None
        return None

    async def midpoint(self, token_id: str) -> float | None:
        """Current CLOB midpoint price for a token (live)."""
        await self._clob.wait()
        try:
            r = await self._client.get(f"{self._cfg.clob_api_url}/midpoint",
                                       params={"token_id": token_id})
            r.raise_for_status()
            return float(r.json().get("mid"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("CLOB midpoint failed: %s", exc)
            return None

    async def spread(self, token_id: str) -> float | None:
        """Current CLOB bid/ask spread for a token (live)."""
        await self._clob.wait()
        try:
            r = await self._client.get(f"{self._cfg.clob_api_url}/spread",
                                       params={"token_id": token_id})
            r.raise_for_status()
            return float(r.json().get("spread"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("CLOB spread failed: %s", exc)
            return None
