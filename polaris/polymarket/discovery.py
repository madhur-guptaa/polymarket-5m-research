"""
polaris/polymarket/discovery.py

Live market discovery for the serving layer: find the CURRENT window's market
and read its CLOB prices, so the API can compute the live edge (our P(up) vs
the market's YES price).

Distinct from backfill.py (historical, exact slugs). Live has to probe a few
timestamp offsets [0, +window, -window] to absorb clock skew / early-created
markets at the boundary, and caches results briefly (markets don't change
within a window; prices refresh a few times a minute).

Strike/CEX-proxy discovery from the old market_discovery.py is intentionally
omitted — it isn't needed for the edge (the YES price already encodes the
market's probability). It can be added later if we want to display the strike.
"""

from __future__ import annotations

import logging
import time

from polaris.polymarket.client import PolymarketClient

logger = logging.getLogger(__name__)

_MARKET_TTL = 20.0   # market identity is stable within a window
_PRICE_TTL = 4.0     # CLOB prices refresh a few times a minute


class LiveDiscovery:
    """Resolves the active market + live YES/NO prices per symbol, with caching."""

    def __init__(self, client: PolymarketClient, window_sec: int) -> None:
        self._client = client
        self._window = window_sec
        self._market: dict[str, tuple[float, object]] = {}   # symbol -> (ts, MarketRef|None)
        self._price: dict[str, tuple[float, dict | None]] = {}

    async def active_market(self, symbol: str):
        """The current window's MarketRef (or None). Probes offsets, cached."""
        now = time.time()
        hit = self._market.get(symbol)
        if hit and now - hit[0] < _MARKET_TTL:
            return hit[1]

        base = int(now) - (int(now) % self._window)
        ref = None
        for off in (0, self._window, -self._window):
            ref = await self._client.resolve_market(symbol, base + off)
            if ref is not None:
                break
        self._market[symbol] = (now, ref)
        if ref is None:
            logger.warning("[%s] no active Polymarket market near %d", symbol, base)
        return ref

    async def prices(self, symbol: str) -> dict | None:
        """Live {yes_price, no_price, spread, market_prob_up} for the active market."""
        now = time.time()
        hit = self._price.get(symbol)
        if hit and now - hit[0] < _PRICE_TTL:
            return hit[1]

        ref = await self.active_market(symbol)
        out = None
        if ref is not None:
            yes = await self._client.midpoint(ref.token_yes)
            spr = await self._client.spread(ref.token_yes)
            if yes is not None:
                out = {
                    "yes_price": round(yes, 4),
                    "no_price": round(1.0 - yes, 4),
                    "spread": round(spr, 4) if spr is not None else None,
                    "market_prob_up": round(yes, 4),
                    "title": ref.title,
                }
        self._price[symbol] = (now, out)
        return out
