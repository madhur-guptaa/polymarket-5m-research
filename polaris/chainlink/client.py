"""
polaris/chainlink/client.py

Free, credential-free access to the Chainlink TWAP-60s series that Polymarket
resolves its crypto Up/Down markets against, via Polymarket's own relay.

MEASURED BEHAVIOUR, which differs from the published docs in three ways that
each break the obvious implementation:

  1. The first frame is an EMPTY STRING, not JSON. json.loads on it raises.
  2. payload.data is an ARRAY of ~58 per-second points -- a snapshot of the
     trailing minute. The docs state "There is no snapshot, history, or replay
     after a disconnect"; there is.
  3. After that one batch the socket pushes NOTHING further. Only
     type="update" yields anything at all -- "subscribe" and "snapshot" both
     return zero points.

So this is a POLLING client, not a streaming one: connect, take the trailing
minute, disconnect, repeat. Polling faster than the ~58s snapshot depth gives
overlapping coverage, which is what closes the gaps a pure listener would leave.

The paid alternative (wss://ws.dataengine.chain.link / api.dataengine.chain.link)
needs CHAINLINK_CLIENT_ID and CHAINLINK_CLIENT_SECRET and a server clock within
five seconds of Chainlink's, but is the only source of HISTORY. This relay
cannot be backfilled.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Final

import websockets

logger = logging.getLogger(__name__)

RELAY_URL: Final[str] = "wss://ws-live-data.polymarket.com"

# Relay symbol -> our internal convention.
SYMBOL_MAP: Final[dict[str, str]] = {"btc/usd": "BTCUSDT", "eth/usd": "ETHUSDT"}
_REVERSE: Final[dict[str, str]] = {v: k for k, v in SYMBOL_MAP.items()}

TOPIC_BY_WINDOW: Final[dict[int, str]] = {
    30: "crypto_prices_twap_thirty",
    60: "crypto_prices_twap_sixty",
}


def _subscription(relay_symbol: str, window_s: int) -> dict:
    # `filters` is a JSON STRING nested inside the JSON message, not an object.
    return {"topic": TOPIC_BY_WINDOW[window_s], "type": "update",
            "filters": json.dumps({"symbol": relay_symbol})}


async def snapshot(symbols: list[str], window_s: int = 60,
                   listen_sec: float = 8.0,
                   timeout_sec: float = 20.0) -> list[tuple[str, int, int, float, str]]:
    """One connect -> trailing-minute snapshot -> disconnect.

    Returns (symbol, ts_seconds, window_s, value, full_accuracy_value) rows.
    Raises nothing: on any failure it returns what it managed to collect, since
    a collector that dies on one bad socket loses data it can never refetch.
    """
    relay_syms = [_REVERSE[s] for s in symbols if s in _REVERSE]
    if not relay_syms:
        return []
    subs = [_subscription(rs, window_s) for rs in relay_syms]
    rows: list[tuple[str, int, int, float, str]] = []
    misaligned = 0
    try:
        async with websockets.connect(RELAY_URL, open_timeout=timeout_sec) as ws:
            await ws.send(json.dumps({"action": "subscribe", "subscriptions": subs}))
            loop = asyncio.get_running_loop()
            deadline = loop.time() + listen_sec
            while loop.time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(),
                                                 timeout=max(0.1, deadline - loop.time()))
                except (asyncio.TimeoutError, websockets.ConnectionClosed):
                    break
                if not raw:          # the empty first frame
                    continue
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                for item in (msg if isinstance(msg, list) else [msg]):
                    payload = (item or {}).get("payload") or {}
                    sym = SYMBOL_MAP.get(str(payload.get("symbol", "")).lower())
                    if sym is None:
                        continue
                    for point in payload.get("data") or []:
                        ts_ms = point.get("timestamp")
                        val = point.get("value")
                        if ts_ms is None or val is None:
                            continue
                        if ts_ms % 1000:
                            misaligned += 1
                        rows.append((sym, int(ts_ms) // 1000, window_s, float(val),
                                     str(point.get("full_accuracy_value") or "")))
    except Exception as exc:  # noqa: BLE001
        logger.warning("chainlink relay snapshot failed: %s: %s",
                       type(exc).__name__, exc)
    if misaligned:
        # Truncating sub-second ticks to whole seconds would silently corrupt
        # any TWAP computed from this table, so it must be loud.
        logger.error("chainlink relay returned %d NON-SECOND-ALIGNED timestamps; "
                     "the whole-second assumption in chainlink_twap no longer holds",
                     misaligned)
    return rows
