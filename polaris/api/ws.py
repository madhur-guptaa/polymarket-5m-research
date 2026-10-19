"""
polaris/api/ws.py

Per-symbol WebSocket: /ws/{symbol}. Two concurrent coroutines — a sender that
pushes data at tiered cadences, and a receiver that answers keepalive pings.

  tick       every 1s   live window state + P(up)
  micro      every 10s  10-second volume bars
  matrix     every 60s  probability-surface grid
  macro      every 60s  1-hour price trend
  polymarket every 5s   market YES/NO price + edge/trade-gate

(polymarket cadence matches polymarket/discovery.py's own 4s price-cache
TTL -- polling faster than that would just re-serve the same cached value;
slower left the "live" edge readout up to 25s stale for no reason, since
the CLOB rate limit (90 req/s) has nowhere near enough load here to justify
it. backtest / calibration messages come with their own future work.)
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from polaris.api.edge import compute_edge
from polaris.api.live import compute_tick
from polaris.core.db import load_10s_volume, load_1h_trend

logger = logging.getLogger(__name__)
ws_router = APIRouter()


async def _sender(ws: WebSocket, pool, cfg, cache, discovery, symbol: str) -> None:
    i = 0
    last_tick: dict = {}
    while True:
        last_tick = await compute_tick(pool, cfg, cache, symbol)
        await ws.send_json({"type": "tick", "data": last_tick})
        if i % 10 == 0:
            await ws.send_json({"type": "micro", "data": await load_10s_volume(pool, symbol)})
        if i % 60 == 0:
            st = await cache.get(symbol)
            if st is not None:
                await ws.send_json({"type": "matrix", "data": {**st.grid, "symbol": symbol}})
            await ws.send_json({"type": "macro", "data": await load_1h_trend(pool, symbol)})
        if i % 5 == 0:
            poly = await discovery.prices(symbol)
            xs = last_tick.get("x") or []
            elapsed = xs[-1] if xs else None
            edge = compute_edge(last_tick.get("prob"), last_tick.get("n_eff"),
                                poly, cfg.trading, elapsed)
            await ws.send_json({"type": "polymarket", "data": {"poly": poly, "edge": edge}})
        i += 1
        await asyncio.sleep(1.0)


async def _receiver(ws: WebSocket) -> None:
    while True:
        msg = await ws.receive_json()
        if msg.get("type") == "ping":
            await ws.send_json({"type": "pong"})


@ws_router.websocket("/ws/{symbol}")
async def ws_endpoint(ws: WebSocket, symbol: str) -> None:
    sym = symbol.upper()
    s = ws.app.state
    await ws.accept()
    send = asyncio.create_task(_sender(ws, s.pool, s.cfg, s.cache, s.discovery, sym))
    recv = asyncio.create_task(_receiver(ws))
    try:
        await asyncio.gather(send, recv)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] ws error: %s", sym, exc)
    finally:
        send.cancel()
        recv.cancel()
