"""
polaris/api/endpoints.py

REST endpoints for initial dashboard state. The WebSocket (ws.py) streams the
same data live; these give a page-load snapshot and let the frontend fetch on
demand. Paths are under /api.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from polaris.api.edge import compute_edge
from polaris.api.live import compute_tick
from polaris.core.db import load_10s_volume, load_1h_trend, load_latest_trades

router = APIRouter(prefix="/api")


def _state(request: Request):
    return request.app.state.pool, request.app.state.cfg, request.app.state.cache


@router.get("/symbols")
async def symbols(request: Request) -> dict:
    return {"symbols": request.app.state.cfg.binance.symbols}


@router.get("/live/{symbol}")
async def live(symbol: str, request: Request) -> dict:
    pool, cfg, cache = _state(request)
    return await compute_tick(pool, cfg, cache, symbol.upper())


@router.get("/matrix/{symbol}")
async def matrix(symbol: str, request: Request) -> dict:
    _, _, cache = _state(request)
    st = await cache.get(symbol.upper())
    if st is None:
        raise HTTPException(404, detail=f"No training data for {symbol.upper()}")
    return {"status": "ready", "symbol": symbol.upper(), **st.grid}


@router.get("/trades/{symbol}")
async def trades(symbol: str, request: Request, limit: int = 50) -> dict:
    pool = request.app.state.pool
    return {"trades": await load_latest_trades(pool, symbol.upper(), limit)}


@router.get("/micro/{symbol}")
async def micro(symbol: str, request: Request) -> dict:
    return {"micro": await load_10s_volume(request.app.state.pool, symbol.upper())}


@router.get("/macro/{symbol}")
async def macro(symbol: str, request: Request) -> dict:
    return {"macro": await load_1h_trend(request.app.state.pool, symbol.upper())}


@router.get("/polymarket/{symbol}")
async def polymarket(symbol: str, request: Request) -> dict:
    """Live market prices + edge vs the current model P(up)."""
    pool, cfg, cache = _state(request)
    sym = symbol.upper()
    poly = await request.app.state.discovery.prices(sym)
    tick = await compute_tick(pool, cfg, cache, sym)
    xs = tick.get("x") or []
    edge = compute_edge(tick.get("prob"), tick.get("n_eff"), poly, cfg.trading,
                        xs[-1] if xs else None)
    return {"symbol": sym, "poly": poly, "edge": edge}
