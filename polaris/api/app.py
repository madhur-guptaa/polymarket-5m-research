"""
polaris/api/app.py

FastAPI application: opens the pool, builds the per-symbol estimator cache, and
serves REST + WebSocket for the dashboard. Runs on default bandwidths — no MLX
or prior calibration required to serve predictions.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from polaris.api.cache import EstimatorCache
from polaris.api.endpoints import router
from polaris.api.ws import ws_router
from polaris.core.config import get_settings
from polaris.core.db import ensure_schema, open_pool
from polaris.polymarket.client import PolymarketClient
from polaris.polymarket.discovery import LiveDiscovery

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    cfg = get_settings()
    pool = await open_pool(cfg.pg, autocommit=True)  # read-only serving
    await ensure_schema(pool)
    pm_client = PolymarketClient(cfg.polymarket)
    app.state.cfg = cfg
    app.state.pool = pool
    app.state.cache = EstimatorCache(pool, cfg)
    app.state.pm_client = pm_client
    app.state.discovery = LiveDiscovery(pm_client, cfg.aggregate.window_sec)
    logger.info("API ready — symbols=%s", cfg.binance.symbols)
    try:
        yield
    finally:
        await pm_client.aclose()
        await pool.close()


def create_app() -> FastAPI:
    app = FastAPI(title="Polaris", lifespan=_lifespan)
    # Dev-friendly CORS; in prod the API serves the built frontend same-origin.
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )
    app.include_router(router)
    app.include_router(ws_router)
    return app


app = create_app()
