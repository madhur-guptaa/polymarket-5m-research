"""
polaris/model/calibrate.py

The two-phase calibration, orchestrated against the ProbabilityModel protocol.

  Step 1 (model-specific): model.optimize -> bandwidths minimising block-CV Brier
  final LOOCV: model.loocv_probabilities -> honest out-of-sample p_up per obs
  Step 2 (model-agnostic): join p_up with poly_historical_prices, optimise risk

Prints the resulting bandwidths + risk params in .env form. This is also the
Phase-0 gate: if the block-CV Brier isn't clearly under ~0.20, the model has no
real predictive power and the rest doesn't matter.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from psycopg_pool import AsyncConnectionPool

from polaris.core.config import Settings
from polaris.model.dataset import load_dataset
from polaris.model.factory import build_model
from polaris.model import risk

logger = logging.getLogger(__name__)




# Bandwidth name -> .env key, per model kind. _print_env emits whichever set
# the calibrated model actually has.
_ENV_KEYS: dict[str, str] = {
    "delta": "BW_DELTA", "time": "BW_TIME", "momentum": "BW_MOMENTUM",
    "imbalance": "BW_IMBALANCE", "volatility": "BW_VOLATILITY", "tod": "BW_TOD",
    "twap_sofar": "BW_TWAP_SOFAR", "req": "BW_REQ", "locked": "BW_LOCKED", "spot": "BW_SPOT",
}


async def _load_poly(pool: AsyncConnectionPool, symbol: str) -> pd.DataFrame:
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT window_start, elapsed_seconds, yes_price, no_price "
            "FROM poly_historical_prices WHERE symbol = %s", (symbol,))
        rows = await cur.fetchall()
    return pd.DataFrame(rows, columns=["window_start", "elapsed_seconds", "yes_price", "no_price"])


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


async def run_calibration(
    pool: AsyncConnectionPool, cfg: Settings, symbol: str,
    days: int | None = None, step1_trials: int = 100, step2_trials: int = 500,
    backend: str = "numpy", train_cadence: int = 1,
) -> dict | None:
    """Full two-phase calibration for one symbol. Returns the tuned params.

    backend="mlx" (Apple GPU) makes Step 1 tractable at real 14-day scale --
    numpy LOOCV is O(N^2) and projects to hours per trial at N~10^6 (measured:
    ~216 min/call at N=928K; MLX is ~9.5x faster at that N, ~23 min/call --
    still too slow for a multi-trial search on its own).

    train_cadence subsamples the SEARCH set (every Nth second by elapsed
    time, same idea as the backtester's cadence thinning) so Step 1's ~100
    trials finish in a practical window; cadence=5 cuts N (and LOOCV's O(N^2)
    cost) by ~25x. The baseline/tuned Brier reported here and Step 2 both run
    on the FULL fs regardless of train_cadence -- thinning only speeds up the
    search, it never touches what gets reported or risk-optimized."""
    fs = await load_dataset(pool, cfg, symbol, days)
    if fs is None:
        return None

    # Through the factory: MODEL_KIND picks the estimator (kde_twap by default).
    model = build_model(cfg, backend=backend)
    search_fs = fs
    if train_cadence > 1:
        mask = fs.times.astype(np.int64) % train_cadence == 0
        search_fs = fs.subset(mask)
        logger.info("[%s] Step-1 search set thinned by cadence=%d: %d -> %d obs",
                    symbol, train_cadence, fs.n, search_fs.n)
        if search_fs.n == 0:
            # Otherwise Optuna's objective silently computes np.mean of an
            # empty array (NaN, no exception) every trial; every trial gets
            # marked FAILED with no error surfaced, and optimize() then dies
            # deep inside Optuna on study.best_params ("Record does not
            # exist") with no hint the real cause was an empty search set.
            raise ValueError(
                f"[{symbol}] train_cadence={train_cadence} thinned the Step-1 search set "
                f"to 0 observations (from {fs.n}). Lower --train-cadence or increase --days."
            )

    # Baseline block-CV Brier at the starting bandwidths, for reference.
    # Always on the FULL set -- this is the honest number, not a search proxy.
    base_brier = brier(model.loocv_probabilities(fs), fs.outcomes)
    logger.info("[%s] baseline block-CV Brier = %.4f (up-rate Brier = %.4f)",
                symbol, base_brier, brier(np.full(fs.n, fs.outcomes.mean()), fs.outcomes))

    # ── Step 1: bandwidths (searched on search_fs, reported on the full fs) ──
    model.fit(search_fs)
    best_bw = model.optimize(search_fs, n_trials=step1_trials)
    model.bw = type(model.bw)(**best_bw)
    model.fit(fs)
    p_up = model.loocv_probabilities(fs)
    tuned_brier = brier(p_up, fs.outcomes)
    logger.info("[%s] tuned block-CV Brier = %.4f", symbol, tuned_brier)

    # ── Step 2: risk (needs Polymarket prices) ──
    poly = await _load_poly(pool, symbol)
    risk_params: dict = {}
    if poly.empty:
        logger.warning("[%s] no poly_historical_prices — skipping Step 2 (run backfill-poly)", symbol)
    else:
        model_df = pd.DataFrame({
            "window_start": fs.window_start_ts,
            "elapsed_seconds": fs.times.astype(int),
            "outcome": fs.outcomes,
            "p_up": p_up,
        })
        merged = pd.merge(model_df, poly, on=["window_start", "elapsed_seconds"], how="inner")
        logger.info("[%s] merged %d rows (%d windows) for backtest",
                    symbol, len(merged), merged["window_start"].nunique() if len(merged) else 0)
        if merged.empty:
            logger.warning("[%s] model/poly merge produced 0 rows — check window overlap", symbol)
        else:
            risk_params = risk.optimize_risk(merged, n_trials=step2_trials)

    result = {"symbol": symbol, "brier": tuned_brier, "bandwidths": best_bw, "risk": risk_params}
    _print_env(result)
    return result


def _print_env(result: dict) -> None:
    bw, rk = result["bandwidths"], result.get("risk", {})
    lines = [f"# {result['symbol']}  (block-CV Brier = {result['brier']:.4f})"]
    lines += [f"{_ENV_KEYS[k]}={v:.4f}" for k, v in bw.items()]
    if rk:
        lines += [
            f"MIN_EDGE={rk['min_edge']:.4f}", f"MAX_KELLY_FRACTION={rk['max_kelly']:.4f}",
            f"TRADE_WINDOW_START={rk['trade_window_start']}",
            f"TRADE_WINDOW_END={rk['trade_window_end']}",
        ]
    print("\n".join(["", "=== calibrated config (copy into .env) ===", *lines, ""]))
