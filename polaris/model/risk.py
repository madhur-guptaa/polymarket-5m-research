"""
polaris/model/risk.py

Step 2 — the money machine. MODEL-AGNOSTIC: it takes an out-of-sample p_up
array (from any ProbabilityModel's loocv_probabilities) joined with historical
Polymarket prices, and tunes risk parameters (min-edge, max-Kelly, trade
window) to maximise the annualised Sharpe of a vectorised backtest.

edge = p_up - yes_price (YES side)  or  (1 - p_up) - no_price (NO side).
Position sizing is Half-Kelly, capped at max_kelly. One trade per window (the
first in-window signal). Ported from the old step2_money.py.
"""

from __future__ import annotations

import logging

import numpy as np
import optuna
import pandas as pd

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 365.25 days x 24 h x 12 five-minute windows per hour.
_ANNUALISATION = 105_120


def _half_kelly(edge: pd.Series, price: pd.Series) -> pd.Series:
    """Half-Kelly fraction for a binary contract bought at `price` with `edge`."""
    safe = price.clip(lower=0.01, upper=0.99)
    b = (1.0 / safe) - 1.0            # decimal odds
    p = edge + safe                    # our win probability
    f = (b * p - (1.0 - p)) / b        # full Kelly
    return (f * 0.5).clip(lower=0.0)


def backtest_sharpe(
    df: pd.DataFrame, min_edge: float, max_kelly: float,
    window_start: int, window_end: int,
) -> float:
    """Annualised Sharpe of the strategy. 0.0 if <10 trades or zero variance."""
    d = df[(df["elapsed_seconds"] >= window_start) & (df["elapsed_seconds"] <= window_end)].copy()
    if d.empty:
        return 0.0
    d["edge_yes"] = d["p_up"] - d["yes_price"]
    d["edge_no"] = (1.0 - d["p_up"]) - d["no_price"]
    d = d[(d["edge_yes"] > min_edge) | (d["edge_no"] > min_edge)]
    if d.empty:
        return 0.0

    trades = d.groupby("window_start").first().reset_index()  # first signal per window
    is_yes = trades["edge_yes"] > trades["edge_no"]
    exec_price = np.where(is_yes, trades["yes_price"], trades["no_price"])
    edge = np.where(is_yes, trades["edge_yes"], trades["edge_no"])
    size = _half_kelly(pd.Series(edge), pd.Series(exec_price)).clip(upper=max_kelly)

    win = np.where(is_yes, trades["outcome"] == 1, trades["outcome"] == 0)
    pnl = np.asarray(size) * np.where(win, 1.0 - exec_price, -exec_price)
    if len(pnl) < 10:
        return 0.0
    sd = float(np.std(pnl, ddof=0))
    if sd == 0.0:
        return 0.0
    return float(np.mean(pnl) / sd) * np.sqrt(_ANNUALISATION)


def optimize_risk(df: pd.DataFrame, n_trials: int = 500) -> dict:
    """TPE search over (min_edge, max_kelly, trade window) maximising Sharpe."""
    def objective(trial: optuna.Trial) -> float:
        me = trial.suggest_float("min_edge", 0.01, 0.10)
        mk = trial.suggest_float("max_kelly", 0.01, 0.15)
        ws = trial.suggest_int("trade_window_start", 0, 180)
        we = trial.suggest_int("trade_window_end", ws + 10, 290)
        return backtest_sharpe(df, me, mk, ws, we)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=7))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("Step 2 best Sharpe=%.2f params=%s", study.best_value, study.best_params)
    return {**study.best_params, "sharpe": study.best_value}
