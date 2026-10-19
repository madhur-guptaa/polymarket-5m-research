"""
polaris/model/backtest.py

Walk-forward paper-trade backtest — the feasibility verdict. READ-ONLY: it
books hypothetical trades against historical Polymarket prices; there is no
execution anywhere.

Per test day D (after a warmup):
  1. fit the model on windows BEFORE D only (rolling `train_days`), so D never
     leaks into its own prediction — the P(up) is honestly out-of-sample.
  2. score every observation-second in D (thinned to `cadence_s` for speed) with
     predict_batch -> P(up), n_eff.
  3. join the historical Polymarket price at that second; edge = P(up)-yes
     (or (1-P(up))-no). The FIRST second where edge>=min_edge AND n_eff>=min_n_eff
     fires ONE paper trade for that window at that second's price.
  4. hold to expiry; payoff = (1-price) if the side won else -price; flat $1 stake.

Aggregate -> P&L, win rate, #trades, avg edge, equity curve.

The model is chosen by `model_kind` and built via model/factory.py, so any
ProbabilityModel (KDE, GBM, ...) runs through this identical harness — same
days, same gates, same market prices — which is what makes a head-to-head
comparison meaningful. KDE bandwidths come from config; GBM params are passed
in. The cadence thinning is a speed approximation, not a leakage issue.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from psycopg_pool import AsyncConnectionPool

from polaris.core.config import Settings
from polaris.model.base import FeatureSet
from polaris.model.factory import build_model
from polaris.model.dataset import load_dataset

logger = logging.getLogger(__name__)



@dataclass
class BacktestResult:
    n_test_windows: int
    n_trades: int
    n_wins: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    avg_edge: float
    trades: pd.DataFrame       # one row per booked trade, time-ordered
    equity: list[float]        # cumulative P&L


def walk_forward_candidates(
    fs: FeatureSet, poly: pd.DataFrame, cfg: Settings, *,
    train_days: int = 7, warmup_days: int = 2, cadence_s: int = 5,
    min_train_obs: int = 5000, model_kind: str | None = None,
    model_params: dict | None = None,
) -> tuple[pd.DataFrame, int]:
    """Run the (expensive) walk-forward once: fit-on-past, score each test-day
    second, join market prices. Returns every scored+priced second (before any
    edge/n_eff filter) plus the number of test windows. min_edge is applied
    later in _book(), so a sweep reuses this without recomputing.
    """
    model_kind = model_kind or cfg.model.kind
    day = (fs.window_start_ts // 86400).astype(np.int64)
    test_days = np.unique(day)[warmup_days:]

    # Which (window, second) rows HAVE a market price. Scoring is restricted to
    # these rather than to a fixed every-Nth-second grid, because a sparse price
    # source defeats the grid: poly_book_snapshots samples every ~5s at an
    # arbitrary phase, so `times % 5 == 0` lines up with only ~20% of snapshots
    # (measured: 598 of 2,677 windows had any gate-time snapshot it could see).
    # The walk-forward then silently ran on a fifth of the data and traded at
    # whichever second happened to align, not at the first second the edge
    # appeared -- which is what the live paper trader does.
    wsec = cfg.aggregate.window_sec
    price_key = set((poly["window_start"].astype(np.int64) * wsec
                     + poly["elapsed_seconds"].astype(np.int64)).tolist())
    row_key = fs.window_start_ts.astype(np.int64) * wsec + fs.times.astype(np.int64)
    priced = np.fromiter((k in price_key for k in row_key), dtype=bool, count=fs.n)
    # Cadence thinning is kept only for DENSE sources (every second priced, e.g.
    # the forward-filled table), where it is a pure speed knob. A sparse source
    # is already thin; thinning it again throws real prices away.
    per_window = poly.groupby("window_start").size().median() if len(poly) else 0
    dense = per_window >= wsec / 2
    logger.info("price rows per window (median) %.0f -> %s source; cadence %s",
                per_window, "dense" if dense else "sparse",
                f"{cadence_s}s" if dense else "ignored (score every priced second)")
    logger.info("backtest walk-forward: model=%s testing %d days (train_days=%d, cadence=%ds)",
                model_kind, len(test_days), train_days, cadence_s)

    parts: list[pd.DataFrame] = []
    n_test_windows = 0
    for D in test_days:
        train_mask = (day < D) & (day >= D - train_days)
        if int(train_mask.sum()) < min_train_obs:
            continue
        q_mask = (day == D) & priced
        if dense:
            q_mask &= fs.times.astype(np.int64) % cadence_s == 0
        if not q_mask.any():
            continue
        # Denominator for %windows = windows that could have traded (had any
        # price), not every window on the day. Counting unpriced windows made the
        # trade rate read 6.5% when it was ~80% of the windows actually priced.
        n_test_windows += len(np.unique(fs.window_start_ts[q_mask]))

        model = build_model(cfg, model_kind, params=model_params).fit(fs.subset(train_mask))
        q = fs.subset(q_mask)
        p_up, n_eff = model.predict_batch(q)
        qdf = pd.DataFrame({
            "window_start": q.window_start_ts, "elapsed_seconds": q.times.astype(int),
            "outcome": q.outcomes, "p_up": p_up, "n_eff": n_eff,
        })
        merged = pd.merge(qdf, poly, on=["window_start", "elapsed_seconds"], how="inner")
        if merged.empty:
            continue
        merged["edge_yes"] = merged["p_up"] - merged["yes_price"]
        merged["edge_no"] = (1.0 - merged["p_up"]) - merged["no_price"]
        merged["is_yes"] = merged["edge_yes"] >= merged["edge_no"]
        merged["edge"] = np.where(merged["is_yes"], merged["edge_yes"], merged["edge_no"])
        parts.append(merged[["window_start", "elapsed_seconds", "is_yes", "edge",
                             "n_eff", "yes_price", "no_price", "outcome", "p_up"]])

    cand = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return cand, n_test_windows


def _book(cand: pd.DataFrame, n_test_windows: int, min_edge: float, min_n_eff: float,
          gate_start: int | None = None, gate_end: int | None = None,
          spread_cost: float = 0.0, fee_rate: float = 0.07) -> BacktestResult:
    """Apply the edge/n_eff/time/spread gate to pre-scored candidates: one
    trade per window (first qualifying second), flat $1, held to expiry.

    gate_start/gate_end mirror TradingSettings.trade_window_start/end (the
    live trade gate in api/edge.py) -- restrict candidates to elapsed_seconds
    in [gate_start, gate_end) before picking the first qualifying second per
    window. None on either side means unbounded on that side (matches the
    session-10 floor-only sweep, which used gate_start alone).

    spread_cost is a FIXED one-sided execution cost in price units (e.g. 0.02
    = 2 cents), applied as a worse fill (price + spread_cost) on whichever
    side is taken, and netted out of edge BEFORE gating -- a real trader
    would size the trade off the cost-adjusted edge, not the raw one. This is
    an assumption swept for sensitivity, not a measured quantity: our only
    Polymarket data is last-TRADE prices (poly_historical_prices), not the
    order book, and yes_price+no_price-1 (the schema's "maker spread") turns
    out to be unusable as a per-tick spread proxy -- averages ~0.0001 with
    p10/p90 of only +-0.005, and the wide tails (-0.88..+0.83) are
    forward-fill/illiquidity artifacts, not real spread signal. Default 0.0
    reproduces the original fair-price (optimistic) fills exactly."""
    empty = BacktestResult(n_test_windows, 0, 0, 0.0, 0.0, 0.0, 0.0, pd.DataFrame(), [])
    if cand.empty:
        return empty
    # Fill price per side = quoted price + spread_cost; the taker fee (Polymarket
    # crypto: fee_rate * p * (1 - p) per share) is netted out of the edge BEFORE
    # gating and out of P&L after, because it is paid on every fill.
    yes_px = cand["yes_price"] + spread_cost
    no_px = cand["no_price"] + spread_cost
    yes_fee = fee_rate * yes_px * (1.0 - yes_px)
    no_fee = fee_rate * no_px * (1.0 - no_px)
    edge_yes = cand["p_up"] - yes_px - yes_fee
    edge_no = (1.0 - cand["p_up"]) - no_px - no_fee
    is_yes = edge_yes >= edge_no
    cand = cand.assign(edge=np.where(is_yes, edge_yes, edge_no), is_yes=is_yes,
                       _fill_price=np.where(is_yes, yes_px, no_px),
                       _fee=np.where(is_yes, yes_fee, no_fee))
    mask = (cand["edge"] >= min_edge) & (cand["n_eff"] >= min_n_eff)
    if gate_start is not None:
        mask &= cand["elapsed_seconds"] >= gate_start
    if gate_end is not None:
        mask &= cand["elapsed_seconds"] < gate_end
    sig = cand[mask]
    if sig.empty:
        return empty
    first = sig.sort_values("elapsed_seconds").groupby("window_start").first().reset_index()
    price = first["_fill_price"].to_numpy()
    won = np.where(first["is_yes"], first["outcome"] == 1, first["outcome"] == 0)
    fee = first["_fee"].to_numpy()
    pnl = np.where(won, 1.0 - price, -price) - fee
    trades = pd.DataFrame({
        "window_start": first["window_start"].astype(int),
        "elapsed": first["elapsed_seconds"].astype(int),
        "side": np.where(first["is_yes"], "YES", "NO"),
        "price": np.round(price, 4), "fee": np.round(fee, 5), "edge": np.round(first["edge"], 4),
        "outcome": first["outcome"].astype(int), "won": won, "pnl": np.round(pnl, 4),
    }).sort_values("window_start").reset_index(drop=True)
    n = len(trades)
    total = float(trades["pnl"].sum())
    return BacktestResult(
        n_test_windows=n_test_windows, n_trades=n, n_wins=int(trades["won"].sum()),
        win_rate=float(trades["won"].mean()), total_pnl=total, avg_pnl=total / n,
        avg_edge=float(trades["edge"].mean()), trades=trades, equity=trades["pnl"].cumsum().tolist(),
    )


def run_backtest(fs: FeatureSet, poly: pd.DataFrame, cfg: Settings, *,
                 min_edge: float = 0.03, min_n_eff: float = 20.0,
                 train_days: int = 7, warmup_days: int = 2, cadence_s: int = 5,
                 gate_start: int | None = None, gate_end: int | None = None,
                 spread_cost: float = 0.0, model_kind: str | None = None,
                 model_params: dict | None = None) -> BacktestResult:
    cand, n_win = walk_forward_candidates(fs, poly, cfg, train_days=train_days,
                                          warmup_days=warmup_days, cadence_s=cadence_s,
                                          model_kind=model_kind, model_params=model_params)
    return _book(cand, n_win, min_edge, min_n_eff, gate_start, gate_end, spread_cost, fee_rate=cfg.trading.taker_fee_rate)


PRICE_SOURCES = ("ffill", "trades", "book")


async def _load_poly(pool: AsyncConnectionPool, symbol: str,
                     source: str = "ffill") -> pd.DataFrame:
    """Market prices per (window, second), as (window_start, elapsed_seconds,
    yes_price, no_price).

    THREE SOURCES, IN ASCENDING ORDER OF TRUSTWORTHINESS -- the choice matters
    more than anything else in this module, because it invalidated an earlier
    headline result:

      "ffill"  poly_historical_prices, from /prices-history. DO NOT TRUST for
               P&L: that endpoint serves 1-MINUTE resolution (~5 real points
               per window) and the backfill forward-filled it to per-second,
               leaving prices a mean 12.9c from the real book inside the trade
               gate. Kept only to reproduce the retracted numbers.
      "trades" poly_trades, real fills at second resolution. Mean 3.5c from the
               book in the gate and essentially unbiased (signed -0.0014).
               Covers 60+ days. Still noisy relative to a 5c min_edge.
      "book"   poly_book_snapshots, the actual quotes we would have traded
               against. The only unimpeachable source, but forward-only --
               it exists from 2026-09-06 onward and nothing can backfill it.
    """
    if source not in PRICE_SOURCES:
        raise ValueError(f"unknown price source {source!r} (expected {PRICE_SOURCES})")

    if source == "ffill":
        sql = ("SELECT window_start, elapsed_seconds, yes_price, no_price "
               "FROM poly_historical_prices WHERE symbol = %s")
        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(sql, (symbol,))
            rows = await cur.fetchall()
        return pd.DataFrame(rows, columns=["window_start", "elapsed_seconds",
                                           "yes_price", "no_price"])

    if source == "book":
        # The YES book's midpoint is the price; the NO side is its complement
        # (both are stored, but using 1-yes keeps the pair internally consistent).
        sql = ("SELECT window_start, elapsed_seconds, mid AS yes_price "
               "FROM poly_book_snapshots WHERE symbol = %s AND side = 'YES' "
               "AND mid IS NOT NULL")
        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(sql, (symbol,))
            rows = await cur.fetchall()
        df = pd.DataFrame(rows, columns=["window_start", "elapsed_seconds", "yes_price"])
        df["no_price"] = 1.0 - df["yes_price"]
        return df

    # "trades": densify real fills to per-second, forward-filling only WITHIN a
    # window (never across, so one window's last print can't leak into the next).
    sql = ("SELECT window_start, elapsed_seconds, "
           "       CASE WHEN outcome = 'Up' THEN price ELSE 1 - price END AS yes_px "
           "FROM poly_trades WHERE symbol = %s")
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(sql, (symbol,))
        rows = await cur.fetchall()
    f = pd.DataFrame(rows, columns=["window_start", "elapsed_seconds", "yes_px"])
    if f.empty:
        return pd.DataFrame(columns=["window_start", "elapsed_seconds", "yes_price", "no_price"])
    # last print wins within a second
    f = (f.sort_values(["window_start", "elapsed_seconds"])
           .groupby(["window_start", "elapsed_seconds"], as_index=False).last())
    # reindex each window onto every second it spans, then forward-fill
    out = []
    for w, g in f.groupby("window_start", sort=False):
        g = g.set_index("elapsed_seconds")["yes_px"]
        dense = g.reindex(range(int(g.index.min()), int(g.index.max()) + 1)).ffill()
        out.append(pd.DataFrame({"window_start": w,
                                 "elapsed_seconds": dense.index.astype(int),
                                 "yes_price": dense.to_numpy()}))
    df = pd.concat(out, ignore_index=True)
    df["no_price"] = 1.0 - df["yes_price"]
    return df


async def run(pool: AsyncConnectionPool, cfg: Settings, symbol: str, *,
              days: int = 30, min_edge: float = 0.03, min_n_eff: float = 20.0,
              train_days: int = 7, cadence_s: int = 5,
              sweep: list[float] | None = None,
              gate_start: int | None = None, gate_end: int | None = None,
              gate_sweep: list[int] | None = None,
              spread_cost: float = 0.0,
              spread_sweep: list[float] | None = None,
              model_kind: str | None = None,
              model_params: dict | None = None,
              price_source: str = "ffill") -> BacktestResult | None:
    """Load data, run the walk-forward once, and print either a single result
    or one sweep (min_edge, gate_start, or spread_cost) -- candidates computed
    once, re-gated each way; the walk-forward itself never reruns."""
    fs = await load_dataset(pool, cfg, symbol, days=days)
    poly = await _load_poly(pool, symbol, price_source)
    logger.info("[%s] price source=%s -> %d priced (window,second) rows",
                symbol, price_source, len(poly))
    if fs is None or poly.empty:
        logger.error("[%s] missing data (obs=%s, poly rows=%s)", symbol, fs and fs.n, len(poly))
        return None

    cand, n_win = walk_forward_candidates(fs, poly, cfg, train_days=train_days,
                                          cadence_s=cadence_s, model_kind=model_kind,
                                          model_params=model_params)

    n_sweeps = sum(1 for s in (spread_sweep, gate_sweep, sweep) if s)
    if n_sweeps > 1:
        logger.warning("multiple sweeps passed (spread_sweep=%s gate_sweep=%s sweep=%s) -- "
                        "only spread_sweep > gate_sweep > sweep is honored, the rest are ignored",
                        bool(spread_sweep), bool(gate_sweep), bool(sweep))

    if spread_sweep:
        # How many cents of one-sided execution cost the current gate/edge can
        # absorb before win rate / P&L go negative. spread_cost is an ASSUMED
        # fixed cost, not measured -- see _book()'s docstring for why
        # yes_price+no_price-1 isn't usable as a real per-tick spread proxy.
        header = (f"\n=== spread_cost sweep · {symbol} (min_edge={min_edge}, n_eff>={min_n_eff}, "
                  f"gate=[{gate_start},{gate_end}), train_days={train_days}, cadence={cadence_s}s) ===")
        print(header)
        print(f"{'spread':>7} {'trades':>7} {'%win':>7} {'avgP&L':>9} {'totP&L':>9} {'%windows':>9}")
        for sc in spread_sweep:
            r = _book(cand, n_win, min_edge, min_n_eff, gate_start, gate_end, sc, fee_rate=cfg.trading.taker_fee_rate)
            wpct = 100 * r.n_trades / max(1, r.n_test_windows)
            print(f"{sc:>6.3f} {r.n_trades:>7} {r.win_rate * 100:>6.1f}% "
                  f"{r.avg_pnl:>+9.4f} {r.total_pnl:>+9.2f} {wpct:>8.1f}%")
        print("")
        return None

    if gate_sweep:
        # Floor-only sweep (gate_end fixed, e.g. None or 90) -- reproduces the
        # session-10 "first qualifying second >= gate" table.
        header = (f"\n=== gate_start sweep · {symbol} (min_edge={min_edge}, n_eff>={min_n_eff}, "
                  f"gate_end={gate_end}, spread_cost={spread_cost}, train_days={train_days}, "
                  f"cadence={cadence_s}s) ===")
        print(header)
        print(f"{'gate>=':>7} {'trades':>7} {'%win':>7} {'avgP&L':>9} {'totP&L':>9} {'%windows':>9}")
        for gs in gate_sweep:
            r = _book(cand, n_win, min_edge, min_n_eff, gs, gate_end, spread_cost, fee_rate=cfg.trading.taker_fee_rate)
            wpct = 100 * r.n_trades / max(1, r.n_test_windows)
            print(f"{gs:>6}s {r.n_trades:>7} {r.win_rate * 100:>6.1f}% "
                  f"{r.avg_pnl:>+9.4f} {r.total_pnl:>+9.2f} {wpct:>8.1f}%")
        print("")
        return None

    if sweep:
        header = (f"\n=== min_edge sweep · {symbol} (n_eff>={min_n_eff}, gate=[{gate_start},{gate_end}), "
                  f"spread_cost={spread_cost}, train_days={train_days}, cadence={cadence_s}s) ===")
        print(header)
        print(f"{'min_edge':>9} {'trades':>7} {'%win':>7} {'avgP&L':>9} {'totP&L':>9} {'%windows':>9}")
        for me in sweep:
            r = _book(cand, n_win, me, min_n_eff, gate_start, gate_end, spread_cost, fee_rate=cfg.trading.taker_fee_rate)
            wpct = 100 * r.n_trades / max(1, r.n_test_windows)
            print(f"{me:>9.3f} {r.n_trades:>7} {r.win_rate * 100:>6.1f}% "
                  f"{r.avg_pnl:>+9.4f} {r.total_pnl:>+9.2f} {wpct:>8.1f}%")
        print("")
        return None

    res = _book(cand, n_win, min_edge, min_n_eff, gate_start, gate_end, spread_cost, fee_rate=cfg.trading.taker_fee_rate)
    print("\n".join([
        "", "=== walk-forward paper-trade backtest ===",
        f"symbol           {symbol}",
        f"test windows     {res.n_test_windows}",
        f"trades           {res.n_trades}  ({res.n_trades / max(1, res.n_test_windows) * 100:.1f}% of windows)",
        f"win rate         {res.win_rate * 100:.1f}%",
        f"avg edge (entry) {res.avg_edge * 100:.2f}%",
        f"total P&L        {res.total_pnl:+.2f}  (flat $1/trade)",
        f"avg P&L / trade  {res.avg_pnl:+.4f}",
        f"min_edge={min_edge} min_n_eff={min_n_eff} gate=[{gate_start},{gate_end}) "
        f"spread_cost={spread_cost} train_days={train_days} cadence={cadence_s}s",
        "",
    ]))
    return res
