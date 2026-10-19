"""
polaris/report.py

Adjudicate the live forward paper trade against what Polymarket ACTUALLY paid.

This exists as a command, not a notebook, because the project's history is full
of numbers that were computed once in a scratch script, quoted for weeks, and
then turned out to rest on a bad join. Anything quoted as a result should be
reproducible with one command.

THREE THINGS THIS GETS RIGHT, each of which was wrong in an earlier pass:

  1. ONE TRADE PER WINDOW. paper_signals stores a row per second, and many
     seconds inside one window pass the gate. They share a single outcome, so
     counting them as independent trades inflates z by ~sqrt(rows per window)
     -- an early pass reported n=2287 and z=-7.6 when the honest figures were
     n=376 and z=-3.0. The first gated second is taken, which is also the
     realistic execution: you fire once, when the signal first appears.

  2. THE REAL ASK, ON THE SIDE ACTUALLY TRADED. Not the midpoint (untradeable)
     and not the YES book when the signal said NO. Matched to the nearest book
     snapshot within +/-10s.

  3. SETTLEMENT, NOT OUR LABEL. window_observations.outcome disagrees with
     Polymarket on 8-11% of windows (see core/schema.sql). Both are reported so
     the difference stays visible, but only the settlement column is real.

Excess wins are measured against what the PRICE implied, not against 50%:
buying a contract at 0.41 should win 41% of the time. Since 2026-10-07 the null
also includes Polymarket's crypto taker fee (0.07 * p * (1-p) per share, ~1.7c
at p=0.41), which every earlier version of this report omitted -- the break-even
is sum(ask + fee). The standard deviation of the win count is sqrt(sum(p(1-p))).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)

_SIGNALS = """
    SELECT symbol, window_start, elapsed_seconds, side, p_up, edge,
           COALESCE(model_kind, 'unknown') AS model_kind
    FROM paper_signals WHERE gates_passed
"""
_BOOK = """
    SELECT symbol, window_start, elapsed_seconds, side, best_ask
    FROM poly_book_snapshots WHERE best_ask IS NOT NULL
"""
_OURS = """
    SELECT symbol, EXTRACT(EPOCH FROM window_start)::BIGINT, MAX(outcome)
    FROM window_observations WHERE outcome IS NOT NULL GROUP BY 1, 2
"""
_SETTLED = "SELECT symbol, window_start, outcome_up FROM poly_resolutions"


@dataclass(frozen=True)
class Verdict:
    """Outcome of one adjudication pass."""
    label: str
    n: int
    win_rate: float
    avg_ask: float
    excess: float
    z: float
    roc: float

    def line(self) -> str:
        return (f"{self.label:24s} n={self.n:5d}  win {100*self.win_rate:5.1f}%  "
                f"ask {self.avg_ask:.3f}  excess {self.excess:+8.2f}  "
                f"z={self.z:+6.2f}  RoC {100*self.roc:+6.1f}%")


# Polymarket crypto taker fee per share: rate * p * (1 - p). Paid on every fill,
# so the break-even win rate for a contract bought at p is p + fee, not p.
TAKER_FEE_RATE = 0.07


def _score(label: str, ask: np.ndarray, win: np.ndarray,
           fee_rate: float = TAKER_FEE_RATE) -> Verdict:
    """Excess wins over what the ask PLUS the taker fee implied, with z and RoC."""
    n = len(ask)
    if n == 0:
        return Verdict(label, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    cost = ask + fee_rate * ask * (1.0 - ask)
    excess = float(win.sum() - cost.sum())
    sd = float(np.sqrt((ask * (1.0 - ask)).sum()))
    return Verdict(label, n, float(win.mean()), float(ask.mean()), excess,
                   excess / sd if sd > 0 else 0.0,
                   excess / float(cost.sum()) if cost.sum() > 0 else 0.0)


async def load_trades(pool: AsyncConnectionPool, tolerance_s: int = 10) -> pd.DataFrame:
    """One gated trade per window, priced at the real ask on the side traded."""
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(_SIGNALS)
        sig = pd.DataFrame(await cur.fetchall(),
                           columns=["symbol", "window_start", "elapsed", "side", "p_up", "edge",
                                    "model_kind"])
        await cur.execute(_BOOK)
        book = pd.DataFrame(await cur.fetchall(),
                            columns=["symbol", "window_start", "elapsed", "side", "best_ask"])
        await cur.execute(_OURS)
        ours = {(s, int(w)): o for s, w, o in await cur.fetchall()}
        await cur.execute(_SETTLED)
        settled = {(s, int(w)): o for s, w, o in await cur.fetchall()}

    if sig.empty:
        return pd.DataFrame()
    # (1) collapse to the FIRST gated second of each window
    sig = (sig.sort_values(["symbol", "window_start", "elapsed"])
              .groupby(["symbol", "window_start"], as_index=False).first())
    # (2) nearest real ask on the same side
    m = pd.merge_asof(sig.sort_values("elapsed"), book.sort_values("elapsed"),
                      by=["symbol", "window_start", "side"], on="elapsed",
                      direction="nearest", tolerance=tolerance_s)
    m = m.dropna(subset=["best_ask"])
    m["y_ours"] = [ours.get((s, int(w)), np.nan) for s, w in zip(m.symbol, m.window_start)]
    m["y_settled"] = [settled.get((s, int(w)), np.nan) for s, w in zip(m.symbol, m.window_start)]
    m["day"] = pd.to_datetime(m.window_start, unit="s").dt.date
    return m


def _wins(m: pd.DataFrame, col: str) -> np.ndarray:
    """A YES buy wins when it settles Up; a NO buy wins when it settles Down."""
    return np.where(m.side == "YES", m[col], 1 - m[col]).astype(float)


def report(m: pd.DataFrame) -> None:
    """Adjudicate each model's live signals separately.

    One section per model_kind. Pooling them would average a retired model's
    record into the current one's, which is precisely the kind of blended
    number this module exists to prevent.
    """
    if m.empty:
        logger.warning("no gated trades with a matching real ask")
        return
    for kind, g in m.groupby("model_kind", sort=False):
        print(f"\n{'=' * 24} model: {kind} {'=' * 24}")
        _report_one(g)


def _report_one(m: pd.DataFrame) -> None:
    full = m.dropna(subset=["y_ours", "y_settled"])
    if full.empty:
        print("no trades with a settled outcome yet")
        return
    ask = full.best_ask.to_numpy()
    print(f"\ngated trades, one per window, priced at the real ask: {len(full)}")
    print(f"label agreement on traded windows: "
          f"{100*(full.y_ours == full.y_settled).mean():.1f}%\n")
    print(_score("our label (NOT real)", ask, _wins(full, "y_ours")).line())
    print(_score("SETTLEMENT (real)", ask, _wins(full, "y_settled")).line())

    print("\nby symbol, on settlement:")
    for sym, g in full.groupby("symbol"):
        print("  " + _score(sym, g.best_ask.to_numpy(), _wins(g, "y_settled")).line())

    print("\nby day, on settlement:")
    for day, g in full.groupby("day"):
        print("  " + _score(str(day), g.best_ask.to_numpy(), _wins(g, "y_settled")).line())

    # The cut that killed the last positive claim: is any single day carrying it?
    days = sorted(full.day.unique())
    if len(days) > 2:
        print("\nleave-one-day-out, on settlement:")
        for day in days:
            g = full[full.day != day]
            print("  " + _score(f"without {day}", g.best_ask.to_numpy(),
                                _wins(g, "y_settled")).line())

    # How many trades would be needed to resolve an edge of the size we care about?
    p = float(np.mean(ask))
    sd = np.sqrt(p * (1 - p))
    for target in (0.05, 0.10):
        need = (2.0 * sd / (target * p)) ** 2
        print(f"\nto detect {100*target:.0f}% RoC at z=2 you need ~{need:,.0f} trades "
              f"(have {len(full)})")
