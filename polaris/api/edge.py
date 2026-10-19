"""
polaris/api/edge.py

Live edge = our P(up) vs the market's price. This is the trading signal (a
readout only — no execution anywhere).

  edge_yes = P(up)      - yes_price - fee(yes_price)     (buy YES / up)
  edge_no  = (1 - P(up)) - no_price - fee(no_price)      (buy NO  / down)

fee(p) = taker_fee_rate * p * (1 - p): Polymarket's crypto taker fee, ~1.75c per
share at p = 0.5. It was left out until 2026-10-07, which let the gate pass
trades whose whole edge was smaller than the fee.

The trade/no-trade gate requires THREE things: a minimum edge, enough effective
sample size (n_eff), AND being inside the tradeable time window. The time gate
comes from the session-10 diagnostic — the edge lives ~45-75s in; earlier is
noise, later the market is already informed.
"""

from __future__ import annotations

from polaris.core.config import TradingSettings


def compute_edge(prob: float | None, n_eff: float | None, poly: dict | None,
                 t: TradingSettings, elapsed: float | None = None) -> dict | None:
    """Edge on both sides + the gated trade decision, or None if inputs missing."""
    if prob is None or poly is None or poly.get("yes_price") is None:
        return None
    yes, no = poly["yes_price"], poly["no_price"]
    r = t.taker_fee_rate
    e_yes = prob - yes - r * yes * (1.0 - yes)
    e_no = (1.0 - prob) - no - r * no * (1.0 - no)
    yes_better = e_yes >= e_no
    edge = e_yes if yes_better else e_no
    in_window = elapsed is not None and t.trade_window_start <= elapsed <= t.trade_window_end
    gated = edge >= t.min_edge and (n_eff or 0.0) >= t.min_n_eff and in_window
    return {
        "edge_yes": round(e_yes, 4),
        "edge_no": round(e_no, 4),
        "side": "YES" if yes_better else "NO",
        "edge": round(edge, 4),
        "min_edge": t.min_edge,
        "min_n_eff": t.min_n_eff,
        "trade_window": [t.trade_window_start, t.trade_window_end],
        "in_window": bool(in_window),
        "trade": bool(gated),
    }
