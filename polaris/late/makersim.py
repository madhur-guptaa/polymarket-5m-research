"""
polaris/late/makersim.py

Queue-aware PAPER simulation of a selective maker on Polymarket's 5-minute markets.

QUESTION
    Historical fills show maker P&L is monotone in a settlement model's edge on
    the maker's side (RESEARCH_LOG session 31): +1.2..1.8c/share with the model
    1-2s stale. But those fills went to whoever was FIRST in the queue. Would
    OUR resting orders have been filled, and would they have been the good ones?

THE STRATEGY BEING SIMULATED
    Every decision second, for each token T of each market: rest a BUY on T at
    the current best bid b_T iff the model's fair value q_T exceeds b_T by at
    least `theta`; otherwise have no order. (Buying both tokens is fine: a YES
    and a NO together are worth exactly $1.) The decision uses the late_signals
    row for that second -- model and book as logged live -- and takes effect
    `latency` seconds after it was logged, to stand in for order round-trips.

FILL MODEL (deliberately conservative)
    A resting BUY of T at b is matched by either
      (a) a taker SELLING T at price <= b, or
      (b) a taker BUYING the complement T' at price >= 1 - b -- Polymarket
          matches a YES bid against a NO buy by minting the pair, and the event
          is reported on T' only. 86% of fills are aggressive BUYs, so (b) is
          the main route; ignoring it would make bids look almost unfillable.
    A trade strictly THROUGH our price fills us in full. A trade AT our price
    first consumes the queue that was ahead of us when the order was placed --
    the larger of T's bid size and T''s ask size at that level, both logged --
    and only then us. Cancellations ahead of us are NOT credited (we stay as far
    back as we started). Every reprice, cancel or completed fill puts the next
    order at the back of the then-current queue.

P&L per filled share: settlement payout of T - b + maker rebate, the rebate
being 20% of the taker fee 0.07 * b * (1 - b) the fill generated. Outcomes are
real settlements from poly_resolutions.

KNOWN OPTIMISM: our own order is assumed not to change anyone's behaviour, and
the decision grid is the 1-second logging cadence (a real quoter re-evaluates
continuously, which helps it). KNOWN PESSIMISM: queue-ahead never shrinks from
cancellations, which in a fast book it constantly does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from psycopg_pool import AsyncConnectionPool

FEE_RATE = 0.07
REBATE_SHARE = 0.20
EPS = 1e-6


@dataclass
class SimConfig:
    theta: float = 0.05          # required model edge over the bid
    latency: float = 1.0         # seconds from logged decision to live order
    order_size: float = 10.0     # shares per resting order
    min_price: float = 0.03      # never quote outside [min, 1-min]: no edge to measure there
    calibrator: object | None = None   # maps raw p_up -> calibrated p_up (fit elsewhere)
    p_col: str = "p"             # which fair-value column of the signal rows to quote on
    # Queue position on (re)placement. "back": behind everything resting at the
    # level, never advanced by cancellations -- the pessimistic bound. "front":
    # nothing ahead -- the optimistic bound (what an order already at the head
    # of the queue, e.g. one that has been resting longest, would see). Reality
    # sits between; a queue that cancels a lot is closer to "front".
    queue_mode: str = "back"


async def load(pool: AsyncConnectionPool, since_epoch: int | None = None) -> tuple:
    """late_signals rows, ws fills and settlements for the collected period."""
    cond = "" if since_epoch is None else f"WHERE window_start >= {int(since_epoch)}"
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(f"""SELECT symbol, window_start, t, wall_t, p_up,
                yes_bid, yes_bid_size, yes_ask, yes_ask_size,
                no_bid, no_bid_size, no_ask, no_ask_size
            FROM late_signals {cond} ORDER BY symbol, window_start, t""")
        sig = pd.DataFrame(await cur.fetchall(), columns=[
            "sym", "ws", "t", "wall_t", "p", "yb", "ybs", "ya", "yas", "nb", "nbs", "na", "nas"])
        await cur.execute(f"""SELECT symbol, window_start, outcome, side, price, size, ts_ms
            FROM poly_ws_trades {cond}""")
        fills = pd.DataFrame(await cur.fetchall(),
                             columns=["sym", "ws", "tok", "side", "x", "s", "ts_ms"])
        await cur.execute("SELECT symbol, window_start, outcome_up FROM poly_resolutions")
        res = {(s, int(w)): int(o) for s, w, o in await cur.fetchall()}
    num = ["wall_t", "p", "yb", "ybs", "ya", "yas", "nb", "nbs", "na", "nas"]
    sig[num] = sig[num].astype(float)
    fills[["x", "s"]] = fills[["x", "s"]].astype(float)
    fills["ts"] = fills["ts_ms"] / 1000.0
    return sig, fills, res


def _simulate_token(rows: pd.DataFrame, fl: pd.DataFrame, tok: str, ws: int,
                    payout: int, cfg: SimConfig) -> list[dict]:
    """One token of one window. Returns one dict per (partial) fill of our orders."""
    is_yes = tok == "YES"
    bid = rows["yb" if is_yes else "nb"].to_numpy()
    bid_sz = rows["ybs" if is_yes else "nbs"].to_numpy()
    comp_ask_sz = rows["nas" if is_yes else "yas"].to_numpy()
    p = rows[cfg.p_col].to_numpy()
    if cfg.calibrator is not None:
        p = cfg.calibrator(p)
    q = p if is_yes else 1.0 - p
    start = ws + rows["wall_t"].to_numpy() + cfg.latency
    end = np.append(start[1:], ws + 300.0 + cfg.latency)
    comp = "NO" if is_yes else "YES"
    # the only fills that can ever reach a BUY of tok: tok sells, comp buys
    rel = fl[((fl.tok == tok) & (fl.side == "SELL")) | ((fl.tok == comp) & (fl.side == "BUY"))]
    rel = rel.sort_values("ts")
    f_ts, f_tok, f_x, f_s = rel.ts.to_numpy(), rel.tok.to_numpy(), rel.x.to_numpy(), rel.s.to_numpy()

    out: list[dict] = []
    cur_price = None          # price of the live order, None = no order
    queue = 0.0               # shares ahead of us at cur_price
    left = 0.0                # unfilled shares of the live order
    j = 0
    for i in range(len(rows)):
        b = bid[i]
        want = (b is not None and np.isfinite(b) and np.isfinite(q[i])
                and cfg.min_price <= b <= 1 - cfg.min_price and q[i] - b >= cfg.theta)
        if not want:
            cur_price = None
        elif cur_price is None or abs(b - cur_price) > EPS or left <= 0:
            # (re)place at the back of the current queue
            cur_price, left = b, cfg.order_size
            ahead_bid = bid_sz[i] if np.isfinite(bid_sz[i]) else 0.0
            ahead_comp = comp_ask_sz[i] if np.isfinite(comp_ask_sz[i]) else 0.0
            queue = max(ahead_bid, ahead_comp) if cfg.queue_mode == "back" else 0.0
        while j < len(f_ts) and f_ts[j] < start[i]:
            j += 1
        k = j
        while k < len(f_ts) and f_ts[k] < end[i]:
            if cur_price is not None and left > 0:
                # translate the fill into "price of tok" terms
                px = f_x[k] if f_tok[k] == tok else 1.0 - f_x[k]
                kind = "through"
                if px < cur_price - EPS:           # traded THROUGH our level
                    got = left
                elif abs(px - cur_price) <= EPS:   # traded AT our level: queue first
                    kind = "at_level"
                    queue -= f_s[k]
                    got = min(left, max(0.0, -queue))
                    queue = min(queue, 0.0) + got if queue < 0 else queue
                else:
                    got = 0.0
                if got > 0:
                    left -= got
                    fee = FEE_RATE * cur_price * (1 - cur_price)
                    out.append({"ws": ws, "tok": tok, "t": int(rows["t"].iloc[i]),
                                "price": cur_price, "shares": got, "q": q[i], "kind": kind,
                                "pnl": got * (payout - cur_price + REBATE_SHARE * fee)})
            k += 1
        j = k
    return out


def simulate(sig: pd.DataFrame, fills: pd.DataFrame, res: dict, cfg: SimConfig) -> pd.DataFrame:
    """All windows with a settlement. One row per simulated fill."""
    out: list[dict] = []
    fills_by = {k: g for k, g in fills.groupby(["sym", "ws"])}
    for (sym, ws), rows in sig.groupby(["sym", "ws"]):
        up = res.get((sym, int(ws)))
        fl = fills_by.get((sym, ws))
        if up is None or fl is None:
            continue
        for tok in ("YES", "NO"):
            payout = up if tok == "YES" else 1 - up
            for r in _simulate_token(rows, fl, tok, int(ws), payout, cfg):
                r["sym"] = sym
                out.append(r)
    return pd.DataFrame(out)


def summarise(f: pd.DataFrame, n_windows: int) -> dict:
    """P&L per share and a window-clustered z (fills within a window share one outcome)."""
    if f.empty:
        return {"fills": 0, "shares": 0.0, "pnl": 0.0, "pnl_per_share": np.nan, "z": np.nan,
                "windows_traded": 0, "of_windows": n_windows}
    per_w = f.groupby(["sym", "ws"]).pnl.sum()
    z = per_w.mean() / (per_w.std(ddof=1) / np.sqrt(len(per_w))) if len(per_w) > 1 else np.nan
    return {"fills": len(f), "shares": float(f.shares.sum()), "pnl": float(f.pnl.sum()),
            "pnl_per_share": float(f.pnl.sum() / f.shares.sum()), "z": float(z),
            "windows_traded": int(len(per_w)), "of_windows": n_windows}


# ── a FAST fair value: Binance-led, Chainlink-anchored ────────────────────────
# Measured 2026-10-08: the Chainlink-only estimate does NOT lead the market --
# in the 2-10c disagreement band where a maker would quote, the market does not
# move toward it at all (corr with the next-second mid change 0.05), because
# Chainlink is an aggregate that reaches us ~1.5s late while the bots pricing
# the book watch exchanges directly. Quoting on that disagreement quotes where
# the MODEL is stale, and the simulation duly lost 3-7c/share. The historical
# maker edge was measured with a Binance-driven model. This builds that, live-
# honest: only Binance trades strictly before the decision instant are used.
async def load_binance(pool: AsyncConnectionPool, symbols: list[str],
                       lo: float, hi: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per symbol: (trade times in epoch seconds, prices), sorted."""
    out = {}
    async with pool.connection() as conn, conn.cursor() as cur:
        for sym in symbols:
            await cur.execute("""SELECT EXTRACT(EPOCH FROM trade_time)::float8, price::float8
                FROM crypto_trades WHERE symbol = %s
                  AND trade_time BETWEEN to_timestamp(%s) AND to_timestamp(%s)
                ORDER BY trade_time""", (sym, lo, hi))
            r = np.array(await cur.fetchall(), dtype=np.float64)
            if len(r):
                out[sym] = (r[:, 0], r[:, 1])
    return out


async def load_spot(pool: AsyncConnectionPool, lo: int, hi: int) -> dict[str, dict[int, float]]:
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute("SELECT symbol, ts, value FROM chainlink_spot WHERE ts BETWEEN %s AND %s",
                          (lo, hi))
        rows = await cur.fetchall()
    spot: dict[str, dict[int, float]] = {}
    for s, ts, v in rows:
        spot.setdefault(s, {})[int(ts)] = float(v)
    return spot


def binance_fair_value(sig: pd.DataFrame, bin_: dict, spot: dict, cl_lag: int = 2,
                       window: int = 300) -> np.ndarray:
    """P(Up) at each late_signals row's WALL instant, Binance-led.

    anchor  : Chainlink, mean spot over [ws-59, ws] -- the exact settlement anchor
    price   : last Binance trade strictly before the decision instant, moved to
              Chainlink's level by the trailing-120s mean basis (Binance - Chainlink)
    final   : the settling mean over [ws+240, ws+299] -- Chainlink seconds already
              observed (those older than cl_lag), Binance-converted seconds since
              then, and the remaining seconds at the current converted price
    sigma   : 1s log-return volatility of Binance over the trailing 120s
    """
    from math import erf, sqrt
    p = np.full(len(sig), np.nan)
    fs = window - 60
    for i, (sym, ws, wall_t) in enumerate(zip(sig.sym.to_numpy(), sig.ws.to_numpy(),
                                              sig.wall_t.to_numpy())):
        if sym not in bin_ or sym not in spot:
            continue
        ts, px = bin_[sym]
        cl = spot[sym]
        ws = int(ws)
        W = ws + float(wall_t)
        k_now = np.searchsorted(ts, W) - 1
        if k_now < 0:
            continue
        sec_now = int(W)
        secs = np.arange(sec_now - 120, sec_now)            # fully elapsed seconds
        idx = np.searchsorted(ts, secs + 1.0) - 1           # last trade <= end of each second
        if (idx < 0).any():
            continue
        b_sec = px[idx]
        pre = [cl[s] for s in range(ws - 59, ws + 1) if s in cl]
        if len(pre) < 50:
            continue
        anchor = sum(pre) / len(pre)
        both = [(b_sec[j], cl[s]) for j, s in enumerate(secs) if s in cl]
        if len(both) < 60:
            continue
        basis = float(np.mean([b - c for b, c in both]))
        P = px[k_now] - basis
        lr = np.diff(np.log(b_sec))
        sigma = float(np.std(lr, ddof=1)) * P
        t = W - ws
        if t >= fs:
            first = ws + fs
            cl_known_to = min(sec_now - cl_lag, ws + window - 1)
            known = [cl[s] for s in range(first, cl_known_to + 1) if s in cl]
            n_cl = len(range(first, cl_known_to + 1)) if cl_known_to >= first else 0
            if n_cl and len(known) < n_cl * 0.8:
                continue
            # pad any missing Chainlink seconds with the Binance-converted price
            bridge_lo = max(first, cl_known_to + 1)
            bridge = [px[max(np.searchsorted(ts, s + 1.0) - 1, 0)] - basis
                      for s in range(bridge_lo, min(sec_now, ws + window))]
            s_known = sum(known) + (n_cl - len(known)) * P + sum(bridge)
            n_known = n_cl + len(bridge)
            k = max(window - n_known, 0)
            mean_final = (s_known + k * P) / 60.0
            var = sigma ** 2 * k * (k + 1) * (2 * k + 1) / 6 / 3600.0
        else:
            mean_final = P
            var = sigma ** 2 * ((fs - t) + 19.5)
        if var <= 0:
            p[i] = float(mean_final >= anchor)
        else:
            p[i] = 0.5 * (1 + erf((mean_final - anchor) / sqrt(var) / sqrt(2)))
    return p


# ── event-driven simulation with a sub-second, book-driven cancel ────────────
# The 1-second simulation above lost almost entirely on TRADE-THROUGH fills:
# our quote left stale at a price the rest of the book had already abandoned.
# Whether a fast quoter escapes them is decided in milliseconds, so this
# version replays, in engine-timestamp order, every touch change on the token
# (poly_ws_touch) and every trade that can reach a bid on it (poly_ws_trades):
#   - our bid follows the best bid, but each move takes `reaction` seconds to
#     take effect; until then the OLD order is still live at the OLD price and
#     can be picked off -- that is the risk being measured;
#   - queue ahead = displayed size at our price when we arrive, shrunk whenever
#     the displayed size shrinks below it (our order is virtual, so the
#     displayed size is all others -- at most that many can be ahead of us);
#   - the quote is only wanted while fair value exceeds the bid by theta, with
#     fair value = the newest p_col value logged at or before that instant.
@dataclass
class FastConfig:
    theta: float = 0.05
    reaction: float = 0.25       # seconds from observing a touch move to our order moving
    order_size: float = 10.0
    min_price: float = 0.03
    p_col: str = "p_fast"
    # "display": join behind the displayed size, advanced as it shrinks (default,
    # realistic-to-pessimistic). "front": nothing ahead -- the optimistic bound.
    queue_mode: str = "display"


async def load_touch(pool: AsyncConnectionPool, symbols: list[str], lo_ms: int,
                     hi_ms: int) -> pd.DataFrame:
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute("""SELECT symbol, window_start, outcome, ts_ms, best_bid, bid_size
            FROM poly_ws_touch WHERE symbol = ANY(%s) AND ts_ms BETWEEN %s AND %s
            ORDER BY symbol, window_start, outcome, ts_ms, id""", (symbols, lo_ms, hi_ms))
        df = pd.DataFrame(await cur.fetchall(),
                          columns=["sym", "ws", "tok", "ts_ms", "bb", "bbs"])
    df[["bb", "bbs"]] = df[["bb", "bbs"]].astype(float)
    df["ts"] = df.ts_ms / 1000.0
    return df


def _fast_token(touch: pd.DataFrame, fl: pd.DataFrame, sig_rows: pd.DataFrame,
                tok: str, ws: int, payout: int, cfg: FastConfig) -> list[dict]:
    comp = "NO" if tok == "YES" else "YES"
    rel = fl[((fl.tok == tok) & (fl.side == "SELL")) | ((fl.tok == comp) & (fl.side == "BUY"))]
    # unified event stream: kind 0 = touch, 1 = trade (touch first on ties: the
    # book update that a trade causes is stamped with the same ms)
    ev_t = np.concatenate([touch.ts.to_numpy(), rel.ts.to_numpy()])
    ev_k = np.concatenate([np.zeros(len(touch), int), np.ones(len(rel), int)])
    a = np.concatenate([touch.bb.to_numpy(), np.where(rel.tok == tok, rel.x, 1.0 - rel.x)])
    b = np.concatenate([touch.bbs.to_numpy(), rel.s.to_numpy()])
    order = np.lexsort((ev_k, ev_t))
    ev_t, ev_k, a, b = ev_t[order], ev_k[order], a[order], b[order]
    # fair value timeline (wall instants)
    fv_t = ws + sig_rows.wall_t.to_numpy()
    fv = sig_rows[cfg.p_col].to_numpy()
    fv = fv if tok == "YES" else 1.0 - fv

    out: list[dict] = []
    live_px = None        # price of our currently LIVE order (None: no order)
    queue = left = 0.0
    pending = None        # (effective_time, new_price or None) -- a move in flight
    best_bid = best_size = None
    for t, k, x, s in zip(ev_t, ev_k, a, b):
        if pending is not None and t >= pending[0]:
            new_px = pending[1]
            pending = None
            if new_px is None:
                live_px = None
            elif live_px is None or abs(new_px - live_px) > EPS or left <= 0:
                live_px, left = new_px, cfg.order_size
                queue = best_size if (cfg.queue_mode == "display" and best_size is not None
                                      and best_bid is not None
                                      and abs(best_bid - new_px) <= EPS) else 0.0
        if k == 0:                                    # touch change
            best_bid, best_size = (x if np.isfinite(x) else None), (s if np.isfinite(s) else 0.0)
            if live_px is not None and best_bid is not None and abs(best_bid - live_px) <= EPS:
                queue = min(queue, best_size)         # people ahead of us left
            j = np.searchsorted(fv_t, t, side="right") - 1
            q = fv[j] if j >= 0 else np.nan
            want = (best_bid is not None and np.isfinite(q)
                    and cfg.min_price <= best_bid <= 1 - cfg.min_price
                    and q - best_bid >= cfg.theta)
            target = best_bid if want else None
            same = (target is None and live_px is None) or (
                target is not None and live_px is not None and abs(target - live_px) <= EPS)
            if not same:
                if pending is None or pending[1] != target:
                    pending = (t + cfg.reaction, target)
            else:
                pending = None
        else:                                         # trade that can reach our bid
            if live_px is None or left <= 0:
                continue
            if x < live_px - EPS:
                got, kind = left, "through"
            elif abs(x - live_px) <= EPS:
                queue -= s
                got, kind = min(left, max(0.0, -queue)), "at_level"
                if queue < 0:
                    queue = 0.0
            else:
                continue
            if got > 0:
                left -= got
                fee = FEE_RATE * live_px * (1 - live_px)
                out.append({"ws": ws, "tok": tok, "price": live_px, "shares": got,
                            "kind": kind, "t": t - ws,
                            "pnl": got * (payout - live_px + REBATE_SHARE * fee)})
    return out


def simulate_fast(sig: pd.DataFrame, touch: pd.DataFrame, fills: pd.DataFrame, res: dict,
                  cfg: FastConfig) -> pd.DataFrame:
    out: list[dict] = []
    t_by = {k: g for k, g in touch.groupby(["sym", "ws", "tok"])}
    f_by = {k: g for k, g in fills.groupby(["sym", "ws"])}
    for (sym, ws), rows in sig.groupby(["sym", "ws"]):
        up = res.get((sym, int(ws)))
        fl = f_by.get((sym, ws))
        if up is None or fl is None:
            continue
        rows = rows.sort_values("wall_t")
        for tok in ("YES", "NO"):
            tt = t_by.get((sym, ws, tok))
            if tt is None:
                continue
            for r in _fast_token(tt, fl, rows, tok, int(ws), up if tok == "YES" else 1 - up, cfg):
                r["sym"] = sym
                out.append(r)
    return pd.DataFrame(out)
