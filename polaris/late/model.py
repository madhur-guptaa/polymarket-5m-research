"""
polaris/late/model.py

P(window settles Up) from Chainlink spot alone, using the contract's own arithmetic.

THE RULE (measured, not assumed -- RESEARCH_LOG session 30):
    Up  <=>  TWAP60(t=299)  >=  TWAP60(t=0)
where TWAP60(t) is the mean of Chainlink spot over the trailing 60 seconds
(agreement with Polymarket settlement: 99.79% of 2,421 windows; TWAP60 vs mean
spot: within 0.1 bp). So
    anchor A = mean spot over [ws-59, ws]           -- fixed before the window opens
    final  F = mean spot over [ws+240, ws+299]      -- the last minute
From t = 240 on, F is part known, part future, and the future part shrinks to
nothing at t = 299.

ESTIMATE. Treat spot as a driftless random walk with per-second volatility
sigma (trailing 120s of 1s log returns). At data-second t:
    t >= 240: known = sum of observed spot in [240, t]; k = 299 - t seconds
              remain, each expected at the current price P:
                  E[F]   = (known + k*P) / 60
                  Var[F] = sigma^2 * k(k+1)(2k+1) / 6 / 60^2
    t <  240: all 60 averaged seconds lie g = 240 - t .. 299 - t seconds ahead:
                  E[F]   = P
                  Var[F] = sigma^2 * (g + 19.5)      (19.5 = sum_{j,l<60} min(j,l) / 60^2)
    p_up = Phi((E[F] - A) / sd)
It is a raw probability: the study logs it with every ingredient and calibrates
offline, because the earlier Binance-based version came out overconfident.
"""

from __future__ import annotations

import math

WINDOW = 300
FINAL_START = 240          # first second inside the settling average
FINAL_N = 60
_MIN_ANCHOR_POINTS = 50
_SIGMA_LOOKBACK = 120


def _ffill(series: dict[int, float], lo: int, hi: int) -> list[float] | None:
    """Values for every second in [lo, hi], forward-filled; None if the start is missing."""
    out, last = [], None
    for s in range(lo, hi + 1):
        v = series.get(s, last)
        if v is None:
            return None
        out.append(v)
        last = v
    return out


def estimate(spot: dict[int, float], window_start: int, t: int,
             window: int = WINDOW) -> dict | None:
    """Ingredients and p_up for data-second t of the window starting at window_start.

    `window` is the market's length in seconds (300, 900, 14400): every series
    settles on TWAP60 at its last second vs TWAP60 at its first, so only the
    position of the final minute changes.
    """
    ws = window_start
    final_start = window - FINAL_N
    # anchor: need most of the minute before the window opened
    pre = [spot[s] for s in range(ws - 59, ws + 1) if s in spot]
    if len(pre) < _MIN_ANCHOR_POINTS:
        return None
    anchor = sum(pre) / len(pre)

    now = ws + t
    hist = _ffill(spot, now - _SIGMA_LOOKBACK, now)
    if hist is None:
        return None
    rets = [math.log(b / a) for a, b in zip(hist, hist[1:]) if a > 0 and b > 0]
    if len(rets) < 60:
        return None
    m = sum(rets) / len(rets)
    sd_ret = math.sqrt(sum((r - m) ** 2 for r in rets) / (len(rets) - 1))
    price = hist[-1]
    sigma = sd_ret * price

    if t >= final_start:
        known = _ffill(spot, ws + final_start, now)
        if known is None:
            return None
        k = (window - 1) - t
        mean_final = (sum(known) + k * price) / FINAL_N
        var = sigma ** 2 * k * (k + 1) * (2 * k + 1) / 6 / FINAL_N ** 2
    else:
        g = final_start - t
        mean_final = price
        var = sigma ** 2 * (g + 19.5)

    sd = math.sqrt(var)
    if sd <= 0:
        p = 1.0 if mean_final >= anchor else 0.0
    else:
        p = 0.5 * (1.0 + math.erf((mean_final - anchor) / sd / math.sqrt(2)))
    return {"anchor": anchor, "mean_final": mean_final, "sd_final": sd,
            "spot": price, "sigma": sigma, "p_up": p}
