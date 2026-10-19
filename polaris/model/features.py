"""
polaris/model/features.py

THE one feature pipeline. Both live inference and calibration derive features
here, so they can never drift apart (the old code had this logic duplicated
between cache.py and dataset_step1.py, with a TODO admitting the hazard — D6).

Inputs are the columns already stored in window_observations (delta_bps,
time_elapsed_s, cum_imbalance, volatility_10s) plus the bucket timestamp. The
derived features are:

  momentum   : price move over the last L seconds, in bps
  log_vol    : log-transformed 10s volatility (heavy tail -> ~Gaussian)
  tod_sin/cos: time-of-day on the unit circle (von Mises kernel)
  imbalance  : cum_imbalance clamped to (eps, 1-eps) for the Beta kernel

MOMENTUM DEFINITION (fixed vs. the old code):
    delta_bps is the *cumulative* displacement from the window open. The move
    over the last L seconds is therefore delta[now] - delta[now-L] — NOT
    cumsum(delta), which the old array version computed and which double-counts
    an already-cumulative series. This is the VWAP-%-change-over-lookback form
    we standardized on (normalized by the window-open price).

NaN convention: NaN means "missing" and the kernels treat it as neutral. Only
momentum uses 0.0 for "no lookback available" (0 = no trend is meaningful).
"""

from __future__ import annotations

import numpy as np

LOG_EPSILON: float = 1e-8
BETA_KERNEL_EPSILON: float = 1e-10


def momentum_over_lookback(
    deltas: np.ndarray, times: np.ndarray, window_ids: np.ndarray,
    lookback_sec: int = 15,
) -> np.ndarray:
    """Move over the last `lookback_sec` seconds, per observation, in bps.

    momentum[i] = delta[i] - delta[j], where j is the observation ~lookback_sec
    earlier in the same window. 0.0 where no valid lookback point exists.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    window_ids = np.asarray(window_ids)
    momenta = np.zeros(len(deltas), dtype=np.float64)

    for wid in np.unique(window_ids):
        idx = np.flatnonzero(window_ids == wid)
        if idx.size == 0:
            continue
        order = np.argsort(times[idx])
        t = times[idx][order]
        d = deltas[idx][order]
        orig = idx[order]

        target = t - lookback_sec
        j = np.clip(np.searchsorted(t, target, side="left"), 0, len(t) - 1)
        # Valid only if the lookback point is real (>=0) and close to target.
        valid = (target >= 0) & (np.abs(t[j] - target) <= 2.0)
        cur = np.arange(len(t))[valid]
        momenta[orig[valid]] = d[cur] - d[j[valid]]
    return momenta


def log_volatility(vol_raw: np.ndarray) -> np.ndarray:
    """log(max(vol, 0) + eps). NaN passes through; +/-Inf -> NaN."""
    vols = np.asarray(vol_raw, dtype=np.float64)
    out = np.log(np.maximum(vols, 0.0) + LOG_EPSILON)
    return np.where(np.isfinite(out), out, np.nan)


def tod_sin_cos(epoch_seconds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Time-of-day (UTC) as (sin, cos) on the unit circle. Avoids midnight jump."""
    secs = np.asarray(epoch_seconds, dtype=np.float64)
    angle = 2.0 * np.pi * (secs % 86400.0) / 86400.0
    sin_a = np.where(np.isfinite(angle), np.sin(angle), np.nan)
    cos_a = np.where(np.isfinite(angle), np.cos(angle), np.nan)
    return sin_a, cos_a


def clamp_imbalance(c: np.ndarray) -> np.ndarray:
    """Clamp to (eps, 1-eps) so the Beta kernel's log(c), log(1-c) stay finite.
    NaN passes through np.clip unchanged."""
    return np.clip(np.asarray(c, dtype=np.float64), BETA_KERNEL_EPSILON, 1.0 - BETA_KERNEL_EPSILON)


def twap_geometry(
    twap_sofar_bps: np.ndarray, spot_vs_anchor_bps: np.ndarray,
    elapsed_s: np.ndarray, window_sec: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """(req_bps, locked_frac): where the contract stands, from the two stored columns.

    Polymarket pays Up iff TWAP(window) >= anchor. By `elapsed_s` seconds in, a
    fraction L of the final average is already fixed (the seconds 0..elapsed_s
    inclusive), so with the running TWAP R and the anchor A:

        L*R + (1-L)*M >= A   <=>   M >= (A - L*R) / (1-L)

    where M is the mean the REMAINING seconds must average. req_bps is that M
    relative to the current price P, in bps -- the binary's distance to strike.
    Large positive: the window must rally to resolve Up. Large negative: it is
    effectively won already. Working in anchor-relative fractions (A = 1,
    R = 1 + twap_sofar, P = 1 + spot_vs_anchor) makes it scale-free.

    NaN when the window is fully locked (no remaining seconds to move it) or
    either input is missing -- the kernels treat NaN as neutral.

    One formula, used by training (dataset.py), live serving and the heatmap grid.
    That is the point: req_bps is a derived feature, so storing it would create a
    second copy of this logic to drift out of sync with the first.
    """
    ts = np.asarray(twap_sofar_bps, dtype=np.float64) / 1e4
    sp = np.asarray(spot_vs_anchor_bps, dtype=np.float64) / 1e4
    t = np.asarray(elapsed_s, dtype=np.float64)
    locked = np.clip((t + 1.0) / window_sec, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        remaining_mean = (1.0 - locked * (1.0 + ts)) / (1.0 - locked)
        req = (remaining_mean / (1.0 + sp) - 1.0) * 1e4
    req = np.where((locked < 1.0) & np.isfinite(req), req, np.nan)
    return req, locked
