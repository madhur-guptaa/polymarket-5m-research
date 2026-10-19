"""
polaris/model/bayesian/kernels.py

The three product-kernel components. Ported verbatim from the old estimator —
the math is minimal and correct, so it stays as-is.

  gaussian    : delta, time, momentum, log-volatility (unbounded real)
  beta        : imbalance (bounded [0,1])
  von_mises   : time-of-day (circular, via sin/cos embedding)

All return UNNORMALISED weights; the Beta-Binomial posterior does the
normalising. The beta kernel is computed in log-space and max-shifted per call
for numerical stability.
"""

from __future__ import annotations

import numpy as np
from scipy.special import betaln

from polaris.model.features import BETA_KERNEL_EPSILON


def gaussian_kernel(obs: np.ndarray, query: float, bandwidth: float) -> np.ndarray:
    """exp(-u^2/2), u = (query - obs) / bandwidth. Weight 1.0 when obs == query."""
    u = (query - obs) / bandwidth
    return np.exp(-0.5 * u * u)


def beta_kernel(obs: np.ndarray, query: float, bandwidth: float) -> np.ndarray:
    """Beta-PDF kernel for [0,1] data. a = q/h + 1, b = (1-q)/h + 1."""
    eps = BETA_KERNEL_EPSILON
    q = max(eps, min(1.0 - eps, query))
    obs_safe = np.clip(obs, eps, 1.0 - eps)
    a = q / bandwidth + 1.0
    b = (1.0 - q) / bandwidth + 1.0
    log_w = (a - 1.0) * np.log(obs_safe) + (b - 1.0) * np.log(1.0 - obs_safe) - betaln(a, b)
    log_w -= np.max(log_w)  # stability; a per-query constant
    return np.exp(log_w)


def von_mises_kernel(
    obs_sin: np.ndarray, obs_cos: np.ndarray,
    query_sin: float, query_cos: float, bandwidth: float,
) -> np.ndarray:
    """exp(-d^2 / 2h^2), d^2 = (sin_q-sin_o)^2 + (cos_q-cos_o)^2 (chordal dist)."""
    d_sin = query_sin - obs_sin
    d_cos = query_cos - obs_cos
    sq = d_sin * d_sin + d_cos * d_cos
    return np.exp(-0.5 * sq / (bandwidth * bandwidth))
