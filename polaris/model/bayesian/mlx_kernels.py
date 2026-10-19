"""
polaris/model/bayesian/mlx_kernels.py

MLX (Apple Silicon GPU) counterparts of estimator.py's batched "_bn" kernels
(query batch b, training set N -> (b, N) weight matrix). Same math, same
masking semantics, ported for one reason: LOOCV bandwidth search is O(N^2) and
numpy tops out around ~88s per full pass at N~76K -- intractable for a real
14-day calibration (N~10^6) x hundreds of Optuna trials. Metal-backed batched
matmul-shaped ops make that tractable.

Two deliberate departures from the numpy version, both because Metal has no
float64 support (Apple GPUs are float32-only in compute shaders):

  1. All arrays are float32. Beta-Binomial posteriors and Gaussian/Beta/von
     Mises kernel weights don't need float64 precision -- this is a
     smoothing/averaging estimator, not an ill-conditioned linear solve.
  2. BETA_KERNEL_EPSILON (1e-10, tuned for float64) is USELESS in float32:
     1.0 - 1e-10 rounds to exactly 1.0 at float32 precision, which would then
     feed log(1 - obs) = log(0) = -inf into the beta kernel. MLX_BETA_EPSILON
     below is float32-appropriate instead.

MLX arrays are functional/immutable (no `w[mask] = x` in-place assignment,
unlike numpy) -- every masked write below is an mx.where(...) instead.

betaln(a, b) (the beta kernel's per-query normalising constant) is computed
in numpy/scipy on the CPU, not ported to MLX: it's shape (batch,), not
(batch, N) -- three orders of magnitude smaller than the matrices this module
exists to accelerate, and scipy already has a correct, fast lgamma. No point
reinventing it in MLX.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from scipy.special import betaln

# float32-safe clip epsilon for the beta kernel (see module docstring, point 2).
MLX_BETA_EPSILON: float = 1e-6


def to_mx(arr: np.ndarray) -> mx.array:
    """NaN-preserving float32 cast: numpy float64 -> mx.array float32."""
    return mx.array(np.asarray(arr, dtype=np.float32))


def gaussian_bn(q: mx.array, o: mx.array, bw: float,
                q_valid: mx.array | None, o_valid: mx.array | None) -> mx.array:
    """(batch,) query vs (N,) obs -> (batch, N) Gaussian kernel weights."""
    o_s = o if o_valid is None else mx.where(o_valid, o, mx.array(0.0, dtype=o.dtype))
    u = (q[:, None] - o_s[None, :]) / bw
    w = mx.exp(-0.5 * u * u)
    if o_valid is not None:
        w = mx.where(o_valid[None, :], w, mx.array(1.0, dtype=w.dtype))
    if q_valid is not None:
        w = mx.where(q_valid[:, None], w, mx.array(1.0, dtype=w.dtype))
    return w


def beta_bn(q: mx.array, o: mx.array, bw: float,
            q_valid: mx.array | None, o_valid: mx.array | None) -> mx.array:
    """(batch,) query vs (N,) obs, both in [0,1] -> (batch, N) Beta-kernel weights.

    a/b (and therefore betaln(a,b)) depend only on the query row, so they're
    computed in numpy on the (batch,)-sized arrays -- see module docstring."""
    eps = MLX_BETA_EPSILON
    q_np = np.asarray(q, dtype=np.float64)
    if q_valid is not None:
        q_np = np.where(np.asarray(q_valid), q_np, 0.5)
    q_np = np.clip(q_np, eps, 1.0 - eps)
    a_np = q_np / bw + 1.0
    b_np = (1.0 - q_np) / bw + 1.0
    betaln_np = betaln(a_np, b_np)

    a = mx.array(a_np.astype(np.float32))
    b = mx.array(b_np.astype(np.float32))
    beta_ln = mx.array(betaln_np.astype(np.float32))

    o_s = o if o_valid is None else mx.where(o_valid, o, mx.array(0.5, dtype=o.dtype))
    o_s = mx.clip(o_s, eps, 1.0 - eps)
    log_w = ((a[:, None] - 1.0) * mx.log(o_s)[None, :]
             + (b[:, None] - 1.0) * mx.log(1.0 - o_s)[None, :]
             - beta_ln[:, None])
    log_w = log_w - mx.max(log_w, axis=1, keepdims=True)
    w = mx.exp(log_w)
    if o_valid is not None:
        w = mx.where(o_valid[None, :], w, mx.array(1.0, dtype=w.dtype))
    if q_valid is not None:
        w = mx.where(q_valid[:, None], w, mx.array(1.0, dtype=w.dtype))
    return w


def von_mises_bn(q_sin: mx.array, q_cos: mx.array, o_sin: mx.array, o_cos: mx.array,
                  bw: float, q_valid: mx.array | None, o_valid: mx.array | None) -> mx.array:
    """(batch,) query sin/cos vs (N,) obs sin/cos -> (batch, N) chordal-distance weights."""
    os_ = mx.where(o_valid, o_sin, mx.array(0.0, dtype=o_sin.dtype)) if o_valid is not None else o_sin
    oc_ = mx.where(o_valid, o_cos, mx.array(0.0, dtype=o_cos.dtype)) if o_valid is not None else o_cos
    ds = q_sin[:, None] - os_[None, :]
    dc = q_cos[:, None] - oc_[None, :]
    w = mx.exp(-0.5 * (ds * ds + dc * dc) / (bw * bw))
    if o_valid is not None:
        w = mx.where(o_valid[None, :], w, mx.array(1.0, dtype=w.dtype))
    if q_valid is not None:
        w = mx.where(q_valid[:, None], w, mx.array(1.0, dtype=w.dtype))
    return w
