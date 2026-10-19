"""
polaris/model/bayesian/estimator.py

The product-kernel Beta-Binomial estimator — a ProbabilityModel.

Given a query (delta, time, and optional momentum/imbalance/log-vol/tod), it
kernel-weights every training observation by similarity and updates a
Beta-Binomial posterior:

    P(up) = (alpha0 + sum_j w_j * outcome_j) / (alpha0 + beta0 + sum_j w_j)

The 95% credible interval comes from that Beta posterior, and n_eff = sum_j w_j
is the effective sample size (the confidence gate for sizing).

Three inference paths, all sharing the same product kernel:
  - predict_proba : one query vs all obs (live hot path, scalar kernels)
  - predict_grid  : a (delta x time) grid via the matrix-mult factorisation
  - loocv_probabilities : leave-one-WINDOW-out over the whole training set,
                          for honest Brier during calibration (block CV, D5)

optimize() tunes the six bandwidths to minimise block-CV Brier (Optuna TPE).

NaN = missing feature. An observation missing a feature contributes neutral
weight (1.0) for that dimension; a query missing a feature drops that dimension
for that query. use_tod=False drops the time-of-day dimension entirely.

Scale note: loocv_probabilities is batched NumPy — fine for the ~10^4-10^5
observations we validate on. A 14-day, ~10^6-row production calibration wants
the MLX/GPU backend (the old gpu_loocv.py); this mirrors its logic and is the
natural place to swap it in.
"""

from __future__ import annotations

import logging

import numpy as np
import optuna
from scipy.special import betaln
from scipy.stats import beta as beta_dist

from polaris.model.base import Bandwidths, FeatureSet, Prediction, Query
from polaris.model.bayesian.kernels import beta_kernel, gaussian_kernel, von_mises_kernel
from polaris.model.features import BETA_KERNEL_EPSILON

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    import mlx.core as mx
    from polaris.model.bayesian import mlx_kernels as _mlxk
    _MLX_AVAILABLE = True
except ImportError:  # pragma: no cover - non-Apple-Silicon hosts
    _MLX_AVAILABLE = False


class KDEModel:
    """Mixed-kernel Beta-Binomial estimator (implements ProbabilityModel)."""

    def __init__(
        self, bw: Bandwidths, prior_alpha: float = 2.0, prior_beta: float = 2.0,
        use_tod: bool = True, backend: str = "numpy",
    ) -> None:
        """backend: "numpy" (default, exact float64) or "mlx" (Apple GPU,
        float32 -- see mlx_kernels.py's docstring for why float32 is safe
        here). Only loocv_probabilities (the O(N^2) bandwidth-search hot
        path) dispatches on it; predict_proba/predict_grid/predict_batch stay
        numpy always -- they're not the bottleneck MLX exists to fix."""
        if backend == "mlx" and not _MLX_AVAILABLE:
            raise RuntimeError("backend='mlx' requested but mlx is not importable")
        self.bw = bw
        self.a0 = prior_alpha
        self.b0 = prior_beta
        self.use_tod = use_tod
        self.backend = backend
        self._fs: FeatureSet | None = None
        # validity masks (True where the feature is present)
        self._m_mom: np.ndarray | None = None
        self._m_imb: np.ndarray | None = None
        self._m_vol: np.ndarray | None = None
        self._m_tod: np.ndarray | None = None
        self._mlx_cache: dict | None = None  # lazily built on first MLX LOOCV call
        self._mlx_cache_fs_id: int | None = None  # id(fs) the cache was built from

    # ── fit ──────────────────────────────────────────────────────────────
    def fit(self, fs: FeatureSet) -> "KDEModel":
        self._fs = fs
        self._m_mom = ~np.isnan(fs.momenta)
        self._m_imb = ~np.isnan(fs.imbalances)
        self._m_vol = ~np.isnan(fs.log_vols)
        self._m_tod = ~np.isnan(fs.tod_sin)
        self._mlx_cache = None  # fs changed -- invalidate
        self._mlx_cache_fs_id = None
        return self

    @property
    def is_ready(self) -> bool:
        return self._fs is not None and self._fs.n > 0

    # ── single-point inference ───────────────────────────────────────────
    def predict_proba(self, q: Query) -> Prediction:
        if not self.is_ready:
            return Prediction(0.5, 0.0, 1.0, 0.0)
        fs, bw = self._fs, self.bw

        w = gaussian_kernel(fs.deltas, q.delta, bw.delta)
        w *= gaussian_kernel(fs.times, q.time, bw.time)
        if q.momentum is not None:
            m = self._m_mom
            w[m] *= gaussian_kernel(fs.momenta[m], q.momentum, bw.momentum)
        if q.imbalance is not None:
            m = self._m_imb
            w[m] *= beta_kernel(fs.imbalances[m], q.imbalance, bw.imbalance)
        if q.log_vol is not None:
            m = self._m_vol
            w[m] *= gaussian_kernel(fs.log_vols[m], q.log_vol, bw.volatility)
        if self.use_tod and q.tod_sin is not None and q.tod_cos is not None:
            m = self._m_tod
            w[m] *= von_mises_kernel(fs.tod_sin[m], fs.tod_cos[m], q.tod_sin, q.tod_cos, bw.tod)

        n_eff = float(w.sum())
        ups = float((w * fs.outcomes).sum())
        alpha, beta = self.a0 + ups, self.b0 + (n_eff - ups)
        return Prediction(
            p_up=alpha / (alpha + beta),
            lo=float(beta_dist.ppf(0.025, alpha, beta)),
            hi=float(beta_dist.ppf(0.975, alpha, beta)),
            n_eff=n_eff,
        )

    # ── grid inference (matrix-mult factorisation) ───────────────────────
    def predict_grid(
        self, delta_range: np.ndarray, time_range: np.ndarray, held: Query,
    ) -> np.ndarray:
        if not self.is_ready:
            return np.full((len(delta_range), len(time_range)), 0.5)
        fs, bw = self._fs, self.bw

        u_d = (delta_range[:, None] - fs.deltas[None, :]) / bw.delta
        w_delta = np.exp(-0.5 * u_d * u_d)                    # (nd, N)
        u_t = (time_range[:, None] - fs.times[None, :]) / bw.time
        w_time = np.exp(-0.5 * u_t * u_t)                     # (nt, N)

        shared = np.ones(fs.n, dtype=np.float64)              # (N,) grid-invariant
        if held.momentum is not None:
            m = self._m_mom
            shared[m] *= gaussian_kernel(fs.momenta[m], held.momentum, bw.momentum)
        if held.imbalance is not None:
            m = self._m_imb
            shared[m] *= beta_kernel(fs.imbalances[m], held.imbalance, bw.imbalance)
        if held.log_vol is not None:
            m = self._m_vol
            shared[m] *= gaussian_kernel(fs.log_vols[m], held.log_vol, bw.volatility)
        if self.use_tod and held.tod_sin is not None and held.tod_cos is not None:
            m = self._m_tod
            shared[m] *= von_mises_kernel(fs.tod_sin[m], fs.tod_cos[m], held.tod_sin, held.tod_cos, bw.tod)

        wd = w_delta * shared[None, :]
        total = wd @ w_time.T                                 # (nd, nt)
        up = wd @ (w_time * fs.outcomes[None, :]).T
        return (self.a0 + up) / (self.a0 + self.b0 + total)

    # ── leave-one-window-out cross-validation (block CV) ─────────────────
    def loocv_probabilities(
        self, fs: FeatureSet, bw: Bandwidths | None = None, batch: int | None = None,
    ) -> np.ndarray:
        """Out-of-sample P(up) for every observation, holding out its whole
        window (not just itself). This is the honest signal Step-1 optimises.
        Dispatches on self.backend; batch defaults to 512 (numpy) or 128
        (mlx -- see _loocv_mlx's docstring for the memory-budget reasoning)."""
        if self.backend == "mlx":
            return self._loocv_mlx(fs, bw, batch or 128)
        return self._loocv_numpy(fs, bw, batch or 512)

    def _loocv_numpy(
        self, fs: FeatureSet, bw: Bandwidths | None, batch: int,
    ) -> np.ndarray:
        bw = bw or self.bw
        n = fs.n
        m_mom, m_imb = ~np.isnan(fs.momenta), ~np.isnan(fs.imbalances)
        m_vol, m_tod = ~np.isnan(fs.log_vols), ~np.isnan(fs.tod_sin)
        p = np.empty(n, dtype=np.float64)

        for s in range(0, n, batch):
            e = min(s + batch, n)
            sl = slice(s, e)
            w = self._gauss_bn(fs.deltas[sl], fs.deltas, bw.delta, None, None)
            w *= self._gauss_bn(fs.times[sl], fs.times, bw.time, None, None)
            w *= self._gauss_bn(fs.momenta[sl], fs.momenta, bw.momentum, m_mom[sl], m_mom)
            w *= self._beta_bn(fs.imbalances[sl], fs.imbalances, bw.imbalance, m_imb[sl], m_imb)
            w *= self._gauss_bn(fs.log_vols[sl], fs.log_vols, bw.volatility, m_vol[sl], m_vol)
            if self.use_tod:
                w *= self._vm_bn(fs.tod_sin[sl], fs.tod_cos[sl], fs.tod_sin, fs.tod_cos,
                                 bw.tod, m_tod[sl], m_tod)
            # block mask: drop every obs in the query's own window
            same = fs.window_ids[sl][:, None] == fs.window_ids[None, :]
            w[same] = 0.0

            n_eff = w.sum(axis=1)
            ups = (w * fs.outcomes[None, :]).sum(axis=1)
            alpha, beta = self.a0 + ups, self.b0 + (n_eff - ups)
            p[sl] = alpha / (alpha + beta)
        return p

    def _mlx_arrays(self, fs: FeatureSet) -> dict:
        """Build (and cache on self._mlx_cache) the float32 mx.array copies of
        fs -- once per fit(), reused across every Optuna trial's LOOCV call so
        we're not re-transferring ~10^6-element arrays host->device on every
        one of ~100+ trials. Keyed by id(fs), not just "is there a cache" --
        a caller that switches fs without an intervening fit() (fit() is what
        clears self._mlx_cache) would otherwise silently reuse the wrong
        arrays: a shape mismatch crash if sizes differ, or silently wrong
        predictions against the wrong training set if they happen to match."""
        if self._mlx_cache is not None and self._mlx_cache_fs_id == id(fs):
            return self._mlx_cache
        c = {
            "deltas": _mlxk.to_mx(fs.deltas), "times": _mlxk.to_mx(fs.times),
            "momenta": _mlxk.to_mx(fs.momenta), "imbalances": _mlxk.to_mx(fs.imbalances),
            "log_vols": _mlxk.to_mx(fs.log_vols),
            "tod_sin": _mlxk.to_mx(fs.tod_sin), "tod_cos": _mlxk.to_mx(fs.tod_cos),
            "outcomes": _mlxk.to_mx(fs.outcomes),
            "window_ids": mx.array(fs.window_ids.astype(np.int32)),
            "m_mom": mx.array(~np.isnan(fs.momenta)), "m_imb": mx.array(~np.isnan(fs.imbalances)),
            "m_vol": mx.array(~np.isnan(fs.log_vols)), "m_tod": mx.array(~np.isnan(fs.tod_sin)),
        }
        self._mlx_cache = c
        self._mlx_cache_fs_id = id(fs)
        return c

    def _loocv_mlx(
        self, fs: FeatureSet, bw: Bandwidths | None, batch: int,
    ) -> np.ndarray:
        """MLX (Apple GPU) counterpart of _loocv_numpy -- identical math and
        masking, see mlx_kernels.py's docstring for the float32 departure.

        batch defaults to 128, not 512: a (batch, N) float32 matrix costs
        4*batch*N bytes and ~6 of them (one per kernel dim) are live per
        iteration. At N~10^6 that's 128*10^6*4 ~ 0.5GB/matrix -- comfortable
        on unified memory; 512 would be ~2GB/matrix, which starts to bite on
        a 16-24GB machine given ~6 matrices in flight per batch."""
        bw = bw or self.bw
        n = fs.n
        c = self._mlx_arrays(fs)
        p = np.empty(n, dtype=np.float64)

        for s in range(0, n, batch):
            e = min(s + batch, n)
            sl = slice(s, e)
            w = _mlxk.gaussian_bn(c["deltas"][sl], c["deltas"], bw.delta, None, None)
            w = w * _mlxk.gaussian_bn(c["times"][sl], c["times"], bw.time, None, None)
            w = w * _mlxk.gaussian_bn(c["momenta"][sl], c["momenta"], bw.momentum,
                                       c["m_mom"][sl], c["m_mom"])
            w = w * _mlxk.beta_bn(c["imbalances"][sl], c["imbalances"], bw.imbalance,
                                   c["m_imb"][sl], c["m_imb"])
            w = w * _mlxk.gaussian_bn(c["log_vols"][sl], c["log_vols"], bw.volatility,
                                       c["m_vol"][sl], c["m_vol"])
            if self.use_tod:
                w = w * _mlxk.von_mises_bn(c["tod_sin"][sl], c["tod_cos"][sl],
                                           c["tod_sin"], c["tod_cos"], bw.tod,
                                           c["m_tod"][sl], c["m_tod"])
            # block mask: drop every obs in the query's own window
            same = c["window_ids"][sl][:, None] == c["window_ids"][None, :]
            w = mx.where(same, mx.array(0.0, dtype=w.dtype), w)

            n_eff = mx.sum(w, axis=1)
            ups = mx.sum(w * c["outcomes"][None, :], axis=1)
            mx.eval(n_eff, ups)  # materialize this batch before the next enqueues
            n_eff_np = np.asarray(n_eff, dtype=np.float64)
            ups_np = np.asarray(ups, dtype=np.float64)
            alpha, beta = self.a0 + ups_np, self.b0 + (n_eff_np - ups_np)
            p[sl] = alpha / (alpha + beta)
        return p

    # ── batched prediction (walk-forward backtest) ──────────────────────
    def predict_batch(self, q: FeatureSet, batch: int = 512) -> tuple[np.ndarray, np.ndarray]:
        """P(up) and n_eff for a batch of query points against the FITTED training
        set. Used by the walk-forward backtester (train = past, query = the test
        day) — no block mask, since queries are disjoint from training in time."""
        fs, bw = self._fs, self.bw
        tm_mom, tm_imb = self._m_mom, self._m_imb
        tm_vol, tm_tod = self._m_vol, self._m_tod
        qm_mom, qm_imb = ~np.isnan(q.momenta), ~np.isnan(q.imbalances)
        qm_vol, qm_tod = ~np.isnan(q.log_vols), ~np.isnan(q.tod_sin)
        m = q.n
        p = np.empty(m, dtype=np.float64)
        ne = np.empty(m, dtype=np.float64)
        for s in range(0, m, batch):
            e = min(s + batch, m)
            sl = slice(s, e)
            w = self._gauss_bn(q.deltas[sl], fs.deltas, bw.delta, None, None)
            w *= self._gauss_bn(q.times[sl], fs.times, bw.time, None, None)
            w *= self._gauss_bn(q.momenta[sl], fs.momenta, bw.momentum, qm_mom[sl], tm_mom)
            w *= self._beta_bn(q.imbalances[sl], fs.imbalances, bw.imbalance, qm_imb[sl], tm_imb)
            w *= self._gauss_bn(q.log_vols[sl], fs.log_vols, bw.volatility, qm_vol[sl], tm_vol)
            if self.use_tod:
                w *= self._vm_bn(q.tod_sin[sl], q.tod_cos[sl], fs.tod_sin, fs.tod_cos,
                                 bw.tod, qm_tod[sl], tm_tod)
            n_eff = w.sum(axis=1)
            ups = (w * fs.outcomes[None, :]).sum(axis=1)
            alpha, beta = self.a0 + ups, self.b0 + (n_eff - ups)
            p[sl] = alpha / (alpha + beta)
            ne[sl] = n_eff
        return p, ne

    # ── bandwidth optimisation (Step 1) ──────────────────────────────────
    def optimize(self, fs: FeatureSet, n_trials: int = 100) -> dict:
        """TPE search over bandwidths minimising block-CV Brier."""
        base = self.bw

        def objective(trial: optuna.Trial) -> float:
            bw = Bandwidths(
                delta=trial.suggest_float("delta", 0.5, 30.0, log=True),
                time=trial.suggest_float("time", 2.0, 90.0, log=True),
                momentum=trial.suggest_float("momentum", 0.3, 20.0, log=True),
                imbalance=trial.suggest_float("imbalance", 0.005, 0.4, log=True),
                volatility=trial.suggest_float("volatility", 0.05, 3.0, log=True),
                tod=trial.suggest_float("tod", 0.05, 3.0, log=True) if self.use_tod else base.tod,
            )
            p = self.loocv_probabilities(fs, bw=bw)
            return float(np.mean((p - fs.outcomes) ** 2))  # Brier

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=7))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = dict(study.best_params)
        if not self.use_tod:
            best["tod"] = base.tod
        logger.info("Step 1 best block-CV Brier=%.4f bandwidths=%s", study.best_value, best)
        return best

    # ── batched masked kernels for LOOCV (query b, obs N) ────────────────
    @staticmethod
    def _gauss_bn(q, o, bw, q_valid, o_valid):
        o_s = o if o_valid is None else np.where(o_valid, o, 0.0)
        u = (q[:, None] - o_s[None, :]) / bw
        w = np.exp(-0.5 * u * u)
        if o_valid is not None:
            w[:, ~o_valid] = 1.0
        if q_valid is not None:
            w[~q_valid, :] = 1.0
        return w

    @staticmethod
    def _beta_bn(q, o, bw, q_valid, o_valid):
        eps = BETA_KERNEL_EPSILON
        q_s = np.clip(np.where(q_valid, q, 0.5) if q_valid is not None else q, eps, 1 - eps)
        o_s = np.clip(np.where(o_valid, o, 0.5) if o_valid is not None else o, eps, 1 - eps)
        a = q_s / bw + 1.0
        b = (1.0 - q_s) / bw + 1.0
        log_w = ((a[:, None] - 1.0) * np.log(o_s)[None, :]
                 + (b[:, None] - 1.0) * np.log(1.0 - o_s)[None, :]
                 - betaln(a, b)[:, None])
        log_w -= log_w.max(axis=1, keepdims=True)
        w = np.exp(log_w)
        if o_valid is not None:
            w[:, ~o_valid] = 1.0
        if q_valid is not None:
            w[~q_valid, :] = 1.0
        return w

    @staticmethod
    def _vm_bn(q_sin, q_cos, o_sin, o_cos, bw, q_valid, o_valid):
        os_ = np.where(o_valid, o_sin, 0.0) if o_valid is not None else o_sin
        oc_ = np.where(o_valid, o_cos, 0.0) if o_valid is not None else o_cos
        ds = q_sin[:, None] - os_[None, :]
        dc = q_cos[:, None] - oc_[None, :]
        w = np.exp(-0.5 * (ds * ds + dc * dc) / (bw * bw))
        if o_valid is not None:
            w[:, ~o_valid] = 1.0
        if q_valid is not None:
            w[~q_valid, :] = 1.0
        return w
