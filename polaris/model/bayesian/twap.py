"""
polaris/model/bayesian/twap.py

Product-kernel Beta-Binomial estimator over the contract's own geometry — a
ProbabilityModel, and the production default (model.kind = "kde_twap").

WHY A SECOND KDE RATHER THAN MORE FEATURES ON THE FIRST
    Polymarket settles on TWAP(window) >= anchor (Chainlink TWAP-60s). A TWAP is
    partially observed as the window runs, so the natural state variables are
    how far the running average sits from the anchor and how much the remaining
    seconds must move to change the answer -- not the Binance-microstructure set
    the original KDE (estimator.py) was built on. Measured on real settlements,
    post-2026-08-07 (RESEARCH_LOG session 27):
        original KDE, 7 dims          Brier .18111 (BTC) / .18678 (ETH)
        these 4 dims                  Brier .15543       / .15268
        both together, 11 dims        Brier .17210       / .17608
    The last line is the reason this is a replacement, not an extension: a
    product kernel multiplies one kernel per dimension, so n_eff decays roughly
    geometrically with dimensionality and the extra seven columns cost more than
    they carry. estimator.KDEModel stays available as kind "kde" -- as the
    baseline, not as a candidate.

    WHAT ACTUALLY CARRIES IT (measured 2026-09-29, session 28). Tuned with wide
    bounds, the search flattens twap_sofar and req completely, and spot +
    locked alone match all four dimensions to the fifth decimal (BTC .13869 vs
    .13870). So the gain over the original KDE is mostly one idea: measure
    price against the SETTLEMENT anchor rather than the window's opening VWAP,
    and stop spending dimensions on microstructure. twap_sofar and req are kept
    because they are the right state variables and a GBM does use them; the
    configured bandwidths simply leave them near-flat.

DIMENSIONS (all Gaussian kernels; every one is unbounded-continuous)
    twap_sofar  running TWAP since window open vs the anchor, bps
    req         what the remaining seconds must average to flip the outcome,
                bps of the current price -- the binary's distance to strike
    locked      fraction of the final average already determined, 0..1
    spot        this second's price vs the anchor, bps
    req and locked are derived (features.twap_geometry) from the other two plus
    elapsed time, both for training rows and at query time -- one formula.

NaN = missing, exactly as in estimator.py: an observation missing a dimension
contributes neutral weight (1.0) for it, and a query missing one drops it.

SCALE
    Every path is batched NumPy with the batch sized to a fixed element budget
    (_BUDGET) rather than a fixed row count, because these run against training
    sets anywhere from ~10^4 rows (a thinned calibration search) to ~10^6 (the
    live 14-day fit). There is no MLX path yet; loocv_probabilities is O(N^2),
    so calibrate a cadence-thinned set (calibrate --train-cadence).
"""

from __future__ import annotations

import logging

import numpy as np
import optuna
from scipy.stats import beta as beta_dist

from polaris.model import features
from polaris.model.base import FeatureSet, Prediction, Query, TWAPBandwidths

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Elements per (batch x N) weight matrix: ~20M float64 = 160MB. Sizes every
# batched path so memory stays flat as N grows.
_BUDGET = 20_000_000

# Column order shared by every path.
_DIMS = ("twap_sofar", "req", "locked", "spot")


def _obs_matrix(fs: FeatureSet) -> np.ndarray:
    """(N, 4) training matrix in _DIMS order."""
    return np.column_stack([fs.twap_sofar, fs.req, fs.locked, fs.spot_vs_anchor])


def _bw_vector(bw: TWAPBandwidths) -> np.ndarray:
    return np.array([bw.twap_sofar, bw.req, bw.locked, bw.spot], dtype=np.float64)


def _log_weights(Q: np.ndarray, O: np.ndarray, h: np.ndarray) -> np.ndarray:
    """(b, N) log product-kernel weights. NaN on either side -> that dim is neutral.

    Accumulated one dimension at a time so only one (b, N) scratch matrix is live
    beyond the result.
    """
    out = np.zeros((Q.shape[0], O.shape[0]), dtype=np.float64)
    for d in range(Q.shape[1]):
        q, o = Q[:, d], O[:, d]
        u = (q[:, None] - o[None, :]) / h[d]
        term = -0.5 * u * u
        # NaN from either side means "missing": kernel value 1.0, log 0.
        out += np.where(np.isnan(term), 0.0, term)
    return out


def _batch_rows(n_obs: int) -> int:
    return max(1, _BUDGET // max(n_obs, 1))


class TWAPKDEModel:
    """TWAP-geometry Beta-Binomial KDE (implements ProbabilityModel)."""

    def __init__(self, bw: TWAPBandwidths, prior_alpha: float = 2.0,
                 prior_beta: float = 2.0, window_sec: int = 300,
                 grid_max_train: int = 20_000, seed: int = 0) -> None:
        """grid_max_train caps the training rows predict_grid weighs (see there)."""
        self.bw = bw
        self.a0 = prior_alpha
        self.b0 = prior_beta
        self.window_sec = window_sec
        self.grid_max_train = grid_max_train
        self.seed = seed
        self._fs: FeatureSet | None = None
        self._O: np.ndarray | None = None

    # ── fit ──────────────────────────────────────────────────────────────
    def fit(self, fs: FeatureSet) -> "TWAPKDEModel":
        self._fs = fs
        self._O = _obs_matrix(fs)
        return self

    @property
    def is_ready(self) -> bool:
        return self._fs is not None and self._fs.n > 0

    # ── query construction ──────────────────────────────────────────────
    def _query_row(self, twap_sofar: float | None, spot: float | None,
                   elapsed: float) -> np.ndarray:
        ts = np.nan if twap_sofar is None else float(twap_sofar)
        sp = np.nan if spot is None else float(spot)
        req, locked = features.twap_geometry(np.array([ts]), np.array([sp]),
                                             np.array([elapsed]), self.window_sec)
        return np.array([ts, req[0], locked[0], sp])

    def _posterior(self, w: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_eff = w.sum(axis=-1)
        ups = w @ y
        return self.a0 + ups, self.b0 + (n_eff - ups), n_eff

    # ── single-point inference (live hot path) ──────────────────────────
    def predict_proba(self, q: Query) -> Prediction:
        if not self.is_ready:
            return Prediction(0.5, 0.0, 1.0, 0.0)
        row = self._query_row(q.twap_sofar, q.spot_vs_anchor, q.time)
        w = np.exp(_log_weights(row[None, :], self._O, _bw_vector(self.bw))[0])
        alpha, beta, n_eff = self._posterior(w, self._fs.outcomes)
        alpha, beta = float(alpha), float(beta)
        return Prediction(
            p_up=alpha / (alpha + beta),
            lo=float(beta_dist.ppf(0.025, alpha, beta)),
            hi=float(beta_dist.ppf(0.975, alpha, beta)),
            n_eff=float(n_eff),
        )

    # ── grid inference (the dashboard surface) ──────────────────────────
    def predict_grid(self, delta_range: np.ndarray, time_range: np.ndarray,
                     held: Query) -> np.ndarray:
        """P(up) over (price vs anchor) x (elapsed time), holding the running TWAP.

        The first axis is read as spot_vs_anchor rather than delta_bps: both are
        "where is price vs the window's reference", and the anchor is the one
        the contract actually settles against. The running TWAP is held at the
        live value while (spot, time) are swept, so the surface answers "if
        price were X at second t, given the average so far" -- exact at the
        current second, an approximation away from it (moving t in reality
        would also move the running average). req and locked are recomputed per
        cell, which is what lets the surface bend with the payoff.

        Unlike estimator.KDEModel.predict_grid this cannot use the matrix-mult
        factorisation -- req couples spot and time, so the kernel does not
        separate over the two axes. It weighs a fixed random subsample of at
        most grid_max_train training rows instead (seeded, so the surface is
        stable between refreshes). The point prediction (predict_proba) always
        uses the full training set.
        """
        nd, nt = len(delta_range), len(time_range)
        if not self.is_ready:
            return np.full((nd, nt), 0.5)
        rng = np.random.default_rng(self.seed)
        n = self._fs.n
        idx = (rng.choice(n, self.grid_max_train, replace=False)
               if n > self.grid_max_train else np.arange(n))
        O, y = self._O[idx], self._fs.outcomes[idx]

        sp, tt = np.meshgrid(delta_range.astype(np.float64),
                             time_range.astype(np.float64), indexing="ij")
        ts_val = np.nan if held.twap_sofar is None else float(held.twap_sofar)
        ts = np.full(sp.shape, ts_val)
        req, locked = features.twap_geometry(ts, sp, tt, self.window_sec)
        Q = np.column_stack([ts.ravel(), req.ravel(), locked.ravel(), sp.ravel()])

        h = _bw_vector(self.bw)
        p = np.empty(len(Q))
        step = _batch_rows(len(O))
        for s in range(0, len(Q), step):
            w = np.exp(_log_weights(Q[s:s + step], O, h))
            alpha, beta, _ = self._posterior(w, y)
            p[s:s + step] = alpha / (alpha + beta)
        return p.reshape(nd, nt)

    # ── leave-one-window-out cross-validation (block CV) ────────────────
    def loocv_probabilities(self, fs: FeatureSet, bw: TWAPBandwidths | None = None,
                            batch: int | None = None) -> np.ndarray:
        """Out-of-sample P(up) per observation, holding out its whole window.

        Same block logic as estimator.KDEModel: the ~300 rows of one window
        share a label, so dropping only the row itself would leak the answer
        through its near-identical neighbours.
        """
        O = _obs_matrix(fs)
        h = _bw_vector(bw or self.bw)
        y = fs.outcomes
        step = batch or _batch_rows(fs.n)
        p = np.empty(fs.n)
        for s in range(0, fs.n, step):
            sl = slice(s, min(s + step, fs.n))
            w = np.exp(_log_weights(O[sl], O, h))
            w[fs.window_ids[sl][:, None] == fs.window_ids[None, :]] = 0.0
            alpha, beta, _ = self._posterior(w, y)
            p[sl] = alpha / (alpha + beta)
        return p

    # ── batched prediction (walk-forward backtest) ──────────────────────
    def predict_batch(self, q: FeatureSet, batch: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """P(up) and n_eff for query rows against the FITTED training set.

        No block mask: queries are disjoint in time from training.
        """
        Q = _obs_matrix(q)
        h = _bw_vector(self.bw)
        step = batch or _batch_rows(self._fs.n)
        p = np.empty(q.n)
        ne = np.empty(q.n)
        for s in range(0, q.n, step):
            sl = slice(s, min(s + step, q.n))
            w = np.exp(_log_weights(Q[sl], self._O, h))
            alpha, beta, n_eff = self._posterior(w, self._fs.outcomes)
            p[sl] = alpha / (alpha + beta)
            ne[sl] = n_eff
        return p, ne

    # ── bandwidth optimisation (Step 1) ──────────────────────────────────
    def optimize(self, fs: FeatureSet, n_trials: int = 100) -> dict:
        """TPE search over the four bandwidths minimising block-CV Brier.

        O(n_trials * N^2): pass a cadence-thinned FeatureSet.
        """
        def objective(trial: optuna.Trial) -> float:
            bw = TWAPBandwidths(
                # Upper bounds are deliberately far past any plausible value: a
                # first search with ceilings of 20/100/0.3/30 put twap_sofar and
                # req right against theirs on both symbols, i.e. the search
                # wanted those kernels close to flat. A ceiling must never be
                # the thing that decides that.
                twap_sofar=trial.suggest_float("twap_sofar", 0.1, 500.0, log=True),
                req=trial.suggest_float("req", 0.2, 5000.0, log=True),
                locked=trial.suggest_float("locked", 0.005, 2.0, log=True),
                spot=trial.suggest_float("spot", 0.1, 200.0, log=True),
            )
            p = self.loocv_probabilities(fs, bw=bw)
            return float(np.mean((p - fs.outcomes) ** 2))

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=7))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = dict(study.best_params)
        logger.info("TWAP KDE best block-CV Brier=%.4f bandwidths=%s", study.best_value, best)
        return best
