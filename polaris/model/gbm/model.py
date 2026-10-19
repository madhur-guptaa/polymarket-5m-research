"""
polaris/model/gbm/model.py

Gradient-boosted challenger to the KDE, behind the same ProbabilityModel
protocol. The point is not to ship a second production model — it is a control:
if a GBM on the identical features, scored the identical way, matches the KDE,
then the KDE's product-kernel independence assumption is costing us nothing; if
it clearly beats it, that assumption is leaving performance on the table.

Same features, same block-grouped evaluation, same walk-forward harness. The
only thing that changes is the estimator.

TWO HONEST DIFFERENCES vs the KDE, both load-bearing when reading a comparison:

  * n_eff has no analogue here. The KDE's n_eff is the summed kernel weight —
    a real "how many similar past observations support this?" measure, and the
    live trade gate uses it. A GBM has no such quantity, so predict_batch
    returns a sentinel large enough to pass any n_eff gate. A head-to-head is
    therefore only apples-to-apples on the edge and time gates; the KDE runs
    with one extra filter the GBM does not have.

  * No credible interval. The Beta posterior gives the KDE (lo, hi) for free.
    Prediction here reports lo = hi = p_up, i.e. "no interval available",
    rather than inventing one.

Missing features are handled natively: LightGBM learns a default split
direction for NaN, which is a genuinely different (and arguably better)
treatment than the KDE's neutral-weight-1.0 masking.
"""

from __future__ import annotations

import logging

import numpy as np

from polaris.model.base import FeatureSet, Prediction, Query

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional at import time
    _LGB_AVAILABLE = False

# Feature order is fixed and shared by every path (fit / predict / grid).
FEATURES = ("delta", "time", "momentum", "imbalance", "log_vol", "tod_sin", "tod_cos")

# Sentinel n_eff: large enough that any min_n_eff gate passes. See module docstring.
_NO_N_EFF = 1e9

# These defaults are FAR more regularized than typical LightGBM settings, and
# deliberately so. Within a 5-minute window the ~300 per-second rows are highly
# autocorrelated and all carry the SAME outcome label, so the effective sample
# size is the number of WINDOWS (~thousands), not the number of rows
# (~millions). Off-the-shelf settings memorize window-level patterns and
# generalize badly: measured on a held-out day, num_leaves=63/
# min_child_samples=200 scores Brier 0.2295 (barely better than the 0.25 base
# rate) and emits saturated 0.000/1.000 probabilities, while the settings below
# score 0.1746. min_child_samples=20000 is roughly 67 windows' worth of rows —
# i.e. a leaf must be supported by many independent windows, not many
# near-duplicate seconds of one.
DEFAULT_PARAMS: dict = {
    "objective": "binary",
    "learning_rate": 0.02,
    "num_leaves": 7,
    "min_child_samples": 20000,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 100.0,
    "verbosity": -1,
    "num_threads": 0,
}
DEFAULT_ROUNDS = 200


def _matrix(fs: FeatureSet) -> np.ndarray:
    """FeatureSet -> (n, 7) float matrix in FEATURES order. NaN preserved."""
    return np.column_stack([
        fs.deltas, fs.times, fs.momenta, fs.imbalances,
        fs.log_vols, fs.tod_sin, fs.tod_cos,
    ]).astype(np.float64)


def _query_row(q: Query, use_tod: bool = True) -> np.ndarray:
    nan = float("nan")
    return np.array([[
        q.delta, q.time,
        nan if q.momentum is None else q.momentum,
        nan if q.imbalance is None else q.imbalance,
        nan if q.log_vol is None else q.log_vol,
        nan if (q.tod_sin is None or not use_tod) else q.tod_sin,
        nan if (q.tod_cos is None or not use_tod) else q.tod_cos,
    ]], dtype=np.float64)


class GBMModel:
    """LightGBM binary classifier implementing ProbabilityModel."""

    def __init__(self, params: dict | None = None, num_rounds: int = DEFAULT_ROUNDS,
                 use_tod: bool = True) -> None:
        if not _LGB_AVAILABLE:
            raise RuntimeError("GBMModel requires lightgbm (uv add lightgbm)")
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.num_rounds = num_rounds
        self.use_tod = use_tod
        self._booster: "lgb.Booster | None" = None

    # ── fit / predict ────────────────────────────────────────────────────
    def fit(self, fs: FeatureSet) -> "GBMModel":
        X = _matrix(fs)
        if not self.use_tod:
            X[:, 5:7] = np.nan       # drop the ToD dimension, mirroring KDE's use_tod
        ds = lgb.Dataset(X, label=fs.outcomes, feature_name=list(FEATURES), free_raw_data=True)
        self._booster = lgb.train(self.params, ds, num_boost_round=self.num_rounds)
        return self

    @property
    def is_ready(self) -> bool:
        return self._booster is not None

    def _predict(self, X: np.ndarray) -> np.ndarray:
        if self._booster is None:
            return np.full(len(X), 0.5)
        if not self.use_tod:
            X = X.copy()
            X[:, 5:7] = np.nan
        return np.asarray(self._booster.predict(X), dtype=np.float64)

    def predict_proba(self, q: Query) -> Prediction:
        p = float(self._predict(_query_row(q, self.use_tod))[0])
        # No posterior => no interval; report the point estimate for lo/hi.
        return Prediction(p_up=p, lo=p, hi=p, n_eff=_NO_N_EFF)

    def predict_batch(self, q: FeatureSet, batch: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """P(up) and n_eff for a batch (batch arg accepted for protocol parity;
        LightGBM batches internally)."""
        p = self._predict(_matrix(q))
        return p, np.full(len(p), _NO_N_EFF)

    def predict_grid(self, delta_range: np.ndarray, time_range: np.ndarray,
                     held: Query) -> np.ndarray:
        """P(up) over the (delta x time) grid with the other features held fixed."""
        nd, nt = len(delta_range), len(time_range)
        base = _query_row(held, self.use_tod)[0]
        X = np.tile(base, (nd * nt, 1))
        dd, tt = np.meshgrid(delta_range, time_range, indexing="ij")
        X[:, 0] = dd.ravel()
        X[:, 1] = tt.ravel()
        return self._predict(X).reshape(nd, nt)

    # ── honest out-of-sample predictions (block CV) ──────────────────────
    def loocv_probabilities(self, fs: FeatureSet, n_splits: int = 5) -> np.ndarray:
        """Out-of-fold P(up) using K-fold CV GROUPED BY WINDOW.

        The KDE's version is true leave-one-window-out (for query i, every other
        second of i's own window is excluded, all other windows retained). A GBM
        cannot do that without one refit per window (~3k fits), so this uses
        grouped K-fold instead: windows are partitioned into K folds, each fold
        predicted by a model trained on the other K-1.

        Both share the property that matters — no observation is ever predicted
        by a model that saw any second of its own window — so the resulting
        Brier scores are comparable. K-fold is the slightly more conservative of
        the two, since each model trains on (K-1)/K of the data rather than
        nearly all of it.
        """
        X, y = _matrix(fs), fs.outcomes
        if not self.use_tod:
            X[:, 5:7] = np.nan
        wid = fs.window_ids
        uniq = np.unique(wid)
        # Deterministic window->fold assignment (no RNG, so runs are reproducible).
        fold_of_window = {w: i % n_splits for i, w in enumerate(uniq)}
        fold = np.array([fold_of_window[w] for w in wid])

        oof = np.empty(len(y), dtype=np.float64)
        for k in range(n_splits):
            tr, te = fold != k, fold == k
            if not te.any() or not tr.any():
                continue
            ds = lgb.Dataset(X[tr], label=y[tr], feature_name=list(FEATURES), free_raw_data=True)
            booster = lgb.train(self.params, ds, num_boost_round=self.num_rounds)
            oof[te] = booster.predict(X[te])
        return oof

    # ── hyperparameter search (Step 1 analogue) ──────────────────────────
    def optimize(self, fs: FeatureSet, n_trials: int = 40) -> dict:
        """Tune GBM hyperparameters to minimise grouped-CV Brier — the same
        objective the KDE's bandwidth search minimises, so the two searches are
        directly comparable."""
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: "optuna.Trial") -> float:
            params = {
                **DEFAULT_PARAMS,
                # Ranges centre on the heavily-regularized region (see
                # DEFAULT_PARAMS): a search capped at the usual
                # min_child_samples<=2000 never reaches the good basin here.
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 3, 63, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 1000, 100000, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 30.0, log=True),
            }
            rounds = trial.suggest_int("num_rounds", 100, 600, log=True)
            probe = GBMModel(params=params, num_rounds=rounds, use_tod=self.use_tod)
            p = probe.loocv_probabilities(fs)
            return float(np.mean((p - fs.outcomes) ** 2))

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=7))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        logger.info("GBM best grouped-CV Brier=%.4f params=%s",
                    study.best_value, study.best_params)
        return dict(study.best_params)
