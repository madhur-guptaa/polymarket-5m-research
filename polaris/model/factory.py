"""
polaris/model/factory.py

One place that turns a model name into a ProbabilityModel. Everything that
runs a model (backtest, calibration, the serving cache) goes through here, so
adding a challenger means adding a branch here rather than touching every
call site.

KDE params come from config (the tuned bandwidths live in .env). GBM params
are passed explicitly — its hyperparameters are a search artifact, not
something worth carrying in the app's env surface.
"""

from __future__ import annotations

from polaris.core.config import Settings
from polaris.model.base import Bandwidths, TWAPBandwidths

MODEL_KINDS = ("kde_twap", "kde", "gbm")


def build_model(cfg: Settings, kind: str | None = None, *, backend: str = "numpy",
                params: dict | None = None):
    """Construct an unfitted model of `kind` (default: cfg.model.kind)."""
    m = cfg.model
    kind = kind or m.kind
    if kind == "kde_twap":
        if backend != "numpy":
            raise ValueError("kde_twap has no MLX backend yet; use backend='numpy'")
        from polaris.model.bayesian.twap import TWAPKDEModel
        bw = TWAPBandwidths(m.bw_twap_sofar, m.bw_req, m.bw_locked, m.bw_spot)
        return TWAPKDEModel(bw, m.prior_alpha, m.prior_beta,
                            window_sec=cfg.aggregate.window_sec)
    if kind == "kde":
        from polaris.model.bayesian.estimator import KDEModel
        bw = Bandwidths(m.bw_delta, m.bw_time, m.bw_momentum,
                        m.bw_imbalance, m.bw_volatility, m.bw_tod)
        return KDEModel(bw, m.prior_alpha, m.prior_beta, m.use_tod, backend=backend)
    if kind == "gbm":
        from polaris.model.gbm import GBMModel
        p = dict(params or {})
        rounds = p.pop("num_rounds", None)
        return GBMModel(params=p or None,
                        num_rounds=rounds if rounds else 300,
                        use_tod=m.use_tod)
    raise ValueError(f"unknown model kind {kind!r} (expected one of {MODEL_KINDS})")
