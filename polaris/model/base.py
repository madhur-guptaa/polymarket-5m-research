"""
polaris/model/base.py

The contract every probability model implements, plus the shared data types.

Everything downstream (calibration Step 2, the cache, the API) depends on this
protocol — never on a concrete model. That is what lets a Bayesian KDE and a
gradient-boosted model (and later a NN, Markov, ...) compete behind one
interface, judged on the same block-CV Brier.

Split of responsibilities (decision D5, refined):
  - fit / predict_proba / predict_grid : inference, model-specific
  - optimize                            : Step-1 hyperparameter search, model-specific
  - loocv_probabilities                 : honest out-of-sample p_up, model-specific
  - Step 2 (risk on p_up + market prices): model-AGNOSTIC (see model/risk.py)
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Bandwidths:
    """Kernel bandwidths, one per feature dimension."""
    delta: float
    time: float
    momentum: float
    imbalance: float
    volatility: float
    tod: float

    def as_dict(self) -> dict[str, float]:
        return {"delta": self.delta, "time": self.time, "momentum": self.momentum,
                "imbalance": self.imbalance, "volatility": self.volatility, "tod": self.tod}


@dataclass(frozen=True)
class TWAPBandwidths:
    """Kernel bandwidths for the TWAP-geometry KDE (model/bayesian/twap.py).

    Four dimensions, not seven: this set REPLACES the Binance-microstructure
    features rather than adding to them, which is what a product kernel needs --
    every extra dimension multiplies another kernel into the weight and n_eff
    decays roughly geometrically with dimensionality.
    """
    twap_sofar: float   # bps
    req: float          # bps
    locked: float       # fraction of the window, 0..1
    spot: float         # bps

    def as_dict(self) -> dict[str, float]:
        return {"twap_sofar": self.twap_sofar, "req": self.req,
                "locked": self.locked, "spot": self.spot}


@dataclass(frozen=True)
class Prediction:
    """A single P(up) estimate with its Beta-posterior credible interval."""
    p_up: float
    lo: float      # 2.5th percentile
    hi: float      # 97.5th percentile
    n_eff: float   # effective sample size (sum of kernel weights)


@dataclass(frozen=True)
class Query:
    """One live query point. Optional features are None when unavailable."""
    delta: float
    time: float
    momentum: float | None = None
    imbalance: float | None = None
    log_vol: float | None = None
    tod_sin: float | None = None
    tod_cos: float | None = None
    # TWAP geometry. req/locked are NOT carried: they are exact functions of
    # these two and `time` (features.twap_geometry), so the model derives them
    # at predict time -- which keeps a caller that only moves `time` (the live
    # tick, the heatmap grid) from holding a stale req.
    twap_sofar: float | None = None
    spot_vs_anchor: float | None = None


@dataclass
class FeatureSet:
    """Fully-processed training arrays (all 1-D, length n). NaN = missing.

    window_ids groups observations by their parent 5-minute window — the basis
    for leave-one-window-out (block) cross-validation. window_start_ts holds the
    Unix-second window start per observation, for joining to Polymarket prices.
    """
    deltas: np.ndarray
    times: np.ndarray
    momenta: np.ndarray
    imbalances: np.ndarray
    log_vols: np.ndarray
    tod_sin: np.ndarray
    tod_cos: np.ndarray
    outcomes: np.ndarray          # 1.0 up, 0.0 down
    window_ids: np.ndarray        # int, 0..n_windows-1
    window_start_ts: np.ndarray   # int64 Unix seconds
    # TWAP geometry: two stored columns and the two derived from them (see
    # features.twap_geometry). NaN where the anchor was unavailable.
    twap_sofar: np.ndarray
    spot_vs_anchor: np.ndarray
    req: np.ndarray
    locked: np.ndarray

    @property
    def n(self) -> int:
        return len(self.deltas)

    def subset(self, mask: np.ndarray) -> "FeatureSet":
        """Rows selected by `mask`, every field included.

        Generic over the dataclass's fields on purpose: calibrate.py and
        backtest.py each used to keep their own hand-written list of field
        names, which meant adding a feature silently dropped it from every
        subset taken by code that was not also updated.
        """
        return FeatureSet(**{f.name: getattr(self, f.name)[mask] for f in fields(self)})

    @property
    def n_windows(self) -> int:
        return int(self.window_ids.max()) + 1 if len(self.window_ids) else 0


@runtime_checkable
class ProbabilityModel(Protocol):
    """Estimates P(up) for a window, with uncertainty."""

    def fit(self, fs: FeatureSet) -> "ProbabilityModel": ...

    def predict_proba(self, q: Query) -> Prediction: ...

    def predict_grid(
        self, delta_range: np.ndarray, time_range: np.ndarray, held: Query,
    ) -> np.ndarray: ...

    def optimize(self, fs: FeatureSet, n_trials: int) -> dict: ...

    def loocv_probabilities(self, fs: FeatureSet) -> np.ndarray: ...
