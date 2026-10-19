"""
polaris/model/featurelab.py

Candidate-feature screening. Answers one question: does feature X carry
incremental signal about the window outcome, beyond the six we already use?

WHY SCREEN WITH THE GBM AND NOT THE KDE
    The production model is a product-kernel KDE, and a KDE cannot be used to
    screen features. Every added dimension multiplies another kernel into the
    weight product, so n_eff collapses exponentially with dimensionality -- a
    genuinely informative feature will still make the KDE worse. Screening
    through it would reject good features for a reason that has nothing to do
    with whether they carry signal.

    LightGBM does not have that problem: it splits on whichever feature helps
    and ignores the rest. So it is the right instrument for "is there signal
    here?", even though the KDE may be the thing we ultimately ship. Winners
    get ported into the KDE one at a time, where each must pay for its own
    dimension.

TWO-STAGE SCORING, and why it is in this order
    Stage 1 (this module): absolute block-CV Brier on ALL labelled data. A
    feature that cannot improve absolute Brier cannot improve the gap to the
    market, and this stage has ~17K BTC windows of statistical power because it
    needs no Polymarket join.
    Stage 2 (separate, on the poly-matched subset): Brier-vs-market gap. This
    is the metric that actually matters -- beating the market, not beating
    0.25 -- but it is available on far fewer windows, so it is a confirmation
    step for stage-1 winners, not a search step.

NO LOOKAHEAD
    Every rolling statistic below is a TRAILING frame (`RANGE BETWEEN
    INTERVAL 'n seconds' PRECEDING AND CURRENT ROW`). RANGE, not ROWS: a second
    with no trades produces no bar, so "10 rows back" is not "10 seconds back".

    The frames are deliberately NOT partitioned by window_start. A 60-second
    lookback taken 10 seconds into a window legitimately reaches back into the
    previous window -- that data was on the tape before the decision point, so
    using it is honest. (The existing volatility_10s IS window-partitioned,
    which is why it is NULL for the first 10 seconds of every window.)
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)

# The six features already in production (model/features.py). tod_sin/tod_cos
# count as one feature conceptually but two columns.
BASE: tuple[str, ...] = (
    "delta_bps", "time_elapsed_s", "mom_15s", "cum_imbalance",
    "log_vol", "tod_sin", "tod_cos",
)


@dataclass(frozen=True)
class Candidate:
    """One candidate feature: its column name and why we think it might work."""
    name: str
    rationale: str


CANDIDATES: tuple[Candidate, ...] = (
    Candidate("mom_5s", "faster momentum than the 15s we use; captures impulse"),
    Candidate("mom_60s", "slower momentum; captures the regime the window sits in"),
    Candidate("accel", "2*mom_15s - mom_30s: is the move speeding up or fading?"),
    Candidate("rv_10s", "stddev of 1s log RETURNS -- volatility_10s is stddev of price "
                        "LEVELS, which conflates trend with volatility"),
    Candidate("rv_60s", "same at a slower scale"),
    Candidate("rv_ratio", "rv_10s/rv_60s -- is volatility spiking right now?"),
    Candidate("park_10s", "Parkinson high-low range vol; ~5x more efficient than "
                          "close-to-close at the same sample size"),
    Candidate("tint_10s", "trades per second -- activity, not volume"),
    Candidate("tint_rel", "10s trade rate vs the trailing 300s rate: activity SURGE"),
    Candidate("avgsz_10s", "log mean trade size; informed flow tends to size up"),
    Candidate("imb_10s", "taker-sell share over the last 10s -- cum_imbalance is "
                         "cumulative from window open and goes stale late in a window"),
    Candidate("imb_30s", "same at 30s"),
    Candidate("imb_delta", "imb_10s - cum_imbalance: is recent flow AGAINST the "
                           "window's flow so far?"),
    Candidate("vwap_dev", "(close-vwap)/vwap this second: intra-second pressure"),
    Candidate("range_bps", "this second's high-low in bps; a liquidity/uncertainty proxy"),
)

# ── extraction ───────────────────────────────────────────────────────────────
# NOTE on mm_buy_volume: it sums quantity where is_buyer_maker is true. If the
# BUYER was the maker then the TAKER was a seller -- so mm_buy_volume is
# taker-SELL volume, and imbalance is the taker-sell share. High = selling
# pressure. Inherited from the aggregator's cum_imbalance; kept consistent here
# on purpose so the two are comparable.
_EXTRACT_SQL = """
COPY (
WITH bars AS (
    SELECT bucket_time,
           vwap::FLOAT8                         AS vwap,
           close_price::FLOAT8                  AS close_price,
           high_price::FLOAT8                   AS high_price,
           low_price::FLOAT8                    AS low_price,
           volume::FLOAT8                       AS volume,
           trade_count::FLOAT8                  AS tc,
           COALESCE(mm_buy_volume, 0)::FLOAT8   AS mm,
           COALESCE(total_taker_volume, volume, 0)::FLOAT8 AS tk
    FROM crypto_1s_bars
    WHERE symbol = %(sym)s AND bucket_time >= %(since)s
),
rets AS (
    SELECT b.*,
           ln(b.vwap / NULLIF(LAG(b.vwap) OVER (ORDER BY b.bucket_time), 0)) AS ret,
           CASE WHEN b.low_price > 0 THEN ln(b.high_price / b.low_price) END AS hl
    FROM bars b
),
roll AS (
    SELECT r.*,
           FIRST_VALUE(r.vwap) OVER w5    AS vwap_5s,
           FIRST_VALUE(r.vwap) OVER w15   AS vwap_15s,
           FIRST_VALUE(r.vwap) OVER w30   AS vwap_30s,
           FIRST_VALUE(r.vwap) OVER w60   AS vwap_60s,
           STDDEV_SAMP(r.ret)  OVER w10   AS rv10,
           STDDEV_SAMP(r.ret)  OVER w60   AS rv60,
           AVG(r.hl * r.hl)    OVER w10   AS hl2_10,
           SUM(r.tc)           OVER w10   AS tc10,
           SUM(r.tc)           OVER w300  AS tc300,
           SUM(r.volume)       OVER w10   AS vol10,
           SUM(r.mm)           OVER w10   AS mm10,
           SUM(r.tk)           OVER w10   AS tk10,
           SUM(r.mm)           OVER w30   AS mm30,
           SUM(r.tk)           OVER w30   AS tk30
    FROM rets r
    WINDOW
        w5   AS (ORDER BY r.bucket_time RANGE BETWEEN INTERVAL  '5 seconds' PRECEDING AND CURRENT ROW),
        w10  AS (ORDER BY r.bucket_time RANGE BETWEEN INTERVAL '10 seconds' PRECEDING AND CURRENT ROW),
        w15  AS (ORDER BY r.bucket_time RANGE BETWEEN INTERVAL '15 seconds' PRECEDING AND CURRENT ROW),
        w30  AS (ORDER BY r.bucket_time RANGE BETWEEN INTERVAL '30 seconds' PRECEDING AND CURRENT ROW),
        w60  AS (ORDER BY r.bucket_time RANGE BETWEEN INTERVAL '60 seconds' PRECEDING AND CURRENT ROW),
        w300 AS (ORDER BY r.bucket_time RANGE BETWEEN INTERVAL '300 seconds' PRECEDING AND CURRENT ROW)
)
SELECT
    EXTRACT(EPOCH FROM o.window_start)::BIGINT              AS window_ts,
    EXTRACT(EPOCH FROM o.bucket_time)::BIGINT               AS bucket_ts,
    o.time_elapsed_s,
    o.delta_bps,
    o.cum_imbalance,
    o.volatility_10s,
    o.outcome,
    -- multi-scale momentum, in bps, off the trailing vwap anchors
    CASE WHEN g.vwap_5s  > 0 THEN (g.vwap - g.vwap_5s)  / g.vwap_5s  * 10000.0 END AS mom_5s,
    CASE WHEN g.vwap_15s > 0 THEN (g.vwap - g.vwap_15s) / g.vwap_15s * 10000.0 END AS mom_15s_u,
    CASE WHEN g.vwap_30s > 0 THEN (g.vwap - g.vwap_30s) / g.vwap_30s * 10000.0 END AS mom_30s,
    CASE WHEN g.vwap_60s > 0 THEN (g.vwap - g.vwap_60s) / g.vwap_60s * 10000.0 END AS mom_60s,
    g.rv10 * 10000.0                                        AS rv_10s,
    g.rv60 * 10000.0                                        AS rv_60s,
    CASE WHEN g.hl2_10 >= 0
         THEN sqrt(g.hl2_10 / (4.0 * ln(2.0))) * 10000.0 END AS park_10s,
    g.tc10                                                  AS tint_10s,
    CASE WHEN g.tc300 > 0 THEN g.tc10 / (g.tc300 / 30.0) END AS tint_rel,
    CASE WHEN g.tc10 > 0 THEN ln(g.vol10 / g.tc10 + 1e-12) END AS avgsz_10s,
    CASE WHEN g.tk10 > 0 THEN g.mm10 / g.tk10 END           AS imb_10s,
    CASE WHEN g.tk30 > 0 THEN g.mm30 / g.tk30 END           AS imb_30s,
    CASE WHEN g.vwap > 0
         THEN (g.close_price - g.vwap) / g.vwap * 10000.0 END AS vwap_dev,
    CASE WHEN g.vwap > 0
         THEN (g.high_price - g.low_price) / g.vwap * 10000.0 END AS range_bps
FROM window_observations o
JOIN roll g ON g.bucket_time = o.bucket_time
WHERE o.symbol = %(sym)s
  AND o.outcome IS NOT NULL
  AND o.window_start >= %(since)s
  AND MOD(o.time_elapsed_s, %(stride)s) = 0
ORDER BY o.window_start, o.bucket_time
) TO STDOUT WITH (FORMAT CSV, HEADER)
"""


async def extract(
    pool: AsyncConnectionPool, symbol: str, days: int, stride: int = 1,
) -> pd.DataFrame:
    """Pull base + candidate features for `symbol` into a DataFrame.

    `stride` subsamples seconds within each window (stride=3 keeps t=0,3,6...).
    Adjacent seconds are ~99% autocorrelated, so striding costs almost no
    information while cutting memory and fit time proportionally. It does NOT
    drop windows -- block CV power is unaffected.
    """
    buf = io.BytesIO()
    sql = _EXTRACT_SQL.replace("%(since)s", "(now() - make_interval(days => %(days)s))")
    async with pool.connection() as conn, conn.cursor() as cur:
        async with cur.copy(sql, {"sym": symbol, "days": days, "stride": stride}) as copy:
            async for chunk in copy:
                buf.write(bytes(chunk))
    buf.seek(0)
    df = pd.read_csv(buf)

    # Derived columns that are cheaper to compute here than in SQL.
    df["log_vol"] = np.log(np.maximum(df["volatility_10s"].to_numpy(), 0.0) + 1e-8)
    df.loc[~np.isfinite(df["log_vol"]), "log_vol"] = np.nan
    ang = 2.0 * np.pi * (df["bucket_ts"].to_numpy() % 86400.0) / 86400.0
    df["tod_sin"], df["tod_cos"] = np.sin(ang), np.cos(ang)
    # Production momentum is window-anchored (delta_bps now minus delta_bps 15s
    # ago), which differs from the unpartitioned mom_15s_u above. Reproduce the
    # production definition so BASE is genuinely the current feature set.
    df["mom_15s"] = _window_anchored_momentum(df, lookback=15)
    df["accel"] = 2.0 * df["mom_15s_u"] - df["mom_30s"]
    df["rv_ratio"] = df["rv_10s"] / df["rv_60s"].replace(0.0, np.nan)
    df["imb_delta"] = df["imb_10s"] - df["cum_imbalance"]
    logger.info("[%s] extracted %d rows / %d windows",
                symbol, len(df), df["window_ts"].nunique())
    return df


def _window_anchored_momentum(df: pd.DataFrame, lookback: int) -> np.ndarray:
    """delta_bps[t] - delta_bps[t-lookback] within the same window (production form)."""
    key = df["window_ts"].to_numpy()
    te = df["time_elapsed_s"].to_numpy()
    d = df["delta_bps"].to_numpy(dtype=np.float64)
    lookup = {(int(k), int(t)): i for i, (k, t) in enumerate(zip(key, te))}
    out = np.zeros(len(df), dtype=np.float64)
    for i, (k, t) in enumerate(zip(key, te)):
        j = lookup.get((int(k), int(t) - lookback))
        if j is not None:
            out[i] = d[i] - d[j]
    return out


# ── screening ────────────────────────────────────────────────────────────────
# Same regularization as the production GBM. These settings are extreme on
# purpose (see gbm/model.py): within a window the ~300 per-second rows share one
# label, so effective N is windows, not rows.
_PARAMS: dict = {
    "objective": "binary", "learning_rate": 0.02, "num_leaves": 7,
    "min_child_samples": 20000, "feature_fraction": 0.9, "bagging_fraction": 0.8,
    "bagging_freq": 1, "lambda_l2": 100.0, "verbosity": -1, "num_threads": 0,
}
_ROUNDS = 200

# Screening capacity. The production GBM's settings (num_leaves=7,
# min_child_samples=20000) were tuned against a SINGLE held-out day; under
# proper walk-forward block CV they turn out to be too tight. Measured on 17.4K
# BTC windows, num_leaves=31/min_child_samples=5000 scores BASE at 0.16502 vs
# the tight config's 0.16609 -- i.e. the tight settings were costing ~0.001
# Brier outright -- and roughly DOUBLES the measured value of candidate
# features, because a 7-leaf tree simply has no capacity left to spend on a
# new column. Screening at the tight setting understates every candidate.
SCREEN_PARAMS: dict = {"num_leaves": 31, "min_child_samples": 5000, "lambda_l2": 50.0}


def _folds(window_ts: np.ndarray, k: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window walk-forward splits, grouped by window and ordered in time.

    Fold i trains on time-blocks [0, i) and tests on block i. This is the honest
    layout for a time series -- a random or shuffled K-fold would let the model
    see the future, and grouping by window (not row) is what keeps the ~300
    autocorrelated rows of one window from straddling the split.
    """
    uniq = np.unique(window_ts)
    edges = np.array_split(uniq, k)
    out = []
    for i in range(1, k):
        train_w = np.concatenate(edges[:i])
        test_w = edges[i]
        out.append((np.isin(window_ts, train_w), np.isin(window_ts, test_w)))
    return out


def cv_predict(df: pd.DataFrame, cols: list[str], k: int = 5,
               params: dict | None = None) -> np.ndarray:
    """Out-of-sample p_up for every row, from the walk-forward folds.

    Rows in the first time-block are never in any test fold and come back NaN --
    there is no earlier data to train on, so there is no honest prediction for
    them.
    """
    import lightgbm as lgb

    params = {**_PARAMS, **(params or {})}
    wts = df["window_ts"].to_numpy()
    y = df["outcome"].to_numpy(dtype=np.float64)
    X = df[cols].to_numpy(dtype=np.float32)
    preds = np.full(len(df), np.nan)
    for tr, te in _folds(wts, k):
        ds = lgb.Dataset(X[tr], label=y[tr], feature_name=list(cols), free_raw_data=True)
        preds[te] = lgb.train(params, ds, num_boost_round=_ROUNDS).predict(X[te])
    return preds


def cv_brier(df: pd.DataFrame, cols: list[str], k: int = 5,
             gate: tuple[int, int] | None = None,
             params: dict | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    """Walk-forward block-CV Brier for a feature set.

    Returns (overall_brier, per_window_brier, window_ids) -- the per-window
    array is what paired significance tests must run on, since rows within a
    window are not independent observations.
    """
    import lightgbm as lgb

    params = {**_PARAMS, **(params or {})}
    wts = df["window_ts"].to_numpy()
    y = df["outcome"].to_numpy(dtype=np.float64)
    X = df[cols].to_numpy(dtype=np.float32)
    preds = np.full(len(df), np.nan)

    for tr, te in _folds(wts, k):
        ds = lgb.Dataset(X[tr], label=y[tr], feature_name=list(cols), free_raw_data=True)
        booster = lgb.train(params, ds, num_boost_round=_ROUNDS)
        preds[te] = booster.predict(X[te])

    mask = ~np.isnan(preds)
    if gate is not None:
        te_s = df["time_elapsed_s"].to_numpy()
        mask &= (te_s >= gate[0]) & (te_s <= gate[1])
    sq = (preds[mask] - y[mask]) ** 2
    w = wts[mask]
    order = np.argsort(w, kind="stable")
    w_sorted, sq_sorted = w[order], sq[order]
    uniq, starts = np.unique(w_sorted, return_index=True)
    per_window = np.array([seg.mean() for seg in np.split(sq_sorted, starts[1:])])
    return float(sq.mean()), per_window, uniq


def paired_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Paired mean difference (a-b) and its t-statistic, over aligned windows."""
    d = a - b
    n = len(d)
    if n < 2:
        return 0.0, 0.0
    sd = d.std(ddof=1)
    return float(d.mean()), float(d.mean() / (sd / np.sqrt(n))) if sd > 0 else 0.0


def greedy_screen(df: pd.DataFrame, k: int = 5, min_gain: float = 1e-4,
                  gate: tuple[int, int] | None = None,
                  params: dict | None = None) -> dict:
    """Forward-select candidates that improve block-CV Brier over BASE.

    Greedy rather than all-subsets because the candidates are correlated with
    each other (three momentum scales, three volatility measures); adding them
    one at a time and re-scoring is what stops a whole correlated family from
    being credited for the same signal.

    A round stops when the best remaining candidate improves overall Brier by
    less than `min_gain`. The paired t-statistic is reported per candidate but
    NOT used as the selection rule -- with ~17K windows even a trivially small
    Brier gain reaches large t, so gain size is the honest criterion and t is
    there to show whether a gain is distinguishable from noise at all.
    """
    selected: list[str] = []
    cols = list(BASE)
    base_b, base_pw, base_w = cv_brier(df, cols, k, gate, params)
    logger.info("BASE (%d features): Brier %.6f", len(cols), base_b)
    history = [{"added": None, "brier": base_b, "gain": 0.0, "t": 0.0}]
    remaining = [c.name for c in CANDIDATES]

    while remaining:
        scored = []
        for name in remaining:
            b, pw, w = cv_brier(df, cols + [name], k, gate, params)
            gain = base_b - b
            _, t = paired_t(base_pw, pw)   # positive t => candidate is better
            scored.append((gain, t, name, b, pw))
            logger.info("  try %-12s Brier %.6f  gain %+.6f  t=%+.2f", name, b, gain, t)
        scored.sort(reverse=True, key=lambda r: r[0])
        gain, t, name, b, pw = scored[0]
        if gain < min_gain:
            logger.info("stop: best remaining (%s) gains only %+.6f < %.6f",
                        name, gain, min_gain)
            break
        selected.append(name)
        cols.append(name)
        base_b, base_pw = b, pw
        remaining.remove(name)
        history.append({"added": name, "brier": b, "gain": gain, "t": t})
        logger.info("ACCEPT %s -> Brier %.6f (t=%+.2f)", name, b, t)

    return {"selected": selected, "final_brier": base_b, "history": history,
            "final_cols": cols}


# ── stage 2: does the feature set widen the gap to the MARKET? ────────────────
# Absolute Brier (stage 1) is the search metric because it has the most windows
# behind it. But beating 0.25 is not the goal -- beating Polymarket is. This
# stage re-scores stage-1 winners against the market's own forecast, using the
# REAL order book (poly_book_snapshots), which is the only price source that
# survived the 2026-09-11 retraction. Forward-only, so the subset is small.
_BOOK_SQL = """
COPY (
    SELECT window_start AS window_ts, elapsed_seconds AS time_elapsed_s,
           mid, best_ask, best_bid
    FROM poly_book_snapshots
    WHERE symbol = %(sym)s AND side = 'YES' AND mid IS NOT NULL
) TO STDOUT WITH (FORMAT CSV, HEADER)
"""


async def load_book(pool: AsyncConnectionPool, symbol: str) -> pd.DataFrame:
    """YES-side book quotes: the market's own P(up) forecast, per window-second."""
    buf = io.BytesIO()
    async with pool.connection() as conn, conn.cursor() as cur:
        async with cur.copy(_BOOK_SQL, {"sym": symbol}) as copy:
            async for chunk in copy:
                buf.write(bytes(chunk))
    buf.seek(0)
    return pd.read_csv(buf)


def market_comparison(train_df: pd.DataFrame, eval_df: pd.DataFrame,
                      book: pd.DataFrame, cols: list[str],
                      params: dict | None = None,
                      gate: tuple[int, int] | None = None) -> dict:
    """Train on everything strictly before the book period; score against the market.

    The split is by time, not by fold: the book only exists from the day
    collection started, so every book row is already after all training data.
    That makes this a genuine out-of-time test rather than a cross-validated one.
    """
    import lightgbm as lgb

    params = {**_PARAMS, **(params or {})}
    cutoff = int(book["window_ts"].min())
    tr = train_df[train_df["window_ts"] < cutoff]
    ds = lgb.Dataset(tr[cols].to_numpy(dtype=np.float32),
                     label=tr["outcome"].to_numpy(dtype=np.float64),
                     feature_name=list(cols), free_raw_data=True)
    booster = lgb.train(params, ds, num_boost_round=_ROUNDS)

    ev = eval_df[eval_df["window_ts"] >= cutoff].copy()
    ev["p_model"] = booster.predict(ev[cols].to_numpy(dtype=np.float32))
    m = ev.merge(book, on=["window_ts", "time_elapsed_s"], how="inner")
    if gate is not None:
        m = m[(m["time_elapsed_s"] >= gate[0]) & (m["time_elapsed_s"] <= gate[1])]

    y = m["outcome"].to_numpy(dtype=np.float64)
    b_model = float(np.mean((m["p_model"].to_numpy() - y) ** 2))
    b_mkt = float(np.mean((m["mid"].to_numpy() - y) ** 2))
    # Per-window pairing: rows inside one window are not independent.
    g = m.assign(sm=(m["p_model"] - y) ** 2, sk=(m["mid"] - y) ** 2).groupby("window_ts")
    pw = g[["sm", "sk"]].mean()
    gap, t = paired_t(pw["sk"].to_numpy(), pw["sm"].to_numpy())  # market - model
    return {"n_rows": len(m), "n_windows": int(m["window_ts"].nunique()),
            "train_windows": int(tr["window_ts"].nunique()),
            "brier_model": b_model, "brier_market": b_mkt,
            "gap": gap, "t": t, "cutoff": cutoff}


# ── the contract's own geometry ──────────────────────────────────────────────
# Polymarket resolves Up iff TWAP(window) >= price at the window's start. A TWAP
# is PARTIALLY OBSERVED as the window runs: by t seconds in, t/300 of the final
# average is already fixed and cannot change. The production feature set ignores
# this entirely -- it predicts "end price vs start price", a near-martingale,
# while the traded contract is "running average vs anchor", which is mechanically
# more determined the later you look.
#
# The sharp feature is the one the payoff is actually a function of: how much the
# REMAINING seconds must average, relative to the current price, for the sign to
# flip. Writing S_t for the sum of per-second prices so far and A for the anchor,
#     TWAP >= A  <=>  (S_t + R) / W >= A  <=>  mean(remaining) >= (W*A - S_t)/(W - t)
# so the distance to the strike, in bps of the current price, is
#     req_bps = ((W*A - S_t)/(W - t) / price - 1) * 1e4
# Large positive = the window must rally hard to resolve Up; large negative = it
# is already won. This is the binary's moneyness, and it was missing.
_TWAP_SQL = """
COPY (
WITH b AS (
    SELECT bucket_time, vwap::FLOAT8 AS v,
           date_bin(make_interval(secs => %(w)s), bucket_time,
                    TIMESTAMPTZ '1970-01-01') AS ws
    FROM crypto_1s_bars
    WHERE symbol = %(sym)s
      AND bucket_time >= (now() - make_interval(days => %(days)s))
),
r AS (
    SELECT b.*,
           -- the Chainlink stream's own quantity: a trailing 60s TWAP. Taken at
           -- the first bar of the window this is the resolution anchor.
           AVG(v) OVER (ORDER BY bucket_time
                        RANGE BETWEEN INTERVAL '60 seconds' PRECEDING AND CURRENT ROW) AS t60,
           SUM(v) OVER (PARTITION BY ws ORDER BY bucket_time
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS s_t,
           COUNT(*) OVER (PARTITION BY ws ORDER BY bucket_time
                          ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS n_t,
           ROW_NUMBER() OVER (PARTITION BY ws ORDER BY bucket_time) AS rn
    FROM b
),
anch AS (
    SELECT ws, MAX(CASE WHEN rn = 1 THEN t60 END) AS anchor FROM r GROUP BY ws
)
SELECT EXTRACT(EPOCH FROM r.ws)::BIGINT                       AS window_ts,
       EXTRACT(EPOCH FROM (r.bucket_time - r.ws))::INT        AS time_elapsed_s,
       -- partial TWAP so far, vs the anchor, in bps
       CASE WHEN a.anchor > 0
            THEN (r.s_t / r.n_t - a.anchor) / a.anchor * 10000.0 END AS twap_sofar_bps,
       -- required average of the REMAINING seconds, in bps of current price
       CASE WHEN r.v > 0 AND a.anchor > 0
             AND EXTRACT(EPOCH FROM (r.bucket_time - r.ws)) < %(w)s - 1
            THEN ((%(w)s * a.anchor - r.s_t)
                  / (%(w)s - EXTRACT(EPOCH FROM (r.bucket_time - r.ws)))
                  / r.v - 1.0) * 10000.0 END                   AS req_bps,
       -- fraction of the final average already locked in
       (r.n_t::FLOAT8 / %(w)s)                                 AS locked_frac,
       CASE WHEN a.anchor > 0
            THEN (r.v - a.anchor) / a.anchor * 10000.0 END     AS spot_vs_anchor_bps
FROM r JOIN anch a USING (ws)
) TO STDOUT WITH (FORMAT CSV, HEADER)
"""

TWAP_FEATURES: tuple[str, ...] = (
    "twap_sofar_bps", "req_bps", "locked_frac", "spot_vs_anchor_bps",
)


async def load_twap_features(pool: AsyncConnectionPool, symbol: str, days: int,
                             window_sec: int = 300) -> pd.DataFrame:
    """Per-second features describing the contract's actual payoff geometry."""
    buf = io.BytesIO()
    async with pool.connection() as conn, conn.cursor() as cur:
        async with cur.copy(_TWAP_SQL,
                            {"sym": symbol, "days": days, "w": window_sec}) as copy:
            async for chunk in copy:
                buf.write(bytes(chunk))
    buf.seek(0)
    return pd.read_csv(buf)


# ── does the KDE class benefit too, or only the GBM? ─────────────────────────
# Everything above is measured on LightGBM, but production runs a product-kernel
# KDE. The two react to a feature set very differently, so a GBM result does not
# transfer: a KDE multiplies one kernel per dimension, so n_eff decays roughly
# geometrically with dimensionality, and the TWAP geometry set is interesting
# here precisely because it REPLACES six features with four rather than adding.
# Fewer, better-aimed dimensions is the one change that helps a product kernel
# structurally rather than in spite of itself.
#
# This mirrors the production estimator's mechanics (product kernel, weights
# summed into a Beta-Binomial posterior) without depending on it, so the feature
# sets can be swapped freely. Gaussian kernels throughout: every TWAP feature is
# unbounded-continuous, and using the same kernel family for both sets keeps the
# comparison about the FEATURES rather than about kernel choice.
def kde_brier(df: pd.DataFrame, cols: list[str], train_frac: float = 0.7,
              n_train: int = 40_000, n_test: int = 15_000,
              scott: float = 1.0, a0: float = 1.0, b0: float = 1.0,
              seed: int = 0) -> tuple[float, int]:
    """Block-split Brier for a product-Gaussian KDE over `cols`.

    Split is by WINDOW and in time order (earlier windows train, later ones
    test) -- the same reason block CV is used everywhere else here: rows inside
    a window share one label, so a row-level split would leak.

    Subsampled because this is O(n_train * n_test) per dimension. Both feature
    sets get the identical budget, so the comparison is fair even though the
    absolute Brier would improve with more points.
    """
    rng = np.random.default_rng(seed)
    wts = df["window_ts"].to_numpy()
    uniq = np.unique(wts)
    cut = uniq[int(len(uniq) * train_frac)]
    tr_mask, te_mask = wts < cut, wts >= cut

    X = df[cols].to_numpy(dtype=np.float64)
    y = df["outcome"].to_numpy(dtype=np.float64)
    ok = np.isfinite(X).all(axis=1)
    tr_idx = np.flatnonzero(tr_mask & ok)
    te_idx = np.flatnonzero(te_mask & ok)
    if len(tr_idx) == 0 or len(te_idx) == 0:
        return float("nan"), 0
    if len(tr_idx) > n_train:
        tr_idx = rng.choice(tr_idx, n_train, replace=False)
    if len(te_idx) > n_test:
        te_idx = rng.choice(te_idx, n_test, replace=False)

    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xte, yte = X[te_idx], y[te_idx]
    # Scott's rule per dimension: the standard default, applied identically to
    # both feature sets so neither is hand-tuned into an advantage.
    d = Xtr.shape[1]
    h = scott * Xtr.std(axis=0, ddof=1) * len(Xtr) ** (-1.0 / (d + 4))
    h[h <= 0] = 1.0

    preds = np.empty(len(Xte))
    Z = Xtr / h
    for lo in range(0, len(Xte), 256):
        q = Xte[lo:lo + 256] / h
        # squared distance in scaled space -> product of Gaussian kernels
        d2 = ((q[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2)
        w = np.exp(-0.5 * d2)
        n_eff = w.sum(axis=1)
        ups = w @ ytr
        preds[lo:lo + 256] = (a0 + ups) / (a0 + b0 + n_eff)
    return float(np.mean((preds - yte) ** 2)), len(te_idx)
