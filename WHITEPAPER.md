# Polaris: A Product-Kernel Beta-Binomial Estimator for Short-Horizon Crypto Binary Markets

**Status: internal research note, tentative.** Single asset (BTCUSDT), single venue (Polymarket),
~48 days of history, one walk-forward evaluation. Not peer reviewed. Numbers below are the best current read of the
evidence, not a claim of a production-ready trading system — see §7 for what would need to hold up before that claim is
reasonable.

---

> **Final conclusion: see §9 (2026-10-08) — no tradeable edge found.**
>
> ## ⛔ RETRACTION (2026-09-12, updated 2026-09-16) — wrong label, wrong target
>
> **This paper optimises and scores against a quantity the market does not pay out on.**
> Polymarket resolves on the **Chainlink BTC/USD TWAP-60s data stream**, comparing the TWAP *over
> the window* against the price at the window's start. `Y` here is an endpoint comparison of
> **Binance** trade prices: wrong source *and* wrong functional form.
>
> §2.2 anticipated the source risk, then asserted that "Chainlink resolves on a **point-in-time**
> price comparison (start vs. end), not an average" — **exactly backwards**. It is an average.
> Nobody had read the market description, which states the rule in plain English.
>
> Polymarket adopted this rule on **2026-08-07**. Measured against 42,000 real settlements, the
> paper's label agrees 95.8% *before* that date and **88.0%** after (BTC; ETH 88.9%), with the
> error concentrated in near-tie windows — **27.9%** disagreement when the window moves `<1bp`,
> falling to ~0% above 10bp. §2.2 identifies the near-tie regime as "exactly the regime this note's
> edge lives in", which is precisely where the label is closest to noise.
>
> **Consequence for §7.** Scoring the market against this label understated it badly: Polymarket's
> true Brier is **0.159**, not the 0.201 reported. On real settlement the comparison reverses —
> model **0.187** vs market **0.159**, paired *t* = **−5.56**. The §7.4 claim that the model
> out-forecasts the market is **withdrawn**. The live test reverses with it: 1,553 gated trades at
> real ask prices give a **34.0%** win rate, *z* = **−4.41**, RoC **−12.7%**. Retraining on actual
> settlements does not close the gap. **Read §7 as void.**
>
> **What the follow-up work established.** About three quarters of the deficit was *structural*
> rather than informational. A TWAP is partially observed as the window runs: by *t* seconds in,
> *t*/300 of the final average is already determined. §3's features predict an endpoint difference —
> a near-martingale — while the contract pays on a running average against an anchor. Four features
> describing the payoff's own geometry (notably the distance-to-strike, how much the remaining
> seconds must average to flip the sign) cut the gap to the market by **70–77%**, to −0.005/−0.008,
> and help the §4 kernel estimator *more* than a GBM because they replace dimensions rather than add
> them. The residual is still significant (*t* ≈ −4) and is plausibly the last structural error: the
> TWAP and its anchor are still reconstructed from Binance rather than read from the settling series.
>
> **This does not rehabilitate the thesis.** Closing the remaining gap buys parity with the market,
> not an edge, and an edge would still have to clear the ~1.1¢ spread. What stands from this paper:
> §4's estimator construction, §6's MLX work, and the observation that the estimator class was never
> the binding constraint.

---

## Abstract

Polymarket lists 5-minute "Bitcoin Up or Down" binary markets that resolve on the sign of the Chainlink BTC/USD price
change over the window. We model `P(up)` at every second of the window as a function of six microstructure features
derived from Binance trade data, using a product-kernel Beta-Binomial estimator — a non-parametric, model-free approach
chosen because the true functional form of `P(up | features)` is unknown and plausibly regime-dependent. The estimator
is calibrated by minimizing Brier score under leave-one-window-out cross-validation, using an MLX (Apple GPU)
-accelerated implementation to make a real ~10⁶-observation calibration tractable. A walk-forward backtest against
Polymarket's own historical quotes finds a statistically strong, timing-entangled edge concentrated in the first 45–90
seconds of each window, while price is still anchored near 0.50: **60.9% realized win rate on 12,571 trades across 48
out-of-time days (~24σ over the 50% null)**, holding up under a simulated one-sided spread cost. We report the full
derivation, the calibration procedure, and — deliberately — every negative result and open caveat along the way, since
several early reads of this data looked like signal and were not.

---

## 1. Problem setup

For a 5-minute window starting at time `t₀`, let `S(t)` be the Binance BTC/USD trade price at elapsed time
`t ∈ [0, 300]` seconds into the window, and let `Y ∈ {0,1}` be the resolution outcome (1 = "up"). Polymarket resolves:

```
Y = 1[ P_chainlink(t₀ + 300) ≥ P_chainlink(t₀) ]
```

using the Chainlink BTC/USD composite, not Binance spot — a distinction that matters and is addressed directly in §2.2.
The market also quotes a live price `yes(t) ∈ (0,1)`, interpretable as the market's own estimate of
`P(Y=1 | information at t)`.

The object of interest is `p(t) = P(Y=1 | x(t))`, where `x(t)` is a feature vector derived from
`S(·)` up to time `t` (§3). Comparing our `p̂(t)` against `yes(t)` gives an edge:

```
edge_yes(t) = p̂(t)      − yes(t)
edge_no(t)  = (1−p̂(t)) − no(t)
```

which is tradeable if the market is slow to incorporate the same information — the central empirical question this note
investigates.

---

## 2. Data

### 2.1 Sources

| source                      | role                                                      | granularity                 |
|-----------------------------|-----------------------------------------------------------|-----------------------------|
| Binance WS / Vision archive | trade prints → 1s OHLCV bars → per-second window features | tick → 1s                   |
| Polymarket Gamma + CLOB API | market discovery + historical YES/NO quote history        | dense, forward-filled to 1s |

BTCUSDT, 2026-07-09 through 2026-08-27 at time of writing: 138M raw trades, 3,706,841 window-observation rows across
14,400 five-minute windows.

### 2.2 Label alignment

> **Superseded (2026-09-12).** Everything in this subsection describes a Binance-derived proxy
> label that was later shown to disagree with actual Polymarket settlement on 14.6% of windows.
> The self-criticism below — that the flips concentrate in near-tie windows — was correct in
> mechanism and badly wrong in magnitude: the real flip rate against settlement is 45.7% for
> windows moving <1bp, not 11.7%. The reassurance that "a walk-forward re-run after the fix
> reproduced the same qualitative result" carries no weight, because the re-run used the same
> wrong label.

Chainlink resolves on a **point-in-time** price comparison (start vs. end), not an average. An earlier version of this
pipeline labeled `Y` using the trade-size-weighted VWAP within the boundary *second* of the window — a subtly different
quantity that a burst of trades in that second can bias. The label was corrected to compare the last trade price
immediately before window start against the last trade price at/before window end (a same-exchange proxy for the
Chainlink composite, since we do not ingest Chainlink directly). The fix changed the outcome on 3.7% of windows,
concentrated almost entirely (11.7% flip rate vs. ≈0%) in windows where the final delta was within 2 bps of zero —
exactly the near-tie regime this note's edge lives in. A walk-forward re-run after the fix reproduced the same
qualitative result (§7.1), which is some evidence the finding is not an artifact of the original labeling error, though
it does not rule out other data issues of the same kind.

A second, unrelated data-integrity issue was found and fixed during this work: an incremental aggregation bug that, for
roughly one second per window, silently overwrote already-correct features/labels with a degenerate recomputation. It
affected 992 of 3,706,841 rows (0.03%) and those rows are never sampled by the cadence-thinned walk-forward query used
for the headline result — see the repository's `RESEARCH_LOG.md` for the full mechanism and fix. Flagged here because
a paper about statistical evidence should disclose data pipeline bugs that were found, not just the ones that were not.

---

## 3. Features

Six features are derived per second, shared verbatim between live inference and calibration (a duplicated feature
pipeline was an actual bug class in an earlier version of this codebase):

| feature           | definition                                                                      | domain      |
|-------------------|---------------------------------------------------------------------------------|-------------|
| `Δ` (delta)       | `(vwap(t) − vwap(0)) / vwap(0) × 10⁴` — cumulative move from window open, bps   | ℝ           |
| `M` (momentum)    | `Δ(t) − Δ(t−L)`, L = 15s — move over the trailing L seconds                     | ℝ           |
| `I` (imbalance)   | cumulative maker-buy volume ÷ cumulative taker volume                           | (0, 1)      |
| `V` (log-vol)     | `log(max(σ₁₀ₛ, 0) + ε)`, ε=10⁻⁸ — log of trailing-10s realized volatility (bps) | ℝ           |
| `Φ` (time-of-day) | `(sin, cos)(2π·s⁄86400)`, s = UTC seconds since midnight                        | unit circle |
| `τ` (elapsed)     | seconds since window open                                                       | [0, 300]    |

`Φ` is encoded on the unit circle rather than as a raw 0–86400 scalar specifically to avoid a discontinuity at the 23:
59→00:00 boundary. A feature missing (NaN) on either the query or an observation contributes a neutral weight of 1 for
that dimension rather than dropping the row.

---

## 4. Model: product-kernel Beta-Binomial estimator

### 4.1 Why non-parametric

The functional form of `p(t)` is not assumed. Kernel density estimation lets the training data determine the shape
directly, at the cost of needing enough data density in a six-dimensional feature space — tractable here at `N ~ 10⁶`
observations but a real constraint on how many features can be added without the estimator starving for local density.

### 4.2 Weighting

For query point `q` and training observation `j`, a weight is a product of one kernel per feature dimension:

```
w_j = K_Δ(δ_q,δ_j; h_Δ) · K_τ(τ_q,τ_j; h_τ) · K_M(m_q,m_j; h_M)
    · K_I(i_q,i_j; h_I) · K_V(v_q,v_j; h_V) · K_Φ(φ_q,φ_j; h_Φ)
```

Three kernel families, chosen to match each feature's domain:

**Gaussian** (Δ, τ, M, V — unbounded reals):

```
K(x,y; h) = exp( −(x−y)² / 2h² )
```

**Beta** (I — bounded [0,1], so the kernel must respect the boundary rather than leak density past it). With
`a = q/h + 1`, `b = (1−q)/h + 1`:

```
K(x,q; h) ∝ x^(a−1) · (1−x)^(b−1) / B(a,b)
```

— i.e. the *query* point and bandwidth parameterize a Beta (a,b) density, evaluated at each training observation `x`.

**von Mises** (Φ — circular). Via the sin/cos embedding, using chordal distance as a stand-in for angular distance:

```
K(x,y; h) = exp( −d² / 2h² ),   d² = (sinₓ−sin_y)² + (cosₓ−cos_y)²
```

### 4.3 Beta-Binomial posterior

Weights combine into a probability through a conjugate Beta-Binomial update:

```
n_eff = Σⱼ wⱼ                                   (effective sample size)
ups   = Σⱼ wⱼ · yⱼ

α = α₀ + ups            β = β₀ + (n_eff − ups)          α₀ = β₀ = 2 (prior)

p̂     = α / (α + β)
CI₉₅  = [ F⁻¹_Beta(0.025; α,β),  F⁻¹_Beta(0.975; α,β) ]
```

`n_eff` — not just `p̂` — is load-bearing throughout the rest of the pipeline: it is the gate that prevents a query far
from any training mass (all `wⱼ ≈ 0`) from producing a falsely confident estimate. In that regime
`p̂ → α₀/(α₀+β₀) = 0.5` with a wide `CI₉₅`, which is the desired degenerate behavior, not a failure mode.

---

## 5. Calibration

### 5.1 Step 1 — bandwidth search

The six bandwidths `h = (h_Δ,h_τ,h_M,h_I,h_V,h_Φ)` are free parameters, tuned to minimize Brier score:

```
h* = argmin_h  (1/N) Σᵢ ( p̂₋ᵢ(h) − yᵢ )²
```

where `p̂₋ᵢ` denotes a held-out prediction. Critically, "held out" here means **leave-one-window-out**, not
leave-one-observation-out: every one of the ~299 other seconds in observation `i`'s own 5-minute window is excluded from
`p̂₋ᵢ`'s training set, not just observation `i` itself. Two consecutive seconds within the same window are
near-duplicates (`Δ`, `τ` differ by ~1 bps and 1s respectively); leaving them in each other's training set at prediction
time understates generalization error by a wide margin. An earlier implementation of this estimator made exactly this
mistake (masked only the self-observation), which the pluggable-model refactor caught and fixed by construction (block
masking on a window ID, not a per-row diagonal).

The search itself uses Optuna's TPE sampler over `h`. Because block-CV LOOCV is `O(N²)` — every query scored against
every other observation — and `N ~ 9×10⁵`–`3.7×10⁶` at the scales used here, a plain NumPy implementation is impractical
(~88s per full pass at N=76K, extrapolating to hours at N~10⁶, times however many trials the search needs). An MLX
(Apple GPU) backend was built for this specific bottleneck; see §6.

### 5.2 Step 2 — risk parameters (not directly used for the headline result)

A second, model-agnostic stage takes Step 1's out-of-sample `p̂`, joins Polymarket's historical quote at the same
second, and tunes `(min_edge, max_kelly, trade_window)` by maximizing the Sharpe ratio of a vectorized backtest. This
stage is **in-sample by construction** — it is an optimizer over historical outcomes, not a walk-forward evaluation —
and its output was checked against the honest walk-forward result before being trusted (§7.2 explains why one of its
outputs was specifically rejected).

---

## 6. Computational note: MLX acceleration

Block-CV LOOCV's `O(N²)` cost is the practical bottleneck standing between "a toy calibration on one day of data" and "a
calibration on the full history." A GPU-backed reimplementation of the kernel-weighting step (product kernels +
Beta-Binomial reduction, batched over query rows)
measured a **9.5× speedup** over NumPy at `N=20K`–`100K`, verified compute-bound (batch size in the 128–512 range made
no measurable difference). Correctness was checked against the NumPy reference before trusting it at scale:
**correlation 0.999989–0.999999, Brier matching to 4 decimal places**, with the residual float32-vs-float64 difference
attributable entirely to precision, not a logic difference (Apple GPU shaders have no float64 path, so all arrays are
cast to float32 for this backend — immaterial for a smoothing estimator like this one). Even at 9.5×, a full-scale
(`N~10⁶`) multi-trial search remains too slow to run un-thinned; the search set is cadence-thinned (every 5th
elapsed-second) for Step 1 specifically, while the baseline/tuned Brier reported below is always computed on the full,
un-thinned set.

---

## 7. Tentative results

> **RETRACTION (2026-09-11).** Every trading result in this section was computed against
> `poly_historical_prices`. That table is built from `/prices-history`, which serves **1-minute
> resolution** (~5 real points per 5-minute window); the backfill forward-filled them to ~300
> per-second rows. Inside the 45–90s trade gate those filled prices sit a mean **12.9¢** from the
> contemporaneous order-book midpoint (n=29,223 matched seconds against live book snapshots), and
> the real price moves ~28 times per window versus ~5 in the filled series.
>
> Edge was therefore computed as a fresh Binance-derived signal minus a *stale* quote. The
> resulting "edge" is substantially a measurement of price staleness, and it is not tradeable at
> the quoted price. §7.1's calibration numbers stand (they score against realised outcomes, not
> prices); §7.2's win rates and P&L, §7.3's cost sweeps and §7.4's model comparison do not.
>
> The live forward paper trade, which quotes real CLOB midpoints, gives 43% win / +10.6% return
> on capital over 107 trades — positive but far weaker. Live and backtest `p_up` correlate at
> 0.9986, so this is a data-quality failure, not a model failure. `book-collect` has recorded
> real per-second books since 2026-09-06; re-running these tests against it is the fix.


### 7.1 Calibration

| bandwidths                                             | block-CV Brier |
|--------------------------------------------------------|----------------|
| untuned (`5, 15, 3, 0.05, 0.3, 0.4`)                   | 0.1871         |
| **tuned** (`0.615, 54.27, 15.40, 0.329, 1.179, 2.463`) | **0.1700**     |
| base rate (always predict 0.5)                         | 0.25           |

Every bandwidth moved under tuning; `h_Δ` moved the most (5 → 0.615, an 8× tightening), which is also the single largest
driver of the sharper, more confident probability surface reported qualitatively downstream — the model became
substantially more decisive along the delta axis, which is consistent with, and probably causal to, the Brier
improvement.

### 7.2 Walk-forward trading result

A walk-forward backtest (train on a rolling window of prior days only, score each test day out-of-sample, take the first
second per window where `edge ≥ min_edge` and `n_eff ≥ min_n_eff`
inside the trade window) was run at `min_edge = 0.05`. Each signal buys **one contract** and holds it to settlement — no
early exit, no shorting; the NO token is bought outright rather than selling YES. Capital at risk is therefore the entry
price (mean 0.470), not a fixed $1, so P&L below is denominated in contracts:

| configuration                                            | trades | win rate  | total P&L | window |
|----------------------------------------------------------|--------|-----------|-----------|--------|
| untuned bw, original label                               | 2,867  | 55.6%     | +266      | 45–90s |
| untuned bw, corrected label                              | 2,869  | 55.9%     | +275      | 45–90s |
| **tuned bw, corrected label, 14-day slice**              | 2,593  | **62.9%** | +382      | 45–90s |
| **tuned bw, corrected label, 48-day out-of-time**        | 12,571 | **60.9%** | +1,755    | 45–90s |
| **strict out-of-time: test days 2026-08-31..09-10 only** | 1,271  | **62.7%** | +215      | 45–90s |

The 48-day row is the one to trust: ~37 of those 48 days were never touched by the model during tuning (not in the Step
1 search, not in Step 2, not in the earlier 14-day-only walk-forward). Win rate settling from 62.9% (same window the
bandwidths were tuned on) to 60.9% on the wider, mostly-unseen set is the expected direction for a slight home-field
advantage disappearing, not a collapse — and both numbers sit far above the untuned baseline (~56%). At 12,571 trades, a
60.9% win rate is approximately:

```
z = (0.609 − 0.5) / sqrt(0.5 × 0.5 / 12571) ≈ 24.4
```

standard deviations above the 50% null. This is a large number and should be read with the appropriate skepticism for a
single-strategy, single-asset backtest — see §7.5.

**Return on capital.** Because each trade risks its entry price rather than a fixed unit, the informative figure is P&L
per dollar staked: 12,571 trades committed 5,902 contract-dollars and returned +1,754.8, i.e. **+29.7%** on capital
deployed. The two sides are balanced and both profitable — NO: 6,151 trades, 61.5% win, +31.1% RoC; YES: 6,420 trades,
60.3% win, +28.5% RoC — so the result does not depend on systematically fading (or following) the market.

**A stricter out-of-time test.** The 48-day row above is "out-of-time" in the sense that most of its days were not in
the tuning set, but all of that data existed when the tuning was done. A later run tested only on 2026-08-31..09-10 —
nine days that were **not yet collected** when the bandwidths, the 45–90s gate and the 60.9% figure were chosen —
training on a rolling five prior days as usual. Result: 1,271 trades, **62.7%** win rate, +215 P&L, Brier 0.1619
(slightly better than the 0.1662 in-period). At the *measured* 1¢ execution cost (§7.3) it holds at 62.8% and +205.

At 1,271 trades the standard error is ≈1.4pp, so 62.7% sits ≈9.4σ above the 50% null, and its 1.8pp gap from the
in-period 60.9% is well inside noise — i.e. the edge did not decay on data that could not have influenced any modelling
choice. This is the single strongest piece of evidence in this note against the "period-specific artifact" explanation.
It does not address single-asset or single-venue risk, and nine days remains a short window that could still be a
favourable regime.

**Why 45–90 seconds.** A three-part diagnostic (documented in `RESEARCH_LOG.md`) found the market price is anchored
near 0.50 for roughly the first 30–60% of each window — a structural feature of these forward-start contracts, not
mispricing — while the model's own out-of-sample reliability curve is nonetheless sloped from very early in the window.
The tradeable edge lives in the overlap: early enough that price hasn't moved yet, late enough that the model's signal
is not pure noise. Discrimination *rises* monotonically with elapsed time, but by the time it's highest (240–300s) the
market has already moved too, and gating trades that late collapses win rate to ~38–44% — the edge is timing-entangled,
not a simple "trade whenever the model is most confident" rule.

### 7.3 Execution cost — measured, then swept

Polymarket publishes last-trade history but no historical order book, so execution cost could not be backfilled: it had
to be collected forward. A book collector (`book-collect` → `poly_book_snapshots`) sampled the live CLOB every 5s for
both tokens of the active market. **Measured** over 2026-09-06..09-10:
121,094 snapshots, 768 windows, 2 symbols.

Inside the 45–90s trade gate (BTCUSDT, n=9,359):

| quantity                  | mean      | p50   | p90   |
|---------------------------|-----------|-------|-------|
| touch spread              | 1.03¢     | 1.00¢ | 1.00¢ |
| **best_ask − last_trade** | **0.51¢** | 1.00¢ | 1.00¢ |

`best_ask − last_trade` is the correction that matters, because the backtest fills at the historical last-trade price:
it is the cost of crossing from where the tape printed to where you could actually buy. Its mean of ~½ tick with a p50
of a full tick reflects a bimodal distribution — the last trade sits at either the bid or the ask, so the gap is ~1¢
or ~0 — not a tight cluster around 0.5¢.

Slippage beyond the touch, walking the ask side (same gate):

| order size | fillable | mean slip vs touch | p90   |
|------------|----------|--------------------|-------|
| $1         | 100%     | 0.00¢              | 0.00¢ |
| $10        | 100%     | 0.03¢              | 0.00¢ |
| $100       | 100%     | 0.35¢              | 0.98¢ |
| $1,000     | 100%     | 3.05¢              | 5.79¢ |

So total realistic taker cost ≈ 0.5¢ at $1–10, ≈0.9¢ at $100, and ≈3.6¢ (p90 ≈6.8¢) at $1,000. Cost is essentially flat
across the window — 1.03¢ from 0–30s through 210–240s, widening only in the final minute (1.21¢ at 270–300s) — so there
is no hidden cost penalty in the early window where the strategy trades. Visible ask depth declines through the window
(≈43.9K → 22K contracts) but stays ample. ETHUSDT agrees closely (spread 1.08¢, cross 0.54¢, n=9,402).

Because the tick size is 1¢, a 1¢ spread is the tightest the venue permits:
these markets sit at the minimum tick essentially always, which is a favourable venue structure rather than a modelling
choice.

**Mapped onto the sensitivity sweep.** The same fixed-cost sweep as before, now read at the measured cost rather than as
an open assumption (netted out of edge before the trade gate, both model classes, tuned bandwidths, full 52-day set):

| assumed one-sided cost | KDE trades | KDE win | KDE P&L | GBM trades | GBM win | GBM P&L |
|------------------------|------------|---------|---------|------------|---------|---------|
| 0¢                     | 12,571     | 60.9%   | +1,755  | 13,107     | 61.2%   | +1,737  |
| 1¢                     | 12,168     | 61.1%   | +1,629  | 12,822     | 61.6%   | +1,646  |
| 2¢                     | 11,690     | 61.7%   | +1,548  | 12,511     | 61.8%   | +1,540  |
| 3¢                     | 11,238     | 62.1%   | +1,453  | 12,161     | 62.1%   | +1,436  |
| 5¢                     | 10,247     | 63.1%   | +1,257  | 11,365     | 62.8%   | +1,239  |
| 7¢                     | 9,212      | 64.0%   | +1,066  | 10,486     | 63.4%   | +1,032  |

At the measured ~0.5–0.9¢ (order sizes up
to $100) the result sits between the 0¢ and 2¢ rows: the edge is substantially intact. At $1,000 per trade (~3.6¢
mean, ~6.8¢ p90) it sits between the 3¢ and 7¢ rows and remains clearly positive. Win rate rises with assumed cost while
trade count falls, which is the mechanical consequence of netting cost out of edge before gating — a higher effective
threshold selects a smaller, higher-conviction subset. Total P&L falls throughout, so this is selection, not a benefit
of cost.

**What this does and does not settle.** It replaces the project's last large open assumption with a measurement, and the
measurement is favourable. It does not make the backtest a live result. The book data covers 2026-09-06..09-10 while the
backtest covers 2026-07-09..08-30, so the measured cost is applied by mapping onto a sweep, not by re-pricing those
specific historical trades — that is not possible without a historical book. It also measures the *visible* book at a
sampled instant: queue position, latency, someone else taking the touch first, adverse selection against a
better-informed counterparty, and the market's reaction to our own order are all still unmodelled, and all push the same
direction (worse). A maker/limit strategy would have a different, likely better cost profile and is untested. See §7.5.

### 7.4 Model-class control: a gradient-boosted challenger

A single model's backtest cannot distinguish "this signal is real" from "this estimator's inductive biases happen to
suit this data." As a control, a LightGBM classifier was run through the *identical* harness — same features, same
walk-forward day splits, same gates, same market prices — with only the estimator swapped (`model/factory.py`,
`backtest --model gbm`):

|             | Brier  | trades | win rate | total P&L | P&L / trade |
|-------------|--------|--------|----------|-----------|-------------|
| KDE (tuned) | 0.1662 | 12,571 | 60.91%   | +1,755    | 0.1396      |
| GBM         | 0.1666 | 13,107 | 61.22%   | +1,737    | 0.1326      |

The two are statistically indistinguishable: a 0.0004 Brier gap, and a 0.31pp win-rate gap against a per-arm standard
error of ≈0.44pp. Two function approximators with very different inductive biases — a product-kernel density estimate
assuming conditional independence across feature dimensions, and an axis-aligned boosted tree ensemble that models
interactions freely — converge on the same performance. Two readings follow, and the second is the more useful one:
the KDE's independence assumption is **not** costing measurable accuracy, and the pair of them plausibly sit near the
ceiling of what *these six features* support. Further gains most likely require better features or better data, not a
better estimator.

Two asymmetries are worth stating rather than hiding. First, the GBM required far heavier regularization than stock
settings to get there (`num_leaves=7`, `min_child_samples=20000`):
within a window the ~300 per-second rows are autocorrelated and share a single label, so the effective sample size is
the number of windows (~14K), not rows (~3.7M). At stock settings (`num_leaves=63`, `min_child_samples=200`) it scored
Brier 0.2295 — barely better than the 0.25 base rate — and emitted saturated 0.000/1.000 probabilities. Second, the GBM
has no `n_eff`
analogue, so its confidence gate is inactive; it therefore took 536 more trades than the KDE and earned *less* in total
(+1,737 vs +1,755). Those marginal trades were net-negative, which is mild evidence that the KDE's `n_eff` gate is doing
real filtering work rather than being decoration — the quantity the GBM cannot reproduce is exactly the one that
improves it.

### 7.5 Threats to validity

- **Single asset, single venue, one time period.** Everything here is BTCUSDT on Polymarket, July–August 2026. No claim
  is made about generalization to other assets, other prediction markets, or other volatility regimes.
- **Execution cost is now measured, but only its taker-cross component, and out of period.**
  §7.3 replaces the pure assumption with 4 days of sampled order book (~0.5–0.9¢ at order sizes up to $100, against a
  sweep that survives 7¢). What remains unmodelled: queue position, latency, losing the touch to someone faster, adverse
  selection, and our own market impact — all of which push cost upward. The book window (September) also post-dates the
  backtest window (July–August), so the mapping is distributional, not trade-by-trade.
- **Step 2's output is explicitly not trusted as reported.** The model-agnostic risk optimizer, run against the tuned
  model's out-of-sample predictions, suggested a *late* trade window (172–219s) and a higher `min_edge` (0.098) —
  directly contradicting every walk-forward diagnostic. This is treated as an overfitting artifact of optimizing
  in-sample over non-walk-forward (block-CV, not chronological) predictions, and was falsified by re-running the honest
  walk-forward at the *original* 45–90s window with the tuned bandwidths, which confirmed the edge got *stronger* in
  place rather than needing to move. It is reported here as a worked example of why an in-sample optimizer's output
  needs an independent, honest check before being trusted — not as evidence against the method in general.
- **No live execution has been attempted.** Every number above is a paper backtest against historical quotes. Slippage,
  partial fills, and the market's own reaction to a real order are entirely unmodeled.
- **One label proxy, not the ground truth.** The training/evaluation label uses a same-exchange point-price proxy for
  Chainlink's composite (§2.2), not Chainlink data directly (not ingested — no simple public REST source exists for it,
  unlike Binance Vision). Any systematic Binance-vs-Chainlink divergence would bias every result in this note in a
  direction not yet measured.

---

## 8. Open items, in priority order

1. Ingest Chainlink data directly to replace the same-exchange label proxy, and measure the proxy-vs-ground-truth
   disagreement rate directly rather than inferring it. With execution cost now measured (§7.3), this is the largest
   remaining unverified assumption in the chain.
2. Re-tune bandwidths on the full 48-day set (currently tuned only on the original 14-day slice and shown to generalize
   acceptably, but not re-optimized) — lower priority given how well the existing tuning has generalized so far.

---

*Full derivations live in the codebase docstrings (`polaris/model/`); the complete, chronological research log —
including the negative results, the diagnostics that overturned earlier "no edge" conclusions, and every intermediate
number — is in `RESEARCH_LOG.md`.*

---

## 9. Conclusion (2026-10-08): what was tested after the retraction

After §7 was withdrawn the question was reopened with everything corrected — real settlement labels,
the measured settlement rule, Polymarket's taker fee, and live (not reconstructed) prices — and widened
from "can this model forecast?" to "can any approach here make money?".

**9.1 The contract, measured.** Settlement is Up iff Chainlink's TWAP-60 at the window's last second is at
least its value at the first (99.79% agreement, 2,421 windows). TWAP-60 equals the trailing-60s mean of
Chainlink spot to within 0.1 bp, and that spot streams free and continuously from Polymarket's relay. So from
second 240 the outcome is progressively locked in, from data anyone can read. Takers pay `0.07·p·(1−p)` per
share; makers pay nothing and receive 20% of it.

**9.2 Forecasting.** Re-targeting the model at the contract's own geometry (price vs. the settlement anchor,
fraction of the final average already fixed) cut the gap to the market by 70–77%, and those two features alone
explain the gain. The model nonetheless stays behind the market (Brier gap −0.005 to −0.008, *t* ≈ −4).

**9.3 Trading.** All after fees, against real settlement:

| Strategy | Result |
|---|---|
| Live paper trading (original / improved model) | −15.0% / −14.9% return on capital |
| Late-window taking, live Chainlink + book, held-out half, 1 s latency | −5.4¢/share (*z* −2.2); Binance-led −2.7¢ (*z* −2.9) |
| Maker, millisecond event replay with sub-second cancels | ≤ break-even at the head of the queue; model selection makes it worse |
| Favourite–longshot bias | negative in every price bucket |
| Endgame sweep (> 0.99) | break-even, negligible depth |
| YES + NO within one market | < $1 in 0.03% of snapshots |
| 5m vs 15m pairs (logically ≥ $1) | riskless in 3.6% of windows; median life 2 s; ~8 shares; not capturable at 1 s |

**9.4 Answer.** Polymarket's 5-minute crypto markets are efficient after fees for anyone without
sub-second, two-legged execution. The one real inefficiency found — cross-series pairs — is combinatorial
and fleeting, as the literature on these markets reports, and is worth on the order of tens of dollars a day
to whoever is fastest. The project's question is answered: not feasible here.

**9.5 Lessons that generalise.** Read the contract before modelling it. Grade against what actually pays
out. Treat every positive result as a data bug until a leave-one-out, a held-out half and a latency test
fail to kill it — in this project every one of them eventually did.

