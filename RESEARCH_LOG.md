# Research log

Chronological record of the Polaris project: design decisions, every experiment, and every
result -- including the ones later withdrawn. Kept as fixed-width text (it was written that way,
and the tables depend on alignment). Start with the **FINAL VERDICT** at the top; the analysis
scripts behind each session are indexed in [`research/README.md`](research/README.md).

```text
Polaris — Project Status
========================

Predicts P(BTC up) for Polymarket's 5-minute "Bitcoin Up or Down" markets from
Binance order-flow features, via a product-kernel Beta-Binomial (KDE)
estimator. One pipeline, read top to bottom: ingest -> aggregate -> model ->
serve. Originally a ~13,700-line, Kafka-based three-service codebase
(binance_ingestion/ + bar_aggregator/ + backend/); refactored (sessions 1-8)
into the single polaris/ package below, then run through real research
(sessions 9-12) to find out whether the model actually has tradeable edge.

FINAL VERDICT (session 35, 2026-10-08): NOT FEASIBLE with the data and
latency available here. Polymarket's 5-minute crypto Up/Down markets are
efficient after the taker fee for anyone without sub-second, two-legged
execution. Every strategy tested, and its result after fees, is listed at the
end of session 35; the write-up is WHITEPAPER.md section 9 and the analysis
drivers are in research/.

The sessions below are chronological and are left as written. Earlier
verdicts -- including claims that the model beat the market, or that an edge
existed in the first 45-90s -- were each later withdrawn; the session that
withdrew them says why. Read any number in context of the sessions after it.

Build Status
------------

Legend: [x] done & verified   [~] in progress   [ ] not started

  [x] Infra: docker-compose.yml (Postgres 16), core/schema.sql, core/config.py
      (pydantic-settings), core/db.py (psycopg3 async pool)
  [x] ingest/     Binance WS -> PG (sink.py, binance.py) + Binance Vision
                  historical backfill (backfill.py)
  [x] aggregate/  incremental trades -> 1s bars -> window observations
                  (sql.py has the 4-CTE feature query; aggregate.py the loop)
  [x] polymarket/ shared Gamma/CLOB client, historical price backfill,
                  discovery.py (live active-market + CLOB price discovery,
                  TTL-cached, wired into api/app.py + endpoints.py + ws.py)
  [x] model/      ProbabilityModel protocol; features.py (one shared feature
                  pipeline for live + calibration); bayesian/ (KDE + kernels
                  + MLX backend); calibrate.py (two-phase); backtest.py
                  (walk-forward paper-trade + gate/spread sweeps); risk.py
      [x] gbm/     gradient-boosted challenger (LightGBM) behind the protocol,
                   run through the identical walk-forward harness via
                   model/factory.py (`backtest --model gbm`) — session 14
      [ ] markov/  parked: the GBM control (session 14) suggests the estimator
                   is not the bottleneck, so another model class is low-value
                   next to better features/data
  [x] api/        FastAPI REST + WebSocket (tick/micro/matrix/macro/polymarket),
                  EstimatorCache, live edge + trade gate
  [x] frontend/   Tailwind rebuild, 3 parts: probability surface (hero),
                  signal band + context panels, Polymarket/edge/diagnostics
      [ ] backtest + calibration panels (deferred, need ws messages for them)
  [x] cleanup     legacy dirs (binance_ingestion/, bar_aggregator/, backend/,
                  validate_idea.py) deleted; confluent-kafka/matplotlib/
                  asyncio dropped from pyproject.toml (session 13)
  [x] docker      Dockerfile + docker-compose --profile app: ingest/aggregate/
                  serve/frontend containerized (dev convenience; calibration
                  stays bare-metal, needs MLX/Metal). mlx moved to an
                  optional dependency accordingly (session 13)

Next up, in priority order:
  1. Ingest Chainlink directly to replace the same-exchange label proxy and
     measure the proxy-vs-truth disagreement rate instead of inferring it.
     With execution cost measured (session 15), this is the largest remaining
     unverified assumption in the chain: the label is a same-exchange proxy
     for the price the market actually settles on, and it matters most at
     small delta -- exactly the near-tie regime the edge lives in.
  2. Better FEATURES, not a better estimator — session 14's GBM control found
     two very different model classes hitting the same ceiling on the current
     six features, which points at the feature set as the binding constraint.
  3. Keep book-collect running. Every extra day makes the cost measurement
     less period-specific, and it is the only way to ever get a book window
     that OVERLAPS a backtest window (which would allow trade-by-trade
     re-pricing instead of the current distributional mapping).
  4. Re-tune bandwidths on the full 48-day set — lower priority; the 14-day-
     tuned bandwidths already generalized well (62.9%->60.9% win, not a
     collapse), and a full re-tune costs ~16x the session-12 run (O(N^2)
     LOOCV). Worth it later, not urgent now.

(polymarket/discovery.py, previously listed here as not-yet-built, turned out
to already exist and be fully wired in — a stale checklist line inherited
from an earlier doc pass. Session 13 caught it while verifying before
"implementing" it; see git history for discovery.py's actual session-8 origin.)


Architecture
============

polaris/
  core/       config.py (pydantic-settings), db.py (psycopg3 pool), schema.sql
  ingest/     binance.py (live WS), backfill.py (Binance Vision history), sink.py
  aggregate/  sql.py (the 4-CTE feature query — the real asset), aggregate.py (loop)
  polymarket/ client.py (shared Gamma/CLOB + rate limiter), backfill.py
  model/      base.py (protocol), features.py (shared feature pipeline),
              bayesian/ (KDE + kernels + mlx_kernels.py), calibrate.py,
              backtest.py, risk.py, dataset.py
  api/        app.py, endpoints.py, ws.py, cache.py, live.py, edge.py,
              query_context.py (shared live-feature-Query builder)
  main.py     unified entry point: ingest|aggregate|backfill|backfill-poly|
              serve|calibrate|backtest
  frontend/   Tailwind + Plotly dashboard (see frontend/README.md)

The tree reads as the flow: core -> ingest -> aggregate -> model -> api.


Key Design Decisions (settled)
===============================

D1. No Kafka. Binance WS writes straight to PostgreSQL; idempotency via
    ON CONFLICT DO NOTHING on (symbol, trade_id). Safe for public market data
    on coarse 5-min windows — a dropped tick is filled by the next one. A
    thin Sink seam keeps Kafka a drop-in option if it's ever needed.
D2. psycopg3 throughout (async + COPY), not asyncpg.
D3. Flow-shaped packages, one file per concern until it genuinely grows —
    not the old 11-flat-files-plus-__init__-re-export-ceremony style.
D4. Pydantic where it pays (config, API/WS models), plain tuples on the hot
    path (trade ingestion) — never Pydantic per-trade, it allocates+validates
    per record and would tax ingest throughput for no benefit.
D5. Modelling is pluggable behind one ProbabilityModel protocol (fit /
    predict_proba / predict_grid / loocv_probabilities / optimize), judged on
    leave-one-WINDOW-out (block CV) Brier, so a KDE and a GBM (and later
    others) compete on the same harness. Block CV, not per-observation LOOCV:
    per-observation leaks ~299 near-identical same-window neighbours into
    training, inflating the score.
D6. ONE feature pipeline (model/features.py) for both live inference and
    calibration — the old code had this duplicated with a TODO admitting the
    hazard. (Session 12 found the same class of bug one level up: cache.py's
    heatmap grid and live.py's point prediction had drifted apart on which
    Query fields they set. Fixed via api/query_context.py — see Research Log.)
D7. Label: Polymarket resolves Up iff the Chainlink BTC/USD price at window
    END >= price at START (data.chain.link), a POINT-IN-TIME comparison —
    not Binance/spot, and not a within-second average. Session 11 aligned the
    training label to boundary point prices accordingly (see Research Log).
D8. Latency levers that actually matter (documented so nobody optimises the
    wrong thing): uvloop, batched COPY over per-row INSERT, a warm connection
    pool, and the estimator's grid-precompute trick (the 61x61 heatmap once
    per window, O(1) per-tick lookup). The model library choice is not a
    latency lever.

Decisions taken during the initial build, still true:
  - Binance backfill lives in ingest/ (feeds crypto_trades); Polymarket
    backfill stays with polymarket/.
  - Kept the staging-table pattern for backfill: it's what makes it both
    fast and idempotent (direct COPY-to-prod aborts a whole day on one dup).
  - Kept chunked catch-up in the aggregator: an unbounded first run after a
    14-day backfill would be one giant query over ~1M+ bars.
  - Binance Vision has a ~2-day publish lag; `backfill --days N` always
    skips the most recent 1-2 days (handled, logs a warning).


Research Log
============

Sessions 1-8 (2026-07-23) — initial build, verified end to end:
  ingestion (live WS + backfill, idempotent), aggregation (window features,
  cross-cycle correctness fixed vs. the old bar_aggregator), Polymarket
  ingestion (CLOB dense prices), the KDE model (ported bit-exact from the old
  estimator, untuned block-CV Brier 0.1946 vs 0.2496 baseline on one day of
  BTC), the API serving layer, and all 3 frontend parts (probability surface,
  signal band + panels, Polymarket/edge/diagnostics). Flagged then: numpy
  block-LOOCV is ~88s/76K obs — too slow for a real 14-day calibration
  (resolved in session 12, see below).

Session 9 — backtester v1 + BTC data foundation:
  Walk-forward paper-trade harness built. First run (default bw): 53.6% win,
  looked inflated (Binance-not-Chainlink label, no spread cost, ~100% trade
  rate). A min_edge sweep then looked NEGATIVE (win rate flat-then-falling as
  threshold rose — the signature of an artifact, not skill) — tentatively
  "no edge."

Session 10 — the "no edge" call was wrong; edge is real but timing-entangled:
  A 3-panel diagnostic overturned it: the market genuinely anchors ~0.50
  early (confirmed artifact in the PRICE), but the out-of-sample reliability
  curve is strongly and monotonically sloped (Brier 0.192 < 0.25 baseline) —
  the model genuinely discriminates. Per-elapsed-time-bucket breakdown
  resolved the apparent contradiction: discrimination rises with time, but by
  then the market has already moved too. The edge lives EARLY, while price is
  still pinned near 0.50. A gate sweep found the sweet spot: 45-90s elapsed
  peaks at 55.6% win / +266 P&L on 2867 trades (~6 sigma over 50%,
  in-sample). Shipped as TradingSettings.trade_window_start/end (45/90) +
  the time gate in api/edge.py.

Session 11 — Chainlink point-price label fix + spread-cost sweep:
  LABEL FIX: the training label compared open_vwap/close_vwap (the average
  price within the window's boundary SECOND) instead of the point-in-time
  comparison Chainlink actually uses (D7). Fixed in aggregate/sql.py:
  outcome now compares the last trade price before window_start (a
  cross-boundary LAG) against the last trade price at/before window end.
  Applied to all 928,289 existing BTCUSDT observations via a full recompute
  (idempotent, no re-ingestion needed — row counts matched exactly before/
  after). Flip rate: 127/3455 windows (3.68%), concentrated almost entirely
  in near-tie windows (<2bps final delta flipped 11.66% of the time; >5bps
  flipped 0%) — confirms the bug only ever mattered where the edge lives.
  Re-ran the 45-90s gate sweep on the corrected label: pattern held
  essentially unchanged (45s: 55.6%/+266 -> 55.9%/+275) — the edge was not a
  label artifact.
  SPREAD-COST SWEEP: yes_price+no_price-1 (poly_historical_prices' "maker
  spread") checked and rejected as a per-tick spread proxy — averages
  ~0.0001 with p10/p90 of only +-0.005 across 1.18M rows; we only have
  last-trade prices, not the order book. Added a fixed, swept one-sided
  execution-cost assumption instead (backtest.py's --spread-cost/
  --spread-sweep), netted out of edge before gating. Result at the
  production gate (45-90s, untuned bw): edge is robust up to a 5c one-sided
  cost (56.4% -> 54.0% win, P&L stays solidly positive) — contradicts the
  earlier worry that spread could halve the edge.

Session 12 — MLX backend, real calibration, data refresh, heatmap fix:
  MLX BACKEND: ported loocv_probabilities to Apple GPU/Metal
  (model/bayesian/mlx_kernels.py; KDEModel gained backend="numpy"|"mlx").
  Float32 throughout (Metal has no float64); BETA_KERNEL_EPSILON (1e-10,
  tuned for float64) replaced by a float32-safe MLX_BETA_EPSILON (1e-6) --
  1.0-1e-10 rounds to exactly 1.0 at float32, which would've fed log(0) to
  the beta kernel. betaln stays on the CPU via scipy (it's (batch,)-sized,
  not (batch,N) -- not worth porting). Correctness verified against numpy
  before trusting it at scale: correlation 0.999989-0.999999, Brier matched
  to 4 decimals at N=2K/20K/100K. Speed: ~9.5x over numpy at N=20K-100K,
  compute-bound (batch size 128/256/512 identical). calibrate.py gained
  --backend/--train-cadence (cadence thins the Step-1 search set only;
  baseline/tuned Brier and Step 2 still run on the full set).
  REAL CALIBRATION: ran the actual 14-day-scale calibration (928K obs,
  cadence=5 search thinning, 50 Optuna trials). Baseline (default bw) Brier
  0.1871 -> tuned Brier 0.1700. Tuned bandwidths: delta=0.615 time=54.27
  momentum=15.40 imbalance=0.329 volatility=1.179 tod=2.463 (every dimension
  moved from the untuned 5/15/3/0.05/0.3/0.4). Step 2 (the risk optimizer)
  suggested min_edge=0.098 and a LATE trade window ([172,219]) — disregarded:
  risk.py is documented as an in-sample, optimistically-biased optimizer over
  non-walk-forward LOOCV predictions, a classic overfitting signature. The
  honest walk-forward re-verification at the SAME 45-90s window confirmed
  the tuned bandwidths instead: win rate 56.4% -> 62.9%, P&L +261 -> +382, on
  FEWER trades (2701->2593) — more selective and more accurate, not just
  more active.
  DATA REFRESH: backfilled BTCUSDT +50 days (Binance trades + Polymarket
  prices), landed gapless through 2026-08-27. Reset the aggregator cursor
  (it had already fast-skipped the gap as empty during session 11's own
  catch-up-to-now) and recomputed window_observations end-to-end: 3.7M obs /
  14,400 windows, ~4x the original dataset. Re-ran the walk-forward gate
  sweep across the full range (48 test days, ~37 never touched by the model
  in any form before): 60.9% win / +1755 P&L on 12,571 trades, ~24 sigma
  over 50%. Tuning did not overfit to the original 14 days.
  HEATMAP CONDITIONING BUG (found via a dashboard screenshot check — "the
  probability surface doesn't have a lot of variance"): api/cache.py built
  the heatmap grid with momentum/imbalance/log_vol/tod all left unset, so it
  silently marginalized over those 4 of 6 dimensions instead of showing the
  live-conditional slice — while api/live.py's point prediction DID condition
  on all 6. The two serving paths had drifted apart (the same D6 hazard, one
  level up). Quantified: conditioning moved individual cells by up to 0.45
  probability points; mean std-of-P(up)-across-time roughly doubled
  (0.049->0.088). Fixed via api/query_context.py (shared by both paths).
  Some of the "flat" look was this bug; the rest is real — tuned bw_delta
  (0.615, 8x narrower than default) makes the surface genuinely sharper
  along the delta axis now, which is why the Brier score improved.
  APPLIED: .env (gitignored) holds the tuned bandwidths + MIN_EDGE=0.05 +
  TRADE_WINDOW=[45,90) — the walk-forward-verified values, not Step 2's
  in-sample suggestions. MAX_KELLY_FRACTION left at the pre-tuning default
  (Kelly sizing is unverified by walk-forward; backtest.py still only does
  flat $1 stakes).

Session 23 — THE LABEL IS WRONG. Everything downstream of it is retracted.
  Started as routine feature work; ended by invalidating the project's one
  surviving positive result.

  HOW IT SURFACED. Screening new features needed a market benchmark, so the
  model was scored against real book quotes by elapsed-time bucket. The gap to
  the market GREW toward window end -- +0.074 Brier at 240-270s, +0.182 at
  270-300s. That is backwards: a market seconds from resolution should be
  nearly perfect, not worse than a coin flip. Chasing that anomaly:
    - the book was NOT stale/wide late (median spread stayed 0.01), so the
      quotes were healthy and the fault had to be on our side;
    - when the market quoted >0.9 or <0.1 twenty seconds before resolution, it
      agreed with our label only 57.4% of the time.

  THE MEASUREMENT. Fetched Polymarket's own settlement (Gamma outcomePrices,
  which pays exactly 1/0 once settled) for 2036 BTC windows:
      our label disagrees with actual settlement on 14.6% of windows.
  Tested six Binance-derived labelling rules to see if it was a convention bug
  (last_close vs prev_close, vs first_open, vs first_vwap, vwap-to-vwap, strict
  vs >=). NONE beat 84.2%; the stored label at 86.9% was already the best of
  them. So it is not a convention mismatch -- POLYMARKET RESOLVES OFF A
  DIFFERENT PRICE FEED, which tracks Binance closely but not tick-for-tick.

  WHY IT IS WORSE THAN 14.6% SOUNDS. The error is not uniform. By Binance move
  size over the window:
      <1bp  45.7% disagree      5-10bp  1.9%
      1-2bp 30.6%               10-20bp 1.5%
      2-5bp 10.6%               >20bp   0.0%
  Our label is right exactly where the answer is obvious and close to a coin
  flip exactly where it is hard -- and ~30% of 5-minute windows move <2bp.

  WHAT THIS DID TO THE RESULTS. Grading the market against our label made the
  market look far worse than it is. Its TRUE Brier is 0.159, not the 0.201 we
  had been reporting. Re-scored on real settlement (BTC, 989 book windows,
  out-of-time):
      graded on OUR label:  model 0.181 vs market 0.201  gap +0.025  t +5.97
      graded on SETTLEMENT: model 0.187 vs market 0.159  gap -0.021  t -5.56
  The sign flips. The "model beats the market as a forecaster" result -- the
  one finding that had survived every previous cut -- was an artifact of
  grading the market against the wrong label. In the trade gate the true gap is
  -0.019 (t=-5.31).

  Retraining on correct labels does NOT rescue it (gap -0.027, t=-6.45, on the
  2036 windows where settlement was known at the time). Worth redoing once the
  full-history backfill lands, but the gap is not marginal.

  THE LIVE FORWARD TEST INVERTS TOO. paper_signals, one trade per window (the
  earlier 2287-row figure double-counted: many seconds in one window pass the
  gate and share a single outcome, which inflates z by ~sqrt(6)):
      n=376, avg ask 0.412
      on OUR label:  win 43.9%  excess +10.07  z=+1.12  RoC  +6.5%
      on SETTLEMENT: win 34.0%  excess -26.93  z=-3.00  RoC -17.4%
  So the strategy is not "no edge" -- it is significantly LOSING. That follows
  directly from the error structure: the gate fires on confident predictions,
  and confidence concentrates in flat windows, which is precisely where our
  label is noise relative to what pays out.

  FIXED IN CODE, not just measured:
    - poly_resolutions table (core/schema.sql) holding actual settlements
    - PolymarketClient.resolution() -- returns None for unsettled, never a guess
    - polymarket/resolutions.py: idempotent backfill + agreement_report()
    - CLI: `polaris backfill-resolutions --days N`, which logs the disagreement
      rate every run so this can never silently drift again

  STANDING VERDICT: the thesis is NOT supported. On the label that actually
  pays out, the market forecasts better than the model and the live gated
  strategy loses. Every prior positive number in this log that rests on
  window_observations.outcome is void.

  NOT INVALIDATED: the infrastructure, and the finding that the production GBM
  is over-regularized (below). Feature selection needs redoing on real labels.

Session 36 — prior art, and two corrections it forces (2026-10-09).
  Read Wang, "Does a TWAP Deter Settlement Manipulation? Evidence from
  Polymarket's Five-minute Bitcoin Markets" (SSRN 7445139, Sept 2026) and Dai,
  Jia, Yu, "Settlement Manipulation in Prediction Markets" (arXiv 2606.31675).
  (Papers are listed in README.md, Related work; the PDFs are not redistributed.)
    - Dai et al.: under SNAPSHOT settlement (Feb-Apr 2026) 821 wallets made
      ~$8.2M pushing Binance in the final seconds. This is what prompted the
      switch our labels broke on.
    - Wang: the switch removed the final-10s volume footprint (5.68% -> 3.02%
      of window volume); a smaller footprint returned from 5 Sep after the
      taker delay was raised; liquidity rewards Aug 7 - Sep 5 were <0.2% of
      turnover; scheduled taker fees ~$3.8M on $188M turnover, Aug 7-31.
  CORRECTIONS TO THIS LOG:
    (1) The rule was not one regime from 2026-08-07. 5m markets settled on a
        30-SECOND TWAP from Aug 7 and a 60-second TWAP from Aug 14 (15m and 4h
        on 60s from Aug 7). Our Chainlink measurements (from Sep 16) are all
        60s regime and stand; but post-Aug-7 training data includes one week
        on the 30s rule, which the 60s TWAP features slightly mis-describe.
    (2) The rule was never undocumented. Polymarket's changelog states that
        "both the price to beat and the final settlement price come from the
        applicable TWAP feed", and Gamma reports the Chainlink price-to-beat
        and settlement value PER CYCLE (from Feb/Mar 2026) -- exact anchors and
        finals for history, free. We never used those fields; they would have
        replaced both the Binance-reconstructed anchor and most of the paid-
        Chainlink-history question.
  NOVELTY, re-assessed:
    - Settlement rule: known (changelog; Wang Sec. 3, App. C).
    - Calibration of 5m markets: covered by Gruener, "Calibrated but not
      sharp" (SSRN 6863546, 209k contracts) -- our no-FLB result likely
      overlaps; not yet read.
    - Plausibly new: (a) maker vs taker P&L against settlement, after fees,
      at fill level, and the latency decay of informational edge; (b) the
      5m/15m logical link and its fleeting riskless mispricings, whose
      executability interacts with the taker delay (250 -> 50 -> 150 ms).
      Both rest on short samples (1-2 days live; 25 days of fills).

Session 35 — the late-window lead fails on live data. Hunt concluded.
  Live re-test (late_signals: live Chainlink spot + live book, 2,106 settled
  coin-windows, 8 coins). Isotonic calibration fit on the first half of
  windows, evaluated on the second; execution at the ask logged L seconds after
  the decision; taker fee included. PRE-REGISTERED rule (t >= 270, edge >= .02
  after fee, first qualifying second):
      Chainlink, all 8 coins, L=1s:   160 trades  -5.4c/share  z -2.19
      Chainlink, BTC+ETH,     L=1s:    51 trades  -5.8c        z -1.46
      Binance-led, BTC+ETH,   L=1s:   266 trades  -2.7c        z -2.91
  Grid (12 cells, Chainlink, 8 coins, L=1s): only t>=240 & edge>=.02/.05 are
  positive (+1.7c / +2.2c, z ~1.1), flanked by negative cells -- what noise
  produces across twelve tries. The session-30 pocket (+5c at zero latency on
  the Binance-proxy backtest) does not survive live data.

  WHERE THE HUNT ENDS. Every strategy tested on these markets, after fees:
    forecasting better than the market (KDE/GBM, Binance features) -- behind it
    late-window taking on the settling TWAP ---------------------- loses live
    market making, naive or model-selected (ms replay) ---------- <= break-even
    favourite-longshot bias, every price bucket ----------------- loses
    endgame sweep (asks > 0.99) --------------------------------- break-even, no depth
    YES+NO within one market ------------------------------------ never < $1 (0.03%)
    5m/15m cross-series pairs ------------------------------------ riskless but ~2s
                                                                  and ~$0.40 each
  These 5-minute crypto markets are efficient after Polymarket's taker fee for
  anyone without sub-second, two-legged execution; what remains goes to bots
  competing in milliseconds. The answer to the project's question -- is it
  mathematically feasible? -- is no, with the data and latency available here.

Session 34 — cross-series (5m vs 15m) arbitrage: real, riskless, and fleeting.
  The last 5m window of each 15m window settles on the SAME Chainlink TWAP60 as
  the 15m market, against a different anchor. If a5 >= a15, "5m Up" implies
  "15m Up", so 15m YES + 5m NO pays >= $1 (pays $2 when F lands between the
  anchors); mirror pair when a5 < a15. cross_signals: 722 15m windows, 8 coins,
  ~22h; 15m settlements fetched via PolymarketClient.resolution(series="15m").
    - The link HOLDS: 0 violations in 698 settled windows (gap >= 0 bp). The
      "middle" ($2) happened in 35% of windows -- and the median pair cost is
      1.35, i.e. the market prices the middle about right on average.
    - Riskless (pair cost incl. both taker fees < $1): 25 windows (3.6%), on 7
      of 8 coins, median touch size 8 shares, first-chance profit +5.4c/pair,
      worst +0.03c. Not concentrated at the very end (first appears anywhere
      from 0 to 232s into the last sub-window).
    - EXECUTABILITY: priced at the book one logged second later, only 24% are
      still riskless and the mean P&L is -0.2c (worst -10c); at 5s, -3.5c.
      Median opportunity life 2s (mean 13.5s, skewed by one 145s case).
  VERDICT: matches the literature (combinatorial > single-market, fleeting).
  Capturable only by a sub-second taker firing both legs at once, with
  leg-risk if one fills and the other does not, and capacity of ~8 shares /
  ~$0.40 per opportunity -- order of $10-25 a day across all 8 coins at best.
  Real, but not material. (Sub-second persistence is unmeasured: poly_ws_touch
  covered only the 5m tokens.)

Session 33 — literature scan, and the favourite-longshot bias is absent here.
  What others report (arXiv + industry write-ups, 2025-26):
    - Single-market arbitrage (YES+NO < $1) is vanishingly rare: an NBA study
      found 7 executable episodes, median life 3.6s. COMBINATORIAL inefficiency
      (prices across logically linked markets) is far more common -- 290
      episodes, concentrated in the FINAL MINUTES, median return 101 bps.
    - IMDEA Networks: ~$40M of arbitrage profit extracted Apr 2024-Apr 2025,
      mainly multi-outcome "negative risk" markets; 14 of the top 20
      leaderboard wallets are bots.
    - "Executable arbitrage" paper: a payoff-space violation need not be
      exploitable before settlement -- check executability, not just prices.
    - Favourite-longshot bias across thousands of Polymarket/Kalshi markets:
      longshots overpriced, favourites underpriced; Kalshi makers out-earn takers.
    - "Endgame sweep": buy near-certain outcomes at 0.997-0.999; ~0.1%/trade.
  TESTED ON OUR 5m DATA (2,106 coin-windows, 8 coins, live asks, after fee,
  window-clustered z): buying at the ask LOSES IN EVERY PRICE BUCKET --
  longshots (-0.9..-2.4c), the middle (-1..-5c), favourites 0.8-0.99 (-2..-6c).
  No favourite-longshot bias to exploit on these markets. The endgame bucket
  (ask > 0.99, last 30s) is break-even (+0.09c, 66 rows) with no capacity, and
  buying at 0.98-0.99 in the last 30s won only 96.1%.
  Most promising per the literature: COMBINATORIAL -- our 5m/15m cross-series
  test, whose data (cross_signals, 706 15m windows) is already collected.

Session 32 — the maker strategy does not survive a live, queue-aware replay.
  polaris/late/makersim.py, two simulators, all P&L per filled share incl. the
  20% maker rebate, real settlement:
  (a) 1-second grid, overnight (all 8 symbols / BTC+ETH): naive quoting at the
      best bid -2.9..-3.4c; selective (model edge >= .05) -4.5..-5.6c. Fills in
      turn were good (+1.3c back of queue, +3.3c front, theta .05); the losses
      were TRADE-THROUGH fills, i.e. our quote left stale.
  (b) Event-driven ms replay (14h, 328 BTC/ETH windows, 18.2M touch changes):
      our bid follows the touch after a `reaction` delay; queue = displayed
      size, shrinking as it shrinks; fair value = Binance-led, Chainlink-
      anchored (binance_fair_value), which does LEAD the book (+0.065 corr,
      vs -0.01 for Chainlink-only).
        reaction 0.05s: naive -1.3c | theta 0 -1.2c | .05 -2.4c | .10 -3.0c
        reaction 1.00s: naive -2.0c | theta 0 -2.1c | .05 -3.9c | .10 -3.4c
      OPTIMISTIC BOUND, front of queue, 0.05s: naive +0.01c (z .08) -- i.e.
      break-even, matching the market-wide +0.15c -- and selective -1.2..-1.7c.
  Selection by the model makes it WORSE in every configuration.
  Cross-check: the historical decomposition (maker P&L monotone in model
  edge, +1.8c at lag 1s over 4,037 windows) does NOT replicate on today's live
  fills: +1.7..2.4c where the model DISAGREES, +0.6c where it agrees strongly
  -- noise, not a gradient (328 windows, so weaker evidence, but consistent
  with the strategy replay).
  VERDICT: no maker edge with our information. A naive maker at the head of the
  queue roughly breaks even; anything realistic loses; and our fair value -- even
  the fast Binance-led one -- does not pick better fills. ms touch logging
  (poly_ws_touch, ~8 GB/day) switched OFF; its 25.7M rows are kept.
  STILL OPEN, all cheap on data already collected: late-window TAKING on live
  Chainlink with real latency; 5m/15m cross-series consistency; thin series.

Session 31 — overnight collection, and the first evidence for a MAKER edge.
  User: will not pay for data; uptime overnight + next day; keep researching
  (GitHub/Reddit/web allowed). Scope unchanged: paper only.

  FREE REAL-TIME FEEDS, measured:
    - Chainlink SPOT via Polymarket relay, topic crypto_prices_chainlink with an
      EMPTY filter: streams all 8 symbols ~1/s, ~1.5s old. (With a symbol filter
      -- what we used before -- it sends one snapshot then nothing; the TWAP
      topic only ever snapshots.) TWAP60 == trailing-60s mean of this spot to
      within 0.1 bp, so this feed alone determines settlement.
    - CLOB market channel: full book + deltas, ~22ms from the engine.
    - Both sockets can STALL silently (open, no data, no error) -> watchdogs.
    - Trade events report fee_rate_bps=0, but market metadata says feeSchedule
      rate .07 taker-only, rebateRate .2 on 5m/15m/4h alike. Liquidity rewards
      have parameters but rates: null -> no pool.
  15m and 4h markets settle on the SAME TWAP60 stream, end vs start, so the 5m
  and 15m markets ending together share one F with different anchors -> a
  logical price link (cross_signals logs both books to test it). Series exist
  for BTC ETH SOL XRP DOGE BNB HYPE ZEC; the relay streams all eight.

  OVERNIGHT COLLECTOR (late-paper service): every data-second, all 8 symbols,
  Chainlink-based settlement estimate + live book (touch, sizes, $10 depth),
  plus every fill (poly_ws_trades) and the 5m/15m cross books. Warm-starts
  its anchor from chainlink_spot so a restart does not cost a window.

  RESEARCH CONTEXT (GitHub, analyst write-ups, Bitquery on-chain study): latency
  arbitrage (Binance -> Polymarket lag of seconds) was THE bot strategy; taker
  fees were introduced in Jan 2026 specifically to neutralise it. The crowd of
  takers breaks even before fees and loses ~the fee after; consensus is that
  makers are now the structural winners.

  MAKER DECOMPOSITION (6.7M BTC fills Aug 28 - Sep 12, real settlement): makers
  as a group earn +0.15c/share -- trading P&L ~0, they keep the rebate.
  CONDITIONED ON A SETTLEMENT MODEL (Binance-proxy structural model, since no
  Chainlink history for that period), maker P&L is MONOTONE in the model's edge
  on the maker's side, d = P_model(maker side) - price:
      d <= -.30  -1.8c   (-.30,-.15] -3.9c   (-.15,-.05] -1.4c   (-.05,0] -0.2c
      (0,.05] +0.8c   (.05,.15] +1.8c   (.15,.30] +3.4c   > .30 +1.8c
  in every minute of the window. LEAKAGE/LATENCY TEST -- classify with the
  model from L seconds earlier:
      L=0 +2.2c | L=1 +1.8c | L=2 +1.25c | L=5 +0.35c | L=10 +0.3c   (d >= .05)
  L>=1 cannot see the fill, so the separation is real -- and it is a SPEED edge
  of ~1-2 seconds, the same window the latency bots exploited. This is why the
  taker results failed and a maker could work: the model's edge (~1-3c) is
  smaller than the cost of crossing (spread + fee ~2.3c) but a maker pays
  neither. Strategy shape: quote passively on the model-favoured side, CANCEL
  within ~1-2s when fair value moves. Needs Binance direct WS (~100ms), not the
  4s DB path.
  NOT YET SHOWN: that OUR resting orders would get those fills. Historical fills
  went to whoever was first in queue. Next: simulate with queue position from
  late_signals sizes, fills from poly_ws_trades (ms), fair value from
  crypto_trades (ms) -- all being collected tonight.

Session 30 — fees, the exact settlement rule, and a late-window lead.
  Brief changed: user gave free rein to find a profitable approach within
  Swedish law. Ground rules kept: research + backtest + paper only; no real
  orders, accounts or keys without the user's explicit go-ahead. Sweden is NOT
  on Polymarket's restricted list; VPN use to dodge a geoblock breaches its ToS
  and is out of scope regardless.

  (1) TAKER FEES WERE MISSING FROM EVERY P&L. Polymarket charges crypto takers
  fee/share = 0.07 * p * (1-p) in USDC -- 1.75c at p=.5, ~3.8% of price paid
  near the middle; makers pay 0 and receive 20% of a market's taker fees,
  pro-rata by filled volume (help.polymarket.com, Trading Fees / Maker Rebates).
  Now in TradingSettings.taker_fee_rate, backtest._book (gating AND P&L),
  report._score (break-even = ask + fee) and api/edge.py (live gate). With fees
  the live records are: kde -15.0% RoC z -5.64 (1,721); kde_twap -14.9% z -3.43
  (501). Every taker strategy near 50c needs > ~2.3c of real edge per share.
  Fees fall toward the extremes (0.33c at p=.95) -- which points LATE.

  (2) YES+NO ARBITRAGE: dead. Over 254,878 paired snapshots, ask_yes + ask_no
  sits at 1.01; it is below 1.00 in 67 (0.03%), and buying both pays two fees.

  (3) THE EXACT SETTLEMENT RULE, from our own Chainlink collection:
      Chainlink TWAP-60s at t=299  >=  Chainlink TWAP-60s at t=0
  agrees with Polymarket settlement on 99.79% of 2,421 windows (the "average
  over the window" reading: 84.7%). So the contract is decided by the AVERAGE
  OF THE LAST 60 SECONDS vs the anchor, which is progressively LOCKED from
  t=239 onward, in a feed we receive free.

  (4) LATE-WINDOW STRUCTURAL MODEL. Final value = mean of the underlying over
  [239,299]; at t the known part is fixed, the remaining k seconds are a random
  walk whose average has closed-form variance. Underlying = Binance 1s VWAP,
  level-shifted to Chainlink; sigma from trailing 120s; isotonic-calibrated on
  the earlier 60% of windows; traded on the later 40% against the real book,
  first qualifying second per window, taker fee included.
    Market's late price is VERY well calibrated (ask 0.928 vs realised 0.930
    etc.), and overall the market still beats the model (Brier .0548 vs .0600).
    But from t>=270 a pocket appears (min_edge .02 after fee): +0.050/share,
    z +2.20 on 138 trades. Stress tests:
      latency 2s +0.039 (z 1.92) | 5s +0.021 (z 1.14) | 10s +0.013 (z 0.89)
      fill at $10 depth +0.041 (z 1.67) | BTC +0.045 | ETH +0.053
      rule chosen on test-half A, judged on untouched half B: +0.051, z +0.94
      (n=39); same at 5s latency: -0.016.
  VERDICT: a lead, not an edge. Positive on both symbols, at depth, and on the
  untouched half -- but n is tiny and it dies at our current ~5s latency.
  Needs (a) far more windows with BOTH Chainlink and book history, and (b) a
  low-latency path (direct Binance WS + Chainlink polled ~1s, bypassing the
  DB) for any live test. pmdata.dev Plus ($49/mo) sells Chainlink TWAP history
  from 2026-08-01 AND UpDown L2 order books -- the dataset that would settle it.
  Not purchased: spending the user's money is their call.

Session 29 — the backtest was running on a fifth of the real prices; fixed.
  Real-book trading backtest (poly_book_snapshots, gate [45,90), min_edge .05)
  first reported ~500 trades per run and a "6.5%" window trade rate, while the
  live paper test on the same model traded ~90% of windows. Replaying the live
  signals offline ruled out the model: live vs offline p_up correlate 1.0000
  (mean |diff| .002). The bug was a JOIN: the walk-forward scored every 5th
  second (t % 5 == 0) and exact-matched book snapshots, which sample every ~5s
  at an ARBITRARY phase -- evenly spread over all five offsets -- so only 20% of
  snapshots were ever seen (598 of 2,677 windows had a visible gate price), and
  it traded at whichever second aligned rather than the first qualifying one.
  The "6.5%" also counted UNPRICED windows in its denominator.
  FIX (backtest.walk_forward_candidates): score exactly the priced seconds;
  cadence thinning only for dense sources (median >= window/2 prices per
  window); %windows over priced windows only.

  RESULT with ~5x the trades, at the ask (+0.55c):
                         trades   win    P&L/trade   total
    BTC kde_twap          2222   42.2%   -0.0130     -28.9
    BTC kde (old)         2450   37.7%   -0.0119     -29.3
    ETH kde_twap          1965   42.6%   -0.0013      -2.5
    ETH kde (old)         2368   37.2%   -0.0118     -28.0
    pooled kde_twap       4187           -0.0075 (z ~ -1.0)
    pooled kde            4818           -0.0119 (z ~ -1.7)
  The earlier "BTC +0.006" was noise and vanished at 5x the sample. The new
  model's 0.017 Brier gain barely moves P&L: trading profit needs the model to
  beat the MARKET where it disagrees with it, and it still does not (the Brier
  gap to the market stays negative). Tellingly, min_edge=.05 passes in 70-80%
  of priced windows -- the model disagrees with the market by 5c almost all the
  time, and the market is usually the one that is right.

  LATENCY (new, and it cuts against live). The live price differed from the
  book mid by 3.7c when matched by elapsed second, but by a median of 0.0000
  when matched by wall-clock within 1s: not a bug, a timing gap. The data the
  model reads lags the clock by 4.3s median (8.3s p90) -- safe_lag_sec=2, the
  1s poll, whole-second flooring and the recompute. Live therefore pairs a ~4s-
  stale model with a CURRENT price, and these markets move ~2.5c in that time;
  the backtest pairs both at the same second and never pays it. So the backtest
  is optimistic by roughly one spread's worth. Reducing safe_lag needs a
  measurement of ingest write latency first (it exists so a second is not
  closed before its trades land). Follow-up, not done.

Session 28 — TWAP geometry is now the PRODUCTION model (MODEL_KIND=kde_twap).
  WHERE THE FEATURES LIVE. The aggregator computes and STORES two new columns
  in window_observations -- twap_sofar_bps (running mean VWAP vs anchor) and
  spot_vs_anchor_bps -- so training, live serving and the heatmap all read the
  same values (D6). anchor = trailing-60s TWAP of per-second VWAP at the
  window's first bar, i.e. the Chainlink stream's own quantity rebuilt from
  Binance; obs queries now read ANCHOR_LOOKBACK_SEC=62s before the window
  floor for it. req_bps and locked_frac are NOT stored: they are exact
  functions of those two plus time, computed by features.twap_geometry in one
  place. The live path, heatmap axis and grid now use price-vs-anchor, not
  delta-vs-open, so the "now" dot sits in the right cell.

  FILLING HISTORY. Raw trades only go back to 2026-08-11 (pruned), and the live
  aggregator skips any period with no trades -- rewinding its cursor would have
  left Jul 9-Aug 10 silently empty. New `polaris recompute-obs` rebuilds
  observations from crypto_1s_bars (kept forever), needs no cursor games, and is
  safe beside the live loop. Ran it over 85 days: 8.8M rows, 0 NULLs remaining.
  Verified on a 2-hour trial slice that every pre-existing column came back
  BIT-IDENTICAL (0 mismatches over 6,575 rows). Seam to watch: rows the OLD
  aggregator wrote in the ~30 min before its restart came out NULL and had to be
  recomputed separately.

  ACCEPTANCE TEST, equal budget: both production classes tuned with 40 Optuna
  trials on the SAME 500-window subsample, fit on the same training set,
  scored on the same 20k held-out later rows (post-Aug-7, real settlement):
                           old kde    kde_twap
    BTC Brier               .1563      .1391
    ETH Brier               .1550      .1373
    BTC gap to market,all  -.0382     -.0105  (t -5.6 -> -3.0, 312 windows)
    BTC gap, trade gate    -.0303     -.0116  (t -3.7 -> -2.1, 153 windows)
    ETH gap to market,all  -.0298     -.0096  (t -4.0 -> -2.4, 288 windows)
    ETH gap, trade gate    -.0063     -.0040  (t -0.6 both: inconclusive, 130 w)
  The new model is clearly better than the old one and still BEHIND the market.

  THE SURPRISE. The first search pinned twap_sofar and req against their
  ceilings on both symbols. With the bounds widened 25-50x the tuner flattens
  both completely, and:
                               BTC      ETH
    4D tuned                 .13870   .13456
    3D, req dropped          .13870   .13454
    2D, spot + locked only   .13869   .13453
  So the whole KDE gain is one idea: measure price against the SETTLEMENT
  ANCHOR instead of the window's opening VWAP, and stop spending dimensions on
  microstructure. twap_sofar/req do carry information (the GBM uses them), but
  a product kernel at this sample size cannot turn it into accuracy.
  Defaults: geometric mean of the two optima (65 / 3000 / 0.157 / 1.0); one
  shared set costs BTC 0.0005 and ETH nothing.

  PRECISION NOTE. load_dataset's window is relative to now(), so each run draws
  a slightly different split: ETH's score at identical bandwidths moved .1345 ->
  .1382 between two runs. Within-run comparisons are paired and reliable;
  absolute Brier across runs wanders ~+/-0.003. Do not quote third-decimal
  differences between separate runs.

  ALSO: cache.py and calibrate.py now build through model/factory.py (both
  used to hard-code KDEModel); FeatureSet.subset() replaces two hand-kept field
  lists that would have silently dropped the new columns; the frontend
  Diagnostics panel renders whichever bandwidths the serving model has.

Session 27 — settlement is now the DEFAULT label, and the KDE benefits from
TWAP geometry more than the GBM does.
  (1) dataset.load_dataset now takes label_source, defaulting to "settlement":
  real Polymarket settlements only, windows without one dropped. Deliberately
  NOT coalesce -- silently backfilling a 12%-wrong proxy into the gaps is how
  this class of bug survives two months. "proxy" and "coalesce" stay available
  but must be asked for and both log a warning naming what they did. Every
  consumer (backtest, calibrate, the API) reaches labels through this one
  function, so it is the single chokepoint. Cost of dropping unsettled windows:
  essentially nothing -- 18,336 BTC / 9,566 ETH windows remain over 70 days.

  (2) prune.py had NO policy for chainlink_twap (added the day before), so it
  fell through to the unlisted default. Marked never=True: it is strictly worse
  to lose than the order book, because Chainlink's own history API is paid with
  no free tier, so a deleted second cannot be bought back at any price.
  poly_resolutions and poly_trades were also unlisted; both now registered.

  (3) THE KDE TEST. Everything in sessions 24-26 was measured on LightGBM, but
  production runs a product-kernel KDE, and a GBM result does not transfer: a
  KDE multiplies one kernel per dimension, so n_eff decays roughly
  geometrically with dimensionality. TWAP geometry is interesting there
  precisely because it REPLACES seven columns with four.
  Post-Aug-7, equal subsample budget for every KDE row:
                              dims   KDE BTC / ETH      GBM BTC / ETH
    BASE (production)            7   .18111 / .18678    .16173 / .15970
    TWAP geometry                4   .15543 / .15268    .14441 / .13898
    BASE + TWAP                 11   .17210 / .17608    .14488 / .13990
  The KDE GAINS MORE than the GBM: .026/.034 vs .017/.021. And adding BASE back
  on top makes the KDE markedly WORSE (.1554 -> .1721) while leaving the GBM
  flat -- the dimensionality penalty, exactly as predicted. Seven largely
  uninformative dimensions were actively hurting the production estimator.
  CAVEAT, do not quote the KDE/GBM columns against each other: the KDE is
  subsampled to 40k training rows (it is O(n_train*n_test)) while the GBM uses
  all of it. The KDE-vs-KDE comparison across feature sets is fair (identical
  budget per row); the KDE-vs-GBM comparison is not, and is unfair to the KDE.

  IMPLICATION FOR PRODUCTION: the production KDE should swap its six features
  for the four TWAP-geometry ones. That is a real, non-Chainlink improvement to
  the shipped model, and it makes the estimator SIMPLER, not more complex. Not
  yet done -- Bandwidths, Query and the kernel assignment in bayesian/ are all
  hard-wired to the current six.

Session 26 — the clean regime triples the TWAP gain, and explains the BTC/ETH
puzzle. Chainlink backfill is PAID ONLY.
  Re-ran session 24 on post-2026-08-07 windows only (the Chainlink regime).
  BLOCK-CV BRIER, BTC:
                            mixed history        post-Aug-7 only
    BASE (production 6)     .16563 / .21578      .16173 / .21515
    TWAP geometry           .16030 / .21205      .14441 / .20359
  BTC's TWAP gain TRIPLES on the clean regime: .0053 -> .0173 (all seconds).
  ETH is unchanged (.0206) because ETH data only begins 2026-08-07 -- it was
  ALREADY clean.
  That resolves the session-24 anomaly I logged as unexplained ("why is ETH's
  gain 4x BTC's?"). It was never a symbol difference. BTC's estimate was
  diluted by 8,300 pre-switch windows whose label followed a DIFFERENT rule, on
  which TWAP geometry is much less relevant. Both symbols now agree at
  .017-.021, which is the coherent picture.

  VS THE MARKET, training restricted to the clean regime, graded on settlement:
                          BTC all    BTC gate    ETH all    ETH gate
    production           -0.02603   -0.01883   -0.03486   -0.02158
    TWAP geometry        -0.00889   -0.00723   -0.01316   -0.00577
    ALL                  -0.00764   -0.00654   -0.01170   -0.00494
  The deficit has fallen ~70-77% from where session 23 left it (-0.021/-0.035).
  It is NOT closed: t still runs -3.7 to -4.5. The market still forecasts
  better. But the trajectory is that every correction -- right target, right
  regime -- has cut the gap substantially, and the remaining ~0.005-0.008 is
  plausibly the last structural error: we still reconstruct the TWAP and its
  anchor from BINANCE rather than reading the Chainlink series the contract
  settles on. chainlink-collect (session 25) is what tests that, and it needs
  ~2 weeks of accumulation.

  CHAINLINK BACKFILL -- ANSWERED, and the answer is no free option:
    - Chainlink Data Streams is PAID ONLY. "All Data Streams subscriptions are
      paid. There is no free account tier." From $150/month, self-service at
      app.chain.link, credentials immediate. Its REST API does serve history:
      GET /api/v1/reports/page?feedID=&startTimestamp=&limit= (HMAC auth via
      Authorization / X-Authorization-Timestamp / X-Authorization-Signature-
      SHA256). Retention depth is undocumented -- would need to be measured.
    - pmdata.dev has a real free tier (600 downloads then 50/month) but
      CHAINLINK HISTORY IS NOT IN IT: it sits in Plus at $49/month, personal
      use only. Coverage starts 2026-08-01, which happens to span the entire
      post-switch regime.
    - Free path: forward collection only, which is already running.
  RECOMMENDATION: do not pay yet. The remaining gap is ~0.006 and closing it
  entirely buys PARITY, not edge -- and every previous fix in this project has
  moved us toward the market, never past it. Let the collector run two weeks and
  test against the real series first. If that closes the gap, THEN $49 for six
  weeks of history to enlarge the training set is a sensible follow-on.

Session 25 — the resolution source CHANGED on 2026-08-07, and the Chainlink
feed is free to ingest (forward-only).
  Polymarket's own docs (docs.polymarket.com/market-data/chainlink-twap) plus a
  third-party note that Up/Down markets "use Chainlink TWAP as their resolution
  source after August 7, 2026" prompted a check for a structural break. There
  is one, and it is unmistakable:
      BTC disagreement by week: 3.4 4.6 4.8 4.5 | 7.6 | 10.8 9.8 10.3 14.8 13.2
                                 <-- before -->  ^Aug7  <-- after -->
      BEFORE 2026-08-07:  n=8341  4.2% disagreement
      ON/AFTER:           n=9647 12.0%
  So our Binance endpoint label was a GOOD proxy (95.8%) under the old regime
  and is a poor one (88%) now. Two consequences:
    - the 8.1% full-history figure is a MIXTURE of two regimes and understates
      the current error; 12.0% is the number that matters;
    - training on all 70 days mixes two different labelling regimes. The clean
      post-switch set is ~9.6K BTC / ~9.2K ETH windows, roughly half what the
      session-24 runs used.
  (Disagreement is still drifting up within the new regime -- 10.8 -> 14.8 --
  which is unexplained. Could be volatility regime, could be further drift.)

  FREE CHAINLINK INGESTION -- VERIFIED WORKING, no credentials:
      wss://ws-live-data.polymarket.com        (Polymarket RTDS relay)
      subscribe: {"action":"subscribe","subscriptions":[
        {"topic":"crypto_prices_twap_sixty","type":"update",
         "filters":"{\"symbol\":\"btc/usd\"}"}]}
  Measured behaviour, which does NOT match the docs:
    - the first frame is an empty string; skip it before json.loads
    - payload.data is an ARRAY of ~58 per-second points (ms timestamps,
      1.0 pts/sec) -- i.e. a SNAPSHOT of the trailing minute, which the docs
      explicitly deny exists ("no snapshot, history, or replay")
    - after that one batch it pushes NOTHING. type="subscribe"/"snapshot"
      return 0 points; only type="update" works.
    - therefore the ingestion pattern is RECONNECT-POLL, not listen: three
      snapshots 30s apart gave 130 unique seconds over a 131s span with two
      2-second gaps. Polling every ~30s gives ~2x redundant per-second cover.
  Direct Chainlink (wss://ws.dataengine.chain.link, https://api.dataengine.
  chain.link) needs CHAINLINK_CLIENT_ID/SECRET and clock sync within 5s; its
  REST API does serve HISTORICAL reports, which the free relay does not.

  THE BACKFILL PROBLEM: the relay is forward-only. Settlements (labels) are
  already free from Gamma, but the per-second TWAP SERIES -- needed for the
  session-24 features, which are the half of the gap we can actually attack --
  cannot be backfilled for free. Paid options: pmdata.dev sells parquet from
  2026-08-01; Chainlink's own REST API with credentials.
  So every hour not collecting is an hour permanently lost, exactly like the
  order book was. That is the argument for standing a collector up promptly
  rather than after the next round of analysis.

Session 24 — half the deficit was STRUCTURAL: we were predicting the wrong
quantity, not merely predicting it badly.
  Follows directly from session 23.5's reading of the resolution rule. The
  contract pays on TWAP(window) >= anchor. A TWAP is PARTIALLY OBSERVED as the
  window runs: by t seconds in, t/300 of the final average is already fixed.
  The production feature set ignores this and predicts "end price vs start
  price" -- a near-martingale, and a strictly harder target than the one being
  traded. That is why the market pulled away from us late in the window.

  Four features, all from data already on disk (model/featurelab.py):
    twap_sofar_bps     partial TWAP so far vs the anchor
    req_bps            what the REMAINING seconds must average for the sign to
                       flip -- the binary's moneyness:
                         TWAP>=A  <=>  mean(remaining) >= (W*A - S_t)/(W - t)
    locked_frac        fraction of the final average already determined
    spot_vs_anchor_bps current price vs anchor

  BLOCK-CV BRIER ON REAL SETTLEMENT (18,057 BTC / 9,160 ETH windows):
                              BTC all   BTC gate   ETH all   ETH gate
    BASE (production 6)        .16563    .21578     .15962    .21321
    BASE + rv_60s + vwap_dev   .16460    .21492     .15875    .21269
    TWAP geometry ALONE        .16030    .21205     .13905    .19797
    BASE + TWAP geometry       .16072    .21289     .13987    .19968
  FOUR mechanical features beat the six production features PLUS every
  microstructure candidate -- and adding BASE on top makes it slightly WORSE,
  so the production features are largely noise for this target. The ETH gain
  (.0206) is more than twenty times the entire microstructure screen (.0009).

  VS THE MARKET, graded on real settlement (out-of-time, book subset):
                       BTC all    BTC gate    ETH all    ETH gate
    production        -0.02582   -0.01769   -0.03503   -0.02172
    TWAP geometry     -0.01264   -0.00938   -0.01317   -0.00587
  The gap HALVES on BTC and falls by two thirds on ETH -- but stays negative
  and significant (t = -4.2 to -7.6). The market still forecasts better.

  THE DIAGNOSIS IS NOW CLEAN, and splits in two:
    ~half the deficit: we were predicting the wrong quantity. Fixed, from data
      we already had.
    ~half the deficit: we compute the right quantity off the WRONG PRICE
      SERIES. The anchor and the running TWAP are reconstructed from Binance;
      the market watches the actual Chainlink TWAP-60s stream. This is the same
      residual that caps every Binance label rule at ~92%.
  So ingesting Chainlink is no longer a hope -- it is a quantified bet with a
  known prize of roughly 0.010-0.013 Brier.

  DO NOT MISREAD THIS AS PROGRESS TOWARD PROFIT. Closing the remaining gap
  entirely buys PARITY with the market, not an edge. Trading needs an edge on
  top of parity, and then that edge must clear the ~1.1c spread. The verdict is
  unchanged: not supported.

Session 23.5 — the actual resolution rule, read from the market description:
  Polymarket states it outright (Gamma `description` / `resolutionSource`):
    "resolve to Up if the TIME-WEIGHTED AVERAGE PRICE (TWAP) of Bitcoin,
     generated by Chainlink, of the time range specified in the title is
     greater than or equal to THE PRICE AT THE BEGINNING of that range"
    source: https://data.chain.link/streams/btc-usd-twap-60s-streams
  So our label was wrong on TWO independent axes: wrong SOURCE (Chainlink
  BTC/USD TWAP-60s stream, not Binance spot) and wrong FUNCTIONAL FORM (an
  average over the range, not an endpoint comparison).
  WHITEPAPER §2.2 asserted the exact opposite -- "Chainlink resolves on a
  point-in-time price comparison (start vs. end), not an average". Backwards.
  Nobody had read the market description.

  FULL-HISTORY DISAGREEMENT, now that poly_resolutions is populated
  (40,315 settlements stored, 20,158 BTC + 20,157 ETH):
      BTC 1407/17403 = 8.1%      ETH 908/8514 = 10.7%
  (The 14.6% quoted in session 23 came from a recent 9-day slice that happened
  to be unusually flat. 8-11% is the honest full-history figure. The SHAPE is
  unchanged: 27.9% disagreement on windows moving <1bp, ~0% above 10bp.)

  LABEL-RULE BAKE-OFF against real settlement, identical windows:
      BTC full 70d:   current(last_close vs prev_close) 92.5% | 60s-TWAP ends 92.2%
      BTC matched:    current 90.9% | 60s-TWAP endpoints 94.7%
      ETH matched:    current 89.6% | 60s-TWAP endpoints 98.0%
  "60s-TWAP endpoints" = trailing-60s TWAP at window END vs at window START,
  i.e. the Chainlink stream's own quantity reconstructed from 1s bars. It wins
  clearly wherever per-second COVERAGE IS DENSE (the live-ingest period) and
  not at all over the full 70 days -- consistent with the 60s TWAP needing
  complete per-second data, which backfilled history does not have. Naive
  TWAP-over-the-whole-window vs spot start scores far worse (82.7-85.4%),
  because that statistic is much closer to zero and therefore flips on tiny
  feed differences.

  DOES BETTER LABELLING RESCUE THE RESULT? NO, and this is settled rather than
  speculative: the model was retrained directly on ACTUAL SETTLEMENT labels
  over 15,808 windows -- the ceiling any proxy could reach -- and the gap to
  the market barely moved (-0.0204 -> -0.0201, t=-5.5). A better proxy label
  cannot beat training on the truth itself. So relabelling is worth doing for
  future work (it recovers ~8% of history that settlement alone cannot cover,
  and the feature set was selected against the wrong target) but it does NOT
  change the verdict.

  THE REAL FIX, if this is ever resumed: ingest the Chainlink BTC/USD/ETH-USD
  TWAP-60s data streams directly and label from the same series the market
  settles on. Binance will not get there -- no Binance-derived rule exceeds
  ~92% over the full history, and the market prices the Chainlink series, which
  is plausibly WHY it forecasts better than a Binance-fed model.

Session 22.5 — the production GBM is over-regularized (incidental finding):
  num_leaves=7 / min_child_samples=20000 were tuned against a SINGLE held-out
  day. Under proper walk-forward block CV, num_leaves=31 / min_child=5000 beats
  them outright on the BASE features -- BTC 0.16502 vs 0.16609, ETH 0.15705 vs
  0.15934 -- and roughly DOUBLES the measured value of any added feature, since
  a 7-leaf tree has no capacity left to spend on a new column. Screening at the
  tight setting understates every candidate. (Measured on the old label, so the
  absolute numbers move; the relative capacity finding should survive.)
  New module: model/featurelab.py -- 15 candidate microstructure features,
  greedy forward selection on block-CV Brier. BTC and ETH independently picked
  the SAME two: rv_60s (stddev of 1s log RETURNS -- the existing volatility_10s
  is stddev of price LEVELS, which conflates trend with vol) and vwap_dev
  (close-minus-vwap within the second). Must be re-run on real labels.

Session 22 — the live edge is not day-robust; it rests on one day:
  Ran the concentration/robustness check that had been skipped twice.
  At the real ask, n=360 (sample grew from 331), total excess +8.6, z=+0.98.
  BY DAY:  Sep10 n=96 excess -0.22 (z=-0.05) | Sep11 n=248 +7.69 (z=+1.06)
           Sep12 n=16 +1.14 (z=+0.60)
  LEAVE-ONE-DAY-OUT: removing Sep 11 drops the excess to +0.92 (z=+0.19).
  => the ENTIRE effect rests on a single day, and that day is not itself
  significant. No day-level robustness. The live test currently shows nothing
  distinguishable from noise.
  METRIC NOTE: the "top-k trades as % of total excess" statistic I reached for
  first is USELESS when the net is near zero -- it is a small difference of
  large gross flows, so top-10 showed 97% and top-25 showed 227%. Use
  leave-one-block-out, not share-of-net, for robustness on near-zero sums.
  CONSEQUENCE for the maker-execution idea: it was pitched as rescuing an edge
  that exists but cannot clear the spread. That framing presumed an edge. It is
  still a structurally interesting question (posting rather than crossing is a
  ~3%-of-capital swing) but it would be improving the economics of a signal
  that is not established.
  SURVIVING RESULT, unchanged by all of this: the model beats the market as a
  FORECASTER (Brier .164 vs .186, holding in every edge bucket and on the gated
  subset). It predicts; it has not been shown to profit.

Session 21 — recovered Sep 10-11 Binance data; two cursor gotchas worth writing down:
  Laptop sleep had left 62 gaps / 1,369 min of missing BTC trades since Sep 11.
  Binance Vision had published Sep 11, so backfill --days 2 recovered 10.1M
  trades across both symbols (dupes correctly skipped). All remaining gaps
  (576 min) are on Sep 12 = today, not yet published.
  GOTCHA 1 -- the aggregator caches cursors IN MEMORY at startup
  (aggregate.run reads them once into a dict), so rewinding aggregator_cursors
  in the DB does nothing to a running container. It must be restarted.
  GOTCHA 2 -- worse, and it bit first: updating the cursor while aggregate is
  RUNNING gets silently clobbered, because the loop writes its in-memory cursor
  back every cycle. Correct order is STOP -> rewind -> START. Doing it the
  other way looked like it worked and quietly didn't.
  Also: rewinding only to the earliest GAP (Sep 11 05:00) left Sep 10 at
  52/288 windows even though the backfill had refilled 3.08M Sep 10 trades --
  refilled data BEFORE the cursor is invisible to the aggregator. Rewound to
  Sep 09 23:59 and re-ran; Sep 10 and Sep 11 are now both 288/288 windows.
  NOT RECOVERABLE, and this is the asymmetry that matters: poly_book_snapshots
  (~1,188 min lost) and paper_signals (~1,206 min) are forward-only. The
  Binance side came back in full; the two instruments the execution-cost
  question actually depends on did not.

Session 20 — pricing at the REAL ASK kills most of the live edge (z 1.83 -> 0.66):
  Session 19's ~2 sigma used the CLOB MIDPOINT. You cannot buy at the midpoint.
  Re-priced every gated live trade at the actual best_ask from
  poly_book_snapshots, matched to the nearest book sample within +-10s
  (331/332 trades matched, 100%):
      at midpoint  n=331  wins 142 vs implied 126.8  excess +15.2  z=+1.83  RoC +12.0%
      at REAL ask  n=331  wins 142 vs implied 136.4  excess  +5.6  z=+0.66  RoC  +4.1%
      by symbol at ask: BTC +1.8 (z=+0.29, RoC +2.5%), ETH +3.8 (z=+0.65, +5.7%)
  MY z~1.75 ESTIMATE WAS WRONG, and instructively so: I deducted the spread
  from P&L only. Paying the ask also RAISES THE BENCHMARK -- implied wins went
  126.8 -> 136.4 (+9.6) while our win count stayed at 142. Execution cost moves
  the bar you must clear, not just the payoff. Worth remembering for any future
  cost adjustment.
  Also corrected: my speculation that spreads are wider at our trade moments
  came from a 68-trade exact-match subset. With full matching, ask-mid = 0.54c,
  consistent with the 1.03c spread measured in session 15. Spreads are NOT
  wider when we trade.
  Both symbols still point positive and agree closely, so this is not one
  asset driving a fluke -- the effect is just small and unproven.

Session 19 — the 43% win rate was NOT a failure; live edge is ~2 sigma positive:
  Chased an apparent paradox -- our model beats the market as a forecaster
  (Brier .164 vs .186) yet wins only 43.2% of its live trades. Two hypotheses
  (selecting on our own model's errors; tail miscalibration) were both REFUTED:
  our Brier beats the market's in EVERY edge bucket and the advantage GROWS
  with edge size (.2156 vs .4473 at edge>0.35), and it still beats it on the
  gated subset specifically (.2190 vs .2267, n=2046).
  THE PARADOX WAS MY FRAMING. Win rate is the wrong success metric here: we
  systematically buy the CHEAPER side (avg entry 0.383), so a sub-50% win rate
  is what success looks like. The right question is whether we win more often
  than the PRICE implied.
  LIVE FORWARD TEST, REAL CLOB PRICES (the most trustworthy measurement in the
  project -- no backfill, no forward-fill, no proxy):
      331 trades, avg entry 0.383
      actual wins        143   (43.2%)
      market-implied     126.8 (38.3%)
      excess            +16.2   sd 8.3   z = +1.95
      P&L +16.22 on 126.8 staked -> RoC +12.8%
  ~2 sigma: suggestive, NOT conclusive (one-sided p ~ .026). Caveats: ~2 days
  of partial-uptime data, BTC+ETH combined, and priced at the MIDPOINT -- real
  buys fill at the ask (~0.5c worse), which drops z to ~1.75.
  CONSEQUENCE: more live data is now the highest-value thing in the project,
  because it is the one measurement that can settle this and it accrues for
  free. At ~165 trades/day of uptime, a few weeks at full uptime gives ~2000+
  trades -- enough to resolve 2 sigma either way.

Session 18 — trade-derived prices ALSO fail; project is now data-limited:
  Backfilled 14 days of fills (6,044,280 over 4,031 windows, 1 failed) and ran
  a 15-gate x 2-threshold sweep on --price-source trades. It FAILED the check
  set in advance: trade rate 94.6% of windows for [0,300) and 89.3% for [0,60),
  vs 6.5% on real book quotes. ~15x more signals = prices still manufacturing
  edge. Tell: [240,300) showed +132% RoC on a 28.2% win rate at avg entry 0.121
  -- i.e. the market supposedly wildly miscalibrated seconds before resolution.
  MECHANISM, measured not inferred (31,480 overlapping priced seconds):
    overall error is UNBIASED (mean +0.0006) but mean|err| 0.0388;
    on the 6.1% of seconds where the trade price sits >=8c BELOW the book,
    trade px averages 0.444 against a real book price of 0.601 -- a 15.7c
    illusion. min_edge does not sample the error on average: it SELECTS
    exactly the seconds where noise made the contract look cheap, then P&L
    credits the full payoff on that fake-cheap entry. Late window is worse
    (mean|err| 0.0784, sd 0.1527), which is why [240,300) looked absurd.
    => unbiased noise still produces biased RESULTS under a selection filter.
  MY 0-15s HYPOTHESIS IS NOT SUPPORTED. [0,15) gave z=+1.8, same as everything
  else early; best cell was [45,60) z=+2.8, unremarkable across 30 configs
  tested. The incumbent 45-90s gate was not obviously wrong after all.
  CREDIBLE RESULT UNCHANGED: book-quote backtest, 52.5% win on 101 trades,
  6.5% of windows, z=0.50 -- not significant.
  STRATEGIC CONSEQUENCE: the project is DATA-limited, not model-limited. More
  features / retuning cannot be evaluated, because the only trustworthy price
  source (poly_book_snapshots) has ~5 days of coverage, and everything else
  has now produced two false positives. Options: (a) let book-collect and
  paper-trade accrue for weeks, then evaluate on real prices only; (b) trade
  only LARGE edges (min_edge >= 0.15-0.20) where a 4c price error matters less
  -- testable on existing book data, but frequency and power collapse.

Session 17 — the edge was a stale-price artifact; real-quote backtest is ~coin flip:
  CHASING the live/backtest divergence found the cause in the BACKTEST.
  poly_historical_prices comes from /prices-history = 1-MINUTE resolution
  (~5 real points per 300s window, verified: fidelity=1 returns the same 5).
  The backfill forward-filled that to ~300 per-second rows, so ~98% of the
  prices every backtest traded against were fabricated. Measured vs 158k live
  book snapshots: mean |book mid - ffill| = 0.137 overall, 0.129 inside the
  45-90s gate; real price moves ~27.6x/window vs ~4.8 distinct filled values.
  => "edge" was a fresh Binance signal minus a STALE quote. Not tradeable.
  NOT a model failure: live vs backtest p_up correlate 0.9986.
  FIX (user was right, my "no historical data" call was wrong -- I concluded
  from one endpoint): data-api.polymarket.com/trades serves individual FILLS at
  second resolution, 60+ days back, ~1000 fills / 51-201 distinct seconds per
  window. Built poly_trades + backfill-poly-trades. Validated before trusting:
  trade-derived price is 3.5c from the book in the gate (vs 12.9c) and
  UNBIASED (signed -0.0014). A side-aware estimator (BUY~ask/SELL~bid) gave no
  improvement -- tested and rejected.
  THE HONEST NUMBER -- backtest on REAL ORDER-BOOK quotes (--price-source book,
  7 test days, the only unimpeachable source):
      1,563 test windows -> 101 trades (6.5% of windows)
      win 52.5% +- 5.0pp  (z=0.50 vs chance -- NOT significant)
      P&L +9.52, +0.094/trade (t~1.8, suggestive at best)
  vs retracted ffill: 62.7% win on 1,271 trades, ~99% of windows traded.
  THE 99% TRADE RATE WAS THE TELL. With real quotes only 6.5% of windows have
  a signal. A strategy finding 5%+ edge in nearly every window was finding
  broken prices, not alpha -- exactly the anomaly flagged in the session-10
  min_edge sweep, now proven.
  THREE independent measurements agree the edge is weak/absent (book 52.5%,
  live paper trade 43%, and the 6.5% trade rate). Only the fabricated-price
  backtest disagreed.
  WHAT STILL STANDS: predictive skill, scored against OUTCOMES not prices --
  block-CV Brier 0.1700 vs 0.25 base. The model does forecast direction better
  than chance. What is NOT established is that this converts to tradeable edge
  after real execution prices.
  ALSO FIXED this session: book() and trades() both swallowed 429s and returned
  empty, so rate-limited runs looked successful (book-collect was at 50%
  cadence; the first trade backfill silently lost 2/3 of its windows). Both now
  retry with backoff and report failures. Third instance of this bug class.

Session 16 — strict out-of-time test PASSED; live/backtest config mismatch found:
  DATA: filled the Aug31-Sep9 BTC gap, backfilled 30d of ETH (309K -> 74.3M
  trades, obs 13K -> 1.76M), aggregated both to present. Added the forward
  paper-trade recorder and moved the long-running collectors into Docker.
  STRICT OUT-OF-TIME TEST (the point of the exercise): test days restricted to
  2026-08-31..09-10 -- nine days that DID NOT EXIST when the bandwidths, the
  45-90s gate or the 60.9% were chosen. Rolling 5-day training, identical
  config to session 14.
      1,271 trades  62.7% win  +215.1 P&L  Brier .1619
      at measured 1c cost: 1,246 trades  62.8%  +205.0
      reference (in-period, 52d, same config): 60.9%, Brier .1662
  => NO DECAY. 62.7% on 1,271 trades is ~9.4 sigma over 50%; the 1.8pp gap vs
  in-period is inside the ~1.4pp standard error. Strongest evidence yet against
  "period-specific artifact". Does NOT address single-asset/single-venue, and
  9 days could still be a favourable regime.
  LIVE/BACKTEST CONFIG MISMATCH (found while reviewing the paper trade) --
  the deployed model is NOT the validated one:
      live serves training_window_days=14; every backtest used train_days=5
      live compares to CLOB MIDPOINT; backtests compare to LAST-TRADE prices
  Corroborating: live n_eff runs 3,800-17,000 vs hundreds-to-low-thousands in
  the backtest -- the signature of averaging over a much larger neighbourhood,
  which pulls p_up toward the base rate. Consistent with the observed live
  behaviour (p_up hovering near 0.5 while the market moved decisively, and all
  gated trades taking the NO side).
  First 18 min of paper trading: 5 resolved would-be trades, 0 wins, all NO,
  all windows resolved UP. Statistically meaningless (5 CONSECUTIVE windows in
  one trending stretch, not independent draws) but the all-NO pattern plus the
  config mismatch are why the mismatch should be reconciled before the forward
  test is compared to any backtest number.
  BOTH MISMATCHES SUBSEQUENTLY RULED OUT -- the hypothesis was wrong:
    (a) price reference: mid vs last_trade measured at the same instant from
        poly_book_snapshots (45-90s gate, n=5,163/side): mean gap +-0.0001,
        p50 +-0.005 (half a tick), mid higher 50.1% of the time. Unbiased
        symmetric noise, not a systematic bias -- cannot explain all-NO.
    (b) training window: reran the identical out-of-time test at train_days=14
        (what live serves) vs 5 (what was validated):
            14d: 1,258 trades  62.6% win  +220.6  Brier .1606
             5d: 1,271 trades  62.7% win  +215.1  Brier .1619
        Identical within noise; 14d marginally BETTER. The flatter-p_up /
        higher-n_eff mechanism I predicted would degrade performance does not.
  So the deployed config is fine, and the live underperformance is best
  explained by sample size: a couple dozen trades over consecutive correlated
  windows in one regime, against a 1,271-trade backtest. Let it accrue.
  SILVER LINING: the strategy is insensitive to the training window (5d ~ 14d),
  which is a robustness result worth having.

Session 15 — execution cost MEASURED (the last big assumption closed):
  The book collector built at the end of session 14 ran 4 days unattended:
  121,094 snapshots, 768 windows, 2 symbols (2026-09-06..09-10), 18,755 rows
  inside the 45-90s trade gate. Survived a Postgres restart mid-run.
  MEASURED (BTCUSDT, 45-90s gate, n=9,359):
    touch spread          mean 1.03c  p50 1.00c  p90 1.00c
    best_ask - last_trade mean 0.51c  p50 1.00c  p90 1.00c   <- the correction
  ask-last is bimodal (last trade sits at bid OR ask, so ~1c or ~0), which is
  why the mean is ~half a tick while the median is a full tick.
  SLIPPAGE vs touch, all 100% fillable:
    $1 0.00c | $10 0.03c | $100 0.35c (p90 0.98c) | $1000 3.05c (p90 5.79c)
  => total taker cost ~0.5c at $1-10, ~0.9c at $100, ~3.6c at $1000.
  FLAT ACROSS THE WINDOW: 1.03c from 0-30s through 210-240s, widening only in
  the last minute (1.21c at 270-300s). No hidden early-window cost penalty in
  the region the strategy actually trades. Ask depth declines 43.9K -> 22K
  contracts but stays ample. ETHUSDT agrees (1.08c / 0.54c, n=9,402).
  1c is the TICK FLOOR, so these markets sit at the tightest spread the venue
  permits essentially always -- favourable venue structure, not an assumption.
  VERDICT: mapped onto the existing sweep, measured cost (~0.5-0.9c up to $100)
  sits between the 0c and 2c rows -> edge substantially intact (+1755 -> +1548).
  Even $1000/trade (~3.6c mean, ~6.8c p90) stays clearly positive (+1453/+1066).
  WHAT IT DOESN'T SETTLE: book data is September, the backtest is July-August,
  so the mapping is distributional, not trade-by-trade (a historical book does
  not exist, so this is not fixable). And it measures the VISIBLE book at a
  sampled instant -- queue position, latency, losing the touch to someone
  faster, adverse selection, and our own market impact are all still
  unmodelled and all push cost upward. Maker/limit execution untested.

Session 14 — GBM challenger (model-class control) + refreshed spread sweep:
  WHY A CHALLENGER: one model's backtest can't separate "the signal is real"
  from "this estimator's inductive biases happen to fit this data". So: same
  features, same walk-forward days, same gates, same market prices — swap only
  the estimator. Built model/factory.py (build_model(cfg, kind)) and made
  backtest.py model-agnostic; `backtest --model {kde,gbm}`.
  RESULT — statistically indistinguishable (full 52-day walk-forward, cadence
  10, train_days 5, min_edge .05, gate 45-90s):
      KDE  brier .1662  12,571 trades  60.91% win  +1754.8  (.1396/trade)
      GBM  brier .1666  13,107 trades  61.22% win  +1737.4  (.1326/trade)
  0.0004 Brier gap; 0.31pp win gap vs a ~0.44pp per-arm standard error. Two
  very different function approximators (product-kernel conditional-
  independence vs axis-aligned boosted trees modelling interactions freely)
  land in the same place.
  READ: (a) the KDE's independence assumption costs nothing measurable;
  (b) more usefully — both are probably near the CEILING of what these six
  features support, so further gains want better features/data, not a better
  estimator. markov/ parked on this basis.
  GBM NEEDED EXTREME REGULARIZATION (num_leaves 7, min_child_samples 20000,
  lr .02) because within a window ~300 per-second rows are autocorrelated and
  share ONE label -> effective N is windows (~14K), not rows (~3.7M). At stock
  settings (63 leaves / mcs 200) it scored brier .2295, barely under the .25
  base rate, emitting saturated 0/1 probabilities. NOTE: the first Optuna
  search range I wrote capped min_child_samples at 2000 and could not have
  reached the good basin — widened to 100000.
  HONEST ASYMMETRY: GBM has no n_eff analogue, so its confidence gate is
  inactive. It took 536 MORE trades and earned LESS in total (+1737 vs +1755)
  -> those marginal trades were net-negative, i.e. the KDE's n_eff gate is
  doing real filtering work, and it's exactly the quantity the GBM can't
  reproduce. Also no credible interval (lo=hi=p_up, not an invented one).
  REGRESSION CHECK: the KDE arm reproduced session 12's published numbers
  EXACTLY (12,571 trades / 60.91% / +1754.8), confirming the factory
  indirection changed nothing.
  SPREAD SWEEP REFRESHED (was open item #1, at tuned bw on the full 52 days,
  both models — candidates cached from the walk-forward, so re-gating each
  cost level was ~free):
      cost   KDE trades / win / P&L        GBM trades / win / P&L
      0c     12571 60.9% +1755             13107 61.2% +1737
      2c     11690 61.7% +1548             12511 61.8% +1540
      5c     10247 63.1% +1257             11365 62.8% +1239
      7c      9212 64.0% +1066             10486 63.4% +1032
  Survives 7c one-sided (14c round-trip) with ~60% of P&L intact. Win rate
  RISES with assumed cost because netting cost out of edge before gating
  raises the effective threshold and selects a smaller, higher-conviction
  subset — not because cost helps (total P&L falls throughout).
  DEPENDENCY FIX: mlx had been added to base dependencies by an earlier
  `uv add`, contradicting the mlx extra's own comment that it must NOT be a
  base dep (breaks the Linux Docker image). Removed from base, kept as the
  extra. lightgbm added as a new `gbm` extra by the same reasoning (the
  container never trains a challenger). macOS also needs `brew install libomp`.
  NOT DONE: calibrate.py and api/cache.py still construct KDEModel directly
  rather than via the factory — harmless duplication, deliberately not
  touched mid-experiment.

Session 13 — repo cleanup + code review + a real data-corruption bug fix:
  CLEANUP: deleted binance_ingestion/, bar_aggregator/, backend/,
  validate_idea.py (fully superseded by polaris/ since session 8; verified
  dead first — no polaris/ module imported from them, and gpu_loocv.py's old
  leave-one-out masking was diffed against the new block-CV masking to
  confirm the migration wasn't just a port but a genuine fix — see below).
  Dropped confluent-kafka, matplotlib, and the pointless "asyncio" PyPI
  package (stdlib since Python 3.4) from pyproject.toml. This doc condensed
  from ~1200 lines of accumulated narrative + the fully-executed pre-refactor
  migration plan into this current-state-plus-changelog form.

  CODE REVIEW caught a real regression from session 11's obs_since -1s->-2s
  change: it also reaches ~2s into the PRECEDING window on every incremental
  cycle, pulling in exactly its last bar with none of that window's other
  ~299 bars in range. WINDOW_OBS_SQL computed that straggler as its own
  degenerate 1-row partition (wrong cum_volume/cum_imbalance/volatility_10s,
  and a spuriously-recomputed outcome comparing the bar to itself), and
  UPSERT_OBS_SQL overwrote the already-correct, already-finalized row for it.
  FIX: WINDOW_OBS_SQL gained min_window_start (the window floor of `since`)
  and filters the final SELECT to it — the extra lookback bar stays visible
  to the CTEs (so prev_close_price's LAG still works) but never becomes an
  output row. Verified against real data: reproduced the leak pre-fix
  (delta_bps=0.0, cum_imbalance=0.0 — visibly degenerate), confirmed 0 leaked
  rows post-fix, confirmed the LAG-based label still works correctly.
  RECOMPUTED to purge the corruption from stored data: reset the BTCUSDT
  cursor, re-ran `aggregate` end-to-end. Row count unchanged (3,706,841 — no
  data loss); 992 rows had corrupted delta_bps/cum_imbalance, 497 had a wrong
  outcome, all now corrected.
  IMPACT ON REPORTED NUMBERS: believed small, not re-verified numerically.
  The walk-forward backtests (56.4%/62.9%/60.9% win rates) use cadence-
  thinned query seconds (multiples of 5 or 10) that structurally never land
  on the corrupted ~t=299 second. Only the full-N baseline/tuned Brier calls
  (0.1871/0.1700) touched every row, at ~992/3.7M ≈ 0.03% contamination —
  likely noise-level. If this matters later (e.g. before trusting Brier to
  4 decimals for a paper/decision), re-run calibrate --backend mlx on the
  now-corrected data.
  Also fixed, same review: KDEModel's MLX cache now keyed by id(fs), not just
  presence (defends future callers that switch fs without an intervening
  fit()); calibrate.py's train_cadence guards against thinning the Step-1
  search set to empty (was an opaque Optuna crash); api/cache.py fetches
  load_dataset + load_current_window_obs concurrently instead of serially;
  backtest.py warns when multiple sweep flags are passed together instead of
  silently honoring only one.

```
