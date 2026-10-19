-- polaris/core/schema.sql
--
-- Consolidated DDL for the entire Polaris pipeline. Applied idempotently on
-- startup by core.db.ensure_schema(). Every statement uses IF NOT EXISTS, so
-- it is safe to run on every boot.
--
-- Pipeline (each table feeds the next):
--
--   crypto_trades          raw ticks (Binance WS live + Binance Vision backfill)
--       -> aggregate step 1
--   crypto_1s_bars         1-second OHLCV bars with taker-volume breakdown
--       -> aggregate step 2
--   window_observations    per-second features inside aligned 5-minute windows
--       -> model training / calibration
--
--   aggregator_cursors     high-water marks for incremental aggregation
--   poly_historical_prices dense Polymarket CLOB prices (calibration Step 2)
--
-- PRECISION: prices/quantities use NUMERIC(18,8) to match Binance exactly and
-- avoid float rounding in financial data. Downstream feature SQL casts to
-- FLOAT8 where statistical features make that appropriate.


-- ═══════════════════════════════════════════════════════════════════════════
-- crypto_trades — raw tick data (root of the pipeline)
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS crypto_trades (
    -- Surrogate key; gives deterministic ordering when trade_time ties
    -- (multiple trades in the same millisecond).
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    symbol      VARCHAR(20)    NOT NULL,
    -- Binance trade id, monotonic per symbol. With symbol it is the idempotency key.
    trade_id    BIGINT         NOT NULL,
    price       NUMERIC(18, 8) NOT NULL,
    quantity    NUMERIC(18, 8) NOT NULL,
    -- TRUE when the buyer was the maker (passive) => taker-initiated SELL (bearish flow).
    is_buyer_mm BOOLEAN        NOT NULL,
    trade_time  TIMESTAMPTZ    NOT NULL,   -- exchange execution time (UTC)
    ingested_at TIMESTAMPTZ    NOT NULL DEFAULT now()
);

-- "latest N trades for symbol X" — the dominant read pattern.
CREATE INDEX IF NOT EXISTS idx_crypto_trades_symbol_time
    ON crypto_trades (symbol, trade_time DESC);

-- Idempotency: rejects duplicate (symbol, trade_id) on retries/restarts.
-- Conflict target for ON CONFLICT DO NOTHING in both live ingest and backfill.
CREATE UNIQUE INDEX IF NOT EXISTS uq_crypto_trades_symbol_trade_id
    ON crypto_trades (symbol, trade_id);


-- ═══════════════════════════════════════════════════════════════════════════
-- crypto_1s_bars — 1-second OHLCV bars
-- ═══════════════════════════════════════════════════════════════════════════
-- vwap: Σ(price·qty)/Σ(qty), the reference price for delta/momentum/volatility.
-- mm_buy_volume: volume where the maker was the buyer (taker SELL volume).
-- mm_buy_volume / total_taker_volume = order-flow imbalance ratio.

CREATE TABLE IF NOT EXISTS crypto_1s_bars (
    symbol             VARCHAR(20)    NOT NULL,
    bucket_time        TIMESTAMPTZ    NOT NULL,
    open_price         DECIMAL(18, 8) NOT NULL,
    high_price         DECIMAL(18, 8) NOT NULL,
    low_price          DECIMAL(18, 8) NOT NULL,
    close_price        DECIMAL(18, 8) NOT NULL,
    volume             DECIMAL(18, 8) NOT NULL,
    trade_count        INTEGER        NOT NULL,
    vwap               DECIMAL(18, 8) NOT NULL,
    mm_buy_volume      DECIMAL(18, 8) NOT NULL DEFAULT 0,
    total_taker_volume DECIMAL(18, 8) NOT NULL DEFAULT 0,
    PRIMARY KEY (symbol, bucket_time)
);

CREATE INDEX IF NOT EXISTS idx_crypto_1s_bars_symbol_time
    ON crypto_1s_bars (symbol, bucket_time DESC);


-- ═══════════════════════════════════════════════════════════════════════════
-- window_observations — per-second features in aligned 5-minute windows
-- ═══════════════════════════════════════════════════════════════════════════
-- Windows align to fixed boundaries via date_bin (12:00, 12:05, ...).
-- cum_imbalance NULL when no trades yet; volatility_10s NULL for first 9s;
-- outcome NULL until the window completes (>= ~295s of data).

CREATE TABLE IF NOT EXISTS window_observations (
    symbol         VARCHAR(20) NOT NULL,
    window_start   TIMESTAMPTZ NOT NULL,
    bucket_time    TIMESTAMPTZ NOT NULL,
    time_elapsed_s INTEGER     NOT NULL,
    delta_bps      FLOAT       NOT NULL,
    cum_imbalance  FLOAT,
    cum_volume     FLOAT       NOT NULL,
    volatility_10s FLOAT,
    outcome        SMALLINT,
    twap_sofar_bps     FLOAT,
    spot_vs_anchor_bps FLOAT,
    PRIMARY KEY (symbol, bucket_time)
);

-- TWAP payoff geometry (added 2026-09-29). Polymarket settles on a TWAP over the
-- window vs a start anchor, so these describe the contract's actual state --
-- see WINDOW_OBS_SQL in polaris/aggregate/sql.py for the definitions. ALTERs so
-- existing databases pick the columns up through ensure_schema; history is
-- filled by `polaris recompute-obs`, which rebuilds from crypto_1s_bars because
-- raw trades older than 30 days are pruned.
ALTER TABLE window_observations ADD COLUMN IF NOT EXISTS twap_sofar_bps     FLOAT;
ALTER TABLE window_observations ADD COLUMN IF NOT EXISTS spot_vs_anchor_bps FLOAT;

CREATE INDEX IF NOT EXISTS idx_wo_symbol_window
    ON window_observations (symbol, window_start, time_elapsed_s);


-- ═══════════════════════════════════════════════════════════════════════════
-- aggregator_cursors — incremental aggregation high-water marks
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS aggregator_cursors (
    symbol    VARCHAR(20) PRIMARY KEY,
    last_time TIMESTAMPTZ NOT NULL
);


-- ═══════════════════════════════════════════════════════════════════════════
-- poly_historical_prices — dense (forward-filled) Polymarket CLOB prices
-- ═══════════════════════════════════════════════════════════════════════════
-- window_start is BIGINT epoch seconds (CLOB API native form). Both yes/no
-- prices stored: yes+no > 1.0 in practice (maker spread), and Step 2 needs
-- both to pick the profitable side.

CREATE TABLE IF NOT EXISTS poly_historical_prices (
    symbol          VARCHAR(20),
    window_start    BIGINT,
    elapsed_seconds INT,
    yes_price       FLOAT8,
    no_price        FLOAT8,
    PRIMARY KEY (symbol, window_start, elapsed_seconds)
);


-- ═══════════════════════════════════════════════════════════════════════════
-- poly_book_snapshots — sampled Polymarket CLOB order book (forward-collected)
-- ═══════════════════════════════════════════════════════════════════════════
-- WHY THIS EXISTS: every execution-cost number in the research so far is a
-- SWEPT ASSUMPTION, not a measurement, because Polymarket's public history
-- (/prices-history -> poly_historical_prices) returns last-TRADE prices only:
-- there is no historical order book to backfill. Spread can therefore only be
-- accumulated going forward, which is what the `book-collect` service does.
--
-- THE MEASUREMENT THAT MATTERS: the backtest fills at the historical
-- last-trade price, so the correction it needs is not the full bid/ask spread
-- but (best_ask - last_trade) -- the cost of crossing from where the tape
-- printed to where you could actually buy. /book returns both in one payload,
-- so the two are captured at the same instant and the gap is exact.
--
-- API TRAP (measured, not theoretical): /book's last_trade_price is
-- MARKET-level, not token-level. Querying the YES token and the NO token at
-- the same instant returns the SAME number, denominated in whichever side
-- traded most recently. Observed live: at one sample YES=0.04/0.05 and
-- NO=0.95/0.96 both reported last_trade_price=0.04; 25s later both reported
-- 0.97. Taken at face value that makes best_ask - last_trade = 0.96 - 0.04 =
-- 0.92 for NO -- garbage, and garbage that looks like a plausible number.
-- Because the pair is complementary, the correct per-side value is L or 1-L;
-- last_trade stores the one consistent with that side's own book, and
-- last_trade_raw preserves the API's value so the choice stays auditable.
--
-- eff_ask_N is the size-weighted average price paid walking the ask side for a
-- $N order (NULL when the book is too thin to fill it) -- the touch price is
-- only the right cost for an infinitesimal order, and the touch is often thin.
-- `levels` keeps the raw top-of-book so later analysis isn't limited to the
-- order sizes we happened to pick today.

CREATE TABLE IF NOT EXISTS poly_book_snapshots (
    symbol          VARCHAR(20) NOT NULL,
    window_start    BIGINT      NOT NULL,   -- epoch seconds, aligned to window_sec
    elapsed_seconds INT         NOT NULL,
    side            VARCHAR(3)  NOT NULL,   -- 'YES' | 'NO' (the token sampled)
    best_bid        FLOAT8,
    best_ask        FLOAT8,
    spread          FLOAT8,                 -- best_ask - best_bid
    mid             FLOAT8,
    last_trade      FLOAT8,                 -- NORMALIZED to this side (see below)
    last_trade_raw  FLOAT8,                 -- exactly as the API returned it
    eff_ask_1       FLOAT8,                 -- avg fill price walking asks for $1
    eff_ask_10      FLOAT8,
    eff_ask_100     FLOAT8,
    eff_ask_1000    FLOAT8,
    bid_depth       FLOAT8,                 -- total contracts resting on bids
    ask_depth       FLOAT8,
    levels          JSONB,                  -- {"bids":[[px,sz]...],"asks":[...]}
    captured_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, window_start, elapsed_seconds, side)
);

-- Analysis reads by elapsed second (how does cost vary across the window, and
-- specifically inside the 45-90s trade gate?).
CREATE INDEX IF NOT EXISTS idx_poly_book_symbol_elapsed
    ON poly_book_snapshots (symbol, elapsed_seconds);


-- ═══════════════════════════════════════════════════════════════════════════
-- paper_signals — forward paper-trade log (read-only research, no execution)
-- ═══════════════════════════════════════════════════════════════════════════
-- Every number in the research so far is a BACKTEST. This is the forward test:
-- the `paper-trade` service records what the LIVE system actually said, tick by
-- tick, so it can later be scored against realized outcomes. Nothing here
-- places an order; there is no execution path in this codebase.
--
-- WHY LOG EVERY EVALUATION, not just the trades that passed the gate: it makes
-- the forward test re-gateable offline at any min_edge / n_eff / trade window,
-- exactly as walk_forward_candidates -> _book lets the backtest re-sweep
-- without recomputing. Logging only fired trades would freeze today's
-- thresholds into the record permanently.
--
-- NO OUTCOME COLUMN ON PURPOSE: the aggregator already writes the resolved
-- label to window_observations.outcome once a window completes, so scoring is
-- a join on (symbol, window_start) at analysis time. Storing a second copy
-- here would be a denormalized field that could silently disagree with it.

CREATE TABLE IF NOT EXISTS paper_signals (
    symbol          VARCHAR(20) NOT NULL,
    window_start    BIGINT      NOT NULL,   -- epoch seconds (matches poly_*)
    elapsed_seconds INT         NOT NULL,   -- as the MODEL saw it, from the data
    p_up            FLOAT8,                 -- live model probability
    n_eff           FLOAT8,
    lower_95        FLOAT8,
    upper_95        FLOAT8,
    delta_bps       FLOAT8,
    yes_price       FLOAT8,                 -- live CLOB midpoint, YES
    no_price        FLOAT8,
    poly_spread     FLOAT8,
    edge            FLOAT8,                 -- best side's edge, per api/edge.py
    side            VARCHAR(3),             -- 'YES' | 'NO'
    in_window       BOOLEAN,                -- inside the configured trade window
    gates_passed    BOOLEAN,                -- edge AND n_eff AND time gates
    captured_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, window_start, elapsed_seconds)
);

-- Which estimator produced each signal (added 2026-09-29, when production moved
-- from the 7-feature "kde" to "kde_twap"). Without it, paper-report would blend
-- two different models into one meaningless live number. Rows written before
-- the column existed were all "kde"; they were tagged once by hand rather than
-- here, so a future NULL still means "unknown", never a guess.
ALTER TABLE paper_signals ADD COLUMN IF NOT EXISTS model_kind VARCHAR(16);

CREATE INDEX IF NOT EXISTS idx_paper_signals_gated
    ON paper_signals (symbol, gates_passed, window_start);


-- ═══════════════════════════════════════════════════════════════════════════
-- poly_trades — Polymarket trade-level history (the fix for stale fill prices)
-- ═══════════════════════════════════════════════════════════════════════════
-- WHY: poly_historical_prices is built from /prices-history, which serves only
-- 1-MINUTE resolution (~5 real points per 5-minute window). Forward-filling
-- that to per-second produced prices a mean 12.9c away from the real order
-- book inside the trade gate, which invalidated every backtest P&L number
-- (see the retraction in WHITEPAPER.md).
--
-- data-api.polymarket.com/trades serves individual FILLS with second-level
-- timestamps, and reaches back at least 60 days: ~1,000 trades and 51-168
-- distinct seconds per 5-minute window, versus ~5 from /prices-history. That
-- makes the research period re-testable on prices someone actually paid.
--
-- STILL NOT THE SAME AS A QUOTE: a fill tells you where a trade happened, not
-- the bid/ask you would have faced. Only poly_book_snapshots has that, and
-- only going forward. The residual gap between "last fill" and "book mid" is
-- therefore measured against the overlapping days rather than assumed.
--
-- Both outcomes are stored. A 'Down' fill at price p implies a YES price of
-- ~(1-p), so both sides can densify one series.
--
-- The wide primary key is deliberate: one transaction can contain several
-- fills, so tx_hash alone is not unique. This combination is, in practice, and
-- it makes re-running the backfill a no-op instead of duplicating rows.

CREATE TABLE IF NOT EXISTS poly_trades (
    symbol          VARCHAR(20) NOT NULL,
    window_start    BIGINT      NOT NULL,   -- epoch seconds, aligned to window_sec
    elapsed_seconds INT         NOT NULL,   -- ts - window_start, 0..window_sec
    ts              BIGINT      NOT NULL,   -- raw fill timestamp (epoch seconds)
    outcome         VARCHAR(8)  NOT NULL,   -- 'Up' | 'Down'
    side            VARCHAR(4)  NOT NULL,   -- 'BUY' | 'SELL' (aggressor direction)
    price           FLOAT8      NOT NULL,
    size            FLOAT8      NOT NULL,
    tx_hash         VARCHAR(80) NOT NULL DEFAULT '',
    PRIMARY KEY (symbol, window_start, ts, outcome, side, price, size, tx_hash)
);

CREATE INDEX IF NOT EXISTS idx_poly_trades_window
    ON poly_trades (symbol, window_start, elapsed_seconds);

-- ─────────────────────────────────────────────────────────────────────────────
-- poly_resolutions: the outcome Polymarket ACTUALLY paid out on.
--
-- THIS IS THE LABEL. window_observations.outcome is derived from Binance trade
-- prices, and on 2026-09-12 it turned out to disagree with Polymarket's own
-- resolution on 14.6% of windows (n=2036, BTC). The disagreement is not a
-- reference-price convention bug -- no Binance-derived rule (last-close vs
-- prev-close, vs first-open, vs VWAP, strict vs >=) scored better than 84%
-- against Polymarket. It is a DIFFERENT PRICE SOURCE: Polymarket resolves
-- these markets off its own oracle feed, which tracks Binance closely but not
-- tick-for-tick.
--
-- The error is entirely concentrated in flat windows -- the ones that matter:
--     move <1bp   45.7% disagreement      move 5-10bp   1.9%
--     move 1-2bp  30.6%                   move >10bp    ~0%
-- i.e. when the window barely moves, our label is close to a coin flip with
-- respect to what the contract actually pays. ~30% of 5-minute windows move
-- less than 2bp.
--
-- Consequences, measured: grading the market against our label made it look
-- 0.201 Brier when its true Brier is 0.159, which manufactured the entire
-- "our model beats the market as a forecaster" result. Under the true label
-- the sign flips: model 0.187 vs market 0.159, t = -5.6 against us.
--
-- outcome_up: 1 if Polymarket resolved "Up", 0 if "Down". Only ever written
-- for markets that have actually resolved (outcomePrices in {"0","1"}) --
-- an unresolved market must be absent, not guessed.
CREATE TABLE IF NOT EXISTS poly_resolutions (
    symbol       VARCHAR(20) NOT NULL,
    window_start BIGINT      NOT NULL,   -- epoch seconds, aligned to window_sec
    outcome_up   SMALLINT    NOT NULL,   -- 1 = Up resolved, 0 = Down
    fetched_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, window_start)
);

-- ─────────────────────────────────────────────────────────────────────────────
-- chainlink_twap: the price series Polymarket ACTUALLY resolves on.
--
-- Polymarket's Up/Down markets settle against the Chainlink BTC/USD (and
-- ETH/USD) TWAP-60s data stream -- "TWAP over the window >= the price at the
-- window's start" -- NOT against Binance. Since 2026-08-07, when the switch
-- happened, our Binance-derived label disagrees with settlement on 12% of
-- windows (it was 4.2% before the switch; see poly_resolutions above).
--
-- Two separate things need this series, and only one of them is covered by
-- poly_resolutions:
--   * the LABEL -- already free from Gamma once a market settles, so this table
--     is not needed for it;
--   * the FEATURES -- the running TWAP and the distance-to-strike (req_bps)
--     that halved the model's deficit to the market. Those must be computed on
--     the SAME series the contract settles on, and that is what this table is
--     for.
--
-- FORWARD-ONLY, and that is the whole reason this is collected rather than
-- backfilled. The free relay (wss://ws-live-data.polymarket.com) serves only a
-- snapshot of the trailing ~60 seconds; there is no history endpoint without
-- paid Chainlink credentials. Every hour not collecting is an hour lost
-- permanently -- the same trap as poly_book_snapshots.
--
-- ts is epoch SECONDS. The stream is second-aligned (verified: every observed
-- timestamp ends in 000 ms); the collector logs a warning if that ever stops
-- being true, because silently truncating sub-second ticks would corrupt the
-- TWAP. full_accuracy_value keeps the exact integer the stream sends (18
-- decimals) so no precision is lost to the FLOAT8.
CREATE TABLE IF NOT EXISTS chainlink_twap (
    symbol              VARCHAR(20) NOT NULL,   -- our convention: BTCUSDT, not btc/usd
    ts                  BIGINT      NOT NULL,   -- epoch seconds
    window_s            INT         NOT NULL DEFAULT 60,
    value               FLOAT8      NOT NULL,
    full_accuracy_value NUMERIC(78, 0),
    captured_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, ts, window_s)
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Late-window research (polaris/late). All three are FORWARD-ONLY and never
-- pruned: none of this can be fetched after the fact for free.
--
-- chainlink_spot: Chainlink's per-second spot price from the Polymarket relay
-- (topic crypto_prices_chainlink, EMPTY filter -- with a symbol filter the relay
-- sends one snapshot and goes silent). Measured 2026-10-07: Chainlink's TWAP-60
-- equals the mean of this series over the trailing 60s to within 0.1 bp, and
-- settlement = TWAP60(t=299) >= TWAP60(t=0) on 99.79% of 2,421 windows. So this
-- series alone determines how every window settles.
CREATE TABLE IF NOT EXISTS chainlink_spot (
    symbol  VARCHAR(20) NOT NULL,
    ts      BIGINT      NOT NULL,   -- epoch seconds (the stream is second-aligned)
    value   FLOAT8      NOT NULL,
    PRIMARY KEY (symbol, ts)
);

-- late_signals: one row per symbol per data-second over the late window, with
-- every ingredient of the estimate and the live book at that instant, so the
-- estimator can be recalibrated and any trading rule replayed offline. No
-- decision is stored: rules are applied afterwards, against poly_resolutions.
CREATE TABLE IF NOT EXISTS late_signals (
    symbol        VARCHAR(20) NOT NULL,
    window_start  BIGINT      NOT NULL,
    t             INT         NOT NULL,   -- data-second within the window (Chainlink clock)
    wall_t        FLOAT8,                 -- wall-clock seconds into the window when logged
    cl_age        FLOAT8,                 -- wall clock minus newest Chainlink point, s
    anchor        FLOAT8,                 -- TWAP60 at t=0 (mean spot over [ws-59, ws])
    mean_final    FLOAT8,                 -- expected settlement TWAP60 at t=299
    sd_final      FLOAT8,
    spot          FLOAT8,
    sigma         FLOAT8,                 -- $ per sqrt(second), trailing 120s
    p_up          FLOAT8,                 -- raw Phi(z); calibrate offline
    yes_ask FLOAT8, yes_ask_size FLOAT8, yes_bid FLOAT8, yes_eff10 FLOAT8,
    no_ask  FLOAT8, no_ask_size  FLOAT8, no_bid  FLOAT8, no_eff10  FLOAT8,
    book_age_ms   FLOAT8,                 -- age of the newest book event used
    captured_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, window_start, t)
);
-- Bid-side depth at the touch, added for the maker study: a resting bid's
-- chance of filling depends on how much queue already sits at that price.
ALTER TABLE late_signals ADD COLUMN IF NOT EXISTS yes_bid_size FLOAT8;
ALTER TABLE late_signals ADD COLUMN IF NOT EXISTS no_bid_size  FLOAT8;

-- poly_ws_trades: every fill the CLOB market channel reports for the current
-- window's tokens. Carries fee_rate_bps per fill -- the fee these 5-minute
-- markets ACTUALLY charge, rather than the category rate on the help page --
-- and is the raw material for a maker (liquidity-providing) study.
CREATE TABLE IF NOT EXISTS poly_ws_trades (
    symbol        VARCHAR(20) NOT NULL,
    window_start  BIGINT      NOT NULL,
    asset_id      VARCHAR(100) NOT NULL,
    outcome       VARCHAR(3)  NOT NULL,   -- YES | NO
    side          VARCHAR(4),             -- aggressor: BUY | SELL
    price         FLOAT8,
    size          FLOAT8,
    fee_rate_bps  INT,
    ts_ms         BIGINT      NOT NULL,
    tx_hash       VARCHAR(80) NOT NULL DEFAULT '',
    PRIMARY KEY (asset_id, ts_ms, tx_hash)
);

-- cross_signals: the 5m and 15m markets that settle at the SAME instant, side
-- by side, every second of the 15m window's last five minutes. Both settle on
-- Chainlink TWAP60 at that instant (one shared F) against different anchors,
-- so their prices are logically linked: if a5 >= a15 then "5m Up" implies
-- "15m Up", and a 15m YES + 5m NO pair always pays >= $1 (mirror case when
-- a5 <= a15). Logged to measure how often -- if ever -- the books violate
-- that by more than the fees. Forward-only; never pruned.
CREATE TABLE IF NOT EXISTS cross_signals (
    symbol      VARCHAR(20) NOT NULL,
    ws15        BIGINT      NOT NULL,
    t15         INT         NOT NULL,     -- 600..899: the last 5m sub-window
    a5          FLOAT8,                   -- anchor of the 5m market (TWAP60 at ws15+600)
    a15         FLOAT8,                   -- anchor of the 15m market (TWAP60 at ws15)
    p5          FLOAT8,                   -- model P(F >= a5)
    p15         FLOAT8,                   -- model P(F >= a15)
    m5_yes_ask FLOAT8, m5_yes_ask_size FLOAT8, m5_yes_bid FLOAT8,
    m5_no_ask  FLOAT8, m5_no_ask_size  FLOAT8, m5_no_bid  FLOAT8,
    m15_yes_ask FLOAT8, m15_yes_ask_size FLOAT8, m15_yes_bid FLOAT8,
    m15_no_ask  FLOAT8, m15_no_ask_size  FLOAT8, m15_no_bid  FLOAT8,
    captured_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, ws15, t15)
);

-- poly_ws_touch: every change of the best bid/ask (price or size) on the current
-- 5m window's tokens, ms-stamped from the CLOB market channel. The maker study
-- (polaris/late/makersim.py) found that resting quotes lose almost entirely on
-- TRADE-THROUGH fills -- our bid left stale after the rest of the book pulled --
-- and that is decided in milliseconds, which the 1-second late_signals grid
-- cannot resolve. Forward-only; never pruned.
CREATE TABLE IF NOT EXISTS poly_ws_touch (
    id           BIGSERIAL PRIMARY KEY,
    symbol       VARCHAR(20) NOT NULL,
    window_start BIGINT      NOT NULL,
    outcome      VARCHAR(3)  NOT NULL,   -- YES | NO
    ts_ms        BIGINT      NOT NULL,   -- engine timestamp of the event
    best_bid     FLOAT8, bid_size FLOAT8,
    best_ask     FLOAT8, ask_size FLOAT8
);
CREATE INDEX IF NOT EXISTS poly_ws_touch_win ON poly_ws_touch (symbol, window_start, ts_ms);
