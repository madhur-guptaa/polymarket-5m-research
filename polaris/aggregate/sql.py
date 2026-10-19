"""
polaris/aggregate/sql.py

The SQL that turns raw trades into model-ready observations. These queries are
the real value of the aggregation stage, ported verbatim from the old
bar_aggregator — the 4-CTE window query in particular is doing subtle work and
is not worth "simplifying".

  AGGREGATE_BARS_SQL   crypto_trades      -> 1-second OHLCV bars (+ volume split)
  WINDOW_OBS_SQL       crypto_1s_bars     -> per-second features per 5-min window
  UPSERT_BAR_SQL       idempotent bar write
  UPSERT_OBS_SQL       idempotent obs write (COALESCE keeps a set outcome)
  GET/SET_CURSOR_SQL   incremental high-water mark
  EARLIEST_TRADE_SQL   first-run bootstrap

Params are psycopg named style %(name)s. Window alignment uses a fixed origin
('2020-01-01 00:00:00+00') so boundaries are stable across restarts — the
Python side (aggregate.py) floors to the SAME origin, so they must stay in sync.
"""

from __future__ import annotations

from typing import Final

# Origin for date_bin window alignment. MUST match WINDOW_ORIGIN in aggregate.py.
WINDOW_ORIGIN_SQL: Final[str] = "2020-01-01 00:00:00+00"


AGGREGATE_BARS_SQL: Final[str] = """
    WITH bucketed AS (
        SELECT
            date_trunc('second', trade_time) AS bucket,
            price::FLOAT8    AS price,
            quantity::FLOAT8 AS quantity,
            is_buyer_mm,
            ROW_NUMBER() OVER (
                PARTITION BY date_trunc('second', trade_time)
                ORDER BY trade_time ASC, id ASC
            ) AS rn_asc,
            ROW_NUMBER() OVER (
                PARTITION BY date_trunc('second', trade_time)
                ORDER BY trade_time DESC, id DESC
            ) AS rn_desc
        FROM crypto_trades
        WHERE symbol = %(sym)s
          AND trade_time > %(since)s
          AND trade_time < %(until)s
    )
    SELECT
        bucket,
        MAX(CASE WHEN rn_asc = 1 THEN price END)  AS open_price,
        MAX(price)                                AS high_price,
        MIN(price)                                AS low_price,
        MAX(CASE WHEN rn_desc = 1 THEN price END) AS close_price,
        SUM(quantity)                             AS volume,
        COUNT(*)::INTEGER                         AS trade_count,
        CASE WHEN SUM(quantity) > 0
             THEN SUM(price * quantity) / SUM(quantity) ELSE 0 END AS vwap,
        SUM(CASE WHEN is_buyer_mm THEN quantity ELSE 0 END)        AS mm_buy_volume,
        SUM(quantity)                             AS total_taker_volume
    FROM bucketed
    GROUP BY bucket
    ORDER BY bucket
"""


UPSERT_BAR_SQL: Final[str] = """
    INSERT INTO crypto_1s_bars (
        symbol, bucket_time, open_price, high_price, low_price,
        close_price, volume, trade_count, vwap, mm_buy_volume, total_taker_volume
    ) VALUES (
        %(symbol)s, %(bucket_time)s, %(open_price)s, %(high_price)s, %(low_price)s,
        %(close_price)s, %(volume)s, %(trade_count)s, %(vwap)s,
        %(mm_buy_volume)s, %(total_taker_volume)s
    )
    ON CONFLICT (symbol, bucket_time) DO UPDATE SET
        open_price = EXCLUDED.open_price, high_price = EXCLUDED.high_price,
        low_price = EXCLUDED.low_price, close_price = EXCLUDED.close_price,
        volume = EXCLUDED.volume, trade_count = EXCLUDED.trade_count,
        vwap = EXCLUDED.vwap, mm_buy_volume = EXCLUDED.mm_buy_volume,
        total_taker_volume = EXCLUDED.total_taker_volume
"""


# 4-CTE feature query. bars -> assign window -> running features -> derive.
# outcome is only set once the window is essentially complete (max_elapsed
# within 5s of the end), so provisional recomputations leave it NULL.
#
# LABEL (outcome) here is a BINANCE PROXY, and a poor one since 2026-08-07.
# Polymarket resolves Up iff the Chainlink BTC/USD TWAP-60s averaged over the
# window is >= the price at the window's start -- an AVERAGE, not the
# point-in-time end-vs-start comparison this column implements (an earlier
# version of this comment claimed the opposite; it was wrong). Against real
# settlement this proxy disagrees on ~12% of post-switch windows. It is kept
# because it is cheap and sometimes useful, but nothing trains on it by
# default: model/dataset.py uses poly_resolutions (real settlements).
#
# TWAP GEOMETRY (twap_sofar_bps, spot_vs_anchor_bps) describes the contract's
# actual payoff. `anchor` reconstructs Chainlink's "price at the beginning of
# the range" as a trailing 60s TWAP of per-second VWAP taken at the window's
# first bar -- the stream's own quantity, rebuilt from Binance (the real series
# is collected into chainlink_twap but has no history yet). The trailing 60s
# deliberately crosses into the PREVIOUS window; that is why obs_since in
# aggregate.py reaches ANCHOR_LOOKBACK_SEC back, and why the anchor for a
# window whose preceding minute is missing comes out slightly short (it is the
# mean of whatever bars exist, never an extrapolation).
#   twap_sofar_bps      running mean of VWAP since window open, vs anchor
#   spot_vs_anchor_bps  this second's VWAP vs anchor
# The derived distance-to-strike (req_bps) and locked fraction are NOT stored:
# they are exact functions of these two plus time_elapsed_s, computed in
# model/features.py so training, live serving and the heatmap grid share one
# formula.
WINDOW_OBS_SQL: Final[str] = f"""
    WITH bars AS (
        SELECT
            b.bucket_time,
            b.vwap::FLOAT8                                      AS vwap,
            b.close_price::FLOAT8                               AS close_price,
            b.volume::FLOAT8                                    AS volume,
            COALESCE(b.mm_buy_volume, 0)::FLOAT8                AS sec_mm_vol,
            COALESCE(b.total_taker_volume, b.volume, 0)::FLOAT8 AS sec_taker_vol
        FROM crypto_1s_bars b
        WHERE b.symbol = %(sym)s
          AND b.bucket_time > %(since)s
          AND b.bucket_time < %(until)s
    ),
    windowed AS (
        SELECT bars.*,
            AVG(bars.vwap) OVER (ORDER BY bars.bucket_time
                RANGE BETWEEN INTERVAL '60 seconds' PRECEDING AND CURRENT ROW) AS t60,
            date_bin(
                make_interval(secs => %(window_sec)s),
                bucket_time,
                '{WINDOW_ORIGIN_SQL}'::TIMESTAMPTZ
            ) AS window_start
        FROM bars
    ),
    enriched AS (
        SELECT w.*,
            EXTRACT(EPOCH FROM (w.bucket_time - w.window_start))::INTEGER AS time_elapsed_s,
            FIRST_VALUE(w.vwap) OVER win_full AS open_vwap,
            FIRST_VALUE(w.t60)  OVER win_full AS anchor,
            AVG(w.vwap)         OVER win_run  AS run_twap,
            LAST_VALUE(w.vwap)  OVER win_full AS close_vwap,
            -- Unpartitioned: the last trade price strictly before THIS bar,
            -- crossing window boundaries on purpose (it's usually the prior
            -- window's last bar).
            LAG(w.close_price) OVER (ORDER BY w.bucket_time) AS prev_close_price,
            SUM(w.volume)       OVER win_run  AS cum_volume,
            SUM(w.sec_mm_vol)   OVER win_run  AS cum_mm_vol,
            SUM(w.sec_taker_vol) OVER win_run AS cum_taker_vol,
            CASE WHEN COUNT(*) OVER win_10s >= 10
                 THEN STDDEV_SAMP(w.vwap) OVER win_10s ELSE NULL END AS vol_10s_raw,
            MAX(EXTRACT(EPOCH FROM (w.bucket_time - w.window_start))::INTEGER)
                OVER win_part AS max_elapsed
        FROM windowed w
        WINDOW
            win_full AS (PARTITION BY w.window_start ORDER BY w.bucket_time
                         ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING),
            win_run  AS (PARTITION BY w.window_start ORDER BY w.bucket_time
                         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
            win_10s  AS (PARTITION BY w.window_start ORDER BY w.bucket_time
                         ROWS BETWEEN 9 PRECEDING AND CURRENT ROW),
            win_part AS (PARTITION BY w.window_start
                         ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
    )
    SELECT
        %(sym)s AS symbol,
        e.window_start,
        e.bucket_time,
        e.time_elapsed_s,
        CASE WHEN e.open_vwap > 0
             THEN (e.vwap - e.open_vwap) / e.open_vwap * 10000.0 ELSE 0 END AS delta_bps,
        CASE WHEN e.cum_taker_vol > 0
             THEN e.cum_mm_vol / e.cum_taker_vol ELSE NULL END AS cum_imbalance,
        e.cum_volume,
        CASE WHEN e.vol_10s_raw IS NOT NULL AND e.open_vwap > 0
             THEN e.vol_10s_raw / e.open_vwap * 10000.0 ELSE NULL END AS volatility_10s,
        CASE WHEN e.max_elapsed >= (%(window_sec)s - 5)
             THEN CASE WHEN LAST_VALUE(e.close_price) OVER win_pt
                            >= COALESCE(FIRST_VALUE(e.prev_close_price) OVER win_pt, e.open_vwap)
                       THEN 1 ELSE 0 END
             ELSE NULL END AS outcome,
        CASE WHEN e.anchor > 0
             THEN (e.run_twap - e.anchor) / e.anchor * 10000.0 END AS twap_sofar_bps,
        CASE WHEN e.anchor > 0
             THEN (e.vwap - e.anchor) / e.anchor * 10000.0 END AS spot_vs_anchor_bps
    FROM enriched e
    -- min_window_start = align_down(since, window_sec): the lookback that
    -- feeds prev_close_price's LAG (see `bars`/obs_since above) reaches ~2s
    -- into the PRECEDING window on purpose, but that one straggler bar must
    -- never become an output row -- with no other bars from its window in
    -- range, it would compute as a degenerate 1-row partition (wrong
    -- cum_volume/cum_imbalance/volatility_10s, and a spuriously-recomputed
    -- outcome from comparing that single bar to itself) and UPSERT_OBS_SQL
    -- would overwrite the already-correct, already-finalized row for it.
    -- This filter keeps it in the CTEs (so LAG/prev_close_price still sees
    -- it) but drops it from what actually gets upserted.
    WHERE e.window_start >= %(min_window_start)s
    WINDOW win_pt AS (PARTITION BY e.window_start ORDER BY e.bucket_time
                      ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
    ORDER BY e.window_start, e.bucket_time
"""


UPSERT_OBS_SQL: Final[str] = """
    INSERT INTO window_observations (
        symbol, window_start, bucket_time, time_elapsed_s,
        delta_bps, cum_imbalance, cum_volume, volatility_10s, outcome,
        twap_sofar_bps, spot_vs_anchor_bps
    ) VALUES (
        %(symbol)s, %(window_start)s, %(bucket_time)s, %(time_elapsed_s)s,
        %(delta_bps)s, %(cum_imbalance)s, %(cum_volume)s, %(volatility_10s)s, %(outcome)s,
        %(twap_sofar_bps)s, %(spot_vs_anchor_bps)s
    )
    ON CONFLICT (symbol, bucket_time) DO UPDATE SET
        window_start   = EXCLUDED.window_start,
        time_elapsed_s = EXCLUDED.time_elapsed_s,
        delta_bps      = EXCLUDED.delta_bps,
        cum_imbalance  = EXCLUDED.cum_imbalance,
        cum_volume     = EXCLUDED.cum_volume,
        volatility_10s = EXCLUDED.volatility_10s,
        twap_sofar_bps     = EXCLUDED.twap_sofar_bps,
        spot_vs_anchor_bps = EXCLUDED.spot_vs_anchor_bps,
        -- Never erase an already-set outcome with a provisional NULL.
        outcome        = COALESCE(EXCLUDED.outcome, window_observations.outcome)
"""


GET_CURSOR_SQL: Final[str] = "SELECT last_time FROM aggregator_cursors WHERE symbol = %(sym)s"

SET_CURSOR_SQL: Final[str] = """
    INSERT INTO aggregator_cursors (symbol, last_time)
    VALUES (%(sym)s, %(ts)s)
    ON CONFLICT (symbol) DO UPDATE SET last_time = EXCLUDED.last_time
"""

EARLIEST_TRADE_SQL: Final[str] = "SELECT MIN(trade_time) FROM crypto_trades WHERE symbol = %(sym)s"
