"""
polaris/main.py

Unified entry point. More subcommands (aggregate, serve, calibrate) land as
those components are built.

    python -m polaris.main ingest
    python -m polaris.main backfill --days 14 --symbol BTCUSDT
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal

from polaris.core.config import get_settings
from polaris.core.db import ensure_schema, open_pool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("polaris")


def _install_signals(stop: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # pragma: no cover (Windows)
            signal.signal(sig, lambda *_: stop.set())


async def _run_ingest() -> None:
    from polaris.ingest.binance import TradeIngester
    from polaris.ingest.sink import PgTradeSink

    cfg = get_settings()
    pool = await open_pool(cfg.pg)
    try:
        if cfg.ingest.ensure_schema:
            await ensure_schema(pool)
        stop = asyncio.Event()
        _install_signals(stop)
        ingester = TradeIngester(cfg, PgTradeSink(pool))
        logger.info("Ingesting symbols=%s", cfg.binance.symbols)
        await ingester.run(stop)
    finally:
        await pool.close()


async def _run_aggregate() -> None:
    from polaris.aggregate import aggregate

    cfg = get_settings()
    # Batch writer: synchronous_commit=off trades a little durability for speed.
    pool = await open_pool(cfg.pg, options="-c synchronous_commit=off")
    try:
        await ensure_schema(pool)
        stop = asyncio.Event()
        _install_signals(stop)
        await aggregate.run(pool, cfg, stop)
    finally:
        await pool.close()


async def _run_backfill(symbols: list[str] | None, days: int) -> None:
    from polaris.ingest.backfill import backfill

    cfg = get_settings()
    # Backfill uses its own sync connections; still ensure the schema exists.
    pool = await open_pool(cfg.pg)
    try:
        await ensure_schema(pool)
    finally:
        await pool.close()
    await backfill(cfg, symbols=symbols, days=days)


async def _run_backtest(symbol: str, days: int, min_edge: float, min_n_eff: float,
                        train_days: int, cadence: int, sweep: str | None,
                        gate_start: int | None, gate_end: int | None,
                        gate_sweep: str | None, spread_cost: float,
                        spread_sweep: str | None, model: str | None = None,
                        price_source: str = "ffill") -> None:
    from polaris.model.backtest import run as run_bt

    cfg = get_settings()
    edges = [float(x) for x in sweep.split(",")] if sweep else None
    gates = [int(x) for x in gate_sweep.split(",")] if gate_sweep else None
    spreads = [float(x) for x in spread_sweep.split(",")] if spread_sweep else None
    pool = await open_pool(cfg.pg)
    try:
        await run_bt(pool, cfg, symbol, days=days, min_edge=min_edge, min_n_eff=min_n_eff,
                     train_days=train_days, cadence_s=cadence, sweep=edges,
                     gate_start=gate_start, gate_end=gate_end, gate_sweep=gates,
                     spread_cost=spread_cost, spread_sweep=spreads, model_kind=model,
                     price_source=price_source)
    finally:
        await pool.close()


async def _run_calibrate(symbol: str, days: int | None, s1: int, s2: int,
                         backend: str, train_cadence: int) -> None:
    from polaris.model.calibrate import run_calibration

    cfg = get_settings()
    pool = await open_pool(cfg.pg)
    try:
        await run_calibration(pool, cfg, symbol, days=days, step1_trials=s1, step2_trials=s2,
                              backend=backend, train_cadence=train_cadence)
    finally:
        await pool.close()


async def _run_backfill_poly_trades(symbols: list[str] | None, days: int) -> None:
    from polaris.polymarket.trades_backfill import backfill as tb

    cfg = get_settings()
    pool = await open_pool(cfg.pg)
    try:
        await ensure_schema(pool)
        await tb(cfg, pool, symbols=symbols, days=days)
    finally:
        await pool.close()


async def _run_backfill_resolutions(symbols: list[str] | None, days: int,
                                    interval: float = 0.0) -> None:
    """Fetch how Polymarket actually SETTLED each window -- the real label.

    With --interval this becomes a service, and it has to be one: settlement is
    the ONLY trustworthy label, it is forward-only (a market must settle before
    it can be read), and when this was left as a manual command it silently went
    four days stale. The backfill is idempotent and fetches only windows missing
    from poly_resolutions, so each pass after the first costs ~1 request per
    window elapsed.
    """
    from polaris.polymarket.resolutions import agreement_report, backfill

    cfg = get_settings()
    pool = await open_pool(cfg.pg)
    log = logging.getLogger(__name__)
    try:
        await ensure_schema(pool)
        while True:
            for sym in (symbols or cfg.binance.symbols):
                try:
                    await backfill(pool, cfg, sym, days=days)
                    rep = await agreement_report(pool, sym)
                    if rep["n"]:
                        log.warning(
                            "[%s] Binance-derived label disagrees with Polymarket "
                            "settlement on %d/%d windows (%.1f%%)",
                            sym, rep["disagree"], rep["n"], rep["disagree_pct"])
                except Exception:  # noqa: BLE001
                    # A loop that dies on one bad pass is worse than no loop:
                    # this is the tripwire, and it must outlive transient Gamma
                    # errors and network changes.
                    log.exception("[%s] resolution pass failed; retrying next tick", sym)
            if interval <= 0:
                return
            await asyncio.sleep(interval)
    finally:
        await pool.close()


async def _run_recompute_obs(symbols: list[str] | None, days: int,
                             lag_minutes: int) -> None:
    """Rebuild window_observations over history from bars -- e.g. to fill a new column."""
    from datetime import datetime, timedelta, timezone

    from polaris.aggregate.aggregate import recompute_observations

    cfg = get_settings()
    pool = await open_pool(cfg.pg)
    try:
        await ensure_schema(pool)
        until = datetime.now(timezone.utc) - timedelta(minutes=lag_minutes)
        since = until - timedelta(days=days)
        for sym in (symbols or cfg.binance.symbols):
            n = await recompute_observations(pool, cfg, sym, since, until)
            logging.getLogger(__name__).info("[%s] recompute-obs done: %d rows", sym, n)
    finally:
        await pool.close()


async def _run_chainlink_collect(symbols: list[str] | None, interval: float,
                                window_s: int) -> None:
    """Collect the Chainlink TWAP series Polymarket actually settles on."""
    from polaris.chainlink.collect import run as collect

    cfg = get_settings()
    pool = await open_pool(cfg.pg)
    try:
        await ensure_schema(pool)
        await collect(pool, cfg, symbols=symbols, window_s=window_s,
                      interval_sec=interval)
    finally:
        await pool.close()


async def _run_paper_report() -> None:
    """Adjudicate the live paper trade against real Polymarket settlements."""
    from polaris.report import load_trades, report

    cfg = get_settings()
    pool = await open_pool(cfg.pg, autocommit=True)
    try:
        report(await load_trades(pool))
    finally:
        await pool.close()


async def _run_prune(days: int | None, tables: list[str] | None, apply: bool) -> None:
    from polaris.core.prune import prune

    cfg = get_settings()
    pool = await open_pool(cfg.pg)
    try:
        await prune(pool, days=days, tables=tables, apply=apply)
    finally:
        await pool.close()


async def _run_paper(interval: float) -> None:
    from polaris import paper

    cfg = get_settings()
    pool = await open_pool(cfg.pg)
    try:
        await ensure_schema(pool)
        stop = asyncio.Event()
        _install_signals(stop)
        await paper.run(pool, cfg, stop, interval=interval)
    finally:
        await pool.close()


async def _run_book_collect() -> None:
    from polaris.polymarket import bookcollect

    cfg = get_settings()
    pool = await open_pool(cfg.pg)
    try:
        await ensure_schema(pool)
        stop = asyncio.Event()
        _install_signals(stop)
        await bookcollect.run(pool, cfg, stop)
    finally:
        await pool.close()


async def _run_late_paper() -> None:
    from polaris.late import trader

    cfg = get_settings()
    pool = await open_pool(cfg.pg)
    try:
        await ensure_schema(pool)
        stop = asyncio.Event()
        _install_signals(stop)
        await trader.run(pool, cfg, stop)
    finally:
        await pool.close()


async def _run_backfill_poly(symbols: list[str] | None, days: int) -> None:
    from polaris.polymarket.backfill import backfill as poly_backfill

    cfg = get_settings()
    pool = await open_pool(cfg.pg)
    try:
        await ensure_schema(pool)
        await poly_backfill(cfg, pool, symbols=symbols, days=days)
    finally:
        await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="polaris")
    sub = parser.add_subparsers(dest="service", required=True)

    sub.add_parser("ingest", help="Live Binance WS -> PostgreSQL")
    sub.add_parser("aggregate", help="trades -> 1s bars -> window observations")

    bf = sub.add_parser("backfill", help="Historical Binance Vision -> PostgreSQL")
    bf.add_argument("--symbol", action="append", dest="symbols",
                    help="Symbol to backfill (repeatable). Defaults to configured symbols.")
    bf.add_argument("--days", type=int, default=14, help="Days back to fetch (default 14).")

    bp = sub.add_parser("backfill-poly", help="Historical Polymarket prices -> PostgreSQL")
    bp.add_argument("--symbol", action="append", dest="symbols",
                    help="Symbol to backfill (repeatable). Defaults to configured symbols.")
    bp.add_argument("--days", type=int, default=14, help="Days back to fetch (default 14).")

    bpt = sub.add_parser("backfill-poly-trades",
                         help="Historical Polymarket FILLS (second resolution) -> poly_trades. "
                              "Use this rather than backfill-poly for backtest prices: "
                              "/prices-history is only 1-minute resolution.")
    bpt.add_argument("--symbol", action="append", dest="symbols")
    bpt.add_argument("--days", type=int, default=30)

    bres = sub.add_parser("backfill-resolutions",
                          help="Polymarket SETTLEMENTS -> poly_resolutions. This is the "
                               "real label: the Binance-derived outcome in "
                               "window_observations disagrees with it on ~15%% of "
                               "windows, concentrated in flat ones.")
    bres.add_argument("--symbol", action="append", dest="symbols")
    bres.add_argument("--days", type=int, default=30)
    bres.add_argument("--interval", type=float, default=0.0,
                      help="Seconds between passes. 0 (default) runs once and exits; "
                           "set it to run as a service (compose uses 3600).")

    sub.add_parser("late-paper",
                   help="PAPER-only late-window collector: Chainlink-based settlement "
                        "estimate vs the live CLOB book, every second of each window's "
                        "end. Forward-only; no orders, no credentials.")

    rco = sub.add_parser("recompute-obs",
                         help="Rebuild window_observations from crypto_1s_bars over "
                              "history (fills new feature columns; bars are kept "
                              "forever, raw trades are not). Safe alongside the live "
                              "aggregator.")
    rco.add_argument("--symbol", action="append", dest="symbols")
    rco.add_argument("--days", type=int, default=90)
    rco.add_argument("--lag-minutes", type=int, default=30,
                     help="Stop this far behind now, so the live aggregator owns the "
                          "windows at the edge (default 30).")

    clc = sub.add_parser("chainlink-collect",
                         help="Collect the Chainlink TWAP-60s series Polymarket "
                              "resolves on. FORWARD-ONLY -- the free relay serves "
                              "only the trailing ~60s and cannot be backfilled.")
    clc.add_argument("--symbol", action="append", dest="symbols")
    clc.add_argument("--interval", type=float, default=30.0,
                     help="Seconds between polls (default 30; must stay below the "
                          "~58s snapshot depth or coverage gaps appear).")
    clc.add_argument("--window", type=int, default=60, choices=(30, 60),
                     help="TWAP window in seconds (default 60, the one Polymarket uses).")

    sub.add_parser("paper-report",
                   help="Score the live paper trade against REAL Polymarket "
                        "settlements (one trade per window, real ask prices). "
                        "Run backfill-resolutions first.")

    pr = sub.add_parser("prune", help="Data retention (dry run unless --apply)")
    pr.add_argument("--days", type=int, default=None,
                    help="Override the per-table default retention.")
    pr.add_argument("--table", action="append", dest="tables",
                    help="Limit to these tables (repeatable).")
    pr.add_argument("--apply", action="store_true",
                    help="Actually delete. Without this, only reports.")

    pt = sub.add_parser("paper-trade",
                        help="Record live signals -> paper_signals (forward test; "
                             "logs only, never executes)")
    pt.add_argument("--interval", type=float, default=5.0, help="Seconds between samples.")

    sub.add_parser("book-collect",
                   help="Sample the live Polymarket CLOB book -> poly_book_snapshots "
                        "(forward-only: spread cannot be backfilled)")

    srv = sub.add_parser("serve", help="Run the FastAPI dashboard server")
    srv.add_argument("--host", default="0.0.0.0")
    srv.add_argument("--port", type=int, default=8000)
    srv.add_argument("--reload", action="store_true", help="Auto-reload (dev).")

    bt = sub.add_parser("backtest", help="Walk-forward paper-trade feasibility backtest")
    bt.add_argument("--symbol", default="BTCUSDT")
    bt.add_argument("--model", default=None, choices=["kde_twap", "kde", "gbm"],
                    help="Which ProbabilityModel to run (default: MODEL_KIND, i.e. kde_twap).")
    bt.add_argument("--days", type=int, default=30, help="History window to load.")
    bt.add_argument("--min-edge", type=float, default=0.03)
    bt.add_argument("--min-n-eff", type=float, default=20.0)
    bt.add_argument("--train-days", type=int, default=7, help="Rolling training window.")
    bt.add_argument("--cadence", type=int, default=5, help="Score every Nth second (speed).")
    bt.add_argument("--price-source", default="ffill", choices=["ffill","trades","book"],
                    help="Market price series. 'ffill' is the RETRACTED 1-minute "
                         "forward-fill (12.9c from the real book); 'trades' is real fills "
                         "at second resolution; 'book' is actual quotes (forward-only, "
                         "from 2026-09-06).")
    bt.add_argument("--sweep", default=None, help="Comma-separated min_edge values, e.g. 0.02,0.05,0.08")
    bt.add_argument("--gate-start", type=int, default=None,
                    help="Trade gate floor (elapsed_seconds >=); None = unbounded.")
    bt.add_argument("--gate-end", type=int, default=None,
                    help="Trade gate ceiling (elapsed_seconds <); None = unbounded.")
    bt.add_argument("--gate-sweep", default=None,
                    help="Comma-separated gate_start values to sweep (gate_end fixed via --gate-end), "
                         "e.g. 0,30,45,60,90,120,180")
    bt.add_argument("--spread-cost", type=float, default=0.0,
                    help="Fixed one-sided execution cost (price units, e.g. 0.02 = 2 cents) added to "
                         "the fill and netted out of edge before gating. Default 0.0 = fair-price fills.")
    bt.add_argument("--spread-sweep", default=None,
                    help="Comma-separated spread_cost values to sweep, e.g. 0,0.01,0.02,0.03,0.05")

    cal = sub.add_parser("calibrate", help="Two-phase calibration (bandwidths + risk)")
    cal.add_argument("--symbol", default="BTCUSDT", help="Symbol to calibrate.")
    cal.add_argument("--days", type=int, default=None, help="Days of history (default: config).")
    cal.add_argument("--step1-trials", type=int, default=100, help="Bandwidth trials.")
    cal.add_argument("--step2-trials", type=int, default=500, help="Risk trials.")
    cal.add_argument("--backend", choices=["numpy", "mlx"], default="numpy",
                     help="LOOCV backend for Step 1. mlx (Apple GPU) is ~9.5x faster at "
                          "N~10^5-10^6 -- needed for a real 14-day-scale search.")
    cal.add_argument("--train-cadence", type=int, default=1,
                     help="Thin the Step-1 SEARCH set to every Nth second (speed; baseline/"
                          "tuned Brier and Step 2 still run on the full set). e.g. 5 cuts "
                          "LOOCV's O(N^2) search cost ~25x.")

    args = parser.parse_args()

    try:
        import uvloop  # type: ignore
        uvloop.install()
    except ImportError:
        pass

    if args.service == "serve":
        import uvicorn
        uvicorn.run("polaris.api.app:app", host=args.host, port=args.port, reload=args.reload)
        return

    if args.service == "ingest":
        asyncio.run(_run_ingest())
    elif args.service == "aggregate":
        asyncio.run(_run_aggregate())
    elif args.service == "backfill":
        symbols = [s.upper() for s in args.symbols] if args.symbols else None
        asyncio.run(_run_backfill(symbols, args.days))
    elif args.service == "backfill-poly-trades":
        symbols = [s.upper() for s in args.symbols] if args.symbols else None
        asyncio.run(_run_backfill_poly_trades(symbols, args.days))
    elif args.service == "backfill-resolutions":
        symbols = [s.upper() for s in args.symbols] if args.symbols else None
        asyncio.run(_run_backfill_resolutions(symbols, args.days, args.interval))
    elif args.service == "late-paper":
        asyncio.run(_run_late_paper())
    elif args.service == "recompute-obs":
        symbols = [s.upper() for s in args.symbols] if args.symbols else None
        asyncio.run(_run_recompute_obs(symbols, args.days, args.lag_minutes))
    elif args.service == "chainlink-collect":
        symbols = [s.upper() for s in args.symbols] if args.symbols else None
        asyncio.run(_run_chainlink_collect(symbols, args.interval, args.window))
    elif args.service == "paper-report":
        asyncio.run(_run_paper_report())
    elif args.service == "prune":
        asyncio.run(_run_prune(args.days, args.tables, args.apply))
    elif args.service == "paper-trade":
        asyncio.run(_run_paper(args.interval))
    elif args.service == "book-collect":
        asyncio.run(_run_book_collect())
    elif args.service == "backfill-poly":
        symbols = [s.upper() for s in args.symbols] if args.symbols else None
        asyncio.run(_run_backfill_poly(symbols, args.days))
    elif args.service == "backtest":
        asyncio.run(_run_backtest(args.symbol.upper(), args.days, args.min_edge,
                                  args.min_n_eff, args.train_days, args.cadence, args.sweep,
                                  args.gate_start, args.gate_end, args.gate_sweep,
                                  args.spread_cost, args.spread_sweep, args.model,
                                  args.price_source))
    elif args.service == "calibrate":
        asyncio.run(_run_calibrate(args.symbol.upper(), args.days, args.step1_trials, args.step2_trials,
                                   args.backend, args.train_cadence))


if __name__ == "__main__":
    main()
