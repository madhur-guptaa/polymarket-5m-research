"""Backtest the EXACT windows the live paper trade covered, and compare p_up directly."""
import asyncio, numpy as np, pandas as pd
from polaris.core.config import get_settings
from polaris.core.db import open_pool
from polaris.model.dataset import load_dataset
from polaris.model.backtest import walk_forward_candidates, _book, _load_poly

CAD, TRAIN, ME, MN, G0, G1 = 10, 5, 0.05, 20.0, 45, 90
SYM = "BTCUSDT"

async def main():
    cfg = get_settings(); pool = await open_pool(cfg.pg)
    fs = await load_dataset(pool, cfg, SYM, days=30)
    poly = await _load_poly(pool, SYM)
    async with pool.connection() as c, c.cursor() as cur:
        await cur.execute("""select window_start, elapsed_seconds, p_up, gates_passed, side
                             from paper_signals where symbol=%s""", (SYM,))
        live = pd.DataFrame(await cur.fetchall(),
                            columns=["window_start","elapsed_seconds","p_up_live","gated_live","side_live"])
    await pool.close()

    day = (fs.window_start_ts // 86400).astype(np.int64)
    warm = len(np.unique(day)) - 2              # test only the last 2 days
    cand, n_win = walk_forward_candidates(fs, poly, cfg, train_days=TRAIN,
                                          cadence_s=CAD, warmup_days=warm)
    # restrict to the exact windows the live recorder saw
    live_windows = set(live["window_start"].unique())
    c2 = cand[cand["window_start"].isin(live_windows)].copy()
    print(f"live windows: {len(live_windows)} | backtest candidates in them: {len(c2)}", flush=True)

    r = _book(c2, len(live_windows), ME, MN, G0, G1)
    if r.n_trades:
        t = r.trades
        print(f"BACKTEST on those windows: {r.n_trades} trades, {100*r.win_rate:.1f}% win, "
              f"P&L {r.total_pnl:+.2f}, avg price {t['price'].mean():.3f}, "
              f"RoC {100*t['pnl'].sum()/t['price'].sum():+.1f}%")
    else:
        print("BACKTEST on those windows: 0 trades")

    # direct p_up comparison on matching (window, second)
    m = pd.merge(c2[["window_start","elapsed_seconds","p_up"]], live,
                 on=["window_start","elapsed_seconds"], how="inner")
    if len(m):
        d = m["p_up"] - m["p_up_live"]
        print(f"p_up matched on {len(m)} (window,second) pairs")
        print(f"  backtest mean {m['p_up'].mean():.4f} | live mean {m['p_up_live'].mean():.4f}")
        print(f"  diff  mean {d.mean():+.4f}  p50 {d.median():+.4f}  "
              f"p10 {d.quantile(.1):+.4f}  p90 {d.quantile(.9):+.4f}  |maxdiff| {d.abs().max():.4f}")
        print(f"  corr {m['p_up'].corr(m['p_up_live']):.4f}")
    else:
        print("no overlapping (window,second) pairs to compare")
    print("LVBTDONE")

asyncio.run(main())
