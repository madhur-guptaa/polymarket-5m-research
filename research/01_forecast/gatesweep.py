"""Where does the edge actually live, on REAL prices? One walk-forward, many gates."""
import asyncio, numpy as np
from polaris.core.config import get_settings
from polaris.core.db import open_pool
from polaris.model.dataset import load_dataset
from polaris.model.backtest import walk_forward_candidates, _book, _load_poly

SRC, CAD, TRAIN, MN = "trades", 10, 5, 20.0

async def main():
    cfg=get_settings(); pool=await open_pool(cfg.pg)
    fs=await load_dataset(pool,cfg,"BTCUSDT",days=14)
    poly=await _load_poly(pool,"BTCUSDT",SRC)
    await pool.close()
    print(f"priced rows {len(poly):,} | obs {fs.n:,} / {fs.n_windows:,} windows", flush=True)
    cand,nw = walk_forward_candidates(fs,poly,cfg,train_days=TRAIN,cadence_s=CAD)
    print(f"candidates {len(cand):,} over {nw:,} test windows", flush=True)

    buckets=[(0,15),(15,30),(30,45),(45,60),(60,75),(75,90),(90,120),
             (120,180),(180,240),(240,300),(0,30),(0,45),(0,60),(45,90),(0,300)]
    for me in (0.05, 0.10):
        print(f"\n=== min_edge {me} | price source {SRC} ===")
        print(f"{'gate':>12} {'trades':>7} {'%win':>7} {'P&L':>9} {'RoC':>8} {'avgpx':>7} {'%wins':>7}")
        for g0,g1 in buckets:
            r=_book(cand,nw,me,MN,g0,g1)
            if not r.n_trades:
                print(f"{f'[{g0},{g1})':>12} {0:>7}"); continue
            t=r.trades; cap=t['price'].sum(); pnl=t['pnl'].sum()
            se=np.sqrt(r.win_rate*(1-r.win_rate)/len(t))
            print(f"{f'[{g0},{g1})':>12} {r.n_trades:>7} {100*r.win_rate:>6.1f}% {pnl:>+9.1f} "
                  f"{100*pnl/cap:>+7.1f}% {t['price'].mean():>7.3f} {100*r.n_trades/nw:>6.1f}%"
                  f"   z={(r.win_rate-0.5)/se:+.1f}")
    print("GATEDONE")
asyncio.run(main())
