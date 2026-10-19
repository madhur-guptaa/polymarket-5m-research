"""Out-of-time test: test days restricted to the never-before-available window."""
import asyncio, json, os, numpy as np
from datetime import datetime, timezone
from polaris.core.config import get_settings
from polaris.core.db import open_pool
from polaris.model.dataset import load_dataset
from polaris.model.backtest import walk_forward_candidates, _book, _load_poly

# Identical to the config behind the published 60.9% (session 14).
CAD, TRAIN_DAYS, ME, MN, G0, G1 = 10, 14, 0.05, 20.0, 45, 90
DAYS, WARMUP = 30, 20   # load Aug 11+, skip 20 days -> test starts ~Aug 31

async def main():
    cfg = get_settings(); pool = await open_pool(cfg.pg)
    fs = await load_dataset(pool, cfg, "BTCUSDT", days=DAYS)
    poly = await _load_poly(pool, "BTCUSDT"); await pool.close()

    day = (fs.window_start_ts // 86400).astype(np.int64)
    alldays = np.unique(day)
    test = alldays[WARMUP:]
    d = lambda x: datetime.fromtimestamp(int(x)*86400, tz=timezone.utc).date()
    print(f"loaded {fs.n:,} obs / {fs.n_windows:,} windows  "
          f"({d(alldays[0])} .. {d(alldays[-1])})", flush=True)
    print(f"TEST DAYS: {d(test[0])} .. {d(test[-1])}  ({len(test)} days)", flush=True)

    cand, n_win = walk_forward_candidates(fs, poly, cfg, train_days=TRAIN_DAYS,
                                          cadence_s=CAD, warmup_days=WARMUP)
    p, y = cand["p_up"].to_numpy(), cand["outcome"].to_numpy()
    out = {"test_from": str(d(test[0])), "test_to": str(d(test[-1])),
           "test_days": int(len(test)), "scored": int(len(cand)),
           "windows": int(n_win), "brier": round(float(((p-y)**2).mean()), 4)}
    # 0c = comparable to the published number; 1c = the measured execution cost
    for cost in (0.0, 0.01):
        r = _book(cand, n_win, ME, MN, G0, G1, spread_cost=cost)
        out[f"cost_{int(cost*100)}c"] = {
            "trades": r.n_trades, "win_rate": round(r.win_rate, 4),
            "pnl": round(r.total_pnl, 1), "avg_pnl": round(r.avg_pnl, 4)}
    print("OOTJSON:" + json.dumps(out))

asyncio.run(main())
