import asyncio, json, numpy as np
from polaris.core.config import get_settings
from polaris.core.db import open_pool
from polaris.model.dataset import load_dataset
from polaris.model.backtest import walk_forward_candidates, _book, _load_poly

async def main():
    cfg = get_settings()
    pool = await open_pool(cfg.pg)
    fs = await load_dataset(pool, cfg, "BTCUSDT", days=30)
    poly = await _load_poly(pool, "BTCUSDT")
    await pool.close()
    cand, n_win = walk_forward_candidates(fs, poly, cfg, train_days=5, cadence_s=10)

    # Panel 3: reliability — out-of-sample p_up deciles vs actual up-rate
    p = cand["p_up"].to_numpy(); y = cand["outcome"].to_numpy()
    bins = np.linspace(0,1,11)
    idx = np.clip(np.digitize(p, bins)-1, 0, 9)
    rel=[]
    for b in range(10):
        m = idx==b
        rel.append({"bin_lo":round(bins[b],2),"bin_hi":round(bins[b+1],2),
                    "n":int(m.sum()),
                    "pred":float(p[m].mean()) if m.any() else None,
                    "actual":float(y[m].mean()) if m.any() else None})
    # Panel 2: trade-entry elapsed distribution at min_edge 0.05
    res = _book(cand, n_win, 0.05, 20.0)
    ent = res.trades["elapsed"].to_numpy() if len(res.trades) else np.array([])
    # histogram in 20s buckets over 0..300
    hist,_ = np.histogram(ent, bins=np.arange(0,321,20))
    entry=[{"t":int(20*i),"count":int(hist[i])} for i in range(len(hist))]

    out={"n_scored":int(len(cand)),"n_windows":int(n_win),
         "reliability":rel,"entry":entry,
         "brier":float(((p-y)**2).mean()),
         "n_trades":int(len(res.trades)),"win_rate":float(res.win_rate)}
    print("WFJSON:"+json.dumps(out))

asyncio.run(main())
