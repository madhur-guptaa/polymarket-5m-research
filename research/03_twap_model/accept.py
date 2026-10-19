"""Acceptance test: production kde vs kde_twap, equal tuning budget, real settlement."""
import asyncio, logging, time, json, numpy as np, pandas as pd, datetime as dt
logging.basicConfig(level=logging.WARNING)
from polaris.core.config import get_settings
from polaris.core.db import open_pool
from polaris.model.dataset import load_dataset
from polaris.model.factory import build_model
from polaris.model.base import Bandwidths, TWAPBandwidths
from polaris.model import featurelab as fl

SWITCH = int(dt.datetime(2026, 8, 7, tzinfo=dt.timezone.utc).timestamp())
RNG = np.random.default_rng(0)

def pick_windows(fs, mask, k, cadence):
    wins = np.unique(fs.window_start_ts[mask])
    chosen = RNG.choice(wins, min(k, len(wins)), replace=False)
    m = mask & np.isin(fs.window_start_ts, chosen) & (fs.times.astype(int) % cadence == 0)
    return m

async def main():
    cfg = get_settings(); pool = await open_pool(cfg.pg, autocommit=True)
    out = {}
    for sym in ("BTCUSDT", "ETHUSDT"):
        fs = await load_dataset(pool, cfg, sym, days=60)            # settlement labels
        fs = fs.subset(fs.window_start_ts >= SWITCH)                  # clean regime only
        wins = np.unique(fs.window_start_ts)
        cut = wins[int(len(wins) * 0.7)]
        tr, te = fs.window_start_ts < cut, fs.window_start_ts >= cut
        search = fs.subset(pick_windows(fs, tr, 500, 10))            # identical for both
        train = fs.subset(tr & (fs.times.astype(int) % 5 == 0))      # identical for both
        test_m = te & (fs.times.astype(int) % 5 == 0)
        test_idx = np.flatnonzero(test_m)
        test_idx = RNG.choice(test_idx, min(20000, len(test_idx)), replace=False)
        tmask = np.zeros(fs.n, bool); tmask[test_idx] = True
        test = fs.subset(tmask)
        res = {"n_search": search.n, "n_train": train.n, "n_test": test.n,
               "train_windows": int(len(np.unique(train.window_start_ts))),
               "test_windows": int(len(np.unique(test.window_start_ts)))}
        preds = {}
        for kind in ("kde", "kde_twap"):
            t0 = time.time()
            m = build_model(cfg, kind).fit(search)
            best = m.optimize(search, n_trials=40)
            m.bw = type(m.bw)(**best)
            m.fit(train)
            p, _ = m.predict_batch(test)
            preds[kind] = p
            res[kind] = {"bw": best, "brier": float(np.mean((p - test.outcomes) ** 2)),
                         "secs": round(time.time() - t0)}
            print(sym, kind, res[kind], flush=True)
        # vs market on the book subset
        bk = await fl.load_book(pool, sym)
        df = pd.DataFrame({"window_ts": test.window_start_ts, "time_elapsed_s": test.times.astype(int),
                           "y": test.outcomes, **{f"p_{k}": v for k, v in preds.items()}})
        mm = df.merge(bk, on=["window_ts", "time_elapsed_s"], how="inner")
        res["market"] = {"n_rows": len(mm), "n_windows": int(mm.window_ts.nunique()),
                         "brier_market": float(np.mean((mm["mid"] - mm.y) ** 2))}
        for gate, gl in [(None, "all"), ((45, 90), "gate")]:
            s = mm if gate is None else mm[(mm.time_elapsed_s >= gate[0]) & (mm.time_elapsed_s <= gate[1])]
            for k in preds:
                g = s.assign(a=(s[f"p_{k}"] - s.y) ** 2, b=(s["mid"] - s.y) ** 2).groupby("window_ts")[["a", "b"]].mean()
                gap, t = fl.paired_t(g.b.to_numpy(), g.a.to_numpy())
                res["market"][f"{k}_{gl}"] = {"model": float(g.a.mean()), "market": float(g.b.mean()),
                                               "gap": gap, "t": t, "n_win": len(g)}
        out[sym] = res
        print(sym, json.dumps(res["market"], indent=1), flush=True)
    json.dump(out, open("research/.cache/accept.json", "w"), indent=1, default=float)
    await pool.close()

asyncio.run(main())
