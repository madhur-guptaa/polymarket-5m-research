"""KDE vs GBM through the identical walk-forward harness."""
import asyncio, json, pickle, numpy as np
from polaris.core.config import get_settings
from polaris.core.db import open_pool
from polaris.model.dataset import load_dataset
from polaris.model.backtest import walk_forward_candidates, _book, _load_poly

CAD, TRAIN_DAYS, ME, MN, G0, G1 = 10, 5, 0.05, 20.0, 45, 90
OUT = __import__("os").environ["CLAUDE_JOB_DIR"] + "/tmp"

async def main():
    cfg = get_settings(); pool = await open_pool(cfg.pg)
    fs = await load_dataset(pool, cfg, "BTCUSDT", days=60)
    poly = await _load_poly(pool, "BTCUSDT"); await pool.close()
    print(f"dataset: {fs.n:,} obs / {fs.n_windows:,} windows", flush=True)

    results = {}
    for kind in ("gbm", "kde"):          # gbm first: it's fast, fails fast
        print(f"--- {kind} walk-forward ---", flush=True)
        cand, n_win = walk_forward_candidates(
            fs, poly, cfg, train_days=TRAIN_DAYS, cadence_s=CAD, model_kind=kind)
        with open(f"{OUT}/cand_{kind}.pkl", "wb") as f:
            pickle.dump({"cand": cand, "n_win": n_win}, f)
        r = _book(cand, n_win, ME, MN, G0, G1)
        p, y = cand["p_up"].to_numpy(), cand["outcome"].to_numpy()
        results[kind] = {
            "scored": int(len(cand)), "windows": int(n_win),
            "brier": round(float(((p - y) ** 2).mean()), 4),
            "trades": r.n_trades, "win_rate": round(r.win_rate, 4),
            "pnl": round(r.total_pnl, 1), "avg_pnl": round(r.avg_pnl, 4),
            "avg_edge": round(r.avg_edge, 4),
        }
        print(kind, results[kind], flush=True)
    json.dump(results, open(f"{OUT}/h2h.json", "w"))
    print("H2HDONE")

asyncio.run(main())
