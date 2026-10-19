import asyncio, json, numpy as np
from polaris.core.config import get_settings
from polaris.core.db import open_pool
from polaris.model.dataset import load_dataset
from polaris.model.backtest import walk_forward_candidates, _load_poly

async def main():
    cfg = get_settings()
    pool = await open_pool(cfg.pg)
    fs = await load_dataset(cfg and pool, cfg, "BTCUSDT", days=30)
    poly = await _load_poly(pool, "BTCUSDT")
    await pool.close()
    cand, n_win = walk_forward_candidates(fs, poly, cfg, train_days=5, cadence_s=10)
    p = cand["p_up"].to_numpy(); y = cand["outcome"].to_numpy()
    el = cand["elapsed_seconds"].to_numpy()

    # ---- per-time-bucket discrimination ----
    buckets = [(0,30),(30,60),(60,120),(120,240),(240,301)]
    per_bucket=[]
    for lo,hi in buckets:
        m=(el>=lo)&(el<hi)
        if m.sum()==0: continue
        pm,ym=p[m],y[m]
        # confidence split: how well does p_up sort outcomes in THIS time window?
        hiconf=pm>=0.6; loconf=pm<=0.4
        per_bucket.append({
            "lo":lo,"hi":hi,"n":int(m.sum()),
            "brier":round(float(((pm-ym)**2).mean()),4),
            "up_rate":round(float(ym.mean()),3),
            "act_when_pgt60": round(float(ym[hiconf].mean()),3) if hiconf.any() else None,
            "n_pgt60": int(hiconf.sum()),
            "act_when_plt40": round(float(ym[loconf].mean()),3) if loconf.any() else None,
            "n_plt40": int(loconf.sum()),
        })

    # ---- time-gate backtest sweep (min_edge fixed, vary earliest entry second) ----
    ME, MN = 0.05, 20.0
    c = cand.copy()
    c["won"] = np.where(c["is_yes"], c["outcome"]==1, c["outcome"]==0)
    c["price"] = np.where(c["is_yes"], c["yes_price"], c["no_price"])
    c["pnl"] = np.where(c["won"], 1.0-c["price"], -c["price"])
    gates=[]
    for g in [0,30,45,60,90,120,150,180]:
        sig = c[(c["edge"]>=ME)&(c["n_eff"]>=MN)&(c["elapsed_seconds"]>=g)]
        if len(sig)==0: gates.append({"gate":g,"trades":0}); continue
        first = sig.sort_values("elapsed_seconds").groupby("window_start").first()
        gates.append({
            "gate":g,"trades":int(len(first)),
            "win":round(float(first["won"].mean()),3),
            "pnl":round(float(first["pnl"].sum()),1),
            "avg_pnl":round(float(first["pnl"].mean()),4),
            "avg_entry":int(first["elapsed_seconds"].mean()),
            "avg_edge":round(float(first["edge"].mean()),4),
        })
    print("TIMEJSON:"+json.dumps({"per_bucket":per_bucket,"gates":gates}))

asyncio.run(main())
