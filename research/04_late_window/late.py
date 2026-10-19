import asyncio, numpy as np, pandas as pd
from polaris.core.config import get_settings
from polaris.core.db import open_pool
FEE=0.07
async def load():
    cfg=get_settings(); pool=await open_pool(cfg.pg, autocommit=True)
    async with pool.connection() as c, c.cursor() as cur:
        await cur.execute("""SELECT symbol, ts-ts%300 ws, ts%300 t, value FROM chainlink_twap WHERE window_s=60""")
        cl=pd.DataFrame(await cur.fetchall(),columns=["sym","ws","t","v"])
        await cur.execute("""SELECT symbol, window_start, elapsed_seconds, side, best_ask, best_bid, eff_ask_10, eff_ask_100
            FROM poly_book_snapshots WHERE elapsed_seconds>=200 AND best_ask IS NOT NULL""")
        bk=pd.DataFrame(await cur.fetchall(),columns=["sym","ws","t","side","ask","bid","ask10","ask100"])
        await cur.execute("SELECT symbol, window_start, outcome_up FROM poly_resolutions")
        res=pd.DataFrame(await cur.fetchall(),columns=["sym","ws","up"])
    await pool.close()
    return cl,bk,res
cl,bk,res=asyncio.run(load())
cl.to_pickle("research/.cache/cl.pkl"); bk.to_pickle("research/.cache/bk_late.pkl"); res.to_pickle("research/.cache/res.pkl")
# per-window arrays: full coverage needed for t=0 and t=299
g=cl.pivot_table(index=["sym","ws"],columns="t",values="v")
g=g[g.notna().sum(axis=1)>=290].dropna(subset=[0,299])
g=g.T.interpolate(limit_direction="both").T          # fill rare single-second gaps
v0=g[0].to_numpy(); v299=g[299].to_numpy()
d=(g.to_numpy()-v0[:,None])/v0[:,None]*1e4          # bps vs anchor, every t
fut=((v299[:,None]-g.to_numpy())/v0[:,None]*1e4)     # remaining move to t=299
idx=g.index.to_frame(index=False)
# volatility normaliser: realised abs change of TWAP60 over the previous 30s
vol=np.abs(np.diff(g.to_numpy(),axis=1)).astype(float)
vol=pd.DataFrame(vol).T.rolling(30,min_periods=10).mean().T.to_numpy()/v0[:,None]*1e4
vol=np.concatenate([np.full((len(g),1),np.nan),vol],axis=1)
order=np.argsort(idx.ws.to_numpy()); n_tr=int(len(order)*0.6)
tr,te=order[:n_tr],order[n_tr:]
print(f"windows: {len(g)}  train {len(tr)}  test {len(te)}")
# empirical p(up | t, z) with z = -d_t / (vol_t * sqrt(299-t)): probability remaining move clears the gap
def zscore(i,t):
    k=max(299-t,1); s=vol[i,t]*np.sqrt(k)+1e-9
    return d[i,t]/s
Z_tr=[]; 
for t in range(200,299):
    k=299-t
    s=vol[tr,t]*np.sqrt(k)+1e-9
    Z_tr.append(pd.DataFrame({"t":t,"z":d[tr,t]/s,"u":fut[tr,t]/s}))
Z_tr=pd.concat(Z_tr).dropna()
# calibrate: P(up) = P(u >= -z); estimate via empirical CDF of u within t-buckets
bins={}
for lo in range(200,299,10):
    u=np.sort(Z_tr[(Z_tr.t>=lo)&(Z_tr.t<lo+10)].u.to_numpy()); bins[lo]=u
def p_up(i,t):
    z=zscore(i,t)
    if not np.isfinite(z): return np.nan
    u=bins[(t//10)*10]
    return 1.0-np.searchsorted(u,-z,side="left")/len(u)
# join with book on test windows
key=idx.iloc[te].assign(i=te)
b=bk.merge(key,on=["sym","ws"]).merge(res,on=["sym","ws"])
b=b[(b.t>=200)&(b.t<=298)]
b["p"]=[p_up(i,t) for i,t in zip(b.i,b.t)]
b=b.dropna(subset=["p"])
b["p_side"]=np.where(b.side=="YES",b.p,1-b.p)
b["win"]=np.where(b.side=="YES",b.up,1-b.up)
for col in ["ask","ask10","ask100"]:
    b[f"fee_{col}"]=FEE*b[col]*(1-b[col])
b.to_pickle("research/.cache/late_rows.pkl")
print(f"test rows (window,second,side) with book: {len(b):,}  windows {b.groupby(['sym','ws']).ngroups}")
# calibration of p on test
cb=pd.cut(b.p_side,[0,.05,.2,.4,.6,.8,.95,.99,1])
print("\ncalibration on test (p_side bucket -> realised win rate, market ask):")
print(b.groupby(cb,observed=True).agg(n=("win","size"),p=("p_side","mean"),win=("win","mean"),ask=("ask","mean")).round(3).to_string())
