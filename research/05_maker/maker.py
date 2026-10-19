"""Maker P&L on historical fills, conditioned on a settlement model's view of the maker's side."""
import asyncio, numpy as np, pandas as pd
from scipy.stats import norm
from polaris.core.config import get_settings
from polaris.core.db import open_pool
async def load():
    cfg=get_settings(); pool=await open_pool(cfg.pg, autocommit=True)
    async with pool.connection() as c, c.cursor() as cur:
        await cur.execute("""SELECT window_start, elapsed_seconds, outcome, side, price, size FROM poly_trades
            WHERE symbol='BTCUSDT' AND window_start < 1789257600 AND price>0 AND price<1""")   # <= Sep 12 (pre-backfill set)
        f=pd.DataFrame(await cur.fetchall(),columns=["ws","e","tok","side","x","s"])
        lo,hi=int(f.ws.min())-200,int(f.ws.max())+300
        await cur.execute("""SELECT EXTRACT(EPOCH FROM bucket_time)::BIGINT, vwap::float8 FROM crypto_1s_bars
            WHERE symbol='BTCUSDT' AND bucket_time BETWEEN to_timestamp(%s) AND to_timestamp(%s)""",(lo,hi))
        px=pd.Series(dict(await cur.fetchall()))
        await cur.execute("SELECT window_start, outcome_up FROM poly_resolutions WHERE symbol='BTCUSDT'")
        res=dict(await cur.fetchall())
    await pool.close(); return f,px,res
f,px,res=asyncio.run(load())
px=px.sort_index(); full=px.reindex(range(int(px.index.min()),int(px.index.max())+1)).ffill()
P={}
for ws in f.ws.unique():
    ws=int(ws)
    a=full.reindex(range(ws-180,ws+300)).to_numpy()
    if np.isnan(a).mean()>0.02: continue
    anchor=np.nanmean(a[121:181])                      # [ws-59, ws]
    lr=np.diff(np.log(a))
    p=np.full(300,np.nan)
    for t in range(300):
        i=180+t; sig=np.nanstd(lr[i-120:i])*a[i]
        if t>=240:
            k=299-t; mf=(np.nansum(a[180+240:i+1])+k*a[i])/60; var=sig**2*k*(k+1)*(2*k+1)/6/3600
        else:
            mf=a[i]; var=sig**2*(240-t+19.5)
        p[t]=norm.cdf((mf-anchor)/np.sqrt(var)) if var>0 else float(mf>=anchor)
    P[ws]=p
f=f[f.ws.isin(P.keys())].copy()
f["up"]=f.ws.map(res); f=f.dropna(subset=["up"])
p_up=np.array([P[int(w)][min(int(e),299)] for w,e in zip(f.ws,f.e)])
tok_up=(f.tok=="Up").to_numpy()
# maker's long token L and price y: aggressor SELL of T -> maker buys T at x; aggressor BUY of T -> maker buys not-T at 1-x
maker_long_up=np.where(f.side=="SELL", tok_up, ~tok_up)
y=np.where(f.side=="SELL", f.x, 1-f.x)
pL=np.where(maker_long_up, p_up, 1-p_up)
payL=np.where(maker_long_up, f.up, 1-f.up)
fee=0.07*f.x*(1-f.x)
f=f.assign(d=pL-y, pnl=payL-y+0.2*fee, t=f.e.clip(0,299))
print(f"fills with model value: {len(f):,} over {f.ws.nunique()} windows")
def table(g):
    return g.apply(lambda q: pd.Series({"fills":len(q),"shares":q.s.sum(),
        "maker_pnl_per_share":(q.pnl*q.s).sum()/q.s.sum()}), include_groups=False)
f["d_bucket"]=pd.cut(f.d,[-1,-.3,-.15,-.05,0,.05,.15,.3,1])
print("\nmaker P&L (incl. 20% fee rebate) by model edge on the maker's side, d = P_model(maker side) - price paid:")
print(table(f.groupby("d_bucket",observed=True)).round(4).to_string())
f["tb"]=(f.t//60)*60
print("\n...and split by time in window, for d >= +0.05 vs d <= -0.05:")
for lab,q in [("d>=+0.05",f[f.d>=0.05]),("d<=-0.05",f[f.d<=-0.05])]:
    print(f"  {lab}:", {int(k):round(v,4) for k,v in table(q.groupby("tb")).maker_pnl_per_share.items()})
f[["ws","t","d","pnl","s","x","side","tok"]].to_pickle(f"research/.cache/maker_fills.pkl")

print("\n=== LEAKAGE / LATENCY TEST: classify each fill with the model from L seconds EARLIER ===")
print(f"{'lag':>4} {'share d>=.05':>12} {'P&L d>=.05':>11} {'P&L d<=-.05':>12} {'P&L .05<d<=.15':>15}")
ws_arr=f.ws.astype(int).to_numpy(); e_arr=f.t.astype(int).to_numpy()
for L in (0,1,2,5,10):
    pu=np.array([P[w][e-L] if e-L>=0 else np.nan for w,e in zip(ws_arr,e_arr)])
    pl=np.where(maker_long_up[f.index.isin(f.index)] if False else maker_long_up[:len(f)], pu, 1-pu)
    d=pl-y[:len(f)]
    ok=~np.isnan(d); s=f.s.to_numpy(); pn=f.pnl.to_numpy()
    def w(m): m=m&ok; return (pn[m]*s[m]).sum()/s[m].sum()
    print(f"{L:4d} {100*s[ok&(d>=.05)].sum()/s[ok].sum():11.1f}% {w(d>=.05):+11.4f} {w(d<=-.05):+12.4f} {w((d>.05)&(d<=.15)):+15.4f}")
