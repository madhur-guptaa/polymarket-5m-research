import asyncio, numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from polaris.core.config import get_settings
from polaris.core.db import open_pool
T="research/.cache"; FEE=0.07
cl=pd.read_pickle(f"{T}/cl.pkl"); bk=pd.read_pickle(f"{T}/bk_late.pkl"); res=pd.read_pickle(f"{T}/res.pkl")
async def bars():
    cfg=get_settings(); pool=await open_pool(cfg.pg, autocommit=True)
    lo=int(cl.ws.min())-120
    async with pool.connection() as c, c.cursor() as cur:
        await cur.execute("""SELECT symbol, EXTRACT(EPOCH FROM bucket_time)::BIGINT, vwap::float8 FROM crypto_1s_bars
                             WHERE bucket_time >= to_timestamp(%s) AND symbol IN ('BTCUSDT','ETHUSDT')""",(lo,))
        r=pd.DataFrame(await cur.fetchall(),columns=["sym","ts","px"])
    await pool.close(); return r
bn=asyncio.run(bars())
g=cl.pivot_table(index=["sym","ws"],columns="t",values="v")
g=g[g.notna().sum(axis=1)>=290].dropna(subset=[0,299]); g=g.T.interpolate(limit_direction="both").T
rows=[]
bn_by={s:d.set_index("ts").px for s,d in bn.groupby("sym")}
for (sym,ws),vals in g.iterrows():
    px=bn_by[sym].reindex(range(ws-130,ws+300)).ffill().to_numpy()   # per-second binance price, ffill gaps
    if np.isnan(px).mean()>0.05: continue
    v=vals.to_numpy(); v0=v[0]
    # binance 1s log-return vol over the 120s before the decision point is computed per t below
    lr=np.diff(np.log(px))
    for t in range(200,299):
        i=t+130                                    # index of second t in px
        # align binance to chainlink level: chainlink v(t) ~ mean(px[t-59..t]) + offset
        off=v[t]-np.nanmean(px[i-59:i+1])
        known_lo=max(i-(t-239),i+1) if t<239 else i-(t-239)
        # seconds of the FINAL averaging range [239..299] already observed (inclusive of t)
        if t>=239:
            known=px[i-(t-239):i+1]+off; n_known=len(known); s_known=known.sum()
        else:
            n_known=0; s_known=0.0
        k=299-t if t>=239 else 61                 # remaining seconds that will enter the average
        p_now=px[i]+off
        sig=np.nanstd(lr[i-120:i])*p_now         # $ per sqrt(second)
        n_tot=n_known+k
        mean_final=(s_known+k*p_now)/n_tot
        # future seconds that enter: if t<239 the range starts later; approximate with
        # a random walk from now whose average over the last 61 seconds is taken
        if t>=239:
            var=sig**2*k*(k+1)*(2*k+1)/6/n_tot**2
        else:
            gap=239-t; var=sig**2*(gap+ (61+1)*(2*61+1)/(6*61))
        z=(mean_final-v0)/np.sqrt(var) if var>0 else np.nan
        rows.append((sym,ws,t,norm.cdf(z) if np.isfinite(z) else np.nan))
P=pd.DataFrame(rows,columns=["sym","ws","t","p"]).merge(res,on=["sym","ws"]).dropna()
order=np.sort(P.ws.unique()); cut=order[int(len(order)*0.6)]
tr,te=P[P.ws<cut],P[P.ws>=cut]
iso=IsotonicRegression(out_of_bounds="clip",y_min=1e-4,y_max=1-1e-4).fit(tr.p,tr.up)
P["pc"]=iso.predict(P.p)
print(f"structural model: {P.groupby(['sym','ws']).ngroups} windows; test from {pd.to_datetime(cut,unit='s')}")
for lab,col in [("raw","p"),("isotonic-calibrated","pc")]:
    d=P[P.ws>=cut]; print(f"  test Brier {lab:20s} {np.mean((d[col]-d.up)**2):.4f}")
P.to_pickle(f"{T}/late_struct.pkl")
# trade simulation on TEST windows only
b=bk.merge(P[P.ws>=cut][["sym","ws","t","pc"]],on=["sym","ws","t"]).merge(res,on=["sym","ws"])
b["p_side"]=np.where(b.side=="YES",b.pc,1-b.pc); b["win"]=np.where(b.side=="YES",b.up,1-b.up)
d=b.assign(mbr=lambda x:(x.ask-x.win)**2)
print(f"\nmarket ask Brier on same rows {np.mean((b.ask-b.win)**2):.4f} vs model {np.mean((b.p_side-b.win)**2):.4f}  (rows {len(b):,})")
print("\nfirst qualifying second per (window, side); P&L per share after taker fee, at best ask:")
print(f"{'t-from':>6} {'min edge':>8} {'trades':>6} {'win':>6} {'avg ask':>7} {'P&L/sh':>8} {'z':>6}")
for t0 in (200,240,260,270,280):
    for me in (0.01,0.02,0.04):
        x=b[(b.t>=t0)].copy()
        x["fee"]=FEE*x.ask*(1-x.ask); x["edge"]=x.p_side-x.ask-x.fee
        x=x[x.edge>=me].sort_values("t").groupby(["sym","ws"]).first()
        if len(x)<20: continue
        pnl=x.win-x.ask-x.fee
        z=pnl.sum()/np.sqrt((x.ask*(1-x.ask)).sum())
        print(f"{t0:6d} {me:8.2f} {len(x):6d} {x.win.mean():6.3f} {x.ask.mean():7.3f} {pnl.mean():+8.4f} {z:+6.2f}")
