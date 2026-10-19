import numpy as np, pandas as pd
T="research/.cache"; FEE=0.07
P=pd.read_pickle(f"{T}/late_struct.pkl"); bk=pd.read_pickle(f"{T}/bk_late.pkl")
order=np.sort(P.ws.unique()); cut=order[int(len(order)*0.6)]
P=P[P.ws>=cut]
def sim(lat=0, price="ask", t0=270, me=0.02, sub=None):
    q=P[["sym","ws","t","pc","up"]].copy(); q["t"]=q.t+lat         # decided with info from t-lat
    b=bk.merge(q,on=["sym","ws","t"])
    if sub is not None: b=b[sub(b)]
    b=b.dropna(subset=[price])
    b["p_side"]=np.where(b.side=="YES",b.pc,1-b.pc); b["win"]=np.where(b.side=="YES",b.up,1-b.up)
    px=b[price]; b["fee"]=FEE*px*(1-px); b["edge"]=b.p_side-px-b.fee
    x=b[(b.t>=t0)&(b.edge>=me)].sort_values("t").groupby(["sym","ws"]).first()
    if len(x)==0: return 0,np.nan,np.nan
    pnl=x.win-x[price]-x.fee
    return len(x), pnl.mean(), pnl.sum()/np.sqrt((x[price]*(1-x[price])).sum())
print("base rule: from t=270, edge>=0.02 after fee")
print(f"{'variant':40s} {'n':>5s} {'P&L/sh':>8s} {'z':>6s}")
for lab,kw in [("zero latency, best ask",{}),
               ("latency 2s",{"lat":2}),("latency 5s",{"lat":5}),("latency 10s",{"lat":10}),
               ("fill at $10 effective ask",{"price":"ask10"}),
               ("fill at $100 effective ask",{"price":"ask100"}),
               ("BTC only",{"sub":lambda b:b.sym=="BTCUSDT"}),("ETH only",{"sub":lambda b:b.sym=="ETHUSDT"})]:
    n,m,z=sim(**kw); print(f"{lab:40s} {n:5d} {m:+8.4f} {z:+6.2f}")
# time split: choose rule on first half of test, evaluate on second
mid=np.sort(P.ws.unique())[len(P.ws.unique())//2]
A=lambda b:b.ws<mid; B=lambda b:b.ws>=mid
print("\nrule selection on test-half A, evaluation on untouched half B:")
best=None
for t0 in (240,250,260,270,280):
    for me in (0.0,0.01,0.02,0.04,0.06):
        n,m,z=sim(t0=t0,me=me,sub=A)
        if n>=30 and (best is None or z>best[0]): best=(z,t0,me,n,m)
print(f"  chosen on A: t0={best[1]} min_edge={best[2]}  (A: n={best[3]}, P&L {best[4]:+.4f}, z={best[0]:+.2f})")
n,m,z=sim(t0=best[1],me=best[2],sub=B); print(f"  untouched B: n={n}, P&L/sh {m:+.4f}, z={z:+.2f}")
n,m,z=sim(t0=best[1],me=best[2],sub=B,lat=5); print(f"  untouched B, 5s latency: n={n}, P&L/sh {m:+.4f}, z={z:+.2f}")
