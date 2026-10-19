import pickle, numpy as np, pandas as pd
x,r5,r15=pickle.load(open("research/.cache/cross.pkl","rb"))
num=[c for c in x.columns if c not in ("symbol","ws15","t15","captured_at")]
x[num]=x[num].astype(float)
x["up5"]=[r5.get((s,int(w)+600)) for s,w in zip(x.symbol,x.ws15)]
x["up15"]=[r15.get((s,int(w))) for s,w in zip(x.symbol,x.ws15)]
x=x.dropna(subset=["up5","up15","a5","a15"])
x["gap_bp"]=(x.a5-x.a15)/x.a15*1e4
w=x.groupby(["symbol","ws15"]).agg(gap=("gap_bp","first"),up5=("up5","first"),up15=("up15","first")).reset_index()
print(f"windows with both settlements: {len(w)}")
for lo in (0.0,0.5,1.0):
    a=w[w.gap>=lo]; b=w[w.gap<=-lo]
    va=((a.up5==1)&(a.up15==0)).sum(); vb=((b.up15==1)&(b.up5==0)).sum()
    print(f"|anchor gap| >= {lo} bp: a5>a15 n={len(a)} violations(5mUp & 15mDown)={va} | a5<a15 n={len(b)} violations(15mUp & 5mDown)={vb}")
# the guaranteed pair at each second
fee=lambda p: 0.07*p*(1-p)
hi=x.gap_bp>=0
x["leg15"]=np.where(hi,x.m15_yes_ask,x.m15_no_ask); x["leg5"]=np.where(hi,x.m5_no_ask,x.m5_yes_ask)
x["sz"]=np.minimum(np.where(hi,x.m15_yes_ask_size,x.m15_no_ask_size),np.where(hi,x.m5_no_ask_size,x.m5_yes_ask_size))
x["cost"]=x.leg15+x.leg5+fee(x.leg15)+fee(x.leg5)
x["payout"]=np.where(hi, x.up15+(1-x.up5), (1-x.up15)+x.up5)       # >= 1 by construction
x=x.dropna(subset=["cost"])
print(f"\npair cost incl. fees (rows {len(x):,}): min {x.cost.min():.4f}  p1 {x.cost.quantile(.01):.4f}  median {x.cost.median():.4f}")
print(f"payout distribution: {x.groupby(['symbol','ws15']).payout.first().value_counts().to_dict()}  (2 = final landed between the anchors)")
for c in (1.0,1.01,1.02,1.03):
    q=x[x.cost<c]
    if q.empty: print(f"cost < {c}: none"); continue
    f=q.sort_values("t15").groupby(["symbol","ws15"]).first()
    print(f"cost < {c:.2f}: rows {len(q):5d}, windows {len(f):3d}, median size {f.sz.median():6.0f} sh | "
          f"first-chance P&L/pair {(f.payout-f.cost).mean():+.4f}, worst {(f.payout-f.cost).min():+.4f}, by symbol {f.reset_index().symbol.value_counts().to_dict()}")

print("\n=== EXECUTABILITY: decide on second s, pay the prices logged at s+L ===")
x=x.sort_values(["symbol","ws15","t15"])
g=x.groupby(["symbol","ws15"])
for L in (1,2,3,5):
    x[f"cost_{L}"]=g.cost.shift(-L); x[f"t_{L}"]=g.t15.shift(-L)
    ok=x[(x.cost<1.0)&((x[f"t_{L}"]-x.t15)==L)]
    f=ok.sort_values("t15").groupby(["symbol","ws15"]).first()
    still=(f[f"cost_{L}"]<1.0).mean()
    pnl=(f.payout-f[f"cost_{L}"])
    print(f"latency {L}s: {len(f)} windows; still riskless at s+{L}: {still:.0%}; P&L/pair at s+{L} prices: mean {pnl.mean():+.4f}, worst {pnl.min():+.4f}")
o=x[x.cost<1.0].copy()
d=o.groupby(["symbol","ws15"]).agg(secs=("t15","size"),t_first=("t15","min"),t_last=("t15","max"),best_edge=("cost",lambda c:1-c.min()),
                                   size_at_best=("sz","median"))
print("\nper opportunity: seconds below $1, when it starts (t15 of 600-899), best edge, median size")
print(d.describe().round(3).loc[["count","mean","50%","min","max"]].to_string())
print("\ntime of first cost<1 within the last-5m sub-window (t15-600):", sorted((d.t_first-600).astype(int).tolist()))
