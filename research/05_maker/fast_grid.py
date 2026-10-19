import pickle, numpy as np, pandas as pd
from polaris.late import makersim as ms
sig,touch,fills,res=pickle.load(open("research/.cache/fast_data.pkl","rb"))
nw=sum((s,int(w)) in res for s,w in sig[["sym","ws"]].drop_duplicates().itertuples(index=False))
print(f"BTC+ETH, {nw} settled windows, ms-resolution replay")
print(f"{'react':>5s} {'theta':>6s} {'fills':>6s} {'shares':>7s} {'P&L/sh':>8s} {'total$':>8s} {'z':>6s} | {'through sh / P&L':>18s} | {'at-level sh / P&L':>18s}")
rows=[]
for th in (-1.0,0.0,0.05,0.10):
    for r in (0.05,0.25,0.5,1.0):
        f=ms.simulate_fast(sig,touch,fills,res,ms.FastConfig(theta=th,reaction=r))
        s=ms.summarise(f,nw); s.update(theta=th,reaction=r)
        parts=[]
        for k in ("through","at_level"):
            g=f[f.kind==k] if len(f) else f
            parts.append(f"{g.shares.sum():7.0f} / {g.pnl.sum()/max(g.shares.sum(),1):+.4f}")
            s[f"{k}_sh"]=float(g.shares.sum()); s[f"{k}_pnl"]=float(g.pnl.sum())
        rows.append(s)
        print(f"{r:5.2f} {th:+6.2f} {s['fills']:6d} {s['shares']:7.0f} {s['pnl_per_share']:+8.4f} {s['pnl']:+8.1f} {s['z']:+6.2f} | {parts[0]:>18s} | {parts[1]:>18s}",flush=True)
pd.DataFrame(rows).to_pickle("research/.cache/fast_grid.pkl")
