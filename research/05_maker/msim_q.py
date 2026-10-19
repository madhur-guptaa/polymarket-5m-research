import pickle, numpy as np, pandas as pd
from polaris.late import makersim as ms
_,fills,res=pickle.load(open("research/.cache/msim_data.pkl","rb"))
sig=pd.read_pickle("research/.cache/sig_fast.pkl")
fills=fills[fills.sym.isin(["BTCUSDT","ETHUSDT"])]
nw=sum((s,int(w)) in res for s,w in sig[["sym","ws"]].drop_duplicates().itertuples(index=False))
print(f"{'queue':6s} {'theta':>6s} {'fills':>6s} {'shares':>7s} {'P&L/sh':>8s} {'z':>6s} | {'through: sh, P&L/sh':>22s} | {'at level: sh, P&L/sh':>22s}")
for qm in ("back","front"):
    for th in (-1.0,0.0,0.05,0.10):
        f=ms.simulate(sig,fills,res,ms.SimConfig(theta=th,latency=1.0,p_col="p_fast",queue_mode=qm))
        s=ms.summarise(f,nw)
        parts=[]
        for k in ("through","at_level"):
            g=f[f.kind==k] if len(f) else f
            parts.append(f"{g.shares.sum():8.0f}, {g.pnl.sum()/max(g.shares.sum(),1):+.4f}")
        print(f"{qm:6s} {th:+6.2f} {s['fills']:6d} {s['shares']:7.0f} {s['pnl_per_share']:+8.4f} {s['z']:+6.2f} | {parts[0]:>22s} | {parts[1]:>22s}",flush=True)
