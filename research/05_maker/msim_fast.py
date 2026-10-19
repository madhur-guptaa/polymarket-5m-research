import pickle, numpy as np, pandas as pd
from polaris.late import makersim as ms
_,fills,res=pickle.load(open("research/.cache/msim_data.pkl","rb"))
sig=pd.read_pickle("research/.cache/sig_fast.pkl")
fills=fills[fills.sym.isin(["BTCUSDT","ETHUSDT"])]
nw=sum((s,int(w)) in res for s,w in sig[["sym","ws"]].drop_duplicates().itertuples(index=False))
print(f"BTC+ETH only, {nw} settled windows")
print(f"{'fair value':14s} {'lat':>4s} {'theta':>6s} {'fills':>6s} {'shares':>7s} {'P&L/sh':>8s} {'total':>9s} {'z':>6s}")
for col,lab in [("p","Chainlink"),("p_fast","Binance-led")]:
    for lat in (0.5,1.0,2.0):
        for th in (-1.0,0.0,0.05,0.10):
            if col=="p" and th!=0.05 and th!=-1.0: continue
            f=ms.simulate(sig,fills,res,ms.SimConfig(theta=th,latency=lat,p_col=col))
            s=ms.summarise(f,nw)
            print(f"{lab:14s} {lat:4.1f} {th:+6.2f} {s['fills']:6d} {s['shares']:7.0f} {s['pnl_per_share']:+8.4f} {s['pnl']:+9.2f} {s['z']:+6.2f}",flush=True)
