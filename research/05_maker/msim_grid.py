import pickle, numpy as np, pandas as pd
from polaris.late import makersim as ms
sig,fills,res=pickle.load(open("research/.cache/msim_data.pkl","rb"))
nw=sum((s,int(w)) in res for s,w in sig[["sym","ws"]].drop_duplicates().itertuples(index=False))
rows=[]
for lat in (0.5,1.0,2.0,5.0):
    for th in (-1.0,0.0,0.02,0.05,0.10):
        f=ms.simulate(sig,fills,res,ms.SimConfig(theta=th,latency=lat))
        s=ms.summarise(f,nw); s.update(latency=lat,theta=th); rows.append(s)
        print(f"lat {lat:3.1f}s theta {th:+.2f}: fills {s['fills']:6d} shares {s['shares']:8.0f} "
              f"P&L/share {s['pnl_per_share']:+.4f} total ${s['pnl']:+8.2f} z {s['z']:+.2f} windows {s['windows_traded']}",flush=True)
pd.DataFrame(rows).to_pickle("research/.cache/msim_grid.pkl")
