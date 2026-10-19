import asyncio, pickle, numpy as np, pandas as pd
from polaris.core.config import get_settings
from polaris.core.db import open_pool
from polaris.late import makersim as ms
sig,fills,res=pickle.load(open("research/.cache/msim_data.pkl","rb"))
sig=sig[sig.sym.isin(["BTCUSDT","ETHUSDT"])].reset_index(drop=True)
async def main():
    cfg=get_settings(); pool=await open_pool(cfg.pg, autocommit=True)
    lo=float(sig.ws.min())-300; hi=float(sig.ws.max())+400
    b=await ms.load_binance(pool,["BTCUSDT","ETHUSDT"],lo,hi)
    sp=await ms.load_spot(pool,int(lo)-200,int(hi))
    await pool.close()
    print({k:len(v[0]) for k,v in b.items()}, "binance trades")
    sig["p_fast"]=ms.binance_fair_value(sig,b,sp)
    sig.to_pickle("research/.cache/sig_fast.pkl")
asyncio.run(main())
s=pd.read_pickle("research/.cache/sig_fast.pkl").dropna(subset=["yb","ya","p_fast"])
s["mid"]=(s.yb+s.ya)/2; s=s[(s.mid>0.05)&(s.mid<0.95)].sort_values(["sym","ws","t"])
g=s.groupby(["sym","ws"])
for h in (1,3,5,10): s[f"dm{h}"]=g.mid.shift(-h)-s.mid
for col,lab in [("p","Chainlink-only"),("p_fast","Binance-led")]:
    d=s[col]-s.mid
    print(f"{lab:15s} rows {d.notna().sum():6d}  corr(model - mid, future mid change):",
          {h: round(d.corr(s[f'dm{h}']),3) for h in (1,3,5,10)})
    b=pd.cut(d,[-1,-.1,-.05,-.02,.02,.05,.1,1])
    print((s.groupby(b,observed=True)[["dm1","dm3","dm5"]].mean()*100).round(2).assign(n=s.groupby(b,observed=True).size()).to_string(),"\n")
