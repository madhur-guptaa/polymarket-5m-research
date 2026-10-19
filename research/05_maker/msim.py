import asyncio, numpy as np, pandas as pd, pickle
from polaris.core.config import get_settings
from polaris.core.db import open_pool
from polaris.late import makersim as ms
async def main():
    cfg=get_settings(); pool=await open_pool(cfg.pg, autocommit=True)
    sig,fills,res=await ms.load(pool); await pool.close()
    pickle.dump((sig,fills,res),open("research/.cache/msim_data.pkl","wb"))
    m=sig.dropna(subset=["yb","na"])
    print(f"rows {len(sig):,}, fills {len(fills):,}, symbol-windows {sig.groupby(['sym','ws']).ngroups}, "
          f"settled {sum((s,int(w)) in res for s,w in sig[['sym','ws']].drop_duplicates().itertuples(index=False))}")
    print(f"mirror check yes_bid + no_ask: median {(m.yb+m.na).median():.4f}, share within 1e-6 of 1: {(abs(m.yb+m.na-1)<1e-6).mean():.1%}")
asyncio.run(main())
