import asyncio, pickle, numpy as np, pandas as pd
from polaris.core.config import get_settings
from polaris.core.db import open_pool
from polaris.polymarket.client import PolymarketClient
async def main():
    cfg=get_settings(); pool=await open_pool(cfg.pg, autocommit=True)
    async with pool.connection() as c, c.cursor() as cur:
        await cur.execute("SELECT * FROM cross_signals")
        cols=[d.name for d in cur.description]; x=pd.DataFrame(await cur.fetchall(),columns=cols)
        await cur.execute("SELECT symbol, window_start, outcome_up FROM poly_resolutions")
        r5={(s,int(w)):int(o) for s,w,o in await cur.fetchall()}
    await pool.close()
    keys=x[["symbol","ws15"]].drop_duplicates().itertuples(index=False)
    r15={}; sem=asyncio.Semaphore(8)
    async with PolymarketClient(cfg.polymarket) as pm:
        async def one(s,w):
            async with sem: r15[(s,int(w))]=await pm.resolution(s,int(w),series="15m")
        await asyncio.gather(*[one(s,w) for s,w in keys])
    pickle.dump((x,r5,r15),open("research/.cache/cross.pkl","wb"))
    print(f"rows {len(x):,}, 15m windows {x[['symbol','ws15']].drop_duplicates().shape[0]}, 15m settlements found {sum(v is not None for v in r15.values())}")
asyncio.run(main())
