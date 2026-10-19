import asyncio, pickle, time, numpy as np, pandas as pd
from polaris.core.config import get_settings
from polaris.core.db import open_pool
from polaris.late import makersim as ms
SYMS=["BTCUSDT","ETHUSDT"]
async def main():
    cfg=get_settings(); pool=await open_pool(cfg.pg, autocommit=True)
    async with pool.connection() as c, c.cursor() as cur:
        await cur.execute("SELECT min(ts_ms), max(ts_ms) FROM poly_ws_touch"); lo_ms,hi_ms=await cur.fetchone()
    lo_ws=(lo_ms//1000//300+1)*300; hi_ws=(hi_ms//1000//300)*300-300      # whole windows only
    sig,fills,res=await ms.load(pool, since_epoch=lo_ws)
    sig=sig[sig.sym.isin(SYMS)&(sig.ws<=hi_ws)].reset_index(drop=True)
    fills=fills[fills.sym.isin(SYMS)&(fills.ws<=hi_ws)]
    t0=time.time()
    b=await ms.load_binance(pool,SYMS,lo_ws-300,hi_ws+400)
    sp=await ms.load_spot(pool,lo_ws-400,hi_ws+400)
    sig["p_fast"]=ms.binance_fair_value(sig,b,sp)
    print(f"fair value for {len(sig):,} rows in {time.time()-t0:.0f}s; NaN {sig.p_fast.isna().mean():.1%}")
    touch=await ms.load_touch(pool,SYMS,lo_ws*1000,(hi_ws+300)*1000)
    await pool.close()
    pickle.dump((sig,touch,fills,res),open("research/.cache/fast_data.pkl","wb"))
    print(f"windows {sig.groupby(['sym','ws']).ngroups}, touch rows {len(touch):,}, fills {len(fills):,}")
asyncio.run(main())
