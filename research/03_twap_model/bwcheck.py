import asyncio, logging, numpy as np, datetime as dt, json
logging.basicConfig(level=logging.WARNING)
from polaris.core.config import get_settings
from polaris.core.db import open_pool
from polaris.model.dataset import load_dataset
from polaris.model.factory import build_model
from polaris.model.base import TWAPBandwidths
SWITCH=int(dt.datetime(2026,8,7,tzinfo=dt.timezone.utc).timestamp())
async def main():
    cfg=get_settings(); pool=await open_pool(cfg.pg, autocommit=True); out={}
    for sym in ("BTCUSDT","ETHUSDT"):
        rng=np.random.default_rng(0)   # same draws as accept.py -> same search/test sets
        fs=await load_dataset(pool,cfg,sym,days=60); fs=fs.subset(fs.window_start_ts>=SWITCH)
        wins=np.unique(fs.window_start_ts); cut=wins[int(len(wins)*0.7)]
        tr,te=fs.window_start_ts<cut, fs.window_start_ts>=cut
        ch=rng.choice(np.unique(fs.window_start_ts[tr]),500,replace=False)
        search=fs.subset(tr&np.isin(fs.window_start_ts,ch)&(fs.times.astype(int)%10==0))
        train=fs.subset(tr&(fs.times.astype(int)%5==0))
        ti=np.flatnonzero(te&(fs.times.astype(int)%5==0)); ti=rng.choice(ti,min(20000,len(ti)),replace=False)
        tm=np.zeros(fs.n,bool); tm[ti]=True; test=fs.subset(tm)
        m=build_model(cfg,"kde_twap")
        shared=TWAPBandwidths(65.0,3000.0,0.157,1.0)   # geometric mean of BTC & ETH optima
        per={"BTCUSDT":TWAPBandwidths(43.12,4213.5,0.1665,0.7004),
             "ETHUSDT":TWAPBandwidths(98.99,2263.5,0.1493,1.4139)}[sym]
        best=shared.as_dict(); res={}
        for lab,bw in [("own-symbol tuned",per),("shared geo-mean",shared)]:
            m.bw=bw; m.fit(train); p,_=m.predict_batch(test)
            res[lab]=float(np.mean((p-test.outcomes)**2))
            print(sym,f"{lab:18s} Brier {res[lab]:.5f}",flush=True)
        print(sym,"best:",best,flush=True); out[sym]=res
    json.dump(out,open("research/.cache/bwcheck.json","w"),indent=1)
    await pool.close()
asyncio.run(main())
