import asyncio, json, httpx, time, pickle, sys
from polaris.core.config import get_settings
cfg=get_settings()
async def main():
    now=int(time.time()); ws0=now-(now%300)
    n_windows=int(sys.argv[1]) if len(sys.argv)>1 else 2600
    out={}; sem=asyncio.Semaphore(8); done=0
    lim=asyncio.Semaphore(1)
    async with httpx.AsyncClient(timeout=25, follow_redirects=True) as cl:
        async def one(ws, sym, prefix):
            nonlocal done
            for attempt in range(4):
                async with sem:
                    try:
                        r=await cl.get(cfg.polymarket.gamma_api_url, params={"slug":f"{prefix}-{ws}"})
                        if r.status_code==429:
                            await asyncio.sleep(1.5*(attempt+1)); continue
                        ev=r.json()
                    except Exception:
                        await asyncio.sleep(1.0*(attempt+1)); continue
                m=(ev[0].get("markets") or [None])[0] if ev else None
                if m:
                    op=m.get("outcomePrices")
                    if isinstance(op,str):
                        try: op=json.loads(op)
                        except Exception: op=None
                    if op and op[0] in ("0","1"):
                        out[(sym,ws)]=int(op[0]=="1")
                break
            done+=1
            if done%200==0: print(f"  {done} fetched, {len(out)} resolved", flush=True)
        tasks=[]
        for sym,prefix in (("BTCUSDT","btc-updown-5m"),("ETHUSDT","eth-updown-5m")):
            tasks += [one(ws0-300*k, sym, prefix) for k in range(1, n_windows)]
        await asyncio.gather(*tasks)
    pickle.dump(out, open(sys.argv[2] if len(sys.argv)>2 else "res.pkl","wb"))
    print("TOTAL resolved:", len(out))
asyncio.run(main())
