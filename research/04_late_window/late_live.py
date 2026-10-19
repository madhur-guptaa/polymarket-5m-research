import asyncio, numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from polaris.core.config import get_settings
from polaris.core.db import open_pool
from polaris.late import makersim as ms
FEE=0.07
async def load():
    cfg=get_settings(); pool=await open_pool(cfg.pg, autocommit=True)
    async with pool.connection() as c, c.cursor() as cur:
        await cur.execute("""SELECT l.symbol, l.window_start, l.t, l.wall_t, l.p_up, l.yes_ask, l.no_ask, l.yes_eff10, l.no_eff10, r.outcome_up
            FROM late_signals l JOIN poly_resolutions r ON r.symbol=l.symbol AND r.window_start=l.window_start
            WHERE l.t >= 200 ORDER BY 1,2,3""")
        d=pd.DataFrame(await cur.fetchall(),columns=["sym","ws","t","wall_t","p","ya","na","y10","n10","up"])
    num=["wall_t","p","ya","na","y10","n10"]; d[num]=d[num].astype(float)
    bt=d[d.sym.isin(["BTCUSDT","ETHUSDT"])]
    lo,hi=float(bt.ws.min())-300,float(bt.ws.max())+400
    b=await ms.load_binance(pool,["BTCUSDT","ETHUSDT"],lo,hi); sp=await ms.load_spot(pool,int(lo)-200,int(hi))
    await pool.close()
    d["p_fast"]=np.nan
    m=d.sym.isin(["BTCUSDT","ETHUSDT"])
    d.loc[m,"p_fast"]=ms.binance_fair_value(d[m].rename(columns={}).reset_index(drop=True),b,sp)
    return d
d=asyncio.run(load())
wins=np.sort(d.ws.unique()); cut=wins[len(wins)//2]
print(f"{d.groupby(['sym','ws']).ngroups} settled coin-windows (t>=200); calibrate on first half, test on second (from {pd.to_datetime(cut,unit='s')})")
def run(col, syms, t0=270, me=0.02, lat=0, price="ask"):
    x=d[d.sym.isin(syms)].dropna(subset=[col]).copy()
    tr=x[x.ws<cut]
    iso=IsotonicRegression(out_of_bounds="clip",y_min=1e-4,y_max=1-1e-4).fit(tr[col],tr.up)
    te=x[x.ws>=cut].copy(); te["pc"]=iso.predict(te[col])
    te=te.sort_values(["sym","ws","t"]); g=te.groupby(["sym","ws"])
    # execution prices L seconds later (only if that row exists exactly L later)
    for s in ("ya","na","y10","n10"): te[s+"_x"]=g[s].shift(-lat) if lat else te[s]
    te["tx"]=g.t.shift(-lat) if lat else te.t
    te=te[(te.tx-te.t)==lat] if lat else te
    rows=[]
    for side in ("YES","NO"):
        q=te.pc if side=="YES" else 1-te.pc
        dec=te.ya if side=="YES" else te.na                       # price seen when deciding
        px=(te.ya_x if price=="ask" else te.y10_x) if side=="YES" else (te.na_x if price=="ask" else te.n10_x)
        win=te.up if side=="YES" else 1-te.up
        e=q-dec-FEE*dec*(1-dec)
        rows.append(pd.DataFrame({"sym":te.sym,"ws":te.ws,"t":te.t,"edge":e,"px":px,"win":win}))
    a=pd.concat(rows).dropna(subset=["edge","px"])
    a=a[(a.t>=t0)&(a.edge>=me)&(a.px>0)&(a.px<1)].sort_values("t").groupby(["sym","ws"]).first()
    if len(a)<5: return len(a),np.nan,np.nan
    pnl=a.win-a.px-FEE*a.px*(1-a.px)
    return len(a), pnl.mean(), pnl.sum()/np.sqrt((a.px*(1-a.px)).sum())
ALL=sorted(d.sym.unique()); BE=["BTCUSDT","ETHUSDT"]
print(f"\nPRE-REGISTERED RULE: t>=270, edge>=0.02 after fee, first qualifying second, buy at the ask")
print(f"{'fair value':22s} {'coins':6s} {'latency':>7s} {'trades':>6s} {'P&L/sh':>8s} {'z':>6s}")
for col,syms,lab in [("p",ALL,"Chainlink (live)"),("p",BE,"Chainlink (live)"),("p_fast",BE,"Binance-led")]:
    for lat in (0,1,2):
        n,m,z=run(col,syms,lat=lat)
        print(f"{lab:22s} {'all 8' if syms is ALL else 'BTC+ETH':6s} {lat:6d}s {n:6d} {m:+8.4f} {z:+6.2f}")
print("\nfull grid (Chainlink, all 8 coins, 1s latency) -- 9 cells, read with multiple-testing in mind:")
for t0 in (240,260,270,280):
    for me in (0.0,0.02,0.05):
        n,m,z=run("p",ALL,t0=t0,me=me,lat=1)
        print(f"  t>={t0} edge>={me:.2f}: trades {n:5d} P&L/sh {m:+.4f} z {z:+.2f}")
