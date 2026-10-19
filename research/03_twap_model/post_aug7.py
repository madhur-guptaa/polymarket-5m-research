import pandas as pd, numpy as np, logging, lightgbm as lgb, datetime as dt
logging.basicConfig(level=logging.WARNING)
from polaris.model import featurelab as fl
T="research/.cache"
SWITCH=int(dt.datetime(2026,8,7,tzinfo=dt.timezone.utc).timestamp())
TW=list(fl.TWAP_FEATURES)
sets={"BASE (production 6)":list(fl.BASE),
      "BASE+rv_60s+vwap_dev":list(fl.BASE)+["rv_60s","vwap_dev"],
      "TWAP geometry":TW,
      "ALL":list(fl.BASE)+["rv_60s","vwap_dev"]+TW}
for sym in ("BTCUSDT","ETHUSDT"):
    df=pd.read_pickle(f"{T}/{sym}_twap.pkl").dropna(subset=["pm"])
    for lab,sub in [("ALL HISTORY (mixed regimes)",df),
                    ("POST-AUG-7 ONLY (clean)",df[df.window_ts>=SWITCH])]:
        d=sub.assign(outcome=sub.pm)
        print(f"\n=== {sym} | {lab} | {d.window_ts.nunique()} windows ===")
        for name,cols in sets.items():
            b,_,_=fl.cv_brier(d,cols,params=fl.SCREEN_PARAMS)
            bg,_,_=fl.cv_brier(d,cols,gate=(45,90),params=fl.SCREEN_PARAMS)
            print(f"   {name:22s} all {b:.5f}   gate {bg:.5f}")
