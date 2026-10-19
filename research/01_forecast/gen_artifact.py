import json, os
d = os.environ["CLAUDE_JOB_DIR"] + "/tmp"
anchor = json.load(open(d+"/anchor.json"))
wf = json.load(open(d+"/wf.json"))

# thin anchor to every 2s for lighter payload
anchor = [a for a in anchor if a["t"] % 2 == 0]
rel = [r for r in wf["reliability"] if r["n"]]
entry = wf["entry"]
DATA = {"anchor": anchor, "rel": rel, "entry": entry,
        "brier": round(wf["brier"],4), "n_trades": wf["n_trades"],
        "win_rate": round(wf["win_rate"],3), "n_windows": wf["n_windows"]}

html = open(d+"/tpl.html").read().replace("__DATA__", json.dumps(DATA))
open(d+"/diagnostic.html","w").write(html)
print("wrote diagnostic.html", len(html), "bytes")
