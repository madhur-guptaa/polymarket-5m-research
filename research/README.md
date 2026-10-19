# research/

The analysis scripts behind the findings in [`../RESEARCH_LOG.md`](../RESEARCH_LOG.md) and
[`../WHITEPAPER.md`](../WHITEPAPER.md) §9. The reusable parts live in the package
(`polaris/model/featurelab.py`, `polaris/late/makersim.py`, `polaris/report.py`, …); these are the
drivers that produced the numbers.

**These are as-run snapshots, not maintained tools.** Several predate later library changes —
notably `load_dataset` now trains on real settlements by default and `MODEL_KIND` defaults to
`kde_twap` — so re-running an early script reproduces the method, not necessarily the exact
number in the log. Each entry names the log session that records the result it produced.

## Running

From the repo root, with the database up:

```sh
mkdir -p research/.cache                       # git-ignored scratch for intermediate pickles
PYTHONPATH=. .venv/bin/python research/05_maker/msim.py
```

Scripts that consume a pickle say so below; run the producer first.

## Index

### 01_forecast — can the model out-forecast the market? (sessions 10–21)
| script | what it did |
|---|---|
| `headtohead.py` | KDE vs GBM through the identical walk-forward harness |
| `gatesweep.py` | where the edge "lived" by trade gate, on real prices |
| `livevsbt.py` | backtested the exact windows the live paper trade covered; live vs backtest `p_up` correlated 0.9986 — proved the September result was a *data* failure |
| `oot.py`, `oot14.py` | out-of-time tests on never-before-available windows |
| `diag_time.py`, `diag_wf.py`, `gen_artifact.py` | per-bucket and walk-forward diagnostics, and the HTML artifact built from them |

### 02_labels — the label was wrong (session 23)
| script | what it did |
|---|---|
| `fetch_res.py` | fetched Polymarket's real settlements at scale; exposed the 8–12% Binance-label disagreement. Superseded by `polaris backfill-resolutions` |

### 03_twap_model — re-targeting at the contract's geometry (sessions 24–29)
| script | what it did |
|---|---|
| `post_aug7.py` | clean-regime re-test; BTC's TWAP gain tripled (.0053 → .0173) |
| `accept.py` | acceptance test: old vs new KDE, equal tuning budget, real settlement, vs the market |
| `bwcheck.py` | widened bandwidth bounds; spot + locked alone match all four dimensions |
| `bt_all.sh` | real-book walk-forward backtests, both models, both symbols, spread sweep |

### 04_late_window — trading the settling TWAP (sessions 30, 35)
| script | what it did | input |
|---|---|---|
| `late.py` | first late-window estimator; measured the market's late calibration | DB → `cl.pkl`, `bk_late.pkl`, `res.pkl` |
| `late2.py` | structural model (Binance underlying, Chainlink anchor), isotonic-calibrated; the t ≥ 270 pocket | those pickles |
| `late3.py` | stress tests: latency, depth, per-symbol, held-out half | `late_struct.pkl` (from `late2.py`) |
| `late_live.py` | **the verdict**: pre-registered rule on live Chainlink + live book, held-out half, with latency — −5.4¢/share | DB |

### 05_maker — market making (sessions 31–32)
| script | what it did | input |
|---|---|---|
| `maker.py` | historical decomposition: maker P&L monotone in model edge; leakage/latency test | DB |
| `msim.py` | loads overnight live data for the 1-second simulator | DB → `msim_data.pkl` |
| `msim_grid.py` | 1-second queue-aware simulation grid (all 8 coins, Chainlink fair value) | `msim_data.pkl` |
| `fastfv.py` | builds the Binance-led fair value; shows it leads the book and Chainlink-only lags | DB → `sig_fast.pkl` |
| `msim_fast.py`, `msim_q.py` | 1-second simulation with the fast fair value; fill-type split, front/back queue bounds | `sig_fast.pkl` |
| `fast_prep.py` | prepares the millisecond replay (touch log, fills, fair value) | DB → `fast_data.pkl` |
| `fast_grid.py` | **the verdict**: event-driven ms replay, reaction × theta grid — no maker edge | `fast_data.pkl` |

### 06_cross_series — 5m vs 15m pairs (session 34)
| script | what it did | input |
|---|---|---|
| `cross.py` | loads `cross_signals`, fetches 15m settlements | DB → `cross.pkl` |
| `cross2.py` | the logical link (0 violations), riskless pair frequency, and executability at 1–5 s latency | `cross.pkl` |
