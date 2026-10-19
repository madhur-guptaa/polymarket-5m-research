# Seven ways we fooled ourselves trying to beat Polymarket's 5-minute markets

*October 2026*

For two months we had a trading strategy that was statistically significant. Scored one way, our live paper
trades showed an edge at **z = +2.53**. Scored against what the market actually paid out, the *same trades*
came in at **z = −4.41**.

Nothing about the trades changed. Only the label did.

This is a write-up of every way our research manufactured an edge that wasn't there — seven of them, each
caught by a specific test — and of the one real inefficiency that survived. If you build backtests, at least
one of these is probably in yours.

---

## The setup

Polymarket runs binary markets on whether Bitcoin will be higher or lower at the end of each five-minute
window. Each window is a contract that pays $1 if you picked the right side. Thousands of them trade every day,
mostly against bots.

Our idea was ordinary: use Binance order flow — price drift, momentum, buy/sell imbalance, volatility — to
estimate P(up) at every second of the window, compare it with the market's price, and trade when we disagreed
enough. We built it properly: a product-kernel Bayesian estimator, a gradient-boosted challenger, block
cross-validation so the ~300 correlated seconds of one window could not leak into each other, walk-forward
backtests, a live paper-trading recorder, a dashboard. Nothing ever placed a real order.

The first headline was **a 60.9% win rate over 12,571 backtested trades — roughly 24 standard deviations
above chance**. A stricter out-of-time test on days that didn't exist when we tuned it said 62.7%.

It was all wrong. Here is how.

---

## 1. Prices you could never have traded at

Our historical market prices came from Polymarket's `/prices-history` endpoint. It returns **one point per
minute**. We forward-filled them to fill every second.

So a model reacting to fresh Binance data was being compared, second by second, with a market price that was
up to a minute stale. Inside our trade window those filled prices sat a mean **12.9¢** away from the real order
book. "Edge" was mostly *the model knowing what happened since the last price print*.

**How we caught it.** Live paper trading quoted real prices and was far weaker. But the model's live and
backtest probabilities correlated at **0.9986** — so the model wasn't the problem; the data was. The tell had
been in plain sight: the backtest found a tradeable signal in **~99% of windows**. On the real order book,
**6.5%**. A strategy that trades almost every window is usually measuring an artifact, not an edge.

On real prices: **52.5% on 101 trades.** Coin flip.

> **Rule:** check the resolution of every input. A price series with fewer distinct values than your decision
> cadence is a lagged series, and a lagged price against a fresh signal is lookahead in disguise.

## 2. Unbiased noise, biased results

Next we tried individual trade prints as the price. These were genuinely second-resolution and, on average,
honest: their mean error against the book was **+0.06¢**.

They still produced a fake edge. A trading rule only acts when the model and the price *disagree* by more
than a threshold — and noise is exactly what makes them disagree. The filter selected the seconds where noise
had made a contract look cheap. Zero-mean noise in, positive-mean "edge" out.

> **Rule:** "the errors average to zero" does not protect you when your strategy *selects* on the error. Any
> threshold rule turns symmetric noise into a one-sided bias.

## 3. The wrong label — the big one

Our outcome label was the obvious one: did the Binance price end the window higher than it started?

Polymarket doesn't settle on Binance. Each market's own description says it resolves on **Chainlink's
time-weighted average price**. When we finally pulled the market's actual settlements and compared, our
label disagreed on **8–12% of windows**. Worse, the errors were not random:

| Binance move over the window | our label wrong |
|---|---|
| under 1 bp | **27.9%** |
| 1–2 bp | 12.6% |
| 5–10 bp | 1.5% |
| over 20 bp | 0.1% |

The label was right when the answer was obvious and close to a coin flip when it was hard — and the hard
windows are the only ones where a model can make money.

It did more than add noise. Grading Polymarket's prices against *our* label made the market look terrible: a
Brier score of 0.201. Against the real outcomes, the market scored **0.159**. Our model, which had appeared to
"beat the market as a forecaster", was in fact **behind it** (0.187 vs 0.159, t = −5.6). And the live paper
trades went from **z = +2.53** to **z = −4.41** — return on capital from +7.3% to −12.7%.

The bad label wasn't hiding a loss behind noise. As the sample grew, it was making the *fake* result more
significant.

**How we caught it.** An anomaly in the last seconds of each window. When the market priced a side above 90¢
twenty seconds before resolution, it was "right" by our label only 57% of the time. A real market that close
to settlement should be right more than nine times in ten. Rather than celebrating a market that seemed
clueless at the finish, we checked our ruler.

The settlement rule turned out to be: Up if Chainlink's 60-second TWAP at the window's last second is at least
its value at the first — 99.79% agreement over 2,421 windows. Polymarket had moved from a single price snapshot
to a time-weighted average on 7 August (30 seconds at first, 60 seconds from 14 August), after researchers
showed traders were pushing the price in the final seconds. Before the switch our Binance proxy agreed with
settlement 95.8% of the time; after it, only 88%. Our training data straddled the switch.

> **Rule:** read the contract before you model it, and grade against **what pays out**, never against a proxy
> for it. If the true outcome is available — and here it was, for every single window — use it.

## 4. Counting seconds as trades

Our first re-score of the live test reported **2,287 trades at z = −7.62**. Every second the signal fired in
a window had been counted as its own trade. But all those seconds share one outcome. The honest count — one
trade per window, the first second it fired — was **376 trades at z = −3.00**. Treating ~6 correlated
seconds as independent inflates z by about √6.

The same mistake, the other way round, inflates fake wins just as easily.

> **Rule:** the unit of evidence is the thing that has an independent outcome. In time series, cluster your
> errors at that level.

## 5. The fee nobody charged

Polymarket charges takers **0.07 × p × (1 − p)** per share: 1.75¢ on a 50¢ contract, close to 4% of the price.
We had modelled the spread carefully and never charged this fee, and every edge we had measured was about
the size of the fee.

With fees, the live results were **−15.0%** (z −5.6) for the original model and **−14.9%** for the improved
one.

> **Rule:** list every cost a real fill pays — spread, fee, slippage, latency — before you look at a single
> P&L number. Fees on these markets were introduced in January 2026 *specifically* to kill the kind of edge we
> were chasing.

## 6. A join that silently used a fifth of the data

Our real-order-book backtest reported about 500 trades per run and a trade rate of 6.5% of windows. The live
paper trader, on the same model, traded in ~90% of windows. We replayed the live signals offline: the two
correlated at 1.0000. So the model agreed with itself; the backtest was wrong.

It scored the model on every 5th second and **exact-matched** order-book snapshots — which are taken every ~5
seconds at an arbitrary offset. Only 20% of snapshots ever lined up: **598 of 2,677 windows**. The backtest
ran on a fifth of the data, and it traded at whichever second happened to align, not the first second the
signal appeared. A "+0.6¢ per trade" result on Bitcoin vanished once the join was fixed and the sample was
five times larger.

> **Rule:** when two systems disagree, find out which one is wrong before you pick the one you like. And
> assert row counts after every join.

## 7. Fills that happened are not fills you would get

The most seductive result came last. Across 6.2 million historical fills, makers' profit rose steadily with
our model's opinion: **−3.9¢ per share where the model disagreed most, +3.4¢ where it agreed most**. That
looked like a recipe — post orders only on the side the model favours.

Two checks took it apart.

**Lookahead inside a second.** The model at second *t* used that second's average Binance price, which
includes trades *after* the fill. Re-running it with the model delayed: +2.2¢ at zero delay, +1.8¢ at one
second, +1.25¢ at two, **+0.35¢ at five**. Partly real — but an edge that lives for two seconds.

**Selection on who got filled.** The historical fills went to whoever was first in the queue. We rebuilt the
strategy as a millisecond-by-millisecond replay — every book change, every trade, our order waiting its turn
and following the best bid after a realistic reaction delay. Result: at best **break-even**, even assuming
we were always first in line with a 50-millisecond reaction. Selecting by the model made it *worse*. Our
orders didn't get the good fills; they got the ones that arrive when the price runs over you.

> **Rule:** an analysis of trades that *happened* is not a simulation of trades *you* would have made.
> Queue position, reaction time and who gets filled are the strategy.

---

## What survived

After fees, with real prices and real outcomes, nothing we tried makes money on these markets:

| Strategy | Result |
|---|---|
| Forecast better than the market | still behind it |
| Trade the last 30 seconds, as the settling average locks in | −5.4¢ per share, live |
| Market making, naive or model-selected | break-even at best |
| Buy favourites / long shots | loses in every price range |
| Buy near-certain outcomes at 99¢ | break-even, no volume |
| Buy YES and NO for under $1 | 0.03% of snapshots |

The one genuine inefficiency was structural. The last 5-minute window of each 15-minute window settles on the
**same** Chainlink value, against a different anchor, so their prices are logically tied: a 15-minute YES plus
a 5-minute NO (or the mirror pair) always pays at least $1. We never saw the link violated in 698 windows. In
**3.6%** of them the pair cost under $1 after fees — truly riskless. But the median opportunity lasted **two
seconds**, offered about **eight shares**, and one second later only a quarter of them were still there. It's
real, and it belongs to whoever is fastest.

## The checklist we wish we'd started with

1. **Read the contract.** What exactly pays out, and when?
2. **Grade against the payout**, not a proxy. Use real outcomes wherever they exist.
3. **Check input resolution.** Forward-filled or bucketed data next to fresh data is lookahead.
4. **Distrust threshold rules on noisy prices** — they turn symmetric noise into one-sided bias.
5. **Cluster at the level of independent outcomes.**
6. **Charge every cost**, especially the ones introduced to stop exactly what you're doing.
7. **Assert row counts after joins.**
8. **Lag your signals** and watch how fast the edge dies.
9. **Simulate the fills you'd get**, not the fills that happened.
10. **Check if someone already published the data.** Polymarket's API reports Chainlink's start and settlement
    values for every window. We spent weeks reconstructing what was sitting in a field we never read.

Every positive result in this project was eventually traced to one of these. The useful habit wasn't any
single test — it was treating each new edge as a bug until a held-out sample, a lag test and a
leave-one-day-out all failed to kill it.

---

*The code, the research log (every session, including the retracted results) and the analysis scripts are on
GitHub: [madhur-guptaa/polymarket-5m-research](https://github.com/madhur-guptaa/polymarket-5m-research). Related work: Dai, Jia & Yu, "Settlement Manipulation in Prediction Markets" (2026);
Wang, "Does a Time-Weighted Average Price Deter Settlement Manipulation?" (2026).*
