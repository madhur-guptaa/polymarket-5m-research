cd "$(git rev-parse --show-toplevel)"
for SYM in BTCUSDT ETHUSDT; do for M in kde_twap kde; do
  echo "######## $SYM $M ########"
  .venv/bin/python -m polaris.main backtest --symbol $SYM --model $M --days 30 --price-source book \
     --min-edge 0.05 --gate-start 45 --gate-end 90 --cadence 5 --spread-sweep 0,0.0055,0.011 2>&1 \
     | grep -vE "httpx|PG pool|Schema"
done; done
