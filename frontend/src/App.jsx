/**
 * App.jsx — dashboard shell.
 *   Part 1: header + probability surface (centerpiece).
 *   Part 2: signal band + context panels.
 *   Part 3: live Polymarket + edge (in the signal band) + diagnostics.
 * (Backtest / calibration panels await the walk-forward backtester.)
 */

import { useEffect, useState } from "react";
import { API_BASE, SYMBOLS, SYMBOL_COLORS, fallbackColor } from "./utils/constants";
import { useWebSocket } from "./hooks/useWebSocket";

import Header from "./components/Header";
import SignalBand from "./components/SignalBand";
import Panel from "./components/ui/Panel";
import ProbabilitySurface from "./components/ProbabilitySurface";
import OrderFlow from "./components/panels/OrderFlow";
import FeatureState from "./components/panels/FeatureState";
import VolumeProfile from "./components/panels/VolumeProfile";
import MacroTrend from "./components/panels/MacroTrend";
import PolymarketPanel from "./components/panels/PolymarketPanel";
import Diagnostics from "./components/panels/Diagnostics";

export default function App() {
  const [symbols, setSymbols] = useState(SYMBOLS);
  const [symbol, setSymbol] = useState("BTCUSDT");
  const { connected, tick, matrix, micro, macro, poly, edge } = useWebSocket(symbol);

  useEffect(() => {
    fetch(`${API_BASE}/symbols`)
      .then((r) => r.json())
      .then((d) => { if (d.symbols?.length) setSymbols(d.symbols); })
      .catch(() => {});
  }, []);

  const c = SYMBOL_COLORS[symbol] || fallbackColor;

  return (
    <div className="flex h-full min-h-0 flex-col gap-3 p-4">
      <Header symbols={symbols} symbol={symbol} onSelect={setSymbol} connected={connected} />
      <SignalBand tick={tick} edge={edge} />

      <div className="grid min-h-0 flex-1 grid-cols-[260px_1fr_270px] gap-3">
        {/* left: microstructure */}
        <div className="flex min-h-0 flex-col gap-3">
          <Panel title="Live Order Flow" className="min-h-0 flex-1">
            <OrderFlow trades={tick?.trades} accent={c.primary} />
          </Panel>
          <Panel title="10s Volume" className="h-40 shrink-0">
            <VolumeProfile data={micro} accent={c.primary} />
          </Panel>
        </div>

        {/* center: the surface */}
        <Panel title={`Probability Surface · ${symbol}`} className="min-h-0">
          <div className="h-full min-h-[440px]">
            <ProbabilitySurface matrix={matrix} tick={tick} accent={c.primary} />
          </div>
        </Panel>

        {/* right: market + model context */}
        <div className="flex min-h-0 flex-col gap-3 overflow-y-auto">
          <Panel title="Polymarket" className="shrink-0">
            <PolymarketPanel poly={poly} edge={edge} />
          </Panel>
          <Panel title="Feature State" className="shrink-0">
            <FeatureState tick={tick} />
          </Panel>
          <Panel title="Diagnostics" className="shrink-0">
            <Diagnostics tick={tick} matrix={matrix} symbol={symbol} />
          </Panel>
          <Panel title="1H Trend" className="h-36 shrink-0">
            <MacroTrend data={macro} accent={c.primary} />
          </Panel>
        </div>
      </div>
    </div>
  );
}
