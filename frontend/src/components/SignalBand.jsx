/**
 * SignalBand — the hero readout above the surface:
 *   P(up) big & colored | 95% CI bar + feature stats | EDGE + trade gate | n_eff + window timer
 * The edge/trade block is a READOUT only (no execution).
 */

import { bps, fmt, pct } from "../utils/format";
import { probColor, probLabel, momColor, sessionColor, sessionLabel } from "../utils/colors";
import Stat from "./ui/Stat";

const WINDOW_SEC = 300;

/** 0..1 probability bar with the 95% CI band, mean marker, and 50% reference. */
function CIBar({ lower, upper, mean }) {
  if (mean == null) return <div className="h-1.5 w-full rounded bg-border" />;
  const L = Math.max(0, lower ?? mean) * 100;
  const U = Math.min(1, upper ?? mean) * 100;
  const M = mean * 100;
  return (
    <div className="relative h-1.5 w-full rounded bg-border">
      <div className="absolute inset-y-0 rounded bg-accent/40" style={{ left: `${L}%`, width: `${Math.max(0, U - L)}%` }} />
      <div className="absolute inset-y-0 w-px bg-muted" style={{ left: "50%" }} />
      <div className="absolute inset-y-[-2px] w-0.5 bg-fg" style={{ left: `${M}%` }} />
    </div>
  );
}

export default function SignalBand({ tick, edge }) {
  const prob = tick?.prob;
  const col = probColor(prob);
  const nEff = tick?.n_eff;
  const elapsed = tick?.x?.length ? tick.x[tick.x.length - 1] : null;
  const ciWidth = tick?.lower_95 != null && tick?.upper_95 != null ? tick.upper_95 - tick.lower_95 : null;
  const progress = elapsed != null ? Math.min(100, (elapsed / WINDOW_SEC) * 100) : 0;

  const edgeVal = edge?.edge;
  const trade = edge?.trade;
  const edgeCol = edgeVal == null ? "var(--color-muted)" : trade ? "var(--color-up)" : "var(--color-fg)";

  return (
    <div className="flex shrink-0 items-stretch gap-4 rounded-lg border border-border bg-panel px-4 py-3">
      {/* hero P(up) */}
      <div className="flex min-w-[130px] flex-col justify-center border-r border-border pr-4">
        <span className="text-[9px] uppercase tracking-[0.15em] text-muted">P(up)</span>
        <span className="text-4xl font-bold leading-none tabular-nums" style={{ color: col }}>
          {prob != null ? (prob * 100).toFixed(1) : "—"}<span className="text-lg">%</span>
        </span>
        <span className="mt-1 text-[10px] font-semibold" style={{ color: col }}>{probLabel(prob)}</span>
      </div>

      {/* CI + feature stats */}
      <div className="flex flex-1 flex-col justify-center gap-2">
        <CIBar lower={tick?.lower_95} upper={tick?.upper_95} mean={prob} />
        <div className="grid grid-cols-3 gap-x-6 gap-y-2 sm:grid-cols-6">
          <Stat label="Price" value={fmt(tick?.current_price, 2)} />
          <Stat label="Δ (bps)" value={bps(tick?.delta_bps)} />
          <Stat label="Momentum" value={bps(tick?.momentum_bps)} color={momColor(tick?.momentum_bps)} />
          <Stat label="Imbalance" value={pct(tick?.cum_imbalance)} />
          <Stat label="95% CI" value={ciWidth != null ? pct(ciWidth) : "—"} />
          <Stat label="Session" value={sessionLabel(tick?.tod_sin, tick?.tod_cos)} color={sessionColor(tick?.tod_sin, tick?.tod_cos)} />
        </div>
      </div>

      {/* edge + trade gate */}
      <div className="flex min-w-[140px] flex-col justify-center gap-1 border-l border-border pl-4">
        <span className="text-[9px] uppercase tracking-[0.15em] text-muted">Edge vs market</span>
        <span className="text-3xl font-bold leading-none tabular-nums" style={{ color: edgeCol }}>
          {edgeVal != null ? `${edgeVal >= 0 ? "+" : ""}${(edgeVal * 100).toFixed(1)}%` : "—"}
          {edge?.side && <span className="ml-1 text-xs text-muted">{edge.side}</span>}
        </span>
        <span
          className="mt-1 w-fit rounded px-2 py-0.5 text-[10px] font-bold tracking-wide"
          style={
            edgeVal == null
              ? { background: "#ffffff08", color: "var(--color-muted)" }
              : trade
              ? { background: "var(--color-up)", color: "#04110a" }
              : { background: "#ffffff10", color: "var(--color-muted)" }
          }
        >
          {edgeVal == null
            ? "NO MARKET"
            : trade
            ? "TRADE ▲"
            : edge?.in_window === false
            ? "WAIT · OUT OF WINDOW"
            : "NO TRADE"}
        </span>
        {edge?.trade_window && (
          <span className="text-[9px] text-muted">trade window {edge.trade_window[0]}–{edge.trade_window[1]}s</span>
        )}
      </div>

      {/* n_eff gate + window timer */}
      <div className="flex min-w-[130px] flex-col justify-center gap-2 border-l border-border pl-4">
        <div className="flex items-center justify-between">
          <span className="text-[9px] uppercase tracking-[0.15em] text-muted">n_eff</span>
          <span className="text-sm font-semibold tabular-nums" style={{ color: nEff >= 20 ? "var(--color-up)" : "var(--color-down)" }}>
            {nEff != null ? Math.round(nEff) : "—"}
          </span>
        </div>
        <div>
          <div className="flex items-center justify-between">
            <span className="text-[9px] uppercase tracking-[0.15em] text-muted">Window</span>
            <span className="text-[10px] tabular-nums text-muted">{elapsed != null ? `${elapsed}s / ${WINDOW_SEC}s` : "—"}</span>
          </div>
          <div className="mt-1 h-1.5 w-full rounded bg-border">
            <div className="h-full rounded bg-accent" style={{ width: `${progress}%` }} />
          </div>
        </div>
      </div>
    </div>
  );
}
