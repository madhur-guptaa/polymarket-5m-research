import { pct } from "../../utils/format";

/** Live Polymarket market: YES/NO prices, spread, and per-side edge. */
export default function PolymarketPanel({ poly, edge }) {
  if (!poly) {
    return <div className="grid h-full place-items-center text-[11px] text-muted">no active market</div>;
  }
  const rows = [
    ["YES (up)", pct(poly.yes_price), "var(--color-up)"],
    ["NO (down)", pct(poly.no_price), "var(--color-down)"],
    ["Spread", poly.spread != null ? pct(poly.spread) : "—", undefined],
    ["Mkt P(up)", pct(poly.market_prob_up), undefined],
  ];
  return (
    <div className="flex flex-col gap-2">
      {poly.title && <div className="truncate text-[10px] text-muted" title={poly.title}>{poly.title}</div>}
      {rows.map(([k, v, c]) => (
        <div key={k} className="flex items-center justify-between border-b border-border/40 py-1 text-[11px]">
          <span className="uppercase tracking-[0.1em] text-muted">{k}</span>
          <span className="font-semibold tabular-nums" style={c ? { color: c } : undefined}>{v}</span>
        </div>
      ))}
      {edge && (
        <div className="mt-1 grid grid-cols-2 gap-2 text-[11px]">
          <div className="rounded bg-up/10 px-2 py-1">
            <div className="text-[9px] uppercase tracking-wide text-muted">edge yes</div>
            <div className="font-semibold tabular-nums" style={{ color: "var(--color-up)" }}>
              {`${edge.edge_yes >= 0 ? "+" : ""}${(edge.edge_yes * 100).toFixed(1)}%`}
            </div>
          </div>
          <div className="rounded bg-down/10 px-2 py-1">
            <div className="text-[9px] uppercase tracking-wide text-muted">edge no</div>
            <div className="font-semibold tabular-nums" style={{ color: "var(--color-down)" }}>
              {`${edge.edge_no >= 0 ? "+" : ""}${(edge.edge_no * 100).toFixed(1)}%`}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
