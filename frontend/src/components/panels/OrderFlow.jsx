import { fmt, fmtTime } from "../../utils/format";

/** Live trade tape — most recent first, freshest row accented. */
export default function OrderFlow({ trades = [], accent }) {
  return (
    <div className="flex h-full flex-col gap-1 text-[11px]">
      <div className="grid grid-cols-3 border-b border-border pb-1 text-[9px] uppercase tracking-[0.1em] text-muted">
        <span>Price</span>
        <span>Size</span>
        <span className="text-right">Time</span>
      </div>
      <div className="flex flex-col gap-1 overflow-hidden">
        {trades.slice(0, 14).map((t, i) => (
          <div key={i} className="grid grid-cols-3 tabular-nums" style={{ opacity: 1 - i * 0.05 }}>
            <span style={{ color: i === 0 ? accent : undefined, fontWeight: i === 0 ? 700 : 400 }}>{fmt(t.price, 2)}</span>
            <span className="text-muted">{fmt(t.quantity, 4)}</span>
            <span className="text-right text-muted">{fmtTime(t.trade_time)}</span>
          </div>
        ))}
        {!trades.length && <span className="text-muted">no trades yet</span>}
      </div>
    </div>
  );
}
