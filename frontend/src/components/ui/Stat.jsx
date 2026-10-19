/** Stat — a compact labelled value cell. */
export default function Stat({ label, value, color, sub }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[9px] uppercase tracking-[0.15em] text-muted">{label}</span>
      <span className="text-sm font-semibold tabular-nums" style={color ? { color } : undefined}>
        {value}
      </span>
      {sub && <span className="text-[9px] text-muted">{sub}</span>}
    </div>
  );
}
