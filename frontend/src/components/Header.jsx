import { SYMBOL_COLORS, fallbackColor } from "../utils/constants";

/** App header: wordmark, symbol tabs, live/offline status. */
export default function Header({ symbols, symbol, onSelect, connected }) {
  return (
    <header className="flex shrink-0 items-center justify-between">
      <div className="flex items-baseline gap-2">
        <span className="text-lg font-bold tracking-[0.2em]">POLARIS</span>
        <span className="text-[10px] tracking-[0.2em] text-muted">// PROBABILITY SURFACE</span>
      </div>

      <div className="flex items-center gap-2">
        {symbols.map((s) => {
          const c = SYMBOL_COLORS[s] || fallbackColor;
          const active = s === symbol;
          return (
            <button
              key={s}
              onClick={() => onSelect(s)}
              className={`rounded-md border px-3 py-1 text-xs font-semibold transition ${active ? "" : "opacity-50 hover:opacity-80"}`}
              style={{
                borderColor: c.primary + "55",
                background: active ? c.primary + "1a" : "transparent",
                color: c.primary,
              }}
            >
              {c.label}
            </button>
          );
        })}
      </div>

      <div className="flex items-center gap-2 text-xs">
        <span className={`h-2 w-2 rounded-full ${connected ? "bg-up" : "bg-down"}`} />
        <span className="text-muted">{connected ? "LIVE" : "OFFLINE"}</span>
      </div>
    </header>
  );
}
