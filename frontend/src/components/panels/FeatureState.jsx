import { bps, pct } from "../../utils/format";
import { momColor, sessionColor, sessionLabel, volColor, volLabel } from "../../utils/colors";

/** The six model features, current values, colored by regime. */
export default function FeatureState({ tick }) {
  const elapsed = tick?.x?.length ? tick.x[tick.x.length - 1] : null;
  const lv = tick?.log_volatility;
  const rows = [
    ["Δ (bps)", bps(tick?.delta_bps), undefined, undefined],
    ["Elapsed", elapsed != null ? `${elapsed}s` : "—", undefined, undefined],
    ["Momentum", bps(tick?.momentum_bps), momColor(tick?.momentum_bps), undefined],
    ["Imbalance", pct(tick?.cum_imbalance), undefined, undefined],
    ["Log vol", lv != null ? lv.toFixed(2) : "—", volColor(lv), volLabel(lv)],
    ["Session", sessionLabel(tick?.tod_sin, tick?.tod_cos), sessionColor(tick?.tod_sin, tick?.tod_cos), undefined],
  ];
  return (
    <div className="flex flex-col">
      {rows.map(([k, v, color, sub]) => (
        <div key={k} className="flex items-center justify-between border-b border-border/40 py-[5px] text-[11px]">
          <span className="uppercase tracking-[0.1em] text-muted">{k}</span>
          <span className="flex items-baseline gap-2 tabular-nums">
            {sub && <span className="text-[9px] text-muted">{sub}</span>}
            <span className="font-semibold" style={color ? { color } : undefined}>{v}</span>
          </span>
        </div>
      ))}
    </div>
  );
}
