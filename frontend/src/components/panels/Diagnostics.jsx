/** Kernel bandwidths + a few window/model diagnostics (tertiary detail). */

// Display label per bandwidth key. The backend sends whichever set the serving
// model has (MODEL_KIND), so rows are built from the keys actually present
// rather than a fixed list -- a fixed list rendered all dashes the moment the
// model changed.
const BW_LABELS = {
  // kde_twap -- the TWAP-geometry KDE (production default)
  twap_sofar: "h · twap",
  req: "h · req",
  locked: "h · lock",
  spot: "h · spot",
  // kde -- the original 6-feature baseline
  delta: "h · δ",
  time: "h · τ",
  momentum: "h · μ",
  imbalance: "h · c",
  volatility: "h · σ",
  tod: "h · φ",
};

const MODEL_LABELS = {
  kde_twap: "KDE · TWAP geometry",
  kde: "KDE · 6-feature",
  gbm: "GBM",
};

export default function Diagnostics({ tick, matrix, symbol }) {
  const bw = tick?.bandwidths || matrix?.bandwidths || {};
  const kind = tick?.model_kind || matrix?.model_kind;
  const fmtBw = (v) => (v != null && !isNaN(v) ? Number(v).toFixed(3) : "—");
  const rows = [
    ["symbol", symbol],
    ["model", MODEL_LABELS[kind] || kind || "—"],
    ["obs", matrix?.n_observations ?? "—"],
    ["windows", matrix?.n_windows ?? "—"],
    ...Object.entries(bw).map(([k, v]) => [BW_LABELS[k] || `h · ${k}`, fmtBw(v)]),
  ];
  return (
    <div className="grid grid-cols-2 gap-x-4">
      {rows.map(([k, v]) => (
        <div key={k} className="flex items-center justify-between border-b border-border/40 py-[3px] text-[10px]">
          <span className="uppercase tracking-[0.1em] text-muted">{k}</span>
          <span className="tabular-nums text-fg">{v}</span>
        </div>
      ))}
    </div>
  );
}
