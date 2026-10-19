import Plot from "react-plotly.js";

// Tuned bandwidths (esp. a narrow delta bandwidth) make P(up) saturate near
// 0/1 fast — correct, confident model behavior, but on a linear 0-1 color
// scale it reads as two solid blocks with a thin transition band. Coloring
// by logit(p) instead stretches that near-0%/near-100% territory back out
// (logit(0.95)=2.9, logit(0.995)=5.3 — clearly different shades, whereas
// linear-0-1 would map both to "basically green") without touching the
// model or the probabilities themselves — display-only.
const LOGIT_CLAMP = 6; // logit(p) at this magnitude ~= p of 0.25% / 99.75%
const EPS = 1e-4;

function logit(p) {
  const c = Math.min(Math.max(p, EPS), 1 - EPS);
  const v = Math.log(c / (1 - c));
  return Math.min(Math.max(v, -LOGIT_CLAMP), LOGIT_CLAMP);
}

// Colorbar tick landmarks, computed in probability space and placed at their
// logit position — so the bar still reads in %, even though the mapping
// underneath it isn't linear.
const COLORBAR_STOPS = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99];
const COLORBAR_TICKVALS = COLORBAR_STOPS.map(logit);
const COLORBAR_TICKTEXT = COLORBAR_STOPS.map((p) => `${Math.round(p * 100)}%`);

/**
 * The hero. A P(up) heatmap over (elapsed time × price delta), with a dotted
 * 50% decision contour, and the current window's path riding across it — the
 * "ride the wave" view. Green = up-lean, red = down-lean, dark at 0.5.
 * Colored on a logit scale (see LOGIT_CLAMP above) — hover still shows the
 * real probability via customdata, only the color mapping is transformed.
 *
 * Props: matrix {x: time[], y: delta[], z: p_up[][]}, tick {x[], y[], ...}, accent.
 */
export default function ProbabilitySurface({ matrix, tick, accent = "#38bdf8" }) {
  if (!matrix?.z) {
    return (
      <div className="grid h-full place-items-center text-sm text-muted">
        waiting for surface…
      </div>
    );
  }

  const zLogit = matrix.z.map((row) => row.map(logit));

  const data = [
    {
      type: "heatmap",
      x: matrix.x, y: matrix.y, z: zLogit, customdata: matrix.z,
      zmin: -LOGIT_CLAMP, zmax: LOGIT_CLAMP, zsmooth: "best",
      colorscale: [[0, "#ff4d6d"], [0.5, "#0d1117"], [1, "#00ff87"]],
      colorbar: {
        thickness: 8, len: 0.9, tickfont: { size: 9, color: "#94a3b8" }, outlinewidth: 0,
        tickvals: COLORBAR_TICKVALS, ticktext: COLORBAR_TICKTEXT,
      },
      hovertemplate: "τ=%{x}s  δ=%{y}bps<br>P(up)=%{customdata:.2f}<extra></extra>",
    },
    {
      type: "contour",
      x: matrix.x, y: matrix.y, z: matrix.z,
      showscale: false, contours: { coloring: "none", start: 0.5, end: 0.5, size: 1 },
      line: { color: "#e2e8f0", width: 1, dash: "dot" }, hoverinfo: "skip",
    },
  ];

  if (tick?.x?.length) {
    const n = tick.x.length;
    data.push({
      type: "scatter", mode: "lines", x: tick.x, y: tick.y,
      line: { color: accent, width: 2 }, hoverinfo: "skip",
    });
    data.push({
      type: "scatter", mode: "markers", x: [tick.x[n - 1]], y: [tick.y[n - 1]],
      marker: { size: 13, color: "#ffffff", line: { color: accent, width: 3 } },
      hovertemplate: `now: τ=%{x}s δ=%{y}bps<extra></extra>`,
    });
  }

  const layout = {
    paper_bgcolor: "transparent", plot_bgcolor: "#0d1117",
    margin: { l: 46, r: 10, t: 8, b: 38 }, showlegend: false,
    xaxis: { title: { text: "elapsed (s)", font: { size: 10 } }, range: [0, 300], color: "#475569", gridcolor: "#1e2937", zeroline: false },
    yaxis: { title: { text: matrix?.y_axis || tick?.y_axis || "δ (bps)", font: { size: 10 } }, color: "#475569", gridcolor: "#1e2937", zeroline: true, zerolinecolor: "#334155" },
    font: { family: "monospace", size: 10, color: "#94a3b8" },
  };

  return (
    <Plot
      data={data} layout={layout} useResizeHandler
      style={{ width: "100%", height: "100%" }}
      config={{ displayModeBar: false, responsive: true }}
    />
  );
}
