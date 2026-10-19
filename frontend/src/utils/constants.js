/**
 * constants.js — connection config + symbol branding.
 *
 * URLs default to same-origin relative paths, which the Vite dev server proxies
 * to the FastAPI backend on :8000 (see vite.config.js). Override in production
 * via VITE_API_BASE / VITE_WS_BASE.
 */

export const API_BASE = import.meta.env.VITE_API_BASE || "/api";

// WebSocket base. Relative by default → resolved against the current origin
// (dev: proxied to :8000; prod: same origin as the served frontend).
export function wsBase() {
  const env = import.meta.env.VITE_WS_BASE;
  if (env) return env;
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}/ws`;
}

// Fallback symbol list; the authoritative list comes from GET /api/symbols.
export const SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"];

export const SYMBOL_COLORS = {
  BTCUSDT: { primary: "#f7931a", label: "BTC" },
  ETHUSDT: { primary: "#627eea", label: "ETH" },
  SOLUSDT: { primary: "#9945ff", label: "SOL" },
  XRPUSDT: { primary: "#00aae4", label: "XRP" },
};

export const fallbackColor = { primary: "#94a3b8", label: "?" };
