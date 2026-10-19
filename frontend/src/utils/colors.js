/**
 * colors.js
 *
 * Color mapping and label generation for the Polaris Quant Engine dashboard.
 *
 * Translates raw numerical feature values from the backend into human-readable
 * labels and color codes for the UI. Used by every visualization component.
 *
 * COLOR PALETTE:
 *   #00ff87  — Green (bullish / low volatility / positive signal)
 *   #ff4d6d  — Red (bearish / high volatility / negative signal)
 *   #f59e0b  — Amber (neutral volatility)
 *   #94a3b8  — Slate (neutral / indeterminate)
 *   #334155  — Dark slate (no data / null)
 *   #9945ff  — Purple (Asia session / SOL)
 *   #627eea  — Blue (Europe session / ETH)
 *   #f7931a  — Orange (US session / BTC)
 *
 * THRESHOLD DESIGN:
 *   Colors use WIDER thresholds than labels intentionally.
 *   Example for probability:
 *     Color: >0.55 = green, <0.45 = red (broad signal)
 *     Label: >0.65 = "STRONG BULL", >0.55 = "MILD BULL", etc. (detailed)
 *   This means the color changes before the label escalates, giving a visual
 *   "early warning" effect on the dashboard.
 *
 * NULL HANDLING:
 *   All functions return a safe default (#334155 for colors, "—" for labels)
 *   when the input is null, undefined, or NaN. This prevents rendering errors
 *   when the backend hasn't computed a feature yet (e.g., first 9 seconds of
 *   a window when volatility is null).
 */

// ═══════════════════════════════════════════════════════════════════
// PROBABILITY (P(Up) from the Bayesian estimator)
// ═══════════════════════════════════════════════════════════════════

/** Color for P(Up) — green above 0.55, red below 0.45, neutral between. */
export const probColor = (p) => {
    if (p == null || isNaN(p)) return "#334155";
    if (p > 0.55) return "#00ff87";
    if (p < 0.45) return "#ff4d6d";
    return "#94a3b8";
};

/** Descriptive label for P(Up) — more granular than probColor. */
export const probLabel = (p) => {
    if (p == null || isNaN(p)) return "NO MODEL";
    if (p > 0.65) return "STRONG BULL";
    if (p > 0.55) return "MILD BULL";
    if (p < 0.35) return "STRONG BEAR";
    if (p < 0.45) return "MILD BEAR";
    return "NEUTRAL";
};

// ═══════════════════════════════════════════════════════════════════
// MOMENTUM (basis points, from compute_momentum_bps)
// ═══════════════════════════════════════════════════════════════════

/** Color for momentum — green above +1.5 bps, red below -1.5 bps. */
export const momColor = (m) => {
    if (m == null || isNaN(m)) return "#334155";
    if (m > 1.5) return "#00ff87";
    if (m < -1.5) return "#ff4d6d";
    return "#94a3b8";
};

/** Descriptive label for momentum — more granular than momColor. */
export const momLabel = (m) => {
    if (m == null || isNaN(m)) return "—";
    if (m > 3) return "STRONG ↑";
    if (m > 1) return "MILD ↑";
    if (m < -3) return "STRONG ↓";
    if (m < -1) return "MILD ↓";
    return "FLAT";
};

// ═══════════════════════════════════════════════════════════════════
// VOLATILITY (log-transformed, from compute_log_volatility)
// ═══════════════════════════════════════════════════════════════════
//
// log_volatility values (typical range):
//   > -5   : High volatility (red — caution)
//   -8 to -5 : Normal volatility (amber)
//   < -8   : Low volatility (green — calm market)
//
// Note: More negative = lower volatility (log scale).

/** Descriptive label for log-volatility. */
export const volLabel = (logVol) => {
    if (logVol == null || isNaN(logVol)) return "—";
    if (logVol > -5) return "HIGH VOL";
    if (logVol > -8) return "NORMAL";
    return "LOW VOL";
};

/** Color for log-volatility — red for high (risky), green for low (calm). */
export const volColor = (logVol) => {
    if (logVol == null || isNaN(logVol)) return "#334155";
    if (logVol > -5) return "#ff4d6d";
    if (logVol > -8) return "#f59e0b";
    return "#00ff87";
};

// ═══════════════════════════════════════════════════════════════════
// TRADING SESSION (from time-of-day sin/cos encoding)
// ═══════════════════════════════════════════════════════════════════
//
// The backend encodes UTC hour as cyclical features:
//   tod_sin = sin(2π × hour / 24)
//   tod_cos = cos(2π × hour / 24)
//
// This function REVERSES the encoding to recover the UTC hour:
//   angle = atan2(sin, cos)    → radians [-π, π]
//   hour  = angle / (2π) × 24  → UTC hour [0, 24)
//
// Session boundaries (UTC):
//   ASIA    :  0:00 –  8:00 UTC  (Tokyo, Shanghai, Sydney)
//   EUROPE  :  8:00 – 14:00 UTC  (London, Frankfurt)
//   US      : 14:00 – 21:00 UTC  (New York, Chicago)
//   ASIA    : 21:00 – 24:00 UTC  (overlap, treated as Asia)

/** Recover UTC trading session from sin/cos time-of-day features. */
export const sessionLabel = (todSin, todCos) => {
    if (todSin == null || todCos == null || isNaN(todSin) || isNaN(todCos)) return "—";

    // Reverse the cyclical encoding: atan2(sin, cos) → angle in radians
    let angle = Math.atan2(todSin, todCos);
    if (angle < 0) angle += 2 * Math.PI; // Normalize to [0, 2π)

    const hour = (angle / (2 * Math.PI)) * 24;

    if (hour >= 0 && hour < 8) return "ASIA";
    if (hour >= 8 && hour < 14) return "EUROPE";
    if (hour >= 14 && hour < 21) return "US";
    return "ASIA"; // 21:00–24:00 UTC wraps to Asia
};

/** Color for the current trading session. */
export const sessionColor = (todSin, todCos) => {
    const label = sessionLabel(todSin, todCos);
    if (label === "ASIA") return "#9945ff";
    if (label === "EUROPE") return "#627eea";
    if (label === "US") return "#f7931a";
    return "#334155";
};
