/**
 * format.js
 *
 * Number and time formatting utilities for the dashboard.
 *
 * All functions handle null/undefined/NaN gracefully, returning "—" (em dash)
 * as a universal placeholder. This prevents rendering errors when the backend
 * hasn't computed a value yet (e.g., during the "waiting" state before any
 * bars arrive, or when the estimator isn't ready).
 *
 * FUNCTIONS:
 *   fmt(n, d)       — Format a number with locale-aware thousands separators
 *   pct(n, d)       — Format a decimal as a percentage (0.5 → "50.0%")
 *   bps(n)          — Format basis points with sign (+3.2 bps, -1.5 bps)
 *   fmtTime(iso)    — Format an ISO timestamp as HH:MM:SS (24-hour)
 */

/**
 * Format a number with locale-aware thousands separators and fixed decimals.
 * @param {number|null} n - The number to format
 * @param {number} d - Decimal places (default 2)
 * @returns {string} Formatted number or "—" if null/NaN
 */
export const fmt = (n, d = 2) => n == null || isNaN(n) ? "—" : Number(n).toLocaleString("en-US", {
    minimumFractionDigits: d, maximumFractionDigits: d,
});

/**
 * Format a decimal probability/ratio as a percentage.
 * @param {number|null} n - The decimal to format (e.g., 0.5 → "50.0%")
 * @param {number} d - Decimal places for the percentage (default 1)
 * @returns {string} Formatted percentage or "—" if null/NaN
 */
export const pct = (n, d = 1) => n == null || isNaN(n) ? "—" : `${(Number(n) * 100).toFixed(d)}%`;

/**
 * Format a basis points value with explicit sign.
 * @param {number|null} n - Basis points value (e.g., 3.2 → "+3.2 bps")
 * @returns {string} Formatted bps or "—" if null/NaN
 */
export const bps = (n) => n == null || isNaN(n) ? "—" : `${n > 0 ? "+" : ""}${Number(n).toFixed(1)} bps`;

/**
 * Format an ISO timestamp as 24-hour time (HH:MM:SS).
 * @param {string|null} iso - ISO 8601 timestamp string
 * @returns {string} Formatted time or "—" if null/empty
 */
export const fmtTime = (iso) => !iso ? "—" : new Date(iso).toLocaleTimeString([], {hour12: false});
