# Polaris Quant Engine — Frontend Dashboard

Real-time Bayesian probability trading dashboard built with React, Plotly, and Recharts.

![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-6-646CFF?logo=vite&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly.js-2.35-3F4F75?logo=plotly&logoColor=white)
![Tailwind](https://img.shields.io/badge/Tailwind_CSS-4-06B6D4?logo=tailwindcss&logoColor=white)

## Overview

This is the visualization layer of the Polaris Quant Engine — a single-page React application
that connects to the FastAPI backend via WebSocket and renders all real-time data:

- **2D Probability Heatmap** — The centerpiece: P(Up) across (time_elapsed, delta_bps)
- **8 KPI Cards** — Price, delta, momentum, model P(Up), market P(Up), edge, imbalance, elapsed
- **6-Feature Radar** — All features feeding the Bayesian estimator, color-coded
- **Order Flow** — Live trade feed + cumulative taker imbalance sparkline
- **Polymarket Integration** — Model vs market probability comparison + edge signal
- **Walk-Forward Backtest** — Sharpe, win rate, cumulative PnL equity curve
- **Calibration Table** — Decile calibration quality check
- **Credible Interval Gauge** — 95% CI with confidence quality indicator

## Dashboard Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              HEADER                                     │
│  POLARIS // QUANT ENGINE    [BTC] [ETH] [SOL] [XRP]       ● WS LIVE   │
├─────────────────────────────────────────────────────────────────────────┤
│                            KPI STRIP                                    │
│  PRICE │ DELTA │ MOM │ MODEL │ MARKET │ EDGE │ IMBAL │ ELAPSED         │
├──────────┬──────────────────┬────────────┬──────────────────────────────┤
│ LEFT     │ CENTER           │ RIGHT 1    │ RIGHT 2                      │
│ 260px    │ 1fr (flexible)   │ 240px      │ 230px                        │
│          │                  │            │                              │
│ Order    │ BAYESIAN         │ P(Up)      │ POLYMARKET                   │
│ Flow     │ PROBABILITY      │ Readout    │                              │
│          │ SURFACE          │            │ Edge Signal                  │
│ 10s Vol  │ (heatmap)        │ Feature    │                              │
│          │                  │ Radar      │ Calibration                  │
│ Imbal    │                  │            │ Table                        │
│          │                  │ 95% CI     │                              │
│ Delta    │                  │ Backtest   │ Window Stats                 │
│ Path     │                  │ 1H Macro   │ + Bandwidths                 │
└──────────┴──────────────────┴────────────┴──────────────────────────────┘
```

## Quick Start

### Prerequisites

- Node.js ≥ 18
- npm or yarn
- Backend API running at the URL specified in `src/utils/constants.js`

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
```

The dev server starts at `http://localhost:5173` with hot module replacement.

### Production Build

```bash
npm run build
npm run preview   # Preview the production build locally
```

### Configuration

Edit `src/utils/constants.js` to set the backend connection URLs:

```javascript
// Development (local network)
export const WS_URL = "ws://localhost:8000/ws";
export const API_URL = "http://localhost:8000/api";

// Production (use environment variables via Vite)
// export const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws";
// export const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api";
```

## Project Structure

```
frontend/
├── public/                       # Static assets
├── src/
│   ├── App.jsx                   # Root component (4-column grid layout)
│   ├── main.jsx                  # React entry point
│   ├── index.css                 # Global dark theme + Tailwind directives
│   ├── hooks/
│   │   └── useWebSocket.js       # Central WS hook (all real-time data)
│   ├── utils/
│   │   ├── colors.js             # Color mapping + label generation
│   │   ├── constants.js          # URLs, symbols, brand colors
│   │   └── format.js             # Number/time formatting (fmt, pct, bps)
│   └── components/
│       ├── layout/
│       │   ├── Header.jsx        # Branding + symbol pills + WS status
│       │   ├── KpiStrip.jsx      # 8 KPI indicator cards
│       │   └── Panel.jsx         # Reusable panel container
│       ├── pills/
│       │   └── SymbolPill.jsx    # Symbol selector button
│       ├── charts/
│       │   ├── HeatmapSurface.jsx  # 2D probability heatmap (Plotly)
│       │   └── OrderFlowChart.jsx  # Imbalance sparkline (Recharts)
│       ├── indicators/
│       │   ├── Tick.jsx              # Animated price with flash
│       │   ├── ProbabilityReadout.jsx # Hero P(Up) display
│       │   ├── FeatureRadar.jsx      # 6-feature grid
│       │   └── CredibleIntervalGauge.jsx # CI gauge
│       └── trading/
│           ├── PolyPanel.jsx         # Polymarket + edge signal
│           ├── BacktestPanel.jsx     # Backtest results + PnL curve
│           └── CalibrationTable.jsx  # Decile calibration quality
├── index.html
├── vite.config.js
├── tailwind.config.js
└── package.json
```

## WebSocket Protocol

A single WebSocket connection per symbol carries all data:

| Message Type   | Cadence | Content                                          |
|----------------|---------|--------------------------------------------------|
| `tick`         | ~1s     | Live window state, features, P(Up), trades       |
| `matrix`       | ~60s    | 2D heatmap grid + 50% contour                    |
| `macro`        | ~60s    | 1-hour price trend data                          |
| `backtest`     | ~60s    | Walk-forward results (Sharpe, win rate, PnL)     |
| `calibration`  | ~60s    | Decile calibration table                         |
| `polymarket`   | ~25s    | Market data + edge signal                        |
| `pong`         | On ping | Keepalive acknowledgment                         |

## Key Design Decisions

- **Single WebSocket**: All data flows through one connection per symbol, reducing connection overhead and simplifying state management.
- **Reconnection**: Exponential backoff (2s → 30s cap) with jitter on disconnect.
- **Cleanup**: Zombie connection prevention via `isUnmounting` flag and listener stripping before close.
- **Monospace Typography**: `JetBrains Mono` / `Fira Code` for precise numerical alignment.
- **Tabular Numbers**: `font-variant-numeric: tabular-nums` prevents layout jitter on price updates.
- **Dark Theme**: #080c10 background with the full Tailwind Slate palette for UI chrome.

## Tech Stack

| Library          | Purpose                                    |
|------------------|--------------------------------------------|
| React 19         | UI framework                               |
| Vite 6           | Build tool + dev server                    |
| Plotly.js        | 2D heatmap visualization                   |
| Recharts         | Area/bar charts (order flow, delta, macro)  |
| Tailwind CSS 4   | Utility classes (minimal usage — mostly inline styles) |

## Color Palette

| Color     | Hex       | Usage                                    |
|-----------|-----------|------------------------------------------|
| Green     | `#00ff87` | Bullish / positive / low volatility       |
| Red       | `#ff4d6d` | Bearish / negative / high volatility      |
| Amber     | `#f59e0b` | Neutral volatility / acceptable quality   |
| Slate     | `#94a3b8` | Neutral / informational                   |
| Dark      | `#334155` | No data / disabled / subtle elements      |
| BTC       | `#f7931a` | Bitcoin orange                            |
| ETH       | `#627eea` | Ethereum blue                             |
| SOL       | `#9945ff` | Solana purple                             |
| XRP       | `#00aae4` | XRP teal                                  |
