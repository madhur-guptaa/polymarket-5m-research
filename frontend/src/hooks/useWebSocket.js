/**
 * useWebSocket.js
 *
 * One WebSocket per symbol (/ws/{symbol}), retargeted to the new api/ contract:
 *   tick   (~1s)  live window state + P(up)   -> tick
 *   micro      (~10s) 10-second volume bars        -> micro
 *   matrix     (~60s) probability-surface grid     -> matrix
 *   macro      (~60s) 1-hour price trend           -> macro
 *   polymarket (~25s) live market prices + edge     -> poly, edge
 *   pong              keepalive ack
 *
 * Reconnects with exponential backoff (2s → 30s).
 */

import { useEffect, useRef, useState } from "react";
import { wsBase } from "../utils/constants";

export function useWebSocket(symbol) {
  const wsRef = useRef(null);
  const reconnect = useRef(null);
  const delay = useRef(2000);

  const [connected, setConnected] = useState(false);
  const [tick, setTick] = useState(null);
  const [matrix, setMatrix] = useState(null);
  const [micro, setMicro] = useState([]);
  const [macro, setMacro] = useState([]);
  const [poly, setPoly] = useState(null);
  const [edge, setEdge] = useState(null);

  useEffect(() => {
    setTick(null); setMatrix(null); setMicro([]); setMacro([]); setPoly(null); setEdge(null);
    delay.current = 2000;
    let unmounting = false;

    function connect() {
      if (unmounting) return;
      const ws = new WebSocket(`${wsBase()}/${symbol}`);
      wsRef.current = ws;

      ws.onopen = () => {
        if (unmounting) return ws.close();
        setConnected(true);
        delay.current = 2000;
      };
      ws.onclose = () => {
        setConnected(false);
        if (!unmounting) {
          reconnect.current = setTimeout(() => {
            delay.current = Math.min(delay.current * 1.5, 30000);
            connect();
          }, delay.current);
        }
      };
      ws.onerror = () => setConnected(false);
      ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        switch (msg.type) {
          case "tick": setTick(msg.data); break;
          case "matrix": setMatrix(msg.data); break;
          case "micro": setMicro(msg.data || []); break;
          case "macro": setMacro(msg.data || []); break;
          case "polymarket": setPoly(msg.data?.poly ?? null); setEdge(msg.data?.edge ?? null); break;
          case "pong": break;
          default: break;
        }
      };
    }

    connect();
    const ping = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "ping" }));
      }
    }, 30000);

    return () => {
      unmounting = true;
      clearInterval(ping);
      clearTimeout(reconnect.current);
      const ws = wsRef.current;
      if (ws) {
        ws.onclose = null;
        ws.onerror = null;
        if (ws.readyState === WebSocket.OPEN) ws.close();
        else if (ws.readyState === WebSocket.CONNECTING) ws.onopen = () => ws.close();
      }
      wsRef.current = null;
    };
  }, [symbol]);

  return { connected, tick, matrix, micro, macro, poly, edge };
}
