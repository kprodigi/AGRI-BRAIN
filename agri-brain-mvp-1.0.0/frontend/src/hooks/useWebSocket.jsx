import { useEffect, useRef, useState, useCallback } from "react";
import { getApiBase } from "@/mvp/api.js";

export function useWebSocket() {
  const [connected, setConnected] = useState(false);
  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const wsRef = useRef(null);
  const listenersRef = useRef([]);

  const addNotification = useCallback((notification) => {
    const item = {
      id: crypto.randomUUID(),
      ...notification,
      timestamp: Date.now(),
      read: false,
    };
    setNotifications((prev) => [item, ...prev].slice(0, 100));
    setUnreadCount((c) => c + 1);
  }, []);

  const markAllRead = useCallback(() => {
    setNotifications((prev) => prev.map((n) => ({ ...n, read: true })));
    setUnreadCount(0);
  }, []);

  const subscribe = useCallback((handler) => {
    listenersRef.current.push(handler);
    return () => {
      listenersRef.current = listenersRef.current.filter((h) => h !== handler);
    };
  }, []);

  useEffect(() => {
    const API = getApiBase();
    const WS_URL = (API || "").replace(/^http/i, "ws") + "/stream";
    let ws;
    let reconnectTimer;

    const connect = () => {
      try {
        ws = new WebSocket(WS_URL);
        wsRef.current = ws;

        ws.onopen = () => {
          setConnected(true);
          console.log("[WS] connected:", WS_URL);
        };

        ws.onclose = () => {
          setConnected(false);
          console.log("[WS] disconnected, reconnecting in 5s...");
          reconnectTimer = setTimeout(connect, 5000);
        };

        ws.onerror = () => setConnected(false);

        ws.onmessage = (ev) => {
          try {
            const msg = JSON.parse(ev.data || "{}");
            const type = msg?.type;
            const payload = msg?.payload;

            // Dispatch to listeners
            listenersRef.current.forEach((fn) => fn(msg));

            // Dispatch DOM events for backward compatibility
            if (type === "decision") {
              document.dispatchEvent(new CustomEvent("decision:new", { detail: payload }));
              addNotification({
                type: "info",
                title: "New Decision",
                message: `${payload?.agent || "Agent"}: ${payload?.action || "decision"} (SLCA: ${(payload?.slca_score ?? payload?.slca ?? 0).toFixed?.(3) || "—"})`,
              });
            }
            if (type === "chain/head") {
              document.dispatchEvent(new CustomEvent("chain:head", { detail: payload }));
            }
            if (type === "agent/error" || type === "chain/error") {
              addNotification({
                type: "error",
                title: "System Error",
                message: payload?.error || "Unknown error",
              });
            }
          } catch (e) {
            console.warn("[WS parse]", e);
          }
        };
      } catch (e) {
        console.warn("[WS] failed to connect:", e);
        reconnectTimer = setTimeout(connect, 5000);
      }
    };

    connect();

    return () => {
      clearTimeout(reconnectTimer);
      try { ws && ws.close(); } catch {}
    };
  }, [addNotification]);

  return { connected, notifications, unreadCount, markAllRead, subscribe };
}
