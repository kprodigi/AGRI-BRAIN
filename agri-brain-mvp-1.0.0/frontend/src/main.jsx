import React, { lazy, Suspense } from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import { getApiBase } from "./mvp/api.js";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { ThemeProvider } from "./hooks/useTheme.jsx";
import { useWebSocket } from "./hooks/useWebSocket.jsx";
import { Toaster } from "sonner";
import MainLayout from "./layouts/MainLayout.jsx";
import OpsPage from "./pages/OpsPage.jsx";

// Lazy-load heavy pages
const QualityPage = lazy(() => import("./pages/QualityPage.jsx"));
const DecisionsPage = lazy(() => import("./pages/DecisionsPage.jsx"));
const MapPage = lazy(() => import("./pages/MapPage.jsx"));
const AnalyticsPage = lazy(() => import("./pages/AnalyticsPage.jsx"));
const AdminPage = lazy(() => import("./pages/AdminPage.jsx"));

function PageLoader() {
  return (
    <div className="flex items-center justify-center h-64">
      <div className="animate-spin rounded-full h-8 w-8 border-2 border-primary border-t-transparent" />
    </div>
  );
}

function AppShell({ children }) {
  const { connected, notifications, unreadCount, markAllRead } = useWebSocket();

  return (
    <MainLayout
      wsConnected={connected}
      notifications={notifications}
      unreadCount={unreadCount}
      onMarkAllRead={markAllRead}
    >
      <Suspense fallback={<PageLoader />}>{children}</Suspense>
    </MainLayout>
  );
}

const router = createBrowserRouter([
  { path: "/", element: <AppShell><OpsPage /></AppShell> },
  { path: "/quality", element: <AppShell><QualityPage /></AppShell> },
  { path: "/decisions", element: <AppShell><DecisionsPage /></AppShell> },
  { path: "/map", element: <AppShell><MapPage /></AppShell> },
  { path: "/analytics", element: <AppShell><AnalyticsPage /></AppShell> },
  { path: "/admin", element: <AppShell><AdminPage /></AppShell> },
]);

// ---------- Global "Take decision" handler (backward compat) ----------
;(function installTakeDecision() {
  if (window.__takeDecisionInstalled) return;
  window.__takeDecisionInstalled = true;

  const base = getApiBase();

  async function callAny() {
    const roleSelect = document.querySelector("select");
    const role = (roleSelect && roleSelect.value) || "farm";
    const payload = JSON.stringify({ agent_id: role, role });
    const hdrs = { "Content-Type": "application/json" };
    const tries = [
      [`${base}/decide`, { method: "POST", headers: hdrs, body: payload }],
      [`${base}/decision/take?role=${encodeURIComponent(role)}`, {}],
      [`${base}/decision/take`, { method: "POST", headers: hdrs, body: payload }],
    ];
    for (const [url, init] of tries) {
      try { const r = await fetch(url, init); if (r.ok) return await r.json(); } catch {}
    }
    throw new Error("No decision endpoint responded with 200");
  }

  document.addEventListener("click", async (e) => {
    const el = e.target.closest('button, a, [role="button"]');
    if (!el) return;
    if (location.pathname.startsWith("/admin")) return;
    if (el.closest("[data-skip-global-take]")) return;
    const label = (el.textContent || "").replace(/\s+/g, " ").trim();
    if (!/^\s*take\s+decision\s*$/i.test(label)) return;
    e.preventDefault();
    try {
      const res = await callAny();
      const memo = res.memo ?? res;
      document.dispatchEvent(new CustomEvent("decision:new", { detail: memo }));
    } catch (err) {
      console.warn(err);
    }
  }, true);
})();

// ---------- Fix "Download Decision Memo (PDF)" button ----------
;(function fixMemoDownloadButton() {
  const base = getApiBase();
  const url = `${base}/report/pdf`;

  function attach() {
    const btn = Array.from(document.querySelectorAll("a,button"))
      .find((el) => /download\s+decision\s+memo\s*\(pdf\)/i.test((el.textContent || "").trim()));
    if (!btn || btn.dataset.memoBound === "1") return;
    if (btn.tagName === "A") {
      btn.setAttribute("href", url);
      btn.setAttribute("target", "_blank");
      btn.setAttribute("rel", "noopener");
    } else {
      btn.addEventListener("click", (e) => { e.preventDefault(); window.open(url, "_blank", "noopener"); });
    }
    btn.dataset.memoBound = "1";
  }
  attach();
  window.addEventListener("load", attach);
  const mo = new MutationObserver(() => attach());
  mo.observe(document.documentElement, { childList: true, subtree: true });
})();

// ---------- Mount app ----------
const rootEl = document.getElementById("root");
if (!rootEl) throw new Error("Root element #root not found");

ReactDOM.createRoot(rootEl).render(
  <ThemeProvider defaultTheme="system">
    <RouterProvider router={router} />
    <Toaster
      position="bottom-right"
      toastOptions={{
        className: "border shadow-lg",
        style: { borderRadius: "0.75rem" },
      }}
    />
  </ThemeProvider>
);
