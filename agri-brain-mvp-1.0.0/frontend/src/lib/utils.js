import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs) {
  return twMerge(clsx(inputs));
}

export function getApiKey() {
  const sessionKey = sessionStorage.getItem("API_KEY");
  if (sessionKey) return sessionKey;
  const legacy = localStorage.getItem("API_KEY");
  if (legacy) {
    sessionStorage.setItem("API_KEY", legacy);
    localStorage.removeItem("API_KEY");
    return legacy;
  }
  return "";
}

// Number formatting helpers
export const n = (v) => (Number.isFinite(+v) ? +v : null);
export const fmt = (v, d = 2) => (Number.isFinite(+v) ? (+v).toFixed(d) : "\u2014");
export const fmtK = (v) => {
  if (!Number.isFinite(+v)) return "\u2014";
  const num = +v;
  if (Math.abs(num) >= 1e6) return (num / 1e6).toFixed(1) + "M";
  if (Math.abs(num) >= 1e3) return (num / 1e3).toFixed(1) + "K";
  return num.toLocaleString();
};
export const last = (arr) => (Array.isArray(arr) && arr.length ? arr[arr.length - 1] : null);
export const short = (s) => (s && s.length > 12 ? `${s.slice(0, 8)}\u2026${s.slice(-4)}` : s || "");

// Shared API key helper — reads from localStorage once per call
function _apiHeaders(extra = {}) {
  const h = { "Content-Type": "application/json", ...extra };
  const key = getApiKey();
  if (key) h["x-api-key"] = key;
  return h;
}

// Authenticated fetch wrapper for pages using direct fetch()
export function authFetch(url, opts = {}) {
  const key = getApiKey();
  const headers = { ...(opts.headers || {}) };
  if (key) headers["x-api-key"] = key;
  if (opts.body && !headers["Content-Type"]) headers["Content-Type"] = "application/json";
  return fetch(url, { ...opts, headers });
}

// Authenticated file download — fetches with API key, opens as blob URL
export async function authDownload(url, filename = "download") {
  const res = await authFetch(url);
  if (!res.ok) throw new Error(`Download failed: ${res.status}`);
  const blob = await res.blob();
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

// API fetch helper
export async function jget(apiBase, path) {
  const r = await fetch(`${apiBase}${path}`, { headers: _apiHeaders() });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

export async function jpost(apiBase, path, body = {}) {
  const r = await fetch(`${apiBase}${path}`, {
    method: "POST",
    headers: _apiHeaders(),
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  try { return await r.json(); } catch { return {}; }
}

// MCP JSON-RPC helpers
// mcpLog is a shared array for protocol interaction logging
export const mcpLog = [];

export async function mcpRaw(apiBase, method, params = {}) {
  const req = { jsonrpc: "2.0", id: Date.now(), method, params };
  const r = await fetch(`${apiBase}/mcp/mcp`, {
    method: "POST",
    headers: _apiHeaders(),
    body: JSON.stringify(req),
  });
  if (!r.ok) throw new Error(`MCP ${r.status} ${r.statusText}`);
  const j = await r.json();
  mcpLog.push({
    ts: new Date().toISOString(), method, params,
    status: j.error ? "error" : "success",
    preview: JSON.stringify(j.result || j.error).substring(0, 200),
  });
  if (mcpLog.length > 200) mcpLog.splice(0, mcpLog.length - 200);
  if (j.error) throw new Error(j.error.message || "MCP error");
  return j.result;
}

export async function mcpCall(apiBase, toolName, args = {}) {
  const result = await mcpRaw(apiBase, "tools/call", { name: toolName, arguments: args });
  const text = result?.content?.[0]?.text;
  if (!text) return result;
  try {
    return JSON.parse(text);
  } catch {
    return result;
  }
}
