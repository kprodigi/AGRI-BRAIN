import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs) {
  return twMerge(clsx(inputs));
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

// API fetch helper
export async function jget(apiBase, path) {
  const r = await fetch(`${apiBase}${path}`);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

export async function jpost(apiBase, path, body = {}) {
  const r = await fetch(`${apiBase}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  try { return await r.json(); } catch { return {}; }
}
