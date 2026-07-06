// frontend/src/mvp/api.js
// -------------------------------------------------------------
// Single API base. The 8111/8100 port-swap retry path was removed
// in 2026-05: the backend never bound 8111 anywhere, so on a real
// network failure the retry doubled the user-visible latency
// (every failed request triggered a second one to a port that was
// also unreachable) and persistently mutated localStorage to a
// dead URL.
//
// API base resolution order (first non-empty wins):
//   1. ``window.API_BASE``      -- runtime override injected by
//                                  the host page (index.html /
//                                  reverse proxy).
//   2. ``localStorage.API_BASE`` -- per-browser pin set via the
//                                   admin panel snippet.
//   3. ``import.meta.env.VITE_API_BASE`` -- build-time pin from
//                                            .env / .env.production
//                                            (added 2026-05 so a
//                                            production build can
//                                            point at a non-localhost
//                                            backend without editing
//                                            index.html or relying
//                                            on localStorage).
//   4. ``http://127.0.0.1:8100`` -- final dev default.
// -------------------------------------------------------------
import { getApiKey } from "@/lib/utils";

const _VITE_API_BASE =
    (typeof import.meta !== 'undefined' && import.meta.env
        ? import.meta.env.VITE_API_BASE
        : '') || '';

let API = (
    window.API_BASE
    || localStorage.getItem('API_BASE')
    || _VITE_API_BASE
    || 'http://127.0.0.1:8100'
).replace(/\/$/, '');

export function getApiBase() {
    return API;
}
export function setApiBase(url) {
    API = (url || '').replace(/\/$/, '');
    localStorage.setItem('API_BASE', API);
}

// Build the PDF URL in one place (used by Admin/Audit buttons)
export function memoPdfUrl() {
    return `${API}/report/pdf`;
}

// -------------------------------------------------------------
// JSON helper. Surfaces the original error directly on failure
// rather than silently retrying against a non-existent port.
// -------------------------------------------------------------
function _headers() {
    const h = { 'Content-Type': 'application/json' };
    const key = getApiKey();
    if (key) h['x-api-key'] = key;
    return h;
}

async function j(method, path, body) {
    const url = `${API}${path}`;
    const res = await fetch(url, {
        method,
        headers: _headers(),
        body: body ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return await res.json();
}

// -------------------------------------------------------------
// Governance
// -------------------------------------------------------------
export const Governance = {
    getPolicy: () => j('GET', '/governance/policy'),
    savePolicy: (p) => j('POST', '/governance/policy', p),
    getChain: () => j('GET', '/governance/chain'),
    saveChain: (c) => j('POST', '/governance/chain', c),
};

// -------------------------------------------------------------
// Audit
// -------------------------------------------------------------
export const Audit = {
    getLogs: () => j('GET', '/audit/logs'),
};

// -------------------------------------------------------------
// Scenarios (new endpoints if present; graceful fallback if not)
// -------------------------------------------------------------
export const Scenarios = {
    async list() {
        // Prefer new backend
        try {
            const res = await fetch(`${API}/scenarios/list`, { headers: _headers() });
            if (res.ok) return await res.json(); // {scenarios:[{id,label,desc}], active}
        } catch { }
        // Fallback — match the backend shape {scenarios: [...], active: ...}
        return {
            scenarios: [
                { id: 'heatwave', label: 'Climate-Induced Heatwave' },
                { id: 'overproduction', label: 'Overproduction / Glut' },
                { id: 'cyber_outage', label: 'Cyber Threat & Node Outage' },
                { id: 'adaptive_pricing', label: 'Adaptive Pricing & Demand Oscillation' },
                { id: 'baseline', label: 'Baseline' },
            ],
            active: null,
        };
    },

    async apply(id, intensity = 1.0) {
        // Try new endpoint
        try {
            const r = await fetch(`${API}/scenarios/run`, {
                method: 'POST',
                headers: _headers(),
                body: JSON.stringify({ name: id, intensity }),
            });
            if (r.ok) return await r.json();
        } catch { }
        // No-op fallback
        return { ok: true };
    },

    async reset() {
        try {
            const r = await fetch(`${API}/scenarios/reset`, { method: 'POST', headers: _headers() });
            if (r.ok) return await r.json();
        } catch { }
        return { ok: true };
    },
};

// -------------------------------------------------------------
// Decide (robust: tries several known endpoints/shapes)
// -------------------------------------------------------------
export const Decide = {
    async once(payload = {}) {
        const res = await fetch(`${API}/decide`, {
            method: 'POST',
            headers: _headers(),
            body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        const raw = await res.json();
        const data = raw.memo ?? raw;
        return {
            agent: data.agent ?? payload.agent ?? 'farm',
            action: data.action ?? data.route ?? 'cold_chain',
            slca_score: data.slca_score ?? data.slca ?? 0,
            carbon_kg: data.carbon_kg ?? data.carbon ?? 0,
            circular_economy_score: data.circular_economy_score ?? null,
            reason: data.reason ?? data.note ?? '—',
            // null when the chain is not configured / the submission failed.
            // Earlier revisions used '0x0' as a sentinel which the UI
            // confused for a successful anchor; the badge now distinguishes
            // null (no anchor attempted), '0x0' (legacy sentinel) and a
            // real 0x-prefixed hash.
            tx_hash: data.tx_hash ?? data.tx ?? null,
            ts: data.ts ?? Math.floor(Date.now() / 1000),
            ...data,
        };
    },
};
