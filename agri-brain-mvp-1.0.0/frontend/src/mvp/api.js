// frontend/src/mvp/api.js
// -------------------------------------------------------------
// Smart, single API base with automatic 8111/8100 fallback
// -------------------------------------------------------------
let API = (window.API_BASE || localStorage.getItem('API_BASE') || 'http://127.0.0.1:8111').replace(/\/$/, '');

export function getApiBase() {
    return API;
}
export function setApiBase(url) {
    API = (url || '').replace(/\/$/, '');
    localStorage.setItem('API_BASE', API);
}

// Build the PDF URL in one place (used by Admin/Audit buttons)
export function memoPdfUrl() {
    return `${API}/audit/memo.pdf`;
}

// Toggle port 8111 <-> 8100 if the current base is unreachable
function togglePort(base) {
    try {
        const u = new URL(base);
        if (u.port === '8111') u.port = '8100';
        else if (u.port === '8100') u.port = '8111';
        else u.port = '8111';
        return u.toString().replace(/\/$/, '');
    } catch {
        return base;
    }
}

// -------------------------------------------------------------
// JSON helper with retry-on-port-swap (keeps your flows running)
// -------------------------------------------------------------
async function j(method, path, body) {
    const url1 = `${API}${path}`;
    try {
        const res = await fetch(url1, {
            method,
            headers: { 'Content-Type': 'application/json' },
            body: body ? JSON.stringify(body) : undefined,
        });
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        return await res.json();
    } catch (e) {
        // network/CORS/port issue? try the alternate port once
        const alt = togglePort(API);
        if (alt !== API) {
            try {
                const url2 = `${alt}${path}`;
                const res2 = await fetch(url2, {
                    method,
                    headers: { 'Content-Type': 'application/json' },
                    body: body ? JSON.stringify(body) : undefined,
                });
                if (!res2.ok) throw new Error(`${res2.status} ${res2.statusText}`);
                // Success on alt — persist it
                setApiBase(alt);
                return await res2.json();
            } catch {
                throw e;
            }
        }
        throw e;
    }
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
            const res = await fetch(`${API}/scenarios/list`);
            if (res.ok) return await res.json(); // {scenarios:[{id,label,desc}], active}
        } catch { }
        // Fallback so Admin still renders something
        return {
            options: [
                { id: 'climate_shock', name: 'Climate-Induced Supply Shock' },
                { id: 'reverse_logistics', name: 'Reverse Logistics of Spoiled Food' },
                { id: 'cyber_outage', name: 'Cyber Threat & Node Outage' },
                { id: 'adaptive_pricing', name: 'Adaptive Pricing & Cooperative Auctions' },
            ],
            selected: null,
        };
    },

    async apply(id, intensity = 1.0) {
        // Try new endpoint
        try {
            const r = await fetch(`${API}/scenarios/run`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: id, intensity }),
            });
            if (r.ok) return await r.json();
        } catch { }
        // No-op fallback
        return { ok: true };
    },

    async reset() {
        try {
            const r = await fetch(`${API}/scenarios/reset`, { method: 'POST' });
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
        const tries = [
            // Primary MVP routes you’ve been using:
            ['/decide', { method: 'POST', body: JSON.stringify(payload) }],
            // Common alternates we’ve seen in your setup:
            ['/decision/take', { method: 'POST' }],
            ['/decision/take', { /* GET */ }],
            ['/decisions/take', { method: 'POST' }],
            ['/case/decide', { method: 'POST', body: JSON.stringify(payload) }],
            ['/api/decision/take', { method: 'POST' }],
        ];

        let lastErr;
        for (const [p, init] of tries) {
            try {
                const res = await fetch(`${API}${p}`, {
                    headers: { 'Content-Type': 'application/json' },
                    ...init,
                });
                if (res.ok) {
                    const data = await res.json();
                    // normalize a minimal memo shape so UI always has fields
                    return {
                        agent: data.agent ?? payload.agent ?? 'farm',
                        action: data.action ?? data.route ?? 'standard_cold_chain',
                        slca_score: data.slca_score ?? data.slca ?? 0,
                        carbon_kg: data.carbon_kg ?? data.carbon ?? 0,
                        reason: data.reason ?? data.note ?? '—',
                        tx_hash: data.tx_hash ?? data.tx ?? '0x0',
                        ts: data.ts ?? Math.floor(Date.now() / 1000),
                        ...data,
                    };
                }
            } catch (e) {
                lastErr = e;
            }
        }
        throw lastErr || new Error('Could not reach any decision endpoint.');
    },
};
