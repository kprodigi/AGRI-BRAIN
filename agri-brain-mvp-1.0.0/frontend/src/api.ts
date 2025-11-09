
export const API_BASE = (localStorage.getItem('API_BASE') || 'http://127.0.0.1:8100').replace(/\/$/, '');

async function j(method: string, path: string, body?: any) {
  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return await res.json();
}

export const Governance = {
  getPolicy: () => j('GET', '/governance/policy'),
  savePolicy: (p: any) => j('POST', '/governance/policy', p),
  getChain: () => j('GET', '/governance/chain'),
  saveChain: (c: any) => j('POST', '/governance/chain', c),
};
export const Audit = { getLogs: () => j('GET', '/audit/logs') };
export const Scenarios = { list: () => j('GET', '/scenarios'), apply: (id: string) => j('POST', '/scenarios', { id }) };
export const Decide = { once: (payload: any) => j('POST', '/decide', payload) };
