// frontend/src/mvp/AdminPanel.tsx
import React, { useEffect, useState } from 'react';
import { Governance, Audit, Scenarios, Decide } from './api';

const API_BASE = (window.API_BASE || localStorage.getItem('API_BASE') || 'http://127.0.0.1:8100').replace(/\/$/, '');

const Card = ({ title, children }) => (
  <div className="p-4 bg-white rounded-2xl shadow mb-6">
    <h2 className="text-xl font-semibold mb-3">{title}</h2>
    <div>{children}</div>
  </div>
);

// -------- helpers ----------
async function getJSON(path: string, init?: RequestInit) {
  const res = await fetch(`${API_BASE}${path}`, init);
  const text = await res.text();
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} :: ${text || '(no body)'}`);
  try { return text ? JSON.parse(text) : {}; } catch { return {}; }
}
async function postJSON(path: string, body: any = {}) {
  return getJSON(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

// ============================================================

export default function AdminPanel() {
  const [tab, setTab] = useState('Policy');

  // ---- Health
  const [apiOk, setApiOk] = useState(true);
  useEffect(() => {
    fetch(`${API_BASE}/health`).then(r => setApiOk(r.ok)).catch(() => setApiOk(false));
  }, []);

  // ---- Policy
  const [policy, setPolicy] = useState<any>(null);
  const [policyErr, setPolicyErr] = useState('');
  const loadPolicy = async () => {
    try { setPolicyErr(''); setPolicy(await Governance.getPolicy()); }
    catch (e) { setPolicyErr(String(e)); setPolicy(null); }
  };
  useEffect(() => { loadPolicy(); }, []);
  const savePolicy = async () => {
    try { const p = await Governance.savePolicy(policy); setPolicy(p); alert('Policy saved.'); }
    catch (e) { alert('Save failed: ' + e); }
  };

  // ---- Chain
  const [chain, setChain] = useState<any>(null);
  const [chainErr, setChainErr] = useState('');
  const loadChain = async () => {
    try { setChainErr(''); setChain(await Governance.getChain()); }
    catch (e) { setChainErr(String(e)); setChain(null); }
  };
  useEffect(() => { loadChain(); }, []);
  const saveChain = async () => {
    try { const c = await Governance.saveChain(chain); setChain(c); alert('Chain config saved.'); }
    catch (e) { alert('Save failed: ' + e); }
  };

  // ---- Audit logs
  const [logs, setLogs] = useState<any[]>([]);
  const refreshAudit = async () => {
    try { const d = await Audit.getLogs(); setLogs(d.items || []); }
    catch (e) { setLogs([]); }
  };
  useEffect(() => { refreshAudit(); }, []);

  // ---- Scenarios with resilient discovery
  const BUILTIN_SCENARIOS = [
    { id: 'climate_shock', label: 'Climate-Induced Supply Shock', desc: '72h heatwave; faster spoilage; reconfigure routes.' },
    { id: 'reverse_logistics', label: 'Reverse Logistics of Spoiled Food', desc: 'Glut/overproduction; prioritize redistribution & recovery.' },
    { id: 'cyber_outage', label: 'Cyber Threat & Node Outage', desc: 'Processor offline; unauthorized tx blocked; reroute flows.' },
    { id: 'adaptive_pricing', label: 'Adaptive Pricing & Coop Auctions', desc: 'Learned pricing policy; equity-aware redistribution.' },
  ];
  const [scList, setScList] = useState<any[]>([]);
  const [scActive, setScActive] = useState<any>(null);
  const [scIntensity, setScIntensity] = useState(1.0);
  const [scMsg, setScMsg] = useState('');

  async function loadScenarios() {
    setScMsg('');
    try {
      const r = await getJSON('/scenarios/list');
      const items = (r.scenarios || []).map((s: any) => ({ id: s.id, label: s.label || s.id, desc: s.desc || '' }));
      setScList(items);
      setScActive(r.active || null);
      return;
    } catch {
      try {
        const d = await Scenarios.list();
        const items = (d?.options || []).map((o: any) => ({ id: o.id, label: o.name || o.id, desc: o.desc || '' }));
        setScList(items.length ? items : BUILTIN_SCENARIOS);
        setScActive(d?.selected ? { name: d.selected, intensity: 1.0 } : null);
        return;
      } catch {
        setScList(BUILTIN_SCENARIOS);
      }
    }
  }
  useEffect(() => { loadScenarios(); }, []);

  async function runScenario(name: string) {
    setScMsg('');
    try {
      const r = await postJSON('/scenarios/run', { name, intensity: Number(scIntensity) });
      setScActive({ name, intensity: Number(scIntensity) });
      const avg = r?.metrics?.avg_tempC;
      setScMsg(`✅ Applied '${name}'${typeof avg === 'number' ? ` (avg temp ${avg.toFixed(2)} °C)` : ''}.`);
      return;
    } catch { }
    try {
      await Scenarios.apply(name);
      const d = await Scenarios.list();
      setScActive({ name, intensity: Number(scIntensity) });
      setScList((d?.options || []).map((o: any) => ({ id: o.id, label: o.name || o.id, desc: o.desc || '' })));
      setScMsg(`✅ Scenario set: ${name}`);
      return;
    } catch { }
    setScMsg('❌ Could not apply scenario.');
  }
  async function resetScenario() {
    setScMsg('');
    try { await postJSON('/scenarios/reset', {}); setScActive(null); setScMsg('✅ Reset to base spinach case.'); }
    catch { setScActive(null); setScMsg('⚠ Reset locally (backend reset not available).'); }
  }

  // ---- Quick decision
  const [role, setRole] = useState('farm');
  const [qdMsg, setQdMsg] = useState('');
  async function quickTake() {
    setQdMsg('');
    try {
      // Post BOTH fields so either backend variant accepts it
      const res = await fetch(`${API_BASE}/decide`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ agent: `admin:${role}`, agent_id: `admin:${role}`, role }),
      });
      const text = await res.text();
      if (!res.ok) {
        setQdMsg(`❌ ${res.status} ${res.statusText}${text ? ' :: ' + text : ''}`);
        return;
      }
      const data = text ? JSON.parse(text) : {};
      const memo = data?.memo || data || {};
      document.dispatchEvent(new CustomEvent('decision:new', { detail: memo }));

      const action = memo.decision ?? memo.action ?? '(unknown)';
      const slca = Number(memo.slca ?? memo.slca_score ?? 0).toFixed(3);
      const co2 = memo.carbon_kg ?? memo.carbon ?? 0;
      setQdMsg(`✅ role=${role} | ${action} | SLCA ${slca} | CO₂ ${co2} kg`);
      refreshAudit();
    } catch (e: any) {
      setQdMsg('❌ ' + e.message);
    }
  }

  // ---- Results (simulation)
  const [resultsData, setResultsData] = useState<any>(null);
  const [resultsLoading, setResultsLoading] = useState(false);
  const [resultsErr, setResultsErr] = useState('');
  const [figFiles, setFigFiles] = useState<string[]>([]);

  async function runSimulation() {
    setResultsLoading(true);
    setResultsErr('');
    try {
      const r = await postJSON('/results/generate');
      setResultsData(r.summary || r);
      // Discover available figures
      try {
        const figNames = [
          'fig2_heatwave.png', 'fig3_reverse.png', 'fig4_cyber.png',
          'fig5_pricing.png', 'fig6_cross.png', 'fig7_ablation.png', 'fig8_green.png',
        ];
        setFigFiles(figNames);
      } catch { setFigFiles([]); }
    } catch (e: any) {
      setResultsErr(e.message || String(e));
    } finally {
      setResultsLoading(false);
    }
  }

  // Optional: keep your advanced simulation button
  const [qd, setQD] = useState({ inventory_units: 500, demand_units: 480, temp_c: 4.6, volatility: 0 });
  const simulateDecision = async () => {
    try {
      const m = await Decide.once(qd);
      document.dispatchEvent(new CustomEvent('decision:new', { detail: m }));
      alert(`Decision: ${m.action} | SLCA=${m.slca_score} | tx=${m.tx_hash}`);
      refreshAudit();
    } catch (e) {
      alert('Simulation failed: ' + e);
    }
  };

  return (
    <div className="max-w-5xl mx-auto p-4">
      {!apiOk && (
        <div className="mb-4 p-3 rounded-md bg-red-100 text-red-700">
          API not reachable at <b>{API_BASE}</b>. In the browser console, run:&nbsp;
          <code>localStorage.setItem('API_BASE','http://127.0.0.1:8111')</code> &nbsp;then reload.
        </div>
      )}

      <div className="flex gap-2 flex-wrap mb-4">
        {['Policy', 'Blockchain', 'Audit', 'Scenarios', 'QuickDecision', 'Results'].map(t => (
          <button key={t} onClick={() => setTab(t)}
            className={`px-3 py-2 rounded-full ${tab === t ? 'bg-black text-white' : 'bg-gray-200'}`}>
            {t}
          </button>
        ))}
      </div>

      {/* POLICY */}
      {tab === 'Policy' && (
        <Card title="Policy thresholds & weights">
          {policyErr && <div className="mb-2 text-sm text-red-600">{policyErr}</div>}
          {!policy && !policyErr && <div className="text-sm text-gray-500">Loading policy…</div>}
          {policy && (
            <>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm">Min shelf for Reroute</label>
                  <input className="border rounded p-2 w-full" type="number" step="0.01"
                    value={policy.min_shelf_reroute}
                    onChange={e => setPolicy({ ...policy, min_shelf_reroute: parseFloat((e.target as HTMLInputElement).value) })} />
                </div>
                <div>
                  <label className="block text-sm">Min shelf for Expedite</label>
                  <input className="border rounded p-2 w-full" type="number" step="0.01"
                    value={policy.min_shelf_expedite}
                    onChange={e => setPolicy({ ...policy, min_shelf_expedite: parseFloat((e.target as HTMLInputElement).value) })} />
                </div>
                <div>
                  <label className="block text-sm">Carbon factor (transport)</label>
                  <input className="border rounded p-2 w-full" type="number" step="0.001"
                    value={(policy.carbon_factors && policy.carbon_factors.transport) || 0}
                    onChange={e => setPolicy({
                      ...policy,
                      carbon_factors: { ...(policy.carbon_factors || {}), transport: parseFloat((e.target as HTMLInputElement).value) }
                    })} />
                </div>
                <div>
                  <label className="block text-sm">Carbon factor (cold_chain)</label>
                  <input className="border rounded p-2 w-full" type="number" step="0.001"
                    value={(policy.carbon_factors && policy.carbon_factors.cold_chain) || 0}
                    onChange={e => setPolicy({
                      ...policy,
                      carbon_factors: { ...(policy.carbon_factors || {}), cold_chain: parseFloat((e.target as HTMLInputElement).value) }
                    })} />
                </div>
              </div>
              <div className="mt-3 flex gap-2">
                <button onClick={savePolicy} className="px-4 py-2 bg-black text-white rounded">Save Policy</button>
                <button onClick={loadPolicy} className="px-4 py-2 bg-gray-200 rounded">Reload</button>
              </div>
            </>
          )}
        </Card>
      )}

      {/* BLOCKCHAIN */}
      {tab === 'Blockchain' && (
        <Card title="Blockchain (Hardhat)">
          {chainErr && <div className="mb-2 text-sm text-red-600">{chainErr}</div>}
          {!chain && !chainErr && <div className="text-sm text-gray-500">Loading chain config…</div>}
          {chain && (
            <>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm">RPC</label>
                  <input className="border rounded p-2 w-full" type="text"
                    value={chain.rpc}
                    onChange={e => setChain({ ...chain, rpc: (e.target as HTMLInputElement).value })} />
                </div>
                <div>
                  <label className="block text-sm">Chain ID</label>
                  <input className="border rounded p-2 w-full" type="number"
                    value={chain.chain_id}
                    onChange={e => setChain({ ...chain, chain_id: parseInt((e.target as HTMLInputElement).value) })} />
                </div>
                <div className="md:col-span-2">
                  <label className="block text-sm">Private Key (optional for demo)</label>
                  <input className="border rounded p-2 w-full" type="password"
                    value={chain.private_key || ''}
                    onChange={e => setChain({ ...chain, private_key: (e.target as HTMLInputElement).value })} />
                </div>
                <div className="md:col-span-2">
                  <label className="block text-sm">Addresses JSON</label>
                  <textarea className="border rounded p-2 w-full" rows={3}
                    value={JSON.stringify(chain.addresses || {}, null, 2)}
                    onChange={e => { try { setChain({ ...chain, addresses: JSON.parse((e.target as HTMLTextAreaElement).value) }); } catch { } }} />
                </div>
              </div>
              <div className="mt-3 flex gap-2">
                <button onClick={saveChain} className="px-4 py-2 bg-black text-white rounded">Save Chain Config</button>
                <button onClick={loadChain} className="px-4 py-2 bg-gray-200 rounded">Reload</button>
              </div>
            </>
          )}
        </Card>
      )}

      {/* AUDIT */}
      {tab === 'Audit' && (
        <Card title="Decision Log (latest first)">
          <div className="flex items-center gap-2 mb-3">
            <button onClick={refreshAudit} className="px-3 py-1 rounded bg-gray-200">Refresh</button>
            <a className="px-3 py-1 rounded bg-black text-white"
              href={`${API_BASE}/audit/memo.pdf`} target="_blank" rel="noopener">
              Download Decision Memo (PDF)
            </a>
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="text-left border-b">
                  <th className="py-1 pr-4">Time</th>
                  <th className="py-1 pr-4">Action</th>
                  <th className="py-1 pr-4">SLCA</th>
                  <th className="py-1 pr-4">Carbon (kg)</th>
                  <th className="py-1 pr-4">Reason</th>
                  <th className="py-1 pr-4">Tx</th>
                </tr>
              </thead>
              <tbody>
                {[...logs].reverse().map((r, i) => {
                  const ts = r.ts ? new Date(r.ts * 1000).toLocaleString() : (r.time || '');
                  return (
                    <tr key={i} className="border-b">
                      <td className="py-1 pr-4">{ts}</td>
                      <td className="py-1 pr-4">{r.action ?? r.decision ?? ''}</td>
                      <td className="py-1 pr-4">{r.slca_score ?? r.slca ?? ''}</td>
                      <td className="py-1 pr-4">{r.carbon_kg ?? ''}</td>
                      <td className="py-1 pr-4">{r.reason ?? r.note ?? ''}</td>
                      <td className="py-1 pr-4">{r.tx_hash ?? r.tx ?? ''}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* SCENARIOS */}
      {tab === 'Scenarios' && (
        <Card title="Research Scenarios (affect KPIs & decision behavior)">
          <div className="flex items-center gap-3 mb-3">
            <label className="text-sm text-gray-600">
              Intensity
              <input
                type="number"
                step="0.1"
                min="0.1"
                className="ml-2 border rounded-md px-2 py-1 w-20"
                value={scIntensity}
                onChange={e => setScIntensity((e.target as HTMLInputElement).value)}
              />
            </label>
            <button className="px-3 py-2 rounded bg-gray-200" onClick={resetScenario}>Reset</button>
          </div>

          <ul className="space-y-3">
            {scList.map(s => (
              <li key={s.id} className="p-4 rounded-md border">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-semibold">{s.label || s.id}</div>
                    {s.desc ? <div className="text-sm text-gray-600">{s.desc}</div> : null}
                  </div>
                  <button className="px-4 py-2 rounded-md bg-black text-white"
                    onClick={() => runScenario(s.id)}>Run</button>
                </div>
              </li>
            ))}
            {scList.length === 0 && (
              <li className="text-sm text-gray-600">No scenarios discovered.</li>
            )}
          </ul>

          {scActive && (
            <div className="mt-2 text-sm">
              Active: <b>{scActive.name}</b> (intensity={String(scActive.intensity)})
            </div>
          )}
          {scMsg && <div className="mt-2 text-sm">{scMsg}</div>}

          <p className="mt-4 text-sm text-gray-500">
            Tip: After running a scenario, open <b>Operations</b>/<b>Quality</b> in the main app
            and try <b>QuickDecision</b> to observe changes. The PDF memo header will reflect updated KPIs.
          </p>
        </Card>
      )}

      {/* QUICK DECISION */}
      {tab === 'QuickDecision' && (
        <Card title="Take a Quick Decision">
          <div className="flex items-center gap-3 mb-4">
            <label className="text-sm">
              Role&nbsp;
              <select className="border rounded px-3 py-2" value={role} onChange={e => setRole((e.target as HTMLSelectElement).value)}>
                <option value="farm">farm</option>
                <option value="processor">processor</option>
                <option value="distributor">distributor</option>
                <option value="retail">retail</option>
              </select>
            </label>
            <button onClick={quickTake} className="px-4 py-2 bg-black text-white rounded" data-skip-global-take="1">
              Take decision
            </button>
            <a className="px-4 py-2 rounded bg-gray-200" href={`${API_BASE}/audit/memo.pdf`} target="_blank" rel="noopener">
              Open PDF
            </a>
          </div>
          {qdMsg && <div className="mb-6 text-sm">{qdMsg}</div>}

          {/* optional advanced simulator retained */}
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm">Inventory Units</label>
              <input className="border rounded p-2 w-full" type="number"
                value={qd.inventory_units}
                onChange={e => setQD({ ...qd, inventory_units: parseFloat((e.target as HTMLInputElement).value) })} />
            </div>
            <div>
              <label className="block text-sm">Demand Units</label>
              <input className="border rounded p-2 w-full" type="number"
                value={qd.demand_units}
                onChange={e => setQD({ ...qd, demand_units: parseFloat((e.target as HTMLInputElement).value) })} />
            </div>
            <div>
              <label className="block text-sm">Temperature (°C)</label>
              <input className="border rounded p-2 w-full" type="number" step="0.1"
                value={qd.temp_c}
                onChange={e => setQD({ ...qd, temp_c: parseFloat((e.target as HTMLInputElement).value) })} />
            </div>
            <div>
              <label className="block text-sm">Volatility (0..1)</label>
              <input className="border rounded p-2 w-full" type="number" step="0.05"
                value={qd.volatility}
                onChange={e => setQD({ ...qd, volatility: parseFloat((e.target as HTMLInputElement).value) })} />
            </div>
          </div>
          <div className="mt-3 flex gap-2">
            <button onClick={simulateDecision} className="px-4 py-2 bg-gray-800 text-white rounded">
              Simulate (advanced)
            </button>
          </div>
        </Card>
      )}

      {/* RESULTS */}
      {tab === 'Results' && (
        <Card title="Simulation Results (5 Scenarios x 5 Modes)">
          <div className="flex items-center gap-3 mb-4">
            <button onClick={runSimulation} disabled={resultsLoading}
              className="px-4 py-2 bg-black text-white rounded disabled:opacity-50">
              {resultsLoading ? 'Running simulation…' : 'Generate Results'}
            </button>
          </div>
          {resultsErr && <div className="text-red-600 text-sm mb-3">{resultsErr}</div>}
          {resultsData && (
            <div className="space-y-4">
              <h3 className="font-semibold">Summary by Scenario x Mode</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm border">
                  <thead>
                    <tr className="bg-gray-100 text-left">
                      <th className="p-2 border">Scenario</th>
                      <th className="p-2 border">Mode</th>
                      <th className="p-2 border">ARI</th>
                      <th className="p-2 border">RLE</th>
                      <th className="p-2 border">Waste</th>
                      <th className="p-2 border">SLCA</th>
                      <th className="p-2 border">Carbon</th>
                      <th className="p-2 border">Equity</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(resultsData).flatMap(([scenario, modes]: [string, any]) =>
                      Object.entries(modes).map(([mode, m]: [string, any]) => (
                        <tr key={`${scenario}-${mode}`} className="border-b hover:bg-gray-50">
                          <td className="p-2 border">{scenario}</td>
                          <td className="p-2 border font-mono">{mode}</td>
                          <td className="p-2 border">{m.ari?.toFixed(3)}</td>
                          <td className="p-2 border">{m.rle?.toFixed(3)}</td>
                          <td className="p-2 border">{m.waste?.toFixed(3)}</td>
                          <td className="p-2 border">{m.slca?.toFixed(3)}</td>
                          <td className="p-2 border">{m.carbon?.toFixed(0)}</td>
                          <td className="p-2 border">{m.equity?.toFixed(3)}</td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}
          {figFiles.length > 0 && (
            <div className="mt-6 space-y-4">
              <h3 className="font-semibold">Generated Figures</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {figFiles.map(f => (
                  <div key={f} className="border rounded p-2">
                    <img src={`${API_BASE}/results/figures/${f}`} alt={f}
                      className="w-full" onError={e => (e.target as HTMLImageElement).style.display = 'none'} />
                    <div className="text-xs text-gray-500 mt-1">{f}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
          <p className="mt-4 text-sm text-gray-500">
            Runs all 5 scenarios (baseline, heatwave, overproduction, cyber_outage, adaptive_pricing) across
            5 modes (static, hybrid_rl, no_pinn, no_slca, agribrain). Results are saved to
            mvp/simulation/results/.
          </p>
        </Card>
      )}
    </div>
  );
}
