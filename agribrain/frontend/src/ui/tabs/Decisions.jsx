import React, { useCallback, useEffect, useState } from "react";
import { authDownload, getApiKey } from "@/lib/utils";

export default function Decisions({ API }) {
  const [decisions, setDecisions] = useState([]);
  const [role, setRole] = useState("farm");

  const load = useCallback(async () => {
    const key = getApiKey();
    const headers = key ? { "x-api-key": key } : {};
    const res = await fetch(`${API}/decisions`, { headers });
    const data = await res.json();
    setDecisions(data.decisions || []);
  }, [API]);

  const act = async () => {
    const key = getApiKey();
    const headers = { "Content-Type": "application/json" };
    if (key) headers["x-api-key"] = key;
    await fetch(`${API}/decide`, {
      method: "POST",
      headers,
      body: JSON.stringify({ agent_id: `demo:${role}`, role }),
    });
    load();
  };

  useEffect(() => {
    void load();
    const id = setInterval(() => { void load(); }, 4000);
    return () => clearInterval(id);
  }, [load]);

  const filtered = decisions.filter((m) => m.role === role);

  return (
    <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div className="card">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold mb-2">Decision Memos</h3>
          <div className="flex items-center gap-2">
            <select className="border rounded px-2 py-1" value={role} onChange={(e) => setRole(e.target.value)}>
              <option>farm</option>
              <option>processor</option>
              <option>cooperative</option>
              <option>distributor</option>
              <option>recovery</option>
            </select>
            <button className="px-3 py-1 rounded bg-slate-900 text-white" onClick={act} data-skip-global-take="1">
              Take decision
            </button>
          </div>
        </div>
        <div className="max-h-72 overflow-auto text-sm">
          {filtered.slice().reverse().map((m, i) => (
            <div key={i} className="border-b py-2">
              <div className="font-semibold">{(m.time || "").replace("T", " ").slice(0, 16)} - {m.decision}</div>
              <div>
                SLCA: <b>{m.slca}</b> | Carbon: {m.carbon_kg} kg | Price: {m.unit_price}
                {m.circular_economy_score != null && ` | Circular: ${m.circular_economy_score}`}
              </div>
              <div>Agent: {m.agent} ({m.role}) | Shelf: {m.shelf_left} | Volatility: {m.volatility} | tx: {m.tx}</div>
              <div className="text-slate-600">{m.note}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <h3 className="font-semibold mb-2">Download Report</h3>
        <button
          className="px-3 py-2 rounded bg-slate-900 text-white inline-block"
          onClick={() => authDownload(`${API}/report/pdf?role=${encodeURIComponent(role)}`, `decision-report-${role}.pdf`)}
        >
          Download Decision Memo (PDF)
        </button>
        <div className="text-xs text-slate-600 mt-2">KPIs + latest decision with SLCA, carbon, tx hash.</div>
      </div>
    </section>
  );
}
