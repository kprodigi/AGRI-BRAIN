// frontend/src/mvp/AdminPanel.jsx
import React, { useEffect, useState } from "react";

const API = (window.API_BASE || localStorage.getItem("API_BASE") || "http://127.0.0.1:8111").replace(/\/$/, "");
const WS_URL = (API || "").replace(/^http/i, "ws") + "/stream";

// ---------------- small JSON helpers ----------------
async function jget(path) {
    const r = await fetch(`${API}${path}`);
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return r.json();
}
async function jpost(path, body = {}) {
    const r = await fetch(`${API}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    try { return await r.json(); } catch { return {}; }
}

// ---------------- minimal JSON-RPC helper ----------------
async function rpc(rpcUrl, method, params = []) {
    const r = await fetch(rpcUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ jsonrpc: "2.0", id: 1, method, params }),
    });
    const j = await r.json();
    if (j.error) throw new Error(j.error.message || "RPC error");
    return j.result;
}
const hexToDec = (h) => (h ? parseInt(h, 16) : 0);
const short = (s) => (s && s.length > 12 ? `${s.slice(0, 8)}…${s.slice(-4)}` : s || "");

// ---------------- UI crumbs ----------------
const Pill = ({ active, children, onClick }) => (
    <button
        onClick={onClick}
        className={active ? "px-5 py-2 rounded-full bg-black text-white" : "px-5 py-2 rounded-full bg-gray-200"}
        style={{ marginRight: 10 }}
    >
        {children}
    </button>
);

export default function AdminPanel() {
    const [tab, setTab] = useState("Policy");
    const [apiOk, setApiOk] = useState(true);

    useEffect(() => {
        fetch(`${API}/health`).then(r => setApiOk(r.ok)).catch(() => setApiOk(false));
    }, []);

    // 🌐 One shared WebSocket for the whole Admin app (decisions + chain head)
    useEffect(() => {
        let ws;
        try {
            ws = new WebSocket(WS_URL);
            ws.onopen = () => console.log("[WS] connected:", WS_URL);
            ws.onclose = () => console.log("[WS] disconnected");
            ws.onmessage = (ev) => {
                try {
                    const msg = JSON.parse(ev.data || "{}");
                    const type = msg?.type;
                    const payload = msg?.payload;

                    if (type === "decision") {
                        // Your existing memo plumbing listens to 'decision:new'
                        document.dispatchEvent(new CustomEvent("decision:new", { detail: payload }));
                    }
                    if (type === "chain/head") {
                        // Fan out as a DOM event so any tab can consume it
                        document.dispatchEvent(new CustomEvent("chain:head", { detail: payload }));
                    }
                    if (type === "agent/error" || type === "chain/error") {
                        console.warn("[agent]", payload?.error || payload);
                    }
                } catch (e) {
                    console.warn("[WS parse]", e);
                }
            };
        } catch (e) {
            console.warn("[WS] failed to connect:", e);
        }
        return () => { try { ws && ws.close(); } catch { } };
    }, []);

    return (
        <div className="max-w-6xl mx-auto p-6 space-y-6">
            {!apiOk && (
                <div className="p-3 rounded-md bg-red-100 text-red-700">
                    API not reachable at <b>{API}</b>. Set it in your browser console with:
                    <code className="ml-1">localStorage.setItem('API_BASE','http://127.0.0.1:8111')</code> then reload.
                </div>
            )}

            <div className="flex items-center gap-3">
                {["Policy", "Blockchain", "Audit", "Scenarios", "QuickDecision"].map(t => (
                    <Pill key={t} active={tab === t} onClick={() => setTab(t)}>{t}</Pill>
                ))}
            </div>

            {tab === "Policy" && <PolicyTab />}
            {tab === "Blockchain" && <ChainTab active={tab === "Blockchain"} />}
            {tab === "Audit" && <AuditTab />}
            {tab === "Scenarios" && <ScenariosTab />}
            {tab === "QuickDecision" && <QuickDecisionTab />}
        </div>
    );
}

/* ---------------------- Policy ---------------------- */
function PolicyTab() {
    const [form, setForm] = useState({
        min_shelf_reroute: 0.70,
        min_shelf_expedite: 0.50,
        carbon_transport: 0.12,
        carbon_cold_chain: 0.08,
    });
    const [loading, setLoading] = useState(true);
    const [msg, setMsg] = useState("");

    useEffect(() => {
        (async () => {
            try {
                const p = await jget("/governance/policy");
                setForm({
                    min_shelf_reroute: p.min_shelf_reroute ?? 0.70,
                    min_shelf_expedite: p.min_shelf_expedite ?? 0.50,
                    carbon_transport: p.carbon_transport ?? 0.12,
                    carbon_cold_chain: p.carbon_cold_chain ?? 0.08,
                });
            } catch { /* defaults ok */ }
            finally { setLoading(false); }
        })();
    }, []);

    const save = async () => {
        setMsg("");
        try { await jpost("/governance/policy", form); setMsg("✅ Saved policy."); }
        catch (e) { setMsg("❌ Failed to save policy."); console.warn(e); }
    };

    if (loading) return <div>Loading policy…</div>;
    const set = (k) => (e) => setForm((s) => ({ ...s, [k]: Number(e.target.value) }));

    return (
        <div className="rounded-xl border p-5">
            <h2 className="text-xl font-semibold mb-4">Policy thresholds & weights</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <LabeledInput label="Min shelf for Reroute" value={form.min_shelf_reroute} onChange={set("min_shelf_reroute")} />
                <LabeledInput label="Min shelf for Expedite" value={form.min_shelf_expedite} onChange={set("min_shelf_expedite")} />
                <LabeledInput label="Carbon factor (transport)" value={form.carbon_transport} onChange={set("carbon_transport")} />
                <LabeledInput label="Carbon factor (cold_chain)" value={form.carbon_cold_chain} onChange={set("carbon_cold_chain")} />
            </div>
            <button className="mt-4 px-4 py-2 rounded-md bg-black text-white" onClick={save}>Save Policy</button>
            {msg && <div className="mt-3 text-sm">{msg}</div>}
        </div>
    );
}

/* ---------------------- Blockchain with auto status + LIVE head ---------------------- */
function ChainTab({ active }) {
    const [form, setForm] = useState({
        rpc: "http://127.0.0.1:8545",
        chain_id: 31337,
        private_key: "",
        addresses_json: "",
    });
    const [msg, setMsg] = useState("");
    const [autoSync, setAutoSync] = useState(true);
    const [status, setStatus] = useState({
        chainIdDec: null,
        blockNumber: null,     // polled via RPC
        _live_block: null,     // live via WS
        decisionLoggerOk: null,
        lastTx: null,
        lastReceiptStatus: null,
        lastReceiptBlock: null,
        error: null,
    });

    // load saved chain config
    useEffect(() => {
        (async () => {
            try {
                const c = await jget("/governance/chain");
                setForm({
                    rpc: c.rpc ?? "http://127.0.0.1:8545",
                    chain_id: c.chain_id ?? 31337,
                    private_key: c.private_key ?? "",
                    addresses_json: c.addresses_json ?? "",
                });
            } catch { /* defaults ok */ }
        })();
    }, []);

    // listen for live heads broadcast by the top-level WS
    useEffect(() => {
        const onHead = (e) => {
            const blk = e?.detail?.block ?? e?.detail?.number ?? null;
            if (blk != null) setStatus((s) => ({ ...s, _live_block: blk }));
        };
        window.addEventListener("chain:head", onHead);
        return () => window.removeEventListener("chain:head", onHead);
    }, []);

    // parse addresses JSON safely
    const parsedAddresses = (() => {
        try { return form.addresses_json ? JSON.parse(form.addresses_json) : {}; }
        catch { return {}; }
    })();

    // save config
    const set = (k) => (e) => setForm((s) => ({ ...s, [k]: e.target.value }));
    const save = async () => {
        setMsg("");
        try { await jpost("/governance/chain", form); setMsg("✅ Saved blockchain config."); }
        catch (e) { setMsg("❌ Failed to save blockchain config."); console.warn(e); }
    };

    // auto-refresh on-chain status
    useEffect(() => {
        if (!active || !autoSync) return;
        let stop = false;

        const refresh = async () => {
            try {
                const [cidHex, blkHex] = await Promise.all([
                    rpc(form.rpc, "eth_chainId").catch(() => null),
                    rpc(form.rpc, "eth_blockNumber").catch(() => null),
                ]);
                const chainIdDec = cidHex ? hexToDec(cidHex) : null;
                const blockNumber = blkHex ? hexToDec(blkHex) : null;

                // find DecisionLogger if provided
                const dlAddr =
                    parsedAddresses.DecisionLogger ||
                    parsedAddresses.DECISION_LOGGER ||
                    parsedAddresses.decisionLogger ||
                    parsedAddresses.decision_logger ||
                    null;

                let decisionLoggerOk = null;
                if (dlAddr) {
                    const code = await rpc(form.rpc, "eth_getCode", [dlAddr, "latest"]).catch(() => "0x");
                    decisionLoggerOk = code && code !== "0x";
                }

                // check last tx hash from audit logs (if any)
                let lastTx = null, lastReceiptStatus = null, lastReceiptBlock = null;
                try {
                    const logs = await jget("/audit/logs");
                    const items = logs.items || [];
                    const lastWithTx = [...items].reverse().find(x => x.tx_hash || x.tx || x.txHash);
                    const txh = lastWithTx?.tx_hash || lastWithTx?.tx || lastWithTx?.txHash || null;
                    if (txh) {
                        lastTx = txh;
                        const receipt = await rpc(form.rpc, "eth_getTransactionReceipt", [txh]).catch(() => null);
                        if (receipt) {
                            lastReceiptStatus = receipt.status === "0x1" ? "success" : (receipt.status ? "failed" : null);
                            lastReceiptBlock = receipt.blockNumber ? hexToDec(receipt.blockNumber) : null;
                        }
                    }
                } catch { /* fine */ }

                if (!stop) {
                    setStatus({
                        chainIdDec,
                        blockNumber,
                        _live_block: null, // will be filled by WS if available
                        decisionLoggerOk,
                        lastTx,
                        lastReceiptStatus,
                        lastReceiptBlock,
                        error: null,
                    });
                }
            } catch (e) {
                if (!stop) setStatus((s) => ({ ...s, error: e.message || "RPC error" }));
            }
        };

        // refresh immediately and every 4s
        refresh();
        const t = setInterval(refresh, 4000);
        return () => { stop = true; clearInterval(t); };
    }, [active, autoSync, form.rpc, form.addresses_json]); // re-run when these change

    return (
        <div className="rounded-xl border p-5">
            <h2 className="text-xl font-semibold mb-4">Blockchain / DAO</h2>

            <div className="grid md:grid-cols-2 gap-4">
                <Text label="RPC" value={form.rpc} onChange={set("rpc")} />
                <Text label="Chain ID (saved)" value={form.chain_id} onChange={set("chain_id")} />
                <Text label="Private Key (optional)" value={form.private_key} onChange={set("private_key")} />
                <TextArea label="Addresses (JSON)" value={form.addresses_json} onChange={set("addresses_json")} rows={5} />
            </div>

            <div className="mt-3 flex items-center gap-3">
                <button className="px-4 py-2 rounded-md bg-black text-white" onClick={save}>Save</button>
                <label className="ml-2 text-sm flex items-center gap-2">
                    <input type="checkbox" checked={autoSync} onChange={e => setAutoSync(e.target.checked)} />
                    Auto-sync on-chain status
                </label>
            </div>

            {msg && <div className="mt-3 text-sm">{msg}</div>}

            {/* Live status */}
            <div className="mt-5 rounded-lg border p-4 bg-gray-50">
                <div className="flex items-center justify-between">
                    <h3 className="font-semibold">On-chain status</h3>
                    <span className="text-xs text-gray-500">{autoSync ? "refreshing every ~4s" : "paused"}</span>
                </div>
                <div className="mt-2 grid md:grid-cols-3 gap-3 text-sm">
                    <Stat label="RPC reachable" value={status.blockNumber !== null ? "yes" : "no"} />
                    <Stat label="Chain ID (RPC)" value={status.chainIdDec ?? "—"} />
                    <Stat label="Block # (RPC poll)" value={status.blockNumber ?? "—"} />
                </div>

                <div className="mt-3 grid md:grid-cols-3 gap-3 text-sm">
                    <Stat label="Block # (live)" value={status._live_block ?? "—"} />
                    <Stat label="DecisionLogger address" value={parsedAddresses.DecisionLogger || parsedAddresses.decisionLogger || "—"} mono />
                    <Stat label="DecisionLogger deployed?" value={status.decisionLoggerOk === null ? "—" : status.decisionLoggerOk ? "yes" : "no"} />
                </div>

                <div className="mt-3 grid md:grid-cols-3 gap-3 text-sm">
                    <Stat label="Last audit tx" value={short(status.lastTx) || "—"} mono />
                    <Stat label="Last tx status" value={status.lastReceiptStatus || "—"} />
                    <Stat label="Last tx block" value={status.lastReceiptBlock ?? "—"} />
                </div>

                <div className="mt-3 grid md:grid-cols-3 gap-3 text-sm">
                    <Stat label="Error" value={status.error || "—"} />
                </div>
            </div>
        </div>
    );
}

/* ---------------------- Audit ---------------------- */
function AuditTab() {
    const [items, setItems] = useState([]);
    const [msg, setMsg] = useState("");

    const load = async () => {
        setMsg("");
        try {
            const r = await jget("/audit/logs");
            setItems(r.items || []);
        } catch (e) {
            setMsg("❌ Could not load logs.");
            console.warn(e);
        }
    };
    useEffect(() => { load(); }, []);

    return (
        <div className="rounded-xl border p-5">
            <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold">Audit</h2>
                <div className="flex items-center gap-3">
                    <a className="px-3 py-2 rounded-md bg-black text-white" href={`${API}/report/pdf`} target="_blank" rel="noopener">
                        Download Decision Memo (PDF)
                    </a>
                    <button className="px-3 py-2 rounded-md bg-gray-200" onClick={load}>Refresh</button>
                </div>
            </div>
            {msg && <div className="mt-3 text-sm">{msg}</div>}
            <ul className="mt-4 space-y-2 text-sm">
                {items.length === 0 && <li className="text-gray-500">No audit events.</li>}
                {items.map((it, i) => (
                    <li key={i} className="p-2 rounded-md bg-gray-50 border">
                        <code className="text-xs">{JSON.stringify(it)}</code>
                    </li>
                ))}
            </ul>
        </div>
    );
}

/* ---------------------- Scenarios (as you have it, with list + fallback) ---------------------- */
function ScenariosTab() {
    const [loading, setLoading] = useState(true);
    const [options, setOptions] = useState([]);          // [{id,name,desc}]
    const [selected, setSelected] = useState(null);      // id | null
    const [intensity, setIntensity] = useState(1.0);     // 0.5 .. 1.5
    const [msg, setMsg] = useState("");

    const FALLBACK = [
        { id: "climate_shock", name: "Climate-Induced Supply Shock", desc: "72h heatwave; accelerated spoilage; reconfigure routes." },
        { id: "reverse_logistics", name: "Reverse Logistics of Spoiled Food", desc: "Glut / overproduction; trigger redistribution and recovery." },
        { id: "cyber_outage", name: "Cyber Threat & Node Outage", desc: "Processor offline; unauthorized tx blocked; reroute flows." },
        { id: "adaptive_pricing", name: "Adaptive Pricing & Cooperative Auctions", desc: "Learned pricing; equity-aware redistribution when saturated." },
    ];

    const normalize = (raw) => {
        if (raw && Array.isArray(raw.scenarios)) {
            return {
                options: raw.scenarios.map(s => ({ id: s.id, name: s.label || s.name || s.id, desc: s.desc || "" })),
                selected: raw.active?.name || null,
            };
        }
        if (raw && Array.isArray(raw.options)) {
            return { options: raw.options.map(s => ({ id: s.id, name: s.name || s.id, desc: s.desc || "" })), selected: raw.selected || null };
        }
        return { options: [], selected: null };
    };

    const loadList = async () => {
        setMsg(""); setLoading(true);
        try {
            const raw = await jget("/scenarios/list");
            const n = normalize(raw);
            if (n.options.length) { setOptions(n.options); setSelected(n.selected); }
            else { setOptions(FALLBACK); setSelected(null); }
        } catch {
            setOptions(FALLBACK); setSelected(null);
        } finally { setLoading(false); }
    };

    useEffect(() => { loadList(); }, []);

    const runScenario = async (id) => {
        setMsg("");
        try {
            try { await jpost("/scenarios/run", { name: id, intensity }); }
            catch { await jpost("/scenarios", { id }); }
            setSelected(id);
            setMsg(`✅ Scenario set: ${id} (intensity ${intensity.toFixed(2)})`);
        } catch (e) {
            setMsg("❌ Failed to apply scenario."); console.warn(e);
        }
    };

    const resetScenario = async () => {
        setMsg("");
        try { await jpost("/scenarios/reset", {}); setSelected(null); setMsg("✅ Scenario reset."); }
        catch (e) { setMsg("❌ Failed to reset scenario."); console.warn(e); }
    };

    const loadSpinach = async () => {
        setMsg("");
        try { await jpost("/case/load", {}); setMsg("✅ Loaded demo case (spinach). Refresh the main app to see KPIs."); }
        catch (e) { setMsg("❌ Failed to load case."); console.warn(e); }
    };

    return (
        <div className="rounded-xl border p-5">
            <h2 className="text-xl font-semibold mb-4">Scenarios</h2>

            {loading ? (
                <div>Loading scenarios…</div>
            ) : (
                <>
                    {options.length === 0 ? (
                        <p className="text-sm text-gray-600">
                            No scenarios available. Open DevTools → Network and check <code>/scenarios/list</code>.
                        </p>
                    ) : (
                        <>
                            <div className="flex flex-wrap gap-2">
                                {options.map(o => (
                                    <button
                                        key={o.id}
                                        onClick={() => runScenario(o.id)}
                                        className={`px-3 py-2 rounded ${selected === o.id ? 'bg-black text-white' : 'bg-gray-200'}`}
                                        title={o.desc || o.name}
                                    >
                                        {o.name}
                                    </button>
                                ))}
                            </div>

                            <div className="mt-4 flex items-center gap-3">
                                <span className="text-sm text-gray-600">Intensity</span>
                                <input
                                    type="range" min="0.5" max="1.5" step="0.1"
                                    value={intensity}
                                    onChange={e => setIntensity(parseFloat(e.target.value))}
                                />
                                <span className="text-sm font-medium w-10 text-center">{intensity.toFixed(2)}</span>

                                <button className="ml-3 px-3 py-1 rounded bg-gray-200" onClick={resetScenario}>Reset</button>
                                <button className="px-3 py-1 rounded bg-gray-200" onClick={loadList}>Refresh</button>
                            </div>
                        </>
                    )}

                    <div className="mt-6">
                        <button className="px-4 py-2 rounded-md bg-black text-white" onClick={loadSpinach}>
                            Load spinach demo data
                        </button>
                    </div>

                    {msg && <div className="mt-3 text-sm">{msg}</div>}

                    <p className="mt-4 text-sm text-gray-600">
                        Tip: tweak policy thresholds in <b>Policy</b>, then try a scenario and run <b>QuickDecision</b> to see behavior shift.
                    </p>
                </>
            )}
        </div>
    );
}

/* ---------------------- Quick Decision ---------------------- */
function QuickDecisionTab() {
    const [role, setRole] = useState("farm");
    const [msg, setMsg] = useState("");

    const take = async () => {
        setMsg("");
        try {
            // Post BOTH fields so either backend variant accepts it
            const res = await jpost("/decide", { agent: `admin:${role}`, agent_id: `admin:${role}`, role });
            const memo = res?.memo || res || {};
            document.dispatchEvent(new CustomEvent("decision:new", { detail: memo }));
            const action = memo.decision ?? memo.action ?? "(unknown)";
            const slca = Number(memo.slca ?? memo.slca_score ?? 0).toFixed(3);
            const co2 = memo.carbon_kg ?? memo.carbon ?? 0;
            setMsg(`✅ role=${role} | ${action} | SLCA ${slca} | CO₂ ${co2} kg`);
        } catch (e) {
            setMsg("❌ Could not take decision.");
            console.warn(e);
        }
    };

    return (
        <div className="rounded-xl border p-5">
            <h2 className="text-xl font-semibold mb-4">Quick Decision</h2>
            <div className="flex items-center gap-3">
                <select className="border rounded-md px-3 py-2" value={role} onChange={e => setRole(e.target.value)}>
                    <option value="farm">farm</option>
                    <option value="processor">processor</option>
                    <option value="distributor">distributor</option>
                    <option value="retail">retail</option>
                </select>
                <button
                    className="px-4 py-2 rounded-md bg-black text-white"
                    onClick={take}
                    data-skip-global-take="1" // prevent any global click hook (if present)
                >
                    Take decision
                </button>
                <a className="px-4 py-2 rounded bg-gray-200" href={`${API}/report/pdf`} target="_blank" rel="noopener">
                    Open PDF
                </a>
            </div>
            {msg && <div className="mt-3 text-sm">{msg}</div>}
        </div>
    );
}

/* ---------------------- small inputs & stats ---------------------- */
function LabeledInput({ label, value, onChange }) {
    return (
        <label className="block">
            <div className="text-sm text-gray-600 mb-1">{label}</div>
            <input type="number" step="0.01" value={value} onChange={onChange}
                className="w-full border rounded-md px-3 py-2" />
        </label>
    );
}
function Text({ label, value, onChange }) {
    return (
        <label className="block">
            <div className="text-sm text-gray-600 mb-1">{label}</div>
            <input type="text" value={value} onChange={onChange}
                className="w-full border rounded-md px-3 py-2" />
        </label>
    );
}
function TextArea({ label, value, onChange, rows = 4 }) {
    return (
        <label className="block">
            <div className="text-sm text-gray-600 mb-1">{label}</div>
            <textarea rows={rows} value={value} onChange={onChange}
                className="w-full border rounded-md px-3 py-2" />
        </label>
    );
}
function Stat({ label, value, mono }) {
    return (
        <div>
            <div className="text-xs text-gray-500">{label}</div>
            <div className={mono ? "font-mono" : ""}>{value}</div>
        </div>
    );
}
