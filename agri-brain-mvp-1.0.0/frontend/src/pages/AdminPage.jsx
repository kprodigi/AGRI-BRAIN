import React, { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Skeleton } from "@/components/ui/skeleton";
import { cn, fmt, short, jget as jgetUtil, jpost as jpostUtil } from "@/lib/utils";
import { getApiBase } from "@/mvp/api.js";
import { toast } from "sonner";
import { motion, AnimatePresence } from "framer-motion";
import {
  Settings, Link2, Shield, Flame, Zap, Save, RefreshCw, Play, RotateCcw,
  ChevronDown, AlertTriangle, CheckCircle2, XCircle, Loader2, HelpCircle,
  FileText, Cloud, ShieldAlert, DollarSign, Layers, Search, Download,
  Info, Plug,
} from "lucide-react";
import McpTab from "@/components/mcp/McpTab";

const API = getApiBase();

async function jget(path) { return jgetUtil(API, path); }
async function jpost(path, body) { return jpostUtil(API, path, body); }

// Minimal JSON-RPC helper
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

// ===================== Policy Tab =====================
function PolicyTab() {
  const [form, setForm] = useState({ min_shelf_reroute: 0.70, min_shelf_expedite: 0.50, carbon_per_km: 0.12, eta: 0.50 });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const p = await jget("/governance/policy");
        setForm({ min_shelf_reroute: p.min_shelf_reroute ?? 0.70, min_shelf_expedite: p.min_shelf_expedite ?? 0.50, carbon_per_km: p.carbon_per_km ?? 0.12, eta: p.eta ?? 0.50 });
      } catch {}
      setLoading(false);
    })();
  }, []);

  const save = async () => {
    setSaving(true);
    try { await jpost("/governance/policy", form); toast.success("Policy saved successfully"); }
    catch { toast.error("Failed to save policy"); }
    setSaving(false);
  };

  const fields = [
    { key: "min_shelf_reroute", label: "Min Shelf for Reroute", unit: "fraction", desc: "Minimum shelf-life fraction before triggering reroute (0-1)", group: "Routing Parameters" },
    { key: "min_shelf_expedite", label: "Min Shelf for Expedite", unit: "fraction", desc: "Minimum shelf-life fraction before expedited delivery (0-1)", group: "Routing Parameters" },
    { key: "carbon_per_km", label: "Carbon per km", unit: "kg CO₂/km", desc: "Carbon emission factor per kilometer of transport", group: "Carbon Parameters" },
    { key: "eta", label: "Waste Penalty (\u03b7)", unit: "weight", desc: "Weight of waste penalty in the objective function (0-1)", group: "SLCA Weights" },
  ];

  if (loading) return <div className="space-y-4">{[...Array(4)].map((_, i) => <Skeleton key={i} className="h-16" />)}</div>;

  const groups = [...new Set(fields.map((f) => f.group))];

  return (
    <div className="space-y-6">
      {groups.map((group) => (
        <Card key={group}>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">{group}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {fields.filter((f) => f.group === group).map((field) => (
              <div key={field.key} className="grid grid-cols-1 sm:grid-cols-3 gap-3 items-center">
                <div className="flex items-center gap-2">
                  <Label className="text-sm">{field.label}</Label>
                  <Tooltip>
                    <TooltipTrigger><HelpCircle className="w-3.5 h-3.5 text-muted-foreground" /></TooltipTrigger>
                    <TooltipContent className="max-w-xs">{field.desc}</TooltipContent>
                  </Tooltip>
                </div>
                <Input
                  type="number"
                  step="0.01"
                  value={form[field.key]}
                  onChange={(e) => setForm((s) => ({ ...s, [field.key]: Number(e.target.value) }))}
                  className="font-mono"
                />
                <span className="text-xs text-muted-foreground">{field.unit}</span>
              </div>
            ))}
          </CardContent>
        </Card>
      ))}
      <Button onClick={save} disabled={saving}>
        {saving ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Save className="w-4 h-4 mr-2" />}
        Save Changes
      </Button>
    </div>
  );
}

// ===================== Blockchain Tab =====================
function BlockchainTab() {
  const [form, setForm] = useState({ rpc: "http://127.0.0.1:8545", chain_id: 31337, private_key: "", addresses_json: "" });
  const [autoSync, setAutoSync] = useState(true);
  const [status, setStatus] = useState({ chainIdDec: null, blockNumber: null, _live_block: null, decisionLoggerOk: null, lastTx: null, lastReceiptStatus: null, error: null });
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const c = await jget("/governance/chain");
        setForm({ rpc: c.rpc ?? "http://127.0.0.1:8545", chain_id: c.chain_id ?? 31337, private_key: c.private_key ?? "", addresses_json: c.addresses_json ?? "" });
      } catch {}
    })();
  }, []);

  useEffect(() => {
    const onHead = (e) => {
      const blk = e?.detail?.block ?? e?.detail?.number ?? null;
      if (blk != null) setStatus((s) => ({ ...s, _live_block: blk }));
    };
    window.addEventListener("chain:head", onHead);
    return () => window.removeEventListener("chain:head", onHead);
  }, []);

  useEffect(() => {
    if (!autoSync) return;
    let stop = false;
    const refresh = async () => {
      try {
        const [cidHex, blkHex] = await Promise.all([
          rpc(form.rpc, "eth_chainId").catch(() => null),
          rpc(form.rpc, "eth_blockNumber").catch(() => null),
        ]);
        if (!stop) {
          setStatus((s) => ({
            ...s,
            chainIdDec: cidHex ? hexToDec(cidHex) : null,
            blockNumber: blkHex ? hexToDec(blkHex) : null,
            error: null,
          }));
        }
      } catch (e) {
        if (!stop) setStatus((s) => ({ ...s, error: e.message }));
      }
    };
    refresh();
    const t = setInterval(refresh, 4000);
    return () => { stop = true; clearInterval(t); };
  }, [autoSync, form.rpc]);

  const save = async () => {
    setSaving(true);
    try { await jpost("/governance/chain", form); toast.success("Blockchain config saved"); }
    catch { toast.error("Failed to save config"); }
    setSaving(false);
  };

  const connected = status.blockNumber !== null;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-base">Connection Status</CardTitle>
            <div className="flex items-center gap-2">
              <Label className="text-sm">Auto-sync</Label>
              <Switch checked={autoSync} onCheckedChange={setAutoSync} />
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-3 rounded-lg bg-muted/50 text-center">
              <div className="flex items-center justify-center gap-2 mb-1">
                {connected ? <CheckCircle2 className="w-4 h-4 text-emerald-500" /> : <XCircle className="w-4 h-4 text-red-500" />}
                <span className="text-xs text-muted-foreground">RPC</span>
              </div>
              <p className="font-medium text-sm">{connected ? "Connected" : "Offline"}</p>
            </div>
            <div className="p-3 rounded-lg bg-muted/50 text-center">
              <p className="text-xs text-muted-foreground mb-1">Chain ID</p>
              <p className="font-mono font-medium text-sm">{status.chainIdDec ?? "\u2014"}</p>
            </div>
            <div className="p-3 rounded-lg bg-muted/50 text-center">
              <p className="text-xs text-muted-foreground mb-1">Block (RPC)</p>
              <p className="font-mono font-medium text-sm">{status.blockNumber ?? "\u2014"}</p>
            </div>
            <div className="p-3 rounded-lg bg-muted/50 text-center">
              <p className="text-xs text-muted-foreground mb-1">Block (Live)</p>
              <p className="font-mono font-medium text-sm">{status._live_block ?? "\u2014"}</p>
            </div>
          </div>
          {status.error && (
            <div className="mt-3 p-2 rounded-md bg-destructive/10 text-sm text-destructive flex items-center gap-2">
              <AlertTriangle className="w-4 h-4" /> {status.error}
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3"><CardTitle className="text-base">Configuration</CardTitle></CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <Label className="text-sm mb-1.5 block">RPC URL</Label>
              <Input value={form.rpc} onChange={(e) => setForm((s) => ({ ...s, rpc: e.target.value }))} className="font-mono text-sm" />
            </div>
            <div>
              <Label className="text-sm mb-1.5 block">Chain ID</Label>
              <Input value={form.chain_id} onChange={(e) => setForm((s) => ({ ...s, chain_id: e.target.value }))} className="font-mono text-sm" />
            </div>
            <div>
              <Label className="text-sm mb-1.5 block">Private Key (optional)</Label>
              <Input type="password" value={form.private_key} onChange={(e) => setForm((s) => ({ ...s, private_key: e.target.value }))} className="font-mono text-sm" />
            </div>
            <div>
              <Label className="text-sm mb-1.5 block">Addresses (JSON)</Label>
              <textarea
                rows={3}
                value={form.addresses_json}
                onChange={(e) => setForm((s) => ({ ...s, addresses_json: e.target.value }))}
                className="flex w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm font-mono shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
              />
            </div>
          </div>
          <Button onClick={save} disabled={saving}>
            {saving ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Save className="w-4 h-4 mr-2" />}
            Save Config
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}

// ===================== Audit Tab =====================
function AuditTab() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [expandedRow, setExpandedRow] = useState(null);

  const load = async () => {
    setLoading(true);
    try { const r = await jget("/audit/logs"); setItems(r.items || []); }
    catch { toast.error("Could not load audit logs"); }
    setLoading(false);
  };

  useEffect(() => { load(); }, []);

  const filtered = items.filter((it) => !search || JSON.stringify(it).toLowerCase().includes(search.toLowerCase()));

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-3">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-2.5 top-2.5 w-4 h-4 text-muted-foreground" />
          <Input placeholder="Search audit logs..." value={search} onChange={(e) => setSearch(e.target.value)} className="pl-8" />
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" asChild>
            <a href={`${API}/report/pdf`} target="_blank" rel="noopener"><FileText className="w-4 h-4 mr-1" /> PDF Report</a>
          </Button>
          <Button variant="outline" size="sm" onClick={load}><RefreshCw className="w-4 h-4 mr-1" /> Refresh</Button>
        </div>
      </div>

      <Card>
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Timestamp</TableHead>
                <TableHead>Agent</TableHead>
                <TableHead>Action</TableHead>
                <TableHead>SLCA</TableHead>
                <TableHead>Carbon</TableHead>
                <TableHead>Tx Hash</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {loading ? (
                [...Array(5)].map((_, i) => (
                  <TableRow key={i}>
                    {[...Array(6)].map((_, j) => <TableCell key={j}><Skeleton className="h-4 w-full" /></TableCell>)}
                  </TableRow>
                ))
              ) : filtered.length === 0 ? (
                <TableRow><TableCell colSpan={6} className="text-center text-muted-foreground py-8">No audit events found.</TableCell></TableRow>
              ) : (
                filtered.map((it, i) => (
                  <React.Fragment key={i}>
                    <TableRow className="cursor-pointer hover:bg-muted/50" onClick={() => setExpandedRow(expandedRow === i ? null : i)}>
                      <TableCell className="font-mono text-xs">{it.time || it.timestamp || "\u2014"}</TableCell>
                      <TableCell className="text-sm">{it.agent || "\u2014"}</TableCell>
                      <TableCell><Badge variant="secondary" className="text-xs">{it.decision || it.action || "\u2014"}</Badge></TableCell>
                      <TableCell className="font-mono text-sm">{fmt(it.slca ?? it.slca_score, 3)}</TableCell>
                      <TableCell className="font-mono text-sm">{fmt(it.carbon_kg, 1)}</TableCell>
                      <TableCell className="font-mono text-xs">{short(it.tx || it.tx_hash)}</TableCell>
                    </TableRow>
                    {expandedRow === i && (
                      <TableRow>
                        <TableCell colSpan={6} className="bg-muted/30">
                          <pre className="text-xs font-mono overflow-x-auto p-2">{JSON.stringify(it, null, 2)}</pre>
                        </TableCell>
                      </TableRow>
                    )}
                  </React.Fragment>
                ))
              )}
            </TableBody>
          </Table>
        </div>
      </Card>
    </div>
  );
}

// ===================== Scenarios Tab =====================
const SCENARIO_CARDS = [
  { id: "climate_shock", name: "Heatwave", desc: "72h heatwave; accelerated spoilage; reconfigure routes.", icon: Flame, color: "#D55E00" },
  { id: "reverse_logistics", name: "Overproduction", desc: "Glut / overproduction; trigger redistribution and recovery.", icon: Layers, color: "#E67E22" },
  { id: "cyber_outage", name: "Cyber Outage", desc: "Processor offline; unauthorized tx blocked; reroute flows.", icon: ShieldAlert, color: "#7570B3" },
  { id: "adaptive_pricing", name: "Adaptive Pricing", desc: "Learned pricing; equity-aware redistribution when saturated.", icon: DollarSign, color: "#0072B2" },
  { id: "baseline", name: "Baseline", desc: "Normal operating conditions for reference comparison.", icon: Shield, color: "#808080" },
];

function ScenariosTab() {
  const [selected, setSelected] = useState(null);
  const [intensity, setIntensity] = useState(1.0);
  const [running, setRunning] = useState(null);
  const [results, setResults] = useState({});

  const runScenario = async (id) => {
    setRunning(id);
    try {
      await jpost("/scenarios/run", { name: id, intensity });
      setSelected(id);
      setResults((r) => ({ ...r, [id]: "complete" }));
      toast.success(`Scenario "${id}" applied (intensity ${intensity.toFixed(2)})`);
    } catch {
      toast.error("Failed to apply scenario");
    }
    setRunning(null);
  };

  const reset = async () => {
    try { await jpost("/scenarios/reset", {}); setSelected(null); setResults({}); toast.success("Scenarios reset"); }
    catch { toast.error("Failed to reset"); }
  };

  const loadCase = async () => {
    try { await jpost("/case/load", {}); toast.success("Demo case loaded — refresh main app to see KPIs"); }
    catch { toast.error("Failed to load case"); }
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {SCENARIO_CARDS.map((s) => (
          <Card key={s.id} className={cn("transition-all", selected === s.id && "border-primary shadow-md")}>
            <CardContent className="p-5">
              <div className="flex items-start gap-3">
                <div className="p-2 rounded-lg" style={{ background: s.color + "15" }}>
                  <s.icon className="w-5 h-5" style={{ color: s.color }} />
                </div>
                <div className="flex-1 min-w-0">
                  <h4 className="font-semibold text-sm">{s.name}</h4>
                  <p className="text-xs text-muted-foreground mt-1">{s.desc}</p>
                  <div className="mt-3 flex items-center gap-2">
                    <Button size="sm" variant={selected === s.id ? "default" : "outline"} onClick={() => runScenario(s.id)} disabled={running === s.id}>
                      {running === s.id ? <Loader2 className="w-3 h-3 mr-1 animate-spin" /> : <Play className="w-3 h-3 mr-1" />}
                      Run
                    </Button>
                    {results[s.id] && <Badge variant="success" className="text-[10px]">Complete</Badge>}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card>
        <CardContent className="p-4 flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-3">
            <Label className="text-sm whitespace-nowrap">Intensity</Label>
            <input type="range" min="0.5" max="1.5" step="0.1" value={intensity} onChange={(e) => setIntensity(parseFloat(e.target.value))} className="w-32" />
            <span className="font-mono text-sm w-10">{intensity.toFixed(2)}</span>
          </div>
          <Separator orientation="vertical" className="h-8 hidden sm:block" />
          <Button variant="outline" size="sm" onClick={reset}><RotateCcw className="w-4 h-4 mr-1" /> Reset All</Button>
          <Button variant="outline" size="sm" onClick={loadCase}><Download className="w-4 h-4 mr-1" /> Load Demo Data</Button>
        </CardContent>
      </Card>
    </div>
  );
}

// ===================== Quick Decision Tab =====================
function QuickDecisionTab() {
  const [role, setRole] = useState("farm");
  const [result, setResult] = useState(null);
  const [taking, setTaking] = useState(false);

  const take = async () => {
    setTaking(true);
    setResult(null);
    try {
      const res = await jpost("/decide", { agent: `admin:${role}`, agent_id: `admin:${role}`, role });
      const memo = res?.memo || res || {};
      document.dispatchEvent(new CustomEvent("decision:new", { detail: memo }));
      setResult(memo);
      toast.success("Decision taken");
    } catch {
      toast.error("Could not take decision");
    }
    setTaking(false);
  };

  return (
    <div className="max-w-lg mx-auto">
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2"><Zap className="w-5 h-5 text-primary" /> Quick Decision</CardTitle>
          <CardDescription>Take an immediate routing decision for a selected agent role.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <Label className="text-sm mb-1.5 block">Agent Role</Label>
            <Select value={role} onValueChange={setRole}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                {["farm", "processor", "cooperative", "distributor", "recovery"].map((r) => (
                  <SelectItem key={r} value={r}>{r.charAt(0).toUpperCase() + r.slice(1)}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="flex gap-2">
            <Button className="flex-1" onClick={take} disabled={taking} data-skip-global-take="1">
              {taking ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Zap className="w-4 h-4 mr-2" />}
              Take Decision
            </Button>
            <Button variant="outline" asChild>
              <a href={`${API}/report/pdf`} target="_blank" rel="noopener"><FileText className="w-4 h-4" /></a>
            </Button>
          </div>

          {result && (
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
              <Card className="bg-primary/5 border-primary/20">
                <CardContent className="p-4 space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Action</span>
                    <Badge variant="teal">{result.decision ?? result.action ?? "\u2014"}</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">SLCA</span>
                    <span className="font-mono font-semibold">{fmt(result.slca ?? result.slca_score, 3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Carbon</span>
                    <span className="font-mono">{fmt(result.carbon_kg ?? result.carbon, 1)} kg</span>
                  </div>
                  {(result.tx_hash || result.tx) && (
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Tx</span>
                      <code className="font-mono text-xs">{short(result.tx_hash || result.tx)}</code>
                    </div>
                  )}
                </CardContent>
              </Card>
            </motion.div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

// ===================== Main Admin Page =====================
export default function AdminPage() {
  const [apiOk, setApiOk] = useState(true);

  useEffect(() => {
    fetch(`${API}/health`).then((r) => setApiOk(r.ok)).catch(() => setApiOk(false));
  }, []);

  return (
    <div className="space-y-6">
      {!apiOk && (
        <Card className="border-destructive/50 bg-destructive/5">
          <CardContent className="p-4 flex items-center gap-3 text-sm text-destructive">
            <AlertTriangle className="w-5 h-5 shrink-0" />
            <div>
              API not reachable at <code className="font-mono">{API}</code>. Set it via{" "}
              <code className="font-mono text-xs">localStorage.setItem('API_BASE','http://127.0.0.1:8111')</code> then reload.
            </div>
          </CardContent>
        </Card>
      )}

      <Tabs defaultValue="policy">
        <TabsList className="w-full justify-start">
          <TabsTrigger value="policy" className="flex items-center gap-1.5"><Settings className="w-4 h-4" /> Policy</TabsTrigger>
          <TabsTrigger value="blockchain" className="flex items-center gap-1.5"><Link2 className="w-4 h-4" /> Blockchain</TabsTrigger>
          <TabsTrigger value="audit" className="flex items-center gap-1.5"><Shield className="w-4 h-4" /> Audit</TabsTrigger>
          <TabsTrigger value="scenarios" className="flex items-center gap-1.5"><Flame className="w-4 h-4" /> Scenarios</TabsTrigger>
          <TabsTrigger value="quick" className="flex items-center gap-1.5"><Zap className="w-4 h-4" /> Quick Decision</TabsTrigger>
          <TabsTrigger value="mcp" className="flex items-center gap-1.5"><Plug className="w-4 h-4" /> MCP</TabsTrigger>
        </TabsList>

        <div className="mt-6">
          <TabsContent value="policy"><PolicyTab /></TabsContent>
          <TabsContent value="blockchain"><BlockchainTab /></TabsContent>
          <TabsContent value="audit"><AuditTab /></TabsContent>
          <TabsContent value="scenarios"><ScenariosTab /></TabsContent>
          <TabsContent value="quick"><QuickDecisionTab /></TabsContent>
          <TabsContent value="mcp"><McpTab /></TabsContent>
        </div>
      </Tabs>
    </div>
  );
}
