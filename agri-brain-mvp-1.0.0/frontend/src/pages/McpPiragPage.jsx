import React, { useEffect, useState, useMemo, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn, fmt, short, jget, mcpCall, mcpRaw, mcpLog, authFetch } from "@/lib/utils";
import { getApiBase } from "@/mvp/api.js";
import { motion, useInView, AnimatePresence } from "framer-motion";
import { toast } from "sonner";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip as ReTooltip, ResponsiveContainer,
  CartesianGrid, Legend, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, PieChart, Pie, Cell, ErrorBar,
} from "recharts";
import {
  Brain, Wrench, BookOpen, Search, Zap, Shield, Hash, GitBranch,
  Copy, Play, Loader2, ScrollText, ArrowRight, Layers, Network,
  CheckCircle2, AlertTriangle, RefreshCw, Database, FileText,
} from "lucide-react";

const API = getApiBase();

const COLORS = { agri: "#009688", pirag: "#2196F3", mcp: "#FF9800" };

const FEATURE_LABELS = [
  { key: "compliance_severity", label: "Compliance", color: "#ef4444", desc: "Regulatory violation severity from MCP compliance checks (0 = compliant, 1 = critical)" },
  { key: "forecast_urgency", label: "Forecast", color: "#f97316", desc: "Spoilage risk urgency from physics-informed Arrhenius forecasting" },
  { key: "retrieval_confidence", label: "Retrieval", color: "#3b82f6", desc: "piRAG document retrieval confidence score (normalized BM25+TF-IDF)" },
  { key: "regulatory_pressure", label: "Regulatory", color: "#a855f7", desc: "Binary regulatory pressure flag from piRAG keyword extraction" },
  { key: "recovery_saturation", label: "Recovery", color: "#22c55e", desc: "Capacity saturation of recovery/composting channels" },
];

const LOGIT_ENTRIES = [
  { key: "cold_chain", label: "Cold Chain", color: "#0072B2" },
  { key: "local_redistribute", label: "Redistribute", color: "#10B981" },
  { key: "recovery", label: "Recovery", color: "#D55E00" },
];

// Ablation data — loaded dynamically from table2_ablation.csv via API
const ABLATION_DATA_FALLBACK = [];

function parseAblationCSV(text, metric = "ARI") {
  const lines = text.trim().split("\n");
  const headers = lines[0].split(",").map((h) => h.trim());
  const rows = lines.slice(1).map((line) => {
    const vals = line.split(",").map((v) => v.trim());
    const obj = {};
    headers.forEach((h, i) => { const n = +vals[i]; obj[h] = Number.isFinite(n) && vals[i] !== "" ? n : vals[i]; });
    return obj;
  });
  const scenarioNames = { heatwave: "Heatwave", overproduction: "Overproduction", cyber_outage: "Cyber Outage", adaptive_pricing: "Pricing", baseline: "Baseline" };
  const scenarios = [...new Set(rows.map((r) => r.Scenario))];
  return scenarios.map((sc) => {
    const scRows = rows.filter((r) => r.Scenario === sc);
    const get = (v) => scRows.find((r) => r.Variant === v)?.[metric] ?? 0;
    return { scenario: scenarioNames[sc] || sc, no_context: get("no_context"), mcp_only: get("mcp_only"), pirag_only: get("pirag_only"), agribrain: get("agribrain") };
  });
}

// Store raw CSV text for re-parsing with different metrics
let _rawAblationCSV = "";

const KB_DOCUMENTS = [
  { id: "regulatory_fda_leafy_greens", category: "Regulatory", title: "FDA Guidelines for Leafy Greens Storage" },
  { id: "regulatory_usda_organic", category: "Regulatory", title: "USDA Organic Cold Chain Standards" },
  { id: "regulatory_fsma_produce", category: "Regulatory", title: "FSMA Produce Safety Rule" },
  { id: "regulatory_codex_alimentarius", category: "Regulatory", title: "Codex Alimentarius Food Hygiene" },
  { id: "sop_cold_chain_transport", category: "SOP", title: "Cold Chain Transport Standard Procedure" },
  { id: "sop_quality_inspection", category: "SOP", title: "Incoming Quality Inspection Protocol" },
  { id: "sop_warehouse_storage", category: "SOP", title: "Warehouse Storage Temperature Management" },
  { id: "sop_last_mile_delivery", category: "SOP", title: "Last Mile Delivery Guidelines" },
  { id: "temperature_excursion_protocol", category: "SOP", title: "Temperature Excursion Response Protocol" },
  { id: "iot_sensor_spec", category: "Technical", title: "IoT Sensor Calibration Specifications" },
  { id: "slca_methodology_leafy", category: "SLCA", title: "Social LCA Methodology for Leafy Greens" },
  { id: "slca_labor_standards", category: "SLCA", title: "Labor Standards in Agricultural Supply Chains" },
  { id: "carbon_footprint_transport", category: "Environmental", title: "Carbon Footprint of Refrigerated Transport" },
  { id: "water_footprint_spinach", category: "Environmental", title: "Water Footprint Analysis for Spinach" },
  { id: "waste_hierarchy_food", category: "Environmental", title: "Food Waste Hierarchy Best Practices" },
  { id: "composting_guidelines", category: "Environmental", title: "Industrial Composting Guidelines" },
  { id: "food_bank_redistribution", category: "Contingency", title: "Food Bank Redistribution Protocols" },
  { id: "animal_feed_conversion", category: "Contingency", title: "Animal Feed Conversion Standards" },
  { id: "emergency_recall_procedure", category: "Contingency", title: "Emergency Product Recall Procedure" },
  { id: "demand_forecasting_guide", category: "Technical", title: "Demand Forecasting Methodology Guide" },
];

const CAT_COLORS = {
  Regulatory: "bg-purple-500/10 text-purple-600 dark:text-purple-400 border-0",
  SOP: "bg-blue-500/10 text-blue-600 dark:text-blue-400 border-0",
  SLCA: "bg-teal-500/10 text-teal-600 dark:text-teal-400 border-0",
  Environmental: "bg-green-500/10 text-green-600 dark:text-green-400 border-0",
  Technical: "bg-gray-500/10 text-gray-600 dark:text-gray-400 border-0",
  Contingency: "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-0",
};

const PIPELINE_STEPS = [
  { icon: Network, label: "Agent", sub: "5 roles", color: "text-gray-600 dark:text-gray-400", bg: "bg-gray-500/10" },
  { icon: Wrench, label: "MCP Tools", sub: "12 tools", color: "text-orange-600 dark:text-orange-400", bg: "bg-orange-500/10" },
  { icon: BookOpen, label: "piRAG", sub: "20 docs", color: "text-blue-600 dark:text-blue-400", bg: "bg-blue-500/10" },
  { icon: Layers, label: "Context", sub: "5D vector", color: "text-teal-600 dark:text-teal-400", bg: "bg-teal-500/10" },
  { icon: Brain, label: "Policy", sub: "Softmax", color: "text-purple-600 dark:text-purple-400", bg: "bg-purple-500/10" },
  { icon: CheckCircle2, label: "Decision", sub: "3 actions", color: "text-emerald-600 dark:text-emerald-400", bg: "bg-emerald-500/10" },
];

const TOOL_PRESETS = {
  check_compliance: { temperature: "14.0", humidity: "85.0", product_type: "spinach" },
  pirag_query: { query: "FDA temperature violation corrective action", k: "4" },
  explain: { action: "local_redistribute", role: "farm", scenario: "heatwave", rho: "0.35", temperature: "14.0" },
};

// ===================== Tab 1: Overview =====================
function OverviewTab({ tools, ablationData = [], benchSummary = null }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true });
  const [ablationMetric, setAblationMetric] = useState("ARI");

  const data = ablationData.length ? ablationData : ABLATION_DATA_FALLBACK;
  const avgNoCtx = data.length ? data.reduce((s, d) => s + d.no_context, 0) / data.length : 0;
  const avgFull = data.length ? data.reduce((s, d) => s + d.agribrain, 0) / data.length : 0;
  const improvement = ((avgFull - avgNoCtx) / avgNoCtx * 100).toFixed(1);

  const metrics = [
    { label: "MCP Tools", value: tools.length || 12, icon: Wrench, color: "text-orange-600", bg: "bg-orange-500/10" },
    { label: "KB Documents", value: 20, icon: BookOpen, color: "text-blue-600", bg: "bg-blue-500/10" },
    { label: "Context Dims", value: "5D", icon: Layers, color: "text-teal-600", bg: "bg-teal-500/10" },
    { label: "Operating Modes", value: 8, icon: Network, color: "text-purple-600", bg: "bg-purple-500/10" },
  ];

  return (
    <div className="space-y-6" ref={ref}>
      {/* Pipeline */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={inView ? { opacity: 1, y: 0 } : {}} transition={{ delay: 0 }}>
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Architecture Pipeline</CardTitle>
            <CardDescription>Data flow from agent observation to context-informed decision</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap items-center justify-center gap-2 md:gap-0">
              {PIPELINE_STEPS.map((step, i) => (
                <React.Fragment key={i}>
                  <div className={cn("flex flex-col items-center gap-1.5 px-4 py-3 rounded-lg", step.bg)}>
                    <step.icon className={cn("w-6 h-6", step.color)} />
                    <span className="text-xs font-semibold">{step.label}</span>
                    <span className="text-[10px] text-muted-foreground">{step.sub}</span>
                  </div>
                  {i < PIPELINE_STEPS.length - 1 && (
                    <ArrowRight className="w-4 h-4 text-muted-foreground hidden md:block shrink-0" />
                  )}
                </React.Fragment>
              ))}
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Metrics */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={inView ? { opacity: 1, y: 0 } : {}} transition={{ delay: 0.1 }}>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {metrics.map((m, i) => (
            <Card key={i}>
              <CardContent className="p-4 flex items-center gap-3">
                <div className={cn("p-2 rounded-lg", m.bg)}>
                  <m.icon className={cn("w-5 h-5", m.color)} />
                </div>
                <div>
                  <p className="text-2xl font-bold">{m.value}</p>
                  <p className="text-xs text-muted-foreground">{m.label}</p>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </motion.div>

      {/* Ablation chart with metric dropdown + error bars */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={inView ? { opacity: 1, y: 0 } : {}} transition={{ delay: 0.2 }}>
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-base">Context Integration Impact ({ablationMetric})</CardTitle>
                <CardDescription>Ablation study: progressive context addition across 5 scenarios</CardDescription>
              </div>
              <div className="flex items-center gap-2">
                <Badge className="bg-teal-500/10 text-teal-600 border-0">+{improvement}% avg ARI</Badge>
                <Select value={ablationMetric} onValueChange={(v) => {
                  setAblationMetric(v);
                }}>
                  <SelectTrigger className="w-24 h-8"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {["ARI", "RLE", "Waste", "SLCA", "Carbon", "Equity"].map((m) => (
                      <SelectItem key={m} value={m}>{m}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            {(() => {
              // Re-parse CSV data for the selected metric
              const metricData = _rawAblationCSV ? parseAblationCSV(_rawAblationCSV, ablationMetric) : data;
              const scenarioKeyMap = { "Heatwave": "heatwave", "Overproduction": "overproduction", "Cyber Outage": "cyber_outage", "Pricing": "adaptive_pricing", "Baseline": "baseline" };
              const metricKey = ablationMetric.toLowerCase();
              // Add CI error data from benchmark
              const chartData = metricData.map((d) => {
                const sc = scenarioKeyMap[d.scenario] || d.scenario;
                const out = { ...d };
                for (const mode of ["no_context", "mcp_only", "pirag_only", "agribrain"]) {
                  const ci = benchSummary?.[sc]?.[mode]?.[metricKey];
                  if (ci && ci.ci_low != null && ci.ci_high != null) {
                    out[mode] = ci.mean;
                    out[`${mode}_err`] = [ci.mean - ci.ci_low, ci.ci_high - ci.mean];
                  }
                }
                return out;
              });
              return (
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData} barGap={2}>
                      <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                      <XAxis dataKey="scenario" tick={{ fontSize: 11 }} />
                      <YAxis tick={{ fontSize: 11 }} />
                      <ReTooltip contentStyle={{ fontSize: 12 }} />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      <Bar dataKey="no_context" name="No Context" fill="#4CAF50" radius={[2, 2, 0, 0]}>
                        {benchSummary && <ErrorBar dataKey="no_context_err" width={3} strokeWidth={1} stroke="#555" />}
                      </Bar>
                      <Bar dataKey="mcp_only" name="MCP Only" fill={COLORS.mcp} radius={[2, 2, 0, 0]}>
                        {benchSummary && <ErrorBar dataKey="mcp_only_err" width={3} strokeWidth={1} stroke="#555" />}
                      </Bar>
                      <Bar dataKey="pirag_only" name="piRAG Only" fill={COLORS.pirag} radius={[2, 2, 0, 0]}>
                        {benchSummary && <ErrorBar dataKey="pirag_only_err" width={3} strokeWidth={1} stroke="#555" />}
                      </Bar>
                      <Bar dataKey="agribrain" name="AGRI-BRAIN" fill={COLORS.agri} radius={[2, 2, 0, 0]}>
                        {benchSummary && <ErrorBar dataKey="agribrain_err" width={3} strokeWidth={1} stroke="#555" />}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              );
            })()}
            <p className="text-xs text-muted-foreground mt-3 text-center">
              Full AGRI-BRAIN (MCP + piRAG) consistently outperforms single-source ablations.
            </p>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}

// ===================== Tab 2: Context Features =====================
function ContextFeaturesTab({ latestExplainability }) {
  if (!latestExplainability) {
    return (
      <Card className="p-8 text-center">
        <Brain className="w-8 h-8 mx-auto text-muted-foreground mb-3" />
        <p className="text-sm text-muted-foreground">Take a decision on the Decisions page to see live context features.</p>
      </Card>
    );
  }

  const cf = latestExplainability.context_features || {};
  const la = latestExplainability.logit_adjustment || {};
  const ctf = latestExplainability.counterfactual;

  const radarData = FEATURE_LABELS.map((f) => ({ axis: f.label, value: cf[f.key] ?? 0 }));

  return (
    <div className="space-y-6">
      {/* Radar + Values */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <Shield className="w-4 h-4 text-teal-600" /> Context Feature Radar
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={radarData}>
                    <PolarGrid className="opacity-30" />
                    <PolarAngleAxis dataKey="axis" tick={{ fontSize: 11 }} />
                    <PolarRadiusAxis tick={{ fontSize: 9 }} domain={[0, 1]} />
                    <Radar name="Context" dataKey="value" stroke="#009688" fill="#009688" fillOpacity={0.25} strokeWidth={2} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <Layers className="w-4 h-4 text-blue-600" /> Feature Values & Logit Adjustment
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                {FEATURE_LABELS.map((f) => (
                  <div key={f.key} className="flex items-center gap-2 text-sm">
                    <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: f.color }} />
                    <span className="w-24 text-muted-foreground">{f.label}</span>
                    <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                      <div className="h-full rounded-full transition-all" style={{ width: `${(cf[f.key] ?? 0) * 100}%`, background: f.color }} />
                    </div>
                    <span className="font-mono w-10 text-right text-xs">{fmt(cf[f.key], 3)}</span>
                  </div>
                ))}
              </div>
              <Separator />
              <div className="space-y-2">
                <p className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider">Logit Adjustment (context → policy)</p>
                {LOGIT_ENTRIES.map((e) => {
                  const val = la[e.key] ?? 0;
                  const pct = Math.min(Math.abs(val) * 50, 50);
                  return (
                    <div key={e.key} className="flex items-center gap-2 text-sm">
                      <span className="w-24 text-muted-foreground">{e.label}</span>
                      <div className="flex-1 h-2.5 rounded-full bg-muted relative overflow-hidden">
                        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-border" />
                        <div
                          className="absolute top-0 bottom-0 rounded-full"
                          style={{
                            ...(val < 0 ? { right: "50%", width: `${pct}%` } : { left: "50%", width: `${pct}%` }),
                            background: e.color, opacity: 0.7,
                          }}
                        />
                      </div>
                      <span className={cn("font-mono w-14 text-right text-xs", val > 0 ? "text-emerald-600" : val < 0 ? "text-red-500" : "")}>
                        {val > 0 ? "+" : ""}{fmt(val, 3)}
                      </span>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Feature descriptions */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          {FEATURE_LABELS.map((f) => (
            <Card key={f.key} className="bg-muted/30">
              <CardContent className="p-3">
                <div className="flex items-center gap-2 mb-1">
                  <span className="w-2.5 h-2.5 rounded-full" style={{ background: f.color }} />
                  <span className="text-xs font-semibold">{f.label}</span>
                </div>
                <p className="text-[11px] text-muted-foreground leading-relaxed">{f.desc}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </motion.div>

      {/* Counterfactual */}
      {ctf && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-600" /> Counterfactual Comparison
              </CardTitle>
              <CardDescription>What would the policy have decided without MCP/piRAG context?</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <p className="text-xs font-semibold text-teal-600">WITH Context</p>
                  {["cold_chain", "local_redistribute", "recovery"].map((a, i) => (
                    <div key={a} className="flex items-center gap-2 text-xs">
                      <span className="w-24 text-muted-foreground capitalize">{a.replace("_", " ")}</span>
                      <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                        <div className="h-full rounded-full bg-teal-500" style={{ width: `${(ctf.probs_with_context?.[i] ?? 0) * 100}%` }} />
                      </div>
                      <span className="font-mono w-14 text-right">{fmt((ctf.probs_with_context?.[i] ?? 0) * 100, 1)}%</span>
                    </div>
                  ))}
                </div>
                <div className="space-y-2">
                  <p className="text-xs font-semibold text-amber-600">WITHOUT Context</p>
                  {["cold_chain", "local_redistribute", "recovery"].map((a, i) => (
                    <div key={a} className="flex items-center gap-2 text-xs">
                      <span className="w-24 text-muted-foreground capitalize">{a.replace("_", " ")}</span>
                      <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                        <div className="h-full rounded-full bg-amber-500" style={{ width: `${(ctf.probs_without_context?.[i] ?? 0) * 100}%` }} />
                      </div>
                      <span className="font-mono w-14 text-right">{fmt((ctf.probs_without_context?.[i] ?? 0) * 100, 1)}%</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="mt-3 text-center">
                <Badge className={ctf.action_changed ? "bg-red-500/10 text-red-600 border-0" : "bg-emerald-500/10 text-emerald-600 border-0"}>
                  {ctf.action_changed ? "Context changed the decision" : "Same decision (context reinforced confidence)"}
                </Badge>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}
    </div>
  );
}

// ===================== Tab 3: Knowledge Base =====================
function KnowledgeBaseTab() {
  const [query, setQuery] = useState("");
  const [role, setRole] = useState("farm");
  const [results, setResults] = useState(null);
  const [searching, setSearching] = useState(false);

  const search = async () => {
    if (!query.trim()) return;
    setSearching(true);
    try {
      const res = await mcpCall(API, "pirag_query", {
        query: query.trim(), k: 5, role, temperature: 14.0, rho: 0.3,
        physics_expansion: true, physics_reranking: true,
      });
      setResults(res);
    } catch (e) {
      toast.error(`Search failed: ${e.message}`);
    }
    setSearching(false);
  };

  const stats = [
    { label: "Top-k", value: "4" },
    { label: "Retrieval", value: "BM25 + TF-IDF" },
    { label: "Reranking", value: "Physics-Informed" },
    { label: "Documents", value: "20" },
  ];

  const grouped = useMemo(() => {
    const g = {};
    KB_DOCUMENTS.forEach((d) => { (g[d.category] = g[d.category] || []).push(d); });
    return g;
  }, []);

  return (
    <div className="space-y-6">
      {/* Stats */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {stats.map((s, i) => (
            <Card key={i} className="bg-muted/30">
              <CardContent className="p-3 text-center">
                <p className="text-lg font-bold text-blue-600">{s.value}</p>
                <p className="text-[11px] text-muted-foreground">{s.label}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </motion.div>

      {/* Document inventory */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <Database className="w-4 h-4 text-blue-600" /> Knowledge Base Inventory
            </CardTitle>
            <CardDescription>20 domain-specific documents across 6 categories</CardDescription>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-64">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="text-xs">Document ID</TableHead>
                    <TableHead className="text-xs">Category</TableHead>
                    <TableHead className="text-xs">Title</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {KB_DOCUMENTS.map((doc) => (
                    <TableRow key={doc.id}>
                      <TableCell className="font-mono text-xs">{doc.id}</TableCell>
                      <TableCell>
                        <Badge className={cn("text-[10px]", CAT_COLORS[doc.category] || "")}>{doc.category}</Badge>
                      </TableCell>
                      <TableCell className="text-xs">{doc.title}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </ScrollArea>
          </CardContent>
        </Card>
      </motion.div>

      {/* Live search */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <Search className="w-4 h-4 text-blue-600" /> Live piRAG Search
            </CardTitle>
            <CardDescription>Query the physics-informed retrieval pipeline with BM25+TF-IDF hybrid scoring and Arrhenius-based reranking</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap items-end gap-3">
              <div className="flex-1 min-w-48">
                <Label className="text-xs mb-1.5 block">Query</Label>
                <div className="relative">
                  <Search className="absolute left-2.5 top-2 w-4 h-4 text-muted-foreground" />
                  <Input
                    placeholder="Search the piRAG knowledge base..."
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && search()}
                    className="pl-8"
                  />
                </div>
              </div>
              <div className="w-32">
                <Label className="text-xs mb-1.5 block">Role</Label>
                <Select value={role} onValueChange={setRole}>
                  <SelectTrigger className="h-9"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {["farm", "processor", "cooperative", "distributor", "recovery"].map((r) => (
                      <SelectItem key={r} value={r}>{r.charAt(0).toUpperCase() + r.slice(1)}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <Button onClick={search} disabled={searching}>
                {searching ? <Loader2 className="w-4 h-4 mr-1 animate-spin" /> : <Search className="w-4 h-4 mr-1" />}
                Search
              </Button>
            </div>

            {results && (
              <div className="space-y-2">
                {results.physics_expanded && (
                  <Badge className="bg-blue-500/10 text-blue-600 border-0 text-[10px]">Physics-expanded query</Badge>
                )}
                {(results.results || []).map((doc, i) => (
                  <Card key={i} className="bg-muted/30">
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between gap-2 mb-2">
                        <div className="flex items-center gap-2">
                          <BookOpen className="w-4 h-4 text-blue-500" />
                          <span className="font-mono text-sm font-semibold">{doc.doc_id}</span>
                        </div>
                        <span className="font-mono text-xs text-muted-foreground">score: {fmt(doc.score, 3)}</span>
                      </div>
                      <div className="h-1.5 rounded-full bg-muted overflow-hidden mb-2">
                        <div className="h-full rounded-full bg-blue-500 transition-all" style={{ width: `${Math.min((doc.score || 0) * 100, 100)}%` }} />
                      </div>
                      {doc.keywords && doc.keywords.length > 0 && (
                        <div className="flex flex-wrap gap-1 mb-2">
                          {doc.keywords.map((kw, j) => (
                            <Badge key={j} className="text-[9px] bg-purple-500/10 text-purple-600 dark:text-purple-400 border-0">{kw}</Badge>
                          ))}
                        </div>
                      )}
                      {doc.passage && <p className="text-xs text-muted-foreground line-clamp-4">{doc.passage}</p>}
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}

// ===================== Tab 4: Protocol & Traces =====================
function ProtocolTab({ tools, decisions }) {
  const [selected, setSelected] = useState("");
  const [args, setArgs] = useState({});
  const [result, setResult] = useState(null);
  const [running, setRunning] = useState(false);
  const [log, setLog] = useState([...mcpLog]);
  const intervalRef = useRef(null);

  useEffect(() => {
    intervalRef.current = setInterval(() => setLog([...mcpLog]), 1000);
    return () => clearInterval(intervalRef.current);
  }, []);

  const tool = tools.find((t) => t.name === selected);
  const schema = tool?.inputSchema?.properties || {};

  const applyPreset = () => { const p = TOOL_PRESETS[selected]; if (p) setArgs(p); };

  const run = async () => {
    setRunning(true);
    try {
      const typed = {};
      for (const [k, v] of Object.entries(args)) {
        const prop = schema[k];
        if (prop?.type === "number" || prop?.type === "integer") typed[k] = Number(v);
        else if (prop?.type === "boolean") typed[k] = v === "true";
        else typed[k] = v;
      }
      setResult(await mcpCall(API, selected, typed));
    } catch (e) {
      setResult({ error: e.message });
      toast.error(`Tool error: ${e.message}`);
    }
    setRunning(false);
  };

  const toolStats = useMemo(() => {
    const counts = {};
    decisions.forEach((d) => {
      (d.explainability?.mcp_tools_invoked || []).forEach((t) => { counts[t] = (counts[t] || 0) + 1; });
    });
    return Object.entries(counts).map(([name, value]) => ({ name, value }));
  }, [decisions]);

  const PIE_COLORS = ["#FF9800", "#2196F3", "#009688", "#E91E63", "#9C27B0", "#4CAF50", "#FF5722", "#607D8B"];

  return (
    <div className="space-y-6">
      {/* Tool invocation */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <Zap className="w-4 h-4 text-orange-600" /> Live MCP Tool Invocation
            </CardTitle>
            <CardDescription>Invoke any of the 12 registered MCP tools via JSON-RPC 2.0</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center gap-2">
              <Select value={selected} onValueChange={(v) => { setSelected(v); setArgs({}); setResult(null); }}>
                <SelectTrigger className="flex-1"><SelectValue placeholder="Select a tool..." /></SelectTrigger>
                <SelectContent>
                  {tools.map((t) => <SelectItem key={t.name} value={t.name}>{t.name}</SelectItem>)}
                </SelectContent>
              </Select>
              {TOOL_PRESETS[selected] && <Button variant="outline" size="sm" onClick={applyPreset}>Preset</Button>}
            </div>

            {tool && Object.keys(schema).length > 0 && (
              <div className="space-y-2">
                {Object.entries(schema).map(([key, prop]) => (
                  <div key={key} className="grid grid-cols-3 gap-2 items-center">
                    <Label className="text-xs">
                      {key}{tool.inputSchema?.required?.includes(key) && <span className="text-red-500">*</span>}
                    </Label>
                    <Input
                      className="col-span-2 h-8 text-xs font-mono"
                      placeholder={prop.description || `${prop.type || "string"}`}
                      value={args[key] || ""}
                      onChange={(e) => setArgs((a) => ({ ...a, [key]: e.target.value }))}
                    />
                  </div>
                ))}
              </div>
            )}

            <Button onClick={run} disabled={!selected || running}>
              {running ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Play className="w-4 h-4 mr-2" />}
              Run Tool
            </Button>

            {result && (
              <div className="mt-2">
                {selected === "check_compliance" && result.compliant !== undefined ? (
                  <div className="space-y-2">
                    <Badge className={result.compliant ? "bg-emerald-500/10 text-emerald-600 border-0" : "bg-red-500/10 text-red-600 border-0"}>
                      {result.compliant ? "Compliant" : "Violation"}
                    </Badge>
                    <pre className="p-3 rounded-md bg-muted text-xs font-mono overflow-x-auto max-h-64">{JSON.stringify(result, null, 2)}</pre>
                  </div>
                ) : (
                  <pre className="p-3 rounded-md bg-muted text-xs font-mono overflow-x-auto max-h-64">{JSON.stringify(result, null, 2)}</pre>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* Stats + Log side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
          <Card className="h-full">
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <Wrench className="w-4 h-4 text-orange-600" /> Tool Invocation Breakdown
              </CardTitle>
            </CardHeader>
            <CardContent>
              {toolStats.length > 0 ? (
                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={toolStats} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`} labelLine={false} fontSize={10}>
                        {toolStats.map((_, i) => <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />)}
                      </Pie>
                      <ReTooltip contentStyle={{ fontSize: 12 }} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <p className="text-sm text-muted-foreground text-center py-8">Take decisions to see tool usage statistics.</p>
              )}
            </CardContent>
          </Card>
        </motion.div>

        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
          <Card className="h-full">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-base flex items-center gap-2">
                  <ScrollText className="w-4 h-4" /> Protocol Log
                </CardTitle>
                <Badge variant="outline" className="text-[10px]">{log.length} entries</Badge>
              </div>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-56">
                {log.length === 0 ? (
                  <p className="text-sm text-muted-foreground text-center py-8">No MCP interactions yet. Use the tool invocation above.</p>
                ) : (
                  <div className="space-y-2">
                    {[...log].reverse().map((entry, i) => (
                      <div key={i} className="flex items-start gap-2 text-xs p-2 rounded-md bg-muted/50">
                        <Badge className={cn("text-[9px] shrink-0 border-0 mt-0.5", entry.status === "success" ? "bg-emerald-500/10 text-emerald-600" : "bg-red-500/10 text-red-600")}>
                          {entry.status}
                        </Badge>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="font-mono font-semibold">{entry.method}</span>
                            <span className="text-muted-foreground">{entry.ts?.split("T")[1]?.split(".")[0]}</span>
                          </div>
                          <p className="font-mono text-muted-foreground truncate mt-0.5">{entry.preview}</p>
                        </div>
                        <button onClick={() => { navigator.clipboard.writeText(JSON.stringify(entry, null, 2)); toast.success("Copied"); }} className="text-muted-foreground hover:text-primary shrink-0">
                          <Copy className="w-3 h-3" />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </ScrollArea>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  );
}

// ===================== Tab 5: Causal Reasoning =====================
function CausalReasoningTab({ latestExplainability, latestMemo }) {
  if (!latestExplainability) {
    return (
      <Card className="p-8 text-center">
        <Brain className="w-8 h-8 mx-auto text-muted-foreground mb-3" />
        <p className="text-sm text-muted-foreground">Take a decision on the Decisions page to see causal reasoning analysis.</p>
      </Card>
    );
  }

  const text = latestExplainability.causal_text || latestExplainability.summary || "";
  const chain = latestExplainability.causal_chain;
  const prov = latestExplainability.provenance;
  const keywords = latestExplainability.keywords;
  const toolsInvoked = latestExplainability.mcp_tools_invoked || [];
  const topDoc = latestExplainability.pirag_top_doc;
  const topScore = latestExplainability.pirag_top_score;

  const renderText = (raw) => {
    const parts = raw.split(/(BECAUSE|WITHOUT|AND)/g);
    return parts.map((part, i) => {
      if (part === "BECAUSE") return <span key={i} className="font-bold text-teal-600 dark:text-teal-400">BECAUSE</span>;
      if (part === "WITHOUT") return <span key={i} className="font-bold text-amber-600 dark:text-amber-400">WITHOUT</span>;
      if (part === "AND") return <span key={i} className="font-semibold">AND</span>;
      const withCites = part.split(/(\[KB:[^\]]+\])/g);
      return withCites.map((seg, j) =>
        seg.startsWith("[KB:") ? (
          <Badge key={`${i}-${j}`} variant="outline" className="mx-0.5 text-[10px] font-mono">{seg}</Badge>
        ) : <span key={`${i}-${j}`}>{seg}</span>
      );
    });
  };

  // Build contribution data for bar chart
  const contributions = chain?.all_contributions || {};
  const contribData = Object.entries(contributions)
    .map(([name, value]) => ({ name, value: +value, fill: name === chain?.primary_cause ? "#009688" : "#94a3b8" }))
    .filter((d) => d.value !== 0)
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

  // Flatten keywords
  const kwCategories = [
    { field: "thresholds", label: "Thresholds", cls: "bg-blue-500/10 text-blue-600 dark:text-blue-400 border-0" },
    { field: "regulations", label: "Regulations", cls: "bg-purple-500/10 text-purple-600 dark:text-purple-400 border-0" },
    { field: "required_actions", label: "Actions", cls: "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-0" },
  ];
  const flatKw = { thresholds: [], regulations: [], required_actions: [] };
  if (keywords) {
    for (const [, data] of Object.entries(keywords)) {
      if (typeof data === "object" && data !== null) {
        for (const cat of kwCategories) {
          for (const item of (data[cat.field] || [])) {
            if (!flatKw[cat.field].includes(item)) flatKw[cat.field].push(item);
          }
        }
      }
    }
  }

  const copyHash = (h) => { navigator.clipboard.writeText(h); toast.success("Hash copied"); };

  return (
    <div className="space-y-6">
      {/* Causal explanation */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <Brain className="w-4 h-4 text-teal-600" /> Causal Explanation
              {chain?.primary_cause && <Badge className="bg-teal-500/10 text-teal-600 border-0 text-[10px]">Primary: {chain.primary_cause}</Badge>}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-sm text-muted-foreground leading-relaxed">
              {text.split("\n\n").map((para, i) => (
                <p key={i} className={i > 0 ? "mt-3" : ""}>{renderText(para)}</p>
              ))}
            </div>
          </CardContent>
        </Card>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Contribution chart */}
        {contribData.length > 0 && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
            <Card className="h-full">
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <Layers className="w-4 h-4 text-teal-600" /> Causal Contribution Breakdown
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={contribData} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                      <XAxis type="number" tick={{ fontSize: 10 }} />
                      <YAxis dataKey="name" type="category" width={120} tick={{ fontSize: 10 }} />
                      <ReTooltip contentStyle={{ fontSize: 12 }} />
                      <Bar dataKey="value" name="Contribution" radius={[0, 4, 4, 0]}>
                        {contribData.map((d, i) => <Cell key={i} fill={d.fill} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {/* Keywords */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}>
          <Card className="h-full">
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <BookOpen className="w-4 h-4 text-purple-600" /> Extracted Keywords
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {kwCategories.map((cat) => {
                const items = flatKw[cat.field];
                if (items.length === 0) return null;
                return (
                  <div key={cat.field} className="flex flex-wrap items-center gap-1.5">
                    <span className="text-xs text-muted-foreground w-20 shrink-0">{cat.label}:</span>
                    {items.map((item, i) => <Badge key={i} className={cn("text-[10px]", cat.cls)}>{item}</Badge>)}
                  </div>
                );
              })}
              {Object.values(flatKw).every((a) => a.length === 0) && (
                <p className="text-sm text-muted-foreground text-center py-4">No keywords extracted from this decision.</p>
              )}
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Provenance chain */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <Hash className="w-4 h-4 text-teal-600" /> Provenance Chain
              {prov?.guards_passed !== false && <Badge className="bg-emerald-500/10 text-emerald-600 border-0 text-[10px]">Guards Passed</Badge>}
            </CardTitle>
            <CardDescription>Cryptographic evidence trail from MCP tools and piRAG retrieval to Merkle-rooted provenance</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="relative pl-6">
              <div className="absolute left-2 top-0 bottom-0 w-px bg-border" />

              {/* MCP tool steps */}
              {toolsInvoked.map((t, i) => (
                <div key={i} className="relative pl-6 pb-3">
                  <div className="absolute left-0 top-0.5 w-4 h-4 rounded-full bg-background border-2 border-orange-500 flex items-center justify-center">
                    <Shield className="w-2.5 h-2.5 text-orange-500" />
                  </div>
                  <div className="text-xs">
                    <span className="font-medium">MCP: {t}</span>
                    {prov?.evidence_hashes?.[i] && (
                      <span className="ml-2 font-mono text-[10px] text-muted-foreground/60">SHA: {short(prov.evidence_hashes[i])}</span>
                    )}
                  </div>
                </div>
              ))}

              {/* piRAG step */}
              {topDoc && (
                <div className="relative pl-6 pb-3">
                  <div className="absolute left-0 top-0.5 w-4 h-4 rounded-full bg-background border-2 border-blue-500 flex items-center justify-center">
                    <BookOpen className="w-2.5 h-2.5 text-blue-500" />
                  </div>
                  <div className="text-xs">
                    <span className="font-medium">piRAG: {topDoc}</span>
                    <span className="ml-2 text-muted-foreground">score={fmt(topScore, 2)}</span>
                  </div>
                </div>
              )}

              {/* Merkle root */}
              {prov?.merkle_root && (
                <div className="relative pl-6 pb-3">
                  <div className="absolute left-0 top-0.5 w-4 h-4 rounded-full bg-background border-2 border-teal-500 flex items-center justify-center">
                    <GitBranch className="w-2.5 h-2.5 text-teal-500" />
                  </div>
                  <div className="text-xs">
                    <span className="font-medium">Merkle Root</span>
                    <button onClick={() => copyHash(prov.merkle_root)} className="ml-2 font-mono text-muted-foreground hover:text-primary">
                      {short(prov.merkle_root)} <Copy className="w-2.5 h-2.5 inline" />
                    </button>
                  </div>
                </div>
              )}

              {/* On-chain */}
              {latestMemo?.tx_hash && latestMemo.tx_hash !== "0x0" && (
                <div className="relative pl-6">
                  <div className="absolute left-0 top-0.5 w-4 h-4 rounded-full bg-background border-2 border-emerald-500 flex items-center justify-center">
                    <CheckCircle2 className="w-2.5 h-2.5 text-emerald-500" />
                  </div>
                  <div className="text-xs">
                    <span className="font-medium">On-chain anchor</span>
                    <button onClick={() => copyHash(latestMemo.tx_hash)} className="ml-2 font-mono text-muted-foreground hover:text-primary">
                      {short(latestMemo.tx_hash)} <Copy className="w-2.5 h-2.5 inline" />
                    </button>
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}


export default function McpPiragPage() {
  const [loading, setLoading] = useState(true);
  const [decisions, setDecisions] = useState([]);
  const [tools, setTools] = useState([]);
  const [ablationData, setAblationData] = useState(ABLATION_DATA_FALLBACK);
  const [benchSummary, setBenchSummary] = useState(null);
  const [activeTab, setActiveTab] = useState("overview");

  useEffect(() => {
    const load = async () => {
      const [decData, toolsData] = await Promise.all([
        jget(API, "/decisions").catch(() => ({ decisions: [] })),
        mcpRaw(API, "tools/list").catch(() => ({ tools: [] })),
      ]);
      const decs = decData.decisions || decData;
      setDecisions(Array.isArray(decs) ? decs : []);
      setTools(toolsData.tools || []);
      // Fetch live ablation data from CSV (with auth)
      try {
        const resp = await authFetch(`${API}/results/figures/table2_ablation.csv`);
        if (resp.ok) {
          _rawAblationCSV = await resp.text();
          setAblationData(parseAblationCSV(_rawAblationCSV));
        }
      } catch (e) { console.warn("Could not load ablation CSV:", e); }
      // Fetch benchmark CI data (with auth)
      try {
        const resp = await authFetch(`${API}/results/figures/benchmark_summary.json`);
        if (resp.ok) setBenchSummary(await resp.json());
      } catch (e) { console.warn("Could not load benchmark data:", e); }
      setLoading(false);
    };
    load();
  }, []);

  const latestMemo = useMemo(() => {
    if (!Array.isArray(decisions) || decisions.length === 0) return null;
    return decisions[0]; // most recent
  }, [decisions]);

  const latestExplainability = useMemo(() => latestMemo?.explainability || null, [latestMemo]);

  if (loading) {
    return (
      <div className="space-y-4">
        {[...Array(4)].map((_, i) => <Skeleton key={i} className="h-32 rounded-lg" />)}
      </div>
    );
  }

  return (
    <div className="space-y-6 pb-12">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex items-center gap-3 mb-1">
          <Brain className="w-6 h-6 text-primary" />
          <h1 className="text-2xl font-bold">MCP/piRAG Context Integration</h1>
          <Badge className="bg-teal-500/10 text-teal-600 border-0 text-xs">Research Contribution</Badge>
        </div>
        <p className="text-sm text-muted-foreground">
          Model Context Protocol interoperability and Physics-informed Retrieval-Augmented Generation pipeline
        </p>
      </motion.div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="w-full justify-start flex-wrap">
          <TabsTrigger value="overview" className="flex items-center gap-1.5"><Network className="w-3.5 h-3.5" /> Overview</TabsTrigger>
          <TabsTrigger value="features" className="flex items-center gap-1.5"><Layers className="w-3.5 h-3.5" /> Context Features</TabsTrigger>
          <TabsTrigger value="knowledge" className="flex items-center gap-1.5"><BookOpen className="w-3.5 h-3.5" /> Knowledge Base</TabsTrigger>
          <TabsTrigger value="protocol" className="flex items-center gap-1.5"><Zap className="w-3.5 h-3.5" /> Protocol</TabsTrigger>
          <TabsTrigger value="causal" className="flex items-center gap-1.5"><Brain className="w-3.5 h-3.5" /> Causal Reasoning</TabsTrigger>
        </TabsList>

        <div className="mt-6">
          <TabsContent value="overview"><OverviewTab tools={tools} ablationData={ablationData} benchSummary={benchSummary} /></TabsContent>
          <TabsContent value="features"><ContextFeaturesTab latestExplainability={latestExplainability} /></TabsContent>
          <TabsContent value="knowledge"><KnowledgeBaseTab /></TabsContent>
          <TabsContent value="protocol"><ProtocolTab tools={tools} decisions={decisions} /></TabsContent>
          <TabsContent value="causal"><CausalReasoningTab latestExplainability={latestExplainability} latestMemo={latestMemo} /></TabsContent>
        </div>
      </Tabs>
    </div>
  );
}
