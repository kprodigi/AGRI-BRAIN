import React, { useEffect, useState, useMemo, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn, fmt, jget, jpost, authFetch } from "@/lib/utils";
import { getApiBase } from "@/mvp/api.js";
import {
  PRIMARY_PUBLICATION_MODES, SECONDARY_PUBLICATION_MODES, canonicalH1Evidence,
  descriptiveCrossScenarioSummary, directionalAdvantagePercent,
  withinPanelRelativeScore,
} from "@/lib/publicationEvidence.js";
import { motion, useInView } from "framer-motion";
import { toast } from "sonner";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip as ReTooltip, ResponsiveContainer,
  CartesianGrid, Legend, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, ErrorBar,
} from "recharts";
import {
  Award, Leaf, Download, Copy, Play, Loader2, FlaskConical,
  Flame, ShieldAlert, DollarSign, Layers,
} from "lucide-react";

const API = getApiBase();

// Color scheme matching paper
const COLORS = {
  static: "#808080",
  agri: "#009688",
  hybrid: "#E67E22",
  noSlca: "#7570B3",
  noContext: "#4CAF50",
  mcpOnly: "#FF9800",
  piragOnly: "#2196F3",
};

const METHOD_COLORS = {
  "Static": COLORS.static,
  "Hybrid RL": COLORS.hybrid,
  "AGRI-BRAIN": COLORS.agri,
  "No social-proxy shaping": COLORS.noSlca,
  "No-external-context": COLORS.noContext,
  "MCP-only": COLORS.mcpOnly,
  "Retrieval-only": COLORS.piragOnly,
};

// Map raw CSV method/variant names to display names
const METHOD_DISPLAY = {
  static: "Static",
  hybrid_rl: "Hybrid RL",
  agribrain: "AGRI-BRAIN",
  no_slca: "No social-proxy shaping",
  no_context: "No-external-context",
  mcp_only: "MCP-only",
  pirag_only: "Retrieval-only",
  // Also handle already-formatted names (no-op)
  "Static": "Static",
  "Hybrid RL": "Hybrid RL",
  "AGRI-BRAIN": "AGRI-BRAIN",
  "No SLCA": "No social-proxy shaping",
  "No-social-performance": "No social-proxy shaping",
  "No social-proxy shaping": "No social-proxy shaping",
  "No Context": "No-external-context",
  "No-external-context": "No-external-context",
  "MCP Only": "MCP-only",
  "MCP-only": "MCP-only",
  "Retrieval-only": "Retrieval-only",
};
const displayMethod = (m) => METHOD_DISPLAY[m] || m;

const METHOD_KEY_MAP = {
  "Static": "static", "Hybrid RL": "hybrid_rl", "No social-proxy shaping": "no_slca",
  "No-external-context": "no_context", "MCP-only": "mcp_only",
  "Retrieval-only": "pirag_only", "AGRI-BRAIN": "agribrain",
};
const VARIANT_KEY_MAP = {
  "Static": "static", "Hybrid RL": "hybrid_rl", "No social-proxy shaping": "no_slca",
  "AGRI-BRAIN": "agribrain", "No-external-context": "no_context", "MCP-only": "mcp_only", "Retrieval-only": "pirag_only",
};

const METRIC_LABELS = {
  ARI: "ARI",
  RLE: "Severity-weighted RLE",
  Waste: "Waste fraction",
  SLCA: "Author-declared social-performance proxy",
  Carbon: "Modeled transport-emissions indicator (kg CO2-eq)",
  Equity: "Temporal social-performance stability proxy",
};
const metricLabel = (key) => METRIC_LABELS[key] || key;

// Animated counter for the headline section
function HeroCounter({ value, suffix = "", prefix = "", label, sublabel, delay = 0 }) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true });
  const nodeRef = useRef(null);

  useEffect(() => {
    if (!isInView || !nodeRef.current) return;
    let frame;
    const start = 0;
    const end = +value;
    const duration = 1500;
    const startTime = Date.now();
    const tick = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = start + (end - start) * eased;
      if (nodeRef.current) nodeRef.current.textContent = prefix + current.toFixed(suffix === "%" ? 1 : 0) + suffix;
      if (progress < 1) frame = requestAnimationFrame(tick);
    };
    const timer = setTimeout(() => { tick(); }, delay);
    return () => { clearTimeout(timer); cancelAnimationFrame(frame); };
  }, [isInView, value, suffix, prefix, delay]);

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 20 }}
      animate={isInView ? { opacity: 1, y: 0 } : {}}
      transition={{ delay: delay / 1000, duration: 0.5 }}
      className="text-center"
    >
      <div className="text-3xl lg:text-4xl font-bold text-primary" ref={nodeRef}>
        {prefix}0{suffix}
      </div>
      <p className="text-sm font-semibold mt-1">{label}</p>
      <p className="text-xs text-muted-foreground">{sublabel}</p>
    </motion.div>
  );
}

// Parse CSV text and normalize method/variant names to display format
function parseCSV(text) {
  const lines = text.trim().split("\n");
  const headers = lines[0].split(",").map((h) => h.trim());
  return lines.slice(1).map((line) => {
    const vals = line.split(",").map((v) => v.trim());
    const obj = {};
    headers.forEach((h, i) => {
      const num = +vals[i];
      obj[h] = Number.isFinite(num) && vals[i] !== "" ? num : vals[i];
    });
    if (obj.Method) obj.Method = displayMethod(obj.Method);
    if (obj.Variant) obj.Variant = displayMethod(obj.Variant);
    return obj;
  });
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border bg-popover p-3 shadow-md text-sm">
      <p className="font-medium mb-1">{label}</p>
      {payload.map((p, i) => (
        <div key={i} className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full" style={{ background: p.color || p.fill }} />
          <span className="text-muted-foreground">{p.name}:</span>
          <span className="font-mono font-medium">{typeof p.value === "number" ? p.value.toFixed(3) : p.value}</span>
        </div>
      ))}
    </div>
  );
}

// Scenario cards
const SCENARIOS = [
  { id: "heatwave", name: "Heatwave", figure: "heatwave.png", icon: Flame, color: "#D55E00",
    findings: [
      "Temperature and humidity excursion over the declared stress window",
      "Mechanistic Arrhenius-lag spoilage-risk trajectory",
      "Policy probabilities and realized routing actions",
      "ARI and emissions-indicator outcomes reported with benchmark uncertainty",
    ],
  },
  { id: "overproduction", name: "Overproduction", figure: "overproduction.png", icon: Layers, color: "#E67E22",
    findings: [
      "Declared inventory surge and recovery-capacity saturation",
      "Routing allocation among cold chain, redistribution, and recovery",
      "Waste fraction per routing opportunity, severity-weighted RLE, and social-performance proxy",
      "Cross-seed uncertainty from the canonical benchmark",
    ],
  },
  // Canonical scenario ids match the backend (agribrain/backend/src/routers/scenarios.py)
  // and every other frontend surface (DemoPage / AdminPage / McpPiragPage / Theater).
  // Pre-2026-05 these used the short "cyber" / "pricing" forms which silently misrouted
  // the lightbox lookup and the deep-link state because table1 / benchmark_significance
  // emit the full "cyber_outage" / "adaptive_pricing" keys.
  { id: "cyber_outage", name: "Cyber Outage", figure: "cyber_outage.png", icon: ShieldAlert, color: "#7570B3",
    findings: [
      "Declared demand and refrigeration disruption after outage onset",
      "Pre/during-outage action-distribution comparison",
      "Routing and outcome changes under the same scenario stream",
      "Local Merkle provenance; on-chain anchoring remains optional",
    ],
  },
  { id: "adaptive_pricing", name: "Adaptive Pricing", figure: "adaptive_pricing.png", icon: DollarSign, color: "#0072B2",
    findings: [
      "Declared oscillatory demand and price-pressure indicator",
      "Policy response across low- and high-demand windows",
      "ARI, waste fraction per routing opportunity, and social-performance proxy",
      "Cross-seed uncertainty from the canonical benchmark",
    ],
  },
];

export default function AnalyticsPage() {
  const [table1, setTable1] = useState([]);
  const [table2, setTable2] = useState([]);
  const [loading, setLoading] = useState(true);
  const [publicationEvidenceReady, setPublicationEvidenceReady] = useState(false);
  const [publicationEvidenceError, setPublicationEvidenceError] = useState("");
  const [selectedScenario, setSelectedScenario] = useState("heatwave");
  const [selectedMetric, setSelectedMetric] = useState("ARI");
  const [ablationMetric, setAblationMetric] = useState("ARI");
  const [radarScenario, setRadarScenario] = useState("Heatwave");
  const [compareA, setCompareA] = useState("AGRI-BRAIN");
  const [compareB, setCompareB] = useState("Static");
  const [simRunning, setSimRunning] = useState(false);
  const [lightboxImg, setLightboxImg] = useState(null);
  const [benchSummary, setBenchSummary] = useState(null);
  const [benchSignificance, setBenchSignificance] = useState(null);

  useEffect(() => {
    const loadData = async () => {
      const fetchText = async (name) => {
        const response = await authFetch(`${API}/results/figures/${name}`);
        if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
        return response.text();
      };
      const fetchJson = async (name) => {
        const response = await authFetch(`${API}/results/figures/${name}`);
        if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
        return response.json();
      };
      try {
        const [t1Text, t2Text, bs, bsig] = await Promise.all([
          fetchText("table1_summary.csv"),
          fetchText("table2_ablation.csv"),
          fetchJson("benchmark_summary.json"),
          fetchJson("benchmark_significance.json"),
        ]);
        const parsedTable1 = parseCSV(t1Text);
        const parsedTable2 = parseCSV(t2Text);
        if (parsedTable1.length !== 35 || parsedTable2.length !== 25) {
          throw new Error(
            `publication panel incomplete (Table 1=${parsedTable1.length}/35, Table 2=${parsedTable2.length}/25)`,
          );
        }
        if (!bs?.summary || !bsig?.significance) {
          throw new Error("canonical benchmark evidence schema is incomplete");
        }
        if (Number(bs?._meta?.n_seeds) !== 20) {
          throw new Error("canonical benchmark evidence must contain exactly 20 seeds");
        }
        // benchmark_summary.json nests scenarios under `summary` (with a top-level
        // `_meta`); flatten so the chart code can read benchSummary[scenario][mode].
        const canonicalSummary = { ...bs.summary, _meta: bs._meta };
        if (!descriptiveCrossScenarioSummary(canonicalSummary)) {
          throw new Error("exact five-scenario AGRI-BRAIN/static summary cells are incomplete");
        }
        setTable1(parsedTable1);
        setTable2(parsedTable2);
        setBenchSummary(canonicalSummary);
        // benchmark_significance.json nests the per-scenario table under `significance`.
        setBenchSignificance(bsig.significance);
        setPublicationEvidenceReady(true);
        setPublicationEvidenceError("");
      } catch (e) {
        console.warn("Validated publication evidence is unavailable:", e);
        setTable1([]);
        setTable2([]);
        setBenchSummary(null);
        setBenchSignificance(null);
        setPublicationEvidenceReady(false);
        setPublicationEvidenceError(e?.message || "validated artifact set not available");
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, []);

  // Grouped bar chart data with CI error bars from benchmark
  const barChartData = useMemo(() => {
    const scenarios = [...new Set(table1.map((r) => r.Scenario))];
    const metricKey = selectedMetric.toLowerCase();
    return scenarios.map((scenario) => {
      const rows = table1.filter((r) => r.Scenario === scenario);
      const obj = { scenario };
      rows.forEach((r) => {
        const val = r[selectedMetric];
        obj[r.Method] = val;
        // Add CI error if benchmark data available
        const rawKey = METHOD_KEY_MAP[r.Method];
        const ci = benchSummary?.[scenario]?.[rawKey]?.[metricKey];
        if (ci && ci.ci_low != null && ci.ci_high != null) {
          obj[r.Method] = ci.mean;
          obj[`${r.Method}_err`] = [ci.mean - ci.ci_low, ci.ci_high - ci.mean];
        }
      });
      return obj;
    });
  }, [table1, selectedMetric, benchSummary]);

  // Ablation bar data with CI error bars
  const ablationData = useMemo(() => {
    const scenarios = [...new Set(table2.map((r) => r.Scenario))];
    const metricKey = ablationMetric.toLowerCase();
    return scenarios.map((scenario) => {
      const rows = table2.filter((r) => r.Scenario === scenario);
      const obj = { scenario };
      rows.forEach((r) => {
        obj[r.Variant] = r[ablationMetric];
        const rawKey = VARIANT_KEY_MAP[r.Variant];
        const ci = benchSummary?.[scenario]?.[rawKey]?.[metricKey];
        if (ci && ci.ci_low != null && ci.ci_high != null) {
          obj[r.Variant] = ci.mean;
          obj[`${r.Variant}_err`] = [ci.mean - ci.ci_low, ci.ci_high - ci.mean];
        }
      });
      return obj;
    });
  }, [table2, ablationMetric, benchSummary]);

  // Radar chart data
  const radarData = useMemo(() => {
    const rows = table1.filter((r) => r.Scenario === radarScenario);
    if (!rows.length) return [];
    const axes = ["ARI", "RLE", "Waste", "SLCA", "Carbon", "Equity"];
    return axes.map((axis) => {
      const panelValues = rows
        .map((r) => Number(r[axis] ?? r[`${axis} (kg)`]))
        .filter(Number.isFinite);
      const obj = { axis: `${metricLabel(axis)} (relative)` };
      rows.forEach((r) => {
        const val = r[axis] ?? r[`${axis} (kg)`];
        obj[r.Method] = withinPanelRelativeScore(val, panelValues, axis);
      });
      return obj;
    });
  }, [table1, radarScenario]);

  // Method comparison
  const comparison = useMemo(() => {
    if (!table1.length) return [];
    const metrics = ["ARI", "RLE", "Waste", "SLCA", "Carbon", "Equity"];
    return metrics.map((metric) => {
      const valuesFor = (method) => table1
        .filter((r) => r.Method === method)
        .map((r) => Number(r[metric] ?? r[`${metric} (kg)`]))
        .filter(Number.isFinite);
      const aVals = valuesFor(compareA);
      const bVals = valuesFor(compareB);
      const aAvg = aVals.length ? aVals.reduce((a, b) => a + b, 0) / aVals.length : null;
      const bAvg = bVals.length ? bVals.reduce((a, b) => a + b, 0) / bVals.length : null;
      const advantagePct = directionalAdvantagePercent(aAvg, bAvg, metric);
      return { metric, label: metricLabel(metric), a: aAvg, b: bAvg, advantagePct };
    });
  }, [table1, compareA, compareB]);

  // Carbon chart data with CI error bars
  const carbonData = useMemo(() => {
    const scenarios = [...new Set(table1.map((r) => r.Scenario))];
    return scenarios.map((scenario) => {
      const rows = table1.filter((r) => r.Scenario === scenario);
      const obj = { scenario };
      rows.forEach((r) => {
        const val = r.Carbon ?? r["Carbon (kg)"] ?? 0;
        obj[r.Method] = val;
        const rawKey = METHOD_KEY_MAP[r.Method];
        const ci = benchSummary?.[scenario]?.[rawKey]?.carbon;
        if (ci && ci.ci_low != null && ci.ci_high != null) {
          obj[r.Method] = ci.mean;
          obj[`${r.Method}_err`] = [ci.mean - ci.ci_low, ci.ci_high - ci.mean];
        }
      });
      return obj;
    });
  }, [table1, benchSummary]);

  const runSimulation = async () => {
    setSimRunning(true);
    try {
      // Kick off background job
      await jpost(API, "/results/generate");
      toast.info("Development run started: 55 retained endpoints, 205 executed episodes, and 59,040 simulated steps. Polling for completion...");

      // Poll /results/status until done
      let status = "running";
      while (status === "running" || status === "started") {
        await new Promise((r) => setTimeout(r, 5000));
        const st = await jget(API, "/results/status");
        status = st.status;
        if (status === "running") {
          toast.info(`Simulation running... (${st.elapsed_s || 0}s elapsed)`, { id: "sim-poll" });
        }
      }

      const st = await jget(API, "/results/status");
      if (st.status === "error") throw new Error(st.error);

      toast.success(`Development-only run complete in ${st.duration_s || "?"}s. Publication evidence was not changed.`);
    } catch (e) {
      toast.error(`Simulation failed: ${e.message || "Check backend logs."}`);
    }
    setSimRunning(false);
  };

  const exportTableCSV = (data, filename) => {
    if (!data.length) return;
    const esc = (value) => {
      const text = String(value ?? "");
      return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
    };
    const columns = Object.keys(data[0]);
    const headers = columns.map((key) => esc(metricLabel(key))).join(",") + "\n";
    const rows = data
      .map((row) => columns.map((key) => esc(row[key])).join(","))
      .join("\n");
    const blob = new Blob([headers + rows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = filename; a.click();
    URL.revokeObjectURL(url);
    toast.success(`${filename} exported`);
  };

  // Helper to format a value with CI range from benchmark data
  const fmtCI = (scenario, method, metric) => {
    const rawMethod = METHOD_KEY_MAP[method] || VARIANT_KEY_MAP[method];
    const metricKey = metric.toLowerCase();
    const ci = benchSummary?.[scenario]?.[rawMethod]?.[metricKey];
    if (!ci || ci.ci_low == null) return null;
    return `[${ci.ci_low.toFixed(3)}, ${ci.ci_high.toFixed(3)}]`;
  };

  const scenarioObj = SCENARIOS.find((s) => s.id === selectedScenario) || SCENARIOS[0];

  // Descriptive UI-only transform of the exact, unrounded JSON scenario means.
  const summaryKPIs = useMemo(() => {
    return descriptiveCrossScenarioSummary(benchSummary);
  }, [benchSummary]);

  if (loading) {
    return (
      <Card className="border-primary/20">
        <CardContent className="p-8 text-center">
          <Loader2 className="w-8 h-8 mx-auto mb-3 animate-spin text-primary" />
          <h2 className="text-xl font-semibold">Verifying publication evidence</h2>
          <p className="text-sm text-muted-foreground mt-2">Checking the validated manifest and exact artifact bytes.</p>
        </CardContent>
      </Card>
    );
  }

  if (!publicationEvidenceReady) {
    return (
      <div className="space-y-6 pb-12">
        <Card className="border-amber-500/40 bg-amber-500/5">
          <CardContent className="p-8 text-center">
            <ShieldAlert className="w-10 h-10 mx-auto mb-3 text-amber-600" />
            <Badge variant="outline" className="mb-3">Publication evidence unavailable</Badge>
            <h2 className="text-xl font-semibold">No canonical benchmark values are displayed</h2>
            <p className="text-sm text-muted-foreground mt-2 max-w-2xl mx-auto">
              A complete schema-v2 manifest and its exact validated artifacts have not been loaded. Historical or development outputs are intentionally not shown as publication results.
            </p>
            <p className="text-xs font-mono text-muted-foreground mt-3">{publicationEvidenceError}</p>
          </CardContent>
        </Card>
        <Card className="border-primary/20">
          <CardContent className="p-6 text-center">
            <FlaskConical className="w-10 h-10 mx-auto mb-3 text-primary" />
            <h3 className="text-lg font-semibold mb-2">Run a development-only simulation</h3>
            <p className="text-sm text-muted-foreground mb-4 max-w-lg mx-auto">
              This one-seed local run exercises all {PRIMARY_PUBLICATION_MODES.length + SECONDARY_PUBLICATION_MODES.length} modes: 55 retained endpoints from 205 executed episodes (59,040 simulated steps). It cannot create or replace publication evidence.
            </p>
            <Button size="lg" onClick={runSimulation} disabled={simRunning}>
              {simRunning ? (
                <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Running simulation...</>
              ) : (
                <><Play className="w-4 h-4 mr-2" /> Generate development results</>
              )}
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-8 pb-12">
      {/* 8.1 Executive Summary Banner */}
      <section>
        <Card className="bg-gradient-to-br from-primary/5 via-background to-primary/5 border-primary/20">
          <CardContent className="py-8 px-6">
            <div className="text-center mb-8">
              <Badge variant="teal" className="mb-2">Backend-accepted artifact set · descriptive UI summary</Badge>
              <h2 className="text-2xl font-bold">AGRI-BRAIN Five-Scenario Summary</h2>
              <p className="text-sm text-muted-foreground mt-1">
                Unweighted arithmetic means of the five exact scenario-level means. Relative differences are descriptive transforms, not pooled confidence intervals or confirmatory effects.
              </p>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6 max-w-4xl mx-auto">
              <HeroCounter value={+summaryKPIs.ariRelativeDifferencePct.toFixed(1)} suffix="%" label="Descriptive ARI difference" sublabel="Ratio of unweighted means vs static" delay={0} />
              <HeroCounter value={+summaryKPIs.wasteRelativeDifferencePct.toFixed(1)} suffix="%" label="Descriptive waste difference" sublabel={`${summaryKPIs.exactUnweightedMeans.waste.agribrain.toFixed(3)} vs ${summaryKPIs.exactUnweightedMeans.waste.static.toFixed(3)} mean fraction`} delay={200} />
              <HeroCounter value={+summaryKPIs.carbonRelativeDifferencePct.toFixed(1)} suffix="%" label="Descriptive emissions difference" sublabel={`${summaryKPIs.exactUnweightedMeans.carbon.agribrain.toFixed(1)} vs ${summaryKPIs.exactUnweightedMeans.carbon.static.toFixed(1)} modeled kg CO2-eq`} delay={400} />
              <HeroCounter value={+summaryKPIs.rleDisplayScore.toFixed(1)} label="RLE display score" sublabel="AGRI-BRAIN unweighted mean ×100; not a percentage" delay={600} />
            </div>
          </CardContent>
        </Card>
      </section>

      {/* 8.2 Interactive Performance Tables */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold">
              <span className="text-xs font-medium tracking-wider uppercase text-muted-foreground mr-2">Table&nbsp;1</span>
              Cross-Scenario Performance
            </h3>
            <p className="text-sm text-muted-foreground italic">Comparison of eight primary modes across five simulated scenarios (288 timesteps each). Paired modes share the declared environmental stream. Source artifact: <code>table1_summary.csv</code>.</p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={() => exportTableCSV(table1, "table1_summary.csv")}>
              <Download className="w-4 h-4 mr-1" /> CSV
            </Button>
            <Button variant="outline" size="sm" onClick={() => { navigator.clipboard.writeText(JSON.stringify(table1, null, 2)); toast.success("Copied"); }}>
              <Copy className="w-4 h-4 mr-1" /> Copy
            </Button>
          </div>
        </div>
        <Card>
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  {["Scenario", "Method", "ARI", "RLE", "Waste", "SLCA", "Carbon", "Equity"].map((h) => (
                    <TableHead key={h} className="font-semibold whitespace-nowrap">{metricLabel(h)}</TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {table1.map((row, i) => {
                  const isAgri = row.Method === "AGRI-BRAIN";
                  return (
                    <TableRow key={i} className={cn(isAgri && "bg-primary/5 border-l-2 border-l-primary")}>
                      <TableCell className="font-medium">{row.Scenario}</TableCell>
                      <TableCell>
                        <Badge variant={isAgri ? "teal" : "secondary"} className="text-xs">{row.Method}</Badge>
                      </TableCell>
                      {["ARI", "RLE", "Waste", "SLCA", "Carbon", "Equity"].map((col) => {
                        const val = row[col] ?? row[`${col} (kg)`];
                        const ci = fmtCI(row.Scenario, row.Method, col);
                        return (
                          <TableCell key={col} className="font-mono text-sm">
                            <span>{typeof val === "number" ? val.toFixed(3) : val ?? "\u2014"}</span>
                            {ci && <span className="block text-[10px] text-muted-foreground">{ci}</span>}
                          </TableCell>
                        );
                      })}
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        </Card>

        <div className="flex items-center justify-between mt-8 mb-4">
          <div>
            <h3 className="text-lg font-semibold">
              <span className="text-xs font-medium tracking-wider uppercase text-muted-foreground mr-2">Table&nbsp;2</span>
              Ablation Study
            </h3>
            <p className="text-sm text-muted-foreground italic">Compact six-mode architectural ablation. Prior and weight sensitivities are separate diagnostics. Source artifact: <code>table2_ablation.csv</code>.</p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={() => exportTableCSV(table2, "table2_ablation.csv")}>
              <Download className="w-4 h-4 mr-1" /> CSV
            </Button>
          </div>
        </div>
        <Card>
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  {["Scenario", "Variant", "ARI", "RLE", "Waste", "SLCA"].map((h) => (
                    <TableHead key={h} className="font-semibold">{metricLabel(h)}</TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {table2.map((row, i) => {
                  const isAgri = row.Variant === "AGRI-BRAIN";
                  return (
                    <TableRow key={i} className={cn(isAgri && "bg-primary/5 border-l-2 border-l-primary")}>
                      <TableCell className="font-medium">{row.Scenario}</TableCell>
                      <TableCell>
                        <Badge
                          className="text-xs border-0"
                          style={{ backgroundColor: (METHOD_COLORS[row.Variant] || "#808080") + "15", color: METHOD_COLORS[row.Variant] || "#808080" }}
                        >
                          {row.Variant}
                        </Badge>
                      </TableCell>
                      {["ARI", "RLE", "Waste", "SLCA"].map((col) => {
                        const ci = fmtCI(row.Scenario, row.Variant, col);
                        return (
                          <TableCell key={col} className="font-mono text-sm">
                            <span>{typeof row[col] === "number" ? row[col].toFixed(3) : row[col] ?? "\u2014"}</span>
                            {ci && <span className="block text-[10px] text-muted-foreground">{ci}</span>}
                          </TableCell>
                        );
                      })}
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        </Card>
      </section>

      {/* 8.3 Interactive Cross-Scenario Charts */}
      <section>
        <h3 className="text-lg font-semibold mb-4">Cross-Scenario Analysis</h3>

        {/* Grouped bar chart */}
        <Card className="mb-6">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">Method Comparison</CardTitle>
              <Select value={selectedMetric} onValueChange={setSelectedMetric}>
                <SelectTrigger className="w-28 h-8"><SelectValue /></SelectTrigger>
                <SelectContent>
                    {["ARI", "RLE", "Waste", "SLCA"].map((m) => (
                      <SelectItem key={m} value={m}>{metricLabel(m)}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <CardDescription>Performance by scenario and method ({metricLabel(selectedMetric)})</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={barChartData} barGap={2} barCategoryGap="20%">
                  <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                  <XAxis dataKey="scenario" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <ReTooltip content={<ChartTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="Static" fill={COLORS.static} radius={[2, 2, 0, 0]} isAnimationActive={false}>
                    <ErrorBar dataKey="Static_err" width={7} strokeWidth={2.5} stroke="#1f2937" />
                  </Bar>
                  <Bar dataKey="Hybrid RL" fill={COLORS.hybrid} radius={[2, 2, 0, 0]} isAnimationActive={false}>
                    <ErrorBar dataKey="Hybrid RL_err" width={7} strokeWidth={2.5} stroke="#1f2937" />
                  </Bar>
                  <Bar dataKey="AGRI-BRAIN" fill={COLORS.agri} radius={[2, 2, 0, 0]} isAnimationActive={false}>
                    <ErrorBar dataKey="AGRI-BRAIN_err" width={7} strokeWidth={2.5} stroke="#1f2937" />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Ablation bar chart */}
        <Card className="mb-6">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">Ablation Study</CardTitle>
              <Select value={ablationMetric} onValueChange={setAblationMetric}>
                <SelectTrigger className="w-28 h-8"><SelectValue /></SelectTrigger>
                <SelectContent>
                    {["ARI", "RLE", "Waste", "SLCA"].map((m) => (
                      <SelectItem key={m} value={m}>{metricLabel(m)}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <CardDescription>Component contribution analysis ({metricLabel(ablationMetric)})</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={ablationData} barGap={1} barCategoryGap="15%">
                  <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                  <XAxis dataKey="scenario" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <ReTooltip content={<ChartTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="Static" fill={COLORS.static} radius={[2, 2, 0, 0]} isAnimationActive={false}>
                    <ErrorBar dataKey="Static_err" width={5} strokeWidth={2.5} stroke="#1f2937" />
                  </Bar>
                  <Bar dataKey="Hybrid RL" fill={COLORS.hybrid} radius={[2, 2, 0, 0]} isAnimationActive={false}>
                    <ErrorBar dataKey="Hybrid RL_err" width={5} strokeWidth={2.5} stroke="#1f2937" />
                  </Bar>
                  <Bar dataKey="No-external-context" fill={COLORS.noContext} radius={[2, 2, 0, 0]} isAnimationActive={false}>
                    <ErrorBar dataKey="No-external-context_err" width={5} strokeWidth={2.5} stroke="#1f2937" />
                  </Bar>
                  <Bar dataKey="No social-proxy shaping" fill={COLORS.noSlca} radius={[2, 2, 0, 0]} isAnimationActive={false}>
                    <ErrorBar dataKey="No social-proxy shaping_err" width={5} strokeWidth={2.5} stroke="#1f2937" />
                  </Bar>
                  <Bar dataKey="AGRI-BRAIN" fill={COLORS.agri} radius={[2, 2, 0, 0]} isAnimationActive={false}>
                    <ErrorBar dataKey="AGRI-BRAIN_err" width={5} strokeWidth={2.5} stroke="#1f2937" />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Statistical Significance (from benchmark) */}
        {benchSignificance && (
          <Card className="mb-6 border-primary/20">
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Confirmatory Directional Evidence ({benchSummary._meta.n_seeds}-Seed Stochastic Benchmark)</CardTitle>
              <CardDescription>Canonical Holm-adjusted directional H1 inference; missing canonical fields fail closed</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2 px-3 font-semibold">Scenario</th>
                      <th className="text-left py-2 px-3 font-semibold">Comparison</th>
                      <th className="text-right py-2 px-3 font-semibold">Holm-adjusted p</th>
                      <th className="text-right py-2 px-3 font-semibold">Paired d_z</th>
                      <th className="text-right py-2 px-3 font-semibold">ARI diff</th>
                      <th className="text-center py-2 px-3 font-semibold">H1 support</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(benchSignificance).map(([scenario, comps]) =>
                      Object.entries(comps).filter(([comp]) => comp === "agribrain_vs_no_context").map(([comp, metrics]) => {
                        const ari = metrics.ari || {};
                        const evidence = canonicalH1Evidence(ari);
                        const pairedDz = Number(ari.cohens_dz);
                        const meanDiff = Number(ari.mean_diff);
                        return (
                          <tr key={`${scenario}-${comp}`} className="border-b border-muted/50 hover:bg-muted/30">
                            <td className="py-1.5 px-3 font-mono text-xs">{scenario}</td>
                            <td className="py-1.5 px-3 text-xs">{comp.replace("agribrain_vs_", "vs ")}</td>
                            <td className="py-1.5 px-3 text-right font-mono text-xs">{evidence.available ? evidence.adjustedP.toFixed(4) : "Unavailable"}</td>
                            <td className="py-1.5 px-3 text-right font-mono text-xs">{Number.isFinite(pairedDz) ? pairedDz.toFixed(2) : "Unavailable"}</td>
                            <td className="py-1.5 px-3 text-right font-mono text-xs">{Number.isFinite(meanDiff) ? `${meanDiff > 0 ? "+" : ""}${meanDiff.toFixed(4)}` : "Unavailable"}</td>
                            <td className="py-1.5 px-3 text-center">{evidence.supported ? <Badge variant="default" className="text-[10px] px-1.5 py-0 bg-emerald-600">Supported</Badge> : <Badge variant="outline" className="text-[10px] px-1.5 py-0">{evidence.available ? "Not supported" : "Unavailable"}</Badge>}</td>
                          </tr>
                        );
                      })
                    )}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Radar chart + Method comparison */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-base">Radar: Multi-Metric Profile</CardTitle>
                <Select value={radarScenario} onValueChange={setRadarScenario}>
                  <SelectTrigger className="w-40 h-8"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {[...new Set(table1.map((r) => r.Scenario))].map((s) => (
                      <SelectItem key={s} value={s}>{s}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <CardDescription>
                Within-scenario display scores only: best observed method = 1 and worst = 0;
                waste and emissions are reversed so higher is favorable. Distances are not effect sizes.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={radarData}>
                    <PolarGrid className="opacity-30" />
                    <PolarAngleAxis dataKey="axis" tick={{ fontSize: 11 }} />
                    <PolarRadiusAxis tick={{ fontSize: 9 }} domain={[0, 1]} />
                    <Radar name="Static" dataKey="Static" stroke={COLORS.static} fill={COLORS.static} fillOpacity={0.1} />
                    <Radar name="Hybrid RL" dataKey="Hybrid RL" stroke={COLORS.hybrid} fill={COLORS.hybrid} fillOpacity={0.1} />
                    <Radar name="AGRI-BRAIN" dataKey="AGRI-BRAIN" stroke={COLORS.agri} fill={COLORS.agri} fillOpacity={0.2} />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <ReTooltip content={<ChartTooltip />} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Method Comparison</CardTitle>
              <CardDescription>
                Direction-adjusted relative advantage of the first method; positive is favorable,
                with lower waste and emissions treated as better.
              </CardDescription>
              <div className="flex items-center gap-2 mt-2">
                <Select value={compareA} onValueChange={setCompareA}>
                  <SelectTrigger className="w-36 h-8"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {["AGRI-BRAIN", "Static", "Hybrid RL"].map((m) => (
                      <SelectItem key={m} value={m}>{m}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <span className="text-xs text-muted-foreground">vs</span>
                <Select value={compareB} onValueChange={setCompareB}>
                  <SelectTrigger className="w-36 h-8"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {["Static", "Hybrid RL", "AGRI-BRAIN"].map((m) => (
                      <SelectItem key={m} value={m}>{m}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-3 mt-2">
                {comparison.map((c) => (
                  <div key={c.metric} className="flex items-center justify-between">
                    <span className="text-sm font-medium w-36">{c.label}</span>
                    <div className="flex-1 mx-4">
                      <div className="flex items-center justify-between text-xs text-muted-foreground mb-1">
                        <span>{Number.isFinite(c.a) ? fmt(c.a, 3) : "Unavailable"}</span>
                        <span>{Number.isFinite(c.b) ? fmt(c.b, 3) : "Unavailable"}</span>
                      </div>
                      <div className="h-2 bg-muted rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all"
                          style={{
                            width: `${Math.min(100, Math.abs(c.advantagePct ?? 0))}%`,
                            backgroundColor: c.advantagePct > 0 ? "#10B981" : c.advantagePct < 0 ? "#D55E00" : "#808080",
                          }}
                        />
                      </div>
                    </div>
                    <span className={cn("text-sm font-mono font-semibold w-24 text-right", c.advantagePct > 0 ? "text-emerald-600" : c.advantagePct < 0 ? "text-[#D55E00]" : "text-muted-foreground")}>
                      {Number.isFinite(c.advantagePct) ? `${c.advantagePct > 0 ? "+" : ""}${c.advantagePct.toFixed(1)}%` : "Unavailable"}
                    </span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* 8.4 Scenario Deep-Dive Gallery */}
      <section>
        <h3 className="text-lg font-semibold mb-4">Scenario Deep-Dive Gallery</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
          {SCENARIOS.map((s) => (
            <button
              key={s.id}
              onClick={() => setSelectedScenario(s.id)}
              className={cn(
                "p-4 rounded-xl border text-left transition-all",
                selectedScenario === s.id
                  ? "border-primary bg-primary/5 shadow-sm"
                  : "hover:border-primary/50 hover:bg-muted/50"
              )}
            >
              <s.icon className="w-6 h-6 mb-2" style={{ color: s.color }} />
              <p className="font-medium text-sm">{s.name}</p>
            </button>
          ))}
        </div>

        <Card>
          <CardContent className="p-6">
            <div className="grid lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <img
                  src={`${API}/results/figures/${scenarioObj.figure}`}
                  alt={`Figure: ${scenarioObj.name}`}
                  className="w-full rounded-lg border cursor-pointer hover:opacity-90 transition-opacity"
                  style={{ imageRendering: "auto" }}
                  onClick={() => setLightboxImg(`${API}/results/figures/${scenarioObj.figure}`)}
                  onError={(e) => { e.target.style.display = "none"; }}
                />
                <p className="text-xs text-muted-foreground italic mt-2">
                  {scenarioObj.name} scenario multi-panel analysis. Click to enlarge.
                </p>
              </div>
              <div>
                <Card className="bg-primary/5 border-primary/20">
                  <CardContent className="p-4">
                    <h4 className="font-semibold text-sm mb-3 flex items-center gap-2">
                      <Award className="w-4 h-4 text-primary" /> Displayed Evidence
                    </h4>
                    <ul className="space-y-2">
                      {scenarioObj.findings.map((f, i) => (
                        <li key={i} className="text-sm flex items-start gap-2">
                          <span className="h-1.5 w-1.5 rounded-full bg-primary mt-1.5 shrink-0" />
                          {f}
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* 8.5 Modeled transport emissions */}
      <section>
        <h3 className="text-lg font-semibold mb-4">Modeled Transport-Emissions Indicator</h3>
        <div className="grid lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Modeled Transport-Emissions Indicator by Scenario</CardTitle>
            </CardHeader>
            <CardContent>
              <img
                src={`${API}/results/figures/transport_emissions.png`}
                alt="Modeled transport-emissions analysis"
                className="w-full rounded-lg border mb-4 cursor-pointer hover:opacity-90"
                style={{ imageRendering: "auto" }}
                onClick={() => setLightboxImg(`${API}/results/figures/transport_emissions.png`)}
                onError={(e) => { e.target.style.display = "none"; }}
              />
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={carbonData} barGap={2}>
                    <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                    <XAxis dataKey="scenario" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} label={{ value: "Modeled kg CO2-eq proxy", angle: -90, position: "insideLeft", fontSize: 11 }} />
                    <ReTooltip content={<ChartTooltip />} />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Bar dataKey="Static" fill={COLORS.static} radius={[2, 2, 0, 0]} isAnimationActive={false}>
                      <ErrorBar dataKey="Static_err" width={7} strokeWidth={2.5} stroke="#1f2937" />
                    </Bar>
                    <Bar dataKey="Hybrid RL" fill={COLORS.hybrid} radius={[2, 2, 0, 0]} isAnimationActive={false}>
                      <ErrorBar dataKey="Hybrid RL_err" width={7} strokeWidth={2.5} stroke="#1f2937" />
                    </Bar>
                    <Bar dataKey="AGRI-BRAIN" fill={COLORS.agri} radius={[2, 2, 0, 0]} isAnimationActive={false}>
                      <ErrorBar dataKey="AGRI-BRAIN_err" width={7} strokeWidth={2.5} stroke="#1f2937" />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-emerald-500/5 border-emerald-500/20">
            <CardContent className="p-6 flex flex-col justify-center h-full">
              <div className="text-center">
                <Leaf className="w-12 h-12 mx-auto mb-4 text-emerald-600" />
                <h4 className="text-xl font-bold mb-2">Interpretation Scope</h4>
                <p className="text-sm text-muted-foreground mb-4">Synthetic modeled kg CO2-eq proxy</p>
              </div>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-3 rounded-lg bg-background border">
                  <span className="text-sm">Definition</span>
                  <span className="font-mono font-bold text-emerald-600">distance × factor × thermal term</span>
                </div>
                <div className="flex items-center justify-between p-3 rounded-lg bg-background border">
                  <span className="text-sm">Payload model</span>
                  <span className="font-mono font-bold">not included</span>
                </div>
                <div className="flex items-center justify-between p-3 rounded-lg bg-background border">
                  <span className="text-sm">Use</span>
                  <span className="font-mono font-bold text-emerald-600">relative scenario comparison</span>
                </div>
              </div>
              <p className="text-xs text-muted-foreground mt-4 text-center italic">
                This modeled kg CO2-eq quantity is not a measured lifecycle-emissions footprint.
              </p>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* 8.6 Run Simulation */}
      <section>
        <Card className="border-primary/20">
          <CardContent className="p-6 text-center">
            <FlaskConical className="w-10 h-10 mx-auto mb-3 text-primary" />
            <h3 className="text-lg font-semibold mb-2">Run One Development Seed</h3>
            <p className="text-sm text-muted-foreground mb-4 max-w-lg mx-auto">
              Runs all 5 scenarios × {PRIMARY_PUBLICATION_MODES.length + SECONDARY_PUBLICATION_MODES.length} modes ({PRIMARY_PUBLICATION_MODES.length} primary + {SECONDARY_PUBLICATION_MODES.length} secondary): 55 retained endpoints, 205 executed episodes, and 59,040 simulated steps. It never replaces validated publication evidence.
            </p>
            <Button size="lg" onClick={runSimulation} disabled={simRunning}>
              {simRunning ? (
                <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Running simulation...</>
              ) : (
                <><Play className="w-4 h-4 mr-2" /> Generate Results</>
              )}
            </Button>
            {simRunning && (
              <p className="text-xs text-muted-foreground mt-2 animate-pulse">This may take 1-2 minutes...</p>
            )}
          </CardContent>
        </Card>
      </section>

      {/* Lightbox */}
      {lightboxImg && (
        <div className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4" onClick={() => setLightboxImg(null)}>
          <img src={lightboxImg} alt="Enlarged figure" className="max-w-full max-h-full object-contain rounded-lg" />
        </div>
      )}
    </div>
  );
}
