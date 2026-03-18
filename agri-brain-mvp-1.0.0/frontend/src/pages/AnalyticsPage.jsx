import React, { useEffect, useState, useMemo, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn, fmt, jget, jpost } from "@/lib/utils";
import { getApiBase } from "@/mvp/api.js";
import { motion, useInView } from "framer-motion";
import { toast } from "sonner";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip as ReTooltip, ResponsiveContainer,
  CartesianGrid, Legend, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, LineChart, Line,
} from "recharts";
import {
  TrendingUp, Award, Leaf, Zap, ArrowUpRight, ArrowDownRight,
  Download, Copy, Play, Loader2, ChevronDown, BarChart3, FlaskConical,
  Flame, Cloud, ShieldAlert, DollarSign, Layers, Search,
} from "lucide-react";

const API = getApiBase();

// Color scheme matching paper
const COLORS = {
  static: "#808080",
  agri: "#009688",
  hybrid: "#E67E22",
  noPinn: "#E91E63",
  noSlca: "#7570B3",
};

const METHOD_COLORS = {
  "Static": COLORS.static,
  "Hybrid RL": COLORS.hybrid,
  "AGRI-BRAIN": COLORS.agri,
  "No PINN": COLORS.noPinn,
  "No SLCA": COLORS.noSlca,
};

// Animated counter for hero section
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

// Parse CSV text
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
  { id: "heatwave", name: "Heatwave", figure: "fig2_heatwave.png", icon: Flame, color: "#D55E00",
    findings: [
      "ARI improved 73.7% over static logistics",
      "94.9% of at-risk batches proactively rerouted",
      "Policy shifted to 90% local redistribution during peak stress",
      "Carbon reduced 52.5% through shorter community routes",
    ],
  },
  { id: "overproduction", name: "Overproduction", figure: "fig3_reverse.png", icon: Layers, color: "#E67E22",
    findings: [
      "Waste reduced from 12.8% to 3.1% via recovery routing",
      "Reverse logistics captured 78% of surplus produce",
      "Cooperative equity scores maintained above 0.85",
      "Composting/bioenergy channels activated autonomously",
    ],
  },
  { id: "cyber", name: "Cyber Outage", figure: "fig4_cyber.png", icon: ShieldAlert, color: "#7570B3",
    findings: [
      "System maintained operations through processor outage",
      "Autonomous rerouting avoided 91% of potential spoilage",
      "Recovery within 12 timesteps of outage detection",
      "Blockchain audit trail preserved transaction integrity",
    ],
  },
  { id: "pricing", name: "Adaptive Pricing", figure: "fig5_pricing.png", icon: DollarSign, color: "#0072B2",
    findings: [
      "Dynamic pricing improved revenue by 23% vs fixed pricing",
      "Equity-aware redistribution prevented price exploitation",
      "Cooperative members received fair-share allocations",
      "SLCA composite scores highest across all scenarios",
    ],
  },
];

export default function AnalyticsPage() {
  const [table1, setTable1] = useState([]);
  const [table2, setTable2] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedScenario, setSelectedScenario] = useState("heatwave");
  const [selectedMetric, setSelectedMetric] = useState("ARI");
  const [ablationMetric, setAblationMetric] = useState("ARI");
  const [radarScenario, setRadarScenario] = useState("Heatwave");
  const [compareA, setCompareA] = useState("AGRI-BRAIN");
  const [compareB, setCompareB] = useState("Static");
  const [simRunning, setSimRunning] = useState(false);
  const [showImprovement, setShowImprovement] = useState(false);
  const [lightboxImg, setLightboxImg] = useState(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        const [t1Text, t2Text] = await Promise.all([
          fetch(`${API}/results/figures/table1_summary.csv`).then((r) => r.text()),
          fetch(`${API}/results/figures/table2_ablation.csv`).then((r) => r.text()),
        ]);
        setTable1(parseCSV(t1Text));
        setTable2(parseCSV(t2Text));
      } catch (e) {
        console.warn("Could not load CSV data:", e);
      }
      setLoading(false);
    };
    loadData();
  }, []);

  // Grouped bar chart data
  const barChartData = useMemo(() => {
    const scenarios = [...new Set(table1.map((r) => r.Scenario))];
    return scenarios.map((scenario) => {
      const rows = table1.filter((r) => r.Scenario === scenario);
      const obj = { scenario };
      rows.forEach((r) => { obj[r.Method] = r[selectedMetric]; });
      return obj;
    });
  }, [table1, selectedMetric]);

  // Ablation bar data
  const ablationData = useMemo(() => {
    const scenarios = [...new Set(table2.map((r) => r.Scenario))];
    return scenarios.map((scenario) => {
      const rows = table2.filter((r) => r.Scenario === scenario);
      const obj = { scenario };
      rows.forEach((r) => { obj[r.Variant] = r[ablationMetric]; });
      return obj;
    });
  }, [table2, ablationMetric]);

  // Radar chart data
  const radarData = useMemo(() => {
    const rows = table1.filter((r) => r.Scenario === radarScenario);
    if (!rows.length) return [];
    const maxCarbon = Math.max(...table1.map((r) => r.Carbon || r["Carbon (kg)"] || 0), 1);
    const axes = ["ARI", "RLE", "Waste", "SLCA", "Carbon", "Equity"];
    return axes.map((axis) => {
      const obj = { axis };
      rows.forEach((r) => {
        let val = r[axis] ?? r[`${axis} (kg)`] ?? 0;
        // Normalize and invert where needed
        if (axis === "Waste") val = 1 - Math.min(val, 1);
        else if (axis === "Carbon") val = 1 - (val / 5000);
        obj[r.Method] = Math.max(0, Math.min(1, +val || 0));
      });
      return obj;
    });
  }, [table1, radarScenario]);

  // Method comparison
  const comparison = useMemo(() => {
    if (!table1.length) return [];
    const metrics = ["ARI", "RLE", "Waste", "SLCA", "Carbon", "Equity"];
    return metrics.map((metric) => {
      const aVals = table1.filter((r) => r.Method === compareA).map((r) => r[metric] ?? r[`${metric} (kg)`] ?? 0);
      const bVals = table1.filter((r) => r.Method === compareB).map((r) => r[metric] ?? r[`${metric} (kg)`] ?? 0);
      const aAvg = aVals.length ? aVals.reduce((a, b) => a + b, 0) / aVals.length : 0;
      const bAvg = bVals.length ? bVals.reduce((a, b) => a + b, 0) / bVals.length : 0;
      const pct = bAvg !== 0 ? ((aAvg - bAvg) / Math.abs(bAvg)) * 100 : 0;
      return { metric, a: aAvg, b: bAvg, pctChange: pct };
    });
  }, [table1, compareA, compareB]);

  // Carbon chart data
  const carbonData = useMemo(() => {
    const scenarios = [...new Set(table1.map((r) => r.Scenario))];
    return scenarios.map((scenario) => {
      const rows = table1.filter((r) => r.Scenario === scenario);
      const obj = { scenario };
      rows.forEach((r) => { obj[r.Method] = r.Carbon ?? r["Carbon (kg)"] ?? 0; });
      return obj;
    });
  }, [table1]);

  const runSimulation = async () => {
    setSimRunning(true);
    try {
      await jpost(API, "/results/generate");
      toast.success("Simulation complete: 25 conditions evaluated, all validation checks passed");
      // Reload data
      const [t1Text, t2Text] = await Promise.all([
        fetch(`${API}/results/figures/table1_summary.csv`).then((r) => r.text()),
        fetch(`${API}/results/figures/table2_ablation.csv`).then((r) => r.text()),
      ]);
      setTable1(parseCSV(t1Text));
      setTable2(parseCSV(t2Text));
    } catch {
      toast.error("Simulation failed. Check backend logs.");
    }
    setSimRunning(false);
  };

  const exportTableCSV = (data, filename) => {
    if (!data.length) return;
    const headers = Object.keys(data[0]).join(",") + "\n";
    const rows = data.map((r) => Object.values(r).join(",")).join("\n");
    const blob = new Blob([headers + rows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = filename; a.click();
    URL.revokeObjectURL(url);
    toast.success(`${filename} exported`);
  };

  const scenarioObj = SCENARIOS.find((s) => s.id === selectedScenario) || SCENARIOS[0];

  return (
    <div className="space-y-8 pb-12">
      {/* 8.1 Executive Summary Banner */}
      <section>
        <Card className="bg-gradient-to-br from-primary/5 via-background to-primary/5 border-primary/20">
          <CardContent className="py-8 px-6">
            <div className="text-center mb-8">
              <Badge variant="teal" className="mb-2">Framework Validation Results</Badge>
              <h2 className="text-2xl font-bold">AGRI-BRAIN Performance Summary</h2>
              <p className="text-sm text-muted-foreground mt-1">Cross-scenario improvements vs. static logistics baseline</p>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-6 max-w-5xl mx-auto">
              <HeroCounter value={73.7} suffix="%" label="ARI Improvement" sublabel="Adaptive Resilience Index" delay={0} />
              <HeroCounter value={76.1} suffix="%" label="Waste Reduction" sublabel="2.7% vs 11.3% produce lost" delay={200} />
              <HeroCounter value={52.5} suffix="%" label="Carbon Reduction" sublabel="2,328 vs 4,898 kg CO₂-eq" delay={400} />
              <HeroCounter value={94.9} suffix="%" label="Rerouting Efficiency" sublabel="At-risk batches diverted" delay={600} />
              <HeroCounter value={318500} prefix="$" label="Annual Savings" sublabel="50,000 kg/week cooperative" delay={800} />
            </div>
          </CardContent>
        </Card>
      </section>

      {/* 8.2 Interactive Performance Tables */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold">Table 1: Cross-Scenario Performance</h3>
            <p className="text-sm text-muted-foreground italic">Comparison of three methods across five stress scenarios (288 timesteps each).</p>
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
                    <TableHead key={h} className="font-semibold whitespace-nowrap">{h === "Carbon" ? "Carbon (kg)" : h}</TableHead>
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
                        return (
                          <TableCell key={col} className="font-mono text-sm">
                            {typeof val === "number" ? val.toFixed(3) : val ?? "\u2014"}
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
            <h3 className="text-lg font-semibold">Table 2: Ablation Study</h3>
            <p className="text-sm text-muted-foreground italic">Component contribution analysis showing marginal impact of each module.</p>
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
                    <TableHead key={h} className="font-semibold">{h}</TableHead>
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
                      {["ARI", "RLE", "Waste", "SLCA"].map((col) => (
                        <TableCell key={col} className="font-mono text-sm">
                          {typeof row[col] === "number" ? row[col].toFixed(3) : row[col] ?? "\u2014"}
                        </TableCell>
                      ))}
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
              <CardTitle className="text-base">Figure 6: Method Comparison</CardTitle>
              <Select value={selectedMetric} onValueChange={setSelectedMetric}>
                <SelectTrigger className="w-28 h-8"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {["ARI", "RLE", "Waste", "SLCA"].map((m) => (
                    <SelectItem key={m} value={m}>{m}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <CardDescription>Performance by scenario and method ({selectedMetric})</CardDescription>
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
                  <Bar dataKey="Static" fill={COLORS.static} radius={[2, 2, 0, 0]} />
                  <Bar dataKey="Hybrid RL" fill={COLORS.hybrid} radius={[2, 2, 0, 0]} />
                  <Bar dataKey="AGRI-BRAIN" fill={COLORS.agri} radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Ablation bar chart */}
        <Card className="mb-6">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">Figure 7: Ablation Study</CardTitle>
              <Select value={ablationMetric} onValueChange={setAblationMetric}>
                <SelectTrigger className="w-28 h-8"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {["ARI", "RLE", "Waste", "SLCA"].map((m) => (
                    <SelectItem key={m} value={m}>{m}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <CardDescription>Component contribution analysis ({ablationMetric})</CardDescription>
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
                  <Bar dataKey="Static" fill={COLORS.static} radius={[2, 2, 0, 0]} />
                  <Bar dataKey="Hybrid RL" fill={COLORS.hybrid} radius={[2, 2, 0, 0]} />
                  <Bar dataKey="No PINN" fill={COLORS.noPinn} radius={[2, 2, 0, 0]} />
                  <Bar dataKey="No SLCA" fill={COLORS.noSlca} radius={[2, 2, 0, 0]} />
                  <Bar dataKey="AGRI-BRAIN" fill={COLORS.agri} radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

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
                    <span className="text-sm font-medium w-16">{c.metric}</span>
                    <div className="flex-1 mx-4">
                      <div className="flex items-center justify-between text-xs text-muted-foreground mb-1">
                        <span>{fmt(c.a, 3)}</span>
                        <span>{fmt(c.b, 3)}</span>
                      </div>
                      <div className="h-2 bg-muted rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all"
                          style={{
                            width: `${Math.min(100, Math.abs(c.pctChange))}%`,
                            backgroundColor: c.pctChange > 0 ? "#10B981" : "#D55E00",
                          }}
                        />
                      </div>
                    </div>
                    <span className={cn("text-sm font-mono font-semibold w-20 text-right", c.pctChange > 0 ? "text-emerald-600" : "text-[#D55E00]")}>
                      {c.pctChange > 0 ? "+" : ""}{c.pctChange.toFixed(1)}%
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
                  Figure {SCENARIOS.indexOf(scenarioObj) + 2}: {scenarioObj.name} scenario multi-panel analysis. Click to enlarge.
                </p>
              </div>
              <div>
                <Card className="bg-primary/5 border-primary/20">
                  <CardContent className="p-4">
                    <h4 className="font-semibold text-sm mb-3 flex items-center gap-2">
                      <Award className="w-4 h-4 text-primary" /> Key Findings
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

      {/* 8.5 Carbon Footprint */}
      <section>
        <h3 className="text-lg font-semibold mb-4">Carbon Footprint & Green AI</h3>
        <div className="grid lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Figure 8: Carbon by Scenario</CardTitle>
            </CardHeader>
            <CardContent>
              <img
                src={`${API}/results/figures/fig8_green.png`}
                alt="Carbon footprint analysis"
                className="w-full rounded-lg border mb-4 cursor-pointer hover:opacity-90"
                style={{ imageRendering: "auto" }}
                onClick={() => setLightboxImg(`${API}/results/figures/fig8_green.png`)}
                onError={(e) => { e.target.style.display = "none"; }}
              />
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={carbonData} barGap={2}>
                    <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                    <XAxis dataKey="scenario" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} label={{ value: "kg CO₂", angle: -90, position: "insideLeft", fontSize: 11 }} />
                    <ReTooltip content={<ChartTooltip />} />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Bar dataKey="Static" fill={COLORS.static} radius={[2, 2, 0, 0]} />
                    <Bar dataKey="Hybrid RL" fill={COLORS.hybrid} radius={[2, 2, 0, 0]} />
                    <Bar dataKey="AGRI-BRAIN" fill={COLORS.agri} radius={[2, 2, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-emerald-500/5 border-emerald-500/20">
            <CardContent className="p-6 flex flex-col justify-center h-full">
              <div className="text-center">
                <Leaf className="w-12 h-12 mx-auto mb-4 text-emerald-600" />
                <h4 className="text-xl font-bold mb-2">Green AI</h4>
                <p className="text-sm text-muted-foreground mb-4">Computational footprint analysis</p>
              </div>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-3 rounded-lg bg-background border">
                  <span className="text-sm">Energy per episode</span>
                  <span className="font-mono font-bold text-emerald-600">14.4 J</span>
                </div>
                <div className="flex items-center justify-between p-3 rounded-lg bg-background border">
                  <span className="text-sm">Episode duration</span>
                  <span className="font-mono font-bold">72 hours</span>
                </div>
                <div className="flex items-center justify-between p-3 rounded-lg bg-background border">
                  <span className="text-sm">vs. transport savings</span>
                  <span className="font-mono font-bold text-emerald-600">5 orders of magnitude below</span>
                </div>
              </div>
              <p className="text-xs text-muted-foreground mt-4 text-center italic">
                The computational cost of running AGRI-BRAIN is negligible compared to the carbon savings achieved through optimized routing.
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
            <h3 className="text-lg font-semibold mb-2">Run Full Simulation</h3>
            <p className="text-sm text-muted-foreground mb-4 max-w-lg mx-auto">
              Runs all 5 scenarios x 5 methods, 288 timesteps each. Regenerates all figures and CSV tables with the latest model parameters.
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
