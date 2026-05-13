import React, { useEffect, useMemo, useState, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { cn, n, fmt, last, jget } from "@/lib/utils";
import { getApiBase } from "@/mvp/api.js";
import { motion, animate } from "framer-motion";
import {
  LineChart, Line, XAxis, YAxis, Tooltip as ReTooltip, ResponsiveContainer,
  CartesianGrid, Legend, Area, AreaChart, ReferenceArea, ReferenceLine,
} from "recharts";
import {
  Thermometer, Activity, Recycle, BarChart3, AlertTriangle, TrendingUp,
  TrendingDown, RefreshCw, Sparkles, X, ArrowRight,
} from "lucide-react";
import { Link } from "react-router-dom";

const API = getApiBase();

// --------------------------------------------------------------------
// Showcase headline banner. Surfaces the canonical 20-seed paper claims
// to a first-time guest landing on the ops dashboard. Numbers are pinned
// against the manuscript's Table 7 / §5.8 — they should match
// mvp/simulation/results/paper_benchmark_table.json and
// mvp/simulation/results/decision_ledger_aggregate.json. Update the
// HEADLINE_CLAIMS list below whenever a new canonical HPC re-run lands.
//
// The banner is dismissable. State persists in localStorage under
// `ops.showcaseBanner.dismissed` so returning operators don't see it,
// but a guest opening a fresh browser will.
// --------------------------------------------------------------------
const HEADLINE_CLAIMS = [
  {
    label: "Integration superiority (H1)",
    value: "+0.019 to +0.035",
    unit: "ΔARI vs no-context",
    detail: "All 5 scenarios, Cohen's dz 3.34–7.40, p_adj < 0.001",
  },
  {
    label: "Component complementarity (H3)",
    value: "20.1%",
    unit: "super-additive integration",
    detail: "n = 23,040 decisions; joint > max single by > 0.005 ARI-eq",
  },
  {
    label: "Provenance integrity",
    value: "100%",
    unit: "verifiable Merkle roots",
    detail: "On-chain anchor via ProvenanceRegistry.sol",
  },
];

function ShowcaseBanner() {
  const [dismissed, setDismissed] = useState(() => {
    try { return localStorage.getItem("ops.showcaseBanner.dismissed") === "1"; }
    catch { return false; }
  });
  if (dismissed) return null;
  const dismiss = () => {
    setDismissed(true);
    try { localStorage.setItem("ops.showcaseBanner.dismissed", "1"); } catch {}
  };
  return (
    <motion.div
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="relative rounded-xl border border-teal-500/30 bg-gradient-to-br from-teal-500/8 via-blue-500/5 to-purple-500/8 p-4 sm:p-5"
    >
      <button
        onClick={dismiss}
        aria-label="Dismiss showcase banner"
        className="absolute top-2 right-2 p-1 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted/50 transition"
      >
        <X className="w-4 h-4" />
      </button>
      <div className="flex items-start gap-3">
        <div className="rounded-lg p-2 bg-teal-500/15">
          <Sparkles className="w-5 h-5 text-teal-600" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h2 className="text-sm sm:text-base font-semibold">
              AGRI-BRAIN — adaptive supply-chain intelligence, evidence-anchored.
            </h2>
            <span className="text-[10px] uppercase tracking-wider rounded-full px-2 py-0.5 bg-teal-500/10 text-teal-700 dark:text-teal-300 border border-teal-500/20">
              canonical 20-seed run
            </span>
          </div>
          <p className="text-xs text-muted-foreground mt-1 leading-relaxed">
            5 perturbation scenarios × 8 routing modes × 20 stochastic seeds × 288 hourly steps =
            230,400 evaluated decisions. Three hypotheses established:
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-3">
            {HEADLINE_CLAIMS.map((c) => (
              <div key={c.label} className="rounded-lg border border-border/50 bg-background/60 p-3">
                <div className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium">
                  {c.label}
                </div>
                <div className="text-lg sm:text-xl font-bold tabular-nums mt-0.5">
                  {c.value}{" "}
                  <span className="text-xs font-normal text-muted-foreground">{c.unit}</span>
                </div>
                <div className="text-[10px] text-muted-foreground mt-1 leading-tight">
                  {c.detail}
                </div>
              </div>
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-2 mt-3">
            <Link to="/mcp-pirag" className="inline-flex items-center gap-1 text-xs font-medium text-teal-700 dark:text-teal-300 hover:underline">
              See H3 mechanism evidence
              <ArrowRight className="w-3 h-3" />
            </Link>
            <span className="text-muted-foreground/50 text-xs">·</span>
            <Link to="/demo" className="inline-flex items-center gap-1 text-xs font-medium text-teal-700 dark:text-teal-300 hover:underline">
              Walkthrough demo
              <ArrowRight className="w-3 h-3" />
            </Link>
            <span className="text-muted-foreground/50 text-xs">·</span>
            <Link to="/analytics" className="inline-flex items-center gap-1 text-xs font-medium text-teal-700 dark:text-teal-300 hover:underline">
              Cross-scenario benchmark
              <ArrowRight className="w-3 h-3" />
            </Link>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

// Animated counter component
function AnimatedCounter({ value, decimals = 0, duration = 1.2, prefix = "", suffix = "" }) {
  const nodeRef = useRef(null);
  const prevValue = useRef(0);

  useEffect(() => {
    const target = Number.isFinite(+value) ? +value : 0;
    const from = prevValue.current;
    prevValue.current = target;

    const controls = animate(from, target, {
      duration,
      onUpdate(v) {
        if (nodeRef.current) {
          nodeRef.current.textContent = prefix + v.toFixed(decimals) + suffix;
        }
      },
    });
    return () => controls.stop();
  }, [value, decimals, duration, prefix, suffix]);

  return <span ref={nodeRef}>{prefix}{Number.isFinite(+value) ? (+value).toFixed(decimals) : "\u2014"}{suffix}</span>;
}

// KPI Card component
function KPICard({ label, value, decimals = 0, suffix = "", icon: Icon, trend, description, tone = "default", className, delay = 0 }) {
  const toneColors = {
    default: "text-foreground",
    good: "text-emerald-600 dark:text-emerald-400",
    warn: "text-amber-600 dark:text-amber-400",
    critical: "text-[#D55E00]",
  };
  const toneBg = {
    default: "bg-primary/10",
    good: "bg-emerald-500/10",
    warn: "bg-amber-500/10",
    critical: "bg-[#D55E00]/10",
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: delay * 0.1, duration: 0.4 }}
      className={className}
    >
      <Card className="h-full">
        <CardContent className="p-5">
          <div className="flex items-start justify-between">
            <div className="space-y-2">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">{label}</p>
              <div className={cn("text-2xl font-bold tabular-nums", toneColors[tone])}>
                <AnimatedCounter value={value} decimals={decimals} suffix={suffix} />
              </div>
              {description && <p className="text-xs text-muted-foreground">{description}</p>}
            </div>
            <div className={cn("p-2.5 rounded-xl", toneBg[tone])}>
              <Icon className={cn("w-5 h-5", toneColors[tone])} />
            </div>
          </div>
          {trend !== undefined && trend !== null && (
            <div className="mt-3 flex items-center gap-1 text-xs">
              {trend >= 0 ? (
                <TrendingUp className="w-3.5 h-3.5 text-emerald-500" />
              ) : (
                <TrendingDown className="w-3.5 h-3.5 text-red-500" />
              )}
              <span className={trend >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-400"}>
                {trend >= 0 ? "+" : ""}{typeof trend === 'number' ? trend.toFixed(1) : trend}%
              </span>
              <span className="text-muted-foreground">vs baseline</span>
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}

// Custom tooltip for charts
function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border bg-popover p-3 shadow-md text-sm">
      <p className="font-medium text-muted-foreground mb-1">{label || "—"}</p>
      {payload.map((p, i) => (
        <div key={i} className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full" style={{ background: p.color }} />
          <span className="text-muted-foreground">{p.name}:</span>
          <span className="font-mono font-medium">{typeof p.value === 'number' ? p.value.toFixed(2) : p.value}</span>
        </div>
      ))}
    </div>
  );
}

// Time range buttons
const TIME_RANGES = [
  { label: "6h", hours: 6 },
  { label: "12h", hours: 12 },
  { label: "24h", hours: 24 },
  { label: "72h", hours: 72 },
  { label: "All", hours: Infinity },
];

export default function OpsPage() {
  const [kpis, setKpis] = useState(null);
  const [tel, setTel] = useState(null);
  const [pred, setPred] = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");
  const [timeRange, setTimeRange] = useState("All");

  useEffect(() => {
    let ok = true;
    (async () => {
      try {
        setErr("");
        const [k, t, p] = await Promise.all([
          jget(API, "/kpis"), jget(API, "/telemetry"), jget(API, "/predictions"),
        ]);
        if (!ok) return;
        setKpis(k || {});
        setTel(t || {});
        setPred(p || {});
        setLoading(false);
      } catch {
        if (ok) { setErr("Could not load data from API."); setLoading(false); }
      }
    })();

    const refresh = async () => {
      try {
        const [k, t, p] = await Promise.all([
          jget(API, "/kpis").catch(() => null),
          jget(API, "/telemetry").catch(() => null),
          jget(API, "/predictions").catch(() => null),
        ]);
        if (ok) {
          if (k) setKpis(k);
          if (t) setTel(t);
          if (p) setPred(p);
        }
      } catch {}
    };
    const id = setInterval(refresh, 15000);

    // WebSocket-driven refresh: when the backend broadcasts a new
    // decision over /stream, useWebSocket dispatches a `decision:new`
    // DOM event. Re-fetch immediately so the dashboard reflects the
    // post-decision state instead of waiting up to 15 s for the next
    // poll. The polling timer above stays as the safety net.
    const onDecision = () => { refresh(); };
    document.addEventListener("decision:new", onDecision);

    return () => {
      ok = false;
      clearInterval(id);
      document.removeEventListener("decision:new", onDecision);
    };
  }, []);

  // Derived
  const records = n(kpis?.records) ?? 0;
  const avgTemp = n(kpis?.avg_tempC ?? kpis?.avg_temp_c);
  const anomalies = n(kpis?.anomaly_points) ?? 0;
  const wasteBaseline = n(kpis?.waste_rate_baseline ?? kpis?.waste_baseline_pct) ?? 0;
  const wasteAgri = n(kpis?.waste_rate_agri ?? kpis?.waste_agri_pct) ?? 0;
  const wasteReduction = wasteBaseline > 0 ? ((wasteBaseline - wasteAgri) / wasteBaseline * 100) : 0;

  const telRows = useMemo(() => {
    if (!tel?.timestamp) return [];
    return tel.timestamp.map((ts, i) => ({
      ts,
      tempC: tel.tempC?.[i],
      RH: tel.RH?.[i],
      ambientC: tel.ambientC?.[i],
      shockG: tel.shockG?.[i],
      inventory: tel.inventory_units?.[i],
      demand: tel.demand_units?.[i],
    }));
  }, [tel]);

  const filteredTelRows = useMemo(() => {
    const range = TIME_RANGES.find((r) => r.label === timeRange);
    if (!range || range.hours === Infinity) return telRows;
    return telRows.slice(-range.hours);
  }, [telRows, timeRange]);

  const shelfSeries = useMemo(
    () => (pred?.shelf_left || []).map((v, i) => ({ i, shelf_left: v })),
    [pred]
  );

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <Card key={i}><CardContent className="p-5"><Skeleton className="h-20" /></CardContent></Card>
          ))}
        </div>
        <Card><CardContent className="p-5"><Skeleton className="h-64" /></CardContent></Card>
      </div>
    );
  }

  if (err) {
    return (
      <Card className="border-destructive/50">
        <CardContent className="p-6 text-center">
          <AlertTriangle className="w-8 h-8 mx-auto mb-2 text-destructive" />
          <p className="text-sm text-destructive">{err}</p>
          <Button variant="outline" size="sm" className="mt-3" onClick={() => window.location.reload()}>
            <RefreshCw className="w-4 h-4 mr-1" /> Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Showcase headline — paper claims at a glance (dismissable; see ShowcaseBanner) */}
      <ShowcaseBanner />

      {/* KPI Bento Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <KPICard
          label="Records Loaded"
          value={records}
          icon={BarChart3}
          description={records ? "rows loaded" : "loading..."}
          delay={0}
        />
        <KPICard
          label="Avg Temperature"
          value={avgTemp}
          decimals={2}
          suffix=" °C"
          icon={Thermometer}
          tone={avgTemp > 8 ? "critical" : avgTemp > 6 ? "warn" : "good"}
          description={avgTemp > 8 ? "Above critical threshold" : avgTemp > 6 ? "Above warning threshold" : "Within safe range (2–8°C)"}
          delay={1}
        />
        <KPICard
          label="Anomaly Points"
          value={anomalies}
          icon={Activity}
          tone={anomalies > 0 ? "warn" : "default"}
          description={anomalies > 0 ? "Shock events detected" : "No anomalies"}
          delay={2}
        />
        <KPICard
          label="Waste Rate"
          value={wasteAgri * 100}
          decimals={1}
          suffix="%"
          icon={Recycle}
          tone={wasteAgri < wasteBaseline ? "good" : "warn"}
          trend={wasteReduction > 0 ? wasteReduction : undefined}
          description={`Baseline: ${fmt(wasteBaseline * 100, 1)}%`}
          delay={3}
        />
      </div>

      {/* Hero: Supply Chain Health */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
        <Card>
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs uppercase tracking-wider text-muted-foreground font-medium">Cold Chain</p>
                <CardTitle className="text-lg">Telemetry Streams</CardTitle>
              </div>
              <div className="flex items-center gap-1">
                {TIME_RANGES.map((r) => (
                  <Button
                    key={r.label}
                    variant={timeRange === r.label ? "default" : "ghost"}
                    size="sm"
                    className="h-7 px-2 text-xs"
                    onClick={() => setTimeRange(r.label)}
                  >
                    {r.label}
                  </Button>
                ))}
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={filteredTelRows}>
                  <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                  <XAxis
                    dataKey="ts"
                    tick={{ fontSize: 11 }}
                    tickFormatter={(v) => {
                      if (!v) return "";
                      const d = new Date(v);
                      return isNaN(d) ? String(v).slice(-5) : d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
                    }}
                  />
                  <YAxis yAxisId="temp" tick={{ fontSize: 11 }} label={{ value: "°C", position: "insideTopLeft", fontSize: 10, offset: -5 }} />
                  <YAxis yAxisId="shock" orientation="right" tick={{ fontSize: 11 }} label={{ value: "g", position: "insideTopRight", fontSize: 10, offset: -5 }} domain={[0, 0.15]} />
                  <ReTooltip content={<ChartTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  {/* Temperature zones */}
                  <ReferenceArea yAxisId="temp" y1={2} y2={8} fill="#10B981" fillOpacity={0.06} label="" />
                  <ReferenceArea yAxisId="temp" y1={8} y2={12} fill="#F59E0B" fillOpacity={0.06} label="" />
                  <ReferenceArea yAxisId="temp" y1={12} y2={50} fill="#D55E00" fillOpacity={0.06} label="" />
                  <ReferenceLine yAxisId="temp" y={8} stroke="#F59E0B" strokeDasharray="5 5" strokeWidth={1} />
                  <ReferenceLine yAxisId="temp" y={12} stroke="#D55E00" strokeDasharray="5 5" strokeWidth={1} />
                  <Line yAxisId="temp" type="monotone" dataKey="tempC" name="Temp (°C)" stroke="#009688" strokeWidth={2} dot={false} />
                  <Line yAxisId="temp" type="monotone" dataKey="ambientC" name="Ambient (°C)" stroke="#0072B2" strokeWidth={1.5} dot={false} />
                  <Line yAxisId="shock" type="monotone" dataKey="shockG" name="Shock (g)" stroke="#D55E00" strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Telemetry grid */}
            <div className="mt-4 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-7 gap-2">
              {[
                { label: "Timestamp", value: last(tel?.timestamp) },
                { label: "Temp °C", value: fmt(last(tel?.tempC)) },
                { label: "RH %", value: fmt(last(tel?.RH)) },
                { label: "Ambient °C", value: fmt(last(tel?.ambientC)) },
                { label: "Shock (g)", value: fmt(last(tel?.shockG)) },
                { label: "Inventory", value: fmt(last(tel?.inventory_units), 0) },
                { label: "Demand", value: fmt(last(tel?.demand_units), 0) },
              ].map((item) => (
                <div key={item.label} className="px-3 py-2 rounded-lg bg-muted/50 border">
                  <span className="text-xs text-muted-foreground">{item.label}</span>
                  <div className="font-mono text-sm font-medium truncate">{item.value ?? "\u2014"}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Predictions */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}>
        <Card>
          <CardHeader className="pb-2">
            <div>
              <p className="text-xs uppercase tracking-wider text-muted-foreground font-medium">Model Output</p>
              <CardTitle className="text-lg">Spoilage & Yield Preview</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid lg:grid-cols-3 gap-4">
              <div className="lg:col-span-2 h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={shelfSeries}>
                    <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                    <XAxis dataKey="i" tick={{ fontSize: 11 }} label={{ value: "Timestep", position: "insideBottom", offset: -5, fontSize: 11 }} />
                    <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} label={{ value: "Shelf Left", angle: -90, position: "insideLeft", fontSize: 11 }} />
                    <ReTooltip content={<ChartTooltip />} />
                    <Area dataKey="shelf_left" name="Shelf Life Remaining" stroke="#009688" fill="#009688" fillOpacity={0.1} strokeWidth={2} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <Card className="bg-muted/30">
                <CardContent className="p-4 space-y-3">
                  <div>
                    <p className="text-xs text-muted-foreground">Last shelf_left</p>
                    <p className="text-xl font-bold font-mono">{fmt(last(pred?.shelf_left))}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Volatility</p>
                    <p className="font-mono text-sm">{last(pred?.volatility) ?? "\u2014"}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Forecast Method</p>
                    <p className="text-sm font-medium">{pred?.demand_forecast?.method ?? "\u2014"}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Forecast Horizon</p>
                    <p className="font-mono text-sm">{pred?.yield_forecast_24h?.length ?? pred?.demand_forecast?.forecast?.length ?? 0}h</p>
                  </div>
                  {avgTemp > 6 && (
                    <div className="flex items-start gap-2 text-amber-600 dark:text-amber-400 text-sm p-2 rounded-lg bg-amber-500/10">
                      <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
                      <span>High average temp — expect faster spoilage.</span>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}
