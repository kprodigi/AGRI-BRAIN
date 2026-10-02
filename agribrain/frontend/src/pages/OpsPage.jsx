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
  TrendingDown, RefreshCw,
} from "lucide-react";

const API = getApiBase();

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
        <CardContent className="p-3">
          <div className="flex items-start justify-between">
            <div className="space-y-2">
              <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">{label}</p>
              <div className={cn("text-2xl font-bold tabular-nums", toneColors[tone])}>
                <AnimatedCounter value={value} decimals={decimals} suffix={suffix} />
              </div>
              {description && <p className="text-xs text-muted-foreground whitespace-nowrap">{description}</p>}
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
      <p className="font-semibold text-muted-foreground mb-1">{label || "—"}</p>
      {payload.map((p, i) => (
        <div key={i} className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full" style={{ background: p.color }} />
          <span className="text-muted-foreground">{p.name}:</span>
          <span className="font-mono font-semibold">{typeof p.value === 'number' ? p.value.toFixed(2) : p.value}</span>
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

function timestampMillis(value) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value < 1e12 ? value * 1000 : value;
  }
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function filterRowsByElapsedHours(rows, hours) {
  if (hours === Infinity) return rows;
  const withTimes = rows
    .map((row) => ({ row, ms: timestampMillis(row.ts) }))
    .filter(({ ms }) => ms !== null);
  if (withTimes.length === 0) return [];
  const latestMs = Math.max(...withTimes.map(({ ms }) => ms));
  const cutoffMs = latestMs - hours * 60 * 60 * 1000;
  return withTimes.filter(({ ms }) => ms >= cutoffMs).map(({ row }) => row);
}

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
    return filterRowsByElapsedHours(telRows, range.hours);
  }, [telRows, timeRange]);

  const shelfSeries = useMemo(() => {
    const rows = (pred?.shelf_left || []).map((v, i) => ({
      i,
      ts: pred?.timestamp?.[i],
      shelf_left: v,
    }));
    const range = TIME_RANGES.find((r) => r.label === timeRange);
    return !range ? rows : filterRowsByElapsedHours(rows, range.hours);
  }, [pred, timeRange]);

  if (loading) {
    return (
      <div className="space-y-2">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <Card key={i}><CardContent className="p-3"><Skeleton className="h-20" /></CardContent></Card>
          ))}
        </div>
        <Card><CardContent className="p-3"><Skeleton className="h-64" /></CardContent></Card>
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
    <div className="space-y-1">
      {/* KPI summary grid */}
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
          description={avgTemp > 8 ? "Above declared 8°C envelope ceiling" : avgTemp > 6 ? "Approaching the 8°C ceiling" : "Within the declared 8°C envelope ceiling"}
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
          description={`Baseline: ${fmt(wasteBaseline * 100, 1)}%`}
          delay={3}
        />
      </div>

      {/* Featured panel: Supply Chain Health */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
        <Card>
          <CardHeader className="pb-0">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs uppercase tracking-wider text-muted-foreground font-semibold">Cold Chain</p>
                <CardTitle className="text-lg font-bold">Benchmark State Series</CardTitle>
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
          <CardContent className="pt-0 px-4 pb-3">
            <div className="h-28">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={filteredTelRows}>
                  <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                  <XAxis
                    dataKey="ts"
                    tick={{ fontSize: 19 }}
                    tickFormatter={(v) => {
                      if (!v) return "";
                      const d = new Date(v);
                      return isNaN(d) ? String(v).slice(-5) : d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
                    }}
                  />
                  <YAxis yAxisId="temp" tick={{ fontSize: 19 }} label={{ value: "°C", position: "insideTopLeft", fontSize: 13, offset: -5 }} />
                  <YAxis yAxisId="shock" orientation="right" tick={{ fontSize: 19 }} label={{ value: "g", position: "insideTopRight", fontSize: 13, offset: -5 }} domain={[0, 0.15]} />
                  <ReTooltip content={<ChartTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 19 }} />
                  {/* Temperature zones */}
                  {/* The declared operating envelope is one-sided: compliance.py checks
                        temperature > temp_max_c (8 C for spinach) and declares no lower
                        bound, so the shaded region runs from the axis floor to the ceiling
                        rather than inventing a 2 C floor. */}
                    <ReferenceArea yAxisId="temp" y2={8} fill="#10B981" fillOpacity={0.16} label="" />
                  <ReferenceArea yAxisId="temp" y1={8} y2={12} fill="#F59E0B" fillOpacity={0.16} label="" />
                  <ReferenceArea yAxisId="temp" y1={12} y2={50} fill="#D55E00" fillOpacity={0.16} label="" />
                  <ReferenceLine yAxisId="temp" y={8} stroke="#F59E0B" strokeDasharray="5 5" strokeWidth={2} />
                  <ReferenceLine yAxisId="temp" y={12} stroke="#D55E00" strokeDasharray="5 5" strokeWidth={2} />
                  <Line yAxisId="temp" type="monotone" dataKey="tempC" name="Temp (°C)" stroke="#009688" strokeWidth={3.5} dot={false} />
                  <Line yAxisId="temp" type="monotone" dataKey="ambientC" name="Ambient (°C)" stroke="#0072B2" strokeWidth={3} dot={false} />
                  <Line yAxisId="shock" type="monotone" dataKey="shockG" name="Shock (g)" stroke="#D55E00" strokeWidth={3} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Telemetry grid */}
            <div className="mt-1 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-7 gap-1.5">
              {[
                { label: "Timestamp", value: last(tel?.timestamp) },
                { label: "Temp °C", value: fmt(last(tel?.tempC)) },
                { label: "RH %", value: fmt(last(tel?.RH)) },
                { label: "Ambient °C", value: fmt(last(tel?.ambientC)) },
                { label: "Shock (g)", value: fmt(last(tel?.shockG)) },
                { label: "Inventory", value: fmt(last(tel?.inventory_units), 0) },
                { label: "Demand", value: fmt(last(tel?.demand_units), 0) },
              ].map((item) => (
                <div key={item.label} className="px-2 py-1 rounded-lg bg-muted/50 border">
                  <span className="text-xs text-muted-foreground">{item.label}</span>
                  <div className="font-mono text-sm font-semibold truncate">{item.value ?? "\u2014"}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Predictions */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}>
        <Card>
          <CardHeader className="pb-0">
            <div>
              <p className="text-xs uppercase tracking-wider text-muted-foreground font-semibold">Model Output</p>
              <CardTitle className="text-lg font-bold">Mechanistic Quality & Forecast Preview</CardTitle>
            </div>
          </CardHeader>
          <CardContent className="pt-0 px-4 pb-3">
            <div className="grid lg:grid-cols-3 gap-4">
              <div className="lg:col-span-2 h-24">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={shelfSeries}>
                    <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                    <XAxis dataKey="ts" tick={{ fontSize: 19 }} label={{ value: "Timestamp", position: "insideBottom", offset: -5, fontSize: 13 }} />
                    <YAxis domain={[0, 1]} tick={{ fontSize: 19 }} label={{ value: "Modeled quality remaining", angle: -90, position: "insideLeft", fontSize: 13 }} />
                    <ReTooltip content={<ChartTooltip />} />
                    <Area dataKey="shelf_left" name="Modeled Quality Remaining" stroke="#00796B" fill="#009688" fillOpacity={0.25} strokeWidth={3.5} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <Card className="bg-muted/30">
                <CardContent className="p-3 space-y-1.5">
                  <div>
                    <p className="text-xs text-muted-foreground">Latest modeled quality remaining</p>
                    <p className="text-xl font-bold font-mono">{fmt(last(pred?.shelf_left))}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Volatility</p>
                    <p className="font-mono text-sm">{last(pred?.volatility) ?? "\u2014"}</p>
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
