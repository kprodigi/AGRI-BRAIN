import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { cn, fmt, last, authFetch } from "@/lib/utils";
import { getApiBase } from "@/mvp/api.js";
import { motion } from "framer-motion";
import {
  LineChart, Line, XAxis, YAxis, Tooltip as ReTooltip, ResponsiveContainer,
  CartesianGrid, Legend, AreaChart, Area, ReferenceArea,
} from "recharts";
import { Thermometer, Droplets, AlertTriangle, Clock, Shield } from "lucide-react";

const API = getApiBase();

// Circular gauge SVG
function SpoilageGauge({ value = 0, size = 160 }) {
  const pct = Math.max(0, Math.min(1, value));
  const radius = (size - 20) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - pct);
  const cx = size / 2;
  const cy = size / 2;

  const color = pct > 0.5 ? "#D55E00" : pct > 0.3 ? "#F59E0B" : "#10B981";

  return (
    <div className="flex flex-col items-center">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {/* Background circle */}
        <circle cx={cx} cy={cy} r={radius} fill="none" stroke="currentColor" strokeWidth={8} className="text-muted/30" />
        {/* Progress arc */}
        <circle
          cx={cx} cy={cy} r={radius} fill="none"
          stroke={color} strokeWidth={8} strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform={`rotate(-90 ${cx} ${cy})`}
          className="transition-all duration-1000"
        />
        {/* Center text */}
        <text x={cx} y={cy - 8} textAnchor="middle" className="fill-foreground text-2xl font-bold" style={{ fontSize: 28 }}>
          {(pct * 100).toFixed(1)}%
        </text>
        <text x={cx} y={cy + 14} textAnchor="middle" className="fill-muted-foreground" style={{ fontSize: 11 }}>
          Spoilage Risk
        </text>
      </svg>
    </div>
  );
}

// Dimensionless modeled remaining-quality fraction C(t), not time-to-threshold.
function RemainingQualityGauge({ value = 0 }) {
  const pct = Math.max(0, Math.min(1, value));
  const color = pct > 0.5 ? "#10B981" : pct > 0.2 ? "#F59E0B" : "#D55E00";

  const radius = 60;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - pct);

  return (
    <div className="flex flex-col items-center">
      <svg width={150} height={150} viewBox="0 0 150 150">
        <circle cx={75} cy={75} r={radius} fill="none" stroke="currentColor" strokeWidth={6} className="text-muted/30" />
        <circle
          cx={75} cy={75} r={radius} fill="none"
          stroke={color} strokeWidth={6} strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform="rotate(-90 75 75)"
          className="transition-all duration-1000"
        />
        <text x={75} y={68} textAnchor="middle" className="fill-foreground font-bold" style={{ fontSize: 22 }}>
          {(pct * 100).toFixed(1)}%
        </text>
        <text x={75} y={88} textAnchor="middle" className="fill-muted-foreground" style={{ fontSize: 11 }}>
          modeled quality
        </text>
      </svg>
    </div>
  );
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border bg-popover p-3 shadow-md text-sm">
      <p className="font-medium text-muted-foreground mb-1">{label || "\u2014"}</p>
      {payload.map((p, i) => (
        <div key={i} className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full" style={{ background: p.color }} />
          <span className="text-muted-foreground">{p.name}:</span>
          <span className="font-mono font-medium">{typeof p.value === 'number' ? p.value.toFixed(3) : p.value}</span>
        </div>
      ))}
    </div>
  );
}

export default function QualityPage() {
  const [tel, setTel] = useState(null);
  const [pred, setPred] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let ok = true;
    const f = async () => {
      try {
        const [tRes, pRes] = await Promise.all([
          authFetch(`${API}/telemetry`),
          authFetch(`${API}/predictions`),
        ]);
        if (!tRes.ok || !pRes.ok) throw new Error("API error");
        const [t, p] = await Promise.all([tRes.json(), pRes.json()]);
        if (ok) { setTel(t); setPred(p); setLoading(false); }
      } catch {
        if (ok) setLoading(false);
      }
    };
    f();
    const id = setInterval(f, 5000);
    // Push refresh on every new decision broadcast, so the residual/ODE
    // overlay reflects the most recent context window without waiting
    // for the next poll.
    const onDecision = () => { f(); };
    document.addEventListener("decision:new", onDecision);
    return () => {
      ok = false;
      clearInterval(id);
      document.removeEventListener("decision:new", onDecision);
    };
  }, []);

  const series = useMemo(() => {
    if (!tel?.timestamp || !pred?.shelf_left) return [];
    return tel.timestamp.map((t, i) => ({
      t,
      idx: i,
      tempC: tel.tempC?.[i],
      RH: tel.RH?.[i],
      ambientC: tel.ambientC?.[i],
      shockG: tel.shockG?.[i],
      demand: tel.demand_units?.[i],
      shelf: pred.shelf_left?.[i],
    }));
  }, [tel, pred]);

  // Spoilage risk (inverse of last shelf_left)
  const lastShelf = last(pred?.shelf_left) ?? 1;
  const spoilageRisk = 1 - Math.max(0, Math.min(1, lastShelf));

  const mechanisticSeries = useMemo(() => {
    const ode = pred?.shelf_left;
    if (!ode) return [];
    return ode.map((value, i) => ({ i, ode: value }));
  }, [pred]);

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <Card><CardContent className="p-6"><Skeleton className="h-40" /></CardContent></Card>
          <Card><CardContent className="p-6"><Skeleton className="h-40" /></CardContent></Card>
          <Card><CardContent className="p-6"><Skeleton className="h-40" /></CardContent></Card>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Top row: Gauges */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0 }}>
          <Card className="text-center">
            <CardHeader className="pb-0">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center justify-center gap-2">
                <Shield className="w-4 h-4" /> Spoilage Risk
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-4 pb-6 flex justify-center">
              <SpoilageGauge value={spoilageRisk} />
            </CardContent>
          </Card>
        </motion.div>

        <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.1 }}>
          <Card className="text-center">
            <CardHeader className="pb-0">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center justify-center gap-2">
                <Clock className="w-4 h-4" /> Modeled Quality Remaining
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-4 pb-6 flex justify-center">
              <RemainingQualityGauge value={lastShelf} />
            </CardContent>
          </Card>
        </motion.div>

        <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.2 }}>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Current Readings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {[
                { icon: Thermometer, label: "Temperature", value: `${fmt(last(tel?.tempC))} °C`, warn: (last(tel?.tempC) ?? 0) > 8 },
                { icon: Droplets, label: "Humidity", value: `${fmt(last(tel?.RH))}%` },
                { icon: Thermometer, label: "Ambient", value: `${fmt(last(tel?.ambientC))} °C` },
                { icon: AlertTriangle, label: "Shock", value: `${fmt(last(tel?.shockG))} g`, warn: (last(tel?.shockG) ?? 0) > 0.5 },
              ].map((item) => (
                <div key={item.label} className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-sm">
                    <item.icon className={cn("w-4 h-4", item.warn ? "text-amber-500" : "text-muted-foreground")} />
                    <span className="text-muted-foreground">{item.label}</span>
                  </div>
                  <span className={cn("font-mono text-sm font-medium", item.warn && "text-amber-600 dark:text-amber-400")}>
                    {item.value}
                  </span>
                </div>
              ))}
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Synthetic Temperature & Humidity</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={series}>
                    <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                    <XAxis dataKey="t" hide />
                    <YAxis yAxisId="temp" tick={{ fontSize: 11 }} label={{ value: "°C", position: "insideTopLeft", fontSize: 10, offset: -5 }} />
                    <YAxis yAxisId="rh" orientation="right" tick={{ fontSize: 11 }} domain={[60, 100]} label={{ value: "%RH", position: "insideTopRight", fontSize: 10, offset: -5 }} />
                    <ReTooltip content={<ChartTooltip />} />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    {/* One-sided: the declared envelope sets only a ceiling (compliance.py
                        temp_max_c = 8 C for spinach); there is no declared lower bound. */}
                    <ReferenceArea yAxisId="temp" y2={8} fill="#10B981" fillOpacity={0.05} />
                    <ReferenceArea yAxisId="temp" y1={8} y2={12} fill="#F59E0B" fillOpacity={0.05} />
                    <Line yAxisId="temp" type="monotone" dataKey="tempC" dot={false} name="Temp °C" stroke="#009688" strokeWidth={2} />
                    <Line yAxisId="rh" type="monotone" dataKey="RH" dot={false} name="Humidity %" stroke="#0072B2" strokeWidth={1.5} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Spoilage: Shelf-life Remaining</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={series}>
                    <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                    <XAxis dataKey="t" hide />
                    <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                    <ReTooltip content={<ChartTooltip />} />
                    <Area type="monotone" dataKey="shelf" fillOpacity={0.15} name="Shelf Left" stroke="#009688" fill="#009688" strokeWidth={2} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Common mechanistic spoilage-risk estimate */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}>
        <Card>
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">Mechanistic Shelf-Life Trajectory</CardTitle>
              <Badge variant="outline">Common to every benchmark arm</Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={mechanisticSeries}>
                  <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                  <XAxis dataKey="i" tick={{ fontSize: 11 }} label={{ value: "Timestep", position: "insideBottom", offset: -5, fontSize: 11 }} />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                  <ReTooltip content={<ChartTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 12, paddingTop: 16 }} />
                  <Line type="monotone" dataKey="ode" name="Modeled shelf life remaining (Arrhenius-lag)" stroke="#009688" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <p className="text-xs text-muted-foreground mt-2 italic">
              The reported trajectory is the common mechanistic Arrhenius estimate with a declared rational lag factor. It is a synthetic modeled-risk trajectory, not an observed shelf-life prediction.
            </p>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}
