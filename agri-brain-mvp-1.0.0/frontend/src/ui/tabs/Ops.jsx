// frontend/src/ui/tabs/Ops.jsx
import React, { useEffect, useMemo, useState } from "react";
import {
    KPIStat,
    Section,
    Button,
    Sparkline,
    useToast,
    Trend,
} from "../components/UiPrimitives.jsx";
import { Thermometer, Activity, Recycle, BarChart3, AlertTriangle } from "lucide-react";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    Tooltip,
    ResponsiveContainer,
    CartesianGrid,
    Legend,
    Area,
    AreaChart,
} from "recharts";

const API =
    (window.API_BASE || localStorage.getItem("API_BASE") || "http://127.0.0.1:8100").replace(
        /\/$/,
        ""
    );

// ---------------- helpers ----------------
async function jget(path) {
    const r = await fetch(`${API}${path}`);
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return r.json();
}
const n = (v) => (Number.isFinite(+v) ? +v : null);
const fmt = (v, d = 2) => (Number.isFinite(+v) ? (+v).toFixed(d) : "—");
const last = (arr) => (Array.isArray(arr) && arr.length ? arr[arr.length - 1] : null);

// Simple local card wrapper (fixes "Card is not defined")
function Card({ className = "", children }) {
    return (
        <div className={`rounded-xl border p-5 bg-white shadow-sm ${className}`}>
            {children}
        </div>
    );
}

export default function Ops() {
    const toast = useToast();
    const [kpis, setKpis] = useState(null);
    const [tel, setTel] = useState(null);
    const [pred, setPred] = useState(null);
    const [loading, setLoading] = useState(true);
    const [err, setErr] = useState("");

    useEffect(() => {
        let ok = true;
        (async () => {
            try {
                setErr("");
                const [k, t, p] = await Promise.all([jget("/kpis"), jget("/telemetry"), jget("/predictions")]);
                if (!ok) return;
                setKpis(k || {});
                setTel(t || {});
                setPred(p || {});
                setLoading(false);
            } catch (e) {
                if (ok) {
                    setErr("Could not load data from API.");
                    setLoading(false);
                }
            }
        })();

        const id = setInterval(async () => {
            try {
                const k = await jget("/kpis");
                if (ok) setKpis(k || {});
            } catch {
                /* ignore periodic refresh errors */
            }
        }, 15000);

        return () => {
            ok = false;
            clearInterval(id);
        };
    }, []);

    // derived values
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

    const shelfSeries = useMemo(
        () => (pred?.shelf_left || []).map((v, i) => ({ i, v })),
        [pred]
    );

    return (
        <div className="max-w-7xl mx-auto p-6 space-y-6">
            {err && <div className="p-3 rounded-md bg-red-100 text-red-700">{err}</div>}

            {/* KPI row */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <KPIStat
                    label="Records"
                    value={records}
                    icon={BarChart3}
                    sub={records ? "rows loaded" : "loading…"}
                    spark={<Sparkline data={telRows.map((r, i) => ({ v: (r.inventory ?? 0) - (r.demand ?? 0), i }))} />}
                />
                <KPIStat
                    label="Avg temp (°C)"
                    value={fmt(avgTemp, 2)}
                    icon={Thermometer}
                    tone={avgTemp > 6 ? "warn" : avgTemp < 4 ? "good" : "default"}
                    spark={<Sparkline data={telRows.map((r) => ({ v: r.tempC ?? 0 }))} />}
                />
                <KPIStat
                    label="Anomaly points"
                    value={anomalies}
                    icon={Activity}
                    spark={<Sparkline data={telRows.map((r) => ({ v: r.shockG ?? 0 }))} />}
                    tone={anomalies > 0 ? "warn" : "default"}
                />
                <KPIStat
                    label="Waste (Agri-Brain)"
                    value={`${fmt(wasteAgri * 100, 1)} %`}
                    sub={<span className="text-xs">Baseline {fmt(wasteBaseline * 100, 1)} %</span>}
                    icon={Recycle}
                    spark={<Sparkline data={shelfSeries} />}
                    tone={wasteAgri < wasteBaseline ? "good" : wasteAgri > wasteBaseline ? "warn" : "default"}
                />
            </div>

            {/* Telemetry */}
            <Section
                title="Telemetry (latest streams)"
                kicker="cold chain"
                right={
                    <Button
                        variant="subtle"
                        onClick={() => toast.show("Refreshing KPIs…", "info")}
                        className="whitespace-nowrap"
                    >
                        Refresh
                    </Button>
                }
            >
                <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={telRows}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="ts" hide />
                            <YAxis domain={["auto", "auto"]} />
                            <Tooltip />
                            <Legend />
                            <Line type="monotone" dataKey="tempC" name="Temp (°C)" strokeWidth={2} dot={false} />
                            <Line type="monotone" dataKey="ambientC" name="Ambient (°C)" strokeWidth={2} dot={false} />
                            <Line type="monotone" dataKey="shockG" name="Shock (g)" strokeWidth={2} dot={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>

                <div className="mt-4 grid sm:grid-cols-2 lg:grid-cols-3 gap-3 text-sm">
                    <Mono label="timestamp" value={last(tel?.timestamp)} />
                    <Mono label="tempC" value={fmt(last(tel?.tempC))} />
                    <Mono label="RH" value={fmt(last(tel?.RH))} />
                    <Mono label="ambientC" value={fmt(last(tel?.ambientC))} />
                    <Mono label="shockG" value={fmt(last(tel?.shockG))} />
                    <Mono label="inventory_units" value={fmt(last(tel?.inventory_units), 0)} />
                    <Mono label="demand_units" value={fmt(last(tel?.demand_units), 0)} />
                </div>
            </Section>

            {/* Predictions */}
            <Section title="Spoilage & Yield (preview)" kicker="model output">
                <div className="grid lg:grid-cols-3 gap-4">
                    <Card className="lg:col-span-2">
                        <div className="h-56">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={(pred?.shelf_left || []).map((v, i) => ({ i, shelf_left: v }))}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="i" />
                                    <YAxis domain={[0, 1]} />
                                    <Tooltip />
                                    <Area dataKey="shelf_left" name="shelf_left" strokeWidth={2} />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </Card>
                    <Card>
                        <div className="space-y-2">
                            <Trend
                                label="Last shelf_left"
                                value={fmt(last(pred?.shelf_left))}
                                delta={(last(pred?.shelf_left) ?? 0) - (pred?.shelf_left?.[0] ?? 0)}
                            />
                            <div className="text-sm text-gray-600">
                                Volatility: <b>{last(pred?.volatility) ?? "—"}</b>
                            </div>
                            <div className="text-sm text-gray-600">
                                Forecast horizon: <b>{pred?.yield_forecast_24h?.length ?? 0}h</b>
                            </div>
                            {avgTemp > 6 && (
                                <div className="flex items-start gap-2 text-amber-700 text-sm">
                                    <AlertTriangle className="w-4 h-4 mt-0.5" /> High average temp — expect faster spoilage.
                                </div>
                            )}
                        </div>
                    </Card>
                </div>
            </Section>

            {/* gentle empty state */}
            {loading && (
                <Card className="text-sm text-gray-600 flex items-center gap-2">
                    Loading data… if this takes too long, ensure the backend is running at <code>{API}</code>.
                </Card>
            )}
        </div>
    );
}

function Mono({ label, value }) {
    return (
        <div className="px-3 py-2 rounded-xl bg-gray-50 border">
            <span className="text-gray-500">{label}: </span>
            <code>{value ?? "—"}</code>
        </div>
    );
}
