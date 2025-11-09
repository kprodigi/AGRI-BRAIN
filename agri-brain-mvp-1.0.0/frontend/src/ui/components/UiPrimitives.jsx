// ui/components/UiPrimitives.jsx
import React, { createContext, useContext, useMemo, useState, useEffect } from "react";
import cx from "classnames";
import { Area, AreaChart, ResponsiveContainer } from "recharts";
import { CheckCircle2, AlertTriangle, Info, TrendingUp, Gauge } from "lucide-react";

/* =============== Card / Section / Button =============== */
export const Card = ({ className, children }) => (
    <div className={cx("rounded-2xl border bg-white shadow-sm p-5", className)}>{children}</div>
);

export const Section = ({ title, kicker, right, children, className }) => (
    <Card className={className}>
        <div className="flex items-center justify-between">
            <div>
                {kicker && <div className="text-xs uppercase tracking-wider text-gray-400">{kicker}</div>}
                <h2 className="text-xl font-semibold">{title}</h2>
            </div>
            {right}
        </div>
        <div className="mt-4">{children}</div>
    </Card>
);

export const Button = ({ variant = "solid", className, ...props }) => {
    const base = "px-4 py-2 rounded-xl text-sm transition";
    const styles = {
        solid: "bg-black text-white hover:opacity-90",
        subtle: "bg-gray-100 hover:bg-gray-200",
        ghost: "hover:bg-gray-50",
    };
    return <button className={cx(base, styles[variant], className)} {...props} />;
};

/* =============== KPI Stat =============== */
export const KPIStat = ({ label, value, sub, icon: Icon = Gauge, tone = "default", spark = null }) => {
    const toneClass =
        tone === "warn" ? "text-amber-600" : tone === "good" ? "text-emerald-600" : "text-gray-900";
    return (
        <Card className="p-4">
            <div className="flex items-center gap-3">
                <div className={cx("rounded-xl p-2 bg-gray-100", tone === "good" && "bg-emerald-50", tone === "warn" && "bg-amber-50")}>
                    <Icon className={cx("w-5 h-5", toneClass)} />
                </div>
                <div className="min-w-0">
                    <div className="text-xs text-gray-500">{label}</div>
                    <div className="text-lg font-semibold leading-tight truncate">{value ?? "—"}</div>
                    {sub && <div className="text-xs text-gray-500">{sub}</div>}
                </div>
            </div>
            {spark}
        </Card>
    );
};

/* =============== Tiny Sparkline =============== */
export const Sparkline = ({ data, dataKey = "v", className }) => (
    <div className={cx("mt-2 h-10", className)}>
        <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data}>
                <Area type="monotone" dataKey={dataKey} strokeWidth={1} fillOpacity={0.15} />
            </AreaChart>
        </ResponsiveContainer>
    </div>
);

/* =============== Toasts =============== */
const ToastCtx = createContext(null);
export const useToast = () => useContext(ToastCtx) || { show: () => { } };

export const ToastProvider = ({ children }) => {
    const [items, setItems] = useState([]);
    const show = (msg, type = "info") =>
        setItems((xs) => [...xs, { id: crypto.randomUUID(), msg, type }]);

    useEffect(() => {
        if (!items.length) return;
        const id = setTimeout(() => setItems((xs) => xs.slice(1)), 3000);
        return () => clearTimeout(id);
    }, [items]);

    const iconFor = (t) =>
        t === "success" ? <CheckCircle2 className="w-4 h-4" /> :
            t === "warn" ? <AlertTriangle className="w-4 h-4" /> : <Info className="w-4 h-4" />;

    return (
        <ToastCtx.Provider value={{ show }}>
            {children}
            <div className="fixed bottom-4 right-4 space-y-2 z-50">
                {items.map((t) => (
                    <div key={t.id} className="flex items-center gap-2 rounded-xl bg-white border shadow px-3 py-2">
                        {iconFor(t.type)}
                        <div className="text-sm">{t.msg}</div>
                    </div>
                ))}
            </div>
        </ToastCtx.Provider>
    );
};

/* =============== Helpers =============== */
export const Trend = ({ label, value, delta }) => {
    const up = (delta ?? 0) >= 0;
    return (
        <div className="text-sm flex items-center gap-2">
            <span className="font-medium">{label}:</span>
            <span className="font-mono">{value}</span>
            <span className={cx("inline-flex items-center gap-1 text-xs", up ? "text-emerald-600" : "text-red-600")}>
                <TrendingUp className="w-3 h-3" />
                {up ? "+" : ""}
                {delta?.toFixed?.(2) ?? delta}
            </span>
        </div>
    );
};
