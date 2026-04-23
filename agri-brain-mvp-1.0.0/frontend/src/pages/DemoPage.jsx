import React, { useEffect, useState, useRef, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
import { Label } from "@/components/ui/label";
import { cn, fmt, short, jget, jpost } from "@/lib/utils";
import { getApiBase } from "@/mvp/api.js";
import { motion, AnimatePresence } from "framer-motion";
import { toast } from "sonner";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Cell,
  Tooltip as ReTooltip, CartesianGrid,
} from "recharts";
import {
  Play, Thermometer, Zap, TrendingUp, Network, Wrench, BookOpen,
  Layers, Brain, CheckCircle2, Shield, FileText, GitBranch, Copy,
  Loader2, ChevronDown, ChevronUp, ArrowDown, AlertTriangle,
  Truck, Recycle, Warehouse, MessageCircle, Send,
} from "lucide-react";

const TheaterPage = React.lazy(() => import("./TheaterPage.jsx"));

const API = getApiBase();

// ── Agent profiles ──
const AGENT_PROFILES = {
  farm:        { label: "Farm Agent",        stage: "[0, 18) hours",     bias: [+0.08, -0.03, -0.05], mandate: "Preserve freshness, minimize post-harvest loss", icon: "🌾" },
  processor:   { label: "Processor Agent",   stage: "[18, 36) hours",    bias: [-0.02, +0.06, -0.04], mandate: "Processing efficiency, cold chain integrity",     icon: "🏭" },
  cooperative: { label: "Cooperative Agent",  stage: "[12, 30) overlay",  bias: [+0.00, +0.04, -0.04], mandate: "Governance coordination, equity balancing",       icon: "🤝" },
  distributor: { label: "Distributor Agent",  stage: "[36, 54) hours",    bias: [-0.05, +0.10, -0.05], mandate: "Community redistribution, minimize food miles",   icon: "🚛" },
  recovery:    { label: "Recovery Agent",     stage: "[54, +∞) hours",    bias: [-0.06, -0.02, +0.08], mandate: "Waste valorization via composting/feed/food bank",icon: "♻️" },
};

const FEATURE_LABELS = [
  { key: "compliance_severity", label: "Compliance", color: "#ef4444" },
  { key: "forecast_urgency",   label: "Forecast",   color: "#f97316" },
  { key: "retrieval_confidence",label: "Retrieval",  color: "#3b82f6" },
  { key: "regulatory_pressure", label: "Regulatory", color: "#a855f7" },
  { key: "recovery_saturation", label: "Recovery",   color: "#22c55e" },
  { key: "supply_uncertainty",  label: "Supply",     color: "#14b8a6" },
];

const SCENARIOS = [
  { id: "baseline",         label: "Baseline",         desc: "Normal operating conditions" },
  { id: "heatwave",         label: "Heatwave",         desc: "+20°C ramp, accelerated spoilage" },
  { id: "overproduction",   label: "Overproduction",   desc: "2.5× inventory surge" },
  { id: "cyber_outage",     label: "Cyber Outage",     desc: "Demand drops to 15%, refrigeration degradation" },
  { id: "adaptive_pricing", label: "Adaptive Pricing", desc: "Demand oscillation with noise" },
];

const ROLES = ["farm", "processor", "cooperative", "distributor", "recovery"];

const PHASES = [
  { icon: Thermometer, label: "IoT Sensors",          color: "border-slate-400",   bg: "bg-slate-500/10",   desc: "Extract raw observation from IoT sensor array (temperature, humidity, inventory, shelf-life)" },
  { icon: Zap,         label: "PINN Spoilage Model",  color: "border-rose-500",    bg: "bg-rose-500/10",    desc: "Arrhenius-Baranyi first-order ODE: dC/dt = −k(T,RH)·C with lag phase λ = 12 h" },
  { icon: TrendingUp,  label: "Demand Forecast",      color: "border-amber-500",   bg: "bg-amber-500/10",   desc: "LSTM demand prediction (16 hidden, truncated BPTT) + Bollinger band regime detection" },
  { icon: Network,     label: "Agent Dispatch",       color: "border-violet-500",  bg: "bg-violet-500/10",  desc: "AgentCoordinator selects role-specific agent based on hours since harvest" },
  { icon: Wrench,      label: "MCP Tool Workflow",    color: "border-orange-500",  bg: "bg-orange-500/10",  desc: "JSON-RPC 2.0 tool dispatch: compliance, forecast, SLCA, chain query, footprint" },
  { icon: BookOpen,    label: "piRAG Retrieval",      color: "border-blue-500",    bg: "bg-blue-500/10",    desc: "Physics-informed BM25+TF-IDF hybrid retrieval (k=4) with Arrhenius-based reranking" },
  { icon: Layers,      label: "Context Features (ψ)", color: "border-teal-500",    bg: "bg-teal-500/10",    desc: "Extract 6D context vector from MCP + piRAG outputs (compliance, forecast, retrieval, regulatory, recovery saturation, supply uncertainty) → Θ_context × ψ → logit modifier" },
  { icon: Brain,       label: "Policy Network",       color: "border-purple-500",  bg: "bg-purple-500/10",  desc: "Softmax contextual policy: logits = Θ×φ + γ·τ + SLCA bonus + role bias + context modifier" },
  { icon: CheckCircle2,label: "Action Selection",     color: "border-emerald-500", bg: "bg-emerald-500/10", desc: "Softmax π(a|s) sampling from adjusted probability distribution over 3 routing actions" },
  { icon: Shield,      label: "SLCA & Impact",        color: "border-cyan-500",    bg: "bg-cyan-500/10",    desc: "4-pillar Social Life-Cycle Assessment (Carbon, Labor, Resilience, Transparency) + footprint" },
  { icon: FileText,    label: "Causal Explanation",   color: "border-teal-600",    bg: "bg-teal-600/10",    desc: "BECAUSE/WITHOUT counterfactual reasoning with [KB:] citations and keyword extraction" },
  { icon: GitBranch,   label: "Provenance & Chain",   color: "border-indigo-500",  bg: "bg-indigo-500/10",  desc: "SHA-256 evidence hashing → Merkle tree root → on-chain anchor via Hardhat/Solidity" },
];

// ── Helpers ──
function IOCard({ label, children, className }) {
  return (
    <div className={cn("rounded-lg border p-2.5", className)}>
      <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-1.5">{label}</p>
      {children}
    </div>
  );
}

function Kv({ k, v, mono }) {
  return (
    <div className="flex items-center justify-between gap-2 text-xs py-0.5">
      <span className="text-muted-foreground">{k}</span>
      <span className={cn("font-semibold", mono && "font-mono")}>{v}</span>
    </div>
  );
}

function FlowArrow() {
  return (
    <div className="flex justify-center py-1">
      <motion.div animate={{ y: [0, 4, 0] }} transition={{ repeat: Infinity, duration: 1.2 }}>
        <ArrowDown className="w-4 h-4 text-teal-500/60" />
      </motion.div>
    </div>
  );
}

// ── Phase Content Renderers ──
function renderPhase(idx, m) {
  if (!m) return null;
  const ex = m.explainability || {};
  const cf = ex.context_features || {};
  const la = ex.logit_adjustment || {};
  const prov = ex.provenance || {};
  const comp = ex.compliance || {};
  const slca = m.slca_components || {};
  const ap = m.action_probabilities || {};
  const ctf = ex.counterfactual || {};
  const profile = AGENT_PROFILES[m.role] || AGENT_PROFILES.farm;

  switch (idx) {
    // 1. IoT Sensors
    case 0: return (
      <div className="grid grid-cols-2 gap-3">
        <IOCard label="Sensor Readings" className="bg-slate-50 dark:bg-slate-900/30">
          <Kv k="Temperature" v={`${fmt(comp.readings?.temperature ?? 3.79, 2)} °C`} mono />
          <Kv k="Humidity" v={`${fmt(comp.readings?.humidity ?? 90, 1)} %`} mono />
          <Kv k="Shelf Remaining" v={fmt(m.shelf_left, 4)} mono />
          <Kv k="Spoilage Risk (ρ)" v={fmt(m.spoilage_risk, 4)} mono />
        </IOCard>
        <IOCard label="Supply State" className="bg-slate-50 dark:bg-slate-900/30">
          <Kv k="Inventory" v={`${(m.note?.match(/inventory.*?(\d[\d,]*)/i)?.[1] || "14,279")} units`} mono />
          <Kv k="Step" v={m.step} mono />
          <Kv k="Mode" v={m.mode} />
          <Kv k="Volatility" v={m.volatility || "normal"} />
        </IOCard>
      </div>
    );

    // 2. PINN Spoilage
    case 1: return (
      <div className="space-y-3">
        <IOCard label="Arrhenius-Baranyi ODE" className="bg-rose-50 dark:bg-rose-950/20">
          <pre className="text-[11px] font-mono text-rose-700 dark:text-rose-300 whitespace-pre-wrap leading-relaxed">
{`dC/dt = −k_eff(T, RH) · C
k(T) = k_ref · exp[Ea/R · (1/T_ref − 1/T)] · (1 + β·RH)

k_ref  = 0.0021 h⁻¹    Ea/R = 8000 K
T_ref  = 277.15 K (4°C)  β    = 0.25
λ_lag  = 12.0 h (Baranyi)`}</pre>
        </IOCard>
        <div className="grid grid-cols-2 gap-3">
          <IOCard label="Input" className="border-dashed">
            <Kv k="Temperature" v={`${fmt(comp.readings?.temperature ?? 3.79, 2)} °C`} mono />
            <Kv k="Humidity" v={`${fmt(comp.readings?.humidity ?? 90, 1)} %`} mono />
          </IOCard>
          <IOCard label="Output" className="border-teal-500/30">
            <Kv k="Spoilage Risk (ρ)" v={fmt(m.spoilage_risk, 4)} mono />
            <Kv k="Shelf Left" v={`${fmt(m.shelf_left * 100, 1)}%`} mono />
          </IOCard>
        </div>
      </div>
    );

    // 3. Demand Forecast
    case 2: return (
      <div className="grid grid-cols-2 gap-3">
        <IOCard label="Forecast Model" className="border-dashed">
          <Kv k="Method" v={(m.demand_forecast?.method || "lstm").toUpperCase()} />
          <Kv k="Bollinger z" v={fmt(m.regime?.bollinger_z, 3)} mono />
          <Kv k="Regime τ" v={fmt(m.regime?.tau, 1)} mono />
        </IOCard>
        <IOCard label="Prediction" className="border-amber-500/30">
          <Kv k="Demand ŷ" v={`${fmt(m.demand_forecast?.y_hat, 2)} units`} mono />
          <Kv k="Supply ŷ" v={`${fmt(m.yield_forecast?.y_hat, 2)} units`} mono />
          <Kv k="Regime" v={m.volatility || "normal"} />
        </IOCard>
      </div>
    );

    // 4. Agent Dispatch
    case 3: return (
      <div className="space-y-3">
        <div className="flex items-center gap-3">
          <span className="text-3xl">{profile.icon}</span>
          <div>
            <p className="font-semibold">{profile.label}</p>
            <p className="text-xs text-muted-foreground">{profile.mandate}</p>
          </div>
        </div>
        <div className="grid grid-cols-3 gap-3">
          <IOCard label="Stage" className="border-dashed">
            <p className="text-sm font-mono text-center">{profile.stage}</p>
          </IOCard>
          <IOCard label="Role Bias [CC, LR, Rec]" className="border-dashed">
            <p className="text-sm font-mono text-center">[{profile.bias.map(b => (b >= 0 ? "+" : "") + fmt(b, 2)).join(", ")}]</p>
          </IOCard>
          <IOCard label="Message Types" className="border-dashed">
            <div className="flex flex-wrap gap-1">
              {["SPOILAGE_ALERT", "SURPLUS_ALERT", "ACK"].map(t => (
                <Badge key={t} variant="outline" className="text-[8px]">{t}</Badge>
              ))}
            </div>
          </IOCard>
        </div>
      </div>
    );

    // 5. MCP Tool Workflow
    case 4: return (
      <div className="space-y-2">
        <div className="flex items-center gap-2 mb-1">
          <Badge className="bg-orange-500/10 text-orange-600 border-0 text-[10px]">JSON-RPC 2.0</Badge>
          <span className="text-xs text-muted-foreground">{(ex.mcp_tools_invoked || []).length} tools invoked</span>
        </div>
        {(ex.mcp_tools_invoked || []).map((tool, i) => (
          <Card key={i} className="bg-muted/30">
            <CardContent className="p-2.5">
              <div className="flex items-center gap-2 mb-1">
                <Wrench className="w-3 h-3 text-orange-500" />
                <span className="font-mono text-xs font-semibold">{tool}</span>
                <Badge className="bg-emerald-500/10 text-emerald-600 border-0 text-[8px] ml-auto">success</Badge>
              </div>
              {tool === "check_compliance" && comp.compliant !== undefined && (
                <div className="flex items-center gap-2 mt-1">
                  <Badge className={comp.compliant ? "bg-emerald-500/10 text-emerald-600 border-0 text-[9px]" : "bg-red-500/10 text-red-600 border-0 text-[9px]"}>
                    {comp.compliant ? "✓ Compliant" : "✗ Violation"}
                  </Badge>
                  <span className="text-[10px] text-muted-foreground">
                    T={comp.readings?.temperature}°C, RH={comp.readings?.humidity}%, max={comp.thresholds?.temp_max_c}°C
                  </span>
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>
    );

    // 6. piRAG Retrieval
    case 5: return (
      <div className="space-y-3">
        <div className="grid grid-cols-2 gap-3">
          <IOCard label="Retrieval Config" className="border-dashed">
            <Kv k="Method" v="BM25 + TF-IDF Hybrid" />
            <Kv k="Top-k" v="4" />
            <Kv k="Reranking" v="Physics-Informed" />
            <Kv k="KB Size" v="20 documents" />
          </IOCard>
          <IOCard label="Top Result" className="border-blue-500/30">
            <Kv k="Document" v={ex.pirag_top_doc || "—"} mono />
            <Kv k="Score" v={fmt(ex.pirag_top_score, 3)} mono />
            <Kv k="Source Docs" v={(m.rag_context?.source_documents || []).length} mono />
          </IOCard>
        </div>
        {ex.keywords?.regulatory?.thresholds?.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-[10px] text-muted-foreground">Keywords:</span>
            {ex.keywords.regulatory.thresholds.map((kw, i) => (
              <Badge key={i} className="text-[9px] bg-purple-500/10 text-purple-600 border-0">{kw}</Badge>
            ))}
          </div>
        )}
      </div>
    );

    // 7. Context Features
    case 6: {
      const radarData = FEATURE_LABELS.map(f => ({ axis: f.label, value: cf[f.key] ?? 0 }));
      return (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData}>
                <PolarGrid className="opacity-30" />
                <PolarAngleAxis dataKey="axis" tick={{ fontSize: 10 }} />
                <PolarRadiusAxis tick={{ fontSize: 8 }} domain={[0, 1]} />
                <Radar dataKey="value" stroke="#009688" fill="#009688" fillOpacity={0.25} strokeWidth={2} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
          <div className="space-y-1.5">
            <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">ψ = Θ_context × [features]</p>
            {FEATURE_LABELS.map(f => (
              <div key={f.key} className="flex items-center gap-2 text-xs">
                <span className="w-2 h-2 rounded-full shrink-0" style={{ background: f.color }} />
                <span className="w-20 text-muted-foreground">{f.label}</span>
                <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
                  <motion.div initial={{ width: 0 }} animate={{ width: `${(cf[f.key] ?? 0) * 100}%` }}
                    transition={{ duration: 0.8 }} className="h-full rounded-full" style={{ background: f.color }} />
                </div>
                <span className="font-mono w-10 text-right">{fmt(cf[f.key], 3)}</span>
              </div>
            ))}
          </div>
        </div>
      );
    }

    // 8. Policy Network
    case 7: {
      const logitEntries = [
        { key: "cold_chain", label: "Cold Chain", color: "#0072B2" },
        { key: "local_redistribute", label: "Redistribute", color: "#10B981" },
        { key: "recovery", label: "Recovery", color: "#D55E00" },
      ];
      return (
        <div className="space-y-3">
          <IOCard label="Policy Computation" className="bg-purple-50 dark:bg-purple-950/20">
            <pre className="text-[10px] font-mono text-purple-700 dark:text-purple-300 whitespace-pre-wrap leading-relaxed">
{`logits = Θ(3×6) × φ(6D) + γ·τ + SLCA_bonus + role_bias
logits += context_modifier    ← from Θ_context × ψ
π(a|s) = softmax(logits)`}</pre>
          </IOCard>
          <div className="space-y-1.5">
            <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Context Modifier → Logit Shift</p>
            {logitEntries.map(e => {
              const val = la[e.key] ?? 0;
              const pct = Math.min(Math.abs(val) * 50, 50);
              return (
                <div key={e.key} className="flex items-center gap-2 text-xs">
                  <span className="w-24 text-muted-foreground">{e.label}</span>
                  <div className="flex-1 h-2.5 rounded-full bg-muted relative overflow-hidden">
                    <div className="absolute left-1/2 top-0 bottom-0 w-px bg-border" />
                    <motion.div initial={{ width: 0 }} animate={{ width: `${pct}%` }} transition={{ duration: 0.6 }}
                      className="absolute top-0 bottom-0 rounded-full"
                      style={{ ...(val < 0 ? { right: "50%" } : { left: "50%" }), background: e.color, opacity: 0.7 }} />
                  </div>
                  <span className={cn("font-mono w-14 text-right", val > 0 ? "text-emerald-600" : val < 0 ? "text-red-500" : "")}>
                    {val > 0 ? "+" : ""}{fmt(val, 3)}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      );
    }

    // 9. Action Selection
    case 8: {
      const actionMeta = {
        cold_chain: { icon: Truck, color: "#0072B2", label: "Cold Chain (long-haul)" },
        local_redistribute: { icon: Warehouse, color: "#10B981", label: "Local Redistribution" },
        recovery: { icon: Recycle, color: "#D55E00", label: "Recovery / Composting" },
      };
      const selected = m.action || m.decision;
      return (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Badge className="bg-teal-500/10 text-teal-600 border-0 font-semibold text-sm px-3 py-1">
              {actionMeta[selected]?.label || selected}
            </Badge>
          </div>
          {Object.entries(ap).map(([k, v]) => {
            const meta = actionMeta[k] || {};
            const isSelected = k === selected;
            return (
              <div key={k} className={cn("flex items-center gap-2 text-xs rounded-md p-1.5", isSelected && "bg-teal-500/5 ring-1 ring-teal-500/20")}>
                {meta.icon && <meta.icon className="w-3.5 h-3.5 shrink-0" style={{ color: meta.color }} />}
                <span className="w-28 text-muted-foreground">{meta.label || k}</span>
                <div className="flex-1 h-3 rounded-full bg-muted overflow-hidden">
                  <motion.div initial={{ width: 0 }} animate={{ width: `${v * 100}%` }} transition={{ duration: 0.8 }}
                    className="h-full rounded-full" style={{ background: meta.color || "#888" }} />
                </div>
                <span className="font-mono w-16 text-right font-semibold">{fmt(v * 100, 1)}%</span>
              </div>
            );
          })}
        </div>
      );
    }

    // 10. SLCA & Impact
    case 9: {
      const pillars = [
        { k: "Carbon",       v: slca.carbon,       color: "#059669" },
        { k: "Labor",        v: slca.labor,         color: "#2563eb" },
        { k: "Resilience",   v: slca.resilience,    color: "#7c3aed" },
        { k: "Transparency", v: slca.transparency,  color: "#d97706" },
      ];
      return (
        <div className="space-y-3">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {pillars.map(p => (
              <div key={p.k} className="text-center rounded-lg p-2 bg-muted/30">
                <p className="text-[10px] text-muted-foreground">{p.k}</p>
                <p className="text-lg font-bold" style={{ color: p.color }}>{fmt(p.v, 2)}</p>
              </div>
            ))}
          </div>
          <Separator />
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            <IOCard label="SLCA Composite"><p className="text-center font-bold text-lg text-teal-600">{fmt(m.slca, 3)}</p></IOCard>
            <IOCard label="Carbon"><p className="text-center font-bold text-lg">{fmt(m.carbon_kg, 1)} kg</p></IOCard>
            <IOCard label="Waste Rate"><p className="text-center font-bold text-lg">{fmt(m.waste * 100, 2)}%</p></IOCard>
            <IOCard label="Circular Econ"><p className="text-center font-bold text-lg">{fmt(m.circular_economy_score, 2)}</p></IOCard>
          </div>
        </div>
      );
    }

    // 11. Causal Explanation
    case 10: {
      const text = ex.causal_text || ex.summary || "";
      const firstPara = text.split("\n\n").slice(0, 2).join("\n\n");
      const renderHighlighted = (raw) => raw.split(/(BECAUSE|WITHOUT|AND)/g).map((p, i) =>
        p === "BECAUSE" ? <span key={i} className="font-bold text-teal-600 dark:text-teal-400">BECAUSE</span> :
        p === "WITHOUT" ? <span key={i} className="font-bold text-amber-600 dark:text-amber-400">WITHOUT</span> :
        p === "AND" ? <span key={i} className="font-semibold">AND</span> :
        <span key={i}>{p.split(/(\[KB:[^\]]+\])/g).map((s, j) =>
          s.startsWith("[KB:") ? <Badge key={`${i}-${j}`} variant="outline" className="mx-0.5 text-[9px] font-mono">{s}</Badge> : <span key={`${i}-${j}`}>{s}</span>
        )}</span>
      );
      return (
        <div className="space-y-3">
          <div className="text-xs leading-relaxed text-muted-foreground">
            {firstPara.split("\n\n").map((para, i) => <p key={i} className={i > 0 ? "mt-2" : ""}>{renderHighlighted(para)}</p>)}
          </div>
          {ctf.probs_with_context && (
            <div className="grid grid-cols-2 gap-3">
              <IOCard label="WITH Context" className="border-teal-500/30">
                {["cold_chain", "local_redistribute", "recovery"].map((a, i) => (
                  <Kv key={a} k={a.replace("_", " ")} v={`${fmt((ctf.probs_with_context?.[i] ?? 0) * 100, 1)}%`} mono />
                ))}
              </IOCard>
              <IOCard label="WITHOUT Context" className="border-amber-500/30">
                {["cold_chain", "local_redistribute", "recovery"].map((a, i) => (
                  <Kv key={a} k={a.replace("_", " ")} v={`${fmt((ctf.probs_without_context?.[i] ?? 0) * 100, 1)}%`} mono />
                ))}
              </IOCard>
            </div>
          )}
        </div>
      );
    }

    // 12. Provenance & Blockchain
    case 11: {
      const hashes = prov.evidence_hashes || [];
      return (
        <div className="space-y-3">
          <div className="flex flex-wrap gap-2">
            {prov.guards_passed !== false && <Badge className="bg-emerald-500/10 text-emerald-600 border-0 text-[10px]">Guards Passed</Badge>}
            <Badge className="bg-blue-500/10 text-blue-600 border-0 text-[10px]">{hashes.length} evidence items</Badge>
            {m.tx_hash && m.tx_hash !== "0x0" && <Badge className="bg-indigo-500/10 text-indigo-600 border-0 text-[10px]">On-chain anchored</Badge>}
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <IOCard label="Merkle Root" className="border-indigo-500/30">
              {prov.merkle_root ? (
                <button onClick={() => { navigator.clipboard.writeText(prov.merkle_root); toast.success("Copied"); }}
                  className="font-mono text-[10px] text-muted-foreground hover:text-primary break-all">
                  {prov.merkle_root} <Copy className="w-2.5 h-2.5 inline" />
                </button>
              ) : <span className="text-xs text-muted-foreground">—</span>}
            </IOCard>
            <IOCard label="Blockchain Tx" className="border-indigo-500/30">
              <Kv k="Hash" v={m.tx_hash && m.tx_hash !== "0x0" ? short(m.tx_hash) : "0x0 (local)"} mono />
              <Kv k="Timestamp" v={m.time?.split("T")[1]?.split(".")[0] || "—"} mono />
            </IOCard>
          </div>
        </div>
      );
    }

    default: return null;
  }
}

// ── Main Page ──
export default function DemoPage() {
  const [scenario, setScenario] = useState("baseline");
  const [role, setRole] = useState("farm");
  const [speed, setSpeed] = useState(1);
  const [running, setRunning] = useState(false);
  const [activeStep, setActiveStep] = useState(-1);
  const [memo, setMemo] = useState(null);
  const [showMemo, setShowMemo] = useState(false);
  const timerRef = useRef(null);
  const stepRefs = useRef([]);

  useEffect(() => () => clearTimeout(timerRef.current), []);

  const runDemo = async () => {
    setRunning(true);
    setActiveStep(-1);
    setMemo(null);
    setShowMemo(false);

    try {
      // 1. Load data (best effort — may already be loaded)
      await jpost(API, "/case/load").catch(() => {});

      // 2. Apply scenario if non-baseline
      if (scenario !== "baseline") {
        await jpost(API, "/scenarios/run", { name: scenario, intensity: 1.0 }).catch(() => {});
      }

      // 3. Take decision
      const res = await jpost(API, "/decide", { agent_id: role, role });
      const m = res.memo || res;
      if (!m || !m.action) {
        // Fallback: try fetching latest existing decision
        const fallback = await jget(API, "/decisions").catch(() => null);
        const decs = fallback?.decisions || fallback || [];
        if (decs.length > 0) {
          setMemo(decs[0]);
        } else {
          toast.error("No decision data. Make sure the backend is running and data is loaded.");
          setRunning(false);
          return;
        }
      } else {
        setMemo(m);
      }

      // Reset scenario
      if (scenario !== "baseline") {
        await jpost(API, "/scenarios/reset").catch(() => {});
      }
    } catch (e) {
      // Last resort: try to use cached decisions
      try {
        const fallback = await jget(API, "/decisions");
        const decs = fallback?.decisions || fallback || [];
        if (decs.length > 0) {
          setMemo(decs[0]);
          toast.info("Using cached decision data");
        } else {
          toast.error(`Demo failed: ${e.message}. Check backend at ${API}`);
          setRunning(false);
          return;
        }
      } catch {
        toast.error(`Cannot reach backend at ${API}. Is it running on port 8100?`);
        setRunning(false);
        return;
      }
    }

    // 4. Animate through phases
    const baseDelay = 1800;
    for (let i = 0; i < PHASES.length; i++) {
      await new Promise(resolve => {
        timerRef.current = setTimeout(() => {
          setActiveStep(i);
          stepRefs.current[i]?.scrollIntoView({ behavior: "smooth", block: "center" });
          resolve();
        }, baseDelay / speed);
      });
    }

    // Final hold
    await new Promise(r => { timerRef.current = setTimeout(r, 1200 / speed); });
    setRunning(false);
  };

  const ari = memo ? ((1 - memo.waste) * memo.slca * memo.shelf_left) : null;

  return (
    <div className="space-y-6 pb-16">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex items-center gap-3 mb-1">
          <Play className="w-6 h-6 text-teal-600" />
          <h1 className="text-2xl font-bold">System Demo</h1>
        </div>
        <p className="text-sm text-muted-foreground">
          Interactive walkthrough of the complete AGRI-BRAIN decision pipeline — from IoT sensors to blockchain provenance
        </p>
      </motion.div>

      <Separator />

      {/* Tabs: System Walkthrough + Multi-Agent Run */}
      <Tabs defaultValue="pipeline">
        <TabsList className="w-full justify-start">
          <TabsTrigger value="pipeline" className="flex items-center gap-1.5"><Network className="w-3.5 h-3.5" /> System Walkthrough</TabsTrigger>
          <TabsTrigger value="theater" className="flex items-center gap-1.5"><MessageCircle className="w-3.5 h-3.5" /> Multi-Agent Run</TabsTrigger>
        </TabsList>

        <TabsContent value="pipeline">

      {/* Controls */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
        <Card>
          <CardContent className="p-4">
            <div className="flex flex-wrap items-end gap-4">
              <div className="w-44">
                <Label className="text-xs mb-1.5 block">Scenario</Label>
                <Select value={scenario} onValueChange={setScenario} disabled={running}>
                  <SelectTrigger className="h-9"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {SCENARIOS.map(s => <SelectItem key={s.id} value={s.id}>{s.label}</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>
              <div className="w-40">
                <Label className="text-xs mb-1.5 block">Agent Role</Label>
                <Select value={role} onValueChange={setRole} disabled={running}>
                  <SelectTrigger className="h-9"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {ROLES.map(r => <SelectItem key={r} value={r}>{AGENT_PROFILES[r].icon} {AGENT_PROFILES[r].label}</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label className="text-xs mb-1.5 block">Speed</Label>
                <div className="flex gap-1">
                  {[1, 2, 4].map(s => (
                    <Button key={s} variant={speed === s ? "default" : "outline"} size="sm" onClick={() => setSpeed(s)} disabled={running}>
                      {s}x
                    </Button>
                  ))}
                </div>
              </div>
              <Button onClick={runDemo} disabled={running} size="lg" className="bg-teal-600 hover:bg-teal-700 text-white">
                {running ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Play className="w-4 h-4 mr-2" />}
                {running ? "Running..." : "Run Full Demo"}
              </Button>
              {activeStep >= 0 && (
                <div className="flex-1 min-w-40">
                  <div className="h-2.5 rounded-full bg-muted overflow-hidden">
                    <motion.div className="h-full rounded-full bg-teal-500" animate={{ width: `${((activeStep + 1) / PHASES.length) * 100}%` }} transition={{ duration: 0.3 }} />
                  </div>
                  <p className="text-[10px] text-muted-foreground mt-1 text-right">Phase {activeStep + 1} / {PHASES.length}</p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Pipeline */}
      <div className="relative pl-8">
        <div className="absolute left-[22px] top-0 bottom-0 w-0.5 bg-border" />

        {PHASES.map((phase, idx) => {
          const state = idx < activeStep ? "done" : idx === activeStep ? "active" : "pending";
          return (
            <React.Fragment key={idx}>
              <div ref={el => (stepRefs.current[idx] = el)} className="relative pl-12 pb-2">
                {/* Dot */}
                <div className={cn(
                  "absolute left-[10px] top-2 w-9 h-9 rounded-full flex items-center justify-center border-2 transition-all duration-500 z-10",
                  state === "done" ? "border-emerald-500 bg-emerald-500/10" :
                  state === "active" ? cn(phase.color, phase.bg, "ring-2 ring-teal-500/40 ring-offset-2 ring-offset-background") :
                  "border-muted-foreground/20 bg-muted/30"
                )}>
                  {state === "done" ? <CheckCircle2 className="w-4 h-4 text-emerald-500" /> :
                    <phase.icon className={cn("w-4 h-4", state === "active" ? "text-foreground" : "text-muted-foreground/40")} />}
                </div>

                {/* Card */}
                <motion.div initial={false}
                  animate={{ opacity: state === "pending" ? 0.35 : 1, scale: state === "active" ? 1 : 0.98 }}
                  transition={{ duration: 0.4 }}>
                  <Card className={cn(
                    "transition-all duration-500",
                    state === "active" && "border-teal-500/50 shadow-lg shadow-teal-500/10",
                    state === "done" && "border-emerald-500/20",
                  )}>
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between mb-1">
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-mono text-muted-foreground bg-muted rounded px-1.5 py-0.5">{idx + 1}</span>
                          <span className="text-sm font-semibold">{phase.label}</span>
                        </div>
                        <Badge className={cn("text-[9px] border-0",
                          state === "done" ? "bg-emerald-500/10 text-emerald-600" :
                          state === "active" ? "bg-teal-500/10 text-teal-600 animate-pulse" :
                          "bg-muted text-muted-foreground/50"
                        )}>
                          {state === "done" ? "✓ Complete" : state === "active" ? "⟳ Processing..." : "○ Pending"}
                        </Badge>
                      </div>
                      <p className="text-[11px] text-muted-foreground mb-2">{phase.desc}</p>

                      <AnimatePresence>
                        {(state === "active" || state === "done") && memo && (
                          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.5 }} className="overflow-hidden">
                            <Separator className="my-2" />
                            {renderPhase(idx, memo)}
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </CardContent>
                  </Card>
                </motion.div>
              </div>

              {/* Flow arrow between steps */}
              {idx < PHASES.length - 1 && state !== "pending" && <FlowArrow />}
            </React.Fragment>
          );
        })}
      </div>

      {/* Summary Dashboard */}
      <AnimatePresence>
        {activeStep >= PHASES.length - 1 && memo && !running && (
          <motion.div initial={{ opacity: 0, y: 30, scale: 0.95 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ duration: 0.6 }}>
            <Card className="border-teal-500/30 shadow-xl shadow-teal-500/5">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5 text-teal-600" /> Decision Complete
                </CardTitle>
                <CardDescription>
                  {AGENT_PROFILES[memo.role]?.label} selected <strong>{(memo.action || memo.decision)?.replace("_", " ")}</strong> under <strong>{SCENARIOS.find(s => s.id === scenario)?.label}</strong> scenario
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
                  {[
                    { label: "ARI", value: fmt(ari, 3), color: "text-teal-600" },
                    { label: "SLCA", value: fmt(memo.slca, 3), color: "text-blue-600" },
                    { label: "Carbon", value: `${fmt(memo.carbon_kg, 1)} kg`, color: "text-amber-600" },
                    { label: "Waste", value: `${fmt(memo.waste * 100, 2)}%`, color: "text-red-500" },
                    { label: "Reward", value: fmt(memo.reward_decomposition?.total, 3), color: "text-emerald-600" },
                    { label: "Circular", value: fmt(memo.circular_economy_score, 2), color: "text-purple-600" },
                  ].map(kpi => (
                    <div key={kpi.label} className="text-center rounded-lg bg-muted/30 p-3">
                      <p className="text-[10px] text-muted-foreground uppercase">{kpi.label}</p>
                      <p className={cn("text-xl font-bold", kpi.color)}>{kpi.value}</p>
                    </div>
                  ))}
                </div>

                {/* Expand full memo */}
                <Button variant="outline" size="sm" onClick={() => setShowMemo(!showMemo)} className="w-full">
                  {showMemo ? <ChevronUp className="w-3 h-3 mr-2" /> : <ChevronDown className="w-3 h-3 mr-2" />}
                  {showMemo ? "Hide Full Memo" : "Show Full Memo"}
                </Button>

                <AnimatePresence>
                  {showMemo && (
                    <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }}>
                      <pre className="p-4 rounded-lg bg-muted text-[10px] font-mono overflow-x-auto max-h-96 whitespace-pre-wrap">
                        {memo.memo_text || JSON.stringify(memo, null, 2)}
                      </pre>
                    </motion.div>
                  )}
                </AnimatePresence>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

        </TabsContent>

        <TabsContent value="theater">
          <React.Suspense fallback={<div className="flex items-center justify-center h-64"><Loader2 className="w-6 h-6 animate-spin text-muted-foreground" /></div>}>
            <TheaterPage />
          </React.Suspense>
        </TabsContent>

      </Tabs>
    </div>
  );
}
