import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { cn, fmt, short } from "@/lib/utils";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer,
} from "recharts";
import {
  Shield, BookOpen, AlertTriangle, CheckCircle2, Copy, Brain,
  Hash, GitBranch,
} from "lucide-react";
import { toast } from "sonner";

const FEATURE_LABELS = [
  { key: "compliance_severity", label: "Compliance", color: "#ef4444" },
  { key: "forecast_urgency", label: "Forecast", color: "#f97316" },
  { key: "retrieval_confidence", label: "Retrieval", color: "#3b82f6" },
  { key: "regulatory_pressure", label: "Regulatory", color: "#a855f7" },
  { key: "recovery_saturation", label: "Recovery", color: "#22c55e" },
];

// --- Section 1a: Causal Explanation ---
function CausalExplanation({ explainability }) {
  const text = explainability.causal_text || explainability.summary || "";
  const primaryCause = explainability.causal_chain?.primary_cause;

  if (!text) {
    // Fail loud, not silent: a missing causal narrative is unusual and
    // the empty field should be visible rather than the section
    // vanishing silently.
    return (
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <Brain className="w-4 h-4 text-teal-600" />
          <h4 className="text-sm font-semibold">Causal Explanation</h4>
          <Badge variant="outline" className="text-[10px]">unavailable</Badge>
        </div>
        <div className="text-xs text-muted-foreground pl-6">
          The backend did not emit a causal narrative for this decision (typical for
          context-disabled modes such as <code>static</code>, <code>hybrid_rl</code>,
          <code>no_pinn</code>, or <code>no_slca</code>). The other panel sections still
          show whatever evidence is available.
        </div>
      </div>
    );
  }

  const renderText = (raw) => {
    const parts = raw.split(/(BECAUSE|WITHOUT|AND)/g);
    return parts.map((part, i) => {
      if (part === "BECAUSE") return <span key={i} className="font-bold text-teal-600 dark:text-teal-400">BECAUSE</span>;
      if (part === "WITHOUT") return <span key={i} className="font-bold text-amber-600 dark:text-amber-400">WITHOUT</span>;
      if (part === "AND") return <span key={i} className="font-semibold">AND</span>;
      // Highlight [KB: ...] citations
      const withCites = part.split(/(\[KB:[^\]]+\])/g);
      return withCites.map((seg, j) =>
        seg.startsWith("[KB:") ? (
          <Badge key={`${i}-${j}`} variant="outline" className="mx-0.5 text-[10px] font-mono">
            {seg}
          </Badge>
        ) : (
          <span key={`${i}-${j}`}>{seg}</span>
        )
      );
    });
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <Brain className="w-4 h-4 text-teal-600" />
        <h4 className="text-sm font-semibold">Causal Explanation</h4>
        {primaryCause && (
          <Badge className="bg-teal-500/10 text-teal-600 dark:text-teal-400 border-0 text-[10px]">
            Primary: {primaryCause}
          </Badge>
        )}
      </div>
      <div className="text-sm text-muted-foreground leading-relaxed pl-6">
        {text.split("\n\n").map((para, i) => (
          <p key={i} className={i > 0 ? "mt-2" : ""}>{renderText(para)}</p>
        ))}
      </div>
    </div>
  );
}

// --- Section 1b: Context Features Radar Chart + Logit Bars ---
function ContextRadar({ explainability }) {
  const cf = explainability.context_features;
  const la = explainability.logit_adjustment;
  if (!cf) {
    return (
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <Brain className="w-4 h-4 text-blue-600" />
          <h4 className="text-sm font-semibold">Context Features (ψ)</h4>
          <Badge variant="outline" className="text-[10px]">unavailable</Badge>
        </div>
        <div className="text-xs text-muted-foreground pl-6">
          No ψ vector was attached to this decision. This is expected for
          context-disabled modes (no_context, hybrid_rl, static, no_pinn, no_slca).
        </div>
      </div>
    );
  }

  const radarData = FEATURE_LABELS.map((f) => ({
    axis: f.label,
    value: cf[f.key] ?? 0,
  }));

  const logitEntries = [
    { key: "cold_chain", label: "ColdChain", color: "#0072B2" },
    { key: "local_redistribute", label: "Redistribute", color: "#10B981" },
    { key: "recovery", label: "Recovery", color: "#D55E00" },
  ];

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Shield className="w-4 h-4 text-blue-600" />
        <h4 className="text-sm font-semibold">Context Features</h4>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Radar chart */}
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart data={radarData}>
              <PolarGrid className="opacity-30" />
              <PolarAngleAxis dataKey="axis" tick={{ fontSize: 10 }} />
              <PolarRadiusAxis tick={{ fontSize: 8 }} domain={[0, 1]} />
              <Radar
                name="Context"
                dataKey="value"
                stroke="#009688"
                fill="#009688"
                fillOpacity={0.2}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Feature values + logit adjustment bars */}
        <div className="space-y-3">
          {/* Feature values */}
          <div className="space-y-1">
            {FEATURE_LABELS.map((f) => (
              <div key={f.key} className="flex items-center gap-2 text-xs">
                <span className="w-2 h-2 rounded-full shrink-0" style={{ background: f.color }} />
                <span className="w-20 text-muted-foreground">{f.label}</span>
                <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all"
                    style={{ width: `${(cf[f.key] ?? 0) * 100}%`, background: f.color }}
                  />
                </div>
                <span className="font-mono w-8 text-right">{fmt(cf[f.key], 2)}</span>
              </div>
            ))}
          </div>

          {/* Logit adjustment */}
          {la && (
            <div className="space-y-1 pt-2 border-t">
              <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Logit Adjustment</p>
              {logitEntries.map((e) => {
                const val = la[e.key] ?? 0;
                const pct = Math.min(Math.abs(val) * 50, 50);
                return (
                  <div key={e.key} className="flex items-center gap-2 text-xs">
                    <span className="w-20 text-muted-foreground">{e.label}</span>
                    <div className="flex-1 h-2 rounded-full bg-muted relative overflow-hidden">
                      <div className="absolute left-1/2 top-0 bottom-0 w-px bg-border" />
                      {val < 0 ? (
                        <div
                          className="absolute top-0 bottom-0 rounded-l-full"
                          style={{ right: "50%", width: `${pct}%`, background: e.color, opacity: 0.7 }}
                        />
                      ) : (
                        <div
                          className="absolute top-0 bottom-0 rounded-r-full"
                          style={{ left: "50%", width: `${pct}%`, background: e.color, opacity: 0.7 }}
                        />
                      )}
                    </div>
                    <span className={cn("font-mono w-12 text-right", val > 0 ? "text-emerald-600" : val < 0 ? "text-red-500" : "")}>
                      {val > 0 ? "+" : ""}{fmt(val, 2)}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// --- Section 1c: Keywords Panel ---
function KeywordsPanel({ keywords }) {
  if (!keywords || Object.keys(keywords).length === 0) return null;

  const categories = [
    { key: "thresholds", field: "thresholds", label: "Thresholds", cls: "bg-blue-500/10 text-blue-600 dark:text-blue-400 border-0" },
    { key: "regulations", field: "regulations", label: "Regulations", cls: "bg-purple-500/10 text-purple-600 dark:text-purple-400 border-0" },
    { key: "required_actions", field: "required_actions", label: "Actions", cls: "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-0" },
  ];

  // Keywords can be nested: keywords.regulatory.thresholds, keywords.sop.thresholds, etc.
  // Flatten them by category
  const flattened = { thresholds: [], regulations: [], required_actions: [] };
  for (const [, data] of Object.entries(keywords)) {
    if (typeof data === "object" && data !== null) {
      for (const cat of categories) {
        const items = data[cat.field] || [];
        for (const item of items) {
          if (!flattened[cat.field].includes(item)) {
            flattened[cat.field].push(item);
          }
        }
      }
    }
  }

  const hasAny = Object.values(flattened).some((arr) => arr.length > 0);
  if (!hasAny) return null;

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <BookOpen className="w-4 h-4 text-purple-600" />
        <h4 className="text-sm font-semibold">Extracted Keywords</h4>
      </div>
      <div className="pl-6 space-y-2">
        {categories.map((cat) => {
          const items = flattened[cat.field];
          if (items.length === 0) return null;
          return (
            <div key={cat.key} className="flex flex-wrap items-center gap-1.5">
              <span className="text-xs text-muted-foreground w-20 shrink-0">{cat.label}:</span>
              {items.slice(0, 8).map((item, i) => (
                <Badge key={i} className={cn("text-[10px]", cat.cls)}>{item}</Badge>
              ))}
              {items.length > 8 && (
                <span className="text-[10px] text-muted-foreground">+{items.length - 8} more</span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// --- Section 1d: Provenance Chain ---
function ProvenanceChain({ explainability, memo }) {
  const prov = explainability.provenance;
  const toolsInvoked = explainability.mcp_tools_invoked || [];
  const compliance = explainability.compliance;
  const forecast = explainability.forecast;
  const topDoc = explainability.pirag_top_doc;
  const topScore = explainability.pirag_top_score;

  const steps = [];

  // MCP tool steps
  if (compliance && typeof compliance === "object") {
    const status = compliance.compliant ? "compliant" : "violation";
    const severity = compliance.violations?.[0]?.severity || "unknown";
    steps.push({
      icon: Shield,
      iconColor: compliance.compliant ? "text-emerald-500" : "text-red-500",
      label: "MCP: check_compliance",
      detail: `${status}${!compliance.compliant ? `, severity=${severity}` : ""}`,
      hash: prov?.evidence_hashes?.[0],
    });
  }

  if (forecast && typeof forecast === "object" && forecast.urgency) {
    steps.push({
      icon: AlertTriangle,
      iconColor: forecast.urgency === "critical" ? "text-red-500" : "text-amber-500",
      label: "MCP: spoilage_forecast",
      detail: `urgency=${forecast.urgency}, rho_6h=${fmt(forecast.forecast_rho, 3)}`,
      hash: prov?.evidence_hashes?.[1],
    });
  }

  for (const tool of toolsInvoked) {
    if (tool !== "check_compliance" && tool !== "spoilage_forecast") {
      steps.push({
        icon: Shield,
        iconColor: "text-muted-foreground",
        label: `MCP: ${tool}`,
        detail: "invoked",
      });
    }
  }

  // piRAG step
  if (topDoc) {
    steps.push({
      icon: BookOpen,
      iconColor: "text-blue-500",
      label: `piRAG: ${topDoc}`,
      detail: `score=${fmt(topScore, 2)}`,
      hash: prov?.evidence_hashes?.[2],
    });
  }

  // Merkle root
  if (prov?.merkle_root) {
    steps.push({
      icon: GitBranch,
      iconColor: "text-teal-500",
      label: "Merkle Root",
      detail: prov.merkle_root,
      isMerkle: true,
    });
  }

  // Blockchain anchor: distinguish three states explicitly so the
  // operator never confuses "not attempted" with "anchored".
  //   * real 0x... hash  -> green "On-chain anchor"
  //   * "0x0" or unset   -> amber "anchor pending — chain not configured"
  //                         (with a hint pointing at HOW_TO_RUN.md §6)
  //   * monitoring phase -> distinct "monitoring preview, no anchor by design"
  const phaseStatus = memo.phase_status || "";
  const txHashIsReal = memo.tx_hash &&
    memo.tx_hash !== "0x0" &&
    /^0x[0-9a-fA-F]{2,}$/.test(memo.tx_hash);
  if (txHashIsReal) {
    steps.push({
      icon: CheckCircle2,
      iconColor: "text-emerald-500",
      label: "On-chain anchor",
      detail: memo.tx_hash,
      isHash: true,
    });
  } else if (phaseStatus === "monitoring_preview") {
    steps.push({
      icon: AlertTriangle,
      iconColor: "text-blue-500",
      label: "Monitoring preview",
      detail: "no on-chain anchor by design (deployment phase = monitoring)",
    });
  } else if (phaseStatus === "advisory_pending") {
    steps.push({
      icon: AlertTriangle,
      iconColor: "text-amber-500",
      label: "Advisory pending",
      detail: "awaiting operator approval; will be anchored on approve",
    });
  } else {
    steps.push({
      icon: AlertTriangle,
      iconColor: "text-amber-500",
      label: "On-chain anchor not attempted",
      detail: "chain not configured (set CHAIN_PRIVKEY + run npx hardhat node)",
    });
  }

  if (steps.length === 0) return null;

  const copyHash = (h) => {
    navigator.clipboard.writeText(h);
    toast.success("Hash copied");
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <Hash className="w-4 h-4 text-teal-600" />
        <h4 className="text-sm font-semibold">Provenance Chain</h4>
        {prov?.guards_passed !== false ? (
          <Badge className="bg-emerald-500/10 text-emerald-600 border-0 text-[10px]">
            Guards Passed
          </Badge>
        ) : (
          <Badge className="bg-amber-500/10 text-amber-600 border-0 text-[10px]">
            Guards Failed
            {prov?.guard_breakdown && (
              <span className="ml-1">
                ({Object.entries(prov.guard_breakdown)
                  .filter(([, v]) => v === false)
                  .map(([k]) => k)
                  .join(", ")})
              </span>
            )}
          </Badge>
        )}
      </div>
      <div className="pl-6 relative">
        <div className="absolute left-8 top-0 bottom-0 w-px bg-border" />
        {steps.map((step, i) => (
          <div key={i} className="relative pl-8 pb-3 last:pb-0">
            <div className={cn("absolute left-6 top-0.5 w-4 h-4 rounded-full bg-background border-2 flex items-center justify-center", step.iconColor)}>
              <step.icon className="w-2.5 h-2.5" />
            </div>
            <div className="text-xs">
              <span className="font-medium">{step.label}</span>
              {step.isMerkle ? (
                <button
                  onClick={() => copyHash(step.detail)}
                  className="ml-2 font-mono text-muted-foreground hover:text-primary"
                >
                  {short(step.detail)} <Copy className="w-2.5 h-2.5 inline" />
                </button>
              ) : step.isHash ? (
                <button
                  onClick={() => copyHash(step.detail)}
                  className="ml-2 font-mono text-muted-foreground hover:text-primary"
                >
                  {short(step.detail)} <Copy className="w-2.5 h-2.5 inline" />
                </button>
              ) : (
                <span className="ml-2 text-muted-foreground">{step.detail}</span>
              )}
              {step.hash && (
                <span className="ml-2 font-mono text-[10px] text-muted-foreground/60">
                  SHA: {short(step.hash)}
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// --- Main Panel ---
export default function ExplainabilityPanel({ explainability, memo }) {
  // Render even when explainability is missing entirely so the operator
  // sees an explicit "no enrichment" message instead of an empty card.
  if (!explainability) {
    return (
      <Card className="bg-muted/30 border-primary/10">
        <CardContent className="p-4 space-y-2">
          <div className="flex items-center gap-2">
            <Brain className="w-4 h-4 text-muted-foreground" />
            <h4 className="text-sm font-semibold">Explainability</h4>
            <Badge variant="outline" className="text-[10px]">no payload</Badge>
          </div>
          <p className="text-xs text-muted-foreground">
            The backend did not attach an explainability blob to this memo. This is
            normal for legacy decisions written before the explainability path was
            wired, and for context-disabled modes where the policy reads only φ(s)
            and there is no ψ vector to explain.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="bg-muted/30 border-primary/10">
      <CardContent className="p-4 space-y-4">
        <CausalExplanation explainability={explainability} />
        <Separator />
        <ContextRadar explainability={explainability} />
        <Separator />
        <KeywordsPanel keywords={explainability.keywords} />
        {explainability.keywords && Object.keys(explainability.keywords).length > 0 && <Separator />}
        <ProvenanceChain explainability={explainability} memo={memo || {}} />
      </CardContent>
    </Card>
  );
}
