// ChannelAttributionPanel
// --------------------------------------------------------------------
// Renders the §5.8 H3-mechanism evidence: cross-seed per-channel logit
// contribution statistics from the canonical 20-seed Path A benchmark.
//
// Source data: `mvp/simulation/results/decision_ledger_aggregate.json`
// produced by `mvp/simulation/benchmarks/aggregate_decision_ledgers.py`.
// The file is fetched at runtime via `/results/figures/<filename>` (the
// backend's results router serves anything under the results dir; the
// `.gitignore` allowlist exposes this JSON as a tracked artifact so a
// fresh CI build / clone has the file available).
//
// What the panel proves to a guest in ~30 seconds:
//   1. The MCP and piRAG channels carry *different* information (their
//      median signed logit shifts on the chosen action live on
//      different scales: MCP median ~0, piRAG median ~0.15 — the piRAG
//      channel's discrete Theta[a, 3] regulatory-pressure step).
//   2. Channel-level attribution is *complementary*: removing either
//      reduces the integrated shift, but the two channels respond to
//      different ψ-features (MCP -> ψ0 compliance; piRAG -> ψ2/ψ3).
//   3. ~20% of decisions show super-additive integration that exceeds
//      the better single channel by >0.005 ARI-equivalent logit shift
//      — neither channel alone can produce that uplift.
//
// The panel layout mirrors the manuscript's supplementary Table S1
// rendering: per-scenario rows for the agribrain mode with median (IQR)
// for MCP, piRAG, and joint shift, plus the sub-additivity fraction.
import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn, fmt, authFetch } from "@/lib/utils";
import { getApiBase } from "@/mvp/api.js";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Legend,
  Tooltip as ReTooltip, ResponsiveContainer, Cell,
} from "recharts";
import {
  Layers, Brain, BookOpen, AlertTriangle, GitBranch, Sparkles,
  TrendingUp, Hash,
} from "lucide-react";

const API = getApiBase();

// --------------------------------------------------------------------
// Feature → channel map. Keep in lockstep with
//   agribrain/backend/pirag/context_to_logits.py THETA_CONTEXT columns
// and
//   mvp/simulation/benchmarks/aggregate_decision_ledgers.py MCP_MASK /
//   PIRAG_MASK.
// MCP-derived features:   ψ0 compliance, ψ1 forecast, ψ4 recovery_sat
// piRAG-derived features: ψ2 retrieval,  ψ3 regulatory
// --------------------------------------------------------------------
const FEATURES = [
  { key: "psi0_compliance",    label: "ψ₀ compliance",    channel: "MCP",   color: "#ef4444" },
  { key: "psi1_forecast",      label: "ψ₁ forecast",      channel: "MCP",   color: "#f97316" },
  { key: "psi4_recovery_sat",  label: "ψ₄ recovery sat.", channel: "MCP",   color: "#22c55e" },
  { key: "psi2_retrieval",     label: "ψ₂ retrieval",     channel: "piRAG", color: "#3b82f6" },
  { key: "psi3_regulatory",    label: "ψ₃ regulatory",    channel: "piRAG", color: "#a855f7" },
];

const SCENARIO_LABELS = {
  heatwave: "Heatwave",
  overproduction: "Overproduction",
  cyber_outage: "Cyber outage",
  adaptive_pricing: "Adaptive pricing",
  baseline: "Baseline",
};

const SCENARIO_ORDER = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing", "baseline"];
const PERTURBED = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing"];

// --------------------------------------------------------------------
// Render helpers
// --------------------------------------------------------------------
function MedianIqr({ median, q25, q75 }) {
  return (
    <span className="font-mono tabular-nums">
      {fmt(median, 3)}{" "}
      <span className="text-muted-foreground">({fmt(q25, 3)}, {fmt(q75, 3)})</span>
    </span>
  );
}

function StatTile({ label, value, hint, accent }) {
  return (
    <div className={cn(
      "rounded-lg border p-3",
      accent === "mcp" && "border-orange-500/30 bg-orange-500/5",
      accent === "pirag" && "border-blue-500/30 bg-blue-500/5",
      accent === "joint" && "border-emerald-500/30 bg-emerald-500/5",
      !accent && "border-border bg-muted/30",
    )}>
      <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className="text-xl font-bold tabular-nums mt-0.5">{value}</div>
      {hint && <div className="text-[10px] text-muted-foreground mt-0.5">{hint}</div>}
    </div>
  );
}

// --------------------------------------------------------------------
// Aggregate the 4 perturbed-scenario agribrain cells into a single
// pooled statistic. The aggregate JSON stores per-cell medians; here we
// average them as a defensible proxy for the pooled distribution since
// every cell has identical n=5760 records (20 seeds × 288 steps), and
// the manuscript's pooled re-computation from the JSONL files produces
// numbers within rounding of this average (see
// section5_8_path_a_filled.md).
// --------------------------------------------------------------------
function poolPerturbed(byScenarioMode) {
  const cells = PERTURBED.map((s) => byScenarioMode?.[s]?.agribrain).filter(Boolean);
  if (cells.length === 0) return null;
  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
  const total = cells.reduce((acc, c) => acc + (c.n_records || 0), 0);
  const nDom = cells.reduce(
    (acc, c) => acc + (c.sub_additivity?.n_steps_joint_exceeds_max_single || 0),
    0,
  );
  return {
    n_records: total,
    mcp:   { median: mean(cells.map((c) => c.mcp_channel_logit_shift.median)),
             q25:    mean(cells.map((c) => c.mcp_channel_logit_shift.q25)),
             q75:    mean(cells.map((c) => c.mcp_channel_logit_shift.q75)) },
    pirag: { median: mean(cells.map((c) => c.pirag_channel_logit_shift.median)),
             q25:    mean(cells.map((c) => c.pirag_channel_logit_shift.q25)),
             q75:    mean(cells.map((c) => c.pirag_channel_logit_shift.q75)) },
    joint: { median: mean(cells.map((c) => c.joint_logit_shift.median)),
             q25:    mean(cells.map((c) => c.joint_logit_shift.q25)),
             q75:    mean(cells.map((c) => c.joint_logit_shift.q75)) },
    subAdditivity: total > 0 ? nDom / total : 0,
    nDominate: nDom,
    featureMeanAbs: FEATURES.reduce((acc, f) => {
      acc[f.key] = mean(cells.map((c) => c.feature_attribution?.[f.key]?.mean_abs || 0));
      return acc;
    }, {}),
  };
}

// --------------------------------------------------------------------
// Main component
// --------------------------------------------------------------------
export default function ChannelAttributionPanel() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const resp = await authFetch(`${API}/results/figures/decision_ledger_aggregate.json`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const json = await resp.json();
        if (!cancelled) setData(json);
      } catch (err) {
        if (!cancelled) setError(err.message || String(err));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const pooled = useMemo(() => (data ? poolPerturbed(data.by_scenario_mode) : null), [data]);

  if (loading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-24 rounded-lg" />
        <Skeleton className="h-64 rounded-lg" />
        <Skeleton className="h-48 rounded-lg" />
      </div>
    );
  }

  if (error || !data || !pooled) {
    return (
      <Card>
        <CardContent className="py-6">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-amber-500 mt-0.5" />
            <div>
              <p className="font-semibold text-sm">Channel-attribution data unavailable</p>
              <p className="text-xs text-muted-foreground mt-1">
                The backend did not serve <code>decision_ledger_aggregate.json</code>. Verify the
                file exists at <code>mvp/simulation/results/decision_ledger_aggregate.json</code>{" "}
                and that the <code>/results/figures/&lt;filename&gt;</code> endpoint is reachable.
                Producing it requires the canonical 20-seed Path A run plus
                <code className="ml-1">python mvp/simulation/benchmarks/aggregate_decision_ledgers.py</code>.
              </p>
              {error && (
                <p className="text-xs text-rose-500 mt-2 font-mono">{error}</p>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  const meta = data._meta || {};
  const featureBars = FEATURES.map((f) => ({
    feature: f.label,
    channel: f.channel,
    value: pooled.featureMeanAbs[f.key] || 0,
    color: f.color,
  }));

  return (
    <div className="space-y-6">
      {/* ────────────────────────────────────────────────────────────── */}
      {/* Headline / H3 claim                                              */}
      {/* ────────────────────────────────────────────────────────────── */}
      <Card className="border-teal-500/30 bg-gradient-to-br from-teal-500/5 to-blue-500/5">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-teal-600" />
            <CardTitle className="text-lg">H3 — Component complementarity established</CardTitle>
            <Badge variant="outline" className="text-[10px]">§5.8</Badge>
          </div>
          <CardDescription>
            Per-decision channel attribution across the canonical 20-seed benchmark.
            n = {pooled.n_records.toLocaleString()} agribrain-mode decisions pooled across
            the four perturbed scenarios (heatwave, overproduction, cyber outage, adaptive pricing).
            The MCP and piRAG channels respond to different ψ-features and integrate
            super-additively on {(pooled.subAdditivity * 100).toFixed(1)}% of decisions.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            <StatTile
              label="MCP channel — median Δlogit"
              value={fmt(pooled.mcp.median, 3)}
              hint={`IQR [${fmt(pooled.mcp.q25, 3)}, ${fmt(pooled.mcp.q75, 3)}]`}
              accent="mcp"
            />
            <StatTile
              label="piRAG channel — median Δlogit"
              value={fmt(pooled.pirag.median, 3)}
              hint={`IQR [${fmt(pooled.pirag.q25, 3)}, ${fmt(pooled.pirag.q75, 3)}]`}
              accent="pirag"
            />
            <StatTile
              label="Joint Δz on chosen action"
              value={fmt(pooled.joint.median, 3)}
              hint={`IQR [${fmt(pooled.joint.q25, 3)}, ${fmt(pooled.joint.q75, 3)}]`}
              accent="joint"
            />
            <StatTile
              label="Super-additive fraction"
              value={`${(pooled.subAdditivity * 100).toFixed(1)}%`}
              hint={`${pooled.nDominate.toLocaleString()} / ${pooled.n_records.toLocaleString()} steps where joint > max single by >0.005`}
            />
          </div>
          <Separator />
          <div className="text-xs leading-relaxed text-muted-foreground">
            The MCP-channel median is ~0 because most decisions sit in low-compliance states (ψ₀ = 0).
            The piRAG-channel median is the discrete Θ<sub>context</sub>[a, ψ₃] regulatory-pressure
            step (~0.15) that fires whenever the retrieved guidance carries a regulatory keyword.
            These two channels are not interchangeable — they live on different scales and respond to
            different ψ features. Removing either reduces the attributable shift; removing both
            collapses the policy to the no-context floor.
          </div>
        </CardContent>
      </Card>

      {/* ────────────────────────────────────────────────────────────── */}
      {/* Supplementary Table S1 — per-scenario agribrain mode             */}
      {/* ────────────────────────────────────────────────────────────── */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Layers className="w-5 h-5 text-primary" />
            <CardTitle className="text-base">Supplementary Table S1 — per-scenario channel statistics</CardTitle>
          </div>
          <CardDescription>
            Median (IQR) signed logit shift on the chosen action under the agribrain mode,
            computed across all {Math.floor(pooled.n_records / 4).toLocaleString()} decisions
            of each scenario (n_seeds = {meta.n_seeds ?? 20} × 288 steps). The "AB &gt; max single"
            column reports the fraction of decisions where the integrated condition exceeds the
            better single channel by &gt;0.005 ARI-equivalent logit shift.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="rounded-lg border overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="font-semibold">Scenario</TableHead>
                  <TableHead className="text-right">n</TableHead>
                  <TableHead>
                    <span className="text-orange-600">MCP</span> Δlogit
                  </TableHead>
                  <TableHead>
                    <span className="text-blue-600">piRAG</span> Δlogit
                  </TableHead>
                  <TableHead>
                    <span className="text-emerald-600">Joint</span> Δz
                  </TableHead>
                  <TableHead className="text-right">AB &gt; max single</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {SCENARIO_ORDER.map((scn) => {
                  const c = data.by_scenario_mode?.[scn]?.agribrain;
                  if (!c) {
                    return (
                      <TableRow key={scn}>
                        <TableCell className="font-medium">{SCENARIO_LABELS[scn]}</TableCell>
                        <TableCell colSpan={5} className="text-xs text-muted-foreground italic">
                          cell missing in aggregate
                        </TableCell>
                      </TableRow>
                    );
                  }
                  const s = c.sub_additivity || {};
                  const isBaseline = scn === "baseline";
                  return (
                    <TableRow key={scn} className={isBaseline ? "bg-muted/30" : ""}>
                      <TableCell className="font-medium">
                        {SCENARIO_LABELS[scn]}
                        {isBaseline && (
                          <Badge variant="outline" className="ml-2 text-[9px]">baseline</Badge>
                        )}
                      </TableCell>
                      <TableCell className="text-right font-mono tabular-nums text-xs">
                        {(c.n_records || 0).toLocaleString()}
                      </TableCell>
                      <TableCell className="text-xs">
                        <MedianIqr {...c.mcp_channel_logit_shift} />
                      </TableCell>
                      <TableCell className="text-xs">
                        <MedianIqr {...c.pirag_channel_logit_shift} />
                      </TableCell>
                      <TableCell className="text-xs">
                        <MedianIqr {...c.joint_logit_shift} />
                      </TableCell>
                      <TableCell className="text-right text-xs">
                        <span className="font-semibold">
                          {((s.fraction_joint_dominates || 0) * 100).toFixed(1)}%
                        </span>
                        <span className="text-muted-foreground ml-1">
                          ({(s.n_steps_joint_exceeds_max_single || 0).toLocaleString()})
                        </span>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
          <div className="text-[10px] text-muted-foreground mt-2 leading-relaxed">
            <strong>Channel masks.</strong> MCP = {"{ψ₀ compliance severity, ψ₁ forecast urgency, ψ₄ recovery saturation}"};
            piRAG = {"{ψ₂ retrieval confidence, ψ₃ regulatory pressure}"}. Joint Δz applies
            Θ<sub>context</sub> · ψ followed by τ<sub>mod</sub> attenuation and [−1, +1] clipping
            (same operation the policy applies at run time).
          </div>
        </CardContent>
      </Card>

      {/* ────────────────────────────────────────────────────────────── */}
      {/* Per-feature attribution bar chart                                */}
      {/* ────────────────────────────────────────────────────────────── */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-primary" />
            <CardTitle className="text-base">Per-feature mean absolute contribution</CardTitle>
          </div>
          <CardDescription>
            |Θ<sub>context</sub>[a, k] · ψ<sub>k</sub>| averaged across all {pooled.n_records.toLocaleString()}
            decisions (4 perturbed scenarios, agribrain mode). MCP channel features in warm colours;
            piRAG channel features in cool colours.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={featureBars} layout="vertical" margin={{ top: 8, right: 24, bottom: 8, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="feature" width={110} tick={{ fontSize: 11 }} />
                <ReTooltip
                  formatter={(v, _name, props) => [fmt(v, 4), `${props.payload.channel} channel`]}
                  labelFormatter={(l) => l}
                />
                <Bar dataKey="value" radius={[0, 6, 6, 0]}>
                  {featureBars.map((d, i) => (
                    <Cell key={i} fill={d.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="text-[10px] text-muted-foreground mt-2">
            ψ₀ compliance dominates the MCP channel; ψ₂ retrieval confidence and ψ₃ regulatory pressure split
            the piRAG channel; ψ₁ forecast urgency and ψ₄ recovery saturation are smaller-magnitude contributors
            that fire scenario-specifically (ψ₁ peaks in heatwave at {fmt(data.by_scenario_mode?.heatwave?.agribrain?.feature_attribution?.psi1_forecast?.mean_abs || 0, 3)};
            falls to {fmt(data.by_scenario_mode?.baseline?.agribrain?.feature_attribution?.psi1_forecast?.mean_abs || 0, 3)} in baseline).
          </div>
        </CardContent>
      </Card>

      {/* ────────────────────────────────────────────────────────────── */}
      {/* Provenance footer                                                */}
      {/* ────────────────────────────────────────────────────────────── */}
      <div className="rounded-lg border border-dashed border-muted-foreground/30 bg-muted/20 px-3 py-2 text-[10px] text-muted-foreground flex items-center gap-2 flex-wrap">
        <Hash className="w-3 h-3" />
        <span className="font-semibold">Source:</span>
        <code>mvp/simulation/results/decision_ledger_aggregate.json</code>
        <span>·</span>
        <span>generated at <code>{meta.generated_at || "unknown"}</code></span>
        <span>·</span>
        <GitBranch className="w-3 h-3 inline" />
        <code>{(meta.git_commit || "unknown").slice(0, 12)}</code>
        <span>·</span>
        <span>{(meta.n_seeds ?? 20)} seeds × {SCENARIO_ORDER.length} scenarios</span>
      </div>
    </div>
  );
}
