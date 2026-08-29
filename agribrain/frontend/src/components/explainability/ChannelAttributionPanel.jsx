// ChannelAttributionPanel
// --------------------------------------------------------------------
// Renders the §5.8 H2-mechanism evidence at the DECISION level (not the
// logit-shift level). Source data:
//   mvp/simulation/results/channel_attribution_aggregate.json
// produced by mvp/simulation/benchmarks/aggregate_channel_attribution.py
// from the instrumented 20-seed agribrain run. Fetched at runtime via
// /results/figures/<filename>.
//
// The observed context modifier is linear-additive in logit space. This panel
// therefore reports conditional argmax sensitivity after algebraically
// masking MCP-derived or retrieval-derived feature groups in each recorded
// full-context state. Retrieval results, guards, and all other state remain
// fixed. These are feature-group reconstructions, not interventions that
// disable a communication channel and not causal effect estimates.
//   - activation orthogonality + conditional magnitude
//   - MCP-derived operating-envelope and tool-feature contribution
import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn, authFetch } from "@/lib/utils";
import { getApiBase } from "@/mvp/api.js";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Legend,
  Tooltip as ReTooltip, ResponsiveContainer, ErrorBar,
} from "recharts";
import {
  Layers, Brain, AlertTriangle, GitBranch, Sparkles,
  ShieldCheck, Hash, Zap,
} from "lucide-react";

const API = getApiBase();

const C_MCP = "#F57C00";
const C_PIRAG = "#1565C0";
const C_SYN = "#8E24AA";
const C_RED = "#9E9E9E";
const C_CTX = "#009688";

const SCENARIO_LABELS = {
  heatwave: "Heatwave",
  overproduction: "Overproduction",
  cyber_outage: "Cyber outage",
  adaptive_pricing: "Adaptive pricing",
  baseline: "Baseline",
};
const SCENARIO_ORDER = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing", "baseline"];

function pct(x, d = 1) {
  const value = Number(x);
  return Number.isFinite(value) ? `${(100 * value).toFixed(d)}%` : "unavailable";
}

function StatTile({ label, value, hint, accent }) {
  return (
    <div className={cn(
      "rounded-lg border p-3",
      accent === "mcp" && "border-orange-500/30 bg-orange-500/5",
      accent === "pirag" && "border-blue-500/30 bg-blue-500/5",
      accent === "ctx" && "border-teal-500/30 bg-teal-500/5",
      accent === "syn" && "border-purple-500/30 bg-purple-500/5",
      !accent && "border-border bg-muted/30",
    )}>
      <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className="text-xl font-bold tabular-nums mt-0.5">{value}</div>
      {hint && <div className="text-[10px] text-muted-foreground mt-0.5">{hint}</div>}
    </div>
  );
}

function RateCi({ node }) {
  if (!node) return <span className="text-muted-foreground">—</span>;
  return (
    <span className="font-mono tabular-nums">
      {pct(node.rate)}{" "}
      <span className="text-muted-foreground text-[10px]">
        [{pct(node.ci_low)}, {pct(node.ci_high)}]
      </span>
    </span>
  );
}

export default function ChannelAttributionPanel() {
  const [data, setData] = useState(null);
  const [compTest, setCompTest] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const resp = await authFetch(`${API}/results/figures/channel_attribution_aggregate.json`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const json = await resp.json();
        const meta = json?._meta;
        const seedCount = Number(meta?.seed_count ?? meta?.n_seeds);
        const expectedScenarios = Object.keys(SCENARIO_LABELS);
        const actualScenarios = Object.keys(json?.by_scenario_mode || {});
        if (seedCount !== 20 || Number(meta?.n_bootstrap) !== 2000) {
          throw new Error("channel attribution is not the canonical 20-seed, 2,000-bootstrap panel");
        }
        if (actualScenarios.length !== expectedScenarios.length
            || expectedScenarios.some((scenario) => !actualScenarios.includes(scenario))) {
          throw new Error("channel attribution does not contain the exact five-scenario panel");
        }
        for (const scenario of expectedScenarios) {
          const cell = json.by_scenario_mode?.[scenario]?.agribrain;
          if (!cell || Number(cell.n_seeds) !== 20) {
            throw new Error(`channel attribution cell ${scenario}/agribrain is incomplete`);
          }
        }
        if (!cancelled) setData(json);
      } catch (err) {
        if (!cancelled) setError(err.message || String(err));
      } finally {
        if (!cancelled) setLoading(false);
      }
      // Dedicated seed-cluster inference for the conditional distinctness
      // index; optional because the aggregate also carries its point value.
      // optional — falls back to the aggregate pooled value if absent.
      try {
        const r2 = await authFetch(`${API}/results/figures/channel_complementarity_test.json`);
        if (r2.ok) { const j2 = await r2.json(); if (!cancelled) setCompTest(j2); }
      } catch { /* ignore; aggregate pooled value is used */ }
    })();
    return () => { cancelled = true; };
  }, []);

  const bsm = useMemo(() => data?.by_scenario_mode || {}, [data]);
  const pooled = data?.agribrain_perturbed_pooled || null;
  const compIndex = (compTest?.conditional_distinctness_index != null)
    ? compTest.conditional_distinctness_index
    : (pooled?.conditional_distinctness_index ?? null);
  const compCi = compTest?.bootstrap_ci || null;
  const scenarios = useMemo(
    () => SCENARIO_ORDER.filter((s) => bsm[s]?.agribrain),
    [bsm],
  );

  const attribBars = useMemo(() => scenarios.map((s) => {
    const a = bsm[s].agribrain.attribution_fraction || {};
    const aci = bsm[s].agribrain.attribution_fraction_ci || {};
    // val in percent; err is the asymmetric seed-cluster bootstrap 95% CI
    // (also in percent), centered on the displayed fraction.
    const cell = (rawKey) => {
      const frac = Number(a[rawKey]);
      if (!Number.isFinite(frac)) return { val: null, err: undefined };
      const c = aci[rawKey];
      const err = (c && c.ci_low != null && c.ci_high != null)
        ? [100 * (frac - c.ci_low), 100 * (c.ci_high - frac)]
        : undefined;
      return { val: 100 * frac, err };
    };
    const p = cell("pirag_group_matches_observed_only"),
          m = cell("mcp_group_matches_observed_only"),
          sy = cell("neither_group_matches_observed"),
          rd = cell("both_groups_match_observed");
    const row = {
      scenario: SCENARIO_LABELS[s] || s,
      "Retrieval group only": p.val, "MCP group only": m.val,
      "Neither single group": sy.val, "Both single groups": rd.val,
    };
    if (p.err) row["Retrieval group only_err"] = p.err;
    if (m.err) row["MCP group only_err"] = m.err;
    if (sy.err) row["Neither single group_err"] = sy.err;
    if (rd.err) row["Both single groups_err"] = rd.err;
    return row;
  }), [scenarios, bsm]);

  if (loading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-24 rounded-lg" />
        <Skeleton className="h-64 rounded-lg" />
        <Skeleton className="h-48 rounded-lg" />
      </div>
    );
  }

  if (error || !data || scenarios.length === 0) {
    return (
      <Card>
        <CardContent className="py-6">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-amber-500 mt-0.5" />
            <div>
              <p className="font-semibold text-sm">Channel-attribution data unavailable</p>
              <p className="text-xs text-muted-foreground mt-1">
                The backend did not serve <code>channel_attribution_aggregate.json</code>. Launch the canonical
                treatment with <code>hpc/hpc_run.sh</code>; its dependent <code>hpc/hpc_publish.sh</code> stage
                consolidates the normal per-seed ledgers and produces this validated artifact.
              </p>
              {error && <p className="text-xs text-rose-500 mt-2 font-mono">{error}</p>}
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  const meta = data._meta || {};

  return (
    <div className="space-y-6">
      {/* Headline */}
      <Card className="border-teal-500/30 bg-gradient-to-br from-teal-500/5 to-blue-500/5">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-teal-600" />
            <CardTitle className="text-lg">H2 — Conditional context-feature sensitivity</CardTitle>
            <Badge variant="outline" className="text-[10px]">§5.8</Badge>
          </div>
          <CardDescription>
            Algebraic feature-group masking across the instrumented {meta.seed_count ?? meta.n_seeds}-seed benchmark
            {pooled ? ` (n = ${pooled.n_instrumented_decisions.toLocaleString()} agribrain decisions, 4 perturbed scenarios)` : ""}.
            The conditional distinctness index is {compIndex != null ? pct(compIndex, 0) : "unavailable"}.
            It summarizes how often the two single-feature-group reconstructions do not both reproduce
            the observed modal route. It does not estimate what would happen if a live channel were disabled.
          </CardDescription>
        </CardHeader>
        {pooled && (
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              <StatTile accent="ctx" label="Observed vs zeroed"
                value={pct(pooled.context_route_change_rate)}
                hint="modal route differs" />
              <StatTile accent="mcp" label="Mask MCP features"
                value={pct(pooled.mcp_feature_group_mask_effect_rate)}
                hint="modal route differs" />
              <StatTile accent="pirag" label="Mask retrieval features"
                value={pct(pooled.pirag_feature_group_mask_effect_rate)}
                hint="modal route differs" />
              <StatTile accent="syn" label="Joint-only route change"
                value={pct(pooled.joint_only_route_change_rate)}
                hint="neither single group reproduces it" />
            </div>
            <Separator />
            <div className="text-xs leading-relaxed text-muted-foreground">
              <strong>Conditional distinctness {pct(compIndex)}{compCi ? ` (seed-cluster bootstrap 95% CI [${pct(compCi[0])}, ${pct(compCi[1])}])` : ""}</strong> — the share of
              observed context-changed modal routes for which the two single-group reconstructions do not
              both match the observed route. This analysis holds the retrieved documents, guards, state,
              and tool outputs fixed; the separate MCP-only, retrieval-only, and No-external-context experimental
              arms provide the actual channel-arm comparisons.
            </div>
            {pooled.decision_movement_concentration &&
              pooled.context_route_change_given_active_rate != null && (
              <div className="text-xs leading-relaxed text-muted-foreground">
                <strong>Conditional concentration.</strong> The {pct(pooled.context_route_change_rate)} unconditional
                rate includes decisions where the combined MCP/piRAG modifier is negligible. Retrieval guards
                withhold only the piRAG term; MCP evidence can remain active. Conditioned on decisions where the
                combined layer is <em>active</em>,
                the observed and zeroed modal routes differ on{" "}
                <strong>{pct(pooled.context_route_change_given_active_rate)}</strong>, and probability movement is
                highly concentrated (Gini{" "}
                <strong>{pooled.decision_movement_concentration.gini.toFixed(3)}</strong>: the route-changing{" "}
                {pct(pooled.context_route_change_rate)} of decisions carry{" "}
                <strong>{pct(pooled.decision_movement_concentration.share_carried_by_decisive)}</strong> of
                all movement). The MCP-feature mask effect can also be inspected when the declared operating-envelope feature is active
                {pooled.mcp_feature_group_mask_effect_given_compliance_rate != null && (
                  <> ({pct(pooled.mcp_feature_group_mask_effect_rate)} overall → {pct(pooled.mcp_feature_group_mask_effect_given_compliance_rate)} conditionally)</>
                )}.
              </div>
            )}
          </CardContent>
        )}
      </Card>

      {/* Per-scenario conditional masking table */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Layers className="w-5 h-5 text-primary" />
            <CardTitle className="text-base">Per-scenario conditional masking sensitivity</CardTitle>
          </div>
          <CardDescription>
            agribrain mode, pooled over {meta.seed_count ?? meta.n_seeds} seeds × 288 steps. Rates are fractions of
            instrumented decisions with seed-cluster bootstrap 95% CIs.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="rounded-lg border overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="font-semibold">Scenario</TableHead>
                  <TableHead className="text-right">n</TableHead>
                  <TableHead><span style={{ color: C_CTX }}>Observed vs zeroed</span></TableHead>
                  <TableHead><span style={{ color: C_MCP }}>Mask MCP features</span></TableHead>
                  <TableHead><span style={{ color: C_PIRAG }}>Mask retrieval features</span></TableHead>
                  <TableHead><span style={{ color: C_SYN }}>Joint-only change</span></TableHead>
                  <TableHead className="text-right">Distinctness</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {scenarios.map((scn) => {
                  const c = bsm[scn].agribrain;
                  const isBaseline = scn === "baseline";
                  return (
                    <TableRow key={scn} className={isBaseline ? "bg-muted/30" : ""}>
                      <TableCell className="font-medium">
                        {SCENARIO_LABELS[scn]}
                        {isBaseline && <Badge variant="outline" className="ml-2 text-[9px]">baseline</Badge>}
                      </TableCell>
                      <TableCell className="text-right font-mono tabular-nums text-xs">
                        {Number.isFinite(Number(c.n_instrumented_decisions)) ? Number(c.n_instrumented_decisions).toLocaleString() : "Unavailable"}
                      </TableCell>
                      <TableCell className="text-xs"><RateCi node={c.context_route_change} /></TableCell>
                      <TableCell className="text-xs"><RateCi node={c.mcp_feature_group_mask_effect} /></TableCell>
                      <TableCell className="text-xs"><RateCi node={c.pirag_feature_group_mask_effect} /></TableCell>
                      <TableCell className="text-xs"><RateCi node={c.joint_only_route_change} /></TableCell>
                      <TableCell className="text-right text-xs font-semibold">
                        {pct(c.conditional_distinctness_index)}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      {/* Attribution stacked bar */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Brain className="w-5 h-5 text-primary" />
            <CardTitle className="text-base">Which single-group reconstruction matches the observed route?</CardTitle>
          </div>
          <CardDescription>
            Each bar partitions observed-versus-zeroed route changes by whether the MCP-feature reconstruction,
            retrieval-feature reconstruction, both, or neither reproduces the observed modal route. Error bars
            are seed-cluster bootstrap 95% confidence intervals. These conditional reconstructions do not assign
            causal responsibility to either communication channel.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={attribBars} margin={{ top: 8, right: 16, bottom: 8, left: 8 }} barGap={1} barCategoryGap="18%">
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis dataKey="scenario" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} unit="%" domain={[0, "auto"]} />
                <ReTooltip formatter={(v) => `${(v || 0).toFixed(1)}%`} />
                <Legend />
                <Bar dataKey="Retrieval group only" fill={C_PIRAG} radius={[2, 2, 0, 0]} isAnimationActive={false}>
                  <ErrorBar dataKey="Retrieval group only_err" width={5} strokeWidth={2} stroke="#1f2937" />
                </Bar>
                <Bar dataKey="MCP group only" fill={C_MCP} radius={[2, 2, 0, 0]} isAnimationActive={false}>
                  <ErrorBar dataKey="MCP group only_err" width={5} strokeWidth={2} stroke="#1f2937" />
                </Bar>
                <Bar dataKey="Neither single group" fill={C_SYN} radius={[2, 2, 0, 0]} isAnimationActive={false}>
                  <ErrorBar dataKey="Neither single group_err" width={5} strokeWidth={2} stroke="#1f2937" />
                </Bar>
                <Bar dataKey="Both single groups" fill={C_RED} radius={[2, 2, 0, 0]} isAnimationActive={false}>
                  <ErrorBar dataKey="Both single groups_err" width={5} strokeWidth={2} stroke="#1f2937" />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* MCP policy-rule and operating-envelope signals */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-5 h-5 text-rose-600" />
            <CardTitle className="text-base">Observed policy-rule and operating-envelope context</CardTitle>
          </div>
          <CardDescription>
            Descriptive rates from the recorded synthetic policy. They identify where author-declared
            threshold features were present; they are not compliance determinations and do not isolate a channel effect.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {scenarios.map((scn) => {
              const g = bsm[scn].agribrain.mcp_governance || {};
              return (
                <div key={scn} className="rounded-lg border p-3 bg-muted/20">
                  <div className="text-xs font-semibold flex items-center gap-1">
                    <Zap className="w-3 h-3 text-amber-500" /> {SCENARIO_LABELS[scn]}
                  </div>
                  <div className="text-[11px] text-muted-foreground mt-1 space-y-0.5">
                    <div>probability-gap override: <span className="font-mono">{pct(g.governance_override_rate, 2)}</span></div>
                    <div>operating-envelope feature active: <span className="font-mono">{pct(g.compliance_active_rate)}</span></div>
                    <div>route differs when feature-active: <span className="font-mono">{pct(g.compliance_decisive_rate)}</span></div>
                  </div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Provenance footer */}
      <div className="rounded-lg border border-dashed border-muted-foreground/30 bg-muted/20 px-3 py-2 text-[10px] text-muted-foreground flex items-center gap-2 flex-wrap">
        <Hash className="w-3 h-3" />
        <span className="font-semibold">Source:</span>
        <code>mvp/simulation/results/channel_attribution_aggregate.json</code>
        <span>·</span>
        <span>generated <code>{meta.generated_at || "unknown"}</code></span>
        <span>·</span>
        <GitBranch className="w-3 h-3 inline" />
        <code>{(meta.git_commit || "unknown").slice(0, 12)}</code>
        <span>·</span>
        <span>{meta.seed_count ?? meta.n_seeds} seeds, {meta.n_bootstrap}-sample bootstrap</span>
      </div>
    </div>
  );
}
