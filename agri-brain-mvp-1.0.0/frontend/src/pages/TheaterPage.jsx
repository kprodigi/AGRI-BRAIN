import React, { useState, useRef, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Label } from "@/components/ui/label";
import { cn, fmt, short, jpost, jget } from "@/lib/utils";
import { getApiBase } from "@/mvp/api.js";
import { motion, AnimatePresence } from "framer-motion";
import { toast } from "sonner";
import {
  Play, Loader2, MessageCircle, ArrowRight, CheckCircle2, AlertTriangle,
  Wrench, BookOpen, Shield, Layers, Brain, Truck, Warehouse, Recycle,
  Send, ChevronDown, ChevronUp,
} from "lucide-react";

const API = getApiBase();

const AGENTS = [
  { role: "farm",        label: "Farm Agent",       emoji: "\uD83C\uDF3E", color: "bg-green-500",  ring: "ring-green-500/30",  bubble: "bg-green-50 dark:bg-green-950/30  border-green-200 dark:border-green-800" },
  { role: "processor",   label: "Processor Agent",  emoji: "\uD83C\uDFED", color: "bg-blue-500",   ring: "ring-blue-500/30",   bubble: "bg-blue-50 dark:bg-blue-950/30    border-blue-200 dark:border-blue-800" },
  { role: "cooperative", label: "Cooperative Agent", emoji: "\uD83E\uDD1D", color: "bg-purple-500", ring: "ring-purple-500/30", bubble: "bg-purple-50 dark:bg-purple-950/30 border-purple-200 dark:border-purple-800" },
  { role: "distributor", label: "Distributor Agent", emoji: "\uD83D\uDE9B", color: "bg-orange-500", ring: "ring-orange-500/30", bubble: "bg-orange-50 dark:bg-orange-950/30 border-orange-200 dark:border-orange-800" },
  { role: "recovery",    label: "Recovery Agent",    emoji: "\u267B\uFE0F",  color: "bg-teal-500",   ring: "ring-teal-500/30",   bubble: "bg-teal-50 dark:bg-teal-950/30    border-teal-200 dark:border-teal-800" },
];

const SCENARIOS = [
  { id: "baseline",         label: "Baseline",         trigger: "Normal operating conditions — routine check-in" },
  { id: "heatwave",         label: "Heatwave",         trigger: "SPOILAGE_ALERT: Temperature spike detected (+20\u00B0C), accelerated decay" },
  { id: "overproduction",   label: "Overproduction",   trigger: "SURPLUS_ALERT: Inventory at 2.5\u00D7 baseline, redistribution needed" },
  { id: "cyber_outage",     label: "Cyber Outage",     trigger: "CAPACITY_UPDATE: Demand dropped to 15%, refrigeration degraded" },
  { id: "adaptive_pricing", label: "Adaptive Pricing", trigger: "REROUTE_REQUEST: Demand oscillation detected, pricing volatility" },
];

const ACTION_META = {
  cold_chain:          { icon: Truck,      color: "#0072B2", label: "Cold Chain",          short: "CC" },
  local_redistribute:  { icon: Warehouse,  color: "#10B981", label: "Local Redistribution", short: "LR" },
  recovery:            { icon: Recycle,     color: "#D55E00", label: "Recovery",             short: "REC" },
};

const OUTGOING_MESSAGES = {
  cold_chain:         "CAPACITY_UPDATE: Routed via long-haul refrigerated transport",
  local_redistribute: "REROUTE_REQUEST: Redirecting to local markets / food banks",
  recovery:           "ACK: Diverted to composting / animal feed / food bank recovery",
};

const FEATURE_LABELS = [
  { key: "compliance_severity", label: "Compliance", color: "#ef4444" },
  { key: "forecast_urgency",   label: "Forecast",   color: "#f97316" },
  { key: "retrieval_confidence",label: "Retrieval",  color: "#3b82f6" },
  { key: "regulatory_pressure", label: "Regulatory", color: "#a855f7" },
  { key: "recovery_saturation", label: "Recovery",   color: "#22c55e" },
];

// ── Agent Chat Bubble ──
function AgentBubble({ agent, memo, isTyping, scenarioTrigger, isFirst }) {
  const [expanded, setExpanded] = useState(false);
  if (!memo && !isTyping) return null;

  const ex = memo?.explainability || {};
  const cf = ex.context_features || {};
  const la = ex.logit_adjustment || {};
  const comp = ex.compliance || {};
  const ap = memo?.action_probabilities || {};
  const action = memo?.action || memo?.decision;
  const meta = ACTION_META[action] || {};

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.4 }}
      className="flex gap-3 items-start"
    >
      {/* Avatar */}
      <div className={cn("w-10 h-10 rounded-full flex items-center justify-center text-lg shrink-0 shadow-md", agent.color, agent.ring, "ring-2 ring-offset-2 ring-offset-background")}>
        {agent.emoji}
      </div>

      {/* Bubble */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-sm font-semibold">{agent.label}</span>
          {memo && <span className="text-[10px] text-muted-foreground">{memo.time?.split("T")[1]?.split(".")[0]}</span>}
        </div>

        {/* Typing indicator */}
        {isTyping && !memo && (
          <div className={cn("rounded-2xl rounded-tl-sm border px-4 py-3 max-w-lg", agent.bubble)}>
            <div className="flex gap-1.5 items-center">
              <motion.div animate={{ opacity: [0.3, 1, 0.3] }} transition={{ repeat: Infinity, duration: 1.2 }} className="w-2 h-2 rounded-full bg-muted-foreground" />
              <motion.div animate={{ opacity: [0.3, 1, 0.3] }} transition={{ repeat: Infinity, duration: 1.2, delay: 0.2 }} className="w-2 h-2 rounded-full bg-muted-foreground" />
              <motion.div animate={{ opacity: [0.3, 1, 0.3] }} transition={{ repeat: Infinity, duration: 1.2, delay: 0.4 }} className="w-2 h-2 rounded-full bg-muted-foreground" />
              <span className="text-xs text-muted-foreground ml-2">analyzing...</span>
            </div>
          </div>
        )}

        {/* Message content */}
        {memo && (
          <div className={cn("rounded-2xl rounded-tl-sm border px-4 py-3 max-w-2xl", agent.bubble)}>
            {/* Incoming trigger */}
            {isFirst && (
              <div className="flex items-start gap-2 mb-2 pb-2 border-b border-dashed">
                <AlertTriangle className="w-3.5 h-3.5 text-amber-500 mt-0.5 shrink-0" />
                <p className="text-[11px] text-amber-700 dark:text-amber-400">{scenarioTrigger}</p>
              </div>
            )}

            {/* Decision announcement */}
            <div className="flex items-center gap-2 mb-2">
              <Badge className="text-white text-xs px-2 py-0.5" style={{ background: meta.color }}>
                {meta.icon && <meta.icon className="w-3 h-3 mr-1 inline" />}
                {meta.label || action}
              </Badge>
              <span className="text-xs text-muted-foreground">P = {fmt((ap[action] ?? 0) * 100, 1)}%</span>
            </div>

            {/* Probability bars */}
            <div className="space-y-1 mb-2">
              {Object.entries(ap).map(([k, v]) => {
                const m = ACTION_META[k] || {};
                return (
                  <div key={k} className="flex items-center gap-1.5 text-[10px]">
                    <span className="w-16 text-muted-foreground truncate">{m.short || k}</span>
                    <div className="flex-1 h-1.5 rounded-full bg-muted/50 overflow-hidden">
                      <motion.div initial={{ width: 0 }} animate={{ width: `${v * 100}%` }} transition={{ duration: 0.6, delay: 0.2 }}
                        className="h-full rounded-full" style={{ background: m.color || "#888" }} />
                    </div>
                    <span className="font-mono w-10 text-right">{fmt(v * 100, 1)}%</span>
                  </div>
                );
              })}
            </div>

            {/* Expandable details */}
            <button onClick={() => setExpanded(!expanded)} className="flex items-center gap-1 text-[10px] text-primary hover:underline">
              {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
              {expanded ? "Hide details" : "Show pipeline details"}
            </button>

            <AnimatePresence>
              {expanded && (
                <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.3 }} className="overflow-hidden">
                  <div className="mt-2 pt-2 border-t space-y-2.5">

                    {/* MCP Tools */}
                    <div>
                      <p className="text-[9px] font-semibold uppercase tracking-wider text-muted-foreground mb-1">MCP Tools (JSON-RPC 2.0)</p>
                      <div className="flex flex-wrap gap-1">
                        {(ex.mcp_tools_invoked || []).map((t, i) => (
                          <Badge key={i} className="bg-orange-500/10 text-orange-600 border-0 text-[9px]">{t}</Badge>
                        ))}
                      </div>
                      {comp.compliant !== undefined && (
                        <div className="flex items-center gap-1.5 mt-1">
                          <Badge className={cn("text-[8px] border-0", comp.compliant ? "bg-emerald-500/10 text-emerald-600" : "bg-red-500/10 text-red-600")}>
                            {comp.compliant ? "\u2713 Compliant" : "\u2717 Violation"}
                          </Badge>
                          <span className="text-[9px] text-muted-foreground">T={comp.readings?.temperature}\u00B0C max={comp.thresholds?.temp_max_c}\u00B0C</span>
                        </div>
                      )}
                    </div>

                    {/* piRAG */}
                    {ex.pirag_top_doc && (
                      <div>
                        <p className="text-[9px] font-semibold uppercase tracking-wider text-muted-foreground mb-1">piRAG Retrieval</p>
                        <div className="flex items-center gap-1.5">
                          <BookOpen className="w-3 h-3 text-blue-500" />
                          <span className="font-mono text-[10px]">{ex.pirag_top_doc}</span>
                          <Badge variant="outline" className="text-[8px]">score: {fmt(ex.pirag_top_score, 3)}</Badge>
                        </div>
                        {ex.keywords?.regulatory?.thresholds?.length > 0 && (
                          <div className="flex flex-wrap gap-1 mt-1">
                            {ex.keywords.regulatory.thresholds.map((kw, i) => (
                              <Badge key={i} className="text-[8px] bg-purple-500/10 text-purple-600 border-0">{kw}</Badge>
                            ))}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Context Features */}
                    <div>
                      <p className="text-[9px] font-semibold uppercase tracking-wider text-muted-foreground mb-1">Context Features (\u03C8)</p>
                      <div className="space-y-0.5">
                        {FEATURE_LABELS.map(f => (
                          <div key={f.key} className="flex items-center gap-1.5 text-[10px]">
                            <span className="w-1.5 h-1.5 rounded-full" style={{ background: f.color }} />
                            <span className="w-16 text-muted-foreground">{f.label}</span>
                            <div className="flex-1 h-1 rounded-full bg-muted/50 overflow-hidden">
                              <div className="h-full rounded-full" style={{ width: `${(cf[f.key] ?? 0) * 100}%`, background: f.color }} />
                            </div>
                            <span className="font-mono w-8 text-right">{fmt(cf[f.key], 2)}</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Impact */}
                    <div className="grid grid-cols-4 gap-1.5">
                      {[
                        ["SLCA", fmt(memo.slca, 3)],
                        ["CO₂", `${fmt(memo.carbon_kg, 1)}kg`],
                        ["Waste", `${fmt(memo.waste * 100, 1)}%`],
                        ["Circular", fmt(memo.circular_economy_score, 2)],
                      ].map(([k, v]) => (
                        <div key={k} className="text-center bg-muted/30 rounded p-1">
                          <p className="text-[8px] text-muted-foreground">{k}</p>
                          <p className="text-[11px] font-mono font-bold">{v}</p>
                        </div>
                      ))}
                    </div>

                    {/* Causal snippet */}
                    {ex.causal_text && (
                      <div>
                        <p className="text-[9px] font-semibold uppercase tracking-wider text-muted-foreground mb-1">Causal Reasoning</p>
                        <p className="text-[10px] text-muted-foreground leading-relaxed">
                          {ex.causal_text.split("\n\n")[0]?.slice(0, 200).split(/(BECAUSE|WITHOUT)/g).map((p, i) =>
                            p === "BECAUSE" ? <span key={i} className="font-bold text-teal-600">BECAUSE</span> :
                            p === "WITHOUT" ? <span key={i} className="font-bold text-amber-600">WITHOUT</span> :
                            <span key={i}>{p}</span>
                          )}
                          {(ex.causal_text?.length || 0) > 200 && "..."}
                        </p>
                      </div>
                    )}

                    {/* Provenance */}
                    {ex.provenance?.merkle_root && (
                      <div className="flex items-center gap-2">
                        <Badge className="bg-emerald-500/10 text-emerald-600 border-0 text-[8px]">Guards Passed</Badge>
                        <span className="font-mono text-[9px] text-muted-foreground">Merkle: {short(ex.provenance.merkle_root)}</span>
                      </div>
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Outgoing message */}
            <div className="flex items-center gap-1.5 mt-2 pt-2 border-t border-dashed">
              <Send className="w-3 h-3 text-muted-foreground shrink-0" />
              <p className="text-[10px] text-muted-foreground italic">{OUTGOING_MESSAGES[action] || "ACK: Decision recorded"}</p>
            </div>
          </div>
        )}
      </div>
    </motion.div>
  );
}

// ── Main Page ──
export default function TheaterPage() {
  const [scenario, setScenario] = useState("heatwave");
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [agentStates, setAgentStates] = useState([]); // [{typing, memo}]
  const [scenarioSummary, setScenarioSummary] = useState(null);
  const timerRef = useRef(null);
  const bottomRef = useRef(null);

  useEffect(() => () => clearTimeout(timerRef.current), []);

  const runTheater = async () => {
    setRunning(true);
    setAgentStates([]);
    setScenarioSummary(null);

    try {
      // Load data + apply scenario
      await jpost(API, "/case/load").catch(() => {});
      if (scenario !== "baseline") {
        await jpost(API, "/scenarios/run", { name: scenario, intensity: 1.0 }).catch(() => {});
      }

      const memos = [];

      for (let i = 0; i < AGENTS.length; i++) {
        const agent = AGENTS[i];

        // Show typing indicator
        setAgentStates(prev => [...prev, { typing: true, memo: null }]);
        await new Promise(r => setTimeout(r, 800 / speed));
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });

        // Make decision
        let memo;
        try {
          const res = await jpost(API, "/decide", { agent_id: agent.role, role: agent.role });
          memo = res.memo || res;
        } catch {
          // Fallback to cached
          const fallback = await jget(API, "/decisions").catch(() => null);
          const decs = fallback?.decisions || fallback || [];
          memo = decs.find(d => d.role === agent.role) || decs[0];
        }

        if (!memo) { toast.error(`No data for ${agent.label}`); continue; }
        memos.push(memo);

        // Replace typing with actual memo
        setAgentStates(prev => {
          const next = [...prev];
          next[i] = { typing: false, memo };
          return next;
        });

        // Wait before next agent
        await new Promise(r => { timerRef.current = setTimeout(r, 1500 / speed); });
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
      }

      // Reset scenario
      if (scenario !== "baseline") {
        await jpost(API, "/scenarios/reset").catch(() => {});
      }

      // Summary
      if (memos.length > 0) {
        const avgSlca = memos.reduce((s, m) => s + (m.slca || 0), 0) / memos.length;
        const totalCarbon = memos.reduce((s, m) => s + (m.carbon_kg || 0), 0);
        const avgWaste = memos.reduce((s, m) => s + (m.waste || 0), 0) / memos.length;
        const actions = memos.map(m => m.action || m.decision);
        setScenarioSummary({ avgSlca, totalCarbon, avgWaste, actions, count: memos.length });
      }
    } catch (e) {
      toast.error(`Theater failed: ${e.message}`);
    }

    setRunning(false);
  };

  const sc = SCENARIOS.find(s => s.id === scenario) || SCENARIOS[0];

  return (
    <div className="space-y-6 pb-16">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex items-center gap-3 mb-1">
          <MessageCircle className="w-6 h-6 text-teal-600" />
          <h1 className="text-2xl font-bold">Agent Decision Theater</h1>
        </div>
        <p className="text-sm text-muted-foreground">
          Watch all 5 supply-chain agents analyze, decide, and communicate in real-time under different scenarios
        </p>
      </motion.div>

      {/* Controls */}
      <Card>
        <CardContent className="p-4">
          <div className="flex flex-wrap items-end gap-4">
            <div className="w-48">
              <Label className="text-xs mb-1.5 block">Scenario</Label>
              <Select value={scenario} onValueChange={setScenario} disabled={running}>
                <SelectTrigger className="h-9"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {SCENARIOS.map(s => <SelectItem key={s.id} value={s.id}>{s.label}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label className="text-xs mb-1.5 block">Speed</Label>
              <div className="flex gap-1">
                {[1, 2, 4].map(s => (
                  <Button key={s} variant={speed === s ? "default" : "outline"} size="sm" onClick={() => setSpeed(s)} disabled={running}>{s}x</Button>
                ))}
              </div>
            </div>
            <Button onClick={runTheater} disabled={running} size="lg" className="bg-teal-600 hover:bg-teal-700 text-white">
              {running ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Play className="w-4 h-4 mr-2" />}
              {running ? "Running..." : "Start Theater"}
            </Button>
            {agentStates.length > 0 && (
              <div className="flex-1 min-w-32">
                <div className="h-2 rounded-full bg-muted overflow-hidden">
                  <motion.div className="h-full rounded-full bg-teal-500"
                    animate={{ width: `${(agentStates.filter(s => s.memo).length / AGENTS.length) * 100}%` }}
                    transition={{ duration: 0.3 }} />
                </div>
                <p className="text-[10px] text-muted-foreground mt-1 text-right">
                  {agentStates.filter(s => s.memo).length} / {AGENTS.length} agents
                </p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Scenario trigger banner */}
      {agentStates.length > 0 && (
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}>
          <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-amber-50 dark:bg-amber-950/20 border border-amber-200 dark:border-amber-800">
            <AlertTriangle className="w-4 h-4 text-amber-500 shrink-0" />
            <span className="text-xs font-medium text-amber-700 dark:text-amber-400">{sc.label}:</span>
            <span className="text-xs text-amber-600 dark:text-amber-500">{sc.trigger}</span>
          </div>
        </motion.div>
      )}

      {/* Chat timeline */}
      <div className="space-y-4">
        {agentStates.map((state, i) => (
          <AgentBubble
            key={i}
            agent={AGENTS[i]}
            memo={state.memo}
            isTyping={state.typing}
            scenarioTrigger={sc.trigger}
            isFirst={i === 0}
          />
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Summary */}
      <AnimatePresence>
        {scenarioSummary && !running && (
          <motion.div initial={{ opacity: 0, y: 30, scale: 0.95 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ duration: 0.5 }}>
            <Card className="border-teal-500/30 shadow-lg">
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5 text-teal-600" />
                  Scenario Complete: {sc.label}
                </CardTitle>
                <CardDescription>
                  {scenarioSummary.count} agents processed the {sc.label.toLowerCase()} scenario
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div className="text-center bg-muted/30 rounded-lg p-3">
                    <p className="text-[10px] text-muted-foreground uppercase">Avg SLCA</p>
                    <p className="text-xl font-bold text-teal-600">{fmt(scenarioSummary.avgSlca, 3)}</p>
                  </div>
                  <div className="text-center bg-muted/30 rounded-lg p-3">
                    <p className="text-[10px] text-muted-foreground uppercase">Total CO₂</p>
                    <p className="text-xl font-bold text-amber-600">{fmt(scenarioSummary.totalCarbon, 1)} kg</p>
                  </div>
                  <div className="text-center bg-muted/30 rounded-lg p-3">
                    <p className="text-[10px] text-muted-foreground uppercase">Avg Waste</p>
                    <p className="text-xl font-bold text-red-500">{fmt(scenarioSummary.avgWaste * 100, 2)}%</p>
                  </div>
                  <div className="text-center bg-muted/30 rounded-lg p-3">
                    <p className="text-[10px] text-muted-foreground uppercase">Actions</p>
                    <div className="flex justify-center gap-1 mt-1">
                      {scenarioSummary.actions.map((a, i) => {
                        const m = ACTION_META[a] || {};
                        return <Badge key={i} className="text-[8px] text-white" style={{ background: m.color }}>{m.short || a}</Badge>;
                      })}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
