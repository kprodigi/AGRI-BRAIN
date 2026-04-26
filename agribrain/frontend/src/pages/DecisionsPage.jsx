import React, { useEffect, useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { cn, fmt, short, authFetch, authDownload } from "@/lib/utils";
import { getApiBase } from "@/mvp/api.js";
import { motion, AnimatePresence } from "framer-motion";
import { toast } from "sonner";
import {
  PieChart, Pie, Cell, ResponsiveContainer, Tooltip as ReTooltip,
} from "recharts";
import {
  Leaf, Copy, Download, FileText, Clock, User, CheckCircle2,
  Filter, Search, Sparkles,
} from "lucide-react";
import ExplainabilityPanel from "@/components/explainability/ExplainabilityPanel";

const API = getApiBase();

const ACTION_COLORS = {
  cold_chain: { bg: "bg-[#0072B2]/10", text: "text-[#0072B2]", color: "#0072B2", label: "Cold Chain" },
  standard_cold_chain: { bg: "bg-[#0072B2]/10", text: "text-[#0072B2]", color: "#0072B2", label: "Cold Chain" },
  redistribution: { bg: "bg-emerald-500/10", text: "text-emerald-600 dark:text-emerald-400", color: "#10B981", label: "Redistribution" },
  local_redistribution: { bg: "bg-emerald-500/10", text: "text-emerald-600 dark:text-emerald-400", color: "#10B981", label: "Redistribution" },
  local_redistribute: { bg: "bg-emerald-500/10", text: "text-emerald-600 dark:text-emerald-400", color: "#10B981", label: "Redistribution" },
  recovery: { bg: "bg-[#D55E00]/10", text: "text-[#D55E00]", color: "#D55E00", label: "Recovery" },
  composting: { bg: "bg-[#D55E00]/10", text: "text-[#D55E00]", color: "#D55E00", label: "Composting" },
};

function getActionStyle(action) {
  const key = (action || "").toLowerCase().replace(/\s+/g, "_");
  for (const [k, v] of Object.entries(ACTION_COLORS)) {
    if (key.includes(k)) return v;
  }
  return { bg: "bg-muted", text: "text-foreground", color: "#808080", label: action || "Unknown" };
}

function DecisionCard({ memo, index }) {
  const [expanded, setExpanded] = useState(false);
  const [showExplain, setShowExplain] = useState(false);
  const actionStyle = getActionStyle(memo.decision || memo.action);
  const ts = memo.time || memo.ts;
  const timeStr = ts ? new Date(ts).toLocaleString([], { dateStyle: "short", timeStyle: "short" }) : "\u2014";

  const copyDetails = () => {
    let text = `Decision: ${memo.decision || memo.action}\nAgent: ${memo.agent} (${memo.role})\nSLCA: ${memo.slca}\nCarbon: ${memo.carbon_kg} kg\nTx: ${memo.tx || memo.tx_hash}\nTime: ${timeStr}`;
    if (memo.memo_text) text += `\n\n${memo.memo_text}`;
    navigator.clipboard.writeText(text);
    toast.success("Copied to clipboard");
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.05, duration: 0.3 }}
      className="relative pl-8"
    >
      {/* Timeline line & dot */}
      <div className="absolute left-3 top-0 bottom-0 w-px bg-border" />
      <div className={cn("absolute left-1.5 top-5 w-3 h-3 rounded-full border-2 border-background", actionStyle.bg.replace("/10", ""))}
        style={{ backgroundColor: actionStyle.color }}
      />

      <Card className="mb-3 hover:shadow-md transition-shadow">
        <CardContent className="p-4">
          <div className="flex items-start justify-between gap-3">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <Badge className={cn(actionStyle.bg, actionStyle.text, "border-0")}>
                  {actionStyle.label}
                </Badge>
                <span className="text-xs text-muted-foreground flex items-center gap-1">
                  <Clock className="w-3 h-3" /> {timeStr}
                </span>
                <span className="text-xs text-muted-foreground flex items-center gap-1">
                  <User className="w-3 h-3" /> {memo.agent} ({memo.role})
                </span>
              </div>

              {/* Metrics row */}
              <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
                <div>
                  <p className="text-xs text-muted-foreground">SLCA</p>
                  <p className="font-mono font-semibold">{fmt(memo.slca, 3)}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Carbon</p>
                  <p className="font-mono font-semibold">{fmt(memo.carbon_kg, 1)} <span className="text-xs font-normal">kg CO₂</span></p>
                </div>
                {memo.unit_price != null && (
                  <div>
                    <p className="text-xs text-muted-foreground">Unit Price</p>
                    <p className="font-mono font-semibold">${fmt(memo.unit_price, 2)}</p>
                  </div>
                )}
                {memo.circular_economy_score != null && (
                  <div>
                    <p className="text-xs text-muted-foreground">Circular Score</p>
                    <p className="font-mono font-semibold">{fmt(memo.circular_economy_score, 3)}</p>
                  </div>
                )}
              </div>

              {/* Blockchain verification */}
              {(memo.tx || memo.tx_hash) && (
                <div className="mt-2 flex items-center gap-2 text-xs">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
                  <span className="text-muted-foreground">Verified</span>
                  <code className="font-mono text-muted-foreground">{short(memo.tx || memo.tx_hash)}</code>
                  <button
                    onClick={() => { navigator.clipboard.writeText(memo.tx || memo.tx_hash); toast.success("Tx hash copied"); }}
                    className="text-primary hover:underline"
                  >
                    <Copy className="w-3 h-3" />
                  </button>
                </div>
              )}

              {memo.note && <p className="mt-2 text-sm text-muted-foreground italic">{memo.note}</p>}

              {/* Expandable detailed memo */}
              {memo.memo_text && (
                <div className="mt-2">
                  <button
                    onClick={() => setExpanded(!expanded)}
                    className="text-xs text-primary hover:underline font-medium"
                  >
                    {expanded ? "Hide details \u25B2" : "Show detailed memo \u25BC"}
                  </button>
                  <AnimatePresence>
                    {expanded && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        transition={{ duration: 0.2 }}
                        className="mt-2 pl-3 border-l-2 border-primary/30 space-y-2"
                      >
                        {memo.memo_text.split("\n\n").map((para, i) => (
                          <p key={i} className="text-sm text-muted-foreground leading-relaxed">{para}</p>
                        ))}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              )}

              {/* Explainability panel */}
              {memo.explainability && (
                <div className="mt-2">
                  <button
                    onClick={() => setShowExplain(!showExplain)}
                    className="text-xs text-teal-600 dark:text-teal-400 hover:underline font-medium flex items-center gap-1"
                  >
                    <Sparkles className="w-3 h-3" />
                    {showExplain ? "Hide explanation \u25B2" : "Show explanation \u25BC"}
                  </button>
                  <AnimatePresence>
                    {showExplain && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        transition={{ duration: 0.25 }}
                        className="mt-2"
                      >
                        <ExplainabilityPanel explainability={memo.explainability} memo={memo} />
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              )}
            </div>

            <Button variant="ghost" size="icon" className="h-7 w-7 shrink-0" onClick={copyDetails}>
              <Copy className="w-3.5 h-3.5" />
            </Button>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}

export default function DecisionsPage() {
  const [decisions, setDecisions] = useState([]);
  const [role, setRole] = useState("all");
  const [actionFilter, setActionFilter] = useState("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [loading, setLoading] = useState(true);

  const load = async () => {
    try {
      const res = await authFetch(`${API}/decisions`);
      if (!res.ok) throw new Error(res.status);
      const r = await res.json();
      setDecisions(r.decisions || []);
      setLoading(false);
    } catch {
      setLoading(false);
    }
  };

  const takeDecision = async () => {
    const selectedRole = role === "all" ? "farm" : role;
    try {
      const res = await authFetch(`${API}/decide`, {
        method: "POST",
        body: JSON.stringify({ agent_id: "demo:" + selectedRole, role: selectedRole }),
      });
      if (!res.ok) throw new Error(res.status);
      toast.success("Decision taken successfully");
      load();
    } catch {
      toast.error("Failed to take decision");
    }
  };

  useEffect(() => {
    load();
    const id = setInterval(load, 4000);
    return () => clearInterval(id);
  }, []);

  const filtered = useMemo(() => {
    let list = [...decisions];
    if (role !== "all") list = list.filter((m) => m.role === role);
    if (actionFilter !== "all") {
      list = list.filter((m) => {
        const action = (m.decision || m.action || "").toLowerCase();
        return action.includes(actionFilter);
      });
    }
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      list = list.filter((m) =>
        JSON.stringify(m).toLowerCase().includes(q)
      );
    }
    return list.reverse();
  }, [decisions, role, actionFilter, searchQuery]);

  // Analytics
  const actionDist = useMemo(() => {
    const counts = {};
    decisions.forEach((d) => {
      const action = d.decision || d.action || "unknown";
      const style = getActionStyle(action);
      const key = style.label;
      counts[key] = (counts[key] || 0) + 1;
    });
    return Object.entries(counts).map(([name, value]) => ({
      name,
      value,
      color: getActionStyle(name).color,
    }));
  }, [decisions]);

  const avgSLCA = useMemo(() => {
    const vals = decisions.map((d) => +d.slca).filter(Number.isFinite);
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
  }, [decisions]);

  const totalCarbon = useMemo(() => {
    return decisions.reduce((sum, d) => sum + (+d.carbon_kg || 0), 0);
  }, [decisions]);

  const exportCSV = () => {
    const esc = (v) => {
      const s = String(v ?? "");
      return s.includes(",") || s.includes('"') || s.includes("\n") ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const headers = "Time,Agent,Role,Action,SLCA,Carbon_kg,Unit_Price,Circular_Score,Tx_Hash,Note,Memo_Text\n";
    const rows = decisions.map((d) =>
      [d.time, d.agent, d.role, d.decision || d.action, d.slca, d.carbon_kg, d.unit_price, d.circular_economy_score, d.tx || d.tx_hash, d.note, d.memo_text].map(esc).join(",")
    ).join("\n");
    const blob = new Blob([headers + rows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "decisions.csv";
    a.click();
    URL.revokeObjectURL(url);
    toast.success("CSV exported");
  };

  if (loading) {
    return (
      <div className="space-y-4">
        {[...Array(3)].map((_, i) => (
          <Card key={i}><CardContent className="p-4"><Skeleton className="h-24" /></CardContent></Card>
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Main timeline */}
      <div className="lg:col-span-2 space-y-4">
        {/* Filters */}
        <Card>
          <CardContent className="p-4">
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-2">
                <Filter className="w-4 h-4 text-muted-foreground" />
                <Select value={role} onValueChange={setRole}>
                  <SelectTrigger className="w-36 h-8"><SelectValue placeholder="Role" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Roles</SelectItem>
                    <SelectItem value="farm">Farm</SelectItem>
                    <SelectItem value="processor">Processor</SelectItem>
                    <SelectItem value="cooperative">Cooperative</SelectItem>
                    <SelectItem value="distributor">Distributor</SelectItem>
                    <SelectItem value="recovery">Recovery</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <Select value={actionFilter} onValueChange={setActionFilter}>
                <SelectTrigger className="w-40 h-8"><SelectValue placeholder="Action" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Actions</SelectItem>
                  <SelectItem value="cold_chain">Cold Chain</SelectItem>
                  <SelectItem value="redistribution">Redistribution</SelectItem>
                  <SelectItem value="recovery">Recovery</SelectItem>
                </SelectContent>
              </Select>
              <div className="relative flex-1 min-w-48">
                <Search className="absolute left-2.5 top-2 w-4 h-4 text-muted-foreground" />
                <Input
                  placeholder="Search decisions..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-8 h-8"
                />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Actions bar */}
        <div className="flex items-center gap-2">
          <Button size="sm" onClick={takeDecision} data-skip-global-take="1">
            <Leaf className="w-4 h-4 mr-1" /> Take Decision
          </Button>
          <Button variant="outline" size="sm" onClick={() => authDownload(`${API}/report/pdf?role=${role === "all" ? "farm" : role}`, "decision-report.pdf")}>
            <FileText className="w-4 h-4 mr-1" /> PDF Report
          </Button>
          <Button variant="outline" size="sm" onClick={exportCSV}>
            <Download className="w-4 h-4 mr-1" /> CSV Export
          </Button>
        </div>

        {/* Timeline */}
        <div>
          {filtered.length === 0 ? (
            <Card>
              <CardContent className="p-8 text-center text-muted-foreground">
                <Leaf className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p>No decisions found matching your filters.</p>
              </CardContent>
            </Card>
          ) : (
            <AnimatePresence>
              {filtered.map((m, i) => (
                <DecisionCard key={`${m.time}-${i}`} memo={m} index={i} />
              ))}
            </AnimatePresence>
          )}
        </div>
      </div>

      {/* Analytics sidebar */}
      <div className="space-y-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Decision Analytics</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              <div className="text-center p-3 rounded-lg bg-muted/50">
                <p className="text-2xl font-bold">{decisions.length}</p>
                <p className="text-xs text-muted-foreground">Total Decisions</p>
              </div>
              <div className="text-center p-3 rounded-lg bg-muted/50">
                <p className="text-2xl font-bold font-mono">{fmt(avgSLCA, 3)}</p>
                <p className="text-xs text-muted-foreground">Mean SLCA</p>
              </div>
              <div className="text-center p-3 rounded-lg bg-muted/50">
                <p className="text-2xl font-bold font-mono">{fmt(totalCarbon, 0)}</p>
                <p className="text-xs text-muted-foreground">Total CO₂ (kg)</p>
              </div>
              <div className="text-center p-3 rounded-lg bg-muted/50">
                <p className="text-2xl font-bold font-mono">{actionDist.length}</p>
                <p className="text-xs text-muted-foreground">Action Types</p>
              </div>
            </div>

            {/* Action distribution */}
            {actionDist.length > 0 && (
              <div>
                <p className="text-xs font-medium text-muted-foreground mb-2">Action Distribution</p>
                <div className="h-40">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={actionDist} cx="50%" cy="50%" innerRadius={35} outerRadius={60} paddingAngle={4} dataKey="value">
                        {actionDist.map((entry, i) => (
                          <Cell key={i} fill={entry.color} />
                        ))}
                      </Pie>
                      <ReTooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="space-y-1 mt-2">
                  {actionDist.map((item) => (
                    <div key={item.name} className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-2">
                        <span className="h-2 w-2 rounded-full" style={{ background: item.color }} />
                        <span>{item.name}</span>
                      </div>
                      <span className="font-mono">{item.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
