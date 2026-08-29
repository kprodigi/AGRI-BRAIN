import React, { useEffect, useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { jget as jgetUtil, jpost as jpostUtil } from "@/lib/utils";
import { runtimePhaseOrDefault } from "@/lib/runtimePhase.js";
import { getApiBase } from "@/mvp/api.js";
import { toast } from "sonner";
import { CheckCircle2, XCircle, Clock, RefreshCw } from "lucide-react";

const API = getApiBase();

const PHASE_DESC = {
  monitoring:
    "Recommendation preview only. The decision is shown but is not added to the decision log and is not submitted to the optional chain logger.",
  advisory:
    "Operator-in-the-loop. Each decision is queued below; approval finalizes and records it. A chain record exists only when chain logging is configured and succeeds.",
  autonomous:
    "Development auto-finalize: each emitted decision is recorded immediately. A chain record exists only when chain logging is configured and succeeds.",
};

const PHASE_LABEL = {
  monitoring: "Monitoring",
  advisory: "Advisory",
  autonomous: "Auto-finalize (development)",
};

export default function PhaseTab() {
  const [phase, setPhase] = useState(null);
  const [queueDepth, setQueueDepth] = useState(0);
  const [ttl, setTtl] = useState(0);
  const [pending, setPending] = useState([]);
  const [history, setHistory] = useState([]);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const p = await jgetUtil(API, "/phase");
      setPhase(runtimePhaseOrDefault(p?.phase));
      setQueueDepth(p?.queue_depth || 0);
      setTtl(p?.advisory_ttl_s || 0);
      const q = await jgetUtil(API, "/phase/advisory/pending");
      setPending(q?.pending || []);
      const h = await jgetUtil(API, "/phase/advisory/history?limit=20");
      setHistory(h?.history || []);
    } catch (e) {
      toast.error(`Phase load failed: ${e.message}`);
    }
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 5000);
    return () => clearInterval(t);
  }, [refresh]);

  async function setActivePhase(p) {
    setBusy(true);
    try {
      await jpostUtil(API, "/phase", { phase: p });
      toast.success(`Development runtime phase set to ${PHASE_LABEL[p] || p}`);
      await refresh();
    } catch (e) {
      toast.error(`Set phase failed: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function approve(id) {
    setBusy(true);
    try {
      await jpostUtil(API, `/phase/advisory/${id}/approve`, {});
      toast.success("Decision approved");
      await refresh();
    } catch (e) {
      toast.error(`Approve failed: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function reject(id) {
    setBusy(true);
    try {
      await jpostUtil(API, `/phase/advisory/${id}/reject`, {});
      toast.success("Decision rejected");
      await refresh();
    } catch (e) {
      toast.error(`Reject failed: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            Development runtime phase
            {phase && (
              <Badge
                variant={
                  phase === "autonomous"
                    ? "default"
                    : phase === "advisory"
                    ? "secondary"
                    : "outline"
                }
              >
                {PHASE_LABEL[phase] || phase}
              </Badge>
            )}
          </CardTitle>
          <CardDescription>
            These settings control the development <code>/decide</code> runtime only.
            Publication benchmark episodes run through the independent simulation protocol
            and do not inherit this dashboard phase setting.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-wrap gap-2">
            {["monitoring", "advisory", "autonomous"].map((p) => (
              <Button
                key={p}
                variant={phase === p ? "default" : "outline"}
                disabled={busy}
                onClick={() => setActivePhase(p)}
              >
                {PHASE_LABEL[p] || p}
              </Button>
            ))}
            <Button variant="ghost" onClick={refresh} disabled={busy} size="icon" title="Refresh">
              <RefreshCw className="w-4 h-4" />
            </Button>
          </div>
          {phase && (
            <div className="text-sm text-muted-foreground">{PHASE_DESC[phase]}</div>
          )}
          <div className="text-xs text-muted-foreground">
            Advisory TTL: {Math.round(ttl)} s &middot; Queue depth: {queueDepth}
          </div>
        </CardContent>
      </Card>

      {phase === "advisory" && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Clock className="w-4 h-4" /> Pending decisions
              <Badge variant="secondary">{pending.length}</Badge>
            </CardTitle>
            <CardDescription>
              Decisions awaiting operator approval. Each entry expires after the TTL above
              if no action is taken.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {pending.length === 0 ? (
              <div className="text-sm text-muted-foreground py-4 text-center">
                No decisions awaiting review.
              </div>
            ) : (
              pending.map((p) => {
                const m = p.memo || {};
                const expIn = Math.max(0, Math.round(p.expires_at - Date.now() / 1000));
                return (
                  <div key={p.decision_id} className="flex items-start justify-between gap-3 border rounded-md p-3">
                    <div className="text-sm space-y-1 min-w-0 flex-1">
                      <div className="font-medium">
                        {m.action} <span className="text-xs text-muted-foreground">({m.role || "—"})</span>
                      </div>
                      <div className="text-xs text-muted-foreground truncate">
                        {m.note || m.summary || ""}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        ρ={typeof m.rho === "number" ? m.rho.toFixed(3) : "—"} &middot; expires in {expIn}s
                      </div>
                    </div>
                    <div className="flex gap-1 shrink-0">
                      <Button size="sm" variant="default" onClick={() => approve(p.decision_id)} disabled={busy}>
                        <CheckCircle2 className="w-4 h-4 mr-1" /> Approve
                      </Button>
                      <Button size="sm" variant="outline" onClick={() => reject(p.decision_id)} disabled={busy}>
                        <XCircle className="w-4 h-4 mr-1" /> Reject
                      </Button>
                    </div>
                  </div>
                );
              })
            )}
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Advisory history</CardTitle>
          <CardDescription>Last 20 advisory outcomes (approved / rejected / expired).</CardDescription>
        </CardHeader>
        <CardContent>
          {history.length === 0 ? (
            <div className="text-sm text-muted-foreground py-4 text-center">No history yet.</div>
          ) : (
            <div className="space-y-1 max-h-64 overflow-y-auto">
              {history.slice().reverse().map((h) => {
                const m = h.memo || {};
                return (
                  <div key={h.decision_id} className="flex items-center justify-between text-xs border-b py-1">
                    <span className="truncate flex-1">{m.action} &middot; {m.role || "—"}</span>
                    <Badge
                      variant={
                        h.outcome === "approve"
                          ? "default"
                          : h.outcome === "reject"
                          ? "outline"
                          : "secondary"
                      }
                    >
                      {h.outcome}
                    </Badge>
                  </div>
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
