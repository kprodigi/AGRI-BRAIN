import React, { useEffect, useState, useCallback } from "react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { jget as jgetUtil } from "@/lib/utils";
import { getApiBase } from "@/mvp/api.js";

const API = getApiBase();

const PHASE_META = {
  monitoring: {
    label: "Monitoring",
    short: "MON",
    bg: "bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-200",
    dot: "bg-blue-500",
    desc: "Recommendations only. Decisions are computed and shown but the supply chain is not mutated and nothing is logged on-chain.",
  },
  advisory: {
    label: "Advisory",
    short: "ADV",
    bg: "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-200",
    dot: "bg-amber-500",
    desc: "Decisions are queued and require operator approval before they are executed and anchored.",
  },
  autonomous: {
    label: "Autonomous",
    short: "AUTO",
    bg: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-200",
    dot: "bg-emerald-500",
    desc: "Decisions are executed and anchored immediately as the policy emits them.",
  },
};

export default function PhasePill() {
  const [phase, setPhase] = useState(null);
  const [queueDepth, setQueueDepth] = useState(0);

  const load = useCallback(async () => {
    try {
      const p = await jgetUtil(API, "/phase");
      setPhase(p?.phase || "autonomous");
      setQueueDepth(p?.queue_depth || 0);
    } catch {
      // Backend unreachable: fall back to neutral display
      setPhase(null);
    }
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, 8000);
    return () => clearInterval(t);
  }, [load]);

  if (!phase) return null;
  const meta = PHASE_META[phase] || PHASE_META.autonomous;

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div
          className={`hidden md:flex items-center gap-1.5 px-2 py-1 rounded-md text-xs font-semibold ${meta.bg}`}
        >
          <span className={`h-1.5 w-1.5 rounded-full ${meta.dot}`} />
          <span>{meta.label}</span>
          {phase === "advisory" && queueDepth > 0 && (
            <span className="ml-1 inline-flex h-4 min-w-4 items-center justify-center rounded-full bg-amber-600 px-1 text-[10px] font-bold text-white">
              {queueDepth > 99 ? "99+" : queueDepth}
            </span>
          )}
        </div>
      </TooltipTrigger>
      <TooltipContent>
        <div className="max-w-xs text-xs">
          <div className="font-semibold mb-1">Deployment phase: {meta.label}</div>
          <div>{meta.desc}</div>
          {phase === "advisory" && (
            <div className="mt-1 text-muted-foreground">
              {queueDepth} decision{queueDepth === 1 ? "" : "s"} awaiting operator review.
            </div>
          )}
        </div>
      </TooltipContent>
    </Tooltip>
  );
}
