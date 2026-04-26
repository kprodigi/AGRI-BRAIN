"""Export structured decision traces for paper evidence.

Captures the full information flow at each decision step:
  Observation -> MCP tool outputs -> piRAG retrieved passages ->
  Context features -> Logit adjustment -> Action -> Explanation -> Provenance

Three export formats:
  1. Per-step trace (detailed, for supplementary material)
  2. Role comparison table (which tools/docs each role uses)
  3. Provenance chain (evidence -> hashes -> Merkle root)
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

_log = logging.getLogger(__name__)


@dataclass
class DecisionTrace:
    """Complete record of one decision step for paper reporting."""

    # Step metadata
    hour: float
    role: str
    scenario: str
    action: str

    # Observation state
    rho: float
    temperature: float
    humidity: float
    inventory: float
    tau: float
    surplus_ratio: float

    # MCP tool outputs (actual content)
    mcp_tools_invoked: List[str] = field(default_factory=list)
    compliance_result: Optional[Dict] = None
    spoilage_forecast: Optional[Dict] = None
    slca_lookup: Optional[Dict] = None
    chain_query_summary: Optional[str] = None

    # piRAG retrieval (actual text)
    pirag_query: str = ""
    pirag_top_doc: str = ""
    pirag_top_score: float = 0.0
    pirag_regulatory_text: str = ""
    pirag_sop_text: str = ""
    pirag_waste_hierarchy_text: str = ""
    pirag_governance_text: str = ""
    pirag_slca_text: str = ""

    # Context features and decision
    context_features: List[float] = field(default_factory=list)
    context_feature_names: List[str] = field(default_factory=list)
    logit_adjustment: List[float] = field(default_factory=list)
    action_probabilities: List[float] = field(default_factory=list)
    governance_override: bool = False

    # Extracted keywords per guidance type
    keywords_regulatory: List[str] = field(default_factory=list)
    keywords_sop: List[str] = field(default_factory=list)
    keywords_waste_hierarchy: List[str] = field(default_factory=list)
    keywords_governance: List[str] = field(default_factory=list)

    # Causal explanation data
    causal_primary_cause: str = ""
    causal_primary_contribution: float = 0.0
    counterfactual_action: str = ""
    counterfactual_prob_shift: List[float] = field(default_factory=list)
    action_changed_by_context: bool = False
    retrieval_metrics: Dict[str, float] = field(default_factory=dict)
    retrieval_counterfactual: Dict[str, Any] = field(default_factory=dict)

    # Explanation
    explanation_summary: str = ""
    explanation_full: str = ""

    # Provenance
    evidence_hashes: List[str] = field(default_factory=list)
    merkle_root: str = ""
    provenance_ready: bool = False


_FEATURE_NAMES = [
    "compliance_severity", "forecast_urgency",
    "retrieval_confidence", "regulatory_pressure", "recovery_saturation",
]


class TraceExporter:
    """Collects and exports decision traces for paper evidence."""

    def __init__(self, max_traces: int = 50):
        self._traces: List[DecisionTrace] = []
        self.max_traces = max_traces
        self._roles_captured: set = set()
        self._action_changes_captured: int = 0

    def should_capture(self, role: str, action_changed: bool, hour: float) -> bool:
        """Decide whether to capture this step's trace."""
        if role not in self._roles_captured:
            return True
        if action_changed and self._action_changes_captured < 10:
            return True
        block = int(hour / 6.0)
        if not any(int(t.hour / 6.0) == block and t.role == role for t in self._traces):
            return True
        return len(self._traces) < self.max_traces

    def capture(
        self,
        obs: Any,
        scenario: str,
        action: str,
        probs: np.ndarray,
        mcp_results: Dict[str, Any],
        rag_context: Dict[str, Any],
        context_features: Optional[np.ndarray],
        logit_adjustment: Optional[np.ndarray],
        explanation: Optional[Dict[str, Any]],
        role: str = "unknown",
        action_changed: bool = False,
        governance_override: bool = False,
    ) -> None:
        """Capture a full decision trace."""
        if not self.should_capture(role, action_changed, obs.hour):
            return

        self._roles_captured.add(role)
        if action_changed:
            self._action_changes_captured += 1

        # Compliance result summary
        compliance = mcp_results.get("check_compliance")
        compliance_summary = None
        if isinstance(compliance, dict):
            compliance_summary = {
                "compliant": compliance.get("compliant"),
                "violations": [
                    {"parameter": v.get("parameter"), "value": v.get("value"),
                     "limit": v.get("limit"), "severity": v.get("severity"),
                     "message": v.get("message", "")}
                    for v in compliance.get("violations", [])
                ],
            }

        # Forecast summary
        forecast = mcp_results.get("spoilage_forecast")
        forecast_summary = None
        if isinstance(forecast, dict):
            forecast_summary = {
                "current_rho": forecast.get("current_rho"),
                "forecast_rho": forecast.get("forecast_rho"),
                "hours_ahead": forecast.get("hours_ahead"),
                "urgency": forecast.get("urgency"),
            }

        # Chain query summary
        chain = mcp_results.get("chain_query", {})
        chain_records = chain.get("records", []) if isinstance(chain, dict) else chain
        chain_summary = None
        if isinstance(chain_records, list) and chain_records:
            actions_count: Dict[str, int] = {}
            for d in chain_records:
                a = d.get("action", "unknown")
                actions_count[a] = actions_count.get(a, 0) + 1
            chain_summary = f"{len(chain_records)} recent decisions: " + ", ".join(
                f"{a}={c}" for a, c in sorted(actions_count.items())
            )

        # Provenance
        ev_hashes: List[str] = []
        if explanation:
            ev_hashes = explanation.get("evidence_hashes", [])

        merkle = ""
        if ev_hashes:
            try:
                from .provenance.merkle import merkle_root
                merkle = merkle_root(ev_hashes)
            except Exception as _exc:
                _log.debug("merkle root for trace skipped: %s", _exc)

        trace = DecisionTrace(
            hour=obs.hour,
            role=role,
            scenario=scenario,
            action=action,
            rho=obs.rho,
            temperature=obs.temp,
            humidity=obs.rh,
            inventory=obs.inv,
            tau=obs.tau,
            surplus_ratio=getattr(obs, "surplus_ratio", 0.0),
            mcp_tools_invoked=mcp_results.get("_tools_invoked", []),
            compliance_result=compliance_summary,
            spoilage_forecast=forecast_summary,
            slca_lookup=mcp_results.get("slca_lookup"),
            chain_query_summary=chain_summary,
            pirag_query=rag_context.get("query", ""),
            pirag_top_doc=rag_context.get("top_doc_id", ""),
            pirag_top_score=rag_context.get("top_citation_score", 0.0),
            pirag_regulatory_text=(rag_context.get("regulatory_guidance", "") or "")[:300],
            pirag_sop_text=(rag_context.get("sop_guidance", "") or "")[:300],
            pirag_waste_hierarchy_text=(rag_context.get("waste_hierarchy_guidance", "") or "")[:300],
            pirag_governance_text=(rag_context.get("governance_guidance", "") or "")[:300],
            pirag_slca_text=(rag_context.get("slca_guidance", "") or "")[:300],
            context_features=context_features.tolist() if context_features is not None else [],
            context_feature_names=_FEATURE_NAMES,
            logit_adjustment=logit_adjustment.tolist() if logit_adjustment is not None else [],
            action_probabilities=probs.tolist() if probs is not None else [],
            governance_override=governance_override,
            explanation_summary=explanation.get("summary", "") if explanation else "",
            explanation_full=(explanation.get("full_explanation", "") or "")[:800] if explanation else "",
            evidence_hashes=ev_hashes,
            merkle_root=merkle,
            provenance_ready=bool(merkle),
        )

        # Populate keyword fields from rag_context (deduplicated)
        keywords = rag_context.get("keywords", {})
        for kw_type in ["regulatory", "sop", "waste_hierarchy", "governance"]:
            kw_data = keywords.get(kw_type, {})
            if isinstance(kw_data, dict):
                all_kw = (kw_data.get("thresholds", [])
                          + kw_data.get("regulations", [])
                          + kw_data.get("required_actions", []))
            elif isinstance(kw_data, list):
                all_kw = kw_data
            else:
                all_kw = []
            # Deduplicate preserving order
            seen: set = set()
            unique: List[str] = []
            for kw in all_kw:
                normalized = kw.strip().lower()
                if normalized not in seen:
                    seen.add(normalized)
                    unique.append(kw)
            setattr(trace, f"keywords_{kw_type}", unique[:5])

        # Populate causal explanation data
        if explanation:
            causal = explanation.get("causal_chain", {})
            trace.causal_primary_cause = causal.get("primary_cause", "")
            trace.causal_primary_contribution = causal.get("primary_contribution", 0.0)
            cf = explanation.get("counterfactual", {})
            trace.counterfactual_action = cf.get("action_without_context", "") or ""
            trace.counterfactual_prob_shift = cf.get("probability_shift", []) or []
            trace.action_changed_by_context = cf.get("action_changed", False)
        trace.retrieval_metrics = rag_context.get("retrieval_metrics", {}) or {}
        trace.retrieval_counterfactual = rag_context.get("counterfactual", {}) or {}

        self._traces.append(trace)

    def export_json(self, filepath: str) -> None:
        """Export all traces as JSON for supplementary material."""
        data = []
        for t in self._traces:
            data.append({
                "step": {
                    "hour": t.hour, "role": t.role, "scenario": t.scenario,
                    "action": t.action, "governance_override": t.governance_override,
                },
                "observation": {
                    "rho": t.rho, "temperature": t.temperature,
                    "humidity": t.humidity, "inventory": t.inventory,
                    "tau": t.tau, "surplus_ratio": t.surplus_ratio,
                },
                "mcp_tools": {
                    "invoked": t.mcp_tools_invoked,
                    "compliance": t.compliance_result,
                    "forecast": t.spoilage_forecast,
                    "slca": t.slca_lookup,
                    "chain_query": t.chain_query_summary,
                },
                "pirag_retrieval": {
                    "query": t.pirag_query,
                    "top_document": t.pirag_top_doc,
                    "top_score": t.pirag_top_score,
                    "regulatory_guidance": t.pirag_regulatory_text,
                    "sop_guidance": t.pirag_sop_text,
                    "waste_hierarchy": t.pirag_waste_hierarchy_text,
                    "governance": t.pirag_governance_text,
                    "slca_guidance": t.pirag_slca_text,
                },
                "context_decision": {
                    "features": dict(zip(t.context_feature_names, t.context_features)) if t.context_features else {},
                    "logit_adjustment": dict(zip(
                        ["ColdChain", "LocalRedistribute", "Recovery"],
                        t.logit_adjustment,
                    )) if t.logit_adjustment else {},
                    "probabilities": dict(zip(
                        ["ColdChain", "LocalRedistribute", "Recovery"],
                        t.action_probabilities,
                    )) if t.action_probabilities else {},
                },
                "keywords": {
                    "regulatory": t.keywords_regulatory,
                    "sop": t.keywords_sop,
                    "waste_hierarchy": t.keywords_waste_hierarchy,
                    "governance": t.keywords_governance,
                },
                "causal_reasoning": {
                    "primary_cause": t.causal_primary_cause,
                    "primary_contribution": t.causal_primary_contribution,
                    "counterfactual_action": t.counterfactual_action,
                    "probability_shift": t.counterfactual_prob_shift,
                    "action_changed": t.action_changed_by_context,
                },
                "retrieval_evaluation": {
                    "metrics": t.retrieval_metrics,
                    "counterfactual": t.retrieval_counterfactual,
                },
                "explanation": {
                    "summary": t.explanation_summary,
                    "full": t.explanation_full,
                },
                "provenance": {
                    "evidence_hashes": t.evidence_hashes[:5],
                    "total_evidence_items": len(t.evidence_hashes),
                    "merkle_root": t.merkle_root,
                    "provenance_ready": t.provenance_ready,
                },
            })
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def export_role_comparison_table(self) -> List[Dict]:
        """Generate a role comparison table for the paper."""
        role_data: Dict[str, Dict[str, Any]] = {}
        for t in self._traces:
            if t.role not in role_data:
                role_data[t.role] = {
                    "mcp_tools": set(),
                    "top_docs": [],
                    "feature_means": np.zeros(5),
                    "logit_means": np.zeros(3),
                    "n": 0,
                    "guidance_types": [],
                    "all_keywords": [],
                    "causal_causes": [],
                }
            rd = role_data[t.role]
            rd["mcp_tools"].update(t.mcp_tools_invoked)
            if t.pirag_top_doc:
                rd["top_docs"].append(t.pirag_top_doc)
            if t.context_features:
                rd["feature_means"] += np.array(t.context_features)
            if t.logit_adjustment:
                rd["logit_means"] += np.array(t.logit_adjustment)
            rd["n"] += 1
            for gtype in ["regulatory", "sop", "waste_hierarchy", "governance", "slca"]:
                if getattr(t, f"pirag_{gtype}_text", ""):
                    rd["guidance_types"].append(gtype)
            # Aggregate keywords
            for kw_field in ["keywords_regulatory", "keywords_sop",
                             "keywords_waste_hierarchy", "keywords_governance"]:
                rd["all_keywords"].extend(getattr(t, kw_field, []))
            # Aggregate causal causes
            if t.causal_primary_cause:
                rd["causal_causes"].append(t.causal_primary_cause)

        table = []
        for role, rd in sorted(role_data.items()):
            n = max(rd["n"], 1)
            doc_counts = Counter(rd["top_docs"])
            top_doc = doc_counts.most_common(1)[0][0] if doc_counts else "none"
            guide_counts = Counter(rd["guidance_types"])
            top_guide = guide_counts.most_common(1)[0][0] if guide_counts else "none"
            kw_counts = Counter(rd["all_keywords"])
            top_keywords = [kw for kw, _ in kw_counts.most_common(5)]
            cause_counts = Counter(rd["causal_causes"])
            cause_distribution = {
                cause: round(count / n, 2) for cause, count in cause_counts.most_common()
            }
            table.append({
                "role": role,
                "mcp_tools": sorted(rd["mcp_tools"]),
                "primary_kb_document": top_doc,
                "primary_guidance_type": top_guide,
                "mean_features": (rd["feature_means"] / n).tolist(),
                "mean_logit_shift": (rd["logit_means"] / n).tolist(),
                "n_traces": rd["n"],
                "top_keywords": top_keywords,
                "primary_cause_distribution": cause_distribution,
            })
        return table

    def export_provenance_chains(self) -> List[Dict]:
        """Export provenance chains showing evidence -> hash -> Merkle root."""
        chains = []
        for t in self._traces:
            if not t.provenance_ready:
                continue
            chains.append({
                "hour": t.hour,
                "role": t.role,
                "action": t.action,
                "explanation_summary": t.explanation_summary,
                "evidence_sources": {
                    "mcp_tools": t.mcp_tools_invoked,
                    "pirag_top_doc": t.pirag_top_doc,
                    "pirag_score": t.pirag_top_score,
                },
                "evidence_hashes": t.evidence_hashes[:5],
                "merkle_root": t.merkle_root,
                "hash_count": len(t.evidence_hashes),
                "chain_complete": bool(t.merkle_root and t.evidence_hashes),
            })
        return chains

    def export_interoperability_trace(self) -> List[Dict]:
        """Export MCP JSON-RPC request/response pairs for interoperability proof."""
        traces = []
        for t in self._traces[:5]:
            step_traces = []
            step_traces.append({
                "request": {
                    "jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "clientInfo": {"name": f"agribrain-{t.role}", "version": "1.0.0"},
                    },
                },
                "response": {
                    "jsonrpc": "2.0", "id": 1,
                    "result": {"capabilities": {"tools": {}, "resources": {}, "prompts": {}}},
                },
            })
            if t.compliance_result:
                step_traces.append({
                    "request": {
                        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                        "params": {
                            "name": "check_compliance",
                            "arguments": {"temperature": t.temperature, "humidity": t.humidity},
                        },
                    },
                    "response_summary": (
                        f"compliant={t.compliance_result.get('compliant')}, "
                        f"violations={len(t.compliance_result.get('violations', []))}"
                    ),
                })
            if t.spoilage_forecast:
                step_traces.append({
                    "request": {
                        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                        "params": {
                            "name": "spoilage_forecast",
                            "arguments": {"current_rho": t.rho, "temperature": t.temperature},
                        },
                    },
                    "response_summary": (
                        f"forecast_rho={t.spoilage_forecast.get('forecast_rho')}, "
                        f"urgency={t.spoilage_forecast.get('urgency')}"
                    ),
                })
            step_traces.append({
                "request": {
                    "jsonrpc": "2.0", "id": 4, "method": "resources/read",
                    "params": {"uri": "agribrain://quality/spoilage_risk"},
                },
                "response_summary": f"rho={t.rho}",
            })
            traces.append({
                "hour": t.hour,
                "role": t.role,
                "mcp_interactions": step_traces,
                "total_protocol_messages": len(step_traces) * 2,
            })
        return traces

    def export_feature_heatmap_data(self) -> Dict[str, Dict[str, List[float]]]:
        """Generate scenario -> role -> mean feature vector for heatmap figure."""
        data: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for t in self._traces:
            if not t.context_features:
                continue
            key = t.scenario
            if key not in data:
                data[key] = {}
            if t.role not in data[key]:
                data[key][t.role] = {"sum": np.zeros(5), "n": 0}
            data[key][t.role]["sum"] += np.array(t.context_features)
            data[key][t.role]["n"] += 1

        result: Dict[str, Dict[str, List[float]]] = {}
        for scenario, roles in data.items():
            result[scenario] = {}
            for role, rd in roles.items():
                n = max(rd["n"], 1)
                result[scenario][role] = (rd["sum"] / n).tolist()
        return result

    def summary(self) -> Dict[str, Any]:
        """Summary statistics."""
        return {
            "total_traces": len(self._traces),
            "roles_captured": sorted(self._roles_captured),
            "action_changes_captured": self._action_changes_captured,
            "provenance_chains": sum(1 for t in self._traces if t.provenance_ready),
            "hours_range": [self._traces[0].hour, self._traces[-1].hour] if self._traces else [],
        }

    def reset(self) -> None:
        self._traces.clear()
        self._roles_captured.clear()
        self._action_changes_captured = 0
