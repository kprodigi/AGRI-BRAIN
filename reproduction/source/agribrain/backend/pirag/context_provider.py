"""Institutional-retrieval context provider for the decision pipeline.

Queries the constructed piR knowledge base for source-labelled context based on
the current scenario conditions, spoilage risk, and temperature, then
returns a structured context dict for use in action selection and
explanation generation.

Supports both the original 3-parameter signature (backward-compatible)
and an extended signature with role, humidity, inventory, surplus, tau,
and hour parameters.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from .strict_validation import handle_unexpected_failure

_log = logging.getLogger(__name__)

_PIPELINE = None


def _get_pipeline():
    """Lazy-initialize the PiR pipeline with knowledge base documents."""
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    try:
        from .agent_pipeline import PiRPipeline

        # PiRPipeline auto-ingests the knowledge_base/ directory on init
        pipeline = PiRPipeline()

        _PIPELINE = pipeline
    except Exception as _exc:
        handle_unexpected_failure(
            "context-provider pipeline initialization", _exc, _log,
        )
        _PIPELINE = None

    return _PIPELINE


def get_policy_context(
    scenario: str = "baseline",
    spoilage_risk: float = 0.0,
    temperature: float = 4.0,
    role: str = "farm",
    humidity: float = 90.0,
    inventory: float = 12000.0,
    surplus_ratio: float = 0.0,
    tau: float = 0.0,
    hour: float = 0.0,
    y_hat: float = 100.0,
    context_mode: str = "full",
    retrieval_kind: str = "pirag",
) -> Dict[str, Any]:
    """Query the constructed knowledge base for source-labelled context.

    Parameters
    ----------
    scenario : current scenario name.
    spoilage_risk : current spoilage risk (rho).
    temperature : current temperature in Celsius.
    role : active agent role (farm, processor, etc.).
    humidity : current relative humidity in percent.
    inventory : current inventory level.
    surplus_ratio : inventory surplus above baseline.
    tau : volatility indicator.
    hour : simulation hour.
    y_hat : demand point forecast used by MCP/context features.
    context_mode : one of ``full``, ``mcp_only``, or ``pirag_only``.
    retrieval_kind : ``pirag`` or the standard-RAG ablation.

    Returns
    -------
    Dict with compatibility keys ``regulatory_guidance``, ``relevant_sops``,
    and ``risk_assessment`` plus source documents, query, and diagnostics.
    These legacy field names contain retrieved text; they do not assert legal,
    regulatory, or independently validated risk status.
    """
    context: Dict[str, Any] = {
        "regulatory_guidance": "",
        "relevant_sops": "",
        "risk_assessment": "",
        "source_documents": [],
        "query": "",
    }

    if context_mode not in {"full", "mcp_only", "pirag_only"}:
        raise ValueError(f"unsupported context_mode: {context_mode!r}")
    pipeline = None
    if context_mode != "mcp_only":
        pipeline = _get_pipeline()
        if pipeline is None:
            return context

    # Try the new role-specific context builder first
    try:
        from .context_builder import retrieve_role_context

        class _FakeObs:
            def __init__(self, rho, temp, rh, inv, y_hat, tau_val, hr, surplus):
                self.rho = rho
                self.temp = temp
                self.rh = rh
                self.inv = inv
                self.y_hat = y_hat
                self.tau = tau_val
                self.hour = hr
                self.surplus_ratio = surplus

        obs = _FakeObs(
            spoilage_risk, temperature, humidity, inventory, y_hat, tau,
            hour, surplus_ratio,
        )

        # Dispatch MCP tools for this role
        mcp_results: Dict[str, Any] = {}
        skip_mcp = context_mode == "pirag_only" or (
            scenario == "cyber_outage" and hour >= 24.0
        )
        if not skip_mcp:
            try:
                from .mcp.tool_dispatch import dispatch_tools
                from .mcp.registry import get_default_registry
                registry = get_default_registry()
                mcp_results = dispatch_tools(role, obs, registry)
            except Exception as _exc:
                handle_unexpected_failure(
                    f"context-provider MCP dispatch for role {role}", _exc, _log,
                )

        result = {}
        if context_mode != "mcp_only":
            result = retrieve_role_context(
                role, obs, scenario, mcp_results, pipeline,
                retrieval_kind=retrieval_kind,
            )

        context["regulatory_guidance"] = result.get("regulatory_guidance", "")
        context["relevant_sops"] = result.get("sop_guidance", "")
        context["risk_assessment"] = result.get("slca_guidance", "")
        context["source_documents"] = [c.get("doc_id", "") for c in result.get("citations", [])]
        context["query"] = result.get("query", "")
        context["mcp_results"] = mcp_results

        # Pass through additional fields, including the per-guard
        # breakdown so the live /decide path surfaces the same
        # diagnostics the simulator does.
        for key in ("waste_hierarchy_guidance", "governance_guidance",
                     "top_fused_score", "top_citation_score",
                     "top_rerank_score", "top_doc_id", "guards_passed",
                     "guard_breakdown", "evidence_hashes"):
            if key in result:
                context[key] = result[key]

        # Compute context modifier for callers who want it
        try:
            from .context_to_logits import compute_context_modifier
            modifier = compute_context_modifier(
                mcp_results, result, obs,
                context_mode=context_mode,
                retrieval_kind=retrieval_kind,
            )
            context["context_modifier"] = modifier.tolist()
        except Exception as _exc:
            handle_unexpected_failure(
                "context-provider logit-modifier calculation", _exc, _log,
            )
            context["context_modifier"] = None

        return context

    except ImportError as _exc:
        handle_unexpected_failure(
            "role-specific context-builder import", _exc, _log,
        )

    # Fallback: original implementation
    conditions = []
    if temperature > 8.0:
        conditions.append("high temperature excursion")
    if spoilage_risk > 0.3:
        conditions.append("elevated spoilage risk")
    if scenario == "heatwave":
        conditions.append("heatwave conditions")
    elif scenario == "cyber_outage":
        conditions.append("system outage contingency")
    elif scenario == "overproduction":
        conditions.append("surplus inventory management")

    query = "cold chain management guidelines for spinach"
    if conditions:
        query += " with " + " and ".join(conditions)
    context["query"] = query

    try:
        response = pipeline.ask(query, k=4, anchor_on_chain=False)

        for citation in response.citations:
            doc_source = citation.meta.get("source", citation.doc_id)
            context["source_documents"].append(doc_source)

            if "regulatory" in citation.doc_id.lower() or "fda" in citation.doc_id.lower():
                context["regulatory_guidance"] = citation.passage[:300]
            elif "sop" in citation.doc_id.lower() or "cold_chain" in citation.doc_id.lower():
                context["relevant_sops"] = citation.passage[:300]
            elif "slca" in citation.doc_id.lower():
                context["risk_assessment"] = citation.passage[:300]

        if not context["regulatory_guidance"] and response.citations:
            context["regulatory_guidance"] = response.citations[0].passage[:200]

    except Exception as _exc:
        handle_unexpected_failure(
            "legacy context-provider retrieval", _exc, _log,
        )

    return context
