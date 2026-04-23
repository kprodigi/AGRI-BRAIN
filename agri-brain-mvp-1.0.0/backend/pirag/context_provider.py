"""RAG context provider for the decision pipeline.

Queries the piRAG knowledge base for relevant policy guidance based on
the current scenario conditions, spoilage risk, and temperature, then
returns a structured context dict for use in action selection and
explanation generation.

Supports both the original 3-parameter signature (backward-compatible)
and an extended signature with role, humidity, inventory, surplus, tau,
and hour parameters.
"""
from __future__ import annotations

import inspect
import logging
from pathlib import Path

_log = logging.getLogger(__name__)
from typing import Any, Dict, Optional

_PIPELINE = None


def _get_pipeline():
    """Lazy-initialize the PiRAG pipeline with knowledge base documents."""
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    try:
        from .agent_pipeline import PiRAGPipeline

        # PiRAGPipeline auto-ingests the knowledge_base/ directory on init
        pipeline = PiRAGPipeline()

        _PIPELINE = pipeline
    except Exception:
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
) -> Dict[str, Any]:
    """Query the knowledge base for relevant policy guidance.

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

    Returns
    -------
    Dict with keys: regulatory_guidance, relevant_sops, risk_assessment,
    source_documents, query, and (when new modules available) additional
    guidance fields.
    """
    context: Dict[str, Any] = {
        "regulatory_guidance": "",
        "relevant_sops": "",
        "risk_assessment": "",
        "source_documents": [],
        "query": "",
    }

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
                self.y_hat = 100.0
                self.tau = tau_val
                self.hour = hr
                self.surplus_ratio = surplus

        obs = _FakeObs(spoilage_risk, temperature, humidity, inventory, 100.0, tau, hour, surplus_ratio)

        # Dispatch MCP tools for this role
        mcp_results: Dict[str, Any] = {}
        try:
            from .mcp.tool_dispatch import dispatch_tools
            from .mcp.registry import get_default_registry
            registry = get_default_registry()
            mcp_results = dispatch_tools(role, obs, registry)
        except Exception as _exc:
            _log.debug("MCP dispatch in context provider skipped for role %s: %s", role, _exc)

        result = retrieve_role_context(role, obs, scenario, mcp_results, pipeline)

        context["regulatory_guidance"] = result.get("regulatory_guidance", "")
        context["relevant_sops"] = result.get("sop_guidance", "")
        context["risk_assessment"] = result.get("slca_guidance", "")
        context["source_documents"] = [c.get("doc_id", "") for c in result.get("citations", [])]
        context["query"] = result.get("query", "")
        context["mcp_results"] = mcp_results

        # Pass through additional fields
        for key in ("waste_hierarchy_guidance", "governance_guidance",
                     "top_citation_score", "top_doc_id", "guards_passed",
                     "evidence_hashes"):
            if key in result:
                context[key] = result[key]

        # Compute context modifier for callers who want it
        try:
            from .context_to_logits import compute_context_modifier
            modifier = compute_context_modifier(mcp_results, result, obs)
            context["context_modifier"] = modifier.tolist()
        except Exception:
            context["context_modifier"] = None

        return context

    except ImportError:
        pass

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

    query = f"cold chain management guidelines for spinach"
    if conditions:
        query += " with " + " and ".join(conditions)
    context["query"] = query

    try:
        response = pipeline.ask(query, k=3, anchor_on_chain=False)

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
        _log.debug("context provider outer path skipped: %s", _exc)

    return context
