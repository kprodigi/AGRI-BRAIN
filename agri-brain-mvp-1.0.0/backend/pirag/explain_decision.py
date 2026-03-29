"""Structured explanation generation for routing decisions.

Combines MCP tool evidence, piRAG citations, and physical state into
a human-readable decision explanation with full provenance chain
(evidence hashes from both piRAG and MCP tool calls).
"""
from __future__ import annotations

from typing import Any, Dict, List

from .provenance.hasher import hash_text, hash_artifact


def explain_decision(
    action: str,
    role: str,
    hour: float,
    obs: Any,
    mcp_results: Dict[str, Any],
    rag_context: Dict[str, Any],
    slca_score: float,
    carbon_kg: float,
    waste: float,
) -> Dict[str, Any]:
    """Generate a structured explanation for a routing decision.

    Parameters
    ----------
    action : selected action name (cold_chain, local_redistribute, recovery).
    role : active agent role.
    hour : simulation hour.
    obs : current Observation.
    mcp_results : results from MCP tool dispatch.
    rag_context : results from piRAG retrieval.
    slca_score : composite SLCA score for this decision.
    carbon_kg : carbon emissions for this decision.
    waste : waste rate for this decision.

    Returns
    -------
    Dict with summary, physical_basis, mcp_evidence, regulatory_context,
    social_performance, full_explanation, evidence_hashes, tools_invoked,
    citations, guards_passed, provenance_ready.
    """
    # Physical basis
    physical_basis = (
        f"At hour {hour:.1f}, spoilage risk rho={obs.rho:.3f}, "
        f"temperature {obs.temp:.1f}C, humidity {obs.rh:.1f}%, "
        f"inventory {obs.inv:.0f} units, "
        f"surplus ratio {obs.surplus_ratio:.2f}."
    )

    # MCP evidence
    tools_invoked = mcp_results.get("_tools_invoked", [])
    mcp_evidence_parts: List[str] = []
    mcp_hashes: List[str] = []

    compliance = mcp_results.get("check_compliance")
    if isinstance(compliance, dict):
        status = "compliant" if compliance.get("compliant") else "non-compliant"
        n_violations = len(compliance.get("violations", []))
        mcp_evidence_parts.append(f"Compliance: {status} ({n_violations} violations)")
        mcp_hashes.append(hash_artifact({"tool": "check_compliance", "result": compliance}))

    forecast = mcp_results.get("spoilage_forecast")
    if isinstance(forecast, dict):
        mcp_evidence_parts.append(f"Spoilage forecast: rho={forecast.get('forecast_rho', '?')} ({forecast.get('urgency', '?')})")
        mcp_hashes.append(hash_artifact({"tool": "spoilage_forecast", "result": forecast}))

    slca_data = mcp_results.get("slca_lookup")
    if isinstance(slca_data, dict):
        mcp_evidence_parts.append(f"SLCA data: product={slca_data.get('product_type', '?')}")
        mcp_hashes.append(hash_artifact({"tool": "slca_lookup", "result": slca_data}))

    mcp_evidence = "; ".join(mcp_evidence_parts) if mcp_evidence_parts else "No MCP tools invoked"

    # Regulatory context from piRAG
    regulatory_context = rag_context.get("regulatory_guidance", "") or rag_context.get("sop_guidance", "")
    if not regulatory_context:
        regulatory_context = "No regulatory context retrieved"

    # Social performance
    social_performance = (
        f"SLCA composite: {slca_score:.3f}, "
        f"carbon: {carbon_kg:.2f} kg, "
        f"waste: {waste:.4f}."
    )

    # Summary
    summary = (
        f"{role} agent selected {action} at hour {hour:.1f}. "
        f"Spoilage risk: {obs.rho:.3f}. "
        f"SLCA: {slca_score:.3f}. "
        f"{'Compliance violations detected. ' if isinstance(compliance, dict) and not compliance.get('compliant') else ''}"
        f"{'Spoilage forecast: ' + forecast.get('urgency', '') + '. ' if isinstance(forecast, dict) else ''}"
    )

    # Full explanation
    full_explanation = f"{summary}\n\nPhysical basis: {physical_basis}\n\nMCP evidence: {mcp_evidence}\n\nRegulatory context: {regulatory_context[:200]}\n\nSocial performance: {social_performance}"

    # Evidence hashes (piRAG + MCP)
    pirag_hashes = rag_context.get("evidence_hashes", [])
    all_hashes = pirag_hashes + mcp_hashes

    # Citations
    citations = rag_context.get("citations", [])

    return {
        "summary": summary.strip(),
        "physical_basis": physical_basis,
        "mcp_evidence": mcp_evidence,
        "regulatory_context": regulatory_context[:200],
        "social_performance": social_performance,
        "full_explanation": full_explanation,
        "evidence_hashes": all_hashes,
        "tools_invoked": tools_invoked,
        "citations": citations,
        "guards_passed": rag_context.get("guards_passed", True),
        "provenance_ready": len(all_hashes) > 0,
    }
