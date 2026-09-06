"""MCP tool: generate a policy-trace explanation for a routing decision.

Enables external systems to request human-readable, source-cited
explanations for any hypothetical or actual routing decision.
"""
from __future__ import annotations

from typing import Any, Dict


def explain(
    action: str = "local_redistribute",
    role: str = "farm",
    hour: float = 0.0,
    rho: float = 0.0,
    temperature: float = 4.0,
    humidity: float = 92.0,
    inventory: float = 10000.0,
    scenario: str = "baseline",
) -> Dict[str, Any]:
    """Generate a policy-trace explanation for a routing decision.

    Runs the in-process project MCP-style dispatch + piRAG retrieval + explanation pipeline
    for the given conditions, producing a human-readable explanation
    with knowledge-base references, feature attribution, and provenance.

    Returns
    -------
    Dict with summary, full_explanation, attribution_chain, keywords, and
    the complete ordered leaf inventory for a local Merkle commitment.
    """
    # Any failure here propagates so the MCP tools/call handler can mark
    # result.isError=True. Returning a success-shaped dict with an "error"
    # key (the previous behavior) made tool failures invisible to the
    # protocol-recorder counter and the MCP Tool Reliability figure.
    from pirag.mcp.registry import get_default_registry
    from pirag.mcp.tool_dispatch import dispatch_tools
    from pirag.context_builder import retrieve_role_context
    from pirag.context_to_logits import extract_context_features, compute_context_modifier
    from pirag.explain_decision import explain_decision
    from pirag.agent_pipeline import PiRAGPipeline

    # Build a minimal observation
    class _Obs:
        pass
    obs = _Obs()
    obs.rho = rho
    obs.temp = temperature
    obs.rh = humidity
    obs.inv = inventory
    obs.y_hat = 100.0
    obs.tau = 0.0
    obs.hour = hour
    obs.surplus_ratio = max(0.0, inventory / 12000.0 - 1.0)
    obs.raw = {"rho": rho, "temp": temperature, "rh": humidity, "inv": inventory}

    registry = get_default_registry()
    pipeline = PiRAGPipeline()

    # MCP dispatch
    mcp_results = dispatch_tools(role, obs, registry)

    # piRAG retrieval
    rag_context = retrieve_role_context(role, obs, scenario, mcp_results, pipeline, None)

    # Context features
    psi = extract_context_features(mcp_results, rag_context, obs)
    modifier = compute_context_modifier(mcp_results, rag_context, obs)

    # Generate explanation
    result = explain_decision(
        action=action, role=role, hour=hour, obs=obs,
        mcp_results=mcp_results, rag_context=rag_context,
        slca_score=0.0, carbon_kg=0.0, waste=0.0,
        context_features=psi, logit_adjustment=modifier,
        keywords=rag_context.get("keywords", {}),
    )

    return {
        "summary": result.get("summary", ""),
        "full_explanation": result.get("full_explanation", ""),
        "attribution_chain": result.get("attribution_chain", {}),
        "causal_chain": result.get("causal_chain", {}),
        "keywords": result.get("keywords", {}),
        "evidence_hashes": result.get("evidence_hashes", []),
        "retrieval_evidence_hashes": result.get(
            "retrieval_evidence_hashes", []
        ),
        "mcp_evidence_hashes": result.get("mcp_evidence_hashes", {}),
        "evidence_hash_count": result.get("evidence_hash_count", 0),
        "evidence_hashes_complete": result.get(
            "evidence_hashes_complete", False
        ),
        "merkle_root": result.get("merkle_root", ""),
        "commitment_type": result.get("commitment_type", "local_merkle_root"),
        "merkle_inclusion_paths_exposed": False,
        "merkle_root_anchored_on_chain": False,
        # Backward-compatible alias. This means only that a local root was
        # computed; it is not a complete proof or an on-chain anchor.
        "provenance_ready": result.get("provenance_ready", False),
    }
