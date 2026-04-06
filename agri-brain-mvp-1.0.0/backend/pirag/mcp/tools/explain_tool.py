"""MCP tool: generate a causal explanation for a routing decision.

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
    """Generate a causal explanation for a routing decision.

    Runs the full MCP dispatch + piRAG retrieval + explanation pipeline
    for the given conditions, producing a human-readable explanation
    with source citations, causal reasoning, and provenance.

    Returns
    -------
    Dict with summary, full_explanation, causal_chain, keywords,
    evidence_hashes, merkle_root, provenance_ready.
    """
    try:
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
            "causal_chain": result.get("causal_chain", {}),
            "keywords": result.get("keywords", {}),
            "evidence_hashes": result.get("evidence_hashes", [])[:5],
            "merkle_root": result.get("merkle_root", ""),
            "provenance_ready": result.get("provenance_ready", False),
        }

    except Exception as e:
        return {
            "summary": f"Explanation generation failed: {e}",
            "full_explanation": "",
            "error": str(e),
        }
