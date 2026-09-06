"""MCP tool: query the piR knowledge base.

Enables external AI systems to retrieve domain-specific guidance from
the AGRI-BRAIN knowledge base through the project's MCP-style interface.
"""
from __future__ import annotations

from typing import Any, Dict, List


# Singleton pipeline (lazy init)
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from pirag.agent_pipeline import PiRPipeline
        _pipeline = PiRPipeline()
    return _pipeline


def pirag_query(
    query: str = "cold-chain operating-envelope context for leafy greens",
    k: int = 4,
    role: str = "farm",
    temperature: float = 4.0,
    rho: float = 0.0,
    humidity: float = 92.0,
    physics_expansion: bool = True,
    physics_reranking: bool = True,
) -> Dict[str, Any]:
    """Query the piR knowledge base with optional physics-informed retrieval.

    Parameters
    ----------
    query : natural language query.
    k : number of documents to retrieve.
    role : agent role (affects query expansion).
    temperature : current temperature for physics expansion.
    rho : current spoilage risk for physics expansion.
    humidity : current humidity for physics reranking.
    physics_expansion : add physics terms to query based on T/rho.
    physics_reranking : rerank results using lexical and Arrhenius-consistency terms.

    Returns
    -------
    Dict with query (possibly expanded), results list, guards_passed, metadata.
    """
    pipeline = _get_pipeline()

    # Track which optional features were unavailable so the caller (and the
    # Tool Reliability figure) can distinguish "fully served" from "served
    # with degraded auxiliary capabilities". These are not errors — the
    # core retrieval still works — but they used to be invisible.
    degraded_features: List[str] = []

    expanded_query = query
    if physics_expansion:
        try:
            from pirag.physics_reranker import expand_query_with_physics
            expanded_query = expand_query_with_physics(query, rho, temperature)
        except ImportError:
            degraded_features.append("physics_expansion")

    # Retrieve
    response = pipeline.ask(expanded_query, k=k, anchor_on_chain=False)

    results: List[Dict[str, Any]] = []
    for citation in response.citations[:k]:
        # Use the real BM25/dense hybrid score that the retriever
        # actually computed (propagated through Citation.score). Earlier
        # revisions hardcoded 0.5 here, which made psi_2 (retrieval
        # score signal) and psi_3 (retrieved-policy signal) constant.
        entry: Dict[str, Any] = {
            "doc_id": citation.doc_id,
            "passage": citation.passage[:500],
            "score": float(getattr(citation, "score", 0.0)),
            "sha256": citation.sha256,
        }
        results.append(entry)

    # Physics reranking
    physics_k_eff = None
    if physics_reranking and results:
        try:
            from pirag.physics_reranker import physics_rerank
            from src.models.spoilage import arrhenius_k
            physics_k_eff = float(arrhenius_k(
                temperature,
                rh_frac=max(0.0, min(float(humidity) / 100.0, 1.0)),
            ))
            passages = [
                {
                    "text": r["passage"],
                    "score": r["score"],
                    "id": r["doc_id"],
                    "meta": {},
                    "sha256": r["sha256"],
                }
                for r in results
            ]
            reranked = physics_rerank(
                passages, temperature, rho, humidity, physics_k_eff,
            )
            results = [
                {
                    "doc_id": r["id"],
                    "passage": r["text"][:500],
                    "score": float(r.get("score", 0.0)),
                    "raw_rrf_score": float(
                        r.get("fused_score", r.get("score", 0.0))
                    ),
                    "sha256": r["sha256"],
                }
                for r in reranked
            ]
        except ImportError:
            degraded_features.append("physics_reranking")

    # Extract keywords from each result
    try:
        from pirag.keyword_extractor import extract_keywords
        for r in results:
            r["keywords"] = extract_keywords(r["passage"])
    except ImportError:
        degraded_features.append("keyword_extraction")

    # Apply the same three declared retrieval-context guards used by the
    # simulator. Non-empty results alone are not evidence that guards passed.
    from pirag.context_builder import DEFAULT_GUARD_CONSTRAINTS
    from pirag.guards.feasibility_guard import within_ranges
    from pirag.guards.retrieval_guard import retrieval_quality_ok
    from pirag.guards.unit_guard import units_consistent

    top_passage = str(results[0]["passage"]) if results else ""
    top_raw_score = float(
        results[0].get("raw_rrf_score", results[0].get("score", 0.0))
    ) if results else 0.0
    retrieval_ok = retrieval_quality_ok(results, top_raw_score)
    unit_ok = units_consistent(top_passage) if top_passage else True
    feasibility_ok = (
        within_ranges(top_passage, DEFAULT_GUARD_CONSTRAINTS)
        if top_passage else True
    )
    guard_breakdown = {
        "retrieval": bool(retrieval_ok),
        "unit": bool(unit_ok),
        "feasibility": bool(feasibility_ok),
    }

    payload: Dict[str, Any] = {
        "query": expanded_query,
        "original_query": query,
        "physics_expanded": physics_expansion and expanded_query != query,
        "results": results,
        "n_results": len(results),
        "guards_passed": all(guard_breakdown.values()),
        "guard_breakdown": guard_breakdown,
        "physics_k_eff_h_inv": physics_k_eff,
    }
    if degraded_features:
        payload["_status"] = "degraded"
        payload["_degraded_features"] = degraded_features
    else:
        payload["_status"] = "ok"
    return payload
