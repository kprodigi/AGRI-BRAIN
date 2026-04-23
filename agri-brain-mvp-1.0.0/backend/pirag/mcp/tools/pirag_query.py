"""MCP tool: query the piRAG knowledge base.

Enables external AI systems to retrieve domain-specific guidance from
the AGRI-BRAIN knowledge base through the standard MCP protocol.
"""
from __future__ import annotations

from typing import Any, Dict, List


# Singleton pipeline (lazy init)
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from pirag.agent_pipeline import PiRAGPipeline
        _pipeline = PiRAGPipeline()
    return _pipeline


def pirag_query(
    query: str = "cold chain compliance for leafy greens",
    k: int = 4,
    role: str = "farm",
    temperature: float = 4.0,
    rho: float = 0.0,
    humidity: float = 92.0,
    physics_expansion: bool = True,
    physics_reranking: bool = True,
) -> Dict[str, Any]:
    """Query the piRAG knowledge base with optional physics-informed retrieval.

    Parameters
    ----------
    query : natural language query.
    k : number of documents to retrieve.
    role : agent role (affects query expansion).
    temperature : current temperature for physics expansion.
    rho : current spoilage risk for physics expansion.
    humidity : current humidity for physics reranking.
    physics_expansion : add physics terms to query based on T/rho.
    physics_reranking : rerank results by physics plausibility.

    Returns
    -------
    Dict with query (possibly expanded), results list, guards_passed, metadata.
    """
    pipeline = _get_pipeline()

    expanded_query = query
    if physics_expansion:
        try:
            from pirag.physics_reranker import expand_query_with_physics
            expanded_query = expand_query_with_physics(query, rho, temperature)
        except ImportError:
            pass

    # Retrieve
    response = pipeline.ask(expanded_query, k=k, anchor_on_chain=False)

    results: List[Dict[str, Any]] = []
    for citation in response.citations[:k]:
        entry: Dict[str, Any] = {
            "doc_id": citation.doc_id,
            "passage": citation.passage[:500],
            "score": 0.5,
            "sha256": citation.sha256,
        }
        results.append(entry)

    # Physics reranking
    if physics_reranking and results:
        try:
            from pirag.physics_reranker import physics_rerank
            passages = [{"text": r["passage"], "score": r["score"],
                         "id": r["doc_id"], "meta": {}} for r in results]
            reranked = physics_rerank(passages, temperature, rho, humidity)
            results = [
                {"doc_id": r["id"], "passage": r["text"][:500],
                 "score": r.get("score", 0.5), "sha256": ""}
                for r in reranked
            ]
        except ImportError:
            pass

    # Extract keywords from each result
    try:
        from pirag.keyword_extractor import extract_keywords
        for r in results:
            r["keywords"] = extract_keywords(r["passage"])
    except ImportError:
        pass

    return {
        "query": expanded_query,
        "original_query": query,
        "physics_expanded": physics_expansion and expanded_query != query,
        "results": results,
        "n_results": len(results),
        "guards_passed": len(results) > 0,
    }
