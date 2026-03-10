"""RAG context provider for the decision pipeline.

Queries the piRAG knowledge base for relevant policy guidance based on
the current scenario conditions, spoilage risk, and temperature, then
returns a structured context dict for use in action selection and
explanation generation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

_PIPELINE = None


def _get_pipeline():
    """Lazy-initialize the PiRAG pipeline with knowledge base documents."""
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    try:
        from .agent_pipeline import PiRAGPipeline

        pipeline = PiRAGPipeline()

        # Ingest knowledge base documents
        kb_dir = Path(__file__).parent / "knowledge_base"
        if kb_dir.exists():
            docs = []
            for f in sorted(kb_dir.iterdir()):
                if f.suffix in (".txt", ".json", ".csv"):
                    text = f.read_text(encoding="utf-8").strip()
                    if text:
                        docs.append({
                            "id": f.stem,
                            "text": text,
                            "metadata": {"source": f.name},
                        })
            if docs:
                pipeline.ingest(docs)

        _PIPELINE = pipeline
    except Exception:
        _PIPELINE = None

    return _PIPELINE


def get_policy_context(
    scenario: str = "baseline",
    spoilage_risk: float = 0.0,
    temperature: float = 4.0,
) -> Dict[str, Any]:
    """Query the knowledge base for relevant policy guidance.

    Parameters
    ----------
    scenario : current scenario name.
    spoilage_risk : current spoilage risk (rho).
    temperature : current temperature in Celsius.

    Returns
    -------
    Dict with keys: regulatory_guidance, relevant_sops, risk_assessment,
    source_documents, and query used.
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

    # Build a contextual query based on conditions
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

        # Extract guidance from retrieved documents
        for citation in response.citations:
            doc_source = citation.meta.get("source", citation.doc_id)
            context["source_documents"].append(doc_source)

            if "regulatory" in citation.doc_id.lower() or "fda" in citation.doc_id.lower():
                context["regulatory_guidance"] = citation.passage[:300]
            elif "sop" in citation.doc_id.lower() or "cold_chain" in citation.doc_id.lower():
                context["relevant_sops"] = citation.passage[:300]
            elif "slca" in citation.doc_id.lower():
                context["risk_assessment"] = citation.passage[:300]

        # Fill any empty fields with general response
        if not context["regulatory_guidance"] and response.citations:
            context["regulatory_guidance"] = response.citations[0].passage[:200]

    except Exception:
        pass

    return context
