"""Lossless, behavior-neutral retrieval-ranking evidence tests."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
from pirag.agent_pipeline import Citation, PiRAGPipeline, PiRAGResponse
from pirag.context_to_logits import compute_context_modifier
from pirag.pyrag.hybrid_retriever import Document, HybridRetriever
from src.agents.coordinator import (
    AgentCoordinator,
    _build_retrieval_channel_evidence,
    _protocol_window,
)
from src.models.policy import Policy


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _obs() -> SimpleNamespace:
    return SimpleNamespace(
        rho=0.4,
        inv=12_000.0,
        temp=12.0,
        rh=90.0,
        y_hat=100.0,
        tau=0.0,
        hour=18.0,
        surplus_ratio=0.2,
        raw={},
    )


def test_hybrid_and_pipeline_expose_existing_score_components(monkeypatch) -> None:
    first = Document("doc-a", "alpha", {"source": "a.txt"})
    second = Document("doc-b", "beta", {"source": "b.txt"})
    retriever = HybridRetriever()
    monkeypatch.setattr(
        retriever.bm25,
        "search",
        lambda query, k: [(first, 8.0), (second, 4.0)],
    )
    monkeypatch.setattr(
        retriever,
        "_dense_search",
        lambda query, k: [(second, 0.9), (first, 0.5)],
    )

    hits = retriever.search("query", k=2)
    assert [hit["id"] for hit in hits] == ["doc-a", "doc-b"]
    assert hits[0]["fused_score"] == hits[0]["score"]
    assert hits[0]["sparse_score"] == 8.0
    assert hits[0]["dense_score"] == 0.5
    assert hits[0]["sparse_rrf"] == 1.0 / 61.0
    assert hits[0]["dense_rrf"] == 1.0 / 62.0
    assert hits[0]["score"] == (
        hits[0]["sparse_rrf"] + hits[0]["dense_rrf"]
    )

    class _Retriever:
        def __init__(self) -> None:
            self.calls = []

        def search(self, question, k):
            self.calls.append((question, k))
            return hits

    pipeline = object.__new__(PiRAGPipeline)
    pipeline.retriever = _Retriever()
    pipeline._answer_inference = lambda question, topk: "bounded answer"
    monkeypatch.setattr("pirag.agent_pipeline.units_consistent", lambda text: True)
    monkeypatch.setattr(
        "pirag.agent_pipeline.within_ranges", lambda text, constraints: True,
    )
    monkeypatch.setattr(
        "pirag.agent_pipeline.verify_with_sim", lambda text, context: None,
    )

    response = pipeline.ask("query", k=2, anchor_on_chain=False)
    assert pipeline.retriever.calls == [("query", 2)]
    citation = response.citations[0]
    assert citation.retrieval_rank == 1
    assert citation.fused_score == hits[0]["fused_score"]
    assert citation.sparse_rank == 1
    assert citation.sparse_score == 8.0
    assert citation.sparse_rrf == 1.0 / 61.0
    assert citation.dense_rank == 2
    assert citation.dense_score == 0.5
    assert citation.dense_rrf == 1.0 / 62.0
    assert citation.fusion == "rrf"
    assert response.retrieval_metadata == {
        "query": "query",
        "requested_k": 2,
        "planner_default_k": 6,
        "effective_k": 2,
        "returned_count": 2,
        "anchor_on_chain": False,
        "fusion_methods": ["rrf"],
    }
    assert response.guard_breakdown == {
        "unit": True,
        "feasibility": True,
        "simulator": None,
        "aggregate": False,
    }


def test_final_ranking_and_step_evidence_are_lossless_and_behavior_neutral(
    monkeypatch,
) -> None:
    from pirag import context_builder, physics_reranker
    from pirag.mcp.tools import spoilage_forecast

    passage_a = "Maintain spinach at 4 degrees Celsius during storage."
    passage_b = "Urgent spoilage diversion at 12 degrees Celsius."
    citation_a = Citation(
        doc_id="sop-a",
        passage=passage_a,
        sha256=_sha(passage_a),
        meta={"source": "a.txt"},
        score=0.030,
        retrieval_rank=1,
        fused_score=0.030,
        sparse_rank=1,
        sparse_score=9.5,
        sparse_rrf=1.0 / 61.0,
        dense_rank=2,
        dense_score=0.5,
        dense_rrf=1.0 / 62.0,
        fusion="rrf",
    )
    citation_b = Citation(
        doc_id="regulatory-b",
        passage=passage_b,
        sha256=_sha(passage_b),
        meta={"source": "b.txt"},
        score=0.026,
        retrieval_rank=2,
        fused_score=0.026,
        sparse_rank=2,
        sparse_score=7.0,
        sparse_rrf=1.0 / 62.0,
        dense_rank=1,
        dense_score=0.8,
        dense_rrf=1.0 / 61.0,
        fusion="rrf",
    )
    response = PiRAGResponse(
        answer="answer",
        citations=[citation_a, citation_b],
        guards_passed=True,
        evidence_hashes=[citation_a.sha256, citation_b.sha256],
        merkle_root="",
        chain_tx=None,
        guard_breakdown={
            "unit": True, "feasibility": True,
            "simulator": True, "aggregate": True,
        },
        retrieval_metadata={
            "query": "base expanded", "requested_k": 4,
            "effective_k": 4, "returned_count": 2,
        },
    )

    class _Pipeline:
        def __init__(self) -> None:
            self.calls = []

        def ask(self, query, **kwargs):
            self.calls.append((query, kwargs))
            return response

    pipeline = _Pipeline()
    monkeypatch.setattr(
        context_builder,
        "build_role_query",
        lambda *args, **kwargs: "base query",
    )
    monkeypatch.setattr(
        physics_reranker,
        "expand_query_with_physics",
        lambda base, rho, temperature, k_eff: base + " expanded",
    )
    monkeypatch.setattr(
        spoilage_forecast,
        "forecast_spoilage",
        lambda *args, **kwargs: {"k_effective": 0.006},
    )

    def _reverse_rerank(passages, *args, **kwargs):
        second = dict(passages[1])
        second.update({
            "score": 0.126,
            "rerank_score": 0.126,
            "lexical_bonus": 0.10,
            "arrhenius_consistency": 0.8,
            "physics_bonus": 0.08,
            "physics_consistency": 0.9,
        })
        first = dict(passages[0])
        first.update({
            "score": 0.031,
            "rerank_score": 0.031,
            "lexical_bonus": 0.001,
            "arrhenius_consistency": 1.0,
            "physics_bonus": 0.001,
            "physics_consistency": 1.0,
        })
        return [second, first]

    monkeypatch.setattr(
        physics_reranker, "lexical_arrhenius_rerank", _reverse_rerank,
    )
    context = context_builder.retrieve_role_context(
        "processor", _obs(), "baseline", {}, pipeline,
        retrieval_kind="pirag",
    )

    assert len(pipeline.calls) == 1
    assert pipeline.calls[0] == (
        "base query expanded", {"k": 4, "anchor_on_chain": False},
    )
    # Compatibility order is unchanged; the explicit final order is reversed.
    assert [item["doc_id"] for item in context["citations"]] == [
        "sop-a", "regulatory-b",
    ]
    ranked = context["ranked_citations"]
    assert [item["doc_id"] for item in ranked] == [
        "regulatory-b", "sop-a",
    ]
    top = ranked[0]
    assert top["rank"] == 1
    assert top["base_rank"] == 2
    assert top["raw_score"] == 0.026
    assert top["fused_score"] == 0.026
    assert top["rerank_score"] == 0.126
    assert top["raw_sparse_score"] == 7.0
    assert top["sparse_rrf"] == 1.0 / 62.0
    assert top["raw_dense_score"] == 0.8
    assert top["dense_rrf"] == 1.0 / 61.0
    assert top["lexical_bonus"] == 0.10
    assert top["arrhenius_consistency"] == 0.8
    assert top["content_sha256"] == citation_b.sha256
    assert top["document_sha256"]
    assert top["metadata_sha256"]
    assert "passage" not in top
    assert context["ranked_evidence_hashes"] == [
        citation_b.sha256, citation_a.sha256,
    ]
    assert context["query_transform_metadata"][
        "physics_expansion_changed_query"
    ] is True
    assert context["ranking_transform_metadata"]["rerank_executed"] is True
    assert context["pipeline_guard_decisions"]["breakdown"][
        "simulator"
    ] is True
    assert context["guard_decisions"]["retrieval_rrf_floor"][
        "evaluated_doc_id"
    ] == "regulatory-b"

    channel = _build_retrieval_channel_evidence(
        rag_context=context,
        integration_trace={
            "retrieval_gate": 1.0,
            "retrieval_blocked_reason": None,
            "temporal_scale": 0.9,
            "physics_scale": 0.8,
            "rag_total_scale": 0.72,
            "effective_psi": np.ones(5),
        },
        protocol_window=_protocol_window(None, (0, 0)),
        attempted=True,
        requested_kind="pirag",
    )
    assert channel["citation_order_source"] == (
        "rag_context.ranked_citations"
    )
    captured_top = channel["ordered_citations"][0]
    for field in (
        "raw_score", "fused_score", "rerank_score", "raw_sparse_score",
        "sparse_rank", "sparse_rrf", "raw_dense_score", "dense_rank",
        "dense_rrf", "lexical_bonus", "arrhenius_consistency",
        "document_sha256", "source_passage_sha256",
    ):
        assert captured_top[field] == top[field if field != "source_passage_sha256" else "content_sha256"]
    assert channel["evidence_hashes"] == [
        citation_b.sha256, citation_a.sha256,
    ]
    assert channel["source_order_evidence_hashes"] == [
        citation_a.sha256, citation_b.sha256,
    ]
    assert channel["query_transform_metadata"] == (
        context["query_transform_metadata"]
    )
    assert channel["guard_decisions"] == context["guard_decisions"]

    # The added exposure fields are observer-only: stripping all of them leaves
    # the exact context modifier unchanged.
    exposure_fields = {
        "ranked_citations", "ranked_evidence_hashes",
        "query_transform_metadata", "ranking_transform_metadata",
        "pipeline_retrieval_metadata", "pipeline_guard_decisions",
        "guard_decisions",
    }
    legacy_context = {
        key: value for key, value in context.items()
        if key not in exposure_fields
    }
    modifier_with_evidence = compute_context_modifier(
        {}, context, _obs(), retrieval_kind="pirag",
    )
    modifier_without_evidence = compute_context_modifier(
        {}, legacy_context, _obs(), retrieval_kind="pirag",
    )
    np.testing.assert_array_equal(
        modifier_with_evidence, modifier_without_evidence,
    )


def test_live_primary_and_cooperative_decision_capture_exactly_one_ranking(
    monkeypatch,
) -> None:
    """Exercise the complete coordinator boundary with the real pipeline."""

    import pirag.context_builder as context_builder
    import pirag.mcp.tool_dispatch as tool_dispatch

    original_dispatch = tool_dispatch.dispatch_tools
    original_retrieve = context_builder.retrieve_role_context
    dispatch_roles = []
    retrieval_roles = []

    def counted_dispatch(role, *args, **kwargs):
        dispatch_roles.append(role)
        return original_dispatch(role, *args, **kwargs)

    def counted_retrieve(role, *args, **kwargs):
        retrieval_roles.append(role)
        return original_retrieve(role, *args, **kwargs)

    monkeypatch.setattr(tool_dispatch, "dispatch_tools", counted_dispatch)
    monkeypatch.setattr(
        context_builder, "retrieve_role_context", counted_retrieve,
    )
    coordinator = AgentCoordinator(context_enabled=True, mode="agribrain")
    env = {
        "rho": 0.25,
        "inv": 12_000.0,
        "temp": 7.0,
        "rh": 88.0,
        "y_hat": 100.0,
        "tau": 0.0,
        "surplus_ratio": 0.8,
        "supply_hat": 12_000.0,
        "supply_std": 100.0,
        "demand_std": 5.0,
        "price_signal": 0.0,
    }
    _action, probabilities, _agent = coordinator.step(
        env, 18.0, "agribrain", Policy(), np.random.default_rng(901),
        "baseline",
    )

    assert dispatch_roles == ["processor", "cooperative"]
    assert retrieval_roles == ["processor", "cooperative"]
    assert np.isclose(np.sum(probabilities), 1.0)
    for section, expected_role in (
        (coordinator._step_channel_evidence["primary"], "processor"),
        (coordinator._step_channel_evidence["cooperative"], "cooperative"),
    ):
        retrieval = section["retrieval"]
        assert retrieval["citation_order_source"] == (
            "rag_context.ranked_citations"
        )
        assert retrieval["query_transform_metadata"]["role"] == expected_role
        assert retrieval["ranking_transform_metadata"][
            "final_count"
        ] == len(retrieval["ordered_citations"])
        assert retrieval["ordered_citations"]
        top = retrieval["ordered_citations"][0]
        assert top["rank"] == 1
        assert top["base_rank"] >= 1
        assert top["raw_score"] == top["fused_score"]
        assert top["rerank_score"] is not None
        assert top["raw_sparse_score"] is not None
        assert top["sparse_rrf"] is not None
        assert top["raw_dense_score"] is not None
        assert top["dense_rrf"] is not None
        assert top["source_passage_sha256"]
        assert top["document_sha256"]
        assert "passage" not in top
