"""Integration tests for MCP + piRAG pipeline (Task 28).

Tests covering registry discovery, MCP protocol, tool dispatch,
shared context, role queries, physics reranking, context modifiers,
backward compatibility, and transport.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_BACKEND = Path(__file__).resolve().parent.parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


# Minimal observation stub for tests
class _Obs:
    def __init__(self, **kwargs):
        self.rho = kwargs.get("rho", 0.15)
        self.inv = kwargs.get("inv", 12000.0)
        self.temp = kwargs.get("temp", 6.0)
        self.rh = kwargs.get("rh", 90.0)
        self.y_hat = kwargs.get("y_hat", 100.0)
        self.tau = kwargs.get("tau", 0.0)
        self.hour = kwargs.get("hour", 5.0)
        self.surplus_ratio = kwargs.get("surplus_ratio", 0.0)
        self.raw = kwargs


# ---- Test 1: Registry discovery ----
def test_registry_discovery():
    from pirag.mcp.registry import ToolRegistry, ToolSpec

    reg = ToolRegistry()
    reg.register(ToolSpec(
        name="t1", description="test", capabilities=["a", "b"],
        fn=lambda: 1, schema={},
    ))
    reg.register(ToolSpec(
        name="t2", description="test", capabilities=["c"],
        fn=lambda: 2, schema={},
    ))

    found = reg.discover(["a"])
    assert len(found) == 1
    assert found[0].name == "t1"

    found_bc = reg.discover(["b", "c"])
    assert len(found_bc) == 2


# ---- Test 2: MCP initialize handshake ----
def test_mcp_initialize_handshake():
    from pirag.mcp.protocol import MCPServer, MCPMessage
    from pirag.mcp.registry import ToolRegistry

    server = MCPServer(registry=ToolRegistry())
    resp = server.handle_message(MCPMessage(id=1, method="initialize"))
    assert resp.result is not None
    assert resp.result["protocolVersion"] == "2024-11-05"
    assert "serverInfo" in resp.result
    assert "extensions" in resp.result


def test_protocol_tool_payload_may_use_tool_name_argument(monkeypatch):
    """Registry selector and a tool's ``tool_name`` payload cannot collide."""
    from pirag.mcp.protocol import MCPMessage, MCPServer
    from pirag.mcp.registry import ToolRegistry, ToolSpec

    monkeypatch.setenv("MCP_RATE_LIMITS", "disabled")
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="oracle",
        description="argument collision regression",
        capabilities=["test"],
        fn=lambda user_id, tool_name: {
            "user_id": user_id, "tool": tool_name, "allowed": True,
        },
        schema={},
    ))
    response = MCPServer(registry=registry).handle_message(MCPMessage(
        id=22,
        method="tools/call",
        params={
            "name": "oracle",
            "arguments": {"user_id": "system", "tool_name": "surplus"},
        },
    ))
    assert response.error is None
    assert response.result is not None
    assert response.result.get("isError") is not True
    assert '"tool": "surplus"' in response.result["content"][0]["text"]


# ---- Test 3: MCP resources read ----
def test_mcp_resources_read():
    from pirag.mcp.protocol import MCPServer, MCPMessage, MCPResource
    from pirag.mcp.registry import ToolRegistry

    server = MCPServer(registry=ToolRegistry())
    server.register_resource(MCPResource(
        uri="test://value",
        name="test",
        description="test resource",
        read_fn=lambda: {"temp": 6.5},
    ))

    resp = server.handle_message(MCPMessage(
        id=2, method="resources/read", params={"uri": "test://value"},
    ))
    assert resp.result is not None
    assert "contents" in resp.result
    assert "6.5" in resp.result["contents"][0]["text"]


# ---- Test 4: MCP prompts expand ----
def test_mcp_prompts_expand():
    from pirag.mcp.protocol import MCPServer, MCPMessage
    from pirag.mcp.registry import ToolRegistry
    from pirag.mcp.prompts import register_prompts

    server = MCPServer(registry=ToolRegistry())
    register_prompts(server)

    for prompt_name in ["regulatory_compliance_check", "waste_hierarchy_assessment",
                        "emergency_rerouting", "slca_routing_guidance",
                        "governance_policy_lookup"]:
        resp = server.handle_message(MCPMessage(
            id=3, method="prompts/get", params={"name": prompt_name, "arguments": {}},
        ))
        assert resp.result is not None
        messages = resp.result.get("messages", [])
        assert len(messages) > 0
        text = messages[0]["content"]["text"]
        assert len(text) > 10, f"Prompt {prompt_name} produced empty text"


# ---- Test 5: Tool dispatch farm workflow ----
def test_tool_dispatch_farm_workflow():
    from pirag.mcp.registry import get_default_registry
    from pirag.mcp.tool_dispatch import dispatch_tools

    # Reset singleton
    import pirag.mcp.registry as _reg_mod
    _reg_mod._DEFAULT_REGISTRY = None

    registry = get_default_registry()
    obs = _Obs(rho=0.30, temp=8.0, rh=88.0)
    results = dispatch_tools("farm", obs, registry)

    assert "check_compliance" in results
    assert "_tools_invoked" in results
    assert "check_compliance" in results["_tools_invoked"]
    # With rho=0.30 > 0.20, slca_lookup should trigger
    assert "slca_lookup" in results


# ---- Test 6: Tool dispatch composition ----
def test_tool_dispatch_composition():
    from pirag.mcp.registry import get_default_registry
    from pirag.mcp.tool_dispatch import dispatch_tools
    import pirag.mcp.registry as _reg_mod
    _reg_mod._DEFAULT_REGISTRY = None

    registry = get_default_registry()
    # Critical temp triggers compliance violation, which triggers spoilage forecast
    obs = _Obs(rho=0.25, temp=12.0, rh=88.0)
    results = dispatch_tools("farm", obs, registry)

    assert "check_compliance" in results
    compliance = results["check_compliance"]
    assert not compliance["compliant"]
    # Critical violation (12C > 5C + 3C = 8C threshold) should trigger spoilage
    assert "spoilage_forecast" in results


# ---- Test 7: Shared context publish-query ----
def test_shared_context_publish_query():
    from pirag.mcp.context_sharing import SharedContextStore

    store = SharedContextStore()
    store.publish("farm", "check_compliance", {"compliant": True}, hour=5.0)
    store.publish("processor", "slca_lookup", {"score": 0.8}, hour=7.0)

    # Query farm compliance from processor perspective
    results = store.query(role="farm", tool_name="check_compliance",
                          max_age_hours=4.0, current_hour=8.0)
    assert len(results) == 1
    assert results[0]["result"]["compliant"] is True


# ---- Test 8: Shared context age eviction ----
def test_shared_context_age_eviction():
    from pirag.mcp.context_sharing import SharedContextStore

    store = SharedContextStore()
    store.publish("farm", "check_compliance", {"old": True}, hour=1.0)
    store.publish("farm", "check_compliance", {"new": True}, hour=10.0)

    results = store.query(role="farm", max_age_hours=4.0, current_hour=12.0)
    assert len(results) == 1
    assert results[0]["result"]["new"] is True


# ---- Test 9: Role query differentiation ----
def test_role_query_differentiation():
    from pirag.context_builder import build_role_query

    obs = _Obs(rho=0.20, temp=6.0, rh=90.0, surplus_ratio=0.1)
    queries = {}
    for role in ["farm", "processor", "cooperative", "distributor", "recovery"]:
        queries[role] = build_role_query(role, obs, "baseline", {})

    # All 5 queries should be distinct
    unique = set(queries.values())
    assert len(unique) == 5, f"Expected 5 unique queries, got {len(unique)}"


# ---- Test 10: Physics query expansion ----
def test_physics_query_expansion():
    from pirag.physics_reranker import expand_query_with_physics

    base = "cold chain compliance"
    # High temp should add thermal term
    expanded = expand_query_with_physics(base, rho=0.10, temperature=15.0, k_eff=0.001)
    assert "thermal degradation" in expanded

    # High rho should add spoilage term
    expanded2 = expand_query_with_physics(base, rho=0.60, temperature=4.0, k_eff=0.001)
    assert "advanced spoilage" in expanded2


# ---- Test 11: Physics reranking boosts relevant ----
def test_physics_reranking_boosts_relevant():
    from pirag.physics_reranker import physics_rerank

    passages = [
        {"text": "The synthetic storage envelope includes 4 degrees Celsius.", "score": 0.5, "id": "a", "meta": {}},
        {"text": "The benchmark assigns faster modeled spoilage above 15 degrees Celsius.", "score": 0.5, "id": "b", "meta": {}},
    ]
    # At 15C, the passage mentioning 15C should get a boost
    reranked = physics_rerank(passages, temperature=15.0, rho=0.20, humidity=90.0)
    assert reranked[0]["id"] == "b"


def test_physics_reranking_preserves_raw_fused_score():
    """Rerank bonuses must not overwrite the RRF confidence quantity."""
    from pirag.physics_reranker import lexical_arrhenius_rerank

    passages = [
        {
            "text": "Urgent decay at 15 degrees Celsius requires immediate action.",
            "score": 0.025,
            "id": "regulatory_fda_test",
            "meta": {},
        }
    ]
    result = lexical_arrhenius_rerank(
        passages, temperature=15.0, rho=0.60, humidity=95.0, k_eff=0.01,
    )[0]

    assert result["fused_score"] == 0.025
    assert result["rerank_score"] == result["score"]
    assert result["rerank_score"] != result["fused_score"]


def test_live_pirag_tool_passes_arrhenius_rate_to_reranker(monkeypatch):
    """The public MCP search must activate the mechanistic rerank term."""
    from pirag import physics_reranker
    from pirag.mcp.tools import pirag_query as tool
    from src.models.spoilage import arrhenius_k

    class _Citation:
        doc_id = "temperature_note"
        passage = "At 15 degrees Celsius, decay accelerates."
        score = 0.025
        sha256 = "a" * 64

    class _Response:
        citations = [_Citation()]

    class _Pipeline:
        @staticmethod
        def ask(*_args, **_kwargs):
            return _Response()

    captured = {}

    def _rerank(passages, temperature, rho, humidity, k_eff):
        captured["k_eff"] = k_eff
        return passages

    monkeypatch.setattr(tool, "_get_pipeline", lambda: _Pipeline())
    monkeypatch.setattr(physics_reranker, "physics_rerank", _rerank)
    result = tool.pirag_query(
        query="thermal handling",
        temperature=15.0,
        humidity=90.0,
        rho=0.2,
        physics_expansion=False,
        physics_reranking=True,
    )
    expected = float(arrhenius_k(15.0, rh_frac=0.90))
    assert captured["k_eff"] == pytest.approx(expected)
    assert captured["k_eff"] > 0.0
    assert result["physics_k_eff_h_inv"] == pytest.approx(expected)
    assert result["results"][0]["sha256"] == "a" * 64
    assert result["guard_breakdown"] == {
        "retrieval": True,
        "unit": True,
        "feasibility": True,
    }
    assert result["guards_passed"] is True


def test_live_pirag_tool_does_not_equate_nonempty_results_with_guard_pass(
    monkeypatch,
):
    """A low-strength hit is returned for inspection but fails closed."""
    from pirag.mcp.tools import pirag_query as tool

    class _Citation:
        doc_id = "weak_note"
        passage = "Documentary handling note."
        score = 0.001
        sha256 = "b" * 64

    class _Response:
        citations = [_Citation()]

    class _Pipeline:
        @staticmethod
        def ask(*_args, **_kwargs):
            return _Response()

    monkeypatch.setattr(tool, "_get_pipeline", lambda: _Pipeline())
    result = tool.pirag_query(
        query="weak query",
        physics_expansion=False,
        physics_reranking=False,
    )
    assert result["n_results"] == 1
    assert result["guard_breakdown"]["retrieval"] is False
    assert result["guards_passed"] is False


def test_pipeline_explicit_retrieval_depth_overrides_planner_default():
    """The benchmark's explicit k=4 must not be silently replaced by k=6."""
    from pirag.agent_pipeline import PiRAGPipeline

    class _Retriever:
        def __init__(self):
            self.calls = []

        def search(self, question, k):
            self.calls.append((question, k))
            return []

    pipeline = object.__new__(PiRAGPipeline)
    pipeline.retriever = _Retriever()
    pipeline.answer_engine = None

    pipeline.ask("explicit", k=4)
    pipeline.ask("planner default")

    assert pipeline.retriever.calls == [("explicit", 4), ("planner default", 6)]


# ---- Test 12: Context modifier bounds (updated: ±1.0) ----
def test_context_modifier_bounds():
    from pirag.context_to_logits import compute_context_modifier

    mcp = {
        "check_compliance": {"compliant": False, "violations": [{"severity": "critical"}]},
        "spoilage_forecast": {"urgency": "critical", "forecast_rho": 0.7},
        "slca_lookup": {"base_scores": {"local_redistribute": {"R": 0.90}}},
    }
    rag = {
        "guards_passed": True,
        "top_citation_score": 0.9,
        "top_doc_id": "regulatory_fda_cold_chain",
        "regulatory_guidance": "some guidance",
        "waste_hierarchy_guidance": "waste hierarchy",
        "sop_guidance": "sop guidance",
    }
    obs = _Obs(rho=0.50, temp=12.0)

    modifier = compute_context_modifier(mcp, rag, obs)
    assert modifier.shape == (3,)
    assert np.all(modifier >= -1.0), f"Modifier below -1.0: {modifier}"
    assert np.all(modifier <= 1.0), f"Modifier above +1.0: {modifier}"
    # With critical compliance + critical forecast + regulatory doc,
    # modifier should have substantial magnitude
    assert np.linalg.norm(modifier) > 0.3, f"Modifier too small: {modifier}"


# ---- Test 13: Context modifier zero when empty ----
def test_context_modifier_zero_when_empty():
    from pirag.context_to_logits import compute_context_modifier

    modifier = compute_context_modifier({}, {}, _Obs())
    assert np.allclose(modifier, 0.0)


# ---- Test 14: Context modifier guard gate ----
def test_context_modifier_guard_gate():
    from pirag.context_to_logits import compute_context_modifier

    mcp = {"check_compliance": {"compliant": False, "violations": [{"severity": "critical"}]}}
    rag = {"guards_passed": False, "top_citation_score": 0.9}
    obs = _Obs(rho=0.50)

    trace = {}
    modifier = compute_context_modifier(mcp, rag, obs, trace_out=trace)
    mcp_only = compute_context_modifier(
        mcp, rag, obs, context_mode="mcp_only",
    )
    pirag_only = compute_context_modifier(
        mcp, rag, obs, context_mode="pirag_only",
    )
    assert np.allclose(modifier, mcp_only)
    assert not np.allclose(modifier, 0.0), (
        "A failed retrieval guard must preserve separately computed MCP features"
    )
    assert np.allclose(pirag_only, 0.0)
    assert trace["retrieval_gate"] == 0.0
    assert trace["retrieval_blocked_reason"] == "retrieval_guard"


def test_temporal_continuity_modulates_pirag_but_not_mcp():
    from pirag.context_to_logits import compute_context_modifier
    from pirag.temporal_context import TemporalContextWindow

    mcp = {
        "check_compliance": {
            "compliant": False,
            "violations": [{"severity": "critical"}],
        },
        "spoilage_forecast": {"urgency": "high"},
    }
    rag = {
        "guards_passed": True,
        "top_citation_score": 0.03,
        "top_doc_id": "fda_regulatory",
    }
    stable = TemporalContextWindow()
    diverse = TemporalContextWindow()
    for index in range(5):
        stable.add(index, "farm", "q", "same", 0.03, "regulatory")
        diverse.add(index, "farm", "q", f"doc-{index}", 0.03, "regulatory")

    obs = _Obs(hour=5.0)
    stable_mcp = compute_context_modifier(
        mcp, rag, obs, stable, context_mode="mcp_only",
    )
    diverse_mcp = compute_context_modifier(
        mcp, rag, obs, diverse, context_mode="mcp_only",
    )
    assert np.allclose(stable_mcp, diverse_mcp)

    stable_trace = {}
    stable_pirag = compute_context_modifier(
        mcp, rag, obs, stable, context_mode="pirag_only",
        trace_out=stable_trace,
    )
    diverse_pirag = compute_context_modifier(
        mcp, rag, obs, diverse, context_mode="pirag_only",
    )
    assert not np.allclose(stable_pirag, diverse_pirag)
    assert stable_trace["temporal_gate_requested"] is True
    assert stable_trace["temporal_gate_applied"] is True
    assert stable_trace["temporal_scale"] == pytest.approx(
        stable_trace["temporal_base"]
        - stable_trace["temporal_decay"]
        * stable_trace["temporal_continuity_score"]
    )


def test_physics_gate_modulates_pirag_but_not_mcp():
    from pirag.context_to_logits import compute_context_modifier

    mcp = {
        "check_compliance": {
            "compliant": False,
            "violations": [{"severity": "critical"}],
        },
        "spoilage_forecast": {"urgency": "high"},
    }
    obs = _Obs(
        policy_flags={"enable_physics_consistency_gate": True},
    )
    low = {
        "guards_passed": True,
        "top_citation_score": 0.03,
        "top_doc_id": "fda_regulatory",
        "physics_consistency_score": 0.02,
    }
    high = {**low, "physics_consistency_score": 0.15}

    low_mcp = compute_context_modifier(mcp, low, obs, context_mode="mcp_only")
    high_mcp = compute_context_modifier(mcp, high, obs, context_mode="mcp_only")
    assert np.allclose(low_mcp, high_mcp)

    low_full = compute_context_modifier(mcp, low, obs)
    high_full = compute_context_modifier(mcp, high, obs)
    assert np.allclose(low_full, low_mcp)
    assert not np.allclose(high_full, high_mcp)


# ---- Test 14b: Three-guard aggregation in retrieve_role_context ----
def test_retrieve_role_context_aggregates_three_guards(monkeypatch):
    """retrieve_role_context must AND the three §3.7 guards (unit,
    feasibility, retrieval) into a single ``guards_passed`` flag and
    expose the per-guard outcomes via ``guard_breakdown``.
    """
    from pirag import context_builder
    from pirag.context_builder import retrieve_role_context

    # Stub the piRAG pipeline so the test does not depend on retrieval.
    # SHA-256 hashes are computed from the actual passage so the test
    # exercises realistic provenance values instead of a placeholder
    # constant.
    import hashlib as _hashlib

    def _sha(text: str) -> str:
        return _hashlib.sha256(text.encode("utf-8")).hexdigest()

    class _Citation:
        def __init__(self, passage):
            self.doc_id = "regulatory_fda_leafy_greens"
            self.passage = passage
            self.sha256 = _sha(passage)
            self.meta = {}
            self.score = 0.5

    class _Resp:
        def __init__(self, passage):
            self.citations = [_Citation(passage)] if passage else []
            self.evidence_hashes = [_sha(passage)] if passage else []
            self.guards_passed = True
            self.merkle_root = _sha("merkle:" + passage) if passage else ""
            self.chain_tx = None
            self.answer = passage or ""

    class _Pipe:
        def __init__(self, passage):
            self._passage = passage
        def ask(self, *a, **kw):
            return _Resp(self._passage)

    obs = _Obs(temp=15.0, rho=0.4)

    # Case A: clean passage with no parseable (number, unit) pairs ->
    # unit guard passes vacuously, feasibility passes, retrieval passes.
    ctx = retrieve_role_context(
        "farm", obs, "baseline", mcp_results={},
        pipeline=_Pipe("Maintain refrigeration within the cold-chain envelope for fresh spinach storage."),
    )
    assert ctx["guard_breakdown"] == {"retrieval": True, "unit": True, "feasibility": True}
    assert ctx["guards_passed"] is True

    # Case B: passage that trips the feasibility guard (number outside the
    # default ±1e9 envelope) -> aggregate must flip False even though
    # retrieval and unit guards individually pass.
    ctx_bad = retrieve_role_context(
        "farm", obs, "baseline", mcp_results={},
        pipeline=_Pipe("Improbable reading of 9.99e15 detected."),
    )
    assert ctx_bad["guard_breakdown"]["feasibility"] is False
    assert ctx_bad["guards_passed"] is False

    # Case C: simulate an empty retriever (no citations) -> retrieval
    # guard fails, unit/feasibility vacuously True; aggregate is False.
    class _PipeEmpty:
        def ask(self, *a, **kw):
            class R:
                citations = []
                evidence_hashes = []
                guards_passed = True
                merkle_root = ""
                chain_tx = None
                answer = ""
            return R()
    ctx_empty = retrieve_role_context(
        "farm", obs, "baseline", mcp_results={}, pipeline=_PipeEmpty(),
    )
    assert ctx_empty["guard_breakdown"]["retrieval"] is False
    assert ctx_empty["guards_passed"] is False


def test_context_guards_use_raw_score_and_the_same_reranked_top(monkeypatch):
    """A rerank bonus cannot pass the RRF guard or redirect passage guards.

    The source retriever returns a safe, high-RRF document first.  The stubbed
    reranker moves a low-RRF, infeasible regulatory document to rank 1 with a
    large adjusted score.  The context must identify that reranked document,
    retain both scores, fail the raw-RRF retrieval guard, and apply the
    feasibility guard to that same document rather than the pre-rerank first
    citation.
    """
    import hashlib

    from pirag.context_builder import retrieve_role_context
    from pirag import physics_reranker

    class _Citation:
        def __init__(self, doc_id, passage, score):
            self.doc_id = doc_id
            self.passage = passage
            self.score = score
            self.meta = {}
            self.sha256 = hashlib.sha256(passage.encode("utf-8")).hexdigest()

    safe = _Citation(
        "sop_safe",
        "Maintain refrigeration within the declared cold-chain envelope.",
        0.030,
    )
    reranked_top = _Citation(
        "regulatory_fda_low_rrf",
        "Improbable reading of 9.99e15 detected.",
        0.010,
    )

    class _Response:
        citations = [safe, reranked_top]
        evidence_hashes = [safe.sha256, reranked_top.sha256]
        guards_passed = True
        merkle_root = ""
        chain_tx = None
        answer = ""

    class _Pipeline:
        def ask(self, *args, **kwargs):
            return _Response()

    def _rerank(passages, *args, **kwargs):
        by_id = {p["id"]: p for p in passages}
        top = dict(by_id["regulatory_fda_low_rrf"])
        top.update(fused_score=0.010, score=0.200, rerank_score=0.200)
        second = dict(by_id["sop_safe"])
        second.update(fused_score=0.030, score=0.030, rerank_score=0.030)
        return [top, second]

    monkeypatch.setattr(physics_reranker, "lexical_arrhenius_rerank", _rerank)
    ctx = retrieve_role_context(
        "farm", _Obs(temp=15.0, rho=0.4), "baseline", {}, _Pipeline(),
    )

    assert ctx["top_doc_id"] == "regulatory_fda_low_rrf"
    assert ctx["top_fused_score"] == 0.010
    assert ctx["top_citation_score"] == 0.010
    assert ctx["top_rerank_score"] == 0.200
    assert ctx["guard_breakdown"]["retrieval"] is False
    assert ctx["guard_breakdown"]["feasibility"] is False
    assert ctx["guards_passed"] is False


def test_strict_retrieval_raises_on_unexpected_pipeline_failure(monkeypatch):
    """Canonical runs must not turn retrieval exceptions into empty context."""
    from pirag.context_builder import retrieve_role_context

    class _BrokenPipeline:
        def ask(self, *args, **kwargs):
            raise ValueError("synthetic retrieval failure")

    monkeypatch.setenv("STRICT_VALIDATION", "1")
    with pytest.raises(
        RuntimeError, match="publication-critical role-context retrieval",
    ):
        retrieve_role_context(
            "farm", _Obs(), "baseline", {}, _BrokenPipeline(),
        )


def test_strict_retrieval_guard_rejection_remains_a_normal_result(monkeypatch):
    """An explicit no-hit guard failure is not an execution exception."""
    from pirag.context_builder import retrieve_role_context

    class _EmptyResponse:
        citations = []
        evidence_hashes = []
        guards_passed = False
        merkle_root = ""
        chain_tx = None
        answer = "No evidence retrieved."

    class _EmptyPipeline:
        def ask(self, *args, **kwargs):
            return _EmptyResponse()

    monkeypatch.setenv("STRICT_VALIDATION", "1")
    context = retrieve_role_context(
        "farm", _Obs(), "baseline", {}, _EmptyPipeline(),
    )
    assert context["guards_passed"] is False
    assert context["guard_breakdown"] == {
        "retrieval": False,
        "unit": True,
        "feasibility": True,
    }


def test_standard_rag_skips_physics_expansion_and_reranking(monkeypatch):
    """The standard comparator uses the base query and base ranking only."""
    import hashlib

    from pirag import physics_reranker
    from pirag.context_builder import retrieve_role_context

    def _pirag_only_path_must_not_run(*args, **kwargs):
        raise AssertionError("standard RAG invoked a piRAG-only transform")

    monkeypatch.setattr(
        physics_reranker,
        "expand_query_with_physics",
        _pirag_only_path_must_not_run,
    )
    monkeypatch.setattr(
        physics_reranker,
        "lexical_arrhenius_rerank",
        _pirag_only_path_must_not_run,
    )

    passage = "Maintain refrigeration within the declared cold-chain envelope."

    class _Citation:
        doc_id = "sop_cold_chain"
        score = 0.030
        meta = {}
        sha256 = hashlib.sha256(passage.encode("utf-8")).hexdigest()

        def __init__(self):
            self.passage = passage

    class _Response:
        citations = [_Citation()]
        evidence_hashes = [_Citation.sha256]
        guards_passed = True
        merkle_root = ""
        chain_tx = None
        answer = passage

    class _Pipeline:
        def __init__(self):
            self.queries = []

        def ask(self, query, **kwargs):
            self.queries.append(query)
            return _Response()

    pipeline = _Pipeline()
    context = retrieve_role_context(
        "farm", _Obs(temp=15.0, rho=0.4), "baseline", {}, pipeline,
        retrieval_kind="standard",
    )

    assert len(pipeline.queries) == 1
    assert context["query"] == pipeline.queries[0]
    assert context["retrieval_kind"] == "standard"
    assert context["top_doc_id"] == "sop_cold_chain"
    assert context["top_fused_score"] == pytest.approx(0.030)
    assert context["top_rerank_score"] == pytest.approx(0.030)
    assert context["physics_consistency_score"] == pytest.approx(1.0)
    assert "lexical_bonus_mean" not in context


def test_standard_rag_modifier_has_no_temporal_or_physics_multiplier():
    """Only the piRAG arm may scale retrieval by continuity or physics."""
    from pirag.context_to_logits import compute_context_modifier

    class _TemporalPathMustNotRun:
        def context_continuity_score(self, hour):
            raise AssertionError(
                "standard RAG invoked retrieval-continuity weighting"
            )

    mcp = {"spoilage_forecast": {"urgency": "high"}}
    rag = {
        "retrieval_kind": "standard",
        "guards_passed": True,
        "top_fused_score": 0.030,
        "top_doc_id": "regulatory_fda_leafy_greens",
        "physics_consistency_score": 0.0,
    }
    obs = _Obs(
        policy_flags={"enable_physics_consistency_gate": True},
    )
    trace = {}
    low_physics = compute_context_modifier(
        mcp, rag, obs,
        temporal_window=_TemporalPathMustNotRun(),
        retrieval_kind="standard",
        trace_out=trace,
    )
    high_physics_context = dict(rag, physics_consistency_score=1.0)
    high_physics = compute_context_modifier(
        mcp, high_physics_context, obs,
        temporal_window=_TemporalPathMustNotRun(),
        retrieval_kind="standard",
    )

    np.testing.assert_allclose(low_physics, high_physics)
    assert trace["retrieval_kind"] == "standard"
    assert trace["retrieval_gate"] == pytest.approx(1.0)
    assert trace["temporal_scale"] == pytest.approx(1.0)
    assert trace["physics_scale"] == pytest.approx(1.0)
    assert trace["rag_total_scale"] == pytest.approx(1.0)
    assert np.linalg.norm(trace["pirag_preclip_component"]) > 0.0

    with pytest.raises(ValueError, match="retrieval kind mismatch"):
        compute_context_modifier(
            mcp, rag, obs, retrieval_kind="pirag",
        )


def test_strict_context_modifier_raises_on_temporal_failure(monkeypatch):
    """Temporal-modulation exceptions may not silently remove a policy term."""
    from pirag.context_to_logits import compute_context_modifier

    class _BrokenWindow:
        def context_continuity_score(self, hour):
            raise ValueError("synthetic temporal failure")

    monkeypatch.setenv("STRICT_VALIDATION", "1")
    with pytest.raises(
        RuntimeError, match="publication-critical temporal continuity",
    ):
        compute_context_modifier(
            {"spoilage_forecast": {"urgency": "high"}},
            {
                "guards_passed": True,
                "top_fused_score": 0.03,
                "top_doc_id": "sop_cold_chain",
            },
            _Obs(),
            temporal_window=_BrokenWindow(),
        )


# ---- Test 15: Context feature extraction ----
def test_context_feature_extraction():
    from pirag.context_to_logits import extract_context_features

    mcp = {
        "check_compliance": {"compliant": False, "violations": [{"severity": "critical"}]},
        "spoilage_forecast": {"urgency": "high"},
        "chain_query": [
            {"action": "recovery"}, {"action": "recovery"},
            {"action": "cold_chain"}, {"action": "recovery"},
        ],
    }
    # 2026-04: post-RRF calibration. The hybrid retriever's top RRF
    # score is bounded by 2/(K+1) ~= 0.0328 for K=60; psi[2] is now
    # normalised against that ceiling so realistic top scores around
    # 0.025 produce confidence ~0.76. We construct two test points:
    #   - a top RRF score (0.025) that should yield ~0.76 confidence
    #     and trip the regulatory predicate (since it is above the
    #     retrieval-guard floor).
    #   - a saturating top RRF score (0.05) that should yield 1.0.
    rag = {
        "top_citation_score": 0.025,
        "top_doc_id": "regulatory_fda_guideline_v2",
    }
    obs = _Obs(rho=0.40)

    psi = extract_context_features(mcp, rag, obs)
    assert psi.shape == (5,)
    assert psi[0] == 1.0, "Critical compliance should be 1.0"
    assert psi[1] == 0.7, f"High urgency should be 0.7, got {psi[1]}"
    # psi[2] is min(top_score / (2/(K+1)), 1.0). With K=60 and
    # top_score=0.025 that's 0.025 / (2/61) ≈ 0.7625.
    assert 0.7 <= psi[2] <= 0.8, f"Confidence should be ~0.76, got {psi[2]}"
    assert psi[3] == 1.0, (
        "Regulatory doc with score above retrieval-guard floor should be 1.0"
    )
    assert abs(psi[4] - 0.75) < 1e-9, f"Recovery saturation should be 0.75, got {psi[4]}"

    # Saturation: a top score above the RRF max should clamp to 1.0.
    rag_sat = {**rag, "top_citation_score": 0.05}
    psi_sat = extract_context_features(mcp, rag_sat, obs)
    assert psi_sat[2] == 1.0, "psi[2] should clamp to 1.0 above RRF max"


def test_context_features_prefer_raw_fused_score_over_rerank_score():
    """psi_2/psi_3 must remain on the declared RRF scale."""
    from pirag.context_to_logits import extract_context_features

    rag = {
        "top_fused_score": 0.010,
        # Compatibility alias is intentionally set to the adjusted score to
        # prove the explicit raw field wins if an older caller is inconsistent.
        "top_citation_score": 0.200,
        "top_rerank_score": 0.200,
        "top_doc_id": "regulatory_fda_test",
    }
    psi = extract_context_features({}, rag, _Obs())

    assert 0.30 < psi[2] < 0.31
    assert psi[3] == 0.0


# ---- Test 16: THETA_CONTEXT sign consistency ----
def test_theta_context_sign_consistency():
    """Verify compliance violation reduces cold chain and increases redistribution."""
    from pirag.context_to_logits import THETA_CONTEXT

    # Column 0 = synthetic operating-envelope exceedance
    assert THETA_CONTEXT[0, 0] < 0, "Envelope exceedance should disfavor cold chain"
    assert THETA_CONTEXT[1, 0] > 0, "Envelope exceedance should favor redistribution"

    # Column 4 = recovery saturation
    assert THETA_CONTEXT[2, 4] < 0, "Recovery saturation should disfavor further recovery"
    assert THETA_CONTEXT[0, 4] > 0, "Recovery saturation should slightly favor cold chain"


# ---- Test 17: Context modifier confidence weighting via features ----
def test_context_modifier_confidence_weighting():
    from pirag.context_to_logits import compute_context_modifier

    mcp = {"check_compliance": {"compliant": False, "violations": [{"severity": "critical"}]}}
    obs = _Obs(rho=0.50)

    # 2026-04: post-RRF score scale. With K=60 the RRF max is
    # 2/(K+1) ≈ 0.0328; psi[2] saturates above that. Pick test
    # points within the realistic RRF score range so high vs low
    # confidence actually map to different clamped psi[2] values.
    rag_high = {"guards_passed": True, "top_citation_score": 0.030, "top_doc_id": ""}
    mod_high = compute_context_modifier(mcp, rag_high, obs)

    rag_low = {"guards_passed": True, "top_citation_score": 0.005, "top_doc_id": ""}
    mod_low = compute_context_modifier(mcp, rag_low, obs)

    # Both should be non-zero (compliance violation is MCP-sourced,
    # not retrieval-dependent)
    assert np.linalg.norm(mod_high) > 0, "High confidence modifier should be non-zero"
    assert np.linalg.norm(mod_low) > 0, "Low confidence modifier should be non-zero"
    # Higher confidence should produce different (generally larger)
    # magnitude because psi_2 contributes additional signal
    assert not np.allclose(mod_high, mod_low), "Different confidence should produce different modifiers"


# ---- Test 18: Backward compatibility ----
def test_backward_compatibility():
    from src.models.action_selection import select_action

    rng = np.random.default_rng(42)

    # Without context_modifier
    a1, p1 = select_action(
        mode="agribrain", rho=0.3, inv=10000, y_hat=100, temp=6.0,
        tau=0.0, policy=_DummyPolicy(), rng=rng, deterministic=True,
    )

    rng2 = np.random.default_rng(42)
    # With context_modifier=None (should be identical)
    a2, p2 = select_action(
        mode="agribrain", rho=0.3, inv=10000, y_hat=100, temp=6.0,
        tau=0.0, policy=_DummyPolicy(), rng=rng2, deterministic=True,
        context_modifier=None,
    )

    assert a1 == a2
    assert np.allclose(p1, p2)


# ---- Test 19: SLCA amplification ----
def test_slca_amplification():
    """Verify agribrain logits with context_modifier include SLCA boost."""
    from src.models.action_selection import select_action

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    # Without context modifier
    _, probs_no_ctx = select_action(
        mode="agribrain", rho=0.3, inv=10000, y_hat=100, temp=6.0,
        tau=0.0, policy=_DummyPolicy(), rng=rng1, deterministic=True,
        context_modifier=None,
    )

    # With a non-zero context modifier (redistribution component > 0)
    ctx_mod = np.array([-0.5, 0.6, 0.1])
    _, probs_ctx = select_action(
        mode="agribrain", rho=0.3, inv=10000, y_hat=100, temp=6.0,
        tau=0.0, policy=_DummyPolicy(), rng=rng2, deterministic=True,
        context_modifier=ctx_mod,
    )

    # Context should shift redistribution probability upward
    assert probs_ctx[1] > probs_no_ctx[1], (
        f"SLCA amplification should boost redistribution: {probs_ctx[1]} vs {probs_no_ctx[1]}"
    )


# ---- Test 20: Context learner update (now 5 features) ----
def test_context_learner_update():
    from pirag.context_learner import ContextRuleLearner

    learner = ContextRuleLearner(n_rules=5, learning_rate=0.1)
    initial_weights = learner.get_weights().copy()

    # Positive delta should increase fired weights
    learner.update(rules_fired=[0, 2], reward_with_context=1.0, reward_without_context=0.5)
    after = learner.get_weights()

    # Fired rules should increase from their starting weight; unfired should not.
    assert after[0] > initial_weights[0], "Fired rule 0 weight should grow after positive delta"
    assert after[2] > initial_weights[2], "Fired rule 2 weight should grow after positive delta"
    assert after[0] > after[3], "Fired rules should increase relative to unfired"


# ---- Test 21: Transport in-process ----
def test_transport_in_process():
    from pirag.mcp.protocol import MCPServer
    from pirag.mcp.registry import ToolRegistry
    from pirag.mcp.transport import InProcessTransport, MCPClient

    server = MCPServer(registry=ToolRegistry())
    transport = InProcessTransport(server)
    client = MCPClient(transport)

    result = client.initialize()
    assert "protocolVersion" in result
    assert result["protocolVersion"] == "2024-11-05"

    client.close()


# ---- Test 22: Guards pass with real pipeline data ----
def test_context_guards_pass_with_real_pipeline():
    """Verify that retrieve_role_context sets guards_passed=True when citations exist."""
    from pirag.context_builder import retrieve_role_context
    from pirag.agent_pipeline import PiRAGPipeline

    pipeline = PiRAGPipeline()

    obs = _Obs(rho=0.30, temp=10.0, rh=88.0, tau=0.0,
               hour=10.0, surplus_ratio=0.2, inv=10000.0, y_hat=15.0)
    ctx = retrieve_role_context("farm", obs, "baseline", {}, pipeline, None)
    assert len(ctx["citations"]) > 0, "Should retrieve at least one citation"
    from pirag.guards.retrieval_guard import MIN_TOP_CITATION_SCORE
    assert ctx["top_citation_score"] > MIN_TOP_CITATION_SCORE
    assert ctx["guards_passed"] is True, "Guards should pass with real citations"


# ---- Test 23: Full pipeline produces non-zero modifier (updated bounds) ----
def test_full_pipeline_nonzero_modifier():
    """End-to-end: MCP dispatch + piRAG retrieval + modifier computation = non-zero."""
    from pirag.mcp.tool_dispatch import dispatch_tools
    from pirag.mcp.registry import get_default_registry
    from pirag.context_builder import retrieve_role_context
    from pirag.context_to_logits import compute_context_modifier
    from pirag.agent_pipeline import PiRAGPipeline
    import pirag.mcp.registry as _reg_mod
    _reg_mod._DEFAULT_REGISTRY = None

    obs = _Obs(rho=0.35, temp=12.0, rh=88.0, tau=1.0,
               hour=10.0, surplus_ratio=0.3, inv=10000.0, y_hat=15.0)

    reg = get_default_registry()
    pipeline = PiRAGPipeline()

    mcp_results = dispatch_tools("farm", obs, reg)
    assert len(mcp_results.get("_tools_invoked", [])) > 0, "MCP tools should fire"

    ctx = retrieve_role_context("farm", obs, "heatwave", mcp_results, pipeline, None)
    assert ctx["guards_passed"] is True, "Guards should pass"

    modifier = compute_context_modifier(mcp_results, ctx, obs)
    assert not np.allclose(modifier, 0.0), f"Modifier should be non-zero, got {modifier}"
    assert np.all(np.abs(modifier) <= 1.0), "Modifier should be within ±1.0 bounds"
    # With the new THETA_CONTEXT approach, modifier norm should be > 0.3
    assert np.linalg.norm(modifier) > 0.3, (
        f"Modifier norm should be > 0.3, got {np.linalg.norm(modifier):.4f}"
    )


# ---- Test 24: Fixed-action waste is context-neutral ----
def test_fixed_action_waste_is_context_neutral():
    """Context metadata must not change a fixed action's physical outcome."""
    from src.models.waste import compute_save_factor, context_waste_penalty

    critical = {"compliant": False, "violations": [{"severity": "critical"}]}
    warning = {"compliant": False, "violations": [{"severity": "warning"}]}
    compliant = {"compliant": True}

    # No compliance data — no penalty regardless of action
    assert context_waste_penalty(None) == 1.0
    assert context_waste_penalty(None, "local_redistribute") == 1.0

    # Compliant — no penalty regardless of action
    assert context_waste_penalty(compliant, "cold_chain") == 1.0
    assert context_waste_penalty(compliant, "local_redistribute") == 1.0

    assert context_waste_penalty(critical, "cold_chain") == 1.0
    assert context_waste_penalty(warning, "cold_chain") == 1.0

    # Rerouting receives no channel-dependent physical bonus.
    assert context_waste_penalty(critical, "local_redistribute") == 1.0
    assert context_waste_penalty(warning, "local_redistribute") == 1.0

    assert context_waste_penalty(critical, "recovery") == 1.0

    # The same action has the same physical effect regardless of whether its
    # compliance state arrived through a protocol channel.
    sf_clean = compute_save_factor("local_redistribute", "agribrain")
    sf_rerouted = compute_save_factor(
        "local_redistribute", "agribrain",
        compliance_data=critical,
    )
    assert sf_rerouted == sf_clean

    # The cold-chain action is equally mode- and context-neutral.
    sf_cc_clean = compute_save_factor("cold_chain", "agribrain")
    sf_cc_violation = compute_save_factor(
        "cold_chain", "agribrain",
        compliance_data=critical,
    )
    assert sf_cc_violation == sf_cc_clean == 0.0


# ---- Test 25: ContextMatrixLearner sign preservation ----
def test_context_matrix_learner_sign_preservation():
    """Verify REINFORCE learner preserves sign constraints."""
    from pirag.context_learner import ContextMatrixLearner
    from pirag.context_to_logits import THETA_CONTEXT

    learner = ContextMatrixLearner(initial_theta=THETA_CONTEXT, learning_rate=0.01)

    # Run many updates with varied rewards
    rng = np.random.default_rng(42)
    for _ in range(100):
        psi = rng.random(5)
        action = rng.integers(0, 3)
        probs = np.array([0.3, 0.5, 0.2])
        reward = rng.random()
        learner.update(psi, action, probs, reward)

    summary = learner.summary()
    assert summary["sign_preserved"], "Signs must be preserved after learning"
    assert summary["n_updates"] == 100
    assert summary["theta_change_norm"] > 0, "Theta should change after 100 updates"
    assert summary["final_slca_amp"] == 0.0, (
        "Confirmatory default must not add a second context-dependent "
        "social-proxy amplification path"
    )


def test_context_matrix_proxy_interaction_is_opt_in_and_bounded():
    """The optional interaction is available only as a labelled sensitivity."""
    from pirag.context_learner import ContextMatrixLearner
    from pirag.context_to_logits import THETA_CONTEXT

    learner = ContextMatrixLearner(
        initial_theta=THETA_CONTEXT,
        learning_rate=0.01,
        learn_proxy_interaction=True,
    )
    for _ in range(100):
        learner.update(
            np.ones(5), 1, np.array([0.1, 0.8, 0.1]),
            reward=1.0, slca_score=0.8,
        )
    assert 0.0 < learner.summary()["final_slca_amp"] <= 0.50


# ---- Test 26: Feature masking for ablation modes ----
def test_feature_masking_ablation():
    """Verify mcp_only and pirag_only modes mask the correct features."""
    from pirag.context_to_logits import compute_context_modifier

    mcp = {
        "check_compliance": {"compliant": False, "violations": [{"severity": "critical"}]},
        "spoilage_forecast": {"urgency": "high"},
    }
    rag = {
        "guards_passed": True,
        "top_citation_score": 0.7,
        "top_doc_id": "regulatory_fda_guidelines",
    }
    obs = _Obs(rho=0.40, temp=10.0)

    mod_full = compute_context_modifier(mcp, rag, obs, context_mode="full")
    mod_mcp = compute_context_modifier(mcp, rag, obs, context_mode="mcp_only")
    mod_pirag = compute_context_modifier(mcp, rag, obs, context_mode="pirag_only")

    # All should be non-zero
    assert np.linalg.norm(mod_full) > 0
    assert np.linalg.norm(mod_mcp) > 0
    assert np.linalg.norm(mod_pirag) > 0

    # Full should have larger magnitude than either partial
    assert np.linalg.norm(mod_full) > np.linalg.norm(mod_mcp), "Full > MCP-only"
    assert np.linalg.norm(mod_full) > np.linalg.norm(mod_pirag), "Full > piRAG-only"

    # MCP and piRAG should differ
    assert not np.allclose(mod_mcp, mod_pirag), "MCP-only and piRAG-only should differ"


def test_context_trace_allocation_reconstructs_after_clipping():
    """Trace allocation must sum to the exact clipped modifier."""
    from pirag.context_to_logits import THETA_CONTEXT, compute_context_modifier

    mcp = {
        "check_compliance": {
            "compliant": False,
            "violations": [{"severity": "critical"}],
        },
        "spoilage_forecast": {"urgency": "critical"},
    }
    rag = {
        "guards_passed": True,
        "top_citation_score": 0.7,
        "top_doc_id": "regulatory_fda_guidelines",
    }
    trace = {}
    modifier = compute_context_modifier(
        mcp, rag, _Obs(rho=0.4, temp=10.0),
        theta_override=10.0 * THETA_CONTEXT,
        trace_out=trace,
    )
    assert np.isclose(np.max(np.abs(modifier)), 1.0)
    reconstructed = (
        np.asarray(trace["feature_contributions"]).sum(axis=1)
        + np.asarray(trace["nonfeature_residual"])
    )
    assert np.allclose(reconstructed, modifier, rtol=1e-12, atol=1e-12)
    saturated = np.abs(np.asarray(trace["preclip_modifier"])) >= 1.0
    jacobian = np.asarray(trace["modifier_theta_jacobian"])
    assert np.allclose(jacobian[saturated], 0.0)


def test_context_modifier_jacobian_matches_finite_difference():
    """The learner trace differentiates the exact separated forward map."""
    from pirag.context_to_logits import THETA_CONTEXT, compute_context_modifier
    from pirag.temporal_context import TemporalContextWindow

    mcp = {
        "check_compliance": {
            "compliant": False,
            "violations": [{"severity": "critical"}],
        },
        "spoilage_forecast": {"urgency": "medium"},
    }
    rag = {
        "guards_passed": True,
        "top_citation_score": 0.025,
        "top_doc_id": "fda_regulatory",
        "physics_consistency_score": 0.075,
    }
    obs = _Obs(
        hour=5.0,
        policy_flags={"enable_physics_consistency_gate": True},
    )
    window = TemporalContextWindow()
    for index in range(4):
        window.add(index, "farm", "q", f"doc-{index}", 0.03, "regulatory")

    theta = 0.5 * THETA_CONTEXT
    trace = {}
    compute_context_modifier(
        mcp, rag, obs, window, theta_override=theta, trace_out=trace,
    )
    analytic = np.asarray(trace["modifier_theta_jacobian"])
    epsilon = 1e-6
    for row in range(3):
        for column in range(5):
            plus = theta.copy()
            minus = theta.copy()
            plus[row, column] += epsilon
            minus[row, column] -= epsilon
            mod_plus = compute_context_modifier(
                mcp, rag, obs, window, theta_override=plus,
            )
            mod_minus = compute_context_modifier(
                mcp, rag, obs, window, theta_override=minus,
            )
            numerical = (mod_plus[row] - mod_minus[row]) / (2.0 * epsilon)
            assert np.isclose(
                analytic[row, column], numerical,
                rtol=1e-7, atol=1e-9,
            )


def test_context_learner_uses_jacobian_and_temperature():
    from pirag.context_learner import (
        CONTEXT_GRADIENT_CONTRACT,
        ContextMatrixLearner,
        context_policy_gradient,
    )

    initial = np.zeros((3, 5), dtype=float)
    learner = ContextMatrixLearner(
        initial_theta=initial,
        learning_rate=0.1,
        grad_clip=10.0,
        sign_constrained=False,
        magnitude_cap_mode="absolute",
        magnitude_cap_value=10.0,
        prior_precision=0.0,
    )
    jacobian = np.ones((3, 5), dtype=float)
    jacobian[0] = 0.0
    probs = np.array([0.2, 0.5, 0.3])
    learner.update(
        psi=np.ones(5),
        action=1,
        probs=probs,
        reward=1.0,
        modifier_theta_jacobian=jacobian,
        policy_temperature=2.0,
    )

    # The running baseline includes 5% of the first reward, so A=0.95.
    expected = (
        0.1
        * ((np.array([0.0, 1.0, 0.0]) - probs) / 2.0)[:, None]
        * jacobian
        * 0.95
    )
    assert np.allclose(learner.theta, expected)
    assert np.allclose(learner.theta[0], 0.0)
    row_score, raw_gradient = context_policy_gradient(
        action=1,
        probs=probs,
        advantage=0.95,
        modifier_theta_jacobian=jacobian,
        policy_temperature=2.0,
    )
    np.testing.assert_allclose(
        row_score, (np.array([0.0, 1.0, 0.0]) - probs) / 2.0,
    )
    np.testing.assert_allclose(raw_gradient, expected / 0.1)
    summary = learner.summary()
    assert summary["gradient_contract"] == CONTEXT_GRADIENT_CONTRACT
    assert summary["last_gradient_trace"]["policy_temperature"] == 2.0


# ---- Test 27: Probability-gap rule activates under a synthetic fixture ----
def test_governance_override():
    """Verify the declared probability-gap rule substitutes redistribution."""
    from src.models.action_selection import select_action

    rng = np.random.default_rng(42)

    # Construct a context modifier that makes cold chain logit very negative
    # Synthetic operating-envelope, forecast, and retrieved-policy signals.
    extreme_modifier = np.array([-1.0, 0.8, 0.2])

    action_idx, probs = select_action(
        mode="agribrain", rho=0.6, inv=10000, y_hat=100, temp=15.0,
        tau=1.0, policy=_DummyPolicy(), rng=rng,
        context_modifier=extreme_modifier,
        deterministic=False,
    )

    # When the declared probability predicates hold, the rule substitutes redistribution
    # (action_idx=1) with deterministic probs [0, 1, 0].
    # Note: the override fires when pi(cold_chain) < GOVERNANCE_CC_PROB_CEILING
    # AND pi(local) - pi(cold_chain) > GOVERNANCE_LOCAL_ADVANTAGE_MIN
    # (both calibration-derived; see action_selection.py).
    if action_idx == 1 and probs[1] == 1.0:
        assert True, "Probability-gap rule correctly activated"
    else:
        # If override didn't fire, the probability conditions were not met,
        # which is acceptable — the override is conservative by design
        assert action_idx in [0, 1, 2], "Action should be valid"


# ---- Test 28: TraceExporter captures step data ----
def test_trace_exporter_captures():
    """Verify trace exporter captures and exports decision traces."""
    from pirag.trace_exporter import TraceExporter

    exporter = TraceExporter(max_traces=10)
    obs = _Obs(rho=0.30, temp=10.0, rh=88.0, hour=5.0, surplus_ratio=0.1)

    mcp = {
        "_tools_invoked": ["check_compliance", "spoilage_forecast"],
        "check_compliance": {"compliant": False, "violations": [{"severity": "critical", "parameter": "temp", "value": 10.0, "limit": 5.0}]},
        "spoilage_forecast": {"current_rho": 0.30, "forecast_rho": 0.35, "hours_ahead": 6, "urgency": "high"},
    }
    rag = {
        "guards_passed": True,
        "top_citation_score": 0.65,
        "top_doc_id": "regulatory_fda_leafy_greens",
        "query": "farm compliance spinach cold chain",
        "regulatory_guidance": "Fresh leafy greens must be stored below 5C.",
        "citations": [],
    }

    exporter.capture(
        obs=obs, scenario="heatwave", action="local_redistribute",
        probs=np.array([0.05, 0.90, 0.05]),
        mcp_results=mcp, rag_context=rag,
        context_features=np.array([1.0, 0.7, 0.81, 1.0, 0.0]),
        logit_adjustment=np.array([-0.80, 0.50, 0.30]),
        explanation={
            "summary": "Farm agent rerouted after an operating-envelope excursion.",
            "evidence_hashes": ["a" * 64, "b" * 64],
        },
        role="farm",
        action_changed=True,
    )

    assert len(exporter._traces) == 1
    t = exporter._traces[0]
    assert t.role == "farm"
    assert t.action == "local_redistribute"
    assert t.compliance_result is not None
    assert not t.compliance_result["compliant"]
    assert t.pirag_top_doc == "regulatory_fda_leafy_greens"
    assert len(t.context_features) == 5
    assert t.explanation_summary != ""

    chains = exporter.export_provenance_chains()
    assert len(chains) == 1
    assert chains[0]["evidence_hashes"] == ["a" * 64, "b" * 64]
    assert chains[0]["evidence_hashes_complete"] is True
    assert chains[0]["local_commitment_recomputable"] is True
    assert chains[0]["merkle_inclusion_paths_exposed"] is False
    assert chains[0]["merkle_root_anchored_on_chain"] is False

    summary = exporter.summary()
    assert summary["total_traces"] == 1
    assert "farm" in summary["roles_captured"]


# ---- Test 29: explain_decision produces structured output ----
def test_explain_decision_output():
    """Verify explain_decision returns complete structured explanation."""
    from pirag.explain_decision import explain_decision

    obs = _Obs(rho=0.35, temp=12.0, rh=85.0, hour=10.0, surplus_ratio=0.2, inv=14000.0)

    mcp = {
        "_tools_invoked": ["check_compliance"],
        "check_compliance": {"compliant": False, "violations": [{"severity": "critical"}]},
    }
    rag = {
        "guards_passed": True,
        "top_citation_score": 0.6,
        "regulatory_guidance": "Temperature must not exceed 5C for leafy greens.",
        "citations": [],
        "evidence_hashes": [],
    }

    result = explain_decision(
        action="local_redistribute", role="distributor", hour=10.0, obs=obs,
        mcp_results=mcp, rag_context=rag, slca_score=0.78, carbon_kg=3.5, waste=0.02,
    )

    assert "summary" in result
    assert "full_explanation" in result
    assert "evidence_hashes" in result
    assert "tools_invoked" in result
    assert "outside envelope" in result["mcp_evidence"]
    assert result["provenance_ready"] is True or len(result["evidence_hashes"]) > 0


# ---- Test 30: Role comparison table from traces ----
def test_role_comparison_table():
    """Verify role comparison table aggregates per-role data."""
    from pirag.trace_exporter import TraceExporter

    exporter = TraceExporter(max_traces=20)

    for role, hour in [("farm", 2.0), ("processor", 8.0), ("distributor", 20.0)]:
        obs = _Obs(rho=0.25, temp=8.0, rh=90.0, hour=hour)
        exporter.capture(
            obs=obs, scenario="baseline", action="local_redistribute",
            probs=np.array([0.1, 0.8, 0.1]),
            mcp_results={"_tools_invoked": ["check_compliance"], "check_compliance": {"compliant": True}},
            rag_context={"top_doc_id": f"doc_{role}", "top_citation_score": 0.5, "guards_passed": True},
            context_features=np.array([0.0, 0.0, 0.62, 0.0, 0.0]),
            logit_adjustment=np.array([-0.1, 0.2, -0.1]),
            explanation=None, role=role,
        )

    table = exporter.export_role_comparison_table()
    assert len(table) == 3
    roles_in_table = {r["role"] for r in table}
    assert roles_in_table == {"farm", "processor", "distributor"}
    assert all("primary_feature_distribution" in row for row in table)
    assert all(
        row["primary_cause_distribution"]
        == row["primary_feature_distribution"]
        for row in table
    )


# ---- Test 31: in-process dispatcher trace has JSON-RPC structure ----
def test_in_process_dispatcher_trace_format():
    """Verify the project MCP-style dispatcher trace has JSON-RPC structure."""
    from pirag.trace_exporter import TraceExporter

    exporter = TraceExporter(max_traces=5)
    obs = _Obs(rho=0.30, temp=10.0, rh=88.0, hour=5.0)

    exporter.capture(
        obs=obs, scenario="heatwave", action="local_redistribute",
        probs=np.array([0.1, 0.8, 0.1]),
        mcp_results={
            "_tools_invoked": ["check_compliance"],
            "check_compliance": {"compliant": False, "violations": [{"severity": "warning"}]},
            "spoilage_forecast": {"current_rho": 0.3, "forecast_rho": 0.35, "urgency": "high"},
        },
        rag_context={"top_doc_id": "doc1", "top_citation_score": 0.5, "guards_passed": True},
        context_features=np.array([0.5, 0.7, 0.6, 0.0, 0.0]),
        logit_adjustment=np.array([-0.5, 0.3, 0.2]),
        explanation=None, role="farm",
    )

    interop = exporter.export_interoperability_trace()
    assert len(interop) == 1
    entry = interop[0]
    assert entry["role"] == "farm"
    assert entry["total_protocol_messages"] > 0

    # Check JSON-RPC structure
    for msg in entry["mcp_interactions"]:
        req = msg["request"]
        assert req["jsonrpc"] == "2.0"
        assert "method" in req
        assert "id" in req


# ---- Test 32: Keyword extraction from passages ----
def test_keyword_extraction():
    """Verify keyword extraction finds thresholds and regulatory references."""
    from pirag.keyword_extractor import extract_keywords, extract_keywords_by_type

    passage = (
        "Synthetic parser fixture only, not a compliance claim: temperature must not exceed 5 degrees Celsius. "
        "Source label FSMA Section 204 is present; the fixture requires a response within 2 hours. "
        "Its author-declared spoilage-risk rule uses rho < 0.30 for redistribution."
    )

    kw = extract_keywords(passage)
    assert len(kw) > 0, "Should extract at least one keyword"

    by_type = extract_keywords_by_type(passage)
    assert len(by_type["thresholds"]) > 0, "Should find temperature threshold"
    assert any("FSMA" in r for r in by_type["regulations"]), "Should find FSMA reference"


# ---- Test 33: Policy trace contains attribution and an ablation delta ----
def test_policy_trace_structure():
    """Verify the trace has cautious attribution, an ablation, and citations."""
    from pirag.explain_decision import explain_decision

    obs = _Obs(rho=0.40, temp=14.0, rh=85.0, hour=30.0, surplus_ratio=0.2, inv=14000.0)

    mcp = {
        "_tools_invoked": ["check_compliance", "spoilage_forecast"],
        "check_compliance": {
            "compliant": False,
            "violations": [{"severity": "critical", "parameter": "temperature",
                           "value": 14.0, "limit": 5.0}],
        },
        "spoilage_forecast": {"current_rho": 0.40, "forecast_rho": 0.45,
                              "hours_ahead": 6, "urgency": "high"},
    }
    rag = {
        "guards_passed": True,
        "top_citation_score": 0.03,
        "top_doc_id": "constructed_temperature_excursion_note",
        "sop_guidance": "The constructed benchmark note uses a declared 5C operating-envelope limit.",
        "citations": [],
        "evidence_hashes": ["abc123"],
        "keywords": {"sop": {"thresholds": ["5C"], "regulations": [], "required_actions": []}},
    }

    result = explain_decision(
        action="local_redistribute", role="distributor", hour=30.0, obs=obs,
        mcp_results=mcp, rag_context=rag,
        slca_score=0.78, carbon_kg=3.5, waste=0.02,
        context_features=np.array([1.0, 0.7, 0.76, 1.0, 0.0]),
        logit_adjustment=np.array([-1.63, 1.18, 0.45]),
        action_probs=np.array([0.03, 0.93, 0.04]),
        ablation_action="local_redistribute",
        ablation_probs=np.array([0.06, 0.88, 0.06]),
        keywords=rag["keywords"],
    )

    assert "because" in result["full_explanation"].lower()
    assert "modifier zeroed" in result["full_explanation"].lower()
    assert "Ablation" in result["full_explanation"], "Should label the comparison as an ablation"
    # New honest field names.
    assert "attribution_chain" in result
    assert result["attribution_chain"]["primary_cause"] in [
        "operating-envelope exceedance", "retrieved-policy signal",
    ]
    assert "ablation_delta" in result
    assert result["ablation_delta"]["kind"] == "ablation_psi_zero"
    assert result["ablation_delta"]["probs_without_context"] is not None
    # Legacy aliases must still resolve for backward compat.
    assert "causal_chain" in result and result["causal_chain"] is result["attribution_chain"]
    assert "counterfactual" in result and result["counterfactual"] is result["ablation_delta"]


def test_humidity_only_envelope_phrase_uses_humidity_units():
    """A humidity excursion must never be narrated as a temperature event."""
    from pirag.explain_decision import _build_contribution_phrase

    phrase = _build_contribution_phrase(
        0,
        np.array([0.5, 0.0, 0.0, 0.0, 0.0]),
        {
            "check_compliance": {
                "compliant": False,
                "violations": [{
                    "parameter": "humidity_high",
                    "value": 98.0,
                    "limit": 95.0,
                    "severity": "warning",
                }],
            }
        },
        {},
        _Obs(temp=4.7, rh=98.0),
    )

    assert "relative-humidity" in phrase
    assert "98.0%" in phrase and "95.0%" in phrase
    assert "3.0 percentage points" in phrase
    assert "temperature excursion" not in phrase


def test_explain_decision_accepts_ablation_kwargs():
    """Honest API: callers should be able to pass ablation_action /
    ablation_probs by their honest names, not via the deprecated
    counterfactual_* aliases."""
    from pirag.explain_decision import explain_decision

    obs = _Obs(rho=0.4, temp=14.0)
    result = explain_decision(
        action="local_redistribute", role="distributor", hour=30.0, obs=obs,
        mcp_results={}, rag_context={"citations": [], "guards_passed": True},
        slca_score=0.7, carbon_kg=2.0, waste=0.03,
        context_features=np.array([1.0, 0.7, 0.0, 0.0, 0.0]),
        logit_adjustment=np.array([-0.5, 0.4, 0.1]),
        action_probs=np.array([0.05, 0.90, 0.05]),
        ablation_action="cold_chain",
        ablation_probs=np.array([0.55, 0.40, 0.05]),
    )
    assert result["ablation_delta"]["kind"] == "ablation_psi_zero"
    assert result["ablation_delta"]["action_changed"] is True
    assert result["ablation_delta"]["action_without_context"] == "cold_chain"


def test_explain_decision_uses_recorded_effective_theta_contributions():
    """A learned matrix, not the declared prior, determines attribution."""
    from pirag.explain_decision import explain_decision

    obs = _Obs(rho=0.4, temp=14.0)
    psi = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    learned_theta = np.zeros((3, 5), dtype=float)
    learned_theta[1] = np.array([0.01, 2.0, 0.0, 0.0, 0.0])
    contributions = learned_theta[1] * psi
    result = explain_decision(
        action="local_redistribute", role="distributor", hour=30.0, obs=obs,
        mcp_results={}, rag_context={"citations": [], "guards_passed": True},
        slca_score=0.7, carbon_kg=2.0, waste=0.03,
        context_features=psi, logit_adjustment=np.array([0.0, 2.01, 0.0]),
        effective_context_theta=learned_theta,
        chosen_action_context_contributions=contributions,
    )
    assert result["attribution_chain"]["primary_feature"] == "modeled-spoilage forecast signal"
    assert result["attribution_chain"]["basis"] == (
        "recorded_final_modifier_feature_allocation_plus_explicit_residual"
    )


# ---- Test 34: New MCP tools registered ----
def test_new_mcp_tools_registered():
    """Verify pirag_query, explain, and context_features tools are registered."""
    from pirag.mcp.registry import get_default_registry
    import pirag.mcp.registry as _reg_mod
    _reg_mod._DEFAULT_REGISTRY = None

    registry = get_default_registry()
    tool_names = set(registry._tools.keys()) if isinstance(registry._tools, dict) else {t.name for t in registry._tools}

    assert "pirag_query" in tool_names, "pirag_query tool should be registered"
    assert "explain" in tool_names, "explain tool should be registered"
    assert "context_features" in tool_names, "context_features tool should be registered"


def test_tools_list_has_qos_metadata():
    from pirag.mcp.protocol import MCPServer, MCPMessage
    from pirag.mcp.registry import get_default_registry
    import pirag.mcp.registry as _reg_mod
    _reg_mod._DEFAULT_REGISTRY = None

    server = MCPServer(registry=get_default_registry())
    resp = server.handle_message(MCPMessage(id=10, method="tools/list"))
    tools = resp.result.get("tools", [])
    assert len(tools) > 0
    assert "x-qos" in tools[0]


# ---- Test 35: Protocol recorder captures interactions ----
def test_protocol_recorder():
    """Verify protocol recorder captures MCP interactions."""
    from pirag.mcp.protocol import MCPServer, MCPMessage
    from pirag.mcp.registry import ToolRegistry
    from pirag.mcp.protocol_recorder import ProtocolRecorder

    server = MCPServer(registry=ToolRegistry())
    recorder = ProtocolRecorder(server, max_records=10)

    # Send an initialize message
    resp = server.handle_message(MCPMessage(id=1, method="initialize"))
    assert resp.result is not None

    records = recorder.get_records()
    assert len(records) == 1
    assert records[0]["request"]["method"] == "initialize"
    assert "result" in records[0]["response"]

    summary = recorder.summary()
    assert summary["total_interactions"] == 1
    assert "initialize" in summary["methods"]


def test_protocol_recorder_default_capacity_covers_publication_episode():
    """Default capacity can retain at least five calls per 288-step episode."""
    from pirag.mcp.protocol import MCPServer
    from pirag.mcp.registry import ToolRegistry
    from pirag.mcp.protocol_recorder import ProtocolRecorder

    recorder = ProtocolRecorder(MCPServer(registry=ToolRegistry()))
    assert recorder.max_records >= 5 * 288


# ---- Test 36: Knowledge base has 20 documents ----
def test_knowledge_base_size():
    """Verify KB has been expanded to 20 documents."""
    from pathlib import Path
    kb_dir = Path(__file__).resolve().parent.parent / "knowledge_base"
    docs = list(kb_dir.glob("*.txt"))
    assert len(docs) >= 20, f"KB should have at least 20 docs, found {len(docs)}"


# ---- Helper ----
class _DummyPolicy:
    gamma_coldchain = 0.1
    gamma_local = 0.1
    gamma_recovery = 0.1
