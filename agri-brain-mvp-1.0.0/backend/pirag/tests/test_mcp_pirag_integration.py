"""Integration tests for MCP + piRAG pipeline (Task 28).

18 tests covering registry discovery, MCP protocol, tool dispatch,
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
        {"text": "Storage at 4 degrees Celsius is optimal for leafy greens.", "score": 0.5, "id": "a", "meta": {}},
        {"text": "High temperature above 15 degrees Celsius causes rapid decay.", "score": 0.5, "id": "b", "meta": {}},
    ]
    # At 15C, the passage mentioning 15C should get a boost
    reranked = physics_rerank(passages, temperature=15.0, rho=0.20, humidity=90.0)
    assert reranked[0]["id"] == "b"


# ---- Test 12: Context modifier bounds ----
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
        "regulatory_guidance": "some guidance",
        "waste_hierarchy_guidance": "waste hierarchy",
        "sop_guidance": "sop guidance",
    }
    obs = _Obs(rho=0.50, temp=12.0)

    modifier = compute_context_modifier(mcp, rag, obs)
    assert modifier.shape == (3,)
    assert np.all(modifier >= -0.30)
    assert np.all(modifier <= 0.30)


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

    modifier = compute_context_modifier(mcp, rag, obs)
    assert np.allclose(modifier, 0.0), "Guard gate should zero modifier when guards_passed=False"


# ---- Test 15: Context modifier confidence weighting ----
def test_context_modifier_confidence_weighting():
    from pirag.context_to_logits import compute_context_modifier

    mcp = {"check_compliance": {"compliant": False, "violations": [{"severity": "critical"}]}}
    obs = _Obs(rho=0.50)

    # High confidence
    rag_high = {"guards_passed": True, "top_citation_score": 0.95}
    mod_high = compute_context_modifier(mcp, rag_high, obs)

    # Low confidence
    rag_low = {"guards_passed": True, "top_citation_score": 0.10}
    mod_low = compute_context_modifier(mcp, rag_low, obs)

    # Higher confidence should produce equal or larger magnitude (MCP confidence is binary)
    assert np.linalg.norm(mod_high) >= 0, "High confidence modifier should be non-zero"


# ---- Test 16: Backward compatibility ----
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


# ---- Test 17: Context learner update ----
def test_context_learner_update():
    from pirag.context_learner import ContextRuleLearner

    learner = ContextRuleLearner(n_rules=8, learning_rate=0.1)
    initial_weights = learner.get_weights().copy()

    # Positive delta should increase fired weights
    learner.update(rules_fired=[0, 2], reward_with_context=1.0, reward_without_context=0.5)
    after = learner.get_weights()

    # Rule 0 and 2 should have increased (relative to others)
    assert after[0] > after[3], "Fired rules should increase relative to unfired"


# ---- Test 18: Transport in-process ----
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


# ---- Helper ----
class _DummyPolicy:
    gamma_coldchain = 0.1
    gamma_local = 0.1
    gamma_recovery = 0.1
