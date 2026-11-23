"""Focused tests for content-addressed per-decision channel evidence."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import src.agents.coordinator as coordinator_module
from src.agents.coordinator import (
    AgentCoordinator,
    _canonical_content_sha256,
)
from src.agents.message import (
    InterAgentMessage,
    MessageType,
    message_bias_from_inbox,
)
from src.models.policy import Policy


def _env(*, surplus_ratio: float = 0.0) -> dict:
    return {
        "rho": 0.25,
        "inv": 12_000.0,
        "temp": 7.0,
        "rh": 88.0,
        "y_hat": 100.0,
        "tau": 0.0,
        "surplus_ratio": surplus_ratio,
        "supply_hat": 12_000.0,
        "supply_std": 100.0,
        "demand_std": 5.0,
        "price_signal": 0.0,
    }


def _assert_sealed(record: dict) -> None:
    unhashed = {
        key: value for key, value in record.items()
        if key != "content_sha256"
    }
    assert record["content_sha256"] == _canonical_content_sha256(unhashed)


def test_peer_evidence_reconstructs_bias_and_detaches_emissions() -> None:
    coordinator = AgentCoordinator(
        context_enabled=False, mode="agribrain",
    )
    processor = coordinator.agents["processor"]
    cooperative = coordinator.agents["cooperative"]
    spoilage_payload = {"rho": 0.72, "temp": 9.0}
    active_message = InterAgentMessage(
        sender="farm_agent",
        recipient=processor.agent_id,
        msg_type=MessageType.SPOILAGE_ALERT,
        payload=spoilage_payload,
        hour=17.75,
    )
    capacity_message = InterAgentMessage(
        sender="recovery_agent",
        recipient=processor.agent_id,
        msg_type=MessageType.CAPACITY_UPDATE,
        payload={"available_capacity": 0.4},
        hour=17.75,
    )
    cooperative_message = InterAgentMessage(
        sender="farm_agent",
        recipient=cooperative.agent_id,
        msg_type=MessageType.SURPLUS_ALERT,
        payload={"surplus_ratio": 0.9},
        hour=17.75,
    )
    processor.receive_message(active_message)
    processor.receive_message(capacity_message)
    cooperative.receive_message(cooperative_message)

    action, probs, active = coordinator.step(
        _env(surplus_ratio=0.8), 18.0, "agribrain", Policy(),
        np.random.default_rng(710), "baseline",
    )
    snapshot_after_decision = coordinator._step_channel_evidence
    peer = snapshot_after_decision["peer"]

    assert peer["consumed_count"] == 3
    active_records = [
        record for record in peer["consumed"]
        if record["consumer_role"] == "processor"
    ]
    cooperative_records = [
        record for record in peer["consumed"]
        if record["consumer_role"] == "cooperative"
    ]
    assert len(active_records) == 2
    assert len(cooperative_records) == 1
    assert all(record["used_for_policy_bias"] for record in active_records)
    assert not cooperative_records[0]["used_for_policy_bias"]

    reconstructed_messages = [
        InterAgentMessage(
            sender=record["sender"],
            recipient=record["recipient"],
            msg_type=MessageType(record["type"]),
            payload=record["payload"],
            hour=record["hour"],
        )
        for record in active_records
    ]
    reconstructed_bias = message_bias_from_inbox(reconstructed_messages)
    np.testing.assert_array_equal(reconstructed_bias, peer["policy_bias"])
    np.testing.assert_array_equal(
        reconstructed_bias, coordinator._step_message_bias,
    )
    np.testing.assert_array_equal(peer["policy_logit_term"], reconstructed_bias)
    assert peer["policy_logit_equation"] == (
        "z_pre_context=z_without_peer+b_peer"
    )

    # End-to-end causal intervention: with every non-peer input and the random
    # seed held fixed, removing only the inbox subtracts exactly b_peer from
    # the pre-context logits and changes the softmax probabilities.
    no_message = AgentCoordinator(context_enabled=False, mode="agribrain")
    _, no_message_probs, _ = no_message.step(
        _env(surplus_ratio=0.8), 18.0, "agribrain", Policy(),
        np.random.default_rng(710), "baseline",
    )
    np.testing.assert_allclose(
        np.asarray(coordinator._step_base_logits)
        - np.asarray(no_message._step_base_logits),
        reconstructed_bias, rtol=0.0, atol=1e-12,
    )
    assert not np.allclose(probs, no_message_probs)

    for record in peer["consumed"]:
        message_body = {
            field: record[field]
            for field in ("sender", "recipient", "type", "payload", "hour")
        }
        assert record["message_sha256"] == _canonical_content_sha256(
            message_body
        )
        _assert_sealed(record)

    # The frozen dataclass contains a mutable payload mapping.  Mutating the
    # caller-owned object after the decision must not change captured evidence.
    spoilage_payload["rho"] = 0.01
    assert active_records[0]["payload"]["rho"] == 0.72
    assert coordinator._step_channel_evidence is snapshot_after_decision

    post_obs = active.observe(_env(surplus_ratio=0.8), 18.0)
    coordinator.post_step(
        active, action, post_obs,
        {"waste": 0.05, "rho": 0.25, "slca": 0.7, "carbon_kg": 1.0},
        hour=18.0, reward=0.5,
    )
    completed = coordinator._step_channel_evidence
    assert completed is not snapshot_after_decision
    assert snapshot_after_decision["peer"]["emitted"] == []
    assert completed["peer"]["emitted_count"] == 2
    assert {
        record["type"] for record in completed["peer"]["emitted"]
    } == {
        MessageType.SURPLUS_ALERT.value,
        MessageType.COORDINATION_UPDATE.value,
    }
    for record in completed["peer"]["emitted"]:
        _assert_sealed(record)
    _assert_sealed(completed["peer"])
    _assert_sealed(completed)
    json.dumps(completed, allow_nan=False)


class _Recorder:
    def __init__(self) -> None:
        self.records: list[dict] = []
        self.dropped = 0

    def append(self, method: str, params: dict, result: dict) -> None:
        sequence = len(self.records) + 1
        self.records.append({
            "timestamp": 1234.0 + sequence,
            "latency_ms": 999.0,
            "_recorder_seq": sequence,
            "request": {
                "jsonrpc": "2.0",
                "id": sequence,
                "method": method,
                "params": params,
            },
            "response": {
                "jsonrpc": "2.0",
                "id": sequence,
                "result": result,
            },
        })


def test_primary_and_cooperative_context_evidence_are_separate_and_passive(
    monkeypatch,
) -> None:
    dispatch_calls: list[str] = []
    retrieval_calls: list[str] = []
    current_recorder: list[_Recorder] = []

    def fake_dispatch(
        role, obs, registry, shared_context, *, mcp_server, dispatch_config,
    ):
        dispatch_calls.append(role)
        tool_name = f"{role}_tool"
        arguments = {"role": role, "hour": float(obs.hour)}
        result = {"role": role, "value": np.float64(0.75)}
        current_recorder[0].append(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
            {"content": [{"type": "text", "text": role}], "isError": False},
        )
        return {
            tool_name: result,
            "_tools_invoked": [tool_name],
            "_tools_skipped": [],
            "_tools_failed": [],
            "_tool_failure_details": [],
        }

    def fake_retrieve(
        role, obs, scenario, mcp_results, pipeline, mcp_server,
        *, retrieval_kind,
    ):
        retrieval_calls.append(role)
        current_recorder[0].append(
            "prompts/get",
            {"name": f"{role}_prompt", "arguments": {"role": role}},
            {"messages": [{"content": {"text": f"{role} prompt"}}]},
        )
        passage = f"complete {role} evidence passage"
        passage_hash = hashlib.sha256(passage.encode("utf-8")).hexdigest()
        document_hash = hashlib.sha256(
            f"{role}-doc-1:{passage_hash}".encode("utf-8")
        ).hexdigest()
        base_score = 0.42 if role == "processor" else 0.31
        return {
            "retrieval_kind": retrieval_kind,
            "query": f"{role} exact query",
            "citations": [{
                "doc_id": f"{role}-doc-1",
                "passage": passage,
                "sha256": passage_hash,
                "score": base_score,
                "fused_score": base_score,
                "rerank_score": base_score + 0.07,
                "lexical_bonus": 0.04,
                "arrhenius_consistency": 0.93,
            }],
            "ranked_citations": [{
                "rank": 1,
                "base_rank": 2,
                "doc_id": f"{role}-doc-1",
                "raw_score": base_score,
                "score": base_score + 0.07,
                "fused_score": base_score,
                "rerank_score": base_score + 0.07,
                "raw_sparse_score": 7.0,
                "sparse_rank": 2,
                "sparse_rrf": 0.016,
                "raw_dense_score": 0.8,
                "dense_rank": 1,
                "dense_rrf": 0.017,
                "fusion": "rrf",
                "lexical_bonus": 0.04,
                "arrhenius_consistency": 0.93,
                "physics_bonus": 0.0372,
                "physics_consistency": 0.9,
                "content_sha256": passage_hash,
                "document_sha256": document_hash,
                "metadata_sha256": "f" * 64,
                "passage_preview_sha256": passage_hash,
                "passage_preview_character_count": len(passage),
                "passage_character_count": len(passage),
            }],
            "top_doc_id": f"{role}-doc-1",
            "top_citation_score": base_score,
            "top_fused_score": base_score,
            "top_rerank_score": base_score + 0.07,
            "lexical_bonus_mean": 0.04,
            "physics_consistency_score": 0.93,
            "guards_passed": True,
            "guard_breakdown": {
                "retrieval": True, "unit": True, "feasibility": True,
            },
            "evidence_hashes": [passage_hash],
            "ranked_evidence_hashes": [passage_hash],
            "query_transform_metadata": {
                "role": role,
                "final_query": f"{role} exact query",
            },
            "ranking_transform_metadata": {
                "final_order_source": "fixture_reranker",
            },
            "pipeline_retrieval_metadata": {"returned_count": 1},
            "pipeline_guard_decisions": {
                "guards_passed": True,
                "breakdown": {"unit": True},
            },
            "guard_decisions": {
                "aggregate": {"passed": True},
            },
            "retrieval_metrics": {"n_citations": 1},
            "regulatory_guidance": passage,
            "sop_guidance": "",
            "slca_guidance": "",
            "waste_hierarchy_guidance": "",
            "governance_guidance": "",
            "keywords": {},
        }

    def fake_modifier(
        mcp_results, rag_context, obs, temporal_window, *,
        theta_override=None, slca_amp_override=None,
        temporal_params_override=None, context_mode="full",
        retrieval_kind="pirag", trace_out=None,
    ):
        is_cooperative = "cooperative_tool" in mcp_results
        modifier = np.array(
            [0.08, -0.02, 0.01]
            if is_cooperative else [0.03, 0.04, -0.01],
            dtype=float,
        )
        if trace_out is not None:
            trace_out.update({
                "effective_psi": np.array([0.2, 0.3, 0.4, 0.1, 0.05]),
                "feature_contributions": np.zeros((3, 5), dtype=float),
                "nonfeature_residual": modifier.copy(),
                "modifier_theta_jacobian": np.zeros((3, 5), dtype=float),
                "retrieval_gate": 1.0,
                "retrieval_blocked_reason": None,
                "temporal_scale": 0.8 if is_cooperative else 0.9,
                "physics_scale": 0.7 if is_cooperative else 0.95,
                "rag_total_scale": 0.56 if is_cooperative else 0.855,
                "retrieval_kind": retrieval_kind,
            })
        return modifier

    import pirag.context_builder as context_builder
    import pirag.context_to_logits as context_to_logits
    import pirag.mcp.tool_dispatch as tool_dispatch

    monkeypatch.setattr(tool_dispatch, "dispatch_tools", fake_dispatch)
    monkeypatch.setattr(
        context_builder, "retrieve_role_context", fake_retrieve,
    )
    monkeypatch.setattr(
        context_to_logits, "compute_context_modifier", fake_modifier,
    )

    def run_once(seed: int, env_state: dict | None = None, hour: float = 18.0):
        recorder = _Recorder()
        current_recorder[:] = [recorder]
        coordinator = AgentCoordinator(
            context_enabled=False, mode="agribrain",
        )
        coordinator.context_enabled = True
        coordinator._registry = object()
        coordinator._mcp_server = object()
        coordinator._shared_context = None
        coordinator._pirag_pipeline = object()
        coordinator._temporal_window = None
        coordinator._context_learner = None
        coordinator._context_evaluator = None
        coordinator._protocol_recorder = recorder
        action, probs, _active = coordinator.step(
            env_state or _env(), hour, "agribrain", Policy(),
            np.random.default_rng(seed), "baseline",
        )
        return coordinator, action, probs

    captured, captured_action, captured_probs = run_once(810)
    assert dispatch_calls == ["processor", "cooperative"]
    assert retrieval_calls == ["processor", "cooperative"]

    primary = captured._step_channel_evidence["primary"]
    cooperative = captured._step_channel_evidence["cooperative"]
    assert primary["mcp"]["tools_invoked"] == ["processor_tool"]
    assert cooperative["mcp"]["tools_invoked"] == ["cooperative_tool"]
    assert primary["mcp"]["returned_matches_effective"] is True
    assert cooperative["mcp"]["returned_matches_effective"] is True
    assert primary["mcp"]["returned_tool_results"] == []
    assert cooperative["mcp"]["returned_tool_results"] == []
    assert primary["mcp"]["effective_tool_results"][0]["result"] == {
        "role": "processor", "value": 0.75,
    }
    assert cooperative["mcp"]["effective_tool_results"][0]["result"] == {
        "role": "cooperative", "value": 0.75,
    }
    assert primary["mcp"]["protocol"]["records_captured"] == 1
    assert cooperative["mcp"]["protocol"]["records_captured"] == 1
    assert primary["mcp"]["protocol"]["records"][0]["tool_name"] == (
        "processor_tool"
    )
    assert cooperative["mcp"]["protocol"]["records"][0]["tool_name"] == (
        "cooperative_tool"
    )

    primary_retrieval = primary["retrieval"]
    cooperative_retrieval = cooperative["retrieval"]
    assert primary_retrieval["query"] == "processor exact query"
    assert cooperative_retrieval["query"] == "cooperative exact query"
    assert primary_retrieval["protocol"]["records_captured"] == 1
    assert cooperative_retrieval["protocol"]["records_captured"] == 1
    primary_citation = primary_retrieval["ordered_citations"][0]
    cooperative_citation = cooperative_retrieval["ordered_citations"][0]
    assert primary_citation["doc_id"] == "processor-doc-1"
    assert cooperative_citation["doc_id"] == "cooperative-doc-1"
    assert primary_retrieval["citation_order_source"] == (
        "rag_context.ranked_citations"
    )
    assert cooperative_retrieval["citation_order_source"] == (
        "rag_context.ranked_citations"
    )
    assert primary_citation["base_rank"] == 2
    assert primary_citation["raw_score"] == 0.42
    assert primary_citation["fused_score"] == 0.42
    assert primary_citation["rerank_score"] == 0.49
    assert primary_citation["lexical_bonus"] == 0.04
    assert primary_citation["arrhenius_consistency"] == 0.93
    assert primary_citation["raw_sparse_score"] == 7.0
    assert primary_citation["sparse_rank"] == 2
    assert primary_citation["sparse_rrf"] == 0.016
    assert primary_citation["raw_dense_score"] == 0.8
    assert primary_citation["dense_rank"] == 1
    assert primary_citation["dense_rrf"] == 0.017
    assert primary_citation["fusion"] == "rrf"
    assert primary_citation["document_sha256"]
    assert primary_citation["source_passage_sha256"]
    assert "passage" not in primary_citation
    assert primary_retrieval["guard_breakdown"] == {
        "retrieval": True, "unit": True, "feasibility": True,
    }
    assert primary_retrieval["temporal_scale"] == 0.9
    assert cooperative_retrieval["temporal_scale"] == 0.8
    assert primary_retrieval["physics_scale"] == 0.95
    assert cooperative_retrieval["physics_scale"] == 0.7
    assert primary_retrieval["query_transform_metadata"]["role"] == (
        "processor"
    )
    assert cooperative_retrieval["query_transform_metadata"]["role"] == (
        "cooperative"
    )
    assert primary_retrieval["guard_decisions"] == {
        "aggregate": {"passed": True},
    }
    assert primary_retrieval["evidence_hashes"] != (
        cooperative_retrieval["evidence_hashes"]
    )
    _assert_sealed(captured._step_channel_evidence)
    json.dumps(captured._step_channel_evidence, allow_nan=False)

    # The H3 post-call fault treatment must preserve both the dispatcher
    # return and the None value that actually reached the policy mapping.
    fault_env = _env()
    fault_env["policy_flags"] = {"enable_failure_injection": True}
    dispatch_calls.clear()
    retrieval_calls.clear()
    faulted, _faulted_action, _faulted_probs = run_once(
        811, fault_env, hour=22.0,
    )
    faulted_mcp = faulted._step_channel_evidence["primary"]["mcp"]
    assert faulted_mcp["returned_matches_effective"] is False
    assert faulted_mcp["returned_tool_results"][0]["result"] == {
        "role": "processor", "value": 0.75,
    }
    assert faulted_mcp["effective_tool_results"][0]["result"] is None
    assert faulted_mcp["dispatcher_metadata"]["_fault_injected"] == (
        "drop_tool_results"
    )

    # Disable only the observer builders.  Identical policy outputs and exact
    # call counts prove evidence capture neither adds calls nor affects logits.
    monkeypatch.setattr(
        coordinator_module,
        "_build_mcp_channel_evidence",
        lambda **_kwargs: {"capture_disabled_for_test": True},
    )
    monkeypatch.setattr(
        coordinator_module,
        "_build_retrieval_channel_evidence",
        lambda **_kwargs: {"capture_disabled_for_test": True},
    )
    dispatch_calls.clear()
    retrieval_calls.clear()
    _uncaptured, uncaptured_action, uncaptured_probs = run_once(810)
    assert dispatch_calls == ["processor", "cooperative"]
    assert retrieval_calls == ["processor", "cooperative"]
    assert uncaptured_action == captured_action
    np.testing.assert_array_equal(uncaptured_probs, captured_probs)
