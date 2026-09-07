from __future__ import annotations

import copy

import pytest

from src.models.episode_evidence_contract import (
    build_episode_evidence_contract,
    reconstruct_episode_evidence,
    validate_episode_evidence_contract,
)


def _record(
    *, latency: float, modifier: list[float] | None, action: int,
    changed: bool | None, protocol_calls: int = 0,
) -> dict:
    active = bool(modifier and max(abs(value) for value in modifier) > 0.10)
    retrieval_attempted = modifier is not None
    primary_tools = [f"tool_{index}" for index in range(protocol_calls)]
    return {
        "decision_latency_ms": latency,
        "action_idx": action,
        "context_modifier": modifier,
        "context_action_changed": changed,
        "context_influence_active": active,
        "context_influence_counted": active and changed is True,
        "mcp_tool_call_count_step": protocol_calls,
        "pirag_query_count_step": int(retrieval_attempted),
        "primary_mcp_tools_invoked_step": primary_tools,
        "cooperative_mcp_tools_invoked_step": [],
        "primary_pirag_query_attempted_step": retrieval_attempted,
        "cooperative_pirag_query_attempted_step": False,
        "dispatcher_tool_failure_count_step": 0,
        "inter_agent_message_count_step": 1,
        "protocol_interaction_count_step": (
            protocol_calls + int(retrieval_attempted)
        ),
        "protocol_tools_call_count_step": protocol_calls,
        "protocol_prompts_get_count_step": int(retrieval_attempted),
        "protocol_jsonrpc_error_count_step": 0,
        "protocol_tool_iserror_count_step": 0,
        "protocol_real_tool_iserror_count_step": 0,
        "protocol_error_count_step": 0,
        "protocol_dropped_interaction_count_step": 0,
    }


def test_episode_evidence_reconstructs_all_activity_aggregates() -> None:
    contract = build_episode_evidence_contract()
    records = [
        _record(
            latency=10.0, modifier=[0.0, 0.2, 0.0], action=1,
            changed=True, protocol_calls=2,
        ),
        _record(
            latency=30.0, modifier=[0.0, 0.2, 0.0], action=0,
            changed=False, protocol_calls=1,
        ),
    ]
    evidence = reconstruct_episode_evidence(records, contract)

    assert evidence["mean_decision_latency_ms"] == 20.0
    assert evidence["p95_decision_latency_ms"] == 29.0
    assert evidence["context_active_steps"] == 2
    assert evidence["context_honored_steps"] == 1
    assert evidence["context_influenced_steps"] == 1
    assert evidence["context_honor_rate"] == 0.5
    assert evidence["context_influence_rate"] == 0.5
    assert evidence["protocol_interaction_count"] == 5
    assert evidence["protocol_tools_call_count"] == 3
    assert evidence["protocol_prompts_get_count"] == 2
    assert evidence["mcp_calls_per_episode"] == 3
    assert evidence["message_count"] == 2
    assert evidence["footprint"]["cumulative_elapsed_seconds"] == 0.04
    assert evidence["footprint"]["cumulative_energy_J"] == 0.4


def test_episode_evidence_rejects_rehashed_semantic_counter_tamper() -> None:
    contract = build_episode_evidence_contract()
    record = _record(
        latency=10.0, modifier=[0.0, 0.2, 0.0], action=1,
        changed=False, protocol_calls=1,
    )
    record["protocol_error_count_step"] = 1

    with pytest.raises(ValueError, match="inconsistent protocol error total"):
        reconstruct_episode_evidence([record], contract)


def test_episode_evidence_rejects_hidden_raw_tool_error() -> None:
    contract = build_episode_evidence_contract()
    record = _record(
        latency=10.0, modifier=[0.0, 0.2, 0.0], action=1,
        changed=False, protocol_calls=1,
    )
    record["protocol_tool_iserror_count_step"] = 1

    with pytest.raises(ValueError, match="raw and real tool error counts differ"):
        reconstruct_episode_evidence([record], contract)


def test_episode_evidence_rejects_more_tool_errors_than_calls() -> None:
    contract = build_episode_evidence_contract()
    record = _record(
        latency=10.0, modifier=[0.0, 0.2, 0.0], action=1,
        changed=False, protocol_calls=1,
    )
    record["protocol_tool_iserror_count_step"] = 2
    record["protocol_real_tool_iserror_count_step"] = 2
    record["protocol_error_count_step"] = 2

    with pytest.raises(ValueError, match="more tool errors than tool calls"):
        reconstruct_episode_evidence([record], contract)


def test_episode_evidence_rejects_parameter_substitution() -> None:
    expected = build_episode_evidence_contract()
    substituted = copy.deepcopy(expected)
    substituted["footprint"]["assumed_active_power_W"] = 1.0

    with pytest.raises(ValueError, match="assumed_active_power_W"):
        validate_episode_evidence_contract(
            substituted, expected_contract=expected,
        )


def _record_with_failed_dispatch(failures: int) -> dict:
    """A step where some dispatched tools returned and some raised.

    ``primary_mcp_tools_invoked_step`` lists only calls that returned, and
    ``mcp_tool_call_count_step`` counts those, because that quantity feeds the
    reported per-episode call metric. The protocol recorder counts every
    dispatched ``tools/call``, so it is larger by exactly the failure count.
    """
    record = _record(
        latency=1.0, modifier=[0.2, 0.0, 0.0], action=1, changed=True,
        protocol_calls=2,
    )
    record["dispatcher_tool_failure_count_step"] = failures
    record["protocol_tools_call_count_step"] = 2 + failures
    record["protocol_interaction_count_step"] = 2 + failures + 1
    return record


def test_dispatched_call_that_failed_reconciles_against_the_protocol_count():
    """A tool that raised after dispatch is still tools/call traffic.

    Before this was reconciled, a single failed dispatch made the protocol
    count exceed the invocation lists by one and the contract rejected an
    otherwise valid step, which aborted every development run that happened to
    hit a failing tool.
    """
    contract = build_episode_evidence_contract()
    reconstruct_episode_evidence([_record_with_failed_dispatch(1)], contract)
    reconstruct_episode_evidence([_record_with_failed_dispatch(3)], contract)


def test_protocol_count_still_rejects_an_unexplained_surplus():
    """The tolerance is exactly the failure count, not a free allowance."""
    record = _record_with_failed_dispatch(1)
    record["protocol_tools_call_count_step"] += 1
    record["protocol_interaction_count_step"] += 1
    contract = build_episode_evidence_contract()
    with pytest.raises(ValueError, match="protocol_tools_call_count_step"):
        reconstruct_episode_evidence([record], contract)


def test_negative_failure_count_is_rejected():
    """The failure count is a count; a negative one cannot excuse a shortfall."""
    record = _record_with_failed_dispatch(1)
    record["dispatcher_tool_failure_count_step"] = -1
    contract = build_episode_evidence_contract()
    with pytest.raises(ValueError, match="dispatcher_tool_failure_count_step"):
        reconstruct_episode_evidence([record], contract)
