"""Focused checks for strict publication channel-evidence validation."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from src.agents.coordinator import (
    AgentCoordinator,
    _build_mcp_channel_evidence,
    _build_peer_channel_evidence,
    _build_retrieval_channel_evidence,
    _empty_channel_evidence,
    _protocol_evidence_record,
    _protocol_window,
    _seal_evidence_record,
)
from src.agents.message import InterAgentMessage, MessageType
from src.models.policy import Policy

from hpc.validate_decision_ledgers import (
    _validate_protocol_evidence,
    _validate_step_channel_evidence,
)


def _record(mode: str, *, bias: list[float] | None = None) -> dict:
    return {
        "hour": 18.0,
        "role": "processor",
        "mode": mode,
        "peer_message_bias": bias or [0.0, 0.0, 0.0],
        "primary_mcp_tools_invoked_step": [],
        "cooperative_mcp_tools_invoked_step": [],
        "primary_pirag_query_attempted_step": False,
        "cooperative_pirag_query_attempted_step": False,
        "context_integration": None,
    }


def _empty_evidence(*, peer_messages=None, bias=None) -> dict:
    primary_mcp, primary_retrieval = _empty_channel_evidence(
        "pirag", "context_capability_disabled",
    )
    coop_mcp, coop_retrieval = _empty_channel_evidence(
        "pirag", "context_capability_disabled",
    )
    messages = peer_messages or []
    policy_bias = np.asarray(bias or [0.0, 0.0, 0.0], dtype=float)
    return _seal_evidence_record({
        "schema_version": "agribrain.step_channel_evidence.v1",
        "hour": 18.0,
        "active_role": "processor",
        "peer": _build_peer_channel_evidence(
            [(message, "processor", True) for message in messages],
            [], policy_bias, enabled=True,
        ),
        "primary": _seal_evidence_record({
            "role": "processor",
            "mcp": primary_mcp,
            "retrieval": primary_retrieval,
        }),
        "cooperative": _seal_evidence_record({
            "active": False,
            "role": "cooperative",
            "mcp": coop_mcp,
            "retrieval": coop_retrieval,
        }),
    })


def test_mcp_only_skipped_retrieval_scores_bind_to_decision_record() -> None:
    """The sealed MCP-only sentinel must match numeric ledger aliases."""

    coordinator = AgentCoordinator(context_enabled=True, mode="mcp_only")
    _action, _probs, active = coordinator.step(
        {
            "rho": 0.25,
            "inv": 12_000.0,
            "temp": 7.0,
            "rh": 88.0,
            "y_hat": 100.0,
            "tau": 0.0,
            "surplus_ratio": 0.0,
            "supply_hat": 12_000.0,
            "supply_std": 100.0,
            "demand_std": 5.0,
            "price_signal": 0.0,
        },
        18.0,
        "mcp_only",
        Policy(),
        np.random.default_rng(193),
        "heatwave",
    )
    context_entry = coordinator.context_log[-1]
    integration = coordinator._step_context_integration_trace
    record = {
        "hour": 18.0,
        "role": active.role,
        "mode": "mcp_only",
        "peer_message_bias": coordinator._step_message_bias.tolist(),
        "primary_mcp_tools_invoked_step": context_entry[
            "primary_mcp_tools_invoked"
        ],
        "cooperative_mcp_tools_invoked_step": context_entry[
            "cooperative_mcp_tools_invoked"
        ],
        "primary_pirag_query_attempted_step": False,
        "cooperative_pirag_query_attempted_step": False,
        "context_integration": integration,
        "psi": integration["primary"]["effective_psi"],
        "retrieval_evidence_hashes": [],
        "retrieval_top_doc_id": "",
        "retrieval_top_score": 0.0,
        "retrieval_top_fused_score": 0.0,
        "retrieval_top_rerank_score": 0.0,
    }

    _validate_step_channel_evidence(
        coordinator._step_channel_evidence,
        record,
        where="mcp_only/heatwave/step_channel_evidence",
    )


def test_protocol_request_hash_is_recomputed_after_valid_resealing() -> None:
    protocol_record = _protocol_evidence_record({
        "_recorder_seq": 1,
        "request": {
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "weather", "arguments": {"hour": 18.0}},
        },
        "response": {"result": {"temperature": 7.0}},
    })
    changed = {
        key: value for key, value in protocol_record.items()
        if key != "content_sha256"
    }
    changed["request_sha256"] = "0" * 64
    protocol = _seal_evidence_record({
        "recorder_present": True,
        "record_index_start": 0,
        "record_index_end": 1,
        "records_captured": 1,
        "records_dropped_during_window": 0,
        "records": [_seal_evidence_record(changed)],
    })
    with pytest.raises(RuntimeError, match="request SHA-256 mismatch"):
        _validate_protocol_evidence(protocol, where="primary.protocol")

