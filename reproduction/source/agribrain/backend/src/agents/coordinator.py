"""Multi-agent coordinator that dispatches decisions to role-specific agents.

The coordinator maps each simulation timestep to the appropriate supply
chain agent based on the lifecycle stage, delegates observation building
and action selection (via ``select_action`` from ``action_selection.py``),
and routes inter-agent messages after each step.

When ``context_enabled=True`` and mode is ``"agribrain"`` (or a structural
ablation), the coordinator integrates MCP
tool dispatch, institutional retrieval, mechanistically derived context modifiers,
online REINFORCE learning, and context quality evaluation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from ..models.action_selection import select_action
from ..models.mode_capabilities import (
    CONTEXT_INFRASTRUCTURE_MODES,
    CONTEXT_MODE_MAP,
    DECISION_OWNER_ROLES,
    capabilities_for,
)
from .base import Observation, SupplyChainAgent
from .message import InterAgentMessage
from .roles import (
    CooperativeAgent,
    DistributorAgent,
    FarmAgent,
    ProcessorAgent,
    RecoveryAgent,
    stage_for_hour,
)

_log = logging.getLogger(__name__)


def _learner_state_sha256(state: Dict[str, Any]) -> str:
    """Stable digest of one JSON-serializable learner checkpoint."""

    payload = json.dumps(
        state, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _context_trace_jsonable(value: Any) -> Any:
    """Convert a context-integration trace to immutable JSON-native values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(key): _context_trace_jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_context_trace_jsonable(item) for item in value]
    return value


def _strict_json_native(value: Any, path: str = "$") -> Any:
    """Return a detached, strictly JSON-native copy of ``value``.

    Evidence hashes must never depend on ``repr`` or ``default=str``.  This
    converter therefore handles the concrete numeric/container types used by
    the simulator and rejects unsupported objects, non-string mapping keys,
    and non-finite floats with a path that identifies the offending value.
    """

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Enum):
        return _strict_json_native(value.value, path)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        native = float(value)
        if not math.isfinite(native):
            raise ValueError(f"non-finite float at {path}")
        return native
    if isinstance(value, np.ndarray):
        return _strict_json_native(value.tolist(), path)
    if isinstance(value, bytes):
        return {
            "byte_length": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _strict_json_native(
                getattr(value, field.name), f"{path}.{field.name}",
            )
            for field in fields(value)
        }
    if isinstance(value, dict):
        native_dict: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"non-string mapping key {key!r} at {path}"
                )
            native_dict[key] = _strict_json_native(
                item, f"{path}.{key}",
            )
        return native_dict
    if isinstance(value, (list, tuple)):
        return [
            _strict_json_native(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"unsupported evidence value {type(value).__name__} at {path}"
    )


def _canonical_content_sha256(value: Any) -> str:
    """Hash one value using canonical, finite, strict JSON encoding."""

    payload = json.dumps(
        _strict_json_native(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _seal_evidence_record(value: Dict[str, Any]) -> Dict[str, Any]:
    """Return a detached content-addressed record without in-place mutation."""

    native = _strict_json_native(value)
    native.pop("content_sha256", None)
    native["content_sha256"] = _canonical_content_sha256(native)
    return native


def _message_evidence_record(
    message: InterAgentMessage,
    *,
    consumer_role: Optional[str] = None,
    used_for_policy_bias: bool = False,
) -> Dict[str, Any]:
    """Serialize one exact peer message and bind it to its consumer."""

    message_body = {
        "sender": message.sender,
        "recipient": message.recipient,
        "type": message.msg_type.value,
        "payload": message.payload,
        "hour": message.hour,
    }
    record = {
        **message_body,
        "consumer_role": consumer_role,
        "used_for_policy_bias": bool(used_for_policy_bias),
        "message_sha256": _canonical_content_sha256(message_body),
    }
    return _seal_evidence_record(record)


def _build_peer_channel_evidence(
    consumed: List[tuple[InterAgentMessage, str, bool]],
    emitted: List[InterAgentMessage],
    policy_bias: Any,
    enabled: bool,
) -> Dict[str, Any]:
    consumed_records = [
        _message_evidence_record(
            message,
            consumer_role=consumer_role,
            used_for_policy_bias=used_for_policy_bias,
        )
        for message, consumer_role, used_for_policy_bias in consumed
    ]
    emitted_records = [
        _message_evidence_record(message) for message in emitted
    ]
    native_bias = _strict_json_native(
        np.asarray(policy_bias, dtype=float).reshape(3)
    )
    return _seal_evidence_record({
        "enabled": bool(enabled),
        "bias_function": "src.agents.message.message_bias_from_inbox",
        "consumed": consumed_records,
        "consumed_count": len(consumed_records),
        "emitted": emitted_records,
        "emitted_count": len(emitted_records),
        "policy_bias": native_bias,
        "policy_bias_sha256": _canonical_content_sha256(native_bias),
        # This explicit alias closes the causal evidence chain: reconstructed
        # consumed messages -> b_peer -> the additive pre-softmax logit term.
        # The independent ledger validator binds it to combined_role_bias and
        # ultimately to the recorded softmax probabilities.
        "policy_logit_term": native_bias,
        "policy_logit_equation": "z_pre_context=z_without_peer+b_peer",
    })


def _protocol_cursor(recorder: Any) -> tuple[int, int]:
    """Read a recorder cursor without copying its complete retained history."""

    if recorder is None:
        return 0, 0
    records = getattr(recorder, "_records", None)
    lock = getattr(recorder, "_lock", None)
    if isinstance(records, list) and lock is not None:
        with lock:
            return len(records), int(getattr(recorder, "_dropped", 0))
    records = getattr(recorder, "records", None)
    if isinstance(records, list):
        return len(records), int(getattr(recorder, "dropped", 0))
    get_records = getattr(recorder, "get_records", None)
    if callable(get_records):
        return len(get_records()), int(getattr(recorder, "_dropped", 0))
    return 0, 0


def _protocol_window(
    recorder: Any,
    start_cursor: tuple[int, int],
) -> Dict[str, Any]:
    """Close one call window and retain only its deterministic records."""

    start_index, dropped_before = start_cursor
    if recorder is None:
        raw_records: List[Dict[str, Any]] = []
        end_index = 0
        dropped_after = 0
    else:
        records = getattr(recorder, "_records", None)
        lock = getattr(recorder, "_lock", None)
        if isinstance(records, list) and lock is not None:
            with lock:
                end_index = len(records)
                dropped_after = int(getattr(recorder, "_dropped", 0))
                raw_records = list(records[start_index:end_index])
        else:
            records = getattr(recorder, "records", None)
            if not isinstance(records, list):
                getter = getattr(recorder, "get_records", None)
                records = list(getter()) if callable(getter) else []
            end_index = len(records)
            dropped_after = int(
                getattr(recorder, "dropped", getattr(recorder, "_dropped", 0))
            )
            raw_records = list(records[start_index:end_index])

    records_out = [
        _protocol_evidence_record(record) for record in raw_records
    ]
    return _seal_evidence_record({
        "recorder_present": recorder is not None,
        "record_index_start": int(start_index),
        "record_index_end": int(end_index),
        "records_captured": len(records_out),
        "records_dropped_during_window": max(
            0, int(dropped_after - dropped_before),
        ),
        "records": records_out,
    })


def _protocol_evidence_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Remove wall-clock fields and canonically describe one MCP record."""

    request = record.get("request") or {}
    response = record.get("response")
    params = request.get("params") or {}
    method = request.get("method", "")
    tool_name = (
        params.get("name")
        if method == "tools/call" and isinstance(params, dict)
        else None
    )
    arguments = (
        params.get("arguments")
        if isinstance(params, dict) and "arguments" in params
        else None
    )
    result = (
        response.get("result")
        if isinstance(response, dict) else None
    )
    error = (
        response.get("error")
        if isinstance(response, dict) else None
    )
    return _seal_evidence_record({
        "protocol_sequence": record.get("_recorder_seq"),
        "notification": bool(record.get("_notification", False)),
        "method": method,
        "request_id": request.get("id"),
        "request": request,
        "request_sha256": _canonical_content_sha256(request),
        "tool_name": tool_name,
        "arguments": arguments,
        # The complete policy-facing tool result is retained once in the MCP
        # block below.  A digest is sufficient at this protocol boundary and
        # avoids duplicating the same payload for every dispatcher record.
        "returned_result_included": False,
        "returned_result_sha256": (
            _canonical_content_sha256(result) if result is not None else None
        ),
        "jsonrpc_error": error,
        "tool_is_error": bool(
            isinstance(result, dict) and result.get("isError") is True
        ),
        # ProtocolRecorder intentionally depth-limits its response copy.  The
        # full dispatcher result and its hash live in the MCP channel block.
        "recorder_response_depth_limited": result is not None,
    })


def _tool_result_records(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    invoked = list(results.get("_tools_invoked", []) or [])
    names = list(invoked)
    names.extend(
        name for name in results
        if not name.startswith("_") and name not in names
    )
    return [
        _seal_evidence_record({
            "tool_name": str(name),
            "result": results.get(name),
            "result_sha256": _canonical_content_sha256(results.get(name)),
        })
        for name in names
    ]


def _build_mcp_channel_evidence(
    *,
    returned_results: Dict[str, Any],
    effective_results: Dict[str, Any],
    protocol_window: Dict[str, Any],
    attempted: bool,
    skip_reason: Optional[str] = None,
    operation_error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Describe one role's dispatch without issuing any additional calls."""

    returned = _strict_json_native(returned_results)
    effective = _strict_json_native(effective_results)
    returned_hash = _canonical_content_sha256(returned)
    effective_hash = _canonical_content_sha256(effective)
    returned_matches_effective = returned_hash == effective_hash
    metadata = {
        key: value for key, value in effective.items() if key.startswith("_")
    }
    jsonrpc_errors = [
        record["jsonrpc_error"]
        for record in protocol_window.get("records", [])
        if record.get("jsonrpc_error") is not None
    ]
    tool_protocol_errors = [
        {
            "tool_name": record.get("tool_name"),
            "returned_result_sha256": record.get("returned_result_sha256"),
        }
        for record in protocol_window.get("records", [])
        if record.get("tool_is_error")
    ]
    return _seal_evidence_record({
        "attempted": bool(attempted),
        "skip_reason": skip_reason,
        "protocol": protocol_window,
        "tools_invoked": list(effective.get("_tools_invoked", []) or []),
        "tools_skipped": list(effective.get("_tools_skipped", []) or []),
        "tools_failed": list(effective.get("_tools_failed", []) or []),
        "tool_failure_details": list(
            effective.get("_tool_failure_details", []) or []
        ),
        # In the ordinary path the dispatcher return is exactly the effective
        # policy input, so retain it once.  When fault treatment changes it,
        # preserve both sides to make the treatment reconstructable.
        "returned_tool_results": (
            []
            if returned_matches_effective else _tool_result_records(returned)
        ),
        "effective_tool_results": _tool_result_records(effective),
        "returned_matches_effective": returned_matches_effective,
        "returned_results_sha256": returned_hash,
        "effective_results_sha256": effective_hash,
        "dispatcher_metadata": metadata,
        "errors": {
            "operation": operation_error,
            "jsonrpc": jsonrpc_errors,
            "tool_is_error": tool_protocol_errors,
        },
    })


def _citation_score(citation: Dict[str, Any], *names: str) -> Any:
    metadata = citation.get("meta") or citation.get("metadata") or {}
    for name in names:
        if name in citation:
            return citation[name]
        if isinstance(metadata, dict) and name in metadata:
            return metadata[name]
    return None


def _citation_evidence_records(
    rag_context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    citations = (
        rag_context.get("ranked_citations")
        or rag_context.get("citations", [])
        or []
    )
    for index, citation in enumerate(citations):
        if not isinstance(citation, dict):
            raise TypeError(
                f"retrieval citation {index} is not a mapping"
            )
        passage = citation.get("passage", citation.get("text"))
        source_hash = (
            citation.get("content_sha256")
            or citation.get("sha256")
            or citation.get("passage_sha256")
            or citation.get("content_hash")
        )
        records.append(_seal_evidence_record({
            "rank": citation.get("rank", index + 1),
            "base_rank": citation.get("base_rank"),
            "doc_id": citation.get("doc_id", citation.get("id", "")),
            "raw_score": _citation_score(citation, "raw_score"),
            "score": _citation_score(citation, "score"),
            "fused_score": _citation_score(
                citation, "fused_score", "rrf_score",
            ),
            "rerank_score": _citation_score(citation, "rerank_score"),
            "lexical_score": _citation_score(citation, "lexical_score"),
            "lexical_bonus": _citation_score(citation, "lexical_bonus"),
            "arrhenius_score": _citation_score(
                citation, "arrhenius_score",
            ),
            "arrhenius_consistency": _citation_score(
                citation, "arrhenius_consistency",
            ),
            "raw_sparse_score": _citation_score(
                citation, "raw_sparse_score", "sparse_score",
            ),
            "sparse_rank": _citation_score(citation, "sparse_rank"),
            "sparse_rrf": _citation_score(citation, "sparse_rrf"),
            "raw_dense_score": _citation_score(
                citation, "raw_dense_score", "dense_score",
            ),
            "dense_rank": _citation_score(citation, "dense_rank"),
            "dense_rrf": _citation_score(citation, "dense_rrf"),
            "fusion": _citation_score(citation, "fusion"),
            "physics_bonus": _citation_score(citation, "physics_bonus"),
            "physics_consistency": _citation_score(
                citation, "physics_consistency",
            ),
            "document_sha256": citation.get("document_sha256"),
            "metadata_sha256": citation.get("metadata_sha256"),
            "source_passage_sha256": source_hash,
            "captured_passage_content_sha256": (
                _text_sha256(passage)
                if isinstance(passage, str)
                else citation.get("passage_preview_sha256")
            ),
            "captured_passage_character_count": (
                len(passage)
                if isinstance(passage, str)
                else citation.get("passage_preview_character_count", 0)
            ),
            "source_passage_character_count": citation.get(
                "passage_character_count"
            ),
            # Legacy contexts carry a 300-character value; ranked contexts
            # expose only its hash. The complete passage is identified by the
            # source hash and is deliberately not duplicated here.
            "captured_passage_scope": (
                "rag_context_value"
                if isinstance(passage, str)
                else "upstream_preview_hash_only"
            ),
            "captured_passage_may_be_preview": bool(
                isinstance(passage, str)
                or citation.get("passage_preview_sha256")
            ),
        }))
    return records


def _build_retrieval_channel_evidence(
    *,
    rag_context: Dict[str, Any],
    integration_trace: Optional[Dict[str, Any]],
    protocol_window: Dict[str, Any],
    attempted: bool,
    requested_kind: str,
    skip_reason: Optional[str] = None,
    operation_error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Describe one retrieval and its guards without retaining passage text."""

    rag = _strict_json_native(rag_context)
    trace = _strict_json_native(integration_trace or {})
    guidance_hashes: Dict[str, Any] = {}
    for field_name in (
        "regulatory_guidance", "sop_guidance", "slca_guidance",
        "waste_hierarchy_guidance", "governance_guidance",
    ):
        value = rag.get(field_name, "")
        guidance_hashes[field_name] = {
            "present": bool(value),
            "character_count": len(value) if isinstance(value, str) else 0,
            "sha256": _text_sha256(value) if isinstance(value, str) else None,
        }
    query = rag.get("query", "")
    return _seal_evidence_record({
        "attempted": bool(attempted),
        "skip_reason": skip_reason,
        "retrieval_kind": rag.get("retrieval_kind", requested_kind),
        "query": query,
        "query_sha256": _text_sha256(query) if isinstance(query, str) else None,
        "protocol": protocol_window,
        "citation_order_source": (
            "rag_context.ranked_citations"
            if rag.get("ranked_citations") else "rag_context.citations"
        ),
        "ordered_citations": _citation_evidence_records(rag),
        "top_doc_id": rag.get("top_doc_id", ""),
        "top_citation_score": rag.get("top_citation_score"),
        "top_fused_score": rag.get("top_fused_score"),
        "top_rerank_score": rag.get("top_rerank_score"),
        "lexical_bonus_mean": rag.get("lexical_bonus_mean"),
        "physics_consistency_score": rag.get(
            "physics_consistency_score"
        ),
        "guards_passed": rag.get("guards_passed"),
        "guard_breakdown": rag.get("guard_breakdown", {}),
        "guard_decisions": rag.get("guard_decisions", {}),
        "pipeline_guard_decisions": rag.get(
            "pipeline_guard_decisions", {}
        ),
        "query_transform_metadata": rag.get(
            "query_transform_metadata", {}
        ),
        "ranking_transform_metadata": rag.get(
            "ranking_transform_metadata", {}
        ),
        "pipeline_retrieval_metadata": rag.get(
            "pipeline_retrieval_metadata", {}
        ),
        "retrieval_gate": trace.get("retrieval_gate"),
        "retrieval_blocked_reason": trace.get(
            "retrieval_blocked_reason"
        ),
        "temporal_scale": trace.get("temporal_scale"),
        "temporal_gate_requested": trace.get("temporal_gate_requested"),
        "temporal_gate_applied": trace.get("temporal_gate_applied"),
        "temporal_continuity_score": trace.get(
            "temporal_continuity_score"
        ),
        "temporal_base": trace.get("temporal_base"),
        "temporal_decay": trace.get("temporal_decay"),
        "physics_scale": trace.get("physics_scale"),
        "rag_total_scale": trace.get("rag_total_scale"),
        "effective_psi": trace.get("effective_psi"),
        "evidence_hashes": rag.get(
            "ranked_evidence_hashes", rag.get("evidence_hashes", [])
        ),
        "source_order_evidence_hashes": rag.get("evidence_hashes", []),
        "guidance_hashes": guidance_hashes,
        "retrieval_metrics": rag.get("retrieval_metrics", {}),
        "raw_context_sha256": _canonical_content_sha256(rag),
        "empty_result": bool(
            attempted
            and not query
            and not (rag.get("ranked_citations") or rag.get("citations"))
        ),
        "errors": {
            "operation": operation_error,
            "reported": rag.get("_error", rag.get("error")),
            "jsonrpc": [
                record["jsonrpc_error"]
                for record in protocol_window.get("records", [])
                if record.get("jsonrpc_error") is not None
            ],
        },
    })


def _empty_channel_evidence(
    requested_kind: str,
    skip_reason: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    empty_protocol = _protocol_window(None, (0, 0))
    mcp = _build_mcp_channel_evidence(
        returned_results={"_tools_invoked": []},
        effective_results={"_tools_invoked": []},
        protocol_window=empty_protocol,
        attempted=False,
        skip_reason=skip_reason,
    )
    retrieval = _build_retrieval_channel_evidence(
        rag_context={},
        integration_trace=None,
        protocol_window=empty_protocol,
        attempted=False,
        requested_kind=requested_kind,
        skip_reason=skip_reason,
    )
    return mcp, retrieval


def _replace_channel_evidence(
    current: Dict[str, Any],
    **sections: Any,
) -> Dict[str, Any]:
    """Replace sections of a sealed decision snapshot and reseal the root."""

    root = {
        key: value for key, value in current.items()
        if key != "content_sha256"
    }
    root.update(sections)
    return _seal_evidence_record(root)

# Compatibility aliases retained for downstream imports. Their content comes
# from the central mode-capability declaration rather than duplicated sets.
_CONTEXT_MODES = set(CONTEXT_INFRASTRUCTURE_MODES)
_CONTEXT_MODE_MAP = dict(CONTEXT_MODE_MAP)

# Cooperative agent overlay window (simulation hours).
# The cooperative agent observes, contributes its role bias, exchanges typed
# peer messages when that channel is enabled, and participates in the declared
# external-context composition while hour is in this half-open interval.
# It is an advisory overlay, not a separate formal voting stage.
COOPERATIVE_OVERLAY_START: float = 12.0
COOPERATIVE_OVERLAY_END: float = 30.0


def _cooperative_window_active(hour: float) -> bool:
    return COOPERATIVE_OVERLAY_START <= hour < COOPERATIVE_OVERLAY_END


def _observe_without_peer_inbox(
    agent: SupplyChainAgent,
    env_state: Dict[str, Any],
    hour: float,
) -> Observation:
    """Observe state without reading or clearing the peer-message inbox.

    Role ``observe`` implementations normally call ``flush_inbox``.  The
    no-peer one-factor arm must disable inbox *consumption* in addition to
    generation and delivery, while leaving the observation schedule and all
    non-message state unchanged.  Temporarily detaching the inbox gives the
    role its normal observation path with an empty message channel, then
    restores any pre-existing messages byte-for-byte and in order.
    """

    pending = list(agent._inbox)
    agent._inbox.clear()
    try:
        observation = agent.observe(env_state, hour)
    finally:
        messages_added_during_observe = list(agent._inbox)
        agent._inbox[:] = pending + messages_added_during_observe
    observation.messages = []
    return observation


def _compose_context_attribution(
    primary_modifier: np.ndarray,
    primary_trace: Dict[str, Any],
    cooperative_modifier: Optional[np.ndarray] = None,
    cooperative_trace: Optional[Dict[str, Any]] = None,
    veto_bias: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, np.ndarray, Dict[str, Any]]:
    """Compose a bounded modifier, attribution, and exact learner Jacobian.

    Feature allocations are blended with the same coefficients as their live
    modifiers.  The legacy ``veto_bias`` argument names the author-declared
    cooperative operating-envelope adjustment: it replaces the primary
    allocation and records its fixed bias as an explicit non-feature residual.
    Jacobians follow the same composition, but a fixed adjustment has zero
    derivative.  A final clip preserves the declared [-1,+1] modifier contract.
    """
    primary_modifier = np.asarray(primary_modifier, dtype=float)
    features = np.asarray(primary_trace["feature_contributions"], dtype=float)
    residual = np.asarray(primary_trace["nonfeature_residual"], dtype=float)
    jacobian = np.asarray(
        primary_trace["modifier_theta_jacobian"], dtype=float,
    )
    modifier = primary_modifier.copy()
    scope = "primary_context"

    if cooperative_modifier is not None and cooperative_trace is not None:
        cooperative_modifier = np.asarray(cooperative_modifier, dtype=float)
        coop_features = np.asarray(
            cooperative_trace["feature_contributions"], dtype=float,
        )
        coop_residual = np.asarray(
            cooperative_trace["nonfeature_residual"], dtype=float,
        )
        coop_jacobian = np.asarray(
            cooperative_trace["modifier_theta_jacobian"], dtype=float,
        )
        if veto_bias is not None:
            veto_bias = np.asarray(veto_bias, dtype=float)
            modifier = cooperative_modifier + veto_bias
            features = coop_features.copy()
            residual = coop_residual + veto_bias
            jacobian = coop_jacobian.copy()
            scope = "cooperative_veto"
        else:
            modifier = 0.7 * primary_modifier + 0.3 * cooperative_modifier
            features = 0.7 * features + 0.3 * coop_features
            residual = 0.7 * residual + 0.3 * coop_residual
            jacobian = 0.7 * jacobian + 0.3 * coop_jacobian
            scope = "cooperative_blend"

    preclip_modifier = modifier.copy()
    modifier = np.clip(preclip_modifier, -1.0, 1.0)
    clip_derivative = (np.abs(preclip_modifier) < 1.0).astype(float)
    jacobian = jacobian * clip_derivative[:, np.newaxis]

    # Scale the attribution (not the Jacobian) so it reconstructs the exact
    # final modifier after the composition-level cap.
    for action_idx in range(3):
        before = float(preclip_modifier[action_idx])
        after = float(modifier[action_idx])
        if abs(before) > 1e-15:
            scale = after / before
            features[action_idx] *= scale
            residual[action_idx] *= scale
    # Absorb only arithmetic residue.
    residual = residual + modifier - (features.sum(axis=1) + residual)
    composition_trace = {
        "scope": scope,
        "clip_applied": True,
        "preclip_modifier": preclip_modifier,
        "clip_derivative": clip_derivative,
        "modifier_theta_jacobian": jacobian.copy(),
        "final_modifier": modifier.copy(),
    }
    return modifier, features, residual, scope, jacobian, composition_trace


class AgentCoordinator:
    """Orchestrates multi-agent decision-making across the supply chain.

    Parameters
    ----------
    agents : optional pre-configured list of agents.  When *None*,
        one of each role is created with default biases.
    context_enabled : whether to activate MCP/piR context injection.
    """

    def __init__(
        self,
        agents: Optional[List[SupplyChainAgent]] = None,
        context_enabled: bool = True,
        context_learner_overrides: Optional[Dict[str, Any]] = None,
        mode: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        context_learner_overrides : optional dict of keyword arguments passed
            verbatim to :class:`ContextMatrixLearner`. Used by the cold-start
            ablation and sensitivity ablation modes to override the default
            ``learning_rate``, ``initial_theta``, ``magnitude_cap_mode`` etc.
            When ``None`` the learner is instantiated with the default
            hand-calibrated initial matrix and declared benchmark hyperparameters.
        mode : optional operating mode. Supplying it constructs exactly the
            learners declared for that mode before a checkpoint is restored.
            Legacy callers may omit it; the first :meth:`step` lazily performs
            the same initialization.
        """
        if agents is None:
            agent_list: List[SupplyChainAgent] = [
                FarmAgent(),
                ProcessorAgent(),
                CooperativeAgent(),
                DistributorAgent(),
                RecoveryAgent(),
            ]
        else:
            agent_list = agents

        self.agents: Dict[str, SupplyChainAgent] = {
            a.role: a for a in agent_list
        }
        self._message_log: List[InterAgentMessage] = []
        self.context_enabled = context_enabled
        self._configured_mode = mode
        self._learning_mode: Optional[str] = None
        self._pending_learner_state: Optional[Dict[str, Any]] = None
        self._context_learner_overrides: Dict[str, Any] = dict(
            context_learner_overrides or {}
        )
        # The private sentinel freezes every learner before evaluation and is
        # removed before the remaining overrides reach ContextMatrixLearner.
        # The ordinary ``freeze`` key remains the context-learner switch.
        self._freeze_all_learners: bool = bool(
            self._context_learner_overrides.pop("_freeze_all_learners", False)
        )
        self._learner_freeze_reason: Optional[str] = (
            "configuration" if self._freeze_all_learners else None
        )
        self._external_policy_learner_ids_frozen: set[int] = set()
        if mode is not None:
            declared_caps = capabilities_for(mode)
            if declared_caps.frozen_learners:
                self._freeze_all_learners = True
                self._learner_freeze_reason = "mode_capability"
                self._context_learner_overrides.setdefault("freeze", True)
                self._context_learner_overrides.setdefault("learning_rate", 0.0)

        # Context infrastructure (lazy init, guarded by try/except)
        self._registry = None
        self._mcp_server = None
        self._shared_context = None
        self._temporal_window = None
        self._pirag_pipeline = None
        self._context_learner = None
        self._theta_learner = None
        self._theta_learners: Dict[str, Any] = {}
        self._reward_shaping_learner = None
        self._governance_skipped_learning_steps: int = 0
        self._context_evaluator = None
        self._context_log: List[Dict[str, Any]] = []
        self._decision_history: List[Dict[str, Any]] = []

        # Current-step context (for post_step use)
        self._step_mcp_results: Dict[str, Any] = {}
        self._step_rag_context: Dict[str, Any] = {}
        self._step_context_modifier: Optional[np.ndarray] = None
        self._step_context_features: Optional[np.ndarray] = None
        # Snapshot of the learned context matrix and the final modifier's
        # feature-resolved attribution before the learner update in post_step.
        # The latter includes temporal/physics scaling, clipping, and any
        # cooperative blend; a separate residual carries any declared fixed
        # cooperative operating-envelope adjustment.
        self._step_effective_context_theta: Optional[np.ndarray] = None
        self._step_context_feature_contributions: Optional[np.ndarray] = None
        self._step_context_nonfeature_residual: Optional[np.ndarray] = None
        # Exact derivative of the final applied context modifier with respect
        # to the shared 3x5 context matrix. This is distinct from the feature
        # attribution above and is the only matrix used by REINFORCE.
        self._step_context_modifier_theta_jacobian: Optional[np.ndarray] = None
        self._step_context_integration_trace: Optional[Dict[str, Any]] = None
        self._step_chosen_action_context_contributions: Optional[np.ndarray] = None
        self._step_chosen_action_context_residual: Optional[float] = None
        self._step_context_attribution_scope: Optional[str] = None
        # Pre-modifier argmax of the base logits, set by
        # ``select_action`` via its ``out`` side-channel when a context
        # modifier is supplied. This is an observer-only diagnostic for
        # policy-surface sensitivity. The reported context-influence rate uses
        # the paired pre-selection-RNG-state counterfactual stored separately
        # below. Stochastic regular-policy calls consume the same categorical
        # variate even when the live probability-gap override discards its draw.
        # None when not applicable (for example, the static path).
        self._step_base_argmax: Optional[int] = None
        # Observer-only ingredients for conditional feature-group masking.
        # ``_step_modifier_mcp`` / ``_step_modifier_pirag`` retain one feature
        # group at a time while reusing the observed MCP/retrieval results,
        # guards, learned matrix, and cooperative blend. They are policy-surface
        # diagnostics, not executions with a communication channel disabled.
        self._step_base_logits: Optional[List[float]] = None
        self._step_post_context_logits_pre_override: Optional[List[float]] = None
        self._step_slca_shaping: Optional[List[float]] = None
        self._step_slca_amp: Optional[float] = None
        self._step_policy_temperature: Optional[float] = None
        self._step_regime_logit_bias: Optional[List[float]] = None
        self._step_modifier_mcp: Optional[np.ndarray] = None
        self._step_modifier_pirag: Optional[np.ndarray] = None
        self._step_probs: Optional[np.ndarray] = None
        self._step_policy_probs_pre_override: Optional[np.ndarray] = None
        self._step_rules_fired: List[int] = []
        self._step_policy: Any = None
        self._step_mode: str = ""
        self._step_peer_messages_enabled: bool = True
        self._step_retrieval_kind: str = "pirag"
        self._step_scenario: str = "baseline"
        self._step_role_bias: Optional[np.ndarray] = None
        self._step_message_bias: Optional[np.ndarray] = None
        self._step_supply_hat: Optional[float] = None
        self._step_supply_std: Optional[float] = None
        self._step_demand_std: Optional[float] = None
        self._step_price_signal: Optional[float] = None
        self._step_rng_state: Optional[Dict[str, Any]] = None
        self._step_phi: Optional[np.ndarray] = None
        self._step_override: bool = False
        self._step_counterfactual_action: int = 0
        self._step_counterfactual_probs: Optional[np.ndarray] = None
        # Per-step benchmark-diagnostic flags for three declared internal
        # checks that can alter or reject a context contribution:
        #   _step_cooperative_veto: cooperative agent's compliance
        #     check found a critical violation the primary missed and
        #     replaced the primary's modifier with a recovery-biased
        #     override. Fires only inside the cooperative window
        #     (12-30h) and only when (coop_critical AND primary_missed).
        #   _step_fault_recovery: a fault-injection event was detected
        #     in this step (mcp_results["_fault_injected"] sentinel)
        #     and the policy fell back to defaults rather than
        #     propagating None tool results into the action.
        #   _step_physics_gate: the physics-consistency gate fired
        #     because retrieved-context physics_score < 0.03; the
        #     modifier was forced to zero so the policy did not act
        #     on inconsistent context.
        # All three are False by default and only set True for the
        # subset of modes that go through ``_compute_step_context``.
        # No-context deliberately initializes the same learners as the full
        # system but bypasses both external channels, so these diagnostic flags
        # remain false for that ablation.
        self._step_cooperative_veto: bool = False
        self._step_fault_recovery: bool = False
        # Number of successfully returned tool results deliberately replaced
        # by the H3 post-call fault treatment on this step.  This is kept
        # separate from the Boolean trigger because a scheduled opportunity
        # can occur when no tool is available (for example after the declared
        # cyber-outage cutoff).
        self._step_fault_injected_result_count: int = 0
        self._step_physics_gate: bool = False
        self._step_keywords: Dict[str, Any] = {}
        self._last_explanation: Optional[Dict[str, Any]] = None
        self._step_dispatch_cfg: Dict[str, Any] = {}
        # Detached, content-addressed causal inputs/outputs for the current
        # decision.  The snapshot is replaced (never mutated in place) when
        # context and post-decision emissions become available.
        self._step_channel_evidence: Dict[str, Any] = {}
        self._step_message_log_start: int = 0

        # Trace exporter and protocol recorder for paper evidence
        self._trace_exporter = None
        self._protocol_recorder = None

        if context_enabled:
            self._init_context_infrastructure()
        if mode is not None:
            self._init_policy_learning(mode)

    def _init_context_infrastructure(self) -> None:
        """Initialize MCP/piR infrastructure. Fails gracefully."""
        try:
            from pirag.context_eval import ContextEvaluator
            from pirag.context_learner import ContextMatrixLearner
            from pirag.context_to_logits import THETA_CONTEXT
            from pirag.mcp.agent_capabilities import register_all_agent_capabilities
            from pirag.mcp.context_sharing import SharedContextStore
            from pirag.mcp.prompts import register_prompts
            from pirag.mcp.protocol import MCPServer
            from pirag.mcp.registry import get_default_registry
            from pirag.mcp.resources import register_agent_resources
            from pirag.temporal_context import TemporalContextWindow

            self._registry = get_default_registry()
            self._mcp_server = MCPServer(registry=self._registry)

            # Register resources (use a closure for live state)
            self._agent_state_snapshot: Dict[str, Any] = {}
            register_agent_resources(
                self._mcp_server,
                lambda: self._agent_state_snapshot,
            )

            # Register prompts
            register_prompts(self._mcp_server)

            # Register agent capabilities
            register_all_agent_capabilities(self._mcp_server, self.agents)

            self._shared_context = SharedContextStore()
            self._temporal_window = TemporalContextWindow()
            learner_kwargs: Dict[str, Any] = {
                "initial_theta": THETA_CONTEXT,
                "learning_rate": 0.02,
                "magnitude_cap_mode": "relative_delta",
                "magnitude_cap_value": 0.5,
                "magnitude_cap_abs_floor": 0.10,
            }
            # Cold-start and sensitivity ablation modes override any of
            # these (e.g. initial_theta=zeros, magnitude_cap_mode="absolute",
            # perturbed initial_theta). Locked benchmark runs operate without
            # overrides, so its behavior is the refined-default above.
            learner_kwargs.update(self._context_learner_overrides)
            # The sign-unconstrained secondary arm changes exactly one rail.
            # Apply the capability after generic sensitivity overrides so a
            # contradictory override cannot silently invalidate the arm.
            if (self._configured_mode is not None
                    and not capabilities_for(
                        self._configured_mode
                    ).sign_constrained_learning):
                learner_kwargs["sign_constrained"] = False
            self._context_learner = ContextMatrixLearner(**learner_kwargs)
            self._context_evaluator = ContextEvaluator()

        except ImportError as exc:
            if os.environ.get("STRICT_VALIDATION", "0") == "1":
                raise RuntimeError(
                    "publication-critical context infrastructure import failed"
                ) from exc
            _log.warning(
                "context infrastructure unavailable; disabling context: %s",
                exc,
            )
            self.context_enabled = False

        try:
            from pirag.agent_pipeline import PiRPipeline
            self._pirag_pipeline = PiRPipeline()
        except ImportError as exc:
            if os.environ.get("STRICT_VALIDATION", "0") == "1":
                raise RuntimeError(
                    "publication-critical piR pipeline import failed"
                ) from exc
            _log.warning("piR pipeline unavailable: %s", exc)

        try:
            from pirag.trace_exporter import TraceExporter
            self._trace_exporter = TraceExporter(max_traces=50)
        except ImportError:
            pass

        try:
            from pirag.mcp.protocol_recorder import ProtocolRecorder
            if self._mcp_server is not None:
                # A 288-step publication episode can issue several MCP calls
                # per step. Keep the cap configurable but large enough to retain
                # a complete benchmark trace.
                record_cap = max(
                    4096, int(os.environ.get("PROTOCOL_MAX_RECORDS", "4096"))
                )
                self._protocol_recorder = ProtocolRecorder(
                    self._mcp_server, max_records=record_cap
                )
        except ImportError:
            pass

    def _init_policy_learning(self, mode: str) -> None:
        """Construct exactly the online learners declared for ``mode``.

        Policy learning is deliberately independent of MCP/retrieval setup.
        Hybrid RL therefore receives the same four decision-owner policy-delta
        learners as the other adaptive arms even though its external context
        channel is disabled. The cooperative agent is an observer/overlay and
        never owns a lifecycle decision stage, so it has no private learner.
        """

        caps = capabilities_for(mode)
        if self._learning_mode is not None:
            if self._learning_mode != mode:
                raise ValueError(
                    "one AgentCoordinator cannot switch learning modes within "
                    f"an episode: initialized={self._learning_mode!r}, got={mode!r}"
                )
            return

        if (self._context_learner is not None
                and not caps.sign_constrained_learning):
            # Covers legacy/lazy construction where the coordinator was
            # created without a mode and the first step selects the arm.
            self._context_learner.sign_constrained = False

        self._learning_mode = mode
        self._theta_learners = {}
        self._theta_learner = None
        self._reward_shaping_learner = None
        freeze = bool(self._freeze_all_learners or caps.frozen_learners)

        if caps.policy_delta_learning:
            try:
                from pirag.context_learner import PolicyDeltaLearner as _PDL

                from ..models.action_selection import THETA as _INITIAL_THETA

                self._theta_learners = {
                    role: _PDL(
                        initial_theta=_INITIAL_THETA,
                        sign_constrained=caps.sign_constrained_learning,
                        freeze=freeze,
                    )
                    for role in DECISION_OWNER_ROLES
                }
                self._theta_learner = self._theta_learners["farm"]
            except ImportError as exc:
                if os.environ.get("STRICT_VALIDATION", "0") == "1":
                    raise RuntimeError(
                        "publication-critical policy-delta learner import failed"
                    ) from exc
                _log.warning("policy-delta learner unavailable: %s", exc)

        if caps.reward_shaping_learning:
            try:
                from pirag.context_learner import RewardShapingLearner as _RSL

                from ..models.action_selection import (
                    NO_SLCA_OFFSET as _INITIAL_NO_SLCA_OFFSET,
                )
                from ..models.action_selection import (
                    SLCA_BONUS as _INITIAL_SLCA_BONUS,
                )
                from ..models.action_selection import (
                    SLCA_RHO_BONUS as _INITIAL_SLCA_RHO_BONUS,
                )

                self._reward_shaping_learner = _RSL(
                    initial_slca_bonus=_INITIAL_SLCA_BONUS,
                    initial_slca_rho_bonus=_INITIAL_SLCA_RHO_BONUS,
                    # Compatibility field only; it is zero and absent from the
                    # current no-SLCA logit equation.
                    initial_no_slca_offset=_INITIAL_NO_SLCA_OFFSET,
                    sign_constrained=caps.sign_constrained_learning,
                    freeze=freeze,
                )
            except ImportError as exc:
                if os.environ.get("STRICT_VALIDATION", "0") == "1":
                    raise RuntimeError(
                        "publication-critical reward-shaping learner import failed"
                    ) from exc
                _log.warning("reward-shaping learner unavailable: %s", exc)

        if self._pending_learner_state is not None:
            pending = self._pending_learner_state
            self._pending_learner_state = None
            self._load_learner_states_now(pending)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def freeze_learners(
        self,
        *external_learners: Any,
        reason: str = "retained_episode_evaluation",
    ) -> Dict[str, Any]:
        """Freeze every learner while preserving its loaded parameters.

        The publication protocol adapts on episodes 0--2 and evaluates the
        retained episode 3 with no updates.  This operation is deliberately
        one-way for the lifetime of a coordinator: it sets the central update
        guard, freezes the context matrix, every role-specific policy delta,
        reward shaping, and any explicitly supplied legacy ``PolicyLearner``.
        It never resets weights, counters, baselines, or replay buffers.

        ``external_learners`` exists because the optional replay-buffer
        ``PolicyLearner`` is owned by the simulator rather than the coordinator.
        Pass that object here when ``ONLINE_LEARNING`` is enabled.
        """

        self._freeze_all_learners = True
        self._learner_freeze_reason = str(reason)

        if self._context_learner is not None:
            self._context_learner.freeze = True
        for learner in self._theta_learners.values():
            learner.freeze = True
        if self._theta_learner is not None and not self._theta_learners:
            self._theta_learner.freeze = True
        if self._reward_shaping_learner is not None:
            self._reward_shaping_learner.freeze = True

        for learner in external_learners:
            if learner is None:
                continue
            freeze_updates = getattr(learner, "freeze_updates", None)
            if callable(freeze_updates):
                freeze_updates()
            elif hasattr(learner, "freeze"):
                learner.freeze = True
            else:
                raise TypeError(
                    "external learner must expose freeze_updates() or freeze"
                )
            self._external_policy_learner_ids_frozen.add(id(learner))

        return self.learner_freeze_summary()

    def learner_freeze_summary(self) -> Dict[str, Any]:
        """Describe the actual update state of every learner family."""

        return {
            "learners_frozen": bool(self._freeze_all_learners),
            "learner_phase": (
                "frozen_evaluation"
                if self._freeze_all_learners else "adaptive_training"
            ),
            "freeze_reason": self._learner_freeze_reason,
            "context_matrix_frozen": (
                bool(self._context_learner.freeze)
                if self._context_learner is not None else None
            ),
            "policy_delta_frozen_by_role": {
                role: bool(learner.freeze)
                for role, learner in self._theta_learners.items()
            },
            "reward_shaping_frozen": (
                bool(self._reward_shaping_learner.freeze)
                if self._reward_shaping_learner is not None else None
            ),
            "external_policy_learners_frozen": int(
                len(self._external_policy_learner_ids_frozen)
            ),
        }

    def reset(self) -> None:
        """Reset all agents, context stores, logs, and per-episode counters."""
        for agent in self.agents.values():
            agent.reset()
        self._message_log.clear()
        self._context_log.clear()
        self._decision_history.clear()
        # Reset the MCP dispatch-id counter so per-episode protocol
        # traces use comparable id ranges (prevents the global counter
        # from growing unboundedly across the simulator's mode/scenario
        # loop).
        try:
            from pirag.mcp.tool_dispatch import reset_dispatch_id_counter
            reset_dispatch_id_counter()
        except Exception:
            pass
        self._step_mcp_results = {}
        self._step_rag_context = {}
        self._step_context_modifier = None
        self._step_context_features = None
        self._step_effective_context_theta = None
        self._step_context_feature_contributions = None
        self._step_context_nonfeature_residual = None
        self._step_context_modifier_theta_jacobian = None
        self._step_context_integration_trace = None
        self._step_chosen_action_context_contributions = None
        self._step_chosen_action_context_residual = None
        self._step_context_attribution_scope = None
        self._step_base_argmax = None
        self._step_base_logits = None
        self._step_post_context_logits_pre_override = None
        self._step_slca_shaping = None
        self._step_slca_amp = None
        self._step_policy_temperature = None
        self._step_regime_logit_bias = None
        self._step_modifier_mcp = None
        self._step_modifier_pirag = None
        self._step_probs = None
        self._step_policy_probs_pre_override = None
        self._step_policy_categorical_uniform = None
        self._step_sampled_action_pre_override = None
        self._step_theta_delta = None
        self._step_slca_bonus_delta = None
        self._step_slca_rho_delta = None
        self._step_no_slca_offset_delta = None
        self._step_combined_role_bias = None
        self._step_rules_fired = []
        self._step_policy = None
        self._step_mode = ""
        self._step_peer_messages_enabled = True
        self._step_retrieval_kind = "pirag"
        self._step_scenario = "baseline"
        self._step_role_bias = None
        self._step_message_bias = None
        self._step_supply_hat = None
        self._step_supply_std = None
        self._step_demand_std = None
        self._step_price_signal = None
        self._step_rng_state = None
        self._step_phi = None
        self._step_override = False
        self._step_counterfactual_action = 0
        self._step_counterfactual_probs = None
        self._step_counterfactual_categorical_uniform = None
        self._step_counterfactual_sampled_action_pre_override = None
        self._step_cooperative_veto = False
        self._step_fault_recovery = False
        self._step_fault_injected_result_count = 0
        self._step_physics_gate = False
        self._step_keywords = {}
        self._last_explanation = None
        self._step_dispatch_cfg = {}
        self._step_channel_evidence = {}
        self._step_message_log_start = 0

        if self._protocol_recorder is not None:
            self._protocol_recorder.reset()
        if self._trace_exporter is not None:
            self._trace_exporter.reset()
        if self._shared_context is not None:
            self._shared_context.reset()
        if self._temporal_window is not None:
            self._temporal_window.reset()
        if self._context_learner is not None:
            self._context_learner.reset()
        # Reset every per-role theta learner (not just the active one)
        # so a new episode starts every role from a clean delta.
        for _learner in self._theta_learners.values():
            _learner.reset()
        if self._theta_learner is not None and not self._theta_learners:
            # Legacy single-learner code path (only triggers if the
            # per-role import failed and we fell back to None).
            self._theta_learner.reset()
        if self._reward_shaping_learner is not None:
            self._reward_shaping_learner.reset()
        self._governance_skipped_learning_steps = 0
        if self._context_evaluator is not None:
            self._context_evaluator.reset()
        if self._registry is not None:
            self._registry.clear_cache()

    # ------------------------------------------------------------------
    # Step logic
    # ------------------------------------------------------------------

    def get_active_agent(self, hour: float) -> SupplyChainAgent:
        """Return the agent responsible for the current lifecycle stage."""
        role = stage_for_hour(hour)
        return self.agents[role]

    def step(
        self,
        env_state: Dict[str, Any],
        hour: float,
        mode: str,
        policy: Any,
        rng: np.random.Generator,
        scenario: str = "baseline",
        rag_context: Optional[Dict[str, Any]] = None,
        policy_temperature: float = 1.0,
        policy_categorical_uniform: float | None = None,
    ) -> tuple:
        """Run one decision step through the active agent.

        The cooperative agent additionally participates as an overlay
        during hours 12-30, observing state and generating messages
        alongside the primary stage agent.

        Returns
        -------
        (action_idx, probs, active_agent)
        """
        self._init_policy_learning(mode)
        caps = capabilities_for(mode)
        active = self.get_active_agent(hour)
        self._step_message_log_start = len(self._message_log)

        # Recovery capacity must reach a decision owner before that owner acts.
        # The lifecycle is forward-only, so a broadcast emitted after hour 54
        # cannot be consumed by farm/processor/distributor later in the same
        # episode. During the distributor stage, solicit one normalized update
        # from the recovery peer and place it in the distributor inbox before
        # observation flushes the inbox into the policy-bias vector.
        if caps.peer_messages and active.role == "distributor":
            recovery_peer = self.agents.get("recovery")
            if recovery_peer is not None:
                capacity_message = recovery_peer.make_capacity_update(
                    hour, recipient=active.agent_id,
                )
                if capacity_message is not None:
                    self._message_log.append(capacity_message)
                    active.receive_message(capacity_message)

        # Route ``_theta_learner`` to the active role's per-role learner
        # so this step's `get_theta_delta()` and `update(...)` calls
        # operate on the role-specific Theta_delta. Cooperative overlay
        # is handled separately below.
        if self._theta_learners and active.role in self._theta_learners:
            self._theta_learner = self._theta_learners[active.role]
        obs = (
            active.observe(env_state, hour)
            if caps.peer_messages
            else _observe_without_peer_inbox(active, env_state, hour)
        )
        _active_consumed_messages = list(
            getattr(obs, "messages", []) or []
        )
        _cooperative_consumed_messages: List[InterAgentMessage] = []

        # Cooperative overlay: observe + generate messages during the
        # cooperative window.
        cooperative = self.agents.get("cooperative")
        if cooperative is not None and cooperative is not active and _cooperative_window_active(hour):
            if caps.peer_messages:
                _cooperative_step_obs = cooperative.observe(env_state, hour)
            else:
                _cooperative_step_obs = _observe_without_peer_inbox(
                    cooperative, env_state, hour,
                )
            _cooperative_consumed_messages = list(
                getattr(_cooperative_step_obs, "messages", []) or []
            )

        # Compute combined role bias: primary agent + cooperative overlay
        # + inter-agent message bias.
        combined_bias = active.role_bias.copy()
        cooperative = self.agents.get("cooperative")
        if cooperative is not None and cooperative is not active and _cooperative_window_active(hour):
            combined_bias = combined_bias + cooperative.role_bias

        # 2026-04 fix: messages received in the active agent's inbox now
        # actually shape the decision. The previous implementation
        # appended messages to ``Observation.messages`` and then did
        # nothing with them, making the documented protocol
        # (SPOILAGE_ALERT / SURPLUS_ALERT / CAPACITY_UPDATE /
        # REROUTE_REQUEST / ACK) non-falsifiable. ``message_bias_from_inbox``
        # converts the flushed inbox into a bounded logit nudge in
        # action space; the bias is added to ``combined_bias`` here so
        # the same code path that consumes role_bias also consumes
        # message-derived bias.
        if caps.peer_messages:
            try:
                from .message import message_bias_from_inbox as _mbias
                inbox_bias = _mbias(getattr(obs, "messages", []) or [])
                combined_bias = combined_bias + inbox_bias
                self._step_message_bias = inbox_bias
                _message_bias_applied = True
            except Exception as exc:
                if os.environ.get("STRICT_VALIDATION", "0") == "1":
                    raise RuntimeError(
                        "publication-critical peer-message bias conversion failed"
                    ) from exc
                _log.warning("peer-message bias conversion skipped: %s", exc)
                self._step_message_bias = np.zeros(3)
                _message_bias_applied = False
        else:
            # Structural zero: no-peer observations cannot consume or convert
            # a queued message into a policy-logit contribution.
            self._step_message_bias = np.zeros(3)
            _message_bias_applied = False

        # Context injection for context-enabled modes
        context_modifier = None
        self._step_mcp_results = {}
        self._step_rag_context = {}
        self._step_context_modifier = None
        self._step_context_features = None
        self._step_effective_context_theta = None
        self._step_context_feature_contributions = None
        self._step_context_nonfeature_residual = None
        self._step_context_modifier_theta_jacobian = None
        self._step_context_integration_trace = None
        self._step_chosen_action_context_contributions = None
        self._step_chosen_action_context_residual = None
        self._step_context_attribution_scope = None
        self._step_base_argmax = None
        self._step_base_logits = None
        self._step_post_context_logits_pre_override = None
        self._step_slca_shaping = None
        self._step_slca_amp = None
        self._step_policy_temperature = None
        self._step_regime_logit_bias = None
        self._step_modifier_mcp = None
        self._step_modifier_pirag = None
        self._step_probs = None
        self._step_policy_probs_pre_override = None
        # These are produced in post_step() from the RNG snapshot captured
        # below. Reset them on every step so a failed or inapplicable
        # context ablation cannot reuse values from the preceding decision.
        self._step_counterfactual_action = 0
        self._step_counterfactual_probs = None
        self._step_rules_fired = []
        self._step_policy = policy
        self._step_mode = mode
        self._step_peer_messages_enabled = bool(caps.peer_messages)
        self._step_retrieval_kind = str(caps.retrieval_kind)
        self._step_scenario = scenario
        self._step_role_bias = combined_bias
        self._step_override = False
        # Reset per-step diagnostic mechanism-activation flags. They get set True by
        # _compute_step_context when triggered; modes that skip the
        # context path leave them at False (structural zero).
        self._step_cooperative_veto = False
        self._step_fault_recovery = False
        self._step_fault_injected_result_count = 0
        self._step_physics_gate = False
        self._step_dispatch_cfg = {
            "enable_qos_routing": bool(getattr(policy, "enable_mcp_qos_routing", False)),
            "enable_reliability": bool(getattr(policy, "enable_mcp_reliability", False)),
            "qos_profile": "heterogeneous" if bool(getattr(policy, "enable_heterogeneous_profiles", False)) else "legacy",
            "retries": 1,
        }
        if bool(getattr(policy, "enable_heterogeneous_profiles", False)):
            role_profile = getattr(active, "profile", {})
            self._step_dispatch_cfg["role_preferred_qos"] = role_profile.get("preferred_qos", "standard")

        context_mode = _CONTEXT_MODE_MAP.get(mode)
        _context_will_run = bool(
            self.context_enabled
            and context_mode is not None
            and self._registry is not None
        )
        if not self.context_enabled:
            _context_skip_reason = "context_disabled"
        elif context_mode is None:
            _context_skip_reason = "mode_has_no_external_context"
        elif self._registry is None:
            _context_skip_reason = "context_registry_unavailable"
        else:
            _context_skip_reason = "pending_context_capture"
        _empty_primary_mcp, _empty_primary_retrieval = (
            _empty_channel_evidence(
                requested_kind=str(caps.retrieval_kind),
                skip_reason=_context_skip_reason,
            )
        )
        _empty_coop_mcp, _empty_coop_retrieval = _empty_channel_evidence(
            requested_kind=str(caps.retrieval_kind),
            skip_reason=(
                _context_skip_reason
                if not _context_will_run
                else "cooperative_overlay_not_executed"
            ),
        )
        _consumed_channel_messages = [
            (message, active.role, _message_bias_applied)
            for message in _active_consumed_messages
        ] + [
            (message, "cooperative", False)
            for message in _cooperative_consumed_messages
        ]
        self._step_channel_evidence = _seal_evidence_record({
            "schema_version": "agribrain.step_channel_evidence.v1",
            "hour": hour,
            "active_role": active.role,
            "peer": _build_peer_channel_evidence(
                _consumed_channel_messages,
                list(self._message_log[self._step_message_log_start:]),
                self._step_message_bias,
                enabled=bool(caps.peer_messages),
            ),
            "primary": _seal_evidence_record({
                "role": active.role,
                "mcp": _empty_primary_mcp,
                "retrieval": _empty_primary_retrieval,
            }),
            "cooperative": _seal_evidence_record({
                "active": bool(
                    cooperative is not None
                    and cooperative is not active
                    and _cooperative_window_active(hour)
                ),
                "role": "cooperative",
                "mcp": _empty_coop_mcp,
                "retrieval": _empty_coop_retrieval,
            }),
        })
        if _context_will_run:
            context_modifier = self._compute_step_context(
                active, obs, scenario, hour,
                context_mode=context_mode,
                retrieval_kind=caps.retrieval_kind,
            )

        # Get learned SLCA amp coefficient
        slca_amp = None
        if self._context_learner is not None and hasattr(self._context_learner, 'get_slca_amp'):
            slca_amp = self._context_learner.get_slca_amp()
        if mode == "no_slca":
            slca_amp = 0.0

        # Supply and demand forecast quantities flow into the state vector
        # (phi_6..phi_8) and the price-volatility proxy feeds phi_9. They
        # are carried on obs.raw; missing keys default to None, in which
        # case build_feature_vector emits zeros on the corresponding
        # channels.
        raw = getattr(obs, "raw", {}) or {}
        supply_hat = raw.get("supply_hat")
        if isinstance(supply_hat, (list, tuple)) and supply_hat:
            supply_hat = supply_hat[0]
        supply_std = raw.get("supply_std")
        demand_std = raw.get("demand_std")
        price_signal = raw.get("price_signal")
        self._step_supply_hat = supply_hat
        self._step_supply_std = supply_std
        self._step_demand_std = demand_std
        self._step_price_signal = price_signal

        # Snapshot the RNG state before the live call consumes from it.
        # The counterfactual in post_step() rebuilds a fresh generator from
        # this state so both stochastic calls consume the same categorical
        # variate. The live probability-gap override may discard the sampled action,
        # but select_action still consumes the draw. The only controlled policy
        # difference is context_modifier (None in the CF, computed live).
        self._step_rng_state = copy.deepcopy(rng.bit_generator.state)

        # Cache phi (10D state feature vector) for the forecast-column
        # learner update in post_step. Cheap to compute; keeps post_step
        # from having to thread the forecast kwargs a second time.
        from ..models.action_selection import build_feature_vector as _bfv
        self._step_phi = _bfv(
            obs.rho, obs.inv, obs.y_hat, obs.temp,
            supply_hat=supply_hat, supply_std=supply_std, demand_std=demand_std,
            price_signal=price_signal,
        )

        theta_delta = (
            self._theta_learner.get_theta_delta()
            if self._theta_learner is not None else None
        )
        if self._reward_shaping_learner is not None:
            _slca_bonus_delta = self._reward_shaping_learner.get_slca_bonus_delta()
            _slca_rho_delta = self._reward_shaping_learner.get_slca_rho_delta()
            _no_slca_offset_delta = self._reward_shaping_learner.get_no_slca_offset_delta()
        else:
            _slca_bonus_delta = None
            _slca_rho_delta = None
            _no_slca_offset_delta = None
        self._step_theta_delta = (
            np.asarray(theta_delta, dtype=float).copy()
            if theta_delta is not None else np.zeros((3, 10), dtype=float)
        )
        self._step_slca_bonus_delta = (
            np.asarray(_slca_bonus_delta, dtype=float).copy()
            if _slca_bonus_delta is not None else np.zeros(3, dtype=float)
        )
        self._step_slca_rho_delta = (
            np.asarray(_slca_rho_delta, dtype=float).copy()
            if _slca_rho_delta is not None else np.zeros(3, dtype=float)
        )
        self._step_no_slca_offset_delta = (
            np.asarray(_no_slca_offset_delta, dtype=float).copy()
            if _no_slca_offset_delta is not None else np.zeros(3, dtype=float)
        )
        self._step_combined_role_bias = np.asarray(
            combined_bias, dtype=float,
        ).copy()

        # Observer-only pre-context argmax for policy/channel diagnostics.
        # Context influence is measured later from a paired context ablation
        # that reuses the saved RNG state; it does not compare this argmax with
        # the sampled live action.
        self._step_base_argmax: Optional[int] = None
        _select_out: Dict[str, object] = {}
        action_idx, probs = select_action(
            mode=mode,
            rho=obs.rho,
            inv=obs.inv,
            y_hat=obs.y_hat,
            temp=obs.temp,
            tau=obs.tau,
            policy=policy,
            rng=rng,
            scenario=scenario,
            hour=hour,
            role_bias=combined_bias,
            context_modifier=context_modifier,
            slca_amp_coeff=slca_amp,
            supply_hat=supply_hat,
            supply_std=supply_std,
            demand_std=demand_std,
            price_signal=price_signal,
            theta_delta=theta_delta,
            slca_bonus_delta=_slca_bonus_delta,
            slca_rho_delta=_slca_rho_delta,
            no_slca_offset_delta=_no_slca_offset_delta,
            policy_temperature=policy_temperature,
            categorical_uniform=policy_categorical_uniform,
            out=_select_out,
        )
        if "base_argmax" in _select_out:
            self._step_base_argmax = int(_select_out["base_argmax"])
        if self._step_context_feature_contributions is not None:
            contribution_matrix = np.asarray(
                self._step_context_feature_contributions, dtype=float,
            )
            if contribution_matrix.shape == (3, 5):
                self._step_chosen_action_context_contributions = (
                    contribution_matrix[int(action_idx)].copy()
                )
                residual = self._step_context_nonfeature_residual
                self._step_chosen_action_context_residual = (
                    float(np.asarray(residual, dtype=float)[int(action_idx)])
                    if residual is not None else 0.0
                )
        # Capture the observer-only decision ingredients (no effect on the
        # chosen action; see select_action ``out`` docstring) for the H2
        # channel-attribution ledger fields.
        self._step_base_logits = _select_out.get("base_logits")
        self._step_post_context_logits_pre_override = _select_out.get(
            "post_context_logits_pre_override"
        )
        self._step_slca_shaping = _select_out.get("slca_shaping")
        self._step_slca_amp = _select_out.get("slca_amp")
        self._step_policy_temperature = _select_out.get("policy_temperature")
        self._step_regime_logit_bias = _select_out.get("regime_logit_bias")
        _pre_override_probs = _select_out.get("policy_probs_pre_override")
        self._step_policy_probs_pre_override = (
            np.asarray(_pre_override_probs, dtype=float)
            if _pre_override_probs is not None else np.asarray(probs, dtype=float)
        )
        self._step_policy_categorical_uniform = _select_out.get(
            "policy_categorical_uniform"
        )
        self._step_sampled_action_pre_override = int(
            _select_out["sampled_action_pre_override"]
        )

        # Store probs for learner update
        self._step_probs = probs

        # Track the author-declared probability-gap override
        self._step_override = bool(_select_out.get("governance_override", False))

        return action_idx, probs, active

    def _compute_step_context(
        self,
        active: SupplyChainAgent,
        obs: Observation,
        scenario: str,
        hour: float,
        context_mode: str = "full",
        retrieval_kind: str = "pirag",
    ) -> Optional[np.ndarray]:
        """Compute MCP/piR context modifier for the current step."""
        try:
            from pirag.context_builder import retrieve_role_context
            from pirag.context_to_logits import (
                THETA_CONTEXT,
                compute_context_modifier,
            )
            from pirag.mcp.tool_dispatch import dispatch_tools

            # Update live state snapshot for MCP resources
            self._agent_state_snapshot = {
                "temp": obs.temp, "rh": obs.rh, "inv": obs.inv,
                "rho": obs.rho, "y_hat": obs.y_hat, "tau": obs.tau,
            }

            # Structural ablation gating. The channel itself is gated:
            # ``pirag_only`` does not
            # call ``dispatch_tools`` and ``mcp_only`` does not call
            # ``retrieve_role_context``. The feature mask is a second
            # containment layer; no mode-specific ablation bias is added.
            # In the cyber-outage scenario the live MCP/tool channel becomes
            # unavailable from hour 24. Locally cached retrieval remains
            # available, so the resulting policy response is produced by the
            # normal context-to-logit pathway rather than a hard-coded
            # mode-specific rerouting probability.
            _cyber_mcp_offline = (scenario == "cyber_outage" and hour >= 24.0)
            _skip_mcp = (context_mode == "pirag_only") or _cyber_mcp_offline
            _skip_rag = (context_mode == "mcp_only")

            # MCP tool dispatch (route through protocol for recording)
            _primary_mcp_cursor = _protocol_cursor(self._protocol_recorder)
            if _skip_mcp:
                mcp_results = {"_tools_invoked": []}
                if _cyber_mcp_offline:
                    mcp_results["_channel_unavailable"] = "cyber_outage"
                else:
                    mcp_results["_ablation_skipped"] = "mcp"
                _primary_mcp_returned = _strict_json_native(mcp_results)
            else:
                mcp_results = dispatch_tools(
                    active.role, obs, self._registry, self._shared_context,
                    mcp_server=self._mcp_server,
                    dispatch_config=self._step_dispatch_cfg,
                )
                # Preserve the dispatcher return before the H3 treatment can
                # replace successful values with None.  This is a detached
                # observer snapshot and is never passed back to policy code.
                _primary_mcp_returned = _strict_json_native(mcp_results)
                if isinstance(obs.raw, dict):
                    flags = obs.raw.get("policy_flags", {})
                    if flags.get("enable_failure_injection", False):
                        # Deterministic injection pattern for reproducibility.
                        if int(hour) % 11 == 0:
                            mcp_results["_fault_injected"] = "drop_tool_results"
                            invoked = list(mcp_results.get("_tools_invoked", []))
                            self._step_fault_injected_result_count = len(invoked)
                            for tool_name in invoked:
                                mcp_results[tool_name] = None
                            # Diagnostic trigger: the simulator detected the
                            # injected fault (the sentinel write above
                            # is the detection event) and falls back to
                            # default-prior policy below rather than
                            # propagating None tool results into the
                            # action. Recorded as one fault-recovery activation.
                            self._step_fault_recovery = True

                # Publish to shared context
                if self._shared_context is not None:
                    for tool_name in mcp_results.get("_tools_invoked", []):
                        self._shared_context.publish(
                            active.role, tool_name, mcp_results.get(tool_name), hour,
                        )
            _primary_mcp_protocol = _protocol_window(
                self._protocol_recorder, _primary_mcp_cursor,
            )

            # piR retrieval
            _primary_retrieval_cursor = _protocol_cursor(
                self._protocol_recorder
            )
            if _skip_rag:
                rag_context = {
                    "query": "",
                    "top_doc_id": "",
                    "top_citation_score": 0.0,
                    # Keep the structural-ablation sentinel numerically
                    # identical to the decision-ledger score aliases.  The
                    # producer records an unavailable retrieval as zero
                    # policy/ordering strength, so the sealed channel evidence
                    # must not serialize these two fields as ``None``.
                    "top_fused_score": 0.0,
                    "top_rerank_score": 0.0,
                    "regulatory_guidance": "",
                    "sop_guidance": "",
                    "waste_hierarchy_guidance": "",
                    "governance_guidance": "",
                    "_ablation_skipped": "pirag",
                }
            else:
                rag_context = retrieve_role_context(
                    active.role, obs, scenario, mcp_results,
                    self._pirag_pipeline, self._mcp_server,
                    retrieval_kind=retrieval_kind,
                )
            _primary_retrieval_protocol = _protocol_window(
                self._protocol_recorder, _primary_retrieval_cursor,
            )
            if self._context_evaluator is not None:
                cf = rag_context.get("counterfactual", {})
                if isinstance(cf, dict) and cf:
                    self._context_evaluator.record_retrieval_counterfactual(
                        hour=hour,
                        role=active.role,
                        top_doc_id=rag_context.get("top_doc_id", ""),
                        cf_top_doc_id=cf.get("top_doc_id", ""),
                    )

            # Record only actual retrieval events. In particular, the MCP-only
            # structural arm must not fill the retrieval-continuity window with
            # blank document identifiers.
            if (self._temporal_window is not None
                    and not _skip_rag
                    and rag_context.get("top_doc_id")):
                guidance_type = ""
                if rag_context.get("regulatory_guidance"):
                    guidance_type = "regulatory"
                elif rag_context.get("sop_guidance"):
                    guidance_type = "sop"
                elif rag_context.get("waste_hierarchy_guidance"):
                    guidance_type = "waste_hierarchy"
                elif rag_context.get("governance_guidance"):
                    guidance_type = "governance"

                self._temporal_window.add(
                    hour, active.role, rag_context.get("query", ""),
                    rag_context.get("top_doc_id", ""),
                    rag_context.get("top_citation_score", 0.0),
                    guidance_type,
                )

            # Get learned parameters from ContextMatrixLearner
            theta_override = None
            slca_amp_override = None
            temporal_params_override = None
            if self._context_learner is not None and hasattr(self._context_learner, 'get_theta'):
                theta_override = self._context_learner.get_theta()
                slca_amp_override = self._context_learner.get_slca_amp()
                temporal_params_override = self._context_learner.get_temporal_params()
            self._step_effective_context_theta = np.asarray(
                theta_override if theta_override is not None else THETA_CONTEXT,
                dtype=float,
            ).copy()

            # Physics-consistency gate detection. compute_context_modifier
            # below will force the piR term to zero if the policy_flags
            # have ``enable_physics_consistency_gate`` set AND the
            # retrieved-context physics_consistency_score is below 0.03
            # (the threshold below which the retrieval is treated as
            # likely-anomalous and not used to nudge the policy). We
            # replicate the same condition here so the coordinator's
            # per-step ``_step_physics_gate`` flag can be set without
            # invasively threading a return value through the modifier
            # function. Recorded as one retrieval physics-gate activation
            # (the policy did not use the inconsistent retrieval term).
            if isinstance(obs.raw, dict):
                _pf = obs.raw.get("policy_flags", {})
                _physics_enabled = bool(
                    _pf.get("enable_physics_consistency_gate", False)
                )
            else:
                _physics_enabled = False
            _physics_score = float(rag_context.get(
                "physics_consistency_score", 1.0,
            ))
            if (retrieval_kind == "pirag"
                    and _physics_enabled and _physics_score < 0.03):
                self._step_physics_gate = True

            primary_trace: Dict[str, Any] = {}
            modifier = compute_context_modifier(
                mcp_results, rag_context, obs,
                self._temporal_window,
                theta_override=theta_override,
                slca_amp_override=slca_amp_override,
                temporal_params_override=temporal_params_override,
                context_mode=context_mode,
                retrieval_kind=retrieval_kind,
                trace_out=primary_trace,
            )
            (modifier, final_feature_contributions,
             final_nonfeature_residual, attribution_scope,
             final_modifier_theta_jacobian, composition_trace) = (
                 _compose_context_attribution(modifier, primary_trace)
             )

            # Cooperative overlay during the 12-30h cooperative window.
            # Two author-declared paths are recorded: (1) ordinary modifier
            # composition, `0.7*primary + 0.3*cooperative`; (2) when the
            # cooperative tool call flags a critical synthetic
            # operating-envelope event that the primary call missed, replace
            # the primary modifier with the cooperative modifier plus the
            # fixed bounded adjustment [-0.20, +0.20, 0.00].  This is distinct
            # from the later probability-gap rule and is not a legal or
            # calibrated safety determination.
            cooperative = self.agents.get("cooperative")
            self._step_cooperative_veto = False
            # Captured cooperative-overlay state, reused by the observer-only
            # per-channel modifier split below so the split applies the same
            # blend / veto the live modifier received (no second dispatch).
            _coop_obs = None
            _coop_mcp = None
            _coop_rag = None
            cooperative_trace_for_record = None
            _coop_context_consumed_messages: List[InterAgentMessage] = []
            _coop_context_error = None
            _coop_mcp_error = None
            _coop_retrieval_error = None
            _coop_mcp_returned: Dict[str, Any] = {"_tools_invoked": []}
            _coop_mcp_effective: Dict[str, Any] = {"_tools_invoked": []}
            _coop_rag_for_evidence: Dict[str, Any] = {}
            _coop_mcp_protocol = _protocol_window(None, (0, 0))
            _coop_retrieval_protocol = _protocol_window(None, (0, 0))
            _coop_mcp_cursor = None
            _coop_retrieval_cursor = None
            _coop_mcp_window_closed = True
            _coop_retrieval_window_closed = True
            _coop_stage = "inactive"
            _coop_overlay_active = bool(
                cooperative is not None
                and cooperative is not active
                and _cooperative_window_active(obs.hour)
            )
            if (cooperative is not None
                    and cooperative is not active
                    and _cooperative_window_active(obs.hour)):
                try:
                    _coop_stage = "observe"
                    coop_obs = (
                        cooperative.observe(obs.raw, obs.hour)
                        if self._step_peer_messages_enabled
                        else _observe_without_peer_inbox(
                            cooperative, obs.raw, obs.hour,
                        )
                    )
                    _coop_context_consumed_messages = list(
                        getattr(coop_obs, "messages", []) or []
                    )
                    # Honor ablation gating in the cooperative overlay
                    # too — otherwise ``pirag_only`` would re-introduce
                    # MCP signals via the cooperative dispatch and
                    # ``mcp_only`` would re-introduce piR via
                    # cooperative retrieval, defeating the structural
                    # ablation the post-audit fix is meant to enforce.
                    _coop_stage = "mcp_dispatch"
                    _coop_mcp_cursor = _protocol_cursor(
                        self._protocol_recorder
                    )
                    _coop_mcp_window_closed = False
                    if _skip_mcp:
                        coop_mcp = {"_tools_invoked": []}
                        if _cyber_mcp_offline:
                            coop_mcp["_channel_unavailable"] = "cyber_outage"
                        else:
                            coop_mcp["_ablation_skipped"] = "mcp"
                    else:
                        coop_mcp = dispatch_tools(
                            "cooperative", coop_obs, self._registry, self._shared_context,
                            mcp_server=self._mcp_server,
                            dispatch_config=self._step_dispatch_cfg,
                        )
                    _coop_mcp_returned = _strict_json_native(coop_mcp)
                    _coop_mcp_effective = coop_mcp
                    _coop_mcp_protocol = _protocol_window(
                        self._protocol_recorder, _coop_mcp_cursor,
                    )
                    _coop_mcp_window_closed = True
                    _coop_stage = "retrieval"
                    _coop_retrieval_cursor = _protocol_cursor(
                        self._protocol_recorder
                    )
                    _coop_retrieval_window_closed = False
                    if _skip_rag:
                        coop_rag = {
                            "query": "",
                            "top_doc_id": "",
                            "top_citation_score": 0.0,
                            # Mirror the primary structural-ablation sentinel;
                            # both role-specific evidence blocks are bound to
                            # the same numeric ledger score convention.
                            "top_fused_score": 0.0,
                            "top_rerank_score": 0.0,
                            "regulatory_guidance": "",
                            "sop_guidance": "",
                            "waste_hierarchy_guidance": "",
                            "governance_guidance": "",
                            "_ablation_skipped": "pirag",
                        }
                    else:
                        coop_rag = retrieve_role_context(
                            "cooperative", coop_obs, "", coop_mcp,
                            self._pirag_pipeline, self._mcp_server,
                            retrieval_kind=retrieval_kind,
                        )
                    _coop_rag_for_evidence = coop_rag
                    _coop_retrieval_protocol = _protocol_window(
                        self._protocol_recorder, _coop_retrieval_cursor,
                    )
                    _coop_retrieval_window_closed = True
                    _coop_obs, _coop_mcp, _coop_rag = coop_obs, coop_mcp, coop_rag
                    _coop_stage = "context_modifier"
                    coop_trace: Dict[str, Any] = {}
                    coop_modifier = compute_context_modifier(
                        coop_mcp, coop_rag, coop_obs,
                        self._temporal_window,
                        theta_override=theta_override,
                        temporal_params_override=temporal_params_override,
                        context_mode=context_mode,
                        retrieval_kind=retrieval_kind,
                        trace_out=coop_trace,
                    )
                    cooperative_trace_for_record = coop_trace

                    # Legacy ``veto`` trace key: the cooperative synthetic
                    # operating-envelope check flags a critical event and the
                    # primary MCP result did not flag the same event.
                    coop_compliance = (coop_mcp.get("check_compliance") or {})
                    primary_compliance = (mcp_results.get("check_compliance") or {})
                    coop_critical = bool(
                        not coop_compliance.get("compliant", True)
                        and any(
                            v.get("severity") == "critical"
                            for v in coop_compliance.get("violations", []) or []
                        )
                    )
                    primary_missed = not (
                        not primary_compliance.get("compliant", True)
                        and any(
                            v.get("severity") == "critical"
                            for v in primary_compliance.get("violations", []) or []
                        )
                    )

                    if coop_critical and primary_missed:
                        # Author-declared cooperative envelope adjustment: the
                        # cooperative modifier replaces the primary modifier
                        # and receives a bounded local-redistribution nudge.
                        veto_bias = np.array([-0.20, +0.20, 0.0])
                        (modifier, final_feature_contributions,
                         final_nonfeature_residual, attribution_scope,
                         final_modifier_theta_jacobian, composition_trace) = (
                            _compose_context_attribution(
                                modifier, primary_trace,
                                coop_modifier, coop_trace, veto_bias,
                            )
                        )
                        self._step_cooperative_veto = True
                    else:
                        (modifier, final_feature_contributions,
                         final_nonfeature_residual, attribution_scope,
                         final_modifier_theta_jacobian, composition_trace) = (
                            _compose_context_attribution(
                                modifier, primary_trace,
                                coop_modifier, coop_trace,
                            )
                        )
                except Exception as _exc:
                    if (_coop_mcp_cursor is not None
                            and not _coop_mcp_window_closed):
                        _coop_mcp_protocol = _protocol_window(
                            self._protocol_recorder, _coop_mcp_cursor,
                        )
                        _coop_mcp_window_closed = True
                    if (_coop_retrieval_cursor is not None
                            and not _coop_retrieval_window_closed):
                        _coop_retrieval_protocol = _protocol_window(
                            self._protocol_recorder, _coop_retrieval_cursor,
                        )
                        _coop_retrieval_window_closed = True
                    _operation_error = {
                        "stage": _coop_stage,
                        "error_type": type(_exc).__name__,
                        "message": str(_exc),
                    }
                    _coop_context_error = _operation_error
                    if _coop_stage == "mcp_dispatch":
                        _coop_mcp_error = _operation_error
                    elif _coop_stage == "retrieval":
                        _coop_retrieval_error = _operation_error
                    if os.environ.get("STRICT_VALIDATION", "0") == "1":
                        raise RuntimeError(
                            "publication-critical cooperative context retrieval "
                            f"failed at hour={hour}, role={active.role}"
                        ) from _exc
                    _log.warning(
                        "cooperative overlay blending skipped: %s", _exc,
                    )

            # Bind the exact call windows and cooperative inbox read to this
            # decision.  These replacements occur only after the live values
            # have already been computed and cannot feed back into the policy.
            if _coop_context_consumed_messages:
                _peer = {
                    key: value
                    for key, value in self._step_channel_evidence["peer"].items()
                    if key != "content_sha256"
                }
                _peer_consumed = list(_peer.get("consumed", []))
                _peer_consumed.extend(
                    _message_evidence_record(
                        message,
                        consumer_role="cooperative",
                        used_for_policy_bias=False,
                    )
                    for message in _coop_context_consumed_messages
                )
                _peer["consumed"] = _peer_consumed
                _peer["consumed_count"] = len(_peer_consumed)
                _peer_evidence = _seal_evidence_record(_peer)
            else:
                _peer_evidence = self._step_channel_evidence["peer"]

            _primary_mcp_evidence = _build_mcp_channel_evidence(
                returned_results=_primary_mcp_returned,
                effective_results=mcp_results,
                protocol_window=_primary_mcp_protocol,
                attempted=not _skip_mcp,
                skip_reason=(
                    "cyber_outage"
                    if _cyber_mcp_offline else (
                        "structural_mcp_ablation" if _skip_mcp else None
                    )
                ),
            )
            _primary_retrieval_evidence = (
                _build_retrieval_channel_evidence(
                    rag_context=rag_context,
                    integration_trace=primary_trace,
                    protocol_window=_primary_retrieval_protocol,
                    attempted=not _skip_rag,
                    requested_kind=retrieval_kind,
                    skip_reason=(
                        "structural_retrieval_ablation"
                        if _skip_rag else None
                    ),
                )
            )
            _coop_mcp_evidence = _build_mcp_channel_evidence(
                returned_results=_coop_mcp_returned,
                effective_results=_coop_mcp_effective,
                protocol_window=_coop_mcp_protocol,
                attempted=bool(_coop_overlay_active and not _skip_mcp),
                skip_reason=(
                    "cooperative_overlay_inactive"
                    if not _coop_overlay_active else (
                        "cyber_outage"
                        if _cyber_mcp_offline else (
                            "structural_mcp_ablation" if _skip_mcp else None
                        )
                    )
                ),
                operation_error=_coop_mcp_error,
            )
            _coop_retrieval_evidence = _build_retrieval_channel_evidence(
                rag_context=_coop_rag_for_evidence,
                integration_trace=cooperative_trace_for_record,
                protocol_window=_coop_retrieval_protocol,
                attempted=bool(_coop_overlay_active and not _skip_rag),
                requested_kind=retrieval_kind,
                skip_reason=(
                    "cooperative_overlay_inactive"
                    if not _coop_overlay_active else (
                        "structural_retrieval_ablation"
                        if _skip_rag else None
                    )
                ),
                operation_error=_coop_retrieval_error,
            )
            self._step_channel_evidence = _replace_channel_evidence(
                self._step_channel_evidence,
                peer=_peer_evidence,
                primary=_seal_evidence_record({
                    "role": active.role,
                    "mcp": _primary_mcp_evidence,
                    "retrieval": _primary_retrieval_evidence,
                }),
                cooperative=_seal_evidence_record({
                    "active": _coop_overlay_active,
                    "role": "cooperative",
                    "context_error": _coop_context_error,
                    "mcp": _coop_mcp_evidence,
                    "retrieval": _coop_retrieval_evidence,
                }),
            )

            # ---- Conditional observed-state feature-group masks ----
            # Recompute the modifier while retaining one psi feature group at a
            # time, reusing all already-dispatched results and guards. No tool or
            # retrieval channel is disabled here. The same cooperative blend and
            # veto are retained so the diagnostic describes the recorded policy
            # surface without a second dispatch or RNG draw.
            def _blend_channel(_cmode: str) -> np.ndarray:
                m = compute_context_modifier(
                    mcp_results, rag_context, obs, self._temporal_window,
                    theta_override=theta_override,
                    slca_amp_override=slca_amp_override,
                    temporal_params_override=temporal_params_override,
                    context_mode=_cmode,
                    retrieval_kind=retrieval_kind,
                )
                if _coop_mcp is not None:
                    cm = compute_context_modifier(
                        _coop_mcp, _coop_rag, _coop_obs, self._temporal_window,
                        theta_override=theta_override,
                        temporal_params_override=temporal_params_override,
                        context_mode=_cmode,
                        retrieval_kind=retrieval_kind,
                    )
                    if self._step_cooperative_veto:
                        veto = (np.array([-0.20, +0.20, 0.0])
                                if _cmode in ("mcp_only", "full") else np.zeros(3))
                        m = cm + veto
                    else:
                        m = 0.7 * m + 0.3 * cm
                # The locked policy always clips the composed context modifier.
                # The former optional unclipped path has been retired, so this
                # diagnostic replay must use the same unconditional boundary.
                return np.clip(m, -1.0, 1.0)

            try:
                self._step_modifier_mcp = _blend_channel("mcp_only")
                self._step_modifier_pirag = _blend_channel("pirag_only")
            except Exception as _exc:  # pragma: no cover - defensive
                if os.environ.get("STRICT_VALIDATION", "0") == "1":
                    raise RuntimeError(
                        "publication-critical context feature-group trace failed "
                        f"at hour={hour}, role={active.role}"
                    ) from _exc
                _log.warning(
                    "per-channel modifier attribution skipped: %s", _exc,
                )
                self._step_modifier_mcp = None
                self._step_modifier_pirag = None

            # Track which features are active (non-zero) for the learner
            psi = np.asarray(primary_trace["effective_psi"], dtype=float).copy()
            rules_fired = [i for i in range(len(psi)) if psi[i] > 0.01]

            # Store for post_step
            self._step_mcp_results = mcp_results
            self._step_rag_context = rag_context
            self._step_context_modifier = modifier
            self._step_keywords = rag_context.get("keywords", {})
            self._step_context_features = psi
            self._step_context_feature_contributions = final_feature_contributions
            self._step_context_nonfeature_residual = final_nonfeature_residual
            self._step_context_modifier_theta_jacobian = (
                final_modifier_theta_jacobian
            )
            self._step_context_integration_trace = {
                "primary": _context_trace_jsonable(primary_trace),
                "cooperative": _context_trace_jsonable(
                    cooperative_trace_for_record
                ),
                "composition": _context_trace_jsonable(composition_trace),
            }
            self._step_context_attribution_scope = attribution_scope
            self._step_rules_fired = rules_fired

            # Log
            primary_tools_invoked = list(
                mcp_results.get("_tools_invoked", [])
            )
            cooperative_tools_invoked = list(
                (_coop_mcp or {}).get("_tools_invoked", [])
            )
            primary_retrieval_attempted = bool(not _skip_rag)
            cooperative_retrieval_attempted = bool(
                not _skip_rag and _coop_rag is not None
            )
            self._context_log.append({
                "hour": obs.hour,
                "role": active.role,
                "primary_mcp_tools_invoked": primary_tools_invoked,
                "cooperative_mcp_tools_invoked": cooperative_tools_invoked,
                "mcp_tools_invoked": (
                    primary_tools_invoked + cooperative_tools_invoked
                ),
                "mcp_tools_skipped": mcp_results.get("_tools_skipped", []),
                "mcp_tools_failed": (
                    list(mcp_results.get("_tools_failed", []))
                    + list((_coop_mcp or {}).get("_tools_failed", []))
                ),
                "mcp_tool_failure_details": (
                    list(mcp_results.get("_tool_failure_details", []))
                    + list((_coop_mcp or {}).get("_tool_failure_details", []))
                ),
                "primary_retrieval_attempted": (
                    primary_retrieval_attempted
                ),
                "cooperative_retrieval_attempted": (
                    cooperative_retrieval_attempted
                ),
                "retrieval_attempted": bool(
                    primary_retrieval_attempted
                    or cooperative_retrieval_attempted
                ),
                "pirag_query_count": int(primary_retrieval_attempted)
                + int(cooperative_retrieval_attempted),
                "pirag_query": rag_context.get("query", ""),
                "pirag_citations": len(rag_context.get("citations", [])),
                "top_doc_id": rag_context.get("top_doc_id", ""),
                "top_citation_score": rag_context.get("top_citation_score", 0.0),
                "context_modifier": modifier.tolist() if modifier is not None else None,
                "modifier_norm": float(np.linalg.norm(modifier)),
                "guards_passed": (
                    rag_context.get("guards_passed", False)
                    if not _skip_rag else None
                ),
                "retrieval_gate": primary_trace.get("retrieval_gate", 0.0),
                "retrieval_blocked_reason": primary_trace.get(
                    "retrieval_blocked_reason"
                ),
                "temporal_scale": primary_trace.get("temporal_scale", 1.0),
                "physics_scale": primary_trace.get("physics_scale", 1.0),
                "rag_total_scale": primary_trace.get("rag_total_scale", 0.0),
                "physics_consistency_score": rag_context.get("physics_consistency_score", 1.0),
                "retrieval_metrics": rag_context.get("retrieval_metrics", {}),
                "retrieval_counterfactual": rag_context.get("counterfactual", {}),
                "rules_fired": self._step_rules_fired,
                "context_mode": context_mode,
                "retrieval_kind": retrieval_kind,
                "peer_messages_enabled": bool(
                    self._step_peer_messages_enabled
                ),
                "governance_override": False,  # Updated after select_action
            })

            return modifier

        except ImportError as exc:
            if os.environ.get("STRICT_VALIDATION", "0") == "1":
                raise RuntimeError(
                    "publication-critical context construction import failed"
                ) from exc
            _log.warning("context construction unavailable for this step: %s", exc)
            return None

    def post_step(
        self,
        agent: SupplyChainAgent,
        action: int,
        obs: Observation,
        outcome: Dict[str, Any],
        hour: float = 0.0,
        reward: float = 0.0,
    ) -> None:
        """Update agent state, enrich and route messages, update learner.

        Parameters
        ----------
        agent : the active agent that took the action.
        action : action index selected.
        obs : the observation built during ``step()``.
        outcome : dict with at least ``waste`` and ``rho`` keys.
        hour : current hour for cooperative overlay check.
        reward : reward received for this step (for context learner).
        """
        agent.update(action, outcome)

        # Tag override on the most recent log entry
        if self._step_override and self._context_log:
            self._context_log[-1]["governance_override"] = True

        messages = (
            agent.generate_messages(obs, action)
            if self._step_peer_messages_enabled else []
        )

        # Cooperative overlay: also update and generate messages during the
        # cooperative window.
        cooperative = self.agents.get("cooperative")
        if cooperative is not None and cooperative is not agent and _cooperative_window_active(hour):
            cooperative.update(action, outcome)
            if self._step_peer_messages_enabled:
                coop_obs = cooperative.observe(obs.raw, hour)
                messages.extend(
                    cooperative.generate_messages(coop_obs, action)
                )

        # Enrich messages with piR context if available
        if self.context_enabled and self._step_rag_context:
            try:
                from pirag.message_enrichment import enrich_message
                messages = [
                    enrich_message(msg, self._step_rag_context, self._step_mcp_results)
                    for msg in messages
                ]
            except ImportError:
                pass

        for msg in messages:
            self._message_log.append(msg)
            if msg.recipient == "broadcast":
                for other in self.agents.values():
                    if other.agent_id != msg.sender:
                        other.receive_message(msg)
            else:
                for other in self.agents.values():
                    if other.agent_id == msg.recipient:
                        other.receive_message(msg)
                        break

        # Complete the same decision snapshot with the exact enriched messages
        # that were actually appended/routed, plus any proactive recovery
        # capacity message emitted before the decision.  Replace and reseal;
        # never mutate the snapshot returned by ``step`` in place.
        if self._step_channel_evidence:
            _peer = {
                key: value
                for key, value in self._step_channel_evidence["peer"].items()
                if key != "content_sha256"
            }
            _emitted_records = [
                _message_evidence_record(message)
                for message in self._message_log[
                    self._step_message_log_start:
                ]
            ]
            _peer["emitted"] = _emitted_records
            _peer["emitted_count"] = len(_emitted_records)
            self._step_channel_evidence = _replace_channel_evidence(
                self._step_channel_evidence,
                peer=_seal_evidence_record(_peer),
            )

        # An action substituted by the probability-gap rule was not sampled
        # from the policy whose
        # probabilities were recorded. Treating its outcome as an on-policy
        # REINFORCE sample would be biased; using the returned one-hot vector
        # would instead count a zero-gradient pseudo-update. Skip all online
        # learner updates for that step and expose the count in provenance.
        if self._step_override and (
            self._context_learner is not None
            or self._theta_learner is not None
            or self._reward_shaping_learner is not None
        ):
            self._governance_skipped_learning_steps += 1

        # Context evaluation and learner update
        if (self.context_enabled
                and self._step_context_modifier is not None
                and self._context_evaluator is not None):
            # Reconstruct the same mode with context removed. This preserves
            # mode-specific terms (notably ``no_slca``), learned deltas,
            # policy temperature, and the random-number draw; the context
            # modifier is the only controlled difference. ``no_context`` belongs to the declared
            # mode set but never reaches this guarded branch because it never
            # creates a context modifier.
            assert self._step_mode in _CONTEXT_MODES, (
                f"context-ablation invariant violated: context_modifier is "
                f"set for non-context mode {self._step_mode!r}"
            )
            # Compute the context-ablated action and probabilities.
            # The replay rebuilds a generator from the snapshot taken in step()
            # so it draws the same categorical variate the live call saw. The
            # live stochastic call consumes that variate before applying any
            # probability-gap override; an override discards the sampled action but
            # does not skip the draw. The only controlled policy difference is
            # then context_modifier.
            try:
                from ..models.action_selection import select_action as _sa
                rng_cf = np.random.default_rng()
                if self._step_rng_state is not None:
                    rng_cf.bit_generator.state = copy.deepcopy(self._step_rng_state)
                theta_delta_cf = (
                    self._theta_learner.get_theta_delta()
                    if self._theta_learner is not None else None
                )
                if self._reward_shaping_learner is not None:
                    _slca_bonus_delta_cf = self._reward_shaping_learner.get_slca_bonus_delta()
                    _slca_rho_delta_cf = self._reward_shaping_learner.get_slca_rho_delta()
                    _no_slca_offset_delta_cf = self._reward_shaping_learner.get_no_slca_offset_delta()
                else:
                    _slca_bonus_delta_cf = None
                    _slca_rho_delta_cf = None
                    _no_slca_offset_delta_cf = None
                _counterfactual_out: Dict[str, object] = {}
                action_without, probs_without = _sa(
                    mode=self._step_mode, rho=obs.rho, inv=obs.inv,
                    y_hat=obs.y_hat, temp=obs.temp, tau=obs.tau,
                    policy=self._step_policy,
                    rng=rng_cf, scenario=self._step_scenario,
                    hour=hour,
                    role_bias=self._step_role_bias,
                    context_modifier=None,
                    supply_hat=self._step_supply_hat,
                    supply_std=self._step_supply_std,
                    demand_std=self._step_demand_std,
                    price_signal=self._step_price_signal,
                    theta_delta=theta_delta_cf,
                    slca_bonus_delta=_slca_bonus_delta_cf,
                    slca_rho_delta=_slca_rho_delta_cf,
                    no_slca_offset_delta=_no_slca_offset_delta_cf,
                    policy_temperature=(
                        float(self._step_policy_temperature)
                        if self._step_policy_temperature is not None else 1.0
                    ),
                    categorical_uniform=(
                        self._step_policy_categorical_uniform
                    ),
                    out=_counterfactual_out,
                )
                counterfactual_uniform = _counterfactual_out.get(
                    "policy_categorical_uniform"
                )
                if counterfactual_uniform != self._step_policy_categorical_uniform:
                    raise RuntimeError(
                        "paired context ablation did not reuse the live "
                        "categorical uniform"
                    )
                self._step_counterfactual_action = action_without
                self._step_counterfactual_probs = probs_without
                self._step_counterfactual_categorical_uniform = (
                    counterfactual_uniform
                )
                self._step_counterfactual_sampled_action_pre_override = int(
                    _counterfactual_out["sampled_action_pre_override"]
                )
            except Exception as exc:
                # Missing paired counterfactuals would bias the publication
                # context-influence numerator downward. Canonical runs set
                # STRICT_VALIDATION=1 and must fail rather than silently turn
                # an unavailable intervention into "no change". Exploratory
                # local runs retain a marked-unavailable record and warning.
                self._step_counterfactual_action = action
                self._step_counterfactual_probs = None
                self._step_counterfactual_categorical_uniform = None
                self._step_counterfactual_sampled_action_pre_override = None
                action_without = action
                message = (
                    "paired pre-selection-state context ablation failed at "
                    f"hour={hour}, role={agent.role}, mode={self._step_mode}: "
                    f"{exc}"
                )
                if os.environ.get("STRICT_VALIDATION", "0") == "1":
                    raise RuntimeError(message) from exc
                _log.warning("%s; marking counterfactual unavailable", message)

            self._context_evaluator.record(
                hour, agent.role, action_without, action,
                reward, self._step_context_modifier,
            )

            # Update ContextMatrixLearner via REINFORCE
            if (self._context_learner is not None
                    and hasattr(self._context_learner, 'get_theta')
                    and self._step_context_features is not None
                    and self._step_context_modifier_theta_jacobian is not None
                    and self._step_probs is not None
                    and not self._freeze_all_learners
                    and not self._step_override):
                self._context_learner.update(
                    psi=self._step_context_features,
                    action=action,
                    probs=self._step_probs,
                    reward=reward,
                    slca_score=(
                        0.0 if self._step_mode == "no_slca"
                        else outcome.get("slca", 0.0)
                    ),
                    modifier_theta_jacobian=(
                        self._step_context_modifier_theta_jacobian
                    ),
                    policy_temperature=(
                        float(self._step_policy_temperature)
                        if self._step_policy_temperature is not None else 1.0
                    ),
                    context_modifier=self._step_context_modifier,
                    slca_shaping=self._step_slca_shaping,
                    slca_amp=(
                        float(self._step_slca_amp)
                        if self._step_slca_amp is not None else 0.0
                    ),
                )

        # Update the active decision owner's PolicyDeltaLearner. Construction
        # is capability-gated, so frozen/static modes never enter this block.
        if (self._theta_learner is not None
                and self._step_phi is not None
                and self._step_probs is not None
                and not self._freeze_all_learners
                and not self._step_override):
            self._theta_learner.update(
                phi=self._step_phi,
                action=action,
                probs=self._step_probs,
                reward=reward,
            )

        # Update RewardShapingLearner only for modes whose logits contain the
        # two active social-proxy vectors and declare them learnable.
        if (self._reward_shaping_learner is not None
                and self._step_probs is not None
                and self._step_mode
                and not self._freeze_all_learners
                and not self._step_override):
            self._reward_shaping_learner.update(
                action=action,
                probs=self._step_probs,
                reward=reward,
                mode=self._step_mode,
                rho=float(getattr(obs, "rho", 0.0)),
            )

        # Generate structured explanation and capture trace
        if self.context_enabled and self._step_mcp_results:
            try:
                from pirag.explain_decision import explain_decision
                action_names = ["cold_chain", "local_redistribute", "recovery"]
                cf_action_name = action_names[self._step_counterfactual_action] if self._step_counterfactual_probs is not None else None
                self._last_explanation = explain_decision(
                    action=action_names[action],
                    role=agent.role,
                    hour=hour,
                    obs=obs,
                    mcp_results=self._step_mcp_results,
                    rag_context=self._step_rag_context,
                    slca_score=outcome.get("slca", 0.0),
                    carbon_kg=outcome.get("carbon_kg", 0.0),
                    waste=outcome.get("waste", 0.0),
                    context_features=self._step_context_features,
                    logit_adjustment=self._step_context_modifier,
                    action_probs=(
                        self._step_policy_probs_pre_override
                        if self._step_policy_probs_pre_override is not None
                        else self._step_probs
                    ),
                    effective_context_theta=self._step_effective_context_theta,
                    chosen_action_context_contributions=(
                        self._step_chosen_action_context_contributions
                    ),
                    chosen_action_context_residual=(
                        self._step_chosen_action_context_residual
                    ),
                    context_attribution_scope=self._step_context_attribution_scope,
                    context_integration_trace=(
                        self._step_context_integration_trace
                    ),
                    counterfactual_action=cf_action_name,
                    counterfactual_probs=self._step_counterfactual_probs,
                    governance_override=self._step_override,
                    keywords=self._step_keywords,
                )
            except Exception as exc:
                self._last_explanation = None
                message = (
                    "decision explanation failed at "
                    f"hour={hour}, role={agent.role}, mode={self._step_mode}: {exc}"
                )
                if os.environ.get("STRICT_VALIDATION", "0") == "1":
                    raise RuntimeError(message) from exc
                _log.warning("%s; explanation marked unavailable", message)

            # Update context cache for MCP resource reads
            try:
                from pirag.mcp.tools.context_features import update_context_cache
                update_context_cache(
                    features=self._step_context_features,
                    modifier=self._step_context_modifier,
                    explanation=self._last_explanation,
                    hour=hour,
                    override=self._step_override,
                    robustness={
                        "dispatch_profile": self._step_dispatch_cfg.get("qos_profile", "legacy"),
                        "reliability_enabled": bool(self._step_dispatch_cfg.get("enable_reliability", False)),
                        "fault_injected": bool(self._step_mcp_results.get("_fault_injected")),
                    },
                )
            except ImportError:
                pass

            if self._trace_exporter is not None:
                # Determine if context changed the action
                action_without = action
                if (self._context_evaluator is not None
                        and self._context_evaluator._records):
                    last_eval = self._context_evaluator._records[-1]
                    action_without = last_eval.get("action_without", action)

                self._trace_exporter.capture(
                    obs=obs,
                    scenario=self._step_scenario,
                    action=action_names[action],
                    probs=self._step_probs,
                    mcp_results=self._step_mcp_results,
                    rag_context=self._step_rag_context,
                    context_features=self._step_context_features,
                    logit_adjustment=self._step_context_modifier,
                    explanation=self._last_explanation,
                    role=agent.role,
                    action_changed=(action != action_without),
                    governance_override=self._step_override,
                    context_integration=(
                        self._step_context_integration_trace
                    ),
                )

        # Decision history retained for the optional retrieval-ingestion
        # diagnostic. It is not learner state and is inactive by default.
        if self.context_enabled:
            self._decision_history.append({
                "hour": hour,
                "action": action,
                "role": agent.role,
                "slca": outcome.get("slca", 0.0),
                "carbon_kg": outcome.get("carbon_kg", 0.0),
                "waste": outcome.get("waste", 0.0),
            })

            # Optional decision-history ingestion (every 24 steps). Summaries
            # of the system's own past actions can create a self-amplifying
            # retrieval loop, so this diagnostic is disabled by default and
            # excluded from the locked publication protocol. Enable it only
            # for a declared ablation via the policy field or environment.
            import os as _os_dyn
            # The environment fallback applies only when no policy object is
            # present. A policy-level setting takes precedence.
            dyn_feedback_enabled = (
                _os_dyn.environ.get("DYNAMIC_KB_FEEDBACK", "false").lower() == "true"
            )
            if self._step_policy is not None:
                # Policy-level setting takes precedence when set.
                dyn_feedback_enabled = bool(
                    getattr(self._step_policy, "enable_dynamic_knowledge_feedback", False)
                )
            if dyn_feedback_enabled and len(self._decision_history) % 24 == 0 and self._pirag_pipeline is not None:
                try:
                    from pirag.dynamic_knowledge import ingest_decision_history
                    ingest_decision_history(
                        self._pirag_pipeline,
                        self._decision_history,
                        "simulation",
                    )
                except ImportError:
                    pass

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def message_log(self) -> List[InterAgentMessage]:
        """Full log of all inter-agent messages this episode."""
        return list(self._message_log)

    @property
    def context_log(self) -> List[Dict[str, Any]]:
        """Full log of context injection events."""
        return list(self._context_log)

    def agent_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Per-agent summary statistics."""
        return {role: dict(agent.state) for role, agent in self.agents.items()}

    def context_summary(self) -> Dict[str, Any]:
        """Summary of MCP and piR activity for paper reporting."""
        caps = (
            capabilities_for(self._learning_mode)
            if self._learning_mode is not None else None
        )
        if not self._context_log:
            return {
                "total_context_steps": 0,
                "total_mcp_tool_calls": 0,
                "total_pirag_queries": 0,
                "dispatcher_tool_failures": 0,
                "retrieval_kind": (
                    caps.retrieval_kind if caps is not None else None
                ),
                "peer_messages_enabled": (
                    caps.peer_messages if caps is not None else None
                ),
            }

        per_role: Dict[str, Dict[str, Any]] = {}
        total_tools = 0
        dispatcher_failures = 0
        guard_failures = 0

        for entry in self._context_log:
            role = entry["role"]
            if role not in per_role:
                per_role[role] = {
                    "mcp_calls": 0, "pirag_queries": 0,
                    "dispatcher_failures": 0,
                    "modifier_magnitudes": [], "guard_failures": 0,
                    "rules_fired_total": 0,
                }
            n_tools = len(entry.get("mcp_tools_invoked", []))
            n_dispatcher_failures = len(entry.get("mcp_tools_failed", []))
            per_role[role]["mcp_calls"] += n_tools
            per_role[role]["dispatcher_failures"] += n_dispatcher_failures
            retrieval_attempted = bool(entry.get(
                "retrieval_attempted", bool(entry.get("pirag_query")),
            ))
            retrieval_count = int(entry.get(
                "pirag_query_count",
                int(bool(entry.get("primary_retrieval_attempted", False)))
                + int(bool(entry.get(
                    "cooperative_retrieval_attempted", False,
                ))),
            ))
            if retrieval_count < 0:
                raise RuntimeError("negative piR query count in context log")
            per_role[role]["pirag_queries"] += retrieval_count
            per_role[role]["modifier_magnitudes"].append(entry.get("modifier_norm", 0.0))
            if retrieval_attempted and not entry.get("guards_passed", False):
                per_role[role]["guard_failures"] += 1
                guard_failures += 1
            per_role[role]["rules_fired_total"] += len(entry.get("rules_fired", []))
            total_tools += n_tools
            dispatcher_failures += n_dispatcher_failures

        # Compute per-role means
        for role in per_role:
            mags = per_role[role].pop("modifier_magnitudes")
            per_role[role]["mean_modifier_magnitude"] = float(np.mean(mags)) if mags else 0.0
            per_role[role]["nonzero_modifier_count"] = sum(1 for m in mags if m > 1e-9)

        modifiers = [e["modifier_norm"] for e in self._context_log]
        physics_scores = [float(e.get("physics_consistency_score", 1.0)) for e in self._context_log]
        citation_overlap_vals = [
            float((e.get("retrieval_metrics", {}) or {}).get(
                "citation_token_overlap_at_3",
                (e.get("retrieval_metrics", {}) or {}).get("faithfulness_at_3", 0.0),
            ))
            for e in self._context_log
            if e.get("retrieval_metrics")
        ]

        return {
            "total_context_steps": len(self._context_log),
            "total_mcp_tool_calls": total_tools,
            "total_pirag_queries": sum(
                int(entry.get(
                    "pirag_query_count",
                    int(bool(entry.get(
                        "primary_retrieval_attempted", False,
                    ))) + int(bool(entry.get(
                        "cooperative_retrieval_attempted", False,
                    ))),
                ))
                for entry in self._context_log
            ),
            "dispatcher_tool_failures": dispatcher_failures,
            "guard_failures": guard_failures,
            "mean_modifier_magnitude": float(np.mean(modifiers)) if modifiers else 0.0,
            "nonzero_modifier_steps": sum(1 for m in modifiers if m > 1e-9),
            "governance_overrides": sum(1 for e in self._context_log if e.get("governance_override")),
            "mean_physics_consistency": float(np.mean(physics_scores)) if physics_scores else 1.0,
            "mean_retrieval_citation_token_overlap_at_3": (
                float(np.mean(citation_overlap_vals)) if citation_overlap_vals else 0.0
            ),
            # Legacy schema alias; this is not a faithfulness evaluation.
            "mean_retrieval_faithfulness_at_3": (
                float(np.mean(citation_overlap_vals)) if citation_overlap_vals else 0.0
            ),
            "retrieval_kind": (
                caps.retrieval_kind if caps is not None else None
            ),
            "peer_messages_enabled": (
                caps.peer_messages if caps is not None else None
            ),
            "per_role": per_role,
        }

    @property
    def trace_exporter(self):
        """Trace exporter for paper evidence (None if not initialized)."""
        return self._trace_exporter

    @property
    def protocol_recorder(self):
        """MCP protocol recorder (None if not initialized)."""
        return self._protocol_recorder

    def learner_summary(self) -> Dict[str, Any]:
        """Context learner statistics."""
        if self._context_learner is not None:
            summary = self._context_learner.summary()
            state = self._context_learner.save_state()
            declared = bool(
                self._learning_mode is not None
                and capabilities_for(self._learning_mode).context_matrix_learning
            )
            summary.update({
                "mode": self._learning_mode,
                "learner_state_schema_version": 2,
                "state_sha256": _learner_state_sha256(state),
                "learning_declared": declared,
                "learning_enabled": bool(
                    declared and not self._freeze_all_learners
                ),
                "learner_frozen": bool(self._context_learner.freeze),
                "learner_phase": self.learner_freeze_summary()["learner_phase"],
                "freeze_reason": self._learner_freeze_reason,
                "governance_skipped_learning_steps": int(
                    self._governance_skipped_learning_steps
                ),
            })
            return summary
        return {}

    def theta_learner_summary(self) -> Dict[str, Any]:
        """Policy-delta statistics across the four decision-owning roles.

        Legacy scalar/matrix keys describe the currently active role. Aggregate
        counters and the role-keyed block prevent the former final-role-only
        summary from hiding three quarters of a canonical episode's updates.
        """
        if not self._theta_learners:
            if self._theta_learner is not None:
                summary = self._theta_learner.summary()
                summary.update({
                    "learner_frozen": bool(self._theta_learner.freeze),
                    "learner_phase": self.learner_freeze_summary()[
                        "learner_phase"
                    ],
                    "freeze_reason": self._learner_freeze_reason,
                })
                return summary
            return {}

        per_role = {}
        for role, learner in self._theta_learners.items():
            role_summary = learner.summary()
            role_summary["learner_frozen"] = bool(learner.freeze)
            per_role[role] = role_summary
        role_states = {
            role: learner.save_state()
            for role, learner in self._theta_learners.items()
        }
        active_role = next(
            (
                role for role, learner in self._theta_learners.items()
                if learner is self._theta_learner
            ),
            "farm",
        )
        summary = dict(per_role[active_role])
        norms = [float(item["delta_frobenius_norm"]) for item in per_role.values()]
        updates = [int(item["n_updates"]) for item in per_role.values()]
        policy_reversals = [
            {
                "role": role,
                **reversal,
            }
            for role, item in per_role.items()
            for reversal in item.get("sign_reversal_coordinates", [])
        ]
        summary.update({
            "active_role_legacy_fields": active_role,
            "mode": self._learning_mode,
            "learner_state_schema_version": 2,
            "learner_frozen": bool(self._freeze_all_learners),
            "learner_phase": self.learner_freeze_summary()["learner_phase"],
            "freeze_reason": self._learner_freeze_reason,
            "governance_skipped_learning_steps": int(
                self._governance_skipped_learning_steps
            ),
            "decision_owner_roles": list(DECISION_OWNER_ROLES),
            "per_role": per_role,
            "per_role_state_sha256": {
                role: _learner_state_sha256(state)
                for role, state in role_states.items()
            },
            "combined_state_sha256": _learner_state_sha256(role_states),
            "n_updates": int(sum(updates)),
            "updates_per_role": {
                role: int(item["n_updates"]) for role, item in per_role.items()
            },
            "delta_frobenius_norm": float(np.linalg.norm(norms)),
            "max_delta_entry": float(max(
                item["max_delta_entry"] for item in per_role.values()
            )),
            "max_fractional_drift": float(max(
                item["max_fractional_drift"] for item in per_role.values()
            )),
            "sign_reversal_count": len(policy_reversals),
            "sign_reversal_coordinates": policy_reversals,
            "worst_sign_reversal": (
                max(
                    policy_reversals,
                    key=lambda item: abs(item["final_weight"]),
                )
                if policy_reversals else None
            ),
        })
        return summary

    def reward_shaping_learner_summary(self) -> Dict[str, Any]:
        """Reward-shaping learner statistics."""
        if self._reward_shaping_learner is not None:
            summary = self._reward_shaping_learner.summary()
            state = self._reward_shaping_learner.save_state()
            summary.update({
                "mode": self._learning_mode,
                "learner_state_schema_version": 2,
                "state_sha256": _learner_state_sha256(state),
                "learner_frozen": bool(self._reward_shaping_learner.freeze),
                "learner_phase": self.learner_freeze_summary()["learner_phase"],
                "freeze_reason": self._learner_freeze_reason,
                "governance_skipped_learning_steps": int(
                    self._governance_skipped_learning_steps
                ),
            })
            return summary
        return {}

    def save_learner_states(self) -> Dict[str, Any]:
        """Serialise all learner states into one JSON-friendly dict.

        Use this at the end of a long HPC episode (or crash-resume point)
        to persist learned weights across runs. The returned dict can be
        written with ``json.dump`` and later restored via
        :meth:`load_learner_states`.
        """
        state: Dict[str, Any] = {
            "schema_version": 2,
            "mode": self._learning_mode,
            "decision_owner_roles": list(DECISION_OWNER_ROLES),
            "learners_frozen": bool(self._freeze_all_learners),
            "learner_phase": self.learner_freeze_summary()["learner_phase"],
            "freeze_reason": self._learner_freeze_reason,
            "governance_skipped_learning_steps": int(
                self._governance_skipped_learning_steps
            ),
        }
        if self._context_learner is not None:
            state["context_learner"] = self._context_learner.save_state()
        # Per-role theta learners. The legacy singleton key remains for old
        # readers, but new loaders never apply it on top of a per-role payload.
        if self._theta_learners:
            state["theta_learners"] = {
                role: lrn.save_state()
                for role, lrn in self._theta_learners.items()
            }
        if self._theta_learner is not None:
            state["theta_learner"] = self._theta_learner.save_state()
        if self._reward_shaping_learner is not None:
            state["reward_shaping_learner"] = self._reward_shaping_learner.save_state()
        return state

    def load_learner_states(self, state: Dict[str, Any]) -> None:
        """Restore learner state produced by :meth:`save_learner_states`.

        Missing keys are tolerated so partial checkpoints (e.g. only the
        theta learner) still work. Attempting to load into a coordinator
        whose learner was never constructed (import-time failure) is a
        no-op for that slot.
        """
        saved_mode = state.get("mode")
        if saved_mode is not None:
            saved_mode = str(saved_mode)
            if self._learning_mode is None:
                self._init_policy_learning(saved_mode)
            elif saved_mode != self._learning_mode:
                raise ValueError(
                    "learner checkpoint mode mismatch: "
                    f"coordinator={self._learning_mode!r}, checkpoint={saved_mode!r}"
                )
        elif self._learning_mode is None:
            # A legacy snapshot does not identify which capability set should be
            # constructed. Defer it until the first step supplies the mode.
            self._pending_learner_state = copy.deepcopy(state)
            return

        self._load_learner_states_now(state)

    def _load_learner_states_now(self, state: Dict[str, Any]) -> None:
        """Apply a checkpoint after mode-specific learners exist."""

        self._governance_skipped_learning_steps = int(
            state.get("governance_skipped_learning_steps", 0)
        )
        ctx = state.get("context_learner")
        if ctx is not None and self._context_learner is not None:
            self._context_learner.load_state(ctx)
        per_role = state.get("theta_learners") or {}
        has_per_role = isinstance(per_role, dict) and bool(per_role)
        if has_per_role:
            for role, role_state in per_role.items():
                if role in self._theta_learners and role_state is not None:
                    self._theta_learners[role].load_state(role_state)
        theta = state.get("theta_learner")
        # Legacy singleton fallback only. Applying it after a per-role payload
        # overwrote Farm with whichever role happened to be active at save time.
        if not has_per_role and theta is not None:
            if self._theta_learners:
                # A schema-v1 singleton represented the shared base-policy
                # correction used by every lifecycle stage. Replicate it to all
                # four decision owners; assigning it only to whichever role was
                # active at load time would create an order-dependent migration.
                for learner in self._theta_learners.values():
                    learner.load_state(theta)
            elif self._theta_learner is not None:
                self._theta_learner.load_state(theta)
        rsl = state.get("reward_shaping_learner")
        if rsl is not None and self._reward_shaping_learner is not None:
            self._reward_shaping_learner.load_state(rsl)
        if bool(state.get("learners_frozen", False)):
            self.freeze_learners(
                reason=str(state.get("freeze_reason") or "loaded_checkpoint"),
            )

    def evaluator_summary(self) -> Dict[str, Any]:
        """Context quality evaluator statistics."""
        if self._context_evaluator is not None:
            return self._context_evaluator.summary()
        return {}
