#!/usr/bin/env python3
"""Validate the exact publication ledger inventory, records, and Merkle roots."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "agribrain" / "backend"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from pirag.mcp.tools.compliance import check_compliance  # noqa: E402
from src.agents.message import (  # noqa: E402
    InterAgentMessage,
    MessageType,
    message_bias_from_inbox,
)
from src.models.action_selection import (  # noqa: E402
    DECLARED_THETA,
    GOVERNANCE_CC_PROB_CEILING,
    GOVERNANCE_LOCAL_ADVANTAGE_MIN,
    categorical_action_from_uniform,
)
from src.models.episode_evidence_contract import (  # noqa: E402
    ACTIVITY_STEP_FIELDS,
    build_episode_evidence_contract,
    reconstruct_episode_evidence,
    validate_episode_evidence_contract,
)
from src.models.forecast import yield_demand_forecast  # noqa: E402
from src.models.mode_capabilities import (  # noqa: E402
    DECISION_OWNER_ROLES,
    capabilities_for,
)
from src.models.outcome_equation_contract import (  # noqa: E402
    build_outcome_equation_contract,
    validate_outcome_equation_contract,
    validate_recorded_spoilage_trajectories,
    validate_recorded_step_outcomes,
)
from src.models.persistence_forecast import persistence_forecast  # noqa: E402
from src.models.pinn_residual import (  # noqa: E402
    build_residual_feature_row,
    load_frozen_checkpoint,
    predict_residual,
)
from src.models.policy import Policy  # noqa: E402
from src.models.policy_equation_contract import (  # noqa: E402
    validate_policy_record,
)
from src.models.resilience import compute_equity, compute_rle  # noqa: E402
from src.models.spoilage import (  # noqa: E402
    advance_spoilage_risk_midpoint,
    arrhenius_k,
)
from src.models.synthetic_spoilage_dgp import (  # noqa: E402
    DEFAULT_PACKAGING_INDEX,
    HANDLING_SHOCK_LOG_RATE_COEFFICIENT,
    PACKAGING_CENTER,
    PACKAGING_LOG_RATE_COEFFICIENT,
    RH_TRANSIENT_LOG_RATE_COEFFICIENT,
    compute_spoilage_independent_synthetic_dgp,
    synthetic_dgp_provenance,
)

from mvp.simulation.benchmarks.trace_contract import (  # noqa: E402
    TRACE_FIELDS,
    TRACE_MODES,
)
from mvp.simulation.generate_results import (  # noqa: E402
    MODES,
    SCENARIOS,
    TRACE_SCHEMA_VERSION,
    _canonical_sha256,
    _policy_categorical_uniform,
    _stream_id,
    _stream_seed,
    apply_scenario,
    policy_theta_for_seed,
)
from mvp.simulation.stochastic import (  # noqa: E402
    StochasticLayer,
)
from mvp.simulation.stochastic import (  # noqa: E402
    canonical_defaults as stochastic_defaults,
)

EXPECTED_SEEDS = (
    42, 1337, 2024, 7, 99, 101, 202, 303, 404, 505,
    606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515,
)
EXPECTED_RECORDS = 288
ACTIONS = ("cold_chain", "local_redistribute", "recovery")
PUBLICATION_DATA_CSV = (
    REPO_ROOT / "agribrain" / "backend" / "src" / "data_spinach.csv"
)


def expected_publication_episode_evidence_contract() -> dict[str, Any]:
    """Return the locked publication timer/activity aggregation contract."""

    return build_episode_evidence_contract()


def _expected_publication_stochastic_layer(
    *,
    benchmark_seed: int,
    scenario: str,
    episode_index: int = 3,
    stochastic_scale: float = 1.0,
) -> StochasticLayer:
    """Recreate the locked environment stream without consuming live state."""

    defaults = stochastic_defaults()
    scale = float(stochastic_scale)
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError("stochastic_scale must be finite and non-negative")

    def scaled(name: str) -> float:
        value = float(defaults[name])
        return value if value == 0.0 else value * scale

    stream_seed = _stream_seed(
        benchmark_seed, scenario, episode_index, "environment",
    )
    return StochasticLayer(
        rng=np.random.default_rng(stream_seed),
        enabled=True,
        temp_std_c=scaled("STOCH_TEMP_STD_C"),
        rh_std=scaled("STOCH_RH_STD"),
        demand_frac_std=scaled("STOCH_DEMAND_FRAC_STD"),
        inventory_frac_std=scaled("STOCH_INVENTORY_FRAC_STD"),
        transport_km_frac_std=scaled("STOCH_TRANSPORT_KM_STD"),
        k_ref_frac_std=scaled("STOCH_K_REF_STD"),
        ea_r_frac_std=scaled("STOCH_EA_R_STD"),
        onset_jitter_hours=scaled("STOCH_ONSET_JITTER_H"),
        theta_noise_std=scaled("STOCH_THETA_NOISE_STD"),
        policy_temp_std=scaled("STOCH_POLICY_TEMP_STD"),
        delay_prob=scaled("STOCH_DELAY_PROB"),
        stream_seed=stream_seed,
    )


def expected_publication_outcome_equation_contract(
    *,
    benchmark_seed: int,
    scenario: str,
    episode_index: int = 3,
    policy: Policy | None = None,
    stochastic_scale: float = 1.0,
    parameter_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the independently derived contract for a locked publication arm."""

    layer = _expected_publication_stochastic_layer(
        benchmark_seed=benchmark_seed,
        scenario=scenario,
        episode_index=episode_index,
        stochastic_scale=stochastic_scale,
    )
    episode_policy = Policy() if policy is None else policy
    effective_k_ref = layer.perturb_k_ref(episode_policy.k_ref, counter=0)
    effective_ea_r = layer.perturb_ea_r(episode_policy.Ea_R, counter=0)
    return build_outcome_equation_contract(
        episode_policy,
        effective_k_ref=effective_k_ref,
        effective_ea_r=effective_ea_r,
        stochastic_layer=layer,
        parameter_overrides=parameter_overrides,
    )


def _canonical_bytes(record: dict[str, Any]) -> bytes:
    return json.dumps(
        record, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")


def _merkle_root(leaves: list[str]) -> str:
    if not leaves:
        return "0" * 64
    layer = [bytes.fromhex(leaf) for leaf in leaves]
    while len(layer) > 1:
        if len(layer) % 2:
            layer.append(layer[-1])
        layer = [
            hashlib.sha256(layer[i] + layer[i + 1]).digest()
            for i in range(0, len(layer), 2)
        ]
    return layer[0].hex()


def _finite_vector(value: Any, length: int) -> bool:
    if not isinstance(value, list) or len(value) != length:
        return False
    try:
        return all(math.isfinite(float(item)) for item in value)
    except (TypeError, ValueError):
        return False


def _finite_matrix(value: Any, rows: int, columns: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == rows
        and all(_finite_vector(row, columns) for row in value)
    )


def _evidence_sha256(value: Any) -> str:
    """Apply the coordinator's strict canonical-JSON evidence hash rule."""

    try:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuntimeError("channel evidence is not strict JSON-native") from exc
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RuntimeError(f"{where} is not a lowercase SHA-256 digest")
    return value


def _validate_sealed_evidence(value: Any, *, where: str) -> None:
    """Recursively verify every content-addressed evidence object."""

    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_sealed_evidence(item, where=f"{where}[{index}]")
        return
    if not isinstance(value, dict):
        return
    if "content_sha256" in value:
        observed = _require_sha256(
            value["content_sha256"], where=f"{where}.content_sha256",
        )
        unhashed = {
            key: item for key, item in value.items()
            if key != "content_sha256"
        }
        if observed != _evidence_sha256(unhashed):
            raise RuntimeError(f"{where} content SHA-256 mismatch")
    for key, item in value.items():
        if key != "content_sha256":
            _validate_sealed_evidence(item, where=f"{where}.{key}")


def _require_sealed_object(value: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{where} is not an object")
    if "content_sha256" not in value:
        raise RuntimeError(f"{where} lacks content_sha256")
    return value


def _validate_protocol_evidence(value: Any, *, where: str) -> None:
    protocol = _require_sealed_object(value, where=where)
    records = protocol.get("records")
    if not isinstance(records, list):
        raise RuntimeError(f"{where}.records is not a list")
    if protocol.get("records_captured") != len(records):
        raise RuntimeError(f"{where} records_captured does not match records")
    for index, record_value in enumerate(records):
        record_where = f"{where}.records[{index}]"
        record = _require_sealed_object(record_value, where=record_where)
        request = record.get("request")
        if not isinstance(request, dict):
            raise RuntimeError(f"{record_where}.request is not an object")
        if _require_sha256(
            record.get("request_sha256"), where=f"{record_where}.request_sha256",
        ) != _evidence_sha256(request):
            raise RuntimeError(f"{record_where} request SHA-256 mismatch")
        params = request.get("params") or {}
        expected_tool = (
            params.get("name")
            if request.get("method") == "tools/call" and isinstance(params, dict)
            else None
        )
        expected_arguments = (
            params.get("arguments")
            if isinstance(params, dict) and "arguments" in params else None
        )
        if record.get("method") != request.get("method") or record.get(
            "request_id"
        ) != request.get("id") or record.get("tool_name") != expected_tool or (
            record.get("arguments") != expected_arguments
        ):
            raise RuntimeError(f"{record_where} request projections are inconsistent")
        returned_hash = record.get("returned_result_sha256")
        if returned_hash is not None:
            _require_sha256(
                returned_hash, where=f"{record_where}.returned_result_sha256",
            )
            if record.get("returned_result_included") is not False:
                raise RuntimeError(
                    f"{record_where} ambiguously claims an included result"
                )


def _tool_results_from_records(value: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{where} is not a list")
    reconstructed: dict[str, Any] = {}
    for index, record_value in enumerate(value):
        record_where = f"{where}[{index}]"
        record = _require_sealed_object(record_value, where=record_where)
        name = record.get("tool_name")
        if not isinstance(name, str) or not name or name in reconstructed:
            raise RuntimeError(f"{record_where} has an invalid tool_name")
        result = record.get("result")
        if _require_sha256(
            record.get("result_sha256"), where=f"{record_where}.result_sha256",
        ) != _evidence_sha256(result):
            raise RuntimeError(f"{record_where} result SHA-256 mismatch")
        reconstructed[name] = result
    return reconstructed


def _validate_mcp_evidence(value: Any, *, where: str) -> None:
    mcp = _require_sealed_object(value, where=where)
    _validate_protocol_evidence(mcp.get("protocol"), where=f"{where}.protocol")
    effective_tools = _tool_results_from_records(
        mcp.get("effective_tool_results"),
        where=f"{where}.effective_tool_results",
    )
    returned_tools = _tool_results_from_records(
        mcp.get("returned_tool_results"),
        where=f"{where}.returned_tool_results",
    )
    metadata = mcp.get("dispatcher_metadata")
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) or not key.startswith("_") for key in metadata
    ):
        raise RuntimeError(f"{where}.dispatcher_metadata is invalid")
    effective = {**effective_tools, **metadata}
    expected_effective_hash = _evidence_sha256(effective)
    if _require_sha256(
        mcp.get("effective_results_sha256"),
        where=f"{where}.effective_results_sha256",
    ) != expected_effective_hash:
        raise RuntimeError(f"{where} effective-results SHA-256 mismatch")
    returned_hash = _require_sha256(
        mcp.get("returned_results_sha256"),
        where=f"{where}.returned_results_sha256",
    )
    matches = mcp.get("returned_matches_effective")
    if not isinstance(matches, bool):
        raise RuntimeError(f"{where}.returned_matches_effective is not boolean")
    if matches:
        if returned_tools or returned_hash != expected_effective_hash:
            raise RuntimeError(f"{where} equal returned/effective results disagree")
    elif not returned_tools or returned_hash == expected_effective_hash:
        raise RuntimeError(f"{where} changed returned results are not preserved")
    tools_invoked = mcp.get("tools_invoked")
    if not isinstance(tools_invoked, list) or tools_invoked != metadata.get(
        "_tools_invoked", []
    ):
        raise RuntimeError(f"{where} invoked-tool metadata is inconsistent")


def _validate_retrieval_evidence(value: Any, *, where: str) -> None:
    retrieval = _require_sealed_object(value, where=where)
    _validate_protocol_evidence(
        retrieval.get("protocol"), where=f"{where}.protocol",
    )
    query = retrieval.get("query")
    expected_query_hash = (
        hashlib.sha256(query.encode("utf-8")).hexdigest()
        if isinstance(query, str) else None
    )
    if retrieval.get("query_sha256") != expected_query_hash:
        raise RuntimeError(f"{where} query SHA-256 mismatch")
    citations = retrieval.get("ordered_citations")
    if not isinstance(citations, list):
        raise RuntimeError(f"{where}.ordered_citations is not a list")
    for index, citation_value in enumerate(citations):
        citation_where = f"{where}.ordered_citations[{index}]"
        citation = _require_sealed_object(citation_value, where=citation_where)
        if citation.get("rank") != index + 1:
            raise RuntimeError(f"{citation_where} rank is inconsistent")
        for field in (
            "source_passage_sha256", "captured_passage_content_sha256",
        ):
            digest = citation.get(field)
            if digest is not None:
                _require_sha256(digest, where=f"{citation_where}.{field}")
    guidance = retrieval.get("guidance_hashes")
    if not isinstance(guidance, dict):
        raise RuntimeError(f"{where}.guidance_hashes is not an object")
    for name, descriptor in guidance.items():
        descriptor_where = f"{where}.guidance_hashes.{name}"
        if not isinstance(descriptor, dict):
            raise RuntimeError(f"{descriptor_where} is not an object")
        digest = descriptor.get("sha256")
        if digest is not None:
            _require_sha256(digest, where=f"{descriptor_where}.sha256")
    _require_sha256(
        retrieval.get("raw_context_sha256"),
        where=f"{where}.raw_context_sha256",
    )


def _validate_step_channel_evidence(
    evidence_value: Any, record: dict[str, Any], *, where: str,
) -> None:
    """Validate a decision's channel snapshot and its policy bindings."""

    evidence = _require_sealed_object(evidence_value, where=where)
    if evidence.get("schema_version") != "agribrain.step_channel_evidence.v1":
        raise RuntimeError(f"{where} has an unsupported schema_version")
    try:
        evidence_hour = float(evidence.get("hour"))
        record_hour = float(record.get("hour"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{where} has an invalid hour binding") from exc
    if not math.isclose(evidence_hour, record_hour, rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError(f"{where} hour does not match the decision")
    role = record.get("role")
    if evidence.get("active_role") != role:
        raise RuntimeError(f"{where} active_role does not match the decision")

    peer = _require_sealed_object(evidence.get("peer"), where=f"{where}.peer")
    if peer.get("bias_function") != "src.agents.message.message_bias_from_inbox":
        raise RuntimeError(f"{where}.peer has an unsupported bias_function")
    consumed = peer.get("consumed")
    emitted = peer.get("emitted")
    if not isinstance(consumed, list) or not isinstance(emitted, list):
        raise RuntimeError(f"{where}.peer message inventories are invalid")
    if peer.get("consumed_count") != len(consumed) or peer.get(
        "emitted_count"
    ) != len(emitted):
        raise RuntimeError(f"{where}.peer message counts are inconsistent")
    bias_messages: list[InterAgentMessage] = []
    for group_name, messages in (("consumed", consumed), ("emitted", emitted)):
        for index, message_value in enumerate(messages):
            message_where = f"{where}.peer.{group_name}[{index}]"
            message = _require_sealed_object(message_value, where=message_where)
            body = {
                key: message.get(key)
                for key in ("sender", "recipient", "type", "payload", "hour")
            }
            if _require_sha256(
                message.get("message_sha256"),
                where=f"{message_where}.message_sha256",
            ) != _evidence_sha256(body):
                raise RuntimeError(f"{message_where} message SHA-256 mismatch")
            if group_name == "consumed" and message.get("used_for_policy_bias"):
                if message.get("consumer_role") != role:
                    raise RuntimeError(
                        f"{message_where} policy-bias consumer is not active_role"
                    )
                try:
                    bias_messages.append(InterAgentMessage(
                        sender=message["sender"], recipient=message["recipient"],
                        msg_type=MessageType(message["type"]),
                        payload=message["payload"], hour=float(message["hour"]),
                    ))
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"{message_where} cannot reconstruct a peer message"
                    ) from exc
    reconstructed_bias = message_bias_from_inbox(bias_messages).tolist()
    if _max_abs_vector_difference(
        peer.get("policy_bias"), reconstructed_bias, 3,
    ) > 0.0:
        raise RuntimeError(f"{where}.peer policy_bias does not reconstruct")
    if _require_sha256(
        peer.get("policy_bias_sha256"),
        where=f"{where}.peer.policy_bias_sha256",
    ) != _evidence_sha256(peer.get("policy_bias")):
        raise RuntimeError(f"{where}.peer policy-bias SHA-256 mismatch")
    if _max_abs_vector_difference(
        peer.get("policy_bias"), record.get("peer_message_bias"), 3,
    ) > 0.0:
        raise RuntimeError(f"{where}.peer bias differs from the decision")
    if peer.get("policy_logit_equation") != (
        "z_pre_context=z_without_peer+b_peer"
    ) or _max_abs_vector_difference(
        peer.get("policy_logit_term"), peer.get("policy_bias"), 3,
    ) > 0.0:
        raise RuntimeError(f"{where}.peer policy-logit binding is inconsistent")

    primary = _require_sealed_object(
        evidence.get("primary"), where=f"{where}.primary",
    )
    cooperative = _require_sealed_object(
        evidence.get("cooperative"), where=f"{where}.cooperative",
    )
    if primary.get("role") != role or cooperative.get("role") != "cooperative":
        raise RuntimeError(f"{where} channel role identities are inconsistent")
    if not isinstance(cooperative.get("active"), bool):
        raise RuntimeError(f"{where}.cooperative.active is not boolean")
    for name, channel in (("primary", primary), ("cooperative", cooperative)):
        _validate_mcp_evidence(channel.get("mcp"), where=f"{where}.{name}.mcp")
        _validate_retrieval_evidence(
            channel.get("retrieval"), where=f"{where}.{name}.retrieval",
        )

    caps = capabilities_for(str(record.get("mode")))
    if not isinstance(peer.get("enabled"), bool) or peer.get("enabled") != bool(
        caps.peer_messages
    ):
        raise RuntimeError(f"{where}.peer enabled flag differs from mode capability")
    for name, count_field, query_field in (
        ("primary", "primary_mcp_tools_invoked_step", "primary_pirag_query_attempted_step"),
        ("cooperative", "cooperative_mcp_tools_invoked_step", "cooperative_pirag_query_attempted_step"),
    ):
        channel = primary if name == "primary" else cooperative
        mcp = channel["mcp"]
        retrieval = channel["retrieval"]
        if record.get(count_field) != mcp.get("tools_invoked"):
            raise RuntimeError(f"{where}.{name} MCP tools differ from decision")
        if bool(record.get(query_field)) != bool(retrieval.get("attempted")):
            raise RuntimeError(f"{where}.{name} retrieval attempt differs from decision")

    integration = record.get("context_integration")
    if caps.context_kind is None:
        for name, channel in (("primary", primary), ("cooperative", cooperative)):
            if channel["mcp"].get("attempted") is not False or channel[
                "retrieval"
            ].get("attempted") is not False or channel["retrieval"].get(
                "effective_psi"
            ) is not None:
                raise RuntimeError(
                    f"{where}.{name} fabricates activity in a non-context mode"
                )
            if (
                channel["mcp"].get("tools_invoked") != []
                or channel["mcp"]["protocol"].get("records") != []
                or channel["retrieval"].get("query") != ""
                or channel["retrieval"].get("ordered_citations") != []
                or channel["retrieval"]["protocol"].get("records") != []
                or not isinstance(channel["mcp"].get("skip_reason"), str)
                or not isinstance(channel["retrieval"].get("skip_reason"), str)
            ):
                raise RuntimeError(
                    f"{where}.{name} non-context evidence is not honestly empty"
                )
    else:
        if not isinstance(integration, dict):
            raise RuntimeError(f"{where} lacks context integration bindings")
        primary_trace = integration.get("primary")
        cooperative_trace = integration.get("cooperative")
        if not isinstance(primary_trace, dict) or _max_abs_vector_difference(
            primary["retrieval"].get("effective_psi"),
            primary_trace.get("effective_psi"), 5,
        ) > 0.0:
            raise RuntimeError(f"{where}.primary effective_psi is unbound")
        if _max_abs_vector_difference(
            primary["retrieval"].get("effective_psi"), record.get("psi"), 5,
        ) > 0.0:
            raise RuntimeError(f"{where}.primary effective_psi differs from psi")
        if cooperative_trace is None:
            if cooperative["retrieval"].get("effective_psi") is not None:
                raise RuntimeError(f"{where}.cooperative fabricates effective_psi")
        elif not isinstance(cooperative_trace, dict) or _max_abs_vector_difference(
            cooperative["retrieval"].get("effective_psi"),
            cooperative_trace.get("effective_psi"), 5,
        ) > 0.0:
            raise RuntimeError(f"{where}.cooperative effective_psi is unbound")
        if primary["retrieval"].get("retrieval_kind") != caps.retrieval_kind:
            raise RuntimeError(f"{where}.primary retrieval kind differs from mode")
        # The decision-record alias preserves the context builder's source
        # order. The channel evidence separately retains ranked order for the
        # policy-facing citation list and source order for this exact binding.
        if primary["retrieval"].get("source_order_evidence_hashes") != record.get(
            "retrieval_evidence_hashes"
        ):
            raise RuntimeError(
                f"{where}.primary source-order retrieval hashes differ from decision"
            )
        for channel_name, channel, trace in (
            ("primary", primary, primary_trace),
            ("cooperative", cooperative, cooperative_trace),
        ):
            if trace is None:
                continue
            retrieval = channel["retrieval"]
            for field in (
                "retrieval_gate", "retrieval_blocked_reason", "temporal_scale",
                "temporal_gate_requested", "temporal_gate_applied",
                "temporal_continuity_score", "temporal_base",
                "temporal_decay", "physics_scale", "rag_total_scale",
                "effective_psi",
            ):
                if retrieval.get(field) != trace.get(field):
                    raise RuntimeError(
                        f"{where}.{channel_name} {field} is unbound"
                    )
        for evidence_field, record_field in (
            ("top_doc_id", "retrieval_top_doc_id"),
            ("top_citation_score", "retrieval_top_score"),
            ("top_fused_score", "retrieval_top_fused_score"),
            ("top_rerank_score", "retrieval_top_rerank_score"),
        ):
            if primary["retrieval"].get(evidence_field) != record.get(record_field):
                raise RuntimeError(
                    f"{where}.primary {evidence_field} differs from decision"
                )

    _validate_sealed_evidence(evidence, where=where)


def _max_abs_vector_difference(left: Any, right: Any, length: int) -> float:
    if not _finite_vector(left, length) or not _finite_vector(right, length):
        return math.inf
    return max(
        abs(float(a) - float(b)) for a, b in zip(left, right, strict=True)
    )


def _validate_forward_context_trace(
    trace: Any, *, theta: Any, expected_retrieval_kind: str, where: str,
) -> None:
    """Validate one primary/cooperative channel-separated forward trace."""

    if not isinstance(trace, dict):
        raise RuntimeError(f"{where} is not an object")
    vector_fields = (
        "mcp_preclip_component", "pirag_preclip_component",
        "preclip_modifier", "clip_derivative", "final_modifier",
        "nonfeature_residual",
    )
    matrix_fields = (
        "linear_feature_contributions",
        "channel_scaled_feature_contributions", "feature_contributions",
        "modifier_theta_jacobian",
    )
    for field in vector_fields:
        if not _finite_vector(trace.get(field), 3):
            raise RuntimeError(f"{where} has invalid {field}")
    for field in matrix_fields:
        if not _finite_matrix(trace.get(field), 3, 5):
            raise RuntimeError(f"{where} has invalid {field}")
    if not _finite_vector(trace.get("effective_psi"), 5):
        raise RuntimeError(f"{where} has invalid effective_psi")
    if not _finite_matrix(trace.get("effective_theta"), 3, 5):
        raise RuntimeError(f"{where} has invalid effective_theta")
    if not _finite_matrix(theta, 3, 5):
        raise RuntimeError(f"{where} has no valid recorded context matrix")
    if trace.get("clip_applied") is not True:
        raise RuntimeError(f"{where} must record the mandatory context clip")
    if trace.get("over_steer") is not False:
        raise RuntimeError(f"{where} activates the retired over-steer path")
    if trace.get("retrieval_kind") != expected_retrieval_kind:
        raise RuntimeError(
            f"{where} retrieval kind does not match the mode capability"
        )

    try:
        retrieval_gate = float(trace["retrieval_gate"])
        temporal_scale = float(trace["temporal_scale"])
        physics_scale = float(trace["physics_scale"])
        rag_total_scale = float(trace["rag_total_scale"])
        global_scale = float(trace["global_scale"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{where} has invalid channel scales") from exc
    if retrieval_gate not in (0.0, 1.0):
        raise RuntimeError(f"{where} retrieval gate is not binary")
    if not all(math.isfinite(value) for value in (
        temporal_scale, physics_scale, rag_total_scale, global_scale,
    )) or not 0.0 <= physics_scale <= 1.0:
        raise RuntimeError(f"{where} has out-of-range channel scales")
    temporal_requested = trace.get("temporal_gate_requested")
    temporal_applied = trace.get("temporal_gate_applied")
    if not isinstance(temporal_requested, bool) or not isinstance(
        temporal_applied, bool,
    ):
        raise RuntimeError(f"{where} temporal gate flags are not boolean")
    if temporal_applied:
        if not temporal_requested or expected_retrieval_kind != "pirag" or (
            trace.get("context_mode") == "mcp_only"
        ):
            raise RuntimeError(f"{where} applies the temporal gate out of scope")
        try:
            continuity = float(trace["temporal_continuity_score"])
            temporal_base = float(trace["temporal_base"])
            temporal_decay = float(trace["temporal_decay"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"{where} lacks temporal gate operands") from exc
        if not all(math.isfinite(value) for value in (
            continuity, temporal_base, temporal_decay,
        )) or temporal_base < 0.0 or temporal_decay < 0.0 or not (
            0.0 <= continuity <= 1.0
        ):
            raise RuntimeError(f"{where} has invalid temporal gate operands")
        expected_temporal_scale = temporal_base - temporal_decay * continuity
        if expected_temporal_scale < 0.0 or not math.isclose(
            temporal_scale, expected_temporal_scale,
            rel_tol=1e-12, abs_tol=1e-12,
        ):
            raise RuntimeError(f"{where} temporal gate does not reconstruct")
    elif (
        trace.get("temporal_continuity_score") is not None
        or trace.get("temporal_base") is not None
        or trace.get("temporal_decay") is not None
        or not math.isclose(temporal_scale, 1.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise RuntimeError(f"{where} records an unapplied temporal gate")
    if not math.isclose(
        rag_total_scale,
        retrieval_gate * temporal_scale * physics_scale,
        rel_tol=1e-12, abs_tol=1e-12,
    ):
        raise RuntimeError(f"{where} piRAG scale does not reconstruct")
    if expected_retrieval_kind == "standard" and not (
        math.isclose(temporal_scale, 1.0, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(physics_scale, 1.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise RuntimeError(
            f"{where} standard RAG applies a piRAG-only multiplier"
        )

    mcp = [float(value) for value in trace["mcp_preclip_component"]]
    rag = [float(value) for value in trace["pirag_preclip_component"]]
    preclip = [float(value) for value in trace["preclip_modifier"]]
    expected_preclip = [mcp[i] + rag[i] for i in range(3)]
    if _max_abs_vector_difference(preclip, expected_preclip, 3) > 1e-10:
        raise RuntimeError(f"{where} channel terms do not reconstruct preclip")

    clip_derivative = [float(value) for value in trace["clip_derivative"]]
    expected_final = (
        [max(-1.0, min(1.0, value)) for value in preclip]
        if trace["clip_applied"] else preclip
    )
    if _max_abs_vector_difference(
        trace["final_modifier"], expected_final, 3,
    ) > 1e-10:
        raise RuntimeError(f"{where} final modifier does not match clip")
    expected_derivative = (
        [1.0 if abs(value) < 1.0 else 0.0 for value in preclip]
        if trace["clip_applied"] else [1.0] * 3
    )
    if _max_abs_vector_difference(
        clip_derivative, expected_derivative, 3,
    ) > 1e-12:
        raise RuntimeError(f"{where} clip derivative is inconsistent")

    psi = [float(value) for value in trace["effective_psi"]]
    effective_theta = trace["effective_theta"]
    linear = trace["linear_feature_contributions"]
    channel_scaled = trace["channel_scaled_feature_contributions"]
    jacobian = trace["modifier_theta_jacobian"]
    for row in range(3):
        for column in range(5):
            if not math.isclose(
                float(effective_theta[row][column]), float(theta[row][column]),
                rel_tol=1e-12, abs_tol=1e-12,
            ):
                raise RuntimeError(f"{where} context matrix snapshot is inconsistent")
            channel_scale = rag_total_scale if column in (2, 3) else 1.0
            expected_linear = float(theta[row][column]) * psi[column]
            if not math.isclose(
                float(linear[row][column]), expected_linear,
                rel_tol=1e-10, abs_tol=1e-12,
            ):
                raise RuntimeError(f"{where} linear context terms are inconsistent")
            expected_scaled = global_scale * channel_scale * expected_linear
            if not math.isclose(
                float(channel_scaled[row][column]), expected_scaled,
                rel_tol=1e-10, abs_tol=1e-12,
            ):
                raise RuntimeError(f"{where} channel-scaled terms are inconsistent")
            expected = (
                clip_derivative[row] * global_scale * channel_scale * psi[column]
            )
            if not math.isclose(
                float(jacobian[row][column]), expected,
                rel_tol=1e-10, abs_tol=1e-12,
            ):
                raise RuntimeError(f"{where} learner Jacobian is inconsistent")

    mcp_columns = (0, 1, 4)
    rag_columns = (2, 3)
    for row in range(3):
        expected_mcp = sum(
            float(channel_scaled[row][column]) for column in mcp_columns
        )
        expected_rag = sum(
            float(channel_scaled[row][column]) for column in rag_columns
        )
        if not math.isclose(
            float(trace["mcp_preclip_component"][row]), expected_mcp,
            rel_tol=1e-10, abs_tol=1e-12,
        ) or not math.isclose(
            float(trace["pirag_preclip_component"][row]), expected_rag,
            rel_tol=1e-10, abs_tol=1e-12,
        ):
            raise RuntimeError(f"{where} channel component sums are inconsistent")
        attributed = (
            sum(float(value) for value in trace["feature_contributions"][row])
            + float(trace["nonfeature_residual"][row])
        )
        if not math.isclose(
            attributed, float(trace["final_modifier"][row]),
            rel_tol=1e-10, abs_tol=1e-12,
        ):
            raise RuntimeError(f"{where} feature attribution does not reconstruct")


def _validate_context_integration(
    integration: Any, *, modifier: Any, jacobian: Any, theta: Any,
    feature_contributions: Any, residual: Any,
    expected_retrieval_kind: str, where: str,
) -> None:
    """Validate hierarchical composition and its exact final Jacobian."""

    if not isinstance(integration, dict):
        raise RuntimeError(f"{where} context integration is not an object")
    primary = integration.get("primary")
    cooperative = integration.get("cooperative")
    composition = integration.get("composition")
    _validate_forward_context_trace(
        primary, theta=theta,
        expected_retrieval_kind=expected_retrieval_kind,
        where=f"{where}/primary",
    )
    if cooperative is not None:
        _validate_forward_context_trace(
            cooperative, theta=theta,
            expected_retrieval_kind=expected_retrieval_kind,
            where=f"{where}/cooperative",
        )
    if not isinstance(composition, dict):
        raise RuntimeError(f"{where} composition trace is missing")
    if not isinstance(composition.get("clip_applied"), bool):
        raise RuntimeError(f"{where} composition clip flag is invalid")
    for field in ("preclip_modifier", "clip_derivative", "final_modifier"):
        if not _finite_vector(composition.get(field), 3):
            raise RuntimeError(f"{where} composition has invalid {field}")
    if not _finite_matrix(composition.get("modifier_theta_jacobian"), 3, 5):
        raise RuntimeError(f"{where} composition Jacobian is invalid")
    scope = composition.get("scope")
    if scope not in {"primary_context", "cooperative_blend", "cooperative_veto"}:
        raise RuntimeError(f"{where} composition scope is invalid")
    if scope != "primary_context" and cooperative is None:
        raise RuntimeError(f"{where} composition omits cooperative trace")

    primary_modifier = [float(value) for value in primary["final_modifier"]]
    if scope == "primary_context":
        expected_preclip = primary_modifier
        expected_features = [
            [float(value) for value in row]
            for row in primary["feature_contributions"]
        ]
        expected_residual = [
            float(value) for value in primary["nonfeature_residual"]
        ]
    else:
        cooperative_modifier = [
            float(value) for value in cooperative["final_modifier"]
        ]
        cooperative_features = cooperative["feature_contributions"]
        cooperative_residual = cooperative["nonfeature_residual"]
        if scope == "cooperative_blend":
            expected_preclip = [
                0.7 * primary_modifier[row] + 0.3 * cooperative_modifier[row]
                for row in range(3)
            ]
            expected_features = [
                [
                    0.7 * float(primary["feature_contributions"][row][column])
                    + 0.3 * float(cooperative_features[row][column])
                    for column in range(5)
                ]
                for row in range(3)
            ]
            expected_residual = [
                0.7 * float(primary["nonfeature_residual"][row])
                + 0.3 * float(cooperative_residual[row])
                for row in range(3)
            ]
        else:
            veto_bias = (-0.20, 0.20, 0.0)
            expected_preclip = [
                cooperative_modifier[row] + veto_bias[row]
                for row in range(3)
            ]
            expected_features = [
                [float(value) for value in row]
                for row in cooperative_features
            ]
            expected_residual = [
                float(cooperative_residual[row]) + veto_bias[row]
                for row in range(3)
            ]

    preclip = [float(value) for value in composition["preclip_modifier"]]
    if _max_abs_vector_difference(preclip, expected_preclip, 3) > 1e-10:
        raise RuntimeError(f"{where} composition inputs do not reconstruct")
    derivative = [float(value) for value in composition["clip_derivative"]]
    expected_final = (
        [max(-1.0, min(1.0, value)) for value in preclip]
        if composition["clip_applied"] else preclip
    )
    if _max_abs_vector_difference(
        composition["final_modifier"], expected_final, 3,
    ) > 1e-10 or _max_abs_vector_difference(
        modifier, expected_final, 3,
    ) > 1e-10:
        raise RuntimeError(f"{where} composed modifier is inconsistent")
    expected_derivative = (
        [1.0 if abs(value) < 1.0 else 0.0 for value in preclip]
        if composition["clip_applied"] else [1.0] * 3
    )
    if _max_abs_vector_difference(
        derivative, expected_derivative, 3,
    ) > 1e-12:
        raise RuntimeError(f"{where} composition clip derivative is inconsistent")

    if not _finite_matrix(feature_contributions, 3, 5) or not _finite_vector(
        residual, 3,
    ):
        raise RuntimeError(f"{where} stored context attribution is invalid")
    for row in range(3):
        before = expected_preclip[row]
        after = expected_final[row]
        scale = after / before if abs(before) > 1e-15 else 1.0
        expected_features[row] = [
            value * scale for value in expected_features[row]
        ]
        expected_residual[row] *= scale
        expected_residual[row] += after - (
            sum(expected_features[row]) + expected_residual[row]
        )
        for column in range(5):
            if not math.isclose(
                float(feature_contributions[row][column]),
                expected_features[row][column],
                rel_tol=1e-10, abs_tol=1e-12,
            ):
                raise RuntimeError(
                    f"{where} context attribution does not reconstruct composition"
                )
        if not math.isclose(
            float(residual[row]), expected_residual[row],
            rel_tol=1e-10, abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"{where} context attribution does not reconstruct composition"
            )

    primary_j = primary["modifier_theta_jacobian"]
    cooperative_j = (
        cooperative["modifier_theta_jacobian"]
        if cooperative is not None else None
    )
    composed_j = composition["modifier_theta_jacobian"]
    for row in range(3):
        for column in range(5):
            if scope == "primary_context":
                before_clip = float(primary_j[row][column])
            elif scope == "cooperative_blend":
                before_clip = (
                    0.7 * float(primary_j[row][column])
                    + 0.3 * float(cooperative_j[row][column])
                )
            else:
                before_clip = float(cooperative_j[row][column])
            expected = derivative[row] * before_clip
            if not math.isclose(
                float(composed_j[row][column]), expected,
                rel_tol=1e-10, abs_tol=1e-12,
            ):
                raise RuntimeError(f"{where} composed Jacobian is inconsistent")
            if not math.isclose(
                float(jacobian[row][column]), expected,
                rel_tol=1e-10, abs_tol=1e-12,
            ):
                raise RuntimeError(f"{where} stored learner Jacobian is inconsistent")


def _softmax(values: list[float]) -> list[float]:
    peak = max(values)
    weights = [math.exp(value - peak) for value in values]
    total = sum(weights)
    return [value / total for value in weights]


def _strict_object(line: str, *, path: Path, line_number: int) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    try:
        value = json.loads(line, parse_constant=reject_constant)
    except Exception as exc:
        raise RuntimeError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object at {path}:{line_number}")
    return value


def _source_replay_float(
    observed: Any,
    expected: float,
    *,
    path: Path,
    step_index: int,
    field: str,
) -> None:
    """Compare one ledger scalar to its independently replayed source value."""

    try:
        value = float(observed)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{path}:{step_index + 2} source-bound replay field {field!r} "
            "is not numeric"
        ) from exc
    if not math.isfinite(value) or not math.isclose(
        value, float(expected), rel_tol=1e-12, abs_tol=1e-12,
    ):
        raise RuntimeError(
            f"{path}:{step_index + 2} source-bound replay mismatch for "
            f"{field}: ledger={value!r}, expected={float(expected)!r}"
        )


def _validate_source_bound_episode_replay(
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    path: Path,
    mode: str,
    scenario: str,
    benchmark_seed: int,
    policy: Policy,
    stochastic_layer: StochasticLayer,
    scenario_frame: pd.DataFrame | None = None,
) -> dict[str, str]:
    """Rebuild a retained episode from committed inputs rather than its hashes.

    The ordinary core path starts at ``PUBLICATION_DATA_CSV`` and reconstructs
    the episode-3 scenario draw, keyed environmental observations, forecasts,
    rolling market state, transport draw, and both spoilage trajectories.
    Structural validation supplies a scenario frame produced while its exact
    LHS overrides are active; all later transformations are still replayed
    here from that explicit frame and stochastic layer.

    H3's declared observation-only dose is applied after the canonical sensor
    stream, in the same non-commuting order as the producer.  Its primitive
    dose is independently checked by the H3 validator; this routine binds the
    resulting policy observations to the source-derived nominal world.
    """

    if len(records) != 288:
        raise RuntimeError(
            f"{path} source-bound replay requires a complete 288-step ledger"
        )
    caps = capabilities_for(mode)
    policy_pinn = load_frozen_checkpoint() if caps.spoilage_residual else None
    if scenario_frame is None:
        if not PUBLICATION_DATA_CSV.is_file():
            raise RuntimeError(
                f"committed publication source is missing: {PUBLICATION_DATA_CSV}"
            )
        base = pd.read_csv(PUBLICATION_DATA_CSV, parse_dates=["timestamp"])
        if len(base) != 288:
            raise RuntimeError(
                f"committed publication source has {len(base)} rows; expected 288"
            )
        scenario_seed = _stream_seed(
            benchmark_seed, scenario, 3, "scenario",
        )
        scenario_frame = apply_scenario(
            base,
            scenario,
            policy,
            np.random.default_rng(scenario_seed),
            stoch=stochastic_layer,
        )
    else:
        scenario_frame = scenario_frame.copy()

    required_columns = {
        "timestamp", "tempC", "RH", "shockG",
        "inventory_units", "demand_units",
    }
    missing_columns = required_columns.difference(scenario_frame.columns)
    if missing_columns:
        raise RuntimeError(
            f"{path} source-bound scenario frame lacks {sorted(missing_columns)}"
        )
    if len(scenario_frame) != 288:
        raise RuntimeError(
            f"{path} source-bound scenario frame has {len(scenario_frame)} "
            "rows; expected 288"
        )
    timestamps = pd.to_datetime(scenario_frame["timestamp"])
    hours = (
        (timestamps - timestamps.iloc[0]).dt.total_seconds() / 3600.0
    ).to_numpy(dtype=float)
    if len(hours) != 288 or not np.all(np.isfinite(hours)):
        raise RuntimeError(f"{path} source-bound timestamps are invalid")

    latent_temp = scenario_frame["tempC"].to_numpy(dtype=float)
    latent_rh = scenario_frame["RH"].to_numpy(dtype=float)
    latent_inventory = scenario_frame["inventory_units"].to_numpy(dtype=float)
    latent_demand = scenario_frame["demand_units"].to_numpy(dtype=float)
    latent_shock = scenario_frame["shockG"].to_numpy(dtype=float)
    for name, values in (
        ("tempC", latent_temp), ("RH", latent_rh),
        ("inventory_units", latent_inventory),
        ("demand_units", latent_demand), ("shockG", latent_shock),
    ):
        if not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"{path} committed source produced non-finite {name}"
            )

    source_temp = (
        scenario_frame["temp_policy_observed"].to_numpy(dtype=float)
        if "temp_policy_observed" in scenario_frame.columns else latent_temp
    )
    source_rh = (
        scenario_frame["rh_policy_observed"].to_numpy(dtype=float)
        if "rh_policy_observed" in scenario_frame.columns else latent_rh
    )
    source_inventory = (
        scenario_frame["inventory_policy_observed"].to_numpy(dtype=float)
        if "inventory_policy_observed" in scenario_frame.columns
        else latent_inventory
    )
    source_demand = (
        scenario_frame["demand_policy_observed"].to_numpy(dtype=float)
        if "demand_policy_observed" in scenario_frame.columns else latent_demand
    )
    # The production supply-proxy forecaster reads the scenario dataframe,
    # before the keyed inventory-observation perturbation is applied.
    supply_history_source = source_inventory

    effective_k_ref = stochastic_layer.perturb_k_ref(
        policy.k_ref, counter=0,
    )
    effective_ea_r = stochastic_layer.perturb_ea_r(
        policy.Ea_R, counter=0,
    )
    _source_replay_float(
        metadata.get("effective_k_ref"), effective_k_ref,
        path=path, step_index=-1, field="effective_k_ref",
    )
    _source_replay_float(
        metadata.get("effective_Ea_R"), effective_ea_r,
        path=path, step_index=-1, field="effective_Ea_R",
    )
    expected_latent_spoilage_model = synthetic_dgp_provenance(
        k_ref=effective_k_ref,
        Ea_R=effective_ea_r,
        T_ref_K=policy.T_ref_K,
        beta=policy.beta_humidity,
        lag_lambda=policy.lag_lambda,
        packaging_index=DEFAULT_PACKAGING_INDEX,
    )
    if metadata.get("latent_spoilage_model") != expected_latent_spoilage_model:
        raise RuntimeError(
            f"{path} source-bound replay has incorrect latent DGP provenance"
        )
    source_latent_spoilage = compute_spoilage_independent_synthetic_dgp(
        scenario_frame,
        k_ref=effective_k_ref,
        Ea_R=effective_ea_r,
        T_ref_K=policy.T_ref_K,
        beta=policy.beta_humidity,
        lag_lambda=policy.lag_lambda,
        packaging_index=DEFAULT_PACKAGING_INDEX,
    )
    source_latent_rho = source_latent_spoilage[
        "spoilage_risk"
    ].to_numpy(dtype=float)

    treatment = metadata.get("observation_treatment")
    if not isinstance(treatment, dict):
        raise RuntimeError(f"{path} source-bound replay lacks observation treatment")
    stressor = str(treatment.get("stressor", "nominal"))
    allowed_stressors = {
        "nominal", "sensor_noise", "missing_data", "telemetry_delay",
        "mcp_fault_injection", "compounded",
    }
    if stressor not in allowed_stressors:
        raise RuntimeError(
            f"{path} source-bound replay has unknown stressor {stressor!r}"
        )

    canonical_temp_history: list[float] = []
    canonical_rh_history: list[float] = []
    predelay_temp_history: list[float] = []
    predelay_rh_history: list[float] = []
    expected_temp_observed: list[float] = []
    expected_rh_observed: list[float] = []
    expected_inventory_observed: list[float] = []
    expected_demand_observed: list[float] = []
    expected_demand_forecast: list[float] = []
    expected_demand_std: list[float] = []
    expected_supply_forecast: list[float] = []
    expected_supply_std: list[float] = []
    expected_regime: list[float] = []
    expected_price: list[float] = []
    expected_transport: list[float] = []
    expected_latent_rho: list[float] = []
    expected_observed_rho: list[float] = []
    latent_quality = 1.0
    previous_latent_rh_transient = 0.0
    observed_mechanistic_rho = 0.0
    observed_deployed_quality = 1.0

    for index, record in enumerate(records):
        temp = stochastic_layer.perturb_temperature(
            float(source_temp[index]), counter=index,
        )
        rh = stochastic_layer.perturb_humidity(
            float(source_rh[index]), counter=index,
        )
        if index > 0 and stochastic_layer.should_delay(counter=index):
            temp = canonical_temp_history[-1]
            rh = canonical_rh_history[-1]
        canonical_temp_history.append(float(temp))
        canonical_rh_history.append(float(rh))

        if stressor in {"sensor_noise", "compounded"}:
            try:
                temp += float(record["h3_temp_noise_c"])
                rh = float(np.clip(
                    rh + float(record["h3_rh_noise_pct"]), 15.0, 100.0,
                ))
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"{path}:{index + 2} source-bound replay lacks H3 noise dose"
                ) from exc
        if stressor in {"missing_data", "compounded"} and bool(
            record.get("h3_missing_observation", False)
        ):
            if index == 0:
                raise RuntimeError(
                    f"{path}:{index + 2} source-bound H3 dose masks step zero"
                )
            temp = predelay_temp_history[-1]
            rh = predelay_rh_history[-1]
        predelay_temp_history.append(float(temp))
        predelay_rh_history.append(float(rh))
        if stressor in {"telemetry_delay", "compounded"}:
            try:
                source_index = int(record["h3_telemetry_source_step_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"{path}:{index + 2} source-bound replay lacks H3 delay dose"
                ) from exc
            if source_index < 0 or source_index > index:
                raise RuntimeError(
                    f"{path}:{index + 2} source-bound H3 delay index is invalid"
                )
            temp = predelay_temp_history[source_index]
            rh = predelay_rh_history[source_index]

        inventory = stochastic_layer.perturb_inventory(
            float(source_inventory[index]), counter=index,
        )
        demand = stochastic_layer.perturb_demand(
            float(source_demand[index]), counter=index,
        )
        expected_temp_observed.append(float(temp))
        expected_rh_observed.append(float(rh))
        expected_inventory_observed.append(float(inventory))
        expected_demand_observed.append(float(demand))

        if index > 0:
            step_h = float(hours[index] - hours[index - 1])
            latent_rh_transient = float(
                abs(latent_rh[index] - latent_rh[index - 1]) / step_h
            )
            mid_time = 0.5 * float(hours[index] + hours[index - 1])
            base_rate = float(arrhenius_k(
                0.5 * float(latent_temp[index] + latent_temp[index - 1]),
                k_ref=effective_k_ref,
                Ea_R=effective_ea_r,
                T_ref_K=policy.T_ref_K,
                rh_frac=0.005 * float(latent_rh[index] + latent_rh[index - 1]),
                beta=policy.beta_humidity,
            ))
            alpha = (
                mid_time / (mid_time + policy.lag_lambda)
                if policy.lag_lambda > 0.0 else 1.0
            )
            handling = 0.5 * float(
                latent_shock[index] + latent_shock[index - 1]
            )
            transient = 0.5 * (
                latent_rh_transient + previous_latent_rh_transient
            )
            log_multiplier = (
                PACKAGING_LOG_RATE_COEFFICIENT
                * (DEFAULT_PACKAGING_INDEX - PACKAGING_CENTER)
                + HANDLING_SHOCK_LOG_RATE_COEFFICIENT * handling
                + RH_TRANSIENT_LOG_RATE_COEFFICIENT * transient
            )
            latent_quality *= math.exp(
                -base_rate * alpha * math.exp(log_multiplier) * step_h
            )
            previous_latent_rh_transient = latent_rh_transient
            observed_mechanistic_rho = advance_spoilage_risk_midpoint(
                observed_mechanistic_rho,
                previous_temp_C=expected_temp_observed[index - 1],
                current_temp_C=float(temp),
                previous_rh_pct=expected_rh_observed[index - 1],
                current_rh_pct=float(rh),
                previous_hour=float(hours[index - 1]),
                current_hour=float(hours[index]),
                k_ref=effective_k_ref,
                Ea_R=effective_ea_r,
                T_ref_K=policy.T_ref_K,
                beta=policy.beta_humidity,
                lag_lambda=policy.lag_lambda,
            )
        latent_rho = 1.0 - latent_quality
        if policy_pinn is None:
            observed_rho = observed_mechanistic_rho
        else:
            observed_rh_transient = 0.0
            if index > 0:
                step_h = float(hours[index] - hours[index - 1])
                if step_h > 0.0:
                    observed_rh_transient = float(
                        abs(
                            expected_rh_observed[index]
                            - expected_rh_observed[index - 1]
                        ) / step_h
                    )
            observed_features = build_residual_feature_row(
                time_h=float(hours[index]),
                temp_c=float(temp),
                rh_pct=float(rh),
                shock_g=float(latent_shock[index]),
                rh_transient_per_h=observed_rh_transient,
                k_ref=effective_k_ref,
                ea_over_r=effective_ea_r,
            )
            observed_delta = float(predict_residual(
                observed_features, policy_pinn,
            )[0])
            observed_deployed_quality = min(
                observed_deployed_quality,
                float(np.clip(
                    1.0 - observed_mechanistic_rho + observed_delta, 0.0, 1.0,
                )),
            )
            observed_rho = 1.0 - observed_deployed_quality
        if not math.isclose(
            float(latent_rho), float(source_latent_rho[index]),
            rel_tol=1e-12, abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"{path}:{index + 2} scalar/vector synthetic DGP replay diverged"
            )
        # Preserve the exact vectorized operation order used by the
        # producer so cryptographic hashes do not differ at one ULP even when
        # the independent scalar reconstruction above is numerically equal.
        latent_rho = float(source_latent_rho[index])
        expected_latent_rho.append(float(latent_rho))
        expected_observed_rho.append(float(observed_rho))

        lookback = min(index + 1, 48)
        demand_tail = expected_demand_observed[-lookback:]
        demand_result = yield_demand_forecast(
            pd.DataFrame({"demand_units": demand_tail}), horizon=1,
        )
        demand_hat = float(demand_result["forecast"][0])
        demand_std = float(demand_result.get("std", 0.0) or 0.0)
        supply_result = persistence_forecast(
            pd.DataFrame({
                "inventory_units": supply_history_source[
                    max(0, index + 1 - lookback):index + 1
                ]
            }),
            horizon=1,
            series_col="inventory_units",
        )
        supply_hat = float(supply_result["forecast"][0])
        supply_std = float(supply_result.get("std", 0.0) or 0.0)
        expected_demand_forecast.append(demand_hat)
        expected_demand_std.append(demand_std)
        expected_supply_forecast.append(supply_hat)
        expected_supply_std.append(supply_std)

        rolling = pd.Series(demand_tail, dtype=float)
        rolling_mean = rolling.rolling(
            int(policy.boll_window), min_periods=1,
        ).mean().iloc[-1]
        rolling_std = rolling.rolling(
            int(policy.boll_window), min_periods=1,
        ).std().fillna(0.0).iloc[-1]
        z_score = (
            (float(rolling.iloc[-1]) - float(rolling_mean))
            / max(float(rolling_std), 1e-6)
        )
        expected_price.append(float(np.clip(z_score, -1.0, 1.0)))
        expected_regime.append(float(abs(z_score) > float(policy.boll_k)))
        expected_transport.append(
            stochastic_layer.perturb_transport_multiplier(counter=index)
        )

    expected_columns = {
        "hour": hours,
        "temp_outcome_environmental": latent_temp,
        "rh_outcome_environmental": latent_rh,
        "shock_g": latent_shock,
        "rho_outcome_environmental": expected_latent_rho,
        "inventory_outcome_environmental": latent_inventory,
        "demand_outcome_environmental": latent_demand,
        "transport_multiplier_outcome_environmental": expected_transport,
        "temp_policy_observed": expected_temp_observed,
        "rh_policy_observed": expected_rh_observed,
        "rho_policy_observed": expected_observed_rho,
        "inventory_policy_observed": expected_inventory_observed,
        "demand_policy_observed": expected_demand_observed,
        "demand_forecast_policy_observed": expected_demand_forecast,
        "demand_forecast_std_policy_observed": expected_demand_std,
        "supply_forecast_policy_observed": expected_supply_forecast,
        "supply_forecast_std_policy_observed": expected_supply_std,
        "bollinger_regime_flag": expected_regime,
        "price_signal": expected_price,
    }
    for field, expected_values in expected_columns.items():
        for index, expected in enumerate(expected_values):
            _source_replay_float(
                records[index].get(field), float(expected),
                path=path, step_index=index, field=field,
            )

    onset = float(
        scenario_frame["scenario_onset_offset_hours"].iloc[0]
        if "scenario_onset_offset_hours" in scenario_frame.columns else 0.0
    )
    _source_replay_float(
        metadata.get("scenario_onset_offset_hours"), onset,
        path=path, step_index=-1, field="scenario_onset_offset_hours",
    )
    latent_payload = {
        "hours": [float(value) for value in hours],
        "temp_outcome_environmental": [float(value) for value in latent_temp],
        "rh_outcome_environmental": [float(value) for value in latent_rh],
        "rho_outcome_environmental": expected_latent_rho,
        "inventory_outcome_environmental": [
            float(value) for value in latent_inventory
        ],
        "demand_outcome_environmental": [float(value) for value in latent_demand],
        "transport_multiplier_outcome_environmental": expected_transport,
        "effective_k_ref": float(effective_k_ref),
        "effective_Ea_R": float(effective_ea_r),
        "scenario_onset_offset_hours": onset,
    }
    observed_payload = {
        "hours": [float(value) for value in hours],
        "temp_policy_observed": expected_temp_observed,
        "rh_policy_observed": expected_rh_observed,
        "rho_policy_observed": expected_observed_rho,
        "inventory_policy_observed": expected_inventory_observed,
        "demand_forecast_policy_observed": expected_demand_forecast,
        "supply_forecast_policy_observed": expected_supply_forecast,
    }
    demand_payload = {
        "hours": [float(value) for value in hours],
        "demand_policy_observed": expected_demand_observed,
        "demand_forecast_policy_observed": expected_demand_forecast,
        "demand_regime_flag": expected_regime,
        "price_signal": expected_price,
    }
    expected_hashes = {
        "latent_environment_sha256": _canonical_sha256(latent_payload),
        "observed_policy_input_sha256": _canonical_sha256(observed_payload),
        "demand_observation_sha256": _canonical_sha256(demand_payload),
    }
    for field, expected_hash in expected_hashes.items():
        if metadata.get(field) != expected_hash:
            raise RuntimeError(
                f"{path} {field} is not derived from the committed source replay"
            )
    return expected_hashes


def validate_ledger(
    path: Path,
    *,
    mode: str,
    scenario: str,
    benchmark_seed: int,
    expected_outcome_equation_contract: dict[str, Any] | None = None,
    expected_policy: Policy | None = None,
    expected_policy_theta: np.ndarray | None = None,
    expected_context_prior: np.ndarray | None = None,
    expected_policy_temperature: float | None = None,
    expected_scenario_frame: pd.DataFrame | None = None,
    expected_stochastic_layer: StochasticLayer | None = None,
) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        lines = list(handle)
    if len(lines) != EXPECTED_RECORDS + 1:
        raise RuntimeError(
            f"{path} has {len(lines) - 1} records; expected {EXPECTED_RECORDS}"
        )

    header = _strict_object(lines[0], path=path, line_number=1)
    if header.get("_header") is not True:
        raise RuntimeError(f"{path} has no canonical header")
    if header.get("n_records") != EXPECTED_RECORDS:
        raise RuntimeError(f"{path} header record count is incorrect")
    metadata = header.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError(f"{path} header metadata is missing")
    if metadata.get("mode") != mode or metadata.get("scenario") != scenario:
        raise RuntimeError(f"{path} header mode/scenario conflicts with filename")
    if metadata.get("benchmark_seed") != benchmark_seed or (
        metadata.get("seed") != benchmark_seed
    ):
        raise RuntimeError(f"{path} header seed conflicts with seed directory")
    if metadata.get("episode_index") != 3:
        raise RuntimeError(f"{path} is not the retained episode-3 ledger")
    expected_policy_stream_id = _stream_id(
        benchmark_seed, scenario, 3, "policy",
    )
    expected_environment_stream_id = _stream_id(
        benchmark_seed, scenario, 3, "environment",
    )
    if metadata.get("policy_stream_id") != expected_policy_stream_id or (
        metadata.get("environment_stream_id")
        != expected_environment_stream_id
    ) or metadata.get("stochastic_stream_id") != expected_environment_stream_id:
        raise RuntimeError(f"{path} stream identity conflicts with its arm")
    policy_stream_seed = _stream_seed(
        benchmark_seed, scenario, 3, "policy",
    )
    expected_policy_theta = np.asarray(
        policy_theta_for_seed(
            np.asarray(DECLARED_THETA, dtype=float), benchmark_seed,
        ) if expected_policy_theta is None else expected_policy_theta,
        dtype=float,
    )
    if expected_policy_theta.shape != (3, 10) or not np.all(
        np.isfinite(expected_policy_theta)
    ):
        raise RuntimeError(f"{path} expected policy prior is not a finite 3x10 matrix")
    if metadata.get("policy_theta_initial_sha256") != _canonical_sha256(
        expected_policy_theta.tolist()
    ):
        raise RuntimeError(f"{path} policy prior does not match its seed")
    if expected_context_prior is None:
        from pirag.context_to_logits import THETA_CONTEXT

        expected_context_prior = np.asarray(THETA_CONTEXT, dtype=float)
    else:
        expected_context_prior = np.asarray(expected_context_prior, dtype=float)
    if expected_context_prior.shape != (3, 5) or not np.all(
        np.isfinite(expected_context_prior)
    ):
        raise RuntimeError(f"{path} expected context prior is not a finite 3x5 matrix")
    if metadata.get("context_prior_sha256") != _canonical_sha256(
        expected_context_prior.tolist()
    ):
        raise RuntimeError(f"{path} context prior does not match the declared arm")
    episode_policy = Policy() if expected_policy is None else expected_policy
    source_replay_layer = (
        _expected_publication_stochastic_layer(
            benchmark_seed=benchmark_seed,
            scenario=scenario,
            episode_index=3,
        )
        if expected_stochastic_layer is None else expected_stochastic_layer
    )
    if not isinstance(source_replay_layer, StochasticLayer):
        raise RuntimeError(f"{path} expected stochastic layer is invalid")
    if expected_policy_temperature is None:
        expected_policy_temperature = source_replay_layer.policy_temperature(
            base=1.0, counter=0,
        )
    try:
        expected_policy_temperature = float(expected_policy_temperature)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{path} expected policy temperature is invalid") from exc
    if not math.isfinite(expected_policy_temperature) or (
        expected_policy_temperature <= 0.0
    ):
        raise RuntimeError(f"{path} expected policy temperature is invalid")
    if metadata.get("learning_enabled") is not False:
        raise RuntimeError(f"{path} retained episode allowed learner updates")
    expected_phase = "fixed_evaluation" if mode == "static" else "frozen_evaluation"
    if metadata.get("episode_phase") != expected_phase:
        raise RuntimeError(
            f"{path} retained episode phase is not {expected_phase!r}"
        )
    if metadata.get("trace_schema_version") != TRACE_SCHEMA_VERSION:
        raise RuntimeError(f"{path} uses an obsolete trace schema")
    if metadata.get("dispatch_opportunity_count") != EXPECTED_RECORDS or not math.isclose(
        float(metadata.get("dispatch_cadence_hours", -1.0)), 0.25,
        abs_tol=1e-12,
    ):
        raise RuntimeError(f"{path} header dispatch accounting is invalid")
    for hash_field in (
        "context_prior_sha256", "policy_theta_initial_sha256",
        "latent_environment_sha256", "observed_policy_input_sha256",
        "demand_observation_sha256",
    ):
        if not isinstance(metadata.get(hash_field), str) or not re.fullmatch(
            r"[0-9a-f]{64}", metadata[hash_field],
        ):
            raise RuntimeError(f"{path} header has invalid {hash_field}")
    if metadata.get("demand_forecast_method") != "holt_linear":
        raise RuntimeError(
            f"{path} did not use the locked Holt-linear demand forecast"
        )
    if metadata.get("supply_forecast_method") != "persistence":
        raise RuntimeError(
            f"{path} did not use the locked persistence supply forecast"
        )
    outcome_contract = metadata.get("outcome_equation_contract")
    try:
        validate_outcome_equation_contract(
            outcome_contract,
            where=f"{path} header outcome_equation_contract",
            expected_contract=expected_outcome_equation_contract,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    episode_evidence_contract = metadata.get("episode_evidence_contract")
    try:
        validate_episode_evidence_contract(
            episode_evidence_contract,
            where=f"{path} header episode_evidence_contract",
            expected_contract=expected_publication_episode_evidence_contract(),
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    arrhenius_contract = outcome_contract["arrhenius"]
    for metadata_field, contract_field in (
        ("effective_k_ref", "effective_k_ref"),
        ("effective_Ea_R", "effective_ea_over_r"),
    ):
        try:
            header_value = float(metadata[metadata_field])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"{path} lacks numeric {metadata_field}") from exc
        if not math.isclose(
            header_value,
            float(arrhenius_contract[contract_field]),
            rel_tol=1e-15,
            abs_tol=1e-15,
        ):
            raise RuntimeError(
                f"{path} {metadata_field} differs from its outcome contract"
            )

    leaves: list[str] = []
    records: list[dict[str, Any]] = []
    reconstructed_outcomes: list[dict[str, Any]] = []
    frozen_theta_delta_by_role: dict[str, np.ndarray] = {}
    frozen_reward_shaping: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    frozen_context_theta: np.ndarray | None = None
    frozen_context_slca_amp: float | None = None
    required = {
        "step_index", "hour", "action", "action_idx", "probs",
        "policy_probs_pre_override", "policy_categorical_uniform",
        "sampled_action_pre_override", "reward",
        "waste", "rho", "slca", "ari",
        "rho_policy_observed", "rho_outcome_environmental",
        "temp_policy_observed", "temp_outcome_environmental",
        "rh_policy_observed", "rh_outcome_environmental",
        "shock_g",
        "inventory_policy_observed", "inventory_outcome_environmental",
        "demand_policy_observed", "demand_forecast_policy_observed",
        "supply_forecast_policy_observed", "bollinger_regime_flag",
        "regime_logit_bias",
        "demand_forecast_std_policy_observed",
        "supply_forecast_std_policy_observed",
        "price_signal",
        "demand_outcome_environmental",
        "transport_multiplier_outcome_environmental",
        "carbon_kg", "mode", "scenario", "role", "phi", "peer_message_bias",
        "combined_role_bias", "effective_theta_delta",
        "effective_slca_bonus_delta", "effective_slca_rho_delta",
        "effective_no_slca_offset_delta", "psi",
        "context_modifier", "base_logits", "post_context_logits_pre_override",
        "slca_shaping", "slca_amp", "policy_temperature",
        "modifier_mcp", "modifier_pirag", "retrieval_top_doc_id",
        "retrieval_top_score", "retrieval_top_fused_score",
        "retrieval_top_rerank_score", "retrieval_evidence_hashes",
        "effective_context_theta", "context_feature_contributions",
        "context_nonfeature_residual", "context_modifier_theta_jacobian",
        "context_integration",
        "chosen_action_context_contributions",
        "chosen_action_context_residual", "context_attribution_basis",
        "context_attribution_scope", "dominant_psi_idx",
        "dominant_context_component", "dominant_action_idx",
        "governance_override", "context_counterfactual_action_idx",
        "context_counterfactual_action", "context_counterfactual_probs",
        "context_counterfactual_categorical_uniform",
        "context_counterfactual_sampled_action_pre_override",
        "context_action_changed", "context_influence_active",
        "context_influence_counted", "context_influence_threshold",
        "simulated_dispatch_accounted",
        "primary_mcp_tools_invoked_step",
        "cooperative_mcp_tools_invoked_step",
        "primary_pirag_query_attempted_step",
        "cooperative_pirag_query_attempted_step",
        "step_channel_evidence",
        *ACTIVITY_STEP_FIELDS,
    }
    for index, line in enumerate(lines[1:], start=2):
        stored = _strict_object(line, path=path, line_number=index)
        missing = required.difference(stored)
        if missing:
            raise RuntimeError(f"{path}:{index} missing fields: {sorted(missing)}")
        leaf = stored.pop("_leaf", None)
        if not isinstance(leaf, str) or not re.fullmatch(r"[0-9a-f]{64}", leaf):
            raise RuntimeError(f"{path}:{index} has an invalid leaf hash")
        actual_leaf = hashlib.sha256(_canonical_bytes(stored)).hexdigest()
        if leaf != actual_leaf:
            raise RuntimeError(f"{path}:{index} leaf hash mismatch")
        leaves.append(leaf)
        records.append(stored)

        try:
            reconstructed_outcomes.append(validate_recorded_step_outcomes(
                stored,
                outcome_contract,
                where=f"{path}:{index}",
                contract_validated=True,
            ))
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

        step_index = index - 2
        if stored.get("step_index") != step_index or not math.isclose(
            float(stored.get("hour", -1.0)), 0.25 * step_index, abs_tol=1e-9,
        ):
            raise RuntimeError(f"{path}:{index} step index/cadence mismatch")
        if not math.isclose(
            float(stored["rho"]), float(stored["rho_policy_observed"]),
            abs_tol=1e-15,
        ):
            raise RuntimeError(f"{path}:{index} rho alias is ambiguous")
        expected_ari = (
            (1.0 - float(stored["waste"])) * float(stored["slca"])
            * (1.0 - float(stored["rho_outcome_environmental"]))
        )
        if not math.isclose(float(stored["ari"]), expected_ari, rel_tol=1e-12,
                            abs_tol=1e-12):
            raise RuntimeError(f"{path}:{index} ARI equation mismatch")
        if step_index > 0:
            for field in ("rho_policy_observed", "rho_outcome_environmental"):
                if float(stored[field]) + 1e-12 < float(records[-2][field]):
                    raise RuntimeError(f"{path}:{index} {field} is not monotone")

        if stored.get("mode") != mode or stored.get("scenario") != scenario:
            raise RuntimeError(f"{path}:{index} mode/scenario mismatch")
        if stored.get("simulated_dispatch_accounted") is not True:
            raise RuntimeError(f"{path}:{index} is not outcome-accounted")
        _validate_step_channel_evidence(
            stored["step_channel_evidence"], stored,
            where=f"{path}:{index} step_channel_evidence",
        )
        action_idx = stored.get("action_idx")
        if not isinstance(action_idx, int) or action_idx not in range(len(ACTIONS)):
            raise RuntimeError(f"{path}:{index} has an invalid action index")
        if stored.get("action") != ACTIONS[action_idx]:
            raise RuntimeError(f"{path}:{index} action label/index mismatch")
        if not _finite_vector(stored.get("probs"), 3):
            raise RuntimeError(f"{path}:{index} has invalid action probabilities")
        if abs(sum(float(value) for value in stored["probs"]) - 1.0) > 1e-8:
            raise RuntimeError(f"{path}:{index} probabilities do not sum to one")
        if not _finite_vector(stored.get("policy_probs_pre_override"), 3):
            raise RuntimeError(
                f"{path}:{index} has invalid pre-override probabilities"
            )
        preoverride_probs = [
            float(value) for value in stored["policy_probs_pre_override"]
        ]
        if abs(sum(preoverride_probs) - 1.0) > 1e-8:
            raise RuntimeError(
                f"{path}:{index} pre-override probabilities do not sum to one"
            )
        sampled_preoverride = stored.get("sampled_action_pre_override")
        if (
            not isinstance(sampled_preoverride, int)
            or sampled_preoverride not in range(len(ACTIONS))
        ):
            raise RuntimeError(
                f"{path}:{index} has invalid pre-override sampled action"
            )
        categorical_uniform = stored.get("policy_categorical_uniform")
        if mode == "static":
            if (
                categorical_uniform is not None
                or sampled_preoverride != 0
                or action_idx != 0
                or stored["probs"] != [1.0, 0.0, 0.0]
                or preoverride_probs != [1.0, 0.0, 0.0]
                or stored.get("governance_override") is not False
            ):
                raise RuntimeError(
                    f"{path}:{index} violates the fixed static policy"
                )
        else:
            try:
                categorical_uniform = float(categorical_uniform)
                expected_uniform = _policy_categorical_uniform(
                    policy_stream_seed, step_index,
                )
                if not math.isclose(
                    categorical_uniform, expected_uniform,
                    rel_tol=0.0, abs_tol=0.0,
                ):
                    raise ValueError(
                        "categorical uniform differs from the keyed policy stream"
                    )
                expected_sample = categorical_action_from_uniform(
                    np.asarray(preoverride_probs, dtype=float),
                    categorical_uniform,
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"{path}:{index} has an invalid categorical sampling record"
                ) from exc
            if sampled_preoverride != expected_sample:
                raise RuntimeError(
                    f"{path}:{index} sampled action is not bound to its uniform"
                )

        caps = capabilities_for(mode)
        try:
            validate_policy_record(
                stored,
                policy=episode_policy,
                policy_theta=expected_policy_theta,
                where=f"{path}:{index}",
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        if mode != "static" and not math.isclose(
            float(stored["policy_temperature"]),
            expected_policy_temperature,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise RuntimeError(
                f"{path}:{index} policy temperature differs from its "
                "environment stream"
            )
        role = str(stored["role"])
        if caps.policy_delta_learning:
            theta_snapshot = np.asarray(
                stored["effective_theta_delta"], dtype=float,
            )
            previous_theta = frozen_theta_delta_by_role.setdefault(
                role, theta_snapshot.copy(),
            )
            if not np.allclose(
                theta_snapshot, previous_theta, rtol=0.0, atol=1e-12,
            ):
                raise RuntimeError(
                    f"{path}:{index} policy-delta learner changed during "
                    "frozen evaluation"
                )
        if caps.reward_shaping_learning:
            shaping_snapshot = (
                np.asarray(stored["effective_slca_bonus_delta"], dtype=float),
                np.asarray(stored["effective_slca_rho_delta"], dtype=float),
                np.asarray(
                    stored["effective_no_slca_offset_delta"], dtype=float,
                ),
            )
            if frozen_reward_shaping is None:
                frozen_reward_shaping = tuple(
                    value.copy() for value in shaping_snapshot
                )
            elif any(
                not np.allclose(current, previous, rtol=0.0, atol=1e-12)
                for current, previous in zip(
                    shaping_snapshot, frozen_reward_shaping, strict=True,
                )
            ):
                raise RuntimeError(
                    f"{path}:{index} reward-shaping learner changed during "
                    "frozen evaluation"
                )
        if caps.context_matrix_learning:
            context_theta = np.asarray(
                stored["effective_context_theta"], dtype=float,
            )
            context_amp = float(stored["slca_amp"])
            if frozen_context_theta is None:
                frozen_context_theta = context_theta.copy()
                frozen_context_slca_amp = context_amp
            elif not np.allclose(
                context_theta, frozen_context_theta, rtol=0.0, atol=1e-12,
            ) or not math.isclose(
                context_amp, float(frozen_context_slca_amp),
                rel_tol=0.0, abs_tol=1e-12,
            ):
                raise RuntimeError(
                    f"{path}:{index} context learner changed during frozen "
                    "evaluation"
                )
        if caps.context_kind is None:
            if stored.get("governance_override") is not False:
                raise RuntimeError(
                    f"{path}:{index} non-context mode has a probability-gap override"
                )
            if max(
                abs(float(left) - float(right))
                for left, right in zip(
                    stored["probs"], preoverride_probs, strict=True,
                )
            ) > 1e-12:
                raise RuntimeError(
                    f"{path}:{index} non-context probabilities changed after sampling"
                )
            if action_idx != sampled_preoverride:
                raise RuntimeError(
                    f"{path}:{index} action is not the recorded categorical sample"
                )
            for field in (
                "context_counterfactual_action_idx",
                "context_counterfactual_action",
                "context_counterfactual_probs",
                "context_counterfactual_categorical_uniform",
                "context_counterfactual_sampled_action_pre_override",
                "context_action_changed",
            ):
                if stored.get(field) is not None:
                    raise RuntimeError(
                        f"{path}:{index} non-context mode fabricates {field}"
                    )
            if stored.get("context_influence_active") is not False or (
                stored.get("context_influence_counted") is not False
            ) or not math.isclose(
                float(stored["context_influence_threshold"]),
                0.10,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise RuntimeError(
                    f"{path}:{index} non-context diagnostics are not inert"
                )
        if not _finite_vector(stored.get("peer_message_bias"), 3):
            raise RuntimeError(f"{path}:{index} has invalid peer-message bias")
        if not caps.peer_messages:
            if any(
                abs(float(value)) > 1e-15
                for value in stored["peer_message_bias"]
            ):
                raise RuntimeError(
                    f"{path}:{index} no-peer arm has nonzero message bias"
                )

        if caps.context_kind is not None:
            for key, length in (
                ("phi", 10), ("peer_message_bias", 3), ("psi", 5),
                ("context_modifier", 3), ("base_logits", 3),
                ("post_context_logits_pre_override", 3),
                ("modifier_mcp", 3), ("modifier_pirag", 3),
                ("context_nonfeature_residual", 3),
                ("chosen_action_context_contributions", 5),
                ("context_counterfactual_probs", 3),
            ):
                if not _finite_vector(stored.get(key), length):
                    raise RuntimeError(
                        f"{path}:{index} has invalid instrumented field {key!r}"
                    )
            if not _finite_matrix(stored.get("effective_context_theta"), 3, 5):
                raise RuntimeError(
                    f"{path}:{index} has invalid effective context matrix"
                )
            if not _finite_matrix(
                stored.get("context_feature_contributions"), 3, 5,
            ):
                raise RuntimeError(
                    f"{path}:{index} has invalid context feature allocation"
                )
            if not _finite_matrix(
                stored.get("context_modifier_theta_jacobian"), 3, 5,
            ):
                raise RuntimeError(
                    f"{path}:{index} has invalid context learner Jacobian"
                )
            _validate_context_integration(
                stored.get("context_integration"),
                modifier=stored.get("context_modifier"),
                jacobian=stored.get("context_modifier_theta_jacobian"),
                theta=stored.get("effective_context_theta"),
                feature_contributions=stored.get(
                    "context_feature_contributions"
                ),
                residual=stored.get("context_nonfeature_residual"),
                expected_retrieval_kind=caps.retrieval_kind,
                where=f"{path}:{index}",
            )
            if stored.get("context_attribution_basis") != (
                "final_modifier_feature_allocation_plus_explicit_residual"
            ):
                raise RuntimeError(f"{path}:{index} has an invalid attribution basis")
            if stored.get("context_attribution_scope") not in {
                "primary_context", "cooperative_blend", "cooperative_veto",
            }:
                raise RuntimeError(f"{path}:{index} has an invalid attribution scope")
            if not isinstance(stored.get("retrieval_evidence_hashes"), list):
                raise RuntimeError(f"{path}:{index} has invalid evidence hashes")
            if any(
                not isinstance(value, str)
                or not re.fullmatch(r"[0-9a-f]{64}", value)
                for value in stored["retrieval_evidence_hashes"]
            ):
                raise RuntimeError(f"{path}:{index} has malformed evidence hashes")

            try:
                base = [float(value) for value in stored["base_logits"]]
                modifier = [float(value) for value in stored["context_modifier"]]
                shaping = [float(value) for value in stored["slca_shaping"]]
                amp = float(stored["slca_amp"])
                temperature = float(stored["policy_temperature"])
                observed_post = [
                    float(value)
                    for value in stored["post_context_logits_pre_override"]
                ]
                feature_allocation = [
                    [float(value) for value in row]
                    for row in stored["context_feature_contributions"]
                ]
                nonfeature_residual = [
                    float(value) for value in stored["context_nonfeature_residual"]
                ]
                chosen_contributions = [
                    float(value)
                    for value in stored["chosen_action_context_contributions"]
                ]
                chosen_residual = float(stored["chosen_action_context_residual"])
                fused_score = float(stored["retrieval_top_fused_score"])
                rerank_score = float(stored["retrieval_top_rerank_score"])
                legacy_score = float(stored["retrieval_top_score"])
                influence_threshold = float(stored["context_influence_threshold"])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"{path}:{index} has non-numeric policy reconstruction fields"
                ) from exc
            if not math.isfinite(amp) or not math.isfinite(temperature) or temperature <= 0:
                raise RuntimeError(f"{path}:{index} has invalid policy scaling")
            if not all(
                math.isfinite(value)
                for value in (
                    chosen_residual, fused_score, rerank_score, legacy_score,
                    influence_threshold,
                )
            ):
                raise RuntimeError(f"{path}:{index} has non-finite trace scalars")
            if not math.isclose(
                influence_threshold, 0.10, rel_tol=0.0, abs_tol=0.0,
            ):
                raise RuntimeError(
                    f"{path}:{index} context influence threshold is not the "
                    "locked 0.10"
                )
            if fused_score < 0.0 or legacy_score != fused_score:
                raise RuntimeError(
                    f"{path}:{index} confuses fused retrieval strength with ordering score"
                )

            reconstructed_modifier = [
                sum(feature_allocation[row]) + nonfeature_residual[row]
                for row in range(3)
            ]
            if max(
                abs(left - right)
                for left, right in zip(
                    reconstructed_modifier, modifier, strict=True,
                )
            ) > 1e-10:
                raise RuntimeError(
                    f"{path}:{index} final context attribution does not reconstruct modifier"
                )
            if max(
                abs(chosen_contributions[column] - feature_allocation[action_idx][column])
                for column in range(5)
            ) > 1e-10 or abs(
                chosen_residual - nonfeature_residual[action_idx]
            ) > 1e-10:
                raise RuntimeError(
                    f"{path}:{index} chosen-action attribution does not match full allocation"
                )
            max_feature = max(abs(value) for value in chosen_contributions)
            if abs(chosen_residual) > max_feature:
                expected_dominant_idx = None
                expected_component = "nonfeature_residual"
            else:
                expected_dominant_idx = max(
                    range(5), key=lambda column: abs(chosen_contributions[column])
                )
                expected_component = f"psi_{expected_dominant_idx}"
            if (
                stored.get("dominant_psi_idx") != expected_dominant_idx
                or stored.get("dominant_context_component") != expected_component
            ):
                raise RuntimeError(f"{path}:{index} dominant attribution is inconsistent")
            expected_dominant_action = max(
                range(3), key=lambda row: modifier[row]
            )
            if stored.get("dominant_action_idx") != expected_dominant_action:
                raise RuntimeError(f"{path}:{index} dominant action modifier is inconsistent")
            amplification = amp * min(abs(modifier[1]), 1.0)
            reconstructed_post = [
                (base[i] + modifier[i] + shaping[i] * amplification) / temperature
                for i in range(3)
            ]
            if max(
                abs(left - right)
                for left, right in zip(
                    reconstructed_post, observed_post, strict=True,
                )
            ) > 1e-10:
                raise RuntimeError(f"{path}:{index} policy-logit reconstruction mismatch")

            unoverridden_probs = _softmax(observed_post)
            if max(
                abs(float(left) - float(right))
                for left, right in zip(
                    preoverride_probs, unoverridden_probs, strict=True,
                )
            ) > 1e-10:
                raise RuntimeError(
                    f"{path}:{index} pre-override probability reconstruction mismatch"
                )
            expected_override = (
                unoverridden_probs[0] < GOVERNANCE_CC_PROB_CEILING
                and unoverridden_probs[1] - unoverridden_probs[0]
                > GOVERNANCE_LOCAL_ADVANTAGE_MIN
            )
            if bool(stored["governance_override"]) != expected_override:
                raise RuntimeError(f"{path}:{index} governance flag mismatch")
            expected_probs = [0.0, 1.0, 0.0] if expected_override else unoverridden_probs
            if max(
                abs(float(left) - float(right))
                for left, right in zip(
                    stored["probs"], expected_probs, strict=True,
                )
            ) > 1e-10:
                raise RuntimeError(f"{path}:{index} probability reconstruction mismatch")
            expected_live_action = 1 if expected_override else sampled_preoverride
            if action_idx != expected_live_action:
                raise RuntimeError(
                    f"{path}:{index} final action does not follow sampling/override"
                )

            counterfactual_probs = [
                float(value) for value in stored["context_counterfactual_probs"]
            ]
            if abs(sum(counterfactual_probs) - 1.0) > 1e-8:
                raise RuntimeError(
                    f"{path}:{index} context-ablation probabilities do not sum to one"
                )
            expected_counterfactual_probs = _softmax(
                [value / temperature for value in base]
            )
            if max(
                abs(left - right)
                for left, right in zip(
                    counterfactual_probs, expected_counterfactual_probs,
                    strict=True,
                )
            ) > 1e-10:
                raise RuntimeError(
                    f"{path}:{index} context-ablation probability reconstruction mismatch"
                )
            counterfactual_idx = stored.get("context_counterfactual_action_idx")
            if not isinstance(counterfactual_idx, int) or counterfactual_idx not in range(3):
                raise RuntimeError(f"{path}:{index} has invalid context-ablation action")
            if stored.get("context_counterfactual_action") != ACTIONS[counterfactual_idx]:
                raise RuntimeError(
                    f"{path}:{index} context-ablation action label/index mismatch"
                )
            counterfactual_uniform = stored.get(
                "context_counterfactual_categorical_uniform"
            )
            if not math.isclose(
                float(counterfactual_uniform), float(categorical_uniform),
                rel_tol=0.0, abs_tol=0.0,
            ):
                raise RuntimeError(
                    f"{path}:{index} context ablation used a different categorical uniform"
                )
            expected_counterfactual_sample = categorical_action_from_uniform(
                np.asarray(counterfactual_probs, dtype=float),
                float(categorical_uniform),
            )
            if (
                stored.get(
                    "context_counterfactual_sampled_action_pre_override"
                ) != expected_counterfactual_sample
                or counterfactual_idx != expected_counterfactual_sample
            ):
                raise RuntimeError(
                    f"{path}:{index} context-ablation action is not bound to the paired uniform"
                )
            expected_changed = action_idx != counterfactual_idx
            if stored.get("context_action_changed") is not expected_changed:
                raise RuntimeError(f"{path}:{index} context action-change flag mismatch")
            expected_active = max(abs(value) for value in modifier) > influence_threshold
            if stored.get("context_influence_active") is not expected_active:
                raise RuntimeError(f"{path}:{index} context-influence active flag mismatch")
            if stored.get("context_influence_counted") is not (
                expected_active and expected_changed
            ):
                raise RuntimeError(f"{path}:{index} context-influence count flag mismatch")

    caps = capabilities_for(mode)
    spoilage_estimator = metadata.get("spoilage_estimator")
    if not isinstance(spoilage_estimator, dict):
        raise RuntimeError(f"{path} lacks frozen spoilage-estimator provenance")
    expected_estimator_kind = (
        "mechanistic_plus_frozen_synthetic_pinn_residual"
        if caps.spoilage_residual else "mechanistic_only_no_pinn"
    )
    if spoilage_estimator.get("kind") != expected_estimator_kind:
        raise RuntimeError(
            f"{path} spoilage estimator conflicts with mode {mode!r}"
        )
    latent_spoilage_model = metadata.get("latent_spoilage_model")
    if not isinstance(latent_spoilage_model, dict):
        raise RuntimeError(f"{path} lacks independent latent-DGP provenance")
    try:
        validate_recorded_spoilage_trajectories(
            records,
            outcome_contract,
            spoilage_estimator=spoilage_estimator,
            latent_spoilage_model=latent_spoilage_model,
            where=str(path),
            contract_validated=True,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    # Every real publication ledger spans the full 288-step episode. Focused
    # unit fixtures deliberately shorten the horizon and cannot traverse every
    # supply-chain stage; the production constant remains 288.
    # Bind the guard to the artifact itself, not the mutable test inventory
    # constant.  A genuine 288-row ledger can never bypass source replay by
    # changing process-local validator configuration.
    full_publication_horizon = len(records) == 288
    if full_publication_horizon and caps.policy_delta_learning and set(frozen_theta_delta_by_role) != set(
        DECISION_OWNER_ROLES
    ):
        raise RuntimeError(
            f"{path} retained episode does not expose every decision-owner "
            "policy snapshot"
        )
    if full_publication_horizon and caps.peer_messages and not any(
        any(abs(float(value)) > 1e-15 for value in record["peer_message_bias"])
        for record in records
    ):
        raise RuntimeError(
            f"{path} peer-enabled arm has no decision-level peer exposure"
        )
    source_replay_hashes: dict[str, str] | None = None
    if full_publication_horizon:
        source_replay_hashes = _validate_source_bound_episode_replay(
            records,
            metadata,
            path=path,
            mode=mode,
            scenario=scenario,
            benchmark_seed=benchmark_seed,
            policy=episode_policy,
            stochastic_layer=source_replay_layer,
            scenario_frame=expected_scenario_frame,
        )

    claimed_root = header.get("merkle_root")
    if not isinstance(claimed_root, str) or not re.fullmatch(
        r"[0-9a-f]{64}", claimed_root,
    ):
        raise RuntimeError(f"{path} has an invalid Merkle root")
    actual_root = _merkle_root(leaves)
    if actual_root != claimed_root:
        raise RuntimeError(f"{path} Merkle root mismatch")

    latent_payload = {
        "hours": [float(record["hour"]) for record in records],
        "temp_outcome_environmental": [
            float(record["temp_outcome_environmental"]) for record in records
        ],
        "rh_outcome_environmental": [
            float(record["rh_outcome_environmental"]) for record in records
        ],
        "rho_outcome_environmental": [
            float(record["rho_outcome_environmental"]) for record in records
        ],
        "inventory_outcome_environmental": [
            float(record["inventory_outcome_environmental"]) for record in records
        ],
        "demand_outcome_environmental": [
            float(record["demand_outcome_environmental"]) for record in records
        ],
        "transport_multiplier_outcome_environmental": [
            float(record["transport_multiplier_outcome_environmental"])
            for record in records
        ],
        "effective_k_ref": float(metadata["effective_k_ref"]),
        "effective_Ea_R": float(metadata["effective_Ea_R"]),
        "scenario_onset_offset_hours": float(
            metadata["scenario_onset_offset_hours"]
        ),
    }
    observed_payload = {
        "hours": [float(record["hour"]) for record in records],
        "temp_policy_observed": [
            float(record["temp_policy_observed"]) for record in records
        ],
        "rh_policy_observed": [
            float(record["rh_policy_observed"]) for record in records
        ],
        "rho_policy_observed": [
            float(record["rho_policy_observed"]) for record in records
        ],
        "inventory_policy_observed": [
            float(record["inventory_policy_observed"]) for record in records
        ],
        "demand_forecast_policy_observed": [
            float(record["demand_forecast_policy_observed"]) for record in records
        ],
        "supply_forecast_policy_observed": [
            float(record["supply_forecast_policy_observed"]) for record in records
        ],
    }
    demand_payload = {
        "hours": [float(record["hour"]) for record in records],
        "demand_policy_observed": [
            float(record["demand_policy_observed"]) for record in records
        ],
        "demand_forecast_policy_observed": [
            float(record["demand_forecast_policy_observed"])
            for record in records
        ],
        "demand_regime_flag": [
            float(record["bollinger_regime_flag"]) for record in records
        ],
        "price_signal": [
            float(record["price_signal"]) for record in records
        ],
    }
    if _canonical_sha256(latent_payload) != metadata["latent_environment_sha256"]:
        raise RuntimeError(f"{path} latent-state hash does not match full records")
    if _canonical_sha256(observed_payload) != metadata["observed_policy_input_sha256"]:
        raise RuntimeError(f"{path} observed-state hash does not match full records")
    if _canonical_sha256(demand_payload) != metadata["demand_observation_sha256"]:
        raise RuntimeError(f"{path} demand-stream hash does not match full records")
    step_ari = [float(record["ari"]) for record in records]
    step_waste = [float(record["waste"]) for record in records]
    step_slca = [float(record["slca"]) for record in records]
    policy = episode_policy
    violation_flags = [
        (
            float(record["temp_outcome_environmental"]) > policy.max_temp_c
            or (1.0 - float(record["rho_outcome_environmental"]))
            < policy.min_shelf_expedite
        )
        for record in records
    ]
    operating_envelope_flags = [
        not bool(check_compliance(
            temperature=float(record["temp_outcome_environmental"]),
            humidity=float(record["rh_outcome_environmental"]),
        ).get("compliant", True))
        for record in records
    ]
    violation_actions = [
        str(record["action"])
        for record, violated in zip(records, violation_flags, strict=True)
        if violated
    ]
    violation_count = len(violation_actions)
    headline_metrics = {
        "ari": float(math.fsum(step_ari) / len(step_ari)),
        "waste": float(math.fsum(step_waste) / len(step_waste)),
        "slca": float(math.fsum(step_slca) / len(step_slca)),
        "carbon": float(math.fsum(
            float(record["carbon_kg"]) for record in records
        )),
        "equity": float(compute_equity(step_slca)),
        "rle": float(compute_rle(
            [float(record["rho_outcome_environmental"]) for record in records],
            [str(record["action"]) for record in records],
        )),
        "constraint_violation_rate": float(
            sum(violation_flags) / len(records)
        ),
        "operational_violation_rate": float(
            sum(violation_flags) / len(records)
        ),
        "compliance_violation_rate": float(
            sum(operating_envelope_flags) / len(records)
        ),
        "operating_envelope_violation_rate": float(
            sum(operating_envelope_flags) / len(records)
        ),
        "regulatory_violation_rate": float(
            sum(operating_envelope_flags) / len(records)
        ),
        "violation_event_count": int(violation_count),
        "downstream_violation_rate": float(
            violation_actions.count("cold_chain") / violation_count
            if violation_count else 0.0
        ),
        "redistribute_violation_rate": float(
            violation_actions.count("local_redistribute") / violation_count
            if violation_count else 0.0
        ),
        "contained_violation_rate": float(
            violation_actions.count("recovery") / violation_count
            if violation_count else 0.0
        ),
    }
    try:
        episode_evidence = reconstruct_episode_evidence(
            records,
            episode_evidence_contract,
            where=str(path),
            contract_validated=True,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    return {
        "latent_environment_sha256": str(
            metadata["latent_environment_sha256"]
        ),
        "spoilage_estimator": dict(spoilage_estimator),
        "latent_spoilage_model": dict(latent_spoilage_model),
        "headline_metrics": headline_metrics,
        "episode_evidence": episode_evidence,
        "source_replay_hashes": source_replay_hashes,
        "trace_binding": (
            _build_trace_binding(records, reconstructed_outcomes)
            if mode in TRACE_MODES else None
        ),
        "learner_snapshots": {
            "theta_delta_by_role": {
                role: value.tolist()
                for role, value in frozen_theta_delta_by_role.items()
            },
            "reward_shaping": (
                {
                    "slca_bonus_delta": frozen_reward_shaping[0].tolist(),
                    "slca_rho_delta": frozen_reward_shaping[1].tolist(),
                    "no_slca_offset_delta": frozen_reward_shaping[2].tolist(),
                }
                if frozen_reward_shaping is not None else None
            ),
            "context_theta": (
                frozen_context_theta.tolist()
                if frozen_context_theta is not None else None
            ),
            "context_slca_amp": frozen_context_slca_amp,
        },
    }


def _build_trace_binding(
    records: list[dict[str, Any]],
    reconstructed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the exact unrounded trace payload implied by one ledger."""

    slca_values = [float(record["slca"]) for record in records]
    equity_trace: list[float] = []
    for index in range(len(records)):
        window = slca_values[max(0, index - 23):index + 1]
        equity_trace.append(float(compute_equity(window)) if len(window) > 1 else 1.0)
    binding = {
        "ari_trace": [float(record["ari"]) for record in records],
        "waste_trace": [float(record["waste"]) for record in records],
        "rho_trace": [float(record["rho_policy_observed"]) for record in records],
        "rho_policy_observed_trace": [
            float(record["rho_policy_observed"]) for record in records
        ],
        "rho_outcome_environmental_trace": [
            float(record["rho_outcome_environmental"]) for record in records
        ],
        "action_trace": [int(record["action_idx"]) for record in records],
        "prob_trace": [
            [float(value) for value in record["probs"]] for record in records
        ],
        "carbon_trace": [float(record["carbon_kg"]) for record in records],
        "hours": [float(record["hour"]) for record in records],
        "temp_trace": [float(record["temp_policy_observed"]) for record in records],
        "rh_trace": [float(record["rh_policy_observed"]) for record in records],
        "inventory_trace": [
            float(record["inventory_policy_observed"]) for record in records
        ],
        "demand_trace": [
            float(record["demand_forecast_policy_observed"]) for record in records
        ],
        "temp_policy_observed_trace": [
            float(record["temp_policy_observed"]) for record in records
        ],
        "temp_outcome_environmental_trace": [
            float(record["temp_outcome_environmental"]) for record in records
        ],
        "rh_policy_observed_trace": [
            float(record["rh_policy_observed"]) for record in records
        ],
        "rh_outcome_environmental_trace": [
            float(record["rh_outcome_environmental"]) for record in records
        ],
        "inventory_policy_observed_trace": [
            float(record["inventory_policy_observed"]) for record in records
        ],
        "inventory_outcome_environmental_trace": [
            float(record["inventory_outcome_environmental"]) for record in records
        ],
        "demand_policy_observed_trace": [
            float(record["demand_policy_observed"]) for record in records
        ],
        "demand_forecast_policy_observed_trace": [
            float(record["demand_forecast_policy_observed"]) for record in records
        ],
        "demand_regime_flag_trace": [
            float(record["bollinger_regime_flag"]) for record in records
        ],
        "price_signal_trace": [float(record["price_signal"]) for record in records],
        "supply_forecast_policy_observed_trace": [
            float(record["supply_forecast_policy_observed"]) for record in records
        ],
        "demand_outcome_environmental_trace": [
            float(record["demand_outcome_environmental"]) for record in records
        ],
        "transport_multiplier_outcome_environmental_trace": [
            float(record["transport_multiplier_outcome_environmental"])
            for record in records
        ],
        "simulated_dispatch_accounted_trace": [
            record["simulated_dispatch_accounted"] for record in records
        ],
        "slca_component_trace": [
            row["slca_component_trace"] for row in reconstructed
        ],
        "slca_trace": slca_values,
        "equity_trace": equity_trace,
        "reward_trace": [float(record["reward"]) for record in records],
    }
    if set(binding) != set(TRACE_FIELDS):
        raise RuntimeError("internal ledger-to-trace binding schema drift")
    return binding


def _compare_trace_cache_value(observed: Any, expected: Any, *, where: str) -> None:
    """Compare a four-decimal cache value to the ledger-derived full precision."""

    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise RuntimeError(f"{where} object schema differs from its ledger")
        for key in expected:
            _compare_trace_cache_value(
                observed[key], expected[key], where=f"{where}/{key}",
            )
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise RuntimeError(f"{where} list shape differs from its ledger")
        for index, (left, right) in enumerate(zip(observed, expected, strict=True)):
            _compare_trace_cache_value(left, right, where=f"{where}[{index}]")
        return
    if isinstance(expected, bool) or isinstance(expected, str) or expected is None:
        if observed != expected:
            raise RuntimeError(f"{where} differs from its ledger")
        return
    if isinstance(expected, int) and not isinstance(expected, bool):
        if isinstance(observed, bool) or observed != expected:
            raise RuntimeError(f"{where} differs from its ledger")
        return
    try:
        cached = float(observed)
        ledger_value = float(expected)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{where} is not numeric") from exc
    if not math.isfinite(cached) or not math.isclose(
        cached, ledger_value, rel_tol=0.0, abs_tol=5.0000001e-5,
    ):
        raise RuntimeError(
            f"{where} differs from its ledger: cached={cached!r}, "
            f"ledger={ledger_value!r}"
        )


def _compare_seed_evidence_value(
    observed: Any, expected: Any, *, where: str,
) -> None:
    """Compare unrounded seed-envelope scalars to ledger reconstruction."""

    if isinstance(expected, dict):
        observed_keys = (
            {str(key): value for key, value in observed.items()}
            if isinstance(observed, dict) else {}
        )
        expected_keys = {str(key): value for key, value in expected.items()}
        if not isinstance(observed, dict) or set(observed_keys) != set(
            expected_keys
        ):
            raise RuntimeError(f"{where} object schema differs from its ledger")
        for key in expected_keys:
            _compare_seed_evidence_value(
                observed_keys[key], expected_keys[key], where=f"{where}/{key}",
            )
        return
    if isinstance(expected, bool) or isinstance(expected, str):
        if observed != expected:
            raise RuntimeError(f"{where} differs from its ledger")
        return
    if isinstance(expected, int):
        if isinstance(observed, bool) or observed != expected:
            raise RuntimeError(f"{where} differs from its ledger")
        return
    try:
        value = float(observed)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{where} is not numeric") from exc
    if not math.isfinite(value) or not math.isclose(
        value, float(expected), rel_tol=1e-12, abs_tol=1e-12,
    ):
        raise RuntimeError(
            f"{where} differs from its ledger: stored={value!r}, "
            f"reconstructed={expected!r}"
        )


def validate_learner_snapshot_binding(
    cell: dict[str, Any],
    snapshots: dict[str, Any],
    *,
    mode: str,
    where: str,
) -> None:
    """Bind frozen decision-level parameters to the retained learner summaries."""

    caps = capabilities_for(mode)
    tolerance = 5.1e-12  # seed envelopes round nested diagnostics to 12 places

    def assert_array(left: Any, right: Any, *, label: str) -> None:
        try:
            observed = np.asarray(left, dtype=float)
            expected = np.asarray(right, dtype=float)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{where}/{label} is not numeric") from exc
        if observed.shape != expected.shape or not np.all(np.isfinite(observed)) or (
            not np.allclose(observed, expected, rtol=0.0, atol=tolerance)
        ):
            raise RuntimeError(
                f"{where}/{label} differs from the frozen decision ledger"
            )

    theta_snapshots = snapshots.get("theta_delta_by_role")
    theta_summary = cell.get("theta_learner_summary")
    if caps.policy_delta_learning:
        if not isinstance(theta_snapshots, dict) or set(theta_snapshots) != set(
            DECISION_OWNER_ROLES
        ) or not isinstance(theta_summary, dict):
            raise RuntimeError(f"{where} lacks policy-delta snapshot binding")
        per_role = theta_summary.get("per_role")
        if not isinstance(per_role, dict) or set(per_role) != set(
            DECISION_OWNER_ROLES
        ):
            raise RuntimeError(f"{where} lacks policy-delta role summaries")
        for role in DECISION_OWNER_ROLES:
            assert_array(
                per_role[role].get("final_theta_delta"),
                theta_snapshots[role],
                label=f"theta_learner_summary/per_role/{role}/final_theta_delta",
            )
    elif theta_snapshots not in ({}, None) or theta_summary not in (None, {}):
        raise RuntimeError(f"{where} exposes an undeclared policy-delta learner")

    reward_snapshot = snapshots.get("reward_shaping")
    reward_summary = cell.get("reward_shaping_learner_summary")
    if caps.reward_shaping_learning:
        if not isinstance(reward_snapshot, dict) or not isinstance(
            reward_summary, dict
        ):
            raise RuntimeError(f"{where} lacks reward-shaping snapshot binding")
        for field in (
            "slca_bonus_delta", "slca_rho_delta", "no_slca_offset_delta",
        ):
            assert_array(
                reward_summary.get(field), reward_snapshot.get(field),
                label=f"reward_shaping_learner_summary/{field}",
            )
    elif reward_snapshot is not None or reward_summary not in (None, {}):
        raise RuntimeError(f"{where} exposes undeclared reward shaping")

    context_snapshot = snapshots.get("context_theta")
    context_amp = snapshots.get("context_slca_amp")
    context_summary = cell.get("learner_summary")
    if caps.context_matrix_learning:
        if not isinstance(context_summary, dict) or context_snapshot is None:
            raise RuntimeError(f"{where} lacks context-learner snapshot binding")
        assert_array(
            context_summary.get("final_theta"), context_snapshot,
            label="learner_summary/final_theta",
        )
        try:
            summary_amp = float(context_summary["final_slca_amp"])
            ledger_amp = float(context_amp)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"{where} lacks context amplification binding") from exc
        if not math.isclose(
            summary_amp, ledger_amp, rel_tol=0.0, abs_tol=tolerance,
        ):
            raise RuntimeError(
                f"{where}/learner_summary/final_slca_amp differs from the ledger"
            )
    elif context_snapshot is not None or context_amp is not None:
        raise RuntimeError(f"{where} exposes an undeclared context learner")


def _validate_seed_headlines(
    seed_root: Path,
    ledger_summaries: dict[tuple[int, str, str], dict[str, Any]],
) -> None:
    """Bind every seed-envelope headline to its retained decision ledger."""
    expected_files = {f"seed_{seed}.json" for seed in EXPECTED_SEEDS}
    observed_files = {path.name for path in seed_root.glob("seed_*.json")}
    if observed_files != expected_files:
        raise RuntimeError(
            "seed-envelope inventory mismatch for ledger headline audit: "
            f"missing={sorted(expected_files - observed_files)}, "
            f"unexpected={sorted(observed_files - expected_files)}"
        )
    for seed in EXPECTED_SEEDS:
        path = seed_root / f"seed_{seed}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"invalid seed envelope: {path}") from exc
        if payload.get("seed") != seed:
            raise RuntimeError(f"{path} seed identity mismatch")
        scenarios = payload.get("scenarios")
        if not isinstance(scenarios, dict) or set(scenarios) != set(SCENARIOS):
            raise RuntimeError(f"{path} scenario inventory mismatch")
        traces = payload.get("traces")
        if not isinstance(traces, dict) or set(traces) != set(SCENARIOS):
            raise RuntimeError(f"{path} trace scenario inventory mismatch")
        for scenario in SCENARIOS:
            modes = scenarios.get(scenario)
            if not isinstance(modes, dict) or set(modes) != set(MODES):
                raise RuntimeError(
                    f"{path} mode inventory mismatch for {scenario}"
                )
            for mode in MODES:
                stored = modes[mode]
                if not isinstance(stored, dict):
                    raise RuntimeError(
                        f"{path} has invalid headline cell {scenario}/{mode}"
                    )
                recomputed = ledger_summaries[(seed, scenario, mode)][
                    "headline_metrics"
                ]
                for metric, expected in recomputed.items():
                    try:
                        observed = float(stored[metric])
                    except (KeyError, TypeError, ValueError) as exc:
                        raise RuntimeError(
                            f"{path} lacks numeric {scenario}/{mode}/{metric}"
                        ) from exc
                    if not math.isclose(
                        observed, expected, rel_tol=1e-10, abs_tol=1e-12,
                    ):
                        raise RuntimeError(
                            f"{path} headline {scenario}/{mode}/{metric} "
                            f"does not match the retained decision ledger: "
                            f"stored={observed!r}, recomputed={expected!r}"
                        )
                for field, expected in ledger_summaries[(
                    seed, scenario, mode,
                )]["episode_evidence"].items():
                    _compare_seed_evidence_value(
                        stored.get(field), expected,
                        where=f"{path}:{scenario}/{mode}/{field}",
                    )
                validate_learner_snapshot_binding(
                    stored,
                    ledger_summaries[(seed, scenario, mode)][
                        "learner_snapshots"
                    ],
                    mode=mode,
                    where=f"{path}:{scenario}/{mode}",
                )
                if (
                    stored.get("mean_decision_latency_ms_descriptive_only")
                    is not True
                    or stored.get("latency_penalty_usd_descriptive_only")
                    is not True
                ):
                    raise RuntimeError(
                        f"{path}:{scenario}/{mode} mislabels measured latency"
                    )
            expected_trace_modes = {
                trace_mode for trace_mode in TRACE_MODES if trace_mode in MODES
            }
            trace_modes = traces.get(scenario)
            if (
                not isinstance(trace_modes, dict)
                or set(trace_modes) != expected_trace_modes
            ):
                raise RuntimeError(
                    f"{path} trace mode inventory mismatch for {scenario}"
                )
            for trace_mode in expected_trace_modes:
                trace_cell = trace_modes[trace_mode]
                expected_trace = ledger_summaries[(
                    seed, scenario, trace_mode,
                )]["trace_binding"]
                if expected_trace is None:
                    raise RuntimeError(
                        f"{path} lacks ledger trace binding for {scenario}/{trace_mode}"
                    )
                _compare_trace_cache_value(
                    trace_cell,
                    expected_trace,
                    where=f"{path}:{scenario}/{trace_mode}",
                )


def validate_inventory(
    ledger_root: Path, seed_root: Path | None = None,
) -> None:
    expected_seed_dirs = {f"seed_{seed}" for seed in EXPECTED_SEEDS}
    found_seed_dirs = {path.name for path in ledger_root.iterdir() if path.is_dir()}
    if found_seed_dirs != expected_seed_dirs:
        raise RuntimeError(
            "ledger seed inventory mismatch: "
            f"missing={sorted(expected_seed_dirs - found_seed_dirs)}, "
            f"unexpected={sorted(found_seed_dirs - expected_seed_dirs)}"
        )
    stray = [
        path for path in ledger_root.rglob("*.jsonl")
        if path.parent.parent != ledger_root
    ]
    if stray:
        raise RuntimeError(f"nested or misplaced ledger files found: {stray[:3]}")

    expected_names = {
        f"{mode}__{scenario}.jsonl"
        for mode in MODES for scenario in SCENARIOS
    }
    ledger_summaries: dict[tuple[int, str, str], dict[str, Any]] = {}
    for seed in EXPECTED_SEEDS:
        seed_dir = ledger_root / f"seed_{seed}"
        found_names = {path.name for path in seed_dir.glob("*.jsonl")}
        if found_names != expected_names:
            raise RuntimeError(
                f"ledger inventory mismatch for seed {seed}: "
                f"missing={sorted(expected_names - found_names)}, "
                f"unexpected={sorted(found_names - expected_names)}"
            )
        latent_hashes: dict[str, set[str]] = {
            scenario: set() for scenario in SCENARIOS
        }
        for mode in MODES:
            for scenario in SCENARIOS:
                ledger_summary = validate_ledger(
                    seed_dir / f"{mode}__{scenario}.jsonl",
                    mode=mode, scenario=scenario, benchmark_seed=seed,
                    expected_outcome_equation_contract=(
                        expected_publication_outcome_equation_contract(
                            benchmark_seed=seed,
                            scenario=scenario,
                        )
                    ),
                )
                ledger_summaries[(seed, scenario, mode)] = ledger_summary
                latent_hashes[scenario].add(
                    ledger_summary["latent_environment_sha256"]
                )
        for scenario, hashes in latent_hashes.items():
            if len(hashes) != 1:
                raise RuntimeError(
                    f"seed {seed}/{scenario} ledgers do not share latent truth"
                )
    if seed_root is not None:
        if not seed_root.is_dir():
            raise RuntimeError(f"seed-envelope root does not exist: {seed_root}")
        _validate_seed_headlines(seed_root, ledger_summaries)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument(
        "--seed-root", type=Path, required=True,
        help="Run-scoped seed envelopes whose headlines must match ledgers.",
    )
    args = parser.parse_args(argv)
    if not args.ledger_root.is_dir():
        raise RuntimeError(f"ledger root does not exist: {args.ledger_root}")
    validate_inventory(args.ledger_root, args.seed_root)
    total = len(EXPECTED_SEEDS) * len(MODES) * len(SCENARIOS)
    print(
        f"[PASS] exact ledger inventory + JSONL/Merkle integrity: "
        f"{total} ledgers, {total * EXPECTED_RECORDS} decisions, "
        f"all paper-facing recomputable endpoints bound to seed envelopes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
