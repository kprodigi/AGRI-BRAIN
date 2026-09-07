"""Executable contract for paper-facing episode activity diagnostics.

The publication seed envelopes contain several episode-level values that are
not outcome equations: measured decision latency, context honor/influence,
MCP execution counts, message activity, and activity-based Green-AI
estimates.  This module makes those values deterministic functions of
Merkle-covered per-decision evidence and a locked measurement contract.

Wall-clock latency remains a descriptive measurement rather than a simulated
or inferential endpoint.  Reconstruction proves that every reported aggregate
is the aggregate of the retained raw timings; it cannot turn a software timer
into independent hardware telemetry.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .action_selection import ACTIONS
from .footprint import (
    DEFAULT_ASSUMED_ACTIVE_POWER_W,
    DEFAULT_ENERGY_PER_PROXY_STEP_J,
    DEFAULT_WATER_PER_PROXY_STEP_L,
    DEFAULT_WATER_RATE_L_PER_SERVER_SECOND,
)


EPISODE_EVIDENCE_CONTRACT_VERSION = 2
EPISODE_EVIDENCE_CONTRACT_TYPE = "agribrain_publication_episode_activity"
DEFAULT_MEASUREMENT_SCOPE = (
    "coordinator.step action-selection wall time only; excludes scenario "
    "construction, forecast preparation, outcome scoring, learner post-step "
    "updates, artifact I/O, and idle allocation"
)
DEFAULT_PROXY_STEP_UNIT = "standardized routing opportunity"
DEFAULT_CONTEXT_INFLUENCE_THRESHOLD = 0.10
DEFAULT_CONTEXT_SENSITIVITY_THRESHOLDS = (0.05, 0.10, 0.15, 0.20)

PROTOCOL_STEP_FIELDS = (
    "protocol_interaction_count_step",
    "protocol_tools_call_count_step",
    "protocol_prompts_get_count_step",
    "protocol_jsonrpc_error_count_step",
    "protocol_tool_iserror_count_step",
    "protocol_real_tool_iserror_count_step",
    "protocol_error_count_step",
    "protocol_dropped_interaction_count_step",
)
ACTIVITY_STEP_FIELDS = (
    "decision_latency_ms",
    "mcp_tool_call_count_step",
    "pirag_query_count_step",
    "dispatcher_tool_failure_count_step",
    "inter_agent_message_count_step",
    *PROTOCOL_STEP_FIELDS,
)


def build_episode_evidence_contract(
    *,
    measurement_scope: str = DEFAULT_MEASUREMENT_SCOPE,
    proxy_step_unit: str = DEFAULT_PROXY_STEP_UNIT,
    influence_threshold: float = DEFAULT_CONTEXT_INFLUENCE_THRESHOLD,
    assumed_active_power_w: float = DEFAULT_ASSUMED_ACTIVE_POWER_W,
    water_rate_l_per_server_second: float = (
        DEFAULT_WATER_RATE_L_PER_SERVER_SECOND
    ),
    energy_per_step_proxy_j: float = DEFAULT_ENERGY_PER_PROXY_STEP_J,
    water_per_step_proxy_l: float = DEFAULT_WATER_PER_PROXY_STEP_L,
) -> dict[str, Any]:
    """Return the exact episode-activity aggregation/measurement contract."""

    contract = {
        "schema_version": EPISODE_EVIDENCE_CONTRACT_VERSION,
        "contract_type": EPISODE_EVIDENCE_CONTRACT_TYPE,
        "latency": {
            "step_field": "decision_latency_ms",
            "unit": "ms",
            "clock": "python_time.perf_counter_elapsed",
            "measurement_scope": str(measurement_scope),
            "descriptive_only": True,
            "mean_equation": "arithmetic_mean(decision_latency_ms)",
            "p95_method": "linear_percentile_rank_0.95_times_n_minus_1",
            "penalty_equation": "sum(max(latency_ms-50,0))*0.0002",
        },
        "context": {
            "influence_threshold": float(influence_threshold),
            "sensitivity_thresholds": [
                float(value) for value in DEFAULT_CONTEXT_SENSITIVITY_THRESHOLDS
            ],
            "dispatch_attempt_equation": "context_modifier_is_a_3_vector",
            "active_equation": "max(abs(context_modifier))>influence_threshold",
            "honor_equation": "active_and_argmax(context_modifier)==action_idx",
            "influence_equation": "active_and_live_action_differs_from_paired_ablation",
        },
        "execution_activity": {
            "mcp_tool_call_step_field": "mcp_tool_call_count_step",
            "primary_mcp_list_step_field": (
                "primary_mcp_tools_invoked_step"
            ),
            "cooperative_mcp_list_step_field": (
                "cooperative_mcp_tools_invoked_step"
            ),
            "pirag_query_step_field": "pirag_query_count_step",
            "primary_pirag_attempt_step_field": (
                "primary_pirag_query_attempted_step"
            ),
            "cooperative_pirag_attempt_step_field": (
                "cooperative_pirag_query_attempted_step"
            ),
            "dispatcher_failure_step_field": (
                "dispatcher_tool_failure_count_step"
            ),
            "message_step_field": "inter_agent_message_count_step",
            "protocol_step_fields": list(PROTOCOL_STEP_FIELDS),
            "aggregation": "integer_sum_over_retained_decisions",
        },
        "footprint": {
            "estimate_basis": "measured_elapsed_seconds_x_declared_rates",
            "measurement_scope": str(measurement_scope),
            "proxy_step_unit": str(proxy_step_unit),
            "estimation_status": (
                "activity-based estimate; not hardware telemetry"
            ),
            "assumed_active_power_W": float(assumed_active_power_w),
            "water_rate_L_per_server_second": float(
                water_rate_l_per_server_second
            ),
            "energy_per_step_proxy_J": float(energy_per_step_proxy_j),
            "water_per_step_proxy_L": float(water_per_step_proxy_l),
            "energy_equation": "sum(latency_ms)/1000*assumed_active_power_W",
            "water_equation": (
                "sum(latency_ms)/1000*water_rate_L_per_server_second"
            ),
        },
    }
    validate_episode_evidence_contract(contract)
    return contract


def _finite(value: Any, *, where: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{where} must be a finite number, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{where} must be a finite number")
    return result


def _exact_keys(value: Any, keys: set[str], *, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        observed = set(value) if isinstance(value, Mapping) else set()
        raise ValueError(
            f"{where} schema mismatch: missing={sorted(keys - observed)}, "
            f"unexpected={sorted(observed - keys)}"
        )
    return value


def _compare_values(observed: Any, expected: Any, *, where: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or set(observed) != set(expected):
            raise ValueError(f"{where} differs from the expected contract")
        for key in expected:
            _compare_values(observed[key], expected[key], where=f"{where}/{key}")
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise ValueError(f"{where} differs from the expected contract")
        for index, (left, right) in enumerate(zip(observed, expected, strict=True)):
            _compare_values(left, right, where=f"{where}[{index}]")
        return
    if isinstance(expected, float):
        actual = _finite(observed, where=where)
        if not math.isclose(actual, expected, rel_tol=1e-15, abs_tol=1e-15):
            raise ValueError(f"{where}={actual!r}, expected {expected!r}")
        return
    if observed != expected:
        raise ValueError(f"{where}={observed!r}, expected {expected!r}")


def validate_episode_evidence_contract(
    contract: Any,
    *,
    where: str = "episode_evidence_contract",
    expected_contract: Mapping[str, Any] | None = None,
) -> None:
    """Fail closed on a malformed or substituted activity contract."""

    top = _exact_keys(contract, {
        "schema_version", "contract_type", "latency", "context",
        "execution_activity", "footprint",
    }, where=where)
    if top["schema_version"] != EPISODE_EVIDENCE_CONTRACT_VERSION or (
        top["contract_type"] != EPISODE_EVIDENCE_CONTRACT_TYPE
    ):
        raise ValueError(f"{where} has an unsupported identity")

    latency = _exact_keys(top["latency"], {
        "step_field", "unit", "clock", "measurement_scope",
        "descriptive_only", "mean_equation", "p95_method",
        "penalty_equation",
    }, where=f"{where}/latency")
    if (
        latency["step_field"] != "decision_latency_ms"
        or latency["unit"] != "ms"
        or latency["clock"] != "python_time.perf_counter_elapsed"
        or latency["descriptive_only"] is not True
        or latency["mean_equation"]
        != "arithmetic_mean(decision_latency_ms)"
        or latency["p95_method"]
        != "linear_percentile_rank_0.95_times_n_minus_1"
        or latency["penalty_equation"]
        != "sum(max(latency_ms-50,0))*0.0002"
        or not isinstance(latency["measurement_scope"], str)
        or not latency["measurement_scope"]
    ):
        raise ValueError(f"{where} changes the locked latency semantics")

    context = _exact_keys(top["context"], {
        "influence_threshold", "sensitivity_thresholds",
        "dispatch_attempt_equation", "active_equation", "honor_equation",
        "influence_equation",
    }, where=f"{where}/context")
    threshold = _finite(
        context["influence_threshold"], where=f"{where}/context/threshold",
    )
    sensitivity_thresholds = context["sensitivity_thresholds"]
    if (
        not isinstance(sensitivity_thresholds, list)
        or sensitivity_thresholds != list(DEFAULT_CONTEXT_SENSITIVITY_THRESHOLDS)
    ):
        raise ValueError(f"{where} changes the locked sensitivity thresholds")
    if threshold < 0.0 or (
        context["dispatch_attempt_equation"]
        != "context_modifier_is_a_3_vector"
        or context["active_equation"]
        != "max(abs(context_modifier))>influence_threshold"
        or context["honor_equation"]
        != "active_and_argmax(context_modifier)==action_idx"
        or context["influence_equation"]
        != "active_and_live_action_differs_from_paired_ablation"
    ):
        raise ValueError(f"{where} changes the locked context diagnostics")

    execution = _exact_keys(top["execution_activity"], {
        "mcp_tool_call_step_field", "primary_mcp_list_step_field",
        "cooperative_mcp_list_step_field", "pirag_query_step_field",
        "primary_pirag_attempt_step_field",
        "cooperative_pirag_attempt_step_field",
        "dispatcher_failure_step_field", "message_step_field",
        "protocol_step_fields", "aggregation",
    }, where=f"{where}/execution_activity")
    expected_execution = {
        "mcp_tool_call_step_field": "mcp_tool_call_count_step",
        "primary_mcp_list_step_field": "primary_mcp_tools_invoked_step",
        "cooperative_mcp_list_step_field": (
            "cooperative_mcp_tools_invoked_step"
        ),
        "pirag_query_step_field": "pirag_query_count_step",
        "primary_pirag_attempt_step_field": (
            "primary_pirag_query_attempted_step"
        ),
        "cooperative_pirag_attempt_step_field": (
            "cooperative_pirag_query_attempted_step"
        ),
        "dispatcher_failure_step_field": "dispatcher_tool_failure_count_step",
        "message_step_field": "inter_agent_message_count_step",
        "protocol_step_fields": list(PROTOCOL_STEP_FIELDS),
        "aggregation": "integer_sum_over_retained_decisions",
    }
    if dict(execution) != expected_execution:
        raise ValueError(f"{where} changes the locked execution aggregation")

    footprint = _exact_keys(top["footprint"], {
        "estimate_basis", "measurement_scope", "proxy_step_unit",
        "estimation_status", "assumed_active_power_W",
        "water_rate_L_per_server_second", "energy_per_step_proxy_J",
        "water_per_step_proxy_L", "energy_equation", "water_equation",
    }, where=f"{where}/footprint")
    for key in (
        "assumed_active_power_W", "water_rate_L_per_server_second",
        "energy_per_step_proxy_J", "water_per_step_proxy_L",
    ):
        if _finite(footprint[key], where=f"{where}/footprint/{key}") < 0.0:
            raise ValueError(f"{where}/footprint/{key} must be non-negative")
    if (
        footprint["estimate_basis"]
        != "measured_elapsed_seconds_x_declared_rates"
        or footprint["measurement_scope"] != latency["measurement_scope"]
        or not isinstance(footprint["proxy_step_unit"], str)
        or not footprint["proxy_step_unit"]
        or footprint["estimation_status"]
        != "activity-based estimate; not hardware telemetry"
        or footprint["energy_equation"]
        != "sum(latency_ms)/1000*assumed_active_power_W"
        or footprint["water_equation"]
        != "sum(latency_ms)/1000*water_rate_L_per_server_second"
    ):
        raise ValueError(f"{where} changes the locked footprint semantics")

    if expected_contract is not None:
        validate_episode_evidence_contract(
            expected_contract, where=f"{where}/expected",
        )
        _compare_values(contract, expected_contract, where=where)


def _nonnegative_int(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where} must be a non-negative integer")
    return value


def _linear_percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    rank = fraction * (len(ordered) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return float(ordered[low] + weight * (ordered[high] - ordered[low]))


def reconstruct_episode_evidence(
    records: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    *,
    where: str = "decision_ledger",
    contract_validated: bool = False,
) -> dict[str, Any]:
    """Rebuild all episode activity scalars from retained decision records."""

    if not contract_validated:
        validate_episode_evidence_contract(contract, where=f"{where}/contract")
    if not records:
        raise ValueError(f"{where} has no retained decisions")

    latencies: list[float] = []
    integer_totals = {
        field: 0
        for field in (
            "mcp_tool_call_count_step", "pirag_query_count_step",
            "dispatcher_tool_failure_count_step",
            "inter_agent_message_count_step", *PROTOCOL_STEP_FIELDS,
        )
    }
    dispatch_attempts = 0
    active_steps = 0
    honored_steps = 0
    influenced_steps = 0
    threshold = float(contract["context"]["influence_threshold"])
    sensitivity_thresholds = [
        float(value) for value in contract["context"]["sensitivity_thresholds"]
    ]
    threshold_counters = {
        value: {"active": 0, "honored": 0, "influenced": 0}
        for value in sensitivity_thresholds
    }
    active_per_recommendation = {
        index: 0 for index in range(len(ACTIONS))
    }
    ignored_per_recommendation = {
        index: 0 for index in range(len(ACTIONS))
    }

    for index, record in enumerate(records):
        row_where = f"{where}:{index}"
        latency = _finite(
            record.get("decision_latency_ms"),
            where=f"{row_where}/decision_latency_ms",
        )
        if latency < 0.0:
            raise ValueError(f"{row_where}/decision_latency_ms is negative")
        latencies.append(latency)
        for field in integer_totals:
            integer_totals[field] += _nonnegative_int(
                record.get(field), where=f"{row_where}/{field}",
            )
        primary_tools = record.get("primary_mcp_tools_invoked_step")
        cooperative_tools = record.get(
            "cooperative_mcp_tools_invoked_step"
        )
        for field, values in (
            ("primary_mcp_tools_invoked_step", primary_tools),
            ("cooperative_mcp_tools_invoked_step", cooperative_tools),
        ):
            if (
                not isinstance(values, list)
                or any(not isinstance(value, str) or not value for value in values)
            ):
                raise ValueError(f"{row_where}/{field} is not a tool-name list")
        expected_mcp_calls = len(primary_tools) + len(cooperative_tools)
        if record["mcp_tool_call_count_step"] != expected_mcp_calls:
            raise ValueError(
                f"{row_where}/mcp_tool_call_count_step does not equal the "
                "primary plus cooperative invocation lists"
            )
        primary_query = record.get("primary_pirag_query_attempted_step")
        cooperative_query = record.get(
            "cooperative_pirag_query_attempted_step"
        )
        if not isinstance(primary_query, bool) or not isinstance(
            cooperative_query, bool,
        ):
            raise ValueError(
                f"{row_where} retrieval-attempt evidence is not boolean"
            )
        if record["pirag_query_count_step"] != (
            int(primary_query) + int(cooperative_query)
        ):
            raise ValueError(
                f"{row_where}/pirag_query_count_step does not equal the "
                "primary plus cooperative attempts"
            )
        # The protocol counter records every dispatched tools/call, so it
        # includes calls that raised after dispatch. The invocation lists hold
        # only calls that returned. The difference between them is exactly the
        # dispatcher's failure count, which the record carries for this
        # reconciliation. Skipped tools are never dispatched and appear in
        # none of the three.
        failed_calls = record.get("dispatcher_tool_failure_count_step", 0)
        if not isinstance(failed_calls, int) or failed_calls < 0:
            raise ValueError(
                f"{row_where}/dispatcher_tool_failure_count_step is not a "
                "non-negative integer"
            )
        if record["protocol_tools_call_count_step"] != (
            expected_mcp_calls + failed_calls
        ):
            raise ValueError(
                f"{row_where}/protocol_tools_call_count_step disagrees with "
                "MCP invocation evidence plus dispatched calls that failed"
            )
        if record["protocol_prompts_get_count_step"] != (
            int(primary_query) + int(cooperative_query)
        ):
            raise ValueError(
                f"{row_where}/protocol_prompts_get_count_step disagrees with "
                "retrieval-attempt evidence"
            )
        if record["protocol_interaction_count_step"] != (
            record["protocol_tools_call_count_step"]
            + record["protocol_prompts_get_count_step"]
        ):
            raise ValueError(
                f"{row_where}/protocol_interaction_count_step is not the "
                "sum of declared protocol methods"
            )
        # The recorder currently declares no by-design ``isError`` exclusions,
        # so the raw and real counters must be identical.  Retaining only an
        # inequality would allow a forged artifact to hide a genuine tool
        # failure by relabelling it as excluded.
        if record["protocol_real_tool_iserror_count_step"] != (
            record["protocol_tool_iserror_count_step"]
        ):
            raise ValueError(
                f"{row_where} raw and real tool error counts differ despite "
                "the empty exclusion contract"
            )
        if record["protocol_tool_iserror_count_step"] > (
            record["protocol_tools_call_count_step"]
        ):
            raise ValueError(f"{row_where} has more tool errors than tool calls")
        if record["protocol_jsonrpc_error_count_step"] > (
            record["protocol_interaction_count_step"]
        ):
            raise ValueError(
                f"{row_where} has more JSON-RPC errors than interactions"
            )
        if record["protocol_error_count_step"] != (
            record["protocol_jsonrpc_error_count_step"]
            + record["protocol_real_tool_iserror_count_step"]
        ):
            raise ValueError(f"{row_where} has an inconsistent protocol error total")

        modifier = record.get("context_modifier")
        attempted = isinstance(modifier, list) and len(modifier) == len(ACTIONS)
        if modifier is not None and not attempted:
            raise ValueError(f"{row_where}/context_modifier has invalid shape")
        if attempted:
            try:
                modifier_values = [float(value) for value in modifier]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{row_where}/context_modifier is not numeric"
                ) from exc
            if not all(math.isfinite(value) for value in modifier_values):
                raise ValueError(f"{row_where}/context_modifier is non-finite")
            dispatch_attempts += 1
            expected_active = max(abs(value) for value in modifier_values) > threshold
            action_idx = record.get("action_idx")
            if (
                isinstance(action_idx, bool)
                or not isinstance(action_idx, int)
                or action_idx not in range(len(ACTIONS))
            ):
                raise ValueError(f"{row_where}/action_idx is not canonical")
            if expected_active:
                active_steps += 1
                recommended = max(
                    range(len(ACTIONS)), key=lambda item: modifier_values[item]
                )
                honored = recommended == action_idx
                honored_steps += int(honored)
                active_per_recommendation[recommended] += 1
                ignored_per_recommendation[recommended] += int(not honored)
            action_changed = record.get("context_action_changed")
            if action_changed is not None and not isinstance(
                action_changed, bool,
            ):
                raise ValueError(f"{row_where}/context_action_changed is invalid")
            expected_influenced = expected_active and action_changed is True
            influenced_steps += int(expected_influenced)
            recommended = max(
                range(len(ACTIONS)), key=lambda item: modifier_values[item]
            )
            for sensitivity_threshold in sensitivity_thresholds:
                sensitivity_active = (
                    max(abs(value) for value in modifier_values)
                    > sensitivity_threshold
                )
                if sensitivity_active:
                    threshold_counters[sensitivity_threshold]["active"] += 1
                    threshold_counters[sensitivity_threshold]["honored"] += int(
                        recommended == action_idx
                    )
                    threshold_counters[sensitivity_threshold][
                        "influenced"
                    ] += int(action_changed is True)
        else:
            expected_active = False
            expected_influenced = False
        if record.get("context_influence_active") is not expected_active or (
            record.get("context_influence_counted") is not expected_influenced
        ):
            raise ValueError(f"{row_where} context diagnostic flags disagree")

    n_records = len(records)
    protocol_errors = integer_totals["protocol_error_count_step"]
    dispatcher_failures = integer_totals[
        "dispatcher_tool_failure_count_step"
    ]
    footprint_contract = contract["footprint"]
    elapsed_samples = [value / 1000.0 for value in latencies]
    # FootprintMeter accumulates in decision order with ``+=``. Built-in sum
    # has the same sequential floating-point semantics, so the rounded audit
    # reconstruction matches the emitted summary byte-for-byte.
    elapsed_seconds = sum(elapsed_samples)
    energy_total = sum(
        float(footprint_contract["assumed_active_power_W"]) * value
        for value in elapsed_samples
    )
    water_total = sum(
        float(footprint_contract["water_rate_L_per_server_second"]) * value
        for value in elapsed_samples
    )
    energy_proxy_total = sum(
        float(footprint_contract["energy_per_step_proxy_J"])
        for _ in records
    )
    water_proxy_total = sum(
        float(footprint_contract["water_per_step_proxy_L"])
        for _ in records
    )
    footprint = {
        "cumulative_energy_J": round(energy_total, 8),
        "cumulative_water_L": round(water_total, 12),
        "cumulative_elapsed_seconds": round(elapsed_seconds, 12),
        "cumulative_energy_per_step_proxy_J": round(energy_proxy_total, 8),
        "cumulative_water_per_step_proxy_L": round(water_proxy_total, 12),
        "total_steps": n_records,
        "timed_call_count": n_records,
        "time_based_estimate_available": True,
        "estimate_basis": footprint_contract["estimate_basis"],
        "measurement_scope": footprint_contract["measurement_scope"],
        "proxy_step_unit": footprint_contract["proxy_step_unit"],
        "estimation_status": footprint_contract["estimation_status"],
        "assumed_active_power_W": float(
            footprint_contract["assumed_active_power_W"]
        ),
        "water_rate_L_per_server_second": float(
            footprint_contract["water_rate_L_per_server_second"]
        ),
    }
    threshold_evidence = {}
    for sensitivity_threshold, counts in threshold_counters.items():
        active = counts["active"]
        threshold_evidence[f"{sensitivity_threshold:.2f}"] = {
            **counts,
            "honor_rate": float(
                counts["honored"] / active if active else 0.0
            ),
            "influence_rate": float(
                counts["influenced"] / active if active else 0.0
            ),
        }
    return {
        "mean_decision_latency_ms": float(math.fsum(latencies) / n_records),
        "p95_decision_latency_ms": _linear_percentile(latencies, 0.95),
        "latency_penalty_usd": float(math.fsum(
            max(value - 50.0, 0.0) for value in latencies
        ) * 0.0002),
        "context_active_steps": active_steps,
        "context_active_fraction": float(active_steps / n_records),
        "context_dispatch_attempt_steps": dispatch_attempts,
        "context_dispatch_attempt_fraction": float(
            dispatch_attempts / n_records
        ),
        "context_honored_steps": honored_steps,
        "context_honor_rate": float(
            honored_steps / active_steps if active_steps else 0.0
        ),
        "context_influenced_steps": influenced_steps,
        "context_influence_rate": float(
            influenced_steps / active_steps if active_steps else 0.0
        ),
        "context_dispatch_influence_rate": float(
            influenced_steps / dispatch_attempts if dispatch_attempts else 0.0
        ),
        "context_active_per_recommendation": active_per_recommendation,
        "context_ignored_per_recommendation": ignored_per_recommendation,
        "context_threshold_counters": threshold_evidence,
        "mcp_calls_per_episode": integer_totals["mcp_tool_call_count_step"],
        "pirag_queries_per_episode": integer_totals["pirag_query_count_step"],
        "message_count": integer_totals["inter_agent_message_count_step"],
        "protocol_interaction_count": integer_totals[
            "protocol_interaction_count_step"
        ],
        "protocol_tools_call_count": integer_totals[
            "protocol_tools_call_count_step"
        ],
        "protocol_prompts_get_count": integer_totals[
            "protocol_prompts_get_count_step"
        ],
        "protocol_jsonrpc_error_count": integer_totals[
            "protocol_jsonrpc_error_count_step"
        ],
        "protocol_tool_iserror_count": integer_totals[
            "protocol_tool_iserror_count_step"
        ],
        "protocol_real_tool_iserror_count": integer_totals[
            "protocol_real_tool_iserror_count_step"
        ],
        "protocol_error_count": protocol_errors,
        "protocol_dropped_interaction_count": integer_totals[
            "protocol_dropped_interaction_count_step"
        ],
        "dispatcher_tool_failure_count": dispatcher_failures,
        "context_execution_error_count": protocol_errors + dispatcher_failures,
        "footprint": footprint,
    }
