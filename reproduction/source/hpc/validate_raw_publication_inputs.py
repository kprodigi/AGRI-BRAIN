#!/usr/bin/env python3
"""Reject incomplete, mixed-run, or unstamped raw publication inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "agribrain" / "backend"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from mvp.simulation.generate_results import (  # noqa: E402
    MODES, SCENARIOS, TRACE_SCHEMA_VERSION, _stream_id,
)
from mvp.simulation.benchmarks.run_stress_suite import (  # noqa: E402
    STRESS_THRESHOLDS, _equivalence_tost,
)
from src.models.mode_capabilities import (  # noqa: E402
    DECISION_OWNER_ROLES, capabilities_for,
)
from src.models.resilience import compute_equity, compute_rle  # noqa: E402
from src.models.synthetic_spoilage_dgp import (  # noqa: E402
    synthetic_dgp_provenance,
)
from mvp.simulation.analysis.experiment_accounting import (  # noqa: E402
    PRIMARY_PUBLICATION_MODES,
    build_episode_accounting,
    build_h3_episode_accounting,
)
from mvp.simulation.benchmarks.trace_contract import (  # noqa: E402
    TRACE_MODES as CANONICAL_TRACE_MODES,
    validate_trace_cell as validate_canonical_trace_cell,
)
from hpc.validate_decision_ledgers import (  # noqa: E402
    expected_publication_episode_evidence_contract,
    expected_publication_outcome_equation_contract,
    validate_learner_snapshot_binding,
    validate_ledger,
)
from src.models.episode_evidence_contract import (  # noqa: E402
    ACTIVITY_STEP_FIELDS,
    reconstruct_episode_evidence,
    validate_episode_evidence_contract,
)
from src.models.outcome_equation_contract import (  # noqa: E402
    validate_outcome_equation_contract,
    validate_recorded_step_outcomes,
)
from hpc.slurm_execution_provenance import (  # noqa: E402
    require_declared_publisher,
    validate_core_array_provenance,
)


EXPECTED_SEEDS = (
    42, 1337, 2024, 7, 99, 101, 202, 303, 404, 505,
    606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515,
)
EPISODE_SCOPE = "final episode per scenario-mode-seed arm"
HISTORY_SCOPE = "earlier decisions in the same episode only"
TRACE_MODES = set(CANONICAL_TRACE_MODES)
STRESSORS = (
    "sensor_noise", "missing_data", "telemetry_delay",
    "mcp_fault_injection", "compounded",
)
STRESS_MODES = {stressor: {"agribrain"} for stressor in STRESSORS}
BASELINE_STRESS_MODES = set().union(*STRESS_MODES.values())
PROTOCOL_COUNT_FIELDS = (
    "protocol_interaction_count",
    "protocol_jsonrpc_error_count",
    "protocol_tool_iserror_count",
    "protocol_real_tool_iserror_count",
    "protocol_error_count",
    "protocol_dropped_interaction_count",
    "dispatcher_tool_failure_count",
    "context_execution_error_count",
)


def _expected_seed_episode_accounting() -> dict[str, Any]:
    """Return the exact one-seed execution budget for the live publication panel."""

    configured_modes = list(MODES)
    primary_modes = [
        mode for mode in PRIMARY_PUBLICATION_MODES if mode in configured_modes
    ]
    accounting = build_episode_accounting(
        scenarios=SCENARIOS,
        configured_modes=configured_modes,
        episode_budget_by_mode={
            mode: int(capabilities_for(mode).episode_count)
            for mode in configured_modes
        },
        n_seeds=1,
        primary_modes=primary_modes,
    )
    accounting["complete_primary_mode_panel"] = (
        tuple(primary_modes) == PRIMARY_PUBLICATION_MODES
    )
    return accounting


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_constant,
        )
    except Exception as exc:
        raise RuntimeError(f"invalid strict JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return payload


def _validate_identity(
    meta: Any, *, source_commit: str, run_tag: str, where: Path,
) -> None:
    if not isinstance(meta, dict):
        raise RuntimeError(f"{where} has no provenance metadata object")
    if meta.get("source_commit") != source_commit:
        raise RuntimeError(f"{where} source_commit does not match the run")
    if meta.get("run_tag") != run_tag:
        raise RuntimeError(f"{where} run_tag does not match the run")


def _validate_context_execution_counts(cell: Any, *, where: str) -> None:
    """Require inspectable, internally consistent zero-error execution."""
    if not isinstance(cell, dict):
        raise RuntimeError(f"{where} is not a metric object")
    counts: dict[str, int] = {}
    for field in PROTOCOL_COUNT_FIELDS:
        value = cell.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(
                f"{where} lacks non-negative integer execution metric {field}"
            )
        counts[field] = value
    if counts["protocol_real_tool_iserror_count"] > counts["protocol_tool_iserror_count"]:
        raise RuntimeError(f"{where} has inconsistent tool isError counters")
    if counts["protocol_error_count"] != (
        counts["protocol_jsonrpc_error_count"]
        + counts["protocol_real_tool_iserror_count"]
    ):
        raise RuntimeError(f"{where} has an inconsistent protocol error total")
    if counts["context_execution_error_count"] != (
        counts["protocol_error_count"] + counts["dispatcher_tool_failure_count"]
    ):
        raise RuntimeError(f"{where} has an inconsistent context execution error total")
    nonzero_failures = {
        field: counts[field]
        for field in (
            "protocol_jsonrpc_error_count",
            "protocol_real_tool_iserror_count",
            "protocol_dropped_interaction_count",
            "dispatcher_tool_failure_count",
            "context_execution_error_count",
        )
        if counts[field]
    }
    if nonzero_failures:
        raise RuntimeError(
            f"{where} contains publication-invalid context execution failures: "
            f"{nonzero_failures}"
        )


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_spoilage_estimator(
    value: Any, *, mode: str, where: str,
) -> None:
    """Require the exact deployed spoilage-estimator provenance contract."""

    expected_keys = {
        "kind", "checkpoint_sha256", "training_dataset_sha256",
        "training_target_origin", "residual_bound_abs",
        "deployment_transform", "synthetic_only", "external_validation",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise RuntimeError(f"{where} has an invalid spoilage-estimator schema")
    if value.get("synthetic_only") is not True or (
        value.get("external_validation") is not False
    ):
        raise RuntimeError(
            f"{where} misstates the synthetic-only/no-external-validation boundary"
        )
    if capabilities_for(mode).spoilage_residual:
        if (
            value.get("kind")
            != "mechanistic_plus_frozen_synthetic_pinn_residual"
            or not _valid_sha256(value.get("checkpoint_sha256"))
            or not _valid_sha256(value.get("training_dataset_sha256"))
            or value.get("training_target_origin") != "independent_synthetic_dgp"
            or value.get("deployment_transform")
            != "clip_quality_to_unit_interval_then_cumulative_minimum"
            or isinstance(value.get("residual_bound_abs"), bool)
            or not isinstance(value.get("residual_bound_abs"), (int, float))
            or not math.isclose(
                float(value["residual_bound_abs"]), 0.08,
                rel_tol=0.0, abs_tol=1e-15,
            )
        ):
            raise RuntimeError(
                f"{where} does not identify the locked frozen synthetic PINN"
            )
        return

    if mode != "no_pinn" or value != {
        "kind": "mechanistic_only_no_pinn",
        "checkpoint_sha256": None,
        "training_dataset_sha256": None,
        "training_target_origin": None,
        "residual_bound_abs": None,
        "deployment_transform": None,
        "synthetic_only": True,
        "external_validation": False,
    }:
        raise RuntimeError(
            f"{where} does not identify the clean mechanistic-only ablation"
        )


def _validate_latent_spoilage_model(
    value: Any,
    *,
    where: str,
    effective_k_ref: Any = None,
    effective_ea_r: Any = None,
) -> None:
    """Require the exact common noise-free independent synthetic DGP."""

    if not isinstance(value, dict) or not isinstance(
        value.get("parameters"), dict,
    ):
        raise RuntimeError(
            f"{where} does not identify the locked independent synthetic DGP"
        )
    parameters = value["parameters"]
    k_ref = parameters.get("k_ref_per_h") if effective_k_ref is None else (
        effective_k_ref
    )
    ea_r = parameters.get("ea_over_r_kelvin") if effective_ea_r is None else (
        effective_ea_r
    )
    try:
        expected = synthetic_dgp_provenance(
            k_ref=float(k_ref),
            Ea_R=float(ea_r),
            T_ref_K=float(parameters.get("reference_temperature_kelvin")),
            beta=float(parameters.get("humidity_coupling")),
            lag_lambda=float(parameters.get("lag_lambda_hours")),
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{where} has invalid independent synthetic DGP parameters"
        ) from exc
    if value != expected:
        raise RuntimeError(
            f"{where} does not identify the locked independent synthetic DGP"
        )


def _integral_count(value: Any, *, upper: int | None = None) -> int | None:
    """Return an exact non-negative integer count, or ``None`` if invalid.

    JSON producers sometimes serialize counters as ``288.0`` even though the
    value is mathematically integral.  Reject booleans, fractional values,
    infinities, negatives, and values above an optional bound, while accepting
    both JSON integer and exactly-integral JSON floating-point encodings.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or not numeric.is_integer():
        return None
    count = int(numeric)
    if upper is not None and count > upper:
        return None
    return count


def _validate_sign_reversal_diagnostics(
    summary: dict[str, Any],
    *,
    where: str,
    sign_constrained: bool,
    feature_key: str | None = None,
    feature_count: int | None = None,
) -> list[dict[str, Any]]:
    """Validate an auditable list of actual, rather than assumed, sign flips."""

    count = _integral_count(summary.get("sign_reversal_count"))
    coordinates = summary.get("sign_reversal_coordinates")
    if count is None or not isinstance(coordinates, list) or len(coordinates) != count:
        raise RuntimeError(f"{where} has invalid sign-reversal diagnostics")
    for index, item in enumerate(coordinates):
        item_where = f"{where}/sign_reversal_coordinates[{index}]"
        if not isinstance(item, dict):
            raise RuntimeError(f"{item_where} is not an object")
        action = _integral_count(item.get("action_index"), upper=2)
        if action is None:
            raise RuntimeError(f"{item_where} has an invalid action index")
        if feature_key is not None:
            feature = _integral_count(
                item.get(feature_key),
                upper=(feature_count - 1 if feature_count is not None else None),
            )
            if feature is None:
                raise RuntimeError(f"{item_where} has an invalid feature index")
        try:
            initial = float(item["initial_weight"])
            final = float(item["final_weight"])
            declared_sign = int(item["declared_sign"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"{item_where} has invalid weights/sign") from exc
        if (
            not math.isfinite(initial)
            or not math.isfinite(final)
            or declared_sign not in (-1, 1)
            or final * declared_sign >= 0.0
        ):
            raise RuntimeError(f"{item_where} does not describe a sign reversal")
    if sign_constrained and count != 0:
        raise RuntimeError(f"{where} reports a reversal under sign projection")
    worst = summary.get("worst_sign_reversal")
    if (not coordinates and worst is not None) or (
        coordinates and worst not in coordinates
    ):
        raise RuntimeError(f"{where} has an inconsistent worst-sign-reversal record")
    if coordinates and abs(float(worst["final_weight"])) != max(
        abs(float(item["final_weight"])) for item in coordinates
    ):
        raise RuntimeError(f"{where} misidentifies the worst sign reversal")
    return coordinates


def _validate_learner_provenance(cell: Any, *, mode: str, where: str) -> None:
    """Bind each learned mode to non-zero, hash-stamped final learner state."""
    if not isinstance(cell, dict):
        raise RuntimeError(f"{where} is not a metric object")
    caps = capabilities_for(mode)
    message_count = _integral_count(cell.get("message_count"))
    if message_count is None or (
        caps.peer_messages and message_count == 0
    ) or (
        not caps.peer_messages and message_count != 0
    ):
        raise RuntimeError(f"{where} has invalid peer-message exposure")

    context_summary = cell.get("learner_summary")
    if caps.context_matrix_learning:
        if not isinstance(context_summary, dict):
            raise RuntimeError(f"{where} lacks context-learner provenance")
        if context_summary.get("mode") != mode or context_summary.get(
            "learner_state_schema_version"
        ) != 2 or not _valid_sha256(context_summary.get("state_sha256")):
            raise RuntimeError(f"{where} has invalid context-learner provenance")
        if context_summary.get("sign_constrained") is not (
            caps.sign_constrained_learning
        ):
            raise RuntimeError(
                f"{where} has the wrong context-learner sign projection"
            )
        if int(context_summary.get("n_updates", 0)) <= 0:
            raise RuntimeError(f"{where} context learner did not update")
        context_state = {
            "theta": context_summary.get("final_theta"),
            "slca_amp_coeff": context_summary.get("final_slca_amp"),
            "learn_proxy_interaction": context_summary.get(
                "learn_proxy_interaction"
            ),
            "sign_constrained": context_summary.get("sign_constrained"),
            "temporal_base": context_summary.get("temporal_base"),
            "temporal_scale": context_summary.get("temporal_scale"),
            "reward_baseline": context_summary.get("reward_baseline"),
            "n_updates": context_summary.get("n_updates"),
        }
        if _canonical_object_sha256(context_state) != context_summary.get(
            "state_sha256"
        ):
            raise RuntimeError(
                f"{where} context learner hash does not reconstruct from state"
            )
        context_reversals = _validate_sign_reversal_diagnostics(
            context_summary,
            where=f"{where}/context_learner",
            sign_constrained=caps.sign_constrained_learning,
            feature_key="context_feature_index",
            feature_count=5,
        )
        if context_summary.get("sign_preserved") is not (
            len(context_reversals) == 0
        ):
            raise RuntimeError(f"{where} has an inconsistent sign-preserved flag")
        compliance_reversals = [
            item for item in context_reversals
            if item["context_feature_index"] == 0
        ]
        if _integral_count(
            context_summary.get("compliance_sign_reversal_count")
        ) != len(compliance_reversals):
            raise RuntimeError(
                f"{where} has an inconsistent compliance-reversal count"
            )
        worst_compliance = context_summary.get(
            "worst_compliance_sign_reversal"
        )
        if (not compliance_reversals and worst_compliance is not None) or (
            compliance_reversals and worst_compliance not in compliance_reversals
        ):
            raise RuntimeError(
                f"{where} has an inconsistent worst compliance reversal"
            )
    elif context_summary not in (None, {}):
        # ``no_context`` and frozen context-prior diagnostics intentionally
        # construct dormant infrastructure for structural parity.  Accept a
        # stamped learner only when the artifact proves that it was disabled
        # and performed exactly zero updates; an active undeclared learner is
        # still publication-invalid.
        if not isinstance(context_summary, dict) or (
            context_summary.get("mode") != mode
            or context_summary.get("learner_state_schema_version") != 2
            or not _valid_sha256(context_summary.get("state_sha256"))
            or context_summary.get("learning_enabled") is not False
            or _integral_count(context_summary.get("n_updates")) != 0
        ):
            raise RuntimeError(f"{where} unexpectedly reports an active context learner")
        dormant_context_state = {
            "theta": context_summary.get("final_theta"),
            "slca_amp_coeff": context_summary.get("final_slca_amp"),
            "learn_proxy_interaction": context_summary.get(
                "learn_proxy_interaction"
            ),
            "sign_constrained": context_summary.get("sign_constrained"),
            "temporal_base": context_summary.get("temporal_base"),
            "temporal_scale": context_summary.get("temporal_scale"),
            "reward_baseline": context_summary.get("reward_baseline"),
            "n_updates": context_summary.get("n_updates"),
        }
        if _canonical_object_sha256(dormant_context_state) != (
            context_summary.get("state_sha256")
        ):
            raise RuntimeError(
                f"{where} dormant context learner hash does not reconstruct"
            )

    theta_summary = cell.get("theta_learner_summary")
    if caps.policy_delta_learning:
        if not isinstance(theta_summary, dict):
            raise RuntimeError(f"{where} lacks policy-delta learner provenance")
        if theta_summary.get("mode") != mode or theta_summary.get(
            "learner_state_schema_version"
        ) != 2 or not _valid_sha256(theta_summary.get("combined_state_sha256")):
            raise RuntimeError(f"{where} has invalid policy-delta provenance")
        if theta_summary.get("sign_constrained") is not (
            caps.sign_constrained_learning
        ):
            raise RuntimeError(
                f"{where} has the wrong policy-delta sign projection"
            )
        roles = theta_summary.get("decision_owner_roles")
        updates = theta_summary.get("updates_per_role")
        hashes = theta_summary.get("per_role_state_sha256")
        per_role = theta_summary.get("per_role")
        if roles != list(DECISION_OWNER_ROLES) or not isinstance(updates, dict) or (
            set(updates) != set(DECISION_OWNER_ROLES)
        ) or not isinstance(hashes, dict) or set(hashes) != set(DECISION_OWNER_ROLES) or (
            not isinstance(per_role, dict)
            or set(per_role) != set(DECISION_OWNER_ROLES)
        ):
            raise RuntimeError(f"{where} has an invalid decision-owner panel")
        update_values = []
        role_states: dict[str, Any] = {}
        for role in DECISION_OWNER_ROLES:
            value = updates[role]
            count = _integral_count(value, upper=288)
            if count is None or count == 0:
                raise RuntimeError(f"{where} has invalid {role} update count")
            if not _valid_sha256(hashes[role]):
                raise RuntimeError(f"{where} has invalid {role} learner hash")
            if (not isinstance(per_role[role], dict)
                    or per_role[role].get("sign_constrained") is not (
                        caps.sign_constrained_learning
                    )):
                raise RuntimeError(
                    f"{where} has the wrong {role} sign projection"
                )
            _validate_sign_reversal_diagnostics(
                per_role[role],
                where=f"{where}/policy_delta/{role}",
                sign_constrained=caps.sign_constrained_learning,
                feature_key="state_feature_index",
                feature_count=10,
            )
            role_state = {
                "theta_delta": per_role[role].get("final_theta_delta"),
                "reward_baseline": per_role[role].get("reward_baseline"),
                "n_updates": per_role[role].get("n_updates"),
                "learning_rate": per_role[role].get("learning_rate"),
                "prior_precision": per_role[role].get("prior_precision"),
                "magnitude_cap_fraction": per_role[role].get(
                    "magnitude_cap_fraction"
                ),
                "sign_constrained": per_role[role].get("sign_constrained"),
            }
            if _canonical_object_sha256(role_state) != hashes[role]:
                raise RuntimeError(
                    f"{where} {role} policy-delta hash does not reconstruct"
                )
            role_states[role] = role_state
            update_values.append(count)
        if _canonical_object_sha256(role_states) != theta_summary.get(
            "combined_state_sha256"
        ):
            raise RuntimeError(
                f"{where} combined policy-delta hash does not reconstruct"
            )
        if int(theta_summary.get("n_updates", -1)) != sum(update_values):
            raise RuntimeError(f"{where} has inconsistent policy-delta update totals")
        per_role_reversals = sum(
            int(per_role[role]["sign_reversal_count"])
            for role in DECISION_OWNER_ROLES
        )
        if _integral_count(theta_summary.get("sign_reversal_count")) != (
            per_role_reversals
        ):
            raise RuntimeError(
                f"{where} has an inconsistent policy-delta reversal total"
            )
    elif theta_summary not in (None, {}):
        raise RuntimeError(f"{where} unexpectedly reports a policy-delta learner")

    shaping_summary = cell.get("reward_shaping_learner_summary")
    if caps.reward_shaping_learning:
        if not isinstance(shaping_summary, dict):
            raise RuntimeError(f"{where} lacks reward-shaping learner provenance")
        if shaping_summary.get("mode") != mode or shaping_summary.get(
            "learner_state_schema_version"
        ) != 2 or not _valid_sha256(shaping_summary.get("state_sha256")):
            raise RuntimeError(f"{where} has invalid reward-shaping provenance")
        if shaping_summary.get("sign_constrained") is not (
            caps.sign_constrained_learning
        ):
            raise RuntimeError(
                f"{where} has the wrong reward-shaping sign projection"
            )
        if int(shaping_summary.get("n_updates", 0)) <= 0:
            raise RuntimeError(f"{where} reward-shaping learner did not update")
        shaping_state = {
            "slca_bonus_delta": shaping_summary.get("slca_bonus_delta"),
            "slca_rho_delta": shaping_summary.get("slca_rho_delta"),
            "no_slca_offset_delta": shaping_summary.get(
                "no_slca_offset_delta"
            ),
            "reward_baseline": shaping_summary.get("reward_baseline"),
            "n_updates": shaping_summary.get("n_updates"),
            "magnitude_cap_fraction": shaping_summary.get(
                "magnitude_cap_fraction"
            ),
            "sign_constrained": shaping_summary.get("sign_constrained"),
        }
        if _canonical_object_sha256(shaping_state) != shaping_summary.get(
            "state_sha256"
        ):
            raise RuntimeError(
                f"{where} reward-shaping hash does not reconstruct from state"
            )
        shaping_reversals = _validate_sign_reversal_diagnostics(
            shaping_summary,
            where=f"{where}/reward_shaping",
            sign_constrained=caps.sign_constrained_learning,
        )
        allowed_parameters = {
            "slca_bonus", "slca_rho_bonus", "no_slca_offset",
        }
        if any(
            item.get("parameter") not in allowed_parameters
            for item in shaping_reversals
        ):
            raise RuntimeError(f"{where} has an unknown reward-shaping vector")
    elif shaping_summary not in (None, {}):
        raise RuntimeError(f"{where} unexpectedly reports reward shaping")


def _seed_panel_keys(panel: Any, *, where: str) -> set[int]:
    if not isinstance(panel, dict):
        raise RuntimeError(f"{where} is not a seed panel")
    try:
        keys = {int(key) for key in panel}
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{where} has a non-integer seed key") from exc
    if keys != set(EXPECTED_SEEDS) or len(panel) != len(EXPECTED_SEEDS):
        raise RuntimeError(
            f"{where} does not contain the exact 20-seed panel: "
            f"missing={sorted(set(EXPECTED_SEEDS) - keys)}, "
            f"unexpected={sorted(keys - set(EXPECTED_SEEDS))}"
        )
    return keys


def _panel_cell(panel: dict[str, Any], seed: int, mode: str) -> dict[str, Any]:
    seed_cell = panel.get(str(seed), panel.get(seed))
    if not isinstance(seed_cell, dict) or not isinstance(seed_cell.get(mode), dict):
        raise RuntimeError(f"missing stress cell seed={seed}/mode={mode}")
    return seed_cell[mode]


def _assert_close(actual: Any, expected: float, *, where: str) -> None:
    try:
        value = float(actual)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{where} is not numeric") from exc
    if not math.isfinite(value) or not math.isclose(
        value, float(expected), rel_tol=1e-10, abs_tol=1e-12,
    ):
        raise RuntimeError(f"{where}={value!r}, expected {expected!r}")


def _validate_trace_cell(cell: Any, *, where: str) -> None:
    try:
        validate_canonical_trace_cell(cell, where=where)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    if not isinstance(cell, dict):
        raise RuntimeError(f"{where} is not a trace object")
    required = {
        "hours", "ari_trace", "waste_trace", "slca_trace",
        "rho_trace", "rho_policy_observed_trace",
        "rho_outcome_environmental_trace", "temp_trace",
        "temp_policy_observed_trace", "temp_outcome_environmental_trace",
        "rh_trace", "rh_policy_observed_trace",
        "rh_outcome_environmental_trace", "inventory_trace",
        "inventory_policy_observed_trace",
        "inventory_outcome_environmental_trace",
        "demand_trace", "demand_forecast_policy_observed_trace",
        "demand_outcome_environmental_trace",
        "transport_multiplier_outcome_environmental_trace",
        "simulated_dispatch_accounted_trace",
    }
    missing = required.difference(cell)
    if missing:
        raise RuntimeError(f"{where} lacks traces {sorted(missing)}")
    for field in required:
        values = cell[field]
        if not isinstance(values, list) or len(values) != 288:
            raise RuntimeError(f"{where}/{field} is not a complete 288-step trace")
        if field == "simulated_dispatch_accounted_trace":
            if any(value is not True for value in values):
                raise RuntimeError(
                    f"{where}/{field} contains an unaccounted opportunity"
                )
            continue
        try:
            numeric = [float(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{where}/{field} is not numeric") from exc
        if not all(math.isfinite(value) for value in numeric):
            raise RuntimeError(f"{where}/{field} contains a non-finite value")

    hours = [float(value) for value in cell["hours"]]
    if not all(
        math.isclose(hours[index] - hours[index - 1], 0.25, abs_tol=1e-9)
        for index in range(1, len(hours))
    ):
        raise RuntimeError(f"{where} does not use the declared 15-minute cadence")
    for field in ("rho_policy_observed_trace", "rho_outcome_environmental_trace"):
        values = [float(value) for value in cell[field]]
        if any(value < 0.0 or value > 1.0 for value in values):
            raise RuntimeError(f"{where}/{field} leaves [0,1]")
        if any(values[index] + 1e-12 < values[index - 1]
               for index in range(1, len(values))):
            raise RuntimeError(f"{where}/{field} is not monotone")
    for alias, explicit in (
        ("rho_trace", "rho_policy_observed_trace"),
        ("temp_trace", "temp_policy_observed_trace"),
        ("rh_trace", "rh_policy_observed_trace"),
        ("inventory_trace", "inventory_policy_observed_trace"),
        ("demand_trace", "demand_forecast_policy_observed_trace"),
    ):
        if cell[alias] != cell[explicit]:
            raise RuntimeError(f"{where} legacy alias {alias} is ambiguous")
    for index, (ari, waste, social, rho) in enumerate(zip(
        cell["ari_trace"], cell["waste_trace"], cell["slca_trace"],
        cell["rho_outcome_environmental_trace"],
    )):
        expected = (1.0 - float(waste)) * float(social) * (1.0 - float(rho))
        if not math.isclose(float(ari), expected, rel_tol=2e-3, abs_tol=2e-4):
            raise RuntimeError(f"{where}/ari_trace[{index}] violates the ARI equation")


def _canonical_object_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _expected_observation_treatment(
    *, scenario: str, stressor: str, seed: int, episode_index: int = 3,
    n_steps: int = 288,
) -> dict[str, Any]:
    """Reconstruct the locked retained-episode H3 dose provenance."""

    key = f"stress|{scenario}|{stressor}|{seed}|{episode_index}".encode("utf-8")
    cell_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    rng = np.random.default_rng(cell_seed)
    treatment: dict[str, Any] = {
        "stressor": stressor,
        "n_steps": n_steps,
        "data_observation_treatment": stressor != "mcp_fault_injection",
        "delay_steps": 0,
        "missing_count": 0,
    }

    def array_hash(values: Any) -> str:
        canonical = json.dumps(
            [float(value) for value in np.asarray(values, dtype=float)],
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    if stressor in {"sensor_noise", "compounded"}:
        temp_noise = rng.normal(0.0, 2.0, size=n_steps)
        rh_noise = rng.normal(0.0, 5.0, size=n_steps)
        treatment["temp_noise_sha256"] = array_hash(temp_noise)
        treatment["rh_noise_sha256"] = array_hash(rh_noise)
    if stressor in {"missing_data", "compounded"}:
        miss = rng.random(n_steps) < 0.10
        if n_steps:
            miss[0] = False
        treatment["missing_count"] = int(np.count_nonzero(miss))
        treatment["missing_mask_sha256"] = hashlib.sha256(
            np.asarray(miss, dtype=np.uint8).tobytes()
        ).hexdigest()
    if stressor in {"telemetry_delay", "compounded"}:
        treatment["delay_steps"] = 4
    treatment["treatment_sha256"] = _canonical_object_sha256(treatment)
    return treatment


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ledger_merkle_root(leaves: list[str]) -> str:
    if not leaves:
        return "0" * 64
    layer = [bytes.fromhex(leaf) for leaf in leaves]
    while len(layer) > 1:
        if len(layer) % 2:
            layer.append(layer[-1])
        layer = [
            hashlib.sha256(layer[index] + layer[index + 1]).digest()
            for index in range(0, len(layer), 2)
        ]
    return layer[0].hex()


def _strict_ledger_object(
    line: str, *, path: Path, line_number: int,
) -> dict[str, Any]:
    try:
        payload = json.loads(line, parse_constant=_reject_constant)
    except Exception as exc:
        raise RuntimeError(
            f"invalid H3 decision-ledger JSON at {path}:{line_number}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"non-object H3 ledger row at {path}:{line_number}")
    return payload


def _reconstruct_h3_ledger(
    path: Path, *, scenario: str, stressor: str, seed: int,
    cell: dict[str, Any], canonical_path: str,
) -> dict[str, Any]:
    """Recompute endpoints and H3 dose from one retained episode-3 ledger."""
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing retained H3 decision ledger: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 289:
        raise RuntimeError(
            f"{path} has {len(lines) - 1} decisions; expected retained 288"
        )
    header = _strict_ledger_object(lines[0], path=path, line_number=1)
    metadata = header.get("metadata")
    if (
        header.get("_header") is not True
        or header.get("n_records") != 288
        or not isinstance(metadata, dict)
        or metadata.get("mode") != "agribrain"
        or metadata.get("scenario") != scenario
        or metadata.get("benchmark_seed") != seed
        or metadata.get("seed") != seed
        or metadata.get("episode_index") != 3
        or metadata.get("learning_enabled") is not False
        or metadata.get("episode_phase") != "frozen_evaluation"
        or metadata.get("trace_schema_version") != TRACE_SCHEMA_VERSION
    ):
        raise RuntimeError(f"{path} is not the canonical retained H3 ledger")
    _validate_spoilage_estimator(
        metadata.get("spoilage_estimator"), mode="agribrain",
        where=f"{path} header",
    )
    _validate_spoilage_estimator(
        cell.get("spoilage_estimator"), mode="agribrain",
        where=f"{path} summary cell",
    )
    _validate_latent_spoilage_model(
        metadata.get("latent_spoilage_model"), where=f"{path} header",
        effective_k_ref=metadata.get("effective_k_ref"),
        effective_ea_r=metadata.get("effective_Ea_R"),
    )
    _validate_latent_spoilage_model(
        cell.get("latent_spoilage_model"), where=f"{path} summary cell",
        effective_k_ref=metadata.get("effective_k_ref"),
        effective_ea_r=metadata.get("effective_Ea_R"),
    )
    outcome_contract = metadata.get("outcome_equation_contract")
    expected_outcome_contract = expected_publication_outcome_equation_contract(
        benchmark_seed=seed,
        scenario=scenario,
    )
    try:
        validate_outcome_equation_contract(
            outcome_contract,
            where=f"{path} header outcome_equation_contract",
            expected_contract=expected_outcome_contract,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    # H3 ledgers are publication decision ledgers with additional treatment
    # fields.  Run the complete shared validator before applying the H3 dose
    # checks below so a hash-valid stress artifact cannot bypass the locked
    # policy equation, keyed categorical draw, paired context ablation,
    # independent-DGP outcome, policy-side spoilage estimate, context
    # integration, or endpoint reconstruction enforced for the core panel.
    generic_ledger_summary = validate_ledger(
        path,
        mode="agribrain",
        scenario=scenario,
        benchmark_seed=seed,
        expected_outcome_equation_contract=expected_outcome_contract,
    )
    validate_learner_snapshot_binding(
        cell,
        generic_ledger_summary["learner_snapshots"],
        mode="agribrain",
        where=str(path),
    )
    episode_evidence_contract = metadata.get("episode_evidence_contract")
    try:
        validate_episode_evidence_contract(
            episode_evidence_contract,
            where=f"{path} header episode_evidence_contract",
            expected_contract=expected_publication_episode_evidence_contract(),
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    if not math.isclose(
        float(metadata.get("effective_k_ref", math.nan)),
        float(outcome_contract["arrhenius"]["effective_k_ref"]),
        rel_tol=1e-15,
        abs_tol=1e-15,
    ) or not math.isclose(
        float(metadata.get("effective_Ea_R", math.nan)),
        float(outcome_contract["arrhenius"]["effective_ea_over_r"]),
        rel_tol=1e-15,
        abs_tol=1e-15,
    ):
        raise RuntimeError(f"{path} effective Arrhenius metadata/contract mismatch")

    if stressor == "nominal":
        expected_treatment = {
            "stressor": "nominal",
            "n_steps": 288,
            "data_observation_treatment": False,
            "delay_steps": 0,
            "missing_count": 0,
        }
    else:
        expected_treatment = _expected_observation_treatment(
            scenario=scenario, stressor=stressor, seed=seed,
        )
    if metadata.get("observation_treatment") != expected_treatment:
        raise RuntimeError(f"{path} header does not contain the locked H3 dose")
    expected_environment_id = _stream_id(seed, scenario, 3, "environment")
    expected_policy_id = _stream_id(seed, scenario, 3, "policy")
    if (
        metadata.get("environment_stream_id") != expected_environment_id
        or metadata.get("stochastic_stream_id") != expected_environment_id
        or metadata.get("policy_stream_id") != expected_policy_id
    ):
        raise RuntimeError(f"{path} header has incorrect retained stream identity")
    for field in (
        "context_prior_sha256", "policy_theta_initial_sha256",
        "latent_environment_sha256", "observed_policy_input_sha256",
        "demand_observation_sha256", "demand_forecast_method",
        "supply_forecast_method", "dispatch_opportunity_count",
        "dispatch_cadence_hours", "spoilage_estimator",
        "latent_spoilage_model",
    ):
        if metadata.get(field) != cell.get(field):
            raise RuntimeError(f"{path} header/cell field mismatch: {field}")

    literal_sha = _sha256_file(path)
    if (
        cell.get("decision_ledger_path") != canonical_path
        or cell.get("decision_ledger_sha256") != literal_sha
        or cell.get("decision_ledger_n_records") != 288
    ):
        raise RuntimeError(f"{path} cell-to-ledger literal-byte binding mismatch")

    records: list[dict[str, Any]] = []
    leaves: list[str] = []
    required = {
        "step_index", "hour", "mode", "scenario", "action", "action_idx",
        "reward", "waste",
        "slca", "ari", "carbon_kg", "rho_outcome_environmental",
        "rho_policy_observed", "shock_g", "temp_policy_observed",
        "temp_outcome_environmental", "rh_policy_observed",
        "rh_outcome_environmental", "inventory_policy_observed",
        "inventory_outcome_environmental", "demand_policy_observed",
        "demand_outcome_environmental", "demand_forecast_policy_observed",
        "supply_forecast_policy_observed", "bollinger_regime_flag",
        "regime_logit_bias",
        "price_signal", "transport_multiplier_outcome_environmental",
        "h3_stressor",
        "h3_data_observation_treatment", "h3_temp_noise_c",
        "h3_rh_noise_pct", "h3_missing_observation",
        "h3_telemetry_source_step_index",
        "h3_fault_injection_scheduled_opportunity",
        "h3_fault_injection_triggered",
        "h3_fault_injected_tool_result_count",
        "context_modifier", "context_action_changed",
        "context_influence_active", "context_influence_counted",
        *ACTIVITY_STEP_FIELDS,
    }
    for index, line in enumerate(lines[1:]):
        stored = _strict_ledger_object(line, path=path, line_number=index + 2)
        missing = required.difference(stored)
        if missing:
            raise RuntimeError(
                f"{path}:{index + 2} lacks H3 fields {sorted(missing)}"
            )
        leaf = stored.pop("_leaf", None)
        if not _valid_sha256(leaf) or leaf != hashlib.sha256(json.dumps(
            stored, sort_keys=True, separators=(",", ":"), default=str,
        ).encode("utf-8")).hexdigest():
            raise RuntimeError(f"{path}:{index + 2} leaf hash mismatch")
        leaves.append(str(leaf))
        records.append(stored)
        try:
            validate_recorded_step_outcomes(
                stored,
                outcome_contract,
                where=f"{path}:{index + 2}",
                contract_validated=True,
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        if (
            stored.get("step_index") != index
            or not math.isclose(float(stored.get("hour", -1)), index * 0.25,
                                abs_tol=1e-9)
            or stored.get("mode") != "agribrain"
            or stored.get("scenario") != scenario
            or stored.get("h3_stressor") != stressor
            or stored.get("h3_data_observation_treatment") is not (
                stressor not in {"nominal", "mcp_fault_injection"}
            )
        ):
            raise RuntimeError(f"{path}:{index + 2} H3 identity/cadence mismatch")
        expected_ari = (
            (1.0 - float(stored["waste"])) * float(stored["slca"])
            * (1.0 - float(stored["rho_outcome_environmental"]))
        )
        if not math.isclose(
            float(stored["ari"]), expected_ari, rel_tol=1e-12, abs_tol=1e-12,
        ):
            raise RuntimeError(f"{path}:{index + 2} ARI equation mismatch")
        scheduled = stored["h3_fault_injection_scheduled_opportunity"]
        triggered = stored["h3_fault_injection_triggered"]
        replaced = stored["h3_fault_injected_tool_result_count"]
        if not isinstance(scheduled, bool) or not isinstance(triggered, bool) or (
            isinstance(replaced, bool) or not isinstance(replaced, int) or replaced < 0
        ):
            raise RuntimeError(f"{path}:{index + 2} invalid H3 fault dose")
        expected_scheduled = (
            stressor in {"mcp_fault_injection", "compounded"}
            and int(float(stored["hour"])) % 11 == 0
        )
        if scheduled is not expected_scheduled or (triggered and not scheduled):
            raise RuntimeError(f"{path}:{index + 2} fault schedule mismatch")
        if (triggered and replaced <= 0) or (not triggered and replaced != 0):
            raise RuntimeError(f"{path}:{index + 2} fault exposure mismatch")

    merkle_root = _ledger_merkle_root(leaves)
    if (
        header.get("merkle_root") != merkle_root
        or cell.get("decision_ledger_merkle_root") != merkle_root
    ):
        raise RuntimeError(f"{path} Merkle-root binding mismatch")

    for field in (
        "effective_k_ref", "effective_Ea_R", "scenario_onset_offset_hours",
    ):
        try:
            value = float(metadata[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"{path} lacks numeric metadata {field}") from exc
        if not math.isfinite(value):
            raise RuntimeError(f"{path} has non-finite metadata {field}")
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
            float(record["demand_forecast_policy_observed"])
            for record in records
        ],
        "supply_forecast_policy_observed": [
            float(record["supply_forecast_policy_observed"])
            for record in records
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
        "price_signal": [float(record["price_signal"]) for record in records],
    }
    for field, payload in (
        ("latent_environment_sha256", latent_payload),
        ("observed_policy_input_sha256", observed_payload),
        ("demand_observation_sha256", demand_payload),
    ):
        expected_hash = _canonical_object_sha256(payload)
        if metadata.get(field) != expected_hash or cell.get(field) != expected_hash:
            raise RuntimeError(f"{path} {field} does not reconstruct from decisions")

    temp_noise = [float(record["h3_temp_noise_c"]) for record in records]
    rh_noise = [float(record["h3_rh_noise_pct"]) for record in records]
    missing_mask = np.asarray([
        bool(record["h3_missing_observation"]) for record in records
    ], dtype=np.uint8)
    source_steps = [
        int(record["h3_telemetry_source_step_index"]) for record in records
    ]
    delay = int(expected_treatment["delay_steps"])
    expected_sources = [max(index - delay, 0) for index in range(288)]
    if source_steps != expected_sources:
        raise RuntimeError(f"{path} telemetry-delay exposure does not reconstruct")

    def array_hash(values: list[float]) -> str:
        return hashlib.sha256(json.dumps(
            values, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest()

    noise_expected = stressor in {"sensor_noise", "compounded"}
    if noise_expected:
        if (
            array_hash(temp_noise) != expected_treatment["temp_noise_sha256"]
            or array_hash(rh_noise) != expected_treatment["rh_noise_sha256"]
        ):
            raise RuntimeError(f"{path} sensor-noise exposure hash mismatch")
    elif any(value != 0.0 for value in (*temp_noise, *rh_noise)):
        raise RuntimeError(f"{path} records undeclared sensor-noise exposure")
    missing_expected = stressor in {"missing_data", "compounded"}
    if missing_expected:
        if (
            int(np.count_nonzero(missing_mask))
            != int(expected_treatment["missing_count"])
            or hashlib.sha256(missing_mask.tobytes()).hexdigest()
            != expected_treatment["missing_mask_sha256"]
        ):
            raise RuntimeError(f"{path} missing-data exposure hash mismatch")
    elif np.count_nonzero(missing_mask):
        raise RuntimeError(f"{path} records undeclared missing-data exposure")

    scheduled_count = sum(
        bool(record["h3_fault_injection_scheduled_opportunity"])
        for record in records
    )
    trigger_count = sum(
        bool(record["h3_fault_injection_triggered"]) for record in records
    )
    replaced_count = sum(
        int(record["h3_fault_injected_tool_result_count"])
        for record in records
    )
    try:
        episode_evidence = reconstruct_episode_evidence(
            records,
            episode_evidence_contract,
            where=str(path),
            contract_validated=True,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    reconstructed = {
        "ari": float(math.fsum(float(record["ari"]) for record in records) / 288),
        "waste": float(math.fsum(float(record["waste"]) for record in records) / 288),
        "slca": float(math.fsum(float(record["slca"]) for record in records) / 288),
        "carbon": float(math.fsum(float(record["carbon_kg"]) for record in records)),
        "equity": float(compute_equity([
            float(record["slca"]) for record in records
        ])),
        "rle": float(compute_rle(
            [float(record["rho_outcome_environmental"]) for record in records],
            [str(record["action"]) for record in records],
        )),
        "fault_injection_scheduled_opportunity_steps": scheduled_count,
        "fault_injection_trigger_steps": trigger_count,
        "fault_injected_tool_result_count": replaced_count,
        "observation_treatment": expected_treatment,
        "latent_environment_sha256": metadata["latent_environment_sha256"],
        "latent_environment_rows": [(
            float(record["temp_outcome_environmental"]),
            float(record["rh_outcome_environmental"]),
            float(record["rho_outcome_environmental"]),
            float(record["inventory_outcome_environmental"]),
            float(record["demand_outcome_environmental"]),
            float(record["transport_multiplier_outcome_environmental"]),
        ) for record in records],
        "observed_policy_rows": [(
            float(record["temp_policy_observed"]),
            float(record["rh_policy_observed"]),
            float(record["inventory_policy_observed"]),
            float(record["demand_forecast_policy_observed"]),
            float(record["supply_forecast_policy_observed"]),
        ) for record in records],
        "temp_noise": temp_noise,
        "rh_noise": rh_noise,
        "missing_mask": [bool(value) for value in missing_mask],
        "source_steps": source_steps,
        "episode_evidence": episode_evidence,
    }
    for field in (
        "ari", "waste", "slca", "rle", "carbon", "equity",
        "fault_injection_scheduled_opportunity_steps",
        "fault_injection_trigger_steps", "fault_injected_tool_result_count",
    ):
        _assert_close(
            cell.get(field), float(reconstructed[field]),
            where=f"{path}/{field} ledger reconstruction",
        )
    _assert_close(
        cell.get("decision_latency_ms"),
        float(episode_evidence["mean_decision_latency_ms"]),
        where=f"{path}/decision_latency_ms ledger reconstruction",
    )
    expected_message_count = generic_ledger_summary["episode_evidence"][
        "message_count"
    ]
    if cell.get("message_count") != expected_message_count:
        raise RuntimeError(
            f"{path}/message_count differs from decision-ledger reconstruction"
        )
    for field in PROTOCOL_COUNT_FIELDS:
        if cell.get(field) != episode_evidence[field]:
            raise RuntimeError(
                f"{path}/{field} differs from decision-ledger reconstruction"
            )
    if stressor != "nominal" and cell.get("observation_treatment") != (
        expected_treatment
    ):
        raise RuntimeError(f"{path} summary treatment differs from ledger dose")
    return reconstructed


def _validate_h3_observation_transform(
    *, nominal: dict[str, Any], stressed: dict[str, Any],
    stressor: str, where: str,
) -> None:
    """Rebuild the exact observed temp/RH stream from dose primitives."""
    if (
        stressed["latent_environment_sha256"]
        != nominal["latent_environment_sha256"]
        or stressed["latent_environment_rows"] != nominal["latent_environment_rows"]
    ):
        raise RuntimeError(f"{where} H3 treatment changed latent truth")

    nominal_rows = nominal["observed_policy_rows"]
    actual_rows = stressed["observed_policy_rows"]
    base_temp = np.asarray([row[0] for row in nominal_rows], dtype=float)
    base_rh = np.asarray([row[1] for row in nominal_rows], dtype=float)
    temp = base_temp.copy()
    rh = base_rh.copy()
    if stressor in {"sensor_noise", "compounded"}:
        temp += np.asarray(stressed["temp_noise"], dtype=float)
        rh = np.clip(
            rh + np.asarray(stressed["rh_noise"], dtype=float), 15.0, 100.0,
        )
    if stressor in {"missing_data", "compounded"}:
        mask = np.asarray(stressed["missing_mask"], dtype=bool)
        temp[mask] = np.nan
        rh[mask] = np.nan
        # Match pandas Series.ffill() exactly for this numeric one-dimensional
        # stream; the locked dose forces mask[0] false.
        for index in range(1, len(temp)):
            if np.isnan(temp[index]):
                temp[index] = temp[index - 1]
            if np.isnan(rh[index]):
                rh[index] = rh[index - 1]
    if stressor in {"telemetry_delay", "compounded"}:
        temp = np.concatenate((np.repeat(temp[0], 4), temp[:-4]))
        rh = np.concatenate((np.repeat(rh[0], 4), rh[:-4]))

    expected_rows = [
        (
            float(temp[index]), float(rh[index]),
            float(nominal_rows[index][2]), float(nominal_rows[index][3]),
            float(nominal_rows[index][4]),
        )
        for index in range(288)
    ]
    for index, (actual, expected) in enumerate(zip(actual_rows, expected_rows)):
        if any(
            not math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
            for left, right in zip(actual, expected)
        ):
            raise RuntimeError(
                f"{where} policy observation does not reconstruct from H3 dose "
                f"at decision {index}"
            )


def _require_h3_directory(path: Path, *, where: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"H3 inventory entry is not a real directory: {where}")


def _require_h3_regular_file(path: Path, *, where: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"H3 inventory entry is not a regular file: {where}")


def _require_exact_h3_entries(
    root: Path,
    expected_names: set[str],
    *,
    where: str,
) -> dict[str, Path]:
    _require_h3_directory(root, where=where)
    entries = {path.name: path for path in root.iterdir()}
    found_names = set(entries)
    if found_names != expected_names:
        raise RuntimeError(
            f"H3 inventory mismatch at {where}: "
            f"missing={sorted(expected_names - found_names)}, "
            f"unexpected={sorted(found_names - expected_names)}"
        )
    return entries


def _validate_h3_ledger_inventory_shape(
    h3_ledger_root: Path,
) -> None:
    """Validate the exact full-evidence inventory emitted by ``hpc_stress.sh``.

    Receipt, manifest, and ledger contents are validated by their dedicated
    provenance and evidence gates after this fail-closed topology check.
    """

    expected_scenarios = set(SCENARIOS)
    scenario_entries = _require_exact_h3_entries(
        h3_ledger_root,
        expected_scenarios,
        where="H3 ledger root",
    )
    expected_seed_dirs = {f"seed_{seed}" for seed in EXPECTED_SEEDS}
    scenario_evidence_names = {
        "complete_episode_evidence_manifest.json",
        "runtime_receipts",
    }
    final_ledger_count = 0
    adaptation_ledger_count = 0
    episode_archive_count = 0

    for scenario in SCENARIOS:
        scenario_root = scenario_entries[scenario]
        scenario_children = _require_exact_h3_entries(
            scenario_root,
            set(STRESSORS) | scenario_evidence_names,
            where=f"H3 scenario {scenario}",
        )
        _require_h3_regular_file(
            scenario_children["complete_episode_evidence_manifest.json"],
            where=f"{scenario}/complete_episode_evidence_manifest.json",
        )
        receipt_root = scenario_children["runtime_receipts"]
        _require_h3_directory(
            receipt_root,
            where=f"{scenario}/runtime_receipts",
        )
        receipt_entries = list(receipt_root.iterdir())
        if not receipt_entries:
            raise RuntimeError(
                f"H3 runtime receipt inventory is empty: {scenario}"
            )
        for receipt_path in receipt_entries:
            if re.fullmatch(
                r"job_[1-9][0-9]*__restart_[0-9]+\.json",
                receipt_path.name,
            ) is None:
                raise RuntimeError(
                    "H3 runtime receipt has an unexpected name: "
                    f"{scenario}/runtime_receipts/{receipt_path.name}"
                )
            _require_h3_regular_file(
                receipt_path,
                where=f"{scenario}/runtime_receipts/{receipt_path.name}",
            )

        for stressor in STRESSORS:
            stressor_root = scenario_children[stressor]
            seed_entries = _require_exact_h3_entries(
                stressor_root,
                expected_seed_dirs,
                where=f"H3 stressor {scenario}/{stressor}",
            )
            for seed in EXPECTED_SEEDS:
                seed_name = f"seed_{seed}"
                seed_root = seed_entries[seed_name]
                arm_name = f"agribrain__{scenario}"
                final_name = f"{arm_name}.jsonl"
                seed_children = _require_exact_h3_entries(
                    seed_root,
                    {
                        final_name,
                        "adaptation_episode_ledgers",
                        "complete_episode_evidence",
                    },
                    where=f"H3 seed {scenario}/{stressor}/{seed_name}",
                )
                _require_h3_regular_file(
                    seed_children[final_name],
                    where=f"{scenario}/{stressor}/{seed_name}/{final_name}",
                )
                final_ledger_count += 1

                evidence_specs = (
                    (
                        "adaptation_episode_ledgers",
                        {f"episode_{index}.jsonl.gz" for index in range(3)},
                    ),
                    (
                        "complete_episode_evidence",
                        {f"episode_{index}.json.gz" for index in range(4)},
                    ),
                )
                for evidence_name, expected_files in evidence_specs:
                    evidence_root = seed_children[evidence_name]
                    arm_entries = _require_exact_h3_entries(
                        evidence_root,
                        {arm_name},
                        where=(
                            f"H3 evidence {scenario}/{stressor}/{seed_name}/"
                            f"{evidence_name}"
                        ),
                    )
                    episode_entries = _require_exact_h3_entries(
                        arm_entries[arm_name],
                        expected_files,
                        where=(
                            f"H3 evidence arm {scenario}/{stressor}/{seed_name}/"
                            f"{evidence_name}/{arm_name}"
                        ),
                    )
                    for filename, path in episode_entries.items():
                        _require_h3_regular_file(
                            path,
                            where=(
                                f"{scenario}/{stressor}/{seed_name}/"
                                f"{evidence_name}/{arm_name}/{filename}"
                            ),
                        )
                    if evidence_name == "adaptation_episode_ledgers":
                        adaptation_ledger_count += len(episode_entries)
                    else:
                        episode_archive_count += len(episode_entries)

    expected_arms = len(SCENARIOS) * len(STRESSORS) * len(EXPECTED_SEEDS)
    expected_counts = {
        "final episode ledger": (final_ledger_count, expected_arms),
        "adaptation episode ledger": (
            adaptation_ledger_count,
            expected_arms * 3,
        ),
        "complete episode archive": (episode_archive_count, expected_arms * 4),
    }
    for label, (actual_count, expected_count) in expected_counts.items():
        if actual_count != expected_count:
            raise RuntimeError(
                f"H3 {label} count mismatch: "
                f"{actual_count} != {expected_count}"
            )


def _h3_ledger_set_binding(seed_panel: dict[str, Any]) -> dict[str, Any]:
    records = []
    for seed in EXPECTED_SEEDS:
        cell = _panel_cell(seed_panel, seed, "agribrain")
        record = {
            "seed": int(seed),
            "path": cell.get("decision_ledger_path"),
            "sha256": cell.get("decision_ledger_sha256"),
            "merkle_root": cell.get("decision_ledger_merkle_root"),
            "n_records": cell.get("decision_ledger_n_records"),
        }
        if (
            not isinstance(record["path"], str)
            or not _valid_sha256(record["sha256"])
            or not _valid_sha256(record["merkle_root"])
            or record["n_records"] != 288
        ):
            raise RuntimeError(f"invalid H3 ledger-set member for seed {seed}")
        records.append(record)
    return {
        "count": len(records),
        "decision_count": len(records) * 288,
        "sha256": _canonical_object_sha256(records),
    }


def _validate_frozen_h3_learner_state(cell: Any, *, where: str) -> None:
    freeze = cell.get("learner_freeze_summary") if isinstance(cell, dict) else None
    if not isinstance(freeze, dict) or (
        freeze.get("learners_frozen") is not True
        or freeze.get("learner_phase") != "frozen_evaluation"
        or freeze.get("freeze_reason") != "retained_episode_3"
        or freeze.get("context_matrix_frozen") is not True
        or freeze.get("reward_shaping_frozen") is not True
        or freeze.get("external_policy_learners_frozen") != 0
    ):
        raise RuntimeError(f"{where} lacks exact frozen-episode learner evidence")
    role_panel = freeze.get("policy_delta_frozen_by_role")
    if not isinstance(role_panel, dict) or set(role_panel) != set(
        DECISION_OWNER_ROLES
    ) or any(value is not True for value in role_panel.values()):
        raise RuntimeError(f"{where} did not freeze every decision-owner learner")


def _validate_primary_nominal_binding(
    *, seed_root: Path, scenario: str, seed: int,
    stress_cell: dict[str, Any], where: str,
) -> None:
    """Bind the copied H3 nominal cell to the exact raw primary envelope."""

    primary_path = seed_root / f"seed_{seed}.json"
    primary_bytes = primary_path.read_bytes()
    primary_payload = json.loads(
        primary_bytes.decode("utf-8"), parse_constant=_reject_constant,
    )
    try:
        primary_cell = primary_payload["scenarios"][scenario]["agribrain"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"{primary_path} lacks primary AGRI-BRAIN cell for {scenario}"
        ) from exc
    if stress_cell.get("primary_seed_envelope_sha256") != hashlib.sha256(
        primary_bytes
    ).hexdigest():
        raise RuntimeError(f"{where} is not bound to the primary seed envelope")
    if stress_cell.get("primary_nominal_cell_sha256") != (
        _canonical_object_sha256(primary_cell)
    ):
        raise RuntimeError(f"{where} is not bound to the primary nominal cell")

    numeric_mapping = {
        "ari": "ari", "waste": "waste", "slca": "slca", "rle": "rle",
        "carbon": "carbon", "equity": "equity",
        "constraint_violation_rate": "constraint_violation_rate",
        "decision_latency_ms": "mean_decision_latency_ms",
        "downstream_violation_rate": "downstream_violation_rate",
        "contained_violation_rate": "contained_violation_rate",
        "dispatch_cadence_hours": "dispatch_cadence_hours",
    }
    for stress_field, primary_field in numeric_mapping.items():
        _assert_close(
            stress_cell.get(stress_field), float(primary_cell.get(primary_field, 0.0)),
            where=f"{where}/{stress_field} primary binding",
        )
    exact_fields = (
        "trace_schema_version", "benchmark_seed", "episode_index",
        "environment_stream_id", "policy_stream_id", "stochastic_stream_id",
        "context_prior_sha256", "policy_theta_initial_sha256",
        "latent_environment_sha256", "observed_policy_input_sha256",
        "demand_observation_sha256", "demand_forecast_method",
        "supply_forecast_method", "learning_enabled", "episode_phase",
        "dispatch_opportunity_count", *PROTOCOL_COUNT_FIELDS,
        "fault_injection_scheduled_opportunity_steps",
        "fault_injection_trigger_steps", "fault_injected_tool_result_count",
        "spoilage_estimator", "latent_spoilage_model",
    )
    for field in exact_fields:
        if stress_cell.get(field) != primary_cell.get(field, 0):
            raise RuntimeError(f"{where}/{field} differs from primary nominal")
    for field in (
        "learner_summary", "theta_learner_summary",
        "reward_shaping_learner_summary", "learner_freeze_summary",
    ):
        if stress_cell.get(field) != primary_cell.get(field):
            raise RuntimeError(f"{where}/{field} differs from primary nominal")


def validate_seed_inputs(
    seed_root: Path, *, source_commit: str, run_tag: str,
    submission_receipt: Mapping[str, Any] | None = None,
) -> None:
    expected_names = {f"seed_{seed}.json" for seed in EXPECTED_SEEDS}
    found_names = {path.name for path in seed_root.glob("seed_*.json")}
    if found_names != expected_names:
        raise RuntimeError(
            "seed envelope inventory mismatch: "
            f"missing={sorted(expected_names - found_names)}, "
            f"unexpected={sorted(found_names - expected_names)}"
        )

    for seed in EXPECTED_SEEDS:
        path = seed_root / f"seed_{seed}.json"
        payload = _load_json(path)
        _validate_identity(
            payload.get("_meta"), source_commit=source_commit,
            run_tag=run_tag, where=path,
        )
        meta = payload["_meta"]
        if submission_receipt is not None:
            declared = submission_receipt.get("slurm_dag", {}).get(
                "seed_array", {}
            ).get("seeds")
            if not isinstance(declared, list) or seed not in declared:
                raise RuntimeError(
                    f"{path} seed/task index differs from the submission receipt"
                )
            seed_index = declared.index(seed)
            try:
                validate_core_array_provenance(
                    meta.get("execution_provenance"),
                    stage="core_seed_array",
                    logical_task_index=seed_index,
                    submission_receipt=submission_receipt,
                )
            except ValueError as exc:
                raise RuntimeError(
                    f"{path} Slurm/source binding is invalid: {exc}"
                ) from exc
        if payload.get("trace_schema_version") != TRACE_SCHEMA_VERSION or (
            meta.get("trace_schema_version") != TRACE_SCHEMA_VERSION
        ):
            raise RuntimeError(f"{path} uses an obsolete trace schema")
        if meta.get("episode_scope") != EPISODE_SCOPE:
            raise RuntimeError(f"{path} has an incorrect episode scope")
        if meta.get("decision_history_scope") != HISTORY_SCOPE:
            raise RuntimeError(f"{path} has an incorrect history scope")
        expected_accounting = _expected_seed_episode_accounting()
        if meta.get("episode_accounting") != expected_accounting:
            raise RuntimeError(
                f"{path} has incorrect retained/executed episode accounting"
            )
        if payload.get("seed") != seed:
            raise RuntimeError(f"{path} seed field does not match its filename")
        scenarios = payload.get("scenarios")
        if not isinstance(scenarios, dict) or set(scenarios) != set(SCENARIOS):
            raise RuntimeError(f"{path} does not contain the exact scenario panel")
        context_prior_hashes: dict[str, set[str]] = {
            mode: set() for mode in MODES
        }
        policy_prior_hashes: dict[str, set[str]] = {
            mode: set() for mode in MODES
        }
        for scenario in SCENARIOS:
            cells = scenarios.get(scenario)
            if not isinstance(cells, dict) or set(cells) != set(MODES):
                raise RuntimeError(
                    f"{path} {scenario!r} does not contain the exact mode panel"
                )
            latent_hashes: set[str] = set()
            latent_model_records: set[str] = set()
            residual_observed_hashes: set[str] = set()
            no_pinn_observed_hashes: set[str] = set()
            for mode, cell in cells.items():
                where = f"{path}:{scenario}/{mode}"
                _validate_spoilage_estimator(
                    cell.get("spoilage_estimator"), mode=mode, where=where,
                )
                _validate_latent_spoilage_model(
                    cell.get("latent_spoilage_model"), where=where,
                    effective_k_ref=cell.get("effective_k_ref"),
                    effective_ea_r=cell.get("effective_Ea_R"),
                )
                _validate_context_execution_counts(
                    cell, where=where,
                )
                _validate_learner_provenance(cell, mode=mode, where=where)
                if cell.get("trace_schema_version") != TRACE_SCHEMA_VERSION:
                    raise RuntimeError(f"{path}:{scenario}/{mode} schema mismatch")
                if cell.get("benchmark_seed") != seed or cell.get("episode_index") != 3:
                    raise RuntimeError(f"{path}:{scenario}/{mode} retained-episode mismatch")
                if cell.get("learning_enabled") is not False:
                    raise RuntimeError(f"{where} retained episode allowed updates")
                expected_phase = (
                    "fixed_evaluation" if mode == "static" else "frozen_evaluation"
                )
                if cell.get("episode_phase") != expected_phase:
                    raise RuntimeError(f"{where} has incorrect evaluation phase")
                freeze = cell.get("learner_freeze_summary") or {}
                if freeze.get("learners_frozen") is not True:
                    raise RuntimeError(f"{where} lacks learner-freeze evidence")
                if cell.get("demand_forecast_method") != "holt_linear" or (
                    cell.get("supply_forecast_method") != "persistence"
                ):
                    raise RuntimeError(f"{where} forecast lock mismatch")
                expected_environment_id = (
                    f"seed={seed};scenario={scenario};episode=3;stream=environment"
                )
                expected_policy_id = (
                    f"seed={seed};scenario={scenario};episode=3;stream=policy"
                )
                if cell.get("environment_stream_id") != expected_environment_id or (
                    cell.get("stochastic_stream_id") != expected_environment_id
                ) or cell.get("policy_stream_id") != expected_policy_id:
                    raise RuntimeError(
                        f"{path}:{scenario}/{mode} has incorrect retained stream IDs"
                    )
                if cell.get("dispatch_opportunity_count") != 288 or not math.isclose(
                    float(cell.get("dispatch_cadence_hours", -1.0)), 0.25,
                    abs_tol=1e-12,
                ):
                    raise RuntimeError(f"{path}:{scenario}/{mode} dispatch accounting mismatch")
                latent_hash = cell.get("latent_environment_sha256")
                observed_hash = cell.get("observed_policy_input_sha256")
                demand_hash = cell.get("demand_observation_sha256")
                if not isinstance(latent_hash, str) or not re.fullmatch(
                    r"[0-9a-f]{64}", latent_hash,
                ) or not isinstance(observed_hash, str) or not re.fullmatch(
                    r"[0-9a-f]{64}", observed_hash,
                ) or not isinstance(demand_hash, str) or not re.fullmatch(
                    r"[0-9a-f]{64}", demand_hash,
                ):
                    raise RuntimeError(f"{path}:{scenario}/{mode} has invalid state hashes")
                context_hash = cell.get("context_prior_sha256")
                policy_hash = cell.get("policy_theta_initial_sha256")
                if not _valid_sha256(context_hash) or not _valid_sha256(policy_hash):
                    raise RuntimeError(
                        f"{path}:{scenario}/{mode} has invalid policy-prior hashes"
                    )
                latent_hashes.add(latent_hash)
                latent_model_records.add(json.dumps(
                    cell["latent_spoilage_model"],
                    sort_keys=True, separators=(",", ":"), allow_nan=False,
                ))
                if capabilities_for(mode).spoilage_residual:
                    residual_observed_hashes.add(observed_hash)
                else:
                    no_pinn_observed_hashes.add(observed_hash)
                context_prior_hashes[mode].add(context_hash)
                policy_prior_hashes[mode].add(policy_hash)
            if len(latent_hashes) != 1:
                raise RuntimeError(
                    f"{path}:{scenario} H1 arms do not share latent truth"
                )
            if len(latent_model_records) != 1:
                raise RuntimeError(
                    f"{path}:{scenario} H1 arms do not share the latent DGP"
                )
            if len(residual_observed_hashes) != 1:
                raise RuntimeError(
                    f"{path}:{scenario} residual-enabled H1 arms do not share "
                    "policy observations"
                )
            if no_pinn_observed_hashes and (
                len(no_pinn_observed_hashes) != 1
                or not no_pinn_observed_hashes.isdisjoint(
                    residual_observed_hashes
                )
            ):
                raise RuntimeError(
                    f"{path}:{scenario} no-PINN policy observation does not "
                    "expose its mechanistic-only spoilage estimate"
                )
        for mode in MODES:
            if len(context_prior_hashes[mode]) != 1:
                raise RuntimeError(
                    f"{path}:{mode} context prior changes across scenarios"
                )
            if len(policy_prior_hashes[mode]) != 1:
                raise RuntimeError(
                    f"{path}:{mode} base-policy prior changes across scenarios"
                )
        for magnitude in (10, 25, 50):
            learned = f"agribrain_pert_{magnitude}"
            frozen = f"agribrain_pert_{magnitude}_static"
            if learned not in context_prior_hashes or frozen not in context_prior_hashes:
                continue
            if context_prior_hashes[learned] != context_prior_hashes[frozen]:
                raise RuntimeError(
                    f"{path} sensitivity pair {learned}/{frozen} has "
                    "different initial context priors"
                )
        primary_modes = (
            "static", "hybrid_rl", "no_pinn", "no_slca", "no_context",
            "mcp_only", "pirag_only", "agribrain",
        )
        primary_policy_hashes = {
            next(iter(policy_prior_hashes[mode]))
            for mode in primary_modes if mode in policy_prior_hashes
        }
        if len(primary_policy_hashes) != 1:
            raise RuntimeError(
                f"{path} primary H1/H2 arms have different base-policy priors"
            )
        traces = payload.get("traces")
        if not isinstance(traces, dict) or set(traces) != set(SCENARIOS):
            raise RuntimeError(f"{path} lacks the exact trace scenario panel")
        for scenario in SCENARIOS:
            mode_panel = traces.get(scenario)
            if not isinstance(mode_panel, dict) or set(mode_panel) != TRACE_MODES:
                raise RuntimeError(f"{path}:{scenario} trace-mode panel mismatch")
            for mode, trace_cell in mode_panel.items():
                _validate_trace_cell(
                    trace_cell, where=f"{path}:{scenario}/{mode}",
                )
        if payload.get("_trace_failures"):
            raise RuntimeError(f"{path} reports trace serialization failures")


def validate_stress_inputs(
    stress_root: Path, *, seed_root: Path,
    source_commit: str, run_tag: str,
    h3_ledger_root: Path | None = None,
    primary_ledger_root: Path | None = None,
    submission_receipt: Mapping[str, Any] | None = None,
    h3_evidence_scope: str = "complete",
) -> None:
    if h3_evidence_scope not in ("complete", "archived-subset"):
        raise ValueError(
            f"unsupported H3 evidence scope: {h3_evidence_scope!r}"
        )
    if not seed_root.is_dir():
        raise RuntimeError(f"primary seed root does not exist: {seed_root}")
    if h3_ledger_root is None:
        h3_ledger_root = stress_root.parent / "decision_ledger_h3"
    if not h3_ledger_root.is_dir():
        raise RuntimeError(f"H3 decision-ledger root does not exist: {h3_ledger_root}")
    if h3_evidence_scope == "complete":
        _validate_h3_ledger_inventory_shape(h3_ledger_root)
    # "archived-subset": the publication archive manifests exactly the
    # stressed decision ledgers, not the adaptation/episode evidence tree,
    # so its extracted copy cannot satisfy the full-evidence topology check.
    # Completeness of that tree is validated on the live results tree by the
    # publisher in the same chain and is bound into the semantic validation
    # receipt, which archive consumers verify independently; every ledger the
    # archive does contain is still opened and reconstructed below.
    if primary_ledger_root is not None and not primary_ledger_root.is_dir():
        raise RuntimeError(
            f"primary decision-ledger root does not exist: {primary_ledger_root}"
        )
    ledger_reconstructions: dict[tuple[str, str, int], dict[str, Any]] = {}
    found = {path.name for path in stress_root.iterdir() if path.is_dir()}
    if found != set(SCENARIOS):
        raise RuntimeError(
            "stress scenario inventory mismatch: "
            f"missing={sorted(set(SCENARIOS) - found)}, "
            f"unexpected={sorted(found - set(SCENARIOS))}"
        )
    required = (
        "stress_summary.json", "stress_degradation.csv",
        "stress_passfail.csv", "stress_h3_test.json",
    )
    for scenario in SCENARIOS:
        root = stress_root / scenario
        for name in required:
            path = root / name
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"missing or empty stress artifact: {path}")

        summary_path = root / "stress_summary.json"
        summary = _load_json(summary_path)
        _validate_identity(
            summary.get("meta"), source_commit=source_commit,
            run_tag=run_tag, where=summary_path,
        )
        stress_meta = summary["meta"]
        if submission_receipt is not None:
            declared = submission_receipt.get("slurm_dag", {}).get(
                "stress_array", {}
            ).get("scenarios")
            if not isinstance(declared, list) or scenario not in declared:
                raise RuntimeError(
                    f"{summary_path} scenario/task index differs from the submission receipt"
                )
            scenario_index = declared.index(scenario)
            try:
                validate_core_array_provenance(
                    stress_meta.get("execution_provenance"),
                    stage="core_stress_array",
                    logical_task_index=scenario_index,
                    submission_receipt=submission_receipt,
                )
            except ValueError as exc:
                raise RuntimeError(
                    f"{summary_path} Slurm/source binding is invalid: {exc}"
                ) from exc
        if stress_meta.get("trace_schema_version") != TRACE_SCHEMA_VERSION:
            raise RuntimeError(f"{summary_path} uses an obsolete trace schema")
        if stress_meta.get("max_rows") is not None:
            raise RuntimeError(f"{summary_path} is not a complete 288-step run")
        if int(stress_meta.get(
            "adaptation_episodes_per_stressed_condition", -1,
        )) != 3 or int(stress_meta.get(
            "frozen_evaluation_episodes_per_stressed_condition", -1,
        )) != 1:
            raise RuntimeError(
                f"{summary_path} does not use 3 adaptation + 1 frozen episode"
            )
        if stress_meta.get("nominal_reference") != (
            "reused_primary_benchmark_episode_3"
        ):
            raise RuntimeError(f"{summary_path} reruns or mislabels H3 nominal")
        if stress_meta.get("thresholds") != STRESS_THRESHOLDS:
            raise RuntimeError(f"{summary_path} stress thresholds are not canonical")
        if stress_meta.get("mcp_reliability_posture") != "false":
            raise RuntimeError(f"{summary_path} changes canonical MCP reliability")
        if "primary nominal endpoint is reused" not in str(
            stress_meta.get("adaptation_posture", "")
        ):
            raise RuntimeError(f"{summary_path} lacks nominal-reuse metadata")
        if "fresh in-memory decision history at every episode" not in str(
            stress_meta.get("decision_history_posture", "")
        ):
            raise RuntimeError(f"{summary_path} lacks fresh-history metadata")
        dose_meta = stress_meta.get("mcp_fault_dose")
        if not isinstance(dose_meta, dict) or (
            dose_meta.get("full_trace_scheduled_opportunity_steps") != 28
            or dose_meta.get("full_trace_total_steps") != 288
        ):
            raise RuntimeError(f"{summary_path} has incorrect MCP fault-dose metadata")
        expected_ledger_design = {
            "stressed_ledgers_per_scenario_task": (
                len(STRESSORS) * len(EXPECTED_SEEDS)
            ),
            "stressed_decisions_per_scenario_task": (
                len(STRESSORS) * len(EXPECTED_SEEDS) * 288
            ),
            "reused_primary_nominal_ledgers_per_scenario_task": len(EXPECTED_SEEDS),
            "newly_executed_nominal_episodes": 0,
            "canonical_stressed_ledger_root": f"decision_ledger_h3/{run_tag}",
            "canonical_nominal_ledger_root": (
                f"decision_ledger_per_seed/{run_tag}"
            ),
        }
        if stress_meta.get("retained_ledger_design") != expected_ledger_design:
            raise RuntimeError(f"{summary_path} has incorrect H3 ledger design")
        if summary["meta"].get("scenarios") != [scenario]:
            raise RuntimeError(f"{summary_path} scenario metadata mismatch")
        results = summary.get("results")
        if not isinstance(results, dict) or set(results) != {scenario}:
            raise RuntimeError(f"{summary_path} result scenario mismatch")
        scenario_results = results[scenario]
        if not isinstance(scenario_results, dict):
            raise RuntimeError(f"{summary_path} scenario result is not an object")
        expected_conditions = {"baseline_seed_list", "baseline_by_seed", *STRESSORS}
        if set(scenario_results) != expected_conditions:
            raise RuntimeError(
                f"{summary_path} condition panel mismatch: "
                f"missing={sorted(expected_conditions - set(scenario_results))}, "
                f"unexpected={sorted(set(scenario_results) - expected_conditions)}"
            )
        if scenario_results.get("baseline_seed_list") != list(EXPECTED_SEEDS):
            raise RuntimeError(f"{summary_path} baseline seed list is not canonical")
        for condition in ("baseline_by_seed", *STRESSORS):
            seed_panel = scenario_results[condition]
            _seed_panel_keys(seed_panel, where=f"{summary_path}:{condition}")
            expected_modes = (
                BASELINE_STRESS_MODES if condition == "baseline_by_seed"
                else STRESS_MODES[condition]
            )
            if not isinstance(seed_panel, dict):
                raise RuntimeError(
                    f"{summary_path} {condition!r} seed panel is not an object"
                )
            for seed, mode_panel in seed_panel.items():
                if not isinstance(mode_panel, dict):
                    raise RuntimeError(
                        f"{summary_path} {condition}/{seed} is not a mode panel"
                    )
                if set(mode_panel) != expected_modes:
                    raise RuntimeError(
                        f"{summary_path} {condition}/{seed} mode panel mismatch"
                    )
                for mode, cell in mode_panel.items():
                    where = f"{summary_path}:{condition}/{seed}/{mode}"
                    _validate_spoilage_estimator(
                        cell.get("spoilage_estimator"), mode=mode, where=where,
                    )
                    _validate_latent_spoilage_model(
                        cell.get("latent_spoilage_model"), where=where,
                    )
                    _validate_context_execution_counts(
                        cell, where=where,
                    )
                    _validate_learner_provenance(cell, mode=mode, where=where)
                    if cell.get("trace_schema_version") != TRACE_SCHEMA_VERSION:
                        raise RuntimeError(
                            f"{summary_path}:{condition}/{seed}/{mode} schema mismatch"
                        )
                    if cell.get("benchmark_seed") != int(seed) or (
                        cell.get("episode_index") != 3
                    ):
                        raise RuntimeError(
                            f"{summary_path}:{condition}/{seed}/{mode} "
                            "seed/retained-episode mismatch"
                        )
                    if cell.get("learning_enabled") is not False or (
                        cell.get("episode_phase") != "frozen_evaluation"
                    ):
                        raise RuntimeError(f"{where} is not a frozen evaluation")
                    _validate_frozen_h3_learner_state(cell, where=where)
                    if cell.get("dispatch_opportunity_count") != 288 or not math.isclose(
                        float(cell.get("dispatch_cadence_hours", -1.0)), 0.25,
                        abs_tol=1e-12,
                    ):
                        raise RuntimeError(
                            f"{summary_path}:{condition}/{seed}/{mode} dispatch mismatch"
                        )
                    for hash_field in (
                        "latent_environment_sha256", "observed_policy_input_sha256",
                        "demand_observation_sha256", "context_prior_sha256",
                        "policy_theta_initial_sha256",
                    ):
                        value = cell.get(hash_field)
                        if not isinstance(value, str) or not re.fullmatch(
                            r"[0-9a-f]{64}", value,
                        ):
                            raise RuntimeError(
                                f"{summary_path}:{condition}/{seed}/{mode} "
                                f"has invalid {hash_field}"
                            )
                    if cell.get("demand_forecast_method") != "holt_linear":
                        raise RuntimeError(
                            f"{where} did not use the locked Holt-linear demand forecast"
                        )
                    if cell.get("supply_forecast_method") != "persistence":
                        raise RuntimeError(
                            f"{where} did not use the locked persistence supply forecast"
                        )
                    expected_environment_id = _stream_id(
                        int(seed), scenario, 3, "environment",
                    )
                    expected_policy_id = _stream_id(
                        int(seed), scenario, 3, "policy",
                    )
                    if (
                        cell.get("environment_stream_id") != expected_environment_id
                        or cell.get("stochastic_stream_id")
                        != expected_environment_id
                        or cell.get("policy_stream_id") != expected_policy_id
                    ):
                        raise RuntimeError(f"{where} has incorrect retained stream IDs")
                    treatment = cell.get("observation_treatment")
                    if not isinstance(treatment, dict):
                        raise RuntimeError(f"{where} lacks observation treatment provenance")
                    expected_stressor = (
                        "nominal" if condition == "baseline_by_seed" else condition
                    )
                    if treatment.get("stressor") != expected_stressor or (
                        treatment.get("n_steps") != 288
                    ):
                        raise RuntimeError(f"{where} has incorrect treatment identity")
                    data_treatment_expected = condition not in {
                        "baseline_by_seed", "mcp_fault_injection",
                    }
                    if treatment.get("data_observation_treatment") is not (
                        data_treatment_expected
                    ):
                        raise RuntimeError(f"{where} has incorrect treatment namespace")
                    if condition == "baseline_by_seed":
                        if treatment != {
                            "stressor": "nominal",
                            "n_steps": 288,
                            "data_observation_treatment": False,
                            "delay_steps": 0,
                            "missing_count": 0,
                            "source": "reused_primary_benchmark",
                        }:
                            raise RuntimeError(
                                f"{where} has incorrect reused-primary treatment metadata"
                            )
                        _validate_primary_nominal_binding(
                            seed_root=seed_root, scenario=scenario, seed=int(seed),
                            stress_cell=cell, where=where,
                        )
                        ledger_path = (
                            primary_ledger_root / f"seed_{seed}"
                            / f"agribrain__{scenario}.jsonl"
                            if primary_ledger_root is not None
                            else seed_root / f"decision_ledger_{seed}"
                            / f"agribrain__{scenario}.jsonl"
                        )
                        canonical_ledger_path = (
                            f"decision_ledger_per_seed/{run_tag}/seed_{seed}/"
                            f"agribrain__{scenario}.jsonl"
                        )
                        ledger_reconstructions[(
                            scenario, condition, int(seed),
                        )] = _reconstruct_h3_ledger(
                            ledger_path, scenario=scenario, stressor="nominal",
                            seed=int(seed), cell=cell,
                            canonical_path=canonical_ledger_path,
                        )
                    else:
                        expected_treatment = _expected_observation_treatment(
                            scenario=scenario, stressor=condition, seed=int(seed),
                        )
                        if treatment != expected_treatment:
                            raise RuntimeError(
                                f"{where} treatment provenance does not match the "
                                "locked seed-indexed H3 dose"
                            )
                        ledger_path = (
                            h3_ledger_root / scenario / condition
                            / f"seed_{seed}" / f"agribrain__{scenario}.jsonl"
                        )
                        canonical_ledger_path = (
                            f"decision_ledger_h3/{run_tag}/{scenario}/{condition}/"
                            f"seed_{seed}/agribrain__{scenario}.jsonl"
                        )
                        reconstructed = _reconstruct_h3_ledger(
                            ledger_path, scenario=scenario, stressor=condition,
                            seed=int(seed), cell=cell,
                            canonical_path=canonical_ledger_path,
                        )
                        ledger_reconstructions[(
                            scenario, condition, int(seed),
                        )] = reconstructed
                        nominal_reconstructed = ledger_reconstructions.get((
                            scenario, "baseline_by_seed", int(seed),
                        ))
                        if nominal_reconstructed is None:
                            raise RuntimeError(
                                f"{where} has no validated reused nominal ledger"
                            )
                        _validate_h3_observation_transform(
                            nominal=nominal_reconstructed,
                            stressed=reconstructed,
                            stressor=condition,
                            where=where,
                        )
                        same_observed_rows = (
                            reconstructed["observed_policy_rows"]
                            == nominal_reconstructed["observed_policy_rows"]
                        )
                        if condition == "mcp_fault_injection" and not (
                            same_observed_rows
                        ):
                            raise RuntimeError(
                                f"{where} MCP-only dose changed ledger observations"
                            )
                        if condition != "mcp_fault_injection" and same_observed_rows:
                            raise RuntimeError(
                                f"{where} data dose produced no ledger-level exposure"
                            )
                    if condition in {"sensor_noise", "compounded"}:
                        for field in ("temp_noise_sha256", "rh_noise_sha256"):
                            if not _valid_sha256(treatment.get(field)):
                                raise RuntimeError(f"{where} lacks {field}")
                    if condition in {"missing_data", "compounded"}:
                        if not _valid_sha256(treatment.get("missing_mask_sha256")) or (
                            not isinstance(treatment.get("missing_count"), int)
                            or treatment["missing_count"] <= 0
                        ):
                            raise RuntimeError(f"{where} has invalid missing-data dose")
                    expected_delay = 4 if condition in {
                        "telemetry_delay", "compounded",
                    } else 0
                    if treatment.get("delay_steps") != expected_delay:
                        raise RuntimeError(f"{where} has incorrect delay dose")
                    dose_values: dict[str, int] = {}
                    for dose_field in (
                        "fault_injection_scheduled_opportunity_steps",
                        "fault_injection_trigger_steps",
                        "fault_injected_tool_result_count",
                    ):
                        dose = cell.get(dose_field)
                        if isinstance(dose, bool) or not isinstance(dose, int) or dose < 0:
                            raise RuntimeError(
                                f"{summary_path}:{condition}/{seed}/{mode} lacks "
                                f"non-negative integer {dose_field}"
                            )
                        dose_values[dose_field] = dose
                    fault_condition = condition in {
                        "mcp_fault_injection", "compounded",
                    }
                    expected_schedule = 28 if fault_condition else 0
                    if dose_values[
                        "fault_injection_scheduled_opportunity_steps"
                    ] != expected_schedule:
                        raise RuntimeError(
                            f"{summary_path}:{condition}/{seed}/{mode} has an "
                            "incorrect fault-schedule opportunity count"
                        )
                    triggers = dose_values["fault_injection_trigger_steps"]
                    replaced = dose_values["fault_injected_tool_result_count"]
                    if triggers > expected_schedule:
                        raise RuntimeError(
                            f"{summary_path}:{condition}/{seed}/{mode} has more "
                            "fault triggers than scheduled opportunities"
                        )
                    if not fault_condition and (triggers or replaced):
                        raise RuntimeError(
                            f"{summary_path}:{condition}/{seed}/{mode} reports "
                            "fault exposure in a non-fault condition"
                        )
                    mcp_active = mode in {"agribrain", "mcp_only"}
                    if fault_condition and mcp_active:
                        if triggers <= 0 or replaced < triggers:
                            raise RuntimeError(
                                f"{summary_path}:{condition}/{seed}/{mode} did "
                                "not receive the declared MCP fault treatment"
                            )
                    if fault_condition and not mcp_active and (triggers or replaced):
                        raise RuntimeError(
                            f"{summary_path}:{condition}/{seed}/{mode} reports MCP "
                            "fault exposure despite having no MCP channel"
                        )

            for seed in EXPECTED_SEEDS:
                mode_panel = scenario_results[condition].get(
                    str(seed), scenario_results[condition].get(seed),
                )
                treatment_records = {
                    json.dumps(
                        cell["observation_treatment"], sort_keys=True,
                        separators=(",", ":"), allow_nan=False,
                    )
                    for cell in mode_panel.values()
                }
                if len(treatment_records) != 1:
                    raise RuntimeError(
                        f"{summary_path}:{condition}/{seed} modes received "
                        "different observation treatments"
                    )
                latent_hashes = {
                    cell["latent_environment_sha256"]
                    for cell in mode_panel.values()
                }
                if len(latent_hashes) != 1:
                    raise RuntimeError(
                        f"{summary_path}:{condition}/{seed} modes do not share latent truth"
                    )
                if condition == "baseline_by_seed":
                    continue
                baseline_mode_panel = scenario_results["baseline_by_seed"].get(
                    str(seed), scenario_results["baseline_by_seed"].get(seed),
                )
                for mode, cell in mode_panel.items():
                    baseline_cell = baseline_mode_panel[mode]
                    if cell["latent_environment_sha256"] != (
                        baseline_cell["latent_environment_sha256"]
                    ):
                        raise RuntimeError(
                            f"{summary_path}:{condition}/{seed}/{mode} changed latent truth"
                        )
                    if condition != "mcp_fault_injection" and (
                        cell["observed_policy_input_sha256"]
                        == baseline_cell["observed_policy_input_sha256"]
                    ):
                        raise RuntimeError(
                            f"{summary_path}:{condition}/{seed}/{mode} was a no-op"
                        )
                    if condition == "mcp_fault_injection" and (
                        cell["observed_policy_input_sha256"]
                        != baseline_cell["observed_policy_input_sha256"]
                    ):
                        raise RuntimeError(
                            f"{summary_path}:{condition}/{seed}/{mode} changed "
                            "the sensor/forecast observation stream"
                        )
                    for common_field in (
                        "environment_stream_id", "stochastic_stream_id",
                        "policy_stream_id", "context_prior_sha256",
                        "policy_theta_initial_sha256", "demand_observation_sha256",
                    ):
                        if cell.get(common_field) != baseline_cell.get(common_field):
                            raise RuntimeError(
                                f"{summary_path}:{condition}/{seed}/{mode} changed "
                                f"paired field {common_field}"
                            )

        h3_path = root / "stress_h3_test.json"
        h3 = _load_json(h3_path)
        if h3.get("execution_provenance") != stress_meta.get(
            "execution_provenance"
        ):
            raise RuntimeError(
                f"{h3_path} execution provenance differs from its raw task summary"
            )
        _validate_identity(
            h3, source_commit=source_commit, run_tag=run_tag, where=h3_path,
        )
        if h3.get("test") != "paired one-sample TOST on seed-level ARI differences":
            raise RuntimeError(f"{h3_path} has an unexpected inferential test")
        if h3.get("nominal_reference") != "reused_primary_benchmark_episode_3":
            raise RuntimeError(f"{h3_path} does not reuse the primary nominal")
        accounting = h3.get("episode_accounting") or {}
        expected_accounting = build_h3_episode_accounting(
            n_seeds=len(EXPECTED_SEEDS), n_scenarios=1,
            n_stressors=len(STRESSORS), episodes_per_condition=4,
            nominal_reference_reused=True,
        )
        if accounting != expected_accounting:
            raise RuntimeError(
                f"{h3_path} does not report exact 100-retained/400-executed "
                "scenario-task H3 accounting"
            )
        if (
            h3.get("confirmatory_method") != "agribrain"
            or h3.get("expected_scenarios") != [scenario]
            or h3.get("expected_stressors") != list(STRESSORS)
            or h3.get("expected_n_cells") != len(STRESSORS)
            or h3.get("adaptation_episodes_per_stressed_condition") != 3
            or h3.get("frozen_evaluation_episodes_per_stressed_condition") != 1
        ):
            raise RuntimeError(f"{h3_path} has incorrect confirmatory H3 design metadata")
        _assert_close(h3.get("alpha"), 0.05, where=f"{h3_path}:alpha")
        _assert_close(
            h3.get("equivalence_margin"),
            STRESS_THRESHOLDS["ari_abs_delta_max"],
            where=f"{h3_path}:equivalence_margin",
        )
        if int(h3.get("n_cells", -1)) != len(STRESSORS):
            raise RuntimeError(f"{h3_path} reports an incorrect H3 cell count")
        cells = h3.get("cells")
        if not isinstance(cells, list) or len(cells) != 5:
            raise RuntimeError(f"{h3_path} must contain five H3 cells")
        if any(cell.get("Scenario") != scenario for cell in cells):
            raise RuntimeError(f"{h3_path} contains a cross-scenario H3 cell")
        cell_by_stressor: dict[str, dict[str, Any]] = {}
        for cell in cells:
            stressor = cell.get("Stressor")
            if stressor in cell_by_stressor:
                raise RuntimeError(f"{h3_path} contains duplicate stressor {stressor!r}")
            if (
                stressor not in STRESSORS
                or cell.get("Method") != "agribrain"
                or cell.get("Confirmatory_H3") is not True
                or cell.get("inferential_status") != "confirmatory_h3"
            ):
                raise RuntimeError(f"{h3_path} contains an unexpected H3 cell")
            if cell.get("treatment_exposure_verified") is not True:
                raise RuntimeError(
                    f"{h3_path}:{stressor} lacks verified treatment exposure"
                )
            cell_by_stressor[str(stressor)] = cell
        if set(cell_by_stressor) != set(STRESSORS):
            raise RuntimeError(f"{h3_path} does not contain the exact stressor panel")

        baseline_panel = scenario_results["baseline_by_seed"]
        for stressor in STRESSORS:
            stressed_panel = scenario_results[stressor]
            diffs = [
                float(_panel_cell(stressed_panel, seed, "agribrain")["ari"])
                - float(_panel_cell(baseline_panel, seed, "agribrain")["ari"])
                for seed in EXPECTED_SEEDS
            ]
            expected_tost = _equivalence_tost(
                diffs, STRESS_THRESHOLDS["ari_abs_delta_max"]
            )
            cell = cell_by_stressor[stressor]
            stressed_binding = _h3_ledger_set_binding(stressed_panel)
            nominal_binding = _h3_ledger_set_binding(baseline_panel)
            for field, expected in (
                ("retained_stressed_decision_ledger_count", stressed_binding["count"]),
                ("retained_stressed_decision_count", stressed_binding["decision_count"]),
                ("retained_stressed_decision_ledger_set_sha256", stressed_binding["sha256"]),
                ("reused_nominal_decision_ledger_count", nominal_binding["count"]),
                ("reused_nominal_decision_count", nominal_binding["decision_count"]),
                ("reused_nominal_decision_ledger_set_sha256", nominal_binding["sha256"]),
            ):
                if cell.get(field) != expected:
                    raise RuntimeError(
                        f"{h3_path}:{stressor}/{field} does not bind retained ledgers"
                    )
            if int(cell.get("n_seeds", -1)) != len(EXPECTED_SEEDS):
                raise RuntimeError(f"{h3_path}:{stressor} has an incorrect seed count")
            for key, expected_value in expected_tost.items():
                actual_key = f"ari_tost_{key}"
                if isinstance(expected_value, bool):
                    if cell.get(actual_key) is not expected_value:
                        raise RuntimeError(f"{h3_path}:{stressor}/{actual_key} mismatch")
                elif isinstance(expected_value, int):
                    if int(cell.get(actual_key, -1)) != expected_value:
                        raise RuntimeError(f"{h3_path}:{stressor}/{actual_key} mismatch")
                else:
                    _assert_close(
                        cell.get(actual_key), float(expected_value),
                        where=f"{h3_path}:{stressor}/{actual_key}",
                    )
            expected_pass = bool(expected_tost["equivalent_alpha_0p05"])
            if (
                cell.get("Pass_Equivalence") is not expected_pass
                or cell.get("Pass") is not expected_pass
                or cell.get("H3_Pass") is not expected_pass
            ):
                raise RuntimeError(f"{h3_path}:{stressor} pass flag contradicts TOST")

        expected_supported = all(
            bool(cell_by_stressor[s]["Pass_Equivalence"])
            and bool(cell_by_stressor[s]["treatment_exposure_verified"])
            for s in STRESSORS
        )
        if bool(h3.get("supported_all_cells")) != expected_supported:
            raise RuntimeError(f"{h3_path} supported_all_cells is inconsistent")
        expected_equivalent_count = sum(
            bool(cell_by_stressor[s]["Pass_Equivalence"]) for s in STRESSORS
        )
        if int(h3.get("n_cells_equivalent", -1)) != expected_equivalent_count:
            raise RuntimeError(f"{h3_path} n_cells_equivalent is inconsistent")
        if int(h3.get("n_cells_with_verified_exposure", -1)) != len(STRESSORS):
            raise RuntimeError(
                f"{h3_path} n_cells_with_verified_exposure is inconsistent"
            )
        if (
            h3.get("retained_stressed_decision_ledger_count")
            != len(STRESSORS) * len(EXPECTED_SEEDS)
            or h3.get("reused_nominal_decision_ledger_references")
            != len(EXPECTED_SEEDS)
            or h3.get("newly_executed_nominal_episodes") != 0
        ):
            raise RuntimeError(f"{h3_path} retained-ledger accounting is incorrect")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--stress-root", type=Path, required=True)
    parser.add_argument("--h3-ledger-root", type=Path, required=True)
    parser.add_argument(
        "--primary-ledger-root", type=Path,
        help=(
            "Optional consolidated decision_ledger_per_seed/<run-tag> root; "
            "otherwise nominal ledgers are read beside the raw seed envelopes."
        ),
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--submission-receipt", type=Path, required=True)
    parser.add_argument("--publisher-slurm-job-id", required=True)
    parser.add_argument(
        "--recovery-receipt",
        type=Path,
        help=(
            "Explicit publication-recovery authorization for a replacement "
            "publisher. When omitted, the running job must remain the "
            "publisher declared by the original submission receipt."
        ),
    )
    parser.add_argument(
        "--publication-commit",
        help="Clean publication-repair commit (required with --recovery-receipt).",
    )
    args = parser.parse_args(argv)

    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        raise RuntimeError("--source-commit must be a full lowercase Git SHA-1")
    if not args.run_tag.strip():
        raise RuntimeError("--run-tag must not be empty")
    if (
        not args.seed_root.is_dir()
        or not args.stress_root.is_dir()
        or not args.h3_ledger_root.is_dir()
    ):
        raise RuntimeError("seed, stress, and H3 ledger roots must all exist")

    from hpc.core_submission_receipt import validate_receipt_file

    submission_receipt = validate_receipt_file(
        args.submission_receipt,
        expected_run_tag=args.run_tag,
        expected_source_commit=args.source_commit,
    )
    if args.recovery_receipt is None:
        if args.publication_commit is not None:
            raise RuntimeError(
                "--publication-commit is invalid without --recovery-receipt"
            )
        require_declared_publisher(
            submission_receipt,
            actual_slurm_job_id=args.publisher_slurm_job_id,
        )
    else:
        if not isinstance(args.publication_commit, str) or re.fullmatch(
            r"[0-9a-f]{40}", args.publication_commit,
        ) is None:
            raise RuntimeError(
                "--publication-commit must be a full lowercase Git SHA-1 "
                "with --recovery-receipt"
            )
        from hpc.publication_recovery_receipt import (
            validate_recovery_receipt_file,
        )

        validate_recovery_receipt_file(
            args.recovery_receipt,
            original_receipt_path=args.submission_receipt,
            expected_kind="core",
            expected_run_tag=args.run_tag,
            expected_simulation_commit=args.source_commit,
            expected_publication_commit=args.publication_commit,
            expected_recovery_job_id=args.publisher_slurm_job_id,
        )

    validate_seed_inputs(
        args.seed_root, source_commit=args.source_commit, run_tag=args.run_tag,
        submission_receipt=submission_receipt,
    )
    validate_stress_inputs(
        args.stress_root, seed_root=args.seed_root,
        h3_ledger_root=args.h3_ledger_root,
        primary_ledger_root=args.primary_ledger_root,
        source_commit=args.source_commit, run_tag=args.run_tag,
        submission_receipt=submission_receipt,
    )
    print(
        f"[PASS] raw publication provenance + exact panel: "
        f"{len(EXPECTED_SEEDS)} seeds, {len(SCENARIOS)} stress scenarios"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
