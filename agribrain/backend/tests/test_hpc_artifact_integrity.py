"""Focused tests for raw-run and DecisionLedger publication gates."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import csv
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, relative: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


raw = _load("validate_raw_publication_inputs", "hpc/validate_raw_publication_inputs.py")
ledgers = _load("validate_decision_ledgers", "hpc/validate_decision_ledgers.py")
manifest_builder = _load(
    "build_artifact_manifest_for_integrity_test",
    "mvp/simulation/analysis/build_artifact_manifest.py",
)
single_seed = _load(
    "run_single_seed_for_integrity_test",
    "mvp/simulation/benchmarks/run_single_seed.py",
)
stress_aggregate = _load(
    "aggregate_stress_outputs_for_integrity_test",
    "mvp/simulation/benchmarks/aggregate_stress_outputs.py",
)
stress_runner = _load(
    "run_stress_suite_for_integrity_test",
    "mvp/simulation/benchmarks/run_stress_suite.py",
)
from src.models.outcome_equation_contract import reconstruct_step_outcomes

from mvp.simulation.benchmarks import trace_contract
from mvp.simulation.validation import validate_publication_artifacts as publication


def _sealed_step_channel_evidence(record: dict) -> dict:
    """Build one internally consistent synthetic coordinator snapshot."""

    from src.agents.coordinator import (
        _build_peer_channel_evidence,
        _build_retrieval_channel_evidence,
        _empty_channel_evidence,
        _protocol_window,
        _seal_evidence_record,
    )
    from src.agents.message import InterAgentMessage, MessageType

    role = str(record["role"])
    peer_bias = list(record["peer_message_bias"])
    consumed = []
    if peer_bias != [0.0, 0.0, 0.0]:
        # A 0.2-capacity update produces the fixture's exact +0.01 recovery
        # bias under the production message_bias_from_inbox implementation.
        message = InterAgentMessage(
            sender="recovery_agent",
            recipient=f"{role}_agent",
            msg_type=MessageType.CAPACITY_UPDATE,
            payload={"available_capacity": 0.2},
            hour=float(record["hour"]),
        )
        consumed.append((message, role, True))
    primary_mcp, _unused_primary_retrieval = _empty_channel_evidence(
        "pirag", "fixture_no_mcp_dispatch",
    )
    cooperative_mcp, cooperative_retrieval = _empty_channel_evidence(
        "pirag", "fixture_cooperative_overlay_inactive",
    )
    retrieval_attempted = bool(
        record.get("primary_pirag_query_attempted_step", False)
    )
    primary_retrieval = _build_retrieval_channel_evidence(
        rag_context={
            "retrieval_kind": "pirag",
            "query": "",
            "citations": [],
            "top_doc_id": record.get("retrieval_top_doc_id", ""),
            "top_citation_score": record.get("retrieval_top_score"),
            "top_fused_score": record.get("retrieval_top_fused_score"),
            "top_rerank_score": record.get("retrieval_top_rerank_score"),
            "evidence_hashes": list(
                record.get("retrieval_evidence_hashes", [])
            ),
        },
        integration_trace=record.get("context_integration", {}).get("primary"),
        protocol_window=_protocol_window(None, (0, 0)),
        attempted=retrieval_attempted,
        requested_kind="pirag",
        skip_reason=None if retrieval_attempted else "fixture_retrieval_not_attempted",
    )
    return _seal_evidence_record({
        "schema_version": "agribrain.step_channel_evidence.v1",
        "hour": float(record["hour"]),
        "active_role": role,
        "peer": _build_peer_channel_evidence(
            consumed, [], np.asarray(peer_bias, dtype=float), enabled=True,
        ),
        "primary": _seal_evidence_record({
            "role": role,
            "mcp": primary_mcp,
            "retrieval": primary_retrieval,
        }),
        "cooperative": _seal_evidence_record({
            "active": False,
            "role": "cooperative",
            "mcp": cooperative_mcp,
            "retrieval": cooperative_retrieval,
        }),
    })


def _mutate_primary_retrieval_binding(
    record: dict, *, record_field: str | None = None,
    evidence_field: str, value,
) -> None:
    """Keep evidence binding valid so a negative test reaches its target."""

    from src.agents.coordinator import _seal_evidence_record

    if record_field is not None:
        record[record_field] = value
    evidence = record["step_channel_evidence"]
    retrieval = {
        key: item for key, item in evidence["primary"]["retrieval"].items()
        if key != "content_sha256"
    }
    retrieval[evidence_field] = value
    primary = {
        key: item for key, item in evidence["primary"].items()
        if key != "content_sha256"
    }
    primary["retrieval"] = _seal_evidence_record(retrieval)
    root = {
        key: item for key, item in evidence.items()
        if key != "content_sha256"
    }
    root["primary"] = _seal_evidence_record(primary)
    record["step_channel_evidence"] = _seal_evidence_record(root)


def _mutate_rag_total_scale_consistently(record: dict) -> None:
    trace = record["context_integration"]["primary"]
    value = float(trace["rag_total_scale"]) + 0.123
    trace["rag_total_scale"] = value
    _mutate_primary_retrieval_binding(
        record, evidence_field="rag_total_scale", value=value,
    )


def _record(step_index: int, contract: dict | None = None) -> dict:
    rho = 0.20 + 0.01 * step_index
    waste = 0.10
    social = 0.80
    zero_matrix = [[0.0] * 5 for _ in range(3)]
    primary_integration = {
        "context_mode": "full",
        "retrieval_kind": "pirag",
        "over_steer": False,
        "clip_applied": True,
        "effective_theta": zero_matrix,
        "effective_psi": [0.0] * 5,
        "linear_feature_contributions": zero_matrix,
        "channel_scaled_feature_contributions": zero_matrix,
        "feature_contributions": zero_matrix,
        "nonfeature_residual": [0.0] * 3,
        "mcp_preclip_component": [0.0] * 3,
        "pirag_preclip_component": [0.0] * 3,
        "retrieval_gate": 0.0,
        "retrieval_blocked_reason": "retrieval_guard",
        "temporal_scale": 1.0,
        "temporal_gate_requested": False,
        "temporal_gate_applied": False,
        "temporal_continuity_score": None,
        "temporal_base": None,
        "temporal_decay": None,
        "physics_scale": 1.0,
        "rag_total_scale": 0.0,
        "global_scale": 1.0,
        "preclip_modifier": [0.0] * 3,
        "clip_derivative": [1.0] * 3,
        "modifier_theta_jacobian": zero_matrix,
        "final_modifier": [0.0] * 3,
        "blocked_reason": None,
    }
    context_integration = {
        "primary": primary_integration,
        "cooperative": None,
        "composition": {
            "scope": "primary_context",
            "clip_applied": True,
            "preclip_modifier": [0.0] * 3,
            "clip_derivative": [1.0] * 3,
            "modifier_theta_jacobian": zero_matrix,
            "final_modifier": [0.0] * 3,
        },
    }
    record = {
        "step_index": step_index,
        "hour": 0.25 * step_index,
        "action": "cold_chain",
        "action_idx": 0,
        "probs": [0.5, 0.3, 0.2],
        "decision_latency_ms": 2.0,
        "reward": 0.7,
        "waste": waste,
        "rho": rho,
        "rho_policy_observed": rho,
        "rho_outcome_environmental": rho,
        "temp_policy_observed": 4.0 + 0.1 * step_index,
        "temp_outcome_environmental": 4.0 + 0.1 * step_index,
        "rh_policy_observed": 90.0,
        "rh_outcome_environmental": 90.0,
        "inventory_policy_observed": 1000.0,
        "inventory_outcome_environmental": 1000.0,
        "demand_policy_observed": 900.0,
        "demand_forecast_policy_observed": 900.0,
        "supply_forecast_policy_observed": 1000.0,
        "bollinger_regime_flag": 0.0,
        "regime_logit_bias": [0.0, 0.0, 0.0],
        "price_signal": 0.0,
        "demand_outcome_environmental": 900.0,
        "transport_multiplier_outcome_environmental": 1.0,
        "simulated_dispatch_accounted": True,
        "slca": social,
        "ari": (1.0 - waste) * social * (1.0 - rho),
        "carbon_kg": 1.5,
        "mode": "agribrain",
        "scenario": "baseline",
        "phi": [0.0] * 10,
        "peer_message_bias": [0.0, 0.0, 0.01],
        "psi": [0.0] * 5,
        "context_modifier": [0.0] * 3,
        "base_logits": [math.log(0.5), math.log(0.3), math.log(0.2)],
        "post_context_logits_pre_override": [
            math.log(0.5), math.log(0.3), math.log(0.2),
        ],
        "slca_shaping": [0.0] * 3,
        "slca_amp": 0.0,
        "policy_temperature": 1.0,
        "modifier_mcp": [0.0] * 3,
        "modifier_pirag": [0.0] * 3,
        "retrieval_top_doc_id": "",
        "retrieval_top_score": 0.0,
        "retrieval_top_fused_score": 0.0,
        "retrieval_top_rerank_score": 0.0,
        "retrieval_evidence_hashes": [],
        "effective_context_theta": zero_matrix,
        "context_feature_contributions": zero_matrix,
        "context_nonfeature_residual": [0.0] * 3,
        "context_modifier_theta_jacobian": zero_matrix,
        "context_integration": context_integration,
        "chosen_action_context_contributions": [0.0] * 5,
        "chosen_action_context_residual": 0.0,
        "context_attribution_basis": (
            "final_modifier_feature_allocation_plus_explicit_residual"
        ),
        "context_attribution_scope": "primary_context",
        "dominant_psi_idx": 0,
        "dominant_context_component": "psi_0",
        "dominant_action_idx": 0,
        "governance_override": False,
        "context_counterfactual_action_idx": 0,
        "context_counterfactual_action": "cold_chain",
        "context_counterfactual_probs": [0.5, 0.3, 0.2],
        "context_action_changed": False,
        "context_influence_active": False,
        "context_influence_counted": False,
        "context_influence_threshold": 0.10,
        "mcp_tool_call_count_step": 0,
        "pirag_query_count_step": 0,
        "dispatcher_tool_failure_count_step": 0,
        "inter_agent_message_count_step": 0,
        "protocol_interaction_count_step": 0,
        "protocol_jsonrpc_error_count_step": 0,
        "protocol_tool_iserror_count_step": 0,
        "protocol_real_tool_iserror_count_step": 0,
        "protocol_error_count_step": 0,
        "protocol_dropped_interaction_count_step": 0,
    }
    if contract is None:
        contract = ledgers.expected_publication_outcome_equation_contract(
            benchmark_seed=42, scenario="baseline",
        )
    reconstructed = reconstruct_step_outcomes(record, contract)
    for field in ("reward", "waste", "slca", "ari", "carbon_kg"):
        record[field] = reconstructed[field]
    return record


_REAL_LEDGER_FIXTURE_BYTES: bytes | None = None


def _write_ledger(path: Path) -> None:
    # Generate the positive fixture through the real retained-episode producer.
    # Hand-authoring rows made this test silently stale whenever the publication
    # ledger contract gained a new independently reconstructable field.
    global _REAL_LEDGER_FIXTURE_BYTES
    path.parent.mkdir(parents=True, exist_ok=True)
    if _REAL_LEDGER_FIXTURE_BYTES is not None:
        path.write_bytes(_REAL_LEDGER_FIXTURE_BYTES)
        return

    import src.models.action_selection as action_selection

    from mvp.simulation import generate_results as gr
    from mvp.simulation.stochastic import make_stochastic_layer

    benchmark_seed = 42
    scenario = "baseline"
    episode_index = 3
    policy = gr.Policy()
    base = pd.read_csv(gr.DATA_CSV, parse_dates=["timestamp"]).head(2)
    scenario_seed = gr._stream_seed(
        benchmark_seed, scenario, episode_index, "scenario",
    )
    environment_seed = gr._stream_seed(
        benchmark_seed, scenario, episode_index, "environment",
    )
    frame = gr.apply_scenario(
        base,
        scenario,
        policy,
        np.random.default_rng(scenario_seed),
        stoch=make_stochastic_layer(
            np.random.default_rng(environment_seed),
            stream_seed=environment_seed,
        ),
    )
    original_theta = np.asarray(action_selection.THETA, dtype=float).copy()
    action_selection.THETA = gr.policy_theta_for_seed(
        np.asarray(action_selection.DECLARED_THETA, dtype=float),
        benchmark_seed,
    )
    try:
        with gr.decision_ledger_scope(path.parent, reset=True):
            gr.run_episode(
                frame,
                "agribrain",
                policy,
                np.random.default_rng(
                    gr._stream_seed(
                        benchmark_seed, scenario, episode_index, "policy",
                    )
                ),
                scenario=scenario,
                stoch=make_stochastic_layer(
                    np.random.default_rng(environment_seed),
                    stream_seed=environment_seed,
                ),
                seed=benchmark_seed,
                benchmark_seed=benchmark_seed,
                episode_index=episode_index,
                environment_stream_id=gr._stream_id(
                    benchmark_seed, scenario, episode_index, "environment",
                ),
                policy_stream_id=gr._stream_id(
                    benchmark_seed, scenario, episode_index, "policy",
                ),
                stochastic_stream_id=gr._stream_id(
                    benchmark_seed, scenario, episode_index, "environment",
                ),
                learner_state_cache={},
                learning_enabled=False,
            )
    finally:
        action_selection.THETA = original_theta
    if not path.is_file():
        raise AssertionError(f"real ledger producer did not create {path}")
    _REAL_LEDGER_FIXTURE_BYTES = path.read_bytes()


def _rehash_after_mutation(path: Path, mutate) -> None:
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    header, records = lines[0], lines[1:]
    for record in records:
        record.pop("_leaf", None)
    mutate(records[0])
    leaves = [
        hashlib.sha256(ledgers._canonical_bytes(record)).hexdigest()
        for record in records
    ]
    header["merkle_root"] = ledgers._merkle_root(leaves)
    path.write_text(
        "\n".join(
            [json.dumps(header)]
            + [json.dumps({**record, "_leaf": leaf})
               for record, leaf in zip(records, leaves, strict=True)]
        ) + "\n",
        encoding="utf-8",
    )


def test_decision_ledger_gate_accepts_exact_inventory_and_merkle(tmp_path, monkeypatch):
    monkeypatch.setattr(ledgers, "EXPECTED_SEEDS", (42,))
    monkeypatch.setattr(ledgers, "MODES", ["agribrain"])
    monkeypatch.setattr(ledgers, "SCENARIOS", ["baseline"])
    monkeypatch.setattr(ledgers, "EXPECTED_RECORDS", 2)
    # This deliberately short unit fixture traverses only the farm stage.
    # Production validation retains the locked four-role, 288-row contract.
    monkeypatch.setattr(ledgers, "DECISION_OWNER_ROLES", ("farm",))
    root = tmp_path / "run"
    _write_ledger(root / "seed_42" / "agribrain__baseline.jsonl")
    ledgers.validate_inventory(root)


def test_decision_ledger_gate_binds_seed_headlines_to_records(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(ledgers, "EXPECTED_SEEDS", (42,))
    monkeypatch.setattr(ledgers, "MODES", ["agribrain"])
    monkeypatch.setattr(ledgers, "SCENARIOS", ["baseline"])
    monkeypatch.setattr(ledgers, "EXPECTED_RECORDS", 2)
    monkeypatch.setattr(ledgers, "DECISION_OWNER_ROLES", ("farm",))
    root = tmp_path / "run"
    ledger = root / "seed_42" / "agribrain__baseline.jsonl"
    _write_ledger(ledger)
    ledger_summary = ledgers.validate_ledger(
        ledger, mode="agribrain", scenario="baseline", benchmark_seed=42,
    )
    recomputed = ledger_summary["headline_metrics"]
    evidence = ledger_summary["episode_evidence"]
    snapshots = ledger_summary["learner_snapshots"]
    learner_envelope = {
        "theta_learner_summary": {
            "per_role": {
                "farm": {
                    "final_theta_delta": snapshots["theta_delta_by_role"][
                        "farm"
                    ],
                },
            },
        },
        "reward_shaping_learner_summary": snapshots["reward_shaping"],
        "learner_summary": {
            "final_theta": snapshots["context_theta"],
            "final_slca_amp": snapshots["context_slca_amp"],
        },
    }
    seed_root = tmp_path / "seeds"
    seed_root.mkdir()
    envelope = {
        "seed": 42,
        "scenarios": {"baseline": {"agribrain": {
            **recomputed,
            **evidence,
            **learner_envelope,
            "mean_decision_latency_ms_descriptive_only": True,
            "latency_penalty_usd_descriptive_only": True,
        }}},
        "traces": {
            "baseline": {"agribrain": ledger_summary["trace_binding"]},
        },
    }
    seed_path = seed_root / "seed_42.json"
    seed_path.write_text(json.dumps(envelope), encoding="utf-8")

    ledgers.validate_inventory(root, seed_root)

    envelope["scenarios"]["baseline"]["agribrain"][
        "mean_decision_latency_ms"
    ] += 0.5
    seed_path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from its ledger"):
        ledgers.validate_inventory(root, seed_root)

    valid_evidence_cell = {
        **recomputed, **evidence, **learner_envelope,
        "mean_decision_latency_ms_descriptive_only": True,
        "latency_penalty_usd_descriptive_only": True,
    }
    for field in (
        "protocol_tools_call_count", "protocol_prompts_get_count",
    ):
        envelope["scenarios"]["baseline"]["agribrain"] = dict(
            valid_evidence_cell
        )
        envelope["scenarios"]["baseline"]["agribrain"].pop(field)
        seed_path.write_text(json.dumps(envelope), encoding="utf-8")
        with pytest.raises(RuntimeError, match=rf"{field} differs from its ledger"):
            ledgers.validate_inventory(root, seed_root)

    for field in (
        "context_honored_steps", "context_influenced_steps",
        "protocol_interaction_count", "protocol_tools_call_count",
        "protocol_prompts_get_count", "mcp_calls_per_episode",
    ):
        envelope["scenarios"]["baseline"]["agribrain"] = dict(
            valid_evidence_cell
        )
        envelope["scenarios"]["baseline"]["agribrain"][field] += 1
        seed_path.write_text(json.dumps(envelope), encoding="utf-8")
        with pytest.raises(RuntimeError, match="differs from its ledger"):
            ledgers.validate_inventory(root, seed_root)

    envelope["scenarios"]["baseline"]["agribrain"] = dict(
        valid_evidence_cell
    )
    envelope["scenarios"]["baseline"]["agribrain"]["ari"] += 0.01
    seed_path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match the retained decision ledger"):
        ledgers.validate_inventory(root, seed_root)

    envelope["scenarios"]["baseline"]["agribrain"] = {
        **recomputed,
        **evidence,
        **learner_envelope,
        "mean_decision_latency_ms_descriptive_only": True,
        "latency_penalty_usd_descriptive_only": True,
    }
    envelope["traces"]["baseline"]["agribrain"]["carbon_trace"][0] += 0.01
    seed_path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"carbon_trace\[0\] differs from its ledger"):
        ledgers.validate_inventory(root, seed_root)


def test_single_seed_exports_protocol_method_counts_from_episode_evidence(
    tmp_path, monkeypatch,
):
    episode = {
        "ari": 0.7,
        "waste": 0.1,
        "rle": 0.8,
        "slca": 0.9,
        "carbon": 1.2,
        "equity": 0.6,
        "carbon_efficiency_ari_per_kgco2e_proxy": 0.7 / 1.2,
        "protocol_interaction_count": 10,
        "protocol_tools_call_count": 7,
        "protocol_prompts_get_count": 3,
    }
    monkeypatch.setattr(
        single_seed, "run_all",
        lambda *, seed: {"results": {"baseline": {"agribrain": episode}}},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_single_seed.py", "42", "--output-dir", str(tmp_path),
        ],
    )
    monkeypatch.delenv("STRICT_VALIDATION", raising=False)

    single_seed.main()

    payload = json.loads(
        (tmp_path / "seed_42.json").read_text(encoding="utf-8")
    )
    cell = payload["scenarios"]["baseline"]["agribrain"]
    assert cell["protocol_tools_call_count"] == 7
    assert cell["protocol_prompts_get_count"] == 3


def test_decision_ledger_gate_rejects_tampered_record(tmp_path, monkeypatch):
    monkeypatch.setattr(ledgers, "EXPECTED_SEEDS", (42,))
    monkeypatch.setattr(ledgers, "MODES", ["agribrain"])
    monkeypatch.setattr(ledgers, "SCENARIOS", ["baseline"])
    monkeypatch.setattr(ledgers, "EXPECTED_RECORDS", 2)
    path = tmp_path / "run" / "seed_42" / "agribrain__baseline.jsonl"
    _write_ledger(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[1])
    first["reward"] = float(first["reward"]) + 0.2
    lines[1] = json.dumps(first)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="leaf hash mismatch"):
        ledgers.validate_inventory(tmp_path / "run")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda record: record["context_feature_contributions"][0].__setitem__(0, 0.1),
            "attribution does not reconstruct",
        ),
        (
            lambda record: record["regime_logit_bias"].__setitem__(0, 0.1),
            "regime_logit_bias violates the locked policy equation",
        ),
        (
            lambda record: record.__setitem__(
                "context_counterfactual_probs", [0.6, 0.2, 0.2],
            ),
            "context-ablation probability reconstruction mismatch",
        ),
        (
            lambda record: _mutate_primary_retrieval_binding(
                record,
                record_field="retrieval_top_score",
                evidence_field="top_citation_score",
                value=0.1,
            ),
            "confuses fused retrieval strength",
        ),
        (
            _mutate_rag_total_scale_consistently,
            "piRAG scale does not reconstruct",
        ),
        (
            lambda record: record[
                "context_modifier_theta_jacobian"
            ][0].__setitem__(0, 0.1),
            "stored learner Jacobian is inconsistent",
        ),
    ],
)
def test_decision_ledger_gate_rejects_semantically_inconsistent_traces(
    tmp_path, monkeypatch, mutate, message,
):
    monkeypatch.setattr(ledgers, "EXPECTED_SEEDS", (42,))
    monkeypatch.setattr(ledgers, "MODES", ["agribrain"])
    monkeypatch.setattr(ledgers, "SCENARIOS", ["baseline"])
    monkeypatch.setattr(ledgers, "EXPECTED_RECORDS", 2)
    path = tmp_path / "run" / "seed_42" / "agribrain__baseline.jsonl"
    _write_ledger(path)
    _rehash_after_mutation(path, mutate)
    with pytest.raises(RuntimeError, match=message):
        ledgers.validate_inventory(tmp_path / "run")


def test_context_trace_validator_rejects_pirag_multiplier_in_standard_rag():
    trace = _record(0)["context_integration"]["primary"]
    trace["retrieval_kind"] = "standard"
    ledgers._validate_forward_context_trace(
        trace,
        theta=[[0.0] * 5 for _ in range(3)],
        expected_retrieval_kind="standard",
        where="fixture/standard",
    )

    trace["temporal_scale"] = 0.8
    with pytest.raises(
        RuntimeError, match="records an unapplied temporal gate",
    ):
        ledgers._validate_forward_context_trace(
            trace,
            theta=[[0.0] * 5 for _ in range(3)],
            expected_retrieval_kind="standard",
            where="fixture/standard",
        )


def test_context_trace_validator_reconstructs_temporal_gate_equation():
    trace = _record(0)["context_integration"]["primary"]
    trace.update({
        "temporal_gate_requested": True,
        "temporal_gate_applied": True,
        "temporal_continuity_score": 0.5,
        "temporal_base": 1.3,
        "temporal_decay": 0.6,
        "temporal_scale": 1.0,
    })
    ledgers._validate_forward_context_trace(
        trace,
        theta=[[0.0] * 5 for _ in range(3)],
        expected_retrieval_kind="pirag",
        where="fixture/pirag",
    )

    trace["temporal_scale"] = 0.9
    with pytest.raises(RuntimeError, match="temporal gate does not reconstruct"):
        ledgers._validate_forward_context_trace(
            trace,
            theta=[[0.0] * 5 for _ in range(3)],
            expected_retrieval_kind="pirag",
            where="fixture/pirag",
        )


def _write_h3_fixture_ledger(
    path: Path, *, seed: int, scenario: str, stressor: str,
) -> dict:
    import src.models.action_selection as action_selection
    from pirag.context_to_logits import THETA_CONTEXT
    from src.agents.roles import ROLE_BIASES, stage_for_hour

    treatment = (
        {
            "stressor": "nominal", "n_steps": 288,
            "data_observation_treatment": False,
            "delay_steps": 0, "missing_count": 0,
        }
        if stressor == "nominal"
        else raw._expected_observation_treatment(
            scenario=scenario, stressor=stressor, seed=seed,
        )
    )
    cell_seed = int.from_bytes(hashlib.sha256(
        f"stress|{scenario}|{stressor}|{seed}|3".encode()
    ).digest()[:8], "big")
    rng = np.random.default_rng(cell_seed)
    temp_noise = np.zeros(288)
    rh_noise = np.zeros(288)
    missing = np.zeros(288, dtype=bool)
    if stressor in {"sensor_noise", "compounded"}:
        temp_noise = rng.normal(0.0, 2.0, 288)
        rh_noise = rng.normal(0.0, 5.0, 288)
    if stressor in {"missing_data", "compounded"}:
        missing = rng.random(288) < 0.10
        missing[0] = False
    delay = 4 if stressor in {"telemetry_delay", "compounded"} else 0

    policy = ledgers.Policy()
    stoch = ledgers._expected_publication_stochastic_layer(
        benchmark_seed=seed, scenario=scenario, episode_index=3,
    )
    base = pd.read_csv(ledgers.PUBLICATION_DATA_CSV, parse_dates=["timestamp"])
    scenario_frame = ledgers.apply_scenario(
        base,
        scenario,
        policy,
        np.random.default_rng(
            ledgers._stream_seed(seed, scenario, 3, "scenario")
        ),
        stoch=stoch,
    )
    timestamps = pd.to_datetime(scenario_frame["timestamp"])
    hours = (
        (timestamps - timestamps.iloc[0]).dt.total_seconds() / 3600.0
    ).to_numpy(dtype=float)
    latent_temp = scenario_frame["tempC"].to_numpy(dtype=float)
    latent_rh = scenario_frame["RH"].to_numpy(dtype=float)
    latent_inventory = scenario_frame["inventory_units"].to_numpy(dtype=float)
    latent_demand = scenario_frame["demand_units"].to_numpy(dtype=float)
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
    effective_k_ref = stoch.perturb_k_ref(policy.k_ref, counter=0)
    effective_ea_r = stoch.perturb_ea_r(policy.Ea_R, counter=0)
    policy_temperature = stoch.policy_temperature(base=1.0, counter=0)

    canonical_temp: list[float] = []
    canonical_rh: list[float] = []
    predelay_temp: list[float] = []
    predelay_rh: list[float] = []
    temp_observed: list[float] = []
    rh_observed: list[float] = []
    inventory_observed: list[float] = []
    demand_observed: list[float] = []
    demand_forecasts: list[float] = []
    demand_stds: list[float] = []
    supply_forecasts: list[float] = []
    supply_stds: list[float] = []
    regimes: list[float] = []
    prices: list[float] = []
    transport: list[float] = []
    latent_rho: list[float] = []
    observed_rho: list[float] = []
    for index in range(288):
        temp = stoch.perturb_temperature(float(source_temp[index]), counter=index)
        rh = stoch.perturb_humidity(float(source_rh[index]), counter=index)
        if index > 0 and stoch.should_delay(counter=index):
            temp = canonical_temp[-1]
            rh = canonical_rh[-1]
        canonical_temp.append(float(temp))
        canonical_rh.append(float(rh))
        if stressor in {"sensor_noise", "compounded"}:
            temp += float(temp_noise[index])
            rh = float(np.clip(rh + float(rh_noise[index]), 15.0, 100.0))
        if stressor in {"missing_data", "compounded"} and bool(missing[index]):
            temp = predelay_temp[-1]
            rh = predelay_rh[-1]
        predelay_temp.append(float(temp))
        predelay_rh.append(float(rh))
        source_step = max(index - delay, 0)
        if delay:
            temp = predelay_temp[source_step]
            rh = predelay_rh[source_step]
        temp_observed.append(float(temp))
        rh_observed.append(float(rh))
        inventory_observed.append(float(stoch.perturb_inventory(
            float(source_inventory[index]), counter=index,
        )))
        demand_observed.append(float(stoch.perturb_demand(
            float(source_demand[index]), counter=index,
        )))
        if index == 0:
            latent_rho.append(0.0)
            observed_rho.append(0.0)
        else:
            latent_rho.append(float(ledgers.advance_spoilage_risk_midpoint(
                latent_rho[-1],
                previous_temp_C=float(latent_temp[index - 1]),
                current_temp_C=float(latent_temp[index]),
                previous_rh_pct=float(latent_rh[index - 1]),
                current_rh_pct=float(latent_rh[index]),
                previous_hour=float(hours[index - 1]),
                current_hour=float(hours[index]),
                k_ref=effective_k_ref, Ea_R=effective_ea_r,
                T_ref_K=policy.T_ref_K, beta=policy.beta_humidity,
                lag_lambda=policy.lag_lambda,
            )))
            observed_rho.append(float(ledgers.advance_spoilage_risk_midpoint(
                observed_rho[-1],
                previous_temp_C=temp_observed[index - 1],
                current_temp_C=temp_observed[index],
                previous_rh_pct=rh_observed[index - 1],
                current_rh_pct=rh_observed[index],
                previous_hour=float(hours[index - 1]),
                current_hour=float(hours[index]),
                k_ref=effective_k_ref, Ea_R=effective_ea_r,
                T_ref_K=policy.T_ref_K, beta=policy.beta_humidity,
                lag_lambda=policy.lag_lambda,
            )))
        lookback = min(index + 1, 48)
        demand_tail = demand_observed[-lookback:]
        demand_result = ledgers.yield_demand_forecast(
            pd.DataFrame({"demand_units": demand_tail}), horizon=1,
        )
        supply_result = ledgers.persistence_forecast(
            pd.DataFrame({
                "inventory_units": source_inventory[
                    max(0, index + 1 - lookback):index + 1
                ]
            }),
            horizon=1,
            series_col="inventory_units",
        )
        demand_forecasts.append(float(demand_result["forecast"][0]))
        demand_stds.append(float(demand_result.get("std", 0.0) or 0.0))
        supply_forecasts.append(float(supply_result["forecast"][0]))
        supply_stds.append(float(supply_result.get("std", 0.0) or 0.0))
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
        prices.append(float(np.clip(z_score, -1.0, 1.0)))
        regimes.append(float(abs(z_score) > float(policy.boll_k)))
        transport.append(float(stoch.perturb_transport_multiplier(counter=index)))

    outcome_contract = ledgers.expected_publication_outcome_equation_contract(
        benchmark_seed=seed, scenario=scenario,
    )
    checkpoint = ledgers.load_frozen_checkpoint()
    latent_dgp = ledgers.compute_spoilage_independent_synthetic_dgp(
        scenario_frame,
        k_ref=effective_k_ref,
        Ea_R=effective_ea_r,
        T_ref_K=policy.T_ref_K,
        beta=policy.beta_humidity,
        lag_lambda=policy.lag_lambda,
    )
    latent_rho = latent_dgp["spoilage_risk"].to_numpy(dtype=float).tolist()
    observed_mechanistic_rho = list(observed_rho)
    shock_values = scenario_frame.get(
        "shockG", pd.Series(np.zeros(288), index=scenario_frame.index),
    ).to_numpy(dtype=float)
    observed_rho = []
    observed_deployed_quality = 1.0
    for index in range(288):
        rh_transient = 0.0
        if index > 0:
            step_h = float(hours[index] - hours[index - 1])
            rh_transient = float(
                abs(rh_observed[index] - rh_observed[index - 1]) / step_h
            )
        features = ledgers.build_residual_feature_row(
            time_h=float(hours[index]),
            temp_c=float(temp_observed[index]),
            rh_pct=float(rh_observed[index]),
            shock_g=float(shock_values[index]),
            rh_transient_per_h=rh_transient,
            k_ref=effective_k_ref,
            ea_over_r=effective_ea_r,
        )
        delta = float(ledgers.predict_residual(features, checkpoint)[0])
        observed_deployed_quality = min(
            observed_deployed_quality,
            float(np.clip(
                1.0 - observed_mechanistic_rho[index] + delta, 0.0, 1.0,
            )),
        )
        observed_rho.append(1.0 - observed_deployed_quality)
    spoilage_estimator = {
        "kind": "mechanistic_plus_frozen_synthetic_pinn_residual",
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "training_dataset_sha256": checkpoint.dataset_sha256,
        "training_target_origin": "independent_synthetic_dgp",
        "residual_bound_abs": 0.08,
        "deployment_transform": (
            "clip_quality_to_unit_interval_then_cumulative_minimum"
        ),
        "synthetic_only": True,
        "external_validation": False,
    }
    theta = ledgers.policy_theta_for_seed(
        np.asarray(ledgers.DECLARED_THETA, dtype=float), seed,
    )
    context_theta = np.asarray(THETA_CONTEXT, dtype=float)
    zero_context_features = [[0.0] * 5 for _ in range(3)]
    zero_theta_delta = [[0.0] * 10 for _ in range(3)]
    records = []
    leaves = []
    for index in range(288):
        source = max(index - delay, 0)
        faulted = stressor in {"mcp_fault_injection", "compounded"}
        scheduled = bool(faulted and int(index * 0.25) % 11 == 0)
        phi = action_selection.build_feature_vector(
            observed_rho[index], inventory_observed[index],
            demand_forecasts[index], temp_observed[index],
            supply_hat=supply_forecasts[index],
            supply_std=supply_stds[index], demand_std=demand_stds[index],
            price_signal=prices[index],
        )
        role = stage_for_hour(float(hours[index]))
        peer_bias = np.asarray(
            [0.0, 0.0, 0.05 * 0.2]
            if index < 4 else [0.0, 0.0, 0.0],
            dtype=float,
        )
        combined_bias = ROLE_BIASES[role].astype(float).copy()
        if 12.0 <= float(hours[index]) < 30.0:
            combined_bias += ROLE_BIASES["cooperative"]
        combined_bias += peer_bias
        shaping = (
            np.asarray(action_selection.SLCA_BONUS, dtype=float)
            + np.asarray(action_selection.SLCA_RHO_BONUS, dtype=float)
            * observed_rho[index]
        )
        regime_bias = np.asarray([
            policy.gamma_coldchain,
            policy.gamma_local,
            policy.gamma_recovery,
        ], dtype=float)
        base_logits = theta @ phi + regime_bias * regimes[index] + shaping
        if observed_rho[index] > action_selection.RHO_RECOVERY_KNEE:
            excess = (
                (observed_rho[index] - action_selection.RHO_RECOVERY_KNEE)
                / (1.0 - action_selection.RHO_RECOVERY_KNEE)
            )
            base_logits[2] += action_selection.RHO_RECOVERY_KNEE_GAIN * excess
            base_logits[1] -= (
                action_selection.RHO_RECOVERY_KNEE_LR_PENALTY * excess
            )
        base_logits += combined_bias
        post_logits = base_logits / policy_temperature
        preoverride_probs = np.asarray(ledgers._softmax(
            [float(value) for value in post_logits]
        ))
        categorical_uniform = ledgers._policy_categorical_uniform(
            ledgers._stream_seed(seed, scenario, 3, "policy"), index,
        )
        sampled_action = ledgers.categorical_action_from_uniform(
            preoverride_probs, categorical_uniform,
        )
        governance_override = bool(
            preoverride_probs[0]
            < action_selection.GOVERNANCE_CC_PROB_CEILING
            and preoverride_probs[1] - preoverride_probs[0]
            > action_selection.GOVERNANCE_LOCAL_ADVANTAGE_MIN
        )
        action_idx = 1 if governance_override else sampled_action
        probs = (
            np.asarray([0.0, 1.0, 0.0])
            if governance_override else preoverride_probs
        )
        primary_integration = {
            "context_mode": "full", "retrieval_kind": "pirag",
            "over_steer": False, "clip_applied": True,
            "effective_theta": context_theta.tolist(),
            "effective_psi": [0.0] * 5,
            "linear_feature_contributions": zero_context_features,
            "channel_scaled_feature_contributions": zero_context_features,
            "feature_contributions": zero_context_features,
            "nonfeature_residual": [0.0] * 3,
            "mcp_preclip_component": [0.0] * 3,
            "pirag_preclip_component": [0.0] * 3,
            "retrieval_gate": 0.0,
            "retrieval_blocked_reason": "retrieval_guard",
            "temporal_scale": 1.0,
            "temporal_gate_requested": False,
            "temporal_gate_applied": False,
            "temporal_continuity_score": None,
            "temporal_base": None,
            "temporal_decay": None,
            "physics_scale": 1.0,
            "rag_total_scale": 0.0, "global_scale": 1.0,
            "preclip_modifier": [0.0] * 3,
            "clip_derivative": [1.0] * 3,
            "modifier_theta_jacobian": zero_context_features,
            "final_modifier": [0.0] * 3, "blocked_reason": None,
        }
        context_integration = {
            "primary": primary_integration,
            "cooperative": None,
            "composition": {
                "scope": "primary_context", "clip_applied": True,
                "preclip_modifier": [0.0] * 3,
                "clip_derivative": [1.0] * 3,
                "modifier_theta_jacobian": zero_context_features,
                "final_modifier": [0.0] * 3,
            },
        }
        retrieval_attempted = bool(index < 10)
        record = {
            "step_index": index,
            "hour": float(hours[index]),
            "mode": "agribrain",
            "scenario": scenario,
            "action": ledgers.ACTIONS[action_idx],
            "action_idx": int(action_idx),
            "probs": probs.tolist(),
            "policy_probs_pre_override": preoverride_probs.tolist(),
            "policy_categorical_uniform": float(categorical_uniform),
            "sampled_action_pre_override": int(sampled_action),
            "decision_latency_ms": 2.0,
            "reward": 0.0,
            "waste": 0.1,
            "slca": 0.8,
            "rho": observed_rho[index],
            "rho_outcome_environmental": latent_rho[index],
            "rho_policy_observed": observed_rho[index],
            "shock_g": float(shock_values[index]),
            "ari": 0.72,
            "carbon_kg": 100.0 / 288.0,
            "temp_policy_observed": temp_observed[index],
            "temp_outcome_environmental": float(latent_temp[index]),
            "rh_policy_observed": rh_observed[index],
            "rh_outcome_environmental": float(latent_rh[index]),
            "inventory_policy_observed": inventory_observed[index],
            "inventory_outcome_environmental": float(latent_inventory[index]),
            "demand_policy_observed": demand_observed[index],
            "demand_outcome_environmental": float(latent_demand[index]),
            "demand_forecast_policy_observed": demand_forecasts[index],
            "demand_forecast_std_policy_observed": demand_stds[index],
            "supply_forecast_policy_observed": supply_forecasts[index],
            "supply_forecast_std_policy_observed": supply_stds[index],
            "bollinger_regime_flag": regimes[index],
            "regime_logit_bias": (
                regime_bias * regimes[index]
            ).tolist(),
            "price_signal": prices[index],
            "transport_multiplier_outcome_environmental": transport[index],
            "role": role,
            "phi": phi.tolist(),
            "peer_message_bias": peer_bias.tolist(),
            "combined_role_bias": combined_bias.tolist(),
            "effective_theta_delta": zero_theta_delta,
            "effective_slca_bonus_delta": [0.0] * 3,
            "effective_slca_rho_delta": [0.0] * 3,
            "effective_no_slca_offset_delta": [0.0] * 3,
            "psi": [0.0] * 5,
            "context_modifier": [0.0, 0.0, 0.0],
            "base_logits": base_logits.tolist(),
            "post_context_logits_pre_override": post_logits.tolist(),
            "slca_shaping": shaping.tolist(),
            "slca_amp": 0.0,
            "policy_temperature": float(policy_temperature),
            "modifier_mcp": [0.0] * 3,
            "modifier_pirag": [0.0] * 3,
            "retrieval_top_doc_id": "",
            "retrieval_top_score": 0.0,
            "retrieval_top_fused_score": 0.0,
            "retrieval_top_rerank_score": 0.0,
            "retrieval_evidence_hashes": [],
            "effective_context_theta": context_theta.tolist(),
            "context_feature_contributions": zero_context_features,
            "context_nonfeature_residual": [0.0] * 3,
            "context_modifier_theta_jacobian": zero_context_features,
            "context_integration": context_integration,
            "chosen_action_context_contributions": [0.0] * 5,
            "chosen_action_context_residual": 0.0,
            "context_attribution_basis": (
                "final_modifier_feature_allocation_plus_explicit_residual"
            ),
            "context_attribution_scope": "primary_context",
            "dominant_psi_idx": 0,
            "dominant_context_component": "psi_0",
            "dominant_action_idx": 0,
            "governance_override": governance_override,
            "context_counterfactual_action_idx": int(sampled_action),
            "context_counterfactual_action": ledgers.ACTIONS[sampled_action],
            "context_counterfactual_probs": preoverride_probs.tolist(),
            "context_counterfactual_categorical_uniform": float(
                categorical_uniform
            ),
            "context_counterfactual_sampled_action_pre_override": int(
                sampled_action
            ),
            "context_action_changed": bool(action_idx != sampled_action),
            "context_influence_active": False,
            "context_influence_counted": False,
            "context_influence_threshold": 0.10,
            "simulated_dispatch_accounted": True,
            "primary_mcp_tools_invoked_step": [],
            "cooperative_mcp_tools_invoked_step": [],
            "primary_pirag_query_attempted_step": retrieval_attempted,
            "cooperative_pirag_query_attempted_step": False,
            "mcp_tool_call_count_step": 0,
            "pirag_query_count_step": int(retrieval_attempted),
            "dispatcher_tool_failure_count_step": 0,
            "inter_agent_message_count_step": int(index < 4),
            "protocol_interaction_count_step": int(retrieval_attempted),
            "protocol_tools_call_count_step": 0,
            "protocol_prompts_get_count_step": int(retrieval_attempted),
            "protocol_jsonrpc_error_count_step": 0,
            "protocol_tool_iserror_count_step": 0,
            "protocol_real_tool_iserror_count_step": 0,
            "protocol_error_count_step": 0,
            "protocol_dropped_interaction_count_step": 0,
            "h3_stressor": stressor,
            "h3_data_observation_treatment": stressor not in {
                "nominal", "mcp_fault_injection",
            },
            "h3_temp_noise_c": float(temp_noise[index]),
            "h3_rh_noise_pct": float(rh_noise[index]),
            "h3_missing_observation": bool(missing[index]),
            "h3_telemetry_source_step_index": source,
            "h3_fault_injection_scheduled_opportunity": scheduled,
            "h3_fault_injection_triggered": scheduled,
            "h3_fault_injected_tool_result_count": 3 if scheduled else 0,
        }
        reconstructed = reconstruct_step_outcomes(record, outcome_contract)
        for field in ("reward", "waste", "slca", "ari", "carbon_kg"):
            record[field] = reconstructed[field]
        record["step_channel_evidence"] = _sealed_step_channel_evidence(record)
        leaf = hashlib.sha256(json.dumps(
            record, sort_keys=True, separators=(",", ":"), default=str,
        ).encode()).hexdigest()
        records.append(record)
        leaves.append(leaf)
    metadata = {
        "mode": "agribrain", "scenario": scenario, "seed": seed,
        "benchmark_seed": seed, "episode_index": 3,
        "learning_enabled": False, "episode_phase": "frozen_evaluation",
        "trace_schema_version": raw.TRACE_SCHEMA_VERSION,
        "environment_stream_id": (
            f"seed={seed};scenario={scenario};episode=3;stream=environment"
        ),
        "stochastic_stream_id": (
            f"seed={seed};scenario={scenario};episode=3;stream=environment"
        ),
        "policy_stream_id": (
            f"seed={seed};scenario={scenario};episode=3;stream=policy"
        ),
        "context_prior_sha256": ledgers._canonical_sha256(
            np.asarray(THETA_CONTEXT, dtype=float).tolist()
        ),
        "policy_theta_initial_sha256": ledgers._canonical_sha256(
            ledgers.policy_theta_for_seed(
                np.asarray(ledgers.DECLARED_THETA, dtype=float), seed,
            ).tolist()
        ),
        "demand_forecast_method": "holt_linear",
        "supply_forecast_method": "persistence",
        "dispatch_opportunity_count": 288,
        "dispatch_cadence_hours": 0.25,
        "effective_k_ref": effective_k_ref,
        "effective_Ea_R": effective_ea_r,
        "scenario_onset_offset_hours": float(
            scenario_frame["scenario_onset_offset_hours"].iloc[0]
            if "scenario_onset_offset_hours" in scenario_frame.columns else 0.0
        ),
        "observation_treatment": treatment,
        "outcome_equation_contract": outcome_contract,
        "episode_evidence_contract": (
            ledgers.expected_publication_episode_evidence_contract()
        ),
        "spoilage_estimator": spoilage_estimator,
        "latent_spoilage_model": ledgers.synthetic_dgp_provenance(
            k_ref=effective_k_ref,
            Ea_R=effective_ea_r,
            T_ref_K=policy.T_ref_K,
            beta=policy.beta_humidity,
            lag_lambda=policy.lag_lambda,
        ),
    }
    latent_payload = {
        "hours": [r["hour"] for r in records],
        "temp_outcome_environmental": [r["temp_outcome_environmental"] for r in records],
        "rh_outcome_environmental": [r["rh_outcome_environmental"] for r in records],
        "rho_outcome_environmental": [r["rho_outcome_environmental"] for r in records],
        "inventory_outcome_environmental": [r["inventory_outcome_environmental"] for r in records],
        "demand_outcome_environmental": [r["demand_outcome_environmental"] for r in records],
        "transport_multiplier_outcome_environmental": [r["transport_multiplier_outcome_environmental"] for r in records],
        "effective_k_ref": metadata["effective_k_ref"],
        "effective_Ea_R": metadata["effective_Ea_R"],
        "scenario_onset_offset_hours": metadata["scenario_onset_offset_hours"],
    }
    observed_payload = {
        "hours": [r["hour"] for r in records],
        "temp_policy_observed": [r["temp_policy_observed"] for r in records],
        "rh_policy_observed": [r["rh_policy_observed"] for r in records],
        "rho_policy_observed": [r["rho_policy_observed"] for r in records],
        "inventory_policy_observed": [r["inventory_policy_observed"] for r in records],
        "demand_forecast_policy_observed": [r["demand_forecast_policy_observed"] for r in records],
        "supply_forecast_policy_observed": [r["supply_forecast_policy_observed"] for r in records],
    }
    demand_payload = {
        "hours": [r["hour"] for r in records],
        "demand_policy_observed": [r["demand_policy_observed"] for r in records],
        "demand_forecast_policy_observed": [r["demand_forecast_policy_observed"] for r in records],
        "demand_regime_flag": [r["bollinger_regime_flag"] for r in records],
        "price_signal": [r["price_signal"] for r in records],
    }
    metadata.update({
        "latent_environment_sha256": raw._canonical_object_sha256(latent_payload),
        "observed_policy_input_sha256": raw._canonical_object_sha256(observed_payload),
        "demand_observation_sha256": raw._canonical_object_sha256(demand_payload),
    })
    header = {
        "_header": True, "n_records": 288,
        "merkle_root": raw._ledger_merkle_root(leaves), "metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(header, sort_keys=True) + "\n")
        for record, leaf in zip(records, leaves, strict=True):
            handle.write(json.dumps({**record, "_leaf": leaf}, sort_keys=True) + "\n")
    step_slca = [float(record["slca"]) for record in records]
    return {
        "ari": math.fsum(float(record["ari"]) for record in records) / 288,
        "waste": math.fsum(float(record["waste"]) for record in records) / 288,
        "slca": math.fsum(step_slca) / 288,
        "rle": raw.compute_rle(
            [float(record["rho_outcome_environmental"]) for record in records],
            [str(record["action"]) for record in records],
        ),
        "carbon": math.fsum(float(record["carbon_kg"]) for record in records),
        "equity": raw.compute_equity(step_slca),
        "context_prior_sha256": metadata["context_prior_sha256"],
        "policy_theta_initial_sha256": metadata[
            "policy_theta_initial_sha256"
        ],
        "latent_environment_sha256": metadata["latent_environment_sha256"],
        "observed_policy_input_sha256": metadata["observed_policy_input_sha256"],
        "demand_observation_sha256": metadata["demand_observation_sha256"],
        "spoilage_estimator": dict(metadata["spoilage_estimator"]),
        "latent_spoilage_model": dict(metadata["latent_spoilage_model"]),
        "decision_ledger_sha256": raw._sha256_file(path),
        "decision_ledger_merkle_root": header["merkle_root"],
        "decision_ledger_n_records": 288,
        "fault_injection_scheduled_opportunity_steps": sum(
            r["h3_fault_injection_scheduled_opportunity"] for r in records
        ),
        "fault_injection_trigger_steps": sum(
            r["h3_fault_injection_triggered"] for r in records
        ),
        "fault_injected_tool_result_count": sum(
            r["h3_fault_injected_tool_result_count"] for r in records
        ),
    }


def _write_raw_fixture(root: Path, *, commit: str, tag: str) -> tuple[Path, Path]:
    seeds = root / "seeds"
    stress = root / "stress"
    h3_ledgers = root / "decision_ledger_h3"
    seeds.mkdir()
    def learner_fields(mode: str) -> dict:
        from pirag.context_to_logits import THETA_CONTEXT

        caps = raw.capabilities_for(mode)
        no_reversals = {
            "sign_reversal_count": 0,
            "sign_reversal_coordinates": [],
            "worst_sign_reversal": None,
        }
        payload = {
            "learner_summary": None,
            "theta_learner_summary": None,
            "reward_shaping_learner_summary": None,
        }
        if caps.context_matrix_learning:
            context_summary = {
                "mode": mode,
                "learner_state_schema_version": 2,
                "final_theta": np.asarray(
                    THETA_CONTEXT, dtype=float,
                ).tolist(),
                "final_slca_amp": 0.0,
                "learn_proxy_interaction": False,
                "temporal_base": 1.0,
                "temporal_scale": 0.0,
                "reward_baseline": 0.5,
                "n_updates": 4,
                "sign_constrained": caps.sign_constrained_learning,
                "sign_preserved": True,
                "compliance_sign_reversal_count": 0,
                "worst_compliance_sign_reversal": None,
                **no_reversals,
            }
            context_state = {
                "theta": context_summary["final_theta"],
                "slca_amp_coeff": context_summary["final_slca_amp"],
                "learn_proxy_interaction": context_summary[
                    "learn_proxy_interaction"
                ],
                "sign_constrained": context_summary["sign_constrained"],
                "temporal_base": context_summary["temporal_base"],
                "temporal_scale": context_summary["temporal_scale"],
                "reward_baseline": context_summary["reward_baseline"],
                "n_updates": context_summary["n_updates"],
            }
            context_summary["state_sha256"] = raw._canonical_object_sha256(
                context_state
            )
            payload["learner_summary"] = context_summary
        if caps.policy_delta_learning:
            updates = {role: 4 for role in raw.DECISION_OWNER_ROLES}
            per_role = {}
            role_states = {}
            per_role_hashes = {}
            for role in raw.DECISION_OWNER_ROLES:
                role_summary = {
                    "final_theta_delta": [[0.0] * 10 for _ in range(3)],
                    "reward_baseline": 0.5,
                    "n_updates": 4,
                    "learning_rate": 0.003,
                    "prior_precision": 1.0,
                    "magnitude_cap_fraction": 0.25,
                    "sign_constrained": caps.sign_constrained_learning,
                    **no_reversals,
                }
                role_state = {
                    "theta_delta": role_summary["final_theta_delta"],
                    "reward_baseline": role_summary["reward_baseline"],
                    "n_updates": role_summary["n_updates"],
                    "learning_rate": role_summary["learning_rate"],
                    "prior_precision": role_summary["prior_precision"],
                    "magnitude_cap_fraction": role_summary[
                        "magnitude_cap_fraction"
                    ],
                    "sign_constrained": role_summary["sign_constrained"],
                }
                per_role[role] = role_summary
                role_states[role] = role_state
                per_role_hashes[role] = raw._canonical_object_sha256(
                    role_state
                )
            payload["theta_learner_summary"] = {
                "mode": mode,
                "learner_state_schema_version": 2,
                "decision_owner_roles": list(raw.DECISION_OWNER_ROLES),
                "updates_per_role": updates,
                "per_role_state_sha256": per_role_hashes,
                "combined_state_sha256": raw._canonical_object_sha256(
                    role_states
                ),
                "n_updates": sum(updates.values()),
                "sign_constrained": caps.sign_constrained_learning,
                **no_reversals,
                "per_role": per_role,
            }
        if caps.reward_shaping_learning:
            shaping_summary = {
                "mode": mode,
                "learner_state_schema_version": 2,
                "slca_bonus_delta": [0.0] * 3,
                "slca_rho_delta": [0.0] * 3,
                "no_slca_offset_delta": [0.0] * 3,
                "reward_baseline": 0.5,
                "n_updates": 4,
                "magnitude_cap_fraction": 0.25,
                "sign_constrained": caps.sign_constrained_learning,
                **no_reversals,
            }
            shaping_state = {
                "slca_bonus_delta": shaping_summary["slca_bonus_delta"],
                "slca_rho_delta": shaping_summary["slca_rho_delta"],
                "no_slca_offset_delta": shaping_summary[
                    "no_slca_offset_delta"
                ],
                "reward_baseline": shaping_summary["reward_baseline"],
                "n_updates": shaping_summary["n_updates"],
                "magnitude_cap_fraction": shaping_summary[
                    "magnitude_cap_fraction"
                ],
                "sign_constrained": shaping_summary["sign_constrained"],
            }
            shaping_summary["state_sha256"] = raw._canonical_object_sha256(
                shaping_state
            )
            payload["reward_shaping_learner_summary"] = shaping_summary
        return payload

    def metric_cell(mode: str, seed: int, *, ari: float = 0.8,
                    latent_hash: str | None = None,
                    observed_hash: str | None = None,
                    stressor: str = "nominal",
                    scheduled: int = 0, triggered: int = 0,
                    replaced: int = 0) -> dict:
        treatment = (
            {
                "stressor": "nominal",
                "n_steps": 288,
                "data_observation_treatment": False,
                "delay_steps": 0,
                "missing_count": 0,
            }
            if stressor == "nominal"
            else raw._expected_observation_treatment(
                scenario="baseline", stressor=stressor, seed=seed,
            )
        )
        freeze_summary = {
            "learners_frozen": True,
            "learner_phase": "frozen_evaluation",
            "freeze_reason": "retained_episode_3",
            "context_matrix_frozen": True,
            "policy_delta_frozen_by_role": {
                role: True for role in raw.DECISION_OWNER_ROLES
            },
            "reward_shaping_frozen": True,
            "external_policy_learners_frozen": 0,
        }
        ledger_path = (
            seeds / f"decision_ledger_{seed}" / "agribrain__baseline.jsonl"
            if stressor == "nominal"
            else h3_ledgers / "baseline" / stressor / f"seed_{seed}"
            / "agribrain__baseline.jsonl"
        )
        ledger = _write_h3_fixture_ledger(
            ledger_path, seed=seed, scenario="baseline", stressor=stressor,
        )
        if stressor != "nominal":
            arm_root = ledger_path.parent
            for episode_index in range(3):
                adaptation_path = (
                    arm_root / "adaptation_episode_ledgers"
                    / "agribrain__baseline"
                    / f"episode_{episode_index}.jsonl.gz"
                )
                adaptation_path.parent.mkdir(parents=True, exist_ok=True)
                adaptation_path.write_bytes(b"fixture adaptation ledger\n")
            for episode_index in range(4):
                archive_path = (
                    arm_root / "complete_episode_evidence"
                    / "agribrain__baseline"
                    / f"episode_{episode_index}.json.gz"
                )
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                archive_path.write_bytes(b"fixture episode archive\n")
        canonical_ledger_path = (
            f"decision_ledger_per_seed/{tag}/seed_{seed}/agribrain__baseline.jsonl"
            if stressor == "nominal"
            else (
                f"decision_ledger_h3/{tag}/baseline/{stressor}/seed_{seed}/"
                "agribrain__baseline.jsonl"
            )
        )
        return {
            "ari": ledger["ari"],
            "waste": ledger["waste"],
            "slca": ledger["slca"],
            "rle": ledger["rle"],
            "carbon": ledger["carbon"],
            "equity": ledger["equity"],
            "message_count": (
                4 if raw.capabilities_for(mode).peer_messages else 0
            ),
            "constraint_violation_rate": 0.05,
            "mean_decision_latency_ms": 2.0,
            "decision_latency_ms": 2.0,
            "downstream_violation_rate": 0.02,
            "contained_violation_rate": 0.98,
            "trace_schema_version": raw.TRACE_SCHEMA_VERSION,
            "benchmark_seed": seed,
            "episode_index": 3,
            "learning_enabled": False,
            "episode_phase": "frozen_evaluation",
            "environment_stream_id": (
                f"seed={seed};scenario=baseline;episode=3;stream=environment"
            ),
            "stochastic_stream_id": (
                f"seed={seed};scenario=baseline;episode=3;stream=environment"
            ),
            "policy_stream_id": (
                f"seed={seed};scenario=baseline;episode=3;stream=policy"
            ),
            "context_prior_sha256": ledger["context_prior_sha256"],
            "policy_theta_initial_sha256": ledger[
                "policy_theta_initial_sha256"
            ],
            "spoilage_estimator": dict(ledger["spoilage_estimator"]),
            "latent_spoilage_model": dict(ledger["latent_spoilage_model"]),
            "dispatch_opportunity_count": 288,
            "dispatch_cadence_hours": 0.25,
            "latent_environment_sha256": ledger["latent_environment_sha256"],
            "observed_policy_input_sha256": ledger[
                "observed_policy_input_sha256"
            ],
            "demand_observation_sha256": ledger["demand_observation_sha256"],
            "demand_forecast_method": "holt_linear",
            "supply_forecast_method": "persistence",
            "protocol_interaction_count": 10,
            "protocol_jsonrpc_error_count": 0,
            "protocol_tool_iserror_count": 0,
            "protocol_real_tool_iserror_count": 0,
            "protocol_error_count": 0,
            "protocol_dropped_interaction_count": 0,
            "dispatcher_tool_failure_count": 0,
            "context_execution_error_count": 0,
            "fault_injection_scheduled_opportunity_steps": ledger[
                "fault_injection_scheduled_opportunity_steps"
            ],
            "fault_injection_trigger_steps": ledger[
                "fault_injection_trigger_steps"
            ],
            "fault_injected_tool_result_count": ledger[
                "fault_injected_tool_result_count"
            ],
            "observation_treatment": treatment,
            "decision_ledger_path": canonical_ledger_path,
            "decision_ledger_sha256": ledger["decision_ledger_sha256"],
            "decision_ledger_merkle_root": ledger[
                "decision_ledger_merkle_root"
            ],
            "decision_ledger_n_records": ledger["decision_ledger_n_records"],
            "learner_freeze_summary": freeze_summary,
            **learner_fields(mode),
        }

    hours = [0.25 * index for index in range(288)]
    rho = [0.001 * index for index in range(288)]
    waste = [0.1] * 288
    social = [0.8] * 288
    actions = [0] * 288
    probabilities = [[0.5, 0.3, 0.2] for _ in range(288)]
    slca_components = [
        {
            "C": 0.8,
            "L": 0.8,
            "R": 0.8,
            "P": 0.8,
            "composite": 0.8,
            "action_family": "cold_chain",
            "slca_quality": 1.0,
            "composite_attenuated": 0.8,
        }
        for _ in range(288)
    ]
    trace = {
        "hours": hours,
        "ari_trace": [
            (1.0 - w) * s * (1.0 - r)
            for w, s, r in zip(waste, social, rho, strict=True)
        ],
        "waste_trace": waste,
        "slca_trace": social,
        "rho_trace": rho,
        "rho_policy_observed_trace": rho,
        "rho_outcome_environmental_trace": rho,
        "action_trace": actions,
        "prob_trace": probabilities,
        "carbon_trace": [1.5] * 288,
        "temp_trace": [4.0] * 288,
        "temp_policy_observed_trace": [4.0] * 288,
        "temp_outcome_environmental_trace": [4.0] * 288,
        "rh_trace": [90.0] * 288,
        "rh_policy_observed_trace": [90.0] * 288,
        "rh_outcome_environmental_trace": [90.0] * 288,
        "inventory_trace": [1000.0] * 288,
        "inventory_policy_observed_trace": [1000.0] * 288,
        "inventory_outcome_environmental_trace": [1000.0] * 288,
        "demand_trace": [900.0] * 288,
        "demand_policy_observed_trace": [900.0] * 288,
        "demand_forecast_policy_observed_trace": [900.0] * 288,
        "demand_regime_flag_trace": [0.0] * 288,
        "price_signal_trace": [0.0] * 288,
        "supply_forecast_policy_observed_trace": [1000.0] * 288,
        "demand_outcome_environmental_trace": [900.0] * 288,
        "transport_multiplier_outcome_environmental_trace": [1.0] * 288,
        "simulated_dispatch_accounted_trace": [True] * 288,
        "slca_component_trace": slca_components,
        "equity_trace": [0.8] * 288,
        "reward_trace": [0.7] * 288,
    }
    assert set(trace) == set(trace_contract.TRACE_FIELDS)

    for seed in raw.EXPECTED_SEEDS:
        latent_hash = hashlib.sha256(f"latent:{seed}:baseline".encode()).hexdigest()
        cells = {
            mode: metric_cell(mode, seed, latent_hash=latent_hash)
            for mode in raw.MODES
        }
        seed_payload = {
            "_meta": {
                "source_commit": commit,
                "run_tag": tag,
                "episode_scope": raw.EPISODE_SCOPE,
                "decision_history_scope": raw.HISTORY_SCOPE,
                "trace_schema_version": raw.TRACE_SCHEMA_VERSION,
                "episode_accounting": raw._expected_seed_episode_accounting(),
            },
            "trace_schema_version": raw.TRACE_SCHEMA_VERSION,
            "seed": seed,
            "scenarios": {"baseline": cells},
            "traces": {
                "baseline": {mode: trace for mode in raw.TRACE_MODES},
            },
            "_trace_failures": [],
        }
        (seeds / f"seed_{seed}.json").write_text(
            json.dumps(seed_payload), encoding="utf-8",
        )

    scenario = stress / "baseline"
    scenario.mkdir(parents=True)
    baseline_by_seed = {}
    for seed in raw.EXPECTED_SEEDS:
        seed_path = seeds / f"seed_{seed}.json"
        seed_bytes = seed_path.read_bytes()
        seed_payload = json.loads(seed_bytes.decode("utf-8"))
        baseline_by_seed[str(seed)] = {}
        for mode in raw.BASELINE_STRESS_MODES:
            cell = metric_cell(mode, seed)
            cell["observation_treatment"]["source"] = (
                "reused_primary_benchmark"
            )
            primary_cell = seed_payload["scenarios"]["baseline"][mode]
            cell["primary_seed_envelope_sha256"] = hashlib.sha256(
                seed_bytes
            ).hexdigest()
            cell["primary_nominal_cell_sha256"] = (
                raw._canonical_object_sha256(primary_cell)
            )
            baseline_by_seed[str(seed)][mode] = cell
    scenario_results = {
        "baseline_seed_list": list(raw.EXPECTED_SEEDS),
        "baseline_by_seed": baseline_by_seed,
    }
    for stressor in raw.STRESSORS:
        faulted = stressor in {"mcp_fault_injection", "compounded"}
        scenario_results[stressor] = {
            str(seed): {
                mode: metric_cell(
                    mode,
                    seed,
                    stressor=stressor,
                    observed_hash=(
                        hashlib.sha256(
                            f"observed:{seed}:{stressor}".encode()
                        ).hexdigest()
                        if stressor != "mcp_fault_injection"
                        else hashlib.sha256(
                            f"observed:{seed}:baseline".encode()
                        ).hexdigest()
                    ),
                    scheduled=28 if faulted else 0,
                    triggered=(
                        28 if faulted and mode in {"agribrain", "mcp_only"}
                        else 0
                    ),
                    replaced=(
                        84 if faulted and mode in {"agribrain", "mcp_only"}
                        else 0
                    ),
                )
                for mode in raw.STRESS_MODES[stressor]
            }
            for seed in raw.EXPECTED_SEEDS
        }
    h3_scenario_root = h3_ledgers / "baseline"
    (h3_scenario_root / "complete_episode_evidence_manifest.json").write_text(
        "{}\n", encoding="utf-8",
    )
    receipt_path = (
        h3_scenario_root / "runtime_receipts"
        / "job_12345__restart_0.json"
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text("{}\n", encoding="utf-8")
    summary = {
        "meta": {
            "source_commit": commit,
            "run_tag": tag,
            "trace_schema_version": raw.TRACE_SCHEMA_VERSION,
            "scenarios": ["baseline"],
            "max_rows": None,
            "thresholds": raw.STRESS_THRESHOLDS,
            "adaptation_episodes_per_stressed_condition": 3,
            "frozen_evaluation_episodes_per_stressed_condition": 1,
            "nominal_reference": "reused_primary_benchmark_episode_3",
            "adaptation_posture": (
                "the primary nominal endpoint is reused; each stressed arm "
                "adapts from the same declared priors on episodes 0-2 and "
                "retains a no-update frozen episode 3"
            ),
            "decision_history_posture": (
                "fresh in-memory decision history at every episode"
            ),
            "mcp_reliability_posture": "false",
            "mcp_fault_dose": {
                "full_trace_scheduled_opportunity_steps": 28,
                "full_trace_total_steps": 288,
            },
            "retained_ledger_design": {
                "stressed_ledgers_per_scenario_task": (
                    len(raw.STRESSORS) * len(raw.EXPECTED_SEEDS)
                ),
                "stressed_decisions_per_scenario_task": (
                    len(raw.STRESSORS) * len(raw.EXPECTED_SEEDS) * 288
                ),
                "reused_primary_nominal_ledgers_per_scenario_task": len(
                    raw.EXPECTED_SEEDS
                ),
                "newly_executed_nominal_episodes": 0,
                "canonical_stressed_ledger_root": f"decision_ledger_h3/{tag}",
                "canonical_nominal_ledger_root": (
                    f"decision_ledger_per_seed/{tag}"
                ),
            },
        },
        "results": {"baseline": scenario_results},
    }
    tost_by_stressor = {
        stressor: raw._equivalence_tost(
            [
                scenario_results[stressor][str(seed)]["agribrain"]["ari"]
                - scenario_results["baseline_by_seed"][str(seed)][
                    "agribrain"
                ]["ari"]
                for seed in raw.EXPECTED_SEEDS
            ],
            raw.STRESS_THRESHOLDS["ari_abs_delta_max"],
        )
        for stressor in raw.STRESSORS
    }
    equivalent_count = sum(
        bool(result["equivalent_alpha_0p05"])
        for result in tost_by_stressor.values()
    )
    h3 = {
        "source_commit": commit,
        "run_tag": tag,
        "test": "paired one-sample TOST on seed-level ARI differences",
        "alpha": 0.05,
        "equivalence_margin": raw.STRESS_THRESHOLDS["ari_abs_delta_max"],
        "confirmatory_method": "agribrain",
        "expected_scenarios": ["baseline"],
        "expected_stressors": list(raw.STRESSORS),
        "expected_n_cells": len(raw.STRESSORS),
        "adaptation_episodes_per_stressed_condition": 3,
        "frozen_evaluation_episodes_per_stressed_condition": 1,
        "nominal_reference": "reused_primary_benchmark_episode_3",
        "episode_accounting": raw.build_h3_episode_accounting(
            n_seeds=len(raw.EXPECTED_SEEDS), n_scenarios=1,
            n_stressors=len(raw.STRESSORS), episodes_per_condition=4,
            nominal_reference_reused=True,
        ),
        "n_cells": len(raw.STRESSORS),
        "n_cells_equivalent": equivalent_count,
        "n_cells_with_verified_exposure": len(raw.STRESSORS),
        "retained_stressed_decision_ledger_count": (
            len(raw.STRESSORS) * len(raw.EXPECTED_SEEDS)
        ),
        "reused_nominal_decision_ledger_references": len(raw.EXPECTED_SEEDS),
        "newly_executed_nominal_episodes": 0,
        "supported_all_cells": equivalent_count == len(raw.STRESSORS),
        "cells": [
            {
                "Scenario": "baseline",
                "Stressor": stressor,
                "Method": "agribrain",
                "n_seeds": len(raw.EXPECTED_SEEDS),
                "Confirmatory_H3": True,
                "inferential_status": "confirmatory_h3",
                "treatment_exposure_verified": True,
                "Pass": bool(tost_by_stressor[stressor][
                    "equivalent_alpha_0p05"
                ]),
                "Pass_Equivalence": bool(tost_by_stressor[stressor][
                    "equivalent_alpha_0p05"
                ]),
                "H3_Pass": bool(tost_by_stressor[stressor][
                    "equivalent_alpha_0p05"
                ]),
                "retained_stressed_decision_ledger_count": len(
                    raw.EXPECTED_SEEDS
                ),
                "retained_stressed_decision_count": (
                    len(raw.EXPECTED_SEEDS) * 288
                ),
                "retained_stressed_decision_ledger_set_sha256": (
                    raw._h3_ledger_set_binding(
                        scenario_results[stressor]
                    )["sha256"]
                ),
                "reused_nominal_decision_ledger_count": len(
                    raw.EXPECTED_SEEDS
                ),
                "reused_nominal_decision_count": (
                    len(raw.EXPECTED_SEEDS) * 288
                ),
                "reused_nominal_decision_ledger_set_sha256": (
                    raw._h3_ledger_set_binding(
                        scenario_results["baseline_by_seed"]
                    )["sha256"]
                ),
                **{
                    f"ari_tost_{key}": value
                    for key, value in tost_by_stressor[stressor].items()
                },
            }
            for stressor in raw.STRESSORS
        ],
    }
    (scenario / "stress_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (scenario / "stress_h3_test.json").write_text(json.dumps(h3), encoding="utf-8")
    (scenario / "stress_degradation.csv").write_text("x\n1\n", encoding="utf-8")
    (scenario / "stress_passfail.csv").write_text("x\n1\n", encoding="utf-8")
    return seeds, stress


def _write_h3_inventory_fixture(root: Path) -> Path:
    root.mkdir()
    for scenario in raw.SCENARIOS:
        scenario_root = root / scenario
        scenario_root.mkdir()
        (scenario_root / "complete_episode_evidence_manifest.json").write_text(
            "{}\n", encoding="utf-8",
        )
        receipt = (
            scenario_root / "runtime_receipts"
            / "job_12345__restart_0.json"
        )
        receipt.parent.mkdir()
        receipt.write_text("{}\n", encoding="utf-8")
        for stressor in raw.STRESSORS:
            for seed in raw.EXPECTED_SEEDS:
                seed_root = scenario_root / stressor / f"seed_{seed}"
                seed_root.mkdir(parents=True)
                arm_name = f"agribrain__{scenario}"
                (seed_root / f"{arm_name}.jsonl").write_text(
                    "{}\n", encoding="utf-8",
                )
                for episode_index in range(3):
                    path = (
                        seed_root / "adaptation_episode_ledgers" / arm_name
                        / f"episode_{episode_index}.jsonl.gz"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"fixture adaptation ledger\n")
                for episode_index in range(4):
                    path = (
                        seed_root / "complete_episode_evidence" / arm_name
                        / f"episode_{episode_index}.json.gz"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"fixture episode archive\n")
    return root


def _one_arm_h3_inventory(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(raw, "SCENARIOS", ("baseline",))
    monkeypatch.setattr(raw, "STRESSORS", ("sensor_noise",))
    monkeypatch.setattr(raw, "EXPECTED_SEEDS", (42,))
    return _write_h3_inventory_fixture(tmp_path / "decision_ledger_h3")


def test_h3_inventory_gate_accepts_exact_full_evidence_tree(
    tmp_path, monkeypatch,
):
    root = _one_arm_h3_inventory(tmp_path, monkeypatch)
    second_receipt = (
        root / "baseline" / "runtime_receipts"
        / "job_12345__restart_1.json"
    )
    second_receipt.write_text("{}\n", encoding="utf-8")

    raw._validate_h3_ledger_inventory_shape(root)


def test_h3_inventory_gate_rejects_unknown_scenario_entry(
    tmp_path, monkeypatch,
):
    root = _one_arm_h3_inventory(tmp_path, monkeypatch)
    (root / "baseline" / "unexpected").mkdir()

    with pytest.raises(RuntimeError, match="unexpected"):
        raw._validate_h3_ledger_inventory_shape(root)


def test_h3_inventory_gate_rejects_final_ledger_only_seed(
    tmp_path, monkeypatch,
):
    root = _one_arm_h3_inventory(tmp_path, monkeypatch)
    seed_root = root / "baseline" / "sensor_noise" / "seed_42"
    for directory in (
        seed_root / "adaptation_episode_ledgers",
        seed_root / "complete_episode_evidence",
    ):
        for path in sorted(directory.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            else:
                path.rmdir()
        directory.rmdir()

    with pytest.raises(RuntimeError, match="missing"):
        raw._validate_h3_ledger_inventory_shape(root)


def test_h3_inventory_gate_rejects_missing_episode_archive(
    tmp_path, monkeypatch,
):
    root = _one_arm_h3_inventory(tmp_path, monkeypatch)
    missing = (
        root / "baseline" / "sensor_noise" / "seed_42"
        / "complete_episode_evidence" / "agribrain__baseline"
        / "episode_3.json.gz"
    )
    missing.unlink()

    with pytest.raises(RuntimeError, match="missing"):
        raw._validate_h3_ledger_inventory_shape(root)


def test_h3_inventory_gate_rejects_missing_adaptation_ledger(
    tmp_path, monkeypatch,
):
    root = _one_arm_h3_inventory(tmp_path, monkeypatch)
    missing = (
        root / "baseline" / "sensor_noise" / "seed_42"
        / "adaptation_episode_ledgers" / "agribrain__baseline"
        / "episode_2.jsonl.gz"
    )
    missing.unlink()

    with pytest.raises(RuntimeError, match="missing"):
        raw._validate_h3_ledger_inventory_shape(root)


def test_h3_inventory_gate_rejects_extra_nested_evidence(
    tmp_path, monkeypatch,
):
    root = _one_arm_h3_inventory(tmp_path, monkeypatch)
    extra = (
        root / "baseline" / "sensor_noise" / "seed_42"
        / "adaptation_episode_ledgers" / "agribrain__baseline"
        / "episode_3.jsonl.gz"
    )
    extra.write_bytes(b"unexpected adaptation ledger\n")

    with pytest.raises(RuntimeError, match="unexpected"):
        raw._validate_h3_ledger_inventory_shape(root)


def test_h3_inventory_gate_rejects_wrong_evidence_type(
    tmp_path, monkeypatch,
):
    root = _one_arm_h3_inventory(tmp_path, monkeypatch)
    manifest = root / "baseline" / "complete_episode_evidence_manifest.json"
    manifest.unlink()
    manifest.mkdir()

    with pytest.raises(RuntimeError, match="not a regular file"):
        raw._validate_h3_ledger_inventory_shape(root)


def test_h3_inventory_gate_rejects_empty_runtime_receipts(
    tmp_path, monkeypatch,
):
    root = _one_arm_h3_inventory(tmp_path, monkeypatch)
    receipt = (
        root / "baseline" / "runtime_receipts"
        / "job_12345__restart_0.json"
    )
    receipt.unlink()

    with pytest.raises(RuntimeError, match="receipt inventory is empty"):
        raw._validate_h3_ledger_inventory_shape(root)


def test_h3_inventory_gate_rejects_runtime_receipt_directory(
    tmp_path, monkeypatch,
):
    root = _one_arm_h3_inventory(tmp_path, monkeypatch)
    receipt = (
        root / "baseline" / "runtime_receipts"
        / "job_12345__restart_0.json"
    )
    receipt.unlink()
    receipt.mkdir()

    with pytest.raises(RuntimeError, match="not a regular file"):
        raw._validate_h3_ledger_inventory_shape(root)


def test_h3_inventory_gate_rejects_malformed_runtime_receipt_name(
    tmp_path, monkeypatch,
):
    root = _one_arm_h3_inventory(tmp_path, monkeypatch)
    malformed = root / "baseline" / "runtime_receipts" / "receipt.json"
    malformed.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unexpected name"):
        raw._validate_h3_ledger_inventory_shape(root)


def test_h3_inventory_gate_rejects_symlinked_stressor_directory(
    tmp_path, monkeypatch,
):
    root = _one_arm_h3_inventory(tmp_path, monkeypatch)
    stressor = root / "baseline" / "sensor_noise"
    real_stressor = tmp_path / "real_sensor_noise"
    stressor.rename(real_stressor)
    try:
        stressor.symlink_to(real_stressor, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are not available to this test user")

    with pytest.raises(RuntimeError, match="not a real directory"):
        raw._validate_h3_ledger_inventory_shape(root)


def test_h3_inventory_gate_rejects_symlinked_final_ledger(
    tmp_path, monkeypatch,
):
    root = _one_arm_h3_inventory(tmp_path, monkeypatch)
    ledger = (
        root / "baseline" / "sensor_noise" / "seed_42"
        / "agribrain__baseline.jsonl"
    )
    target = tmp_path / "outside.jsonl"
    target.write_text("{}\n", encoding="utf-8")
    ledger.unlink()
    try:
        ledger.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are not available to this test user")

    with pytest.raises(RuntimeError, match="not a regular file"):
        raw._validate_h3_ledger_inventory_shape(root)


def test_raw_input_gate_accepts_exact_identity_and_panels(tmp_path, monkeypatch):
    commit = "a" * 40
    tag = "abcdef0_20260819_120000"
    monkeypatch.setattr(raw, "EXPECTED_SEEDS", (42, 43))
    monkeypatch.setattr(raw, "SCENARIOS", ["baseline"])
    monkeypatch.setattr(raw, "MODES", ["agribrain"])
    seeds, stress = _write_raw_fixture(tmp_path, commit=commit, tag=tag)
    monkeypatch.setattr(stress_runner, "PRIMARY_SEEDS_DIR", seeds)
    nominal, identity = stress_runner._load_primary_nominal("baseline", 42)
    assert identity == {"source_commit": commit, "run_tag": tag}
    assert nominal["decision_ledger_path"] == (
        f"decision_ledger_per_seed/{tag}/seed_42/agribrain__baseline.jsonl"
    )
    assert nominal["decision_ledger_n_records"] == 288
    assert nominal["message_count"] == 4
    raw.validate_seed_inputs(seeds, source_commit=commit, run_tag=tag)
    raw.validate_stress_inputs(
        stress, seed_root=seeds, source_commit=commit, run_tag=tag,
    )
    consolidated = tmp_path / "decision_ledger_per_seed" / tag
    for seed in raw.EXPECTED_SEEDS:
        source = seeds / f"decision_ledger_{seed}" / "agribrain__baseline.jsonl"
        target = consolidated / f"seed_{seed}" / "agribrain__baseline.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    raw.validate_stress_inputs(
        stress, seed_root=seeds, source_commit=commit, run_tag=tag,
        primary_ledger_root=consolidated,
    )


def test_h3_stressed_projection_retains_message_count(monkeypatch):
    monkeypatch.setenv("RUN_TAG", "abcdef0_20260819_120000")
    monkeypatch.setattr(
        stress_runner,
        "decision_ledger_scope",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        stress_runner,
        "_ledger_file_binding",
        lambda _path, *, canonical_path: {
            "decision_ledger_path": canonical_path,
            "decision_ledger_sha256": "a" * 64,
            "decision_ledger_merkle_root": "b" * 64,
            "decision_ledger_n_records": 288,
        },
    )
    freeze = {
        "learners_frozen": True,
        "learner_phase": "frozen_evaluation",
        "freeze_reason": "retained_episode_3",
        "context_matrix_frozen": True,
        "policy_delta_frozen_by_role": {"retailer": True},
        "reward_shaping_frozen": True,
    }
    fake_episode = {
        "ari": 0.8,
        "waste": 0.1,
        "slca": 0.9,
        "rle": 0.7,
        "carbon": 1.2,
        "equity": 0.85,
        "message_count": 37,
        "constraint_violation_rate": 0.0,
        "mean_decision_latency_ms": 2.0,
        "downstream_violation_rate": 0.0,
        "contained_violation_rate": 1.0,
        "trace_schema_version": raw.TRACE_SCHEMA_VERSION,
        "benchmark_seed": 42,
        "episode_index": 3,
        "environment_stream_id": "environment",
        "policy_stream_id": "policy",
        "stochastic_stream_id": "environment",
        "context_prior_sha256": "c" * 64,
        "policy_theta_initial_sha256": "d" * 64,
        "spoilage_estimator": {
            "kind": "mechanistic_plus_frozen_synthetic_pinn_residual",
            "checkpoint_sha256": "1" * 64,
            "training_dataset_sha256": "2" * 64,
            "training_target_origin": "independent_synthetic_dgp",
            "residual_bound_abs": 0.08,
            "deployment_transform": (
                "clip_quality_to_unit_interval_then_cumulative_minimum"
            ),
            "synthetic_only": True,
            "external_validation": False,
        },
        "latent_spoilage_model": raw.synthetic_dgp_provenance(),
        "latent_environment_sha256": "e" * 64,
        "observed_policy_input_sha256": "f" * 64,
        "demand_observation_sha256": "0" * 64,
        "demand_forecast_method": "holt_linear",
        "supply_forecast_method": "persistence",
        "learning_enabled": False,
        "episode_phase": "frozen_evaluation",
        "dispatch_opportunity_count": 288,
        "dispatch_cadence_hours": 0.25,
        "learner_summary": {},
        "theta_learner_summary": {},
        "reward_shaping_learner_summary": {},
        "learner_freeze_summary": freeze,
        "decision_ledger_path": "ignored.jsonl",
    }
    monkeypatch.setattr(
        stress_runner,
        "run_episode",
        lambda *_args, **_kwargs: dict(fake_episode),
    )
    frames = {
        episode_index: stress_runner.pd.DataFrame()
        for episode_index in range(4)
    }

    result = stress_runner._run_pair_impl(
        frames,
        "baseline",
        42,
        False,
        ["agribrain"],
        "sensor_noise",
    )

    assert result["agribrain"]["message_count"] == 37


def test_raw_input_gate_rejects_incorrect_primary_episode_accounting(
    tmp_path, monkeypatch,
):
    commit = "a" * 40
    tag = "abcdef0_20260819_120000"
    monkeypatch.setattr(raw, "EXPECTED_SEEDS", (42, 43))
    monkeypatch.setattr(raw, "SCENARIOS", ["baseline"])
    monkeypatch.setattr(raw, "MODES", ["agribrain"])
    seeds, _stress = _write_raw_fixture(tmp_path, commit=commit, tag=tag)
    path = seeds / "seed_42.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_meta"]["episode_accounting"][
        "executed_episodes_all_configured_modes"
    ] -= 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="episode accounting"):
        raw.validate_seed_inputs(seeds, source_commit=commit, run_tag=tag)


def test_raw_input_gate_accepts_exact_integral_float_update_counts(
    tmp_path, monkeypatch,
):
    """Preserved HPC JSON may encode an exact counter as ``288.0``."""
    commit = "a" * 40
    tag = "abcdef0_20260819_120000"
    monkeypatch.setattr(raw, "EXPECTED_SEEDS", (42, 43))
    monkeypatch.setattr(raw, "SCENARIOS", ["baseline"])
    monkeypatch.setattr(raw, "MODES", ["agribrain"])
    seeds, _ = _write_raw_fixture(tmp_path, commit=commit, tag=tag)
    for path in seeds.glob("seed_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for scenario in payload["scenarios"].values():
            for cell in scenario.values():
                summary = cell.get("theta_learner_summary")
                if isinstance(summary, dict):
                    summary["updates_per_role"] = {
                        role: float(value)
                        for role, value in summary["updates_per_role"].items()
                    }
        path.write_text(json.dumps(payload), encoding="utf-8")

    raw.validate_seed_inputs(seeds, source_commit=commit, run_tag=tag)


def test_raw_input_gate_accepts_stamped_dormant_context_infrastructure():
    roles = raw.DECISION_OWNER_ROLES
    dormant_context_state = {
        "theta": [[0.0] * 5 for _ in range(3)],
        "slca_amp_coeff": 0.0,
        "learn_proxy_interaction": False,
        "sign_constrained": True,
        "temporal_base": 1.0,
        "temporal_scale": 0.0,
        "reward_baseline": 0.5,
        "n_updates": 0,
    }
    per_role = {}
    role_states = {}
    per_role_hashes = {}
    for role in roles:
        role_state = {
            "theta_delta": [[0.0] * 10 for _ in range(3)],
            "reward_baseline": 0.5,
            "n_updates": 1,
            "learning_rate": 0.003,
            "prior_precision": 1.0,
            "magnitude_cap_fraction": 0.25,
            "sign_constrained": True,
        }
        role_states[role] = role_state
        per_role_hashes[role] = raw._canonical_object_sha256(role_state)
        per_role[role] = {
            "final_theta_delta": role_state["theta_delta"],
            "reward_baseline": role_state["reward_baseline"],
            "n_updates": role_state["n_updates"],
            "learning_rate": role_state["learning_rate"],
            "prior_precision": role_state["prior_precision"],
            "magnitude_cap_fraction": role_state["magnitude_cap_fraction"],
            "sign_constrained": role_state["sign_constrained"],
            "sign_reversal_count": 0,
            "sign_reversal_coordinates": [],
            "worst_sign_reversal": None,
        }
    shaping_state = {
        "slca_bonus_delta": [0.0] * 3,
        "slca_rho_delta": [0.0] * 3,
        "no_slca_offset_delta": [0.0] * 3,
        "reward_baseline": 0.5,
        "n_updates": 1,
        "magnitude_cap_fraction": 0.25,
        "sign_constrained": True,
    }
    cell = {
        "message_count": 4,
        "learner_summary": {
            "mode": "no_context",
            "learner_state_schema_version": 2.0,
            "state_sha256": raw._canonical_object_sha256(
                dormant_context_state
            ),
            "learning_enabled": False,
            "final_theta": dormant_context_state["theta"],
            "final_slca_amp": dormant_context_state["slca_amp_coeff"],
            "learn_proxy_interaction": dormant_context_state[
                "learn_proxy_interaction"
            ],
            "sign_constrained": dormant_context_state[
                "sign_constrained"
            ],
            "temporal_base": dormant_context_state["temporal_base"],
            "temporal_scale": dormant_context_state["temporal_scale"],
            "reward_baseline": dormant_context_state["reward_baseline"],
            "n_updates": dormant_context_state["n_updates"],
        },
        "theta_learner_summary": {
            "mode": "no_context",
            "learner_state_schema_version": 2,
            "combined_state_sha256": raw._canonical_object_sha256(
                role_states
            ),
            "decision_owner_roles": list(roles),
            "updates_per_role": {role: 1 for role in roles},
            "per_role_state_sha256": per_role_hashes,
            "per_role": per_role,
            "n_updates": len(roles),
            "sign_constrained": True,
            "sign_reversal_count": 0,
        },
        "reward_shaping_learner_summary": {
            "mode": "no_context",
            "learner_state_schema_version": 2,
            "state_sha256": raw._canonical_object_sha256(shaping_state),
            **shaping_state,
            "sign_reversal_count": 0,
            "sign_reversal_coordinates": [],
            "worst_sign_reversal": None,
        },
    }
    raw._validate_learner_provenance(
        cell, mode="no_context", where="fixture/no_context",
    )


def test_raw_input_gate_binds_all_learners_to_unconstrained_sign_arm():
    mode = "agribrain_sign_unconstrained"
    roles = raw.DECISION_OWNER_ROLES
    context_state = {
        "theta": [[0.0] * 5 for _ in range(3)],
        "slca_amp_coeff": 0.0,
        "learn_proxy_interaction": False,
        "sign_constrained": False,
        "temporal_base": 1.0,
        "temporal_scale": 0.0,
        "reward_baseline": 0.5,
        "n_updates": 4,
    }
    per_role = {}
    role_states = {}
    per_role_hashes = {}
    for role in roles:
        role_state = {
            "theta_delta": [[0.0] * 10 for _ in range(3)],
            "reward_baseline": 0.5,
            "n_updates": 1,
            "learning_rate": 0.003,
            "prior_precision": 1.0,
            "magnitude_cap_fraction": 0.25,
            "sign_constrained": False,
        }
        role_states[role] = role_state
        per_role_hashes[role] = raw._canonical_object_sha256(role_state)
        per_role[role] = {
            "final_theta_delta": role_state["theta_delta"],
            "reward_baseline": role_state["reward_baseline"],
            "n_updates": role_state["n_updates"],
            "learning_rate": role_state["learning_rate"],
            "prior_precision": role_state["prior_precision"],
            "magnitude_cap_fraction": role_state["magnitude_cap_fraction"],
            "sign_constrained": role_state["sign_constrained"],
            "sign_reversal_count": 0,
            "sign_reversal_coordinates": [],
            "worst_sign_reversal": None,
        }
    shaping_state = {
        "slca_bonus_delta": [0.0] * 3,
        "slca_rho_delta": [0.0] * 3,
        "no_slca_offset_delta": [0.0] * 3,
        "reward_baseline": 0.5,
        "n_updates": 4,
        "magnitude_cap_fraction": 0.25,
        "sign_constrained": False,
    }
    cell = {
        "message_count": 4,
        "learner_summary": {
            "mode": mode,
            "learner_state_schema_version": 2,
            "state_sha256": raw._canonical_object_sha256(context_state),
            "final_theta": context_state["theta"],
            "final_slca_amp": context_state["slca_amp_coeff"],
            "learn_proxy_interaction": context_state[
                "learn_proxy_interaction"
            ],
            "temporal_base": context_state["temporal_base"],
            "temporal_scale": context_state["temporal_scale"],
            "reward_baseline": context_state["reward_baseline"],
            "n_updates": context_state["n_updates"],
            "sign_constrained": context_state["sign_constrained"],
            "sign_preserved": True,
            "sign_reversal_count": 0,
            "sign_reversal_coordinates": [],
            "worst_sign_reversal": None,
            "compliance_sign_reversal_count": 0,
            "worst_compliance_sign_reversal": None,
        },
        "theta_learner_summary": {
            "mode": mode,
            "learner_state_schema_version": 2,
            "combined_state_sha256": raw._canonical_object_sha256(
                role_states
            ),
            "decision_owner_roles": list(roles),
            "updates_per_role": {role: 1 for role in roles},
            "per_role_state_sha256": per_role_hashes,
            "per_role": per_role,
            "n_updates": len(roles),
            "sign_constrained": False,
            "sign_reversal_count": 0,
        },
        "reward_shaping_learner_summary": {
            "mode": mode,
            "learner_state_schema_version": 2,
            "state_sha256": raw._canonical_object_sha256(shaping_state),
            **shaping_state,
            "sign_reversal_count": 0,
            "sign_reversal_coordinates": [],
            "worst_sign_reversal": None,
        },
    }
    raw._validate_learner_provenance(
        cell, mode=mode, where="fixture/sign-unconstrained",
    )

    cell["theta_learner_summary"]["per_role"][roles[0]][
        "sign_constrained"
    ] = True
    with pytest.raises(RuntimeError, match="wrong .* sign projection"):
        raw._validate_learner_provenance(
            cell, mode=mode, where="fixture/sign-unconstrained",
        )


def test_raw_input_gate_rejects_active_undeclared_context_learner():
    cell = {
        "message_count": 4,
        "learner_summary": {
            "mode": "no_context",
            "learner_state_schema_version": 2,
            "state_sha256": "a" * 64,
            "learning_enabled": True,
            "n_updates": 1,
        },
        "theta_learner_summary": None,
        "reward_shaping_learner_summary": None,
    }
    with pytest.raises(RuntimeError, match="active context learner"):
        raw._validate_learner_provenance(
            cell, mode="no_context", where="fixture/no_context",
        )


@pytest.mark.parametrize("bad_count", [True, -1, 1.5, "4", 289])
def test_raw_input_gate_rejects_non_integral_or_invalid_update_counts(
    tmp_path, monkeypatch, bad_count,
):
    commit = "a" * 40
    tag = "abcdef0_20260819_120000"
    monkeypatch.setattr(raw, "EXPECTED_SEEDS", (42, 43))
    monkeypatch.setattr(raw, "SCENARIOS", ["baseline"])
    monkeypatch.setattr(raw, "MODES", ["agribrain"])
    seeds, _ = _write_raw_fixture(tmp_path, commit=commit, tag=tag)
    path = seeds / "seed_42.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["scenarios"]["baseline"]["agribrain"][
        "theta_learner_summary"
    ]["updates_per_role"][raw.DECISION_OWNER_ROLES[0]] = bad_count
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid .* update count"):
        raw.validate_seed_inputs(seeds, source_commit=commit, run_tag=tag)


def test_raw_input_gate_rejects_tampered_h3_statistic(tmp_path, monkeypatch):
    commit = "a" * 40
    tag = "abcdef0_20260819_120000"
    monkeypatch.setattr(raw, "EXPECTED_SEEDS", (42, 43))
    monkeypatch.setattr(raw, "SCENARIOS", ["baseline"])
    monkeypatch.setattr(raw, "MODES", ["agribrain"])
    seeds, stress = _write_raw_fixture(tmp_path, commit=commit, tag=tag)
    path = stress / "baseline" / "stress_h3_test.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cells"][0]["ari_tost_mean"] = 0.005
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="ari_tost_mean"):
        raw.validate_stress_inputs(
            stress, seed_root=seeds, source_commit=commit, run_tag=tag,
        )


def test_raw_input_gate_rejects_duplicate_h3_stressor(tmp_path, monkeypatch):
    commit = "a" * 40
    tag = "abcdef0_20260819_120000"
    monkeypatch.setattr(raw, "EXPECTED_SEEDS", (42, 43))
    monkeypatch.setattr(raw, "SCENARIOS", ["baseline"])
    monkeypatch.setattr(raw, "MODES", ["agribrain"])
    seeds, stress = _write_raw_fixture(tmp_path, commit=commit, tag=tag)
    path = stress / "baseline" / "stress_h3_test.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cells"][-1]["Stressor"] = payload["cells"][0]["Stressor"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate stressor"):
        raw.validate_stress_inputs(
            stress, seed_root=seeds, source_commit=commit, run_tag=tag,
        )


def test_raw_input_gate_rejects_missing_observed_fault_exposure(
    tmp_path, monkeypatch,
):
    commit = "a" * 40
    tag = "abcdef0_20260819_120000"
    monkeypatch.setattr(raw, "EXPECTED_SEEDS", (42, 43))
    monkeypatch.setattr(raw, "SCENARIOS", ["baseline"])
    monkeypatch.setattr(raw, "MODES", ["agribrain"])
    seeds, stress = _write_raw_fixture(tmp_path, commit=commit, tag=tag)
    path = stress / "baseline" / "stress_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    cell = payload["results"]["baseline"]["mcp_fault_injection"]["42"]["agribrain"]
    cell["fault_injection_trigger_steps"] = 0
    cell["fault_injected_tool_result_count"] = 0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        RuntimeError, match="fault_injection_trigger_steps ledger reconstruction",
    ):
        raw.validate_stress_inputs(
            stress, seed_root=seeds, source_commit=commit, run_tag=tag,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        (
            "decision_latency_ms", 9.0,
            "decision_latency_ms ledger reconstruction",
        ),
        (
            "protocol_interaction_count", 11,
            "protocol_interaction_count differs from decision-ledger",
        ),
        (
            "message_count", 5,
            "message_count differs from decision-ledger",
        ),
    ],
)
def test_raw_input_gate_binds_h3_activity_scalars_to_decision_records(
    tmp_path, monkeypatch, field, replacement, message,
):
    commit = "a" * 40
    tag = "abcdef0_20260819_120000"
    monkeypatch.setattr(raw, "EXPECTED_SEEDS", (42, 43))
    monkeypatch.setattr(raw, "SCENARIOS", ["baseline"])
    monkeypatch.setattr(raw, "MODES", ["agribrain"])
    seeds, stress = _write_raw_fixture(tmp_path, commit=commit, tag=tag)
    path = stress / "baseline" / "stress_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results"]["baseline"]["sensor_noise"]["42"][
        "agribrain"
    ][field] = replacement
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        raw.validate_stress_inputs(
            stress, seed_root=seeds, source_commit=commit, run_tag=tag,
        )


def test_raw_input_gate_rejects_missing_h3_message_count(tmp_path, monkeypatch):
    commit = "a" * 40
    tag = "abcdef0_20260819_120000"
    monkeypatch.setattr(raw, "EXPECTED_SEEDS", (42, 43))
    monkeypatch.setattr(raw, "SCENARIOS", ["baseline"])
    monkeypatch.setattr(raw, "MODES", ["agribrain"])
    seeds, stress = _write_raw_fixture(tmp_path, commit=commit, tag=tag)
    path = stress / "baseline" / "stress_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["results"]["baseline"]["sensor_noise"]["42"][
        "agribrain"
    ]["message_count"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid peer-message exposure"):
        raw.validate_stress_inputs(
            stress, seed_root=seeds, source_commit=commit, run_tag=tag,
        )


def test_raw_input_gate_rejects_truncated_stress_trace(tmp_path, monkeypatch):
    commit = "a" * 40
    tag = "abcdef0_20260819_120000"
    monkeypatch.setattr(raw, "EXPECTED_SEEDS", (42, 43))
    monkeypatch.setattr(raw, "SCENARIOS", ["baseline"])
    monkeypatch.setattr(raw, "MODES", ["agribrain"])
    seeds, stress = _write_raw_fixture(tmp_path, commit=commit, tag=tag)
    path = stress / "baseline" / "stress_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["meta"]["max_rows"] = 8
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="complete 288-step"):
        raw.validate_stress_inputs(
            stress, seed_root=seeds, source_commit=commit, run_tag=tag,
        )


def test_raw_input_gate_binds_reused_nominal_to_primary_envelope(
    tmp_path, monkeypatch,
):
    commit = "a" * 40
    tag = "abcdef0_20260819_120000"
    monkeypatch.setattr(raw, "EXPECTED_SEEDS", (42, 43))
    monkeypatch.setattr(raw, "SCENARIOS", ["baseline"])
    monkeypatch.setattr(raw, "MODES", ["agribrain"])
    seeds, stress = _write_raw_fixture(tmp_path, commit=commit, tag=tag)
    path = stress / "baseline" / "stress_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results"]["baseline"]["baseline_by_seed"]["42"][
        "agribrain"
    ]["ari"] = 0.75
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="ari primary binding"):
        raw.validate_stress_inputs(
            stress, seed_root=seeds, source_commit=commit, run_tag=tag,
        )


def test_raw_input_gate_rejects_self_consistent_but_wrong_stress_dose(
    tmp_path, monkeypatch,
):
    commit = "a" * 40
    tag = "abcdef0_20260819_120000"
    monkeypatch.setattr(raw, "EXPECTED_SEEDS", (42, 43))
    monkeypatch.setattr(raw, "SCENARIOS", ["baseline"])
    monkeypatch.setattr(raw, "MODES", ["agribrain"])
    seeds, stress = _write_raw_fixture(tmp_path, commit=commit, tag=tag)
    path = stress / "baseline" / "stress_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    treatment = payload["results"]["baseline"]["sensor_noise"]["42"][
        "agribrain"
    ]["observation_treatment"]
    treatment["temp_noise_sha256"] = "f" * 64
    treatment_without_hash = dict(treatment)
    treatment_without_hash.pop("treatment_sha256")
    treatment["treatment_sha256"] = raw._canonical_object_sha256(
        treatment_without_hash
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="locked seed-indexed H3 dose"):
        raw.validate_stress_inputs(
            stress, seed_root=seeds, source_commit=commit, run_tag=tag,
        )


def test_raw_input_gate_rejects_changed_paired_demand_stream(
    tmp_path, monkeypatch,
):
    commit = "a" * 40
    tag = "abcdef0_20260819_120000"
    monkeypatch.setattr(raw, "EXPECTED_SEEDS", (42, 43))
    monkeypatch.setattr(raw, "SCENARIOS", ["baseline"])
    monkeypatch.setattr(raw, "MODES", ["agribrain"])
    seeds, stress = _write_raw_fixture(tmp_path, commit=commit, tag=tag)
    path = stress / "baseline" / "stress_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results"]["baseline"]["sensor_noise"]["42"]["agribrain"][
        "demand_observation_sha256"
    ] = "f" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="demand_observation_sha256"):
        raw.validate_stress_inputs(
            stress, seed_root=seeds, source_commit=commit, run_tag=tag,
        )


def test_h3_gate_rejects_self_hashed_observation_not_generated_by_dose(
    tmp_path, monkeypatch,
):
    """Even a rehashed ledger/summary must reconstruct from dose primitives."""
    commit = "a" * 40
    tag = "abcdef0_20260819_120000"
    monkeypatch.setattr(raw, "EXPECTED_SEEDS", (42, 43))
    monkeypatch.setattr(raw, "SCENARIOS", ["baseline"])
    monkeypatch.setattr(raw, "MODES", ["agribrain"])
    seeds, stress = _write_raw_fixture(tmp_path, commit=commit, tag=tag)
    ledger_path = (
        tmp_path / "decision_ledger_h3" / "baseline" / "sensor_noise"
        / "seed_42" / "agribrain__baseline.jsonl"
    )
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    rows = [json.loads(line) for line in lines[1:]]
    rows[10]["temp_policy_observed"] += 0.25
    leaves = []
    for row in rows:
        row.pop("_leaf", None)
        leaf = hashlib.sha256(json.dumps(
            row, sort_keys=True, separators=(",", ":"), default=str,
        ).encode()).hexdigest()
        row["_leaf"] = leaf
        leaves.append(leaf)
    observed_payload = {
        "hours": [float(row["hour"]) for row in rows],
        "temp_policy_observed": [float(row["temp_policy_observed"]) for row in rows],
        "rh_policy_observed": [float(row["rh_policy_observed"]) for row in rows],
        "rho_policy_observed": [float(row["rho_policy_observed"]) for row in rows],
        "inventory_policy_observed": [float(row["inventory_policy_observed"]) for row in rows],
        "demand_forecast_policy_observed": [
            float(row["demand_forecast_policy_observed"]) for row in rows
        ],
        "supply_forecast_policy_observed": [
            float(row["supply_forecast_policy_observed"]) for row in rows
        ],
    }
    observed_hash = raw._canonical_object_sha256(observed_payload)
    header["metadata"]["observed_policy_input_sha256"] = observed_hash
    header["merkle_root"] = raw._ledger_merkle_root(leaves)
    with ledger_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(header, sort_keys=True) + "\n")
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary_path = stress / "baseline" / "stress_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cell = summary["results"]["baseline"]["sensor_noise"]["42"]["agribrain"]
    cell["observed_policy_input_sha256"] = observed_hash
    cell["decision_ledger_merkle_root"] = header["merkle_root"]
    cell["decision_ledger_sha256"] = raw._sha256_file(ledger_path)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    # The shared policy-equation validator may reject this partially coherent
    # forgery before the later H3-dose replay. A separate source-replay test
    # mutates and rebinds the full environmental surface to exercise that later
    # boundary directly.
    with pytest.raises(
        RuntimeError,
        match=(
            "policy observation does not reconstruct from H3 dose"
            "|violates the locked policy equation"
        ),
    ):
        raw.validate_stress_inputs(
            stress, seed_root=seeds, source_commit=commit, run_tag=tag,
        )


def test_raw_input_gate_rejects_incorrect_h3_episode_accounting(
    tmp_path, monkeypatch,
):
    commit = "a" * 40
    tag = "abcdef0_20260819_120000"
    monkeypatch.setattr(raw, "EXPECTED_SEEDS", (42, 43))
    monkeypatch.setattr(raw, "SCENARIOS", ["baseline"])
    monkeypatch.setattr(raw, "MODES", ["agribrain"])
    seeds, stress = _write_raw_fixture(tmp_path, commit=commit, tag=tag)
    path = stress / "baseline" / "stress_h3_test.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["episode_accounting"]["incremental_executed_episodes"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="scenario-task H3 accounting"):
        raw.validate_stress_inputs(
            stress, seed_root=seeds, source_commit=commit, run_tag=tag,
        )


def test_stress_aggregator_rejects_non_agribrain_or_extra_rows(tmp_path):
    path = tmp_path / "stress_passfail.csv"
    exact = stress_aggregate.pd.DataFrame([
        {
            "Scenario": "baseline", "Stressor": stressor,
            "Method": "agribrain", "n_seeds": 20,
        }
        for stressor in stress_aggregate.STRESSORS
    ])
    stress_aggregate._validate_exact_h3_frame(
        exact, scenario="baseline", where=path,
    )
    extra = stress_aggregate.pd.concat([
        exact,
        stress_aggregate.pd.DataFrame([{
            "Scenario": "baseline", "Stressor": "sensor_noise",
            "Method": "hybrid_rl", "n_seeds": 20,
        }]),
    ], ignore_index=True)
    with pytest.raises(RuntimeError, match="AGRI-BRAIN-only"):
        stress_aggregate._validate_exact_h3_frame(
            extra, scenario="baseline", where=path,
        )


def test_stress_aggregator_recomputes_seed_ledger_set_hash(monkeypatch):
    monkeypatch.setattr(stress_aggregate, "CANONICAL_SEEDS", (42, 43))
    panel = {
        str(seed): {
            "agribrain": {
                "decision_ledger_path": f"ledger/{seed}.jsonl",
                "decision_ledger_sha256": f"{seed:064x}",
                "decision_ledger_merkle_root": f"{seed + 1:064x}",
                "decision_ledger_n_records": 288,
            }
        }
        for seed in (42, 43)
    }
    runner_panel = {
        seed: panel[str(seed)]["agribrain"] for seed in (42, 43)
    }
    assert stress_aggregate._ledger_set_sha256(panel) == (
        stress_runner._ledger_set_binding(runner_panel, (42, 43))["sha256"]
    )


def test_confirmatory_h3_rejects_partial_seed_panel():
    assert stress_runner._confirmatory_seed_panel(20) == list(
        stress_runner.CANONICAL_SEEDS
    )
    with pytest.raises(ValueError, match="requires STRESS_N_SEEDS=20"):
        stress_runner._confirmatory_seed_panel(19)


def test_confirmatory_h3_entrypoint_requires_explicit_run_scoped_paths(
    monkeypatch,
):
    for name in (
        "STRESS_OUTPUT_DIR", "STRESS_LEDGER_ROOT", "STRESS_PRIMARY_SEEDS_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="explicit run-scoped paths"):
        stress_runner.main()


def test_raw_input_gate_rejects_mixed_run(tmp_path, monkeypatch):
    commit = "a" * 40
    tag = "abcdef0_20260819_120000"
    monkeypatch.setattr(raw, "EXPECTED_SEEDS", (42, 43))
    monkeypatch.setattr(raw, "SCENARIOS", ["baseline"])
    monkeypatch.setattr(raw, "MODES", ["agribrain"])
    seeds, _ = _write_raw_fixture(tmp_path, commit=commit, tag="wrong_run")
    with pytest.raises(RuntimeError, match="run_tag"):
        raw.validate_seed_inputs(seeds, source_commit=commit, run_tag=tag)


def test_raw_input_gate_rejects_context_execution_failure(tmp_path, monkeypatch):
    commit = "a" * 40
    tag = "abcdef0_20260819_120000"
    monkeypatch.setattr(raw, "EXPECTED_SEEDS", (42, 43))
    monkeypatch.setattr(raw, "SCENARIOS", ["baseline"])
    monkeypatch.setattr(raw, "MODES", ["agribrain"])
    seeds, _ = _write_raw_fixture(tmp_path, commit=commit, tag=tag)
    path = seeds / "seed_42.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    cell = payload["scenarios"]["baseline"]["agribrain"]
    cell["dispatcher_tool_failure_count"] = 1
    cell["context_execution_error_count"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="context execution failures"):
        raw.validate_seed_inputs(seeds, source_commit=commit, run_tag=tag)


def test_final_artifact_gate_rechecks_staged_run_identity(tmp_path, monkeypatch):
    commit = "a" * 40
    tag = "abcdef0_20260819_120000"
    monkeypatch.setattr(publication, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(publication, "EXPECTED_SEEDS", (42,))
    (tmp_path / "benchmark_seeds").mkdir()
    for name in ("benchmark_summary.json", "benchmark_significance.json"):
        (tmp_path / name).write_text(json.dumps({
            "_meta": {"source_commit": commit, "run_tag": tag},
        }), encoding="utf-8")
    (tmp_path / "channel_attribution_aggregate.json").write_text(
        "{}", encoding="utf-8",
    )
    (tmp_path / "stress_passfail.csv").write_text(
        "Scenario,Stressor,Method\n", encoding="utf-8",
    )
    seed_path = tmp_path / "benchmark_seeds" / "seed_42.json"
    seed_path.write_text(json.dumps({
        "_meta": {"source_commit": commit, "run_tag": tag}, "seed": 42,
        "traces": {"baseline": {"agribrain": {
            field: [0.0] for field in (
                "temp_outcome_environmental_trace",
                "rh_outcome_environmental_trace",
                "rho_outcome_environmental_trace",
                "rho_policy_observed_trace", "prob_trace", "ari_trace",
                "inventory_outcome_environmental_trace",
                "demand_outcome_environmental_trace", "waste_trace",
                "action_trace", "slca_component_trace", "equity_trace",
                "reward_trace", "demand_trace",
            )
        }}},
    }), encoding="utf-8")
    seed_record = {
        "file": "benchmark_seeds/seed_42.json",
        "bytes": seed_path.stat().st_size,
        "sha256": hashlib.sha256(seed_path.read_bytes()).hexdigest(),
    }
    aggregate_records = []
    for name in publication.EXPECTED_FIGURE_AGGREGATE_INPUTS:
        aggregate_path = tmp_path / name
        aggregate_records.append({
            "file": name,
            "bytes": aggregate_path.stat().st_size,
            "sha256": hashlib.sha256(aggregate_path.read_bytes()).hexdigest(),
        })
    (tmp_path / "artifact_manifest.json").write_text(json.dumps({
        "git_commit": commit, "artifact_run_tag": tag,
        "artifacts": [seed_record, *aggregate_records],
    }), encoding="utf-8")
    (tmp_path / "stress_summary.json").write_text(json.dumps({
        "meta": {"source_commit": commit, "run_tag": tag},
    }), encoding="utf-8")
    (tmp_path / "stress_h3_test.json").write_text(json.dumps({
        "source_commit": commit, "run_tag": tag,
    }), encoding="utf-8")
    (tmp_path / "channel_saturation_analysis.json").write_text(json.dumps({
        "_meta": {"git_commit": commit, "benchmark_run": tag},
    }), encoding="utf-8")
    (tmp_path / "figure_provenance.json").write_text(json.dumps({
        "schema_version": 3,
        "source_commit": commit,
        "source_commit_semantics": "raw_input_simulation_commit",
        "simulation_source_commit": commit,
        "renderer_code_commit": commit,
        "dual_provenance": False,
        "run_tag": tag,
        "seed_root": "/cluster/repo/mvp/simulation/results/benchmark_seeds",
        "seed_panel": [42],
        "n_seed_envelopes_loaded": 1,
        "seed_input_artifacts": [{**seed_record, "seed": 42}],
        "aggregate_input_artifacts": aggregate_records,
        "render_input_isolated_snapshot": True,
        "illustrative_seed": 42,
        "panels": {
            name: ({
                panel: {
                    "fields": [], "aggregation": "test",
                    "n_seeds": (
                        20 if (
                            (name == "heatwave" and panel == "d")
                            or (name == "overproduction" and panel == "d")
                            or (name == "cyber_outage" and panel in {"b", "c", "d"})
                        ) else 1
                    ),
                }
                for panel in ("a", "b", "c", "d")
            } if name != "cross_scenario_and_secondary" else {
                "fields": list(publication.EXPECTED_FIGURE_AGGREGATE_INPUTS),
                "aggregation": "test",
                "n_seeds": 20,
            })
            for name in (
                "heatwave", "overproduction", "cyber_outage",
                "adaptive_pricing", "cross_scenario_and_secondary",
            )
        },
    }), encoding="utf-8")
    publication._validate_run_provenance()

    original_seed_bytes = seed_path.read_bytes()
    seed_path.write_bytes(original_seed_bytes + b"\n")
    with pytest.raises(SystemExit):
        publication._validate_run_provenance()
    seed_path.write_bytes(original_seed_bytes)

    aggregate_path = tmp_path / "channel_attribution_aggregate.json"
    original_aggregate_bytes = aggregate_path.read_bytes()
    aggregate_path.write_bytes(original_aggregate_bytes + b"\n")
    with pytest.raises(SystemExit):
        publication._validate_run_provenance()
    aggregate_path.write_bytes(original_aggregate_bytes)

    bad = json.loads((tmp_path / "stress_h3_test.json").read_text(encoding="utf-8"))
    bad["run_tag"] = "mixed"
    (tmp_path / "stress_h3_test.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(SystemExit):
        publication._validate_run_provenance()


def _write_significance_fixture(root: Path) -> Path:
    scenarios = publication.EXPECTED_SCENARIOS
    baseline_comparisons = (
        "agribrain_vs_mcp_only", "agribrain_vs_pirag_only",
        "agribrain_vs_no_context", "agribrain_vs_no_pinn",
        "agribrain_vs_no_slca", "agribrain_vs_hybrid_rl",
        "agribrain_vs_static",
    )
    channel_comparisons = (
        "mcp_only_vs_no_context", "pirag_only_vs_no_context",
    )
    h2_comparisons = (
        "mcp_only_vs_no_context", "pirag_only_vs_no_context",
        "agribrain_vs_mcp_only", "agribrain_vs_pirag_only",
    )
    metrics = ("ari", "waste", "rle", "slca", "carbon", "equity")
    significance = {}
    for scenario in scenarios:
        significance[scenario] = {}
        for comparison in (*baseline_comparisons, *channel_comparisons):
            comp = {
                "is_paired_design": True,
                "test_type": "wilcoxon_signed_rank",
                "effect_size_primary": "cohens_dz",
            }
            if comparison in channel_comparisons:
                comp["_family"] = "channel_decomposition"
            for metric in metrics:
                rec = {
                    "p_value": 0.01,
                    "p_value_adj": 0.02,
                    "cohens_d": 0.5,
                    "cohens_dz": 0.5,
                    "mean_diff": 0.01,
                    "mean_diff_ci_low": 0.005,
                    "mean_diff_ci_high": 0.015,
                    "mean_diff_ci_method": "BCa",
                    "effect_size_ci_method": "BCa",
                    "cohens_dz_ci_low": 0.2,
                    "cohens_dz_ci_high": 0.8,
                    "cohens_dz_ci_method": "BCa",
                    "n_seeds": 20,
                    "test_type_actual": "wilcoxon_signed_rank",
                    "correction_method": "by_fdr_within_scenario",
                }
                if comparison == "agribrain_vs_no_context" and metric == "ari":
                    rec["correction_method"] = "holm_bonferroni_across_scenarios"
                    rec["p_value_adj"] = 0.05
                    rec["p_value_directional_greater"] = 0.01
                    rec["h1_raw_p_value_directional_greater"] = 0.01
                    rec["p_value_adj_holm"] = 0.05
                    rec["h1_family_size"] = 5
                    rec["h1_positive_effect_supported"] = False
                    rec["h1_practical_margin"] = 0.005
                    rec["h1_practical_margin_supported"] = False
                if comparison in h2_comparisons and metric == "ari":
                    rec["p_value_directional_greater"] = 0.01
                    rec["correction_method"] = (
                        "holm_bonferroni_h2_directional_20"
                    )
                    rec["p_value_adj"] = 0.20
                    rec["p_value_adj_holm_h2_directional"] = 0.20
                    rec["h2_family_size"] = 20
                    rec["h2_direction"] = comparison.replace("_vs_", " > ")
                    rec["h2_cell_supported"] = False
                    rec["h2_correction_method"] = (
                        "holm_bonferroni_across_20_directional_ari_contrasts"
                    )
                    rec["canonical_raw_p_value_field"] = (
                        "p_value_directional_greater"
                    )
                    if comparison in channel_comparisons:
                        rec["p_value_adj_holm_channel"] = 0.10
                if comparison == "agribrain_vs_no_pinn" and metric == "ari":
                    rec["p_value_directional_greater"] = 0.01
                    rec["pinn_ablation_raw_p_value_directional_greater"] = 0.01
                    rec["pinn_ablation_family_size"] = 5
                    rec["directional_test_type_actual"] = (
                        "wilcoxon_signed_rank"
                    )
                    rec["p_value_adj"] = 0.05
                    rec["p_value_adj_holm_pinn_ablation"] = 0.05
                    rec["correction_method"] = (
                        "holm_bonferroni_pinn_ablation_5"
                    )
                comp[metric] = rec
            significance[scenario][comparison] = comp
        significance[scenario]["h2_synergy_interaction"] = {
            "is_paired_design": True,
            "test_type": "wilcoxon_signed_rank_greater",
            "effect_size_primary": "cohens_dz",
            "exploratory": True,
            "interpretation": "Full - MCP - Retrieval + No-context",
            "ari": {
                "p_value_directional_greater": 0.01,
                "mean_interaction": 0.001,
                "mean_interaction_ci_low": -0.001,
                "mean_interaction_ci_high": 0.003,
                "mean_interaction_ci_method": "BCa",
                "cohens_dz": 0.1,
                "within_pair_sd": 0.01,
                "n_seeds": 20,
                "p_value_adj_holm_exploratory": 0.05,
            },
        }
    h1 = {scenario: 0.05 for scenario in scenarios}
    h2 = {
        f"{scenario}:{comparison}": 0.20
        for scenario in scenarios for comparison in h2_comparisons
    }
    pinn = {scenario: 0.05 for scenario in scenarios}
    h2_rows = []
    for scenario in scenarios:
        for comparison in h2_comparisons:
            left, right = comparison.split("_vs_", 1)
            record = significance[scenario][comparison]["ari"]
            h2_rows.append({
                "source_commit": "a" * 40,
                "run_tag": "fixture_run",
                "scenario": scenario,
                "comparison": comparison,
                "numerator_mode": left,
                "denominator_mode": right,
                "direction": f"{left} > {right}",
                "endpoint": "ari",
                "n_seeds": 20,
                "paired_design": True,
                "test": "wilcoxon_signed_rank",
                "alternative": "greater",
                "mean_difference": record["mean_diff"],
                "mean_difference_ci_low": record["mean_diff_ci_low"],
                "mean_difference_ci_high": record["mean_diff_ci_high"],
                "mean_difference_ci_method": record["mean_diff_ci_method"],
                "cohens_dz": record["cohens_dz"],
                "cohens_dz_ci_low": record["cohens_dz_ci_low"],
                "cohens_dz_ci_high": record["cohens_dz_ci_high"],
                "cohens_dz_ci_method": record["cohens_dz_ci_method"],
                "raw_directional_p_value": record[
                    "p_value_directional_greater"
                ],
                "holm_adjusted_p_value": record[
                    "p_value_adj_holm_h2_directional"
                ],
                "holm_family_size": 20,
                "alpha": 0.05,
                "positive_mean": True,
                "cell_supported": False,
            })
    path = root / "benchmark_significance.json"
    payload = {
        "_meta": {
            "source_commit": "a" * 40,
            "run_tag": "fixture_run",
            "n_seeds": 20,
            "paired": True,
            "wilcoxon_fallback_count": 0,
            "confirmatory_test": "directional_wilcoxon_signed_rank",
            "legacy_sign_flip_resamples": 10_000,
            "n_perm_scope": (
                "legacy descriptive sign-flip only; not the canonical H1/H2 test"
            ),
            "primary_h1_correction": "holm_bonferroni",
            "pinn_ablation_correction": "holm_bonferroni",
            "pinn_ablation_scope": (
                "separate prespecified paired mechanistic-residual ablation; "
                "not part of H1 or H2"
            ),
            "h2_directional_correction": "holm_bonferroni",
            "h2_directional_canonical_field": (
                "p_value_adj_holm_h2_directional"
            ),
            "channel_decomposition_correction": "holm_bonferroni",
        },
        "primary_h1_holm_adjusted": h1,
        "primary_h1_supported_by_cell": {
            scenario: False for scenario in scenarios
        },
        "primary_h1_supported_all_cells": False,
        "pinn_ablation_holm_adjusted": pinn,
        "pinn_ablation_supported_by_cell": {
            scenario: False for scenario in scenarios
        },
        "pinn_ablation_supported_all_cells": False,
        "h2_directional_holm_adjusted": h2,
        "h2_directional_supported_by_cell": {
            key: False for key in h2
        },
        "h2_directional_supported_all_cells": False,
        "h2_synergy_holm_adjusted_exploratory": {
            scenario: 0.05 for scenario in scenarios
        },
        "channel_decomposition_holm_adjusted": {
            f"{scenario}:{comparison}": 0.10
            for scenario in scenarios for comparison in channel_comparisons
        },
        "h2_directional_evidence": h2_rows,
        "significance": significance,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with (root / "h2_directional_evidence.csv").open(
        "w", newline="", encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=publication.H2_PUBLICATION_COLUMNS,
        )
        writer.writeheader()
        writer.writerows(h2_rows)
    return path


def test_final_significance_gate_accepts_exact_h1_h2_families(
    tmp_path, monkeypatch,
):
    _write_significance_fixture(tmp_path)
    monkeypatch.setattr(publication, "RESULTS_DIR", tmp_path)
    publication._validate_significance()


def test_final_significance_gate_rejects_wilcoxon_method_fallback(
    tmp_path, monkeypatch,
):
    path = _write_significance_fixture(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_meta"]["wilcoxon_fallback_count"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(publication, "RESULTS_DIR", tmp_path)
    with pytest.raises(SystemExit):
        publication._validate_significance()


def test_final_significance_gate_recomputes_holm_family(tmp_path, monkeypatch):
    path = _write_significance_fixture(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Keep the top-level map and per-cell canonical field mutually consistent
    # but numerically wrong.  A schema-only gate would accept this pair.
    payload["primary_h1_holm_adjusted"]["baseline"] = 0.04
    payload["significance"]["baseline"]["agribrain_vs_no_context"]["ari"][
        "p_value_adj"
    ] = 0.04
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(publication, "RESULTS_DIR", tmp_path)
    with pytest.raises(SystemExit):
        publication._validate_significance()


def test_final_significance_gate_recomputes_global_h1_support(
    tmp_path, monkeypatch,
):
    path = _write_significance_fixture(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["primary_h1_supported_all_cells"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(publication, "RESULTS_DIR", tmp_path)
    with pytest.raises(SystemExit):
        publication._validate_significance()


def test_final_significance_gate_rejects_h1_claim_from_percentile_fallback(
    tmp_path, monkeypatch,
):
    path = _write_significance_fixture(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    record = payload["significance"]["baseline"][
        "agribrain_vs_no_context"
    ]["ari"]
    record["mean_diff_ci_method"] = "percentile_fallback"
    record["mean_diff_ci_low"] = 0.006
    record["h1_practical_margin_supported"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(publication, "RESULTS_DIR", tmp_path)
    with pytest.raises(SystemExit):
        publication._validate_significance()


def test_final_significance_gate_recomputes_h1_support_flag(
    tmp_path, monkeypatch,
):
    path = _write_significance_fixture(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["significance"]["baseline"]["agribrain_vs_no_context"]["ari"][
        "h1_positive_effect_supported"
    ] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(publication, "RESULTS_DIR", tmp_path)
    with pytest.raises(SystemExit):
        publication._validate_significance()


@pytest.mark.parametrize("tamper", ["cell", "map", "global"])
def test_final_significance_gate_recomputes_h2_support_flags(
    tmp_path, monkeypatch, tamper,
):
    path = _write_significance_fixture(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    key = "baseline:mcp_only_vs_no_context"
    if tamper == "cell":
        payload["significance"]["baseline"]["mcp_only_vs_no_context"]["ari"][
            "h2_cell_supported"
        ] = True
    elif tamper == "map":
        payload["h2_directional_supported_by_cell"][key] = True
    else:
        payload["h2_directional_supported_all_cells"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(publication, "RESULTS_DIR", tmp_path)
    with pytest.raises(SystemExit):
        publication._validate_significance()


def _write_saturation_fixture(root: Path, *, pooled_n: int = 20) -> None:
    def tost(n: int = 20) -> dict:
        return {
            "n": n,
            "sesoi": 0.01,
            "mean_diff": 0.0,
            "ci90_low": -0.001,
            "ci90_high": 0.001,
            "ci95_low": -0.002,
            "ci95_high": 0.002,
            "p_two_sided": 1.0,
            "p_tost": 0.001,
            "verdict": "equivalent_within_margin",
        }

    by_scenario = {
        scenario: {
            "n_seeds": 20,
            "add_pirag_on_mcp": tost(),
            "add_mcp_on_pirag": tost(),
        }
        for scenario in publication.EXPECTED_SCENARIOS
    }
    fit = {
        "n": 4,
        "slope": 0.0,
        "r2": 0.0,
        "p_value": None,
        "inferential": False,
        "unit": "scenario",
        "estimable": True,
    }
    payload = {
        "_meta": {
            "n_seeds": 20,
            "seed_order": list(publication.EXPECTED_SEEDS),
        },
        "by_scenario": by_scenario,
        "pooled_perturbed": {
            "inferential_unit": "seed",
            "scenario_aggregation": (
                "mean paired difference across four scenarios within seed"
            ),
            "scenarios": list(publication.EXPECTED_SCENARIOS[:-1]),
            "add_pirag_on_mcp": tost(pooled_n),
            "add_mcp_on_pirag": tost(pooled_n),
        },
        "moderation": {
            name: {"crossfit": dict(fit), "naive_coupled_bound": dict(fit)}
            for name in (
                "pirag_marginal_vs_mcp_strength",
                "mcp_marginal_vs_pirag_strength",
            )
        },
    }
    (root / "channel_saturation_analysis.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )


def test_channel_saturation_gate_accepts_seed_level_pooling(tmp_path, monkeypatch):
    _write_saturation_fixture(tmp_path)
    monkeypatch.setattr(publication, "RESULTS_DIR", tmp_path)
    publication._validate_channel_saturation()


def test_channel_saturation_gate_rejects_scenario_pseudoreplication(
    tmp_path, monkeypatch,
):
    _write_saturation_fixture(tmp_path, pooled_n=80)
    monkeypatch.setattr(publication, "RESULTS_DIR", tmp_path)
    with pytest.raises(SystemExit):
        publication._validate_channel_saturation()


def test_channel_saturation_gate_rejects_tost_ci_disagreement(tmp_path, monkeypatch):
    _write_saturation_fixture(tmp_path)
    path = tmp_path / "channel_saturation_analysis.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["pooled_perturbed"]["add_mcp_on_pirag"]["p_tost"] = 0.20
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(publication, "RESULTS_DIR", tmp_path)
    with pytest.raises(SystemExit):
        publication._validate_channel_saturation()


def _write_raw_stats_binding_fixture(root: Path, seeds: tuple[int, ...]) -> None:
    seed_root = root / "benchmark_seeds"
    seed_root.mkdir()
    mode_offsets = {
        "agribrain": 0.04,
        "mcp_only": 0.01,
        "pirag_only": 0.02,
        "no_context": 0.00,
        "no_slca": -0.01,
        "hybrid_rl": -0.02,
        "static": -0.05,
    }
    metrics = ("ari", "waste", "rle", "slca", "carbon", "equity")
    for index, seed in enumerate(seeds):
        scenarios = {}
        for scenario in publication.EXPECTED_SCENARIOS:
            scenarios[scenario] = {
                mode: {
                    metric: 0.50 + 0.01 * index + offset
                    for metric in metrics
                }
                for mode, offset in mode_offsets.items()
            }
        (seed_root / f"seed_{seed}.json").write_text(json.dumps({
            "seed": seed,
            "scenarios": scenarios,
        }), encoding="utf-8")

    comparisons = {
        "agribrain_vs_mcp_only": ("agribrain", "mcp_only"),
        "agribrain_vs_pirag_only": ("agribrain", "pirag_only"),
        "agribrain_vs_no_context": ("agribrain", "no_context"),
        "agribrain_vs_no_slca": ("agribrain", "no_slca"),
        "agribrain_vs_hybrid_rl": ("agribrain", "hybrid_rl"),
        "agribrain_vs_static": ("agribrain", "static"),
        "mcp_only_vs_no_context": ("mcp_only", "no_context"),
        "pirag_only_vs_no_context": ("pirag_only", "no_context"),
    }
    significance = {}
    saturation_by_scenario = {}
    differences_by_scenario = {}
    for scenario in publication.EXPECTED_SCENARIOS:
        significance[scenario] = {}
        for comparison, (left_mode, right_mode) in comparisons.items():
            significance[scenario][comparison] = {}
            for metric in metrics:
                left = [
                    0.50 + 0.01 * index + mode_offsets[left_mode]
                    for index, _ in enumerate(seeds)
                ]
                right = [
                    0.50 + 0.01 * index + mode_offsets[right_mode]
                    for index, _ in enumerate(seeds)
                ]
                significance[scenario][comparison][metric] = {
                    "mean_diff": sum(
                        a - b for a, b in zip(left, right, strict=True)
                    ) / len(seeds),
                    "p_value": publication._raw_wilcoxon(left, right),
                }
                if metric == "ari" and comparison in {
                    "agribrain_vs_no_context",
                    "mcp_only_vs_no_context", "pirag_only_vs_no_context",
                    "agribrain_vs_mcp_only", "agribrain_vs_pirag_only",
                }:
                    directional = publication._raw_wilcoxon(
                        left, right, alternative="greater",
                    )
                    significance[scenario][comparison][metric][
                        "p_value_directional_greater"
                    ] = directional
                    if comparison == "agribrain_vs_no_context":
                        significance[scenario][comparison][metric][
                            "h1_raw_p_value_directional_greater"
                        ] = directional
        full = [0.50 + 0.01 * index + mode_offsets["agribrain"]
                for index, _ in enumerate(seeds)]
        mcp = [0.50 + 0.01 * index + mode_offsets["mcp_only"]
               for index, _ in enumerate(seeds)]
        pirag = [0.50 + 0.01 * index + mode_offsets["pirag_only"]
                 for index, _ in enumerate(seeds)]
        differences = {
            "add_pirag_on_mcp": [
                a - b for a, b in zip(full, mcp, strict=True)
            ],
            "add_mcp_on_pirag": [
                a - b for a, b in zip(full, pirag, strict=True)
            ],
        }
        differences_by_scenario[scenario] = differences
        interactions = [
            a - b - c + d
            for a, b, c, d in zip(
                full,
                mcp,
                pirag,
                [
                    0.50 + 0.01 * index + mode_offsets["no_context"]
                    for index, _ in enumerate(seeds)
                ],
                strict=True,
            )
        ]
        significance[scenario]["h2_synergy_interaction"] = {
            "ari": {
                "mean_interaction": sum(interactions) / len(interactions),
                "p_value_directional_greater": publication._raw_wilcoxon(
                    interactions,
                    [0.0] * len(interactions),
                    alternative="greater",
                ),
            },
        }
        saturation_by_scenario[scenario] = {
            name: {
                "n": len(seeds),
                "sesoi": 0.01,
                "mean_diff": stats["mean"],
                "ci90_low": stats["ci90_low"],
                "ci90_high": stats["ci90_high"],
                "ci95_low": stats["ci95_low"],
                "ci95_high": stats["ci95_high"],
                "p_two_sided": stats["p_two_sided"],
                "p_tost": stats["p_tost"],
                "verdict": stats["verdict"],
            }
            for name, values in differences.items()
            for stats in [publication._recompute_tost(values, 0.01)]
        }
    (root / "benchmark_significance.json").write_text(json.dumps({
        "significance": significance,
    }), encoding="utf-8")

    perturbed = publication.EXPECTED_SCENARIOS[:-1]
    pooled = {}
    for name in ("add_pirag_on_mcp", "add_mcp_on_pirag"):
        values = [
            sum(differences_by_scenario[scenario][name][index]
                for scenario in perturbed) / len(perturbed)
            for index in range(len(seeds))
        ]
        stats = publication._recompute_tost(values, 0.01)
        pooled[name] = {
            "n": len(seeds),
            "sesoi": 0.01,
            "mean_diff": stats["mean"],
            "ci90_low": stats["ci90_low"],
            "ci90_high": stats["ci90_high"],
            "ci95_low": stats["ci95_low"],
            "ci95_high": stats["ci95_high"],
            "p_two_sided": stats["p_two_sided"],
            "p_tost": stats["p_tost"],
            "verdict": stats["verdict"],
        }
    (root / "channel_saturation_analysis.json").write_text(json.dumps({
        "by_scenario": saturation_by_scenario,
        "pooled_perturbed": pooled,
    }), encoding="utf-8")


def test_raw_seed_gate_binds_h1_h2_and_saturation_statistics(tmp_path, monkeypatch):
    seeds = (42, 43, 44)
    monkeypatch.setattr(publication, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(publication, "EXPECTED_SEEDS", seeds)
    _write_raw_stats_binding_fixture(tmp_path, seeds)
    publication._validate_h1_h2_against_raw()
    publication._validate_channel_saturation_against_raw()

    significance_path = tmp_path / "benchmark_significance.json"
    significance = json.loads(significance_path.read_text(encoding="utf-8"))
    significance["significance"]["baseline"]["agribrain_vs_no_context"]["ari"][
        "mean_diff"
    ] += 0.001
    significance_path.write_text(json.dumps(significance), encoding="utf-8")
    with pytest.raises(SystemExit):
        publication._validate_h1_h2_against_raw()


def test_raw_seed_gate_rejects_coherently_tampered_directional_h2(
    tmp_path, monkeypatch,
):
    seeds = (42, 43, 44)
    monkeypatch.setattr(publication, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(publication, "EXPECTED_SEEDS", seeds)
    _write_raw_stats_binding_fixture(tmp_path, seeds)

    significance_path = tmp_path / "benchmark_significance.json"
    significance = json.loads(significance_path.read_text(encoding="utf-8"))
    record = significance["significance"]["baseline"][
        "agribrain_vs_mcp_only"
    ]["ari"]
    record["p_value_directional_greater"] = 0.99
    significance_path.write_text(json.dumps(significance), encoding="utf-8")
    with pytest.raises(SystemExit):
        publication._validate_h1_h2_against_raw()


def test_raw_seed_gate_rejects_coherently_tampered_saturation(tmp_path, monkeypatch):
    seeds = (42, 43, 44)
    monkeypatch.setattr(publication, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(publication, "EXPECTED_SEEDS", seeds)
    _write_raw_stats_binding_fixture(tmp_path, seeds)
    path = tmp_path / "channel_saturation_analysis.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    cell = payload["by_scenario"]["baseline"]["add_pirag_on_mcp"]
    cell["mean_diff"] = 0.0
    cell["p_tost"] = 0.001
    cell["verdict"] = "equivalent_within_margin"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit):
        publication._validate_channel_saturation_against_raw()


def test_manifest_accepts_only_current_run_scoped_consolidated_ledgers():
    tag = "abcdef0_20260819_120000"
    current = f"decision_ledger_per_seed/{tag}/seed_42/agribrain__baseline.jsonl"
    old = "decision_ledger_per_seed/old_run/seed_42/agribrain__baseline.jsonl"
    unscoped = "decision_ledger_per_seed/seed_42/agribrain__baseline.jsonl"
    assert manifest_builder._is_canonical_path(current, True, tag)
    assert not manifest_builder._is_canonical_path(old, True, tag)
    assert not manifest_builder._is_canonical_path(unscoped, True, tag)
    h3 = (
        f"decision_ledger_h3/{tag}/baseline/sensor_noise/seed_42/"
        "agribrain__baseline.jsonl"
    )
    h3_old = h3.replace(tag, "old_run")
    h3_extra_mode_dir = h3.replace(
        "/seed_42/agribrain__", "/seed_42/agribrain/agribrain__",
    )
    assert manifest_builder._is_canonical_path(h3, True, tag)
    assert not manifest_builder._is_canonical_path(h3_old, True, tag)
    assert not manifest_builder._is_canonical_path(h3_extra_mode_dir, True, tag)


def test_strict_seed_run_rejects_trace_serialization_failure(monkeypatch):
    monkeypatch.setenv("STRICT_VALIDATION", "1")
    with pytest.raises(RuntimeError, match="trace serialization failures"):
        single_seed._enforce_strict_trace_completion(["baseline/agribrain/ari"], 42)


def test_non_strict_seed_run_preserves_diagnostic_trace_failure(monkeypatch):
    monkeypatch.setenv("STRICT_VALIDATION", "0")
    single_seed._enforce_strict_trace_completion(["baseline/agribrain/ari"], 42)
