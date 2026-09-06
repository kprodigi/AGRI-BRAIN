"""Focused checks for the locked structural sensitivity framework."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mvp.simulation.sensitivity.analyze_structural_sensitivity import analyze_payloads
from mvp.simulation.sensitivity.design import (
    PRIMARY_MODES,
    STRESSORS,
    build_design,
    build_task_manifest,
    canonical_sha256,
    load_locked_protocol,
    validate_design,
    validate_task_manifest,
)
from mvp.simulation.sensitivity.overrides import (
    applied_structural_parameters,
    expected_structural_outcome_equation_contract,
    validate_dynamic_influence,
)
from mvp.simulation.sensitivity.parameters import (
    EXCLUDED_PARAMETERS,
    PARAMETERS,
    validate_parameter_registry,
)
from mvp.simulation.sensitivity.run_structural_sensitivity import (
    _bind_retained_ledger,
    _canonicalize_structural_ledger_paths_for_install,
    _extract_endpoint,
    _structural_episode_evidence_expectations,
    _validate_existing_result,
    run_one_task,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = REPO_ROOT / "mvp" / "simulation" / "experiment_protocol.json"


def _bundle():
    protocol = load_locked_protocol(PROTOCOL_PATH)
    design = build_design(protocol)
    manifest = build_task_manifest(design, protocol)
    return protocol, design, manifest


def test_parameter_registry_has_only_active_audited_factors() -> None:
    report = validate_parameter_registry(REPO_ROOT)
    assert report["status"] == "pass"
    assert report["n_parameters"] == 29
    assert report["n_source_references_checked"] >= 2 * len(PARAMETERS)
    assert "policy.carbon_cap" not in EXCLUDED_PARAMETERS
    assert "frozen PINN training hyperparameters" in EXCLUDED_PARAMETERS
    assert all(parameter.lower < parameter.default < parameter.upper for parameter in PARAMETERS)


def test_every_registered_factor_changes_a_production_observable() -> None:
    report = validate_dynamic_influence(REPO_ROOT)
    assert report["status"] == "pass"
    assert report["n_parameters"] == len(PARAMETERS)
    assert {record["parameter"] for record in report["records"]} == {
        parameter.key for parameter in PARAMETERS
    }


def test_design_is_reproducible_latin_and_seed_balanced() -> None:
    protocol, first, manifest = _bundle()
    second = build_design(protocol)
    assert first == second
    validate_design(first, protocol)
    validate_task_manifest(manifest, protocol)
    assert len(first["points"]) == 100
    assert first["probability_interpretation"] is False

    seeds = [point["seed"] for point in first["points"]]
    assert {seed: seeds.count(seed) for seed in protocol["seeds"]} == {
        seed: 5 for seed in protocol["seeds"]
    }
    for parameter in PARAMETERS:
        strata = sorted(
            int(point["lhs_unit_coordinates"][parameter.key] * 100)
            for point in first["points"]
        )
        assert strata == list(range(100))
        assert all(
            parameter.lower <= point["parameters"][parameter.key] <= parameter.upper
            for point in first["points"]
        )
    for point in first["points"]:
        derived = point["derived_parameters"]
        assert derived["slca_weight_sum"] == 1.0
        assert 0.15 <= derived["slca_weight_price_transparency"] <= 0.35


def test_manifest_exactly_matches_locked_6500_24500_accounting() -> None:
    protocol, _design, manifest = _bundle()
    assert manifest["n_tasks"] == 3000
    assert manifest["accounting"]["per_design_point"] == {
        "primary_retained_cells": 40,
        "primary_executed_episodes": 145,
        "h3_stressed_retained_cells": 25,
        "h3_stressed_executed_episodes": 100,
        "total_retained_cells": 65,
        "total_executed_episodes": 245,
    }
    assert manifest["accounting"]["total"] == {
        "retained_cells": 6500,
        "executed_episodes": 24500,
        "simulated_steps": 7056000,
    }
    primary_tasks = [task for task in manifest["tasks"] if task["panel"] == "primary"]
    stress_tasks = [task for task in manifest["tasks"] if task["panel"] == "h3_stressed"]
    assert len(primary_tasks) == 500
    assert len(stress_tasks) == 2500
    assert all(task["retained_cells"] == 8 and task["executed_episodes"] == 29
               for task in primary_tasks)
    assert all(task["retained_cells"] == 1 and task["executed_episodes"] == 4
               for task in stress_tasks)
    assert protocol["counts"]["structural_sensitivity"] == {
        "latin_hypercube_points": 100,
        "active_factors": 29,
        "retained_cells": 6500,
        "executed_episodes": 24500,
        "simulated_steps": 7056000,
    }


def test_primary_task_evidence_counts_match_all_eight_modes(tmp_path: Path) -> None:
    evidence_root, expected = _structural_episode_evidence_expectations(
        {"panel": "primary"}, tmp_path,
    )
    assert evidence_root == tmp_path / "runtime_artifacts" / "decision_ledger"
    assert expected == {
        "expected_groups": 8,
        "expected_episodes": 29,
        "expected_adaptation_ledgers": 21,
        "expected_final_ledgers": 8,
    }


def _endpoint_fixture(phase: str) -> dict:
    return {
        "ari": 0.7,
        "waste": 0.1,
        "rle": 0.4,
        "slca": 0.8,
        "carbon": 1000.0,
        "equity": 0.6,
        "benchmark_seed": 42,
        "episode_index": 3,
        "episode_phase": phase,
        "learning_enabled": False,
        "learner_freeze_summary": {"learners_frozen": True},
        "effective_Ea_R": 8000.0,
        "spoilage_estimator": {
            "kind": "mechanistic_plus_frozen_synthetic_pinn_residual",
        },
        "latent_spoilage_model": {
            "kind": "independent_synthetic_dgp_v1",
        },
    }


def test_structural_endpoint_accepts_static_fixed_evaluation() -> None:
    endpoint = _extract_endpoint(
        _endpoint_fixture("fixed_evaluation"),
        42,
        expected_phase="fixed_evaluation",
    )
    assert endpoint["episode_phase"] == "fixed_evaluation"
    assert endpoint["effective_Ea_R"] == 8000.0


def test_structural_endpoint_rejects_phase_mismatch() -> None:
    with pytest.raises(ValueError, match="frozen_evaluation"):
        _extract_endpoint(
            _endpoint_fixture("fixed_evaluation"),
            42,
            expected_phase="frozen_evaluation",
        )


@pytest.mark.parametrize(
    "field", ("spoilage_estimator", "latent_spoilage_model"),
)
def test_structural_endpoint_requires_spoilage_provenance(field: str) -> None:
    episode = _endpoint_fixture("fixed_evaluation")
    episode.pop(field)
    with pytest.raises(ValueError, match="missing spoilage provenance"):
        _extract_endpoint(
            episode,
            42,
            expected_phase="fixed_evaluation",
        )


def _transaction_task_fixture() -> dict:
    return {
        "task_sha256": "b" * 64,
        "task_id": "lhs_000__baseline__primary",
        "task_index": 0,
        "output_relpath": "tasks/lhs_000/baseline__primary.json",
        "panel": "primary",
        "point_id": "lhs_000",
        "point_index": 0,
        "scenario": "baseline",
        "seed": 42,
        "modes": ["static"],
        "design_sha256": "e" * 64,
        "parameters_sha256": "f" * 64,
        "retained_cells": 1,
        "executed_episodes": 1,
        "simulated_steps": 288,
    }


def test_structural_binding_preserves_run_prefix_for_h3_attempt(
    tmp_path: Path,
) -> None:
    task = {
        "output_relpath": "tasks/lhs_000/heatwave__h3__sensor_noise.json",
        "panel": "h3_stressed",
        "point_id": "lhs_000",
        "scenario": "heatwave",
        "stressor": "sensor_noise",
        "seed": 42,
        "modes": ["agribrain"],
    }
    attempt_root = (
        tmp_path / "tasks/lhs_000/heatwave__h3__sensor_noise__attempts/attempt_x"
    )
    final_root = (
        tmp_path / "tasks/lhs_000/heatwave__h3__sensor_noise__artifacts"
    )
    suffix = Path(
        "decision_ledgers/heatwave/structural__lhs_000__sensor_noise/"
        "seed_42/agribrain__heatwave.jsonl"
    )
    ledger = attempt_root / suffix
    ledger.parent.mkdir(parents=True)
    merkle_root = "a" * 64
    ledger.write_text(json.dumps({
        "_header": True,
        "merkle_root": merkle_root,
        "n_records": 288,
    }) + "\n", encoding="utf-8")
    endpoint: dict = {}
    _bind_retained_ledger(
        {
            "decision_ledger_path": str(ledger),
            "decision_ledger_root": merkle_root,
            "decision_ledger_n": 288,
        },
        endpoint,
        task_root=attempt_root,
        run_root=tmp_path,
        ledger_path=ledger,
    )
    assert endpoint["decision_ledger_path"] == (
        "tasks/lhs_000/heatwave__h3__sensor_noise__attempts/attempt_x/"
        + suffix.as_posix()
    )

    panel = {"results": {"agribrain": endpoint}}
    _canonicalize_structural_ledger_paths_for_install(
        task=task,
        panel_payload=panel,
        attempt_root=attempt_root,
        final_task_root=final_root,
        run_root=tmp_path,
    )
    assert endpoint["decision_ledger_path"] == (
        "tasks/lhs_000/heatwave__h3__sensor_noise__artifacts/"
        + suffix.as_posix()
    )


def test_structural_attempt_paths_are_canonicalized_only_after_exact_match(
    tmp_path: Path,
) -> None:
    task = _transaction_task_fixture()
    attempt_root = tmp_path / "tasks" / "lhs_000" / "baseline__primary__attempts" / "attempt_x"
    final_root = tmp_path / "tasks" / "lhs_000" / "baseline__primary__artifacts"
    suffix = Path("runtime_artifacts/decision_ledger/static__baseline.jsonl")
    ledger = attempt_root / suffix
    ledger.parent.mkdir(parents=True)
    ledger.write_text("ledger bytes\n", encoding="utf-8")
    panel = {
        "results": {
            "static": {
                "decision_ledger_path": ledger.relative_to(tmp_path).as_posix(),
                "decision_ledger_sha256": hashlib.sha256(
                    ledger.read_bytes()
                ).hexdigest(),
            }
        }
    }
    _canonicalize_structural_ledger_paths_for_install(
        task=task,
        panel_payload=panel,
        attempt_root=attempt_root,
        final_task_root=final_root,
        run_root=tmp_path,
    )
    assert panel["results"]["static"]["decision_ledger_path"] == (
        "tasks/lhs_000/baseline__primary__artifacts/runtime_artifacts/"
        "decision_ledger/static__baseline.jsonl"
    )


def test_structural_resume_recovers_atomic_completion_without_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mvp.simulation.sensitivity.run_structural_sensitivity as runner

    task = _transaction_task_fixture()
    plan = {"source_commit": "a" * 40, "protocol": {"sha256": "c" * 64}}
    point = {"point_id": "lhs_000", "point_index": 0, "parameters_sha256": "f" * 64}
    design = {"points": [point]}
    manifest = {"tasks": [task]}
    plan_path = tmp_path / "run_plan.json"
    plan_path.write_text("{}", encoding="utf-8")
    output = tmp_path / task["output_relpath"]
    task_root = output.parent / "baseline__primary__artifacts"
    completion = task_root / "_completion" / "task_result.json"
    completion.parent.mkdir(parents=True)
    completion_payload = {"result_sha256": "recovered", "marker": 17}
    completion.write_text(json.dumps(completion_payload), encoding="utf-8")

    monkeypatch.delenv("STRICT_VALIDATION", raising=False)
    monkeypatch.setattr(
        runner, "_load_plan_bundle",
        lambda _path: (plan, {}, design, manifest),
    )
    monkeypatch.setattr(runner, "_assert_execution_source", lambda _plan: None)
    monkeypatch.setattr(
        runner, "_validate_existing_result", lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        runner,
        "_run_primary_task",
        lambda *_args, **_kwargs: pytest.fail("recovery reran the simulation"),
    )

    assert run_one_task(plan_path, task_index=0, resume=True) == output
    assert json.loads(output.read_text(encoding="utf-8")) == completion_payload


def test_structural_failed_attempt_is_preserved_for_diagnosis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mvp.simulation.sensitivity.run_structural_sensitivity as runner

    task = _transaction_task_fixture()
    plan = {"source_commit": "a" * 40, "protocol": {"sha256": "c" * 64}}
    point = {"point_id": "lhs_000", "point_index": 0, "parameters_sha256": "f" * 64}
    plan_path = tmp_path / "run_plan.json"
    plan_path.write_text("{}", encoding="utf-8")
    monkeypatch.delenv("STRICT_VALIDATION", raising=False)
    monkeypatch.setattr(
        runner, "_load_plan_bundle",
        lambda _path: (plan, {}, {"points": [point]}, {"tasks": [task]}),
    )
    monkeypatch.setattr(runner, "_assert_execution_source", lambda _plan: None)

    def _fail(*_args, **_kwargs):
        raise RuntimeError("intentional task failure")

    monkeypatch.setattr(runner, "_run_primary_task", _fail)
    with pytest.raises(RuntimeError, match="intentional task failure"):
        run_one_task(plan_path, task_index=0, resume=True)

    attempts = list((tmp_path / "tasks/lhs_000/baseline__primary__attempts").glob("attempt_*"))
    assert len(attempts) == 1
    failure_paths = list(
        (attempts[0] / "_attempt_failures").glob("failure_*.json")
    )
    assert len(failure_paths) == 1
    failure = json.loads(failure_paths[0].read_text(encoding="utf-8"))
    assert failure["status"] == "FAILED_ATTEMPT_RETAINED"
    assert failure["exception_type"] == "RuntimeError"
    assert failure["exception_message"] == "intentional task failure"
    assert len(failure["failure_sha256"]) == 64

    with pytest.raises(RuntimeError, match="intentional task failure"):
        run_one_task(plan_path, task_index=0, resume=True)
    assert len(list(
        (attempts[0] / "_attempt_failures").glob("failure_*.json")
    )) == 2
    assert len(list(
        (tmp_path / "tasks/lhs_000/baseline__primary__attempts").glob("attempt_*")
    )) == 1


def test_structural_task_endpoint_is_recomputed_from_bound_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mvp.simulation.sensitivity.run_structural_sensitivity as structural_runner
    from hpc import validate_decision_ledgers as ledger_validator

    _protocol, design, _manifest = _bundle()
    point = design["points"][0]
    commit = "a" * 40
    task = {
        "task_sha256": "b" * 64,
        "task_id": "lhs_000__baseline__primary",
        "task_index": 0,
        "output_relpath": "tasks/lhs_000/baseline__primary.json",
        "panel": "primary",
        "point_id": "lhs_000",
        "point_index": 0,
        "scenario": "baseline",
        "seed": 42,
        "modes": ["static"],
        "design_sha256": "e" * 64,
        "parameters_sha256": "f" * 64,
        "retained_cells": 1,
        "executed_episodes": 1,
        "simulated_steps": 288,
    }
    ledger_relative = (
        "tasks/lhs_000/baseline__primary__artifacts/runtime_artifacts/"
        "decision_ledger/static__baseline.jsonl"
    )
    ledger = tmp_path / ledger_relative
    ledger.parent.mkdir(parents=True)
    header = {
        "_header": True,
        "merkle_root": "c" * 64,
        "n_records": 288,
        "metadata": {
            "benchmark_seed": 42,
            "episode_index": 3,
            "episode_phase": "fixed_evaluation",
            "learning_enabled": False,
            "latent_environment_sha256": "d" * 64,
            "observed_policy_input_sha256": "1" * 64,
            "demand_observation_sha256": "2" * 64,
            "trace_schema_version": 1,
            "spoilage_estimator": {
                "kind": "mechanistic_plus_frozen_synthetic_pinn_residual",
            },
            "latent_spoilage_model": {
                "kind": "independent_synthetic_dgp_v1",
            },
        },
    }
    ledger.write_text(json.dumps(header) + "\n", encoding="utf-8")
    headlines = {
        "ari": 0.7,
        "waste": 0.1,
        "slca": 0.8,
        "carbon": 1000.0,
        "equity": 0.6,
        "rle": 0.4,
        "constraint_violation_rate": 0.0,
    }
    captured_validation_inputs = {}

    def _validate_ledger_fixture(*_args, **kwargs):
        captured_validation_inputs.update(kwargs)
        return {
            "latent_environment_sha256": "d" * 64,
            "headline_metrics": headlines,
            "learner_snapshots": {
                "theta_delta_by_role": {},
                "reward_shaping": None,
                "context_theta": None,
                "context_slca_amp": None,
            },
        }

    monkeypatch.setattr(
        ledger_validator,
        "validate_ledger",
        _validate_ledger_fixture,
    )
    endpoint = {
        **headlines,
        "message_count": 1,
        "benchmark_seed": 42,
        "episode_index": 3,
        "episode_phase": "fixed_evaluation",
        "learning_enabled": False,
        "latent_environment_sha256": "d" * 64,
        "observed_policy_input_sha256": "1" * 64,
        "demand_observation_sha256": "2" * 64,
        "trace_schema_version": 1,
        "spoilage_estimator": dict(
            header["metadata"]["spoilage_estimator"]
        ),
        "latent_spoilage_model": dict(
            header["metadata"]["latent_spoilage_model"]
        ),
        "decision_ledger_path": ledger_relative,
        "decision_ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
        "decision_ledger_merkle_root": "c" * 64,
        "decision_ledger_n_records": 288,
    }
    evidence_manifest = {
        "schema_version": 1,
        "status": "COMPLETE",
        "ledger_root": "decision_ledger",
        "counts": {
            "episode_groups": 1,
            "executed_episode_archives": 1,
            "adaptation_episode_ledgers": 0,
            "final_episode_ledgers": 1,
            "decision_records": 288,
        },
        "sequences": [],
        "artifacts": [],
        "manifest_sha256": "4" * 64,
    }
    evidence_manifest_path = (
        ledger.parents[2] / "complete_episode_evidence_manifest.json"
    )
    evidence_manifest_path.write_text(
        json.dumps(evidence_manifest), encoding="utf-8",
    )
    monkeypatch.setattr(
        structural_runner,
        "validate_complete_evidence",
        lambda *_args, **_kwargs: evidence_manifest,
    )
    payload = {
        "schema_version": 1,
        "analysis_label": "structural sensitivity",
        "probability_interpretation": False,
        "task_sha256": task["task_sha256"],
        "source_commit": commit,
        "protocol_sha256": "3" * 64,
        "design_sha256": task["design_sha256"],
        "task_id": task["task_id"],
        "task_index": task["task_index"],
        "point_id": task["point_id"],
        "point_index": task["point_index"],
        "seed": task["seed"],
        "scenario": task["scenario"],
        "panel": task["panel"],
        "stressor": None,
        "parameters_sha256": task["parameters_sha256"],
        "retained_cells": task["retained_cells"],
        "executed_episodes": task["executed_episodes"],
        "simulated_steps": task["simulated_steps"],
        "complete_episode_evidence": {
            "status": "COMPLETE",
            "manifest_path": evidence_manifest_path.relative_to(tmp_path).as_posix(),
            "manifest_file_sha256": hashlib.sha256(
                evidence_manifest_path.read_bytes()
            ).hexdigest(),
            "manifest_sha256": evidence_manifest["manifest_sha256"],
            "counts": evidence_manifest["counts"],
        },
        "results": {"static": endpoint},
    }
    payload["result_sha256"] = canonical_sha256(payload)
    result_path = tmp_path / task["output_relpath"]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    assert _validate_existing_result(
        result_path,
        task,
        {"source_commit": commit, "protocol": {"sha256": "3" * 64}},
        run_root=tmp_path,
        point=point,
    )
    assert len(captured_validation_inputs["expected_scenario_frame"]) == 288
    assert captured_validation_inputs["expected_stochastic_layer"].enabled is True

    for field in ("spoilage_estimator", "latent_spoilage_model"):
        original = dict(payload["results"]["static"][field])
        payload["results"]["static"][field] = {"kind": "tampered"}
        payload.pop("result_sha256")
        payload["result_sha256"] = canonical_sha256(payload)
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=f"endpoint/header {field} mismatch"):
            _validate_existing_result(
                result_path,
                task,
                {"source_commit": commit, "protocol": {"sha256": "3" * 64}},
                run_root=tmp_path,
                point=point,
            )
        payload["results"]["static"][field] = original

    payload["results"]["static"]["ari"] = 0.71
    payload.pop("result_sha256")
    payload["result_sha256"] = canonical_sha256(payload)
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="endpoint ari differs from ledger"):
        _validate_existing_result(
            result_path,
            task,
            {"source_commit": commit, "protocol": {"sha256": "3" * 64}},
            run_root=tmp_path,
            point=point,
        )


def test_structural_ledger_source_replay_uses_explicit_lhs_environment(
    tmp_path: Path,
) -> None:
    from hpc.validate_decision_ledgers import (
        PUBLICATION_DATA_CSV,
        validate_ledger,
    )

    _protocol, design, _manifest = _bundle()
    point = design["points"][0]
    seed = int(point["seed"])
    scenario = "overproduction"
    with applied_structural_parameters(point["parameters"], REPO_ROOT) as applied:
        from pirag import context_to_logits
        from src.models import action_selection

        from mvp.simulation import stochastic

        gr = applied["generate_results_module"]
        policy = applied["policy_factory"]()
        environment_seed = gr._stream_seed(seed, scenario, 3, "environment")
        layer = stochastic.make_stochastic_layer(
            np.random.default_rng(environment_seed), stream_seed=environment_seed,
        )
        base = pd.read_csv(PUBLICATION_DATA_CSV, parse_dates=["timestamp"])
        frame = gr.apply_scenario(
            base,
            scenario,
            policy,
            np.random.default_rng(gr._stream_seed(seed, scenario, 3, "scenario")),
            stoch=layer,
        )
        policy_theta = gr.policy_theta_for_seed(
            np.asarray(action_selection.DECLARED_THETA, dtype=float), seed,
        )
        original_theta = np.asarray(action_selection.THETA, dtype=float).copy()
        action_selection.THETA = policy_theta.copy()
        try:
            with gr.decision_ledger_scope(tmp_path, reset=True):
                gr.run_episode(
                    frame,
                    "static",
                    policy,
                    np.random.default_rng(
                        gr._stream_seed(seed, scenario, 3, "policy")
                    ),
                    scenario=scenario,
                    stoch=layer,
                    seed=seed,
                    benchmark_seed=seed,
                    episode_index=3,
                    environment_stream_id=gr._stream_id(
                        seed, scenario, 3, "environment",
                    ),
                    policy_stream_id=gr._stream_id(
                        seed, scenario, 3, "policy",
                    ),
                    stochastic_stream_id=gr._stream_id(
                        seed, scenario, 3, "environment",
                    ),
                    learning_enabled=False,
                )
        finally:
            action_selection.THETA = original_theta
        outcome_contract = gr.build_outcome_equation_contract(
            policy,
            effective_k_ref=layer.perturb_k_ref(policy.k_ref, counter=0),
            effective_ea_r=layer.perturb_ea_r(policy.Ea_R, counter=0),
            stochastic_layer=layer,
        )
        summary = validate_ledger(
            tmp_path / f"static__{scenario}.jsonl",
            mode="static",
            scenario=scenario,
            benchmark_seed=seed,
            expected_outcome_equation_contract=outcome_contract,
            expected_policy=policy,
            expected_policy_theta=policy_theta,
            expected_context_prior=np.asarray(
                context_to_logits.THETA_CONTEXT, dtype=float,
            ),
            expected_policy_temperature=layer.policy_temperature(
                base=1.0, counter=0,
            ),
            expected_scenario_frame=frame,
            expected_stochastic_layer=layer,
        )
    assert summary["source_replay_hashes"][
        "latent_environment_sha256"
    ] == summary["latent_environment_sha256"]


def test_overrides_preserve_slca_simplex_disable_policy_temperature_and_restore() -> None:
    _protocol, design, _manifest = _bundle()
    values = design["points"][0]["parameters"]

    # Imports are deliberately inside the test so override path setup happens
    # before backend absolute imports are resolved.
    with applied_structural_parameters(values, REPO_ROOT) as applied:
        import pirag.context_to_logits as context_to_logits
        import src.models.waste as waste

        policy = applied["policy_factory"]()
        assert abs(policy.w_c + policy.w_l + policy.w_r + policy.w_p - 1.0) < 1e-12
        assert os.environ["STOCH_POLICY_TEMP_STD"] == "0.0"
        assert waste.SAVE_FLOOR["local_redistribute"] == values[
            "local_redistribute_save_floor"
        ]
        expected_norm = np.linalg.norm(context_to_logits.THETA_CONTEXT)
        assert expected_norm > 0.0


def test_structural_outcome_contract_serializes_the_exact_lhs_point() -> None:
    _protocol, design, _manifest = _bundle()
    values = design["points"][0]["parameters"]
    contract = expected_structural_outcome_equation_contract(
        values,
        REPO_ROOT,
        benchmark_seed=int(design["points"][0]["seed"]),
        scenario="baseline",
    )

    assert contract["arrhenius"]["base_k_ref"] == values["spoilage_k_ref"]
    assert contract["arrhenius"]["base_ea_over_r"] == values[
        "spoilage_ea_over_r"
    ]
    assert contract["waste"]["exposure_scale"] == values[
        "waste_exposure_scale"
    ]
    assert contract["waste"]["compression_exponent"] == values[
        "waste_compression_exponent"
    ]
    assert contract["waste"]["action_save_fraction"][
        "local_redistribute"
    ] == values["local_redistribute_save_floor"]
    assert contract["carbon"]["refrigeration_cop_penalty"] == values[
        "refrigeration_cop_penalty"
    ]
    assert contract["carbon"]["route_km_by_action"] == {
        "cold_chain": values["km_coldchain"],
        "local_redistribute": values["km_local"],
        "recovery": values["km_recovery"],
    }
    assert contract["slca"]["carbon_cap"] == values["slca_carbon_cap"]
    assert contract["reward"]["waste_penalty"] == values[
        "waste_reward_penalty"
    ]
    assert contract["stochastic_effective_parameter_provenance"][
        "k_ref_fraction_std"
    ] == pytest.approx(0.20 * values["stochastic_noise_scale"])


def _synthetic_payloads(design, manifest, source_commit: str):
    scenario_order = list(dict.fromkeys(
        task["scenario"] for task in manifest["tasks"] if task["panel"] == "primary"
    ))
    mode_base = {
        "static": 0.42,
        "hybrid_rl": 0.47,
        "no_pinn": 0.56,
        "no_slca": 0.49,
        "no_context": 0.50,
        "mcp_only": 0.53,
        "pirag_only": 0.54,
        "agribrain": 0.58,
    }
    mode_slope = {
        mode: (index - 3) * 0.002 for index, mode in enumerate(PRIMARY_MODES)
    }
    payloads = {}
    for task in manifest["tasks"]:
        point = design["points"][task["point_index"]]
        z = (
            point["parameters"]["spoilage_k_ref"] - 0.0021
        ) / (0.00252 - 0.00168)
        scenario_index = scenario_order.index(task["scenario"])
        latent = f"latent:{task['point_id']}:{task['scenario']}"
        results = {}
        if task["panel"] == "primary":
            for mode in PRIMARY_MODES:
                ari = (
                    mode_base[mode] + 0.004 * scenario_index
                    + mode_slope[mode] * z
                )
                results[mode] = {
                    "ari": ari,
                    "waste": 0.12 - 0.01 * ari + 0.002 * z,
                    "rle": 0.40 + 0.20 * ari,
                    "slca": 0.60 + 0.10 * ari + 0.001 * z,
                    "carbon": 1000.0 - 100.0 * ari + 3.0 * z,
                    "equity": 0.50 + 0.10 * ari,
                    "latent_environment_sha256": latent,
                    "observed_policy_input_sha256": f"nominal:{latent}",
                }
        else:
            stress_index = STRESSORS.index(task["stressor"])
            nominal_ari = (
                mode_base["agribrain"] + 0.004 * scenario_index
                + mode_slope["agribrain"] * z
            )
            delta = 0.001 + stress_index * 0.0007 + 0.0005 * z
            observed = (
                f"nominal:{latent}"
                if task["stressor"] == "mcp_fault_injection"
                else f"stressed:{task['stressor']}:{latent}"
            )
            results["agribrain"] = {
                "ari": nominal_ari + delta,
                "waste": 0.10,
                "rle": 0.52,
                "slca": 0.66,
                "carbon": 940.0,
                "equity": 0.56,
                "latent_environment_sha256": latent,
                "observed_policy_input_sha256": observed,
                "fault_injection_trigger_steps": (
                    4 if task["stressor"] in {"mcp_fault_injection", "compounded"}
                    else 0
                ),
            }
        payload = {
            "schema_version": 1,
            "analysis_label": "structural sensitivity",
            "probability_interpretation": False,
            "source_commit": source_commit,
            "protocol_sha256": "a" * 64,
            "design_sha256": design["design_sha256"],
            "task_sha256": task["task_sha256"],
            "task_id": task["task_id"],
            "task_index": task["task_index"],
            "point_id": task["point_id"],
            "point_index": task["point_index"],
            "seed": task["seed"],
            "scenario": task["scenario"],
            "panel": task["panel"],
            "stressor": task.get("stressor"),
            "parameters_sha256": task["parameters_sha256"],
            "retained_cells": task["retained_cells"],
            "executed_episodes": task["executed_episodes"],
            "simulated_steps": task["simulated_steps"],
            "results": results,
        }
        payload["result_sha256"] = canonical_sha256(payload)
        payloads[task["task_id"]] = payload
    return payloads


def test_analysis_reports_rank_sign_h3_margin_and_associations_without_probability_claim() -> None:
    _protocol, design, manifest = _bundle()
    source_commit = "f" * 40
    report = analyze_payloads(
        design,
        manifest,
        _synthetic_payloads(design, manifest, source_commit),
        source_commit=source_commit,
    )
    assert report["analysis_label"] == "structural sensitivity"
    assert report["probability_interpretation"] is False
    assert report["n_design_points"] == 100
    assert report["n_parameters"] == len(PARAMETERS)
    assert report["rank_stability"]["pooled_scenario_mean"][
        "modal_complete_ordering"
    ][0] == "agribrain"
    assert all(
        cell["positive_sign_fraction"] == 1.0
        for cell in report["h1_sign_stability"].values()
    )
    assert report["h3_margin_stability"][
        "fraction_design_points_all_25_cells_inside_margin"
    ] == 1.0
    assert len(report["rank_associations"]["records"]) > len(PARAMETERS)
    assert len(report["analysis_sha256"]) == 64
