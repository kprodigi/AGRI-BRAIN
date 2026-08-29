"""Fail-closed assembly of the two independently verified evidence scopes."""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from mvp.simulation.validation import validate_publication_artifacts as vpa
from mvp.simulation.validation import validator_source_identity as source_identity

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "hpc" / "build_full_submission_evidence.py"
SPEC = importlib.util.spec_from_file_location("full_submission_evidence", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence)
REAL_CORE_VALIDATION = evidence._run_canonical_core_validation
REAL_SAFE_EXTRACTION = evidence._extract_safe_archive
REAL_VALIDATOR_CHECKOUT = evidence._validate_local_validator_checkout
VIRTUAL_STRUCTURAL_ARCHIVES: dict[Path, dict] = {}

CORE_RECEIPT_PATH = REPO_ROOT / "hpc" / "core_submission_receipt.py"
CORE_SPEC = importlib.util.spec_from_file_location(
    "core_submission_receipt", CORE_RECEIPT_PATH
)
assert CORE_SPEC is not None and CORE_SPEC.loader is not None
core_receipt_module = importlib.util.module_from_spec(CORE_SPEC)
CORE_SPEC.loader.exec_module(core_receipt_module)


def _json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _record(name: str, payload: bytes, *, path_key: str) -> dict:
    return {
        path_key: name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _self_hash(payload: dict, field: str) -> dict:
    payload[field] = evidence._canonical_sha256(payload)
    return payload


def _write_tar(path: Path, members: dict[str, bytes]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                for name, payload in members.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    info.mtime = 0
                    info.mode = 0o644
                    archive.addfile(info, io.BytesIO(payload))


def _core_bundle(
    root: Path,
    *,
    commit: str,
    unsafe_member: bool = False,
    omit_archive_member: str | None = None,
) -> tuple[Path, Path, Path]:
    tag = f"{commit[:7]}_20260828_120000"
    submission = core_receipt_module.build_receipt(
        run_tag=tag,
        source_commit=commit,
        partition="compute",
        seed_job_id="101",
        stress_job_id="102",
        publisher_job_id="103",
        source_snapshot_mode="detached_readonly_git_worktree_v1",
        source_tree_sha256="1" * 64,
    )
    submission_bytes = _json_bytes(submission)
    submission_name = f"core_submission_receipts/{tag}.json"
    validation_name = "publication_validation_receipt.json"
    expected_names = vpa._expected_manifest_paths(tag, include_receipt=True)
    payloads = {
        name: _json_bytes({"fixture": name}) for name in expected_names
    }
    payloads[submission_name] = submission_bytes
    records = [
        _record(name, payloads[name], path_key="file")
        for name in sorted(expected_names)
    ]
    manifest = {
        "schema_version": 2,
        "git_commit": commit,
        "simulation_source_commit": commit,
        "publication_code_commit": commit,
        "dual_provenance": False,
        "git_dirty": False,
        "hash_semantics": {
            "sha256": "SHA-256 of literal file bytes",
            "bytes": "literal file length in bytes",
        },
        "includes_raw_run_artifacts": True,
        "artifact_run_tag": tag,
        "artifact_count": len(records),
        "artifacts": records,
    }
    semantic = {
        **vpa._receipt_contract(manifest, repo_root=REPO_ROOT),
        "generated_at_utc": "2026-08-28T12:00:00+00:00",
        "validated_checks": list(evidence._VALIDATED_CORE_CHECKS),
    }
    semantic_bytes = _json_bytes(semantic)
    payloads[validation_name] = semantic_bytes
    records[sorted(expected_names).index(validation_name)] = _record(
        validation_name, semantic_bytes, path_key="file"
    )
    manifest["artifacts"] = records
    manifest_bytes = _json_bytes(manifest)
    archive_path = root / f"hpc_results_{tag}.tar.gz"
    members = {
        "artifact_manifest.json": manifest_bytes,
        **{name: payloads[name] for name in sorted(expected_names)},
    }
    if omit_archive_member is not None:
        members.pop(omit_archive_member)
    if unsafe_member:
        members["../escape.json"] = b"{}\n"
    _write_tar(archive_path, members)
    receipt = {
        "schema_version": 1,
        "generated_at_utc": "2026-08-28T12:01:00+00:00",
        "derivation_type": "fresh stochastic simulation and publication build",
        "simulation_rerun": True,
        "simulation_source_commit": commit,
        "publication_code_commit": commit,
        "run_tag": tag,
        "parent_archive_sha256": None,
        "validator_source_identity": {
            "head_commit": commit,
            "source_tree_clean_outside_exact_evidence_paths": True,
            "status_includes_untracked_files": True,
            "allowed_evidence_path_count": len(records) + 1,
            "allowed_evidence_path_set_sha256": hashlib.sha256(
                "\n".join(sorted([
                    "mvp/simulation/results/artifact_manifest.json",
                    *(
                        f"mvp/simulation/results/{record['file']}"
                        for record in records
                    ),
                ])).encode("utf-8")
            ).hexdigest(),
        },
        "archive": {
            "file": archive_path.name,
            "bytes": archive_path.stat().st_size,
            "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            "member_count": len(members),
        },
        "manifest": {
            "bytes": len(manifest_bytes),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "artifact_count": len(records),
            "payload_merkle_root": evidence._payload_merkle_root(records),
            "hash_semantics": "literal bytes",
        },
        "validation": {
            "prearchive_payload_hashes": "PASS",
            "postarchive_payload_hashes": "PASS",
            "exact_manifest_membership": "PASS",
            "safe_regular_members_only": "PASS",
            "semantic_validation_receipt_manifested_and_verified": "PASS",
            "validator_checkout_same_clean_commit_outside_exact_evidence": "PASS",
        },
        "evidence_scope": {
            "core_publication_evidence": True,
            "structural_sensitivity_included": False,
            "full_submission_requires_separate_structural_receipt": True,
        },
    }
    receipt_path = root / f"publication_archive_receipt_{tag}.json"
    receipt_path.write_bytes(_json_bytes(receipt))
    ready = {
        "schema_version": 1,
        "status": "READY",
        "archive": {
            "file": archive_path.name,
            "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        },
        "receipt": {
            "file": receipt_path.name,
            "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        },
    }
    ready_path = root / "READY.json"
    ready_path.write_bytes(_json_bytes(ready))
    return archive_path, receipt_path, ready_path


def _structural_accounting() -> dict:
    return {
        "analysis_label": "structural sensitivity",
        "probability_interpretation": False,
        "n_design_points": 100,
        "n_scenarios": 5,
        "primary_modes": [
            "static",
            "hybrid_rl",
            "no_pinn",
            "no_slca",
            "no_context",
            "mcp_only",
            "pirag_only",
            "agribrain",
        ],
        "stress_mode": "agribrain",
        "stressors": [
            "sensor_noise",
            "missing_data",
            "telemetry_delay",
            "mcp_fault_injection",
            "compounded",
        ],
        "episode_budget_by_primary_mode": {
            "static": 1,
            "hybrid_rl": 4,
            "no_pinn": 4,
            "no_slca": 4,
            "no_context": 4,
            "mcp_only": 4,
            "pirag_only": 4,
            "agribrain": 4,
        },
        "episodes_per_stressed_agribrain_cell": 4,
        "steps_per_episode": 288,
        "per_design_point": {
            "primary_retained_cells": 40,
            "primary_executed_episodes": 145,
            "h3_stressed_retained_cells": 25,
            "h3_stressed_executed_episodes": 100,
            "total_retained_cells": 65,
            "total_executed_episodes": 245,
        },
        "total": dict(evidence._STRUCTURAL_TOTALS),
    }


def _structural_bundle(
    root: Path,
    *,
    commit: str,
    n_parameters: int = 29,
    omit_task_output: bool = False,
    omit_retained_ledger: bool = False,
    omit_episode_archive: bool = False,
    omit_scheduler_accounting: bool = False,
    omit_publication_artifact: bool = False,
    misstate_failed_attempt_inventory: bool = False,
) -> tuple[Path, Path]:
    tag = f"sensitivity_{commit[:7]}_20260828_120500"
    design_sha = "2" * 64
    keys = [f"factor_{index:02d}" for index in range(28)] + ["slca_carbon_cap"]
    registry = {
        "analysis_label": "structural sensitivity",
        "probability_interpretation": False,
        "parameters": [{"key": key} for key in keys],
    }
    accounting = _structural_accounting()
    scenarios = list(evidence._SCENARIOS)
    stressors = [
        "sensor_noise",
        "missing_data",
        "telemetry_delay",
        "mcp_fault_injection",
        "compounded",
    ]
    primary_modes = [
        "static",
        "hybrid_rl",
        "no_pinn",
        "no_slca",
        "no_context",
        "mcp_only",
        "pirag_only",
        "agribrain",
    ]
    tasks = []
    task_payloads: dict[str, bytes] = {}
    ledger_payloads: dict[str, bytes] = {}
    episode_payloads: dict[str, bytes] = {}
    adaptation_payloads: dict[str, bytes] = {}
    episode_manifest_payloads: dict[str, bytes] = {}

    def add_complete_task_evidence(
        output_relative: str, *, executed_episodes: int, retained_cells: int,
    ) -> None:
        parent, filename = output_relative.rsplit("/", 1)
        artifact_root = f"{parent}/{filename.removesuffix('.json')}__artifacts"
        episode_payloads.update({
            f"{artifact_root}/complete_episode_evidence/episode_{index:02d}.json.gz":
                b"episode\n"
            for index in range(executed_episodes)
        })
        adaptation_payloads.update({
            f"{artifact_root}/adaptation_episode_ledgers/episode_{index:02d}.jsonl.gz":
                b"ledger\n"
            for index in range(executed_episodes - retained_cells)
        })
        episode_manifest_payloads[
            f"{artifact_root}/complete_episode_evidence_manifest.json"
        ] = b"{}\n"

    for point_index in range(100):
        point_id = f"lhs_{point_index:03d}"
        common = {
            "design_sha256": design_sha,
            "point_index": point_index,
            "point_id": point_id,
            "seed": evidence._SEEDS[point_index % len(evidence._SEEDS)],
            "parameters_sha256": "3" * 64,
        }
        for scenario in scenarios:
            task = {
                **common,
                "task_index": len(tasks),
                "task_id": f"{point_id}__{scenario}__primary",
                "panel": "primary",
                "scenario": scenario,
                "modes": primary_modes,
                "retained_cells": 8,
                "executed_episodes": 29,
                "simulated_steps": 8_352,
                "output_relpath": f"tasks/{point_id}/{scenario}__primary.json",
            }
            _self_hash(task, "task_sha256")
            tasks.append(task)
            task_payloads[task["output_relpath"]] = b"{}\n"
            add_complete_task_evidence(
                task["output_relpath"],
                executed_episodes=task["executed_episodes"],
                retained_cells=task["retained_cells"],
            )
            artifact_root = (
                f"tasks/{point_id}/{scenario}__primary__artifacts/"
                "runtime_artifacts/decision_ledger"
            )
            for mode in primary_modes:
                ledger_payloads[
                    f"{artifact_root}/{mode}__{scenario}.jsonl"
                ] = b"{}\n"
        for scenario in scenarios:
            for stressor in stressors:
                task = {
                    **common,
                    "task_index": len(tasks),
                    "task_id": f"{point_id}__{scenario}__h3__{stressor}",
                    "panel": "h3_stressed",
                    "scenario": scenario,
                    "stressor": stressor,
                    "modes": ["agribrain"],
                    "nominal_reference_task_id": (
                        f"{point_id}__{scenario}__primary"
                    ),
                    "retained_cells": 1,
                    "executed_episodes": 4,
                    "simulated_steps": 1_152,
                    "output_relpath": (
                        f"tasks/{point_id}/{scenario}__h3__{stressor}.json"
                    ),
                }
                _self_hash(task, "task_sha256")
                tasks.append(task)
                task_payloads[task["output_relpath"]] = b"{}\n"
                add_complete_task_evidence(
                    task["output_relpath"],
                    executed_episodes=task["executed_episodes"],
                    retained_cells=task["retained_cells"],
                )
                ledger_payloads[
                    f"tasks/{point_id}/{scenario}__h3__{stressor}__artifacts/"
                    f"decision_ledgers/{scenario}/"
                    f"structural__{point_id}__{stressor}/"
                    f"seed_{common['seed']}/agribrain__{scenario}.jsonl"
                ] = b"{}\n"
    task_manifest = {
        "schema_version": 1,
        "analysis_label": "structural sensitivity",
        "probability_interpretation": False,
        "design_sha256": design_sha,
        "n_tasks": 3_000,
        "accounting": accounting,
        "tasks": tasks,
    }
    _self_hash(task_manifest, "manifest_sha256")
    artifact_payloads = {
        "parameter_registry.json": _json_bytes(registry),
        "lhs_design.json": _json_bytes({"design_sha256": design_sha}),
        "lhs_design.csv": b"point_id\nlhs_000\n",
        "task_manifest.json": _json_bytes(task_manifest),
        "task_manifest.jsonl": b'{"task_index":0}\n',
        "episode_accounting.json": _json_bytes(accounting),
        "experiment_protocol.json": b'{"status":"locked_before_rerun"}\n',
    }
    artifacts = {
        "parameter_registry": "parameter_registry.json",
        "lhs_design": "lhs_design.json",
        "lhs_design_csv": "lhs_design.csv",
        "task_manifest": "task_manifest.json",
        "task_manifest_jsonl": "task_manifest.jsonl",
        "episode_accounting": "episode_accounting.json",
        "locked_protocol": "experiment_protocol.json",
    }
    run_plan = {
        "schema_version": 1,
        "analysis_label": "structural sensitivity",
        "probability_interpretation": False,
        "execution_scope": "structural_sensitivity_only",
        "run_tag": tag,
        "source_commit": commit,
        "source_tracked_tree_clean_at_generation": True,
        "source_tree_clean_at_generation": True,
        "development_only_dirty_plan": False,
        "artifacts": artifacts,
        "artifact_sha256": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in artifact_payloads.items()
        },
    }
    _self_hash(run_plan, "run_plan_sha256")
    run_plan_bytes = _json_bytes(run_plan)
    completion = {
        "status": "complete",
        "n_expected_tasks": 3_000,
        "n_valid_tasks": 3_000,
        "n_missing_tasks": 0,
        "missing_task_ids": [],
    }
    source_tree_sha256 = "1" * 64
    submission = {
        "schema_version": 2,
        "analysis_label": "structural sensitivity",
        "receipt_scope": "submission_only_not_scheduler_completion",
        "scheduler_completion_attested": False,
        "run_tag": tag,
        "source_commit": commit,
        "source_snapshot_mode": "detached_readonly_git_worktree_v1",
        "source_tree_sha256": source_tree_sha256,
        "task_count": 3_000,
        "array_chunk_size_limit": 1_000,
        "max_concurrent_per_array": 50,
        "task_arrays": [
            {"job_id": "201", "offset": 0, "count": 1_000, "afterok_job_id": None},
            {"job_id": "202", "offset": 1_000, "count": 1_000, "afterok_job_id": "201"},
            {"job_id": "203", "offset": 2_000, "count": 1_000, "afterok_job_id": "202"},
        ],
        "publisher": {"job_id": "204", "afterok_job_id": "203"},
    }
    _self_hash(submission, "receipt_sha256")
    scheduler_accounting = {
        "schema_version": 1,
        "status": "COMPLETE",
        "kind": "structural",
        "run_identity": {
            "run_tag": tag,
            "source_commit": commit,
            "source_tree_sha256": source_tree_sha256,
        },
        "arrays": [],
        "task_state_counts": {"COMPLETED": 3_000},
        "scheduler": {"raw_stdout": "", "raw_stdout_sha256": hashlib.sha256(b"").hexdigest()},
        "energy": {
            "site_field_exposed": False,
            "numeric_allocation_rows": 0,
            "summed_consumed_energy_raw_joules": None,
        },
        "row_count": 3_000,
        "rows": [],
    }
    _self_hash(scheduler_accounting, "accounting_sha256")
    publication_artifacts = {
        "structural_sensitivity_summary.csv": b"family,scenario\nH1,baseline\n",
        "structural_sensitivity_summary.png": b"png fixture\n",
        "structural_sensitivity_summary.pdf": b"pdf fixture\n",
    }
    analysis_payload = b"{}\n"
    publication_receipt = {
        "schema_version": 1,
        "receipt_type": "structural_sensitivity_publication_receipt",
        "probability_interpretation": False,
        "source": {
            "name": "structural_sensitivity_analysis.json",
            "bytes": len(analysis_payload),
            "literal_sha256": hashlib.sha256(analysis_payload).hexdigest(),
            "analysis_sha256": None,
            "source_commit": commit,
        },
        "row_count": 1,
        "artifacts": [
            _record(name, payload, path_key="name")
            for name, payload in sorted(publication_artifacts.items())
        ],
    }
    _self_hash(publication_receipt, "receipt_sha256")
    final_payloads = {
        "completion_status.json": _json_bytes(completion),
        "structural_sensitivity_analysis.json": analysis_payload,
        "publication_environment.json": b"{}\n",
        "slurm_submission.json": _json_bytes(submission),
        "slurm_simulation_accounting.json": _json_bytes(scheduler_accounting),
        **publication_artifacts,
        "structural_sensitivity_publication_receipt.json": _json_bytes(
            publication_receipt
        ),
    }
    runtime_payloads = {
        f"runtime_receipts/task_{index}/job_{10_000 + index}__restart_0.json":
            b"{}\n"
        for index in range(3_000)
    }
    runtime_payloads[
        "runtime_receipts/task_0/job_99999__restart_1.json"
    ] = b"{}\n"
    failed_attempt_payloads = {
        "tasks/lhs_000/baseline__primary__attempts/attempt_fixture/"
        "complete_episode_evidence/partial.json.gz": b"partial episode\n",
        "tasks/lhs_000/baseline__primary__attempts/attempt_fixture/"
        "adaptation_episode_ledgers/partial.jsonl.gz": b"partial ledger\n",
        "tasks/lhs_000/baseline__primary__attempts/attempt_fixture/"
        "_attempt_failures/failure_fixture.json": b"{}\n",
    }
    payloads = {
        "run_plan.json": run_plan_bytes,
        **artifact_payloads,
        **task_payloads,
        **ledger_payloads,
        **episode_payloads,
        **adaptation_payloads,
        **episode_manifest_payloads,
        **runtime_payloads,
        **failed_attempt_payloads,
        **final_payloads,
    }
    if omit_task_output:
        payloads.pop(next(iter(task_payloads)))
    if omit_retained_ledger:
        payloads.pop(next(iter(ledger_payloads)))
    if omit_episode_archive:
        payloads.pop(next(iter(episode_payloads)))
    if omit_scheduler_accounting:
        payloads.pop("slurm_simulation_accounting.json")
    if omit_publication_artifact:
        payloads.pop("structural_sensitivity_summary.pdf")
    records = [
        _record(name, payload, path_key="path")
        for name, payload in sorted(payloads.items())
    ]
    manifest = {
        "schema_version": 1,
        "analysis_label": "structural sensitivity",
        "probability_interpretation": False,
        "execution_scope": "structural_sensitivity_only",
        "run_tag": tag,
        "source_commit": commit,
        "source_snapshot_binding": {
            "mode": "detached_readonly_git_worktree_v1",
            "source_tree_sha256": source_tree_sha256,
        },
        "validator_source_identity": {
            "head_commit": commit,
            "source_tree_clean": True,
            "tracked_and_untracked_status_empty": True,
        },
        "design_sha256": design_sha,
        "accounting": accounting,
        "n_parameters": n_parameters,
        "n_tasks": 3_000,
        "retained_decision_ledger_count": 6_500,
        "retained_decision_ledger_set_sha256": evidence._canonical_sha256(
            sorted(
                (
                    record for record in records
                    if record["path"] in ledger_payloads
                ),
                key=lambda record: record["path"],
            )
        ),
        "complete_episode_evidence": {
            "executed_episode_archives": 24_500,
            "adaptation_episode_ledgers": 18_000,
            "final_episode_ledgers": 6_500,
            "per_task_manifests": 3_000,
            "runtime_receipts": {
                "successful_task_receipts": 3_000,
                "failed_attempt_receipts": 1,
                "total_receipts": len(runtime_payloads),
                "summed_task_wall_seconds_nonconcurrent": 3_000.0,
                "summed_child_cpu_seconds": 2_000.0,
            },
            "scheduler_accounting": {
                "array_count": 3,
                "completed_simulation_task_count": 3_000,
                "accounting_row_count": 3_000,
                "energy": scheduler_accounting["energy"],
                "accounting_sha256": scheduler_accounting["accounting_sha256"],
            },
            "failed_attempt_artifacts": {
                "file_count": len(failed_attempt_payloads) + (
                    1 if misstate_failed_attempt_inventory else 0
                ),
                "literal_bytes": sum(map(len, failed_attempt_payloads.values())),
                "retention_policy": (
                    "Retained for diagnosis and audit; excluded from canonical "
                    "episode and ledger counts."
                ),
            },
        },
        "excluded_runtime_material": (
            "temporary files, interpreter caches, and the in-progress publisher "
            "log only; every durable task artifact, episode archive, adaptation "
            "ledger, final ledger, worker runtime receipt, and completed task log "
            "is included"
        ),
        "file_count": len(records),
        "files": records,
    }
    _self_hash(manifest, "manifest_sha256")
    manifest_bytes = _json_bytes(manifest)
    archive_path = root / f"structural_sensitivity_evidence_{tag}.tar.gz"
    prefix = "structural_sensitivity_evidence"
    members = {
        f"{prefix}/structural_sensitivity_artifact_manifest.json": manifest_bytes,
        **{f"{prefix}/{name}": payload for name, payload in payloads.items()},
    }
    # Inventory-focused tests use an in-memory structural archive view.  This
    # keeps the 54,000-plus-member contract tests fast; production tar safety is
    # covered by the finalizer round-trip test and core unsafe-member case.
    archive_path.write_bytes(b"virtual structural evidence fixture\n")
    VIRTUAL_STRUCTURAL_ARCHIVES[archive_path.resolve()] = {
        "manifest_bytes": manifest_bytes,
        "metadata": {
            name: {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for name, payload in members.items()
        },
        "json_members": {
            name: json.loads(payload.decode("utf-8"))
            for name, payload in members.items()
            if name.endswith(".json")
        },
    }
    receipt = {
        "schema_version": 1,
        "receipt_type": "structural_sensitivity_semantic_archive_receipt",
        "analysis_label": "structural sensitivity",
        "validation_status": "PASS",
        "run_tag": tag,
        "source_commit": commit,
        "source_snapshot_binding": manifest["source_snapshot_binding"],
        "validator_source_identity": manifest["validator_source_identity"],
        "publisher_execution": {
            "slurm_job_id": "204",
            "declared_publisher_job_id": "204",
            "identity_match": True,
        },
        "archive": {
            "name": archive_path.name,
            "bytes": archive_path.stat().st_size,
            "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        },
        "artifact_manifest": {
            "name": "structural_sensitivity_artifact_manifest.json",
            "bytes": len(manifest_bytes),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "content_sha256": manifest["manifest_sha256"],
        },
        "archive_member_count": len(members),
        "locked_accounting": {
            "active_factors": 29,
            "latin_hypercube_points": 100,
            "tasks": 3_000,
            "retained_cells": 6_500,
            "retained_decision_ledgers": 6_500,
            "retained_complete_episode_archives": 24_500,
            "retained_adaptation_episode_ledgers": 18_000,
            "successful_worker_runtime_receipts": 3_000,
            "scheduler_accounted_simulation_tasks": 3_000,
            "executed_episodes": 24_500,
            "simulated_steps": 7_056_000,
            "probability_interpretation": False,
        },
        "validation": {
            "complete_task_result_hashes": "PASS",
            "saved_status_matches_fresh_validation": "PASS",
            "saved_analysis_matches_fresh_recomputation": "PASS",
            "run_scoped_environment": "PASS",
            "contiguous_afterok_submission_dag": "PASS",
            "all_task_results_bound_to_slurm_arrays_and_source_snapshot": "PASS",
            "declared_publisher_runtime_identity": "PASS",
            "clean_fixed_validator_source_pre_and_post_archive": "PASS",
            "exact_manifest_membership_and_literal_hashes": "PASS",
            "retained_ledger_equations_and_endpoint_recomputation": "PASS",
            "complete_adaptation_and_evaluation_episode_archives": "PASS",
            "whole_worker_runtime_resource_receipts": "PASS",
            "post_job_scheduler_accounting_for_all_simulation_tasks": "PASS",
            "structural_table_figure_export_bound_to_analysis": "PASS",
        },
        "evidence_scope": {
            "structural_sensitivity_evidence": True,
            "core_publication_evidence_included": False,
            "full_submission_requires_core_receipt": True,
        },
    }
    _self_hash(receipt, "receipt_sha256")
    receipt_path = root / "structural_sensitivity_archive_receipt.json"
    receipt_path.write_bytes(_json_bytes(receipt))
    return archive_path, receipt_path


def _assemble(tmp_path: Path, *, core_commit: str, structural_commit: str):
    core_archive, core_receipt, core_ready = _core_bundle(
        tmp_path, commit=core_commit
    )
    structural_archive, structural_receipt = _structural_bundle(
        tmp_path, commit=structural_commit
    )
    output = tmp_path / "full_submission_evidence_receipt.json"
    payload = evidence.assemble_full_submission_evidence(
        core_archive=core_archive,
        core_receipt=core_receipt,
        core_ready=core_ready,
        structural_archive=structural_archive,
        structural_receipt=structural_receipt,
        output=output,
    )
    return payload, output, (
        core_archive,
        core_receipt,
        core_ready,
        structural_archive,
        structural_receipt,
    )


@pytest.fixture(autouse=True)
def _stub_full_structural_recomputation(monkeypatch: pytest.MonkeyPatch):
    """Tiny fixtures test inventory; production reruns all 3,000 validators."""

    VIRTUAL_STRUCTURAL_ARCHIVES.clear()
    real_extract = evidence._extract_safe_archive
    real_read_archive = evidence._read_archive
    real_json_member = evidence._json_member

    def fixture_read_archive(archive, *, required_manifest_name):
        virtual = VIRTUAL_STRUCTURAL_ARCHIVES.get(Path(archive).resolve())
        if virtual is None:
            return real_read_archive(
                archive, required_manifest_name=required_manifest_name
            )
        if required_manifest_name not in virtual["metadata"]:
            raise ValueError("archive lacks required manifest member")
        return virtual["manifest_bytes"], virtual["metadata"]

    def fixture_json_member(archive, member_name):
        virtual = VIRTUAL_STRUCTURAL_ARCHIVES.get(Path(archive).resolve())
        if virtual is None:
            return real_json_member(archive, member_name)
        try:
            return virtual["json_members"][member_name]
        except KeyError as exc:
            raise ValueError(f"invalid archived JSON: {member_name}") from exc

    def fixture_extract(archive, destination, metadata, **kwargs):
        if metadata and "artifact_manifest.json" in metadata:
            destination.mkdir(parents=True, exist_ok=True)
            return None
        if metadata and all(
            name.startswith("structural_sensitivity_evidence/")
            for name in metadata
        ):
            (destination / "structural_sensitivity_evidence").mkdir(
                parents=True, exist_ok=True
            )
            return None
        return real_extract(archive, destination, metadata, **kwargs)

    monkeypatch.setattr(evidence, "_extract_safe_archive", fixture_extract)
    monkeypatch.setattr(evidence, "_read_archive", fixture_read_archive)
    monkeypatch.setattr(evidence, "_json_member", fixture_json_member)
    monkeypatch.setattr(
        evidence,
        "_run_canonical_core_validation",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        evidence,
        "_run_canonical_structural_validation",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        evidence,
        "_validate_local_validator_checkout",
        lambda expected_commit: {
            "head_commit": expected_commit,
            "source_tree_clean": True,
            "tracked_and_untracked_status_empty": True,
        },
    )
    monkeypatch.setattr(
        evidence, "tracked_source_digest", lambda _root: ("1" * 64, 500),
    )
    yield
    VIRTUAL_STRUCTURAL_ARCHIVES.clear()


def test_assembler_binds_both_complete_scopes_and_self_hashes(tmp_path: Path) -> None:
    commit = "a" * 40
    payload, output, _inputs = _assemble(
        tmp_path, core_commit=commit, structural_commit=commit
    )
    assert output.is_file()
    assert payload["source_commit"] == commit
    assert payload["evidence_scope"] == {
        "core_publication_evidence_present": True,
        "structural_sensitivity_evidence_present": True,
        "full_submission_evidence_present": True,
        "missing_required_scopes": [],
    }
    unsigned = dict(payload)
    digest = unsigned.pop("receipt_sha256")
    assert digest == evidence._canonical_sha256(unsigned)
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored == payload
    structural = payload["structural_sensitivity_evidence"]
    assert structural["complete_episode_evidence"]["runtime_receipts"] == {
        "successful_task_receipts": 3_000,
        "failed_attempt_receipts": 1,
        "total_receipts": 3_001,
        "summed_task_wall_seconds_nonconcurrent": 3_000.0,
        "summed_child_cpu_seconds": 2_000.0,
    }
    assert structural["complete_episode_evidence"][
        "failed_attempt_artifacts"
    ]["file_count"] == 3
    assert structural["complete_episode_evidence"]["scheduler_accounting"][
        "completed_simulation_task_count"
    ] == 3_000
    assert len(structural["structural_publication_artifacts"]) == 4


def test_assembler_rejects_mismatched_source_commits(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="different source commits"):
        _assemble(
            tmp_path,
            core_commit="a" * 40,
            structural_commit="b" * 40,
        )


def test_assembler_rejects_incomplete_structural_factor_panel(tmp_path: Path) -> None:
    commit = "c" * 40
    core_archive, core_receipt, core_ready = _core_bundle(tmp_path, commit=commit)
    structural_archive, structural_receipt = _structural_bundle(
        tmp_path, commit=commit, n_parameters=28
    )
    with pytest.raises(ValueError, match="29 active factors"):
        evidence.assemble_full_submission_evidence(
            core_archive=core_archive,
            core_receipt=core_receipt,
            core_ready=core_ready,
            structural_archive=structural_archive,
            structural_receipt=structural_receipt,
            output=tmp_path / "full.json",
        )


def test_assembler_rejects_one_missing_structural_task_output(tmp_path: Path) -> None:
    commit = "9" * 40
    core_archive, core_receipt, core_ready = _core_bundle(tmp_path, commit=commit)
    structural_archive, structural_receipt = _structural_bundle(
        tmp_path, commit=commit, omit_task_output=True
    )
    with pytest.raises(ValueError, match="exact 3,000 task outputs"):
        evidence.assemble_full_submission_evidence(
            core_archive=core_archive,
            core_receipt=core_receipt,
            core_ready=core_ready,
            structural_archive=structural_archive,
            structural_receipt=structural_receipt,
            output=tmp_path / "full.json",
        )


def test_assembler_rejects_one_missing_structural_retained_ledger(
    tmp_path: Path,
) -> None:
    commit = "5" * 40
    core_archive, core_receipt, core_ready = _core_bundle(tmp_path, commit=commit)
    structural_archive, structural_receipt = _structural_bundle(
        tmp_path, commit=commit, omit_retained_ledger=True
    )
    with pytest.raises(ValueError, match="6,500 retained decision ledgers"):
        evidence.assemble_full_submission_evidence(
            core_archive=core_archive,
            core_receipt=core_receipt,
            core_ready=core_ready,
            structural_archive=structural_archive,
            structural_receipt=structural_receipt,
            output=tmp_path / "full.json",
        )


def test_assembler_rejects_incomplete_structural_episode_archive(
    tmp_path: Path,
) -> None:
    commit = "3" * 40
    core_archive, core_receipt, core_ready = _core_bundle(tmp_path, commit=commit)
    structural_archive, structural_receipt = _structural_bundle(
        tmp_path, commit=commit, omit_episode_archive=True
    )
    with pytest.raises(ValueError, match="complete 24,500-episode evidence"):
        evidence.assemble_full_submission_evidence(
            core_archive=core_archive,
            core_receipt=core_receipt,
            core_ready=core_ready,
            structural_archive=structural_archive,
            structural_receipt=structural_receipt,
            output=tmp_path / "full.json",
        )


@pytest.mark.parametrize(
    "omission",
    ["scheduler", "publication"],
)
def test_assembler_requires_structural_scheduler_and_publication_artifacts(
    tmp_path: Path, omission: str,
) -> None:
    commit = "2" * 40
    core_archive, core_receipt, core_ready = _core_bundle(tmp_path, commit=commit)
    structural_archive, structural_receipt = _structural_bundle(
        tmp_path,
        commit=commit,
        omit_scheduler_accounting=omission == "scheduler",
        omit_publication_artifact=omission == "publication",
    )
    with pytest.raises(ValueError, match="required canonical evidence files"):
        evidence.assemble_full_submission_evidence(
            core_archive=core_archive,
            core_receipt=core_receipt,
            core_ready=core_ready,
            structural_archive=structural_archive,
            structural_receipt=structural_receipt,
            output=tmp_path / "full.json",
        )


def test_assembler_binds_failed_attempt_inventory_exactly(tmp_path: Path) -> None:
    commit = "1" * 40
    core_archive, core_receipt, core_ready = _core_bundle(tmp_path, commit=commit)
    structural_archive, structural_receipt = _structural_bundle(
        tmp_path, commit=commit, misstate_failed_attempt_inventory=True
    )
    with pytest.raises(ValueError, match="failed-attempt evidence"):
        evidence.assemble_full_submission_evidence(
            core_archive=core_archive,
            core_receipt=core_receipt,
            core_ready=core_ready,
            structural_archive=structural_archive,
            structural_receipt=structural_receipt,
            output=tmp_path / "full.json",
        )


def test_assembler_rejects_unsafe_archive_member(tmp_path: Path) -> None:
    commit = "d" * 40
    core_archive, core_receipt, core_ready = _core_bundle(
        tmp_path, commit=commit, unsafe_member=True
    )
    structural_archive, structural_receipt = _structural_bundle(
        tmp_path, commit=commit
    )
    with pytest.raises(ValueError, match="unsafe archive member"):
        evidence.assemble_full_submission_evidence(
            core_archive=core_archive,
            core_receipt=core_receipt,
            core_ready=core_ready,
            structural_archive=structural_archive,
            structural_receipt=structural_receipt,
            output=tmp_path / "full.json",
        )


def test_assembler_rejects_missing_core_h3_ledger(tmp_path: Path) -> None:
    commit = "8" * 40
    missing = (
        f"decision_ledger_h3/{commit[:7]}_20260828_120000/heatwave/"
        "sensor_noise/seed_42/agribrain__heatwave.jsonl"
    )
    core_archive, core_receipt, core_ready = _core_bundle(
        tmp_path, commit=commit, omit_archive_member=missing
    )
    structural_archive, structural_receipt = _structural_bundle(
        tmp_path, commit=commit
    )
    with pytest.raises(ValueError, match="archive membership differs"):
        evidence.assemble_full_submission_evidence(
            core_archive=core_archive,
            core_receipt=core_receipt,
            core_ready=core_ready,
            structural_archive=structural_archive,
            structural_receipt=structural_receipt,
            output=tmp_path / "full.json",
        )


def test_assembler_requires_atomic_core_ready_marker(tmp_path: Path) -> None:
    commit = "6" * 40
    core_archive, core_receipt, core_ready = _core_bundle(
        tmp_path, commit=commit
    )
    ready = json.loads(core_ready.read_text(encoding="utf-8"))
    ready["status"] = "INCOMPLETE"
    core_ready.write_bytes(_json_bytes(ready))
    with pytest.raises(ValueError, match="READY marker"):
        evidence.assemble_full_submission_evidence(
            core_archive=core_archive,
            core_receipt=core_receipt,
            core_ready=core_ready,
            structural_archive=tmp_path / "not-reached.tar.gz",
            structural_receipt=tmp_path / "not-reached.json",
            output=tmp_path / "full.json",
        )


def test_assembler_reruns_core_semantics_not_just_receipt_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "7" * 40
    core_archive, core_receipt, core_ready = _core_bundle(tmp_path, commit=commit)
    structural_archive, structural_receipt = _structural_bundle(
        tmp_path, commit=commit
    )
    monkeypatch.setattr(
        evidence, "_run_canonical_core_validation", REAL_CORE_VALIDATION
    )
    monkeypatch.setattr(
        evidence, "_extract_safe_archive", REAL_SAFE_EXTRACTION
    )
    with pytest.raises(ValueError, match="canonical full core evidence"):
        evidence.assemble_full_submission_evidence(
            core_archive=core_archive,
            core_receipt=core_receipt,
            core_ready=core_ready,
            structural_archive=structural_archive,
            structural_receipt=structural_receipt,
            output=tmp_path / "full.json",
        )


def test_assembler_refuses_to_overwrite_final_receipt(tmp_path: Path) -> None:
    commit = "e" * 40
    payload, output, inputs = _assemble(
        tmp_path, core_commit=commit, structural_commit=commit
    )
    assert payload["receipt_type"] == "full_submission_evidence_set"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        evidence.assemble_full_submission_evidence(
            core_archive=inputs[0],
            core_receipt=inputs[1],
            core_ready=inputs[2],
            structural_archive=inputs[3],
            structural_receipt=inputs[4],
            output=output,
        )


def test_atomic_receipt_install_never_exposes_partial_final_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "full_submission_evidence_receipt.json"

    def fail_install(_source: Path, _target: Path) -> None:
        raise OSError("simulated atomic-install failure")

    monkeypatch.setattr(evidence.os, "link", fail_install)
    with pytest.raises(OSError, match="atomic-install failure"):
        evidence._write_new_file_atomically(output, b'{"complete":true}\n')
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_assembler_rejects_dirty_or_different_validator_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "4" * 40
    core_archive, core_receipt, core_ready = _core_bundle(tmp_path, commit=commit)
    monkeypatch.setattr(
        evidence,
        "_validate_local_validator_checkout",
        lambda _expected: (_ for _ in ()).throw(
            ValueError("validator checkout is not clean")
        ),
    )
    with pytest.raises(ValueError, match="validator checkout is not clean"):
        evidence.assemble_full_submission_evidence(
            core_archive=core_archive,
            core_receipt=core_receipt,
            core_ready=core_ready,
            structural_archive=tmp_path / "not-reached.tar.gz",
            structural_receipt=tmp_path / "not-reached.json",
            output=tmp_path / "full.json",
        )


def test_assembler_rechecks_validator_immediately_before_receipt_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "4" * 40
    core_archive, core_receipt, core_ready = _core_bundle(tmp_path, commit=commit)
    structural_archive, structural_receipt = _structural_bundle(
        tmp_path, commit=commit
    )
    calls = 0

    def clean_then_changed(expected_commit):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("validator source changed during assembly")
        return {
            "head_commit": expected_commit,
            "source_tree_clean": True,
            "tracked_and_untracked_status_empty": True,
        }

    monkeypatch.setattr(
        evidence, "_validate_local_validator_checkout", clean_then_changed
    )
    output = tmp_path / "full.json"
    with pytest.raises(ValueError, match="source changed during assembly"):
        evidence.assemble_full_submission_evidence(
            core_archive=core_archive,
            core_receipt=core_receipt,
            core_ready=core_ready,
            structural_archive=structural_archive,
            structural_receipt=structural_receipt,
            output=output,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("head", "status", "message"),
    [
        ("b" * 40, "", "HEAD differs"),
        (
            "a" * 40,
            " M hpc/build_full_submission_evidence.py\0",
            "changes outside",
        ),
    ],
)
def test_validator_checkout_gate_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    head: str,
    status: str,
    message: str,
) -> None:
    class Completed:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    def fake_run(arguments, **_kwargs):
        if arguments[1:] == ["rev-parse", "--is-inside-work-tree"]:
            return Completed("true")
        if arguments[1:] == ["rev-parse", "--show-toplevel"]:
            return Completed(str(tmp_path))
        if arguments[1:] == ["rev-parse", "HEAD"]:
            return Completed(head)
        if arguments[1:] == [
            "status", "--porcelain=v1", "-z", "--untracked-files=all",
        ]:
            return Completed(status)
        raise AssertionError(arguments)

    monkeypatch.setattr(source_identity.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match=message):
        REAL_VALIDATOR_CHECKOUT("a" * 40, repo_root=tmp_path)


def test_core_submission_receipt_pins_exact_arrays_and_afterok_dag() -> None:
    commit = "f" * 40
    tag = f"{commit[:7]}_20260828_121000"
    payload = core_receipt_module.build_receipt(
        run_tag=tag,
        source_commit=commit,
        partition="compute",
        seed_job_id="201",
        stress_job_id="202",
        publisher_job_id="203",
        source_snapshot_mode="detached_readonly_git_worktree_v1",
        source_tree_sha256="2" * 64,
    )
    assert core_receipt_module.validate_receipt_payload(payload) == payload
    assert payload["slurm_dag"]["seed_array"]["task_count"] == 20
    assert payload["slurm_dag"]["stress_array"]["task_count"] == 5
    assert payload["slurm_dag"]["publisher"]["afterok_job_ids"] == [
        "201", "202"
    ]
    assert payload["receipt_scope"] == "submission_only_not_scheduler_completion"
    assert payload["scheduler_completion_attested"] is False
    assert payload["source_snapshot_mode"] == "detached_readonly_git_worktree_v1"
    assert payload["source_tree_sha256"] == "2" * 64

    payload["slurm_dag"]["publisher"]["afterok_job_ids"] = ["202"]
    payload.pop("receipt_sha256")
    _self_hash(payload, "receipt_sha256")
    with pytest.raises(ValueError, match="publisher DAG stage or dependency"):
        core_receipt_module.validate_receipt_payload(payload)


def test_core_orchestrator_creates_run_scoped_dag_receipt() -> None:
    source = (REPO_ROOT / "hpc" / "hpc_run.sh").read_text(encoding="utf-8")
    assert "core_submission_receipts/${RUN_TAG}.json" in source
    assert "core_submission_receipt.py create" in source
    assert "core_submission_receipt.py validate" in source
    assert '--seed-job-id "$SEED_JOB"' in source
    assert '--stress-job-id "$STRESS_JOB"' in source
    assert '--publisher-job-id "$AGG_JOB"' in source
    assert '--source-snapshot-mode "$AGRIBRAIN_SOURCE_SNAPSHOT_MODE"' in source
    assert '--source-tree-sha256 "$AGRIBRAIN_SOURCE_TREE_SHA256"' in source
    assert 'SEED_JOB="${SEED_SUBMISSION%%;*}"' in source
    assert 'STRESS_JOB="${STRESS_SUBMISSION%%;*}"' in source
    assert 'AGG_JOB="${AGG_SUBMISSION%%;*}"' in source
    assert "--dependency=afterok:${SEED_JOB}" in source
    assert "publication_bundle_${RUN_TAG}/" in source
    assert "complete_run_evidence/${RUN_TAG}/" in source


def test_full_evidence_cli_bootstraps_repo_without_pythonpath(
    tmp_path: Path,
) -> None:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(MODULE_PATH), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--core-ready" in completed.stdout
