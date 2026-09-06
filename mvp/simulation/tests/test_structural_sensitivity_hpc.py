"""Fail-closed Slurm and packaging boundaries for structural sensitivity."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mvp.simulation.sensitivity import finalize_structural_sensitivity as finalizer
from mvp.simulation.sensitivity import run_structural_sensitivity as runner

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_hpc_module(filename: str, module_name: str):
    path = REPO_ROOT / "hpc" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_slurm_orchestrator_is_external_exact_and_fail_closed() -> None:
    orchestrator = (REPO_ROOT / "hpc" / "hpc_sensitivity_run.sh").read_text(
        encoding="utf-8"
    )
    task = (REPO_ROOT / "hpc" / "hpc_sensitivity_task.sh").read_text(
        encoding="utf-8"
    )
    publisher = (REPO_ROOT / "hpc" / "hpc_sensitivity_publish.sh").read_text(
        encoding="utf-8"
    )

    assert "AGRIBRAIN_SENSITIVITY_ROOT" in orchestrator
    assert 'PUBLICATION_PYTHON_BIN="${AGRIBRAIN_PYTHON_BIN:-python3.11}"' in orchestrator
    assert '"$PUBLICATION_PYTHON_BIN" -m venv "$AGRIBRAIN_VENV"' in orchestrator
    assert orchestrator.index(
        '"$PUBLICATION_PYTHON_BIN" hpc/validate_source_checkout.py'
    ) < orchestrator.index("git worktree add")
    assert 'export RUN_TAG="sensitivity_${GIT_COMMIT:0:7}_' in orchestrator
    assert 'export AGRIBRAIN_VENV=".publication_venvs/${RUN_TAG}"' in orchestrator
    assert 'TASK_COUNT" != "3000"' in orchestrator
    assert "AGRIBRAIN_SENSITIVITY_ARRAY_CHUNK_SIZE:-1000" in orchestrator
    assert "AGRIBRAIN_SENSITIVITY_MAX_CONCURRENT:-50" in orchestrator
    assert '--dependency="afterok:${PREVIOUS_JOB}"' in orchestrator
    assert "hpc/hpc_sensitivity_task.sh" in orchestrator
    assert "hpc/hpc_sensitivity_publish.sh" in orchestrator
    assert "--run-tag \"$RUN_TAG\"" in orchestrator
    assert "--allow-dirty" not in orchestrator
    assert "--skip-dynamic-audit" not in orchestrator
    assert 'RESULTS_DIR="mvp/simulation/results"' not in orchestrator
    fresh_gate_start = orchestrator.index(
        "# A fresh structural DAG must not inherit publication-recovery"
    )
    fresh_gate_end = orchestrator.index(
        "source hpc/publication_env.sh", fresh_gate_start,
    )
    fresh_gate = orchestrator[fresh_gate_start:fresh_gate_end]
    for variable in (
        "AGRIBRAIN_RECOVERY_RECEIPT",
        "AGRIBRAIN_RECOVERY_ATTEMPT_ROOT",
        "AGRIBRAIN_SIMULATION_COMMIT",
        "AGRIBRAIN_PUBLICATION_CODE_COMMIT",
        "AGRIBRAIN_SIMULATION_SOURCE_TREE_SHA256",
        "AGRIBRAIN_PUBLICATION_SOURCE_TREE_SHA256",
        "AGRIBRAIN_PRESERVED_RAW_MANIFEST",
        "AGRIBRAIN_RAW_SEEDS_DIR",
        "AGRIBRAIN_RAW_STRESS_DIR",
        "AGRIBRAIN_RAW_H3_LEDGER_DIR",
        "AGRIBRAIN_EXTERNAL_RECOVERY_RECEIPT",
        "AGRIBRAIN_EXTERNAL_RAW_MANIFEST",
        "AGRIBRAIN_ORIGINAL_CORE_RECEIPT",
    ):
        assert variable in fresh_gate

    assert "TASK_INDEX=$((SENSITIVITY_TASK_OFFSET + SLURM_ARRAY_TASK_ID))" in task
    assert '"$TASK_INDEX" -ge 3000' in task
    assert "validate_source_checkout.py" in task
    assert "validate_structural_sensitivity_hpc.py" in task
    assert "--resume" in task
    assert 'RESULTS_DIR="mvp/simulation/results"' not in task

    for gate in (" status ", " analyze ", "finalize_structural_sensitivity"):
        assert gate in publisher
    assert "capture_publication_environment.py --output" in publisher
    assert "capture_slurm_accounting.py" in publisher
    assert "publish_structural_sensitivity" in publisher
    assert "validate_source_checkout.py" in publisher
    assert 'RESULTS_DIR="mvp/simulation/results"' not in publisher
    assert "capture_publication_environment.py\n" not in publisher

    operator_guide = (REPO_ROOT / "HOW_TO_RUN.md").read_text(encoding="utf-8")
    assert "24,500 lossless" in operator_guide
    assert "18,000 adaptation ledgers" in operator_guide
    assert "6,500 final-evaluation" in operator_guide
    assert "structural_sensitivity_publication_receipt.json" in operator_guide
    assert "Failed-attempt artifacts are retained" in operator_guide
    assert "not an 800-episode design" in operator_guide


def test_submission_receipt_heredoc_executes_with_all_runtime_imports(
    tmp_path: Path,
) -> None:
    orchestrator = (REPO_ROOT / "hpc" / "hpc_sensitivity_run.sh").read_text(
        encoding="utf-8"
    )
    receipt_section = orchestrator.split(
        'SUBMISSION_RECEIPT="${SENSITIVITY_RUN_DIR}/slurm_submission.json"', 1
    )[1]
    receipt_program = receipt_section.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    receipt_path = tmp_path / "slurm_submission.json"
    commit = "a" * 40
    tag = f"sensitivity_{commit[:7]}_20260829_120000"
    completed = subprocess.run(
        [
            sys.executable,
            "-",
            str(receipt_path),
            tag,
            commit,
            "3000",
            "1000",
            "50",
            "104",
            "103",
            "101 102 103",
            "0 1000 2000",
            "1000 1000 1000",
            "none 101 102",
            "detached_readonly_git_worktree_v1",
            "b" * 64,
        ],
        input=receipt_program,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    claimed = receipt.pop("receipt_sha256")
    assert claimed == finalizer.canonical_sha256(receipt)
    assert receipt["task_count"] == 3_000
    assert receipt["publisher"]["afterok_job_id"] == "103"


def _valid_environment(repo: Path, external: Path) -> dict[str, str]:
    commit = "a" * 40
    tag = f"sensitivity_{commit[:7]}_20260828_120000"
    run_dir = external / tag
    run_dir.mkdir(parents=True)
    return {
        "AGRIBRAIN_SENSITIVITY_SOURCE_COMMIT": commit,
        "AGRIBRAIN_GIT_COMMIT": commit,
        "RUN_TAG": tag,
        "AGRIBRAIN_SENSITIVITY_ROOT": str(external.resolve()),
        "SENSITIVITY_RUN_DIR": str(run_dir.resolve()),
        "SENSITIVITY_RUN_PLAN": str((run_dir / "run_plan.json").resolve()),
        "AGRIBRAIN_VENV": f".publication_venvs/{tag}",
    }


def test_hpc_identity_validator_accepts_only_external_exact_run_path(
    tmp_path: Path,
) -> None:
    validator = _load_hpc_module(
        "validate_structural_sensitivity_hpc.py", "structural_hpc_validator_ok"
    )
    repo = tmp_path / "repo"
    external = tmp_path / "external"
    repo.mkdir()
    external.mkdir()
    env = _valid_environment(repo, external)
    assert validator.validation_errors(
        env, repo_root=repo, require_plan=False
    ) == []


def test_hpc_identity_validator_rejects_repo_output_and_mismatched_identity(
    tmp_path: Path,
) -> None:
    validator = _load_hpc_module(
        "validate_structural_sensitivity_hpc.py", "structural_hpc_validator_bad"
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    internal = repo / "structural"
    internal.mkdir()
    env = _valid_environment(repo, internal)
    env["AGRIBRAIN_GIT_COMMIT"] = "b" * 40
    errors = validator.validation_errors(env, repo_root=repo, require_plan=False)
    assert any("must equal" in error for error in errors)
    assert any("outside the repository" in error for error in errors)


def test_generated_plan_binds_run_tag_and_carries_portable_protocol_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "c" * 40
    monkeypatch.setattr(
        runner,
        "_git_state",
        lambda _root: {
            "source_commit": commit,
            "tracked_tree_clean": True,
            "tracked_status": [],
            "source_tree_clean": True,
            "source_status": [],
        },
    )
    monkeypatch.setattr(
        runner, "validate_parameter_registry", lambda _root: {"status": "pass"}
    )
    monkeypatch.setattr(
        runner, "validate_dynamic_influence", lambda _root: {"status": "pass"}
    )
    tag = f"sensitivity_{commit[:7]}_20260828_120000"
    plan_path = runner.generate_run_plan(
        tmp_path / tag,
        REPO_ROOT / "mvp" / "simulation" / "experiment_protocol.json",
        run_tag=tag,
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["run_tag"] == tag
    assert plan["execution_scope"] == "structural_sensitivity_only"
    assert plan["source_tree_clean_at_generation"] is True
    assert plan["protocol"]["path"] == "experiment_protocol.json"
    assert not Path(plan["protocol"]["path"]).is_absolute()
    assert (plan_path.parent / plan["protocol"]["path"]).read_bytes() == (
        REPO_ROOT / "mvp" / "simulation" / "experiment_protocol.json"
    ).read_bytes()
    loaded_plan, _protocol, _design, manifest = runner._load_plan_bundle(plan_path)
    assert loaded_plan == plan
    assert manifest["n_tasks"] == 3_000
    validator = _load_hpc_module(
        "validate_structural_sensitivity_hpc.py", "structural_hpc_validator_plan"
    )
    env = {
        "AGRIBRAIN_SENSITIVITY_SOURCE_COMMIT": commit,
        "AGRIBRAIN_GIT_COMMIT": commit,
        "RUN_TAG": tag,
        "AGRIBRAIN_SENSITIVITY_ROOT": str(tmp_path.resolve()),
        "SENSITIVITY_RUN_DIR": str(plan_path.parent.resolve()),
        "SENSITIVITY_RUN_PLAN": str(plan_path.resolve()),
        "AGRIBRAIN_VENV": f".publication_venvs/{tag}",
    }
    assert validator.validation_errors(
        env, repo_root=REPO_ROOT, require_plan=True
    ) == []
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "hpc" / "validate_structural_sensitivity_hpc.py"),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_submission_receipt_requires_contiguous_afterok_chain() -> None:
    commit = "d" * 40
    tag = f"sensitivity_{commit[:7]}_20260828_120000"
    receipt = {
        "schema_version": 2,
        "analysis_label": "structural sensitivity",
        "receipt_scope": "submission_only_not_scheduler_completion",
        "scheduler_completion_attested": False,
        "run_tag": tag,
        "source_commit": commit,
        "source_snapshot_mode": "detached_readonly_git_worktree_v1",
        "source_tree_sha256": "e" * 64,
        "task_count": 3_000,
        "task_arrays": [
            {"job_id": "101", "offset": 0, "count": 1_000, "afterok_job_id": None},
            {"job_id": "102", "offset": 1_000, "count": 1_000, "afterok_job_id": "101"},
            {"job_id": "103", "offset": 2_000, "count": 1_000, "afterok_job_id": "102"},
        ],
        "publisher": {"job_id": "104", "afterok_job_id": "103"},
    }
    receipt["receipt_sha256"] = finalizer.canonical_sha256(receipt)
    finalizer._validate_submission(receipt, run_tag=tag, source_commit=commit)
    finalizer._validate_submission(
        receipt,
        run_tag=tag,
        source_commit=commit,
        publisher_slurm_job_id="104",
    )
    with pytest.raises(ValueError, match="publisher SLURM_JOB_ID"):
        finalizer._validate_submission(
            receipt,
            run_tag=tag,
            source_commit=commit,
            publisher_slurm_job_id="999",
        )
    receipt["task_arrays"][2]["afterok_job_id"] = "101"
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = finalizer.canonical_sha256(receipt)
    with pytest.raises(ValueError, match="afterok chain"):
        finalizer._validate_submission(receipt, run_tag=tag, source_commit=commit)


def test_standalone_finalizer_rejects_dirty_fixed_validator_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "e" * 40
    tag = f"sensitivity_{commit[:7]}_20260828_120000"
    run_root = tmp_path / "run"
    run_root.mkdir()
    plan_path = run_root / "run_plan.json"
    monkeypatch.setattr(
        finalizer,
        "_load_plan_bundle",
        lambda _path: ({
            "run_tag": tag,
            "source_commit": commit,
            "execution_scope": "structural_sensitivity_only",
            "source_tree_clean_at_generation": True,
        }, {}, {}, {}),
    )
    observed_roots: list[Path] = []

    def reject_dirty(_commit: str, *, repo_root: Path, **_kwargs):
        observed_roots.append(repo_root)
        raise ValueError("validator checkout has changes")

    monkeypatch.setattr(
        finalizer, "validate_clean_validator_checkout", reject_dirty,
    )
    with pytest.raises(ValueError, match="validator checkout has changes"):
        finalizer.finalize_run(
            plan_path,
            status_path=run_root / "completion_status.json",
            analysis_path=run_root / "structural_sensitivity_analysis.json",
            environment_path=run_root / "publication_environment.json",
            scheduler_accounting_path=run_root / "slurm_simulation_accounting.json",
            manifest_path=run_root / "structural_sensitivity_artifact_manifest.json",
            archive_path=run_root / f"structural_sensitivity_evidence_{tag}.tar.gz",
            receipt_path=run_root / "structural_sensitivity_archive_receipt.json",
        )
    assert observed_roots == [finalizer.REPO_ROOT]


def test_structural_archive_round_trip_verifies_literal_bytes(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    evidence = run_root / "tasks" / "lhs_000" / "baseline__primary.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text('{"result_sha256":"abc"}\n', encoding="utf-8")
    manifest = run_root / "structural_sensitivity_artifact_manifest.json"
    manifest.write_text('{"manifest_sha256":"def"}\n', encoding="utf-8")
    records = finalizer._records(
        run_root, ["tasks/lhs_000/baseline__primary.json"]
    )
    archive = run_root / "structural_sensitivity_evidence_test.tar.gz"
    finalizer._write_verified_archive(
        archive,
        run_root=run_root,
        manifest_path=manifest,
        records=records,
    )
    assert archive.is_file()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        finalizer._write_verified_archive(
            archive,
            run_root=run_root,
            manifest_path=manifest,
            records=records,
        )


def test_environment_capture_supports_external_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _load_hpc_module(
        "capture_publication_environment.py", "capture_environment_external"
    )
    tag = "sensitivity_eeeeeee_20260828_120000"
    snapshot = {
        "commit": "e" * 40,
        "virtual_environment": {
            "run_scoped": True,
            "path_id": f".publication_venvs/{tag}",
            "isolated_from_base_prefix": True,
        },
        "distribution_validation": {
            "unique_normalized_names": True,
            "lock_versions_match": True,
            "core_version_match": True,
            "unexpected_distributions": [],
        },
        "installed_distributions": ["example==1.0"],
    }
    monkeypatch.setattr(capture, "_validated_snapshot", lambda: (snapshot, []))
    monkeypatch.setenv("RUN_TAG", tag)
    output = tmp_path / "external" / "publication_environment.json"
    assert capture.main(["--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["run_tag"] == tag
    assert payload["git_commit"] == "e" * 40
