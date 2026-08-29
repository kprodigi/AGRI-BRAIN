"""Fail-closed structural publication-recovery boundaries."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mvp.simulation.sensitivity import (
    finalize_structural_sensitivity as finalizer,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SIMULATION_COMMIT = "a" * 40
PUBLICATION_COMMIT = "b" * 40
PUBLICATION_TREE = "c" * 40
SIMULATION_TREE = "d" * 64
RUN_TAG = f"sensitivity_{SIMULATION_COMMIT[:7]}_20260829_120000"
RECOVERY_JOB_ID = "15550001"


def _canonical_recovery_files(
    run_root: Path,
    *,
    job_id: str = RECOVERY_JOB_ID,
) -> tuple[Path, Path, Path, Path]:
    original = run_root / "slurm_submission.json"
    attempt_root = (
        run_root / finalizer.RECOVERY_ATTEMPT_DIRECTORY / job_id
    )
    receipt = (
        attempt_root
        / finalizer.RECOVERY_RECEIPT_DIRECTORY
        / f"{RUN_TAG}.json"
    )
    manifest = (
        attempt_root
        / finalizer.PRESERVED_RAW_MANIFEST_DIRECTORY
        / f"{RUN_TAG}.json"
    )
    receipt.parent.mkdir(parents=True)
    manifest.parent.mkdir(parents=True)
    logs = run_root / "logs"
    logs.mkdir()
    (logs / "publish_14473494.out").write_bytes(b"failed stdout\n")
    (logs / "publish_14473494.err").write_bytes(b"failed stderr\n")
    original.write_text("{}\n", encoding="utf-8")
    receipt.write_text("{}\n", encoding="utf-8")
    manifest.write_text('{"raw":true}\n', encoding="utf-8")
    return original, attempt_root, receipt, manifest


def _fake_recovery_receipt(manifest: Path, *, tree: str = PUBLICATION_TREE) -> dict:
    manifest_bytes = manifest.read_bytes()
    return {
        "receipt_sha256": "e" * 64,
        "source_identity": {"publication_repair_tree": tree},
        "failed_publisher": {
            "job_id": "14473494",
            "stdout": {
                "file": "publish_14473494.out",
                "bytes": len(b"failed stdout\n"),
                "sha256": hashlib.sha256(b"failed stdout\n").hexdigest(),
            },
            "stderr": {
                "file": "publish_14473494.err",
                "bytes": len(b"failed stderr\n"),
                "sha256": hashlib.sha256(b"failed stderr\n").hexdigest(),
            },
        },
        "recovery_publisher": {"job_id": RECOVERY_JOB_ID},
        "preserved_raw_outputs": {
            "file": manifest.name,
            "bytes": len(manifest_bytes),
            "literal_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "manifest_self_hash": "f" * 64,
            "payload_merkle_root": "1" * 64,
            "record_count": 9,
        },
    }


def _fake_raw_manifest() -> dict:
    return {
        "manifest_sha256": "f" * 64,
        "payload_merkle_root": "1" * 64,
        "file_count": 9,
    }


def test_structural_recovery_requires_canonical_files_exact_job_and_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    original, attempt_root, receipt_path, manifest_path = (
        _canonical_recovery_files(run_root)
    )
    receipt = _fake_recovery_receipt(manifest_path)
    observed: dict[str, object] = {}

    def validate_receipt(path: Path, **kwargs):
        observed["receipt"] = path
        observed["original"] = kwargs["original_receipt_path"]
        observed["expected_job"] = kwargs["expected_recovery_job_id"]
        return receipt

    def require_publisher(payload, **kwargs):
        assert payload is receipt
        observed["actual_job"] = kwargs["actual_slurm_job_id"]
        return payload

    monkeypatch.setattr(
        finalizer, "validate_recovery_receipt_file", validate_receipt,
    )
    monkeypatch.setattr(finalizer, "require_authorized_publisher", require_publisher)
    monkeypatch.setattr(finalizer, "_git_tree_sha1", lambda _root: PUBLICATION_TREE)
    monkeypatch.setattr(
        finalizer,
        "_validate_live_recovery_raw_outputs",
        lambda *_args, **_kwargs: _fake_raw_manifest(),
    )

    context = finalizer._validate_structural_recovery(
        run_root=run_root,
        attempt_root=attempt_root,
        run_tag=RUN_TAG,
        simulation_commit=SIMULATION_COMMIT,
        simulation_source_tree_sha256=SIMULATION_TREE,
        publication_commit=PUBLICATION_COMMIT,
        recovery_receipt_path=receipt_path,
        raw_manifest_path=manifest_path,
        publisher_slurm_job_id=RECOVERY_JOB_ID,
    )
    assert context["receipt"] is receipt
    assert context["publication_tree"] == PUBLICATION_TREE
    assert observed == {
        "receipt": receipt_path.resolve(),
        "original": original.resolve(),
        "expected_job": RECOVERY_JOB_ID,
        "actual_job": RECOVERY_JOB_ID,
    }


def test_structural_recovery_rejects_noncanonical_or_changed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    _original, attempt_root, receipt_path, manifest_path = (
        _canonical_recovery_files(run_root)
    )
    receipt = _fake_recovery_receipt(manifest_path)
    monkeypatch.setattr(
        finalizer,
        "validate_recovery_receipt_file",
        lambda *_args, **_kwargs: receipt,
    )
    monkeypatch.setattr(
        finalizer,
        "require_authorized_publisher",
        lambda payload, **_kwargs: payload,
    )
    monkeypatch.setattr(finalizer, "_git_tree_sha1", lambda _root: PUBLICATION_TREE)
    monkeypatch.setattr(
        finalizer,
        "_validate_live_recovery_raw_outputs",
        lambda *_args, **_kwargs: _fake_raw_manifest(),
    )

    copied = tmp_path / "copied-recovery.json"
    copied.write_bytes(receipt_path.read_bytes())
    with pytest.raises(ValueError, match="canonical run-scoped path"):
        finalizer._validate_structural_recovery(
            run_root=run_root,
            attempt_root=attempt_root,
            run_tag=RUN_TAG,
            simulation_commit=SIMULATION_COMMIT,
            simulation_source_tree_sha256=SIMULATION_TREE,
            publication_commit=PUBLICATION_COMMIT,
            recovery_receipt_path=copied,
            raw_manifest_path=manifest_path,
            publisher_slurm_job_id=RECOVERY_JOB_ID,
        )

    manifest_path.write_text('{"raw":false}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="differs from recovery authorization"):
        finalizer._validate_structural_recovery(
            run_root=run_root,
            attempt_root=attempt_root,
            run_tag=RUN_TAG,
            simulation_commit=SIMULATION_COMMIT,
            simulation_source_tree_sha256=SIMULATION_TREE,
            publication_commit=PUBLICATION_COMMIT,
            recovery_receipt_path=receipt_path,
            raw_manifest_path=manifest_path,
            publisher_slurm_job_id=RECOVERY_JOB_ID,
        )


def test_structural_recovery_rejects_wrong_publication_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    _original, attempt_root, receipt_path, manifest_path = (
        _canonical_recovery_files(run_root)
    )
    receipt = _fake_recovery_receipt(manifest_path, tree="9" * 40)
    monkeypatch.setattr(
        finalizer,
        "validate_recovery_receipt_file",
        lambda *_args, **_kwargs: receipt,
    )
    monkeypatch.setattr(
        finalizer,
        "require_authorized_publisher",
        lambda payload, **_kwargs: payload,
    )
    monkeypatch.setattr(finalizer, "_git_tree_sha1", lambda _root: PUBLICATION_TREE)
    with pytest.raises(ValueError, match="tree differs"):
        finalizer._validate_structural_recovery(
            run_root=run_root,
            attempt_root=attempt_root,
            run_tag=RUN_TAG,
            simulation_commit=SIMULATION_COMMIT,
            simulation_source_tree_sha256=SIMULATION_TREE,
            publication_commit=PUBLICATION_COMMIT,
            recovery_receipt_path=receipt_path,
            raw_manifest_path=manifest_path,
            publisher_slurm_job_id=RECOVERY_JOB_ID,
        )


def test_recovery_attempt_root_is_exactly_scoped_to_executing_job(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    first = run_root / finalizer.RECOVERY_ATTEMPT_DIRECTORY / RECOVERY_JOB_ID
    second_job = "15550002"
    second = run_root / finalizer.RECOVERY_ATTEMPT_DIRECTORY / second_job
    first.mkdir(parents=True)
    second.mkdir()
    sentinel = first / "partial-output.json"
    sentinel.write_bytes(b"first-attempt\n")

    assert finalizer._validated_recovery_attempt_root(
        first,
        run_root=run_root,
        publisher_slurm_job_id=RECOVERY_JOB_ID,
    ) == first
    assert finalizer._validated_recovery_attempt_root(
        second,
        run_root=run_root,
        publisher_slurm_job_id=second_job,
    ) == second
    assert sentinel.read_bytes() == b"first-attempt\n"

    with pytest.raises(ValueError, match="job-ID-scoped"):
        finalizer._validated_recovery_attempt_root(
            first,
            run_root=run_root,
            publisher_slurm_job_id=second_job,
        )
    escaped = tmp_path / RECOVERY_JOB_ID
    escaped.mkdir()
    with pytest.raises(ValueError, match="job-ID-scoped"):
        finalizer._validated_recovery_attempt_root(
            escaped,
            run_root=run_root,
            publisher_slurm_job_id=RECOVERY_JOB_ID,
        )


def test_recovery_attempt_root_rejects_symlinked_job_directory(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    attempt_parent = run_root / finalizer.RECOVERY_ATTEMPT_DIRECTORY
    attempt_parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    attempt = attempt_parent / RECOVERY_JOB_ID
    try:
        attempt.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(ValueError, match="symbolic link"):
        finalizer._validated_recovery_attempt_root(
            attempt,
            run_root=run_root,
            publisher_slurm_job_id=RECOVERY_JOB_ID,
        )


def test_recovery_publisher_never_runs_workers_and_revalidates_raw_inputs() -> None:
    script = (REPO_ROOT / "hpc" / "hpc_sensitivity_publish_recovery.sh").read_text(
        encoding="utf-8"
    )
    assert "run_structural_sensitivity task" not in script
    assert "hpc_sensitivity_task.sh" not in script
    assert script.count("\nvalidate_preserved_raw_outputs\n") == 2
    assert '["scontrol", "show", "job", "-o", job_id]' in script
    assert "Slurm {field} would mutate preserved raw inputs" in script
    for name in finalizer.STRUCTURAL_RAW_DIRECTORIES:
        assert f'--input-root "{name}=' in script
    for name in finalizer.STRUCTURAL_RAW_FILES:
        assert f'--input-file "{name}=' in script
    assert "--recovery-publisher-slurm-job-id \"$SLURM_JOB_ID\"" in script
    assert '--recovery-attempt-root "$RECOVERY_ATTEMPT_ROOT"' in script
    assert "--recovery-receipt \"$AGRIBRAIN_RECOVERY_RECEIPT\"" in script
    assert (
        'RECOVERY_ATTEMPT_ROOT="${SENSITIVITY_RUN_DIR}/'
        'publication_recovery_attempts/${SLURM_JOB_ID}"'
    ) in script


def test_partial_recovery_request_fails_before_fresh_finalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    plan = run_root / "run_plan.json"
    monkeypatch.setattr(
        finalizer,
        "_load_plan_bundle",
        lambda _path: (
            {
                "run_tag": RUN_TAG,
                "source_commit": SIMULATION_COMMIT,
                "execution_scope": "structural_sensitivity_only",
                "source_tree_clean_at_generation": True,
            },
            {},
            {},
            {},
        ),
    )
    with pytest.raises(ValueError, match="requires attempt root, receipt, raw manifest"):
        finalizer.finalize_run(
            plan,
            status_path=run_root / "completion_status.json",
            analysis_path=run_root / "structural_sensitivity_analysis.json",
            environment_path=run_root / "publication_environment.json",
            scheduler_accounting_path=run_root / "slurm_simulation_accounting.json",
            manifest_path=run_root / "structural_sensitivity_artifact_manifest.json",
            archive_path=run_root / f"structural_sensitivity_evidence_{RUN_TAG}.tar.gz",
            receipt_path=run_root / "structural_sensitivity_archive_receipt.json",
            recovery_receipt_path=tmp_path / "receipt.json",
        )


def test_structural_publishers_reject_dangling_output_links() -> None:
    for relative in (
        "hpc/hpc_sensitivity_publish.sh",
        "hpc/hpc_sensitivity_publish_recovery.sh",
    ):
        script = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert 'if [ -e "$output" ] || [ -L "$output" ]; then' in script


def test_structural_output_gate_rejects_dangling_leaf_symlink(
    tmp_path: Path,
) -> None:
    output = tmp_path / "structural_sensitivity_archive_receipt.json"
    try:
        output.symlink_to(tmp_path / "outside.json")
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(ValueError, match="symbolic link"):
        finalizer._validated_run_output_path(
            output, output,
            label="structural archive receipt", must_exist=False,
        )


def test_structural_output_gate_rejects_symlinked_parent(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")
    output = alias / "structural_sensitivity_archive_receipt.json"

    with pytest.raises(ValueError, match="symbolic link"):
        finalizer._validated_run_output_path(
            output, output,
            label="structural archive receipt", must_exist=False,
        )
    assert not (outside / output.name).exists()
