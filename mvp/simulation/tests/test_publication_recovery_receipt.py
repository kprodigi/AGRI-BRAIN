"""Focused tests for fail-closed publication-only recovery authorization."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from hpc.capture_failed_publisher_accounting import (
    capture_failed_publisher_accounting,
)
from hpc.core_submission_receipt import SNAPSHOT_MODE, build_receipt
from hpc.preserved_raw_manifest import build_manifest
from hpc.publication_recovery_receipt import (
    RECOVERY_REASON_CODE,
    canonical_sha256,
    create_recovery_receipt,
    main as recovery_main,
    require_authorized_publisher,
    validate_recovery_receipt_file,
)


SIMULATION_COMMIT = "a" * 40
PUBLICATION_COMMIT = "b" * 40
PUBLICATION_TREE = "c" * 40
SOURCE_TREE_SHA256 = "d" * 64


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _core_submission() -> tuple[str, str, dict]:
    run_tag = "aaaaaaa_20260829_105800"
    payload = build_receipt(
        run_tag=run_tag,
        source_commit=SIMULATION_COMMIT,
        partition="compute",
        seed_job_id="101",
        stress_job_id="102",
        publisher_job_id="103",
        source_snapshot_mode=SNAPSHOT_MODE,
        source_tree_sha256=SOURCE_TREE_SHA256,
    )
    return run_tag, "103", payload


def _structural_submission() -> tuple[str, str, dict]:
    run_tag = "sensitivity_aaaaaaa_20260829_105855"
    payload = {
        "schema_version": 2,
        "analysis_label": "structural sensitivity",
        "receipt_scope": "submission_only_not_scheduler_completion",
        "scheduler_completion_attested": False,
        "run_tag": run_tag,
        "source_commit": SIMULATION_COMMIT,
        "source_snapshot_mode": SNAPSHOT_MODE,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "task_count": 3_000,
        "array_chunk_size_limit": 1_000,
        "max_concurrent_per_array": 20,
        "task_arrays": [
            {
                "job_id": "201",
                "offset": 0,
                "count": 1_000,
                "afterok_job_id": None,
            },
            {
                "job_id": "202",
                "offset": 1_000,
                "count": 1_000,
                "afterok_job_id": "201",
            },
            {
                "job_id": "203",
                "offset": 2_000,
                "count": 1_000,
                "afterok_job_id": "202",
            },
        ],
        "publisher": {"job_id": "204", "afterok_job_id": "203"},
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return run_tag, "204", payload


def _failed_accounting(job_id: str) -> dict:
    values = (
        job_id,
        job_id,
        "agribrain-publish",
        "FAILED",
        "1:0",
        "2026-08-29T11:00:00",
        "2026-08-29T11:00:01",
        "2026-08-29T11:00:02",
        "2026-08-29T11:00:03",
        "1",
        "node001",
        "compute",
        "cluster",
    )
    stdout = "|".join(values) + "|\n"

    def runner(command, **_kwargs):
        if command[1] == "--version":
            return subprocess.CompletedProcess(command, 0, "slurm 23.11.0\n", "")
        return subprocess.CompletedProcess(command, 0, stdout, "")

    return capture_failed_publisher_accounting(job_id=job_id, runner=runner)


def _recovery_inputs(
    tmp_path: Path,
    *,
    kind: str,
    run_tag: str,
    publisher_job_id: str,
    submission: dict,
) -> dict[str, Path]:
    if kind == "core":
        run_root = tmp_path / "core_snapshot"
        original = (
            run_root / "mvp" / "simulation" / "results"
            / "core_submission_receipts" / f"{run_tag}.json"
        )
    else:
        run_root = tmp_path / "structural_run"
        original = run_root / "slurm_submission.json"
    _write_json(original, submission)
    failed_accounting = tmp_path / "failed_accounting.json"
    _write_json(failed_accounting, _failed_accounting(publisher_job_id))
    logs = run_root / "logs"
    logs.mkdir(parents=True)
    stdout = logs / f"publish_{publisher_job_id}.out"
    stderr = logs / f"publish_{publisher_job_id}.err"
    stdout.write_bytes(b"publisher validation failed\n")
    stderr.write_bytes(b"ValueError: exact accounting mismatch\n")

    raw_root = tmp_path / "preserved_raw"
    raw_root.mkdir()
    (raw_root / "worker.json").write_bytes(b'{"status":"complete"}\n')
    raw_manifest = build_manifest(
        kind=kind,
        run_tag=run_tag,
        simulation_commit=SIMULATION_COMMIT,
        simulation_source_tree_sha256=SOURCE_TREE_SHA256,
        roots=[("simulation_workers", raw_root)],
        files=[],
    )
    raw_manifest_path = tmp_path / "preserved_raw_manifest.json"
    _write_json(raw_manifest_path, raw_manifest)
    return {
        "original": original,
        "accounting": failed_accounting,
        "stdout": stdout,
        "stderr": stderr,
        "raw_manifest": raw_manifest_path,
    }


@pytest.mark.parametrize(
    ("kind", "submission_factory", "recovery_job_id"),
    [
        ("core", _core_submission, "301"),
        ("structural", _structural_submission, "401"),
    ],
)
def test_real_preserved_manifest_builds_and_validates_recovery_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind,
    submission_factory,
    recovery_job_id,
) -> None:
    run_tag, publisher_job_id, submission = submission_factory()
    paths = _recovery_inputs(
        tmp_path,
        kind=kind,
        run_tag=run_tag,
        publisher_job_id=publisher_job_id,
        submission=submission,
    )
    original_bytes = paths["original"].read_bytes()
    raw_manifest = json.loads(paths["raw_manifest"].read_text(encoding="utf-8"))
    monkeypatch.setattr(
        "hpc.publication_recovery_receipt._git_clean_identity",
        lambda _root: (PUBLICATION_COMMIT, PUBLICATION_TREE),
    )
    output = tmp_path / "publication_recovery_receipt.json"

    created = create_recovery_receipt(
        output=output,
        repo_root=tmp_path,
        kind=kind,
        run_tag=run_tag,
        simulation_commit=SIMULATION_COMMIT,
        original_receipt_path=paths["original"],
        failed_accounting_record_path=paths["accounting"],
        failed_stdout_path=paths["stdout"],
        failed_stderr_path=paths["stderr"],
        raw_output_manifest_path=paths["raw_manifest"],
        held_recovery_publisher_job_id=recovery_job_id,
        reason_code=RECOVERY_REASON_CODE,
        expected_publication_commit=PUBLICATION_COMMIT,
    )
    validated = validate_recovery_receipt_file(
        output,
        original_receipt_path=paths["original"],
        expected_kind=kind,
        expected_run_tag=run_tag,
        expected_simulation_commit=SIMULATION_COMMIT,
        expected_publication_commit=PUBLICATION_COMMIT,
        expected_recovery_job_id=recovery_job_id,
    )

    assert validated == created
    assert validated["simulation_rerun"] is False
    assert validated["original_submission_receipt"]["literal_sha256"] == (
        hashlib.sha256(original_bytes).hexdigest()
    )
    assert validated["original_submission_receipt"]["receipt_sha256"] == (
        submission["receipt_sha256"]
    )
    assert validated["failed_publisher"]["job_id"] == publisher_job_id
    assert validated["failed_publisher"]["accounting_record"][
        "accounting_sha256"
    ] == json.loads(paths["accounting"].read_text(encoding="utf-8"))[
        "accounting_sha256"
    ]
    assert validated["preserved_raw_outputs"]["payload_merkle_root"] == (
        raw_manifest["payload_merkle_root"]
    )
    assert paths["original"].read_bytes() == original_bytes
    assert require_authorized_publisher(
        validated,
        actual_slurm_job_id=recovery_job_id,
        expected_kind=kind,
        expected_run_tag=run_tag,
        expected_simulation_commit=SIMULATION_COMMIT,
        expected_publication_commit=PUBLICATION_COMMIT,
    ) == validated


def test_literal_change_to_original_receipt_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_tag, publisher_job_id, submission = _core_submission()
    paths = _recovery_inputs(
        tmp_path,
        kind="core",
        run_tag=run_tag,
        publisher_job_id=publisher_job_id,
        submission=submission,
    )
    monkeypatch.setattr(
        "hpc.publication_recovery_receipt._git_clean_identity",
        lambda _root: (PUBLICATION_COMMIT, PUBLICATION_TREE),
    )
    output = tmp_path / "recovery.json"
    create_recovery_receipt(
        output=output,
        repo_root=tmp_path,
        kind="core",
        run_tag=run_tag,
        simulation_commit=SIMULATION_COMMIT,
        original_receipt_path=paths["original"],
        failed_accounting_record_path=paths["accounting"],
        failed_stdout_path=paths["stdout"],
        failed_stderr_path=paths["stderr"],
        raw_output_manifest_path=paths["raw_manifest"],
        held_recovery_publisher_job_id="301",
        reason_code=RECOVERY_REASON_CODE,
    )
    # Whitespace preserves the original receipt's canonical self-hash but not
    # the literal bytes sealed by the recovery receipt.
    paths["original"].write_bytes(paths["original"].read_bytes() + b"\n")

    with pytest.raises(ValueError, match="literal bytes changed"):
        validate_recovery_receipt_file(
            output,
            original_receipt_path=paths["original"],
            expected_kind="core",
            expected_run_tag=run_tag,
            expected_simulation_commit=SIMULATION_COMMIT,
            expected_publication_commit=PUBLICATION_COMMIT,
        )


def test_semantically_rehashed_simulation_rerun_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_tag, publisher_job_id, submission = _core_submission()
    paths = _recovery_inputs(
        tmp_path,
        kind="core",
        run_tag=run_tag,
        publisher_job_id=publisher_job_id,
        submission=submission,
    )
    monkeypatch.setattr(
        "hpc.publication_recovery_receipt._git_clean_identity",
        lambda _root: (PUBLICATION_COMMIT, PUBLICATION_TREE),
    )
    output = tmp_path / "recovery.json"
    create_recovery_receipt(
        output=output,
        repo_root=tmp_path,
        kind="core",
        run_tag=run_tag,
        simulation_commit=SIMULATION_COMMIT,
        original_receipt_path=paths["original"],
        failed_accounting_record_path=paths["accounting"],
        failed_stdout_path=paths["stdout"],
        failed_stderr_path=paths["stderr"],
        raw_output_manifest_path=paths["raw_manifest"],
        held_recovery_publisher_job_id="301",
        reason_code=RECOVERY_REASON_CODE,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["simulation_rerun"] = True
    payload.pop("receipt_sha256")
    payload["receipt_sha256"] = canonical_sha256(payload)
    _write_json(output, payload)

    with pytest.raises(ValueError, match="prohibit simulation reruns"):
        validate_recovery_receipt_file(
            output,
            original_receipt_path=paths["original"],
            expected_kind="core",
            expected_run_tag=run_tag,
            expected_simulation_commit=SIMULATION_COMMIT,
            expected_publication_commit=PUBLICATION_COMMIT,
        )


def test_wrong_recovery_slurm_job_is_not_authorized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_tag, publisher_job_id, submission = _core_submission()
    paths = _recovery_inputs(
        tmp_path,
        kind="core",
        run_tag=run_tag,
        publisher_job_id=publisher_job_id,
        submission=submission,
    )
    monkeypatch.setattr(
        "hpc.publication_recovery_receipt._git_clean_identity",
        lambda _root: (PUBLICATION_COMMIT, PUBLICATION_TREE),
    )
    output = tmp_path / "recovery.json"
    receipt = create_recovery_receipt(
        output=output,
        repo_root=tmp_path,
        kind="core",
        run_tag=run_tag,
        simulation_commit=SIMULATION_COMMIT,
        original_receipt_path=paths["original"],
        failed_accounting_record_path=paths["accounting"],
        failed_stdout_path=paths["stdout"],
        failed_stderr_path=paths["stderr"],
        raw_output_manifest_path=paths["raw_manifest"],
        held_recovery_publisher_job_id="301",
        reason_code=RECOVERY_REASON_CODE,
    )

    with pytest.raises(ValueError, match="different Slurm job"):
        require_authorized_publisher(receipt, actual_slurm_job_id="302")


def test_create_rejects_arbitrary_failed_log_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_tag, publisher_job_id, submission = _core_submission()
    paths = _recovery_inputs(
        tmp_path, kind="core", run_tag=run_tag,
        publisher_job_id=publisher_job_id, submission=submission,
    )
    arbitrary = tmp_path / f"publish_{publisher_job_id}.out"
    arbitrary.write_bytes(paths["stdout"].read_bytes())
    monkeypatch.setattr(
        "hpc.publication_recovery_receipt._git_clean_identity",
        lambda _root: (PUBLICATION_COMMIT, PUBLICATION_TREE),
    )
    with pytest.raises(ValueError, match="canonical launcher paths"):
        create_recovery_receipt(
            output=tmp_path / "recovery.json", repo_root=tmp_path,
            kind="core", run_tag=run_tag,
            simulation_commit=SIMULATION_COMMIT,
            original_receipt_path=paths["original"],
            failed_accounting_record_path=paths["accounting"],
            failed_stdout_path=arbitrary,
            failed_stderr_path=paths["stderr"],
            raw_output_manifest_path=paths["raw_manifest"],
            held_recovery_publisher_job_id="301",
            reason_code=RECOVERY_REASON_CODE,
        )


def test_create_rejects_unproven_cause_specific_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_tag, publisher_job_id, submission = _core_submission()
    paths = _recovery_inputs(
        tmp_path, kind="core", run_tag=run_tag,
        publisher_job_id=publisher_job_id, submission=submission,
    )
    monkeypatch.setattr(
        "hpc.publication_recovery_receipt._git_clean_identity",
        lambda _root: (PUBLICATION_COMMIT, PUBLICATION_TREE),
    )
    with pytest.raises(ValueError, match="cause-neutral"):
        create_recovery_receipt(
            output=tmp_path / "recovery.json", repo_root=tmp_path,
            kind="core", run_tag=run_tag,
            simulation_commit=SIMULATION_COMMIT,
            original_receipt_path=paths["original"],
            failed_accounting_record_path=paths["accounting"],
            failed_stdout_path=paths["stdout"],
            failed_stderr_path=paths["stderr"],
            raw_output_manifest_path=paths["raw_manifest"],
            held_recovery_publisher_job_id="301",
            reason_code="declared_parser_failure",
        )


def test_validate_cli_rejects_symlinked_recovery_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_tag, publisher_job_id, submission = _core_submission()
    paths = _recovery_inputs(
        tmp_path, kind="core", run_tag=run_tag,
        publisher_job_id=publisher_job_id, submission=submission,
    )
    monkeypatch.setattr(
        "hpc.publication_recovery_receipt._git_clean_identity",
        lambda _root: (PUBLICATION_COMMIT, PUBLICATION_TREE),
    )
    receipt = tmp_path / "recovery.json"
    create_recovery_receipt(
        output=receipt, repo_root=tmp_path, kind="core", run_tag=run_tag,
        simulation_commit=SIMULATION_COMMIT,
        original_receipt_path=paths["original"],
        failed_accounting_record_path=paths["accounting"],
        failed_stdout_path=paths["stdout"],
        failed_stderr_path=paths["stderr"],
        raw_output_manifest_path=paths["raw_manifest"],
        held_recovery_publisher_job_id="301",
        reason_code=RECOVERY_REASON_CODE,
    )
    alias = tmp_path / "recovery-alias.json"
    try:
        alias.symlink_to(receipt)
    except OSError:
        pytest.skip("symlinks are not available to this test user")
    with pytest.raises(ValueError, match="symbolic link"):
        recovery_main([
            "validate", "--receipt", str(alias),
            "--original-submission-receipt", str(paths["original"]),
            "--kind", "core", "--run-tag", run_tag,
            "--simulation-commit", SIMULATION_COMMIT,
            "--publication-commit", PUBLICATION_COMMIT,
            "--recovery-publisher-slurm-job-id", "301",
        ])
