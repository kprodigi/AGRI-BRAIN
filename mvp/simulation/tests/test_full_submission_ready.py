"""Fail-closed contracts for the promoted combined-submission READY bundle."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hpc.finalizer_submission_authorization import build_authorization, canonical_sha256
from hpc.validate_full_submission_ready import create_ready, validate_ready
from hpc.validate_lexical_path import validate_lexical_path

SIM = "1" * 40
PUB = "2" * 40
FINALIZER = "303"
CORE = "301"
STRUCTURAL = "302"
RUN_TAG = "finalizer_core-run"


def _runner(command, **_kwargs):
    job_id = command[-1]
    if job_id == FINALIZER:
        stdout = (
            f"JobId={FINALIZER} JobState=PENDING Reason=JobHeldUser "
            f"Dependency=afterok:{CORE}:{STRUCTURAL}(unfulfilled)\n"
        )
    else:
        stdout = f"JobId={job_id} JobState=PENDING Reason=JobHeldUser Dependency=(null)\n"
    return subprocess.CompletedProcess(command, 0, stdout.encode(), b"")


def _identity() -> dict[str, str]:
    return {
        "simulation_commit": SIM,
        "publication_commit": PUB,
        "finalizer_job_id": FINALIZER,
        "core_job_id": CORE,
        "structural_job_id": STRUCTURAL,
        "run_tag": RUN_TAG,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _bundle(tmp_path: Path) -> Path:
    directory = tmp_path / "combined"
    directory.mkdir()
    receipt = {
        "receipt_type": "full_submission_evidence_set",
        "dual_provenance": True,
        "simulation_rerun": False,
        "simulation_source_commit": SIM,
        "publication_code_commit": PUB,
        "recovery_authorizations": {
            "core": {"authorized_recovery_publisher_job_id": CORE},
            "structural": {"authorized_recovery_publisher_job_id": STRUCTURAL},
        },
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    authorization = build_authorization(
        finalizer_job_id=FINALIZER,
        core_publisher_job_id=CORE,
        structural_publisher_job_id=STRUCTURAL,
        runner=_runner,
    )
    environment = {"schema_version": 2, "run_tag": RUN_TAG, "git_commit": PUB}
    _write_json(directory / "FULL_SUBMISSION_EVIDENCE_RECEIPT.json", receipt)
    _write_json(directory / "FINALIZER_SUBMISSION_AUTHORIZATION.json", authorization)
    _write_json(directory / "FINALIZER_PUBLICATION_ENVIRONMENT.json", environment)
    create_ready(directory, **_identity())
    return directory


def test_exact_four_file_ready_bundle_validates(tmp_path: Path) -> None:
    directory = _bundle(tmp_path)
    ready = validate_ready(directory, **_identity())
    assert ready["simulation_rerun"] is False
    assert ready["finalizer_publication_environment"]["schema_version"] == 2


@pytest.mark.parametrize(
    ("file_name", "replacement"),
    [
        ("FULL_SUBMISSION_EVIDENCE_RECEIPT.json", b"{}\n"),
        ("FINALIZER_SUBMISSION_AUTHORIZATION.json", b"{}\n"),
        ("FINALIZER_PUBLICATION_ENVIRONMENT.json", b"{}\n"),
        ("READY.json", b"{}\n"),
    ],
)
def test_literal_corruption_fails(tmp_path: Path, file_name: str, replacement: bytes) -> None:
    directory = _bundle(tmp_path)
    (directory / file_name).write_bytes(replacement)
    with pytest.raises(ValueError):
        validate_ready(directory, **_identity())


def test_job_substitution_and_extra_file_fail(tmp_path: Path) -> None:
    directory = _bundle(tmp_path)
    substituted = _identity()
    substituted["core_job_id"] = "999"
    with pytest.raises(ValueError):
        validate_ready(directory, **substituted)
    (directory / "extra.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="exact four-file inventory"):
        validate_ready(directory, **_identity())


def test_dangling_leaf_and_parent_symlinks_fail(tmp_path: Path) -> None:
    dangling_leaf = tmp_path / "dangling"
    linked_parent = tmp_path / "linked-parent"
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    try:
        dangling_leaf.symlink_to(tmp_path / "missing")
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ValueError, match="symlink component"):
        validate_lexical_path(dangling_leaf, kind="absent")
    with pytest.raises(ValueError, match="symlink component"):
        validate_lexical_path(linked_parent / "missing.json", kind="absent")
