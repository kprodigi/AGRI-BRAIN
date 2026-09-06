"""Fail-closed contracts for the held combined-evidence finalizer."""

from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from hpc.finalizer_submission_authorization import (
    build_authorization,
    validate_authorization,
    validate_authorization_file,
)

FINALIZER = "303"
CORE = "301"
STRUCTURAL = "302"


def _record(
    *,
    state: str = "PENDING",
    reason: str = "JobHeldUser",
    dependency: str = "afterok:301:302(unfulfilled)",
) -> bytes:
    return (
        f"JobId={FINALIZER} JobName=agribrain-recovery-final "
        f"JobState={state} Reason={reason} Dependency={dependency}\n"
    ).encode()


def _publisher_record(job_id: str, *, state: str = "PENDING", reason: str = "JobHeldUser") -> bytes:
    return f"JobId={job_id} JobState={state} Reason={reason} Dependency=(null)\n".encode()


def _runner_for(stdout: bytes, *, core: bytes | None = None, structural: bytes | None = None):
    def runner(command, **_kwargs):
        records = {FINALIZER: stdout, CORE: core or _publisher_record(CORE), STRUCTURAL: structural or _publisher_record(STRUCTURAL)}
        return subprocess.CompletedProcess(command, 0, records[command[-1]], b"")

    return runner


def _build(stdout: bytes | None = None) -> dict:
    return build_authorization(
        finalizer_job_id=FINALIZER,
        core_publisher_job_id=CORE,
        structural_publisher_job_id=STRUCTURAL,
        runner=_runner_for(stdout or _record()),
    )


def test_held_finalizer_authorization_binds_exact_jobs_and_literal_record() -> None:
    payload = _build()
    validated = validate_authorization(
        payload,
        finalizer_job_id=FINALIZER,
        core_publisher_job_id=CORE,
        structural_publisher_job_id=STRUCTURAL,
    )
    assert validated["simulation_rerun"] is False
    assert validated["finalizer_slurm_job_id"] == FINALIZER
    assert validated["required_scheduler_state"]["afterok_job_ids"] == [CORE, STRUCTURAL]
    assert (
        validated["observed_held_scheduler_records"]["finalizer"]["parsed"]["dependency_literal"]
        == "afterok:301:302(unfulfilled)"
    )


@pytest.mark.parametrize(
    "record",
    [
        _record(state="RUNNING", reason="None"),
        _record(reason="Priority"),
        _record(dependency="afterany:301:302(unfulfilled)"),
        _record(dependency="afterok:301:302:304(unfulfilled)"),
        _record(dependency="afterok:301(unfulfilled)"),
    ],
)
def test_nonheld_or_inexact_dependency_cannot_be_authorized(record: bytes) -> None:
    with pytest.raises(ValueError):
        _build(record)


def test_authorization_tamper_and_job_substitution_fail() -> None:
    payload = _build()
    tampered = deepcopy(payload)
    tampered["observed_held_scheduler_records"]["finalizer"]["stdout"] = tampered[
        "observed_held_scheduler_records"
    ]["finalizer"]["stdout"].replace("JobHeldUser", "Priority")
    with pytest.raises(ValueError, match="self-hash mismatch"):
        validate_authorization(
            tampered,
            finalizer_job_id=FINALIZER,
            core_publisher_job_id=CORE,
            structural_publisher_job_id=STRUCTURAL,
        )
    with pytest.raises(ValueError, match="different Slurm jobs"):
        validate_authorization(
            payload,
            finalizer_job_id=FINALIZER,
            core_publisher_job_id="300",
            structural_publisher_job_id=STRUCTURAL,
        )


def test_live_revalidation_requires_finalizer_to_remain_held() -> None:
    payload = _build()
    with pytest.raises(ValueError, match="not PENDING"):
        validate_authorization(
            payload,
            finalizer_job_id=FINALIZER,
            core_publisher_job_id=CORE,
            structural_publisher_job_id=STRUCTURAL,
            require_live_held=True,
            runner=_runner_for(_record(state="RUNNING", reason="None")),
        )


def test_live_revalidation_requires_both_publishers_to_remain_held() -> None:
    payload = _build()
    with pytest.raises(ValueError, match="publisher is not PENDING"):
        validate_authorization(
            payload,
            finalizer_job_id=FINALIZER,
            core_publisher_job_id=CORE,
            structural_publisher_job_id=STRUCTURAL,
            require_live_held=True,
            runner=_runner_for(_record(), core=_publisher_record(CORE, state="RUNNING", reason="None")),
        )


def test_authorization_file_must_be_absolute_plain_file(tmp_path: Path) -> None:
    payload = _build()
    receipt = tmp_path / "authorization.json"
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    validate_authorization_file(
        receipt,
        finalizer_job_id=FINALIZER,
        core_publisher_job_id=CORE,
        structural_publisher_job_id=STRUCTURAL,
    )
    link = tmp_path / "authorization-link.json"
    try:
        link.symlink_to(receipt)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ValueError, match="non-symlink regular file"):
        validate_authorization_file(
            link,
            finalizer_job_id=FINALIZER,
            core_publisher_job_id=CORE,
            structural_publisher_job_id=STRUCTURAL,
        )
