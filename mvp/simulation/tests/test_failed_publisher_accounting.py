from __future__ import annotations

import subprocess
from copy import deepcopy

import pytest

from hpc.capture_failed_publisher_accounting import (
    capture_failed_publisher_accounting,
    validate_failed_publisher_accounting,
)


def _runner(command, **_kwargs):
    if command[1] == "--version":
        return subprocess.CompletedProcess(command, 0, "slurm 24.05\n", "")
    rows = (
        "14473471|14473471|agribrain-publish|FAILED|1:0|2026-08-29T10:58:00|"
        "2026-08-29T10:58:00|2026-08-29T12:31:00|2026-08-29T12:31:02|2|"
        "node001|compute|cluster\n"
        "14473471.batch|14473471.batch|batch|FAILED|1:0|2026-08-29T10:58:00|"
        "2026-08-29T10:58:00|2026-08-29T12:31:00|2026-08-29T12:31:02|2|"
        "node001|compute|cluster\n"
    )
    return subprocess.CompletedProcess(command, 0, rows, "")


def test_capture_binds_exact_failed_allocation_and_raw_rows() -> None:
    payload = capture_failed_publisher_accounting(job_id="14473471", runner=_runner)
    validate_failed_publisher_accounting(payload, expected_job_id="14473471")
    assert payload["terminal_state"] == "FAILED"
    assert payload["exit_code"] == "1:0"
    assert len(payload["rows"]) == 2


def test_successful_or_wrong_job_cannot_authorize_recovery() -> None:
    payload = capture_failed_publisher_accounting(job_id="14473471", runner=_runner)
    altered = deepcopy(payload)
    altered["rows"][0]["State"] = "COMPLETED"
    from hpc.capture_failed_publisher_accounting import _canonical_sha256

    altered.pop("accounting_sha256")
    altered["accounting_sha256"] = _canonical_sha256(altered)
    with pytest.raises(ValueError):
        validate_failed_publisher_accounting(altered, expected_job_id="14473471")
    with pytest.raises(ValueError, match="identity is inconsistent"):
        validate_failed_publisher_accounting(payload, expected_job_id="14473494")
