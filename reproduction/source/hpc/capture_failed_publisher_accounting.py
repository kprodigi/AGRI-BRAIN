#!/usr/bin/env python3
"""Capture a self-hashed Slurm record for a declared failed publisher job."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


SCHEMA_VERSION = 1
RECEIPT_TYPE = "failed_declared_publisher_slurm_accounting"
_JOB_ID = re.compile(r"[0-9]+")
_TERMINAL_FAILURE_STATES = frozenset({
    "FAILED", "CANCELLED", "DEADLINE", "NODE_FAIL", "OUT_OF_MEMORY",
    "PREEMPTED", "REVOKED", "TIMEOUT", "BOOT_FAIL",
})
_FIELDS = (
    "JobID", "JobIDRaw", "JobName", "State", "ExitCode", "Submit",
    "Eligible", "Start", "End", "ElapsedRaw", "NodeList", "Partition",
    "Cluster",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _clean_state(raw: object) -> str:
    return str(raw).strip().split("+", 1)[0].split(" ", 1)[0]


def _parse_rows(stdout: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line_number, raw in enumerate(stdout.splitlines(), 1):
        if not raw.strip():
            continue
        values = raw.rstrip("\r\n").split("|")
        if values and values[-1] == "":
            values.pop()
        if len(values) != len(_FIELDS):
            raise ValueError(
                f"sacct row {line_number} has {len(values)} fields; "
                f"expected {len(_FIELDS)}"
            )
        rows.append(dict(zip(_FIELDS, values, strict=True)))
    if not rows:
        raise ValueError("sacct returned no failed-publisher accounting rows")
    return rows


def validate_failed_publisher_accounting(
    payload: Mapping[str, Any], *, expected_job_id: str,
) -> dict[str, Any]:
    if not _JOB_ID.fullmatch(str(expected_job_id)):
        raise ValueError("expected failed publisher job ID must be numeric")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("receipt_type") != RECEIPT_TYPE
        or payload.get("job_id") != str(expected_job_id)
    ):
        raise ValueError("failed publisher accounting identity is inconsistent")
    unsigned = dict(payload)
    claimed = unsigned.pop("accounting_sha256", None)
    if not isinstance(claimed, str) or claimed != _canonical_sha256(unsigned):
        raise ValueError("failed publisher accounting self-hash is invalid")
    rows = payload.get("rows")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("failed publisher accounting rows are malformed")
    allocation = [row for row in rows if row.get("JobID") == str(expected_job_id)]
    if len(allocation) != 1:
        raise ValueError("failed publisher accounting lacks one exact allocation row")
    row = allocation[0]
    state = _clean_state(row.get("State"))
    exit_code = str(row.get("ExitCode", "")).strip()
    if state not in _TERMINAL_FAILURE_STATES:
        raise ValueError(f"declared publisher is not terminally failed: {state!r}")
    if not re.fullmatch(r"[0-9]+:[0-9]+", exit_code) or exit_code == "0:0":
        raise ValueError("failed publisher does not have a nonzero Slurm exit code")
    if payload.get("terminal_state") != state or payload.get(
        "exit_code"
    ) != exit_code:
        raise ValueError("failed publisher accounting summary differs from rows")
    scheduler = payload.get("scheduler")
    if not isinstance(scheduler, dict):
        raise ValueError("failed publisher accounting lacks scheduler evidence")
    raw_stdout = scheduler.get("raw_stdout")
    if not isinstance(raw_stdout, str) or scheduler.get(
        "raw_stdout_sha256"
    ) != hashlib.sha256(raw_stdout.encode("utf-8")).hexdigest():
        raise ValueError("failed publisher raw sacct output hash is invalid")
    if _parse_rows(raw_stdout) != rows:
        raise ValueError("failed publisher parsed rows differ from raw sacct output")
    return dict(payload)


def capture_failed_publisher_accounting(
    *, job_id: str, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    if not _JOB_ID.fullmatch(str(job_id)):
        raise ValueError("failed publisher job ID must be numeric")
    version = runner(
        ["sacct", "--version"], check=False, capture_output=True, text=True,
        timeout=60,
    )
    if version.returncode != 0:
        raise RuntimeError(f"sacct --version failed: {version.stderr.strip()}")
    command = [
        "sacct", "--noheader", "--parsable2", "--allocations", "--local",
        "--jobs", str(job_id), "--format", ",".join(_FIELDS),
    ]
    completed = runner(
        command, check=False, capture_output=True, text=True, timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"sacct failed: {completed.stderr.strip()}")
    rows = _parse_rows(completed.stdout)
    allocation = [row for row in rows if row.get("JobID") == str(job_id)]
    if len(allocation) != 1:
        raise ValueError("sacct lacks one exact failed publisher allocation row")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": RECEIPT_TYPE,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": str(job_id),
        "terminal_state": _clean_state(allocation[0].get("State")),
        "exit_code": str(allocation[0].get("ExitCode", "")).strip(),
        "scheduler": {
            "sacct_version_stdout": version.stdout,
            "command": command,
            "raw_stdout": completed.stdout,
            "raw_stdout_sha256": hashlib.sha256(
                completed.stdout.encode("utf-8")
            ).hexdigest(),
            "raw_stderr": completed.stderr,
        },
        "rows": rows,
    }
    payload["accounting_sha256"] = _canonical_sha256(payload)
    validate_failed_publisher_accounting(payload, expected_job_id=str(job_id))
    return payload


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = path.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite failed publisher accounting: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent,
        prefix=f".{destination.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = capture_failed_publisher_accounting(job_id=args.job_id)
    _write_new_json(args.output, payload)
    print(json.dumps({
        "status": "VALID",
        "job_id": payload["job_id"],
        "terminal_state": payload["terminal_state"],
        "exit_code": payload["exit_code"],
        "accounting_sha256": payload["accounting_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
