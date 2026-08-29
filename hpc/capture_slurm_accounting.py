#!/usr/bin/env python3
"""Capture scheduler accounting for every completed simulation-array task.

The per-worker runtime receipts preserve process-level resource usage while the
worker is alive.  This companion artifact preserves Slurm's post-job view of
the allocation (including energy fields when the site exposes them).  Missing
optional fields are recorded explicitly and are never imputed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_JOB_ID = re.compile(r"[0-9]+")
_ARRAY_ELEMENT_ID = re.compile(r"[0-9]+_[0-9]+")
_TERMINAL_FAILURE_STATES = frozenset({
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
})
_DESIRED_FIELDS = (
    "JobID",
    "JobIDRaw",
    "JobName",
    "State",
    "ExitCode",
    "Submit",
    "Eligible",
    "Start",
    "End",
    "ElapsedRaw",
    "TotalCPU",
    "CPUTimeRAW",
    "AllocCPUS",
    "NTasks",
    "NNodes",
    "ReqCPUS",
    "ReqMem",
    "MaxRSS",
    "MaxVMSize",
    "AveRSS",
    "AveVMSize",
    "MaxDiskRead",
    "MaxDiskWrite",
    "ConsumedEnergyRaw",
    "ConsumedEnergy",
    "AvePower",
    "NodeList",
    "Partition",
    "Cluster",
)
_REQUIRED_FIELDS = {
    "JobID", "JobIDRaw", "State", "ExitCode", "ElapsedRaw", "AllocCPUS",
}
_DEFAULT_ATTEMPTS = 12
_DEFAULT_RETRY_SECONDS = 5.0
_DEFAULT_MAX_RETRY_SECONDS = 120.0
_DEFAULT_QUERY_TIMEOUT_SECONDS = 60.0
_RETRY_BACKOFF_MULTIPLIER = 2.0


class _AccountingNotReady(RuntimeError):
    """A transient sacct result that may become complete on a later query."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _load_self_hashed_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"submission receipt is missing or unsafe: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("submission receipt is not a JSON object")
    unsigned = dict(payload)
    claimed = unsigned.pop("receipt_sha256", None)
    if claimed != _canonical_sha256(unsigned):
        raise ValueError("submission receipt self-hash mismatch")
    return payload


def _expected_arrays(
    receipt: Mapping[str, Any], *, kind: str,
) -> list[dict[str, int | str]]:
    arrays: list[dict[str, int | str]] = []
    if kind == "core":
        dag = receipt.get("slurm_dag")
        if not isinstance(dag, dict):
            raise ValueError("core submission receipt lacks slurm_dag")
        for stage in ("seed_array", "stress_array"):
            record = dag.get(stage)
            if not isinstance(record, dict):
                raise ValueError(f"core submission receipt lacks {stage}")
            arrays.append({
                "stage": stage,
                "job_id": str(record.get("job_id", "")),
                "task_count": int(record.get("task_count", -1)),
            })
    elif kind == "structural":
        raw_arrays = receipt.get("task_arrays")
        if not isinstance(raw_arrays, list) or not raw_arrays:
            raise ValueError("structural submission receipt lacks task arrays")
        for position, record in enumerate(raw_arrays):
            if not isinstance(record, dict):
                raise ValueError("structural task-array receipt is malformed")
            arrays.append({
                "stage": f"structural_chunk_{position}",
                "job_id": str(record.get("job_id", "")),
                "task_count": int(record.get("count", -1)),
            })
    else:
        raise ValueError(f"unsupported accounting kind: {kind}")
    if any(
        not _JOB_ID.fullmatch(str(record["job_id"]))
        or int(record["task_count"]) <= 0
        for record in arrays
    ):
        raise ValueError("submission receipt contains an invalid array identity")
    job_ids = [str(record["job_id"]) for record in arrays]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("submission receipt reuses one Slurm array job id")
    return arrays


def _available_fields(help_text: str) -> dict[str, str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9]*", help_text)
    return {token.casefold(): token for token in tokens}


def _select_fields(help_text: str) -> tuple[list[tuple[str, str]], list[str]]:
    available = _available_fields(help_text)
    selected = [
        (field, available[field.casefold()])
        for field in _DESIRED_FIELDS
        if field.casefold() in available
    ]
    selected_names = {canonical for canonical, _actual in selected}
    missing_required = sorted(_REQUIRED_FIELDS - selected_names)
    if missing_required:
        raise ValueError(
            "sacct does not expose required fields: " + ", ".join(missing_required)
        )
    missing_optional = [
        field for field in _DESIRED_FIELDS if field not in selected_names
    ]
    return selected, missing_optional


def _parse_rows(
    stdout: str, selected: Sequence[tuple[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    canonical_names = [canonical for canonical, _actual in selected]
    for line_number, line in enumerate(stdout.splitlines(), start=1):
        if not line:
            continue
        values = line.split("|")
        if values and values[-1] == "":
            values.pop()
        if len(values) != len(canonical_names):
            raise ValueError(
                f"sacct row {line_number} has {len(values)} fields; "
                f"expected {len(canonical_names)}"
            )
        rows.append(dict(zip(canonical_names, values, strict=True)))
    if not rows:
        raise _AccountingNotReady("sacct returned no accounting rows")
    return rows


def _clean_state(raw: str) -> str:
    return raw.strip().split("+", 1)[0].split(" ", 1)[0]


def _validate_completed_arrays(
    rows: Sequence[Mapping[str, str]],
    arrays: Sequence[Mapping[str, int | str]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    allocation_raw_ids: dict[str, str] = {}
    for array in arrays:
        job_id = str(array["job_id"])
        task_count = int(array["task_count"])
        pattern = re.compile(rf"{re.escape(job_id)}_([0-9]+)")
        task_rows: dict[int, Mapping[str, str]] = {}
        for row in rows:
            # JobID carries Slurm's ArrayJobID_ArrayTaskID display identity.
            # JobIDRaw is the raw numeric JobID assigned to the array element.
            match = pattern.fullmatch(str(row.get("JobID", "")).strip())
            if match is None:
                continue
            raw_job_id = str(row.get("JobIDRaw", "")).strip()
            if not _JOB_ID.fullmatch(raw_job_id):
                raise ValueError(
                    f"Slurm allocation {match.group(0)} has invalid JobIDRaw: "
                    f"{raw_job_id!r}"
                )
            if raw_job_id in allocation_raw_ids:
                raise ValueError(
                    f"Slurm allocations {allocation_raw_ids[raw_job_id]} and "
                    f"{match.group(0)} reuse JobIDRaw {raw_job_id}"
                )
            allocation_raw_ids[raw_job_id] = match.group(0)
            index = int(match.group(1))
            if index in task_rows:
                raise ValueError(f"duplicate sacct allocation row for {job_id}_{index}")
            task_rows[index] = row
        expected = set(range(task_count))
        observed = set(task_rows)
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        if unexpected:
            raise ValueError(
                f"Slurm array {job_id} accounting is contradictory: "
                f"unexpected_count={len(unexpected)}, "
                f"unexpected_first_10={unexpected[:10]}"
            )
        if missing:
            raise _AccountingNotReady(
                f"Slurm array {job_id} accounting is incomplete: "
                f"missing_count={len(missing)}, missing_first_10={missing[:10]}"
            )
        terminal_failures = [
            index
            for index, row in task_rows.items()
            if _clean_state(str(row.get("State", "")))
            in _TERMINAL_FAILURE_STATES
            or (
                _clean_state(str(row.get("State", ""))) == "COMPLETED"
                and str(row.get("ExitCode", "")).strip() != "0:0"
            )
        ]
        if terminal_failures:
            raise ValueError(
                f"Slurm array {job_id} has terminally unsuccessful tasks: "
                f"{terminal_failures[:10]}"
            )
        incomplete = [
            index
            for index, row in task_rows.items()
            if _clean_state(str(row.get("State", ""))) != "COMPLETED"
            or str(row.get("ExitCode", "")).strip() != "0:0"
        ]
        if incomplete:
            raise _AccountingNotReady(
                f"Slurm array {job_id} does not yet show successful tasks: "
                f"{incomplete[:10]}"
            )
        elapsed = [int(str(row["ElapsedRaw"]).strip()) for row in task_rows.values()]
        alloc_cpus = [int(str(row["AllocCPUS"]).strip()) for row in task_rows.values()]
        summaries.append({
            "stage": array["stage"],
            "job_id": job_id,
            "task_count": task_count,
            "completed_task_count": len(task_rows),
            "summed_elapsed_seconds_nonconcurrent": sum(elapsed),
            "summed_allocated_cpu_seconds": sum(
                duration * cpus
                for duration, cpus in zip(elapsed, alloc_cpus, strict=True)
            ),
        })
    return summaries


def _energy_summary(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    values: list[int] = []
    for row in rows:
        job_id = str(row.get("JobID", "")).strip()
        if _ARRAY_ELEMENT_ID.fullmatch(job_id) is None:
            continue
        raw = str(row.get("ConsumedEnergyRaw", "")).strip()
        if raw.isdigit():
            values.append(int(raw))
    return {
        "site_field_exposed": any("ConsumedEnergyRaw" in row for row in rows),
        "numeric_allocation_rows": len(values),
        "summed_consumed_energy_raw_joules": sum(values) if values else None,
        "interpretation": (
            "Slurm/site-reported allocation energy; no imputation. A missing or "
            "zero value is not evidence of zero physical energy use."
        ),
    }


def validate_accounting_payload(
    payload: Mapping[str, Any],
    *,
    kind: str,
    run_tag: str,
    source_commit: str,
    source_tree_sha256: str,
    expected_task_count: int,
) -> dict[str, Any]:
    """Validate a persisted accounting artifact and return its locked summary."""

    unsigned = dict(payload)
    claimed = unsigned.pop("accounting_sha256", None)
    if claimed != _canonical_sha256(unsigned):
        raise ValueError("Slurm accounting self-hash mismatch")
    identity = payload.get("run_identity")
    arrays = payload.get("arrays")
    rows = payload.get("rows")
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "COMPLETE"
        or payload.get("kind") != kind
        or not isinstance(identity, dict)
        or identity.get("run_tag") != run_tag
        or identity.get("source_commit") != source_commit
        or identity.get("source_tree_sha256") != source_tree_sha256
        or not isinstance(arrays, list)
        or not isinstance(rows, list)
    ):
        raise ValueError("Slurm accounting identity or schema mismatch")
    declared = 0
    completed = 0
    job_ids: set[str] = set()
    stages: set[str] = set()
    array_specs: list[dict[str, int | str]] = []
    for array in arrays:
        if not isinstance(array, dict):
            raise ValueError("Slurm accounting array summary is malformed")
        stage = array.get("stage")
        job_id = array.get("job_id")
        task_count = array.get("task_count")
        completed_count = array.get("completed_task_count")
        if (
            not isinstance(stage, str)
            or not stage
            or stage in stages
            or not isinstance(job_id, str)
            or not _JOB_ID.fullmatch(job_id)
            or job_id in job_ids
            or not isinstance(task_count, int)
            or not isinstance(completed_count, int)
            or task_count <= 0
            or completed_count != task_count
        ):
            raise ValueError("Slurm accounting array summary is inconsistent")
        stages.add(stage)
        job_ids.add(job_id)
        array_specs.append({
            "stage": stage,
            "job_id": job_id,
            "task_count": task_count,
        })
        declared += task_count
        completed += completed_count
    if declared != expected_task_count or completed != expected_task_count:
        raise ValueError(
            f"Slurm accounting covers {completed}/{declared} tasks; "
            f"expected {expected_task_count}"
        )
    if payload.get("row_count") != len(rows) or len(rows) < expected_task_count:
        raise ValueError("Slurm accounting row inventory is inconsistent")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Slurm accounting contains a malformed row")
    scheduler = payload.get("scheduler")
    if not isinstance(scheduler, dict):
        raise ValueError("Slurm accounting scheduler evidence is malformed")
    raw_stdout = scheduler.get("raw_stdout")
    selected_fields = scheduler.get("selected_fields")
    if (
        not isinstance(raw_stdout, str)
        or not isinstance(selected_fields, list)
        or not selected_fields
        or any(not isinstance(field, str) for field in selected_fields)
        or len(selected_fields) != len(set(selected_fields))
        or not _REQUIRED_FIELDS.issubset(set(selected_fields))
        or scheduler.get("raw_stdout_sha256")
        != hashlib.sha256(raw_stdout.encode("utf-8")).hexdigest()
    ):
        raise ValueError("Slurm accounting raw-output binding is invalid")
    try:
        reparsed_rows = _parse_rows(
            raw_stdout,
            [(field, field) for field in selected_fields],
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("Slurm accounting raw output cannot be reparsed") from exc
    if reparsed_rows != rows:
        raise ValueError("Slurm accounting rows differ from raw sacct output")
    try:
        reconstructed_arrays = _validate_completed_arrays(rows, array_specs)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"persisted Slurm accounting rows are invalid: {exc}"
        ) from exc
    if reconstructed_arrays != arrays:
        raise ValueError(
            "Slurm accounting array summaries differ from the persisted rows"
        )
    reconstructed_states = Counter(
        _clean_state(str(row.get("State", "")))
        for row in rows
        if _ARRAY_ELEMENT_ID.fullmatch(str(row.get("JobID", "")))
    )
    if dict(sorted(reconstructed_states.items())) != payload.get(
        "task_state_counts"
    ) or payload.get("task_state_counts") != {"COMPLETED": expected_task_count}:
        raise ValueError("Slurm accounting task-state inventory is inconsistent")
    reconstructed_energy = _energy_summary(rows)
    if reconstructed_energy != payload.get("energy"):
        raise ValueError("Slurm accounting energy summary differs from persisted rows")
    return {
        "array_count": len(arrays),
        "completed_simulation_task_count": completed,
        "accounting_row_count": len(rows),
        "energy": reconstructed_energy,
        "accounting_sha256": claimed,
    }


def capture_accounting(
    *,
    submission_receipt: Path,
    output: Path,
    kind: str,
    run_tag: str,
    source_commit: str,
    source_tree_sha256: str,
    attempts: int = _DEFAULT_ATTEMPTS,
    retry_seconds: float = _DEFAULT_RETRY_SECONDS,
    max_retry_seconds: float = _DEFAULT_MAX_RETRY_SECONDS,
    query_timeout_seconds: float = _DEFAULT_QUERY_TIMEOUT_SECONDS,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    if not _HEX40.fullmatch(source_commit):
        raise ValueError("source commit must be a full lowercase SHA-1")
    if not _HEX64.fullmatch(source_tree_sha256):
        raise ValueError("source-tree identity must be a lowercase SHA-256")
    if (
        attempts <= 0
        or not math.isfinite(retry_seconds)
        or not math.isfinite(max_retry_seconds)
        or not math.isfinite(query_timeout_seconds)
        or retry_seconds < 0
        or max_retry_seconds < retry_seconds
        or query_timeout_seconds <= 0
    ):
        raise ValueError("invalid accounting retry policy")
    receipt_path = submission_receipt.absolute()
    receipt = _load_self_hashed_json(receipt_path)
    if (
        receipt.get("run_tag") != run_tag
        or receipt.get("source_commit") != source_commit
        or receipt.get("source_tree_sha256") != source_tree_sha256
    ):
        raise ValueError("submission receipt identity differs from requested run")
    arrays = _expected_arrays(receipt, kind=kind)

    try:
        help_result = runner(
            ["sacct", "--helpformat"],
            check=False,
            capture_output=True,
            text=True,
            timeout=query_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"sacct --helpformat timed out after {query_timeout_seconds:g}s"
        ) from exc
    if help_result.returncode != 0:
        raise RuntimeError(f"sacct --helpformat failed: {help_result.stderr.strip()}")
    selected, missing_optional = _select_fields(help_result.stdout)
    try:
        version_result = runner(
            ["sacct", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=query_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"sacct --version timed out after {query_timeout_seconds:g}s"
        ) from exc
    if version_result.returncode != 0:
        raise RuntimeError(f"sacct --version failed: {version_result.stderr.strip()}")
    job_ids = [str(record["job_id"]) for record in arrays]
    command = [
        "sacct",
        "--noheader",
        "--parsable2",
        "--array",
        "--local",
        "--jobs",
        ",".join(job_ids),
        "--format",
        ",".join(actual for _canonical, actual in selected),
    ]
    rows: list[dict[str, str]] = []
    summaries: list[dict[str, Any]] = []
    raw_stdout = ""
    raw_stderr = ""
    last_error: Exception | None = None
    used_attempts = 0
    retry_delays: list[float] = []
    next_retry_seconds = retry_seconds
    for attempt in range(1, attempts + 1):
        used_attempts = attempt
        try:
            try:
                result = runner(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=query_timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise _AccountingNotReady(
                    f"sacct query timed out after {query_timeout_seconds:g}s"
                ) from exc
            raw_stdout = result.stdout
            raw_stderr = result.stderr
            if result.returncode != 0:
                raise _AccountingNotReady(
                    f"sacct returned {result.returncode}: {result.stderr.strip()}"
                )
            rows = _parse_rows(result.stdout, selected)
            summaries = _validate_completed_arrays(rows, arrays)
            last_error = None
            break
        except _AccountingNotReady as exc:
            last_error = exc
            if attempt < attempts:
                delay = min(next_retry_seconds, max_retry_seconds)
                retry_delays.append(delay)
                print(
                    f"[sacct] attempt {attempt}/{attempts} incomplete: {exc}; "
                    f"retrying in {delay:g}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
                next_retry_seconds = min(
                    delay * _RETRY_BACKOFF_MULTIPLIER,
                    max_retry_seconds,
                )
    if last_error is not None:
        raise RuntimeError(
            f"Slurm accounting remained incomplete after {attempts} attempts: "
            f"{last_error}"
        ) from last_error

    task_states = Counter(
        _clean_state(str(row.get("State", "")))
        for row in rows
        if _ARRAY_ELEMENT_ID.fullmatch(str(row.get("JobID", "")).strip())
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "scope": "completed simulation-array scheduler accounting",
        "kind": kind,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_identity": {
            "run_tag": run_tag,
            "source_commit": source_commit,
            "source_tree_sha256": source_tree_sha256,
            "submission_receipt_sha256": hashlib.sha256(
                receipt_path.read_bytes()
            ).hexdigest(),
        },
        "scheduler": {
            "sacct_version_stdout": version_result.stdout,
            "sacct_version_stderr": version_result.stderr,
            "sacct_version_returncode": version_result.returncode,
            "selected_fields": [canonical for canonical, _actual in selected],
            "actual_requested_fields": [actual for _canonical, actual in selected],
            "missing_optional_fields": missing_optional,
            "command": command,
            "attempts": used_attempts,
            "retry_policy": {
                "maximum_attempts": attempts,
                "initial_delay_seconds": retry_seconds,
                "backoff_multiplier": _RETRY_BACKOFF_MULTIPLIER,
                "maximum_delay_seconds": max_retry_seconds,
                "query_timeout_seconds": query_timeout_seconds,
                "applied_delays_seconds": retry_delays,
                "total_wait_seconds": sum(retry_delays),
            },
            "raw_stdout": raw_stdout,
            "raw_stdout_sha256": hashlib.sha256(raw_stdout.encode("utf-8")).hexdigest(),
            "raw_stderr": raw_stderr,
        },
        "arrays": summaries,
        "task_state_counts": dict(sorted(task_states.items())),
        "energy": _energy_summary(rows),
        "row_count": len(rows),
        "rows": rows,
    }
    payload["accounting_sha256"] = _canonical_sha256(payload)
    validate_accounting_payload(
        payload,
        kind=kind,
        run_tag=run_tag,
        source_commit=source_commit,
        source_tree_sha256=source_tree_sha256,
        expected_task_count=sum(int(record["task_count"]) for record in arrays),
    )

    destination = output.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite Slurm accounting: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    try:
        temporary.replace(destination)
        reread = json.loads(destination.read_text(encoding="utf-8"))
        claimed = reread.pop("accounting_sha256", None)
        if claimed != _canonical_sha256(reread):
            raise RuntimeError("Slurm accounting failed atomic readback validation")
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("core", "structural"), required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--attempts", type=int, default=_DEFAULT_ATTEMPTS)
    parser.add_argument(
        "--retry-seconds", type=float, default=_DEFAULT_RETRY_SECONDS,
        help="initial delay between incomplete sacct queries",
    )
    parser.add_argument(
        "--max-retry-seconds", type=float,
        default=_DEFAULT_MAX_RETRY_SECONDS,
        help="cap for the exponential accounting-retry delay",
    )
    parser.add_argument(
        "--query-timeout-seconds", type=float,
        default=_DEFAULT_QUERY_TIMEOUT_SECONDS,
        help="timeout for each sacct subprocess",
    )
    args = parser.parse_args()
    payload = capture_accounting(
        submission_receipt=args.submission_receipt,
        output=args.output,
        kind=args.kind,
        run_tag=args.run_tag,
        source_commit=args.source_commit,
        source_tree_sha256=args.source_tree_sha256,
        attempts=args.attempts,
        retry_seconds=args.retry_seconds,
        max_retry_seconds=args.max_retry_seconds,
        query_timeout_seconds=args.query_timeout_seconds,
    )
    print(json.dumps({
        "status": payload["status"],
        "row_count": payload["row_count"],
        "arrays": payload["arrays"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
