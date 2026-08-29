#!/usr/bin/env python3
"""Bind a held combined-evidence finalizer to its exact Slurm dependency."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
RECEIPT_TYPE = "held_finalizer_submission_authorization"
WORKFLOW = "sbatch_hold_then_authorization_then_release"
_JOB_ID_RE = re.compile(r"[1-9][0-9]*")
_DEPENDENCY_SUFFIX_RE = re.compile(r"\((?:unfulfilled|satisfied)\)")


def canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_job_id(value: str, label: str) -> str:
    job_id = str(value).strip()
    if _JOB_ID_RE.fullmatch(job_id) is None:
        raise ValueError(f"{label} must be one positive numeric Slurm job id")
    return job_id


def _field(record: str, name: str) -> str:
    match = re.search(rf"(?:^|\s){re.escape(name)}=(\S+)", record)
    if match is None:
        raise ValueError(f"scontrol record lacks {name}")
    return match.group(1)


def _afterok_job_ids(dependency: str) -> list[str]:
    normalized = _DEPENDENCY_SUFFIX_RE.sub("", dependency)
    if not normalized or normalized == "(null)":
        raise ValueError("finalizer has no dependency")
    job_ids: list[str] = []
    for clause in normalized.split(","):
        if not clause.startswith("afterok:"):
            raise ValueError("finalizer dependency is not exclusively afterok")
        values = clause.removeprefix("afterok:").split(":")
        if not values or any(_JOB_ID_RE.fullmatch(value) is None for value in values):
            raise ValueError("finalizer afterok dependency contains an invalid job id")
        job_ids.extend(values)
    if len(job_ids) != 2 or len(set(job_ids)) != 2:
        raise ValueError("finalizer must depend on exactly two distinct afterok jobs")
    return job_ids


def _validate_held_record(
    record: str,
    *,
    finalizer_job_id: str,
    core_publisher_job_id: str,
    structural_publisher_job_id: str,
) -> dict[str, Any]:
    nonempty_lines = [line.strip() for line in record.splitlines() if line.strip()]
    if len(nonempty_lines) != 1:
        raise ValueError("scontrol must return exactly one finalizer allocation record")
    line = nonempty_lines[0]
    observed_job_id = _field(line, "JobId")
    state = _field(line, "JobState")
    reason = _field(line, "Reason")
    dependency = _field(line, "Dependency")
    dependency_job_ids = _afterok_job_ids(dependency)
    if observed_job_id != finalizer_job_id:
        raise ValueError("scontrol returned a different finalizer job id")
    if state != "PENDING" or reason != "JobHeldUser":
        raise ValueError("finalizer is not PENDING with Reason=JobHeldUser")
    expected_ids = {core_publisher_job_id, structural_publisher_job_id}
    if set(dependency_job_ids) != expected_ids:
        raise ValueError("finalizer afterok dependency does not name the two publishers")
    return {
        "job_id": observed_job_id,
        "job_state": state,
        "reason": reason,
        "dependency_literal": dependency,
        "dependency_type": "afterok",
        "afterok_job_ids": dependency_job_ids,
    }


def _validate_held_publisher_record(record: str, *, job_id: str) -> dict[str, Any]:
    nonempty_lines = [line.strip() for line in record.splitlines() if line.strip()]
    if len(nonempty_lines) != 1:
        raise ValueError("scontrol must return exactly one publisher allocation record")
    line = nonempty_lines[0]
    observed_job_id = _field(line, "JobId")
    state = _field(line, "JobState")
    reason = _field(line, "Reason")
    if observed_job_id != job_id:
        raise ValueError("scontrol returned a different publisher job id")
    if state != "PENDING" or reason != "JobHeldUser":
        raise ValueError("publisher is not PENDING with Reason=JobHeldUser")
    return {"job_id": observed_job_id, "job_state": state, "reason": reason}


def _default_runner(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess:
    return subprocess.run(command, **kwargs)  # noqa: S603 - fixed scheduler command


def _query_held_finalizer(
    *,
    finalizer_job_id: str,
    core_publisher_job_id: str,
    structural_publisher_job_id: str,
    scontrol_bin: str = "scontrol",
    runner: Callable[..., subprocess.CompletedProcess] = _default_runner,
) -> tuple[str, dict[str, Any]]:
    if not str(scontrol_bin).strip():
        raise ValueError("scontrol executable cannot be empty")
    command = [str(scontrol_bin), "show", "job", "-o", finalizer_job_id]
    completed = runner(
        command,
        check=False,
        capture_output=True,
        text=False,
        timeout=30,
    )
    if completed.returncode != 0:
        stderr = bytes(completed.stderr or b"").decode("utf-8", errors="replace").strip()
        raise ValueError(f"scontrol query failed for finalizer {finalizer_job_id}: {stderr}")
    stdout_bytes = bytes(completed.stdout or b"")
    stderr_bytes = bytes(completed.stderr or b"")
    if stderr_bytes.strip():
        raise ValueError("successful scontrol query unexpectedly wrote stderr")
    try:
        stdout = stdout_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("scontrol output is not valid UTF-8") from exc
    parsed = _validate_held_record(
        stdout,
        finalizer_job_id=finalizer_job_id,
        core_publisher_job_id=core_publisher_job_id,
        structural_publisher_job_id=structural_publisher_job_id,
    )
    return stdout, parsed


def _query_held_publisher(
    *,
    job_id: str,
    scontrol_bin: str = "scontrol",
    runner: Callable[..., subprocess.CompletedProcess] = _default_runner,
) -> tuple[str, dict[str, Any]]:
    command = [str(scontrol_bin), "show", "job", "-o", job_id]
    completed = runner(command, check=False, capture_output=True, text=False, timeout=30)
    if completed.returncode != 0:
        stderr = bytes(completed.stderr or b"").decode("utf-8", errors="replace").strip()
        raise ValueError(f"scontrol query failed for publisher {job_id}: {stderr}")
    stdout_bytes = bytes(completed.stdout or b"")
    if bytes(completed.stderr or b"").strip():
        raise ValueError("successful scontrol query unexpectedly wrote stderr")
    try:
        stdout = stdout_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("scontrol output is not valid UTF-8") from exc
    return stdout, _validate_held_publisher_record(stdout, job_id=job_id)


def _literal_record(scontrol_bin: str, job_id: str, stdout: str, parsed: dict) -> dict:
    stdout_bytes = stdout.encode("utf-8")
    return {
        "command": [str(scontrol_bin), "show", "job", "-o", job_id],
        "stdout": stdout,
        "stdout_bytes": len(stdout_bytes),
        "stdout_literal_sha256": hashlib.sha256(stdout_bytes).hexdigest(),
        "parsed": parsed,
    }


def build_authorization(
    *,
    finalizer_job_id: str,
    core_publisher_job_id: str,
    structural_publisher_job_id: str,
    scontrol_bin: str = "scontrol",
    runner: Callable[..., subprocess.CompletedProcess] = _default_runner,
) -> dict[str, Any]:
    finalizer = _require_job_id(finalizer_job_id, "finalizer job id")
    core = _require_job_id(core_publisher_job_id, "core publisher job id")
    structural = _require_job_id(structural_publisher_job_id, "structural publisher job id")
    if len({finalizer, core, structural}) != 3:
        raise ValueError("finalizer and publisher Slurm job ids must be distinct")
    finalizer_stdout, finalizer_parsed = _query_held_finalizer(
        finalizer_job_id=finalizer,
        core_publisher_job_id=core,
        structural_publisher_job_id=structural,
        scontrol_bin=scontrol_bin,
        runner=runner,
    )
    core_stdout, core_parsed = _query_held_publisher(
        job_id=core, scontrol_bin=scontrol_bin, runner=runner
    )
    structural_stdout, structural_parsed = _query_held_publisher(
        job_id=structural, scontrol_bin=scontrol_bin, runner=runner
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": RECEIPT_TYPE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_workflow": WORKFLOW,
        "simulation_rerun": False,
        "finalizer_slurm_job_id": finalizer,
        "recovery_publisher_slurm_job_ids": {
            "core": core,
            "structural": structural,
        },
        "required_scheduler_state": {
            "job_state": "PENDING",
            "reason": "JobHeldUser",
            "dependency_type": "afterok",
            "afterok_job_ids": [core, structural],
        },
        "observed_held_scheduler_records": {
            "core": _literal_record(scontrol_bin, core, core_stdout, core_parsed),
            "structural": _literal_record(
                scontrol_bin, structural, structural_stdout, structural_parsed
            ),
            "finalizer": _literal_record(
                scontrol_bin, finalizer, finalizer_stdout, finalizer_parsed
            ),
        },
    }
    payload["authorization_sha256"] = canonical_sha256(payload)
    return payload


def validate_authorization(
    payload: Mapping[str, Any],
    *,
    finalizer_job_id: str,
    core_publisher_job_id: str,
    structural_publisher_job_id: str,
    require_live_held: bool = False,
    scontrol_bin: str = "scontrol",
    runner: Callable[..., subprocess.CompletedProcess] = _default_runner,
) -> dict[str, Any]:
    finalizer = _require_job_id(finalizer_job_id, "finalizer job id")
    core = _require_job_id(core_publisher_job_id, "core publisher job id")
    structural = _require_job_id(structural_publisher_job_id, "structural publisher job id")
    expected_top_level = {
        "schema_version",
        "receipt_type",
        "created_at_utc",
        "authorization_workflow",
        "simulation_rerun",
        "finalizer_slurm_job_id",
        "recovery_publisher_slurm_job_ids",
        "required_scheduler_state",
        "observed_held_scheduler_records",
        "authorization_sha256",
    }
    if set(payload) != expected_top_level:
        raise ValueError("finalizer authorization has ambiguous top-level fields")
    if payload.get("schema_version") != SCHEMA_VERSION or (
        payload.get("receipt_type") != RECEIPT_TYPE
    ):
        raise ValueError("unsupported finalizer authorization schema")
    if payload.get("authorization_workflow") != WORKFLOW:
        raise ValueError("unexpected finalizer authorization workflow")
    if payload.get("simulation_rerun") is not False:
        raise ValueError("finalizer authorization must attest simulation_rerun=false")
    if payload.get("finalizer_slurm_job_id") != finalizer or (
        payload.get("recovery_publisher_slurm_job_ids") != {"core": core, "structural": structural}
    ):
        raise ValueError("finalizer authorization names different Slurm jobs")
    claimed_hash = payload.get("authorization_sha256")
    if not isinstance(claimed_hash, str) or re.fullmatch(r"[0-9a-f]{64}", claimed_hash) is None:
        raise ValueError("finalizer authorization has an invalid self-hash")
    unsigned = dict(payload)
    unsigned.pop("authorization_sha256")
    if canonical_sha256(unsigned) != claimed_hash:
        raise ValueError("finalizer authorization self-hash mismatch")

    required = payload.get("required_scheduler_state")
    expected_required = {
        "job_state": "PENDING",
        "reason": "JobHeldUser",
        "dependency_type": "afterok",
        "afterok_job_ids": [core, structural],
    }
    if required != expected_required:
        raise ValueError("finalizer authorization has a different required scheduler state")
    observed_records = payload.get("observed_held_scheduler_records")
    if not isinstance(observed_records, Mapping) or set(observed_records) != {
        "core",
        "structural",
        "finalizer",
    }:
        raise ValueError("finalizer authorization has invalid scheduler records")
    for role, job_id in (("core", core), ("structural", structural), ("finalizer", finalizer)):
        observed = observed_records.get(role)
        if not isinstance(observed, Mapping) or set(observed) != {
            "command", "stdout", "stdout_bytes", "stdout_literal_sha256", "parsed"
        }:
            raise ValueError(f"finalizer authorization has an invalid {role} scheduler record")
        stdout = observed.get("stdout")
        if not isinstance(stdout, str):
            raise ValueError(f"{role} scheduler stdout must be text")
        stdout_bytes = stdout.encode("utf-8")
        if observed.get("command") != [str(scontrol_bin), "show", "job", "-o", job_id]:
            raise ValueError(f"authorization records a different {role} scontrol command")
        if observed.get("stdout_bytes") != len(stdout_bytes) or (
            observed.get("stdout_literal_sha256") != hashlib.sha256(stdout_bytes).hexdigest()
        ):
            raise ValueError(f"{role} scheduler record literal binding mismatch")
        if role == "finalizer":
            reparsed = _validate_held_record(
                stdout,
                finalizer_job_id=finalizer,
                core_publisher_job_id=core,
                structural_publisher_job_id=structural,
            )
        else:
            reparsed = _validate_held_publisher_record(stdout, job_id=job_id)
        if observed.get("parsed") != reparsed:
            raise ValueError(f"{role} scheduler parsed record differs from literal stdout")
    if require_live_held:
        _query_held_publisher(job_id=core, scontrol_bin=scontrol_bin, runner=runner)
        _query_held_publisher(job_id=structural, scontrol_bin=scontrol_bin, runner=runner)
        _query_held_finalizer(
            finalizer_job_id=finalizer,
            core_publisher_job_id=core,
            structural_publisher_job_id=structural,
            scontrol_bin=scontrol_bin,
            runner=runner,
        )
    return dict(payload)


def _require_plain_absolute_file(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("finalizer authorization path must be absolute")
    if path.is_symlink() or not path.is_file():
        raise ValueError("finalizer authorization must be a non-symlink regular file")
    for parent in path.parents:
        if parent.is_symlink():
            raise ValueError("finalizer authorization has a symlinked parent")
    return path


def validate_authorization_file(
    path: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    receipt = _require_plain_absolute_file(path)
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read finalizer authorization: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("finalizer authorization must be one JSON object")
    return validate_authorization(payload, **kwargs)


def _write_new_authorization(path: Path, payload: Mapping[str, Any]) -> None:
    if not path.is_absolute():
        raise ValueError("finalizer authorization output must be absolute")
    if path.exists() or path.is_symlink():
        raise ValueError("finalizer authorization output already exists")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError("finalizer authorization parent must be a non-symlink directory")
    for parent in path.parents:
        if parent.is_symlink():
            raise ValueError("finalizer authorization output has a symlinked parent")
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("create", "validate"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--receipt", type=Path, required=True)
        sub.add_argument("--finalizer-job-id", required=True)
        sub.add_argument("--core-publisher-job-id", required=True)
        sub.add_argument("--structural-publisher-job-id", required=True)
        sub.add_argument("--scontrol-bin", default="scontrol")
        if command == "validate":
            sub.add_argument("--require-live-held", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = {
        "finalizer_job_id": args.finalizer_job_id,
        "core_publisher_job_id": args.core_publisher_job_id,
        "structural_publisher_job_id": args.structural_publisher_job_id,
        "scontrol_bin": args.scontrol_bin,
    }
    try:
        if args.command == "create":
            payload = build_authorization(**common)
            _write_new_authorization(args.receipt, payload)
            print(f"Saved held-finalizer authorization: {args.receipt}")
        else:
            validate_authorization_file(
                args.receipt,
                require_live_held=args.require_live_held,
                **common,
            )
            print("Held-finalizer authorization OK")
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"BLOCK: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
