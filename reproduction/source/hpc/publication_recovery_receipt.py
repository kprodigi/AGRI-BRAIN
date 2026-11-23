#!/usr/bin/env python3
"""Create and validate a fail-closed publication-recovery authorization.

The normal fresh-run publishers do not use this module.  It exists only for
the narrow case in which all simulation workers completed but their dependent
publisher failed.  The replacement publisher must first be submitted with
``sbatch --hold``.  Its job id is then sealed into this receipt before the job
is released, removing the job-id/receipt creation race without authorizing a
simulation rerun.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hpc.core_submission_receipt import (
    SNAPSHOT_MODE,
    validate_receipt_payload as validate_core_submission_receipt,
)
from hpc.capture_failed_publisher_accounting import (
    validate_failed_publisher_accounting,
)
from hpc.preserved_raw_manifest import validate_manifest_document


SCHEMA_VERSION = 1
RECEIPT_TYPE = "publication_recovery_authorization"
RECOVERY_KINDS = ("core", "structural")
HELD_WORKFLOW = "sbatch_hold_then_receipt_then_release"
RECOVERY_REASON_CODE = "terminal_failed_publisher_publication_only_recovery"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_JOB_ID = re.compile(r"^[1-9][0-9]*$")
_RUN_TAG = re.compile(r"^(?:sensitivity_)?([0-9a-f]{7})_[0-9]{8}_[0-9]{6}$")
_REASON_CODE = re.compile(r"^[a-z][a-z0-9_]{2,95}$")


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 of strict canonical UTF-8 JSON."""

    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _require_hex40(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX40.fullmatch(value) is None:
        raise ValueError(f"{label} must be a full lowercase Git SHA-1")
    return value


def _require_hex64(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_job_id(value: object, label: str) -> str:
    if not isinstance(value, str) or _JOB_ID.fullmatch(value) is None:
        raise ValueError(f"{label} must be a positive decimal Slurm job id")
    return value


def _unresolved_safe_path(path: Path, *, label: str) -> Path:
    """Return an absolute path without resolving away any symlink component."""

    candidate = path.absolute()
    cursor = candidate
    while True:
        if cursor.is_symlink():
            raise ValueError(f"{label} traverses a symbolic link: {cursor}")
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent
    return candidate


def _read_stable_regular_file(
    path: Path, *, label: str, allow_empty: bool = True,
) -> bytes:
    """Read literal bytes while rejecting links and concurrent replacement."""

    candidate = _unresolved_safe_path(path, label=label)
    if not candidate.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {candidate}")
    before = candidate.stat()
    data = candidate.read_bytes()
    after = candidate.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or len(data) != after.st_size:
        raise ValueError(f"{label} changed while it was being read: {candidate}")
    if not allow_empty and not data:
        raise ValueError(f"{label} must not be empty: {candidate}")
    return data


def _load_json_bytes(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain exactly one JSON object")
    return payload


def _validate_timestamp(value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("recovery receipt lacks created_at_utc")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("created_at_utc is not a valid ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("created_at_utc must be timezone aware")


def _validate_structural_submission_receipt(
    payload: Mapping[str, Any], *, run_tag: str, source_commit: str,
) -> dict[str, Any]:
    """Validate the complete structural submission-only DAG contract."""

    if payload.get("schema_version") != 2:
        raise ValueError("structural submission receipt schema must be 2")
    if payload.get("analysis_label") != "structural sensitivity":
        raise ValueError("structural submission receipt has the wrong analysis label")
    if (
        payload.get("receipt_scope")
        != "submission_only_not_scheduler_completion"
        or payload.get("scheduler_completion_attested") is not False
    ):
        raise ValueError("structural receipt must remain submission-only")
    if payload.get("run_tag") != run_tag or payload.get("source_commit") != source_commit:
        raise ValueError("structural submission receipt has the wrong run identity")
    if payload.get("source_snapshot_mode") != SNAPSHOT_MODE:
        raise ValueError("structural submission receipt has the wrong snapshot mode")
    _require_hex64(
        payload.get("source_tree_sha256"),
        "structural source-tree digest",
    )
    if payload.get("task_count") != 3_000:
        raise ValueError("structural submission receipt must cover 3,000 tasks")
    chunk_limit = payload.get("array_chunk_size_limit")
    concurrency = payload.get("max_concurrent_per_array")
    if (
        not isinstance(chunk_limit, int)
        or isinstance(chunk_limit, bool)
        or not 1 <= chunk_limit <= 1_000
    ):
        raise ValueError("structural array chunk limit is invalid")
    if (
        not isinstance(concurrency, int)
        or isinstance(concurrency, bool)
        or concurrency <= 0
    ):
        raise ValueError("structural array concurrency is invalid")

    arrays = payload.get("task_arrays")
    if not isinstance(arrays, list) or not arrays:
        raise ValueError("structural submission receipt lacks task arrays")
    expected_offset = 0
    previous_job_id: str | None = None
    job_ids: list[str] = []
    for record in arrays:
        if not isinstance(record, dict):
            raise ValueError("structural task-array record must be an object")
        if set(record) != {"job_id", "offset", "count", "afterok_job_id"}:
            raise ValueError("structural task-array record has ambiguous fields")
        job_id = _require_job_id(record.get("job_id"), "structural array job id")
        offset = record.get("offset")
        count = record.get("count")
        if offset != expected_offset:
            raise ValueError("structural task-array offsets are not contiguous")
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or not 1 <= count <= chunk_limit
        ):
            raise ValueError("structural task-array count is invalid")
        if record.get("afterok_job_id") != previous_job_id:
            raise ValueError("structural task arrays are not one afterok chain")
        job_ids.append(job_id)
        previous_job_id = job_id
        expected_offset += count
    if expected_offset != 3_000:
        raise ValueError("structural task arrays do not cover 0..2,999")
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("structural submission receipt reuses an array job id")

    publisher = payload.get("publisher")
    if not isinstance(publisher, dict) or set(publisher) != {
        "job_id",
        "afterok_job_id",
    }:
        raise ValueError("structural publisher record is malformed")
    publisher_job_id = _require_job_id(
        publisher.get("job_id"), "structural publisher job id"
    )
    if publisher_job_id in set(job_ids):
        raise ValueError("structural publisher reuses an array job id")
    if publisher.get("afterok_job_id") != previous_job_id:
        raise ValueError("structural publisher is not afterok the final array")

    unsigned = dict(payload)
    claimed = unsigned.pop("receipt_sha256", None)
    if claimed != canonical_sha256(unsigned):
        raise ValueError("structural submission receipt self-hash is invalid")
    return dict(payload)


def _validate_original_submission(
    data: bytes,
    *,
    kind: str,
    run_tag: str,
    simulation_commit: str,
) -> tuple[dict[str, Any], str]:
    payload = _load_json_bytes(data, label="original submission receipt")
    if kind == "core":
        validated = validate_core_submission_receipt(payload)
        if validated.get("run_tag") != run_tag:
            raise ValueError("original core receipt has the wrong run tag")
        if validated.get("source_commit") != simulation_commit:
            raise ValueError("original core receipt has the wrong source commit")
        publisher = validated["slurm_dag"]["publisher"]
    elif kind == "structural":
        validated = _validate_structural_submission_receipt(
            payload, run_tag=run_tag, source_commit=simulation_commit
        )
        publisher = validated["publisher"]
    else:
        raise ValueError(f"unsupported recovery kind: {kind!r}")
    return validated, _require_job_id(
        publisher.get("job_id"), "original publisher job id"
    )


def _failed_accounting_binding(
    path: Path, *, expected_job_id: str,
) -> dict[str, Any]:
    data = _read_stable_regular_file(
        path, label="failed publisher accounting record", allow_empty=False
    )
    record = _load_json_bytes(data, label="failed publisher accounting record")
    validated = validate_failed_publisher_accounting(
        record, expected_job_id=expected_job_id
    )
    return {
        "file": path.name,
        "bytes": len(data),
        "literal_sha256": hashlib.sha256(data).hexdigest(),
        "accounting_sha256": validated["accounting_sha256"],
        "record_utf8": data.decode("utf-8"),
        "record": record,
    }


def _file_hash_binding(path: Path, *, label: str) -> dict[str, Any]:
    data = _read_stable_regular_file(path, label=label)
    return {
        "file": path.name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "hash_semantics": "literal_bytes",
    }


def _expected_failed_log_paths(
    *, kind: str, original_receipt_path: Path, publisher_job_id: str,
) -> tuple[Path, Path]:
    """Return the canonical stdout/stderr paths declared by the launchers."""

    receipt = _unresolved_safe_path(
        original_receipt_path, label="original submission receipt"
    )
    if kind == "core":
        # <snapshot>/mvp/simulation/results/core_submission_receipts/<tag>.json
        try:
            run_root = receipt.parents[4]
        except IndexError as exc:  # pragma: no cover - defensive on exotic paths
            raise ValueError("core submission receipt path is too shallow") from exc
        expected_receipt_parent = (
            run_root / "mvp" / "simulation" / "results" / "core_submission_receipts"
        )
        if receipt.parent != expected_receipt_parent:
            raise ValueError(
                "core submission receipt is not at its canonical snapshot path"
            )
    elif kind == "structural":
        if receipt.name != "slurm_submission.json":
            raise ValueError("structural submission receipt filename is not canonical")
        run_root = receipt.parent
    else:  # validated earlier, retained fail-closed for direct callers
        raise ValueError(f"unsupported recovery kind: {kind!r}")
    logs = run_root / "logs"
    return (
        logs / f"publish_{publisher_job_id}.out",
        logs / f"publish_{publisher_job_id}.err",
    )


def _require_canonical_failed_logs(
    *,
    kind: str,
    original_receipt_path: Path,
    publisher_job_id: str,
    failed_stdout_path: Path,
    failed_stderr_path: Path,
) -> None:
    expected_stdout, expected_stderr = _expected_failed_log_paths(
        kind=kind,
        original_receipt_path=original_receipt_path,
        publisher_job_id=publisher_job_id,
    )
    supplied_stdout = _unresolved_safe_path(
        failed_stdout_path, label="failed publisher stdout"
    )
    supplied_stderr = _unresolved_safe_path(
        failed_stderr_path, label="failed publisher stderr"
    )
    if supplied_stdout != expected_stdout or supplied_stderr != expected_stderr:
        raise ValueError(
            "failed publisher logs do not match the canonical launcher paths for "
            f"job {publisher_job_id}"
        )


def _raw_manifest_binding(
    path: Path,
    *,
    kind: str,
    run_tag: str,
    simulation_commit: str,
    simulation_source_tree_sha256: str,
) -> dict[str, Any]:
    data = _read_stable_regular_file(
        path, label="preserved raw-output manifest", allow_empty=False
    )
    manifest = validate_manifest_document(
        _load_json_bytes(data, label="preserved raw-output manifest"),
        kind=kind,
        run_tag=run_tag,
        simulation_commit=simulation_commit,
        simulation_source_tree_sha256=simulation_source_tree_sha256,
    )
    records = manifest["files"]
    return {
        "file": path.name,
        "bytes": len(data),
        "literal_sha256": hashlib.sha256(data).hexdigest(),
        "manifest_self_hash": manifest["manifest_sha256"],
        "record_count": len(records),
        "normalized_record_set_sha256": canonical_sha256(records),
        "payload_merkle_root": manifest["payload_merkle_root"],
        "hash_semantics": "preserved_simulation_raw_output_manifest_v1",
    }


def _git_clean_identity(repo_root: Path) -> tuple[str, str]:
    unresolved = _unresolved_safe_path(
        repo_root, label="publication-repair repository"
    )
    root = unresolved.resolve(strict=True)
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tree = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot verify clean publication-repair checkout") from exc
    if Path(top).resolve() != root:
        raise RuntimeError("--repo-root is not the Git worktree root")
    _require_hex40(commit, "publication-repair commit")
    _require_hex40(tree, "publication-repair tree")
    if status:
        raise RuntimeError("publication-repair checkout is dirty")
    return commit, tree


def build_recovery_receipt(
    *,
    kind: str,
    run_tag: str,
    simulation_commit: str,
    publication_commit: str,
    publication_tree: str,
    original_receipt_file: str,
    original_receipt_literal_sha256: str,
    original_receipt_self_hash: str,
    original_publisher_job_id: str,
    failed_accounting: Mapping[str, Any],
    failed_stdout: Mapping[str, Any],
    failed_stderr: Mapping[str, Any],
    raw_output_manifest: Mapping[str, Any],
    held_recovery_publisher_job_id: str,
    reason_code: str,
) -> dict[str, Any]:
    """Build a recovery receipt from already verified evidence bindings."""

    if kind not in RECOVERY_KINDS:
        raise ValueError(f"unsupported recovery kind: {kind!r}")
    simulation_commit = _require_hex40(simulation_commit, "simulation commit")
    publication_commit = _require_hex40(publication_commit, "publication commit")
    publication_tree = _require_hex40(publication_tree, "publication tree")
    if publication_commit == simulation_commit:
        raise ValueError("recovery publication commit must differ from simulation commit")
    match = _RUN_TAG.fullmatch(run_tag)
    if match is None or match.group(1) != simulation_commit[:7]:
        raise ValueError("run tag is not bound to the simulation commit")
    if reason_code != RECOVERY_REASON_CODE:
        raise ValueError(
            "reason_code must be the cause-neutral terminal-failure recovery code"
        )
    original_publisher_job_id = _require_job_id(
        original_publisher_job_id, "original publisher job id"
    )
    recovery_job_id = _require_job_id(
        held_recovery_publisher_job_id, "held recovery publisher job id"
    )
    if recovery_job_id == original_publisher_job_id:
        raise ValueError("recovery publisher must have a new Slurm job id")
    _require_hex64(
        original_receipt_literal_sha256, "original receipt literal digest"
    )
    _require_hex64(original_receipt_self_hash, "original receipt self-hash")

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": RECEIPT_TYPE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "recovery_kind": kind,
        "reason_code": reason_code,
        "simulation_rerun": False,
        "run_tag": run_tag,
        "source_identity": {
            "simulation_commit": simulation_commit,
            "publication_repair_commit": publication_commit,
            "publication_repair_tree": publication_tree,
            "publication_repair_checkout_clean": True,
        },
        "original_submission_receipt": {
            "file": original_receipt_file,
            "literal_sha256": original_receipt_literal_sha256,
            "receipt_sha256": original_receipt_self_hash,
            "publisher_job_id": original_publisher_job_id,
            "preserved_without_mutation": True,
        },
        "failed_publisher": {
            "job_id": original_publisher_job_id,
            "accounting_record": dict(failed_accounting),
            "stdout": dict(failed_stdout),
            "stderr": dict(failed_stderr),
        },
        "preserved_raw_outputs": dict(raw_output_manifest),
        "recovery_publisher": {
            "job_id": recovery_job_id,
            "held_at_receipt_creation": True,
            "authorization_workflow": HELD_WORKFLOW,
        },
    }
    validate_recovery_receipt_payload(payload, require_self_hash=False)
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _validate_log_binding(
    binding: object, *, label: str, expected_file: str,
) -> None:
    if not isinstance(binding, dict) or set(binding) != {
        "file",
        "bytes",
        "sha256",
        "hash_semantics",
    }:
        raise ValueError(f"{label} binding is malformed")
    if binding["file"] != expected_file:
        raise ValueError(f"{label} filename is not bound to the failed job id")
    if (
        not isinstance(binding["bytes"], int)
        or isinstance(binding["bytes"], bool)
        or binding["bytes"] < 0
    ):
        raise ValueError(f"{label} byte count is invalid")
    _require_hex64(binding["sha256"], f"{label} digest")
    if binding["hash_semantics"] != "literal_bytes":
        raise ValueError(f"{label} does not use literal-byte hashing")


def validate_recovery_receipt_payload(
    payload: Mapping[str, Any],
    *,
    expected_kind: str | None = None,
    expected_run_tag: str | None = None,
    expected_simulation_commit: str | None = None,
    expected_publication_commit: str | None = None,
    expected_recovery_job_id: str | None = None,
    require_self_hash: bool = True,
) -> dict[str, Any]:
    """Validate the self-contained recovery authorization contract."""

    expected_keys = {
        "schema_version",
        "receipt_type",
        "created_at_utc",
        "recovery_kind",
        "reason_code",
        "simulation_rerun",
        "run_tag",
        "source_identity",
        "original_submission_receipt",
        "failed_publisher",
        "preserved_raw_outputs",
        "recovery_publisher",
    }
    if require_self_hash:
        expected_keys.add("receipt_sha256")
    if set(payload) != expected_keys:
        raise ValueError("recovery receipt fields do not match its schema")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("recovery receipt schema must be 1")
    if payload.get("receipt_type") != RECEIPT_TYPE:
        raise ValueError("receipt is not a publication-recovery authorization")
    _validate_timestamp(payload.get("created_at_utc"))
    kind = payload.get("recovery_kind")
    if kind not in RECOVERY_KINDS:
        raise ValueError("recovery receipt kind is invalid")
    if expected_kind is not None and kind != expected_kind:
        raise ValueError("recovery receipt has the wrong kind")
    reason = payload.get("reason_code")
    if reason != RECOVERY_REASON_CODE:
        raise ValueError("recovery receipt reason code is not cause-neutral")
    if payload.get("simulation_rerun") is not False:
        raise ValueError("publication recovery must explicitly prohibit simulation reruns")

    run_tag = payload.get("run_tag")
    source = payload.get("source_identity")
    if not isinstance(source, dict) or set(source) != {
        "simulation_commit",
        "publication_repair_commit",
        "publication_repair_tree",
        "publication_repair_checkout_clean",
    }:
        raise ValueError("recovery source identity is malformed")
    simulation_commit = _require_hex40(
        source.get("simulation_commit"), "simulation commit"
    )
    publication_commit = _require_hex40(
        source.get("publication_repair_commit"), "publication-repair commit"
    )
    _require_hex40(source.get("publication_repair_tree"), "publication-repair tree")
    if source.get("publication_repair_checkout_clean") is not True:
        raise ValueError("publication-repair checkout was not clean")
    if publication_commit == simulation_commit:
        raise ValueError("recovery receipt is not dual provenance")
    match = _RUN_TAG.fullmatch(str(run_tag))
    if match is None or match.group(1) != simulation_commit[:7]:
        raise ValueError("recovery run tag is not bound to simulation code")
    if expected_run_tag is not None and run_tag != expected_run_tag:
        raise ValueError("recovery receipt has the wrong run tag")
    if (
        expected_simulation_commit is not None
        and simulation_commit != expected_simulation_commit
    ):
        raise ValueError("recovery receipt has the wrong simulation commit")
    if (
        expected_publication_commit is not None
        and publication_commit != expected_publication_commit
    ):
        raise ValueError("recovery receipt has the wrong publication commit")

    original = payload.get("original_submission_receipt")
    if not isinstance(original, dict) or set(original) != {
        "file",
        "literal_sha256",
        "receipt_sha256",
        "publisher_job_id",
        "preserved_without_mutation",
    }:
        raise ValueError("original submission-receipt binding is malformed")
    if not isinstance(original["file"], str) or not original["file"]:
        raise ValueError("original submission receipt filename is invalid")
    _require_hex64(original["literal_sha256"], "original receipt literal digest")
    _require_hex64(original["receipt_sha256"], "original receipt self-hash")
    original_job_id = _require_job_id(
        original["publisher_job_id"], "original publisher job id"
    )
    if original["preserved_without_mutation"] is not True:
        raise ValueError("original submission receipt was not preserved")

    failed = payload.get("failed_publisher")
    if not isinstance(failed, dict) or set(failed) != {
        "job_id",
        "accounting_record",
        "stdout",
        "stderr",
    }:
        raise ValueError("failed publisher binding is malformed")
    if _require_job_id(failed["job_id"], "failed publisher job id") != original_job_id:
        raise ValueError("failed publisher differs from the original publisher")
    accounting = failed["accounting_record"]
    if not isinstance(accounting, dict) or set(accounting) != {
        "file",
        "bytes",
        "literal_sha256",
        "accounting_sha256",
        "record_utf8",
        "record",
    }:
        raise ValueError("failed accounting-record binding is malformed")
    if not isinstance(accounting["file"], str) or not accounting["file"]:
        raise ValueError("failed accounting filename is invalid")
    if not isinstance(accounting["record_utf8"], str):
        raise ValueError("failed accounting literal record is not UTF-8 text")
    raw_record = accounting["record_utf8"].encode("utf-8")
    if accounting["bytes"] != len(raw_record):
        raise ValueError("failed accounting byte count is inconsistent")
    if accounting["literal_sha256"] != hashlib.sha256(raw_record).hexdigest():
        raise ValueError("failed accounting literal digest is invalid")
    parsed_record = _load_json_bytes(raw_record, label="embedded accounting record")
    if accounting["record"] != parsed_record:
        raise ValueError("embedded accounting object differs from its literal record")
    validated_accounting = validate_failed_publisher_accounting(
        parsed_record, expected_job_id=original_job_id
    )
    if accounting["accounting_sha256"] != validated_accounting["accounting_sha256"]:
        raise ValueError("failed accounting self-hash binding is invalid")
    _validate_log_binding(
        failed["stdout"],
        label="failed publisher stdout",
        expected_file=f"publish_{original_job_id}.out",
    )
    _validate_log_binding(
        failed["stderr"],
        label="failed publisher stderr",
        expected_file=f"publish_{original_job_id}.err",
    )

    raw = payload.get("preserved_raw_outputs")
    if not isinstance(raw, dict) or set(raw) != {
        "file",
        "bytes",
        "literal_sha256",
        "manifest_self_hash",
        "record_count",
        "normalized_record_set_sha256",
        "payload_merkle_root",
        "hash_semantics",
    }:
        raise ValueError("preserved raw-output binding is malformed")
    if not isinstance(raw["file"], str) or not raw["file"]:
        raise ValueError("raw-output manifest filename is invalid")
    if (
        not isinstance(raw["bytes"], int)
        or isinstance(raw["bytes"], bool)
        or raw["bytes"] <= 0
    ):
        raise ValueError("raw-output manifest byte count is invalid")
    if (
        not isinstance(raw["record_count"], int)
        or isinstance(raw["record_count"], bool)
        or raw["record_count"] <= 0
    ):
        raise ValueError("raw-output manifest record count is invalid")
    for key in (
        "literal_sha256",
        "normalized_record_set_sha256",
        "payload_merkle_root",
    ):
        _require_hex64(raw[key], f"raw-output {key}")
    _require_hex64(raw["manifest_self_hash"], "raw-output manifest self-hash")
    if raw["hash_semantics"] != "preserved_simulation_raw_output_manifest_v1":
        raise ValueError("raw-output manifest hash semantics are invalid")

    recovery = payload.get("recovery_publisher")
    if not isinstance(recovery, dict) or set(recovery) != {
        "job_id",
        "held_at_receipt_creation",
        "authorization_workflow",
    }:
        raise ValueError("recovery publisher binding is malformed")
    recovery_job_id = _require_job_id(
        recovery["job_id"], "recovery publisher job id"
    )
    if recovery_job_id == original_job_id:
        raise ValueError("recovery publisher reuses the failed publisher job id")
    if recovery["held_at_receipt_creation"] is not True:
        raise ValueError("recovery publisher was not held while authorizing")
    if recovery["authorization_workflow"] != HELD_WORKFLOW:
        raise ValueError("recovery publisher workflow is invalid")
    if (
        expected_recovery_job_id is not None
        and recovery_job_id
        != _require_job_id(expected_recovery_job_id, "expected recovery job id")
    ):
        raise ValueError("recovery receipt authorizes a different Slurm job")

    if require_self_hash:
        unsigned = dict(payload)
        claimed = unsigned.pop("receipt_sha256", None)
        if claimed != canonical_sha256(unsigned):
            raise ValueError("publication-recovery receipt self-hash is invalid")
    return dict(payload)


def create_recovery_receipt(
    *,
    output: Path,
    repo_root: Path,
    kind: str,
    run_tag: str,
    simulation_commit: str,
    original_receipt_path: Path,
    failed_accounting_record_path: Path,
    failed_stdout_path: Path,
    failed_stderr_path: Path,
    raw_output_manifest_path: Path,
    held_recovery_publisher_job_id: str,
    reason_code: str,
    expected_publication_commit: str | None = None,
) -> dict[str, Any]:
    """Create one exclusive receipt after the replacement job is held."""

    publication_commit, publication_tree = _git_clean_identity(repo_root)
    if (
        expected_publication_commit is not None
        and publication_commit != expected_publication_commit
    ):
        raise ValueError("clean checkout differs from expected publication commit")
    original_bytes = _read_stable_regular_file(
        original_receipt_path,
        label="original submission receipt",
        allow_empty=False,
    )
    original, original_job_id = _validate_original_submission(
        original_bytes,
        kind=kind,
        run_tag=run_tag,
        simulation_commit=simulation_commit,
    )
    _require_canonical_failed_logs(
        kind=kind,
        original_receipt_path=original_receipt_path,
        publisher_job_id=original_job_id,
        failed_stdout_path=failed_stdout_path,
        failed_stderr_path=failed_stderr_path,
    )
    payload = build_recovery_receipt(
        kind=kind,
        run_tag=run_tag,
        simulation_commit=simulation_commit,
        publication_commit=publication_commit,
        publication_tree=publication_tree,
        original_receipt_file=original_receipt_path.name,
        original_receipt_literal_sha256=hashlib.sha256(original_bytes).hexdigest(),
        original_receipt_self_hash=original["receipt_sha256"],
        original_publisher_job_id=original_job_id,
        failed_accounting=_failed_accounting_binding(
            failed_accounting_record_path, expected_job_id=original_job_id
        ),
        failed_stdout=_file_hash_binding(
            failed_stdout_path, label="failed publisher stdout"
        ),
        failed_stderr=_file_hash_binding(
            failed_stderr_path, label="failed publisher stderr"
        ),
        raw_output_manifest=_raw_manifest_binding(
            raw_output_manifest_path,
            kind=kind,
            run_tag=run_tag,
            simulation_commit=simulation_commit,
            simulation_source_tree_sha256=original["source_tree_sha256"],
        ),
        held_recovery_publisher_job_id=held_recovery_publisher_job_id,
        reason_code=reason_code,
    )
    # Re-read before writing anything: even a concurrent replacement of the
    # original receipt makes recovery authorization fail closed.
    if _read_stable_regular_file(
        original_receipt_path,
        label="original submission receipt",
        allow_empty=False,
    ) != original_bytes:
        raise ValueError("original submission receipt changed during authorization")
    _write_exclusive_json(output, payload)
    return payload


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = _unresolved_safe_path(path, label="recovery receipt output")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite recovery receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)
    if path.read_bytes() != encoded:
        raise RuntimeError("recovery receipt failed literal-byte readback")


def validate_recovery_receipt_file(
    path: Path,
    *,
    original_receipt_path: Path,
    expected_kind: str,
    expected_run_tag: str,
    expected_simulation_commit: str,
    expected_publication_commit: str,
    expected_recovery_job_id: str | None = None,
) -> dict[str, Any]:
    """Validate a recovery receipt and its immutable original submission receipt."""

    receipt_bytes = _read_stable_regular_file(
        path, label="publication-recovery receipt", allow_empty=False
    )
    payload = _load_json_bytes(receipt_bytes, label="publication-recovery receipt")
    validated = validate_recovery_receipt_payload(
        payload,
        expected_kind=expected_kind,
        expected_run_tag=expected_run_tag,
        expected_simulation_commit=expected_simulation_commit,
        expected_publication_commit=expected_publication_commit,
        expected_recovery_job_id=expected_recovery_job_id,
    )
    original_bytes = _read_stable_regular_file(
        original_receipt_path,
        label="original submission receipt",
        allow_empty=False,
    )
    original, publisher_job_id = _validate_original_submission(
        original_bytes,
        kind=expected_kind,
        run_tag=expected_run_tag,
        simulation_commit=expected_simulation_commit,
    )
    binding = validated["original_submission_receipt"]
    if binding["literal_sha256"] != hashlib.sha256(original_bytes).hexdigest():
        raise ValueError("original submission receipt literal bytes changed")
    if binding["receipt_sha256"] != original["receipt_sha256"]:
        raise ValueError("original submission receipt self-hash changed")
    if binding["publisher_job_id"] != publisher_job_id:
        raise ValueError("original publisher binding differs from submission receipt")
    # Neither validation nor parsing is allowed to rewrite the original.
    if _read_stable_regular_file(
        original_receipt_path,
        label="original submission receipt",
        allow_empty=False,
    ) != original_bytes:
        raise ValueError("original submission receipt changed during validation")
    return validated


def require_authorized_publisher(
    receipt: Mapping[str, Any],
    *,
    actual_slurm_job_id: str,
    expected_kind: str | None = None,
    expected_run_tag: str | None = None,
    expected_simulation_commit: str | None = None,
    expected_publication_commit: str | None = None,
) -> dict[str, Any]:
    """Require the running Slurm job to be the one sealed while held."""

    actual = _require_job_id(actual_slurm_job_id, "actual publisher SLURM_JOB_ID")
    validated = validate_recovery_receipt_payload(
        receipt,
        expected_kind=expected_kind,
        expected_run_tag=expected_run_tag,
        expected_simulation_commit=expected_simulation_commit,
        expected_publication_commit=expected_publication_commit,
        expected_recovery_job_id=actual,
    )
    return validated


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser(
        "create", help="authorize an already submitted and held recovery publisher"
    )
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--repo-root", type=Path, default=Path.cwd())
    create.add_argument("--kind", choices=RECOVERY_KINDS, required=True)
    create.add_argument("--run-tag", required=True)
    create.add_argument("--simulation-commit", required=True)
    create.add_argument("--publication-commit")
    create.add_argument("--original-submission-receipt", type=Path, required=True)
    create.add_argument("--failed-accounting-record", type=Path, required=True)
    create.add_argument("--failed-stdout", type=Path, required=True)
    create.add_argument("--failed-stderr", type=Path, required=True)
    create.add_argument("--raw-output-manifest", type=Path, required=True)
    create.add_argument("--held-recovery-publisher-job-id", required=True)
    create.add_argument("--reason-code", required=True)

    validate = commands.add_parser("validate")
    validate.add_argument("--receipt", type=Path, required=True)
    validate.add_argument("--original-submission-receipt", type=Path, required=True)
    validate.add_argument("--kind", choices=RECOVERY_KINDS, required=True)
    validate.add_argument("--run-tag", required=True)
    validate.add_argument("--simulation-commit", required=True)
    validate.add_argument("--publication-commit", required=True)
    validate.add_argument(
        "--recovery-publisher-slurm-job-id",
        help="when inside Slurm, require this exact running recovery publisher",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "create":
        payload = create_recovery_receipt(
            output=args.output.absolute(),
            repo_root=args.repo_root.absolute(),
            kind=args.kind,
            run_tag=args.run_tag,
            simulation_commit=args.simulation_commit,
            original_receipt_path=args.original_submission_receipt.absolute(),
            failed_accounting_record_path=args.failed_accounting_record.absolute(),
            failed_stdout_path=args.failed_stdout.absolute(),
            failed_stderr_path=args.failed_stderr.absolute(),
            raw_output_manifest_path=args.raw_output_manifest.absolute(),
            held_recovery_publisher_job_id=args.held_recovery_publisher_job_id,
            reason_code=args.reason_code,
            expected_publication_commit=args.publication_commit,
        )
    else:
        payload = validate_recovery_receipt_file(
            args.receipt.absolute(),
            original_receipt_path=args.original_submission_receipt.absolute(),
            expected_kind=args.kind,
            expected_run_tag=args.run_tag,
            expected_simulation_commit=args.simulation_commit,
            expected_publication_commit=args.publication_commit,
            expected_recovery_job_id=args.recovery_publisher_slurm_job_id,
        )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
