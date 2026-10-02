#!/usr/bin/env python3
"""Bind verified core and structural archives into one submission receipt.

This command does not execute or repair simulations.  It independently checks
the literal archive bytes, archive membership, manifests, and validation
receipts produced by the two isolated publication workflows.  A final receipt
is written only when both complete evidence scopes came from the same clean
source commit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hpc.capture_slurm_accounting import validate_accounting_payload  # noqa: E402
from hpc.preserved_raw_manifest import validate_manifest_document  # noqa: E402
from hpc.publication_recovery_receipt import (  # noqa: E402
    validate_recovery_receipt_payload,
)
from hpc.validate_source_snapshot import tracked_source_digest  # noqa: E402
from mvp.simulation.validation.validator_source_identity import (  # noqa: E402
    validate_clean_validator_checkout,
)

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CORE_RUN_TAG = re.compile(r"^([0-9a-f]{7})_[0-9]{8}_[0-9]{6}$")
_STRUCTURAL_RUN_TAG = re.compile(
    r"^sensitivity_([0-9a-f]{7})_[0-9]{8}_[0-9]{6}$"
)
_JOB_ID = re.compile(r"^[1-9][0-9]*$")
_PARTITION = re.compile(r"^[A-Za-z0-9._,+-]+$")
_STRUCTURAL_PREFIX = "structural_sensitivity_evidence"
_STRUCTURAL_MANIFEST = "structural_sensitivity_artifact_manifest.json"
_STRUCTURAL_RECOVERY_ATTEMPTS = "publication_recovery_attempts"
_CORE_VALIDATION_RECEIPT = "publication_validation_receipt.json"

_CORE_ACCOUNTING = {
    "core_unique_retained_cells": 1_600,
    "core_executed_episodes": 6_100,
    "core_simulated_steps": 1_756_800,
    "h1_directional_tests": 5,
    "h2_directional_tests": 20,
    "h3_equivalence_cells": 25,
}
_CORE_EXECUTION_ACCOUNTING = {
    "unique_retained_cells": 1_600,
    "executed_episodes": 6_100,
    "simulated_steps": 1_756_800,
}
_SEEDS = [
    42, 1337, 2024, 7, 99, 101, 202, 303, 404, 505,
    606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515,
]
_SCENARIOS = [
    "heatwave",
    "overproduction",
    "cyber_outage",
    "adaptive_pricing",
    "baseline",
]
_STRUCTURAL_TOTALS = {
    "retained_cells": 6_500,
    "executed_episodes": 24_500,
    "simulated_steps": 7_056_000,
}
_VALIDATED_CORE_CHECKS = [
    "core_slurm_submission_dag",
    "exact_H1_H2_seed_panels_and_inference",
    "deterministic_core_statistical_reaggregation",
    "exact_H3_panel_TOST_and_treatment_exposure",
    "raw_endpoint_and_decision_ledger_recomputation",
    "table_and_paper_export_semantic_projection",
    "forecast_selection_and_predictions",
    "deterministic_derived_artifact_and_H3_replay",
    "figure_provenance_and_exact_inventory",
    "environment_source_and_run_identity",
    "literal_byte_manifest_integrity",
]


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return _sha256_stream(stream)


def _require_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be an existing non-symlink regular file: {path}")
    return path


def _validate_local_validator_checkout(
    expected_commit: str, *, repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Require the executing validator to be the archived clean source commit."""

    return validate_clean_validator_checkout(
        expected_commit, repo_root=repo_root,
    )


def _current_git_tree_sha1(*, repo_root: Path = REPO_ROOT) -> str:
    """Return the literal Git tree of the clean assembler checkout."""

    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=repo_root.resolve(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("cannot establish the assembler Git tree") from exc
    return _require_hex(value, label="assembler Git tree", width=40)


def _validated_new_output_path(path: Path) -> Path:
    """Return an absolute lexical path whose existing components are not links."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute.exists() or absolute.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite final evidence receipt: {absolute}"
        )
    for parent in absolute.parents:
        if parent.is_symlink():
            raise ValueError(
                f"final evidence output has a symlinked parent: {parent}"
            )
        if parent.exists() and not parent.is_dir():
            raise ValueError(
                f"final evidence output parent is not a directory: {parent}"
            )
    return absolute


def _write_new_file_atomically(path: Path, payload: bytes) -> None:
    """Publish complete bytes atomically while preserving no-overwrite safety."""

    path = _validated_new_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Recheck after creating missing parents so a dangling link or parent-path
    # substitution observed before installation is rejected.
    path = _validated_new_output_path(path)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # A same-directory hard-link installation is atomic and, unlike
            # replace(), fails if another process created the final name.
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite final evidence receipt: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _load_json_file(path: Path, label: str) -> dict[str, Any]:
    _require_regular_file(path, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return payload


def _safe_name(raw: object, *, label: str = "archive member") -> str:
    if not isinstance(raw, str) or not raw or "\\" in raw or "\x00" in raw:
        raise ValueError(f"unsafe {label} path: {raw!r}")
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or ":" in path.parts[0]
    ):
        raise ValueError(f"unsafe {label} path: {raw!r}")
    return path.as_posix()


def _structural_recovery_attempt_root(
    authorization: Mapping[str, Any], *, run_tag: str,
) -> str:
    """Return the one canonical archive root for a structural recovery job."""

    job_id = authorization.get("authorized_recovery_publisher_job_id")
    if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
        raise ValueError(
            "structural recovery authorization has an invalid publisher job id"
        )
    root = f"{_STRUCTURAL_RECOVERY_ATTEMPTS}/{job_id}"
    attempt_root = _safe_name(
        authorization.get("attempt_root"),
        label="structural recovery attempt root",
    )
    if attempt_root != root:
        raise ValueError(
            "structural recovery attempt root does not match its publisher job id"
        )
    expected = {
        "receipt_file": f"{root}/publication_recovery_receipts/{run_tag}.json",
        "preserved_raw_manifest_file": (
            f"{root}/preserved_raw_manifests/{run_tag}.json"
        ),
    }
    for field, expected_path in expected.items():
        observed = _safe_name(
            authorization.get(field),
            label=f"structural recovery {field}",
        )
        if observed != expected_path:
            raise ValueError(
                f"structural recovery {field} is outside its job-scoped "
                "attempt root"
            )
    return root


def _publication_member(publication_root: str, name: str) -> str:
    """Return a safe archive-relative publication member path."""

    safe_name = _safe_name(name, label="structural publication member")
    if not publication_root:
        return safe_name
    safe_root = _safe_name(
        publication_root, label="structural publication attempt root"
    )
    return f"{safe_root}/{safe_name}"


def _require_hex(value: object, *, label: str, width: int) -> str:
    pattern = _HEX40 if width == 40 else _HEX64
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase {width}-hex digest")
    return value


def _require_utc_timestamp(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a timezone-aware ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be a timezone-aware ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must be a timezone-aware ISO timestamp")
    return value


def _validate_self_hash(
    payload: Mapping[str, Any], *, field: str, label: str,
) -> str:
    unsigned = dict(payload)
    digest = unsigned.pop(field, None)
    _require_hex(digest, label=f"{label} {field}", width=64)
    expected = _canonical_sha256(unsigned)
    if digest != expected:
        raise ValueError(f"{label} self-hash does not match its canonical content")
    return str(digest)


def _read_archive(
    archive_path: Path, *, required_manifest_name: str,
) -> tuple[bytes, dict[str, dict[str, Any]]]:
    """Return manifest bytes and literal member metadata after safety checks."""

    _require_regular_file(archive_path, "evidence archive")
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
            if not members:
                raise ValueError("evidence archive is empty")
            normalized: dict[str, tarfile.TarInfo] = {}
            for member in members:
                name = _safe_name(member.name)
                if name in normalized:
                    raise ValueError(f"duplicate archive member: {name}")
                if (
                    not member.isfile()
                    or member.issym()
                    or member.islnk()
                    or member.size < 0
                ):
                    raise ValueError(f"archive member is not a regular file: {name}")
                normalized[name] = member
            manifest_member = normalized.get(required_manifest_name)
            if manifest_member is None:
                raise ValueError(
                    f"archive lacks required manifest member {required_manifest_name!r}"
                )
            manifest_stream = archive.extractfile(manifest_member)
            if manifest_stream is None:
                raise ValueError("cannot read archived manifest")
            manifest_bytes = manifest_stream.read()

            metadata: dict[str, dict[str, Any]] = {}
            for name, member in normalized.items():
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"cannot read archive member: {name}")
                metadata[name] = {
                    "bytes": int(member.size),
                    "sha256": _sha256_stream(stream),
                }
            return manifest_bytes, metadata
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise ValueError(f"invalid gzip tar archive: {archive_path}") from exc


def _json_member(
    archive_path: Path, member_name: str,
) -> dict[str, Any]:
    """Read one already safety-validated regular JSON member."""

    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            member = archive.getmember(member_name)
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError(f"archived JSON is not a regular file: {member_name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot read archived JSON: {member_name}")
            payload = json.loads(stream.read().decode("utf-8"))
    except (KeyError, tarfile.TarError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid archived JSON: {member_name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"archived JSON must contain one object: {member_name}")
    return payload


def _archive_member_bytes(
    archive_path: Path,
    member_name: str,
    *,
    metadata: Mapping[str, Mapping[str, Any]],
) -> bytes:
    """Read one already inventoried member and recheck its literal binding."""

    expected = metadata.get(member_name)
    if expected is None:
        raise ValueError(f"archive lacks required recovery member: {member_name}")
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            member = archive.getmember(member_name)
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError(
                    f"archived recovery evidence is not regular: {member_name}"
                )
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot read archived recovery evidence: {member_name}")
            payload = stream.read()
    except (KeyError, tarfile.TarError, OSError, EOFError) as exc:
        raise ValueError(f"cannot read archived recovery evidence: {member_name}") from exc
    if (
        len(payload) != expected.get("bytes")
        or hashlib.sha256(payload).hexdigest() != expected.get("sha256")
    ):
        raise ValueError(f"archived recovery evidence changed: {member_name}")
    return payload


def _load_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _raw_payload_merkle_root(records: list[Mapping[str, Any]]) -> str:
    """Recompute the preserved-raw manifest's canonical Merkle root."""

    level = [
        hashlib.sha256(
            json.dumps(
                {
                    "path": record["path"],
                    "bytes": record["bytes"],
                    "sha256": record["sha256"],
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).digest()
        for record in records
    ]
    if not level:
        return hashlib.sha256(b"").hexdigest()
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            hashlib.sha256(level[index] + level[index + 1]).digest()
            for index in range(0, len(level), 2)
        ]
    return level[0].hex()


def _validate_structural_raw_archive_mapping(
    raw_manifest: Mapping[str, Any],
    *,
    metadata: Mapping[str, Mapping[str, Any]],
    prefix: str,
) -> dict[str, Any]:
    """Reprove every structural raw byte under its archive-prefix path."""

    expected_binding_types = {
        "logs": "directory",
        "runtime_receipts": "directory",
        "tasks": "directory",
        "episode_accounting.json": "file",
        "experiment_protocol.json": "file",
        "lhs_design.csv": "file",
        "lhs_design.json": "file",
        "parameter_registry.json": "file",
        "run_plan.json": "file",
        "slurm_submission.json": "file",
        "task_manifest.json": "file",
        "task_manifest.jsonl": "file",
    }
    bindings = raw_manifest.get("bindings")
    observed_binding_types = {
        str(item.get("name")): item.get("type")
        for item in bindings
        if isinstance(item, dict)
    } if isinstance(bindings, list) else {}
    if observed_binding_types != expected_binding_types:
        raise ValueError(
            "structural preserved-raw manifest has the wrong logical bindings"
        )

    records = raw_manifest.get("files")
    if not isinstance(records, list):
        raise ValueError("structural preserved-raw manifest lacks file records")
    expected_members: dict[str, dict[str, Any]] = {}
    for record in records:
        logical = str(record["path"])
        archive_name = f"{prefix}{logical}"
        if archive_name in expected_members:
            raise ValueError("structural raw mapping produces duplicate archive members")
        expected_members[archive_name] = {
            "bytes": record["bytes"],
            "sha256": record["sha256"],
        }

    raw_domains = (
        f"{prefix}logs/",
        f"{prefix}runtime_receipts/",
        f"{prefix}tasks/",
    )
    singleton_members = {
        f"{prefix}{name}"
        for name, kind in expected_binding_types.items()
        if kind == "file"
    }
    actual_raw_members = {
        name
        for name in metadata
        if name.startswith(raw_domains) or name in singleton_members
    }
    if actual_raw_members != set(expected_members):
        raise ValueError(
            "structural archive raw-member set differs from the preserved manifest: "
            f"missing={sorted(set(expected_members) - actual_raw_members)[:10]}, "
            f"extra={sorted(actual_raw_members - set(expected_members))[:10]}"
        )
    for name, expected in expected_members.items():
        if metadata.get(name) != expected:
            raise ValueError(
                f"structural archived raw byte binding differs: {name}"
            )
    if (
        len(expected_members) != raw_manifest.get("file_count")
        or sum(item["bytes"] for item in expected_members.values())
        != raw_manifest.get("total_bytes")
        or _raw_payload_merkle_root(records)
        != raw_manifest.get("payload_merkle_root")
    ):
        raise ValueError("structural archived raw summary or Merkle root is invalid")
    return {
        "binding_scope": "structural_publication_archive",
        "mapped_member_count": len(expected_members),
        "mapped_bytes": sum(item["bytes"] for item in expected_members.values()),
        "payload_merkle_root": raw_manifest["payload_merkle_root"],
        "exact_member_set": True,
        "literal_sizes_and_sha256": True,
    }


def _validate_archived_recovery(
    archive_path: Path,
    *,
    metadata: Mapping[str, Mapping[str, Any]],
    prefix: str,
    kind: str,
    run_tag: str,
    simulation_commit: str,
    publication_commit: str,
    simulation_source_tree_sha256: str,
    original_submission_member: str,
    manifested_authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one archived recovery receipt and every immutable binding.

    This is deliberately independent of the archive/semantic receipts.  It
    validates the recovery receipt's self-contained failed-publisher evidence,
    compares the literal original submission receipt and preserved-raw
    manifest, and then requires the archive manifest's authorization summary
    to be an exact projection of those independently validated bytes.
    """

    attempt_root: str | None = None
    if kind == "structural":
        attempt_root = _structural_recovery_attempt_root(
            manifested_authorization, run_tag=run_tag,
        )
        recovery_relative = _publication_member(
            attempt_root, f"publication_recovery_receipts/{run_tag}.json",
        )
        raw_relative = _publication_member(
            attempt_root, f"preserved_raw_manifests/{run_tag}.json",
        )
    else:
        recovery_relative = f"publication_recovery_receipts/{run_tag}.json"
        raw_relative = f"preserved_raw_manifests/{run_tag}.json"
    recovery_member = f"{prefix}{recovery_relative}"
    raw_member = f"{prefix}{raw_relative}"
    original_member = f"{prefix}{original_submission_member}"

    recovery_bytes = _archive_member_bytes(
        archive_path, recovery_member, metadata=metadata
    )
    raw_bytes = _archive_member_bytes(archive_path, raw_member, metadata=metadata)
    original_bytes = _archive_member_bytes(
        archive_path, original_member, metadata=metadata
    )
    recovery = validate_recovery_receipt_payload(
        _load_json_bytes(recovery_bytes, label=f"{kind} recovery receipt"),
        expected_kind=kind,
        expected_run_tag=run_tag,
        expected_simulation_commit=simulation_commit,
        expected_publication_commit=publication_commit,
    )
    source_identity = recovery["source_identity"]
    publication_tree = _require_hex(
        source_identity.get("publication_repair_tree"),
        label=f"{kind} publication-repair Git tree",
        width=40,
    )

    original = recovery["original_submission_receipt"]
    original_payload = _load_json_bytes(
        original_bytes, label=f"{kind} original submission receipt"
    )
    if (
        original.get("file") != PurePosixPath(original_submission_member).name
        or original.get("literal_sha256")
        != hashlib.sha256(original_bytes).hexdigest()
        or original.get("receipt_sha256") != original_payload.get("receipt_sha256")
        or original.get("preserved_without_mutation") is not True
    ):
        raise ValueError(
            f"{kind} recovery receipt does not bind the archived original "
            "submission receipt"
        )

    raw_payload = validate_manifest_document(
        _load_json_bytes(raw_bytes, label=f"{kind} preserved-raw manifest"),
        kind=kind,
        run_tag=run_tag,
        simulation_commit=simulation_commit,
        simulation_source_tree_sha256=simulation_source_tree_sha256,
    )
    raw = recovery["preserved_raw_outputs"]
    if (
        raw.get("file") != PurePosixPath(raw_relative).name
        or raw.get("bytes") != len(raw_bytes)
        or raw.get("literal_sha256") != hashlib.sha256(raw_bytes).hexdigest()
        or raw.get("manifest_self_hash") != raw_payload.get("manifest_sha256")
        or raw.get("record_count") != len(raw_payload.get("files", []))
        or raw.get("normalized_record_set_sha256")
        != _canonical_sha256(raw_payload.get("files"))
        or raw.get("payload_merkle_root")
        != raw_payload.get("payload_merkle_root")
    ):
        raise ValueError(
            f"{kind} recovery receipt does not bind the archived preserved raw manifest"
        )

    if kind == "structural":
        raw_archive_binding = _validate_structural_raw_archive_mapping(
            raw_payload, metadata=metadata, prefix=prefix
        )
    else:
        raw_archive_binding = {
            "binding_scope": "separate_complete_run_evidence_bundle_required",
            "mapped_member_count": raw_payload["file_count"],
            "mapped_bytes": raw_payload["total_bytes"],
            "payload_merkle_root": raw_payload["payload_merkle_root"],
            "exact_member_set": None,
            "literal_sizes_and_sha256": None,
        }

    receipt_literal_sha256 = hashlib.sha256(recovery_bytes).hexdigest()
    authorization = dict(manifested_authorization)
    common_projection = {
        "receipt_file": recovery_relative,
        "receipt_literal_sha256": receipt_literal_sha256,
        "preserved_raw_manifest_file": raw_relative,
        "preserved_raw_manifest_literal_sha256": hashlib.sha256(
            raw_bytes
        ).hexdigest(),
        "preserved_raw_payload_merkle_root": raw_payload["payload_merkle_root"],
        "simulation_rerun": False,
    }
    for key, expected in common_projection.items():
        if authorization.get(key) != expected:
            raise ValueError(
                f"{kind} manifest recovery authorization disagrees on {key}"
            )
    receipt_self_hash = recovery["receipt_sha256"]
    if kind == "core":
        expected_core = {
            **common_projection,
            "receipt_self_hash": receipt_self_hash,
            "original_submission_receipt_file": original_submission_member,
            "publication_repair_tree": publication_tree,
            "validated": True,
        }
        if authorization != expected_core:
            raise ValueError(
                "core manifest recovery authorization is not the exact validated projection"
            )
    else:
        assert attempt_root is not None
        structural_projection = {
            "attempt_root": attempt_root,
            "receipt_sha256": receipt_self_hash,
            "preserved_raw_manifest_sha256": raw_payload["manifest_sha256"],
            "original_failed_publisher_job_id": recovery["failed_publisher"][
                "job_id"
            ],
            "authorized_recovery_publisher_job_id": recovery[
                "recovery_publisher"
            ]["job_id"],
            "original_submission_receipt_preserved": True,
        }
        for key, expected in structural_projection.items():
            if authorization.get(key) != expected:
                raise ValueError(
                    f"structural manifest recovery authorization disagrees on {key}"
                )
        if set(authorization) != {
            *common_projection,
            *structural_projection,
        }:
            raise ValueError(
                "structural manifest recovery authorization has ambiguous fields"
            )

    return {
        "kind": kind,
        "run_tag": run_tag,
        "simulation_rerun": False,
        "simulation_commit": simulation_commit,
        "publication_code_commit": publication_commit,
        "publication_repair_tree": publication_tree,
        "receipt_file": recovery_relative,
        "receipt_bytes": len(recovery_bytes),
        "receipt_literal_sha256": receipt_literal_sha256,
        "receipt_self_hash": receipt_self_hash,
        "preserved_raw_manifest_file": raw_relative,
        "preserved_raw_manifest_literal_sha256": hashlib.sha256(
            raw_bytes
        ).hexdigest(),
        "preserved_raw_manifest_self_hash": raw_payload["manifest_sha256"],
        "preserved_raw_payload_merkle_root": raw_payload["payload_merkle_root"],
        "raw_archive_binding": raw_archive_binding,
        "original_failed_publisher_job_id": recovery["failed_publisher"]["job_id"],
        "authorized_recovery_publisher_job_id": recovery["recovery_publisher"][
            "job_id"
        ],
        **(
            {"attempt_root": attempt_root} if attempt_root is not None else {}
        ),
        "validated": True,
        "_raw_manifest_payload": raw_payload,
    }


def _validate_core_complete_run_evidence(
    archive_path: Path,
    receipt_path: Path,
    ready_path: Path,
    *,
    run_tag: str,
    simulation_commit: str,
    simulation_source_tree_sha256: str,
    raw_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind every preserved core/H3 raw byte to the lossless archive."""

    archive_path = _require_regular_file(
        archive_path, "complete core raw-evidence archive"
    )
    receipt_path = _require_regular_file(
        receipt_path, "complete core raw-evidence receipt"
    )
    ready_path = _require_regular_file(
        ready_path, "complete core raw-evidence READY marker"
    )
    manifest_name = "COMPLETE_RUN_EVIDENCE_MANIFEST.json"
    manifest_bytes, metadata = _read_archive(
        archive_path, required_manifest_name=manifest_name
    )
    manifest = _load_json_bytes(
        manifest_bytes, label="complete core raw-evidence manifest"
    )
    manifest_digest = _validate_self_hash(
        manifest,
        field="manifest_sha256",
        label="complete core raw-evidence manifest",
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "COMPLETE"
        or manifest.get("source_commit") != simulation_commit
        or manifest.get("source_tree_sha256")
        != simulation_source_tree_sha256
        or manifest.get("run_tag") != run_tag
    ):
        raise ValueError("complete core raw-evidence manifest identity is inconsistent")
    records = _manifest_records(manifest, key="artifacts", path_key="path")
    if (
        manifest.get("artifact_count") != len(records)
        or manifest.get("artifact_bytes")
        != sum(int(record["bytes"]) for record in records)
    ):
        raise ValueError("complete core raw-evidence artifact summary is inconsistent")
    _verify_exact_membership(
        metadata,
        records,
        manifest_member=manifest_name,
        record_path_key="path",
    )

    receipt = _load_json_file(
        receipt_path, "complete core raw-evidence receipt"
    )
    receipt_digest = _validate_self_hash(
        receipt,
        field="receipt_sha256",
        label="complete core raw-evidence receipt",
    )
    expected_receipt_keys = {
        "schema_version",
        "status",
        "run_tag",
        "source_commit",
        "manifest_sha256",
        "archive",
        "archive_sha256",
        "archive_bytes",
        "artifact_count",
        "episode_totals",
        "runtime_receipt_totals",
        "scheduler_accounting",
        "receipt_sha256",
    }
    archive_sha256 = _sha256_file(archive_path)
    if set(receipt) != expected_receipt_keys or any(
        (
            receipt.get("schema_version") != 1,
            receipt.get("status") != "VERIFIED",
            receipt.get("run_tag") != run_tag,
            receipt.get("source_commit") != simulation_commit,
            receipt.get("manifest_sha256") != manifest_digest,
            receipt.get("archive") != archive_path.name,
            receipt.get("archive_sha256") != archive_sha256,
            receipt.get("archive_bytes") != archive_path.stat().st_size,
            receipt.get("artifact_count") != len(records),
            receipt.get("episode_totals") != manifest.get("episode_totals"),
            receipt.get("runtime_receipt_totals")
            != manifest.get("runtime_receipt_totals"),
            receipt.get("scheduler_accounting")
            != manifest.get("scheduler_accounting"),
        )
    ):
        raise ValueError("complete core raw-evidence receipt is inconsistent")
    ready = _load_json_file(
        ready_path, "complete core raw-evidence READY marker"
    )
    if ready != {
        "status": "READY",
        "archive_sha256": archive_sha256,
        "receipt_sha256": receipt_digest,
    }:
        raise ValueError(
            "complete core raw-evidence READY marker does not bind the bundle"
        )

    expected_binding_types = {
        "benchmark_seed_outputs": "directory",
        "stress_outputs": "directory",
        "h3_decision_ledgers": "directory",
        "core_submission_receipt.json": "file",
    }
    bindings = raw_manifest.get("bindings")
    binding_types = {
        str(item.get("name")): item.get("type")
        for item in bindings
        if isinstance(item, dict)
    } if isinstance(bindings, list) else {}
    if binding_types != expected_binding_types:
        raise ValueError("core preserved-raw manifest has the wrong logical bindings")
    logical_prefixes = {
        "benchmark_seed_outputs/": "inputs/core/",
        "stress_outputs/": "inputs/h3_results/",
        "h3_decision_ledgers/": "inputs/h3_ledgers/",
    }
    raw_records = raw_manifest.get("files")
    if not isinstance(raw_records, list):
        raise ValueError("core preserved-raw manifest lacks file records")
    mapped: dict[str, dict[str, Any]] = {}
    for record in raw_records:
        logical = str(record["path"])
        if logical == "core_submission_receipt.json":
            archive_name = "inputs/core_submission_receipt.json"
        else:
            archive_name = ""
            for logical_prefix, archive_prefix in logical_prefixes.items():
                if logical.startswith(logical_prefix):
                    archive_name = archive_prefix + logical.removeprefix(
                        logical_prefix
                    )
                    break
            if not archive_name:
                raise ValueError(
                    f"unmapped core preserved-raw logical path: {logical}"
                )
        if archive_name in mapped:
            raise ValueError("core raw mapping produces duplicate archive members")
        mapped[archive_name] = {
            "bytes": record["bytes"],
            "sha256": record["sha256"],
        }

    record_by_name = {record["path"]: record for record in records}
    raw_domain_names = {
        name
        for name in record_by_name
        if name.startswith((
            "inputs/core/",
            "inputs/h3_results/",
            "inputs/h3_ledgers/",
        )) or name == "inputs/core_submission_receipt.json"
    }
    if raw_domain_names != set(mapped):
        raise ValueError(
            "complete core raw archive member set differs from preserved inputs: "
            f"missing={sorted(set(mapped) - raw_domain_names)[:10]}, "
            f"extra={sorted(raw_domain_names - set(mapped))[:10]}"
        )
    for name, expected in mapped.items():
        record = record_by_name.get(name)
        if record is None or {
            "bytes": record["bytes"], "sha256": record["sha256"],
        } != expected or metadata.get(name) != expected:
            raise ValueError(f"complete core raw byte binding differs: {name}")

    scheduler_names = set(record_by_name) - set(mapped)
    if scheduler_names != {"inputs/scheduler/slurm_simulation_accounting.json"}:
        raise ValueError(
            "complete recovery archive has unexpected non-raw members: "
            f"{sorted(scheduler_names)}"
        )
    if (
        len(mapped) != raw_manifest.get("file_count")
        or sum(item["bytes"] for item in mapped.values())
        != raw_manifest.get("total_bytes")
        or _raw_payload_merkle_root(raw_records)
        != raw_manifest.get("payload_merkle_root")
    ):
        raise ValueError("complete core raw archive summary or Merkle root is invalid")

    episode_totals = manifest.get("episode_totals")
    if not isinstance(episode_totals, dict) or any(
        episode_totals.get(key) != expected
        for key, expected in {
            "episode_groups": 1_600,
            "executed_episode_archives": 6_100,
            "adaptation_episode_ledgers": 4_500,
            "final_episode_ledgers": 1_600,
        }.items()
    ) or not isinstance(episode_totals.get("decision_records"), int) or (
        episode_totals["decision_records"] <= 0
    ):
        raise ValueError("complete core raw archive has incorrect episode totals")
    runtime = manifest.get("runtime_receipt_totals")
    if not isinstance(runtime, dict) or (
        runtime.get("successful_receipt_count") != 25
        or runtime.get("receipt_count")
        != 25 + int(runtime.get("failed_attempt_receipt_count", -1))
    ):
        raise ValueError("complete core raw archive has incorrect runtime receipts")
    scheduler = manifest.get("scheduler_accounting")
    scheduler_member = "inputs/scheduler/slurm_simulation_accounting.json"
    validated_scheduler = validate_accounting_payload(
        _json_member(archive_path, scheduler_member),
        kind="core",
        run_tag=run_tag,
        source_commit=simulation_commit,
        source_tree_sha256=simulation_source_tree_sha256,
        expected_task_count=25,
    )
    if not isinstance(scheduler, dict) or scheduler != validated_scheduler or (
        scheduler.get("completed_simulation_task_count") != 25
    ):
        raise ValueError("complete core raw archive has incorrect scheduler accounting")

    return {
        "archive": {
            "name": archive_path.name,
            "bytes": archive_path.stat().st_size,
            "sha256": archive_sha256,
            "member_count": len(metadata),
        },
        "receipt": {
            "name": receipt_path.name,
            "bytes": receipt_path.stat().st_size,
            "sha256": _sha256_file(receipt_path),
            "content_sha256": receipt_digest,
        },
        "ready": {
            "name": ready_path.name,
            "bytes": ready_path.stat().st_size,
            "sha256": _sha256_file(ready_path),
            "status": "READY",
        },
        "manifest": {
            "content_sha256": manifest_digest,
            "artifact_count": len(records),
        },
        "raw_mapping": {
            "binding_scope": "complete_run_evidence_bundle",
            "mapped_member_count": len(mapped),
            "mapped_bytes": sum(item["bytes"] for item in mapped.values()),
            "payload_merkle_root": raw_manifest["payload_merkle_root"],
            "exact_member_set": True,
            "literal_sizes_and_sha256": True,
        },
        "episode_totals": dict(episode_totals),
        "runtime_receipt_totals": dict(runtime),
        "scheduler_accounting": dict(scheduler),
    }


def _extract_safe_archive(
    archive_path: Path,
    destination: Path,
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    selected_names: set[str] | None = None,
) -> None:
    """Extract verified regular members without using tarfile.extract()."""

    if destination.exists() and any(destination.iterdir()):
        raise ValueError("safe extraction destination must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            name = _safe_name(member.name)
            expected = metadata.get(name)
            if expected is None or not member.isfile() or member.issym() or member.islnk():
                raise ValueError(f"unsafe or unverified archive member: {name}")
            if selected_names is not None and name not in selected_names:
                continue
            target = destination.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot read archive member: {name}")
            digest = hashlib.sha256()
            written = 0
            with target.open("xb") as output:
                while chunk := stream.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
            if written != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
                raise ValueError(f"archive member changed during safe extraction: {name}")


def _manifest_records(
    manifest: Mapping[str, Any], *, key: str, path_key: str,
) -> list[dict[str, Any]]:
    raw_records = manifest.get(key)
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError(f"manifest {key!r} must be a non-empty list")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict):
            raise ValueError(f"manifest record {index} is not an object")
        name = _safe_name(raw_record.get(path_key), label="manifest")
        if name in seen:
            raise ValueError(f"duplicate manifest path: {name}")
        seen.add(name)
        size = raw_record.get("bytes")
        digest = raw_record.get("sha256")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"manifest byte count is invalid for {name}")
        _require_hex(digest, label=f"manifest SHA-256 for {name}", width=64)
        records.append({path_key: name, "bytes": size, "sha256": digest})
    return records


def _verify_exact_membership(
    metadata: Mapping[str, Mapping[str, Any]],
    records: list[dict[str, Any]],
    *,
    manifest_member: str,
    record_path_key: str,
    prefix: str = "",
) -> None:
    expected: dict[str, dict[str, Any]] = {}
    for record in records:
        name = f"{prefix}{record[record_path_key]}"
        if name == manifest_member:
            raise ValueError("manifest cannot list itself as an evidence payload")
        expected[name] = {
            "bytes": record["bytes"],
            "sha256": record["sha256"],
        }
    expected_names = {manifest_member, *expected}
    actual_names = set(metadata)
    if actual_names != expected_names:
        raise ValueError(
            "archive membership differs from its manifest: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )
    for name, record in expected.items():
        actual = metadata[name]
        if actual["bytes"] != record["bytes"]:
            raise ValueError(f"archived byte count differs from manifest: {name}")
        if actual["sha256"] != record["sha256"]:
            raise ValueError(f"archived SHA-256 differs from manifest: {name}")


def _payload_merkle_root(records: list[dict[str, Any]]) -> str:
    leaves = [
        hashlib.sha256(
            f"{record['file']}\0{record['bytes']}\0{record['sha256']}".encode(
                "utf-8"
            )
        ).digest()
        for record in sorted(records, key=lambda item: item["file"])
    ]
    while len(leaves) > 1:
        if len(leaves) % 2:
            leaves.append(leaves[-1])
        leaves = [
            hashlib.sha256(leaves[index] + leaves[index + 1]).digest()
            for index in range(0, len(leaves), 2)
        ]
    return leaves[0].hex()


def _semantic_artifact_root(records: list[dict[str, Any]]) -> str:
    leaves = [
        hashlib.sha256(
            json.dumps(
                {
                    "file": record["file"],
                    "bytes": record["bytes"],
                    "sha256": record["sha256"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).digest()
        for record in sorted(records, key=lambda item: item["file"])
        if record["file"] != _CORE_VALIDATION_RECEIPT
    ]
    if not leaves:
        return "0" * 64
    while len(leaves) > 1:
        if len(leaves) % 2:
            leaves.append(leaves[-1])
        leaves = [
            hashlib.sha256(leaves[index] + leaves[index + 1]).digest()
            for index in range(0, len(leaves), 2)
        ]
    return leaves[0].hex()


def _validate_core_semantic_receipt(
    payload: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    records: list[dict[str, Any]],
) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("core semantic validation receipt schema must be 1")
    if payload.get("validation_status") != "PASS":
        raise ValueError("core semantic validation did not pass")
    if payload.get("validation_scope") != "core_publication_evidence":
        raise ValueError("core semantic receipt has the wrong validation scope")
    for key in (
        "git_commit",
        "simulation_source_commit",
        "publication_code_commit",
    ):
        if payload.get(key) != manifest.get(key):
            raise ValueError(f"core semantic receipt disagrees on {key}")
    if payload.get("run_tag") != manifest.get("artifact_run_tag"):
        raise ValueError("core semantic receipt run tag differs from the manifest")
    dual = manifest.get("dual_provenance") is True
    if dual:
        if (
            payload.get("fresh_single_commit_run") is not False
            or payload.get("authorized_deterministic_recovery") is not True
            or payload.get("simulation_rerun") is not False
            or payload.get("recovery_authorization")
            != manifest.get("recovery_authorization")
        ):
            raise ValueError(
                "core semantic receipt does not attest the exact authorized "
                "deterministic recovery"
            )
    else:
        if payload.get("fresh_single_commit_run") is not True:
            raise ValueError(
                "core semantic receipt does not attest a fresh single-commit run"
            )
        recovery_fields = {
            "authorized_deterministic_recovery",
            "simulation_rerun",
            "recovery_authorization",
        }
        present_recovery_fields = recovery_fields.intersection(payload)
        if present_recovery_fields and (
            present_recovery_fields != recovery_fields
            or payload.get("authorized_deterministic_recovery") is not False
            or payload.get("simulation_rerun") is not True
            or payload.get("recovery_authorization") is not None
        ):
            raise ValueError(
                "fresh core semantic receipt contains inconsistent recovery fields"
            )
    if payload.get("locked_accounting") != _CORE_ACCOUNTING:
        raise ValueError("core semantic receipt has incorrect locked accounting")
    expected_scope = {
        "included_in_core_receipt": False,
        "required_for_full_submission_evidence": True,
        "required_separate_receipt": "structural_sensitivity_archive_receipt.json",
    }
    if payload.get("structural_sensitivity") != expected_scope:
        raise ValueError("core receipt does not preserve the separate structural scope")
    semantic_records = [
        record for record in records
        if record["file"] != _CORE_VALIDATION_RECEIPT
    ]
    expected_artifact_set = {
        "artifact_count_excluding_receipt": len(semantic_records),
        "merkle_root": _semantic_artifact_root(records),
        "excluded_from_root": [_CORE_VALIDATION_RECEIPT],
        "hash_semantics": "manifested literal bytes",
    }
    if payload.get("semantic_artifact_set") != expected_artifact_set:
        raise ValueError("core semantic receipt is not bound to this artifact set")
    if payload.get("validated_checks") != _VALIDATED_CORE_CHECKS:
        raise ValueError("core semantic receipt lacks the exact validation inventory")
    _require_utc_timestamp(
        payload.get("generated_at_utc"), label="core semantic receipt timestamp"
    )
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("core semantic receipt lacks a protocol binding")
    if protocol.get("file") != "mvp/simulation/experiment_protocol.json":
        raise ValueError("core semantic receipt binds the wrong protocol path")
    if not isinstance(protocol.get("bytes"), int) or protocol["bytes"] <= 0:
        raise ValueError("core semantic receipt has an invalid protocol byte count")
    _require_hex(
        protocol.get("sha256"), label="core protocol SHA-256", width=64
    )


def _validate_core_submission_receipt(
    payload: Mapping[str, Any], *, source_commit: str, run_tag: str,
) -> None:
    if payload.get("schema_version") != 2:
        raise ValueError("core Slurm submission receipt schema must be 2")
    if payload.get("analysis_label") != "core stochastic publication evidence":
        raise ValueError("core Slurm submission receipt has the wrong label")
    if payload.get("execution_scope") != "core_publication_evidence":
        raise ValueError("core Slurm submission receipt has the wrong scope")
    if payload.get("receipt_scope") != (
        "submission_only_not_scheduler_completion"
    ) or payload.get("scheduler_completion_attested") is not False:
        raise ValueError("core Slurm receipt overclaims scheduler completion")
    if payload.get("source_commit") != source_commit:
        raise ValueError("core Slurm submission receipt has the wrong commit")
    if payload.get("run_tag") != run_tag:
        raise ValueError("core Slurm submission receipt has the wrong run tag")
    if payload.get("source_tree_clean_at_submission") is not True:
        raise ValueError("core Slurm submission receipt does not attest clean source")
    if payload.get("source_snapshot_mode") != (
        "detached_readonly_git_worktree_v1"
    ):
        raise ValueError("core Slurm receipt has the wrong source snapshot mode")
    _require_hex(
        payload.get("source_tree_sha256"),
        label="core Slurm source-tree SHA-256",
        width=64,
    )
    if payload.get("deterministic_mode") is not False:
        raise ValueError("core Slurm submission receipt is not stochastic")
    partition = payload.get("partition")
    if not isinstance(partition, str) or not _PARTITION.fullmatch(partition):
        raise ValueError("core Slurm submission receipt has an invalid partition")
    _require_utc_timestamp(
        payload.get("submitted_at_utc"), label="core Slurm submission timestamp"
    )
    if payload.get("locked_core_accounting") != _CORE_EXECUTION_ACCOUNTING:
        raise ValueError("core Slurm submission receipt has incorrect accounting")

    dag = payload.get("slurm_dag")
    if not isinstance(dag, dict) or set(dag) != {
        "seed_array", "stress_array", "publisher",
    }:
        raise ValueError("core Slurm submission receipt lacks the exact DAG")
    seed = dag["seed_array"]
    stress = dag["stress_array"]
    publisher = dag["publisher"]
    if not all(isinstance(stage, dict) for stage in (seed, stress, publisher)):
        raise ValueError("core Slurm DAG stages must be objects")
    job_ids = [stage.get("job_id") for stage in (seed, stress, publisher)]
    if any(not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id)
           for job_id in job_ids):
        raise ValueError("core Slurm DAG has an invalid job id")
    seed_id, stress_id, publisher_id = job_ids
    if len(set(job_ids)) != 3:
        raise ValueError("core Slurm DAG job ids are not distinct")
    expected_seed = {
        "job_id": seed_id,
        "script": "hpc/hpc_seed.sh",
        "array_indices": "0-19",
        "task_count": 20,
        "seeds": _SEEDS,
        "dependency_type": None,
        "afterok_job_ids": [],
    }
    expected_stress = {
        "job_id": stress_id,
        "script": "hpc/hpc_stress.sh",
        "array_indices": "0-4",
        "task_count": 5,
        "scenarios": _SCENARIOS,
        "dependency_type": "afterok",
        "afterok_job_ids": [seed_id],
    }
    expected_publisher = {
        "job_id": publisher_id,
        "script": "hpc/hpc_publish.sh",
        "array_indices": None,
        "task_count": 1,
        "dependency_type": "afterok",
        "afterok_job_ids": [seed_id, stress_id],
    }
    if seed != expected_seed or stress != expected_stress:
        raise ValueError("core Slurm arrays or dependencies are inconsistent")
    if publisher != expected_publisher:
        raise ValueError("core Slurm publisher dependencies are inconsistent")
    _validate_self_hash(
        payload, field="receipt_sha256", label="core Slurm submission receipt"
    )


def _run_canonical_core_validation(
    extracted_results: Path,
    *,
    repo_root: Path,
    recovery_receipt: Path | None = None,
) -> None:
    """Run the same complete semantic gate used by the HPC publisher."""

    from mvp.simulation.validation.validate_publication_artifacts import (
        validate_full_publication_release,
    )

    # The extracted archive carries exactly the manifested artifacts — the
    # stressed H3 ledgers but never the adaptation/episode evidence tree,
    # whose completeness was validated on the live results tree by the
    # publisher and bound into the semantic receipt verified alongside.
    validate_full_publication_release(
        extracted_results,
        repo_root=repo_root,
        recovery_receipt=recovery_receipt,
        evidence_scope="archived-subset",
    )


def _validate_core_evidence(
    archive_path: Path,
    receipt_path: Path,
    ready_path: Path,
    *,
    complete_archive_path: Path | None = None,
    complete_receipt_path: Path | None = None,
    complete_ready_path: Path | None = None,
) -> dict[str, Any]:
    archive_path = _require_regular_file(archive_path, "core evidence archive")
    receipt_path = _require_regular_file(receipt_path, "core archive receipt")
    ready_path = _require_regular_file(ready_path, "core bundle READY marker")
    receipt = _load_json_file(receipt_path, "core archive receipt")
    if receipt.get("schema_version") != 1:
        raise ValueError("core archive receipt schema must be 1")
    fresh_receipt = (
        receipt.get("derivation_type")
        == "fresh stochastic simulation and publication build"
        and receipt.get("simulation_rerun") is True
        and receipt.get("recovery_authorization") is None
    )
    recovery_receipt = (
        receipt.get("derivation_type")
        == "publication-only deterministic recovery"
        and receipt.get("simulation_rerun") is False
        and isinstance(receipt.get("recovery_authorization"), dict)
    )
    if fresh_receipt == recovery_receipt:
        raise ValueError(
            "core archive receipt must prove exactly one fresh or authorized "
            "recovery derivation"
        )
    if receipt.get("parent_archive_sha256") is not None:
        raise ValueError("core evidence must not use an unbound parent archive")

    archive_record = receipt.get("archive")
    if not isinstance(archive_record, dict):
        raise ValueError("core archive receipt lacks archive metadata")
    if archive_record.get("file") != archive_path.name:
        raise ValueError("core archive filename differs from its receipt")
    if archive_record.get("bytes") != archive_path.stat().st_size:
        raise ValueError("core archive byte count differs from its receipt")
    archive_sha = _sha256_file(archive_path)
    if archive_record.get("sha256") != archive_sha:
        raise ValueError("core archive SHA-256 differs from its receipt")
    receipt_sha = _sha256_file(receipt_path)
    ready = _load_json_file(ready_path, "core bundle READY marker")
    expected_ready = {
        "schema_version": 1,
        "status": "READY",
        "archive": {
            "file": archive_path.name,
            "sha256": archive_sha,
        },
        "receipt": {
            "file": receipt_path.name,
            "sha256": receipt_sha,
        },
    }
    if ready != expected_ready:
        raise ValueError("core READY marker does not bind this archive/receipt pair")

    manifest_bytes, metadata = _read_archive(
        archive_path, required_manifest_name="artifact_manifest.json"
    )
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("core artifact manifest is not valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 2:
        raise ValueError("core artifact manifest schema must be 2")
    records = _manifest_records(manifest, key="artifacts", path_key="file")
    if manifest.get("artifact_count") != len(records):
        raise ValueError("core artifact manifest count is inconsistent")
    _verify_exact_membership(
        metadata,
        records,
        manifest_member="artifact_manifest.json",
        record_path_key="file",
    )

    manifest_record = receipt.get("manifest")
    if not isinstance(manifest_record, dict):
        raise ValueError("core archive receipt lacks manifest metadata")
    if manifest_record.get("bytes") != len(manifest_bytes):
        raise ValueError("core manifest byte count differs from archive receipt")
    if manifest_record.get("sha256") != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError("core manifest SHA-256 differs from archive receipt")
    if manifest_record.get("artifact_count") != len(records):
        raise ValueError("core receipt artifact count differs from its manifest")
    if manifest_record.get("payload_merkle_root") != _payload_merkle_root(records):
        raise ValueError("core receipt payload Merkle root is inconsistent")
    if manifest_record.get("hash_semantics") != "literal bytes":
        raise ValueError("core receipt does not declare literal-byte hashing")
    if archive_record.get("member_count") != len(metadata):
        raise ValueError("core receipt archive member count is inconsistent")

    git_commit = _require_hex(
        manifest.get("git_commit"), label="core manifest git_commit", width=40
    )
    source_commit = _require_hex(
        manifest.get("simulation_source_commit"),
        label="core manifest simulation_source_commit",
        width=40,
    )
    publication_commit = _require_hex(
        manifest.get("publication_code_commit"),
        label="core manifest publication_code_commit",
        width=40,
    )
    dual = manifest.get("dual_provenance") is True
    if dual:
        if not (
            recovery_receipt
            and git_commit == source_commit
            and source_commit != publication_commit
            and isinstance(manifest.get("recovery_authorization"), dict)
            and receipt.get("recovery_authorization")
            == manifest.get("recovery_authorization")
        ):
            raise ValueError(
                "core dual provenance lacks one exact recovery authorization"
            )
        provenance_mode = "authorized_publication_recovery"
    elif not (
        fresh_receipt
        and manifest.get("dual_provenance") is False
        and git_commit == source_commit == publication_commit
        and manifest.get("recovery_authorization") is None
    ):
        raise ValueError(
            "core fresh evidence does not use one clean source commit"
        )
    else:
        provenance_mode = "fresh_single_commit"
    if manifest.get("git_dirty") is not False:
        raise ValueError("core evidence was generated from a dirty source tree")
    if manifest.get("includes_raw_run_artifacts") is not True:
        raise ValueError("core evidence archive omits raw run artifacts")
    run_tag = manifest.get("artifact_run_tag")
    match = _CORE_RUN_TAG.fullmatch(str(run_tag))
    if match is None or match.group(1) != source_commit[:7]:
        raise ValueError("core run tag is not bound to its source commit")

    for key, expected in (
        ("simulation_source_commit", source_commit),
        ("publication_code_commit", publication_commit),
        ("run_tag", run_tag),
    ):
        if receipt.get(key) != expected:
            raise ValueError(f"core archive receipt disagrees on {key}")
    validator_paths = [
        "mvp/simulation/results/artifact_manifest.json",
        *(f"mvp/simulation/results/{record['file']}" for record in records),
    ]
    expected_validator_identity = {
        "head_commit": publication_commit,
        "source_tree_clean_outside_exact_evidence_paths": True,
        "status_includes_untracked_files": True,
        "allowed_evidence_path_count": len(validator_paths),
        "allowed_evidence_path_set_sha256": hashlib.sha256(
            "\n".join(sorted(validator_paths)).encode("utf-8")
        ).hexdigest(),
    }
    if receipt.get("validator_source_identity") != expected_validator_identity:
        raise ValueError(
            "core archive receipt does not bind the clean executing validator "
            "checkout to its exact manifested evidence allowlist"
        )
    expected_validation = {
        "prearchive_payload_hashes": "PASS",
        "postarchive_payload_hashes": "PASS",
        "exact_manifest_membership": "PASS",
        "safe_regular_members_only": "PASS",
        "semantic_validation_receipt_manifested_and_verified": "PASS",
        "validator_checkout_same_clean_commit_outside_exact_evidence": "PASS",
    }
    if dual:
        expected_validation["publication_recovery_receipt_and_preserved_inputs"] = (
            "PASS"
        )
        accepted_validations = (expected_validation,)
    else:
        # The recovery-aware archive producer emits an explicit
        # NOT_APPLICABLE projection, while already-issued fresh receipts use
        # the legacy dictionary without that recovery-only key.  Both are
        # exact, fail-closed fresh schemas; no other value or field is allowed.
        producer_fresh_validation = {
            **expected_validation,
            "publication_recovery_receipt_and_preserved_inputs": (
                "NOT_APPLICABLE"
            ),
        }
        accepted_validations = (
            expected_validation,
            producer_fresh_validation,
        )
    if receipt.get("validation") not in accepted_validations:
        raise ValueError("core archive receipt lacks exact passing archive checks")
    expected_scope = {
        "core_publication_evidence": True,
        "structural_sensitivity_included": False,
        "full_submission_requires_separate_structural_receipt": True,
    }
    if receipt.get("evidence_scope") != expected_scope:
        raise ValueError("core archive receipt has an invalid evidence scope")
    _require_utc_timestamp(
        receipt.get("generated_at_utc"), label="core archive receipt timestamp"
    )

    validation_name = _CORE_VALIDATION_RECEIPT
    if validation_name not in metadata:
        raise ValueError("core archive lacks its semantic validation receipt")
    semantic_receipt = _json_member(archive_path, validation_name)
    _validate_core_semantic_receipt(
        semantic_receipt, manifest=manifest, records=records
    )
    submission_name = f"core_submission_receipts/{run_tag}.json"
    if submission_name not in metadata:
        raise ValueError("core archive lacks its immutable Slurm DAG receipt")
    submission_receipt = _json_member(archive_path, submission_name)
    _validate_core_submission_receipt(
        submission_receipt,
        source_commit=source_commit,
        run_tag=str(run_tag),
    )
    recovery_evidence: dict[str, Any] | None = None
    complete_raw_evidence: dict[str, Any] | None = None
    if dual:
        recovery_evidence = _validate_archived_recovery(
            archive_path,
            metadata=metadata,
            prefix="",
            kind="core",
            run_tag=str(run_tag),
            simulation_commit=source_commit,
            publication_commit=publication_commit,
            simulation_source_tree_sha256=submission_receipt[
                "source_tree_sha256"
            ],
            original_submission_member=submission_name,
            manifested_authorization=manifest["recovery_authorization"],
        )
        complete_paths = (
            complete_archive_path,
            complete_receipt_path,
            complete_ready_path,
        )
        if any(path is None for path in complete_paths):
            raise ValueError(
                "core publication recovery requires the complete raw archive, "
                "receipt, and READY marker"
            )
        raw_manifest_payload = recovery_evidence.pop("_raw_manifest_payload")
        complete_raw_evidence = _validate_core_complete_run_evidence(
            complete_archive_path.absolute(),  # type: ignore[union-attr]
            complete_receipt_path.absolute(),  # type: ignore[union-attr]
            complete_ready_path.absolute(),  # type: ignore[union-attr]
            run_tag=str(run_tag),
            simulation_commit=source_commit,
            simulation_source_tree_sha256=submission_receipt[
                "source_tree_sha256"
            ],
            raw_manifest=raw_manifest_payload,
        )
        recovery_evidence["raw_archive_binding"] = complete_raw_evidence[
            "raw_mapping"
        ]
    elif any(
        path is not None
        for path in (
            complete_archive_path,
            complete_receipt_path,
            complete_ready_path,
        )
    ):
        raise ValueError(
            "complete raw bundle arguments are reserved for authorized recovery"
        )
    repo_root = REPO_ROOT
    with tempfile.TemporaryDirectory(prefix="agribrain_core_evidence_") as temp:
        extracted_results = Path(temp) / "results"
        _extract_safe_archive(archive_path, extracted_results, metadata)
        try:
            _run_canonical_core_validation(
                extracted_results,
                repo_root=repo_root,
                recovery_receipt=(
                    extracted_results / recovery_evidence["receipt_file"]
                    if recovery_evidence is not None
                    else None
                ),
            )
        except Exception as exc:
            raise ValueError(
                f"canonical full core evidence validation failed: {exc}"
            ) from exc
    return {
        "source_commit": source_commit,
        "simulation_source_commit": source_commit,
        "publication_code_commit": publication_commit,
        "dual_provenance": dual,
        "simulation_rerun": not dual,
        "provenance_mode": provenance_mode,
        "recovery_authorization": recovery_evidence,
        "complete_raw_run_evidence": complete_raw_evidence,
        "run_tag": str(run_tag),
        "archive": {
            "name": archive_path.name,
            "bytes": archive_path.stat().st_size,
            "sha256": archive_sha,
            "member_count": len(metadata),
        },
        "archive_receipt": {
            "name": receipt_path.name,
            "bytes": receipt_path.stat().st_size,
            "sha256": receipt_sha,
        },
        "atomic_bundle_ready_marker": {
            "name": ready_path.name,
            "bytes": ready_path.stat().st_size,
            "sha256": _sha256_file(ready_path),
            "status": "READY",
        },
        "artifact_manifest": {
            "bytes": len(manifest_bytes),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "artifact_count": len(records),
        },
        "semantic_validation_receipt": {
            "name": validation_name,
            "bytes": metadata[validation_name]["bytes"],
            "sha256": metadata[validation_name]["sha256"],
            "validation_status": "PASS",
            "locked_accounting": dict(_CORE_ACCOUNTING),
        },
        "slurm_submission_receipt": {
            "name": submission_name,
            "bytes": metadata[submission_name]["bytes"],
            "sha256": metadata[submission_name]["sha256"],
            "partition": submission_receipt["partition"],
            "receipt_scope": submission_receipt["receipt_scope"],
            "source_snapshot_mode": submission_receipt["source_snapshot_mode"],
            "source_tree_sha256": submission_receipt["source_tree_sha256"],
            "seed_job_id": submission_receipt["slurm_dag"]["seed_array"][
                "job_id"
            ],
            "stress_job_id": submission_receipt["slurm_dag"]["stress_array"][
                "job_id"
            ],
            "publisher_job_id": submission_receipt["slurm_dag"]["publisher"][
                "job_id"
            ],
        },
    }


def _validate_structural_accounting(accounting: object) -> None:
    if not isinstance(accounting, dict):
        raise ValueError("structural manifest lacks accounting")
    if accounting.get("analysis_label") != "structural sensitivity":
        raise ValueError("structural accounting has the wrong analysis label")
    if accounting.get("probability_interpretation") is not False:
        raise ValueError("structural accounting is incorrectly probabilistic")
    if accounting.get("n_design_points") != 100:
        raise ValueError("structural accounting must contain 100 LHS points")
    if accounting.get("n_scenarios") != 5:
        raise ValueError("structural accounting must contain five scenarios")
    if accounting.get("steps_per_episode") != 288:
        raise ValueError("structural accounting must use 288 steps per episode")
    if accounting.get("episodes_per_stressed_agribrain_cell") != 4:
        raise ValueError("structural H3 cells must execute four episodes")
    if accounting.get("total") != _STRUCTURAL_TOTALS:
        raise ValueError("structural accounting totals do not match the protocol")
    expected_per_point = {
        "primary_retained_cells": 40,
        "primary_executed_episodes": 145,
        "h3_stressed_retained_cells": 25,
        "h3_stressed_executed_episodes": 100,
        "total_retained_cells": 65,
        "total_executed_episodes": 245,
    }
    if accounting.get("per_design_point") != expected_per_point:
        raise ValueError("structural per-design-point accounting is inconsistent")
    if accounting.get("primary_modes") != [
        "static",
        "hybrid_rl",
        "no_pinn",
        "no_slca",
        "no_context",
        "mcp_only",
        "pirag_only",
        "agribrain",
    ]:
        raise ValueError("structural accounting has the wrong primary modes")
    if accounting.get("episode_budget_by_primary_mode") != {
        "static": 1,
        "hybrid_rl": 4,
        "no_pinn": 4,
        "no_slca": 4,
        "no_context": 4,
        "mcp_only": 4,
        "pirag_only": 4,
        "agribrain": 4,
    }:
        raise ValueError("structural primary episode budgets are inconsistent")


def _validate_structural_inventory(
    archive_path: Path,
    *,
    metadata: Mapping[str, Mapping[str, Any]],
    records: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    run_plan: Mapping[str, Any],
    publication_root: str,
) -> None:
    artifacts = run_plan.get("artifacts")
    expected_artifact_keys = {
        "parameter_registry",
        "lhs_design",
        "lhs_design_csv",
        "task_manifest",
        "task_manifest_jsonl",
        "episode_accounting",
        "locked_protocol",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifact_keys:
        raise ValueError("structural run plan lacks the exact artifact bundle")
    artifact_names = {
        key: _safe_name(value, label=f"structural {key}")
        for key, value in artifacts.items()
    }
    artifact_hashes = run_plan.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != set(
        artifact_names.values()
    ):
        raise ValueError("structural run plan artifact hash inventory is incomplete")
    for name, digest in artifact_hashes.items():
        _require_hex(digest, label=f"structural run-plan artifact {name}", width=64)
        archived_name = f"{_STRUCTURAL_PREFIX}/{name}"
        if archived_name not in metadata or metadata[archived_name]["sha256"] != digest:
            raise ValueError(f"structural run-plan artifact is missing or altered: {name}")

    task_manifest_name = (
        f"{_STRUCTURAL_PREFIX}/{artifact_names['task_manifest']}"
    )
    task_manifest = _json_member(archive_path, task_manifest_name)
    _validate_self_hash(
        task_manifest,
        field="manifest_sha256",
        label="structural task manifest",
    )
    if task_manifest.get("schema_version") != 1:
        raise ValueError("structural task manifest schema must be 1")
    if task_manifest.get("analysis_label") != "structural sensitivity":
        raise ValueError("structural task manifest has the wrong label")
    if task_manifest.get("probability_interpretation") is not False:
        raise ValueError("structural task manifest is incorrectly probabilistic")
    if task_manifest.get("design_sha256") != manifest.get("design_sha256"):
        raise ValueError("structural task and artifact manifests bind different designs")
    if task_manifest.get("accounting") != manifest.get("accounting"):
        raise ValueError("structural task and artifact accounting differ")
    tasks = task_manifest.get("tasks")
    if (
        task_manifest.get("n_tasks") != 3_000
        or not isinstance(tasks, list)
        or len(tasks) != 3_000
    ):
        raise ValueError("structural task manifest lacks the exact 3,000 tasks")

    task_paths: set[str] = set()
    ledger_paths: set[str] = set()
    task_ids: set[str] = set()
    point_counts: dict[str, int] = {}
    primary_cells: set[tuple[str, str]] = set()
    stressed_cells: set[tuple[str, str, str]] = set()
    retained_total = 0
    executed_total = 0
    steps_total = 0
    primary_modes = [
        "static", "hybrid_rl", "no_pinn", "no_slca", "no_context",
        "mcp_only", "pirag_only", "agribrain",
    ]
    scenarios = set(_SCENARIOS)
    stressors = {
        "sensor_noise", "missing_data", "telemetry_delay",
        "mcp_fault_injection", "compounded",
    }
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"structural task {index} is not an object")
        _validate_self_hash(
            task, field="task_sha256", label=f"structural task {index}"
        )
        if task.get("task_index") != index:
            raise ValueError("structural task indices are not exactly 0..2,999")
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or not task_id or task_id in task_ids:
            raise ValueError("structural task ids are invalid or duplicated")
        task_ids.add(task_id)
        point_id = task.get("point_id")
        if not isinstance(point_id, str) or not re.fullmatch(r"lhs_[0-9]{3}", point_id):
            raise ValueError("structural task has an invalid LHS point id")
        if task.get("point_index") != int(point_id[-3:]):
            raise ValueError("structural task point id and point index disagree")
        point_counts[point_id] = point_counts.get(point_id, 0) + 1
        scenario = task.get("scenario")
        if scenario not in scenarios:
            raise ValueError("structural task has an invalid scenario")
        relative = _safe_name(task.get("output_relpath"), label="structural task")
        if relative in task_paths:
            raise ValueError("structural task output paths are duplicated")
        task_paths.add(relative)
        panel = task.get("panel")
        if panel == "primary":
            expected_path = f"tasks/{point_id}/{scenario}__primary.json"
            if relative != expected_path or task.get("modes") != primary_modes:
                raise ValueError("structural primary task identity is inconsistent")
            if (
                task.get("retained_cells") != 8
                or task.get("executed_episodes") != 29
                or task.get("simulated_steps") != 8_352
            ):
                raise ValueError("structural primary task accounting is inconsistent")
            primary_cells.add((point_id, str(scenario)))
            artifact_root = PurePosixPath(relative).parent / (
                f"{PurePosixPath(relative).stem}__artifacts"
            )
            ledger_paths.update(
                (
                    artifact_root / "runtime_artifacts" / "decision_ledger"
                    / f"{mode}__{scenario}.jsonl"
                ).as_posix()
                for mode in primary_modes
            )
        elif panel == "h3_stressed":
            stressor = task.get("stressor")
            expected_path = f"tasks/{point_id}/{scenario}__h3__{stressor}.json"
            if stressor not in stressors or relative != expected_path:
                raise ValueError("structural stressed task identity is inconsistent")
            if task.get("modes") != ["agribrain"]:
                raise ValueError("structural stressed task has the wrong mode")
            if (
                task.get("retained_cells") != 1
                or task.get("executed_episodes") != 4
                or task.get("simulated_steps") != 1_152
            ):
                raise ValueError("structural stressed task accounting is inconsistent")
            if task.get("nominal_reference_task_id") != (
                f"{point_id}__{scenario}__primary"
            ):
                raise ValueError("structural stressed task has the wrong nominal reference")
            stressed_cells.add((point_id, str(scenario), str(stressor)))
            artifact_root = PurePosixPath(relative).parent / (
                f"{PurePosixPath(relative).stem}__artifacts"
            )
            ledger_paths.add((
                artifact_root / "decision_ledgers" / str(scenario)
                / f"structural__{point_id}__{stressor}"
                / f"seed_{int(task['seed'])}"
                / f"agribrain__{scenario}.jsonl"
            ).as_posix())
        else:
            raise ValueError("structural task has an invalid panel")
        retained_total += int(task["retained_cells"])
        executed_total += int(task["executed_episodes"])
        steps_total += int(task["simulated_steps"])

    expected_points = {f"lhs_{index:03d}" for index in range(100)}
    if set(point_counts) != expected_points or any(
        count != 30 for count in point_counts.values()
    ):
        raise ValueError("structural task inventory is not 30 tasks per LHS point")
    if len(primary_cells) != 500 or len(stressed_cells) != 2_500:
        raise ValueError("structural task inventory lacks the exact two panels")
    if len(ledger_paths) != 6_500:
        raise ValueError(
            "structural task inventory does not identify 6,500 unique ledgers"
        )
    if {
        "retained_cells": retained_total,
        "executed_episodes": executed_total,
        "simulated_steps": steps_total,
    } != _STRUCTURAL_TOTALS:
        raise ValueError("structural task totals differ from locked accounting")

    required_final = {
        "slurm_submission.json",
        *(
            _publication_member(publication_root, name)
            for name in (
                "completion_status.json",
                "structural_sensitivity_analysis.json",
                "publication_environment.json",
                "slurm_simulation_accounting.json",
                "structural_sensitivity_summary.csv",
                "structural_sensitivity_summary.png",
                "structural_sensitivity_summary.pdf",
                "structural_sensitivity_publication_receipt.json",
            )
        ),
    }
    expected_records = {
        "run_plan.json", *artifact_names.values(), *task_paths, *ledger_paths,
        *required_final,
    }
    observed_records = {str(record["path"]) for record in records}
    if not expected_records.issubset(observed_records):
        raise ValueError(
            "structural archive does not contain the exact 3,000 task outputs, "
            "6,500 retained decision ledgers, and required canonical evidence "
            "files"
        )

    def is_canonical_task_artifact(path: str) -> bool:
        parts = PurePosixPath(path).parts
        return any(part.endswith("__artifacts") for part in parts) and not any(
            part.endswith("__attempts") for part in parts
        )

    episode_archives = {
        path for path in observed_records
        if "/complete_episode_evidence/" in path and path.endswith(".json.gz")
        and is_canonical_task_artifact(path)
    }
    adaptation_ledgers = {
        path for path in observed_records
        if "/adaptation_episode_ledgers/" in path and path.endswith(".jsonl.gz")
        and is_canonical_task_artifact(path)
    }
    episode_manifests = {
        path for path in observed_records
        if path.endswith("__artifacts/complete_episode_evidence_manifest.json")
    }
    runtime_receipts = {
        path for path in observed_records if path.startswith("runtime_receipts/")
        and path.endswith(".json")
    }
    if (
        len(episode_archives) != 24_500
        or len(adaptation_ledgers) != 18_000
        or len(episode_manifests) != 3_000
        or len(runtime_receipts) < 3_000
    ):
        raise ValueError(
            "structural archive lacks the complete 24,500-episode evidence "
            "inventory or one worker receipt per task"
        )
    if manifest.get("retained_decision_ledger_count") != 6_500:
        raise ValueError("structural manifest does not declare 6,500 retained ledgers")
    complete = manifest.get("complete_episode_evidence")
    runtime = complete.get("runtime_receipts", {}) if isinstance(complete, dict) else {}
    scheduler = (
        complete.get("scheduler_accounting", {})
        if isinstance(complete, dict)
        else {}
    )
    failed = (
        complete.get("failed_attempt_artifacts", {})
        if isinstance(complete, dict)
        else {}
    )
    failed_attempt_records = [
        record for record in records
        if any(
            part.endswith("__attempts")
            for part in PurePosixPath(str(record["path"])).parts
        )
    ]
    failed_attempt_bytes = sum(
        int(record["bytes"]) for record in failed_attempt_records
    )
    runtime_wall = runtime.get("summed_task_wall_seconds_nonconcurrent")
    runtime_cpu = runtime.get("summed_child_cpu_seconds")
    runtime_failed = runtime.get("failed_attempt_receipts")
    scheduler_rows = scheduler.get("accounting_row_count")
    scheduler_sha256 = scheduler.get("accounting_sha256")
    failed_file_count = (
        failed.get("file_count") if isinstance(failed, dict) else None
    )
    failed_bytes = (
        failed.get("literal_bytes") if isinstance(failed, dict) else None
    )
    if (
        not isinstance(complete, dict)
        or set(complete) != {
            "executed_episode_archives",
            "adaptation_episode_ledgers",
            "final_episode_ledgers",
            "per_task_manifests",
            "runtime_receipts",
            "scheduler_accounting",
            "failed_attempt_artifacts",
        }
        or complete.get("executed_episode_archives") != 24_500
        or complete.get("adaptation_episode_ledgers") != 18_000
        or complete.get("final_episode_ledgers") != 6_500
        or complete.get("per_task_manifests") != 3_000
        or not isinstance(runtime, dict)
        or set(runtime) != {
            "successful_task_receipts",
            "failed_attempt_receipts",
            "total_receipts",
            "summed_task_wall_seconds_nonconcurrent",
            "summed_child_cpu_seconds",
        }
        or runtime.get("successful_task_receipts") != 3_000
        or isinstance(runtime_failed, bool)
        or not isinstance(runtime_failed, int)
        or runtime_failed != len(runtime_receipts) - 3_000
        or runtime.get("total_receipts") != len(runtime_receipts)
        or isinstance(runtime_wall, bool)
        or not isinstance(runtime_wall, (int, float))
        or not math.isfinite(float(runtime_wall))
        or float(runtime_wall) < 0.0
        or isinstance(runtime_cpu, bool)
        or not isinstance(runtime_cpu, (int, float))
        or not math.isfinite(float(runtime_cpu))
        or float(runtime_cpu) < 0.0
        or not isinstance(scheduler, dict)
        or set(scheduler) != {
            "array_count",
            "completed_simulation_task_count",
            "accounting_row_count",
            "energy",
            "accounting_sha256",
        }
        or scheduler.get("array_count") != 3
        or scheduler.get("completed_simulation_task_count") != 3_000
        or isinstance(scheduler_rows, bool)
        or not isinstance(scheduler_rows, int)
        or scheduler_rows < 3_000
        or not isinstance(scheduler.get("energy"), dict)
        or not isinstance(scheduler_sha256, str)
        or not _HEX64.fullmatch(scheduler_sha256)
        or isinstance(failed_file_count, bool)
        or not isinstance(failed_file_count, int)
        or isinstance(failed_bytes, bool)
        or not isinstance(failed_bytes, int)
        or failed != {
            "file_count": len(failed_attempt_records),
            "literal_bytes": failed_attempt_bytes,
            "retention_policy": (
                "Retained for diagnosis and audit; excluded from canonical "
                "episode and ledger counts."
            ),
        }
    ):
        raise ValueError(
            "structural manifest has incomplete episode, runtime, scheduler, "
            "or failed-attempt evidence"
        )
    expected_excluded_runtime_material = (
        "temporary files, interpreter caches, the in-progress publisher log, "
        "and prior or legacy publication-attempt artifacts only; every durable "
        "simulation task artifact and every artifact of this authorized recovery "
        "attempt is included"
        if publication_root
        else (
            "temporary files, interpreter caches, and the in-progress publisher "
            "log only; every durable task artifact, episode archive, adaptation "
            "ledger, final ledger, worker runtime receipt, and completed task log "
            "is included"
        )
    )
    if manifest.get("excluded_runtime_material") != (
        expected_excluded_runtime_material
    ):
        raise ValueError("structural manifest has an invalid evidence exclusion policy")
    ledger_records = sorted(
        (
            record for record in records
            if str(record["path"]) in ledger_paths
        ),
        key=lambda record: str(record["path"]),
    )
    if (
        len(ledger_records) != 6_500
        or manifest.get("retained_decision_ledger_set_sha256")
        != _canonical_sha256(ledger_records)
    ):
        raise ValueError("structural retained-ledger set binding is invalid")

    completion = _json_member(
        archive_path,
        f"{_STRUCTURAL_PREFIX}/"
        f"{_publication_member(publication_root, 'completion_status.json')}",
    )
    if completion != {
        "status": "complete",
        "n_expected_tasks": 3_000,
        "n_valid_tasks": 3_000,
        "n_missing_tasks": 0,
        "missing_task_ids": [],
    }:
        raise ValueError("structural completion status is not the exact full panel")
    episode_accounting = _json_member(
        archive_path,
        f"{_STRUCTURAL_PREFIX}/{artifact_names['episode_accounting']}",
    )
    if episode_accounting != manifest.get("accounting"):
        raise ValueError("structural episode-accounting artifact is inconsistent")


def _run_canonical_structural_validation(
    extracted_root: Path,
    *,
    publication_root: Path,
    source_commit: str,
    publication_commit: str,
    run_tag: str,
) -> None:
    """Re-run every structural semantic validator without running simulations."""

    extracted_root = extracted_root.resolve(strict=True)
    publication_root = publication_root.resolve(strict=True)
    if not publication_root.is_dir() or not publication_root.is_relative_to(
        extracted_root
    ):
        raise ValueError(
            "structural publication root is outside the extracted evidence"
        )

    from hpc.capture_slurm_accounting import validate_accounting_payload
    from mvp.simulation.sensitivity.analyze_structural_sensitivity import (
        analyze_run,
    )
    from mvp.simulation.sensitivity.finalize_structural_sensitivity import (
        _load_json,
        _runtime_receipt_summary,
        _validate_analysis,
        _validate_environment,
        _validate_status,
        _validate_structural_publication_artifacts,
        _validate_submission,
    )
    from mvp.simulation.sensitivity.run_structural_sensitivity import (
        validate_completed_results_with_ledgers,
    )

    run_plan = extracted_root / "run_plan.json"
    submission = _load_json(extracted_root / "slurm_submission.json")
    _validate_submission(
        submission,
        run_tag=run_tag,
        source_commit=source_commit,
    )
    fresh_status, retained_ledgers = validate_completed_results_with_ledgers(
        run_plan, submission_receipt=submission,
    )
    _validate_status(fresh_status)
    if len(retained_ledgers) != 6_500:
        raise ValueError("structural semantic validation did not bind 6,500 ledgers")
    saved_status = _load_json(publication_root / "completion_status.json")
    _validate_status(saved_status)
    if saved_status != fresh_status:
        raise ValueError("structural saved status differs from fresh task validation")
    saved_analysis = _load_json(
        publication_root / "structural_sensitivity_analysis.json"
    )
    _validate_analysis(saved_analysis, source_commit=source_commit)
    if saved_analysis != analyze_run(run_plan):
        raise ValueError("structural analysis differs from fresh recomputation")
    _validate_environment(
        _load_json(publication_root / "publication_environment.json"),
        run_tag=run_tag,
        source_commit=publication_commit,
    )
    artifact_manifest = _load_json(
        extracted_root / _STRUCTURAL_MANIFEST
    )
    complete = artifact_manifest.get("complete_episode_evidence")
    if not isinstance(complete, dict):
        raise ValueError("structural manifest lacks complete episode evidence")
    runtime_paths, runtime_summary = _runtime_receipt_summary(
        extracted_root,
        run_tag=run_tag,
        source_commit=source_commit,
        source_tree_sha256=submission["source_tree_sha256"],
    )
    if runtime_summary != complete.get("runtime_receipts"):
        raise ValueError(
            "structural runtime-receipt summary differs from fresh validation"
        )
    manifest_paths = {
        str(record.get("path"))
        for record in artifact_manifest.get("files", [])
        if isinstance(record, dict)
    }
    if set(runtime_paths) - manifest_paths:
        raise ValueError("structural manifest omits a runtime receipt")
    scheduler_summary = validate_accounting_payload(
        _load_json(publication_root / "slurm_simulation_accounting.json"),
        kind="structural",
        run_tag=run_tag,
        source_commit=source_commit,
        source_tree_sha256=submission["source_tree_sha256"],
        expected_task_count=3_000,
    )
    if scheduler_summary != complete.get("scheduler_accounting"):
        raise ValueError(
            "structural scheduler-accounting summary differs from fresh validation"
        )
    publication_paths = _validate_structural_publication_artifacts(
        extracted_root,
        publication_root=publication_root,
        analysis_path=publication_root / "structural_sensitivity_analysis.json",
        source_commit=source_commit,
    )
    if set(publication_paths) - manifest_paths:
        raise ValueError("structural manifest omits a publication artifact")


def _validate_structural_evidence(
    archive_path: Path, receipt_path: Path,
) -> dict[str, Any]:
    archive_path = _require_regular_file(
        archive_path, "structural-sensitivity evidence archive"
    )
    receipt_path = _require_regular_file(
        receipt_path, "structural-sensitivity archive receipt"
    )
    receipt = _load_json_file(
        receipt_path, "structural-sensitivity archive receipt"
    )
    if receipt.get("schema_version") != 1:
        raise ValueError("structural archive receipt schema must be 1")
    if receipt.get("receipt_type") != (
        "structural_sensitivity_semantic_archive_receipt"
    ):
        raise ValueError("structural receipt is not the semantic archive receipt")
    if receipt.get("analysis_label") != "structural sensitivity":
        raise ValueError("structural archive receipt has the wrong analysis label")
    if receipt.get("validation_status") != "PASS":
        raise ValueError("structural semantic validation did not pass")
    expected_locked = {
        "active_factors": 29,
        "latin_hypercube_points": 100,
        "tasks": 3_000,
        "retained_cells": 6_500,
        "retained_decision_ledgers": 6_500,
        "retained_complete_episode_archives": 24_500,
        "retained_adaptation_episode_ledgers": 18_000,
        "successful_worker_runtime_receipts": 3_000,
        "scheduler_accounted_simulation_tasks": 3_000,
        "executed_episodes": 24_500,
        "simulated_steps": 7_056_000,
        "probability_interpretation": False,
    }
    if receipt.get("locked_accounting") != expected_locked:
        raise ValueError("structural semantic receipt has incorrect accounting")
    fresh_validation = {
        "complete_task_result_hashes": "PASS",
        "saved_status_matches_fresh_validation": "PASS",
        "saved_analysis_matches_fresh_recomputation": "PASS",
        "run_scoped_environment": "PASS",
        "contiguous_afterok_submission_dag": "PASS",
        "all_task_results_bound_to_slurm_arrays_and_source_snapshot": "PASS",
        "declared_publisher_runtime_identity": "PASS",
        "clean_fixed_validator_source_pre_and_post_archive": "PASS",
        "exact_manifest_membership_and_literal_hashes": "PASS",
        "retained_ledger_equations_and_endpoint_recomputation": "PASS",
        "complete_adaptation_and_evaluation_episode_archives": "PASS",
        "whole_worker_runtime_resource_receipts": "PASS",
        "post_job_scheduler_accounting_for_all_simulation_tasks": "PASS",
        "structural_table_figure_export_bound_to_analysis": "PASS",
    }
    recovery_validation = {
        **fresh_validation,
        "declared_publisher_runtime_identity": "NOT_APPLICABLE_RECOVERY",
        "authorized_recovery_publisher_runtime_identity": "PASS",
        "preserved_raw_outputs_revalidated_before_archive": "PASS",
        "dual_provenance_receipt_and_source_trees": "PASS",
        "simulation_rerun_prohibited": "PASS",
    }
    receipt_validation = receipt.get("validation")
    if receipt_validation not in (fresh_validation, recovery_validation):
        raise ValueError("structural semantic receipt lacks exact passing checks")
    receipt_recovery_mode = receipt_validation == recovery_validation
    expected_scope = {
        "structural_sensitivity_evidence": True,
        "core_publication_evidence_included": False,
        "full_submission_requires_core_receipt": True,
    }
    if receipt.get("evidence_scope") != expected_scope:
        raise ValueError("structural semantic receipt has an invalid evidence scope")
    receipt_digest = _validate_self_hash(
        receipt,
        field="receipt_sha256",
        label="structural archive receipt",
    )

    archive_record = receipt.get("archive")
    if not isinstance(archive_record, dict):
        raise ValueError("structural receipt lacks archive metadata")
    if archive_record.get("name") != archive_path.name:
        raise ValueError("structural archive filename differs from its receipt")
    if archive_record.get("bytes") != archive_path.stat().st_size:
        raise ValueError("structural archive byte count differs from its receipt")
    archive_sha = _sha256_file(archive_path)
    if archive_record.get("sha256") != archive_sha:
        raise ValueError("structural archive SHA-256 differs from its receipt")

    manifest_member = f"{_STRUCTURAL_PREFIX}/{_STRUCTURAL_MANIFEST}"
    manifest_bytes, metadata = _read_archive(
        archive_path, required_manifest_name=manifest_member
    )
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("structural manifest is not valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("structural artifact manifest schema must be 1")
    manifest_digest = _validate_self_hash(
        manifest,
        field="manifest_sha256",
        label="structural artifact manifest",
    )
    records = _manifest_records(manifest, key="files", path_key="path")
    if manifest.get("file_count") != len(records):
        raise ValueError("structural manifest file count is inconsistent")
    _verify_exact_membership(
        metadata,
        records,
        manifest_member=manifest_member,
        record_path_key="path",
        prefix=f"{_STRUCTURAL_PREFIX}/",
    )

    manifest_record = receipt.get("artifact_manifest")
    if not isinstance(manifest_record, dict):
        raise ValueError("structural receipt lacks manifest metadata")
    if manifest_record.get("name") != _STRUCTURAL_MANIFEST:
        raise ValueError("structural receipt names the wrong manifest")
    if manifest_record.get("bytes") != len(manifest_bytes):
        raise ValueError("structural manifest byte count differs from its receipt")
    if manifest_record.get("sha256") != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError("structural manifest SHA-256 differs from its receipt")
    if manifest_record.get("content_sha256") != manifest_digest:
        raise ValueError("structural receipt has the wrong manifest content hash")
    if receipt.get("archive_member_count") != len(metadata):
        raise ValueError("structural receipt archive member count is inconsistent")

    if manifest.get("analysis_label") != "structural sensitivity":
        raise ValueError("structural manifest has the wrong analysis label")
    if manifest.get("probability_interpretation") is not False:
        raise ValueError("structural manifest is incorrectly probabilistic")
    if manifest.get("execution_scope") != "structural_sensitivity_only":
        raise ValueError("structural manifest has the wrong execution scope")
    if manifest.get("n_parameters") != 29:
        raise ValueError("structural manifest must contain 29 active factors")
    if manifest.get("n_tasks") != 3_000:
        raise ValueError("structural manifest must contain 3,000 tasks")
    if manifest.get("retained_decision_ledger_count") != 6_500:
        raise ValueError("structural manifest must contain 6,500 retained ledgers")
    _require_hex(
        manifest.get("retained_decision_ledger_set_sha256"),
        label="structural retained-ledger set SHA-256",
        width=64,
    )
    _validate_structural_accounting(manifest.get("accounting"))

    source_commit = _require_hex(
        manifest.get("source_commit"),
        label="structural manifest source commit",
        width=40,
    )
    dual = manifest.get("dual_provenance") is True
    if dual:
        publication_commit = _require_hex(
            manifest.get("publication_code_commit"),
            label="structural manifest publication code commit",
            width=40,
        )
        if not (
            receipt_recovery_mode
            and manifest.get("simulation_source_commit") == source_commit
            and publication_commit != source_commit
            and manifest.get("simulation_rerun") is False
            and isinstance(manifest.get("recovery_authorization"), dict)
            and receipt.get("simulation_source_commit") == source_commit
            and receipt.get("publication_code_commit") == publication_commit
            and receipt.get("dual_provenance") is True
            and receipt.get("simulation_rerun") is False
            and receipt.get("recovery_authorization")
            == manifest.get("recovery_authorization")
        ):
            raise ValueError(
                "structural dual provenance lacks one exact recovery authorization"
            )
        provenance_mode = "authorized_publication_recovery"
    else:
        publication_commit = source_commit
        if receipt_recovery_mode or any(
            key in manifest
            for key in (
                "simulation_source_commit",
                "publication_code_commit",
                "simulation_rerun",
                "recovery_authorization",
            )
        ) or any(
            key in receipt
            for key in (
                "simulation_source_commit",
                "publication_code_commit",
                "dual_provenance",
                "simulation_rerun",
                "recovery_authorization",
            )
        ):
            raise ValueError(
                "fresh structural evidence contains recovery provenance fields"
            )
        provenance_mode = "fresh_single_commit"
    run_tag = manifest.get("run_tag")
    match = _STRUCTURAL_RUN_TAG.fullmatch(str(run_tag))
    if match is None or match.group(1) != source_commit[:7]:
        raise ValueError("structural run tag is not bound to its source commit")
    publication_root_relative = (
        _structural_recovery_attempt_root(
            manifest["recovery_authorization"], run_tag=str(run_tag),
        )
        if dual
        else ""
    )
    if receipt.get("source_commit") != source_commit:
        raise ValueError("structural receipt source commit differs from its manifest")
    if receipt.get("run_tag") != run_tag:
        raise ValueError("structural receipt run tag differs from its manifest")
    expected_snapshot_binding = {
        "mode": "detached_readonly_git_worktree_v1",
        "source_tree_sha256": manifest.get(
            "source_snapshot_binding", {}
        ).get("source_tree_sha256"),
    }
    _require_hex(
        expected_snapshot_binding["source_tree_sha256"],
        label="structural source-tree SHA-256",
        width=64,
    )
    if manifest.get("source_snapshot_binding") != expected_snapshot_binding or (
        receipt.get("source_snapshot_binding") != expected_snapshot_binding
    ):
        raise ValueError("structural source snapshot binding is inconsistent")
    publication_snapshot_binding: dict[str, Any] | None = None
    if dual:
        if (
            manifest.get("simulation_source_snapshot_binding")
            != expected_snapshot_binding
            or receipt.get("simulation_source_snapshot_binding")
            != expected_snapshot_binding
        ):
            raise ValueError(
                "structural recovery simulation source snapshot is inconsistent"
            )
        publication_binding = manifest.get("publication_source_snapshot_binding")
        if not isinstance(publication_binding, dict):
            raise ValueError(
                "structural recovery lacks a publication source snapshot binding"
            )
        publication_snapshot_binding = {
            "mode": "detached_readonly_git_worktree_v1",
            "source_tree_sha256": _require_hex(
                publication_binding.get("source_tree_sha256"),
                label="structural publication source-tree SHA-256",
                width=64,
            ),
            "git_tree_sha1": _require_hex(
                publication_binding.get("git_tree_sha1"),
                label="structural publication Git tree",
                width=40,
            ),
        }
        if (
            publication_binding != publication_snapshot_binding
            or receipt.get("publication_source_snapshot_binding")
            != publication_snapshot_binding
        ):
            raise ValueError(
                "structural recovery publication source snapshot is inconsistent"
            )
    expected_validator_identity = {
        "head_commit": publication_commit,
        "source_tree_clean": True,
        "tracked_and_untracked_status_empty": True,
    }
    if manifest.get("validator_source_identity") != expected_validator_identity or (
        receipt.get("validator_source_identity") != expected_validator_identity
    ):
        raise ValueError("structural validator source identity is not clean and commit-exact")

    submission = _json_member(
        archive_path, f"{_STRUCTURAL_PREFIX}/slurm_submission.json"
    )
    publisher_execution = receipt.get("publisher_execution")
    expected_publisher_job = (
        submission.get("publisher", {}).get("job_id")
        if isinstance(submission, dict) else None
    )
    if dual:
        recovery_job = receipt["recovery_authorization"].get(
            "authorized_recovery_publisher_job_id"
        )
        expected_execution = {
            "slurm_job_id": recovery_job,
            "declared_publisher_job_id": expected_publisher_job,
            "identity_match": False,
            "authorization_mode": "publication_recovery_authorization",
            "authorized_recovery_publisher_job_id": recovery_job,
            "original_declared_publisher_failed": True,
            "simulation_rerun": False,
        }
    else:
        expected_execution = {
            "slurm_job_id": expected_publisher_job,
            "declared_publisher_job_id": expected_publisher_job,
            "identity_match": True,
        }
    if publisher_execution != expected_execution:
        raise ValueError("structural publisher runtime identity is inconsistent")
    if submission.get("source_snapshot_mode") != expected_snapshot_binding["mode"] or (
        submission.get("source_tree_sha256")
        != expected_snapshot_binding["source_tree_sha256"]
    ):
        raise ValueError("structural submission/source snapshot bindings differ")

    recovery_evidence: dict[str, Any] | None = None
    if dual:
        recovery_evidence = _validate_archived_recovery(
            archive_path,
            metadata=metadata,
            prefix=f"{_STRUCTURAL_PREFIX}/",
            kind="structural",
            run_tag=str(run_tag),
            simulation_commit=source_commit,
            publication_commit=publication_commit,
            simulation_source_tree_sha256=expected_snapshot_binding[
                "source_tree_sha256"
            ],
            original_submission_member="slurm_submission.json",
            manifested_authorization=manifest["recovery_authorization"],
        )
        recovery_evidence.pop("_raw_manifest_payload")
        if (
            publication_snapshot_binding is None
            or recovery_evidence["publication_repair_tree"]
            != publication_snapshot_binding["git_tree_sha1"]
        ):
            raise ValueError(
                "structural recovery receipt and publication source tree differ"
            )

    run_plan_name = f"{_STRUCTURAL_PREFIX}/run_plan.json"
    if run_plan_name not in metadata:
        raise ValueError("structural archive lacks the clean-source run plan")
    run_plan = _json_member(archive_path, run_plan_name)
    _validate_self_hash(
        run_plan, field="run_plan_sha256", label="structural run plan"
    )
    for key, expected in (
        ("analysis_label", "structural sensitivity"),
        ("probability_interpretation", False),
        ("execution_scope", "structural_sensitivity_only"),
        ("run_tag", run_tag),
        ("source_commit", source_commit),
        ("source_tracked_tree_clean_at_generation", True),
        ("source_tree_clean_at_generation", True),
        ("development_only_dirty_plan", False),
    ):
        if run_plan.get(key) != expected:
            raise ValueError(f"structural run plan disagrees on {key}")

    artifacts = run_plan.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("structural run plan lacks its artifact inventory")
    registry_relative = _safe_name(
        artifacts.get("parameter_registry"), label="structural registry"
    )
    registry_name = f"{_STRUCTURAL_PREFIX}/{registry_relative}"
    if registry_name not in metadata:
        raise ValueError("structural archive lacks its parameter registry")
    registry = _json_member(archive_path, registry_name)
    if registry.get("analysis_label") != "structural sensitivity":
        raise ValueError("structural parameter registry has the wrong label")
    if registry.get("probability_interpretation") is not False:
        raise ValueError("structural parameter registry is incorrectly probabilistic")
    parameters = registry.get("parameters")
    if not isinstance(parameters, list) or len(parameters) != 29:
        raise ValueError("structural parameter registry must have 29 factors")
    keys = [
        item.get("key") if isinstance(item, dict) else None
        for item in parameters
    ]
    if any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("structural parameter registry has an invalid factor key")
    if len(keys) != len(set(keys)) or "slca_carbon_cap" not in keys:
        raise ValueError(
            "structural registry must have 29 unique factors including slca_carbon_cap"
        )

    _validate_structural_inventory(
        archive_path,
        metadata=metadata,
        records=records,
        manifest=manifest,
        run_plan=run_plan,
        publication_root=publication_root_relative,
    )
    with tempfile.TemporaryDirectory(prefix="agribrain_structural_evidence_") as temp:
        extracted = Path(temp)
        _extract_safe_archive(archive_path, extracted, metadata)
        _run_canonical_structural_validation(
            extracted / _STRUCTURAL_PREFIX,
            publication_root=(
                extracted / _STRUCTURAL_PREFIX / publication_root_relative
                if publication_root_relative
                else extracted / _STRUCTURAL_PREFIX
            ),
            source_commit=source_commit,
            publication_commit=publication_commit,
            run_tag=str(run_tag),
        )

    return {
        "source_commit": source_commit,
        "simulation_source_commit": source_commit,
        "publication_code_commit": publication_commit,
        "dual_provenance": dual,
        "simulation_rerun": not dual,
        "provenance_mode": provenance_mode,
        "recovery_authorization": recovery_evidence,
        "run_tag": str(run_tag),
        "archive": {
            "name": archive_path.name,
            "bytes": archive_path.stat().st_size,
            "sha256": archive_sha,
            "member_count": len(metadata),
        },
        "archive_receipt": {
            "name": receipt_path.name,
            "bytes": receipt_path.stat().st_size,
            "sha256": _sha256_file(receipt_path),
            "content_sha256": receipt_digest,
        },
        "artifact_manifest": {
            "name": _STRUCTURAL_MANIFEST,
            "bytes": len(manifest_bytes),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "content_sha256": manifest_digest,
            "file_count": len(records),
        },
        "source_snapshot_binding": expected_snapshot_binding,
        "publication_source_snapshot_binding": publication_snapshot_binding,
        "complete_episode_evidence": dict(
            manifest["complete_episode_evidence"]
        ),
        "structural_publication_artifacts": [
            _publication_member(publication_root_relative, name)
            for name in (
                "structural_sensitivity_summary.csv",
                "structural_sensitivity_summary.png",
                "structural_sensitivity_summary.pdf",
                "structural_sensitivity_publication_receipt.json",
            )
        ],
        "design": {
            "latin_hypercube_points": 100,
            "active_factors": 29,
            "tasks": 3_000,
            "retained_decision_ledgers": 6_500,
            "retained_complete_episode_archives": 24_500,
            "retained_adaptation_episode_ledgers": 18_000,
            "successful_worker_runtime_receipts": 3_000,
            "scheduler_accounted_simulation_tasks": 3_000,
            "probability_interpretation": False,
            **_STRUCTURAL_TOTALS,
        },
    }


def assemble_full_submission_evidence(
    *,
    core_archive: Path,
    core_receipt: Path,
    core_ready: Path,
    core_complete_archive: Path | None = None,
    core_complete_receipt: Path | None = None,
    core_complete_ready: Path | None = None,
    structural_archive: Path,
    structural_receipt: Path,
    output: Path,
) -> dict[str, Any]:
    """Validate both scopes and atomically create the self-hashed receipt."""

    output = _validated_new_output_path(output)

    preflight_core_receipt = _load_json_file(
        core_receipt.absolute(), "core archive receipt"
    )
    preflight_simulation_commit = _require_hex(
        preflight_core_receipt.get("simulation_source_commit"),
        label="core receipt simulation source commit",
        width=40,
    )
    preflight_publication_commit = _require_hex(
        preflight_core_receipt.get("publication_code_commit"),
        label="core receipt publication code commit",
        width=40,
    )
    preflight_dual = (
        preflight_simulation_commit != preflight_publication_commit
    )
    if preflight_dual:
        if not (
            preflight_core_receipt.get("derivation_type")
            == "publication-only deterministic recovery"
            and preflight_core_receipt.get("simulation_rerun") is False
            and isinstance(
                preflight_core_receipt.get("recovery_authorization"), dict
            )
        ):
            raise ValueError(
                "core recovery preflight lacks its no-rerun authorization"
            )
    elif not (
        preflight_core_receipt.get("derivation_type")
        == "fresh stochastic simulation and publication build"
        and preflight_core_receipt.get("simulation_rerun") is True
        and preflight_core_receipt.get("recovery_authorization") is None
    ):
        raise ValueError("core fresh preflight provenance is inconsistent")

    validator_source = _validate_local_validator_checkout(
        preflight_publication_commit
    )
    core = _validate_core_evidence(
        core_archive.absolute(),
        core_receipt.absolute(),
        core_ready.absolute(),
        complete_archive_path=core_complete_archive,
        complete_receipt_path=core_complete_receipt,
        complete_ready_path=core_complete_ready,
    )
    if core["publication_code_commit"] != validator_source["head_commit"]:
        raise ValueError(
            "validated core publication commit differs from the validator checkout"
        )
    preflight_structural_receipt = _load_json_file(
        structural_receipt.absolute(), "structural-sensitivity archive receipt"
    )
    preflight_structural_simulation_commit = _require_hex(
        preflight_structural_receipt.get("source_commit"),
        label="structural receipt source commit",
        width=40,
    )
    if preflight_structural_simulation_commit != preflight_simulation_commit:
        raise ValueError(
            "core and structural evidence were generated from different source "
            "commits (simulation identity)"
        )
    if preflight_dual:
        if not (
            preflight_structural_receipt.get("dual_provenance") is True
            and preflight_structural_receipt.get("simulation_rerun") is False
            and preflight_structural_receipt.get("simulation_source_commit")
            == preflight_simulation_commit
            and preflight_structural_receipt.get("publication_code_commit")
            == preflight_publication_commit
            and isinstance(
                preflight_structural_receipt.get("recovery_authorization"), dict
            )
        ):
            raise ValueError(
                "structural evidence lacks matching dual-recovery preflight provenance"
            )
    elif any(
        key in preflight_structural_receipt
        for key in (
            "simulation_source_commit",
            "publication_code_commit",
            "dual_provenance",
            "simulation_rerun",
            "recovery_authorization",
        )
    ):
        raise ValueError(
            "fresh structural evidence cannot carry recovery provenance"
        )
    structural = _validate_structural_evidence(
        structural_archive.absolute(), structural_receipt.absolute()
    )
    if core["provenance_mode"] != structural["provenance_mode"]:
        raise ValueError(
            "core and structural evidence use different provenance modes"
        )
    if (
        core["simulation_source_commit"]
        != structural["simulation_source_commit"]
    ):
        raise ValueError(
            "core and structural evidence were generated from different source "
            "commits (simulation identity)"
        )
    if core["publication_code_commit"] != structural["publication_code_commit"]:
        raise ValueError(
            "core and structural evidence use different publication repair commits"
        )
    if core["publication_code_commit"] != validator_source["head_commit"]:
        raise ValueError(
            "validated publication commit differs from the assembler checkout"
        )
    core_source_tree = core["slurm_submission_receipt"]["source_tree_sha256"]
    structural_source_tree = structural["source_snapshot_binding"][
        "source_tree_sha256"
    ]
    if core_source_tree != structural_source_tree:
        raise ValueError(
            "core and structural evidence bind different literal source trees"
        )
    recomputed_source_tree, tracked_source_count = tracked_source_digest(
        REPO_ROOT
    )
    if tracked_source_count <= 0:
        raise ValueError("assembler checkout has no tracked source files")

    dual = core["dual_provenance"] is True
    core_recovery = core["recovery_authorization"]
    structural_recovery = structural["recovery_authorization"]
    if dual:
        if not (
            isinstance(core_recovery, dict)
            and isinstance(structural_recovery, dict)
            and core_recovery.get("kind") == "core"
            and structural_recovery.get("kind") == "structural"
            and core_recovery.get("simulation_rerun") is False
            and structural_recovery.get("simulation_rerun") is False
            and core_recovery.get("validated") is True
            and structural_recovery.get("validated") is True
        ):
            raise ValueError(
                "dual recovery requires separately validated core and structural receipts"
            )
        if (
            core_recovery["receipt_literal_sha256"]
            == structural_recovery["receipt_literal_sha256"]
            or core_recovery["receipt_self_hash"]
            == structural_recovery["receipt_self_hash"]
        ):
            raise ValueError(
                "core and structural recovery receipts must be distinct"
            )
        if (
            core_recovery["authorized_recovery_publisher_job_id"]
            == structural_recovery["authorized_recovery_publisher_job_id"]
        ):
            raise ValueError(
                "core and structural recovery publishers must have distinct Slurm jobs"
            )
        publication_tree = core_recovery["publication_repair_tree"]
        if publication_tree != structural_recovery["publication_repair_tree"]:
            raise ValueError(
                "core and structural evidence use different publication repair trees"
            )
        structural_publication_snapshot = structural.get(
            "publication_source_snapshot_binding"
        )
        if not isinstance(structural_publication_snapshot, dict) or (
            structural_publication_snapshot.get("git_tree_sha1")
            != publication_tree
        ):
            raise ValueError(
                "structural publication snapshot differs from the shared repair tree"
            )
        if (
            structural_publication_snapshot.get("source_tree_sha256")
            != recomputed_source_tree
        ):
            raise ValueError(
                "publication source-tree SHA-256 differs from the clean assembler checkout"
            )
        if _current_git_tree_sha1() != publication_tree:
            raise ValueError(
                "publication repair Git tree differs from the clean assembler checkout"
            )
        shared_recovery_authorizations: dict[str, Any] | None = {
            "core": core_recovery,
            "structural": structural_recovery,
        }
    else:
        if core_recovery is not None or structural_recovery is not None:
            raise ValueError("fresh evidence cannot carry recovery authorizations")
        publication_tree = None
        shared_recovery_authorizations = None
        if recomputed_source_tree != core_source_tree:
            raise ValueError(
                "evidence source-tree SHA-256 differs from the clean validator checkout"
            )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": "full_submission_evidence_set",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "assembly_action": "verification_and_hash_binding_only",
        "simulations_executed_by_assembler": False,
        "source_commit": core["source_commit"],
        "simulation_source_commit": core["simulation_source_commit"],
        "publication_code_commit": core["publication_code_commit"],
        "dual_provenance": dual,
        "simulation_rerun": not dual,
        "recovery_authorizations": shared_recovery_authorizations,
        "source_identity": {
            "same_clean_source_commit": not dual,
            "same_simulation_source_commit": True,
            "same_publication_code_commit": True,
            "core_git_dirty": False,
            "structural_source_tree_clean_at_generation": True,
            "validator_checkout": validator_source,
            "source_snapshot_mode": "detached_readonly_git_worktree_v1",
            "shared_simulation_source_tree_sha256": core_source_tree,
            "publication_repair_tree": publication_tree,
            "assembler_source_tree_sha256": recomputed_source_tree,
            "tracked_source_file_count": tracked_source_count,
        },
        "evidence_scope": {
            "core_publication_evidence_present": True,
            "structural_sensitivity_evidence_present": True,
            "full_submission_evidence_present": True,
            "missing_required_scopes": [],
        },
        "core_publication_evidence": core,
        "structural_sensitivity_evidence": structural,
        "validation": {
            "literal_archive_hashes_and_sizes": "PASS",
            "atomic_core_bundle_ready_marker": "PASS",
            "safe_regular_archive_members_only": "PASS",
            "exact_manifest_membership_and_payload_hashes": "PASS",
            "fresh_core_semantic_receipt_and_locked_accounting": (
                "NOT_APPLICABLE_RECOVERY" if dual else "PASS"
            ),
            "authorized_dual_publication_recovery": (
                "PASS" if dual else "NOT_APPLICABLE"
            ),
            "distinct_core_and_structural_recovery_receipts": (
                "PASS" if dual else "NOT_APPLICABLE"
            ),
            "complete_core_raw_bundle_exact_manifest_mapping": (
                "PASS" if dual else "NOT_APPLICABLE"
            ),
            "structural_raw_archive_exact_manifest_mapping": (
                "PASS" if dual else "NOT_APPLICABLE"
            ),
            "simulation_rerun_prohibited_in_recovery": (
                "PASS" if dual else "NOT_APPLICABLE"
            ),
            "complete_nonprobabilistic_structural_panel": "PASS",
            "structural_retained_ledgers_and_endpoint_recomputation": "PASS",
            "structural_complete_episode_and_runtime_evidence": "PASS",
            "structural_failed_attempt_artifacts_preserved": "PASS",
            "structural_post_job_scheduler_accounting": "PASS",
            "structural_table_and_figure_artifacts": "PASS",
            "same_clean_source_commit": (
                "NOT_APPLICABLE_DUAL_PROVENANCE" if dual else "PASS"
            ),
            "same_simulation_source_commit": "PASS",
            "same_publication_code_commit": "PASS",
            "same_literal_source_tree_sha256": "PASS",
            "validator_checkout_literal_source_tree_match": "PASS",
            "assembler_validator_checkout_same_clean_commit": "PASS",
        },
    }
    payload["receipt_sha256"] = _canonical_sha256(payload)
    serialized = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    final_validator_source = _validate_local_validator_checkout(
        core["publication_code_commit"]
    )
    if final_validator_source != validator_source:
        raise ValueError(
            "validator checkout changed while assembling full evidence"
        )
    if dual and _current_git_tree_sha1() != publication_tree:
        raise ValueError(
            "publication repair Git tree changed while assembling full evidence"
        )
    _write_new_file_atomically(output, serialized)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--core-archive", type=Path, required=True)
    parser.add_argument("--core-receipt", type=Path, required=True)
    parser.add_argument("--core-ready", type=Path, required=True)
    parser.add_argument(
        "--core-complete-archive",
        type=Path,
        help="Lossless complete_run_evidence archive; required for recovery",
    )
    parser.add_argument(
        "--core-complete-receipt",
        type=Path,
        help="Lossless complete_run_evidence RECEIPT.json; required for recovery",
    )
    parser.add_argument(
        "--core-complete-ready",
        type=Path,
        help="Lossless complete_run_evidence READY.json; required for recovery",
    )
    parser.add_argument("--structural-archive", type=Path, required=True)
    parser.add_argument("--structural-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = assemble_full_submission_evidence(
        core_archive=args.core_archive,
        core_receipt=args.core_receipt,
        core_ready=args.core_ready,
        core_complete_archive=args.core_complete_archive,
        core_complete_receipt=args.core_complete_receipt,
        core_complete_ready=args.core_complete_ready,
        structural_archive=args.structural_archive,
        structural_receipt=args.structural_receipt,
        output=args.output,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
