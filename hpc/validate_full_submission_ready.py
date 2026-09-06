#!/usr/bin/env python3
"""Create or validate the exact four-file combined-submission READY bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # package import in tests; direct import when executed as a script
    from .finalizer_submission_authorization import validate_authorization
    from .validate_lexical_path import validate_lexical_path
except ImportError:  # pragma: no cover - exercised by the HPC shell entry point
    from finalizer_submission_authorization import validate_authorization
    from validate_lexical_path import validate_lexical_path

RECEIPT_NAME = "FULL_SUBMISSION_EVIDENCE_RECEIPT.json"
READY_NAME = "READY.json"
AUTHORIZATION_NAME = "FINALIZER_SUBMISSION_AUTHORIZATION.json"
ENVIRONMENT_NAME = "FINALIZER_PUBLICATION_ENVIRONMENT.json"
INVENTORY = frozenset({RECEIPT_NAME, READY_NAME, AUTHORIZATION_NAME, ENVIRONMENT_NAME})
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    validate_lexical_path(path, kind="file")
    try:
        literal = path.read_bytes()
        payload = json.loads(literal.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be one JSON object")
    return payload, literal


def _self_hash(payload: Mapping[str, Any], field: str, label: str) -> str:
    claimed = payload.get(field)
    if not isinstance(claimed, str) or _SHA256_RE.fullmatch(claimed) is None:
        raise ValueError(f"{label} has an invalid {field}")
    unsigned = dict(payload)
    unsigned.pop(field)
    if canonical_sha256(unsigned) != claimed:
        raise ValueError(f"{label} self-hash mismatch")
    return claimed


def _validate_receipt(
    payload: Mapping[str, Any],
    *,
    simulation_commit: str,
    publication_commit: str,
    core_job_id: str,
    structural_job_id: str,
) -> str:
    claimed = _self_hash(payload, "receipt_sha256", "combined evidence receipt")
    if not (
        payload.get("receipt_type") == "full_submission_evidence_set"
        and payload.get("dual_provenance") is True
        and payload.get("simulation_rerun") is False
        and payload.get("simulation_source_commit") == simulation_commit
        and payload.get("publication_code_commit") == publication_commit
    ):
        raise ValueError("combined evidence receipt has the wrong recovery identity")
    authorizations = payload.get("recovery_authorizations")
    if not isinstance(authorizations, Mapping) or (
        not isinstance(authorizations.get("core"), Mapping)
        or authorizations["core"].get("authorized_recovery_publisher_job_id")
        != core_job_id
        or not isinstance(authorizations.get("structural"), Mapping)
        or authorizations["structural"].get("authorized_recovery_publisher_job_id")
        != structural_job_id
    ):
        raise ValueError("combined evidence receipt authorizes different publisher jobs")
    return claimed


def _expected_ready(
    *,
    receipt: Mapping[str, Any],
    receipt_bytes: bytes,
    authorization: Mapping[str, Any],
    authorization_bytes: bytes,
    environment: Mapping[str, Any],
    environment_bytes: bytes,
    simulation_commit: str,
    publication_commit: str,
    finalizer_job_id: str,
    core_job_id: str,
    structural_job_id: str,
    run_tag: str,
    created_at_utc: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "READY",
        "scope": "validated_core_complete_run_and_structural_submission_evidence",
        "created_at_utc": created_at_utc,
        "simulation_rerun": False,
        "simulation_source_commit": simulation_commit,
        "publication_code_commit": publication_commit,
        "finalizer_slurm_job_id": finalizer_job_id,
        "afterok_recovery_publisher_job_ids": [core_job_id, structural_job_id],
        "finalizer_scheduler_authorization": {
            "file": AUTHORIZATION_NAME,
            "bytes": len(authorization_bytes),
            "literal_sha256": hashlib.sha256(authorization_bytes).hexdigest(),
            "authorization_sha256": authorization["authorization_sha256"],
        },
        "finalizer_publication_environment": {
            "file": ENVIRONMENT_NAME,
            "bytes": len(environment_bytes),
            "literal_sha256": hashlib.sha256(environment_bytes).hexdigest(),
            "schema_version": 2,
            "run_tag": run_tag,
            "git_commit": publication_commit,
        },
        "receipt": {
            "file": RECEIPT_NAME,
            "bytes": len(receipt_bytes),
            "literal_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "receipt_sha256": receipt["receipt_sha256"],
        },
    }


def _validate_components(
    directory: Path,
    *,
    simulation_commit: str,
    publication_commit: str,
    finalizer_job_id: str,
    core_job_id: str,
    structural_job_id: str,
    run_tag: str,
) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes, dict[str, Any], bytes]:
    receipt, receipt_bytes = _load_json(directory / RECEIPT_NAME, "combined receipt")
    authorization, authorization_bytes = _load_json(
        directory / AUTHORIZATION_NAME, "finalizer authorization"
    )
    environment, environment_bytes = _load_json(
        directory / ENVIRONMENT_NAME, "finalizer publication environment"
    )
    _validate_receipt(
        receipt,
        simulation_commit=simulation_commit,
        publication_commit=publication_commit,
        core_job_id=core_job_id,
        structural_job_id=structural_job_id,
    )
    validate_authorization(
        authorization,
        finalizer_job_id=finalizer_job_id,
        core_publisher_job_id=core_job_id,
        structural_publisher_job_id=structural_job_id,
    )
    if not (
        environment.get("schema_version") == 2
        and environment.get("run_tag") == run_tag
        and environment.get("git_commit") == publication_commit
    ):
        raise ValueError("finalizer publication environment identity mismatch")
    return (
        receipt,
        receipt_bytes,
        authorization,
        authorization_bytes,
        environment,
        environment_bytes,
    )


def _validate_identity(identity: Mapping[str, str]) -> None:
    simulation = identity.get("simulation_commit", "")
    publication = identity.get("publication_commit", "")
    if _COMMIT_RE.fullmatch(simulation) is None or _COMMIT_RE.fullmatch(publication) is None:
        raise ValueError("simulation/publication commits must be full lowercase SHA-1 values")
    if simulation == publication:
        raise ValueError("recovered READY evidence requires distinct source commits")
    if not identity.get("run_tag", "").strip():
        raise ValueError("finalizer run tag cannot be empty")


def create_ready(
    directory: Path,
    **identity: str,
) -> dict[str, Any]:
    _validate_identity(identity)
    validate_lexical_path(directory, kind="directory")
    if {item.name for item in directory.iterdir()} != INVENTORY - {READY_NAME}:
        raise ValueError("combined staging directory does not have the exact pre-READY inventory")
    receipt, receipt_bytes, authorization, authorization_bytes, environment, environment_bytes = (
        _validate_components(directory, **identity)
    )
    ready = _expected_ready(
        receipt=receipt,
        receipt_bytes=receipt_bytes,
        authorization=authorization,
        authorization_bytes=authorization_bytes,
        environment=environment,
        environment_bytes=environment_bytes,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        **identity,
    )
    ready["ready_sha256"] = canonical_sha256(ready)
    encoded = (
        json.dumps(ready, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    target = directory / READY_NAME
    with target.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return ready


def validate_ready(directory: Path, **identity: str) -> dict[str, Any]:
    _validate_identity(identity)
    validate_lexical_path(directory, kind="directory")
    if {item.name for item in directory.iterdir()} != INVENTORY:
        raise ValueError("combined READY directory does not have the exact four-file inventory")
    receipt, receipt_bytes, authorization, authorization_bytes, environment, environment_bytes = (
        _validate_components(directory, **identity)
    )
    ready, _ready_bytes = _load_json(directory / READY_NAME, "READY marker")
    _self_hash(ready, "ready_sha256", "READY marker")
    created_at = ready.get("created_at_utc")
    if not isinstance(created_at, str) or not created_at:
        raise ValueError("READY marker lacks created_at_utc")
    expected = _expected_ready(
        receipt=receipt,
        receipt_bytes=receipt_bytes,
        authorization=authorization,
        authorization_bytes=authorization_bytes,
        environment=environment,
        environment_bytes=environment_bytes,
        created_at_utc=created_at,
        **identity,
    )
    expected["ready_sha256"] = canonical_sha256(expected)
    if ready != expected:
        raise ValueError("READY marker does not exactly bind the combined evidence files")
    return ready


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("create", "validate"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--simulation-commit", required=True)
    parser.add_argument("--publication-commit", required=True)
    parser.add_argument("--finalizer-job-id", required=True)
    parser.add_argument("--core-job-id", required=True)
    parser.add_argument("--structural-job-id", required=True)
    parser.add_argument("--run-tag", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    identity = {
        "simulation_commit": args.simulation_commit,
        "publication_commit": args.publication_commit,
        "finalizer_job_id": args.finalizer_job_id,
        "core_job_id": args.core_job_id,
        "structural_job_id": args.structural_job_id,
        "run_tag": args.run_tag,
    }
    try:
        if args.command == "create":
            create_ready(args.directory, **identity)
            print(f"Created combined READY marker: {args.directory / READY_NAME}")
        else:
            validate_ready(args.directory, **identity)
            print("Combined full-submission READY bundle OK")
    except (OSError, ValueError) as exc:
        print(f"BLOCK: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
