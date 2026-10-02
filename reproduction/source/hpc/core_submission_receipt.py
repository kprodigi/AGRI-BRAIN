#!/usr/bin/env python3
"""Create or validate the immutable Slurm DAG receipt for a core run."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = 2
RECEIPT_SCOPE = "submission_only_not_scheduler_completion"
SNAPSHOT_MODE = "detached_readonly_git_worktree_v1"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_RUN_TAG = re.compile(r"^([0-9a-f]{7})_[0-9]{8}_[0-9]{6}$")
_JOB_ID = re.compile(r"^[1-9][0-9]*$")
_PARTITION = re.compile(r"^[A-Za-z0-9._,+-]+$")

SEEDS = [
    42,
    1337,
    2024,
    7,
    99,
    101,
    202,
    303,
    404,
    505,
    606,
    707,
    808,
    909,
    1010,
    1111,
    1212,
    1313,
    1414,
    1515,
]
SCENARIOS = [
    "heatwave",
    "overproduction",
    "cyber_outage",
    "adaptive_pricing",
    "baseline",
]
CORE_ACCOUNTING = {
    "unique_retained_cells": 1_600,
    "executed_episodes": 6_100,
    "simulated_steps": 1_756_800,
}


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _require_job_id(value: object, label: str) -> str:
    if not isinstance(value, str) or not _JOB_ID.fullmatch(value):
        raise ValueError(f"{label} must be a numeric Slurm job id")
    return value


def _validate_timestamp(value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("submission receipt lacks submitted_at_utc")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("submitted_at_utc is not a valid ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("submitted_at_utc must be timezone aware")


def _git_clean_commit(repo_root: Path) -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot verify the source checkout for DAG receipt") from exc
    if not _HEX40.fullmatch(commit):
        raise RuntimeError("git returned an invalid source commit")
    if status:
        raise RuntimeError("source checkout is dirty; refusing to create DAG receipt")
    return commit


def build_receipt(
    *,
    run_tag: str,
    source_commit: str,
    partition: str,
    seed_job_id: str,
    stress_job_id: str,
    publisher_job_id: str,
    source_snapshot_mode: str,
    source_tree_sha256: str,
) -> dict[str, Any]:
    if not _HEX40.fullmatch(source_commit):
        raise ValueError("source_commit must be a full lowercase Git SHA-1")
    match = _RUN_TAG.fullmatch(run_tag)
    if match is None or match.group(1) != source_commit[:7]:
        raise ValueError("run_tag is not bound to source_commit")
    if not isinstance(partition, str) or not _PARTITION.fullmatch(partition):
        raise ValueError("partition has an invalid or ambiguous value")
    if source_snapshot_mode != SNAPSHOT_MODE:
        raise ValueError("source_snapshot_mode is not the locked detached snapshot mode")
    if not isinstance(source_tree_sha256, str) or not _HEX64.fullmatch(
        source_tree_sha256
    ):
        raise ValueError("source_tree_sha256 must be a lowercase SHA-256")
    seed_job_id = _require_job_id(seed_job_id, "seed job id")
    stress_job_id = _require_job_id(stress_job_id, "stress job id")
    publisher_job_id = _require_job_id(publisher_job_id, "publisher job id")
    if len({seed_job_id, stress_job_id, publisher_job_id}) != 3:
        raise ValueError("the three Slurm stages must have distinct job ids")

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis_label": "core stochastic publication evidence",
        "execution_scope": "core_publication_evidence",
        "receipt_scope": RECEIPT_SCOPE,
        "scheduler_completion_attested": False,
        "run_tag": run_tag,
        "source_commit": source_commit,
        "source_tree_clean_at_submission": True,
        "source_snapshot_mode": source_snapshot_mode,
        "source_tree_sha256": source_tree_sha256,
        "partition": partition,
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "deterministic_mode": False,
        "slurm_dag": {
            "seed_array": {
                "job_id": seed_job_id,
                "script": "hpc/hpc_seed.sh",
                "array_indices": "0-19",
                "task_count": 20,
                "seeds": list(SEEDS),
                "dependency_type": None,
                "afterok_job_ids": [],
            },
            "stress_array": {
                "job_id": stress_job_id,
                "script": "hpc/hpc_stress.sh",
                "array_indices": "0-4",
                "task_count": 5,
                "scenarios": list(SCENARIOS),
                "dependency_type": "afterok",
                "afterok_job_ids": [seed_job_id],
            },
            "publisher": {
                "job_id": publisher_job_id,
                "script": "hpc/hpc_publish.sh",
                "array_indices": None,
                "task_count": 1,
                "dependency_type": "afterok",
                "afterok_job_ids": [seed_job_id, stress_job_id],
            },
        },
        "locked_core_accounting": dict(CORE_ACCOUNTING),
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def validate_receipt_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("core submission receipt schema must be 2")
    if payload.get("analysis_label") != "core stochastic publication evidence":
        raise ValueError("core submission receipt has the wrong analysis label")
    if payload.get("execution_scope") != "core_publication_evidence":
        raise ValueError("core submission receipt has the wrong execution scope")
    if payload.get("receipt_scope") != RECEIPT_SCOPE or payload.get(
        "scheduler_completion_attested"
    ) is not False:
        raise ValueError(
            "core submission receipt must remain submission-only and not attest completion"
        )
    source_commit = payload.get("source_commit")
    if not isinstance(source_commit, str) or not _HEX40.fullmatch(source_commit):
        raise ValueError("core submission receipt has an invalid source commit")
    run_tag = payload.get("run_tag")
    match = _RUN_TAG.fullmatch(str(run_tag))
    if match is None or match.group(1) != source_commit[:7]:
        raise ValueError("core submission receipt run tag is not commit-bound")
    if payload.get("source_tree_clean_at_submission") is not True:
        raise ValueError("core submission receipt does not attest a clean checkout")
    if payload.get("source_snapshot_mode") != SNAPSHOT_MODE:
        raise ValueError("core submission receipt has the wrong source snapshot mode")
    if not isinstance(payload.get("source_tree_sha256"), str) or not _HEX64.fullmatch(
        payload["source_tree_sha256"]
    ):
        raise ValueError("core submission receipt has an invalid source-tree digest")
    partition = payload.get("partition")
    if not isinstance(partition, str) or not _PARTITION.fullmatch(partition):
        raise ValueError("core submission receipt has an invalid partition")
    if payload.get("deterministic_mode") is not False:
        raise ValueError("core submission receipt is not stochastic")
    _validate_timestamp(payload.get("submitted_at_utc"))
    if payload.get("locked_core_accounting") != CORE_ACCOUNTING:
        raise ValueError("core submission receipt has incorrect accounting")

    dag = payload.get("slurm_dag")
    if not isinstance(dag, dict) or set(dag) != {
        "seed_array",
        "stress_array",
        "publisher",
    }:
        raise ValueError("core submission receipt lacks the exact three-stage DAG")
    seed = dag["seed_array"]
    stress = dag["stress_array"]
    publisher = dag["publisher"]
    if not all(isinstance(stage, dict) for stage in (seed, stress, publisher)):
        raise ValueError("core submission DAG stages must be objects")
    seed_id = _require_job_id(seed.get("job_id"), "seed job id")
    stress_id = _require_job_id(stress.get("job_id"), "stress job id")
    publisher_id = _require_job_id(
        publisher.get("job_id"), "publisher job id"
    )
    if len({seed_id, stress_id, publisher_id}) != 3:
        raise ValueError("core submission DAG job ids must be distinct")
    expected_seed = {
        "job_id": seed_id,
        "script": "hpc/hpc_seed.sh",
        "array_indices": "0-19",
        "task_count": 20,
        "seeds": SEEDS,
        "dependency_type": None,
        "afterok_job_ids": [],
    }
    expected_stress = {
        "job_id": stress_id,
        "script": "hpc/hpc_stress.sh",
        "array_indices": "0-4",
        "task_count": 5,
        "scenarios": SCENARIOS,
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
    if seed != expected_seed:
        raise ValueError("seed-array DAG stage is inconsistent")
    if stress != expected_stress:
        raise ValueError("stress-array DAG stage or dependency is inconsistent")
    if publisher != expected_publisher:
        raise ValueError("publisher DAG stage or dependency is inconsistent")

    unsigned = dict(payload)
    digest = unsigned.pop("receipt_sha256", None)
    if not isinstance(digest, str) or digest != canonical_sha256(unsigned):
        raise ValueError("core submission receipt self-hash is invalid")
    return dict(payload)


def validate_receipt_file(
    path: Path,
    *,
    expected_run_tag: str | None = None,
    expected_source_commit: str | None = None,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"core submission receipt is not a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("core submission receipt is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("core submission receipt must contain one JSON object")
    validated = validate_receipt_payload(payload)
    if expected_run_tag is not None and validated["run_tag"] != expected_run_tag:
        raise ValueError("core submission receipt has the wrong run tag")
    if (
        expected_source_commit is not None
        and validated["source_commit"] != expected_source_commit
    ):
        raise ValueError("core submission receipt has the wrong source commit")
    return validated


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite core submission receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    try:
        with path.open("xb") as stream:
            stream.write(serialized)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite core submission receipt: {path}"
        ) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--repo-root", type=Path, default=Path.cwd())
    create.add_argument("--run-tag", required=True)
    create.add_argument("--source-commit", required=True)
    create.add_argument("--partition", required=True)
    create.add_argument("--seed-job-id", required=True)
    create.add_argument("--stress-job-id", required=True)
    create.add_argument("--publisher-job-id", required=True)
    create.add_argument("--source-snapshot-mode", required=True)
    create.add_argument("--source-tree-sha256", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--receipt", type=Path, required=True)
    validate.add_argument("--run-tag")
    validate.add_argument("--source-commit")
    validate.add_argument(
        "--publisher-slurm-job-id",
        help=(
            "When running inside the publisher, require this actual "
            "SLURM_JOB_ID to equal the submitted publisher job id"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "create":
        clean_commit = _git_clean_commit(args.repo_root.resolve())
        if clean_commit != args.source_commit:
            raise RuntimeError(
                "current clean checkout commit differs from --source-commit"
            )
        payload = build_receipt(
            run_tag=args.run_tag,
            source_commit=args.source_commit,
            partition=args.partition,
            seed_job_id=args.seed_job_id,
            stress_job_id=args.stress_job_id,
            publisher_job_id=args.publisher_job_id,
            source_snapshot_mode=args.source_snapshot_mode,
            source_tree_sha256=args.source_tree_sha256,
        )
        _write_receipt(args.output.resolve(), payload)
    else:
        payload = validate_receipt_file(
            args.receipt.resolve(),
            expected_run_tag=args.run_tag,
            expected_source_commit=args.source_commit,
        )
        if args.publisher_slurm_job_id is not None:
            actual = _require_job_id(
                args.publisher_slurm_job_id, "publisher SLURM_JOB_ID"
            )
            declared = payload["slurm_dag"]["publisher"]["job_id"]
            if actual != declared:
                raise ValueError(
                    "publisher SLURM_JOB_ID differs from the declared publisher"
                )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
