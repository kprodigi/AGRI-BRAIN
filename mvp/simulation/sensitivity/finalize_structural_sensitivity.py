#!/usr/bin/env python3
"""Manifest-bind and archive one complete structural-sensitivity run."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import tarfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from hpc.capture_slurm_accounting import validate_accounting_payload
from hpc.slurm_execution_provenance import (
    SNAPSHOT_MODE,
    require_declared_publisher,
)
from hpc.validate_source_snapshot import validation_errors as source_snapshot_errors
from mvp.simulation.validation.validator_source_identity import (
    validate_clean_validator_checkout,
)

from .analyze_structural_sensitivity import analyze_run
from .design import canonical_sha256, file_sha256
from .publish_structural_sensitivity import (
    CSV_NAME as STRUCTURAL_CSV_NAME,
)
from .publish_structural_sensitivity import (
    EXPECTED_STRUCTURAL_SUMMARY_ROWS,
    FIGURE_STYLE_CONTRACT,
    _csv_bytes,
    _figure_style_record,
    _inspect_pdf,
    _inspect_png,
    summary_rows,
)
from .publish_structural_sensitivity import (
    PDF_NAME as STRUCTURAL_PDF_NAME,
)
from .publish_structural_sensitivity import (
    PNG_NAME as STRUCTURAL_PNG_NAME,
)
from .publish_structural_sensitivity import (
    RECEIPT_NAME as STRUCTURAL_PUBLICATION_RECEIPT_NAME,
)
from .run_structural_sensitivity import (
    REPO_ROOT,
    _atomic_json,
    _load_plan_bundle,
    validate_completed_results_with_ledgers,
)

MANIFEST_SCHEMA_VERSION = 1
RECEIPT_SCHEMA_VERSION = 1
SUBMISSION_SCHEMA_VERSION = 2
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
ARCHIVE_PREFIX = "structural_sensitivity_evidence"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _safe_file(run_root: Path, relative: str) -> Path:
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute() or any(
        part in {"", ".", ".."} for part in candidate_relative.parts
    ):
        raise ValueError(f"unsafe evidence path: {relative!r}")
    candidate = run_root / candidate_relative
    cursor = run_root
    for part in candidate_relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"evidence path traverses a symbolic link: {relative}")
    if not candidate.is_file():
        raise ValueError(f"evidence file is missing: {candidate}")
    if not candidate.resolve().is_relative_to(run_root.resolve()):
        raise ValueError(f"evidence path escapes the run directory: {relative}")
    return candidate


def _validate_environment(
    payload: Mapping[str, Any], *, run_tag: str, source_commit: str,
) -> None:
    if payload.get("schema_version") != 2:
        raise ValueError("environment receipt schema_version must be 2")
    if payload.get("run_tag") != run_tag:
        raise ValueError("environment receipt run tag does not match the run plan")
    if payload.get("git_commit") != source_commit:
        raise ValueError("environment receipt source commit does not match the run plan")
    virtual = payload.get("virtual_environment", {})
    if virtual.get("run_scoped") is not True:
        raise ValueError("environment receipt does not record a run-scoped venv")
    if virtual.get("path_id") != f".publication_venvs/{run_tag}":
        raise ValueError("environment receipt venv identity does not match RUN_TAG")
    validation = payload.get("distribution_validation", {})
    for key in ("unique_normalized_names", "lock_versions_match", "core_version_match"):
        if validation.get(key) is not True:
            raise ValueError(f"environment receipt failed distribution gate {key!r}")
    if validation.get("unexpected_distributions") != []:
        raise ValueError("environment receipt includes packages outside the locked set")


def _validate_status(payload: Mapping[str, Any]) -> None:
    expected = {
        "status": "complete",
        "n_expected_tasks": 3_000,
        "n_valid_tasks": 3_000,
        "n_missing_tasks": 0,
        "missing_task_ids": [],
    }
    if dict(payload) != expected:
        raise ValueError(f"completion status is not the exact 3,000-task panel: {payload}")


def _validate_submission(
    payload: Mapping[str, Any], *, run_tag: str, source_commit: str,
    publisher_slurm_job_id: str | None = None,
) -> None:
    if payload.get("schema_version") != SUBMISSION_SCHEMA_VERSION:
        raise ValueError("Slurm submission receipt schema_version must be 2")
    if payload.get("analysis_label") != "structural sensitivity":
        raise ValueError("Slurm submission receipt has the wrong analysis label")
    if payload.get("receipt_scope") != (
        "submission_only_not_scheduler_completion"
    ) or payload.get("scheduler_completion_attested") is not False:
        raise ValueError(
            "Slurm receipt must remain submission-only and not attest completion"
        )
    if payload.get("run_tag") != run_tag or payload.get("source_commit") != source_commit:
        raise ValueError("Slurm submission receipt identity does not match the run plan")
    if payload.get("task_count") != 3_000:
        raise ValueError("Slurm submission receipt must schedule exactly 3,000 tasks")
    if payload.get("source_snapshot_mode") != SNAPSHOT_MODE:
        raise ValueError("Slurm receipt has the wrong source snapshot mode")
    digest = payload.get("source_tree_sha256")
    if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
        raise ValueError("Slurm receipt has an invalid source-tree SHA-256")
    arrays = payload.get("task_arrays")
    if not isinstance(arrays, list) or not arrays:
        raise ValueError("Slurm submission receipt lacks task arrays")
    expected_offset = 0
    previous_job_id: str | None = None
    for record in arrays:
        if not isinstance(record, dict):
            raise ValueError("Slurm task-array record must be an object")
        if int(record.get("offset", -1)) != expected_offset:
            raise ValueError("Slurm task-array chunks are not contiguous")
        count = int(record.get("count", 0))
        if count <= 0 or count > 1_000:
            raise ValueError("Slurm task-array chunk size is outside 1..1,000")
        if not str(record.get("job_id", "")).isdigit():
            raise ValueError("Slurm task-array job id is not numeric")
        if record.get("afterok_job_id") != previous_job_id:
            raise ValueError("Slurm task-array chunks are not one fail-closed afterok chain")
        previous_job_id = str(record["job_id"])
        expected_offset += count
    if expected_offset != 3_000:
        raise ValueError("Slurm task-array chunks do not cover exactly 0..2,999")
    publisher = payload.get("publisher")
    if not isinstance(publisher, dict) or not str(publisher.get("job_id", "")).isdigit():
        raise ValueError("Slurm submission receipt lacks a numeric publisher job id")
    if publisher.get("afterok_job_id") != arrays[-1].get("job_id"):
        raise ValueError("publisher is not bound afterok to the final task-array chunk")
    if publisher_slurm_job_id is not None:
        require_declared_publisher(
            payload,
            actual_slurm_job_id=publisher_slurm_job_id,
            structural=True,
        )
    unsigned = dict(payload)
    digest = unsigned.pop("receipt_sha256", None)
    if digest != canonical_sha256(unsigned):
        raise ValueError("Slurm submission receipt self-hash is invalid")


def _validate_analysis(
    payload: Mapping[str, Any], *, source_commit: str,
) -> None:
    unsigned = dict(payload)
    digest = unsigned.pop("analysis_sha256", None)
    if digest != canonical_sha256(unsigned):
        raise ValueError("structural analysis SHA-256 does not match its content")
    expected = {
        "analysis_label": "structural sensitivity",
        "probability_interpretation": False,
        "source_commit": source_commit,
        "n_design_points": 100,
        "n_parameters": 29,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"structural analysis {key!r} is {payload.get(key)!r}, expected {value!r}")


def _validate_structural_publication_artifacts(
    run_root: Path,
    *,
    analysis_path: Path,
    source_commit: str,
) -> list[str]:
    """Verify the deterministic structural CSV/PNG/PDF and their receipt."""

    names = (
        STRUCTURAL_CSV_NAME,
        STRUCTURAL_PNG_NAME,
        STRUCTURAL_PDF_NAME,
        STRUCTURAL_PUBLICATION_RECEIPT_NAME,
    )
    paths = {name: _safe_file(run_root, name) for name in names}
    receipt = _load_json(paths[STRUCTURAL_PUBLICATION_RECEIPT_NAME])
    unsigned = dict(receipt)
    claimed = unsigned.pop("receipt_sha256", None)
    if claimed != canonical_sha256(unsigned):
        raise ValueError("structural publication receipt self-hash mismatch")
    analysis = _load_json(analysis_path)
    source = receipt.get("source")
    artifacts = receipt.get("artifacts")
    if (
        receipt.get("schema_version") != 2
        or receipt.get("receipt_type")
        != "structural_sensitivity_publication_receipt"
        or receipt.get("probability_interpretation") is not False
        or not isinstance(source, dict)
        or source.get("name") != analysis_path.name
        or source.get("bytes") != analysis_path.stat().st_size
        or source.get("literal_sha256") != file_sha256(analysis_path)
        or source.get("analysis_sha256") != analysis.get("analysis_sha256")
        or source.get("source_commit") != source_commit
        or not isinstance(artifacts, list)
    ):
        raise ValueError("structural publication receipt identity mismatch")
    expected_artifacts = {
        STRUCTURAL_CSV_NAME, STRUCTURAL_PNG_NAME, STRUCTURAL_PDF_NAME,
    }
    if {
        record.get("name") for record in artifacts if isinstance(record, dict)
    } != expected_artifacts or len(artifacts) != len(expected_artifacts):
        raise ValueError("structural publication artifact inventory mismatch")
    for record in artifacts:
        if not isinstance(record, dict):
            raise ValueError("structural publication artifact record is malformed")
        path = paths[str(record["name"])]
        if (
            record.get("bytes") != path.stat().st_size
            or record.get("sha256") != file_sha256(path)
        ):
            raise ValueError(f"structural publication artifact hash mismatch: {path}")
    expected_rows = summary_rows(analysis)
    if (
        len(expected_rows) != EXPECTED_STRUCTURAL_SUMMARY_ROWS
        or receipt.get("row_count") != EXPECTED_STRUCTURAL_SUMMARY_ROWS
        or paths[STRUCTURAL_CSV_NAME].read_bytes() != _csv_bytes(expected_rows)
    ):
        raise ValueError("structural CSV is not the exact 55-row analysis projection")
    quality = {
        "png": _inspect_png(paths[STRUCTURAL_PNG_NAME]),
        "pdf": _inspect_pdf(paths[STRUCTURAL_PDF_NAME]),
    }
    expected_style = _figure_style_record(expected_rows, quality)
    if (
        expected_style.get("contract") != FIGURE_STYLE_CONTRACT
        or receipt.get("figure_style") != expected_style
    ):
        raise ValueError("structural figure style or measured quality record mismatch")
    return [path.relative_to(run_root).as_posix() for path in paths.values()]


def _records(run_root: Path, relative_paths: Iterable[str]) -> list[dict[str, Any]]:
    paths = sorted(set(relative_paths))
    records: list[dict[str, Any]] = []
    for relative in paths:
        path = _safe_file(run_root, relative)
        records.append({
            "path": Path(relative).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        })
    return records


def _complete_task_artifact_paths(run_root: Path) -> list[str]:
    """Inventory every durable per-task artifact, excluding no scientific data."""

    paths: list[str] = []
    tasks_root = run_root / "tasks"
    if tasks_root.is_symlink() or not tasks_root.is_dir():
        raise ValueError("structural tasks directory is missing or unsafe")
    for path in sorted(tasks_root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"structural task artifact is a symlink: {path}")
        if path.is_file() and any(
            part.endswith(("__artifacts", "__attempts"))
            for part in path.relative_to(tasks_root).parts
        ):
            paths.append(path.relative_to(run_root).as_posix())
    return paths


def _runtime_receipt_summary(
    run_root: Path, *, run_tag: str, source_commit: str,
    source_tree_sha256: str,
) -> tuple[list[str], dict[str, Any]]:
    receipt_root = run_root / "runtime_receipts"
    if receipt_root.is_symlink() or not receipt_root.is_dir():
        raise ValueError("structural runtime-receipt directory is missing or unsafe")
    paths = sorted(receipt_root.rglob("*.json"))
    successful_tasks: set[int] = set()
    failed_attempts = 0
    summed_wall = 0.0
    summed_cpu = 0.0
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"structural runtime receipt is unsafe: {path}")
        payload = _load_json(path)
        unsigned = dict(payload)
        claimed = unsigned.pop("receipt_sha256", None)
        encoded = json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
        if claimed != hashlib.sha256(encoded).hexdigest():
            raise ValueError(f"structural runtime receipt self-hash mismatch: {path}")
        identity = payload.get("run_identity") or {}
        if (
            payload.get("resource_available") is not True
            or identity.get("run_tag") != run_tag
            or identity.get("source_commit") != source_commit
            or identity.get("source_tree_sha256") != source_tree_sha256
        ):
            raise ValueError(f"structural runtime receipt identity mismatch: {path}")
        label = str(payload.get("label") or "")
        match = re.fullmatch(r"structural_task_(\d+)", label)
        if match is None or not 0 <= int(match.group(1)) < 3_000:
            raise ValueError(f"structural runtime receipt label is invalid: {path}")
        wall = payload.get("wall_seconds")
        if not isinstance(wall, (int, float)) or float(wall) < 0:
            raise ValueError(f"structural runtime receipt wall time is invalid: {path}")
        if payload.get("returncode") == 0:
            task_index = int(match.group(1))
            if task_index in successful_tasks:
                raise ValueError(
                    f"multiple successful runtime receipts for task {task_index}"
                )
            successful_tasks.add(task_index)
            summed_wall += float(wall)
            resource_delta = payload.get("resource_child_delta_or_peak") or {}
            summed_cpu += float(resource_delta.get("ru_utime", 0.0)) + float(
                resource_delta.get("ru_stime", 0.0)
            )
        else:
            failed_attempts += 1
    if successful_tasks != set(range(3_000)):
        missing = sorted(set(range(3_000)).difference(successful_tasks))
        raise ValueError(
            "structural runtime evidence lacks one successful receipt per task; "
            f"missing={missing[:10]}"
        )
    return [path.relative_to(run_root).as_posix() for path in paths], {
        "successful_task_receipts": len(successful_tasks),
        "failed_attempt_receipts": failed_attempts,
        "total_receipts": len(paths),
        "summed_task_wall_seconds_nonconcurrent": summed_wall,
        "summed_child_cpu_seconds": summed_cpu,
    }


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = int(size)
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _write_verified_archive(
    archive: Path,
    *,
    run_root: Path,
    manifest_path: Path,
    records: list[Mapping[str, Any]],
) -> None:
    if archive.exists():
        raise FileExistsError(f"refusing to overwrite structural archive: {archive}")
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    manifest_bytes = manifest_path.read_bytes()
    with temporary.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
                manifest_name = f"{ARCHIVE_PREFIX}/{manifest_path.name}"
                tar.addfile(_tar_info(manifest_name, len(manifest_bytes)), io.BytesIO(manifest_bytes))
                for record in records:
                    relative = str(record["path"])
                    payload = _safe_file(run_root, relative).read_bytes()
                    if len(payload) != int(record["bytes"]):
                        raise ValueError(f"evidence size changed during packaging: {relative}")
                    if hashlib.sha256(payload).hexdigest() != record["sha256"]:
                        raise ValueError(f"evidence hash changed during packaging: {relative}")
                    name = f"{ARCHIVE_PREFIX}/{relative}"
                    tar.addfile(_tar_info(name, len(payload)), io.BytesIO(payload))
    temporary.replace(archive)

    expected = {
        f"{ARCHIVE_PREFIX}/{manifest_path.name}": {
            "bytes": len(manifest_bytes),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
        **{
            f"{ARCHIVE_PREFIX}/{record['path']}": {
                "bytes": int(record["bytes"]),
                "sha256": str(record["sha256"]),
            }
            for record in records
        },
    }
    with tarfile.open(archive, mode="r:gz") as tar:
        members = tar.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError("archive membership does not equal the evidence manifest")
        for member in members:
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError(f"archive contains a non-regular member: {member.name}")
            extracted = tar.extractfile(member)
            if extracted is None:
                raise ValueError(f"cannot read archived member: {member.name}")
            payload = extracted.read()
            record = expected[member.name]
            if len(payload) != record["bytes"]:
                raise ValueError(f"archived size mismatch: {member.name}")
            if hashlib.sha256(payload).hexdigest() != record["sha256"]:
                raise ValueError(f"archived SHA-256 mismatch: {member.name}")


def finalize_run(
    run_plan: Path,
    *,
    status_path: Path,
    analysis_path: Path,
    environment_path: Path,
    scheduler_accounting_path: Path,
    manifest_path: Path,
    archive_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    run_plan = run_plan.resolve()
    run_root = run_plan.parent
    plan, _protocol, _design, manifest = _load_plan_bundle(run_plan)
    run_tag = str(plan.get("run_tag") or "")
    source_commit = str(plan.get("source_commit") or "")
    if not run_tag or plan.get("execution_scope") != "structural_sensitivity_only":
        raise ValueError("only a run-tag-bound structural-only plan can be finalized")
    if plan.get("source_tree_clean_at_generation") is not True:
        raise ValueError("structural run plan was not generated from a clean source tree")
    validator_source_identity = validate_clean_validator_checkout(
        source_commit, repo_root=REPO_ROOT,
    )
    snapshot_failures = source_snapshot_errors(repo_root=REPO_ROOT)
    if snapshot_failures:
        raise ValueError(
            "structural finalizer source snapshot failed: "
            + "; ".join(snapshot_failures[:5])
        )
    submission_path = run_root / "slurm_submission.json"
    submission = _load_json(submission_path)
    publisher_slurm_job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    if not publisher_slurm_job_id:
        raise ValueError(
            "structural finalization requires the actual publisher SLURM_JOB_ID"
        )
    _validate_submission(
        submission,
        run_tag=run_tag,
        source_commit=source_commit,
        publisher_slurm_job_id=publisher_slurm_job_id,
    )
    if os.environ.get("AGRIBRAIN_SOURCE_SNAPSHOT_MODE", "").strip() != (
        submission["source_snapshot_mode"]
    ) or os.environ.get("AGRIBRAIN_SOURCE_TREE_SHA256", "").strip() != (
        submission["source_tree_sha256"]
    ):
        raise ValueError(
            "structural publisher source snapshot differs from the submission receipt"
        )
    expected_outputs = {
        status_path.resolve(): run_root / "completion_status.json",
        analysis_path.resolve(): run_root / "structural_sensitivity_analysis.json",
        environment_path.resolve(): run_root / "publication_environment.json",
        scheduler_accounting_path.resolve(): run_root / "slurm_simulation_accounting.json",
        manifest_path.resolve(): run_root / "structural_sensitivity_artifact_manifest.json",
        archive_path.resolve(): run_root / f"structural_sensitivity_evidence_{run_tag}.tar.gz",
        receipt_path.resolve(): run_root / "structural_sensitivity_archive_receipt.json",
    }
    for actual, expected in expected_outputs.items():
        if actual != expected.resolve():
            raise ValueError(
                f"structural final evidence path {actual} must equal {expected.resolve()}"
            )

    current_status, retained_ledgers = validate_completed_results_with_ledgers(
        run_plan, submission_receipt=submission,
    )
    _validate_status(current_status)
    if len(retained_ledgers) != 6_500:
        raise ValueError("structural finalization requires exactly 6,500 ledgers")
    saved_status = _load_json(status_path.resolve())
    _validate_status(saved_status)
    if saved_status != current_status:
        raise ValueError("saved completion status differs from a fresh hash validation")

    analysis = _load_json(analysis_path.resolve())
    _validate_analysis(analysis, source_commit=source_commit)
    regenerated = analyze_run(run_plan)
    if analysis != regenerated:
        raise ValueError("saved analysis differs from a fresh analysis of task bytes")

    environment = _load_json(environment_path.resolve())
    _validate_environment(environment, run_tag=run_tag, source_commit=source_commit)
    scheduler_accounting = _load_json(scheduler_accounting_path.resolve())
    scheduler_summary = validate_accounting_payload(
        scheduler_accounting,
        kind="structural",
        run_tag=run_tag,
        source_commit=source_commit,
        source_tree_sha256=submission["source_tree_sha256"],
        expected_task_count=3_000,
    )
    structural_publication_paths = _validate_structural_publication_artifacts(
        run_root,
        analysis_path=analysis_path.resolve(),
        source_commit=source_commit,
    )

    required_relative = ["run_plan.json"]
    required_relative.extend(str(name) for name in plan["artifact_sha256"])
    required_relative.extend(str(task["output_relpath"]) for task in manifest["tasks"])
    required_relative.extend(retained_ledgers)
    task_artifact_paths = _complete_task_artifact_paths(run_root)
    required_relative.extend(task_artifact_paths)
    runtime_paths, runtime_summary = _runtime_receipt_summary(
        run_root,
        run_tag=run_tag,
        source_commit=source_commit,
        source_tree_sha256=submission["source_tree_sha256"],
    )
    required_relative.extend(runtime_paths)
    required_relative.extend(structural_publication_paths)
    # Completed upstream task logs are useful operational evidence. The current
    # publisher log is deliberately not read while it is still being written.
    logs_root = run_root / "logs"
    if logs_root.exists():
        if logs_root.is_symlink() or not logs_root.is_dir():
            raise ValueError("structural log directory is unsafe")
        for log_path in sorted(logs_root.glob("task_*")):
            if log_path.is_symlink() or not log_path.is_file():
                raise ValueError(f"structural task log is unsafe: {log_path}")
            required_relative.append(log_path.relative_to(run_root).as_posix())
    for evidence in (
        status_path, analysis_path, environment_path, scheduler_accounting_path,
    ):
        resolved = evidence.resolve()
        if not resolved.is_relative_to(run_root):
            raise ValueError(f"final evidence must be inside the run directory: {evidence}")
        required_relative.append(resolved.relative_to(run_root).as_posix())
    required_relative.append(submission_path.relative_to(run_root).as_posix())
    records = _records(run_root, required_relative)
    ledger_path_set = set(retained_ledgers)
    ledger_records = [
        record for record in records if str(record["path"]) in ledger_path_set
    ]
    episode_archive_records = [
        record for record in records
        if "/complete_episode_evidence/" in str(record["path"])
        and str(record["path"]).endswith(".json.gz")
        and any(
            part.endswith("__artifacts")
            for part in Path(str(record["path"])).parts
        )
    ]
    adaptation_ledger_records = [
        record for record in records
        if "/adaptation_episode_ledgers/" in str(record["path"])
        and str(record["path"]).endswith(".jsonl.gz")
        and any(
            part.endswith("__artifacts")
            for part in Path(str(record["path"])).parts
        )
    ]
    episode_manifest_records = [
        record for record in records
        if str(record["path"]).endswith(
            "__artifacts/complete_episode_evidence_manifest.json"
        )
    ]
    if (
        len(ledger_records) != 6_500
        or len(episode_archive_records) != 24_500
        or len(adaptation_ledger_records) != 18_000
        or len(episode_manifest_records) != 3_000
    ):
        raise ValueError(
            "structural evidence must contain 24,500 episode archives, 18,000 "
            "adaptation ledgers, 6,500 retained ledgers, and 3,000 complete "
            "episode manifests"
        )
    ledger_set_sha256 = canonical_sha256(ledger_records)
    failed_attempt_records = [
        record for record in records
        if any(
            part.endswith("__attempts")
            for part in Path(str(record["path"])).parts
        )
    ]

    if manifest_path.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite an existing structural manifest or receipt")
    if validate_clean_validator_checkout(
        source_commit, repo_root=REPO_ROOT,
    ) != validator_source_identity:
        raise ValueError("structural validator source identity changed before archive creation")
    snapshot_failures = source_snapshot_errors(repo_root=REPO_ROOT)
    if snapshot_failures:
        raise ValueError(
            "structural finalizer source snapshot changed before archive creation: "
            + "; ".join(snapshot_failures[:5])
        )
    artifact_manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "analysis_label": "structural sensitivity",
        "probability_interpretation": False,
        "execution_scope": "structural_sensitivity_only",
        "run_tag": run_tag,
        "source_commit": source_commit,
        "source_snapshot_binding": {
            "mode": submission["source_snapshot_mode"],
            "source_tree_sha256": submission["source_tree_sha256"],
        },
        "validator_source_identity": validator_source_identity,
        "design_sha256": manifest["design_sha256"],
        "accounting": manifest["accounting"],
        "n_parameters": 29,
        "n_tasks": 3_000,
        "retained_decision_ledger_count": 6_500,
        "retained_decision_ledger_set_sha256": ledger_set_sha256,
        "complete_episode_evidence": {
            "executed_episode_archives": len(episode_archive_records),
            "adaptation_episode_ledgers": len(adaptation_ledger_records),
            "final_episode_ledgers": len(ledger_records),
            "per_task_manifests": len(episode_manifest_records),
            "runtime_receipts": runtime_summary,
            "scheduler_accounting": scheduler_summary,
            "failed_attempt_artifacts": {
                "file_count": len(failed_attempt_records),
                "literal_bytes": sum(
                    int(record["bytes"]) for record in failed_attempt_records
                ),
                "retention_policy": (
                    "Retained for diagnosis and audit; excluded from canonical "
                    "episode and ledger counts."
                ),
            },
        },
        "excluded_runtime_material": (
            "temporary files, interpreter caches, and the in-progress publisher "
            "log only; every durable task artifact, episode archive, adaptation "
            "ledger, final ledger, worker runtime receipt, and completed task log "
            "is included"
        ),
        "file_count": len(records),
        "files": records,
    }
    artifact_manifest["manifest_sha256"] = canonical_sha256(artifact_manifest)
    _atomic_json(manifest_path, artifact_manifest)
    _write_verified_archive(
        archive_path,
        run_root=run_root,
        manifest_path=manifest_path,
        records=records,
    )
    if validate_clean_validator_checkout(
        source_commit, repo_root=REPO_ROOT,
    ) != validator_source_identity:
        raise ValueError("structural validator source identity changed during archive creation")
    snapshot_failures = source_snapshot_errors(repo_root=REPO_ROOT)
    if snapshot_failures:
        raise ValueError(
            "structural finalizer source snapshot changed during archive creation: "
            + "; ".join(snapshot_failures[:5])
        )
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "receipt_type": "structural_sensitivity_semantic_archive_receipt",
        "analysis_label": "structural sensitivity",
        "validation_status": "PASS",
        "run_tag": run_tag,
        "source_commit": source_commit,
        "source_snapshot_binding": artifact_manifest["source_snapshot_binding"],
        "validator_source_identity": validator_source_identity,
        "publisher_execution": {
            "slurm_job_id": publisher_slurm_job_id,
            "declared_publisher_job_id": submission["publisher"]["job_id"],
            "identity_match": True,
        },
        "archive": {
            "name": archive_path.name,
            "bytes": archive_path.stat().st_size,
            "sha256": file_sha256(archive_path),
        },
        "artifact_manifest": {
            "name": manifest_path.name,
            "bytes": manifest_path.stat().st_size,
            "sha256": file_sha256(manifest_path),
            "content_sha256": artifact_manifest["manifest_sha256"],
        },
        "archive_member_count": len(records) + 1,
        "locked_accounting": {
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
        },
        "validation": {
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
        },
        "evidence_scope": {
            "structural_sensitivity_evidence": True,
            "core_publication_evidence_included": False,
            "full_submission_requires_core_receipt": True,
        },
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _atomic_json(receipt_path, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-plan", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--scheduler-accounting", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = finalize_run(
        args.run_plan,
        status_path=args.status,
        analysis_path=args.analysis,
        environment_path=args.environment,
        scheduler_accounting_path=args.scheduler_accounting,
        manifest_path=args.manifest,
        archive_path=args.archive,
        receipt_path=args.receipt,
    )
    print(json.dumps(receipt, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
