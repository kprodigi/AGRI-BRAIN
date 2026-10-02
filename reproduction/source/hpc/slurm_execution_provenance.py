#!/usr/bin/env python3
"""Build and validate literal Slurm worker/source-snapshot provenance.

Submission receipts identify array *parents*.  The worker envelopes additionally
record the actual array element and batch job that produced each payload.  This
module keeps that small contract identical across the core and structural
workflows without pretending that a submission receipt is a scheduler-completion
record.
"""
from __future__ import annotations

import os
import re
from typing import Any, Mapping


SCHEMA_VERSION = 1
SNAPSHOT_MODE = "detached_readonly_git_worktree_v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_JOB_ID = re.compile(r"^[1-9][0-9]*$")
CORE_SEEDS = (
    42, 1337, 2024, 7, 99, 101, 202, 303, 404, 505,
    606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515,
)
CORE_SCENARIOS = (
    "heatwave", "overproduction", "cyber_outage", "adaptive_pricing", "baseline",
)


def _required_text(env: Mapping[str, str], name: str) -> str:
    value = str(env.get(name, "")).strip()
    if not value:
        raise RuntimeError(f"{name} is required for Slurm execution provenance")
    return value


def build_array_execution_provenance(
    *,
    stage: str,
    logical_task_index: int,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return the exact source and Slurm identity of one array worker."""

    env = os.environ if environ is None else environ
    if stage not in {
        "core_seed_array",
        "core_stress_array",
        "structural_task_array",
    }:
        raise ValueError(f"unsupported Slurm execution stage: {stage!r}")
    digest = _required_text(env, "AGRIBRAIN_SOURCE_TREE_SHA256")
    if not _HEX64.fullmatch(digest):
        raise RuntimeError("AGRIBRAIN_SOURCE_TREE_SHA256 is not a lowercase SHA-256")
    mode = _required_text(env, "AGRIBRAIN_SOURCE_SNAPSHOT_MODE")
    if mode != SNAPSHOT_MODE:
        raise RuntimeError("AGRIBRAIN_SOURCE_SNAPSHOT_MODE is not the locked mode")
    job_id = _required_text(env, "SLURM_JOB_ID")
    array_job_id = _required_text(env, "SLURM_ARRAY_JOB_ID")
    task_raw = _required_text(env, "SLURM_ARRAY_TASK_ID")
    if not _JOB_ID.fullmatch(job_id) or not _JOB_ID.fullmatch(array_job_id):
        raise RuntimeError("Slurm job/array job ids must be positive decimal integers")
    if not task_raw.isdigit():
        raise RuntimeError("SLURM_ARRAY_TASK_ID must be a non-negative integer")
    array_task_id = int(task_raw)
    if logical_task_index < 0:
        raise ValueError("logical_task_index must be non-negative")
    return {
        "schema_version": SCHEMA_VERSION,
        "execution_platform": "slurm",
        "stage": stage,
        "source_snapshot_mode": mode,
        "source_tree_sha256": digest,
        "slurm_job_id": job_id,
        "slurm_array_job_id": array_job_id,
        "slurm_array_task_id": array_task_id,
        "logical_task_index": int(logical_task_index),
    }


def validate_core_array_provenance(
    provenance: object,
    *,
    stage: str,
    logical_task_index: int,
    submission_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a seed/stress payload to its exact parent array and task index."""

    if not isinstance(provenance, dict):
        raise ValueError("Slurm execution provenance is missing or not an object")
    stage_key = {
        "core_seed_array": "seed_array",
        "core_stress_array": "stress_array",
    }.get(stage)
    if stage_key is None:
        raise ValueError(f"unsupported core provenance stage: {stage!r}")
    dag = submission_receipt.get("slurm_dag")
    if not isinstance(dag, dict) or not isinstance(dag.get(stage_key), dict):
        raise ValueError("core submission receipt lacks the required array stage")
    receipt_stage = dag[stage_key]
    expected = {
        "schema_version": SCHEMA_VERSION,
        "execution_platform": "slurm",
        "stage": stage,
        "source_snapshot_mode": submission_receipt.get("source_snapshot_mode"),
        "source_tree_sha256": submission_receipt.get("source_tree_sha256"),
        "slurm_array_job_id": receipt_stage.get("job_id"),
        "slurm_array_task_id": int(logical_task_index),
        "logical_task_index": int(logical_task_index),
    }
    for field, value in expected.items():
        if provenance.get(field) != value:
            raise ValueError(
                f"Slurm execution provenance {field!r}={provenance.get(field)!r}, "
                f"expected {value!r}"
            )
    job_id = provenance.get("slurm_job_id")
    if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
        raise ValueError("Slurm execution provenance has an invalid task job id")
    return dict(provenance)


def validate_structural_array_provenance(
    provenance: object,
    *,
    logical_task_index: int,
    submission_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind one structural result to its exact receipt chunk/local index."""

    if not isinstance(provenance, dict):
        raise ValueError("structural Slurm execution provenance is missing")
    arrays = submission_receipt.get("task_arrays")
    if not isinstance(arrays, list):
        raise ValueError("structural submission receipt lacks task arrays")
    matching = [
        record
        for record in arrays
        if isinstance(record, dict)
        and int(record.get("offset", -1)) <= logical_task_index
        < int(record.get("offset", -1)) + int(record.get("count", 0))
    ]
    if len(matching) != 1:
        raise ValueError("structural task is not covered by exactly one receipt chunk")
    chunk = matching[0]
    local_index = logical_task_index - int(chunk["offset"])
    expected = {
        "schema_version": SCHEMA_VERSION,
        "execution_platform": "slurm",
        "stage": "structural_task_array",
        "source_snapshot_mode": submission_receipt.get("source_snapshot_mode"),
        "source_tree_sha256": submission_receipt.get("source_tree_sha256"),
        "slurm_array_job_id": str(chunk.get("job_id")),
        "slurm_array_task_id": local_index,
        "logical_task_index": int(logical_task_index),
    }
    for field, value in expected.items():
        if provenance.get(field) != value:
            raise ValueError(
                f"structural Slurm provenance {field!r}={provenance.get(field)!r}, "
                f"expected {value!r}"
            )
    job_id = provenance.get("slurm_job_id")
    if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
        raise ValueError("structural Slurm provenance has an invalid task job id")
    return dict(provenance)


def require_declared_publisher(
    submission_receipt: Mapping[str, Any],
    *,
    actual_slurm_job_id: str,
    structural: bool = False,
) -> None:
    """Require the running publisher to be the publisher submitted in the DAG."""

    actual = str(actual_slurm_job_id).strip()
    if not _JOB_ID.fullmatch(actual):
        raise ValueError("publisher SLURM_JOB_ID must be a positive decimal integer")
    if structural:
        publisher = submission_receipt.get("publisher")
    else:
        dag = submission_receipt.get("slurm_dag")
        publisher = dag.get("publisher") if isinstance(dag, dict) else None
    declared = publisher.get("job_id") if isinstance(publisher, dict) else None
    if actual != declared:
        raise ValueError(
            f"publisher SLURM_JOB_ID {actual!r} differs from declared job {declared!r}"
        )
