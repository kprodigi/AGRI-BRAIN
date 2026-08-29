"""Fail-closed bindings between worker payloads and Slurm receipts."""
from __future__ import annotations

import pytest

from hpc.core_submission_receipt import build_receipt
from hpc.slurm_execution_provenance import (
    build_array_execution_provenance,
    require_declared_publisher,
    validate_core_array_provenance,
    validate_structural_array_provenance,
)


SOURCE_DIGEST = "a" * 64
SNAPSHOT_MODE = "detached_readonly_git_worktree_v1"


def _env(*, job: str, array: str, local_index: int) -> dict[str, str]:
    return {
        "AGRIBRAIN_SOURCE_SNAPSHOT_MODE": SNAPSHOT_MODE,
        "AGRIBRAIN_SOURCE_TREE_SHA256": SOURCE_DIGEST,
        "SLURM_JOB_ID": job,
        "SLURM_ARRAY_JOB_ID": array,
        "SLURM_ARRAY_TASK_ID": str(local_index),
    }


def _core_receipt() -> dict:
    commit = "b" * 40
    return build_receipt(
        run_tag=f"{commit[:7]}_20260828_120000",
        source_commit=commit,
        partition="compute",
        seed_job_id="101",
        stress_job_id="102",
        publisher_job_id="103",
        source_snapshot_mode=SNAPSHOT_MODE,
        source_tree_sha256=SOURCE_DIGEST,
    )


def test_core_seed_provenance_binds_parent_array_and_exact_seed_index() -> None:
    receipt = _core_receipt()
    provenance = build_array_execution_provenance(
        stage="core_seed_array",
        logical_task_index=3,
        environ=_env(job="9001", array="101", local_index=3),
    )
    assert validate_core_array_provenance(
        provenance,
        stage="core_seed_array",
        logical_task_index=3,
        submission_receipt=receipt,
    ) == provenance

    provenance["slurm_array_task_id"] = 4
    with pytest.raises(ValueError, match="slurm_array_task_id"):
        validate_core_array_provenance(
            provenance,
            stage="core_seed_array",
            logical_task_index=3,
            submission_receipt=receipt,
        )


def test_core_provenance_rejects_source_tree_or_array_parent_substitution() -> None:
    receipt = _core_receipt()
    provenance = build_array_execution_provenance(
        stage="core_stress_array",
        logical_task_index=2,
        environ=_env(job="9002", array="102", local_index=2),
    )
    provenance["source_tree_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="source_tree_sha256"):
        validate_core_array_provenance(
            provenance,
            stage="core_stress_array",
            logical_task_index=2,
            submission_receipt=receipt,
        )

    provenance["source_tree_sha256"] = SOURCE_DIGEST
    provenance["slurm_array_job_id"] = "999"
    with pytest.raises(ValueError, match="slurm_array_job_id"):
        validate_core_array_provenance(
            provenance,
            stage="core_stress_array",
            logical_task_index=2,
            submission_receipt=receipt,
        )


def test_structural_provenance_binds_chunk_offset_and_local_index() -> None:
    receipt = {
        "source_snapshot_mode": SNAPSHOT_MODE,
        "source_tree_sha256": SOURCE_DIGEST,
        "task_arrays": [
            {"job_id": "201", "offset": 0, "count": 1_000},
            {"job_id": "202", "offset": 1_000, "count": 1_000},
            {"job_id": "203", "offset": 2_000, "count": 1_000},
        ],
    }
    provenance = build_array_execution_provenance(
        stage="structural_task_array",
        logical_task_index=2_137,
        environ=_env(job="9137", array="203", local_index=137),
    )
    assert validate_structural_array_provenance(
        provenance,
        logical_task_index=2_137,
        submission_receipt=receipt,
    ) == provenance

    provenance["logical_task_index"] = 2_136
    with pytest.raises(ValueError, match="logical_task_index"):
        validate_structural_array_provenance(
            provenance,
            logical_task_index=2_137,
            submission_receipt=receipt,
        )


def test_publisher_runtime_job_must_equal_declared_publisher() -> None:
    receipt = _core_receipt()
    require_declared_publisher(receipt, actual_slurm_job_id="103")
    with pytest.raises(ValueError, match="differs from declared"):
        require_declared_publisher(receipt, actual_slurm_job_id="104")
