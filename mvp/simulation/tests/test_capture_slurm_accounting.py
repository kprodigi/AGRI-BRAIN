from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from hpc.capture_slurm_accounting import capture_accounting


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _write_core_receipt(path: Path) -> tuple[str, str, str]:
    run_tag = "20260828-abcdef0"
    commit = "a" * 40
    tree = "b" * 64
    payload = {
        "schema_version": 2,
        "run_tag": run_tag,
        "source_commit": commit,
        "source_tree_sha256": tree,
        "slurm_dag": {
            "seed_array": {"job_id": "100", "task_count": 2},
            "stress_array": {"job_id": "200", "task_count": 1},
        },
    }
    payload["receipt_sha256"] = _canonical_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return run_tag, commit, tree


def _runner_with_complete_rows(command, **_kwargs):
    if command[1] == "--helpformat":
        return subprocess.CompletedProcess(
            command,
            0,
            "JobIDRaw State ExitCode ElapsedRaw AllocCPUS ConsumedEnergyRaw\n",
            "",
        )
    if command[1] == "--version":
        return subprocess.CompletedProcess(command, 0, "slurm 24.05\n", "")
    # Field order follows the desired-field order selected by the implementation.
    stdout = "\n".join((
        "100_0|COMPLETED|0:0|11|2|101|",
        "100_1|COMPLETED|0:0|13|2|103|",
        "200_0|COMPLETED|0:0|17|4|107|",
    )) + "\n"
    return subprocess.CompletedProcess(command, 0, stdout, "")


def test_capture_slurm_accounting_binds_every_array_task_and_energy(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "submission.json"
    run_tag, commit, tree = _write_core_receipt(receipt)
    output = tmp_path / "accounting.json"
    payload = capture_accounting(
        submission_receipt=receipt,
        output=output,
        kind="core",
        run_tag=run_tag,
        source_commit=commit,
        source_tree_sha256=tree,
        attempts=1,
        retry_seconds=0,
        runner=_runner_with_complete_rows,
    )
    assert output.is_file()
    assert [record["completed_task_count"] for record in payload["arrays"]] == [2, 1]
    assert [record["summed_allocated_cpu_seconds"] for record in payload["arrays"]] == [48, 68]
    assert payload["energy"]["summed_consumed_energy_raw_joules"] == 311
    unsigned = dict(payload)
    claimed = unsigned.pop("accounting_sha256")
    assert claimed == _canonical_sha256(unsigned)


def test_capture_slurm_accounting_rejects_an_incomplete_array(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "submission.json"
    run_tag, commit, tree = _write_core_receipt(receipt)

    def incomplete(command, **kwargs):
        result = _runner_with_complete_rows(command, **kwargs)
        if command[1] not in {"--helpformat", "--version"}:
            result.stdout = "100_0|COMPLETED|0:0|11|2|101|\n200_0|COMPLETED|0:0|17|4|107|\n"
        return result

    with pytest.raises(RuntimeError, match="accounting remained incomplete"):
        capture_accounting(
            submission_receipt=receipt,
            output=tmp_path / "accounting.json",
            kind="core",
            run_tag=run_tag,
            source_commit=commit,
            source_tree_sha256=tree,
            attempts=1,
            retry_seconds=0,
            runner=incomplete,
        )
    assert not (tmp_path / "accounting.json").exists()
