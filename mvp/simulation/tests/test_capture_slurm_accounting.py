from __future__ import annotations

import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from hpc.capture_slurm_accounting import (
    capture_accounting,
    validate_accounting_payload,
)


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
            "JobID JobIDRaw State ExitCode ElapsedRaw AllocCPUS ConsumedEnergyRaw\n",
            "",
        )
    if command[1] == "--version":
        return subprocess.CompletedProcess(command, 0, "slurm 24.05\n", "")
    # Field order follows the desired-field order selected by the implementation.
    stdout = "\n".join((
        "100_0|1000|COMPLETED|0:0|11|2|101|",
        "100_1|1001|COMPLETED|0:0|13|2|103|",
        "200_0|2000|COMPLETED|0:0|17|4|107|",
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
    assert "--array" in payload["scheduler"]["command"]
    assert "--local" in payload["scheduler"]["command"]
    assert payload["rows"][0]["JobID"] == "100_0"
    assert payload["rows"][0]["JobIDRaw"] == "1000"
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
            result.stdout = (
                "100_0|1000|COMPLETED|0:0|11|2|101|\n"
                "200_0|2000|COMPLETED|0:0|17|4|107|\n"
            )
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


def test_capture_slurm_accounting_outlasts_six_delayed_queries_with_a_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = tmp_path / "submission.json"
    run_tag, commit, tree = _write_core_receipt(receipt)
    data_queries = 0

    def delayed(command, **kwargs):
        nonlocal data_queries
        result = _runner_with_complete_rows(command, **kwargs)
        if command[1] not in {"--helpformat", "--version"}:
            data_queries += 1
            if data_queries <= 6:
                result.stdout = (
                    "100_0|1000|COMPLETED|0:0|11|2|101|\n"
                    "200_0|2000|COMPLETED|0:0|17|4|107|\n"
                )
        return result

    observed_delays: list[float] = []
    monkeypatch.setattr(
        "hpc.capture_slurm_accounting.time.sleep", observed_delays.append,
    )
    payload = capture_accounting(
        submission_receipt=receipt,
        output=tmp_path / "accounting.json",
        kind="core",
        run_tag=run_tag,
        source_commit=commit,
        source_tree_sha256=tree,
        attempts=7,
        retry_seconds=1,
        max_retry_seconds=4,
        runner=delayed,
    )

    assert data_queries == 7
    assert observed_delays == [1, 2, 4, 4, 4, 4]
    assert payload["scheduler"]["attempts"] == 7
    assert payload["scheduler"]["retry_policy"] == {
        "maximum_attempts": 7,
        "initial_delay_seconds": 1,
        "backoff_multiplier": 2.0,
        "maximum_delay_seconds": 4,
        "query_timeout_seconds": 60.0,
        "applied_delays_seconds": [1, 2, 4, 4, 4, 4],
        "total_wait_seconds": 19,
    }


def test_capture_slurm_accounting_rejects_reused_raw_allocation_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = tmp_path / "submission.json"
    run_tag, commit, tree = _write_core_receipt(receipt)

    def reused_raw_id(command, **kwargs):
        result = _runner_with_complete_rows(command, **kwargs)
        if command[1] not in {"--helpformat", "--version"}:
            result.stdout = result.stdout.replace("100_1|1001|", "100_1|1000|")
        return result

    def unexpected_sleep(_seconds: float) -> None:
        pytest.fail("permanent accounting contradictions must fail without retrying")

    monkeypatch.setattr(
        "hpc.capture_slurm_accounting.time.sleep", unexpected_sleep,
    )
    with pytest.raises(ValueError, match="reuse JobIDRaw 1000"):
        capture_accounting(
            submission_receipt=receipt,
            output=tmp_path / "accounting.json",
            kind="core",
            run_tag=run_tag,
            source_commit=commit,
            source_tree_sha256=tree,
            runner=reused_raw_id,
        )


def test_capture_slurm_accounting_rejects_terminal_failure_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = tmp_path / "submission.json"
    run_tag, commit, tree = _write_core_receipt(receipt)

    def terminal_failure(command, **kwargs):
        result = _runner_with_complete_rows(command, **kwargs)
        if command[1] not in {"--helpformat", "--version"}:
            result.stdout = result.stdout.replace(
                "100_0|1000|COMPLETED|0:0|", "100_0|1000|FAILED|1:0|"
            )
        return result

    def unexpected_sleep(_seconds: float) -> None:
        pytest.fail("terminal task failure must fail without retrying")

    monkeypatch.setattr(
        "hpc.capture_slurm_accounting.time.sleep", unexpected_sleep,
    )
    with pytest.raises(ValueError, match="terminally unsuccessful tasks"):
        capture_accounting(
            submission_receipt=receipt,
            output=tmp_path / "accounting.json",
            kind="core",
            run_tag=run_tag,
            source_commit=commit,
            source_tree_sha256=tree,
            runner=terminal_failure,
        )


def test_capture_slurm_accounting_retries_a_timed_out_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = tmp_path / "submission.json"
    run_tag, commit, tree = _write_core_receipt(receipt)
    data_queries = 0

    def timeout_once(command, **kwargs):
        nonlocal data_queries
        if command[1] in {"--helpformat", "--version"}:
            return _runner_with_complete_rows(command, **kwargs)
        data_queries += 1
        if data_queries == 1:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return _runner_with_complete_rows(command, **kwargs)

    observed_delays: list[float] = []
    monkeypatch.setattr(
        "hpc.capture_slurm_accounting.time.sleep", observed_delays.append,
    )
    payload = capture_accounting(
        submission_receipt=receipt,
        output=tmp_path / "accounting.json",
        kind="core",
        run_tag=run_tag,
        source_commit=commit,
        source_tree_sha256=tree,
        attempts=2,
        retry_seconds=1,
        max_retry_seconds=1,
        query_timeout_seconds=3,
        runner=timeout_once,
    )

    assert data_queries == 2
    assert observed_delays == [1]
    assert payload["scheduler"]["retry_policy"]["query_timeout_seconds"] == 3


def test_persisted_accounting_is_rederived_from_rows(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "submission.json"
    run_tag, commit, tree = _write_core_receipt(receipt)
    payload = capture_accounting(
        submission_receipt=receipt,
        output=tmp_path / "accounting.json",
        kind="core",
        run_tag=run_tag,
        source_commit=commit,
        source_tree_sha256=tree,
        attempts=1,
        retry_seconds=0,
        runner=_runner_with_complete_rows,
    )

    def resign(candidate: dict) -> dict:
        candidate.pop("accounting_sha256")
        candidate["accounting_sha256"] = _canonical_sha256(candidate)
        return candidate

    def validate(candidate: dict) -> None:
        validate_accounting_payload(
            candidate,
            kind="core",
            run_tag=run_tag,
            source_commit=commit,
            source_tree_sha256=tree,
            expected_task_count=3,
        )

    altered_row = deepcopy(payload)
    altered_row["rows"][0]["State"] = "FAILED"
    with pytest.raises(ValueError, match="rows differ from raw sacct output"):
        validate(resign(altered_row))

    altered_array_summary = deepcopy(payload)
    altered_array_summary["arrays"][0][
        "summed_allocated_cpu_seconds"
    ] += 1
    with pytest.raises(ValueError, match="array summaries differ"):
        validate(resign(altered_array_summary))

    altered_energy = deepcopy(payload)
    altered_energy["energy"]["summed_consumed_energy_raw_joules"] += 1
    with pytest.raises(ValueError, match="energy summary differs"):
        validate(resign(altered_energy))


@pytest.mark.parametrize(
    "publisher", ("hpc_publish.sh", "hpc_sensitivity_publish.sh"),
)
def test_publishers_declare_the_bounded_accounting_retry_policy(
    publisher: str,
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = (repo_root / "hpc" / publisher).read_text(encoding="utf-8")
    assert "--attempts 12" in script
    assert "--retry-seconds 5" in script
    assert "--max-retry-seconds 120" in script
    assert "--query-timeout-seconds 60" in script
