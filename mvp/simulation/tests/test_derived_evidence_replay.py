"""Mutation-negative tests for deterministic derived-evidence replay gates."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mvp.simulation.validation import validate_publication_artifacts as vpa


ALL_REPLAY_OUTPUTS = (
    *vpa.EXPECTED_DERIVED_REPLAY_ARTIFACTS,
    *vpa.EXPECTED_STRESS_TASK_FILES,
)


def _write_matching_outputs(
    canonical: Path,
    replayed: Path,
    names: tuple[str, ...],
) -> None:
    canonical.mkdir(parents=True)
    replayed.mkdir(parents=True)
    for index, name in enumerate(names):
        payload = f"artifact={name};value={index}\n".encode()
        (canonical / name).write_bytes(payload)
        (replayed / name).write_bytes(payload)


@pytest.mark.parametrize("mutated_name", ALL_REPLAY_OUTPUTS)
def test_exact_replay_comparison_rejects_each_derived_output_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutated_name: str,
) -> None:
    canonical = tmp_path / "canonical"
    replayed = tmp_path / "replayed"
    names = (
        vpa.EXPECTED_DERIVED_REPLAY_ARTIFACTS
        if mutated_name in vpa.EXPECTED_DERIVED_REPLAY_ARTIFACTS
        else vpa.EXPECTED_STRESS_TASK_FILES
    )
    _write_matching_outputs(canonical, replayed, names)
    monkeypatch.setattr(vpa, "RESULTS_DIR", canonical)
    vpa._compare_exact_replay_artifacts(
        replayed, names, label="test producer",
    )

    (canonical / mutated_name).write_bytes(b"coherent-but-fabricated\n")
    with pytest.raises(SystemExit):
        vpa._compare_exact_replay_artifacts(
            replayed, names, label="test producer",
        )


def test_exact_replay_comparison_rejects_unexpected_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "canonical"
    replayed = tmp_path / "replayed"
    names = vpa.EXPECTED_DERIVED_REPLAY_ARTIFACTS
    _write_matching_outputs(canonical, replayed, names)
    monkeypatch.setattr(vpa, "RESULTS_DIR", canonical)
    (replayed / "unreviewed_diagnostic.json").write_text("{}")
    with pytest.raises(SystemExit):
        vpa._compare_exact_replay_artifacts(
            replayed, names, label="test producer",
        )


def _raw_h3_manifest(results: Path, run_tag: str) -> dict:
    records = []
    for scenario in vpa.EXPECTED_SCENARIOS:
        for name in vpa.EXPECTED_STRESS_TASK_FILES:
            relative = f"stress_runs/{run_tag}/{scenario}/{name}"
            path = results / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = f"{scenario}/{name}\n".encode()
            path.write_bytes(payload)
            records.append({
                "file": relative,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            })
    return {"artifact_run_tag": run_tag, "artifacts": records}


def test_h3_replay_stages_exact_manifested_task_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = tmp_path / "results"
    run_tag = "abcdef0_20260828_120000"
    manifest = _raw_h3_manifest(results, run_tag)
    monkeypatch.setattr(vpa, "RESULTS_DIR", results)
    staging = tmp_path / "staging"

    vpa._stage_manifested_h3_task_inputs(staging, manifest)
    observed = {
        path.relative_to(staging).as_posix()
        for path in staging.rglob("*") if path.is_file()
    }
    assert observed == {
        f"{scenario}/{name}"
        for scenario in vpa.EXPECTED_SCENARIOS
        for name in vpa.EXPECTED_STRESS_TASK_FILES
    }


def test_h3_replay_rejects_task_mutated_after_manifesting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = tmp_path / "results"
    run_tag = "abcdef0_20260828_120000"
    manifest = _raw_h3_manifest(results, run_tag)
    monkeypatch.setattr(vpa, "RESULTS_DIR", results)
    target = (
        results / "stress_runs" / run_tag / "baseline" / "stress_h3_test.json"
    )
    target.write_bytes(b"coherently edited after manifest\n")

    with pytest.raises(SystemExit):
        vpa._stage_manifested_h3_task_inputs(tmp_path / "staging", manifest)
