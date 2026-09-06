"""Fail-closed tests for deterministic and development evidence boundaries."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SIM_DIR = REPO_ROOT / "mvp" / "simulation"


def _load_guard():
    path = SIM_DIR / "validation" / "run_regression_guard.py"
    spec = importlib.util.spec_from_file_location(
        "agribrain_regression_guard_boundary_test", path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = _load_guard()


def _table(contract: dict, key_name: str) -> pd.DataFrame:
    rows = []
    for scenario in contract["scenarios"]:
        for method in contract["methods"]:
            row = {"Scenario": scenario, key_name: method}
            row.update({metric: 0.5 for metric in contract["metrics"]})
            rows.append(row)
    return pd.DataFrame(rows)


def test_checked_in_snapshot_is_numeric_free_schema_v2_pending_marker():
    snapshot_path = SIM_DIR / "baseline_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    protocol_path = SIM_DIR / "experiment_protocol.json"

    assert snapshot["schema_version"] == 2
    assert snapshot["status"] == "pending"
    assert snapshot["scope"] == "deterministic_development_regression_only"
    assert snapshot["publication_evidence"] is False
    assert snapshot["source"]["commit"] is None
    assert snapshot["tables"] is None
    assert snapshot["key_contract"] == guard._canonical_key_contract()
    assert snapshot["protocol"]["sha256"] == hashlib.sha256(
        protocol_path.read_bytes()
    ).hexdigest()
    assert "no_pinn" in snapshot["key_contract"]["table1"]["methods"]
    assert "no_pinn" in snapshot["key_contract"]["table2"]["methods"]


def test_table_digest_rejects_missing_extra_and_duplicate_cells():
    contract = guard._canonical_key_contract()["table1"]
    complete = _table(contract, "Method")
    assert set(guard._digest_table(complete, "table1")) == guard._expected_keys("table1")

    with pytest.raises(SystemExit):
        guard._digest_table(complete.iloc[:-1].copy(), "table1")

    extra = pd.concat(
        [complete, pd.DataFrame([{**complete.iloc[0].to_dict(), "Method": "legacy"}])],
        ignore_index=True,
    )
    with pytest.raises(SystemExit):
        guard._digest_table(extra, "table1")

    duplicate = pd.concat([complete, complete.iloc[[0]]], ignore_index=True)
    with pytest.raises(SystemExit):
        guard._digest_table(duplicate, "table1")


def test_explicit_init_binds_protocol_commit_keys_and_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    protocol = tmp_path / "experiment_protocol.json"
    protocol.write_bytes((SIM_DIR / "experiment_protocol.json").read_bytes())
    snapshot = tmp_path / "baseline_snapshot.json"
    snapshot.write_bytes((SIM_DIR / "baseline_snapshot.json").read_bytes())
    t1 = tmp_path / "table1_summary.csv"
    t2 = tmp_path / "table2_ablation.csv"
    _table(guard._canonical_key_contract()["table1"], "Method").to_csv(t1, index=False)
    _table(guard._canonical_key_contract()["table2"], "Variant").to_csv(t2, index=False)

    commit = "a" * 40
    monkeypatch.setattr(guard, "DETERMINISTIC_MODE", True)
    monkeypatch.setattr(guard, "PROTOCOL", protocol)
    monkeypatch.setattr(guard, "SNAPSHOT", snapshot)
    monkeypatch.setattr(guard, "T1", t1)
    monkeypatch.setattr(guard, "T2", t2)
    monkeypatch.setattr(guard, "_current_source_commit", lambda: commit)
    monkeypatch.setattr(guard, "_assert_clean_source_tree", lambda: None)
    monkeypatch.delenv("ALLOW_MISSING_BASELINE", raising=False)
    monkeypatch.setenv("REGRESSION_GUARD_INIT", "true")

    guard.main()
    initialized = json.loads(snapshot.read_text(encoding="utf-8"))
    assert initialized["status"] == "validated"
    assert initialized["publication_evidence"] is False
    assert initialized["source"]["commit"] == commit
    assert set(initialized["tables"]) == {"table1", "table2"}

    monkeypatch.delenv("REGRESSION_GUARD_INIT", raising=False)
    guard.main()

    monkeypatch.setattr(guard, "_current_source_commit", lambda: "b" * 40)
    with pytest.raises(SystemExit):
        guard.main()


def test_validated_snapshot_fails_when_protocol_bytes_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    protocol = tmp_path / "experiment_protocol.json"
    protocol.write_bytes((SIM_DIR / "experiment_protocol.json").read_bytes())
    snapshot = tmp_path / "baseline_snapshot.json"
    snapshot.write_bytes((SIM_DIR / "baseline_snapshot.json").read_bytes())
    t1 = tmp_path / "table1_summary.csv"
    t2 = tmp_path / "table2_ablation.csv"
    _table(guard._canonical_key_contract()["table1"], "Method").to_csv(t1, index=False)
    _table(guard._canonical_key_contract()["table2"], "Variant").to_csv(t2, index=False)

    monkeypatch.setattr(guard, "DETERMINISTIC_MODE", True)
    monkeypatch.setattr(guard, "PROTOCOL", protocol)
    monkeypatch.setattr(guard, "SNAPSHOT", snapshot)
    monkeypatch.setattr(guard, "T1", t1)
    monkeypatch.setattr(guard, "T2", t2)
    monkeypatch.setattr(guard, "_current_source_commit", lambda: "a" * 40)
    monkeypatch.setattr(guard, "_assert_clean_source_tree", lambda: None)
    monkeypatch.delenv("ALLOW_MISSING_BASELINE", raising=False)
    monkeypatch.setenv("REGRESSION_GUARD_INIT", "true")
    guard.main()

    protocol.write_text(protocol.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    monkeypatch.delenv("REGRESSION_GUARD_INIT", raising=False)
    with pytest.raises(SystemExit):
        guard.main()


def test_development_runner_refuses_canonical_results_directory():
    runner = SIM_DIR / "benchmarks" / "run_benchmark_suite.py"
    env = os.environ.copy()
    backend = REPO_ROOT / "agribrain" / "backend"
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(backend), env.get("PYTHONPATH", "")) if item
    )
    proc = subprocess.run(
        [sys.executable, str(runner), "--output-dir", str(SIM_DIR / "results")],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode != 0
    assert "development-only" in (proc.stdout + proc.stderr)


def test_development_runner_emits_only_nonpublication_names(
    tmp_path: Path,
):
    runner = SIM_DIR / "benchmarks" / "run_benchmark_suite.py"
    table = tmp_path / "table2_ablation.csv"
    output = tmp_path / "development"
    _table(guard._canonical_key_contract()["table2"], "Variant").to_csv(
        table, index=False,
    )
    # The smoke aggregator also reports descriptive carbon/equity fields when
    # available; include them so every metric has one sample.
    frame = pd.read_csv(table)
    frame["Carbon"] = 1.0
    frame["Equity"] = 0.5
    frame.to_csv(table, index=False)

    env = os.environ.copy()
    backend = REPO_ROOT / "agribrain" / "backend"
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(backend), env.get("PYTHONPATH", "")) if item
    )
    env.pop("BENCHMARK_USE_TABLES", None)
    env.pop("BENCHMARK_WRITE_COMPAT", None)
    proc = subprocess.run(
        [
            sys.executable, str(runner),
            "--output-dir", str(output),
            "--single-run-table", str(table),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert {path.name for path in output.iterdir()} == {
        "development_benchmark_summary.json",
        "development_benchmark_significance.json",
    }
    for path in output.iterdir():
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["_meta"]["publication_evidence"] is False
        assert payload["_meta"]["scope"] == "single_seed_development_smoke"


def test_manual_smoke_workflow_has_no_schedule_or_publication_upload_names():
    workflow = (REPO_ROOT / ".github" / "workflows" / "development-smoke.yml").read_text(
        encoding="utf-8"
    )
    assert "schedule:" not in workflow
    assert "BENCHMARK_USE_TABLES" not in workflow
    assert "BENCHMARK_WRITE_COMPAT" not in workflow
    assert "development_benchmark_summary.json" in workflow
    assert "development_table1_summary.csv" in workflow
    assert "development_table2_ablation.csv" in workflow
    assert "publication_evidence=false" in workflow
    assert "mvp/simulation/results/benchmark_summary.json" not in workflow


def test_ci_skips_absent_current_evidence_without_archive_fallback():
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    assert "current-evidence-boundary:" in workflow
    assert 'manifest = Path("mvp/simulation/results/artifact_manifest.json")' in workflow
    assert 'out.write("has_current=false\\n")' in workflow
    assert "submission-evidence-" not in workflow
    assert "GitHub Release asset tied to the exact publication-source tag" in workflow
    assert 'out.write("recovery_receipt=\\n")' in workflow
    assert 'args+=(--recovery-receipt "$RECOVERY_RECEIPT")' in workflow
    assert "Archived/legacy outputs were not used" in workflow
    assert "if: steps.evidence.outputs.has_current == 'true'" in workflow
    assert "ALLOW_MISSING_BASELINE" not in workflow
    assert "superseded_pre_final" not in workflow
