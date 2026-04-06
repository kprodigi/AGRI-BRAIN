#!/usr/bin/env python3
"""Validate required publication artifacts and schema fields.

Fails fast when key reproducibility/statistics fields are missing.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    raise SystemExit(1)


def _load_json(path: Path) -> Any:
    if not path.exists():
        _fail(f"Missing required file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _fail(f"Invalid JSON in {path}: {exc}")


def _validate_significance() -> None:
    path = RESULTS_DIR / "benchmark_significance.json"
    data = _load_json(path)
    required = {
        "p_value",
        "p_value_adj",
        "cohens_d",
        "cohens_dz",
        "mean_diff",
        "mean_diff_ci_low",
        "mean_diff_ci_high",
    }
    missing = []
    for scenario, comps in data.items():
        if not isinstance(comps, dict):
            missing.append(f"{scenario} (not an object)")
            continue
        for comp, metrics in comps.items():
            if not isinstance(metrics, dict):
                missing.append(f"{scenario}.{comp} (not an object)")
                continue
            for metric, rec in metrics.items():
                if not isinstance(rec, dict):
                    missing.append(f"{scenario}.{comp}.{metric} (not an object)")
                    continue
                absent = sorted(required.difference(rec.keys()))
                if absent:
                    missing.append(f"{scenario}.{comp}.{metric}: missing {', '.join(absent)}")
    if missing:
        _fail("benchmark_significance schema violations:\n  - " + "\n  - ".join(missing[:20]))
    print("[PASS] benchmark_significance.json fields")


def _validate_stress_passfail() -> None:
    path = RESULTS_DIR / "stress_passfail.csv"
    if not path.exists():
        _fail(f"Missing required file: {path}")
    required_cols = {
        "Scenario",
        "Stressor",
        "Method",
        "Pass",
        "ari_delta",
        "waste_delta",
        "slca_delta",
        "rle_delta",
        "carbon_delta",
        "equity_delta",
        "constraint_violation_delta",
        "latency_ms_delta",
        "ARI_Base",
        "ARI_Stressed",
        "Waste_Base",
        "Waste_Stressed",
        "SLCA_Base",
        "SLCA_Stressed",
        "Threshold_ARI",
        "Threshold_Waste",
        "Threshold_SLCA",
        "Threshold_RLE",
        "Threshold_Carbon",
        "Threshold_Equity",
        "Threshold_CVR",
        "Threshold_LatencyMs",
    }
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        missing_cols = sorted(required_cols.difference(cols))
        if missing_cols:
            _fail(f"stress_passfail.csv missing columns: {', '.join(missing_cols)}")
        row_count = sum(1 for _ in reader)
        if row_count == 0:
            _fail("stress_passfail.csv has no rows")
    print("[PASS] stress_passfail.csv schema")


def _validate_manifest() -> None:
    path = RESULTS_DIR / "artifact_manifest.json"
    data = _load_json(path)
    commit = str(data.get("git_commit", "")).strip()
    if not commit or commit == "unknown":
        _fail("artifact_manifest.json missing concrete git_commit")
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        _fail("artifact_manifest.json has empty artifacts list")
    for i, rec in enumerate(artifacts[:10]):
        if not isinstance(rec, dict):
            _fail(f"artifact_manifest.json artifacts[{i}] is not an object")
        for key in ("file", "sha256", "bytes"):
            if key not in rec:
                _fail(f"artifact_manifest.json artifacts[{i}] missing {key}")
    print("[PASS] artifact_manifest.json commit + hashes")


def _validate_external_validity() -> None:
    for name in (
        "external_validity_summary.json",
        "external_validity_summary.csv",
        "external_validity_deltas.csv",
    ):
        path = RESULTS_DIR / name
        if not path.exists():
            _fail(f"Missing required file: {path}")
    print("[PASS] external validity outputs")


def main() -> None:
    if not RESULTS_DIR.exists():
        _fail(f"Missing results directory: {RESULTS_DIR}")
    _validate_significance()
    _validate_stress_passfail()
    _validate_manifest()
    _validate_external_validity()
    print("[PASS] publication artifact validation complete")


if __name__ == "__main__":
    main()
