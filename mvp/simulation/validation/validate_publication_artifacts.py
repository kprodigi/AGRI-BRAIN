#!/usr/bin/env python3
"""Validate required publication artifacts and schema fields.

Fails fast when key reproducibility/statistics fields are missing.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


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
    # 2026-04 schema: per-(scenario, comparison, metric) records are
    # nested under top-level "significance" alongside "_meta" and
    # "primary_h1_holm_adjusted". Unwrap so the traversal works on
    # both wrapped and legacy-flat formats.
    if isinstance(data, dict) and isinstance(data.get("significance"), dict):
        data = data["significance"]
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
    # Comparison-level metadata fields that are NOT per-metric records
    # (so the inner schema check should skip them).
    _COMP_META_KEYS = {
        "is_paired_design", "test_type", "effect_size_primary",
        "_meta",
    }
    for scenario, comps in data.items():
        if not isinstance(comps, dict):
            missing.append(f"{scenario} (not an object)")
            continue
        for comp, metrics in comps.items():
            if not isinstance(metrics, dict):
                missing.append(f"{scenario}.{comp} (not an object)")
                continue
            for metric, rec in metrics.items():
                if metric in _COMP_META_KEYS:
                    continue
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


def _validate_temporal_stability() -> None:
    """Within-trace temporal stability outputs (legacy filenames retained)."""
    for name in (
        "external_validity_summary.json",
        "external_validity_summary.csv",
        "external_validity_deltas.csv",
    ):
        path = RESULTS_DIR / name
        if not path.exists():
            _fail(f"Missing required file: {path}")
    print("[PASS] within-trace temporal stability outputs")


def _validate_threshold_assertions() -> None:
    """Per-claim threshold assertions per docs/CLAIMS_TO_EVIDENCE.md.

    The previous schema check only verified field *presence*; a run
    that produced all-null effects passed. This adds explicit numeric
    thresholds for the headline claims so a clearly-broken run is
    caught at the validator gate.

    Thresholds are intentionally generous (sanity bounds, not the
    hypothesis-confirmation gates that the inferential pipeline keeps
    in the aggregator's CI/p-value path); the goal is to fail when
    the numbers are nonsensical, not when they merely fail to confirm
    the hypothesis.

    Tolerant of n=1 table-fallback aggregator outputs: when p_value /
    CIs are absent because the degenerate-sample-size aggregator
    skipped inferential tests, we accept the record as long as the
    descriptive fields (mean_diff, cohens_d) are present and finite.
    """
    sig = _load_json(RESULTS_DIR / "benchmark_significance.json")
    if isinstance(sig, dict) and isinstance(sig.get("significance"), dict):
        sig = sig["significance"]
    summary = _load_json(RESULTS_DIR / "benchmark_summary.json")
    if isinstance(summary, dict) and "summary" in summary and isinstance(summary["summary"], dict):
        summary = summary["summary"]

    failures = []
    for sc in ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing", "baseline"]:
        # Sanity bound on agribrain ARI mean.
        try:
            ari_mean = float(summary[sc]["agribrain"]["ari"]["mean"])
        except (KeyError, TypeError, ValueError):
            failures.append(f"{sc}/agribrain ari mean missing or non-numeric")
            continue
        if not (0.0 <= ari_mean <= 1.0):
            failures.append(f"{sc}/agribrain ARI mean {ari_mean} out of [0,1]")
        if ari_mean < 0.05:
            failures.append(f"{sc}/agribrain ARI mean {ari_mean} suspiciously low (<0.05)")

        rec = sig.get(sc, {}).get("agribrain_vs_no_context", {}).get("ari")
        if not isinstance(rec, dict):
            failures.append(f"{sc}/agribrain_vs_no_context/ari record missing")
            continue

        # mean_diff must be present and finite.
        try:
            md = float(rec["mean_diff"])
        except (KeyError, TypeError, ValueError):
            failures.append(f"{sc}/agribrain_vs_no_context ari mean_diff missing or non-numeric")
            continue
        import math as _m
        if not _m.isfinite(md):
            failures.append(f"{sc}/agribrain_vs_no_context ari mean_diff non-finite ({md})")

        # p_value: required in [0,1] when present; absence is tolerated
        # for the n=1 table-fallback aggregator (degenerate sample
        # size) which skips significance tests.
        if "p_value" in rec and rec["p_value"] is not None:
            try:
                p = float(rec["p_value"])
                if not (0.0 <= p <= 1.0):
                    failures.append(f"{sc}/agribrain_vs_no_context ari p_value {p} out of [0,1]")
            except (TypeError, ValueError):
                failures.append(f"{sc}/agribrain_vs_no_context ari p_value non-numeric")

        # Effect-size CI bracketed correctly (ci_low <= ci_high).
        lo = rec.get("effect_size_ci_low")
        hi = rec.get("effect_size_ci_high")
        if lo is not None and hi is not None and float(lo) > float(hi):
            failures.append(
                f"{sc}/agribrain_vs_no_context ari effect-size CI inverted: "
                f"low={lo} > high={hi}"
            )

    if failures:
        _fail("threshold assertions failed:\n  - " + "\n  - ".join(failures[:20]))
    print("[PASS] per-claim threshold assertions")


def main() -> None:
    if not RESULTS_DIR.exists():
        _fail(f"Missing results directory: {RESULTS_DIR}")
    _validate_significance()
    _validate_stress_passfail()
    _validate_manifest()
    _validate_temporal_stability()
    _validate_threshold_assertions()
    print("[PASS] publication artifact validation complete")


if __name__ == "__main__":
    main()
