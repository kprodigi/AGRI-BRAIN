#!/usr/bin/env python3
"""Commit-bound deterministic development regression guard.

The checked-in snapshot is intentionally a schema-v2 ``pending`` marker: it
contains no historical numbers. A maintainer may initialize a run-scoped
baseline only by setting ``REGRESSION_GUARD_INIT=true`` after generating the
two deterministic tables from a clean source commit. The resulting snapshot
is bound to the literal experiment-protocol bytes, the exact Git commit, and
the complete table cell/metric contract.

This is a development reproducibility check, not publication evidence. The
stochastic publication pipeline has its own seed-, ledger-, manifest-, and
archive-level validators.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_SIM_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _SIM_DIR.parent.parent
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))
from stochastic import DETERMINISTIC_MODE


RESULTS_DIR = _SIM_DIR / "results"
T1 = RESULTS_DIR / "table1_summary.csv"
T2 = RESULTS_DIR / "table2_ablation.csv"
SNAPSHOT = _SIM_DIR / "baseline_snapshot.json"
PROTOCOL = _SIM_DIR / "experiment_protocol.json"

SCHEMA_VERSION = 2
PENDING_STATUS = "pending"
VALIDATED_STATUS = "validated"
SCOPE = "deterministic_development_regression_only"
PROTOCOL_RELPATH = "mvp/simulation/experiment_protocol.json"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")

SCENARIOS = (
    "heatwave", "overproduction", "cyber_outage",
    "adaptive_pricing", "baseline",
)
TABLE1_METHODS = (
    "static", "hybrid_rl", "no_pinn", "no_slca", "no_context",
    "mcp_only", "pirag_only", "agribrain",
)
TABLE2_VARIANTS = (
    "static", "hybrid_rl", "no_pinn", "no_slca", "no_context", "agribrain",
)
TABLE1_METRICS = ("ARI", "RLE", "Waste", "SLCA", "Carbon", "Equity")
TABLE2_METRICS = ("ARI", "RLE", "Waste", "SLCA")


def _fail(message: str) -> None:
    print(f"[regression-guard] FAILED: {message}")
    raise SystemExit(1)


def _canonical_key_contract() -> dict[str, dict[str, Any]]:
    """Return the complete, ordered table contract locked by the protocol."""

    return {
        "table1": {
            "key_columns": ["Scenario", "Method"],
            "metrics": list(TABLE1_METRICS),
            "scenarios": list(SCENARIOS),
            "methods": list(TABLE1_METHODS),
            "expected_cell_count": len(SCENARIOS) * len(TABLE1_METHODS),
        },
        "table2": {
            "key_columns": ["Scenario", "Variant"],
            "metrics": list(TABLE2_METRICS),
            "scenarios": list(SCENARIOS),
            "methods": list(TABLE2_VARIANTS),
            "expected_cell_count": len(SCENARIOS) * len(TABLE2_VARIANTS),
        },
    }


def _expected_keys(table_name: str) -> set[str]:
    contract = _canonical_key_contract()[table_name]
    return {
        f"{scenario}|{method}"
        for scenario in contract["scenarios"]
        for method in contract["methods"]
    }


def _protocol_sha256() -> str:
    if not PROTOCOL.is_file():
        _fail(f"locked experiment protocol missing: {PROTOCOL}")
    raw = PROTOCOL.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        protocol = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"locked experiment protocol is not valid JSON: {exc}")
    if protocol.get("scenarios") != list(SCENARIOS):
        _fail("experiment protocol scenario order/set differs from the guard contract")
    if protocol.get("primary_modes") != list(TABLE1_METHODS):
        _fail("experiment protocol primary-mode order/set differs from the guard contract")
    expected_secondary = [
        "agribrain_standard_rag",
        "agribrain_no_peer",
        "agribrain_sign_unconstrained",
    ]
    if protocol.get("secondary_ablation_modes") != expected_secondary:
        _fail("experiment protocol secondary-ablation order/set is not canonical")
    return digest


def _digest_table(df: pd.DataFrame, table_name: str) -> dict[str, dict[str, float]]:
    """Validate the exact cell set and return a six-decimal metric digest."""

    contract = _canonical_key_contract()[table_name]
    key_columns = list(contract["key_columns"])
    metrics = list(contract["metrics"])
    missing_columns = sorted(set(key_columns + metrics).difference(df.columns))
    if missing_columns:
        _fail(f"{table_name} missing required columns: {missing_columns}")

    out: dict[str, dict[str, float]] = {}
    duplicate_keys: list[str] = []
    for _, row in df.iterrows():
        if any(pd.isna(row[column]) for column in key_columns):
            _fail(f"{table_name} contains a null key value")
        key = "|".join(str(row[column]) for column in key_columns)
        if key in out:
            duplicate_keys.append(key)
            continue
        record: dict[str, float] = {}
        for metric in metrics:
            try:
                value = float(row[metric])
            except (TypeError, ValueError) as exc:
                _fail(f"{table_name}:{key}:{metric} is not numeric: {exc}")
            if not math.isfinite(value):
                _fail(f"{table_name}:{key}:{metric} is not finite")
            record[metric] = round(value, 6)
        out[key] = record

    if duplicate_keys:
        _fail(f"{table_name} contains duplicate cells: {sorted(set(duplicate_keys))}")
    expected = _expected_keys(table_name)
    actual = set(out)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        _fail(
            f"{table_name} cell set differs from the locked contract; "
            f"missing={missing}, extra={extra}"
        )
    if len(out) != int(contract["expected_cell_count"]):
        _fail(f"{table_name} cell count differs from the locked contract")
    return out


def _load_current() -> dict[str, dict[str, dict[str, float]]]:
    if not T1.is_file() or not T2.is_file():
        _fail(
            "deterministic tables are missing; expected table1_summary.csv "
            "and table2_ablation.csv under mvp/simulation/results"
        )
    return {
        "table1": _digest_table(pd.read_csv(T1), "table1"),
        "table2": _digest_table(pd.read_csv(T2), "table2"),
    }


def _run_git(*args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        _fail(f"cannot establish Git source identity: {exc}")
    return proc.stdout.strip()


def _current_source_commit() -> str:
    commit = _run_git("rev-parse", "HEAD")
    if not _HEX40.fullmatch(commit):
        _fail(f"Git HEAD is not a full lowercase commit SHA: {commit!r}")
    declared = os.environ.get("AGRIBRAIN_GIT_COMMIT", "").strip()
    if declared and declared != commit:
        _fail(
            "AGRIBRAIN_GIT_COMMIT does not exactly match checked-out HEAD: "
            f"{declared!r} != {commit!r}"
        )
    return commit


def _assert_clean_source_tree() -> None:
    """Reject uncommitted source while ignoring generated run outputs."""

    dirty = _run_git(
        "status", "--porcelain=v1", "--untracked-files=all", "--", ".",
        ":(exclude)mvp/simulation/results/**",
        ":(exclude)mvp/simulation/development_results/**",
        ":(exclude)mvp/simulation/baseline_snapshot.json",
    )
    if dirty:
        preview = "\n".join(dirty.splitlines()[:20])
        _fail(
            "source tree is dirty outside generated results/snapshot; "
            f"commit the source before initializing or checking a baseline:\n{preview}"
        )


def _load_snapshot() -> dict[str, Any]:
    if not SNAPSHOT.is_file():
        _fail(
            f"schema-v2 snapshot marker missing: {SNAPSHOT}; restore the "
            "checked-in pending marker before intentional initialization"
        )
    try:
        payload = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fail(f"snapshot is not valid JSON: {exc}")
    if not isinstance(payload, dict):
        _fail("snapshot root must be an object")
    return payload


def _validate_common_snapshot(payload: dict[str, Any], protocol_sha: str) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        _fail("legacy or unknown snapshot schema; schema_version must equal 2")
    if payload.get("scope") != SCOPE:
        _fail("snapshot scope is not deterministic development regression")
    if payload.get("publication_evidence") is not False:
        _fail("snapshot must explicitly declare publication_evidence=false")
    if payload.get("deterministic_mode") is not True:
        _fail("snapshot must explicitly declare deterministic_mode=true")
    expected_protocol = {"path": PROTOCOL_RELPATH, "sha256": protocol_sha}
    if payload.get("protocol") != expected_protocol:
        _fail("snapshot protocol path/hash does not exactly match the locked protocol")
    if payload.get("key_contract") != _canonical_key_contract():
        _fail("snapshot table key/metric contract is not the exact canonical contract")
    source = payload.get("source")
    if not isinstance(source, dict) or source.get("clean_source_required") is not True:
        _fail("snapshot does not require a clean source identity")


def _validate_snapshot_tables(tables: Any) -> None:
    if not isinstance(tables, dict) or set(tables) != {"table1", "table2"}:
        _fail("validated snapshot must contain exactly table1 and table2 digests")
    for table_name in ("table1", "table2"):
        records = tables.get(table_name)
        if not isinstance(records, dict) or set(records) != _expected_keys(table_name):
            _fail(f"snapshot {table_name} cell set is incomplete or contains extras")
        metrics = set(_canonical_key_contract()[table_name]["metrics"])
        for key, record in records.items():
            if not isinstance(record, dict) or set(record) != metrics:
                _fail(f"snapshot {table_name}:{key} metric set is not exact")
            for metric, raw in record.items():
                if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                    _fail(f"snapshot {table_name}:{key}:{metric} is not numeric")
                if not math.isfinite(float(raw)):
                    _fail(f"snapshot {table_name}:{key}:{metric} is not finite")


def _initialized_payload(
    current: dict[str, dict[str, dict[str, float]]],
    *,
    protocol_sha: str,
    source_commit: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": VALIDATED_STATUS,
        "scope": SCOPE,
        "publication_evidence": False,
        "deterministic_mode": True,
        "protocol": {"path": PROTOCOL_RELPATH, "sha256": protocol_sha},
        "source": {
            "commit": source_commit,
            "clean_source_required": True,
        },
        "key_contract": _canonical_key_contract(),
        "tables": current,
    }


def main() -> None:
    if not DETERMINISTIC_MODE:
        print(
            "[regression-guard] SKIPPED: stochastic mode; this development "
            "guard never validates publication evidence"
        )
        return

    if os.environ.get("ALLOW_MISSING_BASELINE", "").lower() in {"1", "true", "yes"}:
        _fail("ALLOW_MISSING_BASELINE is retired; missing/pending baselines fail closed")

    protocol_sha = _protocol_sha256()
    snapshot = _load_snapshot()
    _validate_common_snapshot(snapshot, protocol_sha)
    initialize = os.environ.get("REGRESSION_GUARD_INIT", "").lower() == "true"

    if initialize:
        if snapshot.get("status") != PENDING_STATUS:
            _fail("initialization refuses to overwrite an already validated snapshot")
        if snapshot.get("tables") is not None:
            _fail("pending snapshot unexpectedly contains numeric table values")
        if snapshot.get("source", {}).get("commit") is not None:
            _fail("pending snapshot unexpectedly contains a source commit")
        current = _load_current()
        commit = _current_source_commit()
        _assert_clean_source_tree()
        payload = _initialized_payload(
            current, protocol_sha=protocol_sha, source_commit=commit,
        )
        SNAPSHOT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(
            "[regression-guard] INITIALIZED: commit-bound deterministic "
            f"development baseline at {SNAPSHOT}"
        )
        print("[regression-guard] publication_evidence=false")
        return

    if snapshot.get("status") == PENDING_STATUS:
        if snapshot.get("tables") is not None or snapshot.get("source", {}).get("commit") is not None:
            _fail("pending snapshot must contain neither table values nor a source commit")
        _fail(
            "fresh deterministic baseline is pending; generate the exact tables "
            "from a clean commit and rerun once with REGRESSION_GUARD_INIT=true"
        )
    if snapshot.get("status") != VALIDATED_STATUS:
        _fail(f"unknown snapshot status: {snapshot.get('status')!r}")

    tables = snapshot.get("tables")
    _validate_snapshot_tables(tables)
    source = snapshot.get("source", {})
    snapshot_commit = source.get("commit")
    if not isinstance(snapshot_commit, str) or not _HEX40.fullmatch(snapshot_commit):
        _fail("validated snapshot source.commit is not a full lowercase Git SHA")
    current_commit = _current_source_commit()
    if snapshot_commit != current_commit:
        _fail(
            "snapshot source commit does not exactly match checked-out HEAD: "
            f"{snapshot_commit} != {current_commit}"
        )
    _assert_clean_source_tree()

    current = _load_current()
    if current != tables:
        differences: list[str] = []
        for table_name in ("table1", "table2"):
            for key in sorted(_expected_keys(table_name)):
                for metric in _canonical_key_contract()[table_name]["metrics"]:
                    before = tables[table_name][key][metric]
                    now = current[table_name][key][metric]
                    if before != now:
                        differences.append(
                            f"{table_name}:{key}:{metric} baseline={before!r} now={now!r}"
                        )
        print("[regression-guard] FAILED: exact deterministic metric drift detected")
        for difference in differences[:100]:
            print(" -", difference)
        raise SystemExit(1)

    print(
        "[regression-guard] PASS: exact protocol, commit, key sets, metrics, "
        "and deterministic values match"
    )
    print("[regression-guard] publication_evidence=false")


if __name__ == "__main__":
    main()
