#!/usr/bin/env python3
"""Validate publication result tables without conditioning on outcomes.

The validator enforces artifact presence, balanced-table structure, numeric
finiteness, and construct-level bounds. It deliberately does not require a
preferred method ordering, a minimum effect size, or a hand-selected numerical
range. H1--H3 are evaluated only by the inferential outputs produced by the
aggregation and stress pipelines.

Set ``STRICT_VALIDATION=0`` only for local exploratory work where missing or
invalid tables should be reported without a non-zero exit status.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
TABLE1 = RESULTS_DIR / "table1_summary.csv"
TABLE2 = RESULTS_DIR / "table2_ablation.csv"
STRICT = os.environ.get("STRICT_VALIDATION", "1") == "1"

SCENARIOS = {
    "heatwave", "overproduction", "cyber_outage",
    "adaptive_pricing", "baseline",
}
CORE_METHODS = {
    "static", "hybrid_rl", "no_pinn", "no_slca", "no_context",
    "mcp_only", "pirag_only", "agribrain",
}
ABLATION_METHODS = {
    "static", "hybrid_rl", "no_pinn", "no_slca", "no_context", "agribrain",
}

UNIT_INTERVAL_COLUMNS = {
    "ARI", "Waste", "SLCA", "RLE", "Equity",
    "ConstraintViolationRate", "OperatingEnvelopeViolationRate",
    "DownstreamViolationRate", "ContainedViolationRate",
}
NONNEGATIVE_COLUMNS = {"Carbon", "DecisionLatencyMs"}


def _write_report(errors: list[str], missing: list[str]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "strict": STRICT,
        "validation_scope": (
            "artifact presence, balanced structure, numeric finiteness, "
            "and construct-level bounds; no preferred outcome ordering"
        ),
        "n_errors": len(errors),
        "errors": errors,
        "missing_artifacts": missing,
    }
    (RESULTS_DIR / "validation_report.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _fail_or_report(errors: list[str], missing: list[str] | None = None) -> None:
    missing = missing or []
    _write_report(errors, missing)
    if errors:
        print(f"VALIDATION {'FAILED' if STRICT else 'REPORTED'}: {len(errors)} issue(s)")
        for err in errors:
            print(f"  - {err}")
        if STRICT:
            raise SystemExit(1)
    else:
        print("VALIDATION PASSED")


def _require_columns(df: pd.DataFrame, required: set[str], label: str,
                     errors: list[str]) -> None:
    absent = sorted(required.difference(df.columns))
    if absent:
        errors.append(f"{label} missing columns: {', '.join(absent)}")


def _validate_balanced_panel(
    df: pd.DataFrame,
    scenario_col: str,
    method_col: str,
    expected_methods: set[str],
    label: str,
    errors: list[str],
) -> None:
    if scenario_col not in df or method_col not in df:
        return
    duplicate = df.duplicated([scenario_col, method_col], keep=False)
    if duplicate.any():
        cells = sorted({
            f"{row[scenario_col]}/{row[method_col]}"
            for _, row in df.loc[duplicate, [scenario_col, method_col]].iterrows()
        })
        errors.append(f"{label} duplicate cells: {', '.join(cells)}")

    observed_scenarios = set(df[scenario_col].dropna().astype(str))
    if observed_scenarios != SCENARIOS:
        errors.append(
            f"{label} scenario set mismatch: expected {sorted(SCENARIOS)}, "
            f"observed {sorted(observed_scenarios)}"
        )

    observed_methods = set(df[method_col].dropna().astype(str))
    if observed_methods != expected_methods:
        errors.append(
            f"{label} method set mismatch: expected {sorted(expected_methods)}, "
            f"observed {sorted(observed_methods)}"
        )

    expected_cells = {(s, m) for s in SCENARIOS for m in expected_methods}
    observed_cells = set(zip(df[scenario_col].astype(str), df[method_col].astype(str)))
    missing_cells = sorted(expected_cells.difference(observed_cells))
    extra_cells = sorted(observed_cells.difference(expected_cells))
    if missing_cells:
        errors.append(
            f"{label} missing {len(missing_cells)} cells: "
            + ", ".join(f"{s}/{m}" for s, m in missing_cells[:12])
        )
    if extra_cells:
        errors.append(
            f"{label} has {len(extra_cells)} unexpected cells: "
            + ", ".join(f"{s}/{m}" for s, m in extra_cells[:12])
        )


def _validate_numeric_columns(df: pd.DataFrame, label: str,
                              errors: list[str]) -> None:
    for column in sorted((UNIT_INTERVAL_COLUMNS | NONNEGATIVE_COLUMNS).intersection(df.columns)):
        values = pd.to_numeric(df[column], errors="coerce")
        bad_numeric = values.isna() | ~values.map(math.isfinite)
        if bad_numeric.any():
            errors.append(
                f"{label}.{column} contains {int(bad_numeric.sum())} "
                "missing or non-finite value(s)"
            )
            continue
        if column in UNIT_INTERVAL_COLUMNS:
            bad_bounds = (values < 0.0) | (values > 1.0)
            if bad_bounds.any():
                errors.append(
                    f"{label}.{column} contains {int(bad_bounds.sum())} "
                    "value(s) outside [0, 1]"
                )
        else:
            bad_bounds = values < 0.0
            if bad_bounds.any():
                errors.append(
                    f"{label}.{column} contains {int(bad_bounds.sum())} negative value(s)"
                )


def _validate_summary_json(errors: list[str]) -> None:
    path = RESULTS_DIR / "benchmark_summary.json"
    if not path.exists():
        errors.append("benchmark_summary.json missing")
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"benchmark_summary.json invalid JSON: {exc}")
        return
    summary = payload.get("summary", payload) if isinstance(payload, dict) else None
    if not isinstance(summary, dict):
        errors.append("benchmark_summary.json does not contain a summary object")
        return
    if set(summary) != SCENARIOS:
        errors.append(
            "benchmark_summary.json scenario set mismatch: "
            f"observed {sorted(summary) if isinstance(summary, dict) else 'invalid'}"
        )
        return
    for scenario in sorted(SCENARIOS):
        methods = summary.get(scenario)
        if not isinstance(methods, dict):
            errors.append(f"benchmark_summary.json {scenario} is not an object")
            continue
        missing = CORE_METHODS.difference(methods)
        if missing:
            errors.append(
                f"benchmark_summary.json {scenario} missing core methods: "
                + ", ".join(sorted(missing))
            )
        for method in CORE_METHODS.intersection(methods):
            metrics = methods[method]
            if not isinstance(metrics, dict):
                errors.append(f"benchmark_summary.json {scenario}/{method} is not an object")
                continue
            for metric in ("ari", "waste", "slca", "rle", "equity", "carbon"):
                record = metrics.get(metric)
                if not isinstance(record, dict) or "mean" not in record:
                    errors.append(
                        f"benchmark_summary.json {scenario}/{method}/{metric}.mean missing"
                    )
                    continue
                try:
                    value = float(record["mean"])
                except (TypeError, ValueError):
                    errors.append(
                        f"benchmark_summary.json {scenario}/{method}/{metric}.mean non-numeric"
                    )
                    continue
                if not math.isfinite(value):
                    errors.append(
                        f"benchmark_summary.json {scenario}/{method}/{metric}.mean non-finite"
                    )
                if metric in {"ari", "waste", "slca", "rle", "equity"} and not (0 <= value <= 1):
                    errors.append(
                        f"benchmark_summary.json {scenario}/{method}/{metric}.mean outside [0,1]"
                    )
                if metric == "carbon" and value < 0:
                    errors.append(
                        f"benchmark_summary.json {scenario}/{method}/carbon.mean negative"
                    )


def main() -> None:
    missing = [str(path) for path in (TABLE1, TABLE2) if not path.exists()]
    if missing:
        _fail_or_report([f"missing artifact: {path}" for path in missing], missing)
        return

    errors: list[str] = []
    try:
        table1 = pd.read_csv(TABLE1)
        table2 = pd.read_csv(TABLE2)
    except Exception as exc:
        _fail_or_report([f"failed to read result tables: {exc}"])
        return

    required_metrics = {"ARI", "Waste", "SLCA", "RLE", "Carbon", "Equity"}
    _require_columns(table1, {"Scenario", "Method"} | required_metrics,
                     "table1_summary.csv", errors)
    _require_columns(table2, {"Scenario", "Variant"} | required_metrics,
                     "table2_ablation.csv", errors)
    _validate_balanced_panel(
        table1, "Scenario", "Method", CORE_METHODS, "table1_summary.csv", errors
    )
    _validate_balanced_panel(
        table2, "Scenario", "Variant", ABLATION_METHODS, "table2_ablation.csv", errors
    )
    _validate_numeric_columns(table1, "table1_summary.csv", errors)
    _validate_numeric_columns(table2, "table2_ablation.csv", errors)
    _validate_summary_json(errors)
    _fail_or_report(errors)


if __name__ == "__main__":
    main()
