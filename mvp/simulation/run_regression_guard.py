#!/usr/bin/env python3
"""Strict compatibility regression guard for simulation tables.

Skipped automatically when DETERMINISTIC_MODE=false (stochastic), because
exact-value regression checks are meaningless under seeded perturbation noise.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

_SIM_DIR = Path(__file__).resolve().parent
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))
from stochastic import DETERMINISTIC_MODE


RESULTS_DIR = Path(__file__).resolve().parent / "results"
T1 = RESULTS_DIR / "table1_summary.csv"
T2 = RESULTS_DIR / "table2_ablation.csv"
# Keep snapshot outside generated results so it can be versioned.
SNAPSHOT = Path(__file__).resolve().parent / "baseline_snapshot.json"


def _digest_table(df: pd.DataFrame, keys: list[str], metrics: list[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        k = "|".join(str(row[c]) for c in keys)
        out[k] = {m: round(float(row[m]), 6) for m in metrics}
    return out


def _load_current() -> Dict[str, Dict[str, Dict[str, float]]]:
    if not T1.exists() or not T2.exists():
        raise FileNotFoundError("Expected table1_summary.csv and table2_ablation.csv in results/")
    t1 = pd.read_csv(T1)
    t2 = pd.read_csv(T2)
    return {
        "table1": _digest_table(t1, ["Scenario", "Method"], ["ARI", "RLE", "Waste", "SLCA", "Carbon", "Equity"]),
        "table2": _digest_table(t2, ["Scenario", "Variant"], ["ARI", "RLE", "Waste", "SLCA"]),
    }


def main() -> None:
    if not DETERMINISTIC_MODE:
        print("[regression-guard] SKIPPED: stochastic mode (exact regression checks not applicable)")
        return

    current = _load_current()
    if not SNAPSHOT.exists():
        if os.environ.get("REGRESSION_GUARD_INIT", "false").lower() == "true":
            SNAPSHOT.write_text(json.dumps(current, indent=2), encoding="utf-8")
            print(f"[regression-guard] Created baseline snapshot: {SNAPSHOT}")
            return
        print("[regression-guard] FAILED: baseline snapshot missing")
        print(f"  Expected snapshot: {SNAPSHOT}")
        print("  To initialize intentionally, run with REGRESSION_GUARD_INIT=true")
        raise SystemExit(1)

    baseline = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    diffs = []
    for table_name in ("table1", "table2"):
        for key, vals in current[table_name].items():
            base_vals = baseline.get(table_name, {}).get(key)
            if base_vals is None:
                diffs.append(f"{table_name}:{key} missing in baseline")
                continue
            for metric, val in vals.items():
                base_val = float(base_vals.get(metric, val))
                if abs(val - base_val) > 1e-3:
                    diffs.append(f"{table_name}:{key}:{metric} base={base_val:.6f} now={val:.6f}")

    if diffs:
        print("[regression-guard] FAILED: metric drift detected")
        for d in diffs[:100]:
            print(" -", d)
        raise SystemExit(1)

    print("[regression-guard] PASS: no significant drift detected")


if __name__ == "__main__":
    main()

