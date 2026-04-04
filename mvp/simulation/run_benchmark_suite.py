#!/usr/bin/env python3
"""Multi-seed benchmark suite with bootstrap confidence intervals.

In stochastic mode (default), different seeds produce genuinely different
results, yielding meaningful CIs, p-values, and effect sizes.
In deterministic mode, all seeds produce identical results — statistics
are degenerate (std=0, p=1).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_SIM_DIR = Path(__file__).resolve().parent
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))

from stochastic import DETERMINISTIC_MODE
from generate_results import run_all, SCENARIOS


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _bootstrap_ci(values: List[float], n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(n_boot):
        boot = rng.choice(arr, size=len(arr), replace=True)
        samples.append(float(np.mean(boot)))
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return lo, hi


def _mean_diff_pvalue(a: List[float], b: List[float], n_perm: int = 4000) -> float:
    """Two-sided permutation p-value for difference in means."""
    if not a or not b:
        return 1.0
    x = np.array(a, dtype=float)
    y = np.array(b, dtype=float)
    observed = abs(float(np.mean(x) - np.mean(y)))
    pooled = np.concatenate([x, y])
    n_x = len(x)
    rng = np.random.default_rng(123)
    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        diff = abs(float(np.mean(perm[:n_x]) - np.mean(perm[n_x:])))
        if diff >= observed:
            ge += 1
    return float((ge + 1) / (n_perm + 1))


def _cohens_d(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    x = np.array(a, dtype=float)
    y = np.array(b, dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = ((nx - 1) * sx2 + (ny - 1) * sy2) / max(nx + ny - 2, 1)
    if pooled <= 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / np.sqrt(pooled))


def main() -> None:
    mode_label = "STOCHASTIC" if not DETERMINISTIC_MODE else "DETERMINISTIC"
    print(f"Benchmark suite — mode: {mode_label}")
    if DETERMINISTIC_MODE:
        print("  WARNING: deterministic mode — all seeds produce identical results.")
        print("  Set DETERMINISTIC_MODE=false for meaningful statistics.")

    seeds_env = os.environ.get("BENCHMARK_SEEDS", "").strip()
    if seeds_env:
        seeds = [int(s.strip()) for s in seeds_env.split(",") if s.strip()]
    else:
        seeds = [42, 1337, 2024, 7, 99]
    collected: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    use_tables = os.environ.get("BENCHMARK_USE_TABLES", "false").lower() == "true"

    if use_tables:
        t2_path = RESULTS_DIR / "table2_ablation.csv"
        if not t2_path.exists():
            raise FileNotFoundError(f"Missing {t2_path}; run generate_results.py first.")
        t2 = pd.read_csv(t2_path)
        for scenario in SCENARIOS:
            collected.setdefault(scenario, {})
            for mode in ("agribrain", "mcp_only", "pirag_only", "no_context"):
                row = t2[(t2["Scenario"] == scenario) & (t2["Variant"] == mode)]
                if row.empty:
                    continue
                rec = collected[scenario].setdefault(
                    mode, {"ari": [], "waste": [], "rle": [], "slca": []}
                )
                rec["ari"].append(float(row.iloc[0]["ARI"]))
                rec["waste"].append(float(row.iloc[0]["Waste"]))
                rec["rle"].append(float(row.iloc[0]["RLE"]))
                rec["slca"].append(float(row.iloc[0]["SLCA"]))
    else:
        for seed in seeds:
            run = run_all(seed=seed)
            for scenario in SCENARIOS:
                collected.setdefault(scenario, {})
                for mode in ("agribrain", "mcp_only", "pirag_only", "no_context"):
                    ep = run["results"][scenario][mode]
                    rec = collected[scenario].setdefault(
                        mode, {"ari": [], "waste": [], "rle": [], "slca": []}
                    )
                    rec["ari"].append(float(ep["ari"]))
                    rec["waste"].append(float(ep["waste"]))
                    rec["rle"].append(float(ep["rle"]))
                    rec["slca"].append(float(ep["slca"]))

    summary: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for scenario, modes in collected.items():
        summary[scenario] = {}
        for mode, metrics in modes.items():
            summary[scenario][mode] = {}
            for metric, vals in metrics.items():
                lo, hi = _bootstrap_ci(vals)
                summary[scenario][mode][metric] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "ci_low": lo,
                    "ci_high": hi,
                }

    significance: Dict[str, Dict[str, Dict[str, float]]] = {}
    for scenario, modes in collected.items():
        significance[scenario] = {}
        agri = modes.get("agribrain", {})
        for baseline in ("mcp_only", "pirag_only", "no_context"):
            base = modes.get(baseline, {})
            if not agri or not base:
                continue
            comp_key = f"agribrain_vs_{baseline}"
            significance[scenario][comp_key] = {}
            for metric in ("ari", "waste", "rle", "slca"):
                a_vals = agri.get(metric, [])
                b_vals = base.get(metric, [])
                significance[scenario][comp_key][metric] = {
                    "p_value": _mean_diff_pvalue(a_vals, b_vals),
                    "cohens_d": _cohens_d(a_vals, b_vals),
                    "mean_diff": float(np.mean(a_vals) - np.mean(b_vals)) if a_vals and b_vals else 0.0,
                }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "benchmark_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    sig_out = RESULTS_DIR / "benchmark_significance.json"
    sig_out.write_text(json.dumps(significance, indent=2), encoding="utf-8")
    print(f"Saved benchmark summary: {out}")
    print(f"Saved benchmark significance: {sig_out}")


if __name__ == "__main__":
    main()

