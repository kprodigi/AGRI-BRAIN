#!/usr/bin/env python3
"""Multi-seed context-ablation aggregator with bootstrap CIs.

Ingests per-seed JSONs produced by run_single_seed.py (files named
``seed_<seed>.json`` under the seeds directory) and computes summary
statistics, confidence intervals, paired p-values, and effect sizes for
all eight simulation modes. This script never re-runs run_all() itself;
the HPC pipeline produces the seed JSONs in parallel via a SLURM job
array, and this aggregator assembles them afterwards.

In stochastic mode (default), different seeds produce genuinely different
results, yielding meaningful CIs, p-values, and effect sizes. In
deterministic mode, all seeds produce identical results and statistics
are degenerate (std=0, p=1).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

_SIM_DIR = Path(__file__).resolve().parent.parent
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))

from stochastic import DETERMINISTIC_MODE
from generate_results import SCENARIOS


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DEFAULT_SEEDS_DIR = DEFAULT_RESULTS_DIR / "benchmark_seeds"

MODES = (
    "agribrain", "mcp_only", "pirag_only", "no_context",
    "static", "hybrid_rl", "no_pinn", "no_slca",
)
METRICS = ("ari", "waste", "rle", "slca", "carbon", "equity")
BASELINE_COMPARISONS = ("mcp_only", "pirag_only", "no_context")


def _bootstrap_ci(values: List[float], n_boot: int = 10_000, alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap CI with 10,000 resamples, matching paper Section 3.13."""
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


def _mean_diff_pvalue(a: List[float], b: List[float], n_perm: int = 10_000) -> float:
    """Two-sided permutation p-value for difference in means, 10,000 permutations."""
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


def _parse_seeds_env() -> List[int]:
    raw = os.environ.get("BENCHMARK_SEEDS", "").strip()
    if raw:
        return [int(s.strip()) for s in raw.split(",") if s.strip()]
    return [42, 1337, 2024, 7, 99, 101, 202, 303, 404, 505,
            606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515]


def _discover_seed_files(seeds_dir: Path, seeds: List[int]) -> Dict[int, Path]:
    """Return the subset of requested seeds whose JSON files exist.

    Accepts either ``seeds_dir/seed_<n>.json`` (flat layout, matches the
    default run_single_seed.py output) or
    ``seeds_dir/seed_<n>/seed_<n>.json`` (per-task subdirectory layout,
    matches the SLURM job-array output).
    """
    found: Dict[int, Path] = {}
    for seed in seeds:
        flat = seeds_dir / f"seed_{seed}.json"
        nested = seeds_dir / f"seed_{seed}" / f"seed_{seed}.json"
        if flat.exists():
            found[seed] = flat
        elif nested.exists():
            found[seed] = nested
    return found


def _load_collected(seed_files: Dict[int, Path]) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    collected: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for seed, path in sorted(seed_files.items()):
        data = json.loads(path.read_text(encoding="utf-8"))
        for scenario in SCENARIOS:
            scenario_data = data.get(scenario, {})
            scenario_bucket = collected.setdefault(scenario, {})
            for mode in MODES:
                mode_data = scenario_data.get(mode)
                if mode_data is None:
                    continue
                rec = scenario_bucket.setdefault(
                    mode, {m: [] for m in METRICS}
                )
                for metric in METRICS:
                    value = mode_data.get(metric)
                    if value is None:
                        continue
                    rec[metric].append(float(value))
    return collected


def _build_summary(
    collected: Dict[str, Dict[str, Dict[str, List[float]]]],
    degenerate: bool,
) -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
    summary: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    for scenario, modes in collected.items():
        summary[scenario] = {}
        for mode, metrics in modes.items():
            summary[scenario][mode] = {}
            for metric, vals in metrics.items():
                entry: Dict[str, Any] = {
                    "mean": float(np.mean(vals)) if vals else 0.0,
                    "std": float(np.std(vals)) if vals else 0.0,
                    "n": len(vals),
                }
                if not degenerate and len(vals) >= 2:
                    lo, hi = _bootstrap_ci(vals)
                    entry["ci_low"] = lo
                    entry["ci_high"] = hi
                else:
                    entry["ci_low"] = None
                    entry["ci_high"] = None
                    entry["degenerate"] = True
                summary[scenario][mode][metric] = entry
    return summary


def _build_significance(
    collected: Dict[str, Dict[str, Dict[str, List[float]]]],
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    significance: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for scenario, modes in collected.items():
        significance[scenario] = {}
        agri = modes.get("agribrain", {})
        for baseline in BASELINE_COMPARISONS:
            base = modes.get(baseline, {})
            if not agri or not base:
                continue
            comp_key = f"agribrain_vs_{baseline}"
            significance[scenario][comp_key] = {}
            for metric in METRICS:
                a_vals = agri.get(metric, [])
                b_vals = base.get(metric, [])
                significance[scenario][comp_key][metric] = {
                    "p_value": _mean_diff_pvalue(a_vals, b_vals),
                    "cohens_d": _cohens_d(a_vals, b_vals),
                    "mean_diff": float(np.mean(a_vals) - np.mean(b_vals)) if a_vals and b_vals else 0.0,
                }
    return significance


def _load_from_table2(table2_path: Path) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """Legacy single-run fallback used when BENCHMARK_USE_TABLES=true."""
    t2 = pd.read_csv(table2_path)
    print("  WARNING: BENCHMARK_USE_TABLES=true loads single-run data from CSV.")
    print("  CIs and p-values will be degenerate (n=1). Use multi-seed runs for meaningful statistics.")
    collected: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for scenario in SCENARIOS:
        collected.setdefault(scenario, {})
        for mode in MODES:
            row = t2[(t2["Scenario"] == scenario) & (t2["Variant"] == mode)]
            if row.empty:
                continue
            rec = collected[scenario].setdefault(mode, {m: [] for m in METRICS})
            rec["ari"].append(float(row.iloc[0]["ARI"]))
            rec["waste"].append(float(row.iloc[0]["Waste"]))
            rec["rle"].append(float(row.iloc[0]["RLE"]))
            rec["slca"].append(float(row.iloc[0]["SLCA"]))
            if "Carbon" in row.columns:
                rec["carbon"].append(float(row.iloc[0]["Carbon"]))
            if "Equity" in row.columns:
                rec["equity"].append(float(row.iloc[0]["Equity"]))
    return collected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--seeds-dir",
        type=Path,
        default=DEFAULT_SEEDS_DIR,
        help="Directory containing per-seed JSONs from run_single_seed.py "
             "(default: mvp/simulation/results/benchmark_seeds).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory to write aggregator outputs "
             "(default: mvp/simulation/results).",
    )
    args = parser.parse_args()

    mode_label = "STOCHASTIC" if not DETERMINISTIC_MODE else "DETERMINISTIC"
    print(f"Benchmark suite aggregator, mode: {mode_label}")
    if DETERMINISTIC_MODE:
        print("  WARNING: deterministic mode, all seeds produce identical results.")
        print("  Set DETERMINISTIC_MODE=false for meaningful statistics.")

    seeds = _parse_seeds_env()
    use_tables = os.environ.get("BENCHMARK_USE_TABLES", "false").lower() == "true"

    if use_tables:
        t2_path = args.output_dir / "table2_ablation.csv"
        if not t2_path.exists():
            raise FileNotFoundError(f"Missing {t2_path}; run generate_results.py first.")
        collected = _load_from_table2(t2_path)
        loaded_seeds: List[int] = []
    else:
        seed_files = _discover_seed_files(args.seeds_dir, seeds)
        if not seed_files:
            raise SystemExit(
                f"BLOCK: no per-seed JSONs found under {args.seeds_dir}. "
                f"Run run_single_seed.py for each seed first, or set "
                f"BENCHMARK_USE_TABLES=true for the single-run fallback."
            )
        missing = sorted(set(seeds) - set(seed_files))
        if missing:
            print(f"  WARNING: missing per-seed JSONs for {missing}; aggregating the {len(seed_files)} present")
        collected = _load_collected(seed_files)
        loaded_seeds = sorted(seed_files)

    sample_counts = [len(v) for modes in collected.values() for m in modes.values() for v in m.values()]
    min_samples = min(sample_counts) if sample_counts else 0
    degenerate = min_samples < 2

    summary = _build_summary(collected, degenerate)

    if degenerate:
        print("  Skipping significance tests: degenerate sample size (n < 2)")
        significance: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    else:
        significance = _build_significance(collected)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "benchmark_context_summary.json"
    sig_out = args.output_dir / "benchmark_context_significance.json"
    payload = {
        "_meta": {
            "source": "run_benchmark_suite.py",
            "modes": list(MODES),
            "mode_label": mode_label,
            "seeds_requested": seeds,
            "seeds_loaded": loaded_seeds,
            "seeds_dir": str(args.seeds_dir),
            "use_tables": use_tables,
        },
        "summary": summary,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    sig_out.write_text(json.dumps(significance, indent=2), encoding="utf-8")
    print(f"Saved benchmark summary: {out}")
    print(f"Saved benchmark significance: {sig_out}")

    write_compat = os.environ.get("BENCHMARK_WRITE_COMPAT", "false").lower() == "true"
    if write_compat:
        compat_out = args.output_dir / "benchmark_summary.json"
        compat_sig_out = args.output_dir / "benchmark_significance.json"
        compat_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        compat_sig_out.write_text(json.dumps(significance, indent=2), encoding="utf-8")
        print(f"Saved benchmark summary (compat): {compat_out}")
        print(f"Saved benchmark significance (compat): {compat_sig_out}")


if __name__ == "__main__":
    main()
