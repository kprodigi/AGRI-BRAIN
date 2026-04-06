#!/usr/bin/env python3
"""Aggregate multi-seed benchmark results into summary and significance files.

Reads results/benchmark_seeds/seed_*.json and produces:
  - results/benchmark_summary.json   (means, stds, 95% CIs)
  - results/benchmark_significance.json  (p-values, Cohen's d)

Usage:
    python aggregate_seeds.py
"""
import json
import sys
from pathlib import Path

import numpy as np

SEEDS = [42, 1337, 2024, 7, 99]
SCENARIOS = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing", "baseline"]
MODES = ["agribrain", "mcp_only", "pirag_only", "no_context",
         "static", "hybrid_rl", "no_pinn", "no_slca"]
METRICS = ("ari", "waste", "rle", "slca", "carbon", "equity")

seed_dir = Path("results/benchmark_seeds")


def bootstrap_ci(vals, n_boot=1000, alpha=0.05):
    arr = np.array(vals, dtype=float)
    rng = np.random.default_rng(42)
    boots = [float(np.mean(rng.choice(arr, len(arr), replace=True))) for _ in range(n_boot)]
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def permutation_pvalue(a, b, n_perm=4000):
    x, y = np.array(a, dtype=float), np.array(b, dtype=float)
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


def cohens_d(a, b):
    x, y = np.array(a, dtype=float), np.array(b, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return 0.0
    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = ((len(x) - 1) * sx2 + (len(y) - 1) * sy2) / max(len(x) + len(y) - 2, 1)
    return float((np.mean(x) - np.mean(y)) / np.sqrt(pooled)) if pooled > 0 else 0.0


def main():
    # Load seed results
    all_data = {}
    for seed in SEEDS:
        f = seed_dir / f"seed_{seed}.json"
        if f.exists():
            all_data[seed] = json.loads(f.read_text())
            print(f"Loaded seed {seed}")
        else:
            print(f"WARNING: {f} not found, skipping")

    if len(all_data) < 2:
        print(f"ERROR: Only {len(all_data)} seed(s) found, need at least 2")
        sys.exit(1)

    print(f"Aggregating {len(all_data)} seeds...")

    # Build summary
    summary = {}
    for sc in SCENARIOS:
        summary[sc] = {}
        for mode in MODES:
            summary[sc][mode] = {}
            for met in METRICS:
                vals = [all_data[s][sc][mode][met] for s in all_data if mode in all_data[s].get(sc, {})]
                if not vals:
                    continue
                lo, hi = bootstrap_ci(vals)
                summary[sc][mode][met] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_seeds": len(vals),
                }

    # Build significance
    significance = {}
    for sc in SCENARIOS:
        significance[sc] = {}
        for baseline in ("mcp_only", "pirag_only", "no_context", "hybrid_rl", "static"):
            comp = {}
            for met in METRICS:
                a = [all_data[s][sc]["agribrain"][met] for s in all_data]
                b = [all_data[s][sc][baseline][met] for s in all_data]
                comp[met] = {
                    "p_value": permutation_pvalue(a, b),
                    "cohens_d": cohens_d(a, b),
                    "mean_diff": float(np.mean(a) - np.mean(b)),
                }
            significance[sc][f"agribrain_vs_{baseline}"] = comp

    # Save
    Path("results").mkdir(exist_ok=True)
    Path("results/benchmark_summary.json").write_text(json.dumps(summary, indent=2))
    Path("results/benchmark_significance.json").write_text(json.dumps(significance, indent=2))
    print("Saved benchmark_summary.json")
    print("Saved benchmark_significance.json")

    # Print key results
    print()
    for sc in SCENARIOS:
        a = summary[sc]["agribrain"]["ari"]
        print(f"  {sc}: ARI mean={a['mean']:.4f} CI=[{a['ci_low']:.4f}, {a['ci_high']:.4f}] std={a['std']:.6f}")

    print()
    for sc in SCENARIOS:
        for comp_name in ("agribrain_vs_no_context", "agribrain_vs_hybrid_rl"):
            rec = significance[sc][comp_name]["ari"]
            print(f"  {sc} {comp_name}: p={rec['p_value']:.4f} d={rec['cohens_d']:+.3f}")


if __name__ == "__main__":
    main()
