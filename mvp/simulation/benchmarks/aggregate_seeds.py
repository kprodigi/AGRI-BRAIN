#!/usr/bin/env python3
"""Aggregate multi-seed benchmark results into canonical benchmark files.

Reads results/benchmark_seeds/seed_*.json and produces:
  - results/benchmark_summary.json   (means, stds, 95% CIs)
  - results/benchmark_significance.json  (paired p-values/effect sizes)

Usage:
    python aggregate_seeds.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

SEEDS = [42, 1337, 2024, 7, 99, 101, 202, 303, 404, 505,
         606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515]
SCENARIOS = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing", "baseline"]
MODES = ["agribrain", "mcp_only", "pirag_only", "no_context",
         "static", "hybrid_rl", "no_pinn", "no_slca"]
METRICS = ("ari", "waste", "rle", "slca", "carbon", "equity")

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
seed_dir = _SCRIPT_DIR / "results" / "benchmark_seeds"


def bootstrap_ci(vals, n_boot=1000, alpha=0.05):
    arr = np.array(vals, dtype=float)
    rng = np.random.default_rng(42)
    boots = [float(np.mean(rng.choice(arr, len(arr), replace=True))) for _ in range(n_boot)]
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def bootstrap_mean_diff_ci(a, b, n_boot=1000, alpha=0.05):
    """Bootstrap CI for paired mean difference E[a-b]."""
    x, y = np.array(a, dtype=float), np.array(b, dtype=float)
    if x.shape != y.shape or len(x) == 0:
        return 0.0, 0.0
    idx = np.arange(len(x))
    rng = np.random.default_rng(24)
    boots = []
    for _ in range(n_boot):
        sample_idx = rng.choice(idx, size=len(idx), replace=True)
        d = x[sample_idx] - y[sample_idx]
        boots.append(float(np.mean(d)))
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def paired_permutation_pvalue(a, b, n_perm=4000):
    x, y = np.array(a, dtype=float), np.array(b, dtype=float)
    if x.shape != y.shape or len(x) == 0:
        return 1.0
    d = x - y
    observed = abs(float(np.mean(d)))
    rng = np.random.default_rng(123)
    ge = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(d))
        diff = abs(float(np.mean(d * signs)))
        if diff >= observed:
            ge += 1
    return float((ge + 1) / (n_perm + 1))


def cohens_dz(a, b):
    x, y = np.array(a, dtype=float), np.array(b, dtype=float)
    if x.shape != y.shape or len(x) < 2:
        return 0.0
    d = x - y
    sd = np.std(d, ddof=1)
    return float(np.mean(d) / sd) if sd > 0 else 0.0


def benjamini_hochberg(p_values: dict[str, float]) -> dict[str, float]:
    """Return BH-FDR adjusted p-values, preserving input keys."""
    keys = list(p_values.keys())
    m = len(keys)
    if m == 0:
        return {}
    ordered = sorted(((k, float(p_values[k])) for k in keys), key=lambda kv: kv[1])
    adjusted = {}
    prev = 1.0
    for rank_rev, (k, p) in enumerate(reversed(ordered), start=1):
        i = m - rank_rev + 1
        q = min(prev, (p * m) / max(i, 1))
        adjusted[k] = float(min(max(q, 0.0), 1.0))
        prev = q
    return adjusted


def main():
    seed_csv = os.environ.get(
        "BENCHMARK_SEEDS",
        "42,1337,2024,7,99,101,202,303,404,505,606,707,808,909,1010,1111,1212,1313,1414,1515",
    ).strip()
    seeds = []
    for raw in seed_csv.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            seeds.append(int(raw))
        except ValueError:
            continue
    if not seeds:
        seeds = SEEDS
    print(f"Configured seed count: {len(seeds)}")

    # Load seed results
    all_data = {}
    for seed in seeds:
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

    # Build significance with paired tests and FDR correction.
    significance = {}
    for sc in SCENARIOS:
        significance[sc] = {}
        for baseline in ("mcp_only", "pirag_only", "no_context", "hybrid_rl", "static"):
            comp = {}
            pvals = {}
            for met in METRICS:
                a = [all_data[s][sc]["agribrain"][met] for s in all_data]
                b = [all_data[s][sc][baseline][met] for s in all_data]
                p = paired_permutation_pvalue(a, b)
                d = cohens_dz(a, b)
                key = f"{sc}:{baseline}:{met}"
                pvals[key] = p
                lo_diff, hi_diff = bootstrap_mean_diff_ci(a, b)
                mean_diff = float(np.mean(a) - np.mean(b))
                comp[met] = {
                    "p_value": p,
                    "cohens_d": d,
                    "cohens_dz": d,
                    "mean_diff": mean_diff,
                    "mean_diff_ci_low": lo_diff,
                    "mean_diff_ci_high": hi_diff,
                }
            p_adj = benjamini_hochberg(pvals)
            for met in METRICS:
                key = f"{sc}:{baseline}:{met}"
                comp[met]["p_value_adj"] = float(p_adj.get(key, comp[met]["p_value"]))
            significance[sc][f"agribrain_vs_{baseline}"] = comp

    # Save
    out_dir = _SCRIPT_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "benchmark_significance.json").write_text(json.dumps(significance, indent=2))
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
            print(f"  {sc} {comp_name}: p={rec['p_value']:.4f} p_adj={rec['p_value_adj']:.4f} dz={rec['cohens_dz']:+.3f}")


if __name__ == "__main__":
    main()
