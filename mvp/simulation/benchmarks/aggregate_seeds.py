#!/usr/bin/env python3
"""Aggregate multi-seed benchmark results into canonical benchmark files.

Reads ``results/benchmark_seeds/seed_*.json`` and writes

- ``results/benchmark_summary.json``    , per-(scenario, mode, metric) means,
  standard deviations, and 95 % bootstrap CIs.
- ``results/benchmark_significance.json``, paired permutation p-values, effect
  sizes, and multiplicity-adjusted p-values using two correction families:

  1. Holm-Bonferroni across the five scenario-level primary H1 tests
     (agribrain vs no_context, metric = ARI, one test per scenario). This
     matches the pre-registered multiplicity control declared in the paper.
     Reported as ``p_value_adj_holm`` on the five primary entries and as the
     canonical ``p_value_adj`` on the same entries.
  2. Benjamini-Hochberg FDR within each scenario across all (baseline, metric)
     secondary comparisons. Reported as ``p_value_adj_bh`` on every entry and
     as ``p_value_adj`` on every non-primary entry.

Usage::

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
MODES = ["agribrain", "mcp_only", "pirag_only", "no_context", "no_yield",
         "static", "hybrid_rl", "no_pinn", "no_slca"]
METRICS = ("ari", "waste", "rle", "slca", "carbon", "equity")
BASELINES = ("mcp_only", "pirag_only", "no_context", "no_yield",
             "hybrid_rl", "static")

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
seed_dir = _SCRIPT_DIR / "results" / "benchmark_seeds"


def bootstrap_ci(vals, n_boot=10_000, alpha=0.05):
    """Percentile bootstrap CI with 10,000 resamples, matching paper Section 3.13."""
    arr = np.array(vals, dtype=float)
    rng = np.random.default_rng(42)
    boots = [float(np.mean(rng.choice(arr, len(arr), replace=True))) for _ in range(n_boot)]
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def bootstrap_mean_diff_ci(a, b, n_boot=10_000, alpha=0.05):
    """Paired bootstrap CI for mean(a - b) with 10,000 resamples."""
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


def paired_permutation_pvalue(a, b, n_perm=10_000):
    """Paired sign-flip permutation p-value with 10,000 permutations, matching
    paper Section 3.13."""
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
    """Benjamini-Hochberg step-up FDR correction.

    Controls the false discovery rate at alpha. Preserves input keys.
    Returns each key's BH-adjusted p-value. Order-independent in the output.
    """
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


def holm_bonferroni(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down FWER correction.

    Controls the family-wise error rate. Stricter than BH-FDR. Preserves
    input keys. Matches paper Section 3.13's declared multiplicity control
    for the primary H1 family (the five scenario-level agribrain vs
    no_context comparisons on ARI).
    """
    keys = list(p_values.keys())
    m = len(keys)
    if m == 0:
        return {}
    ordered = sorted(((k, float(p_values[k])) for k in keys), key=lambda kv: kv[1])
    adjusted = {}
    running = 0.0
    for rank_idx, (k, p) in enumerate(ordered):
        # Holm step-down: p_(i) * (m - i + 1), then monotone non-decreasing
        q = min(1.0, p * (m - rank_idx))
        running = max(running, q)
        adjusted[k] = float(running)
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
                vals = [all_data[s][sc][mode][met] for s in all_data
                        if mode in all_data[s].get(sc, {})]
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

    # Build significance with two-level multiplicity control.
    # Pass 1: collect raw p-values for every (scenario, baseline, metric) cell.
    significance: dict = {}
    per_scenario_pvals: dict[str, dict[str, float]] = {sc: {} for sc in SCENARIOS}
    primary_h1_pvals: dict[str, float] = {}

    for sc in SCENARIOS:
        significance[sc] = {}
        for baseline in BASELINES:
            # Paired tests require both modes to come from the same seed.
            # Restrict to seeds that carry both entries; legacy JSONs that
            # predate a given mode (e.g. no_yield before Path B) are skipped
            # cleanly rather than crashed on.
            seeds_paired = sorted(
                s for s in all_data
                if "agribrain" in all_data[s].get(sc, {})
                and baseline in all_data[s].get(sc, {})
            )
            if not seeds_paired:
                continue
            comp: dict = {}
            for met in METRICS:
                a = [all_data[s][sc]["agribrain"][met] for s in seeds_paired]
                b = [all_data[s][sc][baseline][met] for s in seeds_paired]
                p = paired_permutation_pvalue(a, b)
                d = cohens_dz(a, b)
                lo_diff, hi_diff = bootstrap_mean_diff_ci(a, b)
                mean_diff = float(np.mean(a) - np.mean(b))
                comp[met] = {
                    "p_value": p,
                    "cohens_d": d,
                    "cohens_dz": d,
                    "mean_diff": mean_diff,
                    "mean_diff_ci_low": lo_diff,
                    "mean_diff_ci_high": hi_diff,
                    "n_seeds": len(seeds_paired),
                }
                # Accumulate for within-scenario BH-FDR family.
                per_scenario_pvals[sc][f"{baseline}:{met}"] = p
                # Primary H1 family: agribrain vs no_context on ARI per scenario.
                if baseline == "no_context" and met == "ari":
                    primary_h1_pvals[sc] = p
            significance[sc][f"agribrain_vs_{baseline}"] = comp

    # Pass 2a: Holm-Bonferroni across the primary H1 family (5 scenarios).
    primary_h1_holm = holm_bonferroni(primary_h1_pvals)

    # Pass 2b: BH-FDR within each scenario across all (baseline, metric) pairs.
    per_scenario_bh: dict[str, dict[str, float]] = {
        sc: benjamini_hochberg(per_scenario_pvals[sc]) for sc in SCENARIOS
    }

    # Pass 3: write adjusted p-values back into each comparison record. Each
    # cell gets both fields (p_value_adj_bh and, where applicable,
    # p_value_adj_holm) plus a canonical p_value_adj and correction_method.
    for sc in SCENARIOS:
        bh_map = per_scenario_bh.get(sc, {})
        for baseline in BASELINES:
            comp_key = f"agribrain_vs_{baseline}"
            comp = significance[sc].get(comp_key)
            if comp is None:
                continue
            for met in METRICS:
                rec = comp.get(met)
                if rec is None:
                    continue
                key = f"{baseline}:{met}"
                p_bh = float(bh_map.get(key, rec["p_value"]))
                rec["p_value_adj_bh"] = p_bh
                if baseline == "no_context" and met == "ari":
                    p_holm = float(primary_h1_holm.get(sc, rec["p_value"]))
                    rec["p_value_adj_holm"] = p_holm
                    # Canonical p_value_adj on the primary endpoint uses Holm,
                    # matching paper Section 3.13.
                    rec["p_value_adj"] = p_holm
                    rec["correction_method"] = "holm_bonferroni_across_scenarios"
                else:
                    rec["p_value_adj"] = p_bh
                    rec["correction_method"] = "bh_fdr_within_scenario"

    # Save
    out_dir = _SCRIPT_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    payload_summary = {
        "_meta": {
            "n_boot": 10_000,
            "n_perm": 10_000,
            "bootstrap_alpha": 0.05,
            "seeds_loaded": sorted(all_data),
        },
        "summary": summary,
    }
    payload_significance = {
        "_meta": {
            "primary_h1_family": "agribrain_vs_no_context on ARI, 5 scenarios",
            "primary_h1_correction": "holm_bonferroni",
            "secondary_correction": "bh_fdr",
            "secondary_family_scope": "per-scenario, all (baseline, metric) pairs",
            "n_perm": 10_000,
            "paired": True,
        },
        "primary_h1_holm_adjusted": primary_h1_holm,
        "significance": significance,
    }
    (out_dir / "benchmark_summary.json").write_text(
        json.dumps(payload_summary, indent=2)
    )
    (out_dir / "benchmark_significance.json").write_text(
        json.dumps(payload_significance, indent=2)
    )
    print("Saved benchmark_summary.json")
    print("Saved benchmark_significance.json")

    # Print key results
    print()
    for sc in SCENARIOS:
        a = summary[sc]["agribrain"]["ari"]
        print(f"  {sc}: ARI mean={a['mean']:.4f} CI=[{a['ci_low']:.4f}, {a['ci_high']:.4f}] std={a['std']:.6f}")

    print()
    print("Primary H1 family (Holm-Bonferroni across 5 scenarios):")
    for sc in SCENARIOS:
        p_raw = primary_h1_pvals.get(sc)
        p_adj = primary_h1_holm.get(sc)
        if p_raw is None or p_adj is None:
            continue
        print(f"  {sc} agribrain_vs_no_context ARI: p={p_raw:.4f} p_holm={p_adj:.4f}")

    print()
    print("Secondary (per-scenario BH-FDR) selected comparisons, ARI:")
    for sc in SCENARIOS:
        for comp_name in ("agribrain_vs_no_context", "agribrain_vs_no_yield",
                           "agribrain_vs_hybrid_rl"):
            rec = significance[sc].get(comp_name, {}).get("ari")
            if rec is None:
                continue
            print(f"  {sc} {comp_name}: p={rec['p_value']:.4f} p_adj={rec['p_value_adj']:.4f} "
                  f"({rec['correction_method']}) dz={rec['cohens_dz']:+.3f}")


if __name__ == "__main__":
    main()
