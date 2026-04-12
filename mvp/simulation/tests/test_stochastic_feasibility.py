#!/usr/bin/env python3
"""
Stochastic feasibility test (fast version).

Runs ONLY the 3 core methods (static, hybrid_rl, agribrain) across all 5
scenarios with 5 seeds.  Skips the 5 ablation modes to keep runtime manageable.

Reports:
  1. Whether method ordering (agribrain > hybrid_rl > static) holds per seed
  2. Mean +/- std of key metrics across seeds
  3. Comparison against the deterministic baseline
  4. Whether values stay within physically realistic ranges
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# --- Force stochastic mode BEFORE any imports ---
os.environ["DETERMINISTIC_MODE"] = "false"

_BACKEND_SRC = Path(__file__).resolve().parent.parent.parent.parent / "agri-brain-mvp-1.0.0" / "backend"
if str(_BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(_BACKEND_SRC))

_SIM_DIR = Path(__file__).resolve().parent.parent
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))

import numpy as np
import pandas as pd

from generate_results import (
    apply_scenario, run_episode, DATA_CSV, SCENARIOS, Policy,
    _AGRIBRAIN_LOGIT_MODES,
)
from stochastic import make_stochastic_layer, StochasticLayer

_QUICK = "--quick" in sys.argv
SEEDS = [42, 1337] if _QUICK else [42, 1337, 2024, 7, 99]
CORE_METHODS = ["static", "hybrid_rl", "agribrain"]
METRICS = ["ari", "waste", "rle", "slca"]
MAX_ROWS = 50 if _QUICK else 0  # 0 = full data

BOUNDS = {
    "ari":   (0.0, 1.0),
    "waste": (0.0, 0.30),
    "rle":   (0.0, 1.0),
    "slca":  (0.20, 1.0),
}


def run_fast(seed: int, stochastic: bool) -> dict:
    """Run only 3 core methods × 5 scenarios (15 episodes instead of 40)."""
    rng = np.random.default_rng(seed)
    policy = Policy()
    df_base = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])
    if MAX_ROWS > 0:
        df_base = df_base.head(MAX_ROWS)

    results = {}
    for scenario in SCENARIOS:
        results[scenario] = {}
        scenario_rng = np.random.default_rng(rng.integers(0, 2**31))
        df_scenario = apply_scenario(df_base, scenario, policy, scenario_rng)

        ablation_seed = rng.integers(0, 2**31)

        for mode in CORE_METHODS:
            if mode in _AGRIBRAIN_LOGIT_MODES:
                mseed = int(ablation_seed)
            else:
                mseed = int(rng.integers(0, 2**31))

            mode_rng = np.random.default_rng(mseed)

            if stochastic:
                stoch = make_stochastic_layer(np.random.default_rng(mseed + 1))
            else:
                stoch = StochasticLayer(rng=np.random.default_rng(0), enabled=False,
                                       temp_std_c=0.0, rh_std=0.0, demand_frac_std=0.0,
                                       inventory_frac_std=0.0, transport_km_frac_std=0.0,
                                       k_ref_frac_std=0.0, ea_r_frac_std=0.0,
                                       onset_jitter_hours=0.0, theta_noise_std=0.0,
                                       delay_prob=0.0)

            episode = run_episode(df_scenario, mode, policy, mode_rng, scenario, stoch=stoch)
            results[scenario][mode] = {m: episode[m] for m in METRICS}
            print(f"  [{scenario:>20s}] [{mode:>12s}] "
                  f"ARI={episode['ari']:.4f}  waste={episode['waste']:.4f}  "
                  f"RLE={episode['rle']:.4f}  SLCA={episode['slca']:.4f}")

        # Consume the remaining RNG draws to keep seed sequence aligned
        for _ in range(5):  # 5 non-core modes
            rng.integers(0, 2**31)

    return results


def main():
    print("=" * 70)
    mode_str = "QUICK" if _QUICK else "FULL"
    print(f"STOCHASTIC FEASIBILITY TEST ({mode_str}, 3 methods only)")
    print("=" * 70)
    print(f"Seeds: {SEEDS}")
    print(f"Methods: {CORE_METHODS}")
    print(f"Scenarios: {SCENARIOS}")
    if MAX_ROWS > 0:
        print(f"Truncated to {MAX_ROWS} timesteps (--quick mode)")
    print()

    # --- Deterministic baseline ---
    print("DETERMINISTIC BASELINE (seed=42):")
    t0 = time.time()
    baseline = run_fast(seed=42, stochastic=False)
    dt_base = time.time() - t0
    print(f"  Completed in {dt_base:.1f}s")
    print()

    # --- Stochastic runs ---
    all_runs = []
    for seed in SEEDS:
        print(f"STOCHASTIC seed={seed}:")
        t0 = time.time()
        result = run_fast(seed=seed, stochastic=True)
        all_runs.append(result)
        print(f"  Completed in {time.time() - t0:.1f}s")
        print()

    # --- Analysis ---
    print("=" * 70)
    print("COMPARISON: Baseline vs Stochastic (mean +/- std across 5 seeds)")
    print("=" * 70)

    ordering_violations = 0
    total_checks = 0
    bound_violations = 0

    for sc in SCENARIOS:
        print(f"\n--- {sc} ---")
        print(f"{'Metric':>8s}  {'Method':>12s}  {'Baseline':>10s}  "
              f"{'Mean':>10s}  {'Std':>10s}  {'Min':>10s}  {'Max':>10s}  {'OK?':>5s}")
        print("-" * 82)

        for met in METRICS:
            for m in CORE_METHODS:
                vals = [run[sc][m][met] for run in all_runs]
                base = baseline[sc][m][met]
                mean = np.mean(vals)
                std = np.std(vals)
                vmin, vmax = np.min(vals), np.max(vals)
                lo, hi = BOUNDS[met]
                ok = lo <= vmin and vmax <= hi
                if not ok:
                    bound_violations += 1
                print(f"{met:>8s}  {m:>12s}  {base:10.4f}  "
                      f"{mean:10.4f}  {std:10.6f}  {vmin:10.4f}  {vmax:10.4f}  "
                      f"{'OK' if ok else 'FAIL':>5s}")

        # Ordering check per seed
        for i, seed in enumerate(SEEDS):
            run = all_runs[i]
            a = run[sc]["agribrain"]["ari"]
            h = run[sc]["hybrid_rl"]["ari"]
            s = run[sc]["static"]["ari"]
            total_checks += 1
            if not (a >= h >= s):
                ordering_violations += 1
                print(f"  ** ORDERING VIOLATION seed={seed}: "
                      f"agribrain={a:.4f}  hybrid={h:.4f}  static={s:.4f}")

    # --- Summary ---
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Stochastic variation check
    diffs = []
    for sc in SCENARIOS:
        for m in CORE_METHODS:
            for met in METRICS:
                vals = [run[sc][m][met] for run in all_runs]
                base = baseline[sc][m][met]
                max_diff = max(abs(v - base) for v in vals)
                diffs.append(max_diff)
    any_diff = any(d > 1e-6 for d in diffs)
    print(f"Stochastic variation: {'YES' if any_diff else 'NO (perturbations not working!)'}")
    print(f"Mean max-deviation from baseline: {np.mean(diffs):.6f}")
    print(f"Largest single deviation: {np.max(diffs):.6f}")
    print(f"Ordering: {total_checks - ordering_violations}/{total_checks} passed")
    print(f"Bounds: {'ALL OK' if bound_violations == 0 else f'{bound_violations} VIOLATIONS'}")

    print()
    if ordering_violations == 0 and bound_violations == 0 and any_diff:
        print("VERDICT: FEASIBLE")
        print("Stochastic mode produces varied but defensible results.")
    elif not any_diff:
        print("VERDICT: BROKEN")
        print("Perturbations have no effect. Check wiring.")
    else:
        print(f"VERDICT: NEEDS TUNING")
        print(f"{ordering_violations} ordering violations, {bound_violations} bound violations.")

    return ordering_violations == 0 and bound_violations == 0 and any_diff


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
