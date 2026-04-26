#!/usr/bin/env python3
"""
Quick stochastic feasibility test.

Uses truncated data (first 50 timesteps) to prove:
  1. Perturbations create measurable variation across seeds
  2. Method ordering (agribrain > hybrid_rl > static) is preserved
  3. Metric values stay within realistic bounds

Full-length validation should follow once feasibility is confirmed.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ["DETERMINISTIC_MODE"] = "false"
os.environ["FORECAST_METHOD"] = "holt_winters"

_BACKEND_SRC = Path(__file__).resolve().parent.parent.parent.parent / "agri-brain-mvp-1.0.0" / "backend"
if str(_BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(_BACKEND_SRC))

_SIM_DIR = Path(__file__).resolve().parent.parent
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))

import numpy as np
import pandas as pd

from generate_results import (
    apply_scenario, run_episode, DATA_CSV, Policy,
    _AGRIBRAIN_LOGIT_MODES,
)
from stochastic import make_stochastic_layer, StochasticLayer

SCENARIOS = ["heatwave", "baseline"]
METHODS = ["static", "hybrid_rl", "agribrain"]
METRICS = ["ari", "waste", "rle", "slca"]
MAX_ROWS = 50  # Truncate for speed (full data = 288 rows)
SEEDS = [42, 1337, 2024]


def run_mini(seed, stochastic):
    rng = np.random.default_rng(seed)
    policy = Policy()
    df_base = pd.read_csv(DATA_CSV, parse_dates=["timestamp"]).head(MAX_ROWS)
    out = {}
    for sc in SCENARIOS:
        out[sc] = {}
        sc_rng = np.random.default_rng(rng.integers(0, 2**31))
        df_sc = apply_scenario(df_base, sc, policy, sc_rng)
        ab_seed = rng.integers(0, 2**31)
        for m in METHODS:
            ms = int(ab_seed) if m in _AGRIBRAIN_LOGIT_MODES else int(rng.integers(0, 2**31))
            mrng = np.random.default_rng(ms)
            if stochastic:
                stoch = make_stochastic_layer(np.random.default_rng(ms + 1))
            else:
                stoch = StochasticLayer(rng=np.random.default_rng(0), enabled=False,
                                       temp_std_c=0.0, rh_std=0.0, demand_frac_std=0.0,
                                       inventory_frac_std=0.0, transport_km_frac_std=0.0,
                                       k_ref_frac_std=0.0, ea_r_frac_std=0.0,
                                       onset_jitter_hours=0.0, theta_noise_std=0.0,
                                       policy_temp_std=0.0,
                                       delay_prob=0.0)
            ep = run_episode(df_sc, m, policy, mrng, sc, stoch=stoch)
            out[sc][m] = {k: ep[k] for k in METRICS}
            print(f"  [{sc:>15s}] [{m:>12s}] ARI={ep['ari']:.4f} waste={ep['waste']:.4f} "
                  f"RLE={ep['rle']:.4f} SLCA={ep['slca']:.4f}")
        # consume remaining RNG for alignment
        for _ in range(5):
            rng.integers(0, 2**31)
    return out


def main():
    print("=" * 60)
    print("QUICK STOCHASTIC FEASIBILITY TEST")
    print(f"  Data: {MAX_ROWS} timesteps, {len(SCENARIOS)} scenarios, "
          f"{len(METHODS)} methods, {len(SEEDS)} seeds")
    print("=" * 60)
    t_total = time.time()

    print("\n1) DETERMINISTIC baseline (seed=42):")
    t0 = time.time()
    det = run_mini(42, stochastic=False)
    print(f"   ({time.time()-t0:.1f}s)\n")

    stoch_runs = {}
    for seed in SEEDS:
        print(f"2) STOCHASTIC seed={seed}:")
        t0 = time.time()
        stoch_runs[seed] = run_mini(seed, stochastic=True)
        print(f"   ({time.time()-t0:.1f}s)\n")

    # --- Analysis ---
    print("=" * 60)
    print("A) STOCHASTIC vs DETERMINISTIC (does noise have effect?)")
    print("=" * 60)
    for sc in SCENARIOS:
        print(f"\n  {sc}:")
        for m in METHODS:
            for met in METRICS:
                d = det[sc][m][met]
                vals = [stoch_runs[s][sc][m][met] for s in SEEDS]
                mean_s = np.mean(vals)
                std_s = np.std(vals)
                max_diff = max(abs(v - d) for v in vals)
                print(f"    {m:>12s} {met:>5s}: det={d:.5f}  "
                      f"stoch_mean={mean_s:.5f} +/-{std_s:.6f}  "
                      f"max_delta={max_diff:.6f}")

    print("\n" + "=" * 60)
    print("B) METHOD ORDERING CHECK (agribrain >= hybrid >= static)")
    print("=" * 60)
    violations = 0
    checks = 0
    for label, data in [("deterministic", det)] + [(f"stoch_{s}", stoch_runs[s]) for s in SEEDS]:
        for sc in SCENARIOS:
            a = data[sc]["agribrain"]["ari"]
            h = data[sc]["hybrid_rl"]["ari"]
            s = data[sc]["static"]["ari"]
            ok = a >= h >= s
            checks += 1
            if not ok:
                violations += 1
            print(f"  {label:>15s} {sc:>10s}: "
                  f"agri={a:.4f} hybrid={h:.4f} static={s:.4f}  "
                  f"{'OK' if ok else 'FAIL'}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Check variation
    any_diff = False
    for sc in SCENARIOS:
        for m in METHODS:
            for met in METRICS:
                d = det[sc][m][met]
                for s in SEEDS:
                    if abs(stoch_runs[s][sc][m][met] - d) > 1e-6:
                        any_diff = True

    print(f"Stochastic variation: {'YES' if any_diff else 'NO (broken!)'}")
    print(f"Ordering preserved: {checks - violations}/{checks}")
    print(f"Total runtime: {time.time()-t_total:.1f}s")

    print()
    if violations == 0 and any_diff:
        print("VERDICT: FEASIBLE")
    elif not any_diff:
        print("VERDICT: BROKEN - no variation detected")
    else:
        print(f"VERDICT: NEEDS TUNING - {violations} ordering violations")


if __name__ == "__main__":
    main()
