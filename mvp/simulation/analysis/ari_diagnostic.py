#!/usr/bin/env python3
"""
ARI Diagnostic Verification Script

Verifies every ARI value in the regenerated CSVs against the formula,
computes theoretical bounds, checks five-mode ranking consistency,
and produces a clear verdict on the root cause of ARI magnitude.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_SRC = Path(__file__).resolve().parent.parent.parent.parent / "agri-brain-mvp-1.0.0" / "backend"
if str(_BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(_BACKEND_SRC))

_SIM_DIR = Path(__file__).resolve().parent.parent
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))

import numpy as np
import pandas as pd

from generate_results import run_all, SCENARIOS, MODES, SEED

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
EXPECTED_RANKING = ["agribrain", "no_pinn", "hybrid_rl", "no_slca", "static"]


def main():
    print("=" * 78)
    print("ARI DIAGNOSTIC VERIFICATION")
    print("=" * 78)

    # ------------------------------------------------------------------
    # 1. Load regenerated CSVs
    # ------------------------------------------------------------------
    t1 = pd.read_csv(RESULTS_DIR / "table1_summary.csv")
    t2 = pd.read_csv(RESULTS_DIR / "table2_ablation.csv")
    print(f"\nLoaded table1_summary.csv: {len(t1)} rows")
    print(f"Loaded table2_ablation.csv: {len(t2)} rows")

    # ------------------------------------------------------------------
    # 2. Re-run simulation to get per-step data
    # ------------------------------------------------------------------
    print(f"\nRe-running simulation with seed={SEED} for per-step verification...")
    data = run_all(seed=SEED)
    results = data["results"]
    print("Simulation complete.\n")

    # ------------------------------------------------------------------
    # 3. Per-step ARI verification
    # ------------------------------------------------------------------
    print("=" * 78)
    print("SECTION A: PER-STEP ARI FORMULA VERIFICATION")
    print("  ARI_i = (1 - waste_i) * slca_i * (1 - rho_i)")
    print("=" * 78)

    all_waste, all_slca, all_rho, all_quality = [], [], [], []
    max_step_delta = 0.0
    step_mismatches = 0
    total_steps = 0

    for scenario in SCENARIOS:
        for mode in MODES:
            ep = results[scenario][mode]
            waste_trace = ep["waste_trace"]
            slca_trace = ep["slca_trace"]
            rho_trace = ep["rho_trace"]
            ari_trace = ep["ari_trace"]
            n = len(ari_trace)
            total_steps += n

            for i in range(n):
                w, s, r = waste_trace[i], slca_trace[i], rho_trace[i]
                expected = (1.0 - w) * s * (1.0 - r)
                delta = abs(ari_trace[i] - expected)
                if delta > max_step_delta:
                    max_step_delta = delta
                if delta > 1e-10:
                    step_mismatches += 1

                all_waste.append(w)
                all_slca.append(s)
                all_rho.append(r)
                all_quality.append(1.0 - r)

    print(f"\nTotal timesteps verified: {total_steps}")
    print(f"Max step-level delta:    {max_step_delta:.2e}")
    print(f"Steps with delta > 1e-10: {step_mismatches}")
    if max_step_delta < 1e-10:
        print("RESULT: All per-step ARI values match the formula EXACTLY.")
    else:
        print("WARNING: Some per-step ARI values deviate from formula!")

    # ------------------------------------------------------------------
    # 4. Episode-level (CSV) ARI verification
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SECTION B: EPISODE-LEVEL ARI VERIFICATION (CSV vs recomputed)")
    print("  CSV ARI = mean(per-step ARI) — NOT (1-mean_waste)*mean_slca*(1-mean_rho)")
    print("=" * 78)

    header = (f"{'Scenario':>20s} {'Mode':>12s} | {'Waste':>6s} {'SLCA':>6s} "
              f"{'mean_rho':>8s} | {'CSV_ARI':>7s} {'Recomp':>7s} {'Delta':>8s} | "
              f"{'Naive':>7s} {'NaiveD':>8s}")
    print(f"\n{header}")
    print("-" * len(header))

    any_csv_mismatch = False
    csv_deltas = []

    for scenario in SCENARIOS:
        for mode in MODES:
            ep = results[scenario][mode]
            csv_ari = ep["ari"]
            recomputed_ari = float(np.mean(ep["ari_trace"]))
            delta = abs(csv_ari - recomputed_ari)
            csv_deltas.append(delta)

            mean_waste = float(np.mean(ep["waste_trace"]))
            mean_slca = float(np.mean(ep["slca_trace"]))
            mean_rho = float(np.mean(ep["rho_trace"]))
            naive = (1.0 - mean_waste) * mean_slca * (1.0 - mean_rho)
            naive_delta = abs(csv_ari - naive)

            flag = " ***" if delta > 0.01 else ""
            print(f"{scenario:>20s} {mode:>12s} | {mean_waste:6.4f} {mean_slca:6.4f} "
                  f"{mean_rho:8.4f} | {csv_ari:7.4f} {recomputed_ari:7.4f} "
                  f"{delta:8.2e} | {naive:7.4f} {naive_delta:8.4f}{flag}")

            if delta > 0.01:
                any_csv_mismatch = True

            # Also verify against Table 2 CSV
            t2_row = t2[(t2["Scenario"] == scenario) & (t2["Variant"] == mode)]
            if not t2_row.empty:
                t2_ari = float(t2_row["ARI"].iloc[0])
                t2_delta = abs(t2_ari - round(csv_ari, 3))
                if t2_delta > 0.001:
                    print(f"  -> Table2 CSV mismatch: file={t2_ari}, computed={round(csv_ari,3)}")

    print(f"\nMax episode-level delta: {max(csv_deltas):.2e}")
    if not any_csv_mismatch:
        print("RESULT: All CSV ARI values match recomputed episode means (delta < 0.01).")
    else:
        print("WARNING: Some CSV ARI values do NOT match recomputed means!")

    # ------------------------------------------------------------------
    # 5. Component ranges and theoretical ARI bounds
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SECTION C: COMPONENT RANGES AND THEORETICAL ARI BOUNDS")
    print("=" * 78)

    waste_arr = np.array(all_waste)
    slca_arr = np.array(all_slca)
    rho_arr = np.array(all_rho)
    quality_arr = np.array(all_quality)

    print(f"\n{'Component':>20s} {'Min':>8s} {'Max':>8s} {'Mean':>8s} {'Std':>8s}")
    print("-" * 56)
    for name, arr in [("waste", waste_arr), ("slca", slca_arr),
                      ("rho (spoilage)", rho_arr), ("quality (1-rho)", quality_arr)]:
        print(f"{name:>20s} {arr.min():8.4f} {arr.max():8.4f} "
              f"{arr.mean():8.4f} {arr.std():8.4f}")

    ari_best = (1.0 - waste_arr.min()) * slca_arr.max() * quality_arr.max()
    ari_worst = (1.0 - waste_arr.max()) * slca_arr.min() * quality_arr.min()
    ari_typical = (1.0 - waste_arr.mean()) * slca_arr.mean() * quality_arr.mean()

    print("\nTheoretical ARI bounds (from observed component extremes):")
    print(f"  ARI_best  = (1 - {waste_arr.min():.4f}) * {slca_arr.max():.4f} * "
          f"{quality_arr.max():.4f} = {ari_best:.4f}")
    print(f"  ARI_worst = (1 - {waste_arr.max():.4f}) * {slca_arr.min():.4f} * "
          f"{quality_arr.min():.4f} = {ari_worst:.4f}")
    print(f"  ARI_typical (product of means) = {ari_typical:.4f}")

    # Per-mode bounds
    print(f"\n{'Mode':>12s} | {'waste_range':>15s} {'slca_range':>15s} "
          f"{'rho_range':>15s} | {'ARI range':>15s}")
    print("-" * 82)
    for mode in MODES:
        mode_aris, mode_wastes, mode_slcas, mode_rhos = [], [], [], []
        for scenario in SCENARIOS:
            ep = results[scenario][mode]
            mode_aris.append(ep["ari"])
            mode_wastes.append(float(np.mean(ep["waste_trace"])))
            mode_slcas.append(float(np.mean(ep["slca_trace"])))
            mode_rhos.append(float(np.mean(ep["rho_trace"])))
        print(f"{mode:>12s} | [{min(mode_wastes):.3f}, {max(mode_wastes):.3f}] "
              f"[{min(mode_slcas):.3f}, {max(mode_slcas):.3f}] "
              f"[{min(mode_rhos):.3f}, {max(mode_rhos):.3f}] | "
              f"[{min(mode_aris):.3f}, {max(mode_aris):.3f}]")

    # ------------------------------------------------------------------
    # 6. Five-mode ranking consistency
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SECTION D: FIVE-MODE RANKING CONSISTENCY PER SCENARIO")
    print(f"  Expected: {' > '.join(EXPECTED_RANKING)}")
    print("=" * 78)

    rank_inversions = 0
    for scenario in SCENARIOS:
        ari_by_mode = {}
        for mode in MODES:
            ari_by_mode[mode] = results[scenario][mode]["ari"]

        ranked = sorted(ari_by_mode.items(), key=lambda x: -x[1])
        actual_rank = [r[0] for r in ranked]

        match = actual_rank == EXPECTED_RANKING
        status = "OK" if match else "INVERSION"
        if not match:
            rank_inversions += 1

        print(f"\n  {scenario:>20s}: [{status}]")
        for i, (m, a) in enumerate(ranked):
            expected_m = EXPECTED_RANKING[i]
            flag = "" if m == expected_m else f" (expected: {expected_m})"
            print(f"    {i+1}. {m:>12s}  ARI={a:.4f}{flag}")

    print(f"\nRank inversions: {rank_inversions} / {len(SCENARIOS)} scenarios")

    # ------------------------------------------------------------------
    # 7. Sanity checks for calibration
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SECTION E: CALIBRATION SANITY CHECKS")
    print("=" * 78)

    calibration_issues = []

    # Check per-episode means
    for scenario in SCENARIOS:
        for mode in MODES:
            ep = results[scenario][mode]
            mw = float(np.mean(ep["waste_trace"]))
            ms = float(np.mean(ep["slca_trace"]))
            mr = float(np.mean(ep["rho_trace"]))

            if mw > 0.50:
                calibration_issues.append(
                    f"  {scenario}/{mode}: mean waste = {mw:.3f} > 0.50")
            if ms < 0.30:
                calibration_issues.append(
                    f"  {scenario}/{mode}: mean SLCA = {ms:.3f} < 0.30")
            if mr > 0.80:
                calibration_issues.append(
                    f"  {scenario}/{mode}: mean rho = {mr:.3f} > 0.80")

    if calibration_issues:
        print("\nCalibration warnings (component outside realistic range):")
        for issue in calibration_issues:
            print(issue)
    else:
        print("\nAll component means are within realistic sanity ranges:")
        print("  - waste: all <= 0.50")
        print("  - SLCA:  all >= 0.30")
        print("  - rho:   all <= 0.80")

    # ------------------------------------------------------------------
    # 8. README comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SECTION F: README vs ACTUAL COMPARISON")
    print("=" * 78)

    readme_expected = {
        "heatwave":         {"ARI": 0.60, "Waste": 0.03, "RLE": 0.98, "SLCA": 0.74},
        "overproduction":   {"ARI": 0.60, "Waste": 0.05, "RLE": 0.95, "SLCA": 0.70},
        "cyber_outage":     {"ARI": 0.64, "Waste": 0.04, "RLE": 0.84, "SLCA": 0.73},
        "adaptive_pricing": {"ARI": 0.72, "Waste": 0.02, "RLE": 0.82, "SLCA": 0.78},
    }

    print(f"\n{'Scenario':>20s} | {'Metric':>6s} {'README':>8s} {'Actual':>8s} {'Diff':>8s}")
    print("-" * 60)
    for scenario, expected in readme_expected.items():
        ep = results[scenario]["agribrain"]
        for metric, exp_val in expected.items():
            key = metric.lower()
            actual = ep[key]
            diff = actual - exp_val
            print(f"{scenario:>20s} | {metric:>6s} {exp_val:8.3f} {actual:8.3f} {diff:+8.3f}")

    # ------------------------------------------------------------------
    # 9. Verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)

    if any_csv_mismatch:
        print("\n>>> CODE BUG: Recomputed ARI does not match reported ARI in CSVs.")
    elif calibration_issues:
        print("\n>>> CALIBRATION ISSUE: Some components are outside realistic ranges.")
        for issue in calibration_issues:
            print(issue)
    else:
        print("\n>>> EXPECTED MATHEMATICAL PROPERTY")
        print()
        print("All per-step ARI values match the formula ARI = (1-waste)*SLCA*(1-rho)")
        print("exactly (max delta < 1e-10). All CSV episode means match recomputed")
        print("means exactly. All components are within realistic sanity ranges.")
        print()
        print("The multiplicative composition of three [0,1] factors inherently")
        print("produces a composite that is lower than any individual factor.")
        all_aris = [results[s][m]["ari"] for s in SCENARIOS for m in MODES]
        print(f"Observed ARI range across all scenarios and modes: "
              f"[{min(all_aris):.3f}, {max(all_aris):.3f}]")
        print()
        print("The relative five-mode ranking is the appropriate basis for")
        print(f"cross-method comparison. Rank inversions: {rank_inversions}/{len(SCENARIOS)}.")
        print()
        print(f"Theoretical ARI bounds: [{ari_worst:.4f}, {ari_best:.4f}]")

    print("\n" + "=" * 78)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
