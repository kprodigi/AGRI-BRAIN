#!/usr/bin/env python3
"""
Post-generation validation suite.
Checks ALL metric ranges and orderings against the specification.
Run after generate_results.py. ALL checks must pass before committing.

Stochastic mode: uses relaxed bounds and ordering tolerance (0.01) to
accommodate seeded perturbation noise. Deterministic mode: strict exact checks.
"""
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import json

# Import stochastic config (handles env var reading)
_SIM_DIR = Path(__file__).resolve().parent
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))
from stochastic import DETERMINISTIC_MODE

_RESULTS_DIR = Path(__file__).resolve().parent / "results"
t1 = pd.read_csv(_RESULTS_DIR / "table1_summary.csv")
t2 = pd.read_csv(_RESULTS_DIR / "table2_ablation.csv")

# Ordering tolerance: 0.0 for deterministic, 0.01 for stochastic
_TOL = 0.0 if DETERMINISTIC_MODE else 0.01

errors = []

def get(scenario, method, metric):
    row = t1[(t1["Scenario"] == scenario) & (t1["Method"] == method)]
    if len(row) == 0:
        errors.append(f"MISSING: {scenario}/{method}")
        return None
    return float(row[metric].values[0])

def get2(scenario, variant, metric):
    row = t2[(t2["Scenario"] == scenario) & (t2["Variant"] == variant)]
    if len(row) == 0:
        errors.append(f"MISSING ablation: {scenario}/{variant}")
        return None
    return float(row[metric].values[0])

# ============================================================
# CHECK 1: AGRI-BRAIN ARI ranges
# ============================================================
if DETERMINISTIC_MODE:
    ari_ranges = {
        "heatwave": (0.55, 0.65), "overproduction": (0.58, 0.68),
        "cyber_outage": (0.58, 0.68), "adaptive_pricing": (0.66, 0.76),
        "baseline": (0.68, 0.78),
    }
else:
    # Stochastic: widen by ±0.05 to accommodate seed-to-seed variation
    ari_ranges = {
        "heatwave": (0.50, 0.70), "overproduction": (0.53, 0.73),
        "cyber_outage": (0.53, 0.73), "adaptive_pricing": (0.61, 0.81),
        "baseline": (0.63, 0.83),
    }
for sc, (lo, hi) in ari_ranges.items():
    v = get(sc, "agribrain", "ARI")
    if v is not None and not (lo <= v <= hi):
        errors.append(f"ARI range: agribrain/{sc} = {v:.3f}, expected [{lo}, {hi}]")

# ============================================================
# CHECK 2: ARI method ordering in every scenario
# ============================================================
for sc in t1["Scenario"].unique():
    ab = get(sc, "agribrain", "ARI")
    hr = get(sc, "hybrid_rl", "ARI")
    st = get(sc, "static", "ARI")
    if ab is not None and hr is not None and st is not None:
        if not (ab > hr - _TOL and hr > st - _TOL):
            errors.append(f"ARI ordering: {sc}: AB={ab:.3f} > HR={hr:.3f} > ST={st:.3f} VIOLATED")

# ============================================================
# CHECK 3: ARI scenario ordering for AGRI-BRAIN
# ============================================================
ab_ari = {sc: get(sc, "agribrain", "ARI") for sc in t1["Scenario"].unique()}
if all(v is not None for v in ab_ari.values()):
    if not (ab_ari["baseline"] >= ab_ari["adaptive_pricing"]):
        errors.append(f"ARI scenario order: baseline ({ab_ari['baseline']:.3f}) < pricing ({ab_ari['adaptive_pricing']:.3f})")
    if not (ab_ari["adaptive_pricing"] > ab_ari["cyber_outage"]):
        errors.append(f"ARI scenario order: pricing ({ab_ari['adaptive_pricing']:.3f}) <= cyber ({ab_ari['cyber_outage']:.3f})")
    if not (ab_ari["cyber_outage"] >= ab_ari["overproduction"]):
        errors.append(f"ARI scenario order: cyber ({ab_ari['cyber_outage']:.3f}) < overprod ({ab_ari['overproduction']:.3f})")
    if not (ab_ari["overproduction"] > ab_ari["heatwave"]):
        errors.append(f"ARI scenario order: overprod ({ab_ari['overproduction']:.3f}) <= heatwave ({ab_ari['heatwave']:.3f})")
    # Cyber must be meaningfully below baseline
    if ab_ari["baseline"] - ab_ari["cyber_outage"] < 0.03:
        errors.append(f"ARI: cyber ({ab_ari['cyber_outage']:.3f}) too close to baseline ({ab_ari['baseline']:.3f}), gap < 0.03")

# ============================================================
# CHECK 4: RLE rules
# ============================================================
for sc in t1["Scenario"].unique():
    # Static must be 0
    st_rle = get(sc, "static", "RLE")
    if st_rle is not None and st_rle != 0.0:
        errors.append(f"RLE: static/{sc} = {st_rle} (must be 0.0)")
    # AGRI-BRAIN must be > 0
    ab_rle = get(sc, "agribrain", "RLE")
    if ab_rle is not None and ab_rle <= 0.0:
        errors.append(f"RLE: agribrain/{sc} = {ab_rle} (must be > 0)")
    # AGRI-BRAIN >= Hybrid RL
    hr_rle = get(sc, "hybrid_rl", "RLE")
    if ab_rle is not None and hr_rle is not None and ab_rle < hr_rle - 0.01:
        errors.append(f"RLE ordering: {sc}: AB={ab_rle:.3f} < HR={hr_rle:.3f}")

# RLE scenario ordering for AGRI-BRAIN: heatwave > overproduction > cyber
ab_rle_hw = get("heatwave", "agribrain", "RLE")
ab_rle_op = get("overproduction", "agribrain", "RLE")
ab_rle_cy = get("cyber_outage", "agribrain", "RLE")
if all(v is not None for v in [ab_rle_hw, ab_rle_op, ab_rle_cy]):
    if not (ab_rle_hw > ab_rle_op > ab_rle_cy):
        errors.append(f"RLE scenario order: HW={ab_rle_hw:.3f} > OP={ab_rle_op:.3f} > CY={ab_rle_cy:.3f} VIOLATED")

# ============================================================
# CHECK 5: Waste ranges and ordering
# ============================================================
if DETERMINISTIC_MODE:
    waste_ranges_ab = {
        "heatwave": (0.02, 0.05), "overproduction": (0.03, 0.05),
        "cyber_outage": (0.03, 0.05), "adaptive_pricing": (0.019, 0.04),
        "baseline": (0.018, 0.03),
    }
else:
    # Stochastic: widen by ±0.01
    waste_ranges_ab = {
        "heatwave": (0.01, 0.06), "overproduction": (0.02, 0.06),
        "cyber_outage": (0.02, 0.06), "adaptive_pricing": (0.009, 0.05),
        "baseline": (0.008, 0.04),
    }
for sc, (lo, hi) in waste_ranges_ab.items():
    v = get(sc, "agribrain", "Waste")
    if v is not None and not (lo <= v <= hi):
        errors.append(f"Waste range: agribrain/{sc} = {v:.3f}, expected [{lo}, {hi}]")

for sc in t1["Scenario"].unique():
    ab = get(sc, "agribrain", "Waste")
    hr = get(sc, "hybrid_rl", "Waste")
    st = get(sc, "static", "Waste")
    if ab is not None and hr is not None and st is not None:
        if not (st > hr - _TOL and hr > ab - _TOL):
            errors.append(f"Waste ordering: {sc}: ST={st:.3f} > HR={hr:.3f} > AB={ab:.3f} VIOLATED")

# Cyber waste must be higher than baseline waste for AGRI-BRAIN
ab_w_cy = get("cyber_outage", "agribrain", "Waste")
ab_w_bl = get("baseline", "agribrain", "Waste")
if ab_w_cy is not None and ab_w_bl is not None:
    if not (ab_w_cy > ab_w_bl):
        errors.append(f"Waste: cyber ({ab_w_cy:.3f}) must be > baseline ({ab_w_bl:.3f})")

# ============================================================
# CHECK 6: SLCA ordering
# ============================================================
for sc in t1["Scenario"].unique():
    ab = get(sc, "agribrain", "SLCA")
    hr = get(sc, "hybrid_rl", "SLCA")
    st = get(sc, "static", "SLCA")
    if ab is not None and hr is not None and st is not None:
        if not (ab > hr - _TOL and hr > st - _TOL):
            errors.append(f"SLCA ordering: {sc}: AB={ab:.3f} > HR={hr:.3f} > ST={st:.3f} VIOLATED")

# SLCA: cyber must be lower than baseline for AGRI-BRAIN
ab_s_cy = get("cyber_outage", "agribrain", "SLCA")
ab_s_bl = get("baseline", "agribrain", "SLCA")
if ab_s_cy is not None and ab_s_bl is not None:
    if not (ab_s_bl > ab_s_cy):
        errors.append(f"SLCA: baseline ({ab_s_bl:.3f}) must be > cyber ({ab_s_cy:.3f})")

# ============================================================
# CHECK 7: Carbon ordering
# ============================================================
for sc in t1["Scenario"].unique():
    ab = get(sc, "agribrain", "Carbon")
    hr = get(sc, "hybrid_rl", "Carbon")
    st = get(sc, "static", "Carbon")
    if ab is not None and hr is not None and st is not None:
        carbon_tol = 0.0 if DETERMINISTIC_MODE else 50.0
        if not (st > hr - carbon_tol and hr > ab - carbon_tol):
            errors.append(f"Carbon ordering: {sc}: ST={st:.0f} > HR={hr:.0f} > AB={ab:.0f} VIOLATED")

# ============================================================
# CHECK 8: Equity constraints
# ============================================================
for sc in t1["Scenario"].unique():
    st_eq = get(sc, "static", "Equity")
    eq_lo_static = 0.92 if DETERMINISTIC_MODE else 0.88
    if st_eq is not None and st_eq < eq_lo_static:
        errors.append(f"Equity: static/{sc} = {st_eq:.3f} (must be >= {eq_lo_static})")
    eq_range = (0.80, 0.91) if DETERMINISTIC_MODE else (0.75, 0.95)
    for method in ["hybrid_rl", "agribrain"]:
        eq = get(sc, method, "Equity")
        if eq is not None and not (eq_range[0] <= eq <= eq_range[1]):
            errors.append(f"Equity range: {method}/{sc} = {eq:.3f}, expected [{eq_range[0]}, {eq_range[1]}]")

# ============================================================
# CHECK 9: Ablation ordering (ARI) in every scenario
# ============================================================
expected_ablation = ["agribrain", "no_pinn", "hybrid_rl", "no_slca", "static"]
for sc in t2["Scenario"].unique():
    vals = {}
    for var in expected_ablation:
        v = get2(sc, var, "ARI")
        if v is not None:
            vals[var] = v
    if len(vals) == 5:
        for i in range(len(expected_ablation) - 1):
            a, b = expected_ablation[i], expected_ablation[i+1]
            if vals[a] < vals[b] - 0.005:
                errors.append(f"Ablation ARI inversion: {sc}: {a}={vals[a]:.3f} < {b}={vals[b]:.3f}")

# ============================================================
# CHECK 10: Global sanity bounds
# ============================================================
_waste_hi = 0.20 if DETERMINISTIC_MODE else 0.25
_slca_lo = 0.35 if DETERMINISTIC_MODE else 0.30
_carbon_lo = 1500 if DETERMINISTIC_MODE else 1400
_carbon_hi = 5500 if DETERMINISTIC_MODE else 5800
for _, row in t1.iterrows():
    if not (0.01 <= row["Waste"] <= _waste_hi):
        errors.append(f"Waste out of bounds: {row['Method']}/{row['Scenario']} = {row['Waste']}")
    if not (0.0 <= row["ARI"] <= 1.0):
        errors.append(f"ARI out of bounds: {row['Method']}/{row['Scenario']} = {row['ARI']}")
    if not (_slca_lo <= row["SLCA"] <= 0.95):
        errors.append(f"SLCA out of bounds: {row['Method']}/{row['Scenario']} = {row['SLCA']}")
    if not (_carbon_lo <= row["Carbon"] <= _carbon_hi):
        errors.append(f"Carbon out of bounds: {row['Method']}/{row['Scenario']} = {row['Carbon']}")
    if not (0.0 <= row["RLE"] <= 1.0):
        errors.append(f"RLE out of bounds: {row['Method']}/{row['Scenario']} = {row['RLE']}")

# ============================================================
# CHECK 11: Operational feasibility diagnostics (if present)
# ============================================================
if "DecisionLatencyMs" in t1.columns:
    for _, row in t1.iterrows():
        if not (0.0 <= row["DecisionLatencyMs"] <= 5000.0):
            errors.append(
                f"DecisionLatencyMs out of bounds: {row['Method']}/{row['Scenario']} = {row['DecisionLatencyMs']}"
            )

if "ConstraintViolationRate" in t1.columns:
    for _, row in t1.iterrows():
        if not (0.0 <= row["ConstraintViolationRate"] <= 1.0):
            errors.append(
                f"ConstraintViolationRate out of bounds: {row['Method']}/{row['Scenario']} = {row['ConstraintViolationRate']}"
            )

# ============================================================
# REPORT
# ============================================================
bench_path = _RESULTS_DIR / "benchmark_summary.json"
if bench_path.exists():
    try:
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
        for sc in ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing", "baseline"]:
            if sc in bench and "agribrain" in bench[sc]:
                ari_mean = bench[sc]["agribrain"]["ari"]["mean"]
                if not (0.0 <= ari_mean <= 1.0):
                    errors.append(f"Benchmark ARI mean out of bounds: {sc}={ari_mean}")
    except Exception as e:
        errors.append(f"Failed to parse benchmark_summary.json: {e}")

_mode_label = "DETERMINISTIC" if DETERMINISTIC_MODE else "STOCHASTIC"
print(f"\n{'='*70}")
print(f"Validation mode: {_mode_label}")
if errors:
    print(f"VALIDATION FAILED: {len(errors)} issue(s)")
    print(f"{'='*70}")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
else:
    print(f"ALL CHECKS PASSED")
    print(f"{'='*70}")
    print("\nFinal AGRI-BRAIN results:")
    for sc in ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing", "baseline"]:
        r = t1[(t1["Scenario"] == sc) & (t1["Method"] == "agribrain")].iloc[0]
        print(f"  {sc:>20s}: ARI={r['ARI']:.3f} Waste={r['Waste']:.3f} RLE={r['RLE']:.3f} SLCA={r['SLCA']:.3f} Carbon={r['Carbon']:.0f} Eq={r['Equity']:.3f}")
    sys.exit(0)
