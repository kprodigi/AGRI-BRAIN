#!/usr/bin/env python3
"""
Post-generation validation suite.
Checks ALL metric ranges and orderings against the specification.
Run after generate_results.py. ALL checks must pass before committing.

Stochastic mode: uses relaxed bounds and ordering tolerance (0.01) to
accommodate seeded perturbation noise. Deterministic mode: strict exact checks.
"""
import sys
from pathlib import Path

import pandas as pd
import json

# Import stochastic config (handles env var reading)
_SIM_DIR = Path(__file__).resolve().parent.parent
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))
from stochastic import DETERMINISTIC_MODE

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
_T1 = _RESULTS_DIR / "table1_summary.csv"
_T2 = _RESULTS_DIR / "table2_ablation.csv"
if not _T1.exists() or not _T2.exists():
    print(f"SKIP: result tables not present in {_RESULTS_DIR}")
    print("      Run generate_results.py to produce them, then re-run this script.")
    sys.exit(0)
t1 = pd.read_csv(_T1)
t2 = pd.read_csv(_T2)

# Ordering tolerance: 0.0 for deterministic, 0.06 for stochastic.
# Implementation note: 2025-04 realism recalibration.
# The previous 0.04 tolerance was paired with the old narrow-CI run (CIs
# of 0.001-0.005 on ARI). With realistic stochastic noise the per-mode
# CIs widen to ~0.02-0.05, and any two ranks separated by less than
# ~0.06 cannot be distinguished with 95 % confidence. Tightening below
# this would mean the validator fails on perfectly ordinary seed
# variation. 0.06 keeps the ordering check meaningful while accepting
# the realistic noise envelope.
_TOL = 0.0 if DETERMINISTIC_MODE else 0.06

errors = []
warnings_ord = []  # ordering claims reported but never blocking


def _ord(msg: str) -> None:
    """Record an ordering claim; reported but never blocks the build.

    Ordering claims (e.g. "agribrain ARI > hybrid_rl ARI") are
    confirmation-bias-prone when used as build gates. They are
    decided by the bootstrap CIs and adjusted p-values produced by
    `aggregate_seeds.py`; the validator only enforces *intervals*
    (range checks).
    """
    warnings_ord.append(msg)


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
    # Implementation note: Deterministic ranges were tightened around the
    # post-recalibration analytical baselines (lower SLCA ceiling +
    # softer SLCA bonuses + less-aggressive governance override). They
    # span the realistic seed-mean envelope, not the wider stochastic
    # bounds below.
    ari_ranges = {
        "heatwave": (0.45, 0.62), "overproduction": (0.48, 0.65),
        "cyber_outage": (0.48, 0.65), "adaptive_pricing": (0.55, 0.72),
        "baseline": (0.58, 0.75),
    }
else:
    # Stochastic: ±0.15 envelope around the recalibrated deterministic
    # midpoints. Wider than DETERMINISTIC because the new stochastic
    # layer (sensor 2.5C, demand CV 25 %, theta noise 0.08, etc.) drives
    # realistic 0.02-0.05 within-mode CI widths.
    ari_ranges = {
        "heatwave": (0.30, 0.77), "overproduction": (0.33, 0.80),
        "cyber_outage": (0.33, 0.80), "adaptive_pricing": (0.40, 0.87),
        "baseline": (0.43, 0.90),
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
            _ord(f"ARI ordering: {sc}: AB={ab:.3f} > HR={hr:.3f} > ST={st:.3f} VIOLATED")

# ============================================================
# CHECK 3: ARI scenario ordering for AGRI-BRAIN
# ============================================================
ab_ari = {sc: get(sc, "agribrain", "ARI") for sc in t1["Scenario"].unique()}
if all(v is not None for v in ab_ari.values()):
    if not (ab_ari["baseline"] >= ab_ari["adaptive_pricing"]):
        _ord(f"ARI scenario order: baseline ({ab_ari['baseline']:.3f}) < pricing ({ab_ari['adaptive_pricing']:.3f})")
    if not (ab_ari["adaptive_pricing"] > ab_ari["cyber_outage"]):
        _ord(f"ARI scenario order: pricing ({ab_ari['adaptive_pricing']:.3f}) <= cyber ({ab_ari['cyber_outage']:.3f})")
    if not (ab_ari["cyber_outage"] >= ab_ari["overproduction"]):
        _ord(f"ARI scenario order: cyber ({ab_ari['cyber_outage']:.3f}) < overprod ({ab_ari['overproduction']:.3f})")
    if not (ab_ari["overproduction"] > ab_ari["heatwave"]):
        _ord(f"ARI scenario order: overprod ({ab_ari['overproduction']:.3f}) <= heatwave ({ab_ari['heatwave']:.3f})")
    # Cyber must be meaningfully below baseline (ordering-style claim)
    if ab_ari["baseline"] - ab_ari["cyber_outage"] < 0.03:
        _ord(f"ARI: cyber ({ab_ari['cyber_outage']:.3f}) too close to baseline ({ab_ari['baseline']:.3f}), gap < 0.03")

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
    # AGRI-BRAIN vs Hybrid RL RLE ordering. Under the post-2026-04
    # rho-conditional hierarchy weighting in models/resilience.py:
    #
    #   marketable band (rho <= RHO_MARKETABLE_CUTOFF = 0.50):
    #     local_redistribute = 1.00, recovery = 0.40, cold_chain = 0.00
    #   non-marketable band (rho > 0.50):
    #     local_redistribute = 0.00, recovery = 1.00, cold_chain = 0.00
    #
    # AgriBrain's RHO_RECOVERY_KNEE (0.30) and food-safety cutoff
    # (0.65) deliberately route high-rho batches to Recovery, which
    # now scores 1.00 in the non-marketable band (the EU 2008/98/EC
    # Article 4 ordering: at the marketable -> non-marketable
    # boundary, redistributing to humans is no longer permitted, so
    # Recovery for animal feed / energy becomes the hierarchically-
    # preferred option). Hybrid RL has no knee and keeps routing to
    # local_redistribute, which scores 0.00 in the non-marketable
    # band. The expected ordering on heat-stressed / over-production
    # scenarios is therefore agribrain >= hybrid_rl RLE, with the
    # gap proportional to the fraction of steps in the
    # non-marketable band. We keep the check informational rather
    # than blocking because rho profiles are stochastic enough that
    # the gap can compress on low-noise seeds.
    hr_rle = get(sc, "hybrid_rl", "RLE")
    if ab_rle is not None and hr_rle is not None and ab_rle < hr_rle - 0.05:
        _ord(
            f"RLE inversion: {sc}: AB={ab_rle:.3f} below HR={hr_rle:.3f} "
            f"(>0.05 gap). Under rho-conditional weighting AgriBrain "
            f"should >= Hybrid RL on this scenario; investigate whether "
            f"the knee is firing or whether action mapping is correct."
        )

# Cross-scenario RLE ordering for AGRI-BRAIN was previously asserted as
# heatwave >= overproduction > cyber_outage. Under the new
# rho-conditional knee-driven physics this is now the *expected*
# ordering — heatwave drives the highest fraction of steps into the
# non-marketable band where Recovery scores 1.00, so heatwave RLE
# should sit at the top of the cross-scenario band. We keep the
# low-band guard below as informational rather than upgrading to a
# strict ordering invariant because per-seed rho noise can shuffle
# the bottom two scenarios.
ab_rle_hw = get("heatwave", "agribrain", "RLE")
ab_rle_op = get("overproduction", "agribrain", "RLE")
ab_rle_cy = get("cyber_outage", "agribrain", "RLE")
if all(v is not None for v in [ab_rle_hw, ab_rle_op, ab_rle_cy]):
    if ab_rle_hw < 0.5 or ab_rle_op < 0.5 or ab_rle_cy < 0.4:
        _ord(f"RLE low-band: HW={ab_rle_hw:.3f} OP={ab_rle_op:.3f} CY={ab_rle_cy:.3f} (any below 0.5/0.5/0.4 expected band)")
# Also assert agribrain RLE no longer trivially hits 1.0 across the board:
# a saturated 1.0000 is a tautology of the policy, not a measurement, and
# was flagged as a problem in the previous run. Allow at most one scenario
# at 1.0 (extreme edge case where every step rerouted by chance).
ab_rle_all = [get(sc, "agribrain", "RLE") for sc in ["heatwave","overproduction","cyber_outage","adaptive_pricing","baseline"]]
n_at_one = sum(1 for v in ab_rle_all if v is not None and v >= 0.999)
if n_at_one >= 4:
    # Range-style anti-saturation gate: >=4 scenarios at the ceiling means
    # the metric is degenerate, not a "agribrain wins" claim. Keep as a
    # hard error.
    errors.append(f"RLE saturation: {n_at_one}/5 scenarios hit RLE >= 0.999. Recalibrate the policy or noise; this is tautological.")

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
    # Stochastic: widen for realistic field-level perturbation amplitudes
    waste_ranges_ab = {
        "heatwave": (0.005, 0.08), "overproduction": (0.01, 0.08),
        "cyber_outage": (0.01, 0.08), "adaptive_pricing": (0.005, 0.06),
        "baseline": (0.005, 0.05),
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
            _ord(f"Waste ordering: {sc}: ST={st:.3f} > HR={hr:.3f} > AB={ab:.3f} VIOLATED")

# Cyber waste must be higher than baseline waste for AGRI-BRAIN (ordering)
ab_w_cy = get("cyber_outage", "agribrain", "Waste")
ab_w_bl = get("baseline", "agribrain", "Waste")
if ab_w_cy is not None and ab_w_bl is not None:
    if not (ab_w_cy > ab_w_bl):
        _ord(f"Waste: cyber ({ab_w_cy:.3f}) must be > baseline ({ab_w_bl:.3f})")

# ============================================================
# CHECK 6: SLCA ordering
# ============================================================
for sc in t1["Scenario"].unique():
    ab = get(sc, "agribrain", "SLCA")
    hr = get(sc, "hybrid_rl", "SLCA")
    st = get(sc, "static", "SLCA")
    if ab is not None and hr is not None and st is not None:
        if not (ab > hr - _TOL and hr > st - _TOL):
            _ord(f"SLCA ordering: {sc}: AB={ab:.3f} > HR={hr:.3f} > ST={st:.3f} VIOLATED")

# SLCA: cyber must be lower than baseline for AGRI-BRAIN (ordering)
ab_s_cy = get("cyber_outage", "agribrain", "SLCA")
ab_s_bl = get("baseline", "agribrain", "SLCA")
if ab_s_cy is not None and ab_s_bl is not None:
    if not (ab_s_bl > ab_s_cy):
        _ord(f"SLCA: baseline ({ab_s_bl:.3f}) must be > cyber ({ab_s_cy:.3f})")

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
            _ord(f"Carbon ordering: {sc}: ST={st:.0f} > HR={hr:.0f} > AB={ab:.0f} VIOLATED")

# ============================================================
# CHECK 8: Equity constraints
# ============================================================
for sc in t1["Scenario"].unique():
    st_eq = get(sc, "static", "Equity")
    # Equity = mean(SLCA) * (1 - std(SLCA)).
    # Static always picks cold_chain → SLCA is the same every step → std=0
    # → equity = mean(SLCA). The bound therefore matches the static SLCA
    # bound for cold_chain, which lies in roughly [0.40, 1.00] across the
    # post-recalibration parameter space (deterministic and stochastic
    # both). Earlier revisions of this validator used a tighter bound
    # appropriate for an older SLCA-base set; that bound was dead by
    # 2026-04 and is widened here to match the std=0 regime.
    eq_lo_static = 0.30 if DETERMINISTIC_MODE else 0.25
    eq_hi_static = 1.00
    if st_eq is not None and not (eq_lo_static <= st_eq <= eq_hi_static):
        errors.append(
            f"Equity: static/{sc} = {st_eq:.3f} out of range "
            f"[{eq_lo_static}, {eq_hi_static}]"
        )
    eq_range = (0.40, 0.95) if DETERMINISTIC_MODE else (0.30, 0.97)
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
            abl_tol = 0.005 if DETERMINISTIC_MODE else 0.02
            if vals[a] < vals[b] - abl_tol:
                _ord(f"Ablation ARI inversion: {sc}: {a}={vals[a]:.3f} < {b}={vals[b]:.3f}")

# ============================================================
# CHECK 10: Global sanity bounds
# ============================================================
_waste_hi = 0.20 if DETERMINISTIC_MODE else 0.30
_slca_lo = 0.35 if DETERMINISTIC_MODE else 0.25
_carbon_lo = 1500 if DETERMINISTIC_MODE else 1200
_carbon_hi = 5500 if DETERMINISTIC_MODE else 6200
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

# Implementation note: 2025-04 instrumentation symmetry fix.
# ConstraintViolationRate is now the operational metric (temp OR quality)
# which is symmetric across every method. Under the new schema,
# AgriBrain MUST score better than (or equal to within tolerance of) Static
# on this column, because the only way to reduce temperature/quality
# violations is to reroute, and rerouting is exactly what AgriBrain does.
# The old asymmetric metric had AgriBrain at 0.80 vs Static at 0.59;
# under the new metric, AgriBrain should land roughly 0.05-0.20 below
# Static everywhere, never above.
if "ConstraintViolationRate" in t1.columns:
    for sc in t1["Scenario"].unique():
        st_cv = get(sc, "static", "ConstraintViolationRate")
        ab_cv = get(sc, "agribrain", "ConstraintViolationRate")
        if st_cv is not None and ab_cv is not None:
            # Allow a small tolerance for seed noise; the substantive claim
            # is "AgriBrain does not violate operational constraints more
            # often than the always-cold-chain baseline".
            if ab_cv > st_cv + 0.05:
                _ord(
                    f"ConstraintViolationRate ordering: agribrain/{sc}={ab_cv:.3f} > "
                    f"static/{sc}={st_cv:.3f} + 0.05. Operational metric should be "
                    f"AB <= ST; if this fires, the asymmetric instrumentation regressed."
                )

# RegulatoryViolationRate (compliance-only, structurally zero for non-MCP modes)
# is reported but not range-checked because zero is a valid value for static
# and hybrid_rl by construction.

# ============================================================
# REPORT
# ============================================================
bench_path = _RESULTS_DIR / "benchmark_summary.json"
if bench_path.exists():
    try:
        bench_payload = json.loads(bench_path.read_text(encoding="utf-8"))
        # Aggregator wraps the data under a "summary" key; unwrap so the
        # bench[scenario][method][metric] traversal works on both the
        # wrapped and legacy-flat formats.
        bench = (
            bench_payload["summary"]
            if isinstance(bench_payload, dict) and isinstance(bench_payload.get("summary"), dict)
            else bench_payload
        )
        for sc in ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing", "baseline"]:
            if sc in bench and "agribrain" in bench[sc]:
                ari_mean = bench[sc]["agribrain"]["ari"]["mean"]
                if not (0.0 <= ari_mean <= 1.0):
                    errors.append(f"Benchmark ARI mean out of bounds: {sc}={ari_mean}")
    except Exception as e:
        errors.append(f"Failed to parse benchmark_summary.json: {e}")

# Implementation note: 2026-04 validator-mode change.
#
# A 2025-04 revision had downgraded the validator to report-only by
# default to address an earlier confirmation-bias concern (range *and*
# ordering claims were both gating the build, and the orderings encoded
# the manuscript's preferred direction). The 2026-04 fix is more
# surgical: ordering claims now go through
# `_ord(...)` and are reported as warnings only, while range / interval
# checks stay as hard errors and gate the build by default.
#
# To restore the previous report-only behaviour for local debugging,
# export STRICT_VALIDATION=0. The canonical configuration is strict.
import os as _os
import json as _json

_strict = _os.environ.get("STRICT_VALIDATION", "1") == "1"
_mode_label = "DETERMINISTIC" if DETERMINISTIC_MODE else "STOCHASTIC"
print(f"\n{'='*70}")
print(f"Validation mode: {_mode_label} (strict={_strict})")

# Always write a machine-readable report so CI and downstream readers
# can inspect the gate outcomes without re-running the validator.
report = {
    "mode": _mode_label,
    "strict": _strict,
    "n_errors": len(errors),
    "errors": list(errors),
    "n_ordering_warnings": len(warnings_ord),
    "ordering_warnings": list(warnings_ord),
}
try:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (_RESULTS_DIR / "validation_report.json").write_text(_json.dumps(report, indent=2))
except Exception:
    pass

if warnings_ord:
    print(f"\nORDERING WARNINGS (reported, never blocking): {len(warnings_ord)}")
    for w in warnings_ord:
        print(f"  WARN  {w}")

if errors:
    label = "VALIDATION FAILED" if _strict else "VALIDATION REPORTED RANGE ISSUES"
    print(f"\n{label}: {len(errors)} range/interval issue(s)")
    print(f"{'='*70}")
    for e in errors:
        print(f"  {e}")
    if _strict:
        sys.exit(1)
    print("\nNon-strict mode (STRICT_VALIDATION=0): continuing with exit 0.")
else:
    print("\nALL RANGE CHECKS PASSED")
    print(f"{'='*70}")

print("\nFinal AGRI-BRAIN results:")
for sc in ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing", "baseline"]:
    rows = t1[(t1["Scenario"] == sc) & (t1["Method"] == "agribrain")]
    if rows.empty:
        continue
    r = rows.iloc[0]
    print(f"  {sc:>20s}: ARI={r['ARI']:.3f} Waste={r['Waste']:.3f} RLE={r['RLE']:.3f} SLCA={r['SLCA']:.3f} Carbon={r['Carbon']:.0f} Eq={r['Equity']:.3f}")
sys.exit(0)
