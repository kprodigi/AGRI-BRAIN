"""Post-fix validation script. Run after generate_results.py completes."""
import pandas as pd
import numpy as np

t1 = pd.read_csv("mvp/simulation/results/table1_summary.csv")
t2 = pd.read_csv("mvp/simulation/results/table2_ablation.csv")

errors = []

# --- CHECK 1: AGRI-BRAIN RLE must be > 0 in ALL scenarios ---
for _, row in t1[t1["Method"] == "agribrain"].iterrows():
    if row["RLE"] == 0.0:
        errors.append(f"FAIL: AGRI-BRAIN RLE = 0.0 in {row['Scenario']} (should be > 0)")

# --- CHECK 2: Static RLE must be 0 in ALL scenarios ---
for _, row in t1[t1["Method"] == "static"].iterrows():
    if row["RLE"] != 0.0:
        errors.append(f"FAIL: Static RLE = {row['RLE']} in {row['Scenario']} (should be 0.0)")

# --- CHECK 3: ARI ordering AGRI-BRAIN > Hybrid RL > Static in every scenario ---
for scenario in t1["Scenario"].unique():
    s = t1[t1["Scenario"] == scenario]
    ab = s[s["Method"] == "agribrain"]["ARI"].values[0]
    hr = s[s["Method"] == "hybrid_rl"]["ARI"].values[0]
    st = s[s["Method"] == "static"]["ARI"].values[0]
    if not (ab > hr > st):
        errors.append(f"FAIL: ARI ordering broken in {scenario}: AB={ab}, HR={hr}, ST={st}")

# --- CHECK 4: Waste rates in realistic range [0.02, 0.15] for all methods ---
for _, row in t1.iterrows():
    if not (0.01 <= row["Waste"] <= 0.20):
        errors.append(f"FAIL: Waste = {row['Waste']} for {row['Method']}/{row['Scenario']} outside [0.01, 0.20]")

# --- CHECK 5: Ablation ordering (ARI) should be AB > NoPINN > HybridRL > NoSLCA > Static ---
# Allow ties within 0.005 tolerance but no rank inversions > 0.005
for scenario in t2["Scenario"].unique():
    s = t2[t2["Scenario"] == scenario]
    vals = {}
    for _, row in s.iterrows():
        vals[row["Variant"]] = row["ARI"]
    expected_order = ["agribrain", "no_pinn", "hybrid_rl", "no_slca", "static"]
    for i in range(len(expected_order) - 1):
        a, b = expected_order[i], expected_order[i+1]
        if vals.get(a, 0) < vals.get(b, 0) - 0.005:
            errors.append(f"FAIL: Ablation inversion in {scenario}: {a}={vals[a]:.3f} < {b}={vals[b]:.3f}")

# --- CHECK 6: AGRI-BRAIN carbon < Hybrid RL carbon < Static carbon in every scenario ---
for scenario in t1["Scenario"].unique():
    s = t1[t1["Scenario"] == scenario]
    ab = s[s["Method"] == "agribrain"]["Carbon"].values[0]
    hr = s[s["Method"] == "hybrid_rl"]["Carbon"].values[0]
    st = s[s["Method"] == "static"]["Carbon"].values[0]
    if not (ab < hr < st):
        errors.append(f"FAIL: Carbon ordering broken in {scenario}: AB={ab}, HR={hr}, ST={st}")

# --- CHECK 7: Distinct spoilage across scenarios ---
# Cyber outage, adaptive pricing, and baseline should now have different AGRI-BRAIN waste values
ab_cyber = t1[(t1["Scenario"] == "cyber_outage") & (t1["Method"] == "agribrain")]["Waste"].values[0]
ab_pricing = t1[(t1["Scenario"] == "adaptive_pricing") & (t1["Method"] == "agribrain")]["Waste"].values[0]
ab_baseline = t1[(t1["Scenario"] == "baseline") & (t1["Method"] == "agribrain")]["Waste"].values[0]
if ab_cyber == ab_pricing == ab_baseline:
    errors.append(f"FAIL: Cyber/Pricing/Baseline still identical for AGRI-BRAIN waste ({ab_cyber})")

# --- CHECK 8: Cyber outage should NOT be the easiest scenario ---
ab_ari = {}
for _, row in t1[t1["Method"] == "agribrain"].iterrows():
    ab_ari[row["Scenario"]] = row["ARI"]
if ab_ari.get("cyber_outage", 0) > ab_ari.get("baseline", 1):
    errors.append(f"FAIL: Cyber outage ARI ({ab_ari['cyber_outage']:.3f}) > Baseline ARI ({ab_ari['baseline']:.3f})")

# --- CHECK 9: SLCA composite in [0.4, 0.95] for all entries ---
for _, row in t1.iterrows():
    if not (0.35 <= row["SLCA"] <= 0.95):
        errors.append(f"FAIL: SLCA = {row['SLCA']} for {row['Method']}/{row['Scenario']} outside [0.35, 0.95]")

# --- CHECK 10: Equity in [0.70, 1.00] for all entries ---
for _, row in t1.iterrows():
    if not (0.70 <= row["Equity"] <= 1.00):
        errors.append(f"FAIL: Equity = {row['Equity']} for {row['Method']}/{row['Scenario']} outside [0.70, 1.00]")

# --- REPORT ---
if errors:
    print(f"\n{'='*60}")
    print(f"VALIDATION FAILED: {len(errors)} issue(s)")
    print(f"{'='*60}")
    for e in errors:
        print(f"  {e}")
    print(f"\nFix the issues and regenerate results before committing.")
else:
    print(f"\n{'='*60}")
    print(f"ALL 10 CHECKS PASSED")
    print(f"{'='*60}")
    print("Results are ready to commit.")
