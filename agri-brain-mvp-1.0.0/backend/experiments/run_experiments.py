# backend/experiments/run_experiments.py
# Reproducible, large-scale experiment runner for AGRI-BRAIN MVP
# Generates: big CSV + ready-to-publish plots

import argparse, json, math, os, random, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Make sure we can import backend code ----
# Run this script from backend/, and set PYTHONPATH to backend/ (see instructions).
from src.models.spoilage import compute_spoilage, volatility_flags
from src.models.policy import Policy

BASE_DIR = Path(__file__).resolve().parents[1]       # backend/
DATA_PATH = BASE_DIR / "src" / "data_spinach.csv"    # your demo CSV


# ----------------------------- Scenario definitions -----------------------------
@dataclass
class ScenarioCfg:
    name: str
    intensities: List[float]  # e.g., [0.8, 1.0, 1.2, 1.4]
    # knobs interpreted per scenario
    deltaT_per_intensity: float = 0.0        # °C added per 1.0 intensity
    shock_mult_per_intensity: float = 1.0    # multiplier on shock rate
    demand_mult_per_intensity: float = 1.0
    inventory_mult_per_intensity: float = 1.0
    distance_mult_per_intensity: float = 1.0
    cold_burst_temp_add: float = 0.0         # °C added during bursts
    cold_burst_hours: int = 0                # hours per burst
    cold_burst_prob: float = 0.0             # probability a data row is inside a burst window


def scenarios_catalog() -> Dict[str, ScenarioCfg]:
    # You can tune these to match your narrative
    return {
        "climate_shock": ScenarioCfg(
            name="climate_shock",
            intensities=[0.8, 1.0, 1.2, 1.4],
            deltaT_per_intensity=3.0,               # +3C at 1.0 intensity
            shock_mult_per_intensity=1.3,           # 30% more shock events at 1.0 intensity
        ),
        "cold_chain": ScenarioCfg(
            name="cold_chain",
            intensities=[0.8, 1.0, 1.2, 1.4],
            cold_burst_temp_add=3.0, cold_burst_hours=12, cold_burst_prob=0.10,
            # emulate intermittent refrigeration failure
        ),
        "demand_glut": ScenarioCfg(
            name="demand_glut",
            intensities=[0.8, 1.0, 1.2, 1.4],
            demand_mult_per_intensity=1.4,          # +40% demand at 1.0
        ),
        "supply_gap": ScenarioCfg(
            name="supply_gap",
            intensities=[0.8, 1.0, 1.2, 1.4],
            inventory_mult_per_intensity=0.7,       # -30% inventory at 1.0
        ),
        "transport_disruption": ScenarioCfg(
            name="transport_disruption",
            intensities=[0.8, 1.0, 1.2, 1.4],
            distance_mult_per_intensity=1.25,       # +25% distance at 1.0 (detours)
        ),
    }


# ----------------------------- Synthetic perturbations -----------------------------
def apply_scenario(df: pd.DataFrame, sc: ScenarioCfg, intensity: float, rng: random.Random) -> pd.DataFrame:
    """Return a perturbed copy of df according to the scenario+intensity."""
    out = df.copy()

    # baseline: we assume df has tempC, RH, ambientC, shockG, inventory_units, demand_units
    # 1) uniform temperature shift
    if sc.deltaT_per_intensity != 0:
        dT = sc.deltaT_per_intensity * (intensity / 1.0)
        out["tempC"] = out["tempC"] + dT
        out["ambientC"] = out["ambientC"] + dT * 0.7  # ambient tends to move with temp but less

    # 2) shock multiplier (probability of higher shocks)
    if sc.shock_mult_per_intensity != 1.0:
        mult = sc.shock_mult_per_intensity ** (intensity / 1.0)
        # stochastic: if shockG < 0.5, sometimes bump; if already high, maybe bump more
        shock = out["shockG"].to_numpy().copy()
        prob = min(1.0, (mult - 1.0) * 0.3 + 0.1)  # crude probability curve
        mask = np.random.rand(len(shock)) < prob
        shock[mask] = shock[mask] * (1.0 + (mult - 1.0)) + np.random.rand(mask.sum()) * 0.2
        out["shockG"] = shock

    # 3) cold chain burst failures: add tempC for a random contiguous window per probability
    if sc.cold_burst_hours and sc.cold_burst_prob > 0 and sc.cold_burst_temp_add:
        n = len(out)
        window = min(sc.cold_burst_hours, n)
        for i in range(n):
            if rng.random() < sc.cold_burst_prob:
                j = min(n, i + window)
                out.loc[i:j, "tempC"] = out.loc[i:j, "tempC"] + sc.cold_burst_temp_add

    # 4) demand/inventory multipliers
    if sc.demand_mult_per_intensity != 1.0:
        m = sc.demand_mult_per_intensity ** (intensity / 1.0)
        out["demand_units"] = (out["demand_units"] * m).round().astype(int)
    if sc.inventory_mult_per_intensity != 1.0:
        m = sc.inventory_mult_per_intensity ** (intensity / 1.0)
        out["inventory_units"] = (out["inventory_units"] * m).round().astype(int)

    # 5) distance multiplier handled later when computing carbon (route-dependent)
    out.attrs["distance_mult"] = sc.distance_mult_per_intensity ** (intensity / 1.0)

    return out


# ----------------------------- Decision & metrics -----------------------------
def policy_decision(row: pd.Series, p: Policy) -> Tuple[str, float, float, float]:
    """
    Return (action, km, carbon, unit_price) using your backend's simple policy,
    without chain side-effects.
    """
    shelf = float(row["shelf_left"])
    vol = str(row.get("volatility", "ok"))

    if shelf < p.min_shelf_expedite:
        action = "expedite_to_retail"; km = p.km_expedited; price = p.msrp * 0.92
    elif shelf < p.min_shelf_reroute or vol == "anomaly":
        action = "reroute_to_near_dc"; km = p.km_farm_to_dc * 0.6; price = p.msrp * 0.95
    else:
        action = "standard_cold_chain"; km = p.km_farm_to_dc + p.km_dc_to_retail; price = p.msrp

    carbon = km * p.carbon_per_km
    return action, km, carbon, price


def effective_shelf_left(shelf_left: float, action: str) -> float:
    """
    Simple shelf penalty by transit time:
        standard ~36h, reroute ~24h, expedite ~12h
    penalty alpha ~0.015 per hour beyond a nominal 18h.
    This creates a mechanism where faster paths reduce spoilage.
    """
    hours = {"standard_cold_chain": 36, "reroute_to_near_dc": 24, "expedite_to_retail": 12}.get(action, 36)
    alpha = 0.015
    eff = shelf_left - alpha * max(0, hours - 18)
    return max(-1.0, min(1.0, eff))


def run_single(df_base: pd.DataFrame, policy: Policy, distance_mult: float, seed: int) -> Dict:
    """
    Compute baseline vs. Agri-Brain metrics for a single run.
    Baseline: always standard_cold_chain (no policy).
    Agri-Brain: row-wise policy decisions.
    """
    rng = random.Random(seed)
    df = df_base.copy()

    # Compute spoilage/volatility using your backend logic
    df = compute_spoilage(df)
    df["volatility"] = volatility_flags(df)

    # Baseline (always standard); km = p.km_farm_to_dc + p.km_dc_to_retail (with distance_mult)
    p = policy
    km_std = (p.km_farm_to_dc + p.km_dc_to_retail) * distance_mult
    carbon_std = km_std * p.carbon_per_km

    baseline_eff = [effective_shelf_left(float(s), "standard_cold_chain") for s in df["shelf_left"]]
    baseline_waste = float((np.array(baseline_eff) < 0).mean())

    # Agri-Brain decisions
    actions, kms, carbons, prices, eff_shelf = [], [], [], [], []
    for _, row in df.iterrows():
        action, km, c, price = policy_decision(row, p)
        km *= distance_mult
        c *= distance_mult
        actions.append(action); kms.append(km); carbons.append(c); prices.append(price)
        eff_shelf.append(effective_shelf_left(float(row["shelf_left"]), action))

    agri_waste = float((np.array(eff_shelf) < 0).mean())
    avg_temp = float(df["tempC"].mean())
    anomalies = int((df["volatility"] == "anomaly").sum())

    # Carbon & SLCA (very simple)
    mean_carbon = float(np.mean(carbons))
    slca = 1.0 / (1.0 + mean_carbon / 100.0)  # monotone decreasing proxy

    # Action mix
    mix = pd.Series(actions).value_counts(normalize=True)
    mix_std = float(mix.get("standard_cold_chain", 0))
    mix_rer = float(mix.get("reroute_to_near_dc", 0))
    mix_exp = float(mix.get("expedite_to_retail", 0))

    # Demand fulfillment proxy (service level): inventory vs demand
    inv = df["inventory_units"].to_numpy()
    dem = df["demand_units"].to_numpy()
    service = float(np.mean(np.minimum(inv, dem) / np.maximum(1, dem)))

    return {
        "baseline_waste": baseline_waste,
        "agri_waste": agri_waste,
        "delta_waste": baseline_waste - agri_waste,
        "avg_tempC": avg_temp,
        "anomaly_points": anomalies,
        "mean_carbon": mean_carbon,
        "slca_proxy": slca,
        "mix_std": mix_std,
        "mix_reroute": mix_rer,
        "mix_expedite": mix_exp,
        "service_level": service,
    }


# ----------------------------- Experiment driver -----------------------------
def run_grid(
    out_dir: Path,
    scenarios: List[str],
    reps: int,
    reroute_grid: List[float],
    expedite_grid: List[float],
    carbon_per_km: float,
    save_samples: bool = False,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(exist_ok=True)

    # base data
    df0 = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

    rows = []
    for sc_name in scenarios:
        sc_cfg = scenarios_catalog()[sc_name]
        for intensity in sc_cfg.intensities:
            for rer in reroute_grid:
                for exp in expedite_grid:
                    # ensure exp <= rer to be logical
                    if exp > rer:
                        continue
                    for r in range(reps):
                        seed = (hash((sc_name, intensity, rer, exp, r)) & 0xFFFFFFFF)
                        rng = random.Random(seed)

                        # policy for this cell
                        p = Policy()
                        p.min_shelf_reroute = float(rer)
                        p.min_shelf_expedite = float(exp)
                        p.carbon_per_km = float(carbon_per_km)

                        # scenario perturbation
                        df_sc = apply_scenario(df0, sc_cfg, intensity, rng)

                        # run single
                        metrics = run_single(df_sc, p, df_sc.attrs.get("distance_mult", 1.0), seed)

                        row = {
                            "scenario": sc_name,
                            "intensity": intensity,
                            "replicate": r,
                            "min_shelf_reroute": rer,
                            "min_shelf_expedite": exp,
                            "carbon_per_km": carbon_per_km,
                            **metrics,
                        }
                        rows.append(row)

                        if save_samples and r < 3:
                            # save a small sample of the perturbed time series for illustration
                            sample_path = out_dir / "samples" / f"{sc_name}_I{intensity}_rer{rer}_exp{exp}_r{r}.csv"
                            df_sc.to_csv(sample_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    return df


# ----------------------------- Plots -----------------------------
def fig_waste_by_scenario(df: pd.DataFrame, out_dir: Path):
    # aggregate by (scenario,intensity)
    g = df.groupby(["scenario","intensity"], as_index=False).agg(
        baseline_waste=("baseline_waste","mean"),
        agri_waste=("agri_waste","mean"),
        delta_waste=("delta_waste","mean"),
        mean_carbon=("mean_carbon","mean")
    )
    scenarios = sorted(g["scenario"].unique())

    plt.figure(figsize=(10,5))
    for sc in scenarios:
        gi = g[g["scenario"]==sc]
        plt.plot(gi["intensity"], gi["baseline_waste"], marker="o", linestyle="--", label=f"{sc}: baseline")
        plt.plot(gi["intensity"], gi["agri_waste"], marker="o", label=f"{sc}: agri")
    plt.ylabel("Waste rate")
    plt.xlabel("Intensity")
    plt.title("Waste vs. Scenario Intensity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_waste_vs_intensity.png", dpi=160)
    plt.close()

    # Carbon vs Delta-waste (trade-off cloud)
    plt.figure(figsize=(6,5))
    plt.scatter(g["mean_carbon"], g["delta_waste"], alpha=0.7)
    plt.xlabel("Mean carbon (kg/decision)")
    plt.ylabel("Baseline - Agri waste (↓ better)")
    plt.title("Trade-off: Carbon vs. Waste Reduction")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_tradeoff_carbon_vs_delta_waste.png", dpi=160)
    plt.close()


def fig_policy_heatmap(df: pd.DataFrame, out_dir: Path):
    # pick a scenario to illustrate (the worst average baseline waste)
    sc_mean = df.groupby("scenario")["baseline_waste"].mean().sort_values(ascending=False).index[0]
    sub = df[df["scenario"]==sc_mean]

    # grid pivot on reroute x expedite for mean delta_waste
    pivot = sub.pivot_table(index="min_shelf_expedite", columns="min_shelf_reroute", values="delta_waste", aggfunc="mean")
    if pivot.empty:
        return

    plt.figure(figsize=(6,5))
    im = plt.imshow(pivot.values, origin="lower", aspect="auto", cmap="RdYlGn")
    plt.colorbar(im, label="Baseline - Agri waste (↓ better)")
    plt.xticks(range(len(pivot.columns)), [f"{x:.2f}" for x in pivot.columns], rotation=45)
    plt.yticks(range(len(pivot.index)), [f"{y:.2f}" for y in pivot.index])
    plt.xlabel("min_shelf_reroute")
    plt.ylabel("min_shelf_expedite")
    plt.title(f"Policy sweep heatmap — scenario={sc_mean}")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_policy_heatmap.png", dpi=160)
    plt.close()


# ----------------------------- Main -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="experiments/out", help="output dir")
    ap.add_argument("--reps", type=int, default=10, help="replicates per grid cell")
    ap.add_argument("--scenarios", type=str, default="climate_shock,cold_chain,demand_glut,supply_gap,transport_disruption")
    ap.add_argument("--reroute", type=str, default="0.6,0.7,0.8")
    ap.add_argument("--expedite", type=str, default="0.4,0.5,0.6")
    ap.add_argument("--carbon_per_km", type=float, default=0.12)
    ap.add_argument("--save_samples", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    reroute_grid = [float(x) for x in args.reroute.split(",")]
    expedite_grid = [float(x) for x in args.expedite.split(",")]

    df = run_grid(
        out_dir=out_dir,
        scenarios=scenarios,
        reps=args.reps,
        reroute_grid=reroute_grid,
        expedite_grid=expedite_grid,
        carbon_per_km=args.carbon_per_km,
        save_samples=args.save_samples,
    )

    # Plots
    fig_waste_by_scenario(df, out_dir)
    fig_policy_heatmap(df, out_dir)

    print(f"[done] Wrote: {out_dir/'summary.csv'}")
    print(f"[done] Figures: {out_dir}/*.png")


if __name__ == "__main__":
    main()
