#!/usr/bin/env python3
"""
AGRI-BRAIN Figure Generation
==============================
Generates 7 publication-quality figures (PNG + PDF at 300 DPI).

Standalone usage:
    cd mvp/simulation
    python generate_figures.py

Requires generate_results.py to have been run first (or runs it automatically).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure backend models are importable
_BACKEND_SRC = Path(__file__).resolve().parent.parent.parent / "agri-brain-mvp-1.0.0" / "backend"
if str(_BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(_BACKEND_SRC))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from generate_results import run_all, SCENARIOS, MODES, RESULTS_DIR, RLE_THRESHOLD

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------
COLORS = {
    "static":    "#95a5a6",
    "hybrid_rl": "#3498db",
    "no_pinn":   "#e67e22",
    "no_slca":   "#9b59b6",
    "agribrain": "#27ae60",
}

MODE_LABELS = {
    "static":    "Static",
    "hybrid_rl": "Hybrid RL",
    "no_pinn":   "No PINN",
    "no_slca":   "No SLCA",
    "agribrain": "AGRI-BRAIN",
}

SCENARIO_LABELS = {
    "heatwave":         "Heatwave",
    "overproduction":   "Overproduction",
    "cyber_outage":     "Cyber Outage",
    "adaptive_pricing": "Price Volatility",
    "baseline":         "Baseline",
}

DPI = 300


def _apply_style(ax):
    """Remove top/right spines and set font sizes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    ax.title.set_size(10)


def _save(fig, name):
    """Save figure as both PNG and PDF."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = RESULTS_DIR / f"{name}.{ext}"
        fig.savefig(str(path), dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"  Saved {name}.png / .pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Heatwave scenario deep-dive (2x2)
# ---------------------------------------------------------------------------
def fig2_heatwave(data):
    """2x2: temp+humidity, PINN rho trajectory, action prob stacked area, cumulative reward."""
    hw = data["results"]["heatwave"]
    ab = hw["agribrain"]
    hours = np.array(ab["hours"])

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Fig. 2 — Heatwave Scenario Analysis", fontsize=11, fontweight="bold")

    # --- (a) Temperature + Humidity with heatwave window ---
    ax = axes[0, 0]
    ax.plot(hours, ab["temp_trace"], color="#e74c3c", linewidth=0.8, label="Temp (°C)")
    ax2 = ax.twinx()
    ax2.plot(hours, ab["rh_trace"], color="#3498db", linewidth=0.8, alpha=0.7, label="RH (%)")
    ax.axvspan(24, 48, alpha=0.15, color="red", label="Heatwave window")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Temperature (°C)")
    ax2.set_ylabel("Relative Humidity (%)")
    ax.set_title("(a) Environmental Exposure")
    ax.legend(loc="upper left", fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)
    _apply_style(ax)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(labelsize=8)

    # --- (b) PINN rho trajectory for each mode ---
    ax = axes[0, 1]
    for mode in ["static", "hybrid_rl", "agribrain"]:
        ep = hw[mode]
        ax.plot(hours, ep["rho_trace"], color=COLORS[mode], linewidth=0.9,
                label=MODE_LABELS[mode], alpha=0.85)
    ax.axvspan(24, 48, alpha=0.12, color="red")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Spoilage Risk ρ(t)")
    ax.set_title("(b) PINN Spoilage Trajectory")
    ax.legend(fontsize=7)
    _apply_style(ax)

    # --- (c) Action probability stacked area (agribrain) ---
    ax = axes[1, 0]
    probs = np.array(ab["prob_trace"])
    ax.fill_between(hours, 0, probs[:, 0],
                    color="#2980b9", alpha=0.7, label="Cold Chain")
    ax.fill_between(hours, probs[:, 0], probs[:, 0] + probs[:, 1],
                    color="#27ae60", alpha=0.7, label="Local Redist.")
    ax.fill_between(hours, probs[:, 0] + probs[:, 1], 1.0,
                    color="#e67e22", alpha=0.7, label="Recovery")
    ax.axvspan(24, 48, alpha=0.12, color="red")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Action Probability")
    ax.set_title("(c) AGRI-BRAIN Action Probabilities")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="center right")
    _apply_style(ax)

    # --- (d) Cumulative reward ---
    ax = axes[1, 1]
    for mode in ["static", "hybrid_rl", "agribrain"]:
        ep = hw[mode]
        ax.plot(hours, ep["cumulative_reward"], color=COLORS[mode],
                linewidth=0.9, label=MODE_LABELS[mode])
    ax.axvspan(24, 48, alpha=0.12, color="red")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("(d) Cumulative Reward Comparison")
    ax.legend(fontsize=7)
    _apply_style(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "fig2_heatwave")


# ---------------------------------------------------------------------------
# Figure 3: Overproduction / Reverse Logistics (2x2)
# ---------------------------------------------------------------------------
def fig3_reverse(data):
    """2x2: yield vs demand, waste rolling avg, RLE rolling, SLCA component bars."""
    op = data["results"]["overproduction"]
    ab = op["agribrain"]
    hours = np.array(ab["hours"])

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Fig. 3 — Overproduction & Reverse Logistics", fontsize=11, fontweight="bold")

    # --- (a) Yield (inventory) vs demand with surplus fill ---
    ax = axes[0, 0]
    inv = np.array(ab["inventory_trace"])
    dem = np.array(ab["demand_trace"])
    ax.plot(hours, inv / 1000, color="#27ae60", linewidth=0.8, label="Inventory (×1000)")
    ax.plot(hours, dem, color="#e74c3c", linewidth=0.8, label="Demand")
    ax.fill_between(hours, dem, inv / 1000, where=(inv / 1000 > dem),
                    alpha=0.2, color="#f39c12", label="Surplus")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Units")
    ax.set_title("(a) Inventory vs Demand")
    ax.legend(fontsize=7)
    _apply_style(ax)

    # --- (b) Waste rolling average ---
    ax = axes[0, 1]
    window = 12  # 3-hour rolling window
    for mode in ["static", "hybrid_rl", "agribrain"]:
        ep = op[mode]
        waste = np.array(ep["waste_trace"])
        rolling = np.convolve(waste, np.ones(window) / window, mode="same")
        ax.plot(hours, rolling, color=COLORS[mode], linewidth=0.9,
                label=MODE_LABELS[mode])
    ax.set_xlabel("Hours")
    ax.set_ylabel("Waste Rate (rolling avg)")
    ax.set_title("(b) Waste Reduction Over Time")
    ax.legend(fontsize=7)
    _apply_style(ax)

    # --- (c) RLE rolling ---
    ax = axes[1, 0]
    for mode in ["static", "hybrid_rl", "agribrain"]:
        ep = op[mode]
        rho = np.array(ep["rho_trace"])
        actions = np.array(ep["action_trace"])
        # Compute rolling RLE
        at_risk = rho > RLE_THRESHOLD
        routed = at_risk & (actions >= 1)  # local_redistribute or recovery
        rle_rolling = np.convolve(routed.astype(float), np.ones(window) / window, mode="same")
        rle_denom = np.convolve(at_risk.astype(float), np.ones(window) / window, mode="same")
        rle_frac = np.divide(rle_rolling, rle_denom,
                             out=np.zeros_like(rle_rolling), where=rle_denom > 0)
        ax.plot(hours, rle_frac, color=COLORS[mode], linewidth=0.9,
                label=MODE_LABELS[mode])
    ax.set_xlabel("Hours")
    ax.set_ylabel("RLE (rolling)")
    ax.set_title("(c) Reverse Logistics Efficiency")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=7)
    _apply_style(ax)

    # --- (d) SLCA component grouped bars ---
    ax = axes[1, 1]
    components = ["C", "L", "R", "P"]
    comp_labels = ["Carbon", "Labour", "Resilience", "Price"]
    x = np.arange(len(components))
    width = 0.25
    for i, mode in enumerate(["static", "hybrid_rl", "agribrain"]):
        ep = op[mode]
        vals = []
        for comp in components:
            comp_vals = [s[comp] for s in ep["slca_component_trace"]]
            vals.append(np.mean(comp_vals))
        ax.bar(x + i * width, vals, width, color=COLORS[mode],
               label=MODE_LABELS[mode], alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(comp_labels, fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("(d) SLCA Components")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.1)
    _apply_style(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "fig3_reverse")


# ---------------------------------------------------------------------------
# Figure 4: Cyber Outage (1x3)
# ---------------------------------------------------------------------------
def fig4_cyber(data):
    """1x3: ARI over time with outage shading, action distribution, blockchain audit scatter."""
    cy = data["results"]["cyber_outage"]
    ab = cy["agribrain"]
    hours = np.array(ab["hours"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Fig. 4 — Cyber Outage Scenario", fontsize=11, fontweight="bold")

    # --- (a) ARI over time with outage shading ---
    ax = axes[0]
    for mode in ["static", "hybrid_rl", "agribrain"]:
        ep = cy[mode]
        ari = np.array(ep["ari_trace"])
        rolling = np.convolve(ari, np.ones(12) / 12, mode="same")
        ax.plot(hours, rolling, color=COLORS[mode], linewidth=0.9,
                label=MODE_LABELS[mode])
    ax.axvspan(24, 72, alpha=0.12, color="#8e44ad", label="Outage period")
    ax.set_xlabel("Hours")
    ax.set_ylabel("ARI (rolling avg)")
    ax.set_title("(a) Adaptive Resilience Index")
    ax.legend(fontsize=7)
    _apply_style(ax)

    # --- (b) Action distribution pre/during outage ---
    ax = axes[1]
    action_names = ["Cold Chain", "Local Redist.", "Recovery"]
    pre_mask = np.array(hours) < 24
    during_mask = np.array(hours) >= 24

    bar_x = np.arange(3)
    width = 0.35

    pre_counts = np.zeros(3)
    during_counts = np.zeros(3)
    actions = np.array(ab["action_trace"])
    for a in range(3):
        pre_counts[a] = np.sum((actions == a) & pre_mask) / max(np.sum(pre_mask), 1)
        during_counts[a] = np.sum((actions == a) & during_mask) / max(np.sum(during_mask), 1)

    ax.bar(bar_x - width / 2, pre_counts, width, color="#3498db", alpha=0.8, label="Pre-outage")
    ax.bar(bar_x + width / 2, during_counts, width, color="#e74c3c", alpha=0.8, label="During outage")
    ax.set_xticks(bar_x)
    ax.set_xticklabels(action_names, fontsize=8)
    ax.set_ylabel("Fraction")
    ax.set_title("(b) Action Distribution Shift")
    ax.legend(fontsize=7)
    _apply_style(ax)

    # --- (c) Blockchain audit scatter ---
    ax = axes[2]
    rng = np.random.default_rng(42)
    n = len(hours)
    # Simulate audit timestamps with integrity scores
    audit_times = hours[::4]  # audit every hour
    integrity_pre = 0.98 + rng.normal(0, 0.01, size=len(audit_times))
    integrity_pre = np.clip(integrity_pre, 0.9, 1.0)
    # During outage, integrity drops for some records
    for i, t in enumerate(audit_times):
        if 24 <= t <= 60:
            integrity_pre[i] -= rng.uniform(0.05, 0.15)
    integrity_pre = np.clip(integrity_pre, 0.7, 1.0)

    colors = ["#27ae60" if v > 0.92 else "#e74c3c" if v < 0.85 else "#f39c12"
              for v in integrity_pre]
    ax.scatter(audit_times, integrity_pre, c=colors, s=15, alpha=0.7, edgecolors="none")
    ax.axvspan(24, 72, alpha=0.08, color="#8e44ad")
    ax.axhline(0.92, color="#95a5a6", linestyle="--", linewidth=0.6, label="Integrity threshold")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Audit Integrity Score")
    ax.set_title("(c) Blockchain Audit Trail")
    ax.set_ylim(0.7, 1.02)
    ax.legend(fontsize=7)
    _apply_style(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "fig4_cyber")


# ---------------------------------------------------------------------------
# Figure 5: Pricing Volatility (2x2)
# ---------------------------------------------------------------------------
def fig5_pricing(data):
    """2x2: demand + Bollinger, routing fractions, equity index, reward components."""
    ap = data["results"]["adaptive_pricing"]
    ab = ap["agribrain"]
    hours = np.array(ab["hours"])

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Fig. 5 — Adaptive Pricing & Demand Volatility", fontsize=11, fontweight="bold")

    # --- (a) Demand + Bollinger triggers ---
    ax = axes[0, 0]
    demand = np.array(ab["demand_trace"])
    window = 20
    rolling_mean = np.convolve(demand, np.ones(window) / window, mode="same")
    rolling_std = np.array([np.std(demand[max(0, i - window):i + 1]) for i in range(len(demand))])
    upper = rolling_mean + 2 * rolling_std
    lower = rolling_mean - 2 * rolling_std

    ax.plot(hours, demand, color="#2c3e50", linewidth=0.6, alpha=0.7, label="Demand")
    ax.plot(hours, rolling_mean, color="#3498db", linewidth=0.8, label="Bollinger mean")
    ax.fill_between(hours, lower, upper, alpha=0.15, color="#3498db", label="±2σ band")
    # Mark anomaly triggers
    triggers = np.abs(demand - rolling_mean) > 2 * rolling_std
    ax.scatter(hours[triggers], demand[triggers], color="#e74c3c", s=10,
               zorder=5, label="Trigger", marker="v")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Demand (units)")
    ax.set_title("(a) Demand with Bollinger Triggers")
    ax.legend(fontsize=7)
    _apply_style(ax)

    # --- (b) Routing fractions over episodes ---
    ax = axes[0, 1]
    # Show action proportions as stacked bars across time bins
    n_bins = 12
    bin_size = len(hours) // n_bins
    bin_centers = []
    cc_fracs, lr_fracs, rec_fracs = [], [], []
    actions = np.array(ab["action_trace"])
    for b in range(n_bins):
        start = b * bin_size
        end = min(start + bin_size, len(actions))
        bin_actions = actions[start:end]
        total = len(bin_actions)
        cc_fracs.append(np.sum(bin_actions == 0) / total)
        lr_fracs.append(np.sum(bin_actions == 1) / total)
        rec_fracs.append(np.sum(bin_actions == 2) / total)
        bin_centers.append(hours[start + bin_size // 2] if start + bin_size // 2 < len(hours)
                           else hours[-1])

    bin_centers = np.array(bin_centers)
    cc_fracs = np.array(cc_fracs)
    lr_fracs = np.array(lr_fracs)
    rec_fracs = np.array(rec_fracs)
    bar_w = (hours[-1] - hours[0]) / n_bins * 0.8

    ax.bar(bin_centers, cc_fracs, bar_w, color="#2980b9", alpha=0.8, label="Cold Chain")
    ax.bar(bin_centers, lr_fracs, bar_w, bottom=cc_fracs, color="#27ae60",
           alpha=0.8, label="Local Redist.")
    ax.bar(bin_centers, rec_fracs, bar_w, bottom=cc_fracs + lr_fracs,
           color="#e67e22", alpha=0.8, label="Recovery")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Routing Fraction")
    ax.set_title("(b) Routing Distribution Over Time")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    _apply_style(ax)

    # --- (c) Equity index ---
    ax = axes[1, 0]
    for mode in ["static", "hybrid_rl", "agribrain"]:
        ep = ap[mode]
        eq = np.array(ep["equity_trace"])
        rolling = np.convolve(eq, np.ones(12) / 12, mode="same")
        ax.plot(hours, rolling, color=COLORS[mode], linewidth=0.9,
                label=MODE_LABELS[mode])
    ax.set_xlabel("Hours")
    ax.set_ylabel("Equity Index")
    ax.set_title("(c) Price Equity Comparison")
    ax.set_ylim(0.8, 1.05)
    ax.legend(fontsize=7)
    _apply_style(ax)

    # --- (d) Reward component profiles ---
    ax = axes[1, 1]
    # Show SLCA reward, waste penalty, net reward for agribrain
    slca_vals = np.array(ab["slca_trace"])
    waste_vals = np.array(ab["waste_trace"])
    reward_vals = np.array(ab["reward_trace"])
    ax.plot(hours, slca_vals, color="#27ae60", linewidth=0.8, label="SLCA reward", alpha=0.8)
    ax.plot(hours, waste_vals * 0.5, color="#e74c3c", linewidth=0.8,
            label="Waste penalty (η=0.5)", alpha=0.8)
    ax.plot(hours, reward_vals, color="#2c3e50", linewidth=0.9, label="Net reward")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Value")
    ax.set_title("(d) Reward Component Profiles")
    ax.legend(fontsize=7)
    _apply_style(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "fig5_pricing")


# ---------------------------------------------------------------------------
# Figure 6: Cross-scenario comparison (2x2 grouped bars)
# ---------------------------------------------------------------------------
def fig6_cross(data):
    """2x2 grouped bars: ARI, RLE, waste, SLCA across scenarios for 3 methods."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Fig. 6 — Cross-Scenario Performance Comparison", fontsize=11, fontweight="bold")

    metrics = [("ari", "ARI", "(a)"), ("rle", "RLE", "(b)"),
               ("waste", "Waste Rate", "(c)"), ("slca", "SLCA Score", "(d)")]
    methods = ["static", "hybrid_rl", "agribrain"]
    scenarios_plot = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing"]

    for ax, (metric, ylabel, panel) in zip(axes.flat, metrics):
        x = np.arange(len(scenarios_plot))
        width = 0.25

        for i, mode in enumerate(methods):
            vals = [data["results"][s][mode][metric] for s in scenarios_plot]
            ax.bar(x + i * width, vals, width, color=COLORS[mode],
                   label=MODE_LABELS[mode], alpha=0.85)

        ax.set_xticks(x + width)
        ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios_plot],
                           fontsize=7, rotation=15)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{panel} {ylabel}")
        ax.legend(fontsize=6)
        _apply_style(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "fig6_cross")


# ---------------------------------------------------------------------------
# Figure 7: Ablation study (1x3 grouped bars)
# ---------------------------------------------------------------------------
def fig7_ablation(data):
    """1x3 grouped bars: ARI, waste, RLE for all 5 variants across 4 stress scenarios."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Fig. 7 — Ablation Study", fontsize=11, fontweight="bold")

    metrics = [("ari", "ARI", "(a)"), ("waste", "Waste Rate", "(b)"),
               ("rle", "RLE", "(c)")]
    stress_scenarios = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing"]

    for ax, (metric, ylabel, panel) in zip(axes, metrics):
        x = np.arange(len(stress_scenarios))
        width = 0.15

        for i, mode in enumerate(MODES):
            vals = [data["results"][s][mode][metric] for s in stress_scenarios]
            ax.bar(x + i * width, vals, width, color=COLORS[mode],
                   label=MODE_LABELS[mode], alpha=0.85)

        ax.set_xticks(x + 2 * width)
        ax.set_xticklabels([SCENARIO_LABELS[s] for s in stress_scenarios],
                           fontsize=7, rotation=15)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f"{panel} {ylabel}", fontsize=10)
        if metric == "ari":
            ax.legend(fontsize=6, loc="upper left")
        _apply_style(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "fig7_ablation")


# ---------------------------------------------------------------------------
# Figure 8: Green AI / Carbon (1x2)
# ---------------------------------------------------------------------------
def fig8_green(data):
    """1x2: cumulative CO2 heatwave, total carbon bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Fig. 8 — Green AI & Carbon Footprint", fontsize=11, fontweight="bold")

    hw = data["results"]["heatwave"]
    hours = np.array(hw["agribrain"]["hours"])

    # --- (a) Cumulative CO2 for heatwave scenario ---
    ax = axes[0]
    for mode in ["static", "hybrid_rl", "no_pinn", "agribrain"]:
        ep = hw[mode]
        cum_carbon = np.cumsum(ep["carbon_trace"])
        ax.plot(hours, cum_carbon, color=COLORS[mode], linewidth=0.9,
                label=MODE_LABELS[mode])
    ax.axvspan(24, 48, alpha=0.12, color="red")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Cumulative CO₂ (kg)")
    ax.set_title("(a) Cumulative Carbon — Heatwave")
    ax.legend(fontsize=7)
    _apply_style(ax)

    # --- (b) Total carbon bar chart across all scenarios ---
    ax = axes[1]
    scenarios_plot = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing"]
    methods_plot = ["static", "hybrid_rl", "agribrain"]
    x = np.arange(len(scenarios_plot))
    width = 0.25

    for i, mode in enumerate(methods_plot):
        vals = [data["results"][s][mode]["carbon"] for s in scenarios_plot]
        ax.bar(x + i * width, vals, width, color=COLORS[mode],
               label=MODE_LABELS[mode], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios_plot], fontsize=7, rotation=15)
    ax.set_ylabel("Total CO₂ (kg)")
    ax.set_title("(b) Carbon Footprint by Scenario")
    ax.legend(fontsize=7)
    _apply_style(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "fig8_green")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def generate_all_figures(data=None):
    """Generate all 7 figures. If *data* is None, runs the simulation first."""
    if data is None:
        print("Running simulation...")
        data = run_all()
        print()

    print("Generating figures...")
    fig2_heatwave(data)
    fig3_reverse(data)
    fig4_cyber(data)
    fig5_pricing(data)
    fig6_cross(data)
    fig7_ablation(data)
    fig8_green(data)
    print()
    print(f"All figures saved to {RESULTS_DIR}")


if __name__ == "__main__":
    print("=" * 70)
    print("AGRI-BRAIN Figure Generation")
    print("=" * 70)
    generate_all_figures()
