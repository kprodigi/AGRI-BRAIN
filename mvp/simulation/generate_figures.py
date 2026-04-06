#!/usr/bin/env python3
"""
AGRI-BRAIN Figure Generation
==============================
Generates publication-quality figures (Figure 2 through Figure 10)
as PNG + PDF at 800 DPI.

Standalone usage:
    cd mvp/simulation
    python generate_figures.py

Requires generate_results.py to have been run first (or runs it automatically).
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_SRC = Path(__file__).resolve().parent.parent.parent / "agri-brain-mvp-1.0.0" / "backend"
if str(_BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(_BACKEND_SRC))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from generate_results import run_all, SCENARIOS, MODES, RESULTS_DIR
from src.models.resilience import RLE_THRESHOLD

# ---------------------------------------------------------------------------
# Journal-ready global style (Times New Roman, STIX math, 12pt uniform)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 16,
    "figure.titleweight": "bold",
    "figure.dpi": 150,
    "lines.linewidth": 2.0,
    "lines.markersize": 8,
    "axes.linewidth": 1.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.pad": 4,
    "ytick.major.pad": 4,
})

# ---------------------------------------------------------------------------
# Mandatory color, marker, and line style scheme (journal specification)
# ---------------------------------------------------------------------------
COLORS = {
    "static":     "#808080",   # grey
    "hybrid_rl":  "#E67E22",   # orange
    "no_pinn":    "#E91E63",   # pink
    "no_slca":    "#7570b3",   # purple
    "agribrain":  "#009688",   # teal
    "no_context": "#4CAF50",   # green
    "mcp_only":   "#FF9800",   # amber
    "pirag_only": "#2196F3",   # blue
}

MARKERS = {
    "static":     "o",         # circle
    "hybrid_rl":  "s",         # square
    "no_pinn":    "v",         # triangle down
    "no_slca":    "D",         # diamond
    "agribrain":  "^",         # triangle up
    "no_context": "P",         # plus (filled)
    "mcp_only":   "X",         # x (filled)
    "pirag_only": "d",         # thin diamond
}

LINESTYLES = {
    "static":     "-",                       # solid
    "hybrid_rl":  "--",                      # dashed
    "no_pinn":    (0, (3, 1, 1, 1)),         # dash-dot-dot
    "no_slca":    ":",                       # dotted
    "agribrain":  "-.",                      # dash-dot
    "no_context": (0, (5, 2)),               # long dash
    "mcp_only":   (0, (3, 1, 1, 1, 1, 1)),  # dash-dot-dot-dot
    "pirag_only": (0, (1, 1)),               # dotted tight
}

MODE_LABELS = {
    "static":     "Static",
    "hybrid_rl":  "Hybrid RL",
    "no_pinn":    "No PINN",
    "no_slca":    "No SLCA",
    "agribrain":  "AgriBrain",
    "no_context": "No Context",
    "mcp_only":   "MCP Only",
    "pirag_only": "piRAG Only",
}

SCENARIO_LABELS = {
    "heatwave":         "Heatwave",
    "overproduction":   "Overproduction",
    "cyber_outage":     "Cyber Outage",
    "adaptive_pricing": "Price Volatility",
    "baseline":         "Baseline",
}

DPI = 800
MARKER_EVERY = 15


def _apply_style(ax):
    """Apply journal-quality styling to an axes object."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, color="lightgray", linewidth=0.5)
    ax.tick_params(labelsize=12)
    ax.xaxis.label.set_size(12)
    ax.xaxis.label.set_weight("bold")
    ax.yaxis.label.set_size(12)
    ax.yaxis.label.set_weight("bold")
    ax.title.set_size(14)
    ax.title.set_weight("bold")


def _mode_plot(ax, hours, y, mode, **kwargs):
    """Plot a mode's trace with consistent color, marker, and linestyle."""
    ax.plot(
        hours, y,
        color=COLORS[mode],
        marker=MARKERS[mode],
        linestyle=LINESTYLES[mode],
        markevery=MARKER_EVERY,
        markersize=8,
        linewidth=2.0,
        label=MODE_LABELS[mode],
        **kwargs,
    )


def _legend(ax, **kwargs):
    """Add a styled legend with no overlap."""
    defaults = dict(fontsize=12, framealpha=0.95, edgecolor="gray",
                    fancybox=False, shadow=False)
    defaults.update(kwargs)
    ax.legend(**defaults)


def _save(fig, name):
    """Save figure as PNG (800 DPI) and PDF (vector)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = RESULTS_DIR / f"{name}.{ext}"
        fig.savefig(str(path), dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"  Saved {name}.png / .pdf")
    plt.close(fig)


def _annotate_window(ax, x0, x1, color, label, alpha=0.12, ypos=0.95):
    """Add a shaded scenario window with text annotation."""
    ax.axvspan(x0, x1, alpha=alpha, color=color, zorder=0)
    ylim = ax.get_ylim()
    ax.text(
        (x0 + x1) / 2, ylim[0] + ypos * (ylim[1] - ylim[0]),
        label, ha="center", va="top", fontsize=10,
        fontstyle="italic", color=color, alpha=0.8,
    )


# ---------------------------------------------------------------------------
# Figure 2: Heatwave scenario deep-dive (2x2)
# ---------------------------------------------------------------------------
def fig2_heatwave(data):
    """2x2: temp+humidity, observed spoilage risk, action probs, per-step reward."""
    hw = data["results"]["heatwave"]
    ab = hw["agribrain"]
    hours = np.array(ab["hours"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Heatwave Scenario Analysis", fontsize=18, fontweight="bold")

    # --- (a) Temperature + Humidity with heatwave window ---
    ax = axes[0, 0]
    ax.plot(hours, ab["temp_trace"], color="#c0392c", linewidth=1.2, label="Temperature (\u00b0C)")
    ax2 = ax.twinx()
    ax2.plot(hours, ab["rh_trace"], color="#2980b9", linewidth=1.0, alpha=0.7, label="RH (%)")
    ax.axvspan(24, 48, alpha=0.15, color="red", zorder=0)
    ax.set_xlabel("Hours")
    ax.set_ylabel("Temperature (\u00b0C)")
    ax2.set_ylabel("Relative Humidity (%)")
    ax.set_title("(a) Environmental Exposure")
    ax.legend(loc="upper left", fontsize=12, framealpha=0.9, edgecolor="gray")
    ax2.legend(loc="upper right", fontsize=12, framealpha=0.9, edgecolor="gray")
    _apply_style(ax)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(labelsize=12)
    ylims = ax.get_ylim()
    ax.text(36, ylims[0] + 0.08 * (ylims[1] - ylims[0]),
            "Heatwave", ha="center", fontsize=10, fontstyle="italic",
            color="red", alpha=0.8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      alpha=0.7, edgecolor="none"))

    # --- (b) Observed spoilage risk trajectory ---
    # All modes observe the same environmental rho from the scenario data.
    # Only PINN-enhanced modes (AGRI-BRAIN, No SLCA) use the PINN model
    # for predictive routing, but the observed trajectory is identical.
    ax = axes[0, 1]
    rho = np.array(ab["rho_trace"])
    ax.plot(hours, rho, color="#1b9e77", linewidth=1.4, label="Observed \u03c1(t)")
    ax.axhline(RLE_THRESHOLD, color="#e74c3c", linestyle="--", linewidth=0.8,
               alpha=0.6, label=f"RLE threshold (\u03c1={RLE_THRESHOLD})")
    ax.axvspan(24, 48, alpha=0.12, color="red", zorder=0)
    ax.set_xlabel("Hours")
    ax.set_ylabel("Spoilage Risk \u03c1(t)")
    ax.set_title("(b) Spoilage Risk Trajectory")
    _legend(ax)
    _apply_style(ax)

    # --- (c) Action probability stacked area (AGRI-BRAIN) ---
    ax = axes[1, 0]
    probs = np.array(ab["prob_trace"])
    ax.fill_between(hours, 0, probs[:, 0],
                    color="#2980b9", alpha=0.7, label="Cold Chain")
    ax.fill_between(hours, probs[:, 0], probs[:, 0] + probs[:, 1],
                    color="#27ae60", alpha=0.7, label="Local Redist.")
    ax.fill_between(hours, probs[:, 0] + probs[:, 1], 1.0,
                    color="#e67e22", alpha=0.7, label="Recovery")
    ax.axvspan(24, 48, alpha=0.12, color="red", zorder=0)
    ax.set_xlabel("Hours")
    ax.set_ylabel("Action Probability")
    ax.set_title("(c) AGRI-BRAIN Action Probabilities")
    ax.set_ylim(0, 1)
    _legend(ax, loc="center right")
    _apply_style(ax)

    # --- (d) Per-step reward (rolling average) ---
    # Shows actual reward rate, making heatwave-induced dips clearly visible.
    ax = axes[1, 1]
    window = 12  # 3-hour rolling window
    for mode in ["static", "hybrid_rl", "agribrain"]:
        ep = hw[mode]
        reward = np.array(ep["reward_trace"])
        rolling = np.convolve(reward, np.ones(window) / window, mode="same")
        _mode_plot(ax, hours, rolling, mode)
    ax.axvspan(24, 48, alpha=0.12, color="red", zorder=0)
    ax.set_xlabel("Hours")
    ax.set_ylabel("Reward per Step (rolling avg)")
    ax.set_title("(d) Reward Rate During Heatwave")
    _legend(ax, loc="lower left")
    _apply_style(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, "fig2_heatwave")


# ---------------------------------------------------------------------------
# Figure 3: Overproduction / Reverse Logistics (2x2)
# ---------------------------------------------------------------------------
def fig3_overproduction(data):
    """2x2: inventory vs demand (dual axis), waste, RLE with annotation, SLCA bars."""
    op = data["results"]["overproduction"]
    ab = op["agribrain"]
    hours = np.array(ab["hours"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Overproduction & Reverse Logistics",
                 fontsize=18, fontweight="bold")

    # --- (a) Inventory vs Demand (dual y-axis, proper units) ---
    ax = axes[0, 0]
    inv = np.array(ab["inventory_trace"])
    dem = np.array(ab["demand_trace"])
    ax.plot(hours, inv, color="#27ae60", linewidth=1.2, label="Inventory (units)")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Inventory (units)")
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(3, 3))
    ax2 = ax.twinx()
    ax2.plot(hours, dem, color="#e74c3c", linewidth=1.0, alpha=0.8,
             label="Demand (units/step)")
    ax2.set_ylabel("Demand (units/step)")
    # Surplus fill (need to normalize for comparison)
    ax.axvspan(12, 60, alpha=0.08, color="#f39c12", zorder=0)
    ylims = ax.get_ylim()
    ax.text(0.5, 0.8, "Overproduction",
        transform=ax.transAxes, ha="center", va="top", fontsize=11,
        fontstyle="italic", color="#e67e22", alpha=0.9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  alpha=0.8, edgecolor="none"))
    ax.set_title("(a) Inventory vs Demand")
    ax.legend(loc="upper left", fontsize=12, framealpha=0.9, edgecolor="gray")
    ax2.legend(loc="upper right", fontsize=12, framealpha=0.9, edgecolor="gray")
    _apply_style(ax)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(labelsize=12)

    # --- (b) Waste rolling average ---
    ax = axes[0, 1]
    window = 12
    for mode in ["static", "hybrid_rl", "agribrain"]:
        ep = op[mode]
        waste = np.array(ep["waste_trace"])
        rolling = np.convolve(waste, np.ones(window) / window, mode="same")
        _mode_plot(ax, hours, rolling, mode)
    ax.axvspan(12, 60, alpha=0.08, color="#f39c12", zorder=0)
    ax.set_xlabel("Hours")
    ax.set_ylabel("Waste Rate (rolling avg)")
    ax.set_title("(b) Waste Reduction Over Time")
    _legend(ax)
    _apply_style(ax)

    # --- (c) RLE rolling with threshold onset annotation ---
    ax = axes[1, 0]
    for mode in ["static", "hybrid_rl", "agribrain"]:
        ep = op[mode]
        rho = np.array(ep["rho_trace"])
        actions = np.array(ep["action_trace"])
        at_risk = rho > RLE_THRESHOLD
        routed = at_risk & (actions >= 1)
        rle_rolling = np.convolve(routed.astype(float),
                                  np.ones(window) / window, mode="same")
        rle_denom = np.convolve(at_risk.astype(float),
                                np.ones(window) / window, mode="same")
        rle_frac = np.divide(rle_rolling, rle_denom,
                             out=np.zeros_like(rle_rolling), where=rle_denom > 0)
        _mode_plot(ax, hours, rle_frac, mode)

    # Find when rho first exceeds threshold and annotate
    rho_ab = np.array(ab["rho_trace"])
    threshold_idx = np.argmax(rho_ab > RLE_THRESHOLD)
    if threshold_idx > 0 or rho_ab[0] > RLE_THRESHOLD:
        threshold_hour = hours[threshold_idx]
        ax.axvline(threshold_hour, color="#7f8c8d", linestyle="--", linewidth=0.8,
                   alpha=0.7)
        ax.text(threshold_hour + 1, 0.50,
                f"\u03c1 > {RLE_THRESHOLD}\n(h={threshold_hour:.0f})",
                fontsize=10, color="#7f8c8d", va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          alpha=0.8, edgecolor="none"))

    ax.set_xlabel("Hours")
    ax.set_ylabel("RLE (rolling)")
    ax.set_title("(c) Reverse Logistics Efficiency")
    ax.set_ylim(-0.05, 1.1)
    _legend(ax)
    _apply_style(ax)

    # --- (d) SLCA component grouped bars with std error bars ---
    ax = axes[1, 1]
    components = ["C", "L", "R", "P"]
    comp_labels = ["Carbon", "Labour", "Resilience", "Price Transp."]
    x = np.arange(len(components))
    width = 0.25
    for i, mode in enumerate(["static", "hybrid_rl", "agribrain"]):
        ep = op[mode]
        vals = [np.mean([s[comp] for s in ep["slca_component_trace"]])
                for comp in components]
        stds = [np.std([s[comp] for s in ep["slca_component_trace"]])
                for comp in components]
        ax.bar(x + i * width, vals, width, color=COLORS[mode],
               label=MODE_LABELS[mode], alpha=0.85, edgecolor="white",
               linewidth=0.5, yerr=stds, capsize=3,
               error_kw={"linewidth": 0.8, "capthick": 0.8})
    ax.set_xticks(x + width)
    ax.set_xticklabels(comp_labels, fontsize=12)
    ax.set_ylabel("Score")
    ax.set_title("(d) SLCA Components")
    ax.set_ylim(0, 1.1)
    _legend(ax)
    _apply_style(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, "fig3_overproduction")


# ---------------------------------------------------------------------------
# Figure 4: Cyber Outage (1x3)
# ---------------------------------------------------------------------------
def fig4_cyber(data):
    """1x3: ARI over time with outage, action distribution, blockchain audit."""
    cy = data["results"]["cyber_outage"]
    ab = cy["agribrain"]
    hours = np.array(ab["hours"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Cyber Outage Scenario", fontsize=16, fontweight="bold")

    # --- (a) ARI over time with outage shading ---
    ax = axes[0]
    for mode in ["static", "hybrid_rl", "agribrain"]:
        ep = cy[mode]
        ari = np.array(ep["ari_trace"])
        rolling = np.convolve(ari, np.ones(12) / 12, mode="same")
        _mode_plot(ax, hours, rolling, mode)
    ax.axvspan(24, 72, alpha=0.10, color="#8e44ad", zorder=0)
    ylims = ax.get_ylim()
    ax.text(48, ylims[0] + 0.08 * (ylims[1] - ylims[0]),
            "Outage", ha="center", fontsize=10, fontstyle="italic",
            color="#8e44ad", alpha=0.8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      alpha=0.7, edgecolor="none"))
    ax.set_xlabel("Hours")
    ax.set_ylabel("ARI (rolling avg)")
    ax.set_title("(a) Adaptive Resilience Index")
    _legend(ax, loc="lower left")
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
    n_pre = max(np.sum(pre_mask), 1)
    n_during = max(np.sum(during_mask), 1)
    for a in range(3):
        pre_counts[a] = np.sum((actions == a) & pre_mask) / n_pre
        during_counts[a] = np.sum((actions == a) & during_mask) / n_during
    # Binomial standard error for proportions
    pre_se = np.sqrt(pre_counts * (1 - pre_counts) / n_pre)
    during_se = np.sqrt(during_counts * (1 - during_counts) / n_during)

    ax.bar(bar_x - width / 2, pre_counts, width, color="#3498db",
           alpha=0.8, label="Pre-outage", edgecolor="white", linewidth=0.5,
           yerr=1.96 * pre_se, capsize=4, error_kw={"linewidth": 1.0, "capthick": 1.0})
    ax.bar(bar_x + width / 2, during_counts, width, color="#e74c3c",
           alpha=0.8, label="During outage", edgecolor="white", linewidth=0.5,
           yerr=1.96 * during_se, capsize=4, error_kw={"linewidth": 1.0, "capthick": 1.0})
    ax.set_xticks(bar_x)
    ax.set_xticklabels(action_names, fontsize=12)
    ax.set_ylabel("Fraction")
    ax.set_title("(b) Action Distribution Shift")
    _legend(ax)
    _apply_style(ax)

    # --- (c) Policy confidence scatter ---
    ax = axes[2]
    probs = np.array(ab.get("prob_trace", []), dtype=float)
    if probs.size and probs.ndim == 2 and probs.shape[1] > 1:
        eps = 1e-12
        entropy = -np.sum(probs * np.log(np.clip(probs, eps, 1.0)), axis=1)
        max_entropy = np.log(probs.shape[1])
        confidence = 1.0 - np.clip(entropy / max(max_entropy, eps), 0.0, 1.0)
        audit_times = hours[: len(confidence)][::4]
        confidence_scores = confidence[::4]
    else:
        audit_times = hours[::4]
        confidence_scores = np.zeros_like(audit_times, dtype=float)

    colors = ["#27ae60" if v > 0.66 else "#e74c3c" if v < 0.33 else "#f39c12"
              for v in confidence_scores]
    ax.scatter(audit_times, confidence_scores, c=colors, s=18, alpha=0.7,
               edgecolors="none")
    ax.axvspan(24, 72, alpha=0.08, color="#8e44ad", zorder=0)
    ax.axhline(0.66, color="#95a5a6", linestyle="--", linewidth=0.8,
               label="High-confidence band")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Decision Confidence\n(1 - normalized entropy)")
    ax.set_title("(c) Policy Confidence Trace")
    ax.set_ylim(-0.02, 1.02)
    _legend(ax, loc="lower left")
    _apply_style(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "fig4_cyber")


# ---------------------------------------------------------------------------
# Figure 5: Pricing Volatility (2x2)
# ---------------------------------------------------------------------------
def fig5_pricing(data):
    """2x2: demand+Bollinger, routing fractions, equity, reward components."""
    ap = data["results"]["adaptive_pricing"]
    ab = ap["agribrain"]
    hours = np.array(ab["hours"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Adaptive Pricing & Demand Volatility",
                 fontsize=18, fontweight="bold")

    # --- (a) Demand + Bollinger triggers ---
    ax = axes[0, 0]
    demand = np.array(ab["demand_trace"])
    window = 20
    rolling_mean = np.convolve(demand, np.ones(window) / window, mode="same")
    rolling_std = np.array([np.std(demand[max(0, i - window):i + 1])
                            for i in range(len(demand))])
    upper = rolling_mean + 2 * rolling_std
    lower = rolling_mean - 2 * rolling_std

    ax.plot(hours, demand, color="#2c3e50", linewidth=0.6, alpha=0.7, label="Demand")
    ax.plot(hours, rolling_mean, color="#3498db", linewidth=1.0, label="Bollinger mean")
    ax.fill_between(hours, lower, upper, alpha=0.12, color="#3498db",
                    label="\u00b12\u03c3 band")
    triggers = np.abs(demand - rolling_mean) > 2 * rolling_std
    ax.scatter(hours[triggers], demand[triggers], color="#e74c3c", s=12,
               zorder=5, label="Trigger", marker="v")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Demand (units/step)")
    ax.set_title("(a) Demand with Bollinger Triggers")
    _legend(ax)
    _apply_style(ax)

    # --- (b) Routing fractions over time bins ---
    ax = axes[0, 1]
    n_bins = 12
    bin_size = len(hours) // n_bins
    bin_centers, cc_fracs, lr_fracs, rec_fracs = [], [], [], []
    actions = np.array(ab["action_trace"])
    for b in range(n_bins):
        start = b * bin_size
        end = min(start + bin_size, len(actions))
        bin_actions = actions[start:end]
        total = len(bin_actions)
        cc_fracs.append(np.sum(bin_actions == 0) / total)
        lr_fracs.append(np.sum(bin_actions == 1) / total)
        rec_fracs.append(np.sum(bin_actions == 2) / total)
        mid = min(start + bin_size // 2, len(hours) - 1)
        bin_centers.append(hours[mid])

    bin_centers = np.array(bin_centers)
    cc_fracs = np.array(cc_fracs)
    lr_fracs = np.array(lr_fracs)
    rec_fracs = np.array(rec_fracs)
    bar_w = (hours[-1] - hours[0]) / n_bins * 0.8

    ax.bar(bin_centers, cc_fracs, bar_w, color="#2980b9", alpha=0.8,
           label="Cold Chain", edgecolor="white", linewidth=0.5)
    ax.bar(bin_centers, lr_fracs, bar_w, bottom=cc_fracs, color="#27ae60",
           alpha=0.8, label="Local Redist.", edgecolor="white", linewidth=0.5)
    ax.bar(bin_centers, rec_fracs, bar_w, bottom=cc_fracs + lr_fracs,
           color="#e67e22", alpha=0.8, label="Recovery", edgecolor="white",
           linewidth=0.5)
    ax.set_xlabel("Hours")
    ax.set_ylabel("Routing Fraction")
    ax.set_title("(b) Routing Distribution Over Time")
    ax.set_ylim(0, 1.05)
    _legend(ax)
    _apply_style(ax)

    # --- (c) Equity index ---
    ax = axes[1, 0]
    for mode in ["static", "hybrid_rl", "agribrain"]:
        ep = ap[mode]
        eq = np.array(ep["equity_trace"])
        rolling = np.convolve(eq, np.ones(12) / 12, mode="same")
        _mode_plot(ax, hours, rolling, mode)
    ax.set_xlabel("Hours")
    ax.set_ylabel("Equity Index")
    ax.set_title("(c) Price Equity Comparison")
    ax.set_ylim(0.70, 1.02)
    _legend(ax)
    _apply_style(ax)

    # --- (d) Reward decomposition: SLCA vs waste penalty ---
    ax = axes[1, 1]
    slca_vals = np.array(ab["slca_trace"])
    waste_vals = np.array(ab["waste_trace"])
    reward_vals = np.array(ab["reward_trace"])
    window = 12  # 3-hour rolling window for readability
    slca_smooth = np.convolve(slca_vals, np.ones(window) / window, mode="same")
    # Show waste penalty on its own y-scale using twin axis
    reward_smooth = np.convolve(reward_vals, np.ones(window) / window, mode="same")
    l1, = ax.plot(hours, slca_smooth, color="#27ae60", linewidth=1.5,
                  label="SLCA reward", alpha=0.9)
    l2, = ax.plot(hours, reward_smooth, color="#2c3e50", linewidth=1.5,
                  label="Net reward", alpha=0.9)
    ax.set_xlabel("Hours")
    ax.set_ylabel("SLCA / Net Reward")
    ax.set_ylim(0.0, 1.0)
    _apply_style(ax)
    # Twin axis for waste penalty at appropriate scale
    ax2 = ax.twinx()
    waste_smooth = np.convolve(waste_vals, np.ones(window) / window, mode="same")
    l3, = ax2.plot(hours, waste_smooth * 100, color="#e74c3c", linewidth=1.5,
                   linestyle="--", label="Waste rate (%)", alpha=0.9)
    ax2.set_ylabel("Waste Rate (%)", color="#e74c3c")
    ax2.tick_params(axis="y", labelcolor="#e74c3c")
    ax.set_title("(d) Reward Decomposition")
    # Combined legend — one box, bottom right
    ax.legend(handles=[l1, l2, l3], loc="lower right", fontsize=10,
              framealpha=0.95, edgecolor="gray", fancybox=False)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, "fig5_pricing")


# ---------------------------------------------------------------------------
# Figure 6: Cross-scenario comparison (2x2 grouped bars)
# ---------------------------------------------------------------------------
def _load_benchmark_ci() -> dict | None:
    """Load benchmark_summary.json for CI error bars (returns None if unavailable)."""
    bench_file = RESULTS_DIR / "benchmark_summary.json"
    if not bench_file.exists():
        return None
    import json
    return json.loads(bench_file.read_text(encoding="utf-8"))


def _benchmark_complete(bench: dict | None, scenarios: list[str], modes: list[str], metric: str) -> bool:
    if not bench:
        return False
    for s in scenarios:
        for m in modes:
            rec = bench.get(s, {}).get(m, {}).get(metric, {})
            if not rec or any(k not in rec for k in ("mean", "ci_low", "ci_high")):
                return False
    return True


def fig6_cross(data):
    """2x2 grouped bars: ARI, RLE, waste, SLCA across scenarios for 3 methods.
    Adds error bars from benchmark_summary.json when available."""
    bench = _load_benchmark_ci()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Cross-Scenario Performance Comparison",
                 fontsize=18, fontweight="bold")

    metrics = [("ari", "ARI", "(a)"), ("rle", "RLE", "(b)"),
               ("waste", "Waste Rate", "(c)"), ("slca", "SLCA Score", "(d)")]
    methods = ["static", "hybrid_rl", "agribrain"]
    scenarios_plot = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing"]

    for ax, (metric, ylabel, panel) in zip(axes.flat, metrics):
        x = np.arange(len(scenarios_plot))
        width = 0.25

        for i, mode in enumerate(methods):
            vals = [data["results"][s][mode][metric] for s in scenarios_plot]
            # Try to add error bars from benchmark data
            yerr = None
            if _benchmark_complete(bench, scenarios_plot, methods, metric):
                ci_data = []
                for s in scenarios_plot:
                    m_data = bench.get(s, {}).get(mode, {}).get(metric, {})
                    ci_data.append((m_data["mean"], m_data["ci_low"], m_data["ci_high"]))
                means = np.array([c[0] for c in ci_data])
                lows = np.array([c[1] for c in ci_data])
                highs = np.array([c[2] for c in ci_data])
                yerr = np.vstack([means - lows, highs - means])
                vals = means.tolist()

            ax.bar(x + i * width, vals, width, color=COLORS[mode],
                   label=MODE_LABELS[mode], alpha=0.85, edgecolor="white",
                   linewidth=0.5, yerr=yerr,
                   capsize=3 if yerr is not None else 0,
                   error_kw={"linewidth": 1.0, "capthick": 1.0})

        ax.set_xticks(x + width)
        ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios_plot],
                           fontsize=12, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{panel} {ylabel}")
        _apply_style(ax)

    # Single legend at the bottom, shared across all subplots
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(methods),
               fontsize=12, framealpha=0.95, edgecolor="gray",
               fancybox=False, shadow=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    _save(fig, "fig6_cross")


# ---------------------------------------------------------------------------
# Figure 7: Ablation study (1x3 grouped bars)
# ---------------------------------------------------------------------------
def fig7_ablation(data):
    """1x3 grouped bars: ARI, waste, RLE for all 8 variants.
    Adds error bars from benchmark_summary.json when available.
    AGRI-BRAIN is always the last bar in each group."""
    bench = _load_benchmark_ci()

    # Reorder so AGRI-BRAIN is last
    fig7_modes = [m for m in MODES if m != "agribrain"] + ["agribrain"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Ablation Study", fontsize=16, fontweight="bold")

    metrics = [("ari", "ARI", "(a)"), ("waste", "Waste Rate", "(b)"),
               ("rle", "RLE", "(c)")]
    stress_scenarios = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing"]

    for ax, (metric, ylabel, panel) in zip(axes, metrics):
        x = np.arange(len(stress_scenarios))
        width = 0.10

        for i, mode in enumerate(fig7_modes):
            vals = [data["results"][s][mode][metric] for s in stress_scenarios]
            yerr = None
            if _benchmark_complete(bench, stress_scenarios, fig7_modes, metric):
                ci_data = []
                for s in stress_scenarios:
                    m_data = bench.get(s, {}).get(mode, {}).get(metric, {})
                    ci_data.append((m_data["mean"], m_data["ci_low"], m_data["ci_high"]))
                means = np.array([c[0] for c in ci_data])
                lows = np.array([c[1] for c in ci_data])
                highs = np.array([c[2] for c in ci_data])
                yerr = np.vstack([means - lows, highs - means])
                vals = means.tolist()

            ax.bar(x + i * width, vals, width, color=COLORS[mode],
                   label=MODE_LABELS[mode], alpha=0.85, edgecolor="white",
                   linewidth=0.5, yerr=yerr,
                   capsize=2 if yerr is not None else 0,
                   error_kw={"linewidth": 0.8, "capthick": 0.8})

        ax.set_xticks(x + 3.5 * width)
        ax.set_xticklabels([SCENARIO_LABELS[s] for s in stress_scenarios],
                           fontsize=12, rotation=15, ha="right")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_title(f"{panel} {ylabel}", fontsize=14, fontweight="bold")
        _apply_style(ax)

    # Single legend at the bottom, shared across all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(fig7_modes),
               fontsize=12, framealpha=0.95, edgecolor="gray",
               fancybox=False, shadow=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    _save(fig, "fig7_ablation")


# ---------------------------------------------------------------------------
# Figure 8: Green AI / Carbon (1x2)
# ---------------------------------------------------------------------------
def fig8_green_ai(data):
    """1x2: cumulative CO2 heatwave, total carbon bar chart with CI error bars."""
    bench = _load_benchmark_ci()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Green AI & Carbon Footprint",
                 fontsize=14, fontweight="bold")

    hw = data["results"]["heatwave"]
    hours = np.array(hw["agribrain"]["hours"])

    # --- (a) Cumulative CO2 for heatwave scenario ---
    ax = axes[0]
    for mode in ["static", "hybrid_rl", "no_pinn", "agribrain"]:
        ep = hw[mode]
        cum_carbon = np.cumsum(ep["carbon_trace"])
        _mode_plot(ax, hours, cum_carbon, mode)
    ax.axvspan(24, 48, alpha=0.12, color="red", zorder=0)
    ylims = ax.get_ylim()
    ax.text(36, ylims[0] + 0.10 * (ylims[1] - ylims[0]),
            "Heatwave", ha="center", fontsize=10, fontstyle="italic",
            color="red", alpha=0.8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      alpha=0.7, edgecolor="none"))
    ax.set_xlabel("Hours")
    ax.set_ylabel("Cumulative CO\u2082 (kg)")
    ax.set_title("(a) Cumulative Carbon \u2014 Heatwave")
    _legend(ax)
    _apply_style(ax)

    # --- (b) Total carbon bar chart across all scenarios ---
    ax = axes[1]
    scenarios_plot = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing"]
    methods_plot = ["static", "hybrid_rl", "agribrain"]
    x = np.arange(len(scenarios_plot))
    width = 0.25

    for i, mode in enumerate(methods_plot):
        vals = [data["results"][s][mode]["carbon"] for s in scenarios_plot]
        yerr = None
        if _benchmark_complete(bench, scenarios_plot, methods_plot, "carbon"):
            ci_data = []
            for s in scenarios_plot:
                m_data = bench.get(s, {}).get(mode, {}).get("carbon", {})
                ci_data.append((m_data["mean"], m_data["ci_low"], m_data["ci_high"]))
            means = np.array([c[0] for c in ci_data])
            lows = np.array([c[1] for c in ci_data])
            highs = np.array([c[2] for c in ci_data])
            yerr = np.vstack([means - lows, highs - means])
            vals = means.tolist()
        ax.bar(x + i * width, vals, width, color=COLORS[mode],
               label=MODE_LABELS[mode], alpha=0.85, edgecolor="white",
               linewidth=0.5, yerr=yerr,
               capsize=3 if yerr is not None else 0,
               error_kw={"linewidth": 1.0, "capthick": 1.0})

    ax.set_xticks(x + width)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios_plot],
                       fontsize=12, rotation=15, ha="right")
    ax.set_ylabel("Total CO\u2082 (kg)")
    ax.set_title("(b) Carbon Footprint by Scenario")
    _legend(ax)
    _apply_style(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "fig8_green_ai")


def fig9_mcp_pirag_robustness():
    """Robustness figure from protocol logs and benchmark summaries."""
    proto_files = [RESULTS_DIR / f"mcp_protocol_{s}.json" for s in SCENARIOS]
    bench_file = RESULTS_DIR / "benchmark_summary.json"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("MCP + piRAG Robustness", fontsize=14, fontweight="bold")

    # (a) Protocol error counts by scenario
    ax = axes[0]
    scenarios = []
    errs = []
    calls = []
    for s, p in zip(SCENARIOS, proto_files):
        if not p.exists():
            continue
        import json
        rows = json.loads(p.read_text(encoding="utf-8"))
        scenarios.append(SCENARIO_LABELS.get(s, s))
        errs.append(sum(1 for r in rows if r.get("response", {}).get("error")))
        calls.append(len(rows))
    if scenarios:
        x = np.arange(len(scenarios))
        ax.bar(x, calls, color="#90caf9", label="Total calls")
        ax.bar(x, errs, color="#ef5350", label="Errors")
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=20, ha="right")
    ax.set_title("(a) MCP Call Reliability")
    ax.set_ylabel("Count")
    _legend(ax)
    _apply_style(ax)

    # (b) ARI confidence intervals — all context ablation methods
    ax = axes[1]
    ci_methods = [
        ("agribrain", "#009688", "AgriBrain"),
        ("pirag_only", "#2196F3", "piRAG Only"),
        ("mcp_only", "#FF9800", "MCP Only"),
        ("no_context", "#4CAF50", "No Context"),
    ]
    if bench_file.exists():
        import json
        bench = json.loads(bench_file.read_text(encoding="utf-8"))
        scenarios_plot = [s for s in SCENARIOS if s != "baseline"]
        x = np.arange(len(scenarios_plot))
        width = 0.18
        for j, (mode, color, label) in enumerate(ci_methods):
            means, lo_err, hi_err = [], [], []
            for s in scenarios_plot:
                ari = bench.get(s, {}).get(mode, {}).get("ari", {})
                m = float(ari.get("mean", 0.0))
                means.append(m)
                lo_err.append(m - float(ari.get("ci_low", m)))
                hi_err.append(float(ari.get("ci_high", m)) - m)
            if means:
                yerr = np.vstack([lo_err, hi_err])
                ax.bar(x + j * width, means, width, color=color, label=label,
                       alpha=0.85, edgecolor="white", linewidth=0.5,
                       yerr=yerr, capsize=3,
                       error_kw={"linewidth": 1.0, "capthick": 1.0})
        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels([SCENARIO_LABELS.get(s, s) for s in scenarios_plot],
                           rotation=20, ha="right")
    ax.set_title("(b) Context Ablation ARI (95% CI)")
    ax.set_ylabel("ARI")
    _legend(ax)
    _apply_style(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "fig9_mcp_pirag_robustness")


def fig10_latency_quality_frontier(data):
    """Latency-quality frontier with two zones and smart label placement."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1]})
    fig.suptitle("Operational Frontier: Quality vs Latency",
                 fontsize=14, fontweight="bold")

    # Split modes into two groups for clarity
    fast_modes = ["static", "hybrid_rl", "no_pinn", "no_slca", "no_context"]
    context_modes = ["agribrain", "mcp_only", "pirag_only"]

    def _collect(modes):
        pts = []
        for mode in modes:
            ari_vals = [data["results"][s][mode]["ari"] for s in SCENARIOS]
            lat_vals = [data["results"][s][mode].get("mean_decision_latency_ms", 0.0) for s in SCENARIOS]
            pts.append((mode, float(np.mean(lat_vals)), float(np.mean(ari_vals))))
        return pts

    fast_pts = _collect(fast_modes)
    ctx_pts = _collect(context_modes)

    # --- (a) Fast modes (sub-millisecond) ---
    ax = axes[0]
    for mode, x, y in fast_pts:
        ax.scatter(x, y, s=180, color=COLORS[mode], marker=MARKERS[mode],
                   edgecolor="black", linewidth=0.6, alpha=0.9, zorder=5)
    # Smart label placement: sort by ARI and stagger
    fast_sorted = sorted(fast_pts, key=lambda p: p[2])
    for i, (mode, x, y) in enumerate(fast_sorted):
        offset_x = 0.015
        offset_y = 0.008 if i % 2 == 0 else -0.012
        ax.annotate(MODE_LABELS[mode], (x, y), xytext=(x + offset_x, y + offset_y),
                    fontsize=11, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              alpha=0.8, edgecolor="none"))
    ax.set_xlabel("Mean decision latency (ms)")
    ax.set_ylabel("Mean ARI")
    ax.set_title("(a) Lightweight Methods (<1 ms)")
    ax.set_xlim(-0.02, max(p[1] for p in fast_pts) * 2.5)
    _apply_style(ax)

    # --- (b) Context-aware modes (MCP/piRAG overhead) ---
    ax = axes[1]
    # Also show no_context as reference
    ref = [p for p in fast_pts if p[0] == "no_context"][0]
    ax.scatter(ref[1], ref[2], s=140, color=COLORS["no_context"], marker=MARKERS["no_context"],
               edgecolor="black", linewidth=0.6, alpha=0.6, zorder=4)
    ax.annotate("No Context\n(reference)", (ref[1], ref[2]),
                xytext=(ref[1] + 0.5, ref[2] - 0.015),
                fontsize=10, fontstyle="italic", color="gray",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    for mode, x, y in ctx_pts:
        ax.scatter(x, y, s=220, color=COLORS[mode], marker=MARKERS[mode],
                   edgecolor="black", linewidth=0.8, alpha=0.9, zorder=5)
    # Labels with offsets to avoid overlap
    label_offsets = {"agribrain": (0.15, 0.008), "mcp_only": (0.15, -0.015), "pirag_only": (-1.5, 0.010)}
    for mode, x, y in ctx_pts:
        dx, dy = label_offsets.get(mode, (0.15, 0))
        ax.annotate(MODE_LABELS[mode], (x, y), xytext=(x + dx, y + dy),
                    fontsize=11, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              alpha=0.8, edgecolor="none"))
    # Draw arrow showing "context overhead"
    ctx_mean_lat = np.mean([p[1] for p in ctx_pts])
    ctx_mean_ari = np.mean([p[2] for p in ctx_pts])
    ax.annotate("", xy=(ctx_mean_lat, ctx_mean_ari),
                xytext=(ref[1], ref[2]),
                arrowprops=dict(arrowstyle="->", color="#009688", lw=2.0,
                                linestyle="--", alpha=0.5))
    mid_x = (ref[1] + ctx_mean_lat) / 2
    mid_y = (ref[2] + ctx_mean_ari) / 2
    ax.text(mid_x, mid_y + 0.012, f"+{ctx_mean_lat - ref[1]:.1f} ms\n+{(ctx_mean_ari - ref[2]):.3f} ARI",
            fontsize=9, ha="center", color="#009688", fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="#009688"))
    ax.set_xlabel("Mean decision latency (ms)")
    ax.set_ylabel("Mean ARI")
    ax.set_title("(b) Context-Aware Methods (MCP/piRAG overhead)")
    _apply_style(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "fig10_latency_quality_frontier")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def generate_all_figures(data=None):
    """Generate all configured figures. If *data* is None, runs simulation first."""
    if data is None:
        print("Running simulation...")
        data = run_all()
        print()

    print("Generating figures...")
    fig2_heatwave(data)
    fig3_overproduction(data)
    fig4_cyber(data)
    fig5_pricing(data)
    fig6_cross(data)
    fig7_ablation(data)
    fig8_green_ai(data)
    fig9_mcp_pirag_robustness()
    fig10_latency_quality_frontier(data)
    print()
    print(f"All figures saved to {RESULTS_DIR}")


if __name__ == "__main__":
    print("=" * 70)
    print("AGRI-BRAIN Figure Generation")
    print("=" * 70)
    generate_all_figures()
