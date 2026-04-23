#!/usr/bin/env python3
"""
AGRI-BRAIN Figure Generation
==============================
Generates publication-quality figures (Figure 2 through Figure 10)
as PNG + PDF at 800 DPI. The shared style block below is the single
source of truth for typography, palette, and layout so that every
figure in the paper, poster, and slide deck matches exactly.

Styling principles (aligned with Word-document manuscript body text):

- **Font**: Arial for every text element. Fallbacks to Liberation Sans
  (Linux metric-compatible) and DejaVu Sans (matplotlib default) so the
  same code path renders identically on HPC and on local Windows.
- **Size**: body text matches 11 pt Word body; axis labels 12 pt bold;
  subplot titles 13 pt bold; figure super-title 16 pt bold; tick numbers
  11 pt; legends 11 pt.
- **Weight**: titles, axis labels, and annotation emphasis are bold. Body
  text and tick labels remain regular so bold carries contrast.
- **Contrast**: ColorBrewer-derived palette with deeper saturation than
  the Material defaults so colors survive grayscale printing and
  projection.
- **No overlap**: window annotations use axes-fraction placement outside
  the plotting area; long scenario names rotate -20 degrees; legends
  prefer bottom-centre placement or outside-right; tight_layout plus
  explicit padding.
- **Image quality**: 800 DPI PNG and vector PDF; explicit bbox_inches
  tight with pad_inches=0.15 to avoid clipping bold labels.

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
# Unified publication-quality style
# ---------------------------------------------------------------------------
# Body text is 11 pt (Word default for Arial body). Titles and axis labels
# use bold at 12-16 pt so the hierarchy is obvious even at reduced sizes in
# a two-column layout. Every figure in this module inherits these rcParams;
# individual figures should not override them unless a specific layout
# demands it, and any such override must be explicitly justified.
BODY_FONT_SIZE = 11
TICK_FONT_SIZE = 11
AXIS_LABEL_SIZE = 12
SUBPLOT_TITLE_SIZE = 13
FIG_TITLE_SIZE = 16
LEGEND_FONT_SIZE = 11
ANNOT_FONT_SIZE = 10

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
    "mathtext.fontset": "dejavusans",
    "font.size": BODY_FONT_SIZE,
    "axes.labelsize": AXIS_LABEL_SIZE,
    "axes.labelweight": "bold",
    "axes.titlesize": SUBPLOT_TITLE_SIZE,
    "axes.titleweight": "bold",
    "axes.titlepad": 10,
    "xtick.labelsize": TICK_FONT_SIZE,
    "ytick.labelsize": TICK_FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
    "legend.title_fontsize": LEGEND_FONT_SIZE,
    "figure.titlesize": FIG_TITLE_SIZE,
    "figure.titleweight": "bold",
    "figure.dpi": 150,
    "savefig.dpi": 800,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "savefig.facecolor": "white",
    "lines.linewidth": 2.2,
    "lines.markersize": 8,
    "axes.linewidth": 1.3,
    "axes.edgecolor": "#1F1F1F",
    "axes.labelpad": 6,
    "xtick.major.width": 1.3,
    "ytick.major.width": 1.3,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.major.pad": 5,
    "ytick.major.pad": 5,
    "xtick.color": "#1F1F1F",
    "ytick.color": "#1F1F1F",
    "grid.color": "#BDBDBD",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.6,
    "patch.linewidth": 1.0,
    "patch.edgecolor": "white",
    "pdf.fonttype": 42,     # TrueType in PDF, not Type 3
    "ps.fonttype": 42,
})

# ---------------------------------------------------------------------------
# High-contrast, colorblind-safe 9-mode palette
# ---------------------------------------------------------------------------
# Chosen so (a) agribrain stays visually the hero in every comparison, and
# (b) adjacent modes in every chart differ on both hue and value (not just
# hue), so the figures survive greyscale print.
COLORS = {
    "static":     "#4A4A4A",   # charcoal (baseline)
    "hybrid_rl":  "#D95F02",   # burnt orange
    "no_pinn":    "#C2185B",   # deep magenta
    "no_slca":    "#5E35B1",   # deep purple
    "agribrain":  "#009688",   # teal (paper hero, unchanged)
    "no_context": "#2E7D32",   # forest green
    "mcp_only":   "#F57C00",   # vivid amber
    "pirag_only": "#1565C0",   # deep blue
    "no_yield":   "#8D6E63",   # warm brown (Path B ablation)
}

MARKERS = {
    "static":     "o",
    "hybrid_rl":  "s",
    "no_pinn":    "v",
    "no_slca":    "D",
    "agribrain":  "^",
    "no_context": "P",
    "mcp_only":   "X",
    "pirag_only": "d",
    "no_yield":   "*",   # 5-point star for Path B
}

LINESTYLES = {
    "static":     "-",                        # solid
    "hybrid_rl":  "--",                       # dashed
    "no_pinn":    (0, (3, 1, 1, 1)),          # dash-dot-dot
    "no_slca":    ":",                        # dotted
    "agribrain":  "-.",                       # dash-dot
    "no_context": (0, (5, 2)),                # long dash
    "mcp_only":   (0, (3, 1, 1, 1, 1, 1)),   # dash-dot-dot-dot
    "pirag_only": (0, (1, 1)),                # dotted tight
    "no_yield":   (0, (6, 2, 1, 2)),          # dash-short-dash
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
    "no_yield":   "No Yield",
}

SCENARIO_LABELS = {
    "heatwave":         "Heatwave",
    "overproduction":   "Overproduction",
    "cyber_outage":     "Cyber Outage",
    "adaptive_pricing": "Price Volatility",
    "baseline":         "Baseline",
}

# Highlight color used for shaded scenario windows and emphasis text
WINDOW_COLOR = "#B71C1C"      # deep red, high contrast against teal agribrain
WINDOW_ALPHA = 0.12

DPI = 800
MARKER_EVERY = 15


def _apply_style(ax):
    """Apply the shared subplot styling. Safe to call multiple times."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.3)
    ax.spines["bottom"].set_linewidth(1.3)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.6, color="#BDBDBD", alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=TICK_FONT_SIZE, length=5, width=1.3)
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight("bold")
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
    if ax.xaxis.label.get_text():
        ax.xaxis.label.set_size(AXIS_LABEL_SIZE)
        ax.xaxis.label.set_weight("bold")
    if ax.yaxis.label.get_text():
        ax.yaxis.label.set_size(AXIS_LABEL_SIZE)
        ax.yaxis.label.set_weight("bold")
    if ax.get_title():
        ax.title.set_size(SUBPLOT_TITLE_SIZE)
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
        markeredgecolor="white",
        markeredgewidth=0.8,
        linewidth=2.2,
        label=MODE_LABELS[mode],
        **kwargs,
    )


def _legend(ax, **kwargs):
    """Add a styled legend. Bold entries, opaque background, gray border."""
    defaults = dict(
        fontsize=LEGEND_FONT_SIZE,
        framealpha=0.95,
        edgecolor="#757575",
        fancybox=False,
        shadow=False,
        borderpad=0.6,
        handlelength=2.2,
        handletextpad=0.6,
    )
    defaults.update(kwargs)
    leg = ax.legend(**defaults)
    if leg is not None:
        for text in leg.get_texts():
            text.set_fontweight("bold")
        if leg.get_title() is not None:
            leg.get_title().set_fontweight("bold")
    return leg


def _save(fig, name):
    """Save figure as PNG (800 DPI) and PDF (vector, TrueType fonts)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = RESULTS_DIR / f"{name}.{ext}"
        fig.savefig(
            str(path),
            dpi=DPI,
            bbox_inches="tight",
            pad_inches=0.15,
            facecolor="white",
        )
    print(f"  Saved {name}.png / .pdf")
    plt.close(fig)


def _annotate_window(ax, x0, x1, color, label, alpha=WINDOW_ALPHA, ypos=0.93):
    """Shade a scenario window and label it inside the plot at the top.
    A one-shot ylim expansion guarantees the label sits in blank space
    above the data; callers that have locked ylim explicitly (ratio
    axes, for instance) are respected. ``ypos`` is the axes-fraction
    vertical position of the bbox top edge."""
    ax.axvspan(x0, x1, alpha=alpha, color=color, zorder=0)
    # Add top headroom once per axes so the label never occludes data.
    if not getattr(ax, "_window_headroom_applied", False) and ax.get_autoscaley_on():
        y_lo, y_hi = ax.get_ylim()
        span = y_hi - y_lo
        if span > 0:
            ax.set_ylim(y_lo, y_hi + 0.18 * span)
        ax._window_headroom_applied = True
    ax.annotate(
        label,
        xy=((x0 + x1) / 2, ypos),
        xycoords=("data", "axes fraction"),
        ha="center", va="top",
        fontsize=ANNOT_FONT_SIZE,
        fontweight="bold",
        fontstyle="italic",
        color=color,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  alpha=0.95, edgecolor=color, linewidth=1.0),
        zorder=6,
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
    fig.suptitle("Heatwave Scenario Analysis")

    # --- (a) Temperature + Humidity with heatwave window ---
    ax = axes[0, 0]
    ax.plot(hours, ab["temp_trace"], color="#C62828", linewidth=2.0,
            label="Temperature (\u00b0C)")
    ax2 = ax.twinx()
    ax2.plot(hours, ab["rh_trace"], color="#1565C0", linewidth=1.8,
             alpha=0.85, label="RH (%)")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Temperature (\u00b0C)")
    ax2.set_ylabel("Relative Humidity (%)")
    ax.set_title("(a) Environmental Exposure")
    _apply_style(ax)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(labelsize=TICK_FONT_SIZE, length=5, width=1.3)
    ax2.yaxis.label.set_size(AXIS_LABEL_SIZE)
    ax2.yaxis.label.set_weight("bold")
    for lbl in ax2.get_yticklabels():
        lbl.set_fontweight("bold")
    _annotate_window(ax, 24, 48, WINDOW_COLOR, "Heatwave")
    # Combine the two legends into one to avoid corner collisions.
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    _legend(ax, handles=h1 + h2, labels=l1 + l2, loc="upper left")

    # --- (b) Observed spoilage risk trajectory ---
    ax = axes[0, 1]
    rho = np.array(ab["rho_trace"])
    ax.plot(hours, rho, color=COLORS["agribrain"], linewidth=2.2,
            label="Observed \u03c1(t)")
    ax.axhline(RLE_THRESHOLD, color=WINDOW_COLOR, linestyle="--", linewidth=1.6,
               alpha=0.85, label=f"RLE threshold (\u03c1={RLE_THRESHOLD})")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Spoilage Risk \u03c1(t)")
    ax.set_title("(b) Spoilage Risk Trajectory")
    _apply_style(ax)
    _annotate_window(ax, 24, 48, WINDOW_COLOR, "Heatwave")
    _legend(ax, loc="upper left")

    # --- (c) Action probability stacked area (AGRI-BRAIN) ---
    ax = axes[1, 0]
    probs = np.array(ab["prob_trace"])
    ax.fill_between(hours, 0, probs[:, 0],
                    color="#1565C0", alpha=0.85, label="Cold Chain")
    ax.fill_between(hours, probs[:, 0], probs[:, 0] + probs[:, 1],
                    color=COLORS["agribrain"], alpha=0.85, label="Local Redist.")
    ax.fill_between(hours, probs[:, 0] + probs[:, 1], 1.0,
                    color="#F57C00", alpha=0.85, label="Recovery")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Action Probability")
    ax.set_title("(c) AgriBrain Action Probabilities")
    ax.set_ylim(0, 1)
    _apply_style(ax)
    _annotate_window(ax, 24, 48, WINDOW_COLOR, "Heatwave")
    _legend(ax, loc="center right")

    # --- (d) Per-step reward (rolling average) ---
    ax = axes[1, 1]
    window = 12
    for mode in ["static", "hybrid_rl", "agribrain"]:
        ep = hw[mode]
        reward = np.array(ep["reward_trace"])
        rolling = np.convolve(reward, np.ones(window) / window, mode="same")
        _mode_plot(ax, hours, rolling, mode)
    ax.set_xlabel("Hours")
    ax.set_ylabel("Reward per Step (rolling avg)")
    ax.set_title("(d) Reward Rate During Heatwave")
    _apply_style(ax)
    _annotate_window(ax, 24, 48, WINDOW_COLOR, "Heatwave")
    _legend(ax, loc="lower left")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
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
    fig.suptitle("Overproduction & Reverse Logistics")

    # --- (a) Inventory vs Demand (dual y-axis) ---
    ax = axes[0, 0]
    inv = np.array(ab["inventory_trace"])
    dem = np.array(ab["demand_trace"])
    ax.plot(hours, inv, color=COLORS["agribrain"], linewidth=2.0,
            label="Inventory (units)")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Inventory (units)")
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(3, 3))
    ax2 = ax.twinx()
    ax2.plot(hours, dem, color=COLORS["hybrid_rl"], linewidth=1.8,
             alpha=0.85, label="Demand (units/step)")
    ax2.set_ylabel("Demand (units/step)")
    ax.set_title("(a) Inventory vs Demand")
    _apply_style(ax)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(labelsize=TICK_FONT_SIZE, length=5, width=1.3)
    ax2.yaxis.label.set_size(AXIS_LABEL_SIZE)
    ax2.yaxis.label.set_weight("bold")
    for lbl in ax2.get_yticklabels():
        lbl.set_fontweight("bold")
    _annotate_window(ax, 12, 60, WINDOW_COLOR, "Overproduction")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    _legend(ax, handles=h1 + h2, labels=l1 + l2, loc="upper left")

    # --- (b) Waste rolling average ---
    ax = axes[0, 1]
    window = 12
    for mode in ["static", "hybrid_rl", "agribrain"]:
        ep = op[mode]
        waste = np.array(ep["waste_trace"])
        rolling = np.convolve(waste, np.ones(window) / window, mode="same")
        _mode_plot(ax, hours, rolling, mode)
    ax.set_xlabel("Hours")
    ax.set_ylabel("Waste Rate (rolling avg)")
    ax.set_title("(b) Waste Reduction Over Time")
    _apply_style(ax)
    _annotate_window(ax, 12, 60, WINDOW_COLOR, "Overproduction")
    _legend(ax, loc="upper right")

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

    # Mark threshold onset with an axvline but push the text annotation to
    # a fixed axes-fraction slot so it never overlaps the traces.
    rho_ab = np.array(ab["rho_trace"])
    threshold_idx = int(np.argmax(rho_ab > RLE_THRESHOLD))
    if threshold_idx > 0 or rho_ab[0] > RLE_THRESHOLD:
        threshold_hour = hours[threshold_idx]
        ax.axvline(threshold_hour, color="#616161", linestyle="--",
                   linewidth=1.2, alpha=0.8)
        ax.annotate(
            f"\u03c1 > {RLE_THRESHOLD}  (h={threshold_hour:.0f})",
            xy=(threshold_hour, 1.02), xycoords=("data", "axes fraction"),
            ha="left", va="bottom", fontsize=ANNOT_FONT_SIZE,
            fontweight="bold", color="#424242",
        )

    ax.set_xlabel("Hours")
    ax.set_ylabel("RLE (rolling)")
    ax.set_title("(c) Reverse Logistics Efficiency")
    ax.set_ylim(-0.05, 1.15)
    _apply_style(ax)
    _annotate_window(ax, 12, 60, WINDOW_COLOR, "Overproduction")
    _legend(ax, loc="lower right")

    # --- (d) SLCA component grouped bars with std error bars ---
    ax = axes[1, 1]
    components = ["C", "L", "R", "P"]
    comp_labels = ["Carbon", "Labour", "Resilience", "Price Transp."]
    x = np.arange(len(components))
    width = 0.26
    for i, mode in enumerate(["static", "hybrid_rl", "agribrain"]):
        ep = op[mode]
        vals = [np.mean([s[comp] for s in ep["slca_component_trace"]])
                for comp in components]
        stds = [np.std([s[comp] for s in ep["slca_component_trace"]])
                for comp in components]
        ax.bar(x + i * width, vals, width, color=COLORS[mode],
               label=MODE_LABELS[mode], alpha=0.92, edgecolor="white",
               linewidth=0.8, yerr=stds, capsize=4,
               error_kw={"linewidth": 1.0, "capthick": 1.0})
    ax.set_xticks(x + width)
    ax.set_xticklabels(comp_labels)
    ax.set_ylabel("Score")
    ax.set_title("(d) SLCA Components")
    ax.set_ylim(0, 1.15)
    _apply_style(ax)
    _legend(ax, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "fig3_overproduction")


# ---------------------------------------------------------------------------
# Figure 4: Cyber Outage (1x3)
# ---------------------------------------------------------------------------
def fig4_cyber(data):
    """1x3: ARI over time with outage, action distribution, blockchain audit."""
    cy = data["results"]["cyber_outage"]
    ab = cy["agribrain"]
    hours = np.array(ab["hours"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle("Cyber Outage Scenario")

    # --- (a) ARI over time with outage shading ---
    ax = axes[0]
    for mode in ["static", "hybrid_rl", "agribrain"]:
        ep = cy[mode]
        ari = np.array(ep["ari_trace"])
        rolling = np.convolve(ari, np.ones(12) / 12, mode="same")
        _mode_plot(ax, hours, rolling, mode)
    ax.set_xlabel("Hours")
    ax.set_ylabel("ARI (rolling avg)")
    ax.set_title("(a) Adaptive Resilience Index")
    _apply_style(ax)
    _annotate_window(ax, 24, 72, WINDOW_COLOR, "Outage")
    _legend(ax, loc="lower left")

    # --- (b) Action distribution pre/during outage ---
    ax = axes[1]
    action_names = ["Cold Chain", "Local Redist.", "Recovery"]
    pre_mask = np.array(hours) < 24
    during_mask = np.array(hours) >= 24

    bar_x = np.arange(3)
    width = 0.38

    pre_counts = np.zeros(3)
    during_counts = np.zeros(3)
    actions = np.array(ab["action_trace"])
    n_pre = max(np.sum(pre_mask), 1)
    n_during = max(np.sum(during_mask), 1)
    for a in range(3):
        pre_counts[a] = np.sum((actions == a) & pre_mask) / n_pre
        during_counts[a] = np.sum((actions == a) & during_mask) / n_during
    pre_se = np.sqrt(pre_counts * (1 - pre_counts) / n_pre)
    during_se = np.sqrt(during_counts * (1 - during_counts) / n_during)

    ax.bar(bar_x - width / 2, pre_counts, width, color="#1565C0",
           alpha=0.92, label="Pre-outage", edgecolor="white", linewidth=0.8,
           yerr=1.96 * pre_se, capsize=4,
           error_kw={"linewidth": 1.2, "capthick": 1.2})
    ax.bar(bar_x + width / 2, during_counts, width, color=WINDOW_COLOR,
           alpha=0.92, label="During outage", edgecolor="white", linewidth=0.8,
           yerr=1.96 * during_se, capsize=4,
           error_kw={"linewidth": 1.2, "capthick": 1.2})
    ax.set_xticks(bar_x)
    ax.set_xticklabels(action_names)
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, max(max(pre_counts + pre_se * 2), max(during_counts + during_se * 2)) * 1.25 + 0.02)
    ax.set_title("(b) Action Distribution Shift")
    _apply_style(ax)
    _legend(ax, loc="upper right")

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

    # Higher-contrast scatter colors bound to the same three-tier band
    dot_colors = ["#2E7D32" if v > 0.66 else "#C62828" if v < 0.33 else "#F57C00"
                  for v in confidence_scores]
    ax.scatter(audit_times, confidence_scores, c=dot_colors, s=36, alpha=0.85,
               edgecolors="white", linewidths=0.6)
    ax.axhline(0.66, color="#424242", linestyle="--", linewidth=1.2,
               label="High-confidence band")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Decision Confidence\n(1 \u2212 normalized entropy)")
    ax.set_title("(c) Policy Confidence Trace")
    ax.set_ylim(-0.02, 1.02)
    _apply_style(ax)
    _annotate_window(ax, 24, 72, WINDOW_COLOR, "Outage")
    _legend(ax, loc="lower left")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
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
    fig.suptitle("Adaptive Pricing & Demand Volatility")

    # --- (a) Demand + Bollinger triggers ---
    ax = axes[0, 0]
    demand = np.array(ab["demand_trace"])
    window = 20
    rolling_mean = np.convolve(demand, np.ones(window) / window, mode="same")
    rolling_std = np.array([np.std(demand[max(0, i - window):i + 1])
                            for i in range(len(demand))])
    upper = rolling_mean + 2 * rolling_std
    lower = rolling_mean - 2 * rolling_std

    ax.plot(hours, demand, color="#37474F", linewidth=1.0, alpha=0.75, label="Demand")
    ax.plot(hours, rolling_mean, color="#1565C0", linewidth=2.0, label="Bollinger mean")
    ax.fill_between(hours, lower, upper, alpha=0.22, color="#1565C0",
                    label="\u00b12\u03c3 band", linewidth=0)
    triggers = np.abs(demand - rolling_mean) > 2 * rolling_std
    ax.scatter(hours[triggers], demand[triggers], color=WINDOW_COLOR, s=42,
               zorder=5, label="Trigger", marker="v",
               edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Hours")
    ax.set_ylabel("Demand (units/step)")
    ax.set_title("(a) Demand with Bollinger Triggers")
    _apply_style(ax)
    _legend(ax, loc="upper right")

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

    ax.bar(bin_centers, cc_fracs, bar_w, color="#1565C0", alpha=0.92,
           label="Cold Chain", edgecolor="white", linewidth=0.8)
    ax.bar(bin_centers, lr_fracs, bar_w, bottom=cc_fracs, color=COLORS["agribrain"],
           alpha=0.92, label="Local Redist.", edgecolor="white", linewidth=0.8)
    ax.bar(bin_centers, rec_fracs, bar_w, bottom=cc_fracs + lr_fracs,
           color="#F57C00", alpha=0.92, label="Recovery", edgecolor="white",
           linewidth=0.8)
    ax.set_xlabel("Hours")
    ax.set_ylabel("Routing Fraction")
    ax.set_title("(b) Routing Distribution Over Time")
    ax.set_ylim(0, 1.15)
    _apply_style(ax)
    _legend(ax, loc="upper right", ncol=3)

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
    _apply_style(ax)
    _legend(ax, loc="lower right")

    # --- (d) Reward decomposition: SLCA vs waste penalty ---
    ax = axes[1, 1]
    slca_vals = np.array(ab["slca_trace"])
    waste_vals = np.array(ab["waste_trace"])
    reward_vals = np.array(ab["reward_trace"])
    window = 12
    slca_smooth = np.convolve(slca_vals, np.ones(window) / window, mode="same")
    reward_smooth = np.convolve(reward_vals, np.ones(window) / window, mode="same")
    l1, = ax.plot(hours, slca_smooth, color=COLORS["agribrain"], linewidth=2.2,
                  label="SLCA reward", alpha=0.95)
    l2, = ax.plot(hours, reward_smooth, color="#263238", linewidth=2.0,
                  label="Net reward", alpha=0.95)
    ax.set_xlabel("Hours")
    ax.set_ylabel("SLCA / Net Reward")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("(d) Reward Decomposition")
    _apply_style(ax)
    ax2 = ax.twinx()
    waste_smooth = np.convolve(waste_vals, np.ones(window) / window, mode="same")
    l3, = ax2.plot(hours, waste_smooth * 100, color=WINDOW_COLOR, linewidth=2.0,
                   linestyle="--", label="Waste rate (%)", alpha=0.95)
    ax2.set_ylabel("Waste Rate (%)", color=WINDOW_COLOR)
    ax2.tick_params(axis="y", labelcolor=WINDOW_COLOR,
                    labelsize=TICK_FONT_SIZE, length=5, width=1.3)
    ax2.spines["top"].set_visible(False)
    ax2.yaxis.label.set_size(AXIS_LABEL_SIZE)
    ax2.yaxis.label.set_weight("bold")
    for lbl in ax2.get_yticklabels():
        lbl.set_fontweight("bold")
    _legend(ax, handles=[l1, l2, l3],
            labels=[l1.get_label(), l2.get_label(), l3.get_label()],
            loc="lower right")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
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


# Bold error bar styling so tight 20-seed CIs remain visible at figure scale.
_ERR_KW = {"linewidth": 1.8, "capthick": 1.8, "ecolor": "#1F1F1F", "alpha": 0.9}
_ERR_CAPSIZE = 5

_CI_NOTE = "Error bars: 95% bootstrap CI across 20 seeds"


def _stamp_ci_note(fig) -> None:
    """Small footer note documenting the CI basis on benchmark figures."""
    fig.text(0.5, 0.005, _CI_NOTE, ha="center", va="bottom",
             fontsize=ANNOT_FONT_SIZE - 1, color="#424242",
             style="italic", fontweight="bold")


def _bar_xticklabels(ax, scenarios_plot):
    """Bold, slightly rotated scenario names that never overlap."""
    ax.set_xticklabels(
        [SCENARIO_LABELS[s] for s in scenarios_plot],
        rotation=20, ha="right",
    )


def fig6_cross(data):
    """2x2 grouped bars: ARI, RLE, waste, SLCA across scenarios for 3 methods.
    Adds error bars from benchmark_summary.json when available."""
    bench = _load_benchmark_ci()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Cross-Scenario Performance Comparison")

    metrics = [("ari", "ARI", "(a)"), ("rle", "RLE", "(b)"),
               ("waste", "Waste Rate", "(c)"), ("slca", "SLCA Score", "(d)")]
    methods = ["static", "hybrid_rl", "agribrain"]
    scenarios_plot = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing"]

    for ax, (metric, ylabel, panel) in zip(axes.flat, metrics):
        x = np.arange(len(scenarios_plot))
        width = 0.26

        for i, mode in enumerate(methods):
            vals = [data["results"][s][mode][metric] for s in scenarios_plot]
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
                   label=MODE_LABELS[mode], alpha=0.92, edgecolor="white",
                   linewidth=0.8, yerr=yerr,
                   capsize=_ERR_CAPSIZE if yerr is not None else 0,
                   error_kw=_ERR_KW)

        ax.set_xticks(x + width)
        _bar_xticklabels(ax, scenarios_plot)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{panel} {ylabel}")
        _apply_style(ax)

    # Single legend at the bottom, shared across all subplots
    handles, labels = axes.flat[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=len(methods),
                     fontsize=LEGEND_FONT_SIZE, framealpha=0.95,
                     edgecolor="#757575", fancybox=False, shadow=False,
                     bbox_to_anchor=(0.5, 0.015))
    for text in leg.get_texts():
        text.set_fontweight("bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    _stamp_ci_note(fig)
    _save(fig, "fig6_cross")


# ---------------------------------------------------------------------------
# Figure 7: Ablation study (1x3 grouped bars)
# ---------------------------------------------------------------------------
def fig7_ablation(data):
    """1x3 grouped bars: ARI, waste, RLE for all 9 variants.
    Adds error bars from benchmark_summary.json when available. AgriBrain
    is always the last bar in each group so the comparison reads left to
    right from simplest baseline to full system."""
    bench = _load_benchmark_ci()

    # Reorder so AgriBrain is last, and no_yield (Path B ablation) sits
    # immediately before it for direct visual comparison.
    ordered = [m for m in MODES if m not in ("agribrain", "no_yield")] + ["no_yield", "agribrain"]
    fig7_modes = ordered

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
    fig.suptitle("Ablation Study")

    metrics = [("ari", "ARI", "(a)"), ("waste", "Waste Rate", "(b)"),
               ("rle", "RLE", "(c)")]
    stress_scenarios = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing"]

    n_modes = len(fig7_modes)
    # Give each group enough horizontal room: total group width ~ 0.9, split
    # across n_modes bars. Slight group gap by scaling x by 1.15.
    width = 0.9 / n_modes
    x_scale = 1.25

    for ax, (metric, ylabel, panel) in zip(axes, metrics):
        x = np.arange(len(stress_scenarios)) * x_scale

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
                   label=MODE_LABELS[mode], alpha=0.92, edgecolor="white",
                   linewidth=0.7, yerr=yerr,
                   capsize=_ERR_CAPSIZE if yerr is not None else 0,
                   error_kw=_ERR_KW)

        ax.set_xticks(x + (n_modes - 1) * width / 2)
        _bar_xticklabels(ax, stress_scenarios)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{panel} {ylabel}")
        _apply_style(ax)

    # One shared legend, bottom center, split across two rows to keep the
    # bars visible and the legend readable even with nine entries.
    handles, labels = axes[0].get_legend_handles_labels()
    ncol = min(5, (n_modes + 1) // 2)
    leg = fig.legend(handles, labels, loc="lower center", ncol=ncol,
                     fontsize=LEGEND_FONT_SIZE, framealpha=0.95,
                     edgecolor="#757575", fancybox=False, shadow=False,
                     bbox_to_anchor=(0.5, 0.02))
    for text in leg.get_texts():
        text.set_fontweight("bold")
    fig.tight_layout(rect=[0, 0.13, 1, 0.95])
    _stamp_ci_note(fig)
    _save(fig, "fig7_ablation")


# ---------------------------------------------------------------------------
# Figure 8: Green AI / Carbon (1x2)
# ---------------------------------------------------------------------------
def fig8_green_ai(data):
    """1x2: cumulative CO2 heatwave, total carbon bar chart with CI error bars."""
    bench = _load_benchmark_ci()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Green AI & Carbon Footprint")

    hw = data["results"]["heatwave"]
    hours = np.array(hw["agribrain"]["hours"])

    # --- (a) Cumulative CO2 for heatwave scenario ---
    ax = axes[0]
    for mode in ["static", "hybrid_rl", "no_pinn", "agribrain"]:
        ep = hw[mode]
        cum_carbon = np.cumsum(ep["carbon_trace"])
        _mode_plot(ax, hours, cum_carbon, mode)
    ax.set_xlabel("Hours")
    ax.set_ylabel(r"Cumulative $\mathrm{CO_2}$ (kg)")
    ax.set_title("(a) Cumulative Carbon \u2014 Heatwave")
    _apply_style(ax)
    _annotate_window(ax, 24, 48, WINDOW_COLOR, "Heatwave")
    _legend(ax, loc="upper left")

    # --- (b) Total carbon bar chart across all scenarios ---
    ax = axes[1]
    scenarios_plot = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing"]
    methods_plot = ["static", "hybrid_rl", "agribrain"]
    x = np.arange(len(scenarios_plot))
    width = 0.26

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
               label=MODE_LABELS[mode], alpha=0.92, edgecolor="white",
               linewidth=0.8, yerr=yerr,
               capsize=_ERR_CAPSIZE if yerr is not None else 0,
               error_kw=_ERR_KW)

    ax.set_xticks(x + width)
    _bar_xticklabels(ax, scenarios_plot)
    ax.set_ylabel(r"Total $\mathrm{CO_2}$ (kg)")
    ax.set_title("(b) Carbon Footprint by Scenario")
    _apply_style(ax)
    _legend(ax, loc="upper right")

    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    _stamp_ci_note(fig)
    _save(fig, "fig8_green_ai")


def fig9_mcp_pirag_robustness():
    """Robustness figure from protocol logs and benchmark summaries."""
    proto_files = [RESULTS_DIR / f"mcp_protocol_{s}.json" for s in SCENARIOS]
    bench_file = RESULTS_DIR / "benchmark_summary.json"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("MCP + piRAG Robustness")

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
        ax.bar(x, calls, color="#1565C0", alpha=0.92, label="Total calls",
               edgecolor="white", linewidth=0.8)
        ax.bar(x, errs, color=WINDOW_COLOR, alpha=0.92, label="Errors",
               edgecolor="white", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=20, ha="right")
    ax.set_title("(a) MCP Call Reliability")
    ax.set_ylabel("Count")
    _apply_style(ax)
    # Leave room so the legend sits in empty space above the tallest bar.
    if calls:
        ax.set_ylim(0, max(calls) * 1.30)
    _legend(ax, loc="upper left")

    # (b) ARI confidence intervals — Path B-aware ablation methods
    ax = axes[1]
    ci_methods = [
        ("agribrain",  COLORS["agribrain"],  "AgriBrain"),
        ("no_yield",   COLORS["no_yield"],   "No Yield"),
        ("pirag_only", COLORS["pirag_only"], "piRAG Only"),
        ("mcp_only",   COLORS["mcp_only"],   "MCP Only"),
        ("no_context", COLORS["no_context"], "No Context"),
    ]
    if bench_file.exists():
        import json
        bench = json.loads(bench_file.read_text(encoding="utf-8"))
        scenarios_plot = [s for s in SCENARIOS if s != "baseline"]
        x = np.arange(len(scenarios_plot))
        width = 0.9 / len(ci_methods)
        for j, (mode, color, label) in enumerate(ci_methods):
            means, lo_err, hi_err = [], [], []
            for s in scenarios_plot:
                ari = bench.get(s, {}).get(mode, {}).get("ari", {})
                m = float(ari.get("mean", 0.0))
                means.append(m)
                lo_raw = ari.get("ci_low", m)
                hi_raw = ari.get("ci_high", m)
                lo = float(m if lo_raw is None else lo_raw)
                hi = float(m if hi_raw is None else hi_raw)
                lo_err.append(max(0.0, m - lo))
                hi_err.append(max(0.0, hi - m))
            if means:
                yerr = np.vstack([lo_err, hi_err])
                ax.bar(x + j * width, means, width, color=color, label=label,
                       alpha=0.92, edgecolor="white", linewidth=0.7,
                       yerr=yerr, capsize=_ERR_CAPSIZE,
                       error_kw=_ERR_KW)
        ax.set_xticks(x + (len(ci_methods) - 1) * width / 2)
        ax.set_xticklabels([SCENARIO_LABELS.get(s, s) for s in scenarios_plot],
                           rotation=20, ha="right")
    ax.set_title("(b) Context Ablation ARI (95% CI)")
    ax.set_ylabel("ARI")
    _apply_style(ax)
    # Leave headroom above the tallest bar so the legend sits in empty
    # space, and place the legend at the top to avoid colliding with the
    # scenario-name labels below.
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.22)
    _legend(ax, loc="upper right", ncol=3)

    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    _stamp_ci_note(fig)
    _save(fig, "fig9_mcp_pirag_robustness")


def fig10_latency_quality_frontier(data):
    """Latency-quality frontier with two zones, fully matching the shared
    figure style (Arial, bold titles and axis labels, 800 DPI, no label
    overlaps). Panel (a) shows the lightweight methods (sub-millisecond);
    panel (b) shows the MCP/piRAG-enabled methods with the no-context
    reference point and an overhead annotation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                             gridspec_kw={"width_ratios": [1, 1], "wspace": 0.32})
    fig.suptitle("Latency vs ARI Frontier", fontsize=FIG_TITLE_SIZE,
                 fontweight="bold", y=1.00)

    bench = _load_benchmark_ci() or {}

    fast_modes = ["static", "hybrid_rl", "no_pinn", "no_slca", "no_context"]
    context_modes = ["agribrain", "mcp_only", "pirag_only", "no_yield"]

    def _collect(modes):
        pts = []
        for mode in modes:
            ari_vals = [data["results"][s][mode]["ari"] for s in SCENARIOS]
            lat_vals = [data["results"][s][mode].get("mean_decision_latency_ms", 0.0)
                        for s in SCENARIOS]
            yerr = (0.0, 0.0)
            ci_recs = [bench.get(s, {}).get(mode, {}).get("ari", {}) for s in SCENARIOS]
            if ci_recs and all("mean" in r and "ci_low" in r and "ci_high" in r for r in ci_recs):
                means = np.array([float(r["mean"]) for r in ci_recs], dtype=float)
                lows = np.array([float(r["mean"] if r["ci_low"] is None else r["ci_low"])
                                 for r in ci_recs], dtype=float)
                highs = np.array([float(r["mean"] if r["ci_high"] is None else r["ci_high"])
                                  for r in ci_recs], dtype=float)
                mean_y = float(np.mean(means))
                yerr = (max(0.0, mean_y - float(np.mean(lows))),
                        max(0.0, float(np.mean(highs)) - mean_y))
                pts.append((mode, float(np.mean(lat_vals)), mean_y, yerr))
            else:
                pts.append((mode, float(np.mean(lat_vals)), float(np.mean(ari_vals)), yerr))
        return pts

    fast_pts = _collect(fast_modes)
    ctx_pts = _collect([m for m in context_modes
                        if m in data["results"].get(SCENARIOS[0], {})])

    # --- (a) Fast modes (sub-millisecond) ---
    ax = axes[0]
    ax.set_title("(a) Lightweight Methods (<1 ms)")
    for mode, x, y, yerr in fast_pts:
        ax.scatter(x, y, s=220, color=COLORS[mode], marker=MARKERS[mode],
                   edgecolor="white", linewidth=1.4, alpha=0.95, zorder=5,
                   label=MODE_LABELS[mode])
        if yerr[0] > 0 or yerr[1] > 0:
            ax.errorbar([x], [y], yerr=np.array([[yerr[0]], [yerr[1]]]), fmt="none",
                        ecolor=COLORS[mode], elinewidth=1.6, capsize=4, alpha=0.85, zorder=4)
    ax.set_xlabel("Mean Decision Latency (ms)")
    ax.set_ylabel("Mean ARI")
    x_max_a = max(p[1] for p in fast_pts)
    ax.set_xlim(-0.005, x_max_a * 1.35 + 0.02)
    y_vals_a = [p[2] for p in fast_pts]
    ax.set_ylim(min(y_vals_a) - 0.05, max(y_vals_a) + 0.05)
    _legend(ax, loc="lower right", ncol=1, title="Method")
    _apply_style(ax)

    # --- (b) Context-aware methods (MCP/piRAG overhead) ---
    ax = axes[1]
    ax.set_title("(b) Context-Aware Methods")

    # No-context reference point (shared with panel a) plotted with lower
    # opacity so context-enabled points read as the primary subject.
    ref = next((p for p in fast_pts if p[0] == "no_context"), None)
    handles_b = []
    if ref is not None:
        ref_handle = ax.scatter(ref[1], ref[2], s=180,
                                color=COLORS["no_context"],
                                marker=MARKERS["no_context"],
                                edgecolor="white", linewidth=1.2,
                                alpha=0.55, zorder=4,
                                label="No Context (reference)")
        handles_b.append(ref_handle)

    for mode, x, y, yerr in ctx_pts:
        h = ax.scatter(x, y, s=260, color=COLORS[mode], marker=MARKERS[mode],
                       edgecolor="white", linewidth=1.4, alpha=0.95, zorder=5,
                       label=MODE_LABELS[mode])
        handles_b.append(h)
        if yerr[0] > 0 or yerr[1] > 0:
            ax.errorbar([x], [y], yerr=np.array([[yerr[0]], [yerr[1]]]), fmt="none",
                        ecolor=COLORS[mode], elinewidth=1.8, capsize=4,
                        alpha=0.9, zorder=4)

    # Overhead arrow between the reference and the context-enabled centroid.
    if ref is not None and ctx_pts:
        ctx_mean_lat = float(np.mean([p[1] for p in ctx_pts]))
        ctx_mean_ari = float(np.mean([p[2] for p in ctx_pts]))
        ax.annotate("", xy=(ctx_mean_lat, ctx_mean_ari),
                    xytext=(ref[1], ref[2]),
                    arrowprops=dict(arrowstyle="->", color=COLORS["agribrain"],
                                    lw=2.0, linestyle="--", alpha=0.55))
        # Place the overhead badge in the upper-left corner where no data
        # or legend sits, so it never overlaps the scatter points, the
        # dashed arrow, or the legend at lower-right.
        ax.text(0.03, 0.97,
                f"Context overhead\n+{ctx_mean_lat - ref[1]:.1f} ms  |  "
                f"{ctx_mean_ari - ref[2]:+.3f} ARI",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=ANNOT_FONT_SIZE, fontweight="bold",
                color=COLORS["agribrain"],
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          alpha=0.95, edgecolor=COLORS["agribrain"], linewidth=1.2),
                zorder=6)

    ax.set_xlabel("Mean Decision Latency (ms)")
    ax.set_ylabel("Mean ARI")
    lat_all = [ref[1]] + [p[1] for p in ctx_pts] if ref is not None else [p[1] for p in ctx_pts]
    ari_all = [ref[2]] + [p[2] for p in ctx_pts] if ref is not None else [p[2] for p in ctx_pts]
    ax.set_xlim(min(lat_all) - 0.3, max(lat_all) + 0.8)
    ax.set_ylim(min(ari_all) - 0.015, max(ari_all) + 0.020)
    _legend(ax, loc="lower right", ncol=1, title="Method")
    _apply_style(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
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
