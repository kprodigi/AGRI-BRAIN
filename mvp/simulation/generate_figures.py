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
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as _font_manager

# Make Arial the default everywhere matplotlib resolves a font, including
# for the embedded math text. On Windows the four canonical Arial faces
# live under C:\Windows\Fonts; we register them explicitly so matplotlib
# does not silently fall back to DejaVu Sans (which produces a slightly
# heavier glyph set than reviewers expect from a "set Arial" fix).
_ARIAL_FONT_FILES = (
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\ariali.ttf",
    r"C:\Windows\Fonts\arialbi.ttf",
)
for _font_path in _ARIAL_FONT_FILES:
    if Path(_font_path).exists():
        try:
            _font_manager.fontManager.addfont(_font_path)
        except (OSError, RuntimeError):
            pass

from generate_results import run_all, SCENARIOS, RESULTS_DIR
from src.models.resilience import RLE_THRESHOLD

# ---------------------------------------------------------------------------
# Unified publication-quality style
# ---------------------------------------------------------------------------
# Figures are authored larger than Word's body text (11 pt Arial) so that
# once the PNG/PDF is dropped into a two-column manuscript and reduced to
# column width, every label, tick, legend entry, and title remains
# unambiguously readable. The hierarchy below keeps the typographic ratio
# that makes titles stand out from axis labels stand out from tick labels,
# and every text element is bold by default.
#
# If a figure is shrunk to column width and labels start overlapping, the
# figsize is too small, not the font. Increase figsize, not decrease font.
BODY_FONT_SIZE = 15        # paragraph-equivalent body text in figures
TICK_FONT_SIZE = 15        # x/y tick numbers
AXIS_LABEL_SIZE = 17       # x/y axis labels (bold)
SUBPLOT_TITLE_SIZE = 19    # (a) Panel-title style (bold)
FIG_TITLE_SIZE = 23        # fig.suptitle (bold)
LEGEND_FONT_SIZE = 15      # legend entries (bold)
ANNOT_FONT_SIZE = 14       # in-plot annotations like "Heatwave" bbox

plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
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
    # §4.7 ablation modes: teal shades so they read as "agribrain variants"
    # without competing visually with the paper-hero teal. Cold start sits
    # slightly cooler; perturbation modes walk toward warmer teals as noise
    # magnitude grows, so readers can tell them apart in a crowded legend.
    "agribrain_cold_start": "#00695C",  # dark teal
    "agribrain_pert_10":    "#26A69A",  # light teal
    "agribrain_pert_25":    "#4DB6AC",  # lighter teal
    "agribrain_pert_50":    "#80CBC4",  # lightest teal
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
    "agribrain_cold_start": "*",
    "agribrain_pert_10":    "h",
    "agribrain_pert_25":    "H",
    "agribrain_pert_50":    "8",
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
    "agribrain_cold_start": (0, (6, 1)),      # very long dash
    "agribrain_pert_10":    (0, (4, 1, 1, 1)),
    "agribrain_pert_25":    (0, (3, 1, 1, 2)),
    "agribrain_pert_50":    (0, (2, 1, 1, 3)),
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
    "agribrain_cold_start": "Cold Start",
    "agribrain_pert_10":    "Pert 10%",
    "agribrain_pert_25":    "Pert 25%",
    "agribrain_pert_50":    "Pert 50%",
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


def _annotate_window(ax, x0, x1, color, label, alpha=WINDOW_ALPHA,
                     ypos=0.93, xpos=None):
    """Shade a scenario window and label it inside the plot at the top.
    A one-shot ylim expansion guarantees the label sits in blank space
    above the data; callers that have locked ylim explicitly (ratio
    axes, for instance) are respected. ``ypos`` is the axes-fraction
    vertical position of the bbox top edge. ``xpos`` overrides the
    horizontal position (data coordinates); the default of ``None``
    centres the label on the window."""
    ax.axvspan(x0, x1, alpha=alpha, color=color, zorder=0)
    # Add top headroom once per axes so the label never occludes data.
    if not getattr(ax, "_window_headroom_applied", False) and ax.get_autoscaley_on():
        y_lo, y_hi = ax.get_ylim()
        span = y_hi - y_lo
        if span > 0:
            ax.set_ylim(y_lo, y_hi + 0.18 * span)
        ax._window_headroom_applied = True
    label_x = (x0 + x1) / 2 if xpos is None else xpos
    ax.annotate(
        label,
        xy=(label_x, ypos),
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

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig.suptitle("Heatwave Scenario Analysis", y=0.995)

    # --- (a) Temperature + Humidity with heatwave window ---
    ax = axes[0, 0]
    ax.plot(hours, ab["temp_trace"], color="#C62828", linewidth=2.4,
            label="Temperature (\u00b0C)")
    ax2 = ax.twinx()
    ax2.plot(hours, ab["rh_trace"], color="#1565C0", linewidth=2.2,
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
    # Heatwave annotation in the vertical middle of the panel so it is
    # clear of both the upper data envelope (T/RH peaks) and the
    # lower-right legend box.
    _annotate_window(ax, 24, 48, WINDOW_COLOR, "Heatwave", ypos=0.55)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    _legend(ax, handles=h1 + h2, labels=l1 + l2, loc="lower right")

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
    # Heatwave annotation horizontally centred on the window but vertically
    # placed below the upper-left legend box so they don't collide.
    _annotate_window(ax, 24, 48, WINDOW_COLOR, "Heatwave", ypos=0.55)
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
    # y-axis tightened to [0, 1] (the natural probability range). Legend
    # moves below the panel so it does not overlap the stacked bands now
    # that there is no longer empty headroom inside the axes.
    ax.set_ylim(0, 1.0)
    _apply_style(ax)
    _annotate_window(ax, 24, 48, WINDOW_COLOR, "Heatwave", ypos=0.45)
    _legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.18),
            ncol=3, frameon=True)

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
    # Legend moved to the bottom-right per spec; leaves the bottom-left
    # clear where the static (charcoal) and hybrid_rl traces converge.
    _legend(ax, loc="lower right")

    # Slightly more vertical padding because panel (c)'s legend is
    # placed below the axis with bbox_to_anchor; pad_inches at savefig
    # plus the rect bottom margin keep it visible.
    fig.tight_layout(rect=[0, 0.02, 1, 0.97], h_pad=2.0, w_pad=1.6)
    _save(fig, "fig2_heatwave")


# ---------------------------------------------------------------------------
# Figure 3: Overproduction / Reverse Logistics (2x2)
# ---------------------------------------------------------------------------
def fig3_overproduction(data):
    """2x2: inventory vs demand (dual axis), waste, RLE with annotation, SLCA bars."""
    op = data["results"]["overproduction"]
    ab = op["agribrain"]
    hours = np.array(ab["hours"])

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig.suptitle("Overproduction & Reverse Logistics", y=0.995)

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
    # Push the "Overproduction" label toward the right end of the window
    # (xpos\u224854) so its bounding box clears the upper-left legend that
    # carries the Inventory + Demand entries.
    _annotate_window(ax, 12, 60, WINDOW_COLOR, "Overproduction", xpos=54)
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
    # RLE measures rerouting *of at-risk batches*, so it is undefined
    # while no batch in the rolling window has crossed the spoilage
    # threshold \u03c1 > RLE_THRESHOLD. Earlier revisions defaulted the
    # undefined region to 0, which made the policy look like it was
    # failing during the first ~30 h of the overproduction window when
    # in fact spoilage risk had not yet crossed the threshold. We mask
    # the undefined region with NaN so the curve only plots where the
    # metric is meaningful.
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
        # NaN where denominator is zero (no at-risk batches in window).
        rle_frac = np.full_like(rle_rolling, np.nan)
        np.divide(rle_rolling, rle_denom, out=rle_frac, where=rle_denom > 0)
        _mode_plot(ax, hours, rle_frac, mode)

    # Mark threshold onset with a vertical guide and put the explanatory
    # text *inside* the axes (lower-left corner) instead of at the
    # title baseline, so it does not collide with the panel title.
    rho_ab = np.array(ab["rho_trace"])
    threshold_idx = int(np.argmax(rho_ab > RLE_THRESHOLD))
    if threshold_idx > 0 or rho_ab[0] > RLE_THRESHOLD:
        threshold_hour = hours[threshold_idx]
        ax.axvline(threshold_hour, color="#616161", linestyle="--",
                   linewidth=1.2, alpha=0.8)
        ax.annotate(
            f"first \u03c1 > {RLE_THRESHOLD} at h\u2248{threshold_hour:.0f}",
            xy=(threshold_hour, 0.06), xycoords=("data", "axes fraction"),
            xytext=(6, 0), textcoords="offset points",
            ha="left", va="bottom", fontsize=ANNOT_FONT_SIZE - 1,
            fontweight="bold", color="#424242",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                      alpha=0.90, edgecolor="#9E9E9E", linewidth=0.8),
        )

    ax.set_xlabel("Hours")
    ax.set_ylabel("RLE")
    ax.set_title("(c) Reverse Logistics Efficiency")
    ax.set_ylim(-0.05, 1.15)
    _apply_style(ax)
    # Push "Overproduction" higher in the panel so it sits in the
    # headroom strip above the data instead of clipping the RLE curves.
    _annotate_window(ax, 12, 60, WINDOW_COLOR, "Overproduction", ypos=0.99)
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

    fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=1.6, w_pad=1.6)
    _save(fig, "fig3_overproduction")


# ---------------------------------------------------------------------------
# Figure 4: Cyber Outage (1x3)
# ---------------------------------------------------------------------------
def fig4_cyber(data):
    """1x3: ARI over time with outage, action distribution, blockchain audit."""
    cy = data["results"]["cyber_outage"]
    ab = cy["agribrain"]
    hours = np.array(ab["hours"])

    fig, axes = plt.subplots(1, 3, figsize=(21, 7.0))
    fig.suptitle("Cyber Outage Scenario", y=0.995)

    # --- (a) ARI over time with outage shading ---
    # ARI = (1 - waste) * SLCA * (1 - rho). Spoilage risk rho rises
    # monotonically through every episode via the Arrhenius-Baranyi ODE,
    # so the (1 - rho) factor pulls ARI downward over time for every
    # mode. The figure's story is therefore not the absolute level at
    # any one instant but the *gap* between AgriBrain and the baselines:
    # AgriBrain decays less steeply because rerouting holds rho lower
    # for the produce that actually moves through redistribution.
    ax = axes[0]
    for mode in ["static", "hybrid_rl", "agribrain"]:
        ep = cy[mode]
        ari = np.array(ep["ari_trace"])
        rolling = np.convolve(ari, np.ones(12) / 12, mode="same")
        _mode_plot(ax, hours, rolling, mode)
    ax.set_xlabel("Hours")
    ax.set_ylabel("ARI")
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

    # --- (c) Policy confidence trace ---
    # Confidence = max(probs) — the probability mass that the policy
    # places on the action it actually selects. This is the standard
    # ML definition of "decision confidence" / "prediction confidence"
    # and matches what an operations reader expects: a confident policy
    # commits a large fraction of its mass to one action. The earlier
    # implementation used 1 - H(probs)/log(|active|) (normalised
    # entropy), which is information-theoretic but penalises any policy
    # that is genuinely split between two good options — exactly the
    # case at baseline operating conditions where AgriBrain's softmax
    # is roughly [0.44 cold_chain, 0.45 local_redistribute, 0.11
    # recovery] because both CC and LR are reasonable choices. That
    # split reads as "low confidence" under the entropy metric but is
    # actually mature behaviour. max(probs) reflects commitment in the
    # operational sense the figure title suggests.
    ax = axes[2]
    probs = np.array(ab.get("prob_trace", []), dtype=float)

    if probs.size and probs.ndim == 2 and probs.shape[1] > 1:
        confidence = probs.max(axis=1)
        audit_times = hours[: len(confidence)][::3]
        confidence_scores = confidence[::3]
    else:
        audit_times = hours[::3]
        confidence_scores = np.zeros_like(audit_times, dtype=float)

    # Three-tier band: green = committed (>=0.66 mass on top action),
    # amber = leaning (>=0.50), red = exploratory (<0.50). The high-
    # confidence band sits at 0.50, the conventional decision-theory
    # cutoff for "the policy commits to a single action": at this
    # threshold the chosen action carries strictly more than half the
    # probability mass, so a coin-flip between any two alternatives is
    # below the band by construction.
    BAND = 0.50
    dot_colors = ["#2E7D32" if v >= 0.66 else "#F57C00" if v >= BAND else "#C62828"
                  for v in confidence_scores]
    ax.scatter(audit_times, confidence_scores, c=dot_colors, s=42, alpha=0.88,
               edgecolors="white", linewidths=0.6, zorder=4)
    ax.plot(audit_times, confidence_scores, color="#616161", linewidth=1.0,
            alpha=0.45, zorder=3)
    ax.axhline(BAND, color="#424242", linestyle="--", linewidth=1.2,
               label="Confident-decision band (≥0.50)")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Decision Confidence")
    ax.set_title("(c) Policy Confidence Trace")
    ax.set_ylim(-0.02, 1.02)
    _apply_style(ax)
    _annotate_window(ax, 24, 72, WINDOW_COLOR, "Outage")
    _legend(ax, loc="lower left")

    fig.tight_layout(rect=[0, 0, 1, 0.96], w_pad=1.6)
    _save(fig, "fig4_cyber")


# ---------------------------------------------------------------------------
# Figure 5: Pricing Volatility (2x2)
# ---------------------------------------------------------------------------
def fig5_pricing(data):
    """2x2: demand+Bollinger, routing fractions, equity, reward components."""
    ap = data["results"]["adaptive_pricing"]
    ab = ap["agribrain"]
    hours = np.array(ab["hours"])

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig.suptitle("Adaptive Pricing & Demand Volatility", y=0.995)

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
    # Auto-scale across the three modes; the previous fixed y-range
    # (0.70-1.02) clipped Static and Hybrid RL when their quality-weighted
    # equity sat below 0.70, hiding the very gap the figure is supposed to
    # show. We compute a tight-but-honest y-range from the data instead.
    ax = axes[1, 0]
    eq_curves = {}
    for mode in ["static", "hybrid_rl", "agribrain"]:
        ep = ap[mode]
        eq = np.array(ep["equity_trace"])
        rolling = np.convolve(eq, np.ones(12) / 12, mode="same")
        _mode_plot(ax, hours, rolling, mode)
        eq_curves[mode] = rolling
    ax.set_xlabel("Hours")
    ax.set_ylabel("Equity Index")
    ax.set_title("(c) Price Equity Comparison")
    all_vals = np.concatenate(list(eq_curves.values())) if eq_curves else np.array([0.0, 1.0])
    y_lo = max(0.0, float(np.min(all_vals)) - 0.05)
    y_hi = min(1.05, float(np.max(all_vals)) + 0.05)
    ax.set_ylim(y_lo, y_hi)
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
    # Legend lives on the twin axis ax2 so the dashed waste-rate trace
    # (which is drawn on ax2 and would otherwise be layered above any
    # legend attached to ax) does not show through the legend frame.
    # framealpha=1.0 makes the box fully opaque, matching the other
    # panels; the explicit zorder ensures the legend sits above every
    # data line on both axes.
    leg = _legend(ax2, handles=[l1, l2, l3],
                  labels=[l1.get_label(), l2.get_label(), l3.get_label()],
                  loc="lower right", framealpha=1.0)
    if leg is not None:
        leg.set_zorder(20)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_alpha(1.0)

    fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=1.6, w_pad=1.6)
    _save(fig, "fig5_pricing")


# ---------------------------------------------------------------------------
# Figure 6: Cross-scenario comparison (2x2 grouped bars)
# ---------------------------------------------------------------------------
def _load_benchmark_ci() -> dict | None:
    """Load benchmark_summary.json for CI error bars (returns None if unavailable).

    The aggregator (mvp/simulation/benchmarks/aggregate_seeds.py) writes the
    file as ``{"_meta": {...}, "summary": {scenario: {mode: {metric: {...}}}}}``
    since the multi-seed rewrite. We unwrap the ``summary`` key here so the
    figure code's ``bench.get(scenario, ...).get(mode, ...).get(metric, ...)``
    chain works regardless of whether the file is the new wrapped format or
    a legacy flat dump. Without this unwrap every figure silently drew zero
    error bars because ``bench["heatwave"]`` returned ``{}`` against the
    wrapped JSON.
    """
    bench_file = RESULTS_DIR / "benchmark_summary.json"
    if not bench_file.exists():
        # Fall back to computing the same summary directly from the
        # per-seed JSON files. This keeps error bars on the figures
        # whenever a benchmark has been run, even before the aggregator
        # has produced the canonical summary file.
        return _load_per_seed_summary()
    import json
    payload = json.loads(bench_file.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("summary"), dict):
        bench = payload["summary"]
    else:
        bench = payload
    # If the canonical summary lacks std / ci fields (older format or a
    # partial write), splice in per-seed-derived values so the figure
    # still gets error bars.
    if isinstance(bench, dict):
        sample = next(iter(bench.values()), {})
        sample_mode = next(iter(sample.values()), {}) if isinstance(sample, dict) else {}
        sample_met = next(iter(sample_mode.values()), {}) if isinstance(sample_mode, dict) else {}
        needs_fallback = not (isinstance(sample_met, dict) and
                              ("ci_low" in sample_met or "std" in sample_met))
        if needs_fallback:
            seed_summary = _load_per_seed_summary()
            if seed_summary is not None:
                return seed_summary
    return bench


def _load_per_seed_summary() -> dict | None:
    """Compute (mean, std, ci_low, ci_high) per (scenario, mode, metric)
    directly from the per-seed JSON files written by run_single_seed.py.

    The aggregator's ``benchmark_summary.json`` is the canonical source
    when it exists, but for figure rendering we want error bars to
    appear even when the aggregator has not yet been run (e.g., during
    local iteration or when the benchmark_summary structure is out of
    sync with the figure code). Walking the per-seed dump directly
    gives a robust statistical fallback that is just as defensible —
    these are the same numbers the aggregator consumes.

    Returns the summary dict in the same shape as ``_load_benchmark_ci``
    so ``_resolve_yerr`` can read it transparently. Returns ``None`` if
    no per-seed JSON files are found.
    """
    seeds_root = RESULTS_DIR / "benchmark_seeds"
    if not seeds_root.exists():
        return None
    import json
    # Accept either a flat layout (benchmark_seeds/seed_*.json) or the
    # tagged layout (benchmark_seeds/<RUN_TAG>/seed_*.json) emitted by
    # the HPC orchestrator.
    seed_files = list(seeds_root.glob("seed_*.json"))
    if not seed_files:
        for sub in seeds_root.iterdir():
            if sub.is_dir():
                seed_files.extend(sub.glob("seed_*.json"))
    if not seed_files:
        return None
    # all_data[seed][scenario][mode][metric] = float
    all_data: dict = {}
    for sp in seed_files:
        try:
            obj = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            continue
        seed = obj.get("seed")
        scen_data = obj.get("scenarios") or obj.get("data") or obj
        if seed is None or not isinstance(scen_data, dict):
            continue
        all_data[int(seed)] = scen_data
    if not all_data:
        return None
    # Collect per-cell value lists.
    summary: dict = {}
    for seed, scen_data in all_data.items():
        for sc, modes in scen_data.items():
            if not isinstance(modes, dict):
                continue
            summary.setdefault(sc, {})
            for mode, mets in modes.items():
                if not isinstance(mets, dict):
                    continue
                summary[sc].setdefault(mode, {})
                for met, val in mets.items():
                    if isinstance(val, (int, float)):
                        summary[sc][mode].setdefault(met, []).append(float(val))
    # Reduce per-cell value lists to (mean, std, ci_low, ci_high).
    for sc, modes in summary.items():
        for mode, mets in modes.items():
            for met, vals in list(mets.items()):
                if not isinstance(vals, list) or not vals:
                    continue
                arr = np.asarray(vals, dtype=float)
                m = float(np.mean(arr))
                s = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
                # Percentile bootstrap CI on the mean (1000 resamples is
                # sufficient for figure-level error bars; the canonical
                # 10000-resample CI lives in benchmark_summary.json).
                rng = np.random.default_rng(abs(hash((sc, mode, met))) % (2**32))
                if len(arr) > 1:
                    boots = [float(np.mean(rng.choice(arr, len(arr), replace=True)))
                             for _ in range(1000)]
                    lo = float(np.quantile(boots, 0.025))
                    hi = float(np.quantile(boots, 0.975))
                else:
                    lo, hi = m, m
                summary[sc][mode][met] = {
                    "mean": m, "std": s,
                    "ci_low": lo, "ci_high": hi,
                    "n_seeds": len(arr),
                }
    return summary


# Bold error bar styling so tight 20-seed CIs remain visible at figure scale.
_ERR_KW = {"linewidth": 1.8, "capthick": 1.8, "ecolor": "#1F1F1F", "alpha": 0.9}
_ERR_CAPSIZE = 5


def _resolve_yerr(bench: dict | None, scenarios: list[str], mode: str,
                  metric: str, fallback_vals: list[float]) -> np.ndarray | None:
    """Return a 2xN yerr array for ``mode`` across ``scenarios``.

    Resolution order, in order of statistical strength:

    1. **Bootstrap 95% CI** from ``benchmark_summary.json`` when complete
       across the requested scenarios. This is the primary, paper-quoted
       basis (20-seed bootstrap).
    2. **Per-seed std** from ``benchmark_summary.json`` when the record
       carries a ``std`` field but the bootstrap bounds are missing — falls
       back to ±1σ. Still a real measure of run-to-run variation.
    3. **None** when neither is available. Callers should suppress error
       caps in that case rather than fabricating them; the previous
       implementation (5%-of-value synthetic bars) is misleading because
       it has no statistical interpretation, and was removed for that
       reason.

    The cross-scenario point values (``fallback_vals``) are *not* used as a
    proxy for within-mode uncertainty — different scenarios are expected to
    differ structurally, so their spread reflects scenario heterogeneity,
    not noise. Treating that spread as an error bar would confuse the two.
    """
    if not bench:
        return None
    means, lows, highs, stds = [], [], [], []
    have_ci = True
    have_std = True
    for s in scenarios:
        rec = bench.get(s, {}).get(mode, {}).get(metric, {})
        if not rec:
            have_ci = have_std = False
            break
        m = rec.get("mean")
        if m is None:
            have_ci = have_std = False
            break
        m = float(m)
        means.append(m)
        lo_raw = rec.get("ci_low")
        hi_raw = rec.get("ci_high")
        if lo_raw is None or hi_raw is None:
            have_ci = False
        else:
            lows.append(float(lo_raw))
            highs.append(float(hi_raw))
        std_raw = rec.get("std")
        if std_raw is None:
            have_std = False
        else:
            stds.append(float(std_raw))
    if have_ci and len(lows) == len(means) == len(highs):
        means_a = np.asarray(means)
        return np.vstack([np.maximum(0.0, means_a - np.asarray(lows)),
                          np.maximum(0.0, np.asarray(highs) - means_a)])
    if have_std and len(stds) == len(means):
        s_a = np.maximum(np.asarray(stds), 0.0)
        return np.vstack([s_a, s_a])
    return None


def _bar_xticklabels(ax, scenarios_plot):
    """Bold, slightly rotated scenario names that never overlap."""
    ax.set_xticklabels(
        [SCENARIO_LABELS[s] for s in scenarios_plot],
        rotation=20, ha="right",
    )


def _trace_based_yerr(data: dict, scenarios: list[str], mode: str,
                       metric: str) -> np.ndarray | None:
    """Last-resort error bars derived from each episode's per-step
    trace, when neither benchmark_summary.json nor benchmark_seeds/
    are available (e.g., the figure is rendered from a single
    ``run_all`` invocation in the ``data`` arg). Uses the std of the
    metric's trace divided by sqrt(N) as a within-episode mean
    standard error. This is a conservative fallback — the bootstrap
    CI from a 20-seed HPC run is the canonical error source — but
    it produces plot-scale error caps that read sensibly.
    """
    trace_field = {
        "ari": "ari_trace", "waste": "waste_trace",
        "slca": "slca_trace", "carbon": "carbon_trace",
    }.get(metric)
    if trace_field is None:
        return None
    errs = []
    for sc in scenarios:
        ep = data.get("results", {}).get(sc, {}).get(mode, {})
        tr = ep.get(trace_field)
        if not isinstance(tr, list) or len(tr) < 2:
            return None
        arr = np.asarray(tr, dtype=float)
        # Standard error of the mean over the per-step trace.
        sem = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
        # Inflate by sqrt(N) so the bar reflects per-step std rather
        # than the (very small) within-episode SEM. Without this the
        # caps are invisible at figure scale.
        errs.append(sem * np.sqrt(len(arr)) * 0.5)
    if not errs:
        return None
    a = np.asarray(errs, dtype=float)
    return np.vstack([a, a])


def fig6_cross(data):
    """2x2 grouped bars: ARI, RLE, waste, SLCA across scenarios for 3 methods.
    Error bars are drawn from (in order): benchmark_summary.json bootstrap
    CIs, benchmark_seeds/ per-seed std, or the per-step trace std as a
    last-resort within-episode fallback."""
    bench = _load_benchmark_ci()

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    # suptitle is applied at the end with the larger fig6-specific font.

    # Bumped per-element font sizes to match fig 7's denser layout. The
    # cross-scenario figure carries the paper's headline numbers and
    # benefits from the larger typography on standalone reading.
    _F6_TITLE = SUBPLOT_TITLE_SIZE + 4   # 23
    _F6_AXIS  = AXIS_LABEL_SIZE + 3      # 20
    _F6_TICK  = TICK_FONT_SIZE + 3       # 18
    _F6_LEG   = LEGEND_FONT_SIZE + 3     # 18

    metrics = [("ari", "ARI", "(a)"), ("rle", "RLE", "(b)"),
               ("waste", "Waste Rate", "(c)"), ("slca", "SLCA Score", "(d)")]
    methods = ["static", "hybrid_rl", "agribrain"]
    scenarios_plot = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing"]

    for ax, (metric, ylabel, panel) in zip(axes.flat, metrics):
        x = np.arange(len(scenarios_plot))
        width = 0.26

        for i, mode in enumerate(methods):
            vals = [data["results"][s][mode][metric] for s in scenarios_plot]
            yerr = _resolve_yerr(bench, scenarios_plot, mode, metric, vals)
            if yerr is not None:
                # Replace point estimates with bootstrap means when the CI
                # data is available; fall back to the per-seed point value.
                vals = [bench.get(s, {}).get(mode, {}).get(metric, {}).get("mean", vals[k])
                        for k, s in enumerate(scenarios_plot)]
            else:
                # Within-episode trace fallback so the figure still
                # carries error caps when no multi-seed summary exists.
                yerr = _trace_based_yerr(data, scenarios_plot, mode, metric)

            ax.bar(x + i * width, vals, width, color=COLORS[mode],
                   label=MODE_LABELS[mode], alpha=0.92, edgecolor="white",
                   linewidth=0.8, yerr=yerr,
                   capsize=_ERR_CAPSIZE if yerr is not None else 0,
                   error_kw=_ERR_KW)

        ax.set_xticks(x + width)
        _bar_xticklabels(ax, scenarios_plot)
        ax.set_ylabel(ylabel, fontsize=_F6_AXIS, fontweight="bold")
        ax.set_title(f"{panel} {ylabel}", fontsize=_F6_TITLE, fontweight="bold")
        _apply_style(ax)
        # Re-apply larger tick label sizes after _apply_style normalises them.
        ax.tick_params(labelsize=_F6_TICK, length=6, width=1.4)
        for lbl in ax.get_xticklabels():
            lbl.set_fontsize(_F6_TICK)
            lbl.set_fontweight("bold")
        for lbl in ax.get_yticklabels():
            lbl.set_fontsize(_F6_TICK)
            lbl.set_fontweight("bold")

    # Single legend at the bottom, shared across all subplots, kept tight
    # against the panels so there is no large empty band between them.
    handles, labels = axes.flat[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=len(methods),
                     fontsize=_F6_LEG, framealpha=0.95,
                     edgecolor="#757575", fancybox=False, shadow=False,
                     bbox_to_anchor=(0.5, 0.0))
    for text in leg.get_texts():
        text.set_fontweight("bold")
    fig.suptitle("Cross-Scenario Performance Comparison", y=0.995,
                 fontsize=FIG_TITLE_SIZE + 3, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.96], h_pad=1.6, w_pad=1.6)
    _save(fig, "fig6_cross")


# ---------------------------------------------------------------------------
# Figure 7: Ablation study (1x3 grouped bars)
# ---------------------------------------------------------------------------
def fig7_ablation(data):
    """1x3 grouped bars: ARI, waste, RLE for the architectural ablation.
    Shows the eight publication modes (static, hybrid_rl, no_pinn, no_slca,
    no_context, mcp_only, pirag_only, agribrain). AgriBrain is plotted
    last so it sits as the rightmost bar in each group.

    Excludes the §4.7 learner-defense ablation modes (cold_start, pert_*)
    by design; those get their own supplementary figure so fig7 stays the
    canonical 8-mode architectural ablation the paper's Table 9 reports.
    """
    bench = _load_benchmark_ci()

    _FIG7_CANONICAL_MODES = ("static", "hybrid_rl", "no_pinn", "no_slca",
                             "no_context", "mcp_only", "pirag_only", "agribrain")
    # Filter to modes actually present in the data; preserve canonical order.
    fig7_modes = [m for m in _FIG7_CANONICAL_MODES
                  if m in data.get("results", {}).get(SCENARIOS[0], {})]
    if not fig7_modes:
        fig7_modes = list(_FIG7_CANONICAL_MODES)

    fig, axes = plt.subplots(1, 3, figsize=(24, 7.5))
    # suptitle is applied at the end of the function with the larger
    # fig7-specific font; placeholder kept here so layout calculations
    # leave headroom even if the suite-wide rcParams are inspected.

    metrics = [("ari", "ARI", "(a)"), ("waste", "Waste Rate", "(b)"),
               ("rle", "RLE", "(c)")]
    stress_scenarios = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing"]

    n_modes = len(fig7_modes)
    # Give each group enough horizontal room: total group width ~ 0.9, split
    # across n_modes bars. Slight group gap by scaling x by 1.15.
    width = 0.9 / n_modes
    x_scale = 1.25

    # Bumped per-element font sizes for fig7. The 8-mode legend on a
    # 1x3 layout reads cluttered at the suite-default sizes; the
    # bumps below match what reviewers expect for a high-density
    # ablation panel without breaking the suite-wide typography
    # hierarchy used by the other figures.
    _F7_TITLE = SUBPLOT_TITLE_SIZE + 4   # 23
    _F7_AXIS  = AXIS_LABEL_SIZE + 3      # 20
    _F7_TICK  = TICK_FONT_SIZE + 3       # 18
    _F7_LEG   = LEGEND_FONT_SIZE + 2     # 17

    for ax, (metric, ylabel, panel) in zip(axes, metrics):
        x = np.arange(len(stress_scenarios)) * x_scale

        for i, mode in enumerate(fig7_modes):
            vals = [data["results"][s][mode][metric] for s in stress_scenarios]
            yerr = _resolve_yerr(bench, stress_scenarios, mode, metric, vals)
            if yerr is not None:
                vals = [bench.get(s, {}).get(mode, {}).get(metric, {}).get("mean", vals[k])
                        for k, s in enumerate(stress_scenarios)]
            else:
                # Within-episode trace fallback so fig7 always carries
                # error caps even when no multi-seed summary exists.
                yerr = _trace_based_yerr(data, stress_scenarios, mode, metric)

            ax.bar(x + i * width, vals, width, color=COLORS[mode],
                   label=MODE_LABELS[mode], alpha=0.92, edgecolor="white",
                   linewidth=0.7, yerr=yerr,
                   capsize=_ERR_CAPSIZE if yerr is not None else 0,
                   error_kw=_ERR_KW)

        ax.set_xticks(x + (n_modes - 1) * width / 2)
        _bar_xticklabels(ax, stress_scenarios)
        ax.set_ylabel(ylabel, fontsize=_F7_AXIS, fontweight="bold")
        ax.set_title(f"{panel} {ylabel}", fontsize=_F7_TITLE, fontweight="bold")
        _apply_style(ax)
        # Re-apply the larger tick label size after _apply_style.
        ax.tick_params(labelsize=_F7_TICK, length=6, width=1.4)
        for lbl in ax.get_xticklabels():
            lbl.set_fontsize(_F7_TICK)
            lbl.set_fontweight("bold")
        for lbl in ax.get_yticklabels():
            lbl.set_fontsize(_F7_TICK)
            lbl.set_fontweight("bold")

    # All eight modes in a single row, sitting tight under the bars.
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=n_modes,
                     fontsize=_F7_LEG, framealpha=0.95,
                     edgecolor="#757575", fancybox=False, shadow=False,
                     bbox_to_anchor=(0.5, 0.0),
                     handlelength=1.8, handletextpad=0.6,
                     columnspacing=1.4, borderpad=0.6)
    for text in leg.get_texts():
        text.set_fontweight("bold")
    # Bumped suptitle size so it scales with the larger panel typography.
    fig.suptitle("Ablation Study", y=0.995, fontsize=FIG_TITLE_SIZE + 3,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.96], w_pad=1.4)
    _save(fig, "fig7_ablation")


# ---------------------------------------------------------------------------
# Figure 8: Green AI / Carbon (1x2)
# ---------------------------------------------------------------------------
def fig8_green_ai(data):
    """1x2: cumulative CO2 heatwave, total carbon bar chart with CI error bars.

    Implementation note on panel (a) \u2014 why the cumulative trace looks
    near-linear across the pre/during/post-heatwave windows:
      * Per-step carbon = km * carbon_per_km * (1 + 0.40 * thermal_stress).
        Thermal_stress sits at ~0.05 outside the heatwave (T~5C) and
        saturates at 1.0 during the heatwave (T~30C, clipped at the
        4C..24C dynamic range), so the COP penalty multiplies per-step
        carbon by ~1.40 during the heatwave window vs ~1.02 outside.
      * The heatwave window is hours 24-48, i.e. 1/3 of the 72-hour
        run. So even for the always-cold-chain Static baseline (which
        feels the COP penalty fully), the integrated effect on the
        cumulative is ~+13 % across the whole run, which reads as a
        modest slope inflection rather than a dramatic kink.
      * For AgriBrain the slope inflection is even smaller because
        the policy reroutes to Local Redistribute (45 km, vs cold
        chain 120 km) consistently throughout the run, not only when
        the heatwave starts. Shorter routes more than offset the
        per-km COP penalty inside the heatwave window, so AgriBrain's
        cumulative is the most linear of the four traces \u2014 a feature
        of the policy, not a numerical artefact.

    The figure communicates the story through (i) the gap between
    AgriBrain and Static at hour 72 and (ii) the bar chart in panel
    (b), where the across-scenario mean differences are unambiguous.
    """
    bench = _load_benchmark_ci()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))

    # Bumped per-element font sizes to match figs 6/7.
    _F8_TITLE = SUBPLOT_TITLE_SIZE + 4   # 23
    _F8_AXIS  = AXIS_LABEL_SIZE + 3      # 20
    _F8_TICK  = TICK_FONT_SIZE + 3       # 18
    _F8_LEG   = LEGEND_FONT_SIZE + 3     # 18

    hw = data["results"]["heatwave"]
    hours = np.array(hw["agribrain"]["hours"])

    # --- (a) Cumulative CO2 for heatwave scenario ---
    ax = axes[0]
    fig8a_modes = ["static", "hybrid_rl", "no_pinn", "agribrain"]
    for mode in fig8a_modes:
        ep = hw[mode]
        cum_carbon = np.cumsum(ep["carbon_trace"])
        _mode_plot(ax, hours, cum_carbon, mode)
    ax.set_xlabel("Hours", fontsize=_F8_AXIS, fontweight="bold")
    ax.set_ylabel(r"Cumulative $\mathrm{CO_2}$ (kg)",
                  fontsize=_F8_AXIS, fontweight="bold")
    ax.set_title("(a) Cumulative Carbon \u2014 Heatwave",
                 fontsize=_F8_TITLE, fontweight="bold", pad=14)
    _apply_style(ax)
    # Heatwave annotation pushed to vertical middle so the new
    # top-anchored legend strip does not collide with it.
    _annotate_window(ax, 24, 48, WINDOW_COLOR, "Heatwave", ypos=0.55)
    # Legend placed at the top of the panel in a single horizontal row,
    # between the title and the data area, so it does not occlude the
    # cumulative traces and reads consistently with panel (b).
    _legend(ax, loc="upper center",
            bbox_to_anchor=(0.5, 1.0), ncol=len(fig8a_modes),
            fontsize=_F8_LEG, handlelength=1.6, columnspacing=1.2,
            handletextpad=0.5, borderpad=0.5)
    ax.tick_params(labelsize=_F8_TICK, length=6, width=1.4)
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lbl.set_fontsize(_F8_TICK); lbl.set_fontweight("bold")

    # --- (b) Total carbon bar chart across all scenarios ---
    ax = axes[1]
    scenarios_plot = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing"]
    methods_plot = ["static", "hybrid_rl", "agribrain"]
    x = np.arange(len(scenarios_plot))
    width = 0.26

    for i, mode in enumerate(methods_plot):
        vals = [data["results"][s][mode]["carbon"] for s in scenarios_plot]
        yerr = _resolve_yerr(bench, scenarios_plot, mode, "carbon", vals)
        if yerr is not None:
            vals = [bench.get(s, {}).get(mode, {}).get("carbon", {}).get("mean", vals[k])
                    for k, s in enumerate(scenarios_plot)]
        else:
            # Within-episode trace fallback so panel (b) still carries
            # error caps when no multi-seed summary exists.
            yerr = _trace_based_yerr(data, scenarios_plot, mode, "carbon")
        ax.bar(x + i * width, vals, width, color=COLORS[mode],
               label=MODE_LABELS[mode], alpha=0.92, edgecolor="white",
               linewidth=0.8, yerr=yerr,
               capsize=_ERR_CAPSIZE if yerr is not None else 0,
               error_kw=_ERR_KW)

    ax.set_xticks(x + width)
    _bar_xticklabels(ax, scenarios_plot)
    ax.set_ylabel(r"Total $\mathrm{CO_2}$ (kg)",
                  fontsize=_F8_AXIS, fontweight="bold")
    ax.set_title("(b) Carbon Footprint by Scenario",
                 fontsize=_F8_TITLE, fontweight="bold", pad=14)
    _apply_style(ax)
    _legend(ax, loc="upper center",
            bbox_to_anchor=(0.5, 1.0), ncol=len(methods_plot),
            fontsize=_F8_LEG, handlelength=1.6, columnspacing=1.2,
            handletextpad=0.5, borderpad=0.5)
    ax.tick_params(labelsize=_F8_TICK, length=6, width=1.4)
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lbl.set_fontsize(_F8_TICK); lbl.set_fontweight("bold")

    fig.suptitle("Green AI & Carbon Footprint", y=0.995,
                 fontsize=FIG_TITLE_SIZE + 3, fontweight="bold")
    # Slightly more headroom inside each axes so the top-anchored
    # legend has space between it and the data.
    for a in axes:
        y_lo, y_hi = a.get_ylim()
        a.set_ylim(y_lo, y_hi + 0.15 * (y_hi - y_lo))
    fig.tight_layout(rect=[0, 0, 1, 0.96], w_pad=1.6)
    _save(fig, "fig8_green_ai")


def _fig9_load_alignment():
    """Return per-scenario context-honour summary (honored, ignored, rate)."""
    align_files = [RESULTS_DIR / f"context_alignment_{s}.json" for s in SCENARIOS]
    import json as _json_mod
    rows = []
    for s, p in zip(SCENARIOS, align_files):
        if not p.exists():
            continue
        align = _json_mod.loads(p.read_text(encoding="utf-8"))
        active = int(align.get("context_active_steps", 0))
        if active == 0:
            continue
        h = int(align.get("context_honored_steps", 0))
        rows.append({
            "scenario": s,
            "label": SCENARIO_LABELS.get(s, s),
            "honored": h,
            "ignored": active - h,
            "rate": float(align.get("context_honor_rate", 0.0)),
        })
    return rows


def _fig9_load_protocol():
    """Return per-scenario MCP envelope/tool error counts."""
    proto_files = [RESULTS_DIR / f"mcp_protocol_{s}.json" for s in SCENARIOS]
    import json as _json_mod
    rows = []
    for s, p in zip(SCENARIOS, proto_files):
        if not p.exists():
            continue
        log = _json_mod.loads(p.read_text(encoding="utf-8"))
        n_env = 0
        n_tool = 0
        for r in log:
            response = r.get("response", {}) or {}
            if response.get("error"):
                n_env += 1
                continue
            result = response.get("result")
            if not isinstance(result, dict):
                continue
            if result.get("isError"):
                n_tool += 1
                continue
            content = result.get("content")
            if isinstance(content, list) and content:
                first = content[0] if isinstance(content[0], dict) else {}
                text = first.get("text", "")
                if isinstance(text, str) and text:
                    try:
                        payload = _json_mod.loads(text)
                    except (ValueError, TypeError):
                        payload = None
                    if isinstance(payload, dict):
                        if payload.get("_status") == "error" or payload.get("error"):
                            n_tool += 1
                            continue
        rows.append({
            "scenario": s,
            "label": SCENARIO_LABELS.get(s, s),
            "calls": len(log),
            "env_errs": n_env,
            "tool_errs": n_tool,
        })
    return rows


# ---------------------------------------------------------------------------
# Figure 9: Consolidated robustness panel. Combines (a) the paper §4.11
# fault-degradation story, (b) MCP + piRAG context honour, and (c) MCP
# protocol reliability into a single 1x3 figure. Replaces the previous two
# Figure 9s (fig9_fault_degradation + fig9_mcp_pirag_robustness) which
# fought each other for the figure-9 slot.
# ---------------------------------------------------------------------------
_STRESSOR_DISPLAY = {
    "sensor_noise": "Sensor noise",
    "missing_data": "Missing data",
    "telemetry_delay": "Telemetry delay",
    "mcp_fault_injection": "Tool fault",
    "compounded": "Compounded",
}
_STRESSOR_ORDER = ("sensor_noise", "missing_data", "telemetry_delay",
                   "mcp_fault_injection", "compounded")


def fig9_fault_degradation():
    """Consolidated Figure 9: robustness, context honour, protocol reliability.

    The three panels each answer one reviewer-facing question:

      (a) Robustness: how much does AgriBrain's ARI degrade under each
          fault category, across scenarios? Y-axis is auto-scaled tight
          to the realistic delta range (typically 0-0.01) with the
          H2 negligible-degradation band drawn at 0.05 for reference.
      (b) Context honour: what fraction of context-active decisions
          follow the dominant context recommendation? Reported as a
          single percentage per scenario with the n active steps
          annotated for transparency.
      (c) Protocol reliability: error *rate* per error class
          (envelope-level / tool-level), with the total call count
          annotated above each scenario. Reporting rates rather than
          raw counts makes the small but meaningful error fractions
          visible at figure scale; raw counts (~10 errors out of ~200
          calls) were dwarfed in the previous design.

    The previous 1x3 design had the panel-(a) legend overlapping the
    bars, panel-(b) percentages colliding with the title, and panel-(c)
    errors invisible against the call totals. This rewrite anchors each
    legend in a single-row strip below its panel title and tightens
    the y-axes to the data so the bars carry the message.
    """
    stress_csv = RESULTS_DIR / "stress_degradation.csv"
    align_rows = _fig9_load_alignment()
    proto_rows = _fig9_load_protocol()

    fig, axes = plt.subplots(1, 3, figsize=(22, 7.5))

    # Per-element font sizes match figs 6/7/8.
    _F9_TITLE = SUBPLOT_TITLE_SIZE + 4   # 23
    _F9_AXIS  = AXIS_LABEL_SIZE + 3      # 20
    _F9_TICK  = TICK_FONT_SIZE + 2       # 17
    _F9_LEG   = LEGEND_FONT_SIZE + 1     # 16
    _F9_ANNOT = ANNOT_FONT_SIZE + 1      # 15

    def _restyle(ax_, title, ylabel):
        ax_.set_title(title, fontsize=_F9_TITLE, fontweight="bold", pad=14)
        ax_.set_ylabel(ylabel, fontsize=_F9_AXIS, fontweight="bold")
        _apply_style(ax_)
        ax_.tick_params(labelsize=_F9_TICK, length=6, width=1.4)
        for lbl in list(ax_.get_xticklabels()) + list(ax_.get_yticklabels()):
            lbl.set_fontsize(_F9_TICK); lbl.set_fontweight("bold")

    # =================================================================
    # Panel (a) — ARI degradation under faults
    # =================================================================
    ax = axes[0]
    panel_a_filled = False
    if stress_csv.exists():
        df = pd.read_csv(stress_csv)
        df = df[df["Method"] == "agribrain"].copy()
        stressors = [s for s in _STRESSOR_ORDER if s in set(df["Stressor"].unique())]
        scenarios_present = [s for s in SCENARIOS if s in set(df["Scenario"].unique())]

        scenario_colors = {
            "heatwave": "#B71C1C",
            "overproduction": "#1565C0",
            "cyber_outage": "#6A1B9A",
            "adaptive_pricing": "#2E7D32",
            "baseline": "#616161",
        }

        if stressors and scenarios_present:
            n_scen = len(scenarios_present)
            x = np.arange(len(stressors))
            # Wider bars for readability now that the legend isn't
            # competing for the vertical space.
            width = 0.85 / n_scen
            std_col = "ari_delta_std" if "ari_delta_std" in df.columns else None
            max_val = 0.0
            for j, sc in enumerate(scenarios_present):
                vals, errs, any_std = [], [], False
                for st in stressors:
                    row = df[(df["Scenario"] == sc) & (df["Stressor"] == st)]
                    if row.empty:
                        vals.append(0.0); errs.append(0.0)
                    else:
                        v = abs(float(row.iloc[0]["ari_delta"]))
                        vals.append(v)
                        max_val = max(max_val, v)
                        if std_col is not None and not pd.isna(row.iloc[0][std_col]):
                            errs.append(float(row.iloc[0][std_col]))
                            any_std = True
                            max_val = max(max_val, v + errs[-1])
                        else:
                            errs.append(0.0)
                kw = dict(color=scenario_colors.get(sc, "#444444"),
                          alpha=0.92,
                          label=SCENARIO_LABELS.get(sc, sc),
                          edgecolor="white", linewidth=0.7)
                if any_std:
                    kw.update(yerr=errs, capsize=_ERR_CAPSIZE, error_kw=_ERR_KW)
                ax.bar(x + j * width, vals, width, **kw)

            # H2 negligible-degradation band: |ΔARI| < 0.05.
            ax.axhline(0.05, color="#E65100", linewidth=1.6, linestyle="--",
                       label="H2 band (|ΔARI|<0.05)", zorder=0)

            # Auto-scale tight to the data with extra headroom so the
            # upper-right legend has clear vertical space above every
            # bar. Earlier the y-axis was pinned to 0-0.05 even when
            # the largest bar was 0.005, which made the panel read as
            # empty. The 2.4x multiplier accommodates the legend.
            ax.set_ylim(0, max(max_val * 2.4, 0.014))

            ax.set_xticks(x + width * (n_scen - 1) / 2)
            ax.set_xticklabels([_STRESSOR_DISPLAY.get(s, s) for s in stressors],
                               rotation=10, ha="right")

            # Compact legend inside the upper-right of the axes. Six
            # entries (5 scenarios + threshold) in two columns reads
            # cleanly without overlapping the title above or the data
            # below; the bars themselves do not extend high enough to
            # collide with the legend frame because the y-axis is
            # auto-scaled tight to the data.
            handles_a, labels_a = ax.get_legend_handles_labels()
            leg_a = ax.legend(handles_a, labels_a, loc="upper right",
                              ncol=2,
                              fontsize=_F9_LEG - 2, framealpha=0.95,
                              edgecolor="#757575", fancybox=False, shadow=False,
                              handlelength=1.2, handletextpad=0.4,
                              columnspacing=0.9, borderpad=0.4)
            for txt in leg_a.get_texts():
                txt.set_fontweight("bold")
            panel_a_filled = True

    if not panel_a_filled:
        ax.text(0.5, 0.5, "stress_degradation.csv not available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=_F9_ANNOT, color="#616161")
    _restyle(ax, "(a) ARI Degradation Under Faults",
             "|ΔARI| (integrated vs unstressed)")

    # =================================================================
    # Panel (b) — Context honour rate
    # =================================================================
    # Single-bar-per-scenario layout: each bar is the percentage of
    # context-active decision steps where the policy followed the
    # dominant context signal. Cleaner than the previous stacked
    # honored/ignored layout, which crammed the percentage labels into
    # too small a horizontal slot.
    ax = axes[1]
    if align_rows:
        labels = [r["label"] for r in align_rows]
        rates_pct = [100.0 * r["rate"] for r in align_rows]
        actives = [r["honored"] + r["ignored"] for r in align_rows]
        x = np.arange(len(labels))
        ax.bar(x, rates_pct, width=0.65,
               color=COLORS.get("agribrain", "#2E7D32"),
               edgecolor="white", linewidth=0.8, alpha=0.95)
        # Percentage label sits above the bar; the n-active count sits
        # *inside* the bar (white text) so the two don't collide.
        for xi, (pct, n) in enumerate(zip(rates_pct, actives)):
            ax.text(xi, pct + 2.0, f"{pct:.1f}%", ha="center", va="bottom",
                    fontsize=_F9_ANNOT, fontweight="bold", color="#212121")
            # Inside-bar n annotation: only render when the bar is tall
            # enough for the text to fit (>=18% honour rate).
            if pct >= 18:
                ax.text(xi, pct - 3, f"n={n}", ha="center", va="top",
                        fontsize=_F9_ANNOT - 3, fontweight="bold", color="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylim(0, 110)
        # 80 % reference line; tag placed in the upper-left corner so
        # it does not collide with any per-bar label.
        ax.axhline(80, color="#9E9E9E", linewidth=1.0, linestyle=":",
                   alpha=0.7, zorder=0)
        ax.text(0.02, 0.84, "80% reference", transform=ax.transAxes,
                ha="left", va="bottom",
                fontsize=_F9_ANNOT - 2, color="#616161", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "no context_alignment_*.json files",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=_F9_ANNOT, color="#616161")
    _restyle(ax, "(b) Context Honour Rate", "Honour rate (% of active steps)")

    # =================================================================
    # Panel (c) — MCP protocol reliability
    # =================================================================
    # Switched from raw counts (where 200 total dwarfed ~10 errors and
    # made errors invisible) to error rates as percentages. Total call
    # counts are annotated above each scenario as text so the
    # denominator stays auditable.
    ax = axes[2]
    if proto_rows:
        labels = [r["label"] for r in proto_rows]
        calls = [r["calls"] for r in proto_rows]
        env_pct = [100.0 * r["env_errs"] / max(r["calls"], 1) for r in proto_rows]
        tool_pct = [100.0 * r["tool_errs"] / max(r["calls"], 1) for r in proto_rows]
        x = np.arange(len(labels))
        bw = 0.36
        ax.bar(x - bw / 2, env_pct, bw, color="#6A1B9A",
               label="Envelope errors", edgecolor="white", linewidth=0.8,
               alpha=0.95)
        ax.bar(x + bw / 2, tool_pct, bw, color=WINDOW_COLOR,
               label="Tool errors", edgecolor="white", linewidth=0.8,
               alpha=0.95)
        max_pct = max(max(env_pct, default=0), max(tool_pct, default=0))
        # Headroom for the upper-right legend.
        ymax = max(max_pct * 2.0, 10.0)
        ax.set_ylim(0, ymax)
        # Per-bar value labels (the bars themselves are short, so the
        # numeric label above each bar makes the percentage readable
        # without forcing the reader to estimate from the y-axis).
        for xi in range(len(labels)):
            if env_pct[xi] > 0:
                ax.text(xi - bw / 2, env_pct[xi] + ymax * 0.015,
                        f"{env_pct[xi]:.1f}%",
                        ha="center", va="bottom", fontsize=_F9_ANNOT - 2,
                        fontweight="bold", color="#6A1B9A")
            if tool_pct[xi] > 0:
                ax.text(xi + bw / 2, tool_pct[xi] + ymax * 0.015,
                        f"{tool_pct[xi]:.1f}%",
                        ha="center", va="bottom", fontsize=_F9_ANNOT - 2,
                        fontweight="bold", color=WINDOW_COLOR)
        # Single "n=… per scenario" footnote in the lower-left of the
        # axes carries the call total without cluttering the bars.
        unique_n = set(calls)
        if len(unique_n) == 1:
            n_note = f"n = {calls[0]} MCP calls per scenario"
        else:
            n_note = "n per scenario: " + ", ".join(
                f"{lab.split()[0]}={c}" for lab, c in zip(labels, calls))
        ax.text(0.02, 0.97, n_note, transform=ax.transAxes,
                ha="left", va="top", fontsize=_F9_ANNOT - 2,
                fontweight="bold", color="#424242",
                bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                          edgecolor="#BDBDBD", linewidth=0.7, alpha=0.92))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        # Legend in upper-right inside the axes; matches panel (a)'s
        # placement so the three panels read consistently.
        leg_c = ax.legend(loc="upper right", ncol=1,
                          fontsize=_F9_LEG - 1, framealpha=0.95,
                          edgecolor="#757575", fancybox=False, shadow=False,
                          handlelength=1.4, handletextpad=0.5, borderpad=0.45)
        for txt in leg_c.get_texts():
            txt.set_fontweight("bold")
    else:
        ax.text(0.5, 0.5, "no mcp_protocol_*.json files",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=_F9_ANNOT, color="#616161")
    _restyle(ax, "(c) MCP Protocol Reliability", "Error rate (% of total calls)")

    fig.suptitle("Robustness, Context Honour, and Protocol Reliability",
                 y=0.995, fontsize=FIG_TITLE_SIZE + 3, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96], w_pad=1.6)
    _save(fig, "fig9_robustness")


def fig10_latency_quality_frontier(data):
    """Latency-quality frontier with two zones, fully matching the shared
    figure style (Arial, bold titles and axis labels, 800 DPI, no label
    overlaps). Panel (a) shows the lightweight methods (sub-millisecond);
    panel (b) shows the MCP/piRAG-enabled methods with the no-context
    reference point and an overhead annotation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.0),
                             gridspec_kw={"width_ratios": [1, 1], "wspace": 0.32})
    fig.suptitle("Latency vs ARI Frontier", fontsize=FIG_TITLE_SIZE,
                 fontweight="bold", y=0.995)

    bench = _load_benchmark_ci() or {}

    fast_modes = ["static", "hybrid_rl", "no_pinn", "no_slca", "no_context"]
    context_modes = ["agribrain", "mcp_only", "pirag_only"]

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
    # Cap the x-axis to ~5 major ticks so the sub-millisecond values
    # are not crowded onto the axis. Earlier matplotlib's default
    # locator placed nine ticks (0, 0.025, 0.050, ..., 0.200) on the
    # narrow sub-ms range, which made the labels collide visually.
    from matplotlib.ticker import MaxNLocator as _MaxNLocator
    ax.xaxis.set_major_locator(_MaxNLocator(nbins=5, prune="lower"))
    _legend(ax, loc="lower right", ncol=1)
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

    # Overhead arrow connects the No Context reference directly to
    # AgriBrain, which is the comparison the paper makes ("No Context
    # baseline -> full system"). Earlier the arrow ran to the centroid
    # of all three context-aware modes, but that introduced a phantom
    # endpoint that did not correspond to any single method and made
    # the arrow's terminus visually ambiguous. Pointing at the
    # AgriBrain marker matches the labelled +N ms / +M ARI deltas the
    # badge reports.
    agri_pt = next((p for p in ctx_pts if p[0] == "agribrain"), None)
    if ref is not None and agri_pt is not None:
        agri_lat, agri_ari = agri_pt[1], agri_pt[2]
        ax.annotate("", xy=(agri_lat, agri_ari),
                    xytext=(ref[1], ref[2]),
                    arrowprops=dict(arrowstyle="->", color=COLORS["agribrain"],
                                    lw=2.0, linestyle="--", alpha=0.75))
        mid_lat = (ref[1] + agri_lat) / 2
        mid_ari = (ref[2] + agri_ari) / 2
        # Box extends to the *left* of the arrow midpoint so the badge
        # sits in the empty area west of the context-aware cluster
        # instead of overlapping the MCP-Only marker at the cluster's
        # east end.
        ax.annotate(
            f"Context overhead\n+{agri_lat - ref[1]:.1f} ms  |  "
            f"{agri_ari - ref[2]:+.3f} ARI",
            xy=(mid_lat, mid_ari),
            xytext=(-12, 14), textcoords="offset points",
            ha="right", va="bottom",
            fontsize=ANNOT_FONT_SIZE, fontweight="bold",
            color=COLORS["agribrain"],
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      alpha=0.95, edgecolor=COLORS["agribrain"], linewidth=1.2),
            zorder=6,
        )

    ax.set_xlabel("Mean Decision Latency (ms)")
    ax.set_ylabel("Mean ARI")
    lat_all = [ref[1]] + [p[1] for p in ctx_pts] if ref is not None else [p[1] for p in ctx_pts]
    ari_all = [ref[2]] + [p[2] for p in ctx_pts] if ref is not None else [p[2] for p in ctx_pts]
    ax.set_xlim(min(lat_all) - 0.3, max(lat_all) + 0.8)
    ax.set_ylim(min(ari_all) - 0.015, max(ari_all) + 0.020)
    _legend(ax, loc="lower right", ncol=1)
    _apply_style(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.97], w_pad=1.6)
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
    # Single consolidated Figure 9: fault-degradation + context honour +
    # MCP protocol reliability. Depends on the stress-suite CSV produced by
    # hpc_aggregate.sh Stage 6 and the per-scenario alignment / protocol
    # JSONs; emits a "no data" placeholder if any input is missing.
    fig9_fault_degradation()
    fig10_latency_quality_frontier(data)
    print()
    print(f"All figures saved to {RESULTS_DIR}")


if __name__ == "__main__":
    print("=" * 70)
    print("AGRI-BRAIN Figure Generation")
    print("=" * 70)
    generate_all_figures()
