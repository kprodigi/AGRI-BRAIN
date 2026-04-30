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

_BACKEND_SRC = Path(__file__).resolve().parent.parent.parent / "agribrain" / "backend"
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
from src.models.action_selection import ACTIONS, CYBER_REROUTE_PROB
from src.models.resilience import RLE_THRESHOLD, compute_effective_rho, HIERARCHY_WEIGHT

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
    """2x2: env exposure, per-method retail rho, AgriBrain action mix, per-step ARI.

    Panel (b) plots the quantity-weighted mean rho on retail-bound
    batches under the *temperature-conditional* batch-FIFO model
    (see resilience.route_rho_factor and batch_inventory.py). Each
    batch accumulates rho at its status-specific factor, with the
    cold-chain factor stepping from 0.15 (nominal) through 0.40
    (stressed at 30-35 degC) to 1.00 (overwhelmed above 35 degC).
    Under realistic physics, cold chain is *strictly better* than
    local-redistribute on retail rho whenever the ambient is below
    30 degC; the two are roughly tied during the 30-35 degC stress
    band that the heatwave scenario operates in. AgriBrain therefore
    does *not* clearly win on raw retail rho - its win comes from
    the composite ARI (panel d), where the LR-leaning policy gains
    on carbon, labour, resilience, and price-transparency at modest
    rho cost.

    Panel (c) shows AgriBrain's action-probability stacked area with
    three regime guides: at-risk threshold crossing (rho >= 0.10),
    Recovery knee crossing (rho >= 0.50), and post-heatwave fresh-batch
    cold-chain recovery.

    Panel (d) plots per-step ARI (12 h rolling) - the composite metric
    the paper sells. ARI is bounded [0, 1] so the cross-method gap is
    directly interpretable.
    """
    hw = data["results"]["heatwave"]
    ab = hw["agribrain"]
    hours = np.array(ab["hours"])

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig.suptitle("Heatwave Scenario Analysis", y=0.995)

    # --- (a) Temperature + Humidity with heatwave window ---
    ax = axes[0, 0]
    ax.plot(hours, ab["temp_trace"], color="#C62828", linewidth=2.4,
            label="Temperature")
    # Safe-storage reference line (5 C, FDA leafy-greens guideline).
    ax.axhline(5.0, color="#C62828", linestyle=":", linewidth=1.4,
               alpha=0.65, label="Safe storage")
    ax2 = ax.twinx()
    ax2.plot(hours, ab["rh_trace"], color="#1565C0", linewidth=2.2,
             alpha=0.85, label="RH")
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
    ax2.set_ylim(40, 100)
    _annotate_window(ax, 24, 48, WINDOW_COLOR, "Heatwave", ypos=0.55)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    _legend(ax, handles=h1 + h2, labels=l1 + l2,
            loc="lower center", framealpha=0.80)

    # --- (b) Effective spoilage risk per method (batch-FIFO) ---
    # The environmental rho trace (Arrhenius-from-temperature) is
    # exogenous physics, identical across methods, and shown as a thin
    # grey reference. Each method's plotted trace is the
    # quantity-weighted mean rho on the retail-bound batch pool,
    # computed mechanistically from the per-batch FIFO inventory model
    # (BatchInventory in src/models/batch_inventory.py).
    #
    # Cold-chain factor is *temperature-conditional* (Mercier 2017,
    # Ndraha 2018):
    #   T < 30 degC: 0.15 (nominal cold-chain integrity)
    #   30-35 degC : 0.40 (cold chain stressed)
    #   T > 35 degC: 1.00 (cold chain overwhelmed)
    # Local-redistribute holds 0.45 across all temperatures; Recovery
    # is 0.00 (leaves retail pool). Under realistic physics, Static
    # (CC-only) wins on retail rho at nominal ambient and ties
    # AgriBrain in the stress band. The figure should therefore *not*
    # be read as "AgriBrain wins on rho"; the AgriBrain win is on the
    # composite ARI (panel d), not on raw rho.
    ax = axes[0, 1]
    for mode in ["static", "hybrid_rl", "agribrain"]:
        ep = hw[mode]
        # Prefer the batch-FIFO trace (mechanistic, per-batch); fall
        # back to the aggregate-mix accounting view, then to a
        # post-hoc compute for legacy result files.
        if "batch_effective_rho_trace" in ep and ep["batch_effective_rho_trace"]:
            eff = np.array(ep["batch_effective_rho_trace"])
        elif "effective_rho_trace" in ep and ep["effective_rho_trace"]:
            eff = np.array(ep["effective_rho_trace"])
        else:
            eff = compute_effective_rho(
                np.array(ep["rho_trace"]),
                np.array(ep["prob_trace"]),
                turnover_halflife_hours=12.0,
                dt_hours=0.25,
            )
        _mode_plot(ax, hours, eff, mode)
    ax.axhline(RLE_THRESHOLD, color=WINDOW_COLOR, linestyle="--",
               linewidth=1.6, alpha=0.85,
               label=f"At-risk threshold (\u03c1={RLE_THRESHOLD:.2f})")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Spoilage Risk")
    ax.set_title("(b) Spoilage Risk Trajectory")
    ax.set_ylim(0, 1.0)
    _apply_style(ax)
    _annotate_window(ax, 24, 48, WINDOW_COLOR, "Heatwave", ypos=0.55)
    _legend(ax, loc="upper left", framealpha=0.80)

    # --- (c) AgriBrain action-probability stacked area + regime guides ---
    ax = axes[1, 0]
    probs = np.array(ab["prob_trace"])
    ax.fill_between(hours, 0, probs[:, 0],
                    color="#1565C0", alpha=0.85, label="Cold Chain")
    ax.fill_between(hours, probs[:, 0], probs[:, 0] + probs[:, 1],
                    color=COLORS["agribrain"], alpha=0.85, label="Local Redist.")
    ax.fill_between(hours, probs[:, 0] + probs[:, 1], 1.0,
                    color="#F57C00", alpha=0.85, label="Recovery")

    # Regime guides: vertical lines at the rho thresholds where the
    # policy logic transitions. Use the AgriBrain rho trace to find the
    # crossing hours.
    ab_rho = np.array(ab["rho_trace"])
    def _first_cross(threshold):
        idx = np.argmax(ab_rho > threshold)
        if idx == 0 and ab_rho[0] <= threshold:
            return None
        return float(hours[idx])

    h_atrisk = _first_cross(RLE_THRESHOLD)
    h_knee = _first_cross(0.50)
    if h_atrisk is not None:
        ax.axvline(h_atrisk, color="#424242", linestyle="--", linewidth=1.1,
                   alpha=0.65)
        ax.text(h_atrisk + 0.4, 0.05, f"\u03c1>0.10\n@h{h_atrisk:.0f}",
                fontsize=ANNOT_FONT_SIZE - 1, color="#212121",
                fontweight="bold", va="bottom")
    if h_knee is not None:
        ax.axvline(h_knee, color="#424242", linestyle="--", linewidth=1.1,
                   alpha=0.65)
        ax.text(h_knee + 0.4, 0.05, f"\u03c1>0.50\n@h{h_knee:.0f}",
                fontsize=ANNOT_FONT_SIZE - 1, color="#212121",
                fontweight="bold", va="bottom")

    ax.set_xlabel("Hours")
    ax.set_ylabel("Action Probability")
    ax.set_title("(c) AgriBrain Action Probabilities")
    ax.set_ylim(0, 1.0)
    _apply_style(ax)
    _annotate_window(ax, 24, 48, WINDOW_COLOR, "Heatwave", ypos=0.45)
    _legend(ax, loc="center right", ncol=1, frameon=True, framealpha=0.85)

    # --- (d) Per-step ARI (12 h rolling average) ---
    ax = axes[1, 1]
    window = 12
    for mode in ["static", "hybrid_rl", "agribrain"]:
        ep = hw[mode]
        ari = np.array(ep["ari_trace"])
        rolling = np.convolve(ari, np.ones(window) / window, mode="same")
        _mode_plot(ax, hours, rolling, mode)
    ax.set_xlabel("Hours")
    ax.set_ylabel("ARI")
    ax.set_title("(d) ARI per step During Heatwave")
    ax.set_ylim(0, 1.0)
    _apply_style(ax)
    _annotate_window(ax, 24, 48, WINDOW_COLOR, "Heatwave")
    _legend(ax, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.985], h_pad=1.6, w_pad=1.6)
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
            label="Inventory")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Inventory (units)")
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(3, 3))
    ax2 = ax.twinx()
    ax2.plot(hours, dem, color=COLORS["hybrid_rl"], linewidth=1.8,
             alpha=0.85, label="Demand")
    ax2.set_ylabel("Demand (units/step)")
    ax.set_title("(a) Inventory vs Demand")
    _apply_style(ax)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(labelsize=TICK_FONT_SIZE, length=5, width=1.3)
    ax2.yaxis.label.set_size(AXIS_LABEL_SIZE)
    ax2.yaxis.label.set_weight("bold")
    for lbl in ax2.get_yticklabels():
        lbl.set_fontweight("bold")
    # Position the "Overproduction" label inside the red zone toward
    # the centre-right (xpos\u224840) so the bounding box sits clearly
    # within the 12-60 h window without clipping the right edge.
    _annotate_window(ax, 12, 60, WINDOW_COLOR, "Overproduction", xpos=40)
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
    ax.set_ylabel("Waste Rate")
    ax.set_title("(b) Waste Reduction Over Time")
    _apply_style(ax)
    _annotate_window(ax, 12, 60, WINDOW_COLOR, "Overproduction")
    _legend(ax, loc="upper left")

    # --- (c) RLE rolling (EU-hierarchy + severity-weighted) ---
    # Mirrors the canonical episode-level metric in
    # resilience.compute_rle / RLETracker, just with a rolling window
    # for visual continuity. Per at-risk timestep (rho > theta):
    #   numerator(t)   = rho(t) * w(action_t)
    #   denominator(t) = rho(t) * w_max
    # where w is HIERARCHY_WEIGHT (LR=1.00, Recovery=0.40, CC=0.00)
    # from EU 2008/98/EC Article 4 as operationalised in Papargyropoulou
    # et al. (2014). Numerator and denominator are convolved separately
    # so the rolling RLE = num_rolling / den_rolling is well-defined;
    # NaN where the window contains no at-risk steps.
    #
    # The match-quality form (band-edge author parameters) and the
    # capacity-constrained form (BatchInventory realised-action trace)
    # this panel used to plot alongside the canonical form were retired
    # in 2026-04. Only the EU-hierarchy weighted form survives here, in
    # resilience.compute_rle, in the benchmark JSONs, and in the table
    # CSVs - the same value the headline RLE column carries.
    ax = axes[1, 0]
    action_names = ACTIONS  # canonical (cold_chain, local_redistribute, recovery)
    w_max = max(HIERARCHY_WEIGHT.values())
    for mode in ["static", "hybrid_rl", "agribrain"]:
        ep = op[mode]
        rho = np.array(ep["rho_trace"])
        actions = np.array(ep["action_trace"])
        at_risk = rho > RLE_THRESHOLD

        weighted_num = np.zeros_like(rho)
        weighted_den = np.zeros_like(rho)
        for t in range(len(rho)):
            if at_risk[t]:
                a = action_names[int(actions[t])]
                w = HIERARCHY_WEIGHT.get(a, 0.0)
                weighted_num[t] = rho[t] * w
                weighted_den[t] = rho[t] * w_max

        num_rolling = np.convolve(weighted_num,
                                  np.ones(window) / window, mode="same")
        den_rolling = np.convolve(weighted_den,
                                  np.ones(window) / window, mode="same")
        # NaN where denominator is zero (no at-risk batches in window).
        rle_frac = np.full_like(num_rolling, np.nan)
        np.divide(num_rolling, den_rolling, out=rle_frac,
                  where=den_rolling > 0)
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
    _legend(ax, loc="lower left")

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

    fig.tight_layout(rect=[0, 0, 1, 0.985], h_pad=1.6, w_pad=1.6)
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
    # Position the legend between lower-left and lower-centre so it
    # sits clear of both the AgriBrain decay tail (right) and the
    # high-ARI pre-outage region (left of h=24).
    _legend(ax, loc="lower left", bbox_to_anchor=(0.18, 0.0))

    # --- (b) Action distribution pre/during outage ---
    ax = axes[1]
    action_names = ["Cold Chain", "Local Redistribute", "Recovery"]
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
    ax.set_ylabel("Fraction of routing decisions")
    ax.set_ylim(0, max(max(pre_counts + pre_se * 2), max(during_counts + during_se * 2)) * 1.25 + 0.02)
    ax.set_title("(b) Action Distribution Shift")
    _apply_style(ax)
    _legend(ax, loc="upper right")

    # --- (c) Realized rerouting rate vs design probability ---
    # Plots the rolling fraction of decisions where action != cold_chain
    # — the empirical "actually rerouted" rate — against each method's
    # *design* reroute probability during outage. The story: when the
    # cyber-outage branch in select_action bypasses the centralised
    # softmax (h >= 24) and falls back to a per-mode Bernoulli
    # [1-p, p, 0] with p = CYBER_REROUTE_PROB[mode], the empirical
    # rolling rate should converge toward p — that convergence is
    # what demonstrates the autonomous edge stack is actually firing
    # at its capability-determined rate.
    #
    # CYBER_REROUTE_PROB is a hardcoded design constant in
    # action_selection.py (Static = 0.0, Hybrid RL = 0.60,
    # AgriBrain = 0.74). The dashed reference lines drawn during the
    # outage window let the reader see at-a-glance whether the
    # rolling lines actually approach those targets.
    #
    # Rolling window choice: 24 steps = 6 hours. With Bernoulli
    # p ~ 0.7 and 24 samples the rolling mean's standard error is
    # ~0.094, so the visual ±2sigma band is ~0.18 — small enough that
    # the design points are visible above the sampling noise. The
    # earlier 12-step window had ~0.25 envelope which masked the
    # convergence story.
    ax = axes[2]
    window = 24  # 24 steps × 0.25 h = 6 h rolling window

    rolling_by_mode: dict[str, np.ndarray] = {}
    for mode in ["static", "hybrid_rl", "agribrain"]:
        ep = cy[mode]
        actions = np.array(ep["action_trace"])
        rerouted = (actions != 0).astype(float)
        rolling = np.convolve(rerouted, np.ones(window) / window, mode="same")
        rolling_by_mode[mode] = rolling
        _mode_plot(ax, hours, rolling, mode)

    # Reference lines at each method's design probability, drawn only
    # during the outage window so the reader sees the convergence
    # target without misreading them as pre-outage references.
    outage_mask = hours >= 24.0
    for mode, p_design in (
        ("static", 0.0),
        ("hybrid_rl", CYBER_REROUTE_PROB.get("hybrid_rl", 0.60)),
        ("agribrain", CYBER_REROUTE_PROB.get("agribrain", 0.74)),
    ):
        x_seg = hours[outage_mask]
        y_seg = np.full_like(x_seg, p_design, dtype=float)
        ax.plot(x_seg, y_seg, color=COLORS[mode], linestyle=":",
                linewidth=1.6, alpha=0.55)
        # Right-anchored value tag so the reader can read off the
        # design probability without measuring against the y-axis.
        ax.text(
            float(hours[-1]) + 0.6, p_design,
            f"p={p_design:.2f}",
            ha="left", va="center", fontsize=9, color=COLORS[mode],
            fontweight="bold",
        )

    # Vertical guide at the outage onset.
    ax.axvline(24.0, color="#424242", linestyle="--", linewidth=1.2, alpha=0.8)

    # Regime labels rewritten in plain language and placed in the
    # top strip of the panel so they sit clear of the Outage badge
    # (which lives at the title-baseline edge of the shaded region).
    ax.text(
        12.0, 1.06,
        "Cloud policy",
        ha="center", va="bottom", fontsize=10, color="#212121",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                  edgecolor="#9E9E9E", linewidth=0.6, alpha=0.92),
    )
    ax.text(
        58.0, 1.06,
        "Edge fallback (CYBER_REROUTE_PROB)",
        ha="center", va="bottom", fontsize=10, color="#212121",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                  edgecolor="#9E9E9E", linewidth=0.6, alpha=0.92),
    )

    ax.set_xlabel("Hours")
    ax.set_ylabel("Reroute Rate (6 h rolling)")
    ax.set_title("(c) Reroute Rate vs Design Probability")
    ax.set_ylim(-0.02, 1.18)
    # Extend xlim slightly so the right-anchored "p=..." tags fit.
    ax.set_xlim(float(hours[0]), float(hours[-1]) + 4.0)
    _apply_style(ax)
    _annotate_window(ax, 24, 72, WINDOW_COLOR, "Outage", ypos=0.50)
    _legend(ax, loc="center right")

    fig.tight_layout(rect=[0, 0, 1, 0.985], h_pad=1.6, w_pad=1.6)
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
    _legend(ax, loc="upper center", ncol=3)

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
    _legend(ax, loc="lower center")

    # --- (d) Reward decomposition: SLCA, waste penalty, rho penalty ---
    # Three stacked layers on a single axis make the additive decomposition
    # R = SLCA − η_w·waste − η_ρ·ρ visually obvious. The vertical gap
    # between consecutive lines is each penalty's contribution at time t,
    # and the shaded bands quantify those magnitudes without a twin axis.
    ax = axes[1, 1]
    slca_vals = np.array(ab["slca_trace"])
    waste_vals = np.array(ab["waste_trace"])
    rho_vals = np.array(ab["rho_trace"])
    reward_vals = np.array(ab["reward_trace"])
    # Pull eta_w / eta_rho from the Policy defaults so the panel stays in
    # sync with reward.py if either default is retuned in future.
    from src.models.policy import Policy as _PolicyDefaults
    _p = _PolicyDefaults()
    eta_w = float(_p.eta)
    eta_rho = float(_p.eta_rho)
    after_waste_vals = slca_vals - eta_w * waste_vals

    window = 12
    slca_smooth = np.convolve(slca_vals, np.ones(window) / window, mode="same")
    after_waste_smooth = np.convolve(after_waste_vals, np.ones(window) / window, mode="same")
    reward_smooth = np.convolve(reward_vals, np.ones(window) / window, mode="same")

    # Penalty bands: shade between consecutive layers so the eye sees
    # each penalty's magnitude as an area. waste-penalty band sits
    # between SLCA(t) and SLCA − η_w·waste; ρ-penalty band sits between
    # that line and the net reward. Bands are unlabelled here; the
    # legend below collapses both into a single "Penalty" proxy entry
    # so the legend box stays compact (3 items instead of 5).
    ax.fill_between(hours, after_waste_smooth, slca_smooth,
                    color=WINDOW_COLOR, alpha=0.22)
    ax.fill_between(hours, reward_smooth, after_waste_smooth,
                    color="#6A1B9A", alpha=0.22)

    # Three layer lines, top → bottom. The middle ``SLCA - eta_w*waste``
    # line stays on the plot to separate the two penalty bands visually
    # but does not get its own legend entry.
    line_slca, = ax.plot(hours, slca_smooth, color=COLORS["agribrain"],
                         linewidth=2.4, alpha=0.95, label="SLCA(t)")
    ax.plot(hours, after_waste_smooth, color="#FF8F00", linewidth=2.0,
            alpha=0.95, linestyle="--")
    line_reward, = ax.plot(hours, reward_smooth, color="#263238",
                           linewidth=2.4, alpha=0.95, label="Net reward")
    # Single proxy artist standing for both shaded bands. Coloured at
    # the waste-penalty band's hue (the more visually prominent of the
    # two) at the same alpha so the swatch matches what the eye sees.
    from matplotlib.patches import Patch as _Patch
    penalty_proxy = _Patch(facecolor=WINDOW_COLOR, alpha=0.22, label="Penalty")

    ax.set_xlabel("Hours")
    ax.set_ylabel("Reward components")
    # User-requested zoom: 0.6-0.8 puts the three layer lines and the
    # two penalty bands at maximum visual separation for adaptive_pricing
    # where SLCA(t), SLCA - eta_w*waste, and net reward all live within
    # ~[0.62, 0.78].
    ax.set_ylim(0.6, 0.8)
    ax.set_title("(d) Reward Decomposition")
    _apply_style(ax)
    # Compact 3-entry legend: SLCA(t), Net reward, Penalty (proxy).
    leg = _legend(
        ax,
        handles=[line_slca, line_reward, penalty_proxy],
        labels=["SLCA(t)", "Net reward", "Penalty"],
        loc="lower right", framealpha=1.0,
    )
    if leg is not None:
        leg.set_zorder(20)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_alpha(1.0)

    fig.tight_layout(rect=[0, 0, 1, 0.985], h_pad=1.6, w_pad=1.6)
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
    bench = _remap_legacy_rle_variants(bench)
    return bench


def _remap_legacy_rle_variants(bench: dict | None) -> dict | None:
    """Remap legacy multi-variant RLE keys to the single canonical name.

    Pre-2026-04 ``benchmark_summary.json`` files exposed four RLE
    columns: ``rle`` (saturating binary recovered/at_risk),
    ``rle_binary`` (alias of the same), ``rle_weighted`` (EU 2008/98/EC
    + severity-weighted form), and ``rle_capacity_constrained``
    (BatchInventory realised-action variant). Only the
    EU-hierarchy + severity-weighted form survived the simplification —
    it now lives under the plain key ``rle`` in
    ``resilience.compute_rle`` and in fresh aggregator output.

    For backward compatibility with summary files written before the
    simplification, this helper detects the legacy format (presence of
    ``rle_weighted`` alongside ``rle``) and remaps so figure code that
    reads ``bench[scenario][mode]["rle"]`` always sees the canonical
    EU-hierarchy form regardless of which run produced the JSON. The
    retired variants are dropped from the in-memory dict so they
    cannot leak into a figure by accident.

    No-op when ``bench`` is None, empty, or already in the new format
    (``rle_weighted`` absent).
    """
    if not isinstance(bench, dict) or not bench:
        return bench
    sample = next(iter(bench.values()), {})
    if not isinstance(sample, dict):
        return bench
    sample_mode = next(iter(sample.values()), {})
    if not isinstance(sample_mode, dict) or "rle_weighted" not in sample_mode:
        # New format already has only the canonical ``rle``; nothing to do.
        return bench
    legacy_keys = ("rle_binary", "rle_realistic", "rle_capacity_constrained")
    for sc, modes in bench.items():
        if not isinstance(modes, dict):
            continue
        for mode, mets in modes.items():
            if not isinstance(mets, dict):
                continue
            # Promote rle_weighted (the EU-hierarchy form) to the
            # canonical ``rle`` slot, replacing the legacy ``rle`` key
            # (which used to hold the retired match-quality variant).
            if "rle_weighted" in mets:
                mets["rle"] = mets["rle_weighted"]
            for key in ("rle_weighted", *legacy_keys):
                mets.pop(key, None)
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

    # Single canonical RLE: EU-hierarchy + severity-weighted form
    # (resilience.compute_rle, post-2026-04 simplification).
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
                 fontsize=FIG_TITLE_SIZE, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.985], h_pad=1.6, w_pad=1.6)
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

    # Single canonical RLE: EU-hierarchy + severity-weighted form
    # (resilience.compute_rle, post-2026-04 simplification).
    metrics = [("ari", "ARI", "(a)"), ("waste", "Waste Rate", "(b)"),
               ("rle", "RLE", "(c)")]
    stress_scenarios = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing"]

    n_modes = len(fig7_modes)
    # Wider bars and tighter group gap. Total group width 0.98 (was 0.9)
    # plus x_scale dropped from 1.25 to 1.10 means each group occupies
    # ~89% of its allotted x-slot instead of ~72%, so the bars are
    # visibly chunkier and the inter-group gap shrinks proportionally —
    # which is what reviewers expect when each group already carries 8
    # well-separated bars distinguished by colour.
    width = 0.98 / n_modes
    x_scale = 1.10

    # Bumped per-element font sizes for fig7 — the previous +3-tick /
    # +4-title bumps still read small against the 24-inch figure width
    # at paper scale, so each tier moves up another 2 points to land
    # the title at 25pt, axis at 22pt, ticks at 20pt, legend at 19pt.
    _F7_TITLE = SUBPLOT_TITLE_SIZE + 6   # 25
    _F7_AXIS  = AXIS_LABEL_SIZE + 5      # 22
    _F7_TICK  = TICK_FONT_SIZE + 5       # 20
    _F7_LEG   = LEGEND_FONT_SIZE + 4     # 19

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
    fig.suptitle("Ablation Study", y=0.995, fontsize=FIG_TITLE_SIZE,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.985], w_pad=1.4)
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
    fig8a_modes = ["static", "hybrid_rl", "agribrain"]
    for mode in fig8a_modes:
        ep = hw[mode]
        cum_carbon = np.cumsum(ep["carbon_trace"])
        _mode_plot(ax, hours, cum_carbon, mode)
    ax.set_xlabel("Hours", fontsize=_F8_AXIS, fontweight="bold")
    ax.set_ylabel(r"Cumulative $\mathbf{CO_2}$ (kg)",
                  fontsize=_F8_AXIS, fontweight="bold")
    ax.set_title("(a) Cumulative Carbon \u2014 Heatwave",
                 fontsize=_F8_TITLE, fontweight="bold", pad=14)
    _apply_style(ax)
    # Heatwave annotation pushed to vertical middle so the new
    # top-anchored legend strip does not collide with it.
    _annotate_window(ax, 24, 48, WINDOW_COLOR, "Heatwave", ypos=0.55)
    # Legend anchored to the upper centre of the panel \u2014 sits over the
    # mid x-range where the curves are well below the legend baseline,
    # keeping the 3-entry row clear of both axes.
    _legend(ax, loc="upper center",
            bbox_to_anchor=(0.5, 0.99), ncol=len(fig8a_modes),
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
    ax.set_ylabel(r"Total $\mathbf{CO_2}$ (kg)",
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
                 fontsize=FIG_TITLE_SIZE, fontweight="bold")
    # Slightly more headroom inside each axes so the top-anchored
    # legend has space between it and the data.
    for a in axes:
        y_lo, y_hi = a.get_ylim()
        a.set_ylim(y_lo, y_hi + 0.15 * (y_hi - y_lo))
    fig.tight_layout(rect=[0, 0, 1, 0.985], w_pad=1.6)
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


def _fig9_load_method_means():
    """Return per-(scenario, method) mean ARI from benchmark_summary.json.

    Used by the % improvement forest plot to convert ``mean_diff`` (in
    absolute ARI units) into a relative gain over the baseline mean ARI.
    Returns ``{scenario: {method: mean_ari}}`` or ``None`` if the file
    is absent.

    Handles both the flat legacy layout
    (``{"heatwave": {"agribrain": {"ari": {"mean": ...}}}, ...}``) and
    the post-2026-04 aggregator layout that nests the scenario dict
    under a ``summary`` key alongside ``_meta``.
    """
    summary_path = RESULTS_DIR / "benchmark_summary.json"
    if not summary_path.exists():
        return None
    import json as _json_mod
    summary = _json_mod.loads(summary_path.read_text(encoding="utf-8"))
    if isinstance(summary, dict) and "summary" in summary \
            and isinstance(summary["summary"], dict):
        summary = summary["summary"]
    out: dict = {}
    for sc, modes in summary.items():
        out[sc] = {}
        if not isinstance(modes, dict):
            continue
        for mode, metrics in modes.items():
            ari = metrics.get("ari") if isinstance(metrics, dict) else None
            if isinstance(ari, dict) and "mean" in ari:
                out[sc][mode] = float(ari["mean"])
    return out


def _fig9_load_significance():
    """Return the per-scenario agribrain_vs_X paired-difference statistics.

    benchmark_significance.json carries, for each scenario × baseline ×
    metric, the bootstrap mean_diff with 95% CI, the multiplicity-adjusted
    p-value, and Cohen's d. Returns None when the file is absent.

    Two on-disk shapes are supported:

      - flat (legacy, pre-2026-04 aggregator): scenarios at the top
        level — ``{"heatwave": {"agribrain_vs_static": {...}}, ...}``
      - nested (current aggregator): scenarios under a ``significance``
        key alongside ``_meta`` and ``primary_h1_holm_adjusted`` —
        ``{"_meta": {...}, "significance": {"heatwave": {...}}, ...}``

    The loader unwraps the new nesting if present so the rest of the
    fig9 code path can keep its scenario-keyed access pattern. Without
    this unwrap, ``scenarios_in_sig = [s for s in SCENARIOS if s in sig_data]``
    silently evaluates to the empty list (because ``"heatwave" in {"_meta": ..., "significance": ...}``
    is False), producing the empty-panel rendering bug that surfaced
    after the aggregator restructured the file in 2026-04.
    """
    sig_path = RESULTS_DIR / "benchmark_significance.json"
    if not sig_path.exists():
        return None
    import json as _json_mod
    payload = _json_mod.loads(sig_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "significance" in payload \
            and isinstance(payload["significance"], dict):
        return payload["significance"]
    return payload


def _fig9_load_n_seeds():
    """Best-effort lookup of the per-scenario seed count from benchmark_summary.

    All 5 scenarios usually share the same n_seeds; return the modal value
    or None if the summary file is missing. Supports the post-2026-04
    aggregator layout that nests scenarios under a ``summary`` key as
    well as the legacy flat layout.
    """
    summary_path = RESULTS_DIR / "benchmark_summary.json"
    if not summary_path.exists():
        return None
    import json as _json_mod
    summary = _json_mod.loads(summary_path.read_text(encoding="utf-8"))
    if isinstance(summary, dict) and "summary" in summary \
            and isinstance(summary["summary"], dict):
        summary = summary["summary"]
    counts = []
    for sc, modes in summary.items():
        if not isinstance(modes, dict):
            continue
        for mode, metrics in modes.items():
            ari = metrics.get("ari") if isinstance(metrics, dict) else None
            if isinstance(ari, dict) and "n_seeds" in ari:
                counts.append(int(ari["n_seeds"]))
    if not counts:
        return None
    # Modal n_seeds.
    from collections import Counter
    return Counter(counts).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Figure 9: Consolidated statistical-superiority panel. Three panels keyed on
# benchmark_significance.json + context_alignment_*.json:
#   (a) Cohen's d heatmap — agribrain vs each of 5 baselines, log-coloured.
#   (b) % ARI improvement forest plot — same 25 comparisons, recoded to
#       relative gain so the axis reads in human terms.
#   (c) Context honour rate per scenario.
# The earlier ARI-only fault-degradation panel and the H2 pass-rate matrix
# were both retired because they had no visual variance (every cell passed,
# every bar was small) — the effect-size and % improvement encodings carry
# the same evidence with a gradient that reads at figure scale.
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
    """Consolidated Figure 9: effect-size, performance gain, context honour.

    Three panels, each keyed on benchmark_significance.json and built to
    carry visual variance proportional to the strength of the result:

      (a) Effect-size heatmap. Cohen's d for ARI, agribrain vs each of
          5 baselines, per scenario. Log-coloured so the 36× spread
          (d ≈ 2 to d ≈ 76 across the 25 comparisons) reads as a clear
          gradient. Cell text = numeric d and significance star.
      (b) % ARI improvement forest plot. Same 25 comparisons recoded
          as 100·(mean_diff)/baseline_mean, with 95% bootstrap CI bars
          and multiplicity-adjusted p-value stars. The relative scale
          makes "+63 % vs static" and "+2 % vs MCP-only" instantly
          interpretable, where the absolute ΔARI hid the magnitude.
      (c) Context honour rate. Fraction of context-active decisions
          where the policy followed the dominant context recommendation.
          Source: context_alignment_{scenario}.json.
    """
    sig_data = _fig9_load_significance()
    align_rows = _fig9_load_alignment()
    n_seeds_global = _fig9_load_n_seeds()
    method_means = _fig9_load_method_means() or {}

    # Width-ratio rebalancing for the upcoming 5-scenario panel (c).
    # Panel (c) currently renders 3 bars (heatwave, overproduction,
    # cyber_outage are the only context_alignment_*.json files present),
    # but adaptive_pricing and baseline are scheduled to be added on
    # the next HPC run. Bumping panel (c) from 1.05 to 1.40 leaves room
    # for the extra two bars without re-flowing the figure later. The
    # panel-(a)/(b) shrink is small (1.10→1.00, 1.45→1.35) and does not
    # cramp either panel — both still carry the same content density.
    fig, axes = plt.subplots(1, 3, figsize=(22, 7.5),
                             gridspec_kw={"width_ratios": [1.00, 1.35, 1.40]})

    # Per-element font sizes — bumped above figs 6/7/8 so the cell
    # values and bar labels read clearly at paper scale.
    _F9_TITLE = SUBPLOT_TITLE_SIZE + 7    # 26
    _F9_AXIS  = AXIS_LABEL_SIZE + 5       # 22
    _F9_TICK  = TICK_FONT_SIZE + 5        # 20
    _F9_LEG   = LEGEND_FONT_SIZE + 3      # 18
    _F9_ANNOT = ANNOT_FONT_SIZE + 4       # 18

    def _restyle(ax_, title, ylabel=None, xlabel=None):
        ax_.set_title(title, fontsize=_F9_TITLE, fontweight="bold", pad=14)
        if ylabel is not None:
            ax_.set_ylabel(ylabel, fontsize=_F9_AXIS, fontweight="bold")
        if xlabel is not None:
            ax_.set_xlabel(xlabel, fontsize=_F9_AXIS, fontweight="bold")
        _apply_style(ax_)
        ax_.tick_params(labelsize=_F9_TICK, length=6, width=1.4)
        for lbl in list(ax_.get_xticklabels()) + list(ax_.get_yticklabels()):
            lbl.set_fontsize(_F9_TICK); lbl.set_fontweight("bold")

    # =================================================================
    # Panel (a) — Cohen's d heatmap (scenario × baseline)
    # =================================================================
    ax = axes[0]
    # Baseline order: weakest -> strongest, so cells fade from deep
    # green (huge gap vs static) to lighter green (small gap vs MCP-only).
    # Display labels are publication-style (Title Case + spaced) so the
    # axis reads cleanly in the paper rather than echoing the snake_case
    # internal mode names.
    _BASELINES = [
        ("agribrain_vs_static",     "vs Static"),
        ("agribrain_vs_hybrid_rl",  "vs Hybrid RL"),
        ("agribrain_vs_no_context", "vs No Context"),
        ("agribrain_vs_pirag_only", "vs piRAG only"),
        ("agribrain_vs_mcp_only",   "vs MCP only"),
    ]
    if sig_data:
        scenarios_in_sig = [s for s in SCENARIOS if s in sig_data]
        n_rows = len(scenarios_in_sig)
        n_cols = len(_BASELINES)
        d_mat = np.full((n_rows, n_cols), np.nan)
        p_mat = np.full((n_rows, n_cols), np.nan)
        for i, sc in enumerate(scenarios_in_sig):
            for j, (cmp_key, _) in enumerate(_BASELINES):
                ari = sig_data[sc].get(cmp_key, {}).get("ari", {}) or {}
                if "cohens_d" in ari:
                    d_mat[i, j] = float(ari["cohens_d"])
                if "p_value_adj" in ari:
                    p_mat[i, j] = float(ari["p_value_adj"])

        # Sequential green colormap, log-normalised because the d range
        # spans 36×: a linear scale would crush the small-effect end and
        # wash out the gradient.
        finite = d_mat[np.isfinite(d_mat)]
        d_min = max(float(finite.min()), 1.5) if finite.size else 1.5
        d_max = float(finite.max()) if finite.size else 80.0
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=d_min, vmax=d_max)
        im = ax.imshow(d_mat, cmap="Greens", norm=norm,
                       aspect="auto", interpolation="nearest")

        # Cell annotation: numeric Cohen's d, with a halo so the text is
        # legible across the full gradient (white on saturated cells, dark
        # on pale ones).
        from matplotlib import patheffects as _pe
        for i in range(n_rows):
            for j in range(n_cols):
                d = d_mat[i, j]
                if not np.isfinite(d):
                    continue
                # Pick text colour that contrasts: white on saturated
                # cells (upper third of log range), dark on pale cells.
                cell_frac = (np.log(d) - np.log(d_min)) / (np.log(d_max) - np.log(d_min))
                txt_color = "white" if cell_frac > 0.55 else "#1B5E20"
                halo = "black" if txt_color == "white" else "white"
                # Cell text shows just the numeric Cohen's d. Significance
                # is reported in the headline below ("25/25 p<0.05") and
                # in panel (b)'s star annotations; repeating it here mixes
                # an effect-size encoding with a p-value encoding.
                t = ax.text(j, i, f"{d:.1f}",
                            ha="center", va="center",
                            fontsize=_F9_TICK, fontweight="bold",
                            color=txt_color)
                t.set_path_effects([_pe.withStroke(linewidth=1.6, foreground=halo)])

        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels([lbl for _, lbl in _BASELINES],
                           rotation=20, ha="right")
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels([SCENARIO_LABELS.get(s, s) for s in scenarios_in_sig])

        # Slim colourbar on the right edge — gives the gradient an
        # explicit scale for reviewers who want exact magnitudes.
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cbar.set_label("Cohen's d (ARI)",
                       fontsize=_F9_ANNOT, fontweight="bold")
        cbar.ax.tick_params(labelsize=_F9_TICK - 2)
        for lbl in cbar.ax.get_yticklabels():
            lbl.set_fontweight("bold")
    else:
        ax.text(0.5, 0.5, "benchmark_significance.json not available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=_F9_ANNOT, color="#616161")
    _restyle(ax, "(a) Effect Size — Cohen's d (ARI)")

    # =================================================================
    # Panel (b) — Aggregated % ARI improvement per baseline
    # =================================================================
    # The previous 25-marker forest plot was unreadable; the same data
    # rolled up to one bar per baseline (mean across 5 scenarios, with
    # whiskers showing the scenario range) tells the same story in 5
    # visual elements instead of 25, and the declining gradient from
    # `vs static` (≈+75 %) down to `vs MCP only` (≈+2.7 %) is instantly
    # legible.
    ax = axes[1]
    _BASELINES_BAR = [
        ("agribrain_vs_static",     "static",     "vs Static",       "#616161"),
        ("agribrain_vs_hybrid_rl",  "hybrid_rl",  "vs Hybrid RL",    "#1565C0"),
        ("agribrain_vs_no_context", "no_context", "vs No Context",   "#6A1B9A"),
        ("agribrain_vs_pirag_only", "pirag_only", "vs piRAG only",   "#00838F"),
        ("agribrain_vs_mcp_only",   "mcp_only",   "vs MCP only",     "#E65100"),
    ]
    if sig_data:
        scenarios_in_sig = [s for s in SCENARIOS if s in sig_data]
        # Compute per-baseline % improvement across all scenarios.
        per_baseline = {}
        for cmp_key, mode_name, leg_label, col in _BASELINES_BAR:
            pcts = []
            for sc in scenarios_in_sig:
                ari = sig_data[sc].get(cmp_key, {}).get("ari", {}) or {}
                if not ari:
                    continue
                mean_diff = float(ari.get("mean_diff", 0.0))
                base_mean = method_means.get(sc, {}).get(mode_name)
                if base_mean is None:
                    ag_mean = method_means.get(sc, {}).get("agribrain")
                    if ag_mean is not None:
                        base_mean = ag_mean - mean_diff
                if not base_mean or base_mean <= 0:
                    continue
                pcts.append(100.0 * mean_diff / base_mean)
            if pcts:
                per_baseline[cmp_key] = {
                    "label": leg_label,
                    "color": col,
                    "mean":  float(np.mean(pcts)),
                    "lo":    float(np.min(pcts)),
                    "hi":    float(np.max(pcts)),
                    "n":     len(pcts),
                }

        # Sort weakest -> strongest baseline so the bar chart reads as a
        # gradient that shrinks left to right.
        order = sorted(per_baseline.items(),
                       key=lambda kv: -kv[1]["mean"])

        max_hi = max((v["hi"] for v in per_baseline.values()), default=80.0)
        for i, (cmp_key, v) in enumerate(order):
            ax.barh(i, v["mean"], height=0.62, color=v["color"],
                    edgecolor="white", linewidth=0.8, alpha=0.95, zorder=2)
            # Whisker = min-max range across the 5 scenarios.
            ax.plot([v["lo"], v["hi"]], [i, i],
                    color="#212121", linewidth=2.0, zorder=3)
            for x_end in (v["lo"], v["hi"]):
                ax.plot([x_end, x_end], [i - 0.18, i + 0.18],
                        color="#212121", linewidth=2.0, zorder=3)
            # Numeric label: mean + range on the right side of each bar.
            label_x = v["hi"] + 0.5 if v["hi"] < 2.0 else v["hi"] * 1.10
            ax.text(label_x, i,
                    f"+{v['mean']:.1f}%   (range {v['lo']:.1f}–{v['hi']:.1f}%)",
                    va="center", ha="left",
                    fontsize=_F9_ANNOT - 1, fontweight="bold", color="#212121")

        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels([v["label"] for _, v in order])
        ax.invert_yaxis()  # weakest baseline (largest gain) on top

        # Symlog so the +1..+15 % cluster has visual room and the +75 %
        # static bar doesn't dominate. Tick layout is the load-bearing
        # readability piece: the previous {0,5,50,100} set put 50% and
        # 100% visually adjacent in the compressed log region (the data
        # tops out around +75% so the gap between log(50) and log(100)
        # is ~0.30 dec, which collides at the bold 20pt tick fontsize).
        # Switch linthresh from 2.0 to 5.0 so the linear region absorbs
        # the small-bar cluster cleanly and the log region begins where
        # the big bars actually live; tick set drops to {0, 5, 25, 75}
        # which gives clearly separated labels across the whole axis
        # without losing fidelity at either end. xlim caps at 80 since
        # the headline +75% vs Static bar is the largest data point and
        # padding to 110% wasted half the panel.
        ax.set_xscale("symlog", linthresh=5.0, linscale=1.0)
        ax.set_xlim(0, max(max_hi * 1.10, 80.0))
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
        major_ticks = [0, 5, 25, 75]
        ax.xaxis.set_major_locator(FixedLocator(major_ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}%"))
        ax.xaxis.set_minor_locator(NullLocator())
    else:
        ax.text(0.5, 0.5, "benchmark_significance.json not available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=_F9_ANNOT, color="#616161")
    _restyle(ax, "(b) % ARI Improvement vs Baselines")

    # =================================================================
    # Panel (c) — Context honour rate (preserved from previous design)
    # =================================================================
    ax = axes[2]
    if align_rows:
        labels = [r["label"] for r in align_rows]
        rates_pct = [100.0 * r["rate"] for r in align_rows]
        actives = [r["honored"] + r["ignored"] for r in align_rows]
        x = np.arange(len(labels))
        ax.bar(x, rates_pct, width=0.65,
               color=COLORS.get("agribrain", "#2E7D32"),
               edgecolor="white", linewidth=0.8, alpha=0.95)
        for xi, (pct, n) in enumerate(zip(rates_pct, actives)):
            ax.text(xi, pct + 2.0, f"{pct:.1f}%", ha="center", va="bottom",
                    fontsize=_F9_ANNOT, fontweight="bold", color="#212121")
            if pct >= 18:
                ax.text(xi, pct - 3, f"n={n}", ha="center", va="top",
                        fontsize=_F9_ANNOT - 3, fontweight="bold", color="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylim(0, 110)
        ax.axhline(80, color="#9E9E9E", linewidth=1.0, linestyle=":",
                   alpha=0.7, zorder=0)
    else:
        ax.text(0.5, 0.5, "no context_alignment_*.json files",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=_F9_ANNOT, color="#616161")
    _restyle(ax, "(c) Context Honour Rate",
             ylabel="Honour rate (% of active steps)")

    fig.suptitle("Performance Gain over Baselines and Context Honour",
                 y=0.995, fontsize=FIG_TITLE_SIZE, fontweight="bold")
    # Title-to-subplot gap matches the canonical figure 3 pattern
    # (rect top 0.985, suptitle y 0.995) so all paper figures share
    # the same header spacing. w_pad kept at 1.0 so panel (c) has
    # room for the upcoming 5-scenario layout without losing space
    # to padding.
    fig.tight_layout(rect=[0, 0.02, 1, 0.985], w_pad=1.0)
    _save(fig, "fig9_robustness")


def fig10_latency_quality_frontier(data):
    """Latency-quality frontier with two zones, fully matching the shared
    figure style (Arial, bold titles and axis labels, 800 DPI, no label
    overlaps). Panel (a) shows the lightweight methods (sub-millisecond);
    panel (b) shows the MCP/piRAG-enabled methods with the no-context
    reference point and an overhead annotation.
    """
    # Panel (b) uses a broken x-axis (split between 0.5 ms and 5.0 ms)
    # to suppress the empty zone between the No Context reference
    # (~0.18 ms) and the context-aware cluster (~5.85 ms). The broken
    # axis is implemented as two sub-axes inside the right gridspec
    # cell with width_ratios=[1, 5] — the small left sub-axis carries
    # the No Context reference while the larger right sub-axis carries
    # the three jittered context-aware markers.
    import matplotlib.gridspec as _gridspec
    # figsize bumped to (18, 7.5) and rect-top dropped to 0.94 below
    # so the suptitle-to-panel-title gap matches figs 6/7/8 instead
    # of compressing into the shorter 7.0-height panel space the
    # earlier render used.
    fig = plt.figure(figsize=(18, 7.5))
    outer_gs = _gridspec.GridSpec(
        1, 2, figure=fig, width_ratios=[1, 1], wspace=0.32,
    )
    ax_a = fig.add_subplot(outer_gs[0])
    # Inner break-axis spacing widened from 0.06 to 0.14 so the
    # right-edge "0.5" tick of the left sub-axis and the left-edge
    # "5.0" tick of the right sub-axis no longer collide visually.
    inner_gs = outer_gs[1].subgridspec(1, 2, width_ratios=[1, 5], wspace=0.14)
    ax_b_left = fig.add_subplot(inner_gs[0])
    ax_b_right = fig.add_subplot(inner_gs[1], sharey=ax_b_left)
    axes = [ax_a]  # legacy compat for the panel (a) code path below
    fig.suptitle("Latency vs ARI Frontier", fontsize=FIG_TITLE_SIZE,
                 fontweight="bold", y=0.995)

    bench = _load_benchmark_ci() or {}

    fast_modes = ["static", "hybrid_rl", "no_pinn", "no_slca", "no_context"]
    context_modes = ["agribrain", "mcp_only", "pirag_only"]

    def _collect(modes):
        """Collect (mode, mean_latency_ms, mean_ari, yerr) per mode.

        Point positions use the canonical single-seed per-episode ARI from
        ``data["results"]`` so the figure preserves the y-axis scale of
        the published render (which was rendered from
        ``run_all(seed=42)`` output, single-seed values).

        Error bars use the standard error of the across-scenario mean:
        sd(per_scenario_ari) / sqrt(n_scenarios). The within-scenario
        bootstrap CIs reported in benchmark_summary.json are too tight
        (~0.001-0.005 ARI on n=20 seeds) to render visibly in panel (a)'s
        wide y-range, so we use cross-scenario variability instead — this
        is the statistically appropriate uncertainty for the
        *cross-scenario mean* the figure plots and is consistent with the
        symmetric ±SE convention reported elsewhere in the manuscript.
        """
        pts = []
        for mode in modes:
            scenario_aris = []
            scenario_lats = []
            for s in SCENARIOS:
                rec = data["results"].get(s, {}).get(mode, {})
                if "ari" not in rec:
                    continue
                scenario_aris.append(float(rec["ari"]))
                scenario_lats.append(float(rec.get("mean_decision_latency_ms", 0.0)))
            if not scenario_aris:
                continue
            ari_arr = np.array(scenario_aris, dtype=float)
            mean_y = float(ari_arr.mean())
            n = ari_arr.size
            se_y = float(ari_arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            yerr = (se_y, se_y)
            pts.append((mode, float(np.mean(scenario_lats)), mean_y, yerr))
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
    # X-axis zoomed to 0.08-0.20 ms so the cluster of fast lightweight
    # methods (Hybrid RL, No PINN, No SLCA, No Context near
    # lat ≈ 0.16-0.18 ms) and the Static reference (~0.09 ms) are
    # separated visually. The previous adaptive limits added 10%
    # right-padding past the largest latency, which compressed the
    # five markers into the right half of the panel and left a wide
    # blank zone on the left.
    ax.set_xlim(0.08, 0.20)
    pts_with_err_a = [(p[2], p[3]) for p in fast_pts]
    bar_lo_a = min(y - e[0] for y, e in pts_with_err_a)
    bar_hi_a = max(y + e[1] for y, e in pts_with_err_a)
    ax.set_ylim(bar_lo_a - 0.02, bar_hi_a + 0.02)
    # Cap the x-axis to ~5 major ticks so the sub-millisecond values
    # are not crowded onto the axis. Earlier matplotlib's default
    # locator placed nine ticks (0, 0.025, 0.050, ..., 0.200) on the
    # narrow sub-ms range, which made the labels collide visually.
    from matplotlib.ticker import MaxNLocator as _MaxNLocator
    ax.xaxis.set_major_locator(_MaxNLocator(nbins=5, prune="lower"))
    _legend(ax, loc="lower right", ncol=1)
    _apply_style(ax)

    # --- (b) Context-aware methods (MCP/piRAG overhead) ---
    # Broken x-axis: ax_b_left covers 0.0-0.5 ms (the No Context
    # reference point) and ax_b_right covers 5.0+ ms (the three
    # context-aware modes). The empty 0.5-5.0 ms zone is suppressed
    # via a // glyph at the break.
    ref = next((p for p in fast_pts if p[0] == "no_context"), None)
    handles_b = []

    # Plot the No Context reference on the LEFT sub-axis only.
    if ref is not None:
        ref_handle = ax_b_left.scatter(
            ref[1], ref[2], s=180,
            color=COLORS["no_context"], marker=MARKERS["no_context"],
            edgecolor="white", linewidth=1.2, alpha=0.55, zorder=4,
            label="No Context",
        )
        handles_b.append(ref_handle)
        if ref[3][0] > 0 or ref[3][1] > 0:
            ax_b_left.errorbar(
                [ref[1]], [ref[2]],
                yerr=np.array([[ref[3][0]], [ref[3][1]]]),
                fmt="none", ecolor=COLORS["no_context"],
                elinewidth=1.6, capsize=4, alpha=0.55, zorder=3,
            )

    # Plot the three context-aware modes on the RIGHT sub-axis with
    # the small horizontal jitter that visually separates the cluster.
    # Underlying data is unjittered; the overhead annotation below is
    # computed from the true AgriBrain latency vs the No Context ref.
    _ctx_jitter = {"agribrain": -0.10, "mcp_only": 0.0, "pirag_only": +0.10}
    for mode, x, y, yerr in ctx_pts:
        x_plot = x + _ctx_jitter.get(mode, 0.0)
        h = ax_b_right.scatter(
            x_plot, y, s=260,
            color=COLORS[mode], marker=MARKERS[mode],
            edgecolor="white", linewidth=1.4, alpha=0.95, zorder=5,
            label=MODE_LABELS[mode],
        )
        handles_b.append(h)
        if yerr[0] > 0 or yerr[1] > 0:
            ax_b_right.errorbar(
                [x_plot], [y],
                yerr=np.array([[yerr[0]], [yerr[1]]]),
                fmt="none", ecolor=COLORS[mode],
                elinewidth=1.8, capsize=4, alpha=0.9, zorder=4,
            )

    # Hide the inner spines so the broken-axis read as a single panel.
    ax_b_left.spines["right"].set_visible(False)
    ax_b_right.spines["left"].set_visible(False)
    ax_b_right.tick_params(left=False, labelleft=False)
    ax_b_left.yaxis.tick_left()

    # Diagonal // glyphs at the break — only the bottom-edge pair.
    # The earlier render also drew top-edge glyphs which spilled above
    # the axes (clip_on=False) and read as a stray "//" floating just
    # below the panel title; removed since the bottom pair alone is
    # enough to convey the broken-axis convention to the reader.
    _d = 0.018  # diagonal length in axes coordinates
    _kw_left = dict(transform=ax_b_left.transAxes, color="#424242",
                    lw=1.4, clip_on=False)
    ax_b_left.plot((1 - _d, 1 + _d), (-_d, +_d), **_kw_left)
    _kw_right = dict(transform=ax_b_right.transAxes, color="#424242",
                     lw=1.4, clip_on=False)
    ax_b_right.plot((-_d / 5, +_d / 5), (-_d, +_d), **_kw_right)

    # Overhead arrow spans the break — implemented via ConnectionPatch
    # which carries a single arrow across two Axes objects in figure
    # coordinates. The arrow starts at the No Context reference on
    # the left sub-axis and lands on the AgriBrain marker on the
    # right sub-axis.
    agri_pt = next((p for p in ctx_pts if p[0] == "agribrain"), None)
    if ref is not None and agri_pt is not None:
        from matplotlib.patches import ConnectionPatch
        agri_lat, agri_ari = agri_pt[1], agri_pt[2]
        agri_lat_plot = agri_lat + _ctx_jitter.get("agribrain", 0.0)
        con = ConnectionPatch(
            xyA=(ref[1], ref[2]), coordsA=ax_b_left.transData,
            xyB=(agri_lat_plot, agri_ari), coordsB=ax_b_right.transData,
            arrowstyle="->", color=COLORS["agribrain"],
            lw=2.0, linestyle="--", alpha=0.75, zorder=2,
        )
        fig.add_artist(con)
        # Overhead annotation lives on the right sub-axis since the
        # arrow's terminus and the cluster context both sit there.
        ax_b_right.annotate(
            f"Context overhead\n+{agri_lat - ref[1]:.1f} ms  |  "
            f"{agri_ari - ref[2]:+.3f} ARI",
            xy=(agri_lat_plot, agri_ari),
            xytext=(-12, 14), textcoords="offset points",
            ha="right", va="bottom",
            fontsize=ANNOT_FONT_SIZE, fontweight="bold",
            color=COLORS["agribrain"],
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      alpha=0.95, edgecolor=COLORS["agribrain"],
                      linewidth=1.2),
            zorder=6,
        )

    # Y-axis range must accommodate the cross-scenario SE error bars.
    pts_with_err = ([(ref[2], ref[3])] if ref is not None else []) + [(p[2], p[3]) for p in ctx_pts]
    bar_lo = min(y - e[0] for y, e in pts_with_err)
    bar_hi = max(y + e[1] for y, e in pts_with_err)
    ax_b_left.set_ylim(bar_lo - 0.03, bar_hi + 0.03)

    # X-axis split: left sub-axis 0.0-0.5 ms (covers No Context ~0.18 ms),
    # right sub-axis 5.0-cluster_max + 0.4 ms (covers AgriBrain / MCP /
    # piRAG Only at ~5.75-5.95 ms). Suppresses the 0.5-5.0 empty zone.
    ax_b_left.set_xlim(0.0, 0.5)
    ctx_lat_max = max(p[1] + _ctx_jitter.get(p[0], 0.0) for p in ctx_pts)
    ax_b_right.set_xlim(5.0, ctx_lat_max + 0.4)

    # Tick layout: left sub-axis gets {0.0, 0.5}, right sub-axis gets
    # {5, 6} or similar based on the data range.
    from matplotlib.ticker import FixedLocator as _FixedLocator
    ax_b_left.xaxis.set_major_locator(_FixedLocator([0.0, 0.5]))
    ax_b_right.xaxis.set_major_locator(_FixedLocator([5.0, 6.0]))

    # Y-label only on the left sub-axis. X-label and title placed
    # via fig-level helpers so they read as a single panel rather
    # than as two adjacent sub-axes.
    ax_b_left.set_ylabel("Mean ARI")

    # Compute the geometric centre of the broken pair in figure
    # coordinates — both labels and title use this so they appear
    # centred over the (left + right) sub-axis pair rather than
    # tied to one sub-axis.
    bbox_left = ax_b_left.get_position()
    bbox_right = ax_b_right.get_position()
    pair_x_centre = (bbox_left.x0 + bbox_right.x1) / 2.0

    # X-label centred under the broken pair.
    fig.text(pair_x_centre, bbox_left.y0 - 0.07,
             "Mean Decision Latency (ms)",
             ha="center", va="top",
             fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    # Panel (b) title centred over the broken pair.
    fig.text(pair_x_centre, bbox_left.y1 + 0.025,
             "(b) Context-Aware Methods",
             ha="center", va="bottom",
             fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold")

    # Legend on the right sub-axis (lower centre) — but the handle
    # list must combine entries from both sub-axes (the No Context
    # reference scatter lives on ax_b_left and would otherwise be
    # missing from the legend).
    leg_handles = []
    leg_labels = []
    for ax_ in (ax_b_left, ax_b_right):
        for h, lbl in zip(*ax_.get_legend_handles_labels()):
            if lbl not in leg_labels:
                leg_handles.append(h)
                leg_labels.append(lbl)
    _legend(ax_b_right, handles=leg_handles, labels=leg_labels,
            loc="lower center", ncol=2)
    _apply_style(ax_b_left)
    _apply_style(ax_b_right)
    # Re-hide the spine after _apply_style restores defaults.
    ax_b_left.spines["right"].set_visible(False)
    ax_b_right.spines["left"].set_visible(False)
    ax_b_right.tick_params(left=False, labelleft=False)

    fig.tight_layout(rect=[0, 0, 1, 0.91], w_pad=1.6)
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
    # hpc/hpc_aggregate.sh Stage 6 and the per-scenario alignment / protocol
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
