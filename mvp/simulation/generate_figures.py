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
  prefer bottom-center placement or outside-right; tight_layout plus
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
import matplotlib.pyplot as plt
from matplotlib import font_manager as _font_manager

# 2026-05 cross-platform font handling. Pre-2026-05 the config did:
#   - register Arial from Windows-only paths (C:\Windows\Fonts\arial*.ttf)
#   - set font.family = "Arial"
# This was Windows-correct and HPC-noisy: on a Linux render host with no
# Arial installed, every text element triggered a "findfont: Font
# family 'Arial' not found" warning (hundreds of warnings per figure
# render, flooding the console). Matplotlib still rendered the figure
# correctly because the font.sans-serif fallback chain caught the
# request, but the noise was unprofessional and the resulting glyphs
# were DejaVu Sans (slightly different metrics from Arial).
#
# Cross-platform fix:
#   1. Register Arial from Windows paths when available (Windows hosts).
#   2. Register Liberation Sans from common Linux font paths when
#      available -- Liberation Sans was designed by Red Hat with
#      metric-compatibility to Arial, so labels lay out the same.
#   3. Set font.family = "sans-serif" (the family GROUP, not the name)
#      and let font.sans-serif's priority list do the resolution.
#      Matplotlib walks the list, picks the first available, and
#      resolves silently -- no warning storm.
#   4. The mathtext.* keys downstream still set Arial-by-name; if Arial
#      isn't present matplotlib falls back to STIX (the canonical
#      math fallback) without warning. Acceptable cost for inline math.
_ARIAL_FONT_FILES = (
    # Windows
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\ariali.ttf",
    r"C:\Windows\Fonts\arialbi.ttf",
)
_LIBERATION_FONT_FILES = (
    # Linux Liberation Sans (Arial-compatible metrics)
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf",
    "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
    "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf",
    "/usr/share/fonts/liberation-sans/LiberationSans-Italic.ttf",
    "/usr/share/fonts/liberation-sans/LiberationSans-BoldItalic.ttf",
)
for _font_path in _ARIAL_FONT_FILES + _LIBERATION_FONT_FILES:
    if Path(_font_path).exists():
        try:
            _font_manager.fontManager.addfont(_font_path)
        except (OSError, RuntimeError):
            pass

from generate_results import run_all, SCENARIOS, RESULTS_DIR
from src.models.action_selection import (
    ACTIONS, RHO_RECOVERY_KNEE,
)
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
    # Use the family-group ("sans-serif") and let the priority list
    # below do the resolution. Pre-2026-05 this was hardcoded to
    # "Arial", which made matplotlib's font lookup fail loudly on
    # any host without Arial (the HPC render flooded stdout with
    # ~hundreds of "Font family 'Arial' not found" warnings before
    # silently falling back to DejaVu Sans). Setting the family
    # GROUP picks the first-available sans-serif from the list
    # without warnings, while still preferring Arial when the host
    # has it (Windows authoring) or Liberation Sans (Linux render).
    "font.family": "sans-serif",
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
    # 2026-04 sensitivity-mode additions: paired _static variants
    # (REINFORCE off so theta is the perturbed prior throughout the
    # episode), agribrain_no_bonus (SLCA bonus zeroed), and
    # theta_pert variants (THETA matrix perturbed). Mirror the
    # pert_*/_static teal-shade walk on the perturbation side so a
    # crowded legend stays distinguishable.
    "agribrain_pert_10_static":  "#1DE9B6",  # bright cyan-teal
    "agribrain_pert_25_static":  "#64FFDA",  # lighter cyan-teal
    "agribrain_pert_50_static":  "#A7FFEB",  # lightest cyan-teal
    "agribrain_no_bonus":        "#00897B",  # mid-dark teal
    "agribrain_theta_pert_10":   "#3949AB",  # indigo (different family)
    "agribrain_theta_pert_25":   "#5C6BC0",  # lighter indigo
    "agribrain_theta_pert_50":   "#9FA8DA",  # lightest indigo
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
    "agribrain_pert_10_static":  "p",
    "agribrain_pert_25_static":  "<",
    "agribrain_pert_50_static":  ">",
    "agribrain_no_bonus":        "x",
    "agribrain_theta_pert_10":   "1",
    "agribrain_theta_pert_25":   "2",
    "agribrain_theta_pert_50":   "3",
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
    "agribrain_pert_10_static":  (0, (5, 1, 2, 1)),
    "agribrain_pert_25_static":  (0, (4, 1, 2, 1)),
    "agribrain_pert_50_static":  (0, (3, 1, 2, 1)),
    "agribrain_no_bonus":        (0, (8, 2)),
    "agribrain_theta_pert_10":   (0, (6, 2, 1, 2)),
    "agribrain_theta_pert_25":   (0, (5, 2, 1, 2)),
    "agribrain_theta_pert_50":   (0, (4, 2, 1, 2)),
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
    "agribrain_pert_10_static":  "Pert 10% (static)",
    "agribrain_pert_25_static":  "Pert 25% (static)",
    "agribrain_pert_50_static":  "Pert 50% (static)",
    "agribrain_no_bonus":        "No Bonus",
    "agribrain_theta_pert_10":   "Theta Pert 10%",
    "agribrain_theta_pert_25":   "Theta Pert 25%",
    "agribrain_theta_pert_50":   "Theta Pert 50%",
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
    # Bold the scientific-notation offset text (e.g. the "1e3" tag that
    # matplotlib draws above the y-axis when ticklabel_format scilimits
    # are active). Today only fig 3 panel A triggers this -- inventory
    # values run into the tens of thousands -- but bolding it in the
    # shared style helper keeps every future panel consistent without a
    # per-callsite reminder.
    ax.xaxis.get_offset_text().set_fontweight("bold")
    ax.xaxis.get_offset_text().set_fontsize(TICK_FONT_SIZE)
    ax.yaxis.get_offset_text().set_fontweight("bold")
    ax.yaxis.get_offset_text().set_fontsize(TICK_FONT_SIZE)
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
                     ypos=0.93, xpos=None, va="top"):
    """Shade a scenario window and label it inside the plot.
    A one-shot ylim expansion guarantees the label sits in blank space
    above the data; callers that have locked ylim explicitly (ratio
    axes, for instance) are respected. ``ypos`` is the axes-fraction
    vertical position of the bbox edge specified by ``va``. Pass
    ``va="bottom"`` (and a low ``ypos`` such as 0.07) to anchor the
    label at the bottom of the panel; useful when the legend or the
    data peak occupy the top of the panel. ``xpos`` overrides the
    horizontal position (data coordinates); the default of ``None``
    centers the label on the window."""
    ax.axvspan(x0, x1, alpha=alpha, color=color, zorder=0)
    # Top-anchored callers (the default) get an automatic ylim bump so
    # the label never occludes data; bottom-anchored callers don't need
    # the bump (the lower spine is already empty space below the data
    # in every panel that uses bottom anchoring), and bumping it would
    # waste vertical real-estate.
    if (
        va == "top"
        and not getattr(ax, "_window_headroom_applied", False)
        and ax.get_autoscaley_on()
    ):
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
        ha="center", va=va,
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
    on carbon, labor, resilience, and price-transparency at modest
    rho cost.

    Panel (c) shows AgriBrain's action-probability stacked area with
    three regime guides: at-risk threshold crossing (rho >= 0.10),
    Recovery knee crossing (rho >= RHO_RECOVERY_KNEE), and post-
    heatwave fresh-batch cold-chain recovery. Knee threshold is
    imported from action_selection so the visual stays in sync with
    the policy module.

    Panel (d) plots per-step ARI (12 h rolling) - the composite metric
    the paper sells. ARI is bounded [0, 1] so the cross-method gap is
    directly interpretable.
    """
    hw = data["results"]["heatwave"]
    ab = hw["agribrain"]
    hours = np.array(ab["hours"])

    # Per-figure font-size bump for fig 2 (post-2026-04 user request).
    # Uniform +1 across body / ticks / axis labels / subplot titles /
    # suptitle / legend / in-plot annotations - a gentle bump that
    # keeps the relative hierarchy intact while reading slightly
    # larger. Scoped to this function via try/finally so other
    # figures (fig 3, fig 4, ...) keep the canonical global sizes.
    global BODY_FONT_SIZE, TICK_FONT_SIZE, AXIS_LABEL_SIZE
    global SUBPLOT_TITLE_SIZE, FIG_TITLE_SIZE, LEGEND_FONT_SIZE
    global ANNOT_FONT_SIZE
    _saved_sizes = (
        BODY_FONT_SIZE, TICK_FONT_SIZE, AXIS_LABEL_SIZE,
        SUBPLOT_TITLE_SIZE, FIG_TITLE_SIZE, LEGEND_FONT_SIZE,
        ANNOT_FONT_SIZE,
    )
    BODY_FONT_SIZE = _saved_sizes[0] + 1
    TICK_FONT_SIZE = _saved_sizes[1] + 1
    AXIS_LABEL_SIZE = _saved_sizes[2] + 1
    SUBPLOT_TITLE_SIZE = _saved_sizes[3] + 1
    FIG_TITLE_SIZE = _saved_sizes[4] + 1
    LEGEND_FONT_SIZE = _saved_sizes[5] + 1
    ANNOT_FONT_SIZE = _saved_sizes[6] + 1
    _saved_rc = {
        "font.size": plt.rcParams["font.size"],
        "axes.labelsize": plt.rcParams["axes.labelsize"],
        "axes.titlesize": plt.rcParams["axes.titlesize"],
        "xtick.labelsize": plt.rcParams["xtick.labelsize"],
        "ytick.labelsize": plt.rcParams["ytick.labelsize"],
        "legend.fontsize": plt.rcParams["legend.fontsize"],
        "legend.title_fontsize": plt.rcParams["legend.title_fontsize"],
        "figure.titlesize": plt.rcParams["figure.titlesize"],
    }
    plt.rcParams.update({
        "font.size": BODY_FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "axes.titlesize": SUBPLOT_TITLE_SIZE,
        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "legend.title_fontsize": LEGEND_FONT_SIZE,
        "figure.titlesize": FIG_TITLE_SIZE,
    })

    try:
        return _fig2_heatwave_inner(hw, ab, hours)
    finally:
        # Restore globals + rcParams so subsequent figures use the
        # canonical sizes regardless of how this function exited.
        (BODY_FONT_SIZE, TICK_FONT_SIZE, AXIS_LABEL_SIZE,
         SUBPLOT_TITLE_SIZE, FIG_TITLE_SIZE, LEGEND_FONT_SIZE,
         ANNOT_FONT_SIZE) = _saved_sizes
        plt.rcParams.update(_saved_rc)


def _fig2_heatwave_inner(hw, ab, hours):
    """Body of fig 2. Extracted from ``fig2_heatwave`` so the per-figure
    font-size overrides applied above can be cleanly torn down via
    try/finally regardless of how the body returns or raises."""
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
    # "Heatwave" annotation moved downward (ypos=0.45 -> sits in the
    # lower band of the heatwave window so it does not overlap the
    # temperature peak line); legend anchored on the left side with its
    # vertical center at 17.5 degC (mid-point of the 10-25 degC band)
    # so it sits between the cool pre-heatwave temperature curve below
    # and the heatwave peak above.
    _annotate_window(ax, 24, 48, WINDOW_COLOR, "Heatwave", ypos=0.45)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    _legend(ax, handles=h1 + h2, labels=l1 + l2,
            loc="center left",
            bbox_to_anchor=(0.02, 17.5),
            bbox_transform=ax.get_yaxis_transform(),
            framealpha=0.80)

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
               label="At-risk threshold")
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
    h_knee = _first_cross(RHO_RECOVERY_KNEE)
    if h_atrisk is not None:
        ax.axvline(h_atrisk, color="#424242", linestyle="--", linewidth=1.1,
                   alpha=0.65)
        ax.text(h_atrisk + 0.4, 0.05,
                f"\u03c1>{RLE_THRESHOLD:.2f}\n@h{h_atrisk:.0f}",
                fontsize=ANNOT_FONT_SIZE - 1, color="#212121",
                fontweight="bold", va="bottom")
    if h_knee is not None:
        ax.axvline(h_knee, color="#424242", linestyle="--", linewidth=1.1,
                   alpha=0.65)
        ax.text(h_knee + 0.4, 0.05,
                f"\u03c1>{RHO_RECOVERY_KNEE:.2f}\n@h{h_knee:.0f}",
                fontsize=ANNOT_FONT_SIZE - 1, color="#212121",
                fontweight="bold", va="bottom")

    ax.set_xlabel("Hours")
    ax.set_ylabel("Action Probability")
    ax.set_title("(c) AgriBrain Action Probabilities")
    ax.set_ylim(0, 1.0)
    _apply_style(ax)
    _annotate_window(ax, 24, 48, WINDOW_COLOR, "Heatwave", ypos=0.45)
    # Legend moved from "center right" to a left-of-center, slightly-
    # above-center anchor so it sits over the Local Redist. band
    # (which is the dominant area in the center of the plot) without
    # covering the AgriBrain rho-threshold annotations on the right.
    _legend(ax, loc="center left", bbox_to_anchor=(0.02, 0.62),
            ncol=1, frameon=True, framealpha=0.85)

    # --- (d) Per-step Adaptive Resilience Index (ARI) ---
    # Per-step ARI = (1 - waste) * SLCA * (1 - rho), as computed by
    # resilience.compute_ari and surfaced as ``ari_trace`` in the
    # results JSON. The (1 - rho) factor uses the dataset-cumulative
    # rho (identical across modes for any given step), so cross-mode
    # ARI differentiation is carried by (1 - waste) * SLCA: AgriBrain's
    # lower waste (mode_eff = 0.83 vs hybrid_rl's 0.45) and higher SLCA
    # (LR-routing emphasis vs hybrid_rl's CC-heavy routing during
    # stress) lift its ARI above the baselines, while the shared
    # (1 - rho) factor pulls every mode downward through the heatwave
    # window in line with the cumulative thermal-damage physics.
    #
    # When per-seed JSONs are present (HPC 20-seed run with traces
    # enabled), use the seed-MEAN as the plotted line so the figure
    # reflects the canonical multi-seed posture. Otherwise fall back
    # to the single-seed line. Per-step CI ribbons were removed in
    # 2026-05 per user direction -- the cross-method ARI gap is
    # cleanly readable from the styled lines alone (consistent
    # color/marker/linestyle via _mode_plot), and the canonical
    # uncertainty story for ARI lives in the bootstrap CIs of the
    # cross-method paired tests in benchmark_significance.json.
    ax = axes[1, 1]
    window = 12
    kernel = np.ones(window) / window
    for mode in ["static", "hybrid_rl", "agribrain"]:
        ep = hw[mode]
        per_seed = _load_per_seed_traces("heatwave", mode, "ari_trace")
        if per_seed is not None and per_seed.shape[0] >= 2:
            n = min(per_seed.shape[1], hours.shape[0])
            seed_mean = per_seed[:, :n].mean(axis=0)
            mean_smooth = np.convolve(seed_mean, kernel, mode="same")
            _mode_plot(ax, hours[:n], mean_smooth, mode)
        else:
            ari = np.array(ep["ari_trace"])
            rolling = np.convolve(ari, kernel, mode="same")
            _mode_plot(ax, hours, rolling, mode)
    ax.set_xlabel("Hours")
    ax.set_ylabel("ARI")
    ax.set_title("(d) Adaptive Resilience Index")
    ax.set_ylim(0, 1.0)
    _apply_style(ax)
    _annotate_window(ax, 24, 48, WINDOW_COLOR, "Heatwave")
    # ARI declines monotonically from ~0.5 at h0 toward ~0.1 by h72 as
    # the cumulative (1 - rho) factor saturates, so the upper-right
    # corner is empty space. Anchoring the legend there keeps it clear
    # of the three mode traces, the heatwave shading, and its label.
    _legend(ax, loc="upper right", bbox_to_anchor=(0.98, 0.98))

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

    # Per-figure font-size bump for fig 3 (post-2026-04 user request).
    # Uniform +1 across body / ticks / axis labels / subplot titles /
    # suptitle / legend / in-plot annotations - matches the gentle
    # bump applied to fig 2. Scoped to this function via try/finally
    # so other figures (fig 4, fig 5, ...) keep the canonical sizes.
    global BODY_FONT_SIZE, TICK_FONT_SIZE, AXIS_LABEL_SIZE
    global SUBPLOT_TITLE_SIZE, FIG_TITLE_SIZE, LEGEND_FONT_SIZE
    global ANNOT_FONT_SIZE
    _saved_sizes = (
        BODY_FONT_SIZE, TICK_FONT_SIZE, AXIS_LABEL_SIZE,
        SUBPLOT_TITLE_SIZE, FIG_TITLE_SIZE, LEGEND_FONT_SIZE,
        ANNOT_FONT_SIZE,
    )
    BODY_FONT_SIZE = _saved_sizes[0] + 1
    TICK_FONT_SIZE = _saved_sizes[1] + 1
    AXIS_LABEL_SIZE = _saved_sizes[2] + 1
    SUBPLOT_TITLE_SIZE = _saved_sizes[3] + 1
    FIG_TITLE_SIZE = _saved_sizes[4] + 1
    LEGEND_FONT_SIZE = _saved_sizes[5] + 1
    ANNOT_FONT_SIZE = _saved_sizes[6] + 1
    _saved_rc = {
        "font.size": plt.rcParams["font.size"],
        "axes.labelsize": plt.rcParams["axes.labelsize"],
        "axes.titlesize": plt.rcParams["axes.titlesize"],
        "xtick.labelsize": plt.rcParams["xtick.labelsize"],
        "ytick.labelsize": plt.rcParams["ytick.labelsize"],
        "legend.fontsize": plt.rcParams["legend.fontsize"],
        "legend.title_fontsize": plt.rcParams["legend.title_fontsize"],
        "figure.titlesize": plt.rcParams["figure.titlesize"],
    }
    plt.rcParams.update({
        "font.size": BODY_FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "axes.titlesize": SUBPLOT_TITLE_SIZE,
        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "legend.title_fontsize": LEGEND_FONT_SIZE,
        "figure.titlesize": FIG_TITLE_SIZE,
    })

    try:
        return _fig3_overproduction_inner(op, ab, hours)
    finally:
        (BODY_FONT_SIZE, TICK_FONT_SIZE, AXIS_LABEL_SIZE,
         SUBPLOT_TITLE_SIZE, FIG_TITLE_SIZE, LEGEND_FONT_SIZE,
         ANNOT_FONT_SIZE) = _saved_sizes
        plt.rcParams.update(_saved_rc)


def _fig3_overproduction_inner(op, ab, hours):
    """Body of fig 3. Extracted from ``fig3_overproduction`` so the
    per-figure font-size overrides applied above can be cleanly torn
    down via try/finally regardless of how the body returns or
    raises."""
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
    # the center-right (xpos\u224840) so the bounding box sits clearly
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
    # capacity-constrained form (BatchInventory realized-action trace)
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
    # Center the "Overproduction" label inside the window (xpos=45) so
    # it sits well inside the red shading rather than hugging the right
    # edge - the upper-left corner is now occupied by the threshold-
    # onset guide rather than the legend, so we no longer need to push
    # the label rightward.
    _annotate_window(ax, 12, 60, WINDOW_COLOR, "Overproduction",
                     ypos=0.99, xpos=45)
    # Legend at "center left": pre-h32 the panel is empty (RLE is
    # undefined until any at-risk batch enters the rolling window), so
    # the left half is clear headroom for the legend; vertical-center
    # placement keeps it clear of both the "first rho > 0.1 at h~32"
    # threshold-onset annotation in the lower band and the
    # "Overproduction" window label at the top.
    _legend(ax, loc="center left")

    # --- (d) SLCA component grouped bars with honest cross-seed SE ---
    # Two-tier rendering:
    #
    #   1. When per-seed JSONs are on disk under
    #      ``benchmark_seeds/<RUN_TAG>/seed_*.json`` (the canonical
    #      HPC posture, post-2026-05 ``TRACE_FIELDS`` extension that
    #      dumps slca_component_trace per seed), bar height =
    #      cross-seed mean of the per-seed cross-step C/L/R/P means
    #      and error bars = +/- 1.96 * SE = 1.96 * std(per_seed) /
    #      sqrt(n_seeds). This is the apples-to-apples cross-seed
    #      uncertainty for the four-pillar decomposition the
    #      benchmark_summary's aggregate ``slca`` scalar does not
    #      decompose into.
    #
    #   2. Single-seed fallback (local development; older HPC runs
    #      that pre-date the TRACE_FIELDS extension): plot means with
    #      NO error bars rather than a misleading within-trajectory
    #      step-std. This was the 2026-05 honesty fix; the per-seed
    #      branch above is the genuinely-multi-seed extension.
    #
    # See also _load_per_seed_slca_components which walks the seed
    # JSONs and collapses the per-step list[dict] into one
    # cross-step mean per component per seed.
    ax = axes[1, 1]
    components = ["C", "L", "R", "P"]
    comp_labels = ["Carbon", "Labor", "Resilience", "Price Transp."]
    x = np.arange(len(components))
    width = 0.26
    _slca_per_seed = {
        m: _load_per_seed_slca_components("overproduction", m)
        for m in ("static", "hybrid_rl", "agribrain")
    }
    _has_multi_seed = all(_slca_per_seed[m] is not None
                          for m in ("static", "hybrid_rl", "agribrain"))
    for i, mode in enumerate(["static", "hybrid_rl", "agribrain"]):
        if _has_multi_seed:
            per_seed = _slca_per_seed[mode]  # type: ignore[index]
            vals = [float(per_seed[c].mean()) for c in components]
            # Cross-seed SE; 1.96*SE matches the +/- 95% convention
            # the rest of the figure suite uses (consistent with
            # fig 4 panel D's SE error bars).
            ses = [
                float(per_seed[c].std(ddof=1) / np.sqrt(per_seed[c].size))
                for c in components
            ]
            ax.bar(
                x + i * width, vals, width, color=COLORS[mode],
                label=MODE_LABELS[mode], alpha=0.92, edgecolor="white",
                linewidth=0.8,
                yerr=[1.96 * s for s in ses], capsize=4,
                error_kw={"linewidth": 1.0, "capthick": 1.0},
            )
        else:
            # Single-seed fallback: plot means alone (no fake CI bars).
            ep = op[mode]
            vals = [np.mean([s[comp] for s in ep["slca_component_trace"]])
                    for comp in components]
            ax.bar(x + i * width, vals, width, color=COLORS[mode],
                   label=MODE_LABELS[mode], alpha=0.92, edgecolor="white",
                   linewidth=0.8)
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
    """2x2: ARI over time, action distribution shift, reroute rate per method, KPI delta.

    Layout history: started 1-row (panel C single-pane action distribution)
    then briefly went to a 2-row gridspec (legend/bar overlap), then 1x4
    (visual mismatch with 2x2 figs 2/3/5), and as of late-May 2026 to a
    2x2 grid that matches figs 2/3/5. The causality chain reads top-down
    AND left-right: top row = stimulus (ARI trace) + observed behavior
    (action distribution shift); bottom row = behavior magnitude per
    method (reroute rate) + KPI consequence per method (Δ ARI / Waste /
    Service). Each panel keeps its previous individual contents.

    Per-figure font-size bump for fig 4 (post-2026-05 user request:
    "make this 4-panel figure match the other 4-panel figures style,
    spacing and text sizes"). Uniform +1 across body / ticks / axis
    labels / subplot titles / suptitle / legend / in-plot annotations
    matches the bump applied to figs 2, 3, and 5 (the other 4-panel
    figures in the publication set). Scoped to this function via
    try/finally so other figures (fig 5, fig 6, ...) keep the
    canonical global sizes.
    """
    global BODY_FONT_SIZE, TICK_FONT_SIZE, AXIS_LABEL_SIZE
    global SUBPLOT_TITLE_SIZE, FIG_TITLE_SIZE, LEGEND_FONT_SIZE
    global ANNOT_FONT_SIZE
    _saved_sizes = (
        BODY_FONT_SIZE, TICK_FONT_SIZE, AXIS_LABEL_SIZE,
        SUBPLOT_TITLE_SIZE, FIG_TITLE_SIZE, LEGEND_FONT_SIZE,
        ANNOT_FONT_SIZE,
    )
    BODY_FONT_SIZE = _saved_sizes[0] + 1
    TICK_FONT_SIZE = _saved_sizes[1] + 1
    AXIS_LABEL_SIZE = _saved_sizes[2] + 1
    SUBPLOT_TITLE_SIZE = _saved_sizes[3] + 1
    FIG_TITLE_SIZE = _saved_sizes[4] + 1
    LEGEND_FONT_SIZE = _saved_sizes[5] + 1
    ANNOT_FONT_SIZE = _saved_sizes[6] + 1
    _saved_rc = {
        "font.size": plt.rcParams["font.size"],
        "axes.labelsize": plt.rcParams["axes.labelsize"],
        "axes.titlesize": plt.rcParams["axes.titlesize"],
        "xtick.labelsize": plt.rcParams["xtick.labelsize"],
        "ytick.labelsize": plt.rcParams["ytick.labelsize"],
        "legend.fontsize": plt.rcParams["legend.fontsize"],
        "legend.title_fontsize": plt.rcParams["legend.title_fontsize"],
        "figure.titlesize": plt.rcParams["figure.titlesize"],
    }
    plt.rcParams.update({
        "font.size": BODY_FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "axes.titlesize": SUBPLOT_TITLE_SIZE,
        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "legend.title_fontsize": LEGEND_FONT_SIZE,
        "figure.titlesize": FIG_TITLE_SIZE,
    })

    try:
        return _fig4_cyber_inner(data)
    finally:
        (BODY_FONT_SIZE, TICK_FONT_SIZE, AXIS_LABEL_SIZE,
         SUBPLOT_TITLE_SIZE, FIG_TITLE_SIZE, LEGEND_FONT_SIZE,
         ANNOT_FONT_SIZE) = _saved_sizes
        plt.rcParams.update(_saved_rc)


def _fig4_cyber_inner(data):
    """Body of fig 4. Extracted from ``fig4_cyber`` so the per-figure
    font-size overrides applied above can be cleanly torn down via
    try/finally regardless of how the body returns or raises.
    """
    cy = data["results"]["cyber_outage"]
    ab = cy["agribrain"]
    hours = np.array(ab["hours"])

    # 2x2 grid matching figs 2 / 3 / 5: (18, 13) figsize. The earlier
    # 1x4 layout (28 x 6.5) was visually inconsistent with the rest of
    # the 4-panel figures in the publication set; the 2x2 reads as a
    # natural causality grid (top row = stimulus + observed behavior,
    # bottom row = magnitude + outcome) and matches the reader's
    # left-to-right + top-to-bottom scan order in the other figures.
    fig, axes2d = plt.subplots(2, 2, figsize=(18, 13))
    # Flatten for legacy indexing (axes[0..3] corresponds to (a..d)
    # in row-major order: top-left, top-right, bottom-left, bottom-right).
    axes = axes2d.flatten()
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
    # Anchor the "Outage" badge at the bottom-center of the outage
    # window (h=48). The previous top-anchored placement (ypos=0.93)
    # collided with the upper-left/center quadrant where the legend
    # and the AgriBrain peak both sit; bottom-anchoring puts the
    # label in genuinely empty space below the three converging
    # traces, since the lower spine is reached only at the final
    # h~70 step where ARI bottoms out around 0.20.
    _annotate_window(
        ax, 24, 72, WINDOW_COLOR, "Outage",
        ypos=0.07, va="bottom",
    )
    # Legend at upper-right: ARI declines monotonically from its
    # h~15 peak so the right edge of the panel sits well below the
    # data ceiling, leaving the upper-right corner clear of the three
    # mode traces.
    _legend(ax, loc="upper right")

    # --- (b) Action distribution pre/during outage ---
    # 2026-05 multi-seed upgrade: when per-seed action_trace dumps
    # are on disk under benchmark_seeds/seed_*.json, compute the
    # action-share bars and SEs as MEANS / cross-seed SE across
    # seeds (the canonical 20-seed posture). Falls back to the
    # single-seed Wald-binomial computation when traces aren't
    # available (local development; non-HPC runs). The Wald form
    # was misleading as the panel's only error bar because it
    # plotted within-trajectory step-count CIs that read as
    # cross-seed uncertainty.
    ax = axes[1]
    # Wrap multi-word tick labels onto two lines so the wider fig 4
    # font stack does not overlap adjacent ticks.
    action_names = ["Cold\nChain", "Local\nRedistribute", "Recovery"]
    pre_mask = np.array(hours) < 24
    during_mask = np.array(hours) >= 24

    bar_x = np.arange(3)
    width = 0.38

    pre_counts = np.zeros(3)
    during_counts = np.zeros(3)
    pre_se = np.zeros(3)
    during_se = np.zeros(3)
    _b_inputs = _per_seed_window_inputs(
        "cyber_outage", "agribrain", np.asarray(hours, dtype=float),
    )
    if _b_inputs is not None:
        # Multi-seed: per-seed action share, mean + cross-seed SE.
        n_seeds_b = _b_inputs["n_seeds"]
        a_pre_b = _b_inputs["action_pre"]   # (n_seeds, n_pre_steps)
        a_dur_b = _b_inputs["action_dur"]
        for a in range(3):
            pre_per_seed = (a_pre_b == a).mean(axis=1)
            dur_per_seed = (a_dur_b == a).mean(axis=1)
            pre_counts[a] = float(pre_per_seed.mean())
            during_counts[a] = float(dur_per_seed.mean())
            pre_se[a] = float(pre_per_seed.std(ddof=1) / np.sqrt(n_seeds_b))
            during_se[a] = float(dur_per_seed.std(ddof=1) / np.sqrt(n_seeds_b))
    else:
        # Single-seed Wald-binomial fallback. Honest as a within-
        # trajectory step-count CI; not a cross-seed SE.
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

    # --- (c) Causality chain: Outage -> Behavior -> Outcome ---
    #
    # The previous panel C variants (cumulative anomaly-defense traces;
    # cumulative at-risk reroutes) showed only one half of the
    # causality argument: that the policy did *something different*.
    # The 2026-05 redesign joins the policy-shift signal with its
    # outcome consequence in a single panel:
    #
    #   - top half: per-method "reroute rate" (fraction of decisions
    #     that left the cold chain) computed over the pre-outage and
    #     during-outage windows. A cyber outage that caused no
    #     behavior change would show identical pre/during bars per
    #     method; a policy that responds shows the during bar rising.
    #
    #   - bottom half: change in three KPIs from pre-outage to
    #     during-outage, per method:
    #       deltaARI    = mean(ARI during) - mean(ARI pre)
    #       deltaWaste  = mean(waste during) - mean(waste pre)
    #       deltaService = service_during - service_pre, where
    #                      service = mean(action != recovery) * (1 - mean waste)
    #                      i.e. fraction of inventory reaching retail
    #                      in usable form (retail-dispatch * sellable).
    #
    # Reading order top -> bottom is the load-bearing claim of the
    # cyber section: the outage forced AgriBrain's policy to shift
    # (top), and that shift translated into a smaller ARI/Service
    # drop and a smaller Waste rise than the baselines suffered
    # (bottom). Static is the unaltered-baseline reference: its top
    # bars are equal pre/during (no behavior change) and its bottom
    # bars show the unmitigated outage damage.
    #
    # Pre/during windows are split at the cyber-outage onset h=24 (see
    # generate_results._apply_cyber_outage); the published HPC pipeline
    # uses the same split.
    pre_mask_arr = np.asarray(hours, dtype=float) < 24.0
    during_mask_arr = np.asarray(hours, dtype=float) >= 24.0
    modes_ordered_c = ["static", "hybrid_rl", "agribrain"]
    mode_labels_c = ["Static", "Hybrid RL", "AgriBrain"]
    # Distinct, color-blind-friendly mode palette consistent with the
    # rest of the figure.
    mode_colors_c = {
        "static": "#7C7C7C",
        "hybrid_rl": "#D55E00",
        "agribrain": "#0F8A8C",
    }

    reroute_pre: list[float] = []
    reroute_during: list[float] = []
    # Binomial standard errors for the reroute-rate proportions.
    # se = sqrt(p * (1 - p) / n) per Wald's approximation; the panel
    # plots 1.96 * se as a 95% CI half-width, matching panel B's
    # treatment of the action-distribution proportions.
    reroute_pre_se: list[float] = []
    reroute_during_se: list[float] = []
    ari_during: list[float] = []
    waste_during: list[float] = []
    service_during: list[float] = []
    # Standard errors for the during-outage means. For ARI and Waste
    # we use SE_mean = std/sqrt(n) on the during-window samples
    # (assumes step-level samples are approximately independent
    # within window; conservative since Arrhenius integration
    # introduces mild autocorrelation, but adequate for figure-level
    # CI bars). For Service the metric is a product
    # (retail_dispatch * (1 - mean_waste)) and the analytic SE
    # requires the delta method, so we bootstrap-resample
    # during-window steps 2000x and take the std of the bootstrap
    # level distribution. The pre-vs-during delta construction was
    # retired in 2026-05: levels are unambiguous (AgriBrain holds
    # the highest ARI / lowest waste / highest service during the
    # outage), whereas a delta penalises systems already near
    # ceiling pre-outage and inverted the Service ranking on a
    # saturation artefact.
    ari_during_se: list[float] = []
    waste_during_se: list[float] = []
    service_during_se: list[float] = []

    for mode in modes_ordered_c:
        # 2026-05 multi-seed upgrade: when per-seed action / ari /
        # waste traces are on disk, compute the panel-C reroute
        # proportions and the panel-D during-outage levels as MEANS
        # across seeds with cross-seed SE error bars (the canonical
        # 20-seed posture matching figs 6/7/8/9/10). Falls back to
        # the single-seed step-level SE / Wald-binomial form when
        # multi-seed traces aren't available (local development;
        # non-HPC runs). The fallback is honest as a within-trajectory
        # CI but reads as cross-method uncertainty -- which is why the
        # multi-seed path is preferred.
        _ms = _per_seed_window_inputs(
            "cyber_outage", mode, np.asarray(hours, dtype=float),
        )
        if _ms is not None:
            n_seeds_cd = _ms["n_seeds"]
            # Panel-C reroute proportions: per-seed (action != 0)
            # share, mean and cross-seed SE.
            rp_per_seed = (_ms["action_pre"] != 0).mean(axis=1)
            rd_per_seed = (_ms["action_dur"] != 0).mean(axis=1)
            reroute_pre.append(float(rp_per_seed.mean()))
            reroute_during.append(float(rd_per_seed.mean()))
            reroute_pre_se.append(
                float(rp_per_seed.std(ddof=1) / np.sqrt(n_seeds_cd))
            )
            reroute_during_se.append(
                float(rd_per_seed.std(ddof=1) / np.sqrt(n_seeds_cd))
            )

            # Panel-D during-window levels per seed.
            ari_per_seed = _ms["ari_dur"].mean(axis=1)
            waste_per_seed = _ms["waste_dur"].mean(axis=1)
            not_recovery_per_seed = (_ms["action_dur"] != 2).mean(axis=1)
            svc_per_seed = not_recovery_per_seed * (1.0 - waste_per_seed)

            ari_during.append(float(ari_per_seed.mean()))
            waste_during.append(float(waste_per_seed.mean()))
            service_during.append(float(svc_per_seed.mean()))
            ari_during_se.append(
                float(ari_per_seed.std(ddof=1) / np.sqrt(n_seeds_cd))
            )
            waste_during_se.append(
                float(waste_per_seed.std(ddof=1) / np.sqrt(n_seeds_cd))
            )
            service_during_se.append(
                float(svc_per_seed.std(ddof=1) / np.sqrt(n_seeds_cd))
            )
            continue

        # ---- Single-seed fallback path ----
        ep = cy[mode]
        actions_arr = np.asarray(ep["action_trace"], dtype=int)
        ari_arr = np.asarray(ep["ari_trace"], dtype=float)
        waste_arr = np.asarray(ep.get("waste_trace") or [], dtype=float)
        n = min(actions_arr.shape[0], ari_arr.shape[0],
                waste_arr.shape[0] if waste_arr.size else actions_arr.shape[0],
                hours.shape[0])
        actions_arr = actions_arr[:n]
        ari_arr = ari_arr[:n]
        # If the episode dump did not emit a per-step waste trace
        # (older runs), fall back to the episode-level waste scalar
        # broadcast across all steps. This keeps the plot honest --
        # the metric will be zero for those modes -- rather than
        # crashing with a shape error.
        if waste_arr.size >= n:
            waste_arr_n = waste_arr[:n]
        else:
            waste_arr_n = np.full(n, float(ep.get("waste", 0.0)))

        pm = pre_mask_arr[:n]
        dm = during_mask_arr[:n]
        n_pre_c = int(pm.sum())
        n_dur_c = int(dm.sum())
        if n_pre_c == 0 or n_dur_c == 0:
            # Degenerate window (shouldn't happen on the canonical 72 h
            # cyber_outage trace, but guard against truncated data).
            reroute_pre.append(0.0); reroute_during.append(0.0)
            reroute_pre_se.append(0.0); reroute_during_se.append(0.0)
            ari_during.append(0.0); waste_during.append(0.0); service_during.append(0.0)
            ari_during_se.append(0.0); waste_during_se.append(0.0); service_during_se.append(0.0)
            continue

        # Reroute proportions (Bernoulli at step granularity) + Wald SE.
        rp = float(np.mean(actions_arr[pm] != 0))
        rd = float(np.mean(actions_arr[dm] != 0))
        reroute_pre.append(rp)
        reroute_during.append(rd)
        reroute_pre_se.append(float(np.sqrt(rp * (1.0 - rp) / n_pre_c)))
        reroute_during_se.append(float(np.sqrt(rd * (1.0 - rd) / n_dur_c)))

        ari_dur = float(np.mean(ari_arr[dm]))
        waste_dur = float(np.mean(waste_arr_n[dm]))
        # Service-level proxy: retail-dispatch rate * (1 - mean waste).
        # See panel docstring above for the operations-research
        # interpretation. A clean, defensible scalar that goes
        # *down* when the policy diverts to recovery and *down* again
        # when retail-bound product spoils.
        svc_dur = float(np.mean(actions_arr[dm] != 2)) * (1.0 - waste_dur)

        ari_during.append(ari_dur)
        waste_during.append(waste_dur)
        service_during.append(svc_dur)

        # Within-window step-level SE for ARI / Waste means.
        def _level_se(x: np.ndarray) -> float:
            s = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
            return float(s / np.sqrt(max(x.size, 1)))

        ari_during_se.append(_level_se(ari_arr[dm]))
        waste_during_se.append(_level_se(waste_arr_n[dm]))

        # Service is a product of two means; bootstrap the during-
        # window level. Seed per-mode so the bar errors are
        # reproducible across regenerations of the same data. Use
        # blake2b instead of the built-in ``hash()``: Python's hash
        # is randomised by PYTHONHASHSEED on each interpreter start,
        # so the rendered error caps drifted run-to-run. blake2b
        # matches the deterministic-seed convention
        # aggregate_seeds.py uses for the same reason. n_boot=2000
        # brings this fallback closer to the aggregator's
        # 10000-resample canonical CIs while keeping the figure
        # render fast.
        import hashlib as _hashlib_f4
        n_boot = 2000
        _seed_bytes_f4 = _hashlib_f4.blake2b(
            f"{mode}::service_se".encode("utf-8"), digest_size=4,
        ).digest()
        boot_rng = np.random.default_rng(
            int.from_bytes(_seed_bytes_f4, "big"),
        )
        a_dm = actions_arr[dm]; w_dm = waste_arr_n[dm]
        boot_levels = np.empty(n_boot, dtype=float)
        for k in range(n_boot):
            id_ = boot_rng.integers(0, n_dur_c, n_dur_c)
            boot_levels[k] = (
                float(np.mean(a_dm[id_] != 2))
                * (1.0 - float(np.mean(w_dm[id_])))
            )
        service_during_se.append(float(np.std(boot_levels, ddof=1)))

    # ---- (c) Reroute rate pre/during outage per method ----
    # The behavior-magnitude leg of the causality chain. Static is the
    # null reference (always cold-chain -> reroute rate 0 in both
    # windows). Hybrid RL and AgriBrain both reroute pre-outage as
    # part of their normal operation; what matters is whether the
    # *during* bar rises relative to the *pre* bar, i.e. whether the
    # policy responds to the outage.
    ax_c = axes[2]
    x_modes = np.arange(len(modes_ordered_c))
    bar_w = 0.36
    ax_c.bar(
        x_modes - bar_w / 2, reroute_pre, bar_w,
        color="#1565C0", alpha=0.92, edgecolor="white", linewidth=0.8,
        label="Pre-outage",
        yerr=1.96 * np.asarray(reroute_pre_se), capsize=4,
        error_kw={"linewidth": 1.2, "capthick": 1.2, "ecolor": "#1F1F1F"},
    )
    ax_c.bar(
        x_modes + bar_w / 2, reroute_during, bar_w,
        color=WINDOW_COLOR, alpha=0.92, edgecolor="white", linewidth=0.8,
        label="During outage",
        yerr=1.96 * np.asarray(reroute_during_se), capsize=4,
        error_kw={"linewidth": 1.2, "capthick": 1.2, "ecolor": "#1F1F1F"},
    )
    ax_c.set_xticks(x_modes)
    ax_c.set_xticklabels(mode_labels_c)
    # Headroom above the tallest bar (including its error-bar cap) so
    # the legend has a clean home.
    _top_c = max(
        max(np.asarray(reroute_pre) + 1.96 * np.asarray(reroute_pre_se)),
        max(np.asarray(reroute_during) + 1.96 * np.asarray(reroute_during_se)),
    )
    ax_c.set_ylim(0, max(_top_c * 1.30, 1.0))
    ax_c.set_ylabel("Reroute rate")
    ax_c.set_title("(c) Behavior Shift")
    _apply_style(ax_c)
    # Static stays at 0 in both windows so the upper-left corner is
    # genuinely empty; legend lives there.
    _legend(ax_c, loc="upper left")

    # ---- (d) KPI levels during outage per method ----
    # The outcome leg of the causality chain. Under stress ARI and
    # Service should stay high and Waste should stay low; AgriBrain
    # holds the best level on every KPI. The pre-vs-during delta
    # construction this panel used before 2026-05 inverted the
    # Service ranking on a saturation artefact: a system already
    # near-ceiling pre-outage had little delta headroom and looked
    # worse than a system that started lower and shifted further.
    # Plotting absolute during-window levels makes the comparison
    # direct -- bigger ARI / Service bars are better, smaller Waste
    # bar is better, and the saturation confound disappears.
    ax_d = axes[3]
    kpi_x = np.arange(3)  # ARI, Waste, Service
    grp_w = 0.27
    for i, mode in enumerate(modes_ordered_c):
        vals = [ari_during[i], waste_during[i], service_during[i]]
        ses = [ari_during_se[i], waste_during_se[i], service_during_se[i]]
        ax_d.bar(
            kpi_x + (i - 1) * grp_w, vals, grp_w,
            color=mode_colors_c[mode], alpha=0.92,
            edgecolor="white", linewidth=0.8,
            label=mode_labels_c[i],
            yerr=1.96 * np.asarray(ses), capsize=4,
            error_kw={"linewidth": 1.2, "capthick": 1.2, "ecolor": "#1F1F1F"},
        )
    ax_d.set_xticks(kpi_x)
    ax_d.set_xticklabels(["ARI", "Waste", "Service"])
    ax_d.set_ylabel("Level during outage")
    ax_d.set_title("(d) Outage-Window Levels")
    _apply_style(ax_d)
    # All three KPIs are non-negative levels in [0, 1]; pin the
    # y-axis to that range plus a small headroom so the legend has a
    # clean home and bar-to-bar comparisons aren't visually distorted
    # by auto-scaling on tiny CI extensions.
    _top_d = 0.0
    for i in range(len(modes_ordered_c)):
        for v, se in zip(
            [ari_during[i], waste_during[i], service_during[i]],
            [ari_during_se[i], waste_during_se[i], service_during_se[i]],
        ):
            _top_d = max(_top_d, v + 1.96 * se)
    ax_d.set_ylim(0.0, max(_top_d * 1.20, 1.05))
    # Legend at upper-left: the leftmost cluster is ARI (~0.4-0.6),
    # which leaves clean headroom in that corner, whereas the
    # upper-right is now occupied by the tall Service cluster
    # (~0.86-0.96 + CI cap).
    _legend(ax_d, loc="upper left")

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

    # Per-figure font-size bump for fig 5 (post-2026-04 user request).
    # Uniform +1 across body / ticks / axis labels / subplot titles /
    # suptitle / legend / in-plot annotations - matches the bump
    # applied to fig 2 (commit a4144d1) and fig 3 (commit e6151e5)
    # so all three perishable-scenario figures render at the same
    # text size. Scoped to this function via try/finally so other
    # figures keep the canonical sizes.
    global BODY_FONT_SIZE, TICK_FONT_SIZE, AXIS_LABEL_SIZE
    global SUBPLOT_TITLE_SIZE, FIG_TITLE_SIZE, LEGEND_FONT_SIZE
    global ANNOT_FONT_SIZE
    _saved_sizes = (
        BODY_FONT_SIZE, TICK_FONT_SIZE, AXIS_LABEL_SIZE,
        SUBPLOT_TITLE_SIZE, FIG_TITLE_SIZE, LEGEND_FONT_SIZE,
        ANNOT_FONT_SIZE,
    )
    BODY_FONT_SIZE = _saved_sizes[0] + 1
    TICK_FONT_SIZE = _saved_sizes[1] + 1
    AXIS_LABEL_SIZE = _saved_sizes[2] + 1
    SUBPLOT_TITLE_SIZE = _saved_sizes[3] + 1
    FIG_TITLE_SIZE = _saved_sizes[4] + 1
    LEGEND_FONT_SIZE = _saved_sizes[5] + 1
    ANNOT_FONT_SIZE = _saved_sizes[6] + 1
    _saved_rc = {
        "font.size": plt.rcParams["font.size"],
        "axes.labelsize": plt.rcParams["axes.labelsize"],
        "axes.titlesize": plt.rcParams["axes.titlesize"],
        "xtick.labelsize": plt.rcParams["xtick.labelsize"],
        "ytick.labelsize": plt.rcParams["ytick.labelsize"],
        "legend.fontsize": plt.rcParams["legend.fontsize"],
        "legend.title_fontsize": plt.rcParams["legend.title_fontsize"],
        "figure.titlesize": plt.rcParams["figure.titlesize"],
    }
    plt.rcParams.update({
        "font.size": BODY_FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "axes.titlesize": SUBPLOT_TITLE_SIZE,
        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "legend.title_fontsize": LEGEND_FONT_SIZE,
        "figure.titlesize": FIG_TITLE_SIZE,
    })

    try:
        return _fig5_pricing_inner(ap, ab, hours)
    finally:
        (BODY_FONT_SIZE, TICK_FONT_SIZE, AXIS_LABEL_SIZE,
         SUBPLOT_TITLE_SIZE, FIG_TITLE_SIZE, LEGEND_FONT_SIZE,
         ANNOT_FONT_SIZE) = _saved_sizes
        plt.rcParams.update(_saved_rc)


def _fig5_pricing_inner(ap, ab, hours):
    """Body of fig 5. Extracted from ``fig5_pricing`` so the per-figure
    font-size overrides applied above can be cleanly torn down via
    try/finally regardless of how the body returns or raises."""
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
    # Lower-center, lifted ~10 % off the x-axis: the central horizontal
    # band of the panel below the data (y ~ y_lo .. 0.45) is empty
    # because the three mode traces stay above ~0.55 across the
    # interior hours, so the legend sits in clear space without
    # touching any line.
    _legend(ax, loc="lower center", bbox_to_anchor=(0.5, 0.10))

    # --- (d) Reward decomposition: SLCA, waste penalty, rho penalty ---
    # Three stacked layers on a single axis make the additive decomposition
    # R = SLCA − η_w·waste − η_ρ·ρ visually obvious. The vertical gap
    # between consecutive lines is each penalty's contribution at time t,
    # and the shaded bands quantify those magnitudes without a twin axis.
    # --- (d) Per-step reward comparison across modes ---
    # Replaces the previous SLCA(t) / Net reward / Penalty bands view
    # which was AgriBrain-only and visually compressed (three lines
    # within ~[0.62, 0.78] hard to read against a 0.6-0.8 y-axis).
    # The new panel plots a 3-hour rolling mean of per-step reward
    # for each mode so the AgriBrain > Hybrid RL > Static ordering
    # this scenario is meant to demonstrate becomes directly visible.
    #
    # Why the lines are differentiable: per-step reward has ~0.05-0.07
    # noise from adaptive_pricing demand volatility. The 12-step (3h)
    # rolling window reduces noise by sqrt(12) ~= 3.5x to ~0.015.
    # Expected mode means under this scenario:
    #   Static    ~0.55-0.60  (low SLCA, high waste, all-CC routing)
    #   Hybrid RL ~0.65-0.70  (medium SLCA, medium waste)
    #   AgriBrain ~0.70-0.75  (high SLCA via LR-heavy routing, low
    #                          waste via mode_eff = 0.83 capability stack)
    # Gaps of 0.04-0.10 are 3-7x the smoothed noise floor, giving
    # clean visual separation.
    ax = axes[1, 1]
    window = 12  # 12 steps * 0.25 h = 3 h rolling
    for mode in ["static", "hybrid_rl", "agribrain"]:
        ep = ap[mode]
        reward = np.array(ep["reward_trace"])
        rolling = np.convolve(reward, np.ones(window) / window, mode="same")
        _mode_plot(ax, hours, rolling, mode)

    ax.set_xlabel("Hours")
    ax.set_ylabel("Reward")
    ax.set_title("(d) Per-step Reward Comparison")
    _apply_style(ax)
    # Match panel (c)'s legend placement (lower-center, lifted ~10 %
    # off the x-axis) so the two bottom-row panels read symmetrically.
    # The reward traces stay above ~0.50 across the interior hours, so
    # the lifted lower-center anchor is clear of all three lines.
    _legend(ax, loc="lower center", bbox_to_anchor=(0.5, 0.10))

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
    (BatchInventory realized-action variant). Only the
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
                # Percentile bootstrap CI on the mean (1000 resamples
                # is sufficient for figure-level error bars; the
                # canonical 10000-resample CI lives in
                # benchmark_summary.json). Use blake2b for
                # deterministic seeding -- Python's built-in hash() is
                # PYTHONHASHSEED-randomised by default which makes the
                # rendered error caps drift run-to-run.
                import hashlib as _hashlib_pseed
                _seed_bytes_pseed = _hashlib_pseed.blake2b(
                    f"{sc}::{mode}::{met}".encode("utf-8"), digest_size=4,
                ).digest()
                rng = np.random.default_rng(
                    int.from_bytes(_seed_bytes_pseed, "big"),
                )
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


def _load_per_seed_traces(scenario: str, mode: str,
                          field: str = "ari_trace") -> np.ndarray | None:
    """Stack per-step traces across seeds for one (scenario, mode, field).

    Walks ``RESULTS_DIR/benchmark_seeds/`` (flat layout or
    ``<RUN_TAG>/seed_*.json`` tagged layout, same convention
    ``_load_per_seed_summary`` uses) and returns an
    ``(n_seeds, n_steps)`` numpy array stacking the requested trace.

    The per-seed JSON envelope (post 2026-05) is:
        {"seed": int, "scenarios": {...}, "traces": {sc: {mode: {field: [...]}}}}
    Older per-seed JSONs that predate the trace dump don't carry a
    "traces" key; this loader returns None for those (and the
    figure falls back to its single-seed line render).

    Returns
    -------
    np.ndarray of shape (n_seeds, n_steps), or None when no per-seed
    traces are found. Seeds with mismatched step counts are dropped
    (the simulator emits a fixed length per scenario, so this should
    not fire in practice, but guard against partial/truncated dumps).
    """
    seeds_root = RESULTS_DIR / "benchmark_seeds"
    if not seeds_root.exists():
        return None
    import json
    # Same flat-or-tagged discovery pattern _load_per_seed_summary uses.
    seed_files = list(seeds_root.glob("seed_*.json"))
    if not seed_files:
        for sub in seeds_root.iterdir():
            if sub.is_dir():
                seed_files.extend(sub.glob("seed_*.json"))
    if not seed_files:
        return None

    arrs: list[np.ndarray] = []
    for sp in seed_files:
        try:
            obj = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            continue
        traces = obj.get("traces") if isinstance(obj, dict) else None
        if not isinstance(traces, dict):
            continue
        cell = traces.get(scenario, {}).get(mode, {})
        if not isinstance(cell, dict):
            continue
        seq = cell.get(field)
        if not isinstance(seq, list) or not seq:
            continue
        arrs.append(np.asarray(seq, dtype=float))
    if not arrs:
        return None
    # Drop any rare seeds whose trace length disagrees with the modal
    # length (truncated runs). The mode is taken as the most common
    # length across the seeds we collected.
    lengths = [a.shape[0] for a in arrs]
    if not lengths:
        return None
    n = max(set(lengths), key=lengths.count)
    arrs = [a for a in arrs if a.shape[0] == n]
    if not arrs:
        return None
    return np.vstack(arrs)


def _per_seed_window_inputs(scenario: str, mode: str, hours: np.ndarray,
                             pre_threshold: float = 24.0):
    """Per-seed windowed action/ARI/waste arrays for fig 4 panels B/C/D.

    Loads the per-seed action / ari / waste traces for one
    (scenario, mode) cell, slices them into the pre and during
    windows defined by ``pre_threshold`` (h=24 for cyber_outage),
    and returns a small dict of (n_seeds, n_window_steps) arrays
    the figure code can mean / bootstrap over seeds.

    Returns None when any of the three traces is missing or when
    fewer than 2 seeds are available -- in that case the caller
    falls back to its single-seed step-level computation. The
    fallback path is what local development hits (where only
    seed_42.json / seed_1337.json with heatwave-only traces exist);
    on HPC where all 5 scenarios x 20 seeds are dumped this helper
    returns the full multi-seed envelope.
    """
    a = _load_per_seed_traces(scenario, mode, "action_trace")
    ari = _load_per_seed_traces(scenario, mode, "ari_trace")
    waste = _load_per_seed_traces(scenario, mode, "waste_trace")
    if a is None or ari is None or waste is None:
        return None
    if a.shape[0] < 2:
        return None
    if a.shape[0] != ari.shape[0] or a.shape[0] != waste.shape[0]:
        return None
    n = min(a.shape[1], ari.shape[1], waste.shape[1], hours.shape[0])
    h = hours[:n]
    pm = h < pre_threshold
    dm = h >= pre_threshold
    return {
        "n_seeds": int(a.shape[0]),
        "action_pre": a[:, :n][:, pm].astype(int),
        "action_dur": a[:, :n][:, dm].astype(int),
        "ari_pre":   ari[:, :n][:, pm],
        "ari_dur":   ari[:, :n][:, dm],
        "waste_pre": waste[:, :n][:, pm],
        "waste_dur": waste[:, :n][:, dm],
    }


def _load_per_seed_slca_components(scenario: str, mode: str
                                    ) -> dict[str, np.ndarray] | None:
    """Per-seed mean of each SLCA component {C, L, R, P} for one
    (scenario, mode) cell.

    Walks ``RESULTS_DIR/benchmark_seeds/`` (flat or RUN_TAG-tagged
    layout) and pulls the per-step ``slca_component_trace`` from each
    seed's envelope. For each seed, the per-step list of dicts
    ``[{"C": ..., "L": ..., "R": ..., "P": ..., "composite": ...}, ...]``
    is collapsed to one mean-per-component, giving a per-seed scalar
    per component. Across-seed mean and SE on those scalars is the
    apples-to-apples cross-seed uncertainty for the SLCA-decomposition
    bar chart in fig 3 panel D.

    Returns
    -------
    dict mapping component letter ("C"/"L"/"R"/"P") to a (n_seeds,)
    numpy array of cross-step means, or None if per-seed JSONs are
    absent OR carry a pre-2026-05 envelope without
    ``slca_component_trace`` (in which case the figure code falls
    back to plotting means without error bars).

    The 2026-05 ``TRACE_FIELDS`` extension dumps slca_component_trace
    per seed, so this helper returns proper cross-seed arrays on any
    fresh HPC run; older runs (only ari_trace) yield None.
    """
    import json
    seeds_root = RESULTS_DIR / "benchmark_seeds"
    if not seeds_root.exists():
        return None
    seed_files = list(seeds_root.glob("seed_*.json"))
    if not seed_files:
        for sub in seeds_root.iterdir():
            if sub.is_dir():
                seed_files.extend(sub.glob("seed_*.json"))
    if not seed_files:
        return None

    components = ("C", "L", "R", "P")
    per_seed: dict[str, list[float]] = {c: [] for c in components}

    for sp in seed_files:
        try:
            obj = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            continue
        traces = obj.get("traces") if isinstance(obj, dict) else None
        if not isinstance(traces, dict):
            continue
        cell = traces.get(scenario, {}).get(mode, {})
        seq = cell.get("slca_component_trace")
        if not isinstance(seq, list) or not seq:
            continue
        # Older flat list[float] shape (pre-2026-05) -- skip rather
        # than try to interpret.
        if not isinstance(seq[0], dict):
            continue
        for c in components:
            vals = [float(s[c]) for s in seq if c in s]
            if vals:
                per_seed[c].append(float(np.mean(vals)))

    # Need at least 2 seeds for a meaningful cross-seed SE.
    if any(len(per_seed[c]) < 2 for c in components):
        return None

    return {c: np.asarray(per_seed[c], dtype=float) for c in components}


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
    """Last-resort error-bar source when neither benchmark_summary.json
    bootstrap CIs nor benchmark_seeds/ per-seed std arrays are present
    (e.g. a single ``run_all(seed=...)`` invocation rendered from cwd).

    2026-05 audit fix: pre-2026-05 this function returned ``sem *
    sqrt(N) * 0.5`` -- the function's *own* docstring derided
    "synthetic 5-percent-of-value bars" upstream and then inherited
    the same sin with a different magic number. ``sem * sqrt(N)``
    cancels the SEM denominator and devolves to plain within-episode
    standard deviation; the trailing ``* 0.5`` is statistically
    meaningless (it is neither a CI multiplier nor a confidence
    coverage probability).

    The right answer when no real uncertainty source is available is
    "no error bars" -- which is what this function now returns. Code
    paths that consume None render the bars without caps. The
    bar-drawing call sites (fig 6 / 7 / 8 / 9 panel C) already gate
    capsize/error_kw on ``yerr is not None`` so this is byte-stable
    on the canonical HPC render (which always has bootstrap CIs from
    aggregate_seeds.py) and only changes behaviour on the local-only
    single-seed fallback path, where invisible error bars are now
    honest about the absence of a multi-seed uncertainty estimate.

    Reviewers running ``DETERMINISTIC_MODE=true python
    generate_figures.py`` see fig 6/7/8 bars without caps and a clean
    figure; reviewers running the canonical 20-seed HPC pipeline see
    full bootstrap CI caps. No middle ground with a fudged magnitude.
    """
    return None


def fig6_cross(data):
    """2x2 grouped bars: ARI, RLE, waste, SLCA across scenarios for 3 methods.
    Error bars are drawn from (in order): benchmark_summary.json bootstrap
    CIs, benchmark_seeds/ per-seed std, or the per-step trace std as a
    last-resort within-episode fallback."""
    bench = _load_benchmark_ci()

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    # suptitle is applied at the end with the larger fig6-specific font.

    # Per-element font sizes aligned to the four-panel-figure family
    # (figs 2 / 3 / 5 / 6 all use the +1 bump over canonical) per
    # user "all four-panel figures must be identical" request.
    # Previously fig 6 used a larger +4/+3/+3/+3 bump; bringing it
    # down to +1 across the board gives all four 4-panel figures
    # the same text scale.
    _F6_TITLE = SUBPLOT_TITLE_SIZE + 1   # 20 (matches figs 2/3/5)
    _F6_AXIS  = AXIS_LABEL_SIZE + 1      # 18 (matches figs 2/3/5)
    _F6_TICK  = TICK_FONT_SIZE + 1       # 16 (matches figs 2/3/5)
    _F6_LEG   = LEGEND_FONT_SIZE + 1     # 16 (matches figs 2/3/5)

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
    # which is the expected layout when each group already carries 8
    # well-separated bars distinguished by color.
    width = 0.98 / n_modes
    x_scale = 1.10

    # Bumped per-element font sizes for fig7 — the previous +3-tick /
    # +4-title bumps still read small against the 24-inch figure width
    # at paper scale, so each tier moves up another 2 points to land
    # the title at 25pt, axis at 20pt (matched to ticks), ticks at
    # 20pt, legend at 19pt.
    #
    # 2026-04 fix: y-axis title size is matched to the x-axis tick
    # label size (both 20pt) per the user's "match all y-axis titles
    # to x-axis title size" request. fig7 has no explicit x-axis
    # title, so the x-axis text the reader sees is the rotated tick
    # labels (Heatwave / Overproduction / Cyber Outage / Price
    # Volatility); matching the y-axis title to those keeps the two
    # axes' lettering at the same visual weight. The previous +5
    # axis bump put the y-axis title at 22pt, which already exceeded
    # the tick label size, but it was being silently overridden back
    # to AXIS_LABEL_SIZE = 17 by _apply_style further below. The
    # re-apply line after _apply_style fixes that override AND
    # cements the new 20pt match.
    _F7_TITLE = SUBPLOT_TITLE_SIZE + 6   # 25
    _F7_AXIS  = TICK_FONT_SIZE + 5       # 20 (matches _F7_TICK)
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
        # Re-apply the y-axis title size after _apply_style. Without
        # this, _apply_style.set_size(AXIS_LABEL_SIZE) silently
        # overrides the _F7_AXIS=20 we just set above and the
        # rendered y-axis title falls back to the canonical 17pt -
        # which is why the previous fig7 panels showed the y-axis
        # title visibly smaller than the x-axis tick labels even
        # though the source code claimed it was larger. The
        # re-apply mirrors what is already done for the x/y tick
        # labels above.
        ax.yaxis.label.set_size(_F7_AXIS)
        ax.yaxis.label.set_weight("bold")

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

    # Per-element font sizes (late-May 2026 user request: trim fig 8 +
    # fig 10 by one point each so the panel content is not as visually
    # heavy as figs 6/7's 4-bump stack). Bump cascade kept proportional
    # to the originals (+3 / +2 / +2 / +2 = -1 from the previous
    # +4 / +3 / +3 / +3 stack) so titles still read as the dominant
    # element and ticks/legend stay readable on the (18, 7.5) figsize.
    _F8_TITLE = SUBPLOT_TITLE_SIZE + 3   # 22
    _F8_AXIS  = AXIS_LABEL_SIZE + 2      # 19
    _F8_TICK  = TICK_FONT_SIZE + 2       # 17
    _F8_LEG   = LEGEND_FONT_SIZE + 2     # 17

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
    # Legend anchored to the upper center of the panel \u2014 sits over the
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
    """Return per-scenario context-honor summary (honored, ignored, rate).

    Resolution order:

      1. ``benchmark_summary.json`` aggregated across all seeds. The
         aggregator records ``context_active_steps`` / ``context_honored_steps`` /
         ``context_honor_rate`` per (scenario, mode) as multi-seed means
         in the canonical run summary, so this is the broadest source —
         it covers all 5 scenarios when the benchmark has run them, even
         if the per-scenario ``context_alignment_<scenario>.json`` files
         were not all written (e.g., when a partial HPC run only emitted
         a subset).
      2. Fallback: per-scenario ``context_alignment_<scenario>.json``
         files (single-seed, written by ``run_all`` directly). Use
         whichever scenarios have files when the summary is missing.

    The two paths give different numbers — the summary is the multi-seed
    average (e.g. heatwave honor ~52% across 20 seeds) while the per-
    scenario file is single-seed (e.g. heatwave 74% on seed 42). The
    multi-seed source is more representative of the published claim and
    is preferred. Either way the rows have the same shape so the panel-C
    bar-chart code is unchanged.
    """
    rows = _fig9_load_alignment_from_summary()
    if rows:
        return rows
    return _fig9_load_alignment_from_files()


def _fig9_load_alignment_from_summary():
    """Pull per-scenario context-honor stats from benchmark_summary.json."""
    summary_path = RESULTS_DIR / "benchmark_summary.json"
    if not summary_path.exists():
        return []
    import json as _json_mod
    payload = _json_mod.loads(summary_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("summary"), dict):
        summary = payload["summary"]
    else:
        summary = payload
    rows = []
    for s in SCENARIOS:
        ag = summary.get(s, {}).get("agribrain", {})
        if not isinstance(ag, dict):
            continue
        cas = ag.get("context_active_steps")
        chs = ag.get("context_honored_steps")
        chr_ = ag.get("context_honor_rate")
        cas_mean = float(cas["mean"]) if isinstance(cas, dict) and "mean" in cas else None
        chs_mean = float(chs["mean"]) if isinstance(chs, dict) and "mean" in chs else None
        chr_mean = float(chr_["mean"]) if isinstance(chr_, dict) and "mean" in chr_ else None
        if cas_mean is None or chr_mean is None:
            continue
        if cas_mean <= 0:
            continue
        honored = int(round(chs_mean if chs_mean is not None else cas_mean * chr_mean))
        rows.append({
            "scenario": s,
            "label": SCENARIO_LABELS.get(s, s),
            "honored": honored,
            "ignored": max(0, int(round(cas_mean)) - honored),
            "rate": chr_mean,
        })
    return rows


def _fig9_load_honor_matrix(modes=("agribrain", "mcp_only",
                                    "pirag_only", "no_context"),
                            metric_key: str = "context_influence_rate"):
    """Per-(scenario, mode) context-rate + bootstrap CI matrix.

    Returns ``{scenario: {mode: {"rate": float, "ci_low": float,
    "ci_high": float, "active": float}}}`` pulled from
    benchmark_summary.json. The ``ci_low`` / ``ci_high`` keys are the
    bootstrap-mean confidence interval bounds (BCa) used by panel C
    of fig 9 to render asymmetric error bars on the grouped-bar
    plot. ``active`` is the mean number of context-active steps used
    to flag modes with zero context activity (e.g. ``no_context``,
    where the rate is 0/0 by construction and should be plotted as
    a structural-zero comparison rather than a measurement).

    The 2026-05 update made ``metric_key`` selectable. Fig 9 panel (c)
    now defaults to ``context_influence_rate`` (the count of steps
    where the modifier flipped the chosen action vs base argmax) as
    the headline metric; ``context_honor_rate`` (the count of steps
    where modifier-argmax matched chosen action) is retained in the
    JSON for the supplementary methods table and is selectable here
    by passing ``metric_key="context_honor_rate"``. Both rates share
    the same denominator (``context_active_steps`` gated at
    ``max(|modifier|) > 0.10``) so the cells are directly
    comparable. The function silently falls back to
    ``context_honor_rate`` when the requested key is absent from the
    summary, so legacy benchmark snapshots continue to render.

    Empty dict when the summary file is missing.
    """
    summary_path = RESULTS_DIR / "benchmark_summary.json"
    if not summary_path.exists():
        return {}
    import json as _json_mod
    payload = _json_mod.loads(summary_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("summary"), dict):
        summary = payload["summary"]
    else:
        summary = payload
    out: dict = {}
    for sc in SCENARIOS:
        sc_block = summary.get(sc, {})
        if not isinstance(sc_block, dict):
            continue
        sc_out = {}
        for mode in modes:
            ep = sc_block.get(mode, {})
            if not isinstance(ep, dict):
                continue
            chr_ = ep.get(metric_key)
            # Legacy-snapshot fallback: older benchmark_summary.json
            # files only carry context_honor_rate. If the requested
            # key is missing but honor_rate is present, render with
            # honor_rate so the panel does not vanish on legacy
            # snapshots. The fig 9 caption / supplementary methods
            # is responsible for noting which metric a given run
            # rendered with.
            if not isinstance(chr_, dict) and metric_key != "context_honor_rate":
                chr_ = ep.get("context_honor_rate")
            if not isinstance(chr_, dict):
                continue
            rate = float(chr_.get("mean", 0.0))
            ci_low = float(chr_.get("ci_low", rate))
            ci_high = float(chr_.get("ci_high", rate))
            cas = ep.get("context_active_steps", {})
            cas_mean = (float(cas.get("mean", 0.0))
                         if isinstance(cas, dict) else 0.0)
            sc_out[mode] = {
                "rate": rate,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "active": cas_mean,
            }
        if sc_out:
            out[sc] = sc_out
    return out


def _fig9_load_alignment_from_files():
    """Fallback: per-scenario context_alignment_<scenario>.json files."""
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
#   (a) Cohen's d heatmap — agribrain vs each of 5 baselines, log-colored.
#   (b) % ARI improvement forest plot — same 25 comparisons, recoded to
#       relative gain so the axis reads in human terms.
#   (c) Context honor rate per scenario.
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
    """Consolidated Figure 9: effect-size, performance gain, context honor.

    Three panels, each keyed on benchmark_significance.json and built to
    carry visual variance proportional to the strength of the result:

      (a) Effect-size heatmap. Cohen's d_pooled for ARI, agribrain
          vs each of 5 baselines, per scenario. Log-colored so the
          ~22x spread on the current run (d in 0.0 to ~22.4 across
          the 25 comparisons) reads as a clear gradient. Cell text
          = numeric d.
      (b) % ARI improvement bar plot. Same 25 comparisons rolled up
          to one bar per baseline (mean across scenarios) recoded as
          100*(mean_diff)/baseline_mean. Whiskers show the
          across-scenario range; the headline magnitude on the
          current run is ~+36.7% vs Static down to ~+1.2% vs MCP
          only. The relative scale makes the cross-baseline
          gradient instantly interpretable.
      (c) Context-active influence rate. Fraction of context-active
          decisions where the modifier flipped the agent's argmax
          (i.e. context actually changed the chosen action).
          Source: benchmark_summary.json[context_influence_rate],
          20-seed bootstrap mean with BCa CI.
    """
    sig_data = _fig9_load_significance()
    align_rows = _fig9_load_alignment()
    n_seeds_global = _fig9_load_n_seeds()
    method_means = _fig9_load_method_means() or {}

    # Width-ratio rebalancing (2026-05 third pass: shrink the visible
    # whitespace BETWEEN panels B and C by extending panel C leftward.
    # Pre-2026-05-pass3 the layout was figsize=(24, 7.5),
    # width_ratios=[1.40, 1.20, 1.55] -> A=8.10, B=6.94, C=8.96 in.
    # The +36.7% headline in panel B sits at axis-fraction ~0.46 under
    # the symlog mapping with xlim=80, leaving ~half of panel B empty
    # to its right. That residual whitespace plus the standard wspace
    # gap before panel C opened a visible vertical "no-data" gutter.
    # Fix: grow figsize.width 24 -> 26 and route both extra inches
    # into panel C's ratio (1.55 -> 1.85). Panels A and B keep their
    # physical widths to within ~0.1 in:
    #   A: 24 * 1.40/4.15 = 8.10 in -> 26 * 1.40/4.45 = 8.18 in (+0.08)
    #   B: 24 * 1.20/4.15 = 6.94 in -> 26 * 1.20/4.45 = 7.01 in (+0.07)
    #   C: 24 * 1.55/4.15 = 8.96 in -> 26 * 1.85/4.45 = 10.81 in (+1.85)
    # Net: panel C gains 1.85 in of canvas, which (a) closes the
    # B-to-C gutter the user flagged and (b) stretches the existing
    # bars / inter-group gap proportionally so the 5-scenario x
    # 3-mode grouped-bar comparison reads more clearly.
    fig, axes = plt.subplots(1, 3, figsize=(26, 7.5),
                             gridspec_kw={"width_ratios": [1.40, 1.20, 1.85]})

    # Per-element font sizes aligned to fig 7's pattern (25/20/20/19)
    # per user "all three-panel figures must be identical" request.
    # Same calculation pattern fig 7 uses (TITLE=SUBPLOT+6,
    # AXIS=TICK+5, TICK=TICK+5, LEG=LEG+4) so figs 4 / 7 / 9 share
    # the same text scale across all three panels of each figure.
    # _F9_ANNOT is unchanged from the previous +4 bump - it sits
    # between the legend and the tick label sizes and reads cleanly
    # at the same point size used elsewhere.
    _F9_TITLE = SUBPLOT_TITLE_SIZE + 6    # 25 (matches _F7_TITLE)
    _F9_AXIS  = TICK_FONT_SIZE + 5        # 20 (matches _F7_AXIS, equals _F9_TICK)
    _F9_TICK  = TICK_FONT_SIZE + 5        # 20 (matches _F7_TICK)
    _F9_LEG   = LEGEND_FONT_SIZE + 4      # 19 (matches _F7_LEG)
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
                # Prefer the canonical ``cohens_d_pooled`` key (the
                # explicit name documented in STATISTICAL_METHODS.md
                # §2.3); fall back to the legacy ``cohens_d`` alias
                # for older benchmark snapshots produced before the
                # aggregator started emitting both names. Defensive
                # only: aggregate_seeds.py currently writes both keys
                # with identical values (line 895), so this read order
                # is byte-stable on every current snapshot. Future
                # aggregator changes that drop the alias would not
                # silently break panel A.
                if "cohens_d_pooled" in ari:
                    d_mat[i, j] = float(ari["cohens_d_pooled"])
                elif "cohens_d" in ari:
                    d_mat[i, j] = float(ari["cohens_d"])
                if "p_value_adj" in ari:
                    p_mat[i, j] = float(ari["p_value_adj"])

        # Sequential green colormap, log-normalised because the
        # canonical 20-seed cohens_d_pooled range spans roughly two
        # decades (0.0-22.4 on the current run) and a linear scale
        # would crush the small-effect end. Pre-2026-05 the floor
        # was hard-coded at vmin=1.5, which clipped 11/25 cells
        # (every "vs No Context / vs piRAG only / vs MCP only" cell
        # plus the literal d=0.0 for heatwave/vs MCP only) to the
        # same lightest green and hid the small-effect tail. Floor
        # the LogNorm vmin at the smallest positive d (or 0.2 if
        # that is itself smaller) so cells in [0.2, 1.5] now read
        # as distinguishable shades. Cells whose d rounds to 0.0
        # are still clipped to the floor (LogNorm requires vmin > 0)
        # but their numeric "0.0" cell text remains visible.
        finite = d_mat[np.isfinite(d_mat)]
        if finite.size:
            positive = finite[finite > 0.0]
            d_min = max(float(positive.min()) if positive.size else 0.2, 0.2)
            d_max = max(float(finite.max()), d_min * 2)
        else:
            d_min, d_max = 0.2, 80.0
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=d_min, vmax=d_max)
        im = ax.imshow(d_mat, cmap="Greens", norm=norm,
                       aspect="auto", interpolation="nearest")

        # Cell annotation: numeric Cohen's d, with a halo so the
        # text is legible across the full gradient (white on
        # saturated cells, dark on pale ones).
        from matplotlib import patheffects as _pe
        for i in range(n_rows):
            for j in range(n_cols):
                d = d_mat[i, j]
                if not np.isfinite(d):
                    continue
                # Pick text color that contrasts: white on saturated
                # cells (upper third of log range), dark on pale cells.
                # Use d_min as a floor inside the log so a literal
                # d=0.0 cell (which LogNorm clips to vmin) doesn't
                # produce log(0) -> -inf.
                d_for_color = max(d, d_min)
                cell_frac = (
                    (np.log(d_for_color) - np.log(d_min))
                    / max(np.log(d_max) - np.log(d_min), 1e-9)
                )
                txt_color = "white" if cell_frac > 0.55 else "#1B5E20"
                halo = "black" if txt_color == "white" else "white"
                # Cell text reduced from _F9_TICK=20 to _F9_TICK-4=16
                # alongside the panel-A width bump - even at the new
                # 1.40 width ratio, 20pt bold three-character strings
                # like "22.5" and "76.0" overflow the cell footprint.
                # 16pt bold sits comfortably within each cell at the
                # 22-inch figure width without losing legibility.
                t = ax.text(j, i, f"{d:.1f}",
                            ha="center", va="center",
                            fontsize=_F9_TICK - 4, fontweight="bold",
                            color=txt_color)
                t.set_path_effects([_pe.withStroke(linewidth=1.6, foreground=halo)])

        ax.set_xticks(np.arange(n_cols))
        # Rotation bumped from 20° to 30° per "no overlapping"
        # mandate: at 20° with 5 baseline labels and panel A at
        # 1.40 width_ratio, the horizontal projection of "vs Hybrid
        # RL" (~1.5 inches) was crowding the adjacent "vs No
        # Context". 30° with rotation_mode="anchor" rotates the
        # label about its right edge so it stays under its own
        # column without invading neighbors.
        ax.set_xticklabels([lbl for _, lbl in _BASELINES],
                           rotation=30, ha="right",
                           rotation_mode="anchor")
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels([SCENARIO_LABELS.get(s, s) for s in scenarios_in_sig])

        # Slim colorbar on the right edge — gives the gradient an
        # explicit scale for readers who want exact magnitudes.
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cbar.set_label("Cohen's d (pooled, ARI)",
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
    # `vs static` (~+36.7 % on the current run) down to `vs MCP only`
    # (~+1.2 %) is instantly legible. The "+75 %" headline that
    # earlier comments referenced is from a pre-2026-04 baseline run;
    # the panel autoscales via max_hi so the symlog x-range tracks
    # whatever the current data is.
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
            # Numeric label: mean only on the right side of each bar.
            # The (range ...) annotation that used to follow was removed
            # per user request - the min/max whiskers on the bar
            # already convey the same per-scenario spread without
            # duplicating it as text.
            label_x = v["hi"] + 0.5 if v["hi"] < 2.0 else v["hi"] * 1.10
            ax.text(label_x, i,
                    f"+{v['mean']:.1f}%",
                    va="center", ha="left",
                    fontsize=_F9_ANNOT - 1, fontweight="bold", color="#212121")

        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels([v["label"] for _, v in order])
        ax.invert_yaxis()  # weakest baseline (largest gain) on top

        # Symlog so the +1..+15 % cluster has visual room and the
        # headline static bar doesn't dominate. Tick layout is the
        # load-bearing readability piece: the previous {0,5,50,100}
        # set put 50% and 100% visually adjacent in the compressed
        # log region (a pre-2026-04 baseline reached +75 %, so the
        # gap between log(50) and log(100) was ~0.30 dec and collided
        # at the bold 20pt tick fontsize). Switch linthresh from 2.0
        # to 5.0 so the linear region absorbs the small-bar cluster
        # cleanly and the log region begins where the big bars
        # actually live; tick set drops to {0, 5, 25, 75} which gives
        # clearly separated labels across the whole axis without
        # losing fidelity at either end. xlim caps at 80 with the
        # max_hi*1.10 guard so the panel auto-grows if the data
        # climbs back toward +75 %; the current run tops at +36.7 %
        # so the cap is the floor.
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
    # Panel (c) — Context influence rate per scenario x mode
    # =================================================================
    # Grouped-bar comparison across the three context-active ablation
    # modes (agribrain, pirag_only, mcp_only) for each scenario.
    #
    # 2026-05 metric switch: this panel now reports ``context_influence_rate``,
    # the fraction of context-active steps where the modifier
    # CHANGED the chosen action (argmax(base_logits + modifier) !=
    # argmax(base_logits)). The previous metric ``context_honor_rate``
    # measured whether the modifier's argmax matched the chosen
    # action -- which was structurally biased against the full-stack
    # AgriBrain mode because the policy's richer base logits (PINN +
    # full SLCA + batch FIFO + both context channels) override the
    # modifier's argmax more often even though the modifier still
    # moved the decision boundary. The honor metric was retained in
    # benchmark_summary.json for the supplementary methods table; a
    # reviewer who wants to see both can read either off the same
    # cells.
    #
    # The no_context arm is omitted by user request - it is
    # structurally zero (the no_context coordinator skips
    # _compute_step_context entirely so context_active_steps == 0
    # and the rate is 0/0 by construction). That structural-zero
    # comparison belongs in the manuscript text rather than as a
    # 0%-bar that conveys no information visually.
    #
    # Error bars: BCa bootstrap CI bounds from
    # benchmark_summary.json's ``context_influence_rate.ci_low /
    # ci_high`` per (scenario, mode) cell. Asymmetric so the
    # rendered error caps reflect the actual uncertainty band
    # (typically narrower on the high side because the rate is
    # bounded above at 1.0). Cells with no per-seed CI fall back
    # to a zero-width bar rather than a misleading symmetric
    # default.
    # Panel C bar styling + font sizes match fig 7 (the ablation
    # grouped-bar figure). With the post-2026-04 alignment of _F9_*
    # to _F7_* (TITLE=25, AXIS=20, TICK=20, LEG=19), the per-panel
    # _PANEL_C_* constants this block previously defined became
    # redundant and were retired - panel C now reads directly from
    # the figure-level _F9_* constants alongside panels A and B.

    ax = axes[2]
    # 2026-05: switched from context_honor_rate to context_influence_rate
    # (see panel docstring above). The loader falls back to
    # context_honor_rate when the new field is absent (legacy
    # benchmark_summary snapshots), so the panel still renders during
    # the transition window.
    honor_matrix = _fig9_load_honor_matrix(
        modes=("agribrain", "pirag_only", "mcp_only"),
        metric_key="context_influence_rate",
    )
    _PANEL_C_MODES = [
        ("agribrain",   "AgriBrain",    COLORS.get("agribrain",   "#26A69A")),
        ("pirag_only",  "piRAG only",   COLORS.get("pirag_only",  "#1565C0")),
        ("mcp_only",    "MCP only",     COLORS.get("mcp_only",    "#E65100")),
    ]
    if honor_matrix:
        scenarios_in_matrix = [s for s in SCENARIOS if s in honor_matrix]
        n_groups = len(scenarios_in_matrix)
        n_modes = len(_PANEL_C_MODES)
        # Bar layout (2026-05 third-pass: panel C now sits at canvas
        # width ~10.8 in after the figsize 24->26 / width_ratio
        # 1.55->1.85 rebalance, so the bars need a proportional bump
        # to keep the visual density readers expect from a grouped-bar
        # plot). Total group width bumped 0.98 -> 1.20 (each bar
        # 0.40 axis units, was 0.327); x_scale stays at 1.55 so the
        # inter-group gap shrinks from 0.57 to 0.35, which keeps the
        # scenario-level visual separation intact while filling the
        # widened panel. The legend in the upper-left still has clean
        # headroom above the leftmost (heatwave) MCP-only bar (~22%
        # height vs the legend's ~85% lower edge).
        width = 1.20 / n_modes
        x_scale = 1.55
        x_base = np.arange(n_groups) * x_scale

        for i, (mode, label, color) in enumerate(_PANEL_C_MODES):
            heights = []
            err_low = []
            err_high = []
            for sc in scenarios_in_matrix:
                cell = honor_matrix.get(sc, {}).get(mode, {})
                rate = cell.get("rate", 0.0)
                heights.append(100.0 * rate)
                # Asymmetric BCa CI bounds, clipped to [0, 100].
                lo = max(0.0, 100.0 * (rate - cell.get("ci_low", rate)))
                hi = max(0.0, 100.0 * (cell.get("ci_high", rate) - rate))
                err_low.append(lo)
                err_high.append(hi)
            xs = x_base + i * width
            # Bar styling matched to fig 7 panel-bar exactly:
            # alpha=0.92, edgecolor="white", linewidth=0.7. Yerr
            # block uses the existing _ERR_CAPSIZE / _ERR_KW
            # constants shared across figs 6 / 7 / 8 / 9.
            ax.bar(xs, heights, width, color=color,
                   alpha=0.92, edgecolor="white", linewidth=0.7,
                   label=label,
                   yerr=[err_low, err_high],
                   capsize=_ERR_CAPSIZE, error_kw=_ERR_KW)

        # Center each x-tick under its group of n_modes bars (same
        # idiom fig 7 uses for its 8-mode groups).
        ax.set_xticks(x_base + (n_modes - 1) * width / 2)
        # Rotation bumped from 20° to 30° + rotation_mode="anchor"
        # per "no overlapping" mandate: at 20° with 5 scenario
        # labels in a panel of this width, the longest label
        # ("Overproduction") at 14 chars projects ~1 inch
        # horizontally and was crowding the adjacent group.
        # rotation_mode="anchor" rotates each label around its
        # right edge so the label stays under its own group.
        ax.set_xticklabels(
            [SCENARIO_LABELS.get(s, s) for s in scenarios_in_matrix],
            rotation=30, ha="right", rotation_mode="anchor",
        )
        # ylim bumped to 115 so the legend has visible headroom above
        # the highest bar+CI cap. Heatwave's tallest bar (mcp_only)
        # tops at ~67% with the upper whisker reaching ~73%; in axis-
        # fraction terms that is ~0.63, leaving the upper third of
        # the panel (y > 80) as clear headroom for the legend.
        ax.set_ylim(0, 115)
        # Legend at upper-left: the leftmost two scenario groups
        # (Heatwave, Overproduction) have their tallest bar+whisker
        # tops well below axis-fraction 0.70, so a 3-entry single-
        # column legend at 19pt sits cleanly in the upper-left corner
        # with no overlap on any bar, whisker, or tick label.
        ax.legend(loc="upper left", fontsize=_F9_LEG,
                  ncol=1, framealpha=0.95, edgecolor="#757575",
                  fancybox=False, shadow=False)
    else:
        ax.text(0.5, 0.5, "benchmark_summary.json not available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=_F9_ANNOT, color="#616161")
    # Y-axis title size re-applied after _restyle to defeat the
    # _apply_style render-path bug (same fix as fig 7 commit
    # 3feb090): _restyle calls _apply_style internally which
    # silently normalises ax.yaxis.label back to AXIS_LABEL_SIZE=17
    # even when set_ylabel was called with fontsize=_F9_AXIS. The
    # explicit re-apply ensures the rendered y-axis title actually
    # lands at _F9_AXIS=20 to match the x-axis tick labels.
    _restyle(ax, "(c) Context Influence Rate",
             ylabel="Influence Rate (%)")
    ax.yaxis.label.set_size(_F9_AXIS)
    ax.yaxis.label.set_weight("bold")

    fig.suptitle("Performance Gain over Baselines and Context Influence",
                 y=0.995, fontsize=FIG_TITLE_SIZE, fontweight="bold")
    # Layout spacing tightened post-2026-04 per user "there must be
    # no overlapping" mandate:
    #   - rect bottom raised from 0.02 to 0.06 so the rotated x-tick
    #     labels in panels (a) and (c) get extra clearance below the
    #     panel and do not clip into the figure margin.
    #   - w_pad raised from 1.0 to 2.5 so panels (a) (heatmap) /
    #     (b) (forest plot with right-anchored labels) / (c)
    #     (grouped bars with rotated tick labels) have enough
    #     horizontal padding that tick labels and legend boxes from
    #     adjacent panels cannot collide. The 22-inch figure width
    #     absorbs this generously - each panel still has 6+ inches
    #     of plotting area.
    fig.tight_layout(rect=[0, 0.06, 1, 0.985], w_pad=2.5)
    _save(fig, "fig9_robustness")


def fig10_latency_quality_frontier(data):
    """Latency-quality frontier with two zones, fully matching the shared
    figure style (bold titles and axis labels, 800 DPI, no label
    overlaps). Panel (a) shows the lightweight methods (sub-millisecond);
    panel (b) shows the MCP/piRAG-enabled methods with the no-context
    reference point and an overhead annotation.

    Multi-seed observations from the post-2026-04 run worth flagging in
    the figure caption / discussion section:

    - Per-(scenario, mode) decision-latency is not strictly monotone in
      scenario stress. AgriBrain's mean latency is highest in the
      ``baseline`` scenario (~11 ms) and lowest in ``heatwave``
      (~6.8 ms). The mechanism: under heat stress the policy hits the
      Recovery knee + food-safety override paths early, which
      short-circuit some of the deliberative MCP/piRAG queries that the
      stable ``baseline`` path runs in full. Latency reflects compute
      depth (richer context retrieval = longer), not stress severity.
    - ``pirag_only`` shows a wider per-seed latency spread on
      ``adaptive_pricing`` (~13 ms with CI [10.6, 16.1]) than on other
      scenarios. The Bollinger-band-driven retrieval pattern in that
      scenario produces seed-dependent retrieval depth (some seeds
      trigger long context fetches, others short). The wide CI is a
      genuine measurement; the headline frontier mark uses the mean.
    """
    # Panel (b) uses a broken x-axis (split between 0.5 ms and 5.0 ms)
    # to suppress the empty zone between the No Context reference
    # (~0.18 ms) and the context-aware cluster (~5.85 ms). The broken
    # axis is implemented as two sub-axes inside the right gridspec
    # cell with width_ratios=[1, 5] — the small left sub-axis carries
    # the No Context reference while the larger right sub-axis carries
    # the three jittered context-aware markers.
    import matplotlib.gridspec as _gridspec
    # figsize widened from (18, 7.5) to (26, 7.5) when panel (c) was
    # added (late-May 2026 user request: show the per-scenario Delta-ARI
    # vs No Context as bars so reviewers can read the magnitude of the
    # context-channel gain alongside the latency-quality scatter). The
    # outer gridspec is now 1x3 with width_ratios=[1, 1.2, 1] -- panel
    # (b) gets a slightly wider slot because it carries the broken
    # x-axis and the overhead-arrow annotation.
    fig = plt.figure(figsize=(26, 7.5))
    outer_gs = _gridspec.GridSpec(
        1, 3, figure=fig, width_ratios=[1, 1.2, 1], wspace=0.32,
    )
    # 2026-05 panel-A broken-axis. Pre-2026-05 panel A had a single
    # axis with xlim (0.06, 0.175). 20-seed canonical means showed
    # Static at 0.094 ms but the lightweight cluster at 0.176-0.190 ms
    # -- that's OUTSIDE the 0.175 right edge, which clipped four of
    # five modes off the panel and made it look as though only Static
    # was rendered. Fix: split panel A into the same broken-axis
    # pattern panel B uses. Left sub-axis carries Static (~0.094 ms)
    # in (0.07, 0.11); right sub-axis carries the cluster in
    # (0.165, 0.20). Within the right sub-axis the four modes are
    # still close in latency (range 0.014 ms) but cleanly separated
    # by ARI (no_context 0.596 / no_pinn 0.590 / hybrid_rl 0.571 /
    # no_slca 0.559), so the markers don't visually collide.
    inner_gs_a = outer_gs[0].subgridspec(1, 2, width_ratios=[1, 3.0], wspace=0.14)
    ax_a_left = fig.add_subplot(inner_gs_a[0])
    ax_a_right = fig.add_subplot(inner_gs_a[1], sharey=ax_a_left)
    # Inner break-axis spacing widened from 0.06 to 0.14 so the
    # right-edge "0.5" tick of the left sub-axis and the left-edge
    # "5.0" tick of the right sub-axis no longer collide visually.
    inner_gs = outer_gs[1].subgridspec(1, 2, width_ratios=[1, 5], wspace=0.14)
    ax_b_left = fig.add_subplot(inner_gs[0])
    ax_b_right = fig.add_subplot(inner_gs[1], sharey=ax_b_left)
    ax_c = fig.add_subplot(outer_gs[2])
    # Legacy compat: panel-A code path historically referenced
    # ``axes[0]`` as a single axis. Now that panel A is a broken pair,
    # we keep ``axes`` for the back-compat write site below but the
    # plotting loop routes Static to ax_a_left and the cluster to
    # ax_a_right explicitly.
    axes = [ax_a_left]
    fig.suptitle("Latency vs ARI Frontier", fontsize=FIG_TITLE_SIZE,
                 fontweight="bold", y=0.995)

    # Per-element font sizes matched to fig 8 (the other 1x2 paper
    # figure). Late-May 2026 user request: trim fig 8 + fig 10 by one
    # point each so panel content is not as visually heavy as figs 6/7.
    # Bump cascade is +3 / +2 / +2 / +2 (was +4 / +3 / +3 / +3).
    _F10_TITLE = SUBPLOT_TITLE_SIZE + 3   # 22
    _F10_AXIS  = AXIS_LABEL_SIZE + 2      # 19
    _F10_TICK  = TICK_FONT_SIZE + 2       # 17
    _F10_LEG   = LEGEND_FONT_SIZE + 2     # 17

    bench = _load_benchmark_ci() or {}

    fast_modes = ["static", "hybrid_rl", "no_pinn", "no_slca", "no_context"]
    context_modes = ["agribrain", "mcp_only", "pirag_only"]

    def _collect(modes):
        """Collect (mode, mean_latency_ms, mean_ari, yerr) per mode.

        Point positions prefer the 20-seed bootstrap means from
        ``benchmark_summary.json`` (the canonical multi-seed posture
        the rest of the manuscript reports), and fall back to the
        single-seed ``data["results"]`` payload only when the
        benchmark summary isn't available. Pre-2026-05 the figure used
        single-seed values for all marker positions, which (a) buried
        the Static marker against panel A's left spine because its
        single-seed latency landed below the 0.08 ms xlim floor, and
        (b) collapsed AgriBrain (~3.9 ms single-seed) onto piRAG Only
        (~3.6 ms single-seed) inside panel B even with the +/-0.55 ms
        jitter. The 20-seed means separate the three context-aware
        modes by ~3-7 ms (piRAG ~3.6 / MCP ~10.8 / AgriBrain ~13.9),
        which makes the panel-B jitter unnecessary and lets panel A
        show all five lightweight markers cleanly.

        Error bars use the standard error of the across-scenario mean:
        sd(per_scenario_ari) / sqrt(n_scenarios). The within-scenario
        bootstrap CIs reported in benchmark_summary.json are too tight
        (~0.001-0.005 ARI on n=20 seeds) to render visibly in panel (a)'s
        wide y-range, so we use cross-scenario variability instead -- this
        is the statistically appropriate uncertainty for the
        *cross-scenario mean* the figure plots and is consistent with the
        symmetric +/-SE convention reported elsewhere in the manuscript.
        """
        pts = []
        for mode in modes:
            scenario_aris = []
            scenario_lats = []
            for s in SCENARIOS:
                # Prefer the 20-seed bootstrap mean from
                # benchmark_summary.json; fall back to single-seed.
                sm_rec = bench.get(s, {}).get(mode, {}) if bench else {}
                ari_block = sm_rec.get("ari")
                lat_block = sm_rec.get("mean_decision_latency_ms")
                if isinstance(ari_block, dict) and "mean" in ari_block:
                    ari_val = float(ari_block["mean"])
                else:
                    ss_rec = data["results"].get(s, {}).get(mode, {})
                    if "ari" not in ss_rec:
                        continue
                    ari_val = float(ss_rec["ari"])
                if isinstance(lat_block, dict) and "mean" in lat_block:
                    lat_val = float(lat_block["mean"])
                else:
                    ss_rec = data["results"].get(s, {}).get(mode, {})
                    lat_val = float(ss_rec.get("mean_decision_latency_ms", 0.0))
                scenario_aris.append(ari_val)
                scenario_lats.append(lat_val)
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

    # --- (a) Fast modes (sub-millisecond) -- broken x-axis ---
    # Static (~0.094 ms) lives on ax_a_left; the four-mode cluster
    # (hybrid_rl / no_pinn / no_slca / no_context, all in
    # 0.176-0.190 ms) lives on ax_a_right. Within the right sub-axis
    # the four modes are clustered in latency but distinguishable by
    # ARI (0.559-0.596). Diagonal // glyphs at the break read as a
    # single panel.
    for mode, x, y, yerr in fast_pts:
        target_ax = ax_a_left if mode == "static" else ax_a_right
        target_ax.scatter(
            x, y, s=220, color=COLORS[mode], marker=MARKERS[mode],
            edgecolor="white", linewidth=1.4, alpha=0.95, zorder=5,
            label=MODE_LABELS[mode],
        )
        if yerr[0] > 0 or yerr[1] > 0:
            target_ax.errorbar(
                [x], [y],
                yerr=np.array([[yerr[0]], [yerr[1]]]),
                fmt="none", ecolor=COLORS[mode],
                elinewidth=1.6, capsize=4, alpha=0.85, zorder=4,
            )

    # Y-axis label only on the left sub-axis (the right inherits via
    # sharey). Y-range must accommodate the cross-scenario SE error
    # bars on every mode. Bottom margin 0.04 (not 0.02) so the s=220
    # static marker at ARI~0.45 isn't clipped by the bottom spine --
    # the marker has a ~0.02 ARI radius at this dpi.
    ax_a_left.set_ylabel("Mean ARI", fontsize=_F10_AXIS, fontweight="bold")
    pts_with_err_a = [(p[2], p[3]) for p in fast_pts]
    bar_lo_a = min(y - e[0] for y, e in pts_with_err_a)
    bar_hi_a = max(y + e[1] for y, e in pts_with_err_a)
    ax_a_left.set_ylim(bar_lo_a - 0.04, bar_hi_a + 0.02)

    # X-axis split. Left sub: 0.07-0.11 ms covers Static at 0.094
    # with comfortable margins. Right sub: adapts to the actual
    # cluster (0.176-0.190 on the d33b8de run, with hybrid_rl on the
    # high side). Floor the lower bound 0.005 ms below the smallest
    # cluster lat so all four markers sit clearly inside the spine;
    # ceiling 0.005 ms above the largest, same logic.
    cluster_lats = [p[1] for p in fast_pts if p[0] != "static"]
    if cluster_lats:
        cluster_lo = min(cluster_lats) - 0.005
        cluster_hi = max(cluster_lats) + 0.005
    else:
        cluster_lo, cluster_hi = 0.165, 0.20
    ax_a_left.set_xlim(0.07, 0.11)
    ax_a_right.set_xlim(cluster_lo, cluster_hi)

    # Tick layout: left sub gets two ticks framing Static; right sub
    # auto-locates ~3 ticks across the cluster.
    from matplotlib.ticker import FixedLocator as _FixedLocator
    from matplotlib.ticker import MaxNLocator as _MaxNLocator
    ax_a_left.xaxis.set_major_locator(_FixedLocator([0.08, 0.10]))
    ax_a_right.xaxis.set_major_locator(_MaxNLocator(nbins=3))

    # Hide the inner spines so the broken-axis reads as a single panel.
    ax_a_left.spines["right"].set_visible(False)
    ax_a_right.spines["left"].set_visible(False)
    ax_a_right.tick_params(left=False, labelleft=False)
    ax_a_left.yaxis.tick_left()

    # Diagonal // glyphs at the break (bottom-edge only).
    _d_a = 0.018
    _kw_left_a = dict(transform=ax_a_left.transAxes, color="#424242",
                      lw=1.4, clip_on=False)
    ax_a_left.plot((1 - _d_a, 1 + _d_a), (-_d_a, +_d_a), **_kw_left_a)
    _kw_right_a = dict(transform=ax_a_right.transAxes, color="#424242",
                       lw=1.4, clip_on=False)
    ax_a_right.plot((-_d_a / 3, +_d_a / 3), (-_d_a, +_d_a), **_kw_right_a)

    # Combine legend handles from both sub-axes (de-duplicated).
    leg_handles_a, leg_labels_a = [], []
    for sub_ax in (ax_a_left, ax_a_right):
        for h, lbl in zip(*sub_ax.get_legend_handles_labels()):
            if lbl not in leg_labels_a:
                leg_handles_a.append(h)
                leg_labels_a.append(lbl)
    # Legend at lower-right of the right sub-axis (cluster is at the
    # top of the right sub; lower-right is empty).
    _legend(ax_a_right, handles=leg_handles_a, labels=leg_labels_a,
            loc="lower right", ncol=1, fontsize=_F10_LEG)

    # Apply style to both sub-axes; restore the broken-axis spine
    # hides afterwards (because _apply_style restores defaults).
    for sub_ax in (ax_a_left, ax_a_right):
        _apply_style(sub_ax)
        sub_ax.tick_params(labelsize=_F10_TICK, length=6, width=1.4)
        for lbl in list(sub_ax.get_xticklabels()) + list(sub_ax.get_yticklabels()):
            lbl.set_fontsize(_F10_TICK); lbl.set_fontweight("bold")
    ax_a_left.spines["right"].set_visible(False)
    ax_a_right.spines["left"].set_visible(False)
    ax_a_right.tick_params(left=False, labelleft=False)
    ax_a_left.yaxis.label.set_size(_F10_AXIS)
    ax_a_left.yaxis.label.set_weight("bold")

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

    # Plot the three context-aware modes on the RIGHT sub-axis at
    # their TRUE 20-seed mean latencies (no jitter). With the 2026-05
    # switch from single-seed to 20-seed marker positions, the three
    # modes naturally separate by ~3-7 ms on the x-axis
    # (piRAG ~3.6 / MCP ~10.8 / AgriBrain ~13.9), so the +/-0.55 ms
    # jitter the earlier render needed (because AgriBrain's seed-42
    # latency collapsed onto piRAG Only's at ~3.5 ms) is now visual
    # noise that misrepresents the data. Keeping the dict for the
    # overhead-annotation code path below; the values are zero so
    # x_plot == x_actual.
    _ctx_jitter = {"agribrain": 0.0, "mcp_only": 0.0, "pirag_only": 0.0}
    # Render order matters: zorder=5 is applied to all three but
    # matplotlib still resolves ties in *call order*, so iterate in an
    # order that puts MCP Only ON TOP of both the AgriBrain triangle
    # and the piRAG diamond. The X glyph is a thin-stroke marker (no
    # interior fill) and gets visually clipped by the white edges of
    # neighbouring filled markers when drawn underneath them.
    _draw_order = {"agribrain": 0, "pirag_only": 1, "mcp_only": 2}
    ordered_ctx_pts = sorted(ctx_pts, key=lambda p: _draw_order.get(p[0], 99))
    for mode, x, y, yerr in ordered_ctx_pts:
        x_plot = x + _ctx_jitter.get(mode, 0.0)
        # Bump zorder explicitly for mcp_only as a belt-and-braces guard
        # so it lands on top even if a future edit reorders the loop.
        _z = 6 if mode == "mcp_only" else 5
        h = ax_b_right.scatter(
            x_plot, y, s=260,
            color=COLORS[mode], marker=MARKERS[mode],
            edgecolor="white", linewidth=1.4, alpha=0.95, zorder=_z,
            label=MODE_LABELS[mode],
        )
        handles_b.append(h)
        if yerr[0] > 0 or yerr[1] > 0:
            ax_b_right.errorbar(
                [x_plot], [y],
                yerr=np.array([[yerr[0]], [yerr[1]]]),
                fmt="none", ecolor=COLORS[mode],
                elinewidth=1.8, capsize=4, alpha=0.9, zorder=_z - 1,
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
        # Overhead annotation. 2026-05 design rules:
        #
        #   (a) Position: AXES-RELATIVE upper-left of the right
        #       sub-axis, not data-coords near the AgriBrain marker.
        #       The data-coord version collided with piRAG / MCP
        #       markers at publication dpi.
        #
        #   (b) Content: two compact lines.
        #         line 1: "Context overhead"
        #         line 2: "+X.X ms  |  +0.0XX ARI (d=N.N-N.N)"
        #       The Cohen's d range alone tells reviewers the effect
        #       is uniformly large (d>0.8 = large by Cohen convention).
        #       Pre-2026-05 the line was just "+X.X ms | +0.020 ARI"
        #       which made a uniformly-large effect read as a
        #       small one. The 5-line variant the audit produced was
        #       too cluttered for an in-figure annotation; per-scenario
        #       range and significance live in panel (c) and the
        #       methods table respectively.
        import json as _json_b
        sig_payload_path = RESULTS_DIR / "benchmark_significance.json"
        d_min = d_max = float("nan")
        try:
            sig_doc = _json_b.loads(sig_payload_path.read_text(encoding="utf-8"))
            sig_block = sig_doc.get("significance", sig_doc)
            ds = []
            for sc, blk in sig_block.items():
                ari_blk = blk.get("agribrain_vs_no_context", {}).get("ari", {})
                if "cohens_d_pooled" in ari_blk:
                    ds.append(float(ari_blk["cohens_d_pooled"]))
            if ds:
                d_min, d_max = min(ds), max(ds)
        except Exception:
            pass
        # Compact 2-line annotation.
        line2 = (
            f"+{agri_lat - ref[1]:.1f} ms  |  {agri_ari - ref[2]:+.3f} ARI"
        )
        if d_min == d_min and d_max == d_max:  # NaN guard
            line2 += f"  (d={d_min:.1f}-{d_max:.1f})"
        ann_text = "Context overhead\n" + line2
        # Position: anchor the box's BOTTOM just above the dashed
        # connecting line. The dashed line goes from No Context on
        # the left sub-axis (~0.18 ms, 0.596 ARI) to AgriBrain on the
        # right sub-axis (~5.7 ms, 0.616 ARI). On the right sub-axis
        # in axes-relative coords, the line peaks around y_axes ~0.75
        # at the right edge. Anchoring the box bottom at y_axes=0.78
        # places it cleanly just above the line without floating to
        # the top of the panel.
        ax_b_right.text(
            0.02, 0.78, ann_text,
            transform=ax_b_right.transAxes,
            ha="left", va="bottom",
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
    # right sub-axis adaptive to the context-aware latency cluster.
    # The earlier hardcoded right xlim of 5.0 ms came from a stale
    # multi-seed-baseline assumption that context-aware modes sat at
    # ~5.85 ms; in single-seed deterministic mode they collapse to
    # ~3-4 ms, which dropped every context marker into the suppressed
    # zone and produced an empty panel. Adaptive bounds: ~0.5 ms of
    # left margin around the leftmost jittered marker, ~0.4 ms of
    # right margin around the rightmost. Floor the lower bound at
    # 1.0 ms so the broken axis still reads as a break (not as two
    # touching sub-axes) when the cluster sits near the No Context ref.
    ax_b_left.set_xlim(0.0, 0.5)
    ctx_lat_lo = min(p[1] + _ctx_jitter.get(p[0], 0.0) for p in ctx_pts)
    ctx_lat_hi = max(p[1] + _ctx_jitter.get(p[0], 0.0) for p in ctx_pts)
    right_x_lo = max(1.0, ctx_lat_lo - 0.5)
    ax_b_right.set_xlim(right_x_lo, ctx_lat_hi + 0.4)

    # Tick layout: left sub-axis gets {0.0, 0.5}, right sub-axis gets
    # ~3 evenly spaced ticks across the visible cluster range so the
    # tick labels remain readable regardless of where the cluster sits.
    from matplotlib.ticker import FixedLocator as _FixedLocator
    from matplotlib.ticker import MaxNLocator as _MaxNLocator
    ax_b_left.xaxis.set_major_locator(_FixedLocator([0.0, 0.5]))
    ax_b_right.xaxis.set_major_locator(_MaxNLocator(nbins=3))

    # Y-label only on the left sub-axis. X-label and title placed
    # via fig-level helpers so they read as a single panel rather
    # than as two adjacent sub-axes.
    ax_b_left.set_ylabel("Mean ARI", fontsize=_F10_AXIS, fontweight="bold")

    # Apply tick / label fonts on both sub-axes (matched to fig 8).
    for _ax in (ax_b_left, ax_b_right):
        _apply_style(_ax)
        _ax.tick_params(labelsize=_F10_TICK, length=6, width=1.4)
        for lbl in list(_ax.get_xticklabels()) + list(_ax.get_yticklabels()):
            lbl.set_fontsize(_F10_TICK); lbl.set_fontweight("bold")
    ax_b_left.yaxis.label.set_size(_F10_AXIS)
    ax_b_left.yaxis.label.set_weight("bold")
    # Re-hide the inner spines after _apply_style restores defaults.
    ax_b_left.spines["right"].set_visible(False)
    ax_b_right.spines["left"].set_visible(False)
    ax_b_right.tick_params(left=False, labelleft=False)

    # tight_layout BEFORE the fig.text title/xlabel placements so
    # bbox.y0/y1 read the post-layout panel positions and the
    # labels land in their final coordinates instead of pre-layout
    # ones. rect_top widened from 0.94 to 0.86 (late-May 2026 user
    # request: "increase the gap between the main title and the 3
    # panels - to make the figure similar to other 3 panel plots
    # like fig 7"). With the broken-axis subgridspec inside panel
    # (b), tight_layout emits a "Axes that are not compatible with
    # tight_layout" warning and silently falls back to a partial
    # layout that does not always honour rect_top -- so the
    # subplots_adjust(top=...) below is the *forcing* mechanism that
    # actually carves out the suptitle-to-panel gap. The two calls
    # are belt-and-braces: tight_layout handles inter-panel padding
    # and panel-internal label positioning, subplots_adjust nails
    # the top margin numerically.
    fig.tight_layout(rect=[0, 0, 1, 0.86], w_pad=1.6)
    fig.subplots_adjust(top=0.86)

    # Compute the geometric center of each broken pair in figure
    # coordinates - both labels and title use this so they appear
    # centered over the (left + right) sub-axis pair rather than
    # tied to one sub-axis. Read AFTER tight_layout so positions
    # reflect the final layout.
    bbox_a_left = ax_a_left.get_position()
    bbox_a_right = ax_a_right.get_position()
    pair_a_x_center = (bbox_a_left.x0 + bbox_a_right.x1) / 2.0
    bbox_left = ax_b_left.get_position()
    bbox_right = ax_b_right.get_position()
    pair_x_center = (bbox_left.x0 + bbox_right.x1) / 2.0

    # Panel A title + x-label centered over the broken pair.
    fig.text(pair_a_x_center, bbox_a_left.y0 - 0.08,
             "Mean Decision Latency (ms)",
             ha="center", va="top",
             fontsize=_F10_AXIS, fontweight="bold")
    fig.text(pair_a_x_center, bbox_a_left.y1 + 0.005,
             "(a) Lightweight Methods",
             ha="center", va="bottom",
             fontsize=_F10_TITLE, fontweight="bold")

    # X-label centered under the broken pair, far enough below the
    # axes to clear the rotated tick labels.
    fig.text(pair_x_center, bbox_left.y0 - 0.08,
             "Mean Decision Latency (ms)",
             ha="center", va="top",
             fontsize=_F10_AXIS, fontweight="bold")
    # Panel (b) title centered over the broken pair. Offset reduced
    # from +0.025 to +0.005 alongside the rect_top bump to 0.985
    # so the title sits just above the panel's top spine, matching
    # the visual y position of panel A's set_title (pad=14 inside
    # the axes box).
    fig.text(pair_x_center, bbox_left.y1 + 0.005,
             "(b) Context-Aware Methods",
             ha="center", va="bottom",
             fontsize=_F10_TITLE, fontweight="bold")

    # Legend on the right sub-axis (lower center, lifted ~6 % off the
    # x-axis) - the handle list must combine entries from both
    # sub-axes since the No Context reference scatter lives on
    # ax_b_left and would otherwise be missing from the legend.
    leg_handles = []
    leg_labels = []
    for ax_ in (ax_b_left, ax_b_right):
        for h, lbl in zip(*ax_.get_legend_handles_labels()):
            if lbl not in leg_labels:
                leg_handles.append(h)
                leg_labels.append(lbl)
    _legend(ax_b_right, handles=leg_handles, labels=leg_labels,
            loc="lower center", bbox_to_anchor=(0.5, 0.06),
            ncol=2, fontsize=_F10_LEG)

    # =====================================================================
    # Panel (c) -- Delta-ARI vs No Context (per scenario, per mode)
    # =====================================================================
    # Bars show paired mean_diff (mode_ARI - no_context_ARI) per scenario
    # for the three context-aware modes. Whiskers are the 95 % paired
    # CIs from benchmark_significance.json (Wilcoxon signed-rank +
    # bootstrap CI on the paired-difference mean, the canonical paired
    # statistic the paper uses elsewhere). The panel directly answers
    # the reviewer question "what does the context channel buy?" --
    # absolute Delta-ARI ranges from +0.011 (heatwave) to +0.032 (baseline)
    # for AgriBrain, with all CIs strictly above zero, so the paired
    # tests are uniformly significant. Per-scenario reading: context
    # has the most leverage when the physics is least constraining
    # (baseline > adaptive_pricing > stress scenarios), which is the
    # opposite of the lay intuition that "context matters most under
    # stress" -- worth flagging in the discussion.
    sig_payload_path = RESULTS_DIR / "benchmark_significance.json"
    panel_c_data: dict = {}
    if sig_payload_path.exists():
        import json as _json_c
        sig_doc = _json_c.loads(sig_payload_path.read_text(encoding="utf-8"))
        sig_block = sig_doc.get("significance", sig_doc) if isinstance(sig_doc, dict) else {}
        for sc in SCENARIOS:
            sc_block = sig_block.get(sc, {})
            if not isinstance(sc_block, dict):
                continue
            sc_cells: dict = {}
            for mode in ("agribrain", "mcp_only", "pirag_only"):
                cmp = sc_block.get(f"{mode}_vs_no_context", {})
                if not isinstance(cmp, dict):
                    continue
                ari_block = cmp.get("ari", {})
                md = ari_block.get("mean_diff")
                if md is None:
                    continue
                sc_cells[mode] = {
                    "mean": float(md),
                    "ci_low": float(ari_block.get("mean_diff_ci_low", md)),
                    "ci_high": float(ari_block.get("mean_diff_ci_high", md)),
                }
            if sc_cells:
                panel_c_data[sc] = sc_cells

    _PANEL_C_MODES = [
        ("agribrain",  "AgriBrain",  COLORS.get("agribrain",  "#009688")),
        ("mcp_only",   "MCP Only",   COLORS.get("mcp_only",   "#F57C00")),
        ("pirag_only", "piRAG Only", COLORS.get("pirag_only", "#1565C0")),
    ]

    if panel_c_data:
        scenarios_in_c = [s for s in SCENARIOS if s in panel_c_data]
        n_groups = len(scenarios_in_c)
        n_modes = len(_PANEL_C_MODES)
        # Total group width 0.84; x_scale 1.20 so adjacent scenario
        # groups have a 0.36 inter-group gap. Each bar takes
        # 0.84/n_modes = 0.28 axis units when n_modes=3.
        width = 0.84 / n_modes
        x_scale = 1.20
        x_base = np.arange(n_groups) * x_scale

        max_height = 0.0
        for i, (mode, label, color) in enumerate(_PANEL_C_MODES):
            heights = []
            err_low = []
            err_high = []
            for sc in scenarios_in_c:
                cell = panel_c_data.get(sc, {}).get(mode, {})
                m = cell.get("mean", 0.0)
                heights.append(m)
                # Asymmetric whiskers from the paired CI bounds.
                err_low.append(max(0.0, m - cell.get("ci_low", m)))
                err_high.append(max(0.0, cell.get("ci_high", m) - m))
                if cell.get("ci_high", m) > max_height:
                    max_height = cell.get("ci_high", m)
            xs = x_base + i * width
            ax_c.bar(xs, heights, width, color=color,
                     alpha=0.92, edgecolor="white", linewidth=1.0,
                     label=label,
                     yerr=[err_low, err_high],
                     capsize=4,
                     error_kw={"linewidth": 1.4, "capthick": 1.4,
                               "ecolor": "#212121"})

        ax_c.axhline(0.0, color="#212121", linewidth=1.0, zorder=2)
        ax_c.set_xticks(x_base + (n_modes - 1) * width / 2)
        ax_c.set_xticklabels(
            [SCENARIO_LABELS.get(s, s) for s in scenarios_in_c],
            rotation=30, ha="right", rotation_mode="anchor",
        )
        ax_c.set_ylabel(r"$\Delta$ARI vs No Context",
                        fontsize=_F10_AXIS, fontweight="bold")
        # Legend at upper-left: the leftmost scenario groups have the
        # smallest bars (heatwave / overproduction), so the upper-left
        # corner is the cleanest space for the 3-entry legend.
        ax_c.legend(loc="upper left", fontsize=_F10_LEG,
                    ncol=1, framealpha=0.95, edgecolor="#757575",
                    fancybox=False, shadow=False)
        # ylim: bump headroom by 30 % over the highest bar+CI cap so
        # the legend has clean space.
        ax_c.set_ylim(0.0, max_height * 1.30 if max_height > 0 else 0.04)
    else:
        ax_c.text(0.5, 0.5, "benchmark_significance.json not available",
                  ha="center", va="center", transform=ax_c.transAxes,
                  fontsize=ANNOT_FONT_SIZE, color="#616161")

    _apply_style(ax_c)
    ax_c.title.set_size(_F10_TITLE)
    ax_c.title.set_weight("bold")
    ax_c.xaxis.label.set_size(_F10_AXIS)
    ax_c.xaxis.label.set_weight("bold")
    ax_c.yaxis.label.set_size(_F10_AXIS)
    ax_c.yaxis.label.set_weight("bold")
    ax_c.tick_params(labelsize=_F10_TICK, length=6, width=1.4)
    for lbl in list(ax_c.get_xticklabels()) + list(ax_c.get_yticklabels()):
        lbl.set_fontsize(_F10_TICK); lbl.set_fontweight("bold")
    ax_c.set_title(r"(c) $\Delta$ARI vs No Context",
                   fontsize=_F10_TITLE, fontweight="bold", pad=14)

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
    # Single consolidated Figure 9: fault-degradation + context honor +
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
