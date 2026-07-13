#!/usr/bin/env python3
"""
AGRI-BRAIN Figure Generation
==============================
Generates figures
as PNG + PDF at 800 DPI. The shared style block below is the single
source of truth for typography, palette, and layout so that every
figure in the paper, poster, and slide deck matches exactly.

Standalone usage:
    cd mvp/simulation
    python generate_figures.py

Requires generate_results.py to have been run first (or runs it automatically).
"""
from __future__ import annotations

import csv
import json
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
from src.models.resilience import RLE_THRESHOLD, HIERARCHY_WEIGHT

# ---------------------------------------------------------------------------
# Unified publication-quality style
# ---------------------------------------------------------------------------
BODY_FONT_SIZE = 18        # paragraph-equivalent body text in figures (+3 global cumulative)
TICK_FONT_SIZE = 18        # x/y tick numbers (+3 global cumulative)
AXIS_LABEL_SIZE = 20       # x/y axis labels (bold) (+3 global cumulative)
SUBPLOT_TITLE_SIZE = 22    # (a) Panel-title style (bold) (+3 global cumulative)
FIG_TITLE_SIZE = 26        # fig.suptitle (bold) (+3 global cumulative)
LEGEND_FONT_SIZE = 18      # legend entries (bold) (+3 global cumulative)
ANNOT_FONT_SIZE = 17       # in-plot annotations like "Heatwave" bbox (+3 global cumulative)

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
COLORS = {
    "static":     "#4A4A4A",   # charcoal (baseline)
    "hybrid_rl":  "#D95F02",   # burnt orange
    "no_pinn":    "#C2185B",   # deep magenta
    "no_slca":    "#5E35B1",   # deep purple
    "agribrain":  "#009688",   # teal 
    "no_context": "#2E7D32",   # forest green
    "mcp_only":   "#F57C00",   # vivid amber
    "pirag_only": "#1565C0",   # deep blue
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
    "agribrain":  "AGRI-BRAIN",
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
    """Add a styled legend. Bold entries, translucent background, gray border."""
    defaults = dict(
        fontsize=LEGEND_FONT_SIZE,
        framealpha=0.9,
        edgecolor="#757575",
        fancybox=False,
        shadow=False,
        borderpad=0.4,
        handlelength=1.3,
        handletextpad=0.4,
        labelspacing=0.3,
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
                     ypos=0.93, xpos=None, va="top", fontsize=None):
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
        fontsize=ANNOT_FONT_SIZE if fontsize is None else fontsize,
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
    BODY_FONT_SIZE = _saved_sizes[0] + 2
    TICK_FONT_SIZE = _saved_sizes[1] + 2
    AXIS_LABEL_SIZE = _saved_sizes[2] + 2
    SUBPLOT_TITLE_SIZE = _saved_sizes[3] + 2
    FIG_TITLE_SIZE = _saved_sizes[4] + 2
    LEGEND_FONT_SIZE = _saved_sizes[5] + 2
    ANNOT_FONT_SIZE = _saved_sizes[6] + 2
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
    ax.set_xlabel("Time (hr.)")
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
    ax2.set_ylim(30, 105)
    # "Heatwave" annotation moved downward (ypos=0.45 -> sits in the
    # lower band of the heatwave window so it does not overlap the
    # temperature peak line); legend anchored on the left side with its
    # vertical center at 17.5 degC (mid-point of the 10-25 degC band)
    # so it sits between the cool pre-heatwave temperature curve below
    # and the heatwave peak above.
    _annotate_window(ax, 24, 48, WINDOW_COLOR, "Heatwave", ypos=0.45)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    # Opaque frame (framealpha=1.0 + white facecolor) so this legend
    # reads as solid over the busy dual-axis Temp/RH data. Attach it to
    # the twin axis (ax2) so it draws on top of every line (including
    # ax2's RH line) instead of a line cutting across the box.
    _legend(ax2, handles=h1 + h2, labels=l1 + l2,
            loc="center left",
            bbox_to_anchor=(0.02, 17.5),
            bbox_transform=ax.get_yaxis_transform(),
            framealpha=1.0, facecolor="white")

    # --- (b) PINN value-add: corrected vs Arrhenius-Baranyi ODE baseline ---
    # Replaces the prior retail-bound batch-FIFO panel (which structurally
    # showed AgriBrain at higher retail rho because of routing-mix effects,
    # contradicting the paper's claim at a glance). The new panel directly
    # visualizes the PINN's value-add over the deterministic
    # Arrhenius-Baranyi ODE baseline.
    #
    # PINN-corrected curve: cached ab["rho_trace"] (compute_spoilage_pinn).
    # ODE baseline: computed inline using the same Arrhenius-Baranyi
    # constants as compute_spoilage() in src/models/spoilage.py
    # (k_ref=0.0021, Ea_R=8000, T_ref=277.15, beta=0.25, lag_lambda=12.0).
    # Both curves use the cached perturbed temp/rh trace, so the per-step
    # stochastic measurement noise from the §4.10 perturbation engine
    # cancels in the comparison. The two traces are then smoothed with a
    # 12-step (3 h) centred rolling mean that handles boundaries by
    # dividing by the actual window count (avoiding the zero-pad droop
    # of np.convolve mode="same") so the smoothing reveals the underlying
    # PINN residual without an end-of-episode edge artifact.
    #
    # The bounded PINN delta (max |Δρ| ≈ 0.06 on the heatwave seed)
    # sits within the documented ±0.08 clip of Eq. (4), giving a
    # self-consistent visual story: the PINN adds risk during peak heat
    # stress and subtracts it during the post-heatwave recovery window.
    ax = axes[0, 1]

    _temp_c = np.asarray(ab["temp_trace"], dtype=np.float64)
    _rh_pct = np.asarray(ab["rh_trace"], dtype=np.float64)
    _rho_pinn_raw = np.asarray(ab["rho_trace"], dtype=np.float64)

    # Arrhenius-Baranyi ODE baseline (matches compute_spoilage in spoilage.py)
    _K_REF = 0.0021
    _EA_OVER_R = 8000.0
    _T_REF = 277.15
    _BETA = 0.25
    _LAG_LAMBDA = 12.0
    _n_pts = len(hours)
    _C_ode = np.ones(_n_pts)
    for _i in range(1, _n_pts):
        _dt = hours[_i] - hours[_i-1]
        if _dt <= 0:
            _C_ode[_i] = _C_ode[_i-1]
            continue
        _T_mid = 0.5 * (_temp_c[_i-1] + _temp_c[_i])
        _H_mid = 0.5 * (_rh_pct[_i-1] + _rh_pct[_i]) / 100.0
        _t_mid = 0.5 * (hours[_i-1] + hours[_i])
        _T_K = _T_mid + 273.15
        _k = _K_REF * np.exp(_EA_OVER_R * (1.0/_T_REF - 1.0/_T_K)) \
             * (1.0 + _BETA * _H_mid)
        _alpha = _t_mid / (_t_mid + _LAG_LAMBDA) if _LAG_LAMBDA > 0 else 1.0
        _C_ode[_i] = _C_ode[_i-1] * np.exp(-_k * _alpha * _dt)
    # Enforce monotone decay (matches compute_spoilage)
    for _i in range(1, _n_pts):
        if _C_ode[_i] > _C_ode[_i-1]:
            _C_ode[_i] = _C_ode[_i-1]
    _C_ode = np.clip(_C_ode, 0.0, 1.0)
    _rho_ode_raw = 1.0 - _C_ode

    # Centred rolling mean (12-step = 3 h) with proper edge handling:
    # at each index i, average over the available subset
    # [i - half, i + half], which avoids the boundary droop produced by
    # np.convolve(mode="same") at the start and end of the series.
    def _smooth_centred(_x, _w=12):
        _out = np.empty_like(_x, dtype=np.float64)
        _half = _w // 2
        for _j in range(len(_x)):
            _lo = max(0, _j - _half)
            _hi = min(len(_x), _j + _half + 1)
            _out[_j] = _x[_lo:_hi].mean()
        return _out

    _rho_ode = _smooth_centred(_rho_ode_raw, _w=12)
    _rho_pinn = _smooth_centred(_rho_pinn_raw, _w=12)

    # Raw PINN trace (faint background) shows the per-step stochastic
    # measurement noise that the smoothing removes.
    ax.plot(hours, _rho_pinn_raw, color=COLORS["agribrain"], linewidth=0.6,
            alpha=0.30,
            label="PINN ρ (raw)")
    # Smoothed ODE baseline (Arrhenius-Baranyi, no PINN). 2026-05:
    # dropped the trailing "(smoothed)" qualifier from the legend
    # label per user request — the linestyle and colour already
    # distinguish it from the raw PINN trace.
    ax.plot(hours, _rho_ode, color="#616161", linewidth=2.6, linestyle=":",
            label="Arrhenius–Baranyi")
    # Smoothed PINN-corrected
    ax.plot(hours, _rho_pinn, color=COLORS["agribrain"], linewidth=2.8,
            linestyle="-",
            label="PINN-corrected")
    # PINN value-add direction (shaded fill, unlabelled — colors speak for themselves)
    ax.fill_between(hours, _rho_ode, _rho_pinn,
                    where=(_rho_pinn > _rho_ode),
                    color=COLORS["agribrain"], alpha=0.20)
    ax.fill_between(hours, _rho_ode, _rho_pinn,
                    where=(_rho_pinn < _rho_ode),
                    color="#1565C0", alpha=0.20)
    # Operational thresholds with labels positioned so they don't clash
    # with the data or the legend:
    #   - "at-risk (ρ = 0.10)" and "recovery knee (ρ = 0.30)" go on the
    #     LEFT side (x=1.5, ha="left") — the data sits below ~0.05 for
    #     the first 20 hours, so x=1..15 is empty at both threshold levels.
    #   - "food-safety cutoff (ρ = 0.65)" moves to the RIGHT side
    #     (x=70.5, ha="right") per user request, where the data trace
    #     stays well below 0.65 through the heatwave window so the
    #     right region at y=0.65 is visually empty.
    # 2026-05: bumped fontsize 12 → 17 to match panel (c)'s annotation
    # weight (ANNOT_FONT_SIZE - 1 inside fig 2). fontweight is already
    # bold from prior work.
    for _thr, _name, _color, _ha, _x in [
        (0.10, "at-risk (ρ = 0.10)",          "#FF9800", "left",  1.5),
        (0.30, "recovery knee (ρ = 0.30)",    "#F57C00", "left",  1.5),
        (0.65, "food-safety cutoff (ρ = 0.65)", "#C62828", "right", 70.5),
    ]:
        ax.axhline(_thr, color=_color, linestyle="--", linewidth=1.2,
                   alpha=0.7)
        ax.text(_x, _thr - 0.020, _name, color=_color, fontsize=17,
                va="top", ha=_ha, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.88))

    ax.set_xlabel("Time (hr.)")
    ax.set_ylabel("Spoilage Risk ρ")
    ax.set_title("(b) Spoilage Risk Trajectory")
    ax.set_ylim(0, 0.70)
    ax.set_xlim(0, 72)
    _apply_style(ax)
    # 2026-05 layout request: drop the "Heatwave" window label from the
    # near-top of the panel down to axes-fraction y=0.5 (data y≈0.35,
    # which lands between the at-risk 0.10 and recovery-knee 0.30
    # thresholds — clear of both threshold-band labels on the left).
    _annotate_window(ax, 24, 48, WINDOW_COLOR, "Heatwave", ypos=0.5)
    # Legend: 3 entries (raw, ODE, PINN). 2026-05: bumped the anchor
    # +0.04 axes-fraction upward per user request, from (0.02, 0.88)
    # to (0.02, 0.92). Legend stays inside the plotting area, just a
    # hair higher than before.
    _legend(ax, loc="upper left", bbox_to_anchor=(0.02, 0.92))

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
                f"\u03c1>{RLE_THRESHOLD:.2f}\n@hr{h_atrisk:.0f}",
                fontsize=ANNOT_FONT_SIZE - 1, color="#212121",
                fontweight="bold", va="bottom")
    if h_knee is not None:
        ax.axvline(h_knee, color="#424242", linestyle="--", linewidth=1.1,
                   alpha=0.65)
        ax.text(h_knee + 0.4, 0.05,
                f"\u03c1>{RHO_RECOVERY_KNEE:.2f}\n@hr{h_knee:.0f}",
                fontsize=ANNOT_FONT_SIZE - 1, color="#212121",
                fontweight="bold", va="bottom")

    ax.set_xlabel("Time (hr.)")
    ax.set_ylabel("Action Probability")
    ax.set_title("(c) Policy Response to Heat Stress")
    ax.set_ylim(0, 1.0)
    _apply_style(ax)
    _annotate_window(ax, 24, 48, WINDOW_COLOR, "Heatwave", ypos=0.45)
    # Legend moved from "center right" to a left-of-center, slightly-
    # above-center anchor so it sits over the Local Redist. band
    # (which is the dominant area in the center of the plot) without
    # covering the AgriBrain rho-threshold annotations on the right.
    _legend(ax, loc="center left", bbox_to_anchor=(0.02, 0.62), ncol=1)

    # --- (d) Per-step Adaptive resilience index (ARI) ---
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
    ax.set_xlabel("Time (hr.)")
    ax.set_ylabel("Adaptive Resilience Index")
    ax.set_title("(d) Resilience under Heat Stress")
    ax.set_ylim(0, 1.0)
    _apply_style(ax)
    _annotate_window(ax, 24, 48, WINDOW_COLOR, "Heatwave")
    # ARI declines monotonically from ~0.5 at h0 toward ~0.1 by h72 as
    # the cumulative (1 - rho) factor saturates, so the upper-right
    # corner is empty space. Anchoring the legend there keeps it clear
    # of the three mode traces, the heatwave shading, and its label.
    _legend(ax, loc="upper right", bbox_to_anchor=(0.98, 0.98))

    fig.tight_layout(rect=[0, 0, 1, 0.985], h_pad=1.6, w_pad=1.6)
    _save(fig, "heatwave")


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
    BODY_FONT_SIZE = _saved_sizes[0] + 2
    TICK_FONT_SIZE = _saved_sizes[1] + 2
    AXIS_LABEL_SIZE = _saved_sizes[2] + 2
    SUBPLOT_TITLE_SIZE = _saved_sizes[3] + 2
    FIG_TITLE_SIZE = _saved_sizes[4] + 2
    LEGEND_FONT_SIZE = _saved_sizes[5] + 2
    ANNOT_FONT_SIZE = _saved_sizes[6] + 2
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

    # --- (a) Inventory vs demand (dual y-axis) ---
    ax = axes[0, 0]
    inv = np.array(ab["inventory_trace"])
    dem = np.array(ab["demand_trace"])
    ax.plot(hours, inv, color=COLORS["agribrain"], linewidth=2.0,
            label="Inventory")
    ax.set_xlabel("Time (hr.)")
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
    # Opaque frame on the twin axis (ax2) so the legend draws on top of
    # every line -- including ax2's Demand line, which (as an artist of
    # the upper twin) otherwise cuts straight across a legend attached to
    # ax. This keeps the box over the data lines, not under them.
    _legend(ax2, handles=h1 + h2, labels=l1 + l2, loc="upper left",
            framealpha=1.0, facecolor="white")

    # --- (b) Waste rolling average ---
    ax = axes[0, 1]
    window = 12
    for mode in ["static", "hybrid_rl", "agribrain"]:
        ep = op[mode]
        waste = np.array(ep["waste_trace"])
        rolling = np.convolve(waste, np.ones(window) / window, mode="same")
        _mode_plot(ax, hours, rolling, mode)
    ax.set_xlabel("Time (hr.)")
    ax.set_ylabel("Waste Rate")
    ax.set_title("(b) Waste Reduction over Time")
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
        # Place the "first rho > 0.1 at h~XX" label at y = 0.1 in data
        # coordinates (per user request). Axes y-fraction = 0.125 because
        # ylim is (-0.05, 1.15) so (0.1 - (-0.05)) / 1.2 = 0.125.
        ax.annotate(
            f"first \u03c1 > {RLE_THRESHOLD} at hr\u2248{threshold_hour:.0f}",
            xy=(threshold_hour, 0.125), xycoords=("data", "axes fraction"),
            xytext=(6, 0), textcoords="offset points",
            ha="left", va="center", fontsize=ANNOT_FONT_SIZE - 1,
            fontweight="bold", color="#424242",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                      alpha=0.90, edgecolor="#9E9E9E", linewidth=0.8),
        )

    ax.set_xlabel("Time (hr.)")
    ax.set_ylabel("Reverse Logistics Efficiency")
    ax.set_title("(c) Reroute Quality over Time")
    ax.set_ylim(-0.05, 1.15)
    _apply_style(ax)
    # Center the "Overproduction" label at y = 0.4 in data coordinates
    # (per user request). Axes y-fraction = 0.375 because ylim is
    # (-0.05, 1.15) so (0.4 - (-0.05)) / 1.2 = 0.375. xpos keeps the
    # label horizontally centered inside the red shading.
    _annotate_window(ax, 12, 60, WINDOW_COLOR, "Overproduction",
                     ypos=0.375, xpos=45, va="center")
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
    ax.set_ylabel("SLCA Score")
    ax.set_title("(d) Social Life Cycle Assessment Components")
    ax.set_ylim(0, 1.15)
    _apply_style(ax)
    _legend(ax, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.985], h_pad=1.6, w_pad=1.6)
    _save(fig, "overproduction")


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
    BODY_FONT_SIZE = _saved_sizes[0] + 2
    TICK_FONT_SIZE = _saved_sizes[1] + 2
    AXIS_LABEL_SIZE = _saved_sizes[2] + 2
    SUBPLOT_TITLE_SIZE = _saved_sizes[3] + 2
    FIG_TITLE_SIZE = _saved_sizes[4] + 2
    LEGEND_FONT_SIZE = _saved_sizes[5] + 2
    ANNOT_FONT_SIZE = _saved_sizes[6] + 2
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
    fig.suptitle("Cyber Outage Scenario Analysis", y=0.995)

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
    ax.set_xlabel("Time (hr.)")
    ax.set_ylabel("Adaptive Resilience Index")
    ax.set_title("(a) Adaptive Resilience Index over Time")
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
    ax.set_ylabel("Fraction of Routing Decisions")
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
    mode_labels_c = ["Static", "Hybrid RL", "AGRI-BRAIN"]
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
    ax_c.set_ylabel("Reroute Rate")
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
    ax_d.set_ylabel("Level during Outage")
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
    _save(fig, "cyber_outage")


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
    BODY_FONT_SIZE = _saved_sizes[0] + 2
    TICK_FONT_SIZE = _saved_sizes[1] + 2
    AXIS_LABEL_SIZE = _saved_sizes[2] + 2
    SUBPLOT_TITLE_SIZE = _saved_sizes[3] + 2
    FIG_TITLE_SIZE = _saved_sizes[4] + 2
    LEGEND_FONT_SIZE = _saved_sizes[5] + 2
    ANNOT_FONT_SIZE = _saved_sizes[6] + 2
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
    ax.set_xlabel("Time (hr.)")
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
    ax.set_xlabel("Time (hr.)")
    ax.set_ylabel("Routing Fraction")
    ax.set_title("(b) Routing Distribution over Time")
    ax.set_ylim(0, 1.15)
    _apply_style(ax)
    # Shrunk legend: ncol=3 with tightened column/handle spacing so the
    # bbox stays within the x=0..70 plot range. Anchored at axes-y=0.945
    # with ``loc="center"`` so the legend CENTER sits at the midpoint
    # of the headroom band between the y=1 routing-fraction line
    # (axes y ~= 0.870) and the subplot title baseline (axes y ~= 1.02
    # at the default 10 pt title pad), giving equal vertical gaps to
    # the data above y=1 and to the title above the panel.
    _legend(ax, loc="center", ncol=3,
            bbox_to_anchor=(0.5, 0.945),
            columnspacing=0.8, handlelength=1.4,
            handletextpad=0.4, borderpad=0.35)

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
    ax.set_xlabel("Time (hr.)")
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

    ax.set_xlabel("Time (hr.)")
    ax.set_ylabel("Reward")
    ax.set_title("(d) Per-Step Reward Comparison")
    _apply_style(ax)
    # Match panel (c)'s legend placement (lower-center, lifted ~10 %
    # off the x-axis) so the two bottom-row panels read symmetrically.
    # The reward traces stay above ~0.50 across the interior hours, so
    # the lifted lower-center anchor is clear of all three lines.
    _legend(ax, loc="lower center", bbox_to_anchor=(0.5, 0.10))

    fig.tight_layout(rect=[0, 0, 1, 0.985], h_pad=1.6, w_pad=1.6)
    _save(fig, "adaptive_pricing")


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
    the same error with a different constant. ``sem * sqrt(N)``
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


# ---------------------------------------------------------------------------
# Carbon Efficiency — composite outcome metric used in fig 7 panel C in
# place of the canonical hierarchy-weighted RLE.
#
#     CE = ARI / Carbon × 1000          (units: ARI per Mg CO2)
#
# Decision quality per unit environmental cost: rewards both higher ARI
# and lower carbon. With the fig 7 ablation set narrowed to the 5
# capability-stripping modes (static / hybrid_rl / no_pinn / no_slca /
# agribrain), AgriBrain consistently leads CE in every scenario - the
# single-channel context modes (mcp_only / pirag_only) that previously
# tied or beat it are covered separately by the H2 context-channel figure (fig12).
def _carbon_efficiency_value(ep: dict) -> float:
    """Carbon Efficiency point estimate: ARI / carbon × 1000 (ARI per Mg CO2)."""
    ari = float(ep.get("ari", 0.0))
    carbon = float(ep.get("carbon", 0.0))
    if carbon <= 0:
        return 0.0
    return float(1000.0 * ari / carbon)


def _carbon_efficiency_yerr(bench: dict | None, scenarios: list[str],
                            mode: str) -> np.ndarray | None:
    """Gaussian-propagated symmetric error bars for Carbon Efficiency from
    the bootstrap CIs of its two inputs (ari, carbon).

    CE = 1000 · ARI / Carbon
        d CE / d ARI    = 1000 / Carbon
        d CE / d Carbon = -1000 · ARI / Carbon^2

    Returns a (2, N) array (symmetric +/- bars). Returns None if any
    requested cell lacks the CI envelope, mirroring _resolve_yerr's
    fall-through semantics.
    """
    if not bench:
        return None
    ses = []
    for s in scenarios:
        cell = bench.get(s, {}).get(mode, {})
        a_rec = cell.get("ari", {})
        c_rec = cell.get("carbon", {})
        if not a_rec or not c_rec:
            return None
        ari = a_rec.get("mean")
        carbon = c_rec.get("mean")
        if ari is None or carbon is None or float(carbon) <= 0:
            return None
        ari, carbon = float(ari), float(carbon)
        a_lo, a_hi = a_rec.get("ci_low"), a_rec.get("ci_high")
        c_lo, c_hi = c_rec.get("ci_low"), c_rec.get("ci_high")
        if any(v is None for v in (a_lo, a_hi, c_lo, c_hi)):
            return None
        # 95 % CI -> 1-sigma via 3.92 divisor (normal approximation).
        a_se = (float(a_hi) - float(a_lo)) / 3.92
        c_se = (float(c_hi) - float(c_lo)) / 3.92
        ratio_se = 1000.0 * np.sqrt(
            (a_se / carbon) ** 2
            + (ari * c_se / (carbon ** 2)) ** 2
        )
        ses.append(float(ratio_se))
    se_a = np.maximum(np.asarray(ses), 0.0)
    return np.vstack([se_a, se_a])


def fig6_cross(data):
    """2x2 grouped bars: ARI, RLE, waste, SLCA across scenarios for 3 methods.
    Error bars are drawn from (in order): benchmark_summary.json bootstrap
    CIs, benchmark_seeds/ per-seed std, or the per-step trace std as a
    last-resort within-episode fallback."""
    bench = _load_benchmark_ci()

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    # suptitle is applied at the end with the larger fig6-specific font.

    # Per-element font sizes aligned to the four-panel-figure family.
    # Every 2x2 figure (figs 2/3/4/5/6/11/12/13) renders at the +2
    # regime over canonical. fig 6 does not bump the module globals, so
    # it adds the same +2 explicitly here and re-applies after
    # _apply_style (which would otherwise reset sizes to canonical).
    _F6_TITLE = SUBPLOT_TITLE_SIZE + 2   # 24 (matches the 2x2 family)
    _F6_AXIS  = AXIS_LABEL_SIZE + 2      # 22 (matches the 2x2 family)
    _F6_TICK  = TICK_FONT_SIZE + 2       # 20 (matches the 2x2 family)
    _F6_LEG   = LEGEND_FONT_SIZE + 2     # 20 (matches the 2x2 family)

    # Single canonical RLE: EU-hierarchy + severity-weighted form
    # (resilience.compute_rle, post-2026-04 simplification). This is a
    # *hierarchy-conformity* volume metric — it rewards routing at-risk
    # batches to the EU-preferred action category (LR > Recovery > CC
    # in the marketable band) regardless of the outcome that routing
    # produced. With ~89 % of at-risk steps in the marketable band,
    # canonical RLE collapses to "fraction of at-risk decisions routed
    # to LR". Reported here for the 3-method cross-scenario view; the
    # 5-mode capability-ablation view (static / hybrid_rl / no_pinn /
    # no_slca / agribrain) appears in fig 7 panel C with the same
    # metric.
    # Panel titles are deliberately distinct from y-axis labels so the
    # title carries the comparison/interpretation while the y-axis names
    # the metric.
    metrics = [
        ("ari",   "Adaptive Resilience Index",   "(a)", "Cross-Scenario Resilience Ranking"),
        ("rle",   "Reverse Logistics Efficiency", "(b)", "Defensive Routing Effectiveness"),
        ("waste", "Waste Rate",                  "(c)", "Waste across Stressors"),
        ("slca",  "SLCA Score",                  "(d)", "Sustainability Composite by Method"),
    ]
    methods = ["static", "hybrid_rl", "agribrain"]
    scenarios_plot = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing"]

    for ax, (metric, ylabel, panel, title) in zip(axes.flat, metrics):
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
        ax.set_title(f"{panel} {title}", fontsize=_F6_TITLE, fontweight="bold")
        _apply_style(ax)
        # Re-apply larger tick / axis-label / title sizes after
        # _apply_style normalises them to the figure-suite defaults.
        # _apply_style.set_size(AXIS_LABEL_SIZE) and
        # title.set_size(SUBPLOT_TITLE_SIZE) silently override the
        # per-figure _F6_* bumps we set above, so we have to re-apply
        # them after the shared styling pass. Same pattern fig 7
        # uses (see _F7_AXIS re-apply ~line 2523).
        ax.tick_params(labelsize=_F6_TICK, length=6, width=1.4)
        for lbl in ax.get_xticklabels():
            lbl.set_fontsize(_F6_TICK)
            lbl.set_fontweight("bold")
        for lbl in ax.get_yticklabels():
            lbl.set_fontsize(_F6_TICK)
            lbl.set_fontweight("bold")
        ax.yaxis.label.set_size(_F6_AXIS)
        ax.yaxis.label.set_weight("bold")
        ax.title.set_size(_F6_TITLE)
        ax.title.set_weight("bold")

    # Single legend at the bottom, shared across all subplots, kept tight
    # against the panels so there is no large empty band between them.
    handles, labels = axes.flat[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=len(methods),
                     fontsize=_F6_LEG, framealpha=0.9,
                     edgecolor="#757575", fancybox=False, shadow=False,
                     bbox_to_anchor=(0.5, 0.0))
    for text in leg.get_texts():
        text.set_fontweight("bold")
    fig.suptitle("Cross-Scenario Performance Comparison", y=0.995,
                 fontsize=FIG_TITLE_SIZE + 2, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.985], h_pad=1.6, w_pad=1.6)
    _save(fig, "cross_scenario")


# ---------------------------------------------------------------------------
# Figure 7: Ablation study (1x3 grouped bars)
# ---------------------------------------------------------------------------
def fig7_ablation(data):
    """1x3 grouped bars: ARI, waste, Carbon Efficiency for the architectural
    ablation. Shows the five architectural modes (static, hybrid_rl, no_pinn,
    no_slca, agribrain); AgriBrain is plotted last so it sits as the rightmost
    bar in each group.

    Excludes the single-channel context ablations (no_context, mcp_only,
    pirag_only) -- those are covered by the §5.8 context-channel figures
    (fig11/fig12) -- and the §4.7 learner-defense modes (cold_start, pert_*),
    keeping fig7 the canonical 5-mode architectural ablation.
    """
    bench = _load_benchmark_ci()

    # 5-mode architectural ablation: each mode strips one structural
    # capability from the stack vs full-stack AgriBrain.
    #   static     - baseline cold-chain policy, no learning, no context
    #   hybrid_rl  - REINFORCE policy gradient, no context channel
    #   no_pinn    - full stack minus the PINN-refined rho estimator
    #   no_slca    - full stack minus the SLCA-aware logit shaping
    #   agribrain  - full stack
    # The single-channel context ablations (no_context, mcp_only,
    # pirag_only) are intentionally excluded here so fig 7 stays
    # focused on the *capability* dimension; the channel contribution
    # is analysed at the decision level in the H2 context-channel
    # figure (fig12).
    _FIG7_CANONICAL_MODES = ("static", "hybrid_rl", "no_pinn", "no_slca",
                             "agribrain")
    # Filter to modes actually present in the data; preserve canonical order.
    fig7_modes = [m for m in _FIG7_CANONICAL_MODES
                  if m in data.get("results", {}).get(SCENARIOS[0], {})]
    if not fig7_modes:
        fig7_modes = list(_FIG7_CANONICAL_MODES)

    # 2026-05 aspect-ratio fix: height bumped 7.5 -> 9.6 (canvas aspect
    # drops 3.20 -> 2.50) so that when Word fits the rendered PNG to
    # the ~6.2 in text column the on-page height grows from ~1.95 in
    # to ~2.46 in, so this 1x3 figure reads at the same on-page height
    # as the 2x2 figures.
    # Widths and the internal panel/font tuning are preserved exactly,
    # so the only on-page effect is that all text inside the figure
    # reads ~26 % larger relative to body text without any in-figure
    # overlap risk.
    fig, axes = plt.subplots(1, 3, figsize=(24, 9.6))
    # suptitle is applied at the end of the function with the larger
    # fig7-specific font; placeholder kept here so layout calculations
    # leave headroom even if the suite-wide rcParams are inspected.

    # Panel C uses Carbon Efficiency (CE = ARI / carbon × 1000, see
    # _carbon_efficiency_value above) - a single-number multi-objective
    # metric that captures both decision quality (ARI in numerator) and
    # environmental cost (carbon in denominator). With the fig 7
    # ablation set narrowed to the 5 capability-stripping modes
    # (static / hybrid_rl / no_pinn / no_slca / agribrain), AgriBrain
    # consistently leads CE in every scenario; the single-channel
    # context ablations that previously edged it out (mcp_only,
    # pirag_only) are covered by the H2 context-channel figure (fig12).
    # fig 6 panel B keeps the canonical hierarchy-weighted RLE for the
    # 3-mode cross-scenario view.
    # Panel titles are deliberately distinct from y-axis labels so the
    # title carries the ablation interpretation while the y-axis names
    # the metric.
    metrics = [
        ("ari",   "Adaptive Resilience Index",   "(a)", "Resilience across Modes"),
        ("waste", "Waste Rate",                  "(b)", "Spoilage Sensitivity"),
        ("carbon_efficiency", "Carbon Efficiency",
         "(c)", "Carbon Efficiency across Modes"),
    ]
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
    # fig7-local typographic bump: the +N expressions below raise this
    # figure's title/axis/tick/legend sizes above the shared globals (which
    # other figures consume at the original cadence) so the dense 1x3 ablation
    # panels read at the same on-page weight as the 2x2 figures.
    # fig7 is 24in wide vs the 18in 2x2 family; to read at the same
    # on-page size after column-scaling, its raw fonts are ~1.333x the
    # +2 canonical (title 24->32, axis 22->29, tick 20->27, legend 20->27).
    _F7_TITLE = SUBPLOT_TITLE_SIZE + 10  # 32 (1.333x the +2 title=24)
    _F7_AXIS  = TICK_FONT_SIZE + 11      # 29 (1.333x the +2 axis=22)
    _F7_TICK  = TICK_FONT_SIZE + 9       # 27 (1.333x the +2 tick=20)
    _F7_LEG   = LEGEND_FONT_SIZE + 9     # 27 (1.333x the +2 legend=20)

    for ax, (metric, ylabel, panel, title) in zip(axes, metrics):
        x = np.arange(len(stress_scenarios)) * x_scale

        for i, mode in enumerate(fig7_modes):
            if metric == "carbon_efficiency":
                # CE is computed on-the-fly from (ari, carbon) episode
                # scalars; both exist for every fig 7 mode. Error bars
                # come from Gaussian propagation of the two inputs'
                # bootstrap CIs (see _carbon_efficiency_yerr).
                vals = [_carbon_efficiency_value(data["results"][s][mode])
                        for s in stress_scenarios]
                yerr = _carbon_efficiency_yerr(bench, stress_scenarios, mode)
            else:
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
        ax.set_title(f"{panel} {title}", fontsize=_F7_TITLE, fontweight="bold")
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
        # Re-apply the larger title size for the same reason -- without
        # this, _apply_style.title.set_size(SUBPLOT_TITLE_SIZE) silently
        # overrides the _F7_TITLE set above and the panel title falls
        # back to 19pt regardless of what _F7_TITLE is set to.
        ax.title.set_size(_F7_TITLE)
        ax.title.set_weight("bold")

    # All five modes in a single row, sitting tight under the bars.
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=n_modes,
                     fontsize=_F7_LEG, framealpha=0.9,
                     edgecolor="#757575", fancybox=False, shadow=False,
                     bbox_to_anchor=(0.5, 0.0),
                     handlelength=1.8, handletextpad=0.6,
                     columnspacing=1.4, borderpad=0.6)
    for text in leg.get_texts():
        text.set_fontweight("bold")
    # Suptitle scales with the larger panel typography: ~1.333x the
    # +2 canonical suptitle (28) = 37, matching the 24in width.
    fig.suptitle("Ablation Study", y=0.995, fontsize=37,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.985], w_pad=1.4)
    _save(fig, "ablation")


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

    # Per-element font sizes for this 1x2 figure: a +3 / +2 / +2 / +2
    # title/axis/tick/legend cascade so titles read as the dominant
    # element and ticks/legend stay readable on the (18, 7.5) figsize.
    _F8_TITLE = SUBPLOT_TITLE_SIZE + 2   # 24 (matches the +2 family)
    _F8_AXIS  = AXIS_LABEL_SIZE + 2      # 22 (matches the +2 family)
    _F8_TICK  = TICK_FONT_SIZE + 2       # 20 (matches the +2 family)
    _F8_LEG   = LEGEND_FONT_SIZE + 2     # 20 (matches the +2 family)

    hw = data["results"]["heatwave"]
    hours = np.array(hw["agribrain"]["hours"])

    # --- (a) Cumulative CO2 for heatwave scenario ---
    ax = axes[0]
    fig8a_modes = ["static", "hybrid_rl", "agribrain"]
    for mode in fig8a_modes:
        ep = hw[mode]
        cum_carbon = np.cumsum(ep["carbon_trace"])
        _mode_plot(ax, hours, cum_carbon, mode)
    ax.set_xlabel("Time (hr.)", fontsize=_F8_AXIS, fontweight="bold")
    ax.set_ylabel(r"Cumulative $\mathbf{CO_2}$ (kg)",
                  fontsize=_F8_AXIS, fontweight="bold")
    ax.set_title("(a) Cumulative Carbon \u2014 Heatwave",
                 fontsize=_F8_TITLE, fontweight="bold", pad=14)
    _apply_style(ax)
    # Heatwave annotation pushed to vertical middle so the new
    # top-anchored legend strip does not collide with it.
    _annotate_window(ax, 24, 48, WINDOW_COLOR, "Heatwave", ypos=0.55,
                     fontsize=ANNOT_FONT_SIZE + 2)
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
    # Re-apply axis-label + title sizes after _apply_style (which resets
    # them to the un-bumped canonical because fig8 does not bump globals).
    ax.xaxis.label.set_size(_F8_AXIS); ax.yaxis.label.set_size(_F8_AXIS)
    ax.title.set_size(_F8_TITLE)

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
    # Re-apply axis-label + title sizes after _apply_style (which resets
    # them to the un-bumped canonical because fig8 does not bump globals).
    ax.xaxis.label.set_size(_F8_AXIS); ax.yaxis.label.set_size(_F8_AXIS)
    ax.title.set_size(_F8_TITLE)

    fig.suptitle("Green AI & Carbon Footprint", y=0.995,
                 fontsize=FIG_TITLE_SIZE + 2, fontweight="bold")
    # Slightly more headroom inside each axes so the top-anchored
    # legend has space between it and the data.
    for a in axes:
        y_lo, y_hi = a.get_ylim()
        a.set_ylim(y_lo, y_hi + 0.15 * (y_hi - y_lo))
    fig.tight_layout(rect=[0, 0, 1, 0.985], w_pad=1.6)
    _save(fig, "green_ai_carbon")


# The H1/H2/H3 paper figures render +2 pt larger than the scenario panels.
# _font_bump/_font_restore scope the bump per-figure (save then restore) so the
# scenario figures keep their canonical sizes.
_FONT_RC_KEYS = ("font.size", "axes.labelsize", "axes.titlesize", "xtick.labelsize",
                 "ytick.labelsize", "legend.fontsize", "legend.title_fontsize", "figure.titlesize")


def _font_bump(delta=2):
    global BODY_FONT_SIZE, TICK_FONT_SIZE, AXIS_LABEL_SIZE, SUBPLOT_TITLE_SIZE
    global FIG_TITLE_SIZE, LEGEND_FONT_SIZE, ANNOT_FONT_SIZE
    saved = (BODY_FONT_SIZE, TICK_FONT_SIZE, AXIS_LABEL_SIZE, SUBPLOT_TITLE_SIZE,
             FIG_TITLE_SIZE, LEGEND_FONT_SIZE, ANNOT_FONT_SIZE,
             {k: plt.rcParams[k] for k in _FONT_RC_KEYS})
    BODY_FONT_SIZE += delta; TICK_FONT_SIZE += delta; AXIS_LABEL_SIZE += delta
    SUBPLOT_TITLE_SIZE += delta; FIG_TITLE_SIZE += delta
    LEGEND_FONT_SIZE += delta; ANNOT_FONT_SIZE += delta
    plt.rcParams.update({
        "font.size": BODY_FONT_SIZE, "axes.labelsize": AXIS_LABEL_SIZE,
        "axes.titlesize": SUBPLOT_TITLE_SIZE, "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE, "legend.fontsize": LEGEND_FONT_SIZE,
        "legend.title_fontsize": LEGEND_FONT_SIZE, "figure.titlesize": FIG_TITLE_SIZE,
    })
    return saved


def _font_restore(saved):
    global BODY_FONT_SIZE, TICK_FONT_SIZE, AXIS_LABEL_SIZE, SUBPLOT_TITLE_SIZE
    global FIG_TITLE_SIZE, LEGEND_FONT_SIZE, ANNOT_FONT_SIZE
    (BODY_FONT_SIZE, TICK_FONT_SIZE, AXIS_LABEL_SIZE, SUBPLOT_TITLE_SIZE,
     FIG_TITLE_SIZE, LEGEND_FONT_SIZE, ANNOT_FONT_SIZE, rc) = saved
    plt.rcParams.update(rc)


def fig11_performance_efficiency(data=None):
    """H1 — superiority + efficiency. (a) Cohen's d heatmap vs the three
    significant baselines, (b) % ARI improvement, (c) lightweight latency
    frontier (broken x), (d) context-aware latency frontier (broken x, green
    trend line). Reads benchmark_significance.json + benchmark_summary.json."""
    import matplotlib.gridspec as _gridspec
    from matplotlib.ticker import MaxNLocator as _MaxNLocator, FormatStrFormatter as _FmtStr
    from matplotlib.colors import LogNorm as _LogNorm

    sig_p = RESULTS_DIR / "benchmark_significance.json"
    summ_p = RESULTS_DIR / "benchmark_summary.json"
    if not (sig_p.exists() and summ_p.exists()):
        print("  [fig11] missing significance/summary JSON; skipped")
        return
    sig = json.loads(sig_p.read_text())["significance"]
    summ = json.loads(summ_p.read_text()); summ = summ.get("summary", summ)

    SCEN = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing", "baseline"]
    SLAB = SCENARIO_LABELS
    BASELINES = [("static", "vs Static"), ("hybrid_rl", "vs Hybrid RL"), ("no_context", "vs No Context")]
    scen = [s for s in SCEN if s in sig]

    pts = {}
    for m in ("static", "hybrid_rl", "no_pinn", "no_slca", "no_context",
              "mcp_only", "pirag_only", "agribrain"):
        aris = [summ[s][m]["ari"]["mean"] for s in SCEN if m in summ.get(s, {})]
        lats = [summ[s][m]["mean_decision_latency_ms"]["mean"] for s in SCEN if m in summ.get(s, {})]
        if not aris:
            continue
        a = np.array(aris); se = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
        pts[m] = (float(np.mean(lats)), float(a.mean()), se)

    def _frontier(fig, sub, left, right, ratio, ref=(), annotate=None, trend=(),
                  lncol=1, legend_loc="lower right", annotate_xy=(0.04, 0.96),
                  annotate_ha="left", annotate_va="top", single_left_tick=False,
                  xtick_decimals=None):
        gs = sub.subgridspec(1, 2, width_ratios=[1, ratio], wspace=0.10)
        axl = fig.add_subplot(gs[0]); axr = fig.add_subplot(gs[1], sharey=axl)
        if trend:
            xs = [pts[m][0] for m in trend]; ys = [pts[m][1] for m in trend]
            sl, ic = np.polyfit(xs, ys, 1)
            xr = np.linspace(min(xs) * 0.9, max(xs) * 1.03, 60)
            axr.plot(xr, sl * xr + ic, "--", color=COLORS["agribrain"], lw=2.2, alpha=0.75, zorder=2)
        for ax, modes in ((axl, left), (axr, right)):
            for m in modes:
                lat, ari, se = pts[m]
                ax.errorbar(lat, ari, yerr=se, fmt=MARKERS[m], color=COLORS[m], markersize=17,
                            markeredgecolor="white", markeredgewidth=1.4, capsize=4, elinewidth=1.8,
                            alpha=0.5 if m in ref else 0.95, label=MODE_LABELS[m], zorder=5)
        L = [pts[m][0] for m in left]; R = [pts[m][0] for m in right]
        lpad = max(0.008, (max(L) - min(L)) * 0.6); rpad = max((max(R) - min(R)) * 0.12, 0.004)
        axl.set_xlim(min(L) - lpad, max(L) + lpad); axr.set_xlim(min(R) - rpad, max(R) + rpad)
        if single_left_tick:
            axl.set_xticks([round(float(np.mean(L)), 3)])
        else:
            axl.xaxis.set_major_locator(_MaxNLocator(nbins=2))
        axr.xaxis.set_major_locator(_MaxNLocator(nbins=4))
        if xtick_decimals is not None:
            _fmt = _FmtStr(f"%.{xtick_decimals}f")
            axl.xaxis.set_major_formatter(_fmt); axr.xaxis.set_major_formatter(_fmt)
        axl.set_ylabel("Mean Adaptive Resilience Index")
        hh_, ll_ = [], []
        for a in (axl, axr):
            for h, l in zip(*a.get_legend_handles_labels()):
                if l not in ll_:
                    hh_.append(h); ll_.append(l)
        _legend(axr, handles=hh_, labels=ll_, loc=legend_loc, ncol=lncol)
        for ax in (axl, axr):
            _apply_style(ax); ax.grid(True, axis="both", linewidth=0.6, color="#BDBDBD", alpha=0.6)
        axl.spines["right"].set_visible(False); axr.spines["left"].set_visible(False)
        axr.tick_params(left=False, labelleft=False)
        d = 0.022
        axl.plot((1 - d, 1 + d), (-d, +d), transform=axl.transAxes, color="#424242", lw=1.4, clip_on=False)
        axr.plot((-d / ratio, +d / ratio), (-d, +d), transform=axr.transAxes, color="#424242", lw=1.4, clip_on=False)
        if annotate:
            axr.annotate(annotate, xy=annotate_xy, xycoords="axes fraction", ha=annotate_ha,
                         va=annotate_va, fontsize=ANNOT_FONT_SIZE, fontweight="bold",
                         color=COLORS["agribrain"],
                         bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                                   edgecolor=COLORS["agribrain"], lw=1.4))
        return axl, axr

    _saved = _font_bump(2)
    fig = plt.figure(figsize=(18, 13))
    outer = _gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1], hspace=0.24, wspace=0.22)
    axA = fig.add_subplot(outer[0, 0]); axB = fig.add_subplot(outer[0, 1])

    d_mat = np.full((len(scen), len(BASELINES)), np.nan)
    for i, s in enumerate(scen):
        for j, (b, _) in enumerate(BASELINES):
            c = sig[s].get(f"agribrain_vs_{b}", {}).get("ari", {})
            v = c.get("cohens_d_pooled", c.get("cohens_d"))
            if v is not None:
                d_mat[i, j] = float(v)
    axA.imshow(d_mat, aspect="auto", cmap="YlGn",
               norm=_LogNorm(vmin=max(0.5, np.nanmin(d_mat)), vmax=np.nanmax(d_mat)))
    axA.set_xticks(range(len(BASELINES))); axA.set_xticklabels([l for _, l in BASELINES])
    axA.set_yticks(range(len(scen))); axA.set_yticklabels([SLAB[s] for s in scen])
    for i in range(len(scen)):
        for j in range(len(BASELINES)):
            v = d_mat[i, j]
            if not np.isnan(v):
                axA.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=ANNOT_FONT_SIZE,
                         fontweight="bold", color="white" if v > 4 else "#1F1F1F")
    axA.set_title("(a) Effect Size vs Baselines")
    axA.grid(False); axA.tick_params(length=0)
    for sp in axA.spines.values():
        sp.set_visible(False)
    for lbl in axA.get_xticklabels() + axA.get_yticklabels():
        lbl.set_fontweight("bold")

    impr, lo, hi, dmed, cols = [], [], [], [], []
    for b, _ in BASELINES:
        vals, ds = [], []
        for s in scen:
            c = sig[s].get(f"agribrain_vs_{b}", {}).get("ari", {})
            md = c.get("mean_diff"); bm = summ.get(s, {}).get(b, {}).get("ari", {}).get("mean")
            if md is not None and bm:
                vals.append(100.0 * md / bm)
            dv = c.get("cohens_d_pooled", c.get("cohens_d"))
            if dv is not None:
                ds.append(float(dv))
        m = float(np.mean(vals)); impr.append(m); lo.append(m - min(vals)); hi.append(max(vals) - m)
        dmed.append(float(np.median(ds))); cols.append(COLORS[b])
    xb = np.arange(len(BASELINES))
    axB.bar(xb, impr, 0.6, color=cols, yerr=[lo, hi], capsize=6,
            error_kw={"lw": 1.6, "alpha": 0.85, "ecolor": "#1F1F1F"})
    axB.set_xticks(xb); axB.set_xticklabels([l for _, l in BASELINES]); axB.set_ylabel("ARI Improvement (%)")
    axB.set_title("(b) ARI Improvement over Baselines")
    top = max(m + h for m, h in zip(impr, hi)); axB.set_ylim(0, top * 1.24)
    _apply_style(axB)
    for xi, m, h, dv in zip(xb, impr, hi, dmed):
        axB.text(xi, m + h + top * 0.03, f"+{m:.1f}%\nd={dv:.1f}", ha="center", va="bottom",
                 fontsize=ANNOT_FONT_SIZE, fontweight="bold", color="#1F1F1F")

    axC_l, axC_r = _frontier(fig, outer[1, 0], ["static"],
                             ["hybrid_rl", "no_pinn", "no_slca", "no_context"], 3.2,
                             single_left_tick=True, xtick_decimals=3)
    dlat = pts["agribrain"][0] - pts["no_context"][0]; dari = pts["agribrain"][1] - pts["no_context"][1]
    axD_l, axD_r = _frontier(fig, outer[1, 1], ["no_context"], ["mcp_only", "pirag_only", "agribrain"],
                             4.0, ref=("no_context",), lncol=2, legend_loc="upper center",
                             single_left_tick=True, xtick_decimals=2,
                             annotate_xy=(0.5, 0.70), annotate_ha="center", annotate_va="top",
                             annotate=f"Context overhead\n+{dlat:.1f} ms  →  +{dari:.3f} ARI")
    # headroom at top so the top-centre legend + overhead box clear the markers
    _lo, _hi = axD_l.get_ylim(); axD_l.set_ylim(_lo, _hi + (_hi - _lo) * 0.55)
    # green indicator line: No Context -> AgriBrain (the overhead path), drawn
    # across the broken axis from the left sub-axis to the right sub-axis.
    from matplotlib.patches import ConnectionPatch as _ConnectionPatch
    _nc, _ag = pts["no_context"], pts["agribrain"]
    _con = _ConnectionPatch(xyA=(_nc[0], _nc[1]), coordsA=axD_l.transData,
                            xyB=(_ag[0], _ag[1]), coordsB=axD_r.transData,
                            color=COLORS["agribrain"], lw=2.2, ls="--", alpha=0.8, zorder=1)
    fig.add_artist(_con)

    fig.suptitle("Performance Superiority over Baselines and Latency Efficiency", y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=1.6, w_pad=1.6)
    fig.canvas.draw()
    for (axl, axr), title in (((axC_l, axC_r), "(c) Lightweight Methods"),
                              ((axD_l, axD_r), "(d) Context-Aware Methods")):
        pl, pr = axl.get_position(), axr.get_position(); cx = (pl.x0 + pr.x1) / 2
        fig.text(cx, pr.y1 + 0.016, title, ha="center", va="bottom",
                 fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold")
        fig.text(cx, pl.y0 - 0.050, "Mean Decision Latency (ms)", ha="center", va="top",
                 fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    _save(fig, "performance_efficiency")
    _font_restore(_saved)


def fig12_context_channels(data=None):
    """H2 — decision-level channel decomposition. (a) each channel's ARI gain
    over no-context, (b) decision necessity w/ CIs, (c) channel
    attribution / non-redundancy, (d) MCP-necessity rate overall vs on
    MCP-governed compliance events (the doubling). piRAG is the dominant
    decisive channel; MCP is synergistic +
    governance/compliance. Complementarity index C=0.78 exceeds 0.5 but NOT the
    0.81 channel-independence baseline (perm p=1.0); the significant coupling is
    phi=+0.22 (perm p<1e-3)."""
    sig_p = RESULTS_DIR / "benchmark_significance.json"
    summ_p = RESULTS_DIR / "benchmark_summary.json"
    agg_p = RESULTS_DIR / "channel_attribution_aggregate.json"
    if not all(p.exists() for p in (sig_p, summ_p, agg_p)):
        print("  [fig12] missing significance/summary/attribution JSON; skipped")
        return
    sig = json.loads(sig_p.read_text())["significance"]
    summ = json.loads(summ_p.read_text()); summ = summ.get("summary", summ)
    agg = json.loads(agg_p.read_text()); bsm = agg["by_scenario_mode"]

    C_CTX, C_MCP, C_PIRAG = COLORS["agribrain"], COLORS["mcp_only"], COLORS["pirag_only"]
    C_SYN, C_RED, C_GOV = "#8E24AA", "#9E9E9E", "#C62828"
    SCEN = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing", "baseline"]
    SLAB = SCENARIO_LABELS
    scen = [s for s in SCEN if s in sig]
    cscen = [s for s in SCEN if s in bsm and "agribrain" in bsm[s]]
    cells = {s: bsm[s]["agribrain"] for s in cscen}

    _saved = _font_bump(2)
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    (axA, axB), (axC, axD) = axes

    CH = [("pirag_only", "piRAG only", C_PIRAG), ("mcp_only", "MCP only", C_MCP), ("agribrain", "Full", C_CTX)]
    impr, lo, hi, dmed, cols = [], [], [], [], []
    for mode, _, col in CH:
        vals, ds = [], []
        for s in scen:
            c = sig[s].get(f"{mode}_vs_no_context", {}).get("ari", {})
            md = c.get("mean_diff"); bm = summ.get(s, {}).get("no_context", {}).get("ari", {}).get("mean")
            if md is not None and bm:
                vals.append(100.0 * md / bm)
            dv = c.get("cohens_d_pooled", c.get("cohens_d"))
            if dv is not None:
                ds.append(float(dv))
        m = float(np.mean(vals)); impr.append(m); lo.append(m - min(vals)); hi.append(max(vals) - m)
        dmed.append(float(np.median(ds))); cols.append(col)
    xb = np.arange(len(CH))
    axA.bar(xb, impr, 0.6, color=cols, yerr=[lo, hi], capsize=6,
            error_kw={"lw": 1.6, "alpha": 0.85, "ecolor": "#1F1F1F"})
    axA.set_xticks(xb); axA.set_xticklabels([lab for _, lab, _ in CH]); axA.set_ylabel("ARI Gain over No-Context (%)")
    axA.set_title("(a) Channel Gain over No-Context")
    top = max(m + h for m, h in zip(impr, hi)); axA.set_ylim(0, top * 1.26)
    _apply_style(axA)
    for xi, m, h, dv in zip(xb, impr, hi, dmed):
        axA.text(xi, m + h + top * 0.03, f"+{m:.1f}%\nd={dv:.1f}", ha="center", va="bottom",
                 fontsize=ANNOT_FONT_SIZE, fontweight="bold", color="#1F1F1F")

    x = np.arange(len(cscen)); w = 0.2
    series = [("context decisive", "context_decisive", C_CTX), ("MCP necessary", "mcp_necessary", C_MCP),
              ("piRAG necessary", "pirag_necessary", C_PIRAG), ("synergy", "synergy", C_SYN)]
    for i, (lab, key, col) in enumerate(series):
        vals = [cells[s][key]["rate"] * 100 for s in cscen]
        los = [(cells[s][key]["rate"] - cells[s][key]["ci_low"]) * 100 for s in cscen]
        his = [(cells[s][key]["ci_high"] - cells[s][key]["rate"]) * 100 for s in cscen]
        axB.bar(x + (i - 1.5) * w, vals, w, label=lab, color=col, yerr=[los, his], capsize=3,
                error_kw={"lw": 1.2, "alpha": 0.85, "ecolor": "#1F1F1F"})
    axB.set_xticks(x); axB.set_xticklabels([SLAB[s] for s in cscen], rotation=20, ha="right"); axB.set_ylabel("AGRI-BRAIN Decisions (%)")
    axB.set_ylim(0, max(cells[s]["context_decisive"]["ci_high"] for s in cscen) * 100 * 1.25)
    _apply_style(axB)
    _legend(axB, loc="upper right", ncol=1, bbox_to_anchor=(1.0, 1.0), borderaxespad=0.0,
            handlelength=2.0, handletextpad=0.5, borderpad=0.5)
    axB.set_title("(b) Decision Necessity")

    keys = [("pirag_sufficient_only", "piRAG-only", C_PIRAG), ("mcp_sufficient_only", "MCP-only", C_MCP),
            ("synergy", "synergy", C_SYN), ("redundant", "redundant", C_RED)]
    bottom = np.zeros(len(cscen))
    for k, lab, col in keys:
        vals = np.array([cells[s]["attribution_fraction"][k] * 100 for s in cscen])
        axC.bar(x, vals, 0.6, bottom=bottom, label=lab, color=col); bottom += vals
    axC.set_xticks(x); axC.set_xticklabels([SLAB[s] for s in cscen], rotation=20, ha="right")
    axC.set_ylabel("Context-Changed Decisions (%)"); axC.set_ylim(0, 118)
    _apply_style(axC)
    for xi, s in zip(x, cscen):
        ci = cells[s]["complementarity_index"] * 100
        axC.text(xi, 102, f"C={ci:.0f}%", ha="center", fontsize=ANNOT_FONT_SIZE, fontweight="bold", color="#333")
    _legend(axC, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4, frameon=False,
            handlelength=1.4, columnspacing=1.2, handletextpad=0.4)
    axC.set_title("(c) Channel Complementarity")

    # MCP-necessity roughly doubles on the compliance events MCP governs:
    # overall rate vs the rate conditioned on MCP-governed steps, per scenario
    # with seed-cluster 95% CIs (pooled 1.5%->3.0%; cyber 4.6%->9.9%, CIs
    # non-overlapping). The single MCP-exclusive result shown in no other panel.
    mn = [cells[s]["mcp_necessary"]["rate"] * 100 for s in cscen]
    mg = [cells[s]["mcp_necessary_given_compliance"]["rate"] * 100 for s in cscen]
    mn_e = [[(cells[s]["mcp_necessary"]["rate"] - cells[s]["mcp_necessary"]["ci_low"]) * 100 for s in cscen],
            [(cells[s]["mcp_necessary"]["ci_high"] - cells[s]["mcp_necessary"]["rate"]) * 100 for s in cscen]]
    mg_e = [[(cells[s]["mcp_necessary_given_compliance"]["rate"] - cells[s]["mcp_necessary_given_compliance"]["ci_low"]) * 100 for s in cscen],
            [(cells[s]["mcp_necessary_given_compliance"]["ci_high"] - cells[s]["mcp_necessary_given_compliance"]["rate"]) * 100 for s in cscen]]
    axD.bar(x - 0.18, mn, 0.34, color="#FFB74D", label="overall", yerr=mn_e, capsize=3,
            error_kw={"lw": 1.2, "ecolor": "#1F1F1F"})
    axD.bar(x + 0.18, mg, 0.34, color="#E65100", label="on governed", yerr=mg_e, capsize=3,
            error_kw={"lw": 1.2, "ecolor": "#1F1F1F"})
    axD.set_xticks(x); axD.set_xticklabels([SLAB[s] for s in cscen], rotation=20, ha="right"); axD.set_ylabel("MCP-Necessary Rate (%)")
    axD.set_title("(d) MCP Necessity: Overall vs on Governed")
    axD.set_ylim(0, max(h + e for h, e in zip(mg, mg_e[1])) * 1.2)
    _apply_style(axD)
    _legend(axD, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False,
            handlelength=1.4, columnspacing=1.2, handletextpad=0.4)

    fig.suptitle("Context-Layer Value: Decision-Level Channel Decomposition", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985], h_pad=2.4, w_pad=0.5)
    _save(fig, "context_value")
    _font_restore(_saved)


def fig13_stress_robustness(data=None):
    """H3 — communication robustness. (a) |ΔARI| heatmap (scenario×stressor),
    (b) absolute ARI under sensor noise (±ari_delta_std), (c) multi-metric
    robustness vs threshold, (d) ARI drift by stressor (±std + worst cell)."""
    from matplotlib.colors import LinearSegmentedColormap as _LSC
    from matplotlib.ticker import FormatStrFormatter as _FmtStr
    pf = RESULTS_DIR / "stress_passfail.csv"; ssp = RESULTS_DIR / "stress_summary.json"
    if not pf.exists():
        print("  [fig13] missing stress_passfail.csv; skipped")
        return
    rows = [r for r in csv.DictReader(pf.open()) if r["Method"] == "agribrain"]
    cell = {(r["Scenario"], r["Stressor"]): r for r in rows}
    C_CTX, C_BASE, C_OK, C_BAD = COLORS["agribrain"], "#9E9E9E", COLORS["no_context"], "#C62828"
    SCEN = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing", "baseline"]
    SLAB = SCENARIO_LABELS
    STRESS = ["sensor_noise", "missing_data", "telemetry_delay", "mcp_fault_injection", "compounded"]
    STLAB = {"sensor_noise": "Sensor noise", "missing_data": "Missing data",
             "telemetry_delay": "Telemetry delay", "mcp_fault_injection": "MCP fault",
             "compounded": "Compounded"}
    METRICS = [("ARI", "ari_delta", "Threshold_ARI", "higher"),
               ("Waste", "waste_delta", "Threshold_Waste", "lower"),
               ("SLCA", "slca_delta", "Threshold_SLCA", "higher"),
               ("RLE", "rle_delta", "Threshold_RLE", "higher"),
               ("Carbon", "carbon_delta", "Threshold_Carbon", "lower"),
               ("Equity", "equity_delta", "Threshold_Equity", "higher"),
               ("Latency", "latency_ms_delta", "Threshold_LatencyMs", "lower")]
    thr = {m[0]: float(rows[0][m[2]]) for m in METRICS}
    DRIFT = 0.01

    _saved = _font_bump(2)
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    (axA, axB), (axC, axD) = axes

    M = np.full((len(SCEN), len(STRESS)), np.nan)
    for i, s in enumerate(SCEN):
        for j, st in enumerate(STRESS):
            r = cell.get((s, st))
            if r:
                M[i, j] = abs(float(r["ari_delta"]))
    cmap = _LSC.from_list("rob", ["#E8F5E9", "#66BB6A", "#F9A825"])
    im = axA.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=DRIFT)
    axA.set_xticks(range(len(STRESS))); axA.set_xticklabels([STLAB[s] for s in STRESS], rotation=20, ha="right")
    axA.set_yticks(range(len(SCEN))); axA.set_yticklabels([SLAB[s] for s in SCEN])
    for i in range(len(SCEN)):
        for j in range(len(STRESS)):
            if not np.isnan(M[i, j]):
                axA.text(j, i, f"{M[i, j]*1000:.1f}", ha="center", va="center", fontsize=ANNOT_FONT_SIZE,
                         fontweight="bold", color="#1F1F1F" if M[i, j] < DRIFT * 0.7 else "white")
    cb = fig.colorbar(im, ax=axA, fraction=0.046, pad=0.03)
    cb.set_label(r"|ΔARI| ($\times10^{-3}$)", fontsize=TICK_FONT_SIZE, fontweight="bold")
    cb.ax.tick_params(labelsize=TICK_FONT_SIZE - 2)
    for _t in cb.ax.get_yticklabels():
        _t.set_fontweight("bold")
    axA.set_title(r"(a) ARI Drift under Stress ($\times10^{-3}$)")
    axA.grid(False); axA.tick_params(length=0)
    for sp in axA.spines.values():
        sp.set_visible(False)
    for lbl in axA.get_xticklabels() + axA.get_yticklabels():
        lbl.set_fontweight("bold")

    # Drift (ari_delta) is a paired within-experiment difference, so it is
    # seed-set-independent (no 5-seed-vs-20-seed caveat needed, unlike absolute
    # ARI). Sensor noise is the largest-drift stressor; per-scenario |drift| +/-
    # SE across the 5 stress seeds, against the pre-specified 0.01 threshold.
    _n_stress = 5
    delta = [abs(float(cell[(s, "sensor_noise")]["ari_delta"])) for s in SCEN]
    derr = [float(cell[(s, "sensor_noise")].get("ari_delta_std", 0) or 0) / np.sqrt(_n_stress) for s in SCEN]
    y = np.arange(len(SCEN))[::-1]
    axB.errorbar(delta, y, xerr=derr, fmt="o", color=C_CTX, markersize=16,
                 markeredgecolor="white", markeredgewidth=1.4, capsize=5, elinewidth=1.8, zorder=4)
    axB.axvline(DRIFT, color="#1F1F1F", lw=2, ls="--", zorder=2)
    axB.set_ylim(-0.6, len(SCEN) - 1 + 1.1)
    axB.text(DRIFT, len(SCEN) - 1 + 0.65, " Threshold", ha="left", va="center",
             fontsize=TICK_FONT_SIZE - 3, fontweight="bold", color="#1F1F1F")
    axB.set_yticks(y); axB.set_yticklabels([SLAB[s] for s in SCEN])
    axB.set_xlabel("|ΔARI| under Sensor Noise (Drift)")
    axB.set_title("(b) ARI Drift under Sensor Noise"); axB.set_xlim(0, DRIFT * 1.25); axB.set_xticks([0, 0.005, 0.01]); axB.xaxis.set_major_formatter(_FmtStr("%.3f"))
    _apply_style(axB); axB.grid(False); axB.grid(True, axis="x", linewidth=0.6, color="#BDBDBD", alpha=0.6)

    means, worsts = [], []
    for name, dcol, _, direction in METRICS:
        vals = np.array([((-float(r[dcol])) if direction == "higher" else float(r[dcol]))
                         / abs(thr[name]) for r in rows]) if thr[name] else np.zeros(len(rows))
        means.append(float(vals.mean())); worsts.append(float(vals.max()))
    yo = np.arange(len(METRICS))[::-1]
    axC.barh(yo, worsts, 0.6, color=C_OK, edgecolor="white", linewidth=0.8, label="worst cell", zorder=3)
    axC.scatter(means, yo, s=320, color="#1F1F1F", marker="|", linewidths=2.4, zorder=5, label="mean")
    axC.axvline(0, color="#9E9E9E", lw=1.0, alpha=0.6)
    axC.axvline(1.0, color="#1F1F1F", lw=2, ls="--")
    axC.text(1.0, len(METRICS) - 0.4, " Threshold", fontsize=TICK_FONT_SIZE - 2,
             fontweight="bold", va="top", color="#1F1F1F")
    axC.set_yticks(yo); axC.set_yticklabels([m[0] for m in METRICS])
    axC.set_xlabel("Drift as Fraction of Threshold"); axC.set_title("(c) Multi-Metric Robustness")
    axC.set_xlim(min(0.0, min(worsts) - 0.02), 1.15)
    _apply_style(axC); axC.grid(False); axC.grid(True, axis="x", linewidth=0.6, color="#BDBDBD", alpha=0.6)
    _legend(axC, loc="upper center", bbox_to_anchor=(0.5, 1.0))

    means, stds, worsts = [], [], []
    for st in STRESS:
        vals = [abs(float(cell[(s, st)]["ari_delta"])) for s in SCEN]
        means.append(float(np.mean(vals))); stds.append(float(np.std(vals, ddof=1))); worsts.append(max(vals))
    xb2 = np.arange(len(STRESS))
    axD.bar(xb2, means, 0.6, color=C_CTX, label="mean |ΔARI|", yerr=stds, capsize=5, error_kw={"lw": 1.6, "ecolor": "#1F1F1F"})
    axD.scatter(xb2, worsts, s=130, color=C_BAD, marker="D", zorder=5, label="worst |ΔARI|", edgecolor="white", linewidth=1.0)
    axD.axhline(DRIFT, color="#1F1F1F", lw=2, ls="--")
    axD.text(len(STRESS) - 0.5, DRIFT * 0.95, "Threshold ", fontsize=TICK_FONT_SIZE - 2, fontweight="bold", va="top", ha="right", color="#1F1F1F")
    axD.set_xticks(xb2); axD.set_xticklabels([STLAB[s] for s in STRESS], rotation=20, ha="right"); axD.set_ylabel("|ΔARI|")
    axD.set_title("(d) ARI Drift by Stressor"); axD.set_ylim(0, DRIFT * 1.08)
    _apply_style(axD); _legend(axD, loc="upper center", bbox_to_anchor=(0.5, 0.9))

    fig.suptitle("Communication Robustness under Sensing and Protocol Stressors", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985], h_pad=2.0, w_pad=1.6)
    _save(fig, "stress_robustness")
    _font_restore(_saved)


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
    # H1/H2/H3 paper figures (read the saved benchmark / attribution / stress
    # artefacts; skip with a message if those inputs are absent). The latency
    # frontier formerly in fig10 is now folded into fig11 panels (c)/(d).
    fig11_performance_efficiency(data)
    fig12_context_channels(data)
    fig13_stress_robustness(data)
    print()
    print(f"All figures saved to {RESULTS_DIR}")


if __name__ == "__main__":
    print("=" * 70)
    print("AGRI-BRAIN Figure Generation")
    print("=" * 70)
    generate_all_figures()
