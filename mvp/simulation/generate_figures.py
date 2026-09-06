#!/usr/bin/env python3
"""
AGRI-BRAIN Figure Generation
==============================
Generates figures
as PNG + PDF at 800 DPI. The shared style block below is the single
source of truth for typography, palette, and layout so that every
figure in the paper, poster, and slide deck matches exactly.

This module is a renderer library. Publication rendering is orchestrated by
``regenerate_figures_from_cache.py`` after it validates a complete, identified
seed cache. Direct execution fails closed and never runs a one-seed simulation.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

_BACKEND_SRC = Path(__file__).resolve().parent.parent.parent / "agribrain" / "backend"
if str(_BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(_BACKEND_SRC))

import matplotlib

matplotlib.use("Agg")

import contextlib as _contextlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as _patheffects
import numpy as np
from analysis.publication_figure_style import (
    ANNOT_FONT_SIZE,
    AXIS_LABEL_SIZE,
    BODY_FONT_SIZE,  # noqa: F401  read by name in _SCALED_NAMES/_SCALED_RC
    FIG_TITLE_SIZE,
    LEGEND_FONT_SIZE,
    PANEL_KEY_FONT_CAP,
    PANEL_KEY_FONT_SIZE,
    PANEL_KEY_OVERHANG,
    MARKER_EVERY,
    PUBLICATION_DPI,
    SEMANTIC_COLORS,
    SEMANTIC_HATCHES,
    SEMANTIC_LINESTYLES,
    SEMANTIC_MARKERS,
    SUBPLOT_TITLE_SIZE,
    TICK_FONT_SIZE,
    accessible_legend,
    apply_publication_style,
    save_figure_pair,
    style_axes,
)
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

from benchmarks.trace_contract import (
    TRACE_LENGTH,
    validate_trace_cell,
)
from benchmarks.trace_contract import (
    TRACE_MODES as CANONICAL_TRACE_MODES,
)
from generate_results import RESULTS_DIR, SCENARIOS, Policy
from src.models.action_selection import ACTIONS
from src.models.carbon import compute_carbon_efficiency
from src.models.resilience import (
    HIERARCHY_WEIGHT,
    RLE_THRESHOLD,
    hierarchy_weight,
)

# ---------------------------------------------------------------------------
# Unified publication-quality style
# ---------------------------------------------------------------------------
apply_publication_style()


# A four-panel figure carries four axes in the printed width a strip figure
# gives to one, so its text has to be set larger to stay readable at journal
# column size. One scaler, one delta, every 2x2 figure: figs 2-5 previously
# bumped by hand, figs 11-13 called a bump of zero, and fig 6 read the
# unbumped globals through local aliases, so the family was never actually
# uniform despite the comments claiming it was.
#
# The move is uniform across all seven sizes, which preserves the hierarchy
# between a tick label, an axis label and a panel title. Both the module
# globals and the rcParams are shifted: helpers such as _apply_style read the
# globals at call time, while anything drawn straight through matplotlib reads
# the rcParams.
FOUR_PANEL_FONT_BUMP = 4

_SCALED_NAMES = (
    "BODY_FONT_SIZE", "TICK_FONT_SIZE", "AXIS_LABEL_SIZE",
    "SUBPLOT_TITLE_SIZE", "FIG_TITLE_SIZE", "LEGEND_FONT_SIZE",
    "ANNOT_FONT_SIZE",
)
_SCALED_RC = (
    ("font.size", "BODY_FONT_SIZE"),
    ("axes.labelsize", "AXIS_LABEL_SIZE"),
    ("axes.titlesize", "SUBPLOT_TITLE_SIZE"),
    ("xtick.labelsize", "TICK_FONT_SIZE"),
    ("ytick.labelsize", "TICK_FONT_SIZE"),
    ("legend.fontsize", "LEGEND_FONT_SIZE"),
    ("legend.title_fontsize", "LEGEND_FONT_SIZE"),
    ("figure.titlesize", "FIG_TITLE_SIZE"),
)


class panel_fonts(_contextlib.ContextDecorator):
    """Add ``delta`` points to every figure font size for the duration.

    Used as a decorator so a figure keeps the enlarged sizes for its whole
    body and gives them back on the way out, whether it returns or raises; a
    figure that leaked its sizes would silently enlarge whichever figure was
    drawn next.
    """

    def __init__(self, delta):
        self.delta = delta

    def __enter__(self):
        g = globals()
        self._saved = {name: g[name] for name in _SCALED_NAMES}
        for name in _SCALED_NAMES:
            g[name] = self._saved[name] + self.delta
        self._saved_rc = {key: plt.rcParams[key] for key, _ in _SCALED_RC}
        plt.rcParams.update({key: g[name] for key, name in _SCALED_RC})
        return self

    def __exit__(self, *exc):
        globals().update(self._saved)
        plt.rcParams.update(self._saved_rc)
        return False


# ---------------------------------------------------------------------------
# High-contrast palette for the exact eight primary and three secondary arms
# ---------------------------------------------------------------------------
COLORS = dict(SEMANTIC_COLORS)
HATCHES = dict(SEMANTIC_HATCHES)
MARKERS = dict(SEMANTIC_MARKERS)
LINESTYLES = dict(SEMANTIC_LINESTYLES)

# Figure keys carry the short arm name only. What each arm ablates, and how,
# is stated in the caption and the methods text where there is room to say it
# precisely; repeating it inside a six-inch panel is what pushed the earlier
# keys on top of the data.
MODE_LABELS = {
    "static":     "Static",
    "hybrid_rl":  "Hybrid RL",
    "no_pinn":    "No-PINN",
    "no_slca":    "No-sLCA",
    "agribrain":  "AGRI-BRAIN",
    "no_context": "No-context",
    "mcp_only":   "MCP",
    "pirag_only": "Retrieval",
    "agribrain_standard_rag":       "Standard-RAG",
    "agribrain_no_peer":            "No-peer",
    "agribrain_sign_unconstrained": "Sign-free",
}

SCENARIO_LABELS = {
    "heatwave":         "Heatwave",
    "overproduction":   "Overproduction",
    "cyber_outage":     "Cyber Outage",
    "adaptive_pricing": "Adaptive Pricing",
    "baseline":         "Baseline",
}

# Categorical tick form of the same names. Wrapping to two lines keeps every
# tick horizontal, which reads better than a rotated label and is immune to
# the neighbour collisions rotation produces once a panel gets narrow.
SCENARIO_TICKS = {
    "heatwave":         "Heatwave",
    "overproduction":   "Over-\nproduction",
    "cyber_outage":     "Cyber\nOutage",
    "adaptive_pricing": "Adaptive\nPricing",
    "baseline":         "Baseline",
}

# Highlight color used for shaded scenario windows and emphasis text
WINDOW_COLOR = "#B71C1C"      # deep red, high contrast against teal agribrain
WINDOW_ALPHA = 0.12
ACTION_COLORS = {
    "cold_chain": "#332288",
    "local_redistribution": "#009E73",
    "recovery": "#D55E00",
}
ACTION_HATCHES = {
    "cold_chain": "///",
    "local_redistribution": "\\\\",
    "recovery": "xx",
}
PERIOD_HATCHES = {"before": "//", "during": "xx"}

DPI = PUBLICATION_DPI


def _apply_style(ax):
    """Apply the shared subplot styling. Safe to call multiple times."""
    style_axes(ax)
    # Per-figure layout functions may temporarily bump the imported size
    # aliases. Reapply those live values after the shared helper so a local
    # readability adjustment cannot be silently reset to the base sizes.
    ax.tick_params(labelsize=TICK_FONT_SIZE)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontsize(TICK_FONT_SIZE)
        label.set_fontweight("bold")
    for axis in (ax.xaxis, ax.yaxis):
        axis.get_offset_text().set_fontsize(TICK_FONT_SIZE)
        axis.get_offset_text().set_fontweight("bold")
    if ax.xaxis.label.get_text():
        ax.xaxis.label.set(size=AXIS_LABEL_SIZE, weight="bold")
    if ax.yaxis.label.get_text():
        ax.yaxis.label.set(size=AXIS_LABEL_SIZE, weight="bold")
    if ax.get_title():
        ax.title.set(size=SUBPLOT_TITLE_SIZE, weight="bold")


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


def _rolling_mean(values, window: int, *, centered: bool = True) -> np.ndarray:
    """Edge-truncated rolling mean with no implicit zero padding.

    ``numpy.convolve(..., mode="same")`` pads both ends with zeros and creates
    artificial endpoint drops. Descriptive trajectory panels use a centred
    window; online monitoring summaries can request a trailing window.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or window < 1:
        raise ValueError("rolling mean requires a 1D array and window >= 1")
    out = np.empty_like(array, dtype=float)
    left = (window - 1) // 2
    right = window // 2
    for index in range(len(array)):
        if centered:
            lo = max(0, index - left)
            hi = min(len(array), index + right + 1)
        else:
            lo = max(0, index - window + 1)
            hi = index + 1
        out[index] = float(np.mean(array[lo:hi]))
    return out


def _rolling_sum(values, window: int, *, centered: bool = True) -> np.ndarray:
    """Edge-truncated rolling sum paired with :func:`_rolling_mean`."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or window < 1:
        raise ValueError("rolling sum requires a 1D array and window >= 1")
    out = np.empty_like(array, dtype=float)
    left = (window - 1) // 2
    right = window // 2
    for index in range(len(array)):
        if centered:
            lo = max(0, index - left)
            hi = min(len(array), index + right + 1)
        else:
            lo = max(0, index - window + 1)
            hi = index + 1
        out[index] = float(np.sum(array[lo:hi]))
    return out


def _legend(ax, **kwargs):
    """Add a bold, high-contrast publication legend."""
    kwargs.setdefault("fontsize", LEGEND_FONT_SIZE)
    kwargs.setdefault("title_fontsize", LEGEND_FONT_SIZE)
    return accessible_legend(ax, **kwargs)


# ---------------------------------------------------------------------------
# Shared multi-panel layout contract
# ---------------------------------------------------------------------------
# Every figure is drawn on the same 18-inch canvas, so one page width renders
# identical type in all of them, and every panel key is placed the same way.
# A key is never drawn inside the data area: it goes in reserved blank space
# directly above its axes, under the panel title, where it cannot occlude a
# line, a bar, or an annotation no matter where the data falls.
GRID_FIGSIZE = (18.0, 13.5)      # 2x2 figures
TRIPTYCH_FIGSIZE = (18.0, 7.6)   # 1x3 figures
PAIR_FIGSIZE = (18.0, 8.0)       # 1x2 figures
# The figure title is anchored just under the layout rect it shares with the
# panels. Anchoring it at the very top of the canvas instead leaves the whole
# unused reserve as a blank band under it -- about an inch on these figures --
# so the two values are kept together here and applied by the finishers rather
# than passed in at each call site.
GRID_RECT_TOP = 0.985            # 2x2 figures
STRIP_RECT_TOP = 0.955           # single-row figures
SUPTITLE_DROP = 0.005            # title anchor sits this far under the rect top
SUPTITLE_Y = GRID_RECT_TOP - SUPTITLE_DROP
# Room reserved above the axes for the key, expressed as multiples of the key's
# own font size rather than in absolute points: a fixed reservation silently
# stops clearing the title as soon as the type is set larger, and the key rides
# up into it. The two factors reproduce the previous 25 pt row and 12 pt gap at
# the 20 pt key font they were tuned against.
_KEY_ROW_LEADING = 1.25          # vertical room one key row needs, per pt of font
_KEY_TITLE_GAP = 2.10            # gap between the key block and the title, per pt
# Figure 11 fixes its own grid rather than reflowing, so the block above its
# panels -- one key row, the gap, the panel title and the figure title -- is
# subtracted from the canvas here instead. Stated as a formula so it tracks the
# constants above rather than needing retuning whenever they move.
_F11_CANVAS_PT = GRID_FIGSIZE[1] * 72.0
_F11_TITLE_BLOCK = (
    PANEL_KEY_FONT_SIZE * (_KEY_ROW_LEADING + _KEY_TITLE_GAP)
    + SUBPLOT_TITLE_SIZE + FIG_TITLE_SIZE
)
_F11_SUPTITLE_Y = 0.988
_F11_GRID_TOP = _F11_SUPTITLE_Y - _F11_TITLE_BLOCK / _F11_CANVAS_PT


#: Labels that repeat what their own panel title or axis already says. Cutting
#: them is what lets every short key share one readable size.
_KEY_LABEL_SHORT = {
    # (a) Environmental Exposure: the y-axis already reads Temperature.
    "Temp (latent)": "Latent",
    "Temp (observed)": "Observed",
    # (b) Feature-Group Masking: the panel title already says masking.
    "Observed vs zeroed": "Observed",
    "MCP mask": "MCP",
    "Retrieval mask": "Retrieval",
    "Joint-only change": "Joint-only",
}
_KEY_SINGLE_ROW_MAX = 4
_KEY_SPACING = {"handlelength": 1.6, "handletextpad": 0.4, "columnspacing": 1.1}


def _key_room(ax, renderer):
    """Width a key centred on this panel may take before it meets a neighbour.

    A key centred over its panel may reach into the gutter beside it, but not
    past the midpoint of the gap to whatever sits next to it in the same
    horizontal band. Measuring that gap panel by panel is what lets the type
    stay large: a flat fraction of the panel width has to be set for the
    tightest panel in the set and then costs every other panel the size it
    could have had. PANEL_KEY_OVERHANG survives as the ceiling for a panel with
    no neighbour at all.
    """
    me = ax.get_window_extent(renderer)
    canvas = ax.figure.get_window_extent(renderer)
    left, right = canvas.x0, canvas.x1
    for other in ax.figure.axes:
        if other is ax:
            continue
        box = other.get_window_extent(renderer)
        if box.y1 <= me.y0 or box.y0 >= me.y1:
            continue                      # not in the same band
        if box.x1 <= me.x0:
            left = max(left, (box.x1 + me.x0) / 2.0)
        elif box.x0 >= me.x1:
            right = min(right, (box.x0 + me.x1) / 2.0)
    centre = (me.x0 + me.x1) / 2.0
    return min(2.0 * (centre - left), 2.0 * (right - centre),
               me.width * PANEL_KEY_OVERHANG)


def _panel_key(ax, *, handles=None, labels=None, ncol=None, **kwargs):
    """Draw one panel's key in reserved space above its axes.

    Four entries or fewer always occupy a single row, at one absolute size
    shared by every key in every figure, so a two-entry key and a four-entry
    key read as the same object.

    A longer key is sized to the room it has rather than held to that size: it
    starts at the cap and gives up half a point at a time until it fits beside
    its neighbour, down to the shared size as a floor. Only if it still does not
    fit there does it take a second row. Wrapping a five- or six-entry key at
    the shared size would cost a row on panels that had the width for one line.
    """
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    labels = [_KEY_LABEL_SHORT.get(text, text) for text in labels]
    count = len(handles)
    short = count <= _KEY_SINGLE_ROW_MAX
    if short:
        ncol = count
    elif ncol is None:
        ncol = (count + 1) // 2
    rows = -(-count // ncol)
    kwargs.pop("fontsize", None)
    fontsize = PANEL_KEY_FONT_SIZE if short else PANEL_KEY_FONT_CAP
    anchor = kwargs.pop("bbox_to_anchor", (0.5, 1.0))

    def draw(columns):
        return ax.legend(
            handles, labels,
            loc="lower center", bbox_to_anchor=anchor, ncol=columns,
            frameon=False, borderaxespad=0.0, labelspacing=0.3,
            fontsize=fontsize, **_KEY_SPACING, **kwargs,
        )

    legend = draw(ncol)
    renderer = ax.figure.canvas.get_renderer()
    fits = lambda lg: lg.get_window_extent(renderer).width <= _key_room(ax, renderer)
    if not short:
        # Type size first, down to the size the short keys share; a second row
        # only if that floor is still too wide.
        while fontsize > PANEL_KEY_FONT_SIZE and not fits(legend):
            fontsize = max(float(PANEL_KEY_FONT_SIZE), fontsize - 0.5)
            legend.remove()
            legend = draw(ncol)
        if not fits(legend) and ncol > 1:
            ncol = max(1, ncol - 1)
            rows = -(-count // ncol)
            legend.remove()
            legend = draw(ncol)

    for text in legend.get_texts():
        text.set_fontweight("bold")
    pad = fontsize * (_KEY_ROW_LEADING * rows + _KEY_TITLE_GAP)
    ax._agri_title_pad = pad
    ax.set_title(
        ax.get_title(), pad=pad,
        fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold",
    )
    return legend


def _align_panel_titles(fig):
    """Give every titled panel in a figure the same title offset.

    Panels carrying a two-row key would otherwise sit their titles higher
    than their key-less neighbours, which reads as a misaligned grid.
    """
    titled = [ax for ax in fig.axes if ax.get_title()]
    if not titled:
        return
    # A key-less panel still needs the plain title gap, scaled the same way.
    plain_gap = PANEL_KEY_FONT_SIZE * _KEY_TITLE_GAP
    pad = max(getattr(ax, "_agri_title_pad", plain_gap) for ax in titled)
    for ax in titled:
        ax.set_title(ax.get_title(), pad=pad,
                     fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold")


def _cat_ticks(ax, positions, labels, axis="x"):
    """Set horizontal categorical tick labels from the wrapped label forms."""
    if axis == "x":
        ax.set_xticks(list(positions))
        ax.set_xticklabels(list(labels), rotation=0, ha="center")
    else:
        ax.set_yticks(list(positions))
        ax.set_yticklabels(list(labels), rotation=0)


# A category slot is as wide as the data allows; enlarging the type widens the
# label inside it but not the slot, so labels that cleared each other at the
# canonical size can run together. Rather than let them collide, the offending
# axis steps its own tick text down just far enough to clear, leaving the rest
# of the figure's enlarged type alone. Numeric axes are measured too and are
# simply left as they are, since their labels do not touch.
_TICK_FIT_FLOOR = 0.72           # never shrink tick text below this much of its size
_TICK_FIT_GAP = 0.35             # clear gap between neighbours, per pt of font


def _fit_tick_labels(fig):
    """Shrink any x axis whose tick labels overlap, until they do not."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        labels = [t for t in ax.get_xticklabels() if t.get_text()]
        if len(labels) < 2:
            continue
        size = floor = labels[0].get_fontsize()
        floor *= _TICK_FIT_FLOOR
        while size > floor:
            gap = _TICK_FIT_GAP * size * fig.dpi / 72.0
            boxes = sorted((t.get_window_extent(renderer) for t in labels),
                           key=lambda b: b.x0)
            if all(a.x1 + gap <= b.x0 for a, b in zip(boxes, boxes[1:])):
                break
            size *= 0.94
            for text in labels:
                text.set_fontsize(size)


def _seat_suptitle(fig, rect_top):
    """Anchor the figure title directly above the panels."""
    if fig._suptitle is not None:
        fig._suptitle.set_y(rect_top - SUPTITLE_DROP)


# Row gap for the shared 2x2 layout, in multiples of the font size. It has to
# hold the upper row's tick labels and axis name above the lower row's key and
# panel title, and no more: anything beyond that reads as a band of dead space
# across the middle of the figure. Figure 11 states its own geometry and does
# not come through here.
GRID_H_PAD = 1.2
GRID_W_PAD = 3.2


def _finish_grid(fig, *, bottom=0.0):
    """Shared outer layout for every 2x2 figure."""
    _align_panel_titles(fig)
    fig.tight_layout(rect=[0, bottom, 1, GRID_RECT_TOP],
                     h_pad=GRID_H_PAD, w_pad=GRID_W_PAD)
    _seat_suptitle(fig, GRID_RECT_TOP)
    _fit_tick_labels(fig)


def _finish_strip(fig, *, bottom=0.0):
    """Shared outer layout for every single-row (1x2 / 1x3) figure."""
    _align_panel_titles(fig)
    fig.tight_layout(rect=[0, bottom, 1, STRIP_RECT_TOP], w_pad=3.2)
    _seat_suptitle(fig, STRIP_RECT_TOP)
    _fit_tick_labels(fig)


def _save(fig, name):
    """Save figure as PNG (800 DPI) and PDF (vector, TrueType fonts)."""
    output_raw = os.environ.get("FIGURE_OUTPUT_DIR", "").strip()
    if not output_raw:
        raise RuntimeError(
            "FIGURE_OUTPUT_DIR is required; render through "
            "regenerate_figures_from_cache.py after evidence validation"
        )
    output_dir = Path(output_raw)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure_pair(fig, output_dir, name, dpi=DPI)
    print(f"  Saved {name}.png / .pdf")
    plt.close(fig)


def _twin_axes(ax):
    """The axes sharing this one's frame -- what twinx() produced."""
    box = ax.get_position().bounds
    return [other for other in ax.figure.axes
            if other is not ax and other.get_position().bounds == box]


def _label_axes_fraction(ax, ann, fontsize):
    """Height of a drawn annotation, including its box, as a share of the axes.

    Measured rather than computed from the font: the box adds padding and the
    text's own ascent and descent vary with the glyphs, and an estimate that is
    a few percent short puts the badge's edge exactly on the data it was meant
    to clear.
    """
    fig = ax.figure
    try:
        renderer = fig.canvas.get_renderer()
        text_px = ann.get_window_extent(renderer).height
        axes_px = ax.get_window_extent(renderer).height
    except Exception:
        return None
    if axes_px <= 0:
        return None
    size = ANNOT_FONT_SIZE if fontsize is None else fontsize
    # get_window_extent covers the text; the round boxstyle's pad is a multiple
    # of the font size, applied on both edges.
    pad_px = 2 * 0.25 * size * fig.dpi / 72.0
    return (text_px + pad_px) / axes_px


def _window_headroom(ax, ann, fontsize, ypos, va):
    """Factor to grow a y span by so a window label clears the data.

    For a top-anchored label the span must grow by 1 / (ypos - box_frac) to put
    the box's bottom edge on the old data top; a bottom-anchored one needs
    1 / (1 - ypos - box_frac) growing the other way. A small margin turns
    "exactly touching" into a visible gap. Clamped so an unusually tall label
    in a short panel cannot blow the scale out.
    """
    # A label the caller has deliberately placed down among the traces is not
    # trying to clear them, so it gets the modest legacy breathing room rather
    # than a reservation sized to push every data point clear of it.
    if va == "top" and ypos < 0.85:
        return 1.18
    box_frac = _label_axes_fraction(ax, ann, fontsize)
    if box_frac is None:
        return 1.18
    clear = (ypos - box_frac) if va == "top" else (1.0 - ypos - box_frac)
    if clear <= 0.25:
        return 1.55
    return min(1.55, 1.06 / clear)


def _annotate_window(ax, x0, x1, color, label, alpha=WINDOW_ALPHA,
                     ypos=0.93, xpos=None, va="top", fontsize=None,
                     headroom=None):
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
    label_x = (x0 + x1) / 2 if xpos is None else xpos
    ann = ax.annotate(
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

    # The badge is opaque, so whatever it lands on is hidden rather than merely
    # crowded. A one-shot span expansion moves the data clear of it -- upward
    # for a top-anchored label, downward for a bottom-anchored one. Callers
    # that locked their ylim (ratio axes, say) are respected and skipped.
    # ``headroom`` overrides the autoscale test, for a panel that pins its own
    # limits -- otherwise pinning them silently opts out of the reservation and
    # the badge lands on the data.
    want = ax.get_autoscaley_on() if headroom is None else headroom
    if (
        va in ("top", "bottom")
        and not getattr(ax, "_window_headroom_applied", False)
        and want
    ):
        factor = _window_headroom(ax, ann, fontsize, ypos, va)
        y_lo, y_hi = ax.get_ylim()
        span = y_hi - y_lo
        if span > 0:
            if va == "top":
                ax.set_ylim(y_lo, y_lo + span * factor)
            else:
                ax.set_ylim(y_hi - span * factor, y_hi)
        ax._window_headroom_applied = True
        # A twin y-axis carries its own curve, which the expansion above does
        # not move; without this the label clears the primary axis's data and
        # is still crossed by the twin's. Callers whose twin has a natural
        # ceiling (a percentage, say) must anchor at the bottom instead -- see
        # figure 2 panel (a), whose twin is relative humidity.
        for twin in _twin_axes(ax):
            if getattr(twin, "_window_headroom_applied", False):
                continue
            t_lo, t_hi = twin.get_ylim()
            t_span = t_hi - t_lo
            if t_span > 0:
                if va == "top":
                    twin.set_ylim(t_lo, t_lo + t_span * factor)
                else:
                    twin.set_ylim(t_hi - t_span * factor, t_hi)
            twin._window_headroom_applied = True


# ---------------------------------------------------------------------------
# Figure 2: Heatwave scenario deep-dive (2x2)
# ---------------------------------------------------------------------------
@panel_fonts(FOUR_PANEL_FONT_BUMP)
def fig2_heatwave(data):
    """2x2: latent/observed state, policy response, and per-step ARI.

    Panel (b) distinguishes the common latent environmental spoilage state
    used for endpoints from the noisy/delayed state available to routing.

    Panel (c) shows AgriBrain's action-probability stacked area and the
    severity-weighted RLE at-risk trigger (rho > 0.10). It does not display
    retired route-conditioned freshness or disposition cutoffs.

    Panel (d) plots per-step ARI (12-step rolling mean). ARI is bounded
    [0, 1].
    """
    hw = data["results"]["heatwave"]
    ab = hw["agribrain"]
    hours = np.array(ab["hours"])

    return _fig2_heatwave_inner(hw, ab, hours)


def _fig2_heatwave_inner(hw, ab, hours):
    """Body of fig 2. Extracted from ``fig2_heatwave`` so the per-figure
    font-size overrides applied above can be cleanly torn down via
    try/finally regardless of how the body returns or raises."""
    fig, axes = plt.subplots(2, 2, figsize=GRID_FIGSIZE)
    fig.suptitle("Heatwave Scenario Analysis", y=SUPTITLE_Y)

    # --- (a) Temperature + Humidity with heatwave window ---
    ax = axes[0, 0]
    ax.plot(hours, ab["temp_outcome_environmental_trace"],
            color="#B71C1C", linewidth=2.4, linestyle="-",
            label="Temp (latent)")
    ax.plot(hours, ab["temp_policy_observed_trace"], color="#882255",
            linewidth=1.3, alpha=1.0, linestyle="--", marker="o",
            markevery=MARKER_EVERY, label="Temp (observed)")
    policy_ceiling = float(Policy().max_temp_c)
    ax.axhline(policy_ceiling, color="#C62828", linestyle=":", linewidth=1.4,
               alpha=0.65, label="Policy ceiling")
    ax2 = ax.twinx()
    ax2.plot(hours, ab["rh_outcome_environmental_trace"], color="#332288",
             linewidth=2.2, alpha=0.9, linestyle="-.", label="RH")
    ax.set_xlabel("Time (hr.)")
    ax.set_ylabel("Temperature (\u00b0C)")
    ax2.set_ylabel("Relative Humidity (%)")
    ax.set_title("(a) Environmental Exposure")
    _apply_style(ax)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(labelsize=TICK_FONT_SIZE, length=5, width=1.3)
    ax2.yaxis.label.set_size(AXIS_LABEL_SIZE)
    ax2.yaxis.label.set_weight("normal")
    for lbl in ax2.get_yticklabels():
        lbl.set_fontweight("normal")
    ax2.set_ylim(30, 105)
    # The window tag is anchored at the BOTTOM here, unlike every other panel.
    # This panel's twin axis is relative humidity, which runs at 90-100% and so
    # occupies the top of the frame; the usual top-anchored placement put the
    # RH trace straight through the label's glyphs. The twin cannot be given
    # headroom either -- that would print a tick above 100% RH. Below the
    # traces is genuinely empty: across the 24-48 h window the temperature
    # never drops under 15 degC.
    _annotate_window(ax, 24, 48, WINDOW_COLOR, "Heatwave",
                     ypos=0.05, va="bottom")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    # Both axes' series share one key above the panel. Nothing has to be
    # drawn on top of the dual-axis traces to stay readable.
    _panel_key(ax, handles=h1 + h2, labels=l1 + l2, ncol=2)

    # --- (b) Independent-DGP outcome and policy estimate ---
    # The solid curve is the common noise-free synthetic DGP outcome used for
    # scoring. The dashed curve is the frozen residual-enabled estimate made
    # available to the policy. This is internal synthetic validation, not
    # observed-quality or external shelf-life evidence.
    ax = axes[0, 1]
    _rho_latent = np.asarray(
        ab["rho_outcome_environmental_trace"], dtype=np.float64,
    )
    _rho_observed = np.asarray(
        ab["rho_policy_observed_trace"], dtype=np.float64,
    )

    ax.plot(hours, _rho_latent, color="#212121", linewidth=2.6,
            label="Scored outcome")
    ax.plot(hours, _rho_observed, color=COLORS["agribrain"], linewidth=1.8,
            alpha=0.9, linestyle="--", marker=MARKERS["agribrain"],
            markevery=MARKER_EVERY, label="PINN estimate")
    # The only threshold shown is the severity-weighted RLE event trigger. The
    # legacy 0.30 route knee and 0.65 disposition cutoff are excluded from the
    # confirmatory model and therefore must not appear as policy evidence.
    ax.axhline(RLE_THRESHOLD, color="#A66F00", linestyle=":", linewidth=1.4,
               alpha=0.7)
    # Above the line, not below it: the spoilage curve climbs from the left and
    # crosses the threshold only around hr 37, so the band above the line is
    # empty exactly where the label sits, while below it the rising curve ran
    # under the label's near-opaque box and its markers were clipped.
    ax.text(1.5, RLE_THRESHOLD + 0.020,
            f"RLE trigger (ρ > {RLE_THRESHOLD:.2f})",
            color="#A66F00", fontsize=17, va="bottom", ha="left",
            fontweight="bold")
    # No box: the label runs past the start of the shaded window, and an opaque
    # ground punched a white notch out of the shading. Nothing is plotted here
    # -- the curve is still near zero this early -- so the text needs no ground
    # of its own.

    ax.set_xlabel("Time (hr.)")
    ax.set_ylabel("Spoilage Risk")
    ax.set_title("(b) Spoilage Risk Trajectory")
    ax.set_ylim(0, 0.70)
    ax.set_xlim(0, 72)
    _apply_style(ax)
    _annotate_window(ax, 24, 48, WINDOW_COLOR, "Heatwave")
    _panel_key(ax)

    # --- (c) AgriBrain action-probability stacked area + regime guides ---
    ax = axes[1, 0]
    probs = np.array(ab["prob_trace"])
    ax.fill_between(
        hours, 0, probs[:, 0], color=ACTION_COLORS["cold_chain"],
        alpha=1.0, hatch=ACTION_HATCHES["cold_chain"],
        edgecolor="#1F1F1F", linewidth=0.5, label="Cold Chain",
    )
    ax.fill_between(hours, probs[:, 0], probs[:, 0] + probs[:, 1],
                    color=ACTION_COLORS["local_redistribution"], alpha=1.0,
                    hatch=ACTION_HATCHES["local_redistribution"],
                    edgecolor="#1F1F1F", linewidth=0.5, label="Local Redist.")
    ax.fill_between(hours, probs[:, 0] + probs[:, 1], 1.0,
                    color=ACTION_COLORS["recovery"], alpha=1.0,
                    hatch=ACTION_HATCHES["recovery"],
                    edgecolor="#1F1F1F", linewidth=0.5, label="Recovery")

    # Show when the policy-observed risk first enters the RLE event set. This
    # is an interpretation guide, not a hard action-selection threshold.
    ab_rho = np.array(ab["rho_policy_observed_trace"])
    def _first_cross(threshold):
        idx = np.argmax(ab_rho > threshold)
        if idx == 0 and ab_rho[0] <= threshold:
            return None
        return float(hours[idx])

    h_atrisk = _first_cross(RLE_THRESHOLD)
    if h_atrisk is not None:
        # Drawn across the stack only, not the full frame. The band above 1.0
        # is reserved for the window tag, and a full-height marker ran under
        # the tag's opaque box, which broke the line into two stubs.
        ax.plot([h_atrisk, h_atrisk], [0.0, 1.0], color="#424242",
                linestyle="--", linewidth=1.1, alpha=0.65)
        # This panel is a stacked area summing to 1, so there is no empty
        # ground anywhere inside the frame to put a label on. A filled box
        # either lets the band's hatching run through the glyphs (if it is
        # translucent) or hides a block of the stack (if it is not), so the
        # glyphs carry a stroked outline instead: legible against the hatch
        # while covering only the width of the outline itself.
        ax.text(h_atrisk + 0.4, 0.05,
                f"\u03c1>{RLE_THRESHOLD:.2f}\n@hr{h_atrisk:.0f}",
                fontsize=ANNOT_FONT_SIZE - 1, color="#111111",
                fontweight="bold", va="bottom", zorder=6,
                path_effects=[_patheffects.withStroke(linewidth=4.0,
                                                      foreground="white")])

    ax.set_xlabel("Time (hr.)")
    ax.set_ylabel("Action Probability")
    ax.set_title("(c) Policy Response to Heat Stress")
    # The three action probabilities sum to 1, so the band above 1.0 is blank
    # by construction and is where the window tag goes. How much of that band
    # the tag needs depends on the tag, so the ceiling is asked for rather than
    # guessed -- the hand-tuned 1.18 that used to sit here stopped clearing the
    # stack once the type was set larger, and pinning the limit at all had
    # quietly opted this panel out of the reservation.
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    _apply_style(ax)
    _annotate_window(ax, 24, 48, WINDOW_COLOR, "Heatwave", headroom=True)
    _panel_key(ax, ncol=3)

    # --- (d) Per-step Adaptive resilience index (ARI) ---
    # Per-step ARI = (1 - waste) * social-performance term * (1 - rho),
    # computed by resilience.compute_ari and surfaced as ``ari_trace`` in the
    # results JSON. The (1 - rho) factor uses the common latent environmental
    # rho. Fixed actions have mode-neutral outcome equations; any between-mode
    # difference arises from selected actions and learned/contextual policy
    # state, not a method-specific physical efficiency multiplier.
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
    for mode in ["static", "hybrid_rl", "agribrain"]:
        ep = hw[mode]
        per_seed = _load_per_seed_traces("heatwave", mode, "ari_trace")
        if per_seed is not None and per_seed.shape[0] >= 2:
            n = min(per_seed.shape[1], hours.shape[0])
            seed_mean = per_seed[:, :n].mean(axis=0)
            mean_smooth = _rolling_mean(seed_mean, window)
            _mode_plot(ax, hours[:n], mean_smooth, mode)
        else:
            ari = np.array(ep["ari_trace"])
            rolling = _rolling_mean(ari, window)
            _mode_plot(ax, hours, rolling, mode)
    ax.set_xlabel("Time (hr.)")
    ax.set_ylabel("Adaptive Resilience Index")
    ax.set_title("(d) Resilience under Heat Stress")
    ax.set_ylim(0, 1.0)
    _apply_style(ax)
    _annotate_window(ax, 24, 48, WINDOW_COLOR, "Heatwave")
    _panel_key(ax)

    _finish_grid(fig)
    _save(fig, "heatwave")


# ---------------------------------------------------------------------------
# Figure 3: Overproduction / Reverse Logistics (2x2)
# ---------------------------------------------------------------------------
@panel_fonts(FOUR_PANEL_FONT_BUMP)
def fig3_overproduction(data):
    """2x2: inventory vs demand (dual axis), waste, RLE with annotation, SLCA bars."""
    op = data["results"]["overproduction"]
    ab = op["agribrain"]
    hours = np.array(ab["hours"])

    return _fig3_overproduction_inner(op, ab, hours)


def _fig3_overproduction_inner(op, ab, hours):
    """Body of fig 3. Extracted from ``fig3_overproduction`` so the
    per-figure font-size overrides applied above can be cleanly torn
    down via try/finally regardless of how the body returns or
    raises."""
    fig, axes = plt.subplots(2, 2, figsize=GRID_FIGSIZE)
    fig.suptitle("Overproduction & Reverse Logistics", y=SUPTITLE_Y)

    # --- (a) Inventory vs demand (dual y-axis) ---
    ax = axes[0, 0]
    inv = np.array(ab["inventory_outcome_environmental_trace"])
    dem = np.array(ab["demand_outcome_environmental_trace"])
    ax.plot(
        hours, inv, color=COLORS["agribrain"], linewidth=2.0,
        linestyle=LINESTYLES["agribrain"], marker=MARKERS["agribrain"],
        markevery=MARKER_EVERY, label="Inventory",
    )
    ax.set_xlabel("Time (hr.)")
    ax.set_ylabel("Inventory (units)")
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(3, 3))
    ax2 = ax.twinx()
    ax2.plot(
        hours, dem, color=COLORS["hybrid_rl"], linewidth=1.8,
        linestyle=LINESTYLES["hybrid_rl"], marker=MARKERS["hybrid_rl"],
        markevery=MARKER_EVERY, alpha=0.9, label="Demand",
    )
    ax2.set_ylabel("Demand (units/step)")
    ax.set_title("(a) Inventory vs Demand")
    _apply_style(ax)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(labelsize=TICK_FONT_SIZE, length=5, width=1.3)
    ax2.yaxis.label.set_size(AXIS_LABEL_SIZE)
    ax2.yaxis.label.set_weight("normal")
    for lbl in ax2.get_yticklabels():
        lbl.set_fontweight("normal")
    # Position the "Overproduction" label inside the red zone toward
    # the center-right (xpos\u224840) so the bounding box sits clearly
    # within the 12-60 h window without clipping the right edge.
    _annotate_window(ax, 12, 60, WINDOW_COLOR, "Overproduction", xpos=40)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    _panel_key(ax, handles=h1 + h2, labels=l1 + l2, ncol=2)

    # --- (b) Waste rolling average ---
    ax = axes[0, 1]
    window = 12
    for mode in ["static", "hybrid_rl", "agribrain"]:
        ep = op[mode]
        waste = np.array(ep["waste_trace"])
        rolling = _rolling_mean(waste, window)
        _mode_plot(ax, hours, rolling, mode)
    ax.set_xlabel("Time (hr.)")
    ax.set_ylabel("Waste Fraction")
    ax.set_title("(b) Waste Reduction over Time")
    _apply_style(ax)
    _annotate_window(ax, 12, 60, WINDOW_COLOR, "Overproduction")
    _panel_key(ax)

    # --- (c) RLE rolling (EU-hierarchy + severity-weighted) ---
    # Mirrors the canonical episode-level metric in
    # resilience.compute_rle / RLETracker, just with a rolling window
    # for visual continuity. Per at-risk timestep (rho > theta):
    #   numerator(t)   = rho(t) * w(action_t, rho_t)
    #   denominator(t) = rho(t) * w_max
    # where w is the declared rho-conditional hierarchy mapping,
    # qualitatively motivated by EU 2008/98/EC Article 4 and the food-waste
    # hierarchy literature. Numerator and denominator are accumulated with
    # the same trailing window so rolling RLE = num_rolling / den_rolling;
    # NaN where the window contains no at-risk steps.
    #
    # Only the declared rho-conditional hierarchy form is used here,
    # in resilience.compute_rle, in the benchmark JSONs, and in the table
    # CSVs: it is the same value carried by the headline RLE column.
    ax = axes[1, 0]
    action_names = ACTIONS  # canonical (cold_chain, local_redistribute, recovery)
    w_max = max(HIERARCHY_WEIGHT.values())
    for mode in ["static", "hybrid_rl", "agribrain"]:
        ep = op[mode]
        rho = np.array(ep["rho_outcome_environmental_trace"])
        actions = np.array(ep["action_trace"])
        at_risk = rho > RLE_THRESHOLD

        weighted_num = np.zeros_like(rho)
        weighted_den = np.zeros_like(rho)
        for t in range(len(rho)):
            if at_risk[t]:
                a = action_names[int(actions[t])]
                w = hierarchy_weight(a, float(rho[t]))
                weighted_num[t] = rho[t] * w
                weighted_den[t] = rho[t] * w_max

        num_rolling = _rolling_sum(weighted_num, window, centered=False)
        den_rolling = _rolling_sum(weighted_den, window, centered=False)
        # NaN where denominator is zero (no at-risk opportunities in window).
        rle_frac = np.full_like(num_rolling, np.nan)
        np.divide(num_rolling, den_rolling, out=rle_frac,
                  where=den_rolling > 0)
        _mode_plot(ax, hours, rle_frac, mode)

    # Mark threshold onset with a vertical guide and put the explanatory
    # text *inside* the axes (lower-left corner) instead of at the
    # title baseline, so it does not collide with the panel title.
    rho_ab = np.array(ab["rho_outcome_environmental_trace"])
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
            # Set to the LEFT of the marker. To its right the panel has only
            # about two points of slack: enough to clear the marker line or to
            # stay inside the frame, not both -- offset far enough to clear the
            # line and the box's right border overhangs the axes. To the left
            # the band is empty at this height, since RLE is undefined until
            # the first at-risk batch arrives at this very hour.
            xytext=(-14, 0), textcoords="offset points",
            ha="right", va="center", fontsize=ANNOT_FONT_SIZE - 1,
            fontweight="bold", color="#424242",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                      alpha=0.90, edgecolor="#9E9E9E", linewidth=0.8),
        )

    ax.set_xlabel("Time (hr.)")
    ax.set_ylabel("Severity-Weighted RLE")
    ax.set_title("(c) Reroute Quality over Time")
    ax.set_ylim(-0.05, 1.15)
    _apply_style(ax)
    # Center the "Overproduction" label at y = 0.4 in data coordinates
    # (per user request). Axes y-fraction = 0.375 because ylim is
    # (-0.05, 1.15) so (0.4 - (-0.05)) / 1.2 = 0.375. xpos sits it in the left
    # half of the shading, which is empty: RLE is undefined until the first
    # at-risk batch enters the rolling window around hr 47. Centred at 45 the
    # opaque box covered the hr-47 event line and the start of the hybrid-RL
    # trace, both of which begin exactly there.
    _annotate_window(ax, 12, 60, WINDOW_COLOR, "Overproduction",
                     ypos=0.375, xpos=28, va="center")
    # Legend at "center left": pre-h32 the panel is empty (RLE is
    # undefined until any at-risk batch enters the rolling window), so
    # the left half is clear headroom for the legend; vertical-center
    # placement keeps it clear of both the "first rho > 0.1 at h~32"
    # threshold-onset annotation in the lower band and the
    # "Overproduction" window label at the top.
    _panel_key(ax)

    # --- (d) Declared social-proxy component bars with cross-seed SE ---
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
    comp_labels = ["Inverse\nemissions", "Labour\npractice",
                   "Community\nnetwork", "Price\ninformation"]
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
                label=MODE_LABELS[mode], alpha=1.0, hatch=HATCHES[mode],
                edgecolor="#1F1F1F", linewidth=0.7,
                yerr=[1.96 * s for s in ses], capsize=4,
                error_kw={"linewidth": 1.0, "capthick": 1.0},
            )
        else:
            # Single-seed fallback: plot means alone (no fake CI bars).
            ep = op[mode]
            vals = [np.mean([s[comp] for s in ep["slca_component_trace"]])
                    for comp in components]
            ax.bar(x + i * width, vals, width, color=COLORS[mode],
                   label=MODE_LABELS[mode], alpha=1.0, hatch=HATCHES[mode],
                   edgecolor="#1F1F1F", linewidth=0.7)
    ax.set_xticks(x + width)
    ax.set_xticklabels(comp_labels)
    ax.set_ylabel("Social-Performance Proxy")
    ax.set_title("(d) Social-Proxy Components")
    ax.set_ylim(0, 1.15)
    _apply_style(ax)
    _panel_key(ax)

    _finish_grid(fig)
    _save(fig, "overproduction")


# ---------------------------------------------------------------------------
# Figure 4: Cyber Outage (1x3)
# ---------------------------------------------------------------------------
@panel_fonts(FOUR_PANEL_FONT_BUMP)
def fig4_cyber(data):
    """2x2: ARI over time, action distribution shift, reroute rate per method, KPI delta.

    Layout history: started 1-row (panel C single-pane action distribution)
    then briefly went to a 2-row gridspec (legend/bar overlap), then 1x4
    (visual mismatch with 2x2 figs 2/3/5), and as of late-May 2026 to a
    2x2 grid that matches figs 2/3/5. The descriptive sequence reads top-down
    AND left-right: top row = stimulus (ARI trace) + observed behavior
    (action distribution shift); bottom row = behavior magnitude per
    method (reroute rate) + KPI consequence per method (Δ ARI / Waste /
    Service). Each panel keeps its previous individual contents.
    """
    return _fig4_cyber_inner(data)


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
    # natural evidence grid (top row = stimulus + observed behavior,
    # bottom row = magnitude + outcome) and matches the reader's
    # left-to-right + top-to-bottom scan order in the other figures.
    fig, axes2d = plt.subplots(2, 2, figsize=GRID_FIGSIZE)
    # Flatten for legacy indexing (axes[0..3] corresponds to (a..d)
    # in row-major order: top-left, top-right, bottom-left, bottom-right).
    axes = axes2d.flatten()
    fig.suptitle("Cyber Outage Scenario Analysis", y=SUPTITLE_Y)

    # --- (a) ARI over time with outage shading ---
    # ARI = (1 - waste) * SLCA * (1 - rho). Spoilage risk rho rises
    # monotonically through every episode via the Arrhenius-lag ODE,
    # so the (1 - rho) factor pulls ARI downward over time for every
    # mode. The figure's story is therefore not the absolute level at
    # any one instant but the *gap* between AgriBrain and the baselines:
    # Any gap arises through the action-dependent waste and social-performance terms;
    # paired modes share the same latent environmental rho.
    ax = axes[0]
    for mode in ["static", "hybrid_rl", "agribrain"]:
        ep = cy[mode]
        ari = np.array(ep["ari_trace"])
        rolling = _rolling_mean(ari, 12)
        _mode_plot(ax, hours, rolling, mode)
    ax.set_xlabel("Time (hr.)")
    ax.set_ylabel("Adaptive Resilience Index")
    ax.set_title("(a) Adaptive Resilience Index over Time")
    _apply_style(ax)
    # Top-anchored, as everywhere else. This was bottom-anchored to dodge a
    # legend that used to sit inside the upper-left quadrant; the legend has
    # since moved out of the axes into a key above them, and the bottom is now
    # the crowded end -- the badge's opaque box was covering a stretch of the
    # static trace as it declines through the outage.
    _annotate_window(ax, 24, 72, WINDOW_COLOR, "Outage")
    _panel_key(ax)

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

    ax.bar(bar_x - width / 2, pre_counts, width, color="#332288",
           alpha=1.0, label="Pre-outage", hatch=PERIOD_HATCHES["before"],
           edgecolor="#1F1F1F", linewidth=0.7,
           yerr=1.96 * pre_se, capsize=4,
           error_kw={"linewidth": 1.2, "capthick": 1.2})
    ax.bar(bar_x + width / 2, during_counts, width, color=WINDOW_COLOR,
           alpha=1.0, label="During outage", hatch=PERIOD_HATCHES["during"],
           edgecolor="#1F1F1F", linewidth=0.7,
           yerr=1.96 * during_se, capsize=4,
           error_kw={"linewidth": 1.2, "capthick": 1.2})
    ax.set_xticks(bar_x)
    ax.set_xticklabels(action_names)
    ax.set_ylabel("Fraction of Routing Decisions")
    ax.set_ylim(0, max(max(pre_counts + pre_se * 2), max(during_counts + during_se * 2)) * 1.25 + 0.02)
    ax.set_title("(b) Action Distribution Shift")
    _apply_style(ax)
    _panel_key(ax)

    # --- (c,d) Descriptive behavior and outcome panels ---
    # Pre/during windows are split at the declared outage onset (h=24).
    # Panel (c) reports reroute fractions; panel (d) reports absolute
    # during-outage levels. Their temporal juxtaposition is descriptive and is
    # not presented as causal identification.
    pre_mask_arr = np.asarray(hours, dtype=float) < 24.0
    during_mask_arr = np.asarray(hours, dtype=float) >= 24.0
    modes_ordered_c = ["static", "hybrid_rl", "agribrain"]
    mode_labels_c = ["Static", "Hybrid RL", "AGRI-BRAIN"]
    # Distinct, color-blind-friendly mode palette consistent with the
    # rest of the figure.
    mode_colors_c = {mode: COLORS[mode] for mode in modes_ordered_c}

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
    # Standard errors for the during-outage means. For ARI and Waste,
    # the single-trajectory development fallback uses std/sqrt(n) on
    # during-window samples. Because those steps are autocorrelated, this
    # fallback is descriptive and can understate uncertainty; publication
    # figures use the seed-level path below. For Service the metric is a product
    # (retail_dispatch * (1 - mean_waste)) and the analytic SE
    # requires the delta method, so we bootstrap-resample
    # during-window steps 2000x and take the std of the bootstrap
    # level distribution. The pre-vs-during delta construction was
    # retired in 2026-05 because levels avoid a headroom-dependent delta
    # interpretation. Any cross-method ordering is supplied only by validated
    # validated inputs; it is not assumed by this plotting routine.
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
        # Service-level indicator: retail-dispatch rate * (1 - mean waste).
        # See panel docstring above for the operations-research
        # interpretation. This explicitly declared scalar decreases when the
        # policy selects recovery and when modeled waste increases.
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
    # The behavior-magnitude panel. Static is the
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
        color="#332288", alpha=1.0, hatch=PERIOD_HATCHES["before"],
        edgecolor="#1F1F1F", linewidth=0.7,
        label="Pre-outage",
        yerr=1.96 * np.asarray(reroute_pre_se), capsize=4,
        error_kw={"linewidth": 1.2, "capthick": 1.2, "ecolor": "#1F1F1F"},
    )
    ax_c.bar(
        x_modes + bar_w / 2, reroute_during, bar_w,
        color=WINDOW_COLOR, alpha=1.0, hatch=PERIOD_HATCHES["during"],
        edgecolor="#1F1F1F", linewidth=0.7,
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
    _panel_key(ax_c)

    # ---- (d) KPI levels during outage per method ----
    # The outcome panel reports absolute during-outage levels. The pre-vs-during delta
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
            color=mode_colors_c[mode], alpha=1.0, hatch=HATCHES[mode],
            edgecolor="#1F1F1F", linewidth=0.7,
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
    _panel_key(ax_d)

    _finish_grid(fig)
    _save(fig, "cyber_outage")


# ---------------------------------------------------------------------------
# Figure 5: Pricing Volatility (2x2)
# ---------------------------------------------------------------------------
@panel_fonts(FOUR_PANEL_FONT_BUMP)
def fig5_pricing(data):
    """2x2: demand+Bollinger, routing fractions, equity, reward components."""
    ap = data["results"]["adaptive_pricing"]
    ab = ap["agribrain"]
    hours = np.array(ab["hours"])

    return _fig5_pricing_inner(ap, ab, hours)


def _fig5_pricing_inner(ap, ab, hours):
    """Body of fig 5. Extracted from ``fig5_pricing`` so the per-figure
    font-size overrides applied above can be cleanly torn down via
    try/finally regardless of how the body returns or raises."""
    fig, axes = plt.subplots(2, 2, figsize=GRID_FIGSIZE)
    fig.suptitle("Adaptive Pricing & Demand Volatility", y=SUPTITLE_Y)

    # --- (a) Demand + Bollinger triggers ---
    ax = axes[0, 0]
    demand = np.array(ab["demand_trace"])
    policy = Policy()
    window = int(policy.boll_window)
    rolling_mean = _rolling_mean(demand, window, centered=False)
    rolling_std = np.array([
        np.std(demand[max(0, i - window + 1):i + 1], ddof=1)
        if i - max(0, i - window + 1) + 1 > 1 else 0.0
        for i in range(len(demand))
    ])
    upper = rolling_mean + float(policy.boll_k) * rolling_std
    lower = rolling_mean - float(policy.boll_k) * rolling_std

    ax.plot(hours, demand, color="#37474F", linewidth=1.2, alpha=0.8,
            linestyle="-", label="Demand")
    ax.plot(hours, rolling_mean, color=COLORS["agribrain"], linewidth=2.0,
            linestyle="--", marker=MARKERS["agribrain"],
            markevery=MARKER_EVERY, label="Bollinger mean")
    ax.fill_between(hours, lower, upper, alpha=0.18, color=COLORS["agribrain"],
                    label="\u00b12\u03c3 band", linewidth=0)
    triggers = np.abs(demand - rolling_mean) > float(policy.boll_k) * rolling_std
    ax.scatter(hours[triggers], demand[triggers], color=WINDOW_COLOR, s=42,
               zorder=5, label="Trigger", marker="v",
               edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Time (hr.)")
    ax.set_ylabel("Demand (units/step)")
    ax.set_title("(a) Demand with Bollinger Triggers")
    _apply_style(ax)
    _panel_key(ax)

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

    ax.bar(bin_centers, cc_fracs, bar_w, color=ACTION_COLORS["cold_chain"],
           alpha=1.0, label="Cold Chain", hatch=ACTION_HATCHES["cold_chain"],
           edgecolor="#1F1F1F", linewidth=0.7)
    ax.bar(bin_centers, lr_fracs, bar_w, bottom=cc_fracs,
           color=ACTION_COLORS["local_redistribution"], alpha=1.0,
           label="Local Redist.", hatch=ACTION_HATCHES["local_redistribution"],
           edgecolor="#1F1F1F", linewidth=0.7)
    ax.bar(bin_centers, rec_fracs, bar_w, bottom=cc_fracs + lr_fracs,
           color=ACTION_COLORS["recovery"], alpha=1.0, label="Recovery",
           hatch=ACTION_HATCHES["recovery"], edgecolor="#1F1F1F",
           linewidth=0.7)
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
    _panel_key(ax, ncol=3)

    # --- (c) Temporal social-performance stability proxy ---
    # Auto-scale across the three modes; the previous fixed y-range
    # (0.70-1.02) clipped Static and Hybrid RL when their quality-weighted
    # equity sat below 0.70, hiding the very gap the figure is supposed to
    # show. We compute a tight-but-honest y-range from the data instead.
    ax = axes[1, 0]
    eq_curves = {}
    for mode in ["static", "hybrid_rl", "agribrain"]:
        ep = ap[mode]
        eq = np.array(ep["equity_trace"])
        rolling = _rolling_mean(eq, 12)
        _mode_plot(ax, hours, rolling, mode)
        eq_curves[mode] = rolling
    ax.set_xlabel("Time (hr.)")
    ax.set_ylabel("Proxy Stability")
    ax.set_title("(c) Proxy Stability over Time")
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
    _panel_key(ax)

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
    # for each mode. The 12-step mean is a display smoother only.
    ax = axes[1, 1]
    window = 12  # 12 steps * 0.25 h = 3 h rolling
    for mode in ["static", "hybrid_rl", "agribrain"]:
        ep = ap[mode]
        reward = np.array(ep["reward_trace"])
        rolling = _rolling_mean(reward, window)
        _mode_plot(ax, hours, rolling, mode)

    ax.set_xlabel("Time (hr.)")
    ax.set_ylabel("Reward")
    ax.set_title("(d) Per-Step Reward Comparison")
    _apply_style(ax)
    # Match panel (c)'s legend placement (lower-center, lifted ~10 %
    # off the x-axis) so the two bottom-row panels read symmetrically.
    # The reward traces stay above ~0.50 across the interior hours, so
    # the lifted lower-center anchor is clear of all three lines.
    _panel_key(ax)

    _finish_grid(fig)
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
    + severity-weighted form), and a retired capacity-constrained variant.
    Only the
    EU-hierarchy + severity-weighted form survived the simplification —
    it now lives under the plain key ``rle`` in
    ``resilience.compute_rle`` and in current aggregator output.

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


def _discover_seed_files() -> list[Path]:
    """Resolve exactly one seed-artifact scope without mixing run tags.

    Publication rendering sets ``FIGURE_SEED_ROOT`` to the validated tagged
    run. Local development may use flat ``benchmark_seeds/`` files or one
    unambiguous tagged directory. Multiple tagged directories without an
    explicit selection are rejected rather than merged.
    """
    explicit = os.environ.get("FIGURE_SEED_ROOT", "").strip()
    seeds_root = Path(explicit) if explicit else RESULTS_DIR / "benchmark_seeds"
    if not seeds_root.exists():
        return []
    flat = sorted(seeds_root.glob("seed_*.json"))
    if explicit or flat:
        return flat
    run_tag = os.environ.get("RUN_TAG", "").strip()
    if run_tag:
        tagged = seeds_root / run_tag
        if tagged.is_dir():
            return sorted(tagged.glob("seed_*.json"))
    candidates = [
        sorted(path.glob("seed_*.json"))
        for path in sorted(seeds_root.iterdir()) if path.is_dir()
    ]
    candidates = [files for files in candidates if files]
    if len(candidates) > 1:
        raise RuntimeError(
            "ambiguous benchmark seed cache: multiple tagged runs exist; "
            "set FIGURE_SEED_ROOT to one validated run directory"
        )
    return candidates[0] if candidates else []


def _load_seed_payloads() -> list[tuple[Path, dict]]:
    """Load the selected seed scope and enforce publication identity."""
    files = _discover_seed_files()
    strict = os.environ.get("STRICT_VALIDATION", "0") == "1"
    expected_raw = os.environ.get("BENCHMARK_SEEDS", "").strip()
    expected = (
        {int(value) for value in expected_raw.split(",") if value.strip()}
        if expected_raw else set()
    )
    found_names = {path.name for path in files}
    if expected:
        expected_names = {f"seed_{seed}.json" for seed in expected}
        if found_names != expected_names:
            raise RuntimeError(
                "figure seed inventory mismatch: "
                f"missing={sorted(expected_names - found_names)}, "
                f"unexpected={sorted(found_names - expected_names)}"
            )
    payloads: list[tuple[Path, dict]] = []
    seen: set[int] = set()
    run_tag = os.environ.get("RUN_TAG", "").strip()
    commit = os.environ.get("AGRIBRAIN_GIT_COMMIT", "").strip()
    for path in files:
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            if strict:
                raise RuntimeError(f"invalid figure seed envelope {path}: {exc}") from exc
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("seed"), int):
            if strict:
                raise RuntimeError(f"invalid figure seed envelope schema: {path}")
            continue
        seed = int(obj["seed"])
        if path.name != f"seed_{seed}.json" or seed in seen:
            raise RuntimeError(f"duplicate or misnamed figure seed envelope: {path}")
        seen.add(seed)
        if strict:
            meta = obj.get("_meta")
            if not isinstance(meta, dict):
                raise RuntimeError(f"figure seed envelope has no provenance: {path}")
            if commit and meta.get("source_commit") != commit:
                raise RuntimeError(f"figure seed envelope commit mismatch: {path}")
            if run_tag and meta.get("run_tag") != run_tag:
                raise RuntimeError(f"figure seed envelope run-tag mismatch: {path}")
            traces = obj.get("traces")
            if not isinstance(traces, dict) or set(traces) != set(SCENARIOS):
                raise RuntimeError(
                    f"figure seed envelope lacks the exact scenario traces: {path}"
                )
            for scenario in SCENARIOS:
                mode_panel = traces.get(scenario)
                if not isinstance(mode_panel, dict) or set(mode_panel) != set(
                    CANONICAL_TRACE_MODES
                ):
                    raise RuntimeError(
                        f"figure seed trace-mode panel mismatch: {path}/{scenario}"
                    )
                for mode in CANONICAL_TRACE_MODES:
                    try:
                        validate_trace_cell(
                            mode_panel[mode],
                            where=f"{path}:{scenario}/{mode}",
                        )
                    except ValueError as exc:
                        raise RuntimeError(str(exc)) from exc
        payloads.append((path, obj))
    if strict and expected and seen != expected:
        raise RuntimeError("loaded figure seed IDs differ from the declared panel")
    return payloads


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
    seed_payloads = _load_seed_payloads()
    if not seed_payloads:
        return None
    # all_data[seed][scenario][mode][metric] = float
    all_data: dict = {}
    for _path, obj in seed_payloads:
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
    seed_payloads = _load_seed_payloads()
    if not seed_payloads:
        return None

    strict = os.environ.get("STRICT_VALIDATION", "0") == "1"
    arrs: list[np.ndarray] = []
    for path, obj in seed_payloads:
        traces = obj.get("traces") if isinstance(obj, dict) else None
        if not isinstance(traces, dict):
            if strict:
                raise RuntimeError(f"missing trace block: {path}")
            continue
        cell = traces.get(scenario, {}).get(mode, {})
        if not isinstance(cell, dict):
            if strict:
                raise RuntimeError(f"missing trace cell: {path}/{scenario}/{mode}")
            continue
        seq = cell.get(field)
        if not isinstance(seq, list) or not seq:
            if strict:
                raise RuntimeError(
                    f"missing figure trace: {path}/{scenario}/{mode}/{field}"
                )
            continue
        try:
            array = np.asarray(seq, dtype=float)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"nonnumeric figure trace: {path}/{scenario}/{mode}/{field}"
            ) from exc
        if strict and array.shape[0] != TRACE_LENGTH:
            raise RuntimeError(
                f"incomplete figure trace: {path}/{scenario}/{mode}/{field}"
            )
        arrs.append(array)
    if not arrs:
        return None
    if strict and len(arrs) != len(seed_payloads):
        raise RuntimeError(
            f"figure trace {scenario}/{mode}/{field} is not present for every seed"
        )
    # Exploratory rendering may drop rare seeds whose trace length disagrees with the modal
    # length (truncated runs). The mode is taken as the most common
    # length across the seeds we collected.
    lengths = [a.shape[0] for a in arrs]
    if not lengths:
        return None
    n = max(set(lengths), key=lengths.count)
    if strict and any(a.shape != arrs[0].shape for a in arrs):
        raise RuntimeError(
            f"figure trace {scenario}/{mode}/{field} has inconsistent shapes"
        )
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
    strict = os.environ.get("STRICT_VALIDATION", "0") == "1"
    if a is None or ari is None or waste is None:
        if strict:
            raise RuntimeError(f"missing strict window traces for {scenario}/{mode}")
        return None
    if a.shape[0] < 2:
        if strict:
            raise RuntimeError(f"strict window traces have fewer than 20 seeds: {scenario}/{mode}")
        return None
    if a.shape[0] != ari.shape[0] or a.shape[0] != waste.shape[0]:
        if strict:
            raise RuntimeError(f"strict window trace seed counts differ: {scenario}/{mode}")
        return None
    if strict and a.shape[0] != 20:
        raise RuntimeError(f"strict window traces require exactly 20 seeds: {scenario}/{mode}")
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
    seed_payloads = _load_seed_payloads()
    if not seed_payloads:
        return None

    components = ("C", "L", "R", "P")
    per_seed: dict[str, list[float]] = {c: [] for c in components}

    strict = os.environ.get("STRICT_VALIDATION", "0") == "1"
    for path, obj in seed_payloads:
        traces = obj.get("traces") if isinstance(obj, dict) else None
        if not isinstance(traces, dict):
            if strict:
                raise RuntimeError(f"missing SLCA trace block: {path}")
            continue
        cell = traces.get(scenario, {}).get(mode, {})
        seq = cell.get("slca_component_trace")
        if not isinstance(seq, list) or not seq:
            if strict:
                raise RuntimeError(
                    f"missing SLCA component trace: {path}/{scenario}/{mode}"
                )
            continue
        # Older flat list[float] shape (pre-2026-05) -- skip rather
        # than try to interpret.
        if not isinstance(seq[0], dict):
            if strict:
                raise RuntimeError(
                    f"invalid SLCA component trace: {path}/{scenario}/{mode}"
                )
            continue
        for c in components:
            vals = [float(s[c]) for s in seq if c in s]
            if vals:
                per_seed[c].append(float(np.mean(vals)))

    # Need at least 2 seeds for a meaningful cross-seed SE.
    if strict and any(len(per_seed[c]) != 20 for c in components):
        raise RuntimeError(
            f"strict SLCA components require exactly 20 seeds: {scenario}/{mode}"
        )
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
    indicator of within-mode uncertainty — different scenarios are expected to
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
    """Horizontal, two-line scenario names that never overlap."""
    ax.set_xticklabels(
        [SCENARIO_TICKS[s] for s in scenarios_plot],
        rotation=0, ha="center",
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
# ARI per modeled emissions indicator — retained as an exploratory helper and not used in the
# canonical publication figures.
#
#     CE = mean ARI / episode Carbon_kg
#        [ARI·kg⁻¹ CO2e]
#
# ARI per unit modeled transport-emissions indicator: higher values
# reflect higher ARI, lower episode carbon, or both.
def _carbon_efficiency_value(ep: dict) -> float:
    """Exploratory ratio: mean ARI / cumulative modeled indicator."""
    ari = float(ep.get("ari", 0.0))
    carbon = float(ep.get("carbon", 0.0))
    return compute_carbon_efficiency(ari, carbon)


def _carbon_efficiency_yerr(bench: dict | None, scenarios: list[str],
                            mode: str) -> np.ndarray | None:
    """Return paired-seed BCa bounds for the exploratory ARI/carbon ratio.

    ``run_single_seed.py`` computes the ratio within each matched seed, and
    ``aggregate_seeds.py`` bootstraps those ratios directly.  This preserves
    numerator/denominator covariance.  Reconstructing uncertainty from the
    two marginal ARI and carbon intervals (the former implementation) is not
    a valid confidence interval for their ratio and is intentionally refused.
    """
    if not bench:
        return None
    means, lows, highs = [], [], []
    for s in scenarios:
        rec = bench.get(s, {}).get(mode, {}).get(
            "carbon_efficiency_ari_per_kgco2e_proxy", {}
        )
        if not rec or rec.get("ci_method") != "BCa":
            return None
        mean, low, high = rec.get("mean"), rec.get("ci_low"), rec.get("ci_high")
        if any(value is None for value in (mean, low, high)):
            return None
        means.append(float(mean)); lows.append(float(low)); highs.append(float(high))
    means_a = np.asarray(means)
    return np.vstack([
        np.maximum(0.0, means_a - np.asarray(lows)),
        np.maximum(0.0, np.asarray(highs) - means_a),
    ])


@panel_fonts(FOUR_PANEL_FONT_BUMP)
def fig6_cross(data):
    """2x2 grouped bars: ARI, RLE, waste, SLCA across scenarios for 3 methods.
    Error bars are drawn from (in order): benchmark_summary.json bootstrap
    CIs, benchmark_seeds/ per-seed std, or the per-step trace std as a
    last-resort within-episode fallback."""
    bench = _load_benchmark_ci()

    fig, axes = plt.subplots(2, 2, figsize=GRID_FIGSIZE)
    # suptitle is applied at the end with the larger fig6-specific font.

    # Local aliases of the enlarged sizes the decorator has already put in
    # place. They are captured here and re-applied after _apply_style, which
    # would otherwise reset the axis text to the canonical sizes.
    _F6_TITLE = SUBPLOT_TITLE_SIZE
    _F6_AXIS  = AXIS_LABEL_SIZE
    _F6_TICK  = TICK_FONT_SIZE
    _F6_LEG   = LEGEND_FONT_SIZE

    # Single canonical RLE: author-defined, hierarchy-inspired,
    # severity-weighted form (resilience.compute_rle). It scores at-risk
    # routing against declared rho-conditional action weights and is not a
    # measure of legal or regulatory conformity. Reported here for the
    # 3-method cross-scenario view; the
    # 5-mode capability-ablation view (static / hybrid_rl / no_slca /
    # no_context / agribrain) appears in fig 7 panel C with the same
    # metric.
    # Panel titles are deliberately distinct from y-axis labels so the
    # title carries the comparison/interpretation while the y-axis names
    # the metric.
    metrics = [
        ("ari",   "Adaptive Resilience Index",   "(a)", "Cross-Scenario Resilience Ranking"),
        ("rle",   "Severity-Weighted RLE", "(b)", "Defensive Routing Effectiveness"),
        ("waste", "Waste Fraction", "(c)", "Waste across Stressors"),
        ("slca",  "Social-Performance Proxy", "(d)", "Social Proxy by Method"),
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
                   label=MODE_LABELS[mode], alpha=1.0, hatch=HATCHES[mode],
                   edgecolor="#1F1F1F", linewidth=0.7, yerr=yerr,
                   capsize=_ERR_CAPSIZE if yerr is not None else 0,
                   error_kw=_ERR_KW)

        ax.set_xticks(x + width)
        _bar_xticklabels(ax, scenarios_plot)
        ax.set_ylabel(ylabel, fontsize=_F6_AXIS, fontweight="normal")
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
            lbl.set_fontweight("normal")
        for lbl in ax.get_yticklabels():
            lbl.set_fontsize(_F6_TICK)
            lbl.set_fontweight("normal")
        ax.yaxis.label.set_size(_F6_AXIS)
        ax.yaxis.label.set_weight("normal")
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
        text.set_fontweight("normal")
    fig.suptitle("Cross-Scenario Performance Comparison", y=0.995,
                 fontsize=FIG_TITLE_SIZE + 2, fontweight="bold")
    _finish_grid(fig, bottom=0.06)
    _save(fig, "cross_scenario")


# ---------------------------------------------------------------------------
# Figure 7: Ablation study (1x3 grouped bars)
# ---------------------------------------------------------------------------
def fig7_ablation(data):
    """1x3 grouped bars: ARI, waste, and carbon for the architectural
    ablation. Shows the six architectural modes (static, hybrid_rl, no_pinn,
    no_slca, no_context, agribrain); AgriBrain is plotted last so it sits as the rightmost
    bar in each group.

    The separate single-channel arms (mcp_only and pirag_only) are covered by
    the H2 context-channel figure, keeping this the canonical six-mode
    capability view.
    """
    bench = _load_benchmark_ci()

    # 6-mode architectural ablation: each mode strips one structural
    # capability from the stack vs full-stack AgriBrain.
    #   static     - fixed policy, no learning or external context
    #   hybrid_rl  - learned base-policy correction, no external context
    #   no_pinn    - full stack with the frozen residual disabled
    #   no_slca    - full stack minus social-performance logit shaping
    #   no_context - same learned policy without external MCP/retrieval context
    #   agribrain  - full stack
    # The single-channel context arms (mcp_only and pirag_only) are excluded
    # here so fig 7 stays
    # focused on the *capability* dimension; the channel contribution
    # is analysed at the decision level in the H2 context-channel
    # figure (fig12).
    _FIG7_CANONICAL_MODES = (
        "static", "hybrid_rl", "no_pinn", "no_slca", "no_context",
        "agribrain",
    )
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
    fig, axes = plt.subplots(1, 3, figsize=TRIPTYCH_FIGSIZE)
    # suptitle is applied at the end of the function with the larger
    # fig7-specific font; placeholder kept here so layout calculations
    # leave headroom even if the suite-wide rcParams are inspected.

    # Panel C reports the emissions indicator directly; no derived
    # ratio is introduced solely for presentation.
    # Panel titles are deliberately distinct from y-axis labels so the
    # title carries the ablation interpretation while the y-axis names
    # the metric.
    metrics = [
        ("ari",   "Adaptive Resilience Index",   "(a)", "Resilience across Modes"),
        ("waste", "Waste Fraction", "(b)", "Spoilage Sensitivity"),
        ("carbon", "Emissions Indicator",
         "(c)", "Emissions across Modes"),
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
    # This figure used to be drawn on a 24-inch canvas and compensated with a
    # 1.333x type scale so that it matched the others once the page shrank it.
    # It is now the same 18 inches wide as every other figure, so it uses the
    # shared sizes directly and no longer prints with oversized axis text.
    _F7_TITLE = SUBPLOT_TITLE_SIZE
    _F7_AXIS  = AXIS_LABEL_SIZE
    _F7_TICK  = TICK_FONT_SIZE
    _F7_LEG   = LEGEND_FONT_SIZE

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
                if yerr is not None:
                    vals = [
                        bench[s][mode][
                            "carbon_efficiency_ari_per_kgco2e_proxy"
                        ]["mean"]
                        for s in stress_scenarios
                    ]
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
                   label=MODE_LABELS[mode], alpha=1.0, hatch=HATCHES[mode],
                   edgecolor="#1F1F1F", linewidth=0.7, yerr=yerr,
                   capsize=_ERR_CAPSIZE if yerr is not None else 0,
                   error_kw=_ERR_KW)

        ax.set_xticks(x + (n_modes - 1) * width / 2)
        _bar_xticklabels(ax, stress_scenarios)
        ax.set_ylabel(ylabel, fontsize=_F7_AXIS, fontweight="normal")
        ax.set_title(f"{panel} {title}", fontsize=_F7_TITLE, fontweight="bold")
        _apply_style(ax)
        # Re-apply the larger tick label size after _apply_style.
        ax.tick_params(labelsize=_F7_TICK, length=6, width=1.4)
        for lbl in ax.get_xticklabels():
            lbl.set_fontsize(_F7_TICK)
            lbl.set_fontweight("normal")
        for lbl in ax.get_yticklabels():
            lbl.set_fontsize(_F7_TICK)
            lbl.set_fontweight("normal")
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
        ax.yaxis.label.set_weight("normal")
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
        text.set_fontweight("normal")
    # Suptitle scales with the larger panel typography: ~1.333x the
    # +2 canonical suptitle (28) = 37, matching the 24in width.
    fig.suptitle("Ablation Study", y=SUPTITLE_Y, fontsize=FIG_TITLE_SIZE,
                 fontweight="bold")
    _finish_strip(fig, bottom=0.13)
    _save(fig, "ablation")


# ---------------------------------------------------------------------------
# Figure 8: Modeled transport-emissions indicator (1x2)
# ---------------------------------------------------------------------------
def fig8_transport_emissions(data):
    """1x2: cumulative emissions indicator and route-indicator totals with CIs.

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

    fig, axes = plt.subplots(1, 2, figsize=PAIR_FIGSIZE)

    # Per-element font sizes for this 1x2 figure: a +3 / +2 / +2 / +2
    # title/axis/tick/legend cascade so titles read as the dominant
    # element and ticks/legend stay readable on the (18, 7.5) figsize.
    _F8_TITLE = SUBPLOT_TITLE_SIZE
    _F8_AXIS  = AXIS_LABEL_SIZE
    _F8_TICK  = TICK_FONT_SIZE
    _F8_LEG   = LEGEND_FONT_SIZE

    hw = data["results"]["heatwave"]
    hours = np.array(hw["agribrain"]["hours"])

    # --- (a) Cumulative emissions indicator for heatwave scenario ---
    ax = axes[0]
    fig8a_modes = ["static", "hybrid_rl", "agribrain"]
    for mode in fig8a_modes:
        ep = hw[mode]
        cum_carbon = np.cumsum(ep["carbon_trace"])
        _mode_plot(ax, hours, cum_carbon, mode)
    ax.set_xlabel("Time (hr.)", fontsize=_F8_AXIS, fontweight="normal")
    ax.set_ylabel("Cumulative Emissions",
                  fontsize=_F8_AXIS, fontweight="normal")
    ax.set_title("(a) Cumulative Emissions \u2014 Heatwave",
                 fontsize=_F8_TITLE, fontweight="bold", pad=14)
    _apply_style(ax)
    _annotate_window(ax, 24, 48, WINDOW_COLOR, "Heatwave", ypos=0.55,
                     fontsize=ANNOT_FONT_SIZE)
    _panel_key(ax, ncol=len(fig8a_modes), fontsize=_F8_LEG)
    ax.tick_params(labelsize=_F8_TICK, length=6, width=1.4)
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lbl.set_fontsize(_F8_TICK); lbl.set_fontweight("normal")
    # Re-apply axis-label + title sizes after _apply_style (which resets
    # them to the un-bumped canonical because fig8 does not bump globals).
    ax.xaxis.label.set_size(_F8_AXIS); ax.yaxis.label.set_size(_F8_AXIS)
    ax.title.set_size(_F8_TITLE)

    # --- (b) Emissions-indicator bar chart across all scenarios ---
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
               label=MODE_LABELS[mode], alpha=1.0, hatch=HATCHES[mode],
               edgecolor="#1F1F1F", linewidth=0.7, yerr=yerr,
               capsize=_ERR_CAPSIZE if yerr is not None else 0,
               error_kw=_ERR_KW)

    ax.set_xticks(x + width)
    _bar_xticklabels(ax, scenarios_plot)
    ax.set_ylabel("Emissions Indicator",
                  fontsize=_F8_AXIS, fontweight="normal")
    ax.set_title("(b) Emissions by Scenario",
                 fontsize=_F8_TITLE, fontweight="bold", pad=14)
    _apply_style(ax)
    _panel_key(ax, ncol=len(methods_plot), fontsize=_F8_LEG)
    ax.tick_params(labelsize=_F8_TICK, length=6, width=1.4)
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lbl.set_fontsize(_F8_TICK); lbl.set_fontweight("normal")
    # Re-apply axis-label + title sizes after _apply_style (which resets
    # them to the un-bumped canonical because fig8 does not bump globals).
    ax.xaxis.label.set_size(_F8_AXIS); ax.yaxis.label.set_size(_F8_AXIS)
    ax.title.set_size(_F8_TITLE)

    fig.suptitle("Modeled Transport-Emissions Indicator", y=SUPTITLE_Y,
                 fontsize=FIG_TITLE_SIZE, fontweight="bold")
    _finish_strip(fig)
    _save(fig, "transport_emissions")




@panel_fonts(FOUR_PANEL_FONT_BUMP)
def fig11_performance_efficiency(data=None):
    """H1 plus descriptive comparators and efficiency. (a) Effect-size
    heatmap for the prespecified H1 contrast and two descriptive baselines,
    (b) relative ARI difference, (c) lightweight latency
    frontier (broken x), (d) context-aware latency frontier (broken x, green
    trend line). Reads benchmark_significance.json + benchmark_summary.json."""
    import matplotlib.gridspec as _gridspec
    from matplotlib.colors import LinearSegmentedColormap as _LSC
    from matplotlib.colors import TwoSlopeNorm as _TwoSlopeNorm
    from matplotlib.ticker import FormatStrFormatter as _FmtStr
    from matplotlib.ticker import MaxNLocator as _MaxNLocator

    sig_p = RESULTS_DIR / "benchmark_significance.json"
    summ_p = RESULTS_DIR / "benchmark_summary.json"
    if not (sig_p.exists() and summ_p.exists()):
        print("  [fig11] missing significance/summary JSON; skipped")
        return
    sig = json.loads(sig_p.read_text())["significance"]
    summ = json.loads(summ_p.read_text()); summ = summ.get("summary", summ)

    SCEN = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing", "baseline"]
    SLAB = SCENARIO_LABELS
    BASELINES = [
        ("static", "vs Static"),
        ("hybrid_rl", "vs Hybrid RL"),
        ("no_context", "vs No-context"),
    ]
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
                            markerfacecolor="white" if m in ref else COLORS[m],
                            markeredgecolor=COLORS[m] if m in ref else "white",
                            markeredgewidth=1.4, capsize=4, elinewidth=1.8,
                            alpha=1.0, label=MODE_LABELS[m], zorder=5)
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
        axl.set_ylabel("Mean ARI (\u00b1SE)")
        hh_, ll_ = [], []
        for a in (axl, axr):
            for h, l in zip(*a.get_legend_handles_labels()):
                if l not in ll_:
                    hh_.append(h); ll_.append(l)
        # Three across keeps the key inside the pair's own width, so it
        # never spills sideways into the neighbouring panel, and the anchor
        # centres it on the pair rather than on the wider right sub-axes.
        _pl, _pr = axl.get_position(), axr.get_position()
        _panel_key(axr, handles=hh_, labels=ll_, ncol=min(len(ll_), 3),
                   bbox_to_anchor=((_pl.x0 + _pr.x1) / 2, _pr.y1),
                   bbox_transform=fig.transFigure)
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

    fig = plt.figure(figsize=GRID_FIGSIZE)
    # tight_layout cannot reflow the broken-axis sub-gridspecs in row 2, so
    # this figure states its geometry directly. The generous row gap is what
    # the row-2 panel titles and keys sit in, so it has to grow with them: the
    # ratio reproduces the tuned 0.46 exactly at the canonical title and key
    # sizes and opens up from there.
    outer = _gridspec.GridSpec(
        2, 2, figure=fig, height_ratios=[1, 1],
        left=0.075, right=0.985, top=_F11_GRID_TOP, bottom=0.085,
        hspace=0.62 * (SUBPLOT_TITLE_SIZE + LEGEND_FONT_SIZE) / 40.0,
        wspace=0.24,
    )
    axA = fig.add_subplot(outer[0, 0]); axB = fig.add_subplot(outer[0, 1])

    d_mat = np.full((len(scen), len(BASELINES)), np.nan)
    for i, s in enumerate(scen):
        for j, (b, _) in enumerate(BASELINES):
            c = sig[s].get(f"agribrain_vs_{b}", {}).get("ari", {})
            v = c.get("cohens_dz", c.get("cohens_d"))
            if v is not None:
                d_mat[i, j] = float(v)
    _dmax = max(float(np.nanmax(np.abs(d_mat))), 1e-6)
    effect_cmap = _LSC.from_list(
        "agribrain_signed_effect", ["#B35806", "#F7F7F7", "#542788"],
    )
    axA.pcolormesh(
        np.arange(len(BASELINES) + 1) - 0.5,
        np.arange(len(scen) + 1) - 0.5,
        d_mat,
        cmap=effect_cmap,
        norm=_TwoSlopeNorm(vmin=-_dmax, vcenter=0.0, vmax=_dmax),
        shading="flat",
        rasterized=False,
        edgecolors="white",
        linewidth=0.4,
    )
    axA.set_xlim(-0.5, len(BASELINES) - 0.5)
    axA.set_ylim(len(scen) - 0.5, -0.5)
    axA.set_xticks(range(len(BASELINES))); axA.set_xticklabels([l for _, l in BASELINES])
    axA.set_yticks(range(len(scen))); axA.set_yticklabels([SLAB[s] for s in scen])
    for i in range(len(scen)):
        for j in range(len(BASELINES)):
            v = d_mat[i, j]
            if not np.isnan(v):
                axA.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=ANNOT_FONT_SIZE,
                         fontweight="bold", color="white" if abs(v) > 0.65 * _dmax else "#1F1F1F")
    axA.set_title("(a) Paired Effect Sizes (d$_z$)")
    axA.grid(False); axA.tick_params(length=0)
    for sp in axA.spines.values():
        sp.set_visible(False)
    for lbl in axA.get_xticklabels() + axA.get_yticklabels():
        lbl.set_fontweight("normal")

    impr, lo, hi, dmed, cols = [], [], [], [], []
    for b, _ in BASELINES:
        vals, ds = [], []
        for s in scen:
            c = sig[s].get(f"agribrain_vs_{b}", {}).get("ari", {})
            md = c.get("mean_diff"); bm = summ.get(s, {}).get(b, {}).get("ari", {}).get("mean")
            if md is not None and bm:
                vals.append(100.0 * md / bm)
            dv = c.get("cohens_dz", c.get("cohens_d"))
            if dv is not None:
                ds.append(float(dv))
        m = float(np.mean(vals)); impr.append(m); lo.append(m - min(vals)); hi.append(max(vals) - m)
        dmed.append(float(np.median(ds))); cols.append(COLORS[b])
    xb = np.arange(len(BASELINES))
    bars_b = axB.bar(
        xb, impr, 0.6, color=cols, yerr=[lo, hi], capsize=6,
        edgecolor="#1F1F1F", linewidth=0.7,
        error_kw={"lw": 1.6, "alpha": 0.85, "ecolor": "#1F1F1F"},
    )
    for patch, (mode, _) in zip(bars_b, BASELINES, strict=True):
        patch.set_hatch(HATCHES[mode])
    axB.set_xticks(xb); axB.set_xticklabels([l for _, l in BASELINES]); axB.set_ylabel("Relative ARI Difference (%)")
    axB.set_title("(b) Relative ARI Differences")
    low_b = min(m - l for m, l in zip(impr, lo))
    high_b = max(m + h for m, h in zip(impr, hi))
    span_b = max(high_b - low_b, 1e-6)
    axB.set_ylim(min(0.0, low_b - 0.18 * span_b), max(0.0, high_b + 0.22 * span_b))
    axB.axhline(0.0, color="#616161", linewidth=1.0)
    _apply_style(axB)
    for xi, m, l, h, dv in zip(xb, impr, lo, hi, dmed):
        y = m + h + 0.04 * span_b if m >= 0 else m - l - 0.04 * span_b
        # A ground of its own: these sit inside the plot area, and without one
        # the horizontal gridlines run straight through the glyphs.
        axB.text(xi, y, f"{m:+.1f}%\nmedian dz={dv:.1f}", ha="center",
                 va="bottom" if m >= 0 else "top",
                 fontsize=ANNOT_FONT_SIZE, fontweight="bold", color="#1F1F1F",
                 zorder=6,
                 bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                           edgecolor="none", alpha=1.0))

    axC_l, axC_r = _frontier(fig, outer[1, 0], ["static"],
                             ["hybrid_rl", "no_pinn", "no_slca", "no_context"], 3.2,
                             single_left_tick=True, xtick_decimals=3)
    dlat = pts["agribrain"][0] - pts["no_context"][0]; dari = pts["agribrain"][1] - pts["no_context"][1]
    axD_l, axD_r = _frontier(fig, outer[1, 1], ["no_context"], ["mcp_only", "pirag_only", "agribrain"],
                             4.0, ref=("no_context",), lncol=4,
                             single_left_tick=True, xtick_decimals=2,
                             annotate_xy=(0.5, 0.90), annotate_ha="center", annotate_va="top",
                             annotate=f"Context-associated change\n{dlat:+.1f} ms  →  {dari:+.3f} ARI")
    # headroom so the in-panel annotation clears the markers
    _lo, _hi = axD_l.get_ylim(); axD_l.set_ylim(_lo, _hi + (_hi - _lo) * 0.45)
    # green indicator line: No-external-context -> AGRI-BRAIN (overhead path), drawn
    # across the broken axis from the left sub-axis to the right sub-axis.
    from matplotlib.patches import ConnectionPatch as _ConnectionPatch
    _nc, _ag = pts["no_context"], pts["agribrain"]
    _con = _ConnectionPatch(xyA=(_nc[0], _nc[1]), coordsA=axD_l.transData,
                            xyB=(_ag[0], _ag[1]), coordsB=axD_r.transData,
                            color=COLORS["agribrain"], lw=2.2, ls="--", alpha=0.8, zorder=1)
    fig.add_artist(_con)

    # This figure's GridSpec tops out at 0.885 and its panel titles and keys
    # sit above that, so the title is seated relative to them, not to the
    # shared grid rect.
    fig.suptitle("Paired Comparisons and Decision Latency", y=_F11_SUPTITLE_Y)
    _align_panel_titles(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # (c) and (d) are broken-axis pairs, so their shared title and x-axis name
    # are placed in figure coordinates once the layout is settled. The title
    # clears the pair's own key, which sits directly above the axes.
    for (axl, axr), title in (((axC_l, axC_r), "(c) Lightweight Methods"),
                              ((axD_l, axD_r), "(d) Context-Aware Methods")):
        pl, pr = axl.get_position(), axr.get_position(); cx = (pl.x0 + pr.x1) / 2
        # The pair's key occupies one or more rows directly above the axes and
        # its height follows the key's font size, so the title is seated above
        # whatever the key actually measures rather than a fixed offset.
        gap = SUBPLOT_TITLE_SIZE * 0.5 / (fig.get_figheight() * 72.0)
        top = pr.y1
        for pair_ax in (axl, axr):
            key = pair_ax.get_legend()
            if key is not None:
                top = max(top, key.get_window_extent(renderer).y1 / fig.bbox.height)
        fig.text(cx, top + gap, title, ha="center", va="bottom",
                 fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold")
        # Likewise the shared x-axis name clears the lowest tick label drawn.
        bottom = pl.y0
        for pair_ax in (axl, axr):
            for label in pair_ax.get_xticklabels():
                if label.get_text():
                    bottom = min(bottom,
                                 label.get_window_extent(renderer).y0 / fig.bbox.height)
        fig.text(cx, bottom - gap, "Mean Decision Latency (ms)", ha="center", va="top",
                 fontsize=AXIS_LABEL_SIZE, fontweight="normal")
    _save(fig, "performance_efficiency")


@panel_fonts(FOUR_PANEL_FONT_BUMP)
def fig12_context_channels(data=None):
    """H2 channel-arm contrasts and conditional feature-group masking.

    Panel (a) reads experimental arm differences. Panels (b-d) summarize
    algebraic masks applied to the recorded full-system policy surface; they do
    not represent disabled communication channels. The function encodes no
    expected direction.
    """
    sig_p = RESULTS_DIR / "benchmark_significance.json"
    summ_p = RESULTS_DIR / "benchmark_summary.json"
    agg_p = RESULTS_DIR / "channel_attribution_aggregate.json"
    if not all(p.exists() for p in (sig_p, summ_p, agg_p)):
        print("  [fig12] missing significance/summary/attribution JSON; skipped")
        return
    sig = json.loads(sig_p.read_text())["significance"]
    summ = json.loads(summ_p.read_text()); summ = summ.get("summary", summ)
    agg = json.loads(agg_p.read_text()); bsm = agg["by_scenario_mode"]

    C_CTX, C_MCP, C_PIR = COLORS["agribrain"], COLORS["mcp_only"], COLORS["pirag_only"]
    C_SYN, C_RED, C_GOV = "#882255", "#4D4D4D", "#A66F00"
    SCEN = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing", "baseline"]
    SLAB = SCENARIO_LABELS
    scen = [s for s in SCEN if s in sig]
    cscen = [s for s in SCEN if s in bsm and "agribrain" in bsm[s]]
    cells = {s: bsm[s]["agribrain"] for s in cscen}

    fig, axes = plt.subplots(2, 2, figsize=GRID_FIGSIZE)
    (axA, axB), (axC, axD) = axes

    # The confirmatory H2 family contains all four directional contrasts in
    # every scenario.  Earlier versions showed only the two single-channel
    # additions and mixed in Full-vs-No-external-context (the H1 contrast), which made
    # the panel an incomplete visual representation of H2.
    CH = [
        ("mcp_only", "no_context", "MCP\n\u2212 No-context", C_MCP),
        ("pirag_only", "no_context", "Retrieval\n\u2212 No-context", C_PIR),
        ("agribrain", "mcp_only", "Full\n\u2212 MCP", C_CTX),
        ("agribrain", "pirag_only", "Full\n\u2212 Retrieval", C_SYN),
    ]
    impr, lo, hi, dmed, cols = [], [], [], [], []
    for left_mode, right_mode, _, col in CH:
        vals, ds = [], []
        for s in scen:
            comparison = f"{left_mode}_vs_{right_mode}"
            c = sig[s].get(comparison, {}).get("ari", {})
            md = c.get("mean_diff")
            bm = summ.get(s, {}).get(right_mode, {}).get("ari", {}).get("mean")
            if md is not None and bm:
                vals.append(100.0 * md / bm)
            dv = c.get("cohens_dz", c.get("cohens_d"))
            if dv is not None:
                ds.append(float(dv))
        m = float(np.mean(vals)); impr.append(m); lo.append(m - min(vals)); hi.append(max(vals) - m)
        dmed.append(float(np.median(ds))); cols.append(col)
    xb = np.arange(len(CH))
    bars_a = axA.bar(
        xb, impr, 0.6, color=cols, yerr=[lo, hi], capsize=6,
        edgecolor="#1F1F1F", linewidth=0.7,
        error_kw={"lw": 1.6, "alpha": 0.85, "ecolor": "#1F1F1F"},
    )
    for patch, hatch in zip(bars_a, ("//", "\\\\", "xx", "oo"), strict=True):
        patch.set_hatch(hatch)
    _cat_ticks(axA, xb, [lab for _, _, lab, _ in CH])
    axA.set_ylabel("\u0394ARI vs Comparison (%)")
    axA.set_title("(a) H2 Directional Contrasts")
    low_a = min(m - l for m, l in zip(impr, lo))
    high_a = max(m + h for m, h in zip(impr, hi))
    span_a = max(high_a - low_a, 1e-6)
    axA.set_ylim(min(0.0, low_a - 0.18 * span_a), max(0.0, high_a + 0.24 * span_a))
    axA.axhline(0.0, color="#616161", linewidth=1.0)
    _apply_style(axA)
    for xi, m, l, h, dv in zip(xb, impr, lo, hi, dmed):
        y = m + h + 0.04 * span_a if m >= 0 else m - l - 0.04 * span_a
        # A ground of its own, as in figure 11 panel (b): these sit inside the
        # plot area, and without one the horizontal gridlines run through the
        # glyphs. Covering a gridline is the right trade -- it is decoration,
        # and it reads perfectly well resuming either side of the label.
        axA.text(xi, y, f"{m:+.1f}%\nmedian\ndz={dv:.1f}", ha="center",
                 va="bottom" if m >= 0 else "top",
                 fontsize=ANNOT_FONT_SIZE, fontweight="bold", color="#1F1F1F",
                 zorder=6,
                 bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                           edgecolor="none", alpha=1.0))

    x = np.arange(len(cscen)); w = 0.2
    series = [
        ("Observed vs zeroed", "context_route_change", C_CTX, "xx"),
        ("MCP mask", "mcp_feature_group_mask_effect", C_MCP, "//"),
        ("Retrieval mask", "pirag_feature_group_mask_effect", C_PIR, "oo"),
        ("Joint-only change", "joint_only_route_change", C_SYN, "\\\\"),
    ]
    for i, (lab, key, col, hatch) in enumerate(series):
        vals = [cells[s][key]["rate"] * 100 for s in cscen]
        los = [(cells[s][key]["rate"] - cells[s][key]["ci_low"]) * 100 for s in cscen]
        his = [(cells[s][key]["ci_high"] - cells[s][key]["rate"]) * 100 for s in cscen]
        axB.bar(x + (i - 1.5) * w, vals, w, label=lab, color=col,
                hatch=hatch, edgecolor="#1F1F1F", linewidth=0.7,
                yerr=[los, his], capsize=3,
                error_kw={"lw": 1.2, "alpha": 0.85, "ecolor": "#1F1F1F"})
    _cat_ticks(axB, x, [SCENARIO_TICKS[s] for s in cscen])
    axB.set_ylabel("AGRI-BRAIN Decisions (%)")
    axB.set_ylim(0, max(cells[s]["context_route_change"]["ci_high"] for s in cscen) * 100 * 1.25)
    _apply_style(axB)
    axB.set_title("(b) Feature-Group Masking")
    _panel_key(axB, ncol=2)

    keys = [
        ("pirag_group_matches_observed_only", "Retrieval only", C_PIR, "oo"),
        ("mcp_group_matches_observed_only", "MCP only", C_MCP, "//"),
        ("neither_group_matches_observed", "Neither", C_SYN, "\\\\"),
        ("both_groups_match_observed", "Both", C_RED, "xx"),
    ]
    bottom = np.zeros(len(cscen))
    for k, lab, col, hatch in keys:
        vals = np.array([cells[s]["attribution_fraction"][k] * 100 for s in cscen])
        axC.bar(
            x, vals, 0.6, bottom=bottom, label=lab, color=col,
            hatch=hatch, edgecolor="#1F1F1F", linewidth=0.7,
        )
        bottom += vals
    _cat_ticks(axC, x, [SCENARIO_TICKS[s] for s in cscen])
    axC.set_ylabel("Context-Changed (%)"); axC.set_ylim(0, 118)
    _apply_style(axC)
    for xi, s in zip(x, cscen):
        di = cells[s]["conditional_distinctness_index"] * 100
        axC.text(xi, 102, f"D={di:.0f}%", ha="center", fontsize=ANNOT_FONT_SIZE, fontweight="bold", color="#333")
    axC.set_title("(c) Single-Group Route Agreement")
    _panel_key(axC, ncol=4)

    # Overall rate vs the rate conditioned on MCP-governed steps, per scenario,
    # with seed-cluster 95% CIs. Direction and magnitude are data-driven.
    mn = [cells[s]["mcp_feature_group_mask_effect"]["rate"] * 100 for s in cscen]
    mg = [cells[s]["mcp_feature_group_mask_effect_given_compliance"]["rate"] * 100 for s in cscen]
    mn_e = [[(cells[s]["mcp_feature_group_mask_effect"]["rate"] - cells[s]["mcp_feature_group_mask_effect"]["ci_low"]) * 100 for s in cscen],
            [(cells[s]["mcp_feature_group_mask_effect"]["ci_high"] - cells[s]["mcp_feature_group_mask_effect"]["rate"]) * 100 for s in cscen]]
    mg_e = [[(cells[s]["mcp_feature_group_mask_effect_given_compliance"]["rate"] - cells[s]["mcp_feature_group_mask_effect_given_compliance"]["ci_low"]) * 100 for s in cscen],
            [(cells[s]["mcp_feature_group_mask_effect_given_compliance"]["ci_high"] - cells[s]["mcp_feature_group_mask_effect_given_compliance"]["rate"]) * 100 for s in cscen]]
    axD.bar(x - 0.18, mn, 0.34, color="#007C91", label="Overall",
            hatch="//", edgecolor="#1F1F1F", linewidth=0.7, yerr=mn_e, capsize=3,
            error_kw={"lw": 1.2, "ecolor": "#1F1F1F"})
    axD.bar(x + 0.18, mg, 0.34, color="#A66F00", label="On governed steps",
            hatch="xx", edgecolor="#1F1F1F", linewidth=0.7, yerr=mg_e, capsize=3,
            error_kw={"lw": 1.2, "ecolor": "#1F1F1F"})
    _cat_ticks(axD, x, [SCENARIO_TICKS[s] for s in cscen])
    axD.set_ylabel("MCP Mask Effect (%)")
    axD.set_title("(d) MCP Masking: Overall vs Governed")
    axD.set_ylim(0, max(h + e for h, e in zip(mg, mg_e[1])) * 1.2)
    _apply_style(axD)
    _panel_key(axD, ncol=2)

    fig.suptitle("Context-Layer Contrasts and Feature Masks", y=SUPTITLE_Y)
    _finish_grid(fig)
    _save(fig, "context_value")


@panel_fonts(FOUR_PANEL_FONT_BUMP)
def fig13_stress_robustness(data=None):
    """H3 robustness. (a) |ΔARI| and TOST outcome by cell, (b) signed
    sensor-noise ΔARI with 90% TOST intervals, (c) descriptive metric
    drift, and (d) descriptive cross-scenario ARI drift by stressor."""
    from matplotlib.colors import LinearSegmentedColormap as _LSC
    from matplotlib.ticker import FormatStrFormatter as _FmtStr
    pf = RESULTS_DIR / "stress_passfail.csv"
    if not pf.exists():
        print("  [fig13] missing stress_passfail.csv; skipped")
        return
    rows = [r for r in csv.DictReader(pf.open()) if r["Method"] == "agribrain"]
    cell = {(r["Scenario"], r["Stressor"]): r for r in rows}
    C_CTX, C_OK, C_BAD = COLORS["agribrain"], "#007C91", "#A66F00"
    SCEN = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing", "baseline"]
    SLAB = SCENARIO_LABELS
    STRESS = ["sensor_noise", "missing_data", "telemetry_delay", "mcp_fault_injection", "compounded"]
    STLAB = {"sensor_noise": "Sensor\nnoise", "missing_data": "Missing\ndata",
             "telemetry_delay": "Telemetry\ndelay", "mcp_fault_injection": "MCP\nfault",
             "compounded": "Com-\npounded"}
    METRICS = [("ARI", "ari_delta", "Threshold_ARI", "higher"),
               ("Waste fraction", "waste_delta", "Threshold_Waste", "lower"),
               ("Social proxy", "slca_delta", "Threshold_SLCA", "higher"),
               ("RLE", "rle_delta", "Threshold_RLE", "higher"),
               ("Emissions", "carbon_delta", "Threshold_Carbon", "lower"),
               ("Proxy stability", "equity_delta", "Threshold_Equity", "higher"),
               ("Latency", "latency_ms_delta", "Threshold_LatencyMs", "lower")]
    thr = {m[0]: float(rows[0][m[2]]) for m in METRICS}
    DRIFT = 0.01

    fig, axes = plt.subplots(2, 2, figsize=GRID_FIGSIZE)
    (axA, axB), (axC, axD) = axes

    M = np.full((len(SCEN), len(STRESS)), np.nan)
    for i, s in enumerate(SCEN):
        for j, st in enumerate(STRESS):
            r = cell.get((s, st))
            if r:
                M[i, j] = abs(float(r["ari_delta"]))
    cmap = _LSC.from_list(
        "agribrain_stress_magnitude", ["#F7FBFF", "#6BAED6", "#08306B"],
    )
    color_max = max(DRIFT, float(np.nanmax(M)))
    im = axA.pcolormesh(
        np.arange(len(STRESS) + 1) - 0.5,
        np.arange(len(SCEN) + 1) - 0.5,
        M,
        cmap=cmap,
        vmin=0,
        vmax=color_max,
        shading="flat",
        rasterized=False,
        edgecolors="white",
        linewidth=0.4,
    )
    axA.set_xlim(-0.5, len(STRESS) - 0.5)
    axA.set_ylim(len(SCEN) - 0.5, -0.5)
    _cat_ticks(axA, range(len(STRESS)), [STLAB[s] for s in STRESS])
    _cat_ticks(axA, range(len(SCEN)), [SLAB[s] for s in SCEN], axis="y")
    for i in range(len(SCEN)):
        for j in range(len(STRESS)):
            if not np.isnan(M[i, j]):
                passed = str(cell[(SCEN[i], STRESS[j])].get("Pass_Equivalence", "")).lower() == "true"
                axA.text(j, i, f"{M[i, j]*1000:.1f}\n{'EQ' if passed else 'NE'}", ha="center", va="center", fontsize=ANNOT_FONT_SIZE,
                         fontweight="bold", color="#1F1F1F" if M[i, j] < color_max * 0.7 else "white")
    cb = fig.colorbar(im, ax=axA, fraction=0.046, pad=0.03)
    # Colorbar gradients are rasterized by default in vector backends, which
    # embeds a raster image XObject and violates the all-vector publication
    # PDF contract; draw the solids as vector quads with face-matched edges
    # so quad seams stay invisible.
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor("face")
    cb.set_label(r"|ΔARI| ($\times10^{-3}$)", fontsize=TICK_FONT_SIZE, fontweight="normal")
    # The cells are annotated as M * 1000 under a title that declares 1e-3
    # units, but the mesh is mapped from the raw values, so the bar has to be
    # relabelled into the units of the numbers it is colouring; left alone its
    # ticks read a thousand times smaller than the cells beside them.
    from matplotlib.ticker import FuncFormatter as _FuncFmt
    cb.ax.yaxis.set_major_formatter(_FuncFmt(lambda _v, _p: f"{_v * 1000:g}"))
    cb.ax.tick_params(labelsize=TICK_FONT_SIZE - 2)
    for _t in cb.ax.get_yticklabels():
        _t.set_fontweight("normal")
    axA.set_title(r"(a) ARI Drift under Stress ($\times10^{-3}$)", pad=14)
    axA.grid(False); axA.tick_params(length=0)
    for sp in axA.spines.values():
        sp.set_visible(False)
    for lbl in axA.get_xticklabels() + axA.get_yticklabels():
        lbl.set_fontweight("normal")

    # Signed paired mean changes with the formal 90% TOST intervals.
    delta = [float(cell[(s, "sensor_noise")]["ari_delta"]) for s in SCEN]
    ci_lo = [float(cell[(s, "sensor_noise")]["ari_tost_ci90_low"]) for s in SCEN]
    ci_hi = [float(cell[(s, "sensor_noise")]["ari_tost_ci90_high"]) for s in SCEN]
    derr = [
        [max(0.0, m - lo) for m, lo in zip(delta, ci_lo)],
        [max(0.0, hi - m) for m, hi in zip(delta, ci_hi)],
    ]
    y = np.arange(len(SCEN))[::-1]
    axB.errorbar(delta, y, xerr=derr, fmt="o", color=C_CTX, markersize=16,
                 markeredgecolor="white", markeredgewidth=1.4, capsize=5, elinewidth=1.8, zorder=4)
    axB.axvline(-DRIFT, color="#1F1F1F", lw=2, ls="--", zorder=2)
    axB.axvline(DRIFT, color="#1F1F1F", lw=2, ls="--", zorder=2)
    axB.set_ylim(-0.6, len(SCEN) - 1 + 1.1)
    axB.text(0.0, len(SCEN) - 1 + 0.72, "Equivalence margin", ha="center", va="center",
             fontsize=TICK_FONT_SIZE - 3, fontweight="bold", color="#1F1F1F")
    axB.set_yticks(y); axB.set_yticklabels([SLAB[s] for s in SCEN])
    axB.set_xlabel("ΔARI under Sensor Noise (90% CI)")
    b_lo = min(min(ci_lo), -DRIFT); b_hi = max(max(ci_hi), DRIFT)
    b_pad = max((b_hi - b_lo) * 0.12, 0.001)
    axB.set_title("(b) Paired ARI Change: Sensor Noise", pad=14)
    axB.set_xlim(b_lo - b_pad, b_hi + b_pad); axB.xaxis.set_major_formatter(_FmtStr("%.3f"))
    _apply_style(axB); axB.grid(False); axB.grid(True, axis="x", linewidth=0.6, color="#BDBDBD", alpha=0.6)

    means, worsts = [], []
    for name, dcol, _, direction in METRICS:
        vals = np.array([((-float(r[dcol])) if direction == "higher" else float(r[dcol]))
                         / abs(thr[name]) for r in rows]) if thr[name] else np.zeros(len(rows))
        means.append(float(vals.mean())); worsts.append(float(vals.max()))
    yo = np.arange(len(METRICS))[::-1]
    worst_bars = axC.barh(
        yo, worsts, 0.6,
        color=[C_BAD if value > 1.0 else C_OK for value in worsts],
        edgecolor="#1F1F1F", linewidth=0.7, label="Worst cell", zorder=3,
    )
    for patch, value in zip(worst_bars, worsts, strict=True):
        patch.set_hatch("xx" if value > 1.0 else "//")
    axC.scatter(means, yo, s=320, color="#1F1F1F", marker="|", linewidths=2.4, zorder=5, label="Mean")
    axC.axvline(0, color="#9E9E9E", lw=1.0, alpha=0.6)
    axC.axvline(1.0, color="#1F1F1F", lw=2, ls="--")
    axC.text(1.0, len(METRICS) - 0.4, " Declared bound", fontsize=TICK_FONT_SIZE - 2,
             fontweight="bold", va="top", color="#1F1F1F")
    axC.set_yticks(yo); axC.set_yticklabels([m[0] for m in METRICS])
    axC.set_xlabel("Change / Declared Bound")
    axC.set_title("(c) Metric Sensitivity")
    c_lo = min(0.0, min(worsts), min(means)); c_hi = max(1.0, max(worsts), max(means))
    c_pad = max((c_hi - c_lo) * 0.08, 0.05)
    axC.set_xlim(c_lo - c_pad, c_hi + c_pad)
    _apply_style(axC); axC.grid(False); axC.grid(True, axis="x", linewidth=0.6, color="#BDBDBD", alpha=0.6)
    _panel_key(axC, ncol=2)

    means, stds, worsts = [], [], []
    for st in STRESS:
        vals = [abs(float(cell[(s, st)]["ari_delta"])) for s in SCEN]
        means.append(float(np.mean(vals))); stds.append(float(np.std(vals, ddof=1))); worsts.append(max(vals))
    xb2 = np.arange(len(STRESS))
    axD.bar(xb2, means, 0.6, color=C_CTX, hatch="//",
            edgecolor="#1F1F1F", linewidth=0.7,
            label="Mean \u00b1 scenario SD", yerr=stds, capsize=5,
            error_kw={"lw": 1.6, "ecolor": "#1F1F1F"})
    axD.scatter(xb2, worsts, s=130, color=C_BAD, marker="D", zorder=5, label="Worst |\u0394ARI|", edgecolor="white", linewidth=1.0)
    axD.axhline(DRIFT, color="#1F1F1F", lw=2, ls="--")
    axD.text(len(STRESS) - 0.45, DRIFT * 0.94, "Equivalence margin",
             fontsize=TICK_FONT_SIZE - 3, fontweight="bold", va="top",
             ha="right", color="#1F1F1F")
    _cat_ticks(axD, xb2, [STLAB[s] for s in STRESS])
    axD.set_ylabel("|ΔARI|")
    d_hi = max(DRIFT, max(m + s for m, s in zip(means, stds)), max(worsts))
    axD.set_title("(d) ARI Drift by Stressor"); axD.set_ylim(0, d_hi * 1.12)
    _apply_style(axD)
    _panel_key(axD, ncol=2)

    fig.suptitle("Robustness under Sensing and Protocol Stressors", y=SUPTITLE_Y)
    _finish_grid(fig)
    _save(fig, "stress_robustness")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def generate_all_figures(data=None):
    """Render figures from an explicitly supplied validated data mapping."""
    if data is None:
        raise RuntimeError(
            "validated data is required; direct one-seed simulation rendering "
            "is retired"
        )

    print("Generating figures...")
    fig2_heatwave(data)
    fig3_overproduction(data)
    fig4_cyber(data)
    fig5_pricing(data)
    fig6_cross(data)
    fig7_ablation(data)
    fig8_transport_emissions(data)
    # H1/H2/H3 paper figures (read the saved benchmark / attribution / stress
    # artefacts; skip with a message if those inputs are absent). The latency
    # frontier formerly in fig10 is now folded into fig11 panels (c)/(d).
    fig11_performance_efficiency(data)
    fig12_context_channels(data)
    fig13_stress_robustness(data)
    print()
    print(f"All figures saved to {Path(os.environ['FIGURE_OUTPUT_DIR'])}")


if __name__ == "__main__":
    print(
        "RETIRED: direct generate_figures.py execution cannot run a simulation "
        "or write publication figures. Use the validated "
        "mvp/simulation/regenerate_figures_from_cache.py workflow.",
        file=sys.stderr,
    )
    raise SystemExit(2)
