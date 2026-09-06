"""Shared accessible, deterministic style for publication figures."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

BODY_FONT_SIZE = 18
TICK_FONT_SIZE = 18
AXIS_LABEL_SIZE = 20
SUBPLOT_TITLE_SIZE = 22
FIG_TITLE_SIZE = 26
LEGEND_FONT_SIZE = 19
#: Absolute size for every panel key, deliberately not scaled by the per-figure
#: font bump: a two-entry key and a four-entry key have to read as the same
#: object across the whole set. 19 pt is the largest at which no key in the set
#: touches another key, an axes, a panel title or the canvas edge; at 20 pt the
#: performance figure's key leaves the canvas.
PANEL_KEY_FONT_SIZE = 19
#: Starting size for a key of five entries or more, which is sized to the room
#: it has rather than held to the shared size; it gives up half a point at a
#: time down to PANEL_KEY_FONT_SIZE as a floor.
PANEL_KEY_FONT_CAP = 30.0
#: Ceiling on how far a key may reach past its own panel when nothing sits
#: beside it. Where a neighbour does sit in the same band, the usable width is
#: measured against that gap instead, panel by panel; see _key_room.
PANEL_KEY_OVERHANG = 1.60
ANNOT_FONT_SIZE = 17
PUBLICATION_DPI = 800
MINIMUM_PUBLICATION_DPI = 400
PUBLICATION_STYLE_SCHEMA_VERSION = 1
MARKER_EVERY = 15

# Okabe-Ito plus three high-contrast extensions. Meanings are stable across
# every core/H1/H2/H3 figure; color is backed by a series-appropriate marker,
# line pattern, hatch, or grouping position rather than carrying meaning alone.
SEMANTIC_COLORS = {
    "static": "#4D4D4D",
    "hybrid_rl": "#A66F00",
    "no_pinn": "#117733",
    "no_slca": "#CC79A7",
    "agribrain": "#0072B2",
    "no_context": "#009E73",
    "mcp_only": "#D55E00",
    "pirag_only": "#007C91",
    "agribrain_standard_rag": "#882255",
    "agribrain_no_peer": "#332288",
    "agribrain_sign_unconstrained": "#999933",
}
SEMANTIC_MARKERS = {
    "static": "o", "hybrid_rl": "s", "no_pinn": "<", "no_slca": "D",
    "agribrain": "^",
    "no_context": "P", "mcp_only": "X", "pirag_only": "d",
    "agribrain_standard_rag": "v", "agribrain_no_peer": "h",
    "agribrain_sign_unconstrained": "*",
}
SEMANTIC_HATCHES = {
    "static": "",
    "hybrid_rl": "//",
    "no_pinn": "--",
    "no_slca": "\\\\",
    "agribrain": "xx",
    "no_context": "..",
    "mcp_only": "++",
    "pirag_only": "oo",
    "agribrain_standard_rag": "**",
    "agribrain_no_peer": "OO",
    "agribrain_sign_unconstrained": "||",
}
SEMANTIC_LINESTYLES = {
    "static": "-", "hybrid_rl": "--", "no_pinn": (0, (2, 1, 2, 1, 6, 1)),
    "no_slca": ":", "agribrain": "-.",
    "no_context": (0, (5, 2)), "mcp_only": (0, (3, 1, 1, 1, 1, 1)),
    "pirag_only": (0, (1, 1)),
    "agribrain_standard_rag": (0, (6, 2, 1, 2)),
    "agribrain_no_peer": (0, (8, 2)),
    "agribrain_sign_unconstrained": (0, (4, 2, 1, 2)),
}


def publication_style_contract() -> dict[str, Any]:
    """Return the JSON-safe rendering contract bound into figure provenance."""

    line_styles: dict[str, Any] = {}
    for mode, style in SEMANTIC_LINESTYLES.items():
        if isinstance(style, str):
            line_styles[mode] = style
        else:
            offset, pattern = style
            line_styles[mode] = [offset, list(pattern)]
    return {
        "schema_version": PUBLICATION_STYLE_SCHEMA_VERSION,
        "png": {
            "dpi": PUBLICATION_DPI,
            "minimum_dpi": MINIMUM_PUBLICATION_DPI,
            "background": "#FFFFFF",
            "transparent": False,
        },
        "pdf": {
            "fonttype": 42,
            "use_14_core_fonts": False,
            "one_page_per_figure": True,
            "vector_primitives_required": True,
        },
        "accessibility": {
            "minimum_series_contrast_against_white": 3.0,
            "redundant_series_encodings": {
                "line_series": ["color", "marker", "linestyle"],
                "bar_series": ["color", "hatch", "position"],
            },
            "semantic_colors": dict(SEMANTIC_COLORS),
            "semantic_markers": dict(SEMANTIC_MARKERS),
            "semantic_hatches": dict(SEMANTIC_HATCHES),
            "semantic_linestyles": line_styles,
        },
        "typography": {
            "default_weight": "bold",
            "external_tex": False,
            "axis_labels": "bold",
            "tick_labels": "bold",
            "legend": "bold",
            "annotations": "bold",
            "panel_titles": "bold",
            "figure_titles": "bold",
        },
        "panel_key": {
            "placement": "reserved space above the axes, under the panel title",
            "font_size": PANEL_KEY_FONT_SIZE,
            "font_size_is_absolute": True,
            "single_row_max_entries": 4,
            "axes_width_overhang": PANEL_KEY_OVERHANG,
        },
    }


def apply_publication_style() -> None:
    """Apply the shared print-legible Matplotlib defaults."""
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "dejavusans",
        "font.size": BODY_FONT_SIZE,
        "font.weight": "bold",
        "text.usetex": False,
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
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "text.color": "#1F1F1F",
        "axes.labelcolor": "#1F1F1F",
        "axes.titlecolor": "#1F1F1F",
        "savefig.dpi": PUBLICATION_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "savefig.facecolor": "#FFFFFF",
        "savefig.transparent": False,
        "lines.linewidth": 2.2,
        "lines.markersize": 8,
        "axes.linewidth": 1.3,
        "axes.edgecolor": "#1F1F1F",
        "axes.labelpad": 6,
        "xtick.major.width": 1.3,
        "ytick.major.width": 1.3,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.color": "#1F1F1F",
        "ytick.color": "#1F1F1F",
        "grid.color": "#C7C7C7",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.55,
        "patch.linewidth": 1.0,
        "patch.edgecolor": "white",
        "pdf.fonttype": 42,
        "pdf.use14corefonts": False,
        "ps.fonttype": 42,
    })


def style_axes(ax: Any) -> None:
    """Apply subtle grids, open spines, and bold print-weight labels to one axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.6, color="#C7C7C7", alpha=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=TICK_FONT_SIZE, length=5, width=1.3)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontweight("bold")
    for axis in (ax.xaxis, ax.yaxis):
        axis.get_offset_text().set_fontweight("bold")
        axis.get_offset_text().set_fontsize(TICK_FONT_SIZE)
    if ax.xaxis.label.get_text():
        ax.xaxis.label.set(size=AXIS_LABEL_SIZE, weight="bold")
    if ax.yaxis.label.get_text():
        ax.yaxis.label.set(size=AXIS_LABEL_SIZE, weight="bold")
    if ax.get_title():
        ax.title.set(size=SUBPLOT_TITLE_SIZE, weight="bold")


def accessible_legend(ax: Any, **kwargs: Any) -> Any:
    """Create a high-contrast legend with sufficiently long style samples."""
    defaults = {
        "fontsize": LEGEND_FONT_SIZE,
        "framealpha": 0.95,
        "facecolor": "white",
        "edgecolor": "#666666",
        "fancybox": False,
        "shadow": False,
        "borderpad": 0.45,
        "handlelength": 2.4,
        "handletextpad": 0.55,
        "labelspacing": 0.35,
    }
    defaults.update(kwargs)
    legend = ax.legend(**defaults)
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontweight("bold")
        if legend.get_title() is not None:
            legend.get_title().set_fontweight("bold")
    return legend


def _set_every_text_bold(fig: Any) -> None:
    """Set every piece of text in a finished figure bold, in one final pass.

    The style defaults and the helpers above carry titles, labels, ticks and
    keys, but a figure also holds text they never touch: bar annotations, axis
    offset text, inline callouts, and anything a panel adds directly. Weight is
    applied here, at the moment before writing, so nothing is left behind.
    """
    from matplotlib.text import Text

    for text in fig.findobj(Text):
        text.set_fontweight("bold")
    for axes in fig.axes:
        for axis in (axes.xaxis, axes.yaxis):
            axis.get_offset_text().set_fontweight("bold")
        legend = axes.get_legend()
        if legend is None:
            continue
        for text in legend.get_texts():
            text.set_fontweight("bold")
        if legend.get_title() is not None:
            legend.get_title().set_fontweight("bold")


def _lift_figure_title_clear_of_panels(fig: Any) -> None:
    """Keep the key-to-title gap from driving panel titles into the figure title.

    Most figures reflow and absorb the reserved key space on their own. One
    fixes its grid geometry in absolute figure coordinates and cannot, so its
    panel titles ride up into the suptitle. The suptitle is lifted as far as the
    canvas allows, and if that is still not enough the panels are lowered by the
    shortfall.
    """
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is None:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    height = fig.get_window_extent(renderer).height
    titles = [axes.title for axes in fig.axes if axes.get_title()]
    if not titles:
        return
    margin = 0.012 * height
    highest = max(title.get_window_extent(renderer).y1 for title in titles)
    if suptitle.get_window_extent(renderer).y0 >= highest + margin:
        return
    want = ((highest + margin) / height
            + (suptitle.get_window_extent(renderer).height / height) / 2)
    suptitle.set_y(min(0.995, want))
    fig.canvas.draw()
    shortfall = highest + margin - suptitle.get_window_extent(renderer).y0
    if shortfall <= 0:
        return
    drop = shortfall / height
    for axes in fig.axes:
        box = axes.get_position()
        axes.set_position([box.x0, box.y0 - drop * box.y1,
                           box.width, box.height * (1 - drop)])


def save_figure_pair(fig: Any, output_dir: Path, name: str, *, dpi: int = PUBLICATION_DPI) -> None:
    """Write deterministic high-resolution PNG and vector PDF companions."""
    if dpi < MINIMUM_PUBLICATION_DPI:
        raise ValueError(f"publication PNG dpi must be >= {MINIMUM_PUBLICATION_DPI}")
    _set_every_text_bold(fig)
    _lift_figure_title_clear_of_panels(fig)
    output_dir.mkdir(parents=True, exist_ok=True)
    common = {"bbox_inches": "tight", "pad_inches": 0.15, "facecolor": "white"}
    fig.savefig(
        output_dir / f"{name}.png",
        dpi=dpi,
        metadata={"Software": "AGRI-BRAIN deterministic Matplotlib renderer"},
        **common,
    )
    fig.savefig(
        output_dir / f"{name}.pdf",
        metadata={
            "Creator": "AGRI-BRAIN deterministic Matplotlib renderer",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
        **common,
    )
