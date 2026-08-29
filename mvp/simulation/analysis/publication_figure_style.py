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
LEGEND_FONT_SIZE = 18
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
            "default_weight": "normal",
            "external_tex": False,
            "axis_labels": "normal",
            "tick_labels": "normal",
            "legend": "normal",
            "panel_titles": "bold",
            "figure_titles": "bold",
        },
    }


def apply_publication_style() -> None:
    """Apply the shared print-legible Matplotlib defaults."""
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "dejavusans",
        "font.size": BODY_FONT_SIZE,
        "font.weight": "normal",
        "text.usetex": False,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "axes.labelweight": "normal",
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
    """Apply subtle grids, open spines, and print-weight labels to one axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.6, color="#C7C7C7", alpha=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=TICK_FONT_SIZE, length=5, width=1.3)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontweight("normal")
    for axis in (ax.xaxis, ax.yaxis):
        axis.get_offset_text().set_fontweight("normal")
        axis.get_offset_text().set_fontsize(TICK_FONT_SIZE)
    if ax.xaxis.label.get_text():
        ax.xaxis.label.set(size=AXIS_LABEL_SIZE, weight="normal")
    if ax.yaxis.label.get_text():
        ax.yaxis.label.set(size=AXIS_LABEL_SIZE, weight="normal")
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
            text.set_fontweight("normal")
        if legend.get_title() is not None:
            legend.get_title().set_fontweight("normal")
    return legend


def save_figure_pair(fig: Any, output_dir: Path, name: str, *, dpi: int = PUBLICATION_DPI) -> None:
    """Write deterministic high-resolution PNG and vector PDF companions."""
    if dpi < MINIMUM_PUBLICATION_DPI:
        raise ValueError(f"publication PNG dpi must be >= {MINIMUM_PUBLICATION_DPI}")
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
