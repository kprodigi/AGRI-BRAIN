from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pypdf import PdfReader

from mvp.simulation.analysis.publication_figure_style import (
    MINIMUM_PUBLICATION_DPI,
    PUBLICATION_DPI,
    SEMANTIC_COLORS,
    SEMANTIC_HATCHES,
    SEMANTIC_LINESTYLES,
    SEMANTIC_MARKERS,
    accessible_legend,
    apply_publication_style,
    save_figure_pair,
    style_axes,
)

EXPECTED_MODES = {
    "static", "hybrid_rl", "no_pinn", "no_slca", "no_context", "mcp_only",
    "pirag_only", "agribrain", "agribrain_standard_rag",
    "agribrain_no_peer", "agribrain_sign_unconstrained",
}


def _contrast_against_white(hex_color: str) -> float:
    channels = [int(hex_color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in channels
    ]
    luminance = sum(
        weight * value
        for weight, value in zip((0.2126, 0.7152, 0.0722), linear, strict=True)
    )
    return 1.05 / (luminance + 0.05)


def _render(output: Path, name: str) -> None:
    apply_publication_style()
    figure, axes = plt.subplots(figsize=(5, 3))
    for index, mode in enumerate(("agribrain", "no_context", "mcp_only")):
        axes.plot(
            [0, 1, 2], [index, index + 0.5, index + 0.25],
            color=SEMANTIC_COLORS[mode], marker=SEMANTIC_MARKERS[mode],
            linestyle=SEMANTIC_LINESTYLES[mode], label=mode,
        )
    axes.set(title="Synthetic accessibility QA", xlabel="Step", ylabel="Score")
    style_axes(axes)
    accessible_legend(axes)
    save_figure_pair(figure, output, name)
    plt.close(figure)


def test_semantic_style_covers_every_publication_mode_with_redundancy():
    assert set(SEMANTIC_COLORS) == EXPECTED_MODES
    assert set(SEMANTIC_MARKERS) == EXPECTED_MODES
    assert set(SEMANTIC_HATCHES) == EXPECTED_MODES
    assert set(SEMANTIC_LINESTYLES) == EXPECTED_MODES
    assert len(set(SEMANTIC_COLORS.values())) == len(EXPECTED_MODES)
    assert len(set(SEMANTIC_MARKERS.values())) == len(EXPECTED_MODES)
    assert len(set(SEMANTIC_HATCHES.values())) == len(EXPECTED_MODES)
    assert all(_contrast_against_white(color) >= 3.0 for color in SEMANTIC_COLORS.values())
    assert PUBLICATION_DPI >= MINIMUM_PUBLICATION_DPI


def test_save_pair_is_deterministic_and_high_resolution(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    _render(first, "fixture")
    _render(second, "fixture")
    for extension in ("png", "pdf"):
        assert (first / f"fixture.{extension}").read_bytes() == (
            second / f"fixture.{extension}"
        ).read_bytes()
    with Image.open(first / "fixture.png") as image:
        dpi = image.info.get("dpi")
        assert dpi is not None
        assert min(dpi) >= MINIMUM_PUBLICATION_DPI - 1
    assert (first / "fixture.pdf").read_bytes().startswith(b"%PDF-")
    reader = PdfReader(first / "fixture.pdf", strict=True)
    page = reader.pages[0]
    fonts = page["/Resources"]["/Font"].get_object()
    assert fonts
    assert all(
        str(reference.get_object().get("/Subtype", "")) != "/Type3"
        for reference in fonts.values()
    )
    content = page.get_contents().get_data()
    assert b" m\n" in content or b" l\n" in content


def test_typographic_hierarchy_sets_every_element_bold():
    matplotlib.rcParams.update({
        "axes.facecolor": "black",
        "figure.facecolor": "red",
        "font.weight": "bold",
        "pdf.use14corefonts": True,
        "savefig.transparent": True,
        "text.color": "yellow",
        "text.usetex": True,
        "xtick.color": "white",
        "ytick.color": "white",
    })
    apply_publication_style()
    assert matplotlib.rcParams["axes.facecolor"] == "#FFFFFF"
    assert matplotlib.rcParams["figure.facecolor"] == "#FFFFFF"
    assert matplotlib.rcParams["text.color"] == "#1F1F1F"
    # The set is printed bold throughout, so the default weight carries it and
    # the hierarchy is expressed by size alone. This reverses the earlier
    # contract, under which bold was reserved for titles.
    assert matplotlib.rcParams["font.weight"] == "bold"
    assert matplotlib.rcParams["text.usetex"] is False
    assert matplotlib.rcParams["pdf.use14corefonts"] is False
    assert matplotlib.rcParams["savefig.transparent"] is False
    assert matplotlib.rcParams["xtick.color"] == "#1F1F1F"
    assert matplotlib.rcParams["ytick.color"] == "#1F1F1F"
    figure, axes = plt.subplots()
    axes.plot([0, 1], [0, 1], label="Series")
    axes.set(title="Panel title", xlabel="Regular x label", ylabel="Regular y label")
    style_axes(axes)
    legend = accessible_legend(axes)
    figure.canvas.draw()
    assert axes.title.get_fontweight() == "bold"
    assert axes.xaxis.label.get_fontweight() == "bold"
    assert axes.yaxis.label.get_fontweight() == "bold"
    assert all(label.get_fontweight() == "bold" for label in axes.get_xticklabels())
    assert all(text.get_fontweight() == "bold" for text in legend.get_texts())
    plt.close(figure)
