"""Regression gates for the canonical figure renderer's visual encodings."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SIMULATION_DIR = Path(__file__).resolve().parents[1]
FIGURE_SOURCE = SIMULATION_DIR / "generate_figures.py"


def _load_renderer():
    module_name = "generate_figures_accessibility_test"
    spec = importlib.util.spec_from_file_location(module_name, FIGURE_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SIMULATION_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SIMULATION_DIR))
    return module


def test_per_figure_font_bump_survives_shared_axes_and_legend_helpers():
    renderer = _load_renderer()
    # Read the base sizes before entering the bump: panel_fonts raises the
    # module globals for the duration, so these have to be captured first.
    expected_tick = renderer.TICK_FONT_SIZE + 2
    expected_axis = renderer.AXIS_LABEL_SIZE + 2
    expected_title = renderer.SUBPLOT_TITLE_SIZE + 2
    expected_legend = renderer.LEGEND_FONT_SIZE + 2
    with renderer.panel_fonts(2):
        figure, axes = renderer.plt.subplots()
        try:
            axes.plot([0, 1], [0, 1], label="Series")
            axes.set(title="Panel", xlabel="Step", ylabel="Score")
            renderer._apply_style(axes)
            legend = renderer._legend(axes)
            figure.canvas.draw()
            assert axes.title.get_fontsize() == expected_title
            assert axes.xaxis.label.get_fontsize() == expected_axis
            assert axes.yaxis.label.get_fontsize() == expected_axis
            assert all(
                label.get_fontsize() == expected_tick
                for label in axes.get_xticklabels()
            )
            assert all(
                text.get_fontsize() == expected_legend
                for text in legend.get_texts()
            )
            # The set is printed bold throughout; hierarchy is carried by size.
            assert axes.title.get_fontweight() == "bold"
            assert axes.xaxis.label.get_fontweight() == "bold"
            assert all(text.get_fontweight() == "bold" for text in legend.get_texts())
        finally:
            renderer.plt.close(figure)


def test_renderer_uses_pattern_redundancy_and_vector_accessible_heatmaps():
    source = FIGURE_SOURCE.read_text(encoding="utf-8")
    assert "cmap=\"RdYlGn\"" not in source
    assert ".imshow(" not in source
    assert source.count(".pcolormesh(") == 2
    assert source.count("hatch=HATCHES[mode]") >= 6
    assert source.count("hatch=ACTION_HATCHES[") >= 6
    assert "#1565C0" not in source
    assert "alpha=0.92" not in source
    assert "alpha=0.85" not in source
