"""Focused tests for deterministic structural publication exports."""
from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from PIL import Image
from pypdf import PdfReader

from mvp.simulation.sensitivity.design import canonical_sha256
from mvp.simulation.sensitivity.publish_structural_sensitivity import (
    CSV_NAME,
    FIGURE_STYLE_CONTRACT,
    PDF_NAME,
    PNG_NAME,
    RECEIPT_NAME,
    _draw_figure,
    publish_analysis,
    summary_rows,
)


def _summary(mean: float) -> dict:
    return {
        "n": 100,
        "mean": mean,
        "std": 0.002,
        "min": mean - 0.004,
        "q05": mean - 0.003,
        "median": mean,
        "q95": mean + 0.003,
        "max": mean + 0.004,
    }


def _analysis() -> dict:
    payload = {
        "schema_version": 1,
        "analysis_label": "structural sensitivity",
        "probability_interpretation": False,
        "interpretation_boundary": "Descriptive stability, not probability.",
        "source_commit": "f" * 40,
        "design_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "n_design_points": 100,
        "n_parameters": 29,
        "h1_sign_stability": {
            "baseline": {
                "contrast": "agribrain - no_context",
                "descriptive_over_structural_points": _summary(0.012),
                "positive_sign_fraction": 0.98,
                "point_difference_above_0p005_fraction": 0.91,
            },
        },
        "h2_sign_stability": {
            "baseline": {
                "agribrain_minus_mcp_only": {
                    "descriptive_over_structural_points": _summary(0.006),
                    "positive_sign_fraction": 0.84,
                },
            },
        },
        "h3_margin_stability": {
            "strict_margin": 0.01,
            "cells": {
                "baseline": {
                    "sensor_noise": {
                        "contrast": "stressed agribrain - primary nominal agribrain",
                        "descriptive_over_structural_points": _summary(0.001),
                        "inside_strict_0p01_margin_fraction": 0.99,
                        "max_absolute_delta": 0.008,
                        "all_cells_have_nonzero_exposure": True,
                    },
                },
            },
        },
    }
    payload["analysis_sha256"] = canonical_sha256(payload)
    return payload


def _complete_panel_analysis() -> dict:
    payload = _analysis()
    payload.pop("analysis_sha256")
    scenarios = (
        "adaptive_pricing", "baseline", "cyber_outage", "heatwave",
        "overproduction",
    )
    contrasts = (
        "mcp_only_minus_no_context",
        "pirag_only_minus_no_context",
        "agribrain_minus_mcp_only",
        "agribrain_minus_pirag_only",
        "synergy_full_minus_mcp_minus_retrieval_plus_no_context",
    )
    stressors = (
        "sensor_noise", "missing_data", "telemetry_delay",
        "mcp_fault_injection", "compounded",
    )
    payload["h1_sign_stability"] = {
        scenario: {
            "contrast": "agribrain - no_context",
            "descriptive_over_structural_points": _summary(0.004 + 0.002 * index),
            "positive_sign_fraction": 0.8 + 0.01 * index,
            "point_difference_above_0p005_fraction": 0.7 + 0.01 * index,
        }
        for index, scenario in enumerate(scenarios)
    }
    payload["h2_sign_stability"] = {
        scenario: {
            contrast: {
                "descriptive_over_structural_points": _summary(
                    -0.004 + 0.002 * contrast_index + 0.0005 * scenario_index
                ),
                "positive_sign_fraction": 0.55 + 0.02 * contrast_index,
            }
            for contrast_index, contrast in enumerate(contrasts)
        }
        for scenario_index, scenario in enumerate(scenarios)
    }
    payload["h3_margin_stability"]["cells"] = {
        scenario: {
            stressor: {
                "contrast": "stressed agribrain - primary nominal agribrain",
                "descriptive_over_structural_points": _summary(
                    0.0002 * (stressor_index + 1)
                ),
                "inside_strict_0p01_margin_fraction": 0.99,
                "max_absolute_delta": 0.003 + 0.001 * stressor_index,
                "all_cells_have_nonzero_exposure": True,
            }
            for stressor_index, stressor in enumerate(stressors)
        }
        for scenario in scenarios
    }
    payload["analysis_sha256"] = canonical_sha256(payload)
    return payload


def _write_analysis(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_publication_is_deterministic_and_provenance_bound(tmp_path: Path) -> None:
    analysis = tmp_path / "structural_sensitivity_analysis.json"
    _write_analysis(analysis, _complete_panel_analysis())
    left = publish_analysis(analysis, tmp_path / "left")
    right = publish_analysis(analysis, tmp_path / "right")

    assert left == right
    assert left["schema_version"] == 2
    assert left["probability_interpretation"] is False
    assert left["row_count"] == 55
    for name in (CSV_NAME, PNG_NAME, PDF_NAME, RECEIPT_NAME):
        assert (tmp_path / "left" / name).read_bytes() == (
            tmp_path / "right" / name
        ).read_bytes()
    unsigned = dict(left)
    digest = unsigned.pop("receipt_sha256")
    assert digest == canonical_sha256(unsigned)
    assert {record["name"] for record in left["artifacts"]} == {
        CSV_NAME, PNG_NAME, PDF_NAME,
    }
    for record in left["artifacts"]:
        artifact = tmp_path / "left" / record["name"]
        assert record["bytes"] == artifact.stat().st_size
        assert record["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()

    style = left["figure_style"]
    assert style["contract"] == FIGURE_STYLE_CONTRACT
    assert style["requested_png_dpi"] == 800
    assert style["minimum_png_dpi"] == 400
    assert style["pdf_fonttype"] == 42
    assert style["typography"] == {
        "panel_and_figure_titles": "bold",
        "axis_tick_and_legend_text": "normal",
    }
    assert style["redundant_encodings"]["data_series"] == [
        "high-contrast colour", "marker shape",
    ]

    png_path = tmp_path / "left" / PNG_NAME
    with Image.open(png_path) as image:
        image.load()
        assert image.format == "PNG"
        assert image.mode in {"RGB", "RGBA"}
        assert min(image.size) >= 3000
        assert min(image.info["dpi"]) >= 799.0
        assert style["quality_checks"]["png"]["width_px"] == image.width
        assert style["quality_checks"]["png"]["height_px"] == image.height

    pdf_path = tmp_path / "left" / PDF_NAME
    reader = PdfReader(pdf_path, strict=True)
    assert not reader.is_encrypted
    assert len(reader.pages) == 1
    page = reader.pages[0]
    assert list(page.images) == []
    assert style["quality_checks"]["pdf"]["type3_font_count"] == 0
    assert style["quality_checks"]["pdf"]["embedded_font_programs"] >= 1
    assert style["quality_checks"]["pdf"][
        "vector_drawing_operators_present"
    ] is True
    assert "/CIDFontType2" in style["quality_checks"]["pdf"]["font_subtypes"]

    with (tmp_path / "left" / CSV_NAME).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert sum(row["family"] == "H1" for row in rows) == 5
    assert sum(row["family"] == "H2" for row in rows) == 25
    assert sum(row["family"] == "H3" for row in rows) == 25
    assert {
        (row["scenario"], row["contrast"])
        for row in rows if row["family"] == "H2"
    } == {
        (scenario, contrast)
        for scenario in (
            "adaptive_pricing", "baseline", "cyber_outage", "heatwave",
            "overproduction",
        )
        for contrast in (
            "mcp_only_minus_no_context",
            "pirag_only_minus_no_context",
            "agribrain_minus_mcp_only",
            "agribrain_minus_pirag_only",
            "synergy_full_minus_mcp_minus_retrieval_plus_no_context",
        )
    }


def test_complete_structural_panels_use_legible_redundant_encoding() -> None:
    figure = _draw_figure(summary_rows(_complete_panel_analysis()))
    try:
        assert len(figure.axes) == 3
        assert figure._suptitle is not None
        assert figure._suptitle.get_fontweight() == "bold"
        for axis in figure.axes:
            assert axis.title.get_fontweight() == "bold"
            assert axis.xaxis.label.get_fontweight() == "bold"
            assert all(
                label.get_fontweight() == "bold"
                for label in (*axis.get_xticklabels(), *axis.get_yticklabels())
            )
            legend = axis.get_legend()
            if legend is not None:
                assert all(
                    text.get_fontweight() == "bold"
                    for text in legend.get_texts()
                )

        h2_markers = {
            line.get_marker()
            for line in figure.axes[1].lines
            if line.get_marker() not in {None, "", "None", "none", "|", "_"}
        }
        h3_markers = {
            line.get_marker()
            for line in figure.axes[2].lines
            if line.get_marker() not in {None, "", "None", "none", "|", "_"}
        }
        assert len(h2_markers) == 5
        assert len(h3_markers) == 5
        strict_margin = next(
            line for line in figure.axes[2].lines
            if line.get_label() == "Margin 0.01"
        )
        assert strict_margin.get_linestyle() == "--"
    finally:
        plt.close(figure)


@pytest.mark.parametrize("mutation", ["hash", "schema", "quantile"])
def test_publication_fails_closed_on_invalid_analysis(
    tmp_path: Path, mutation: str,
) -> None:
    payload = _complete_panel_analysis()
    if mutation == "hash":
        payload["analysis_sha256"] = "0" * 64
    elif mutation == "schema":
        payload["schema_version"] = 2
        payload["analysis_sha256"] = canonical_sha256({
            key: value for key, value in payload.items()
            if key != "analysis_sha256"
        })
    else:
        payload["h1_sign_stability"]["baseline"][
            "descriptive_over_structural_points"
        ]["q05"] = 99.0
        unsigned = {
            key: value for key, value in payload.items()
            if key != "analysis_sha256"
        }
        payload["analysis_sha256"] = canonical_sha256(unsigned)
    analysis = tmp_path / "structural_sensitivity_analysis.json"
    _write_analysis(analysis, payload)
    with pytest.raises(ValueError):
        publish_analysis(analysis, tmp_path / "output")
    assert not (tmp_path / "output" / RECEIPT_NAME).exists()


def test_publication_rejects_an_incomplete_55_cell_inventory(tmp_path: Path) -> None:
    payload = _complete_panel_analysis()
    payload.pop("analysis_sha256")
    del payload["h2_sign_stability"]["baseline"]["mcp_only_minus_no_context"]
    payload["analysis_sha256"] = canonical_sha256(payload)
    analysis = tmp_path / "structural_sensitivity_analysis.json"
    _write_analysis(analysis, payload)
    with pytest.raises(ValueError, match="exact five contrasts"):
        publish_analysis(analysis, tmp_path / "output")
