#!/usr/bin/env python3
"""Publish deterministic tables and figures from a structural analysis JSON."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402
from pypdf import PdfReader  # noqa: E402

# The 800-DPI QC read-back is ~94.4 MP, above PIL's default decompression-bomb
# warning threshold (89.5 MP) though below its hard-error threshold; declare
# bounded headroom so the intentional read-back stays warning-free.
Image.MAX_IMAGE_PIXELS = 120_000_000

from ..analysis.publication_figure_style import (  # noqa: E402
    MINIMUM_PUBLICATION_DPI,
    PUBLICATION_DPI,
    SEMANTIC_COLORS,
    SEMANTIC_LINESTYLES,
    SEMANTIC_MARKERS,
    accessible_legend,
    apply_publication_style,
    save_figure_pair,
    style_axes,
)
from .design import canonical_sha256

ANALYSIS_SCHEMA_VERSION = 1
PUBLICATION_SCHEMA_VERSION = 2
CSV_NAME = "structural_sensitivity_summary.csv"
PNG_NAME = "structural_sensitivity_summary.png"
PDF_NAME = "structural_sensitivity_summary.pdf"
RECEIPT_NAME = "structural_sensitivity_publication_receipt.json"
SUMMARY_FIELDS = ("n", "mean", "std", "min", "q05", "median", "q95", "max")
CSV_FIELDS = (
    "family", "scenario", "contrast", "stressor", *SUMMARY_FIELDS,
    "positive_sign_fraction", "point_difference_above_0p005_fraction",
    "inside_strict_0p01_margin_fraction", "max_absolute_delta",
    "all_cells_have_nonzero_exposure",
)
FIGURE_STYLE_CONTRACT = "agribrain_publication_figure_style_v2"
REFERENCE_COLOR = "#4D4D4D"
H2_STYLE_KEYS = {
    "mcp_only_minus_no_context": "mcp_only",
    "pirag_only_minus_no_context": "pirag_only",
    "agribrain_minus_mcp_only": "agribrain",
    "agribrain_minus_pirag_only": "no_slca",
    "synergy_full_minus_mcp_minus_retrieval_plus_no_context": (
        "agribrain_standard_rag"
    ),
}
H3_STYLE_KEYS = {
    "sensor_noise": "agribrain",
    "missing_data": "no_context",
    "telemetry_delay": "hybrid_rl",
    "mcp_fault_injection": "mcp_only",
    "compounded": "agribrain_no_peer",
}
H2_LABELS = {
    "mcp_only_minus_no_context": "MCP − No-context",
    "pirag_only_minus_no_context": "Retrieval − No-context",
    "agribrain_minus_mcp_only": "Full − MCP",
    "agribrain_minus_pirag_only": "Full − Retrieval",
    "synergy_full_minus_mcp_minus_retrieval_plus_no_context": "Synergy",
}
H3_LABELS = {
    "sensor_noise": "Sensor noise",
    "missing_data": "Missing data",
    "telemetry_delay": "Telemetry delay",
    "mcp_fault_injection": "MCP fault",
    "compounded": "Compounded",
}
STRUCTURAL_SCENARIOS = (
    "adaptive_pricing", "baseline", "cyber_outage", "heatwave",
    "overproduction",
)
STRUCTURAL_H2_CONTRASTS = tuple(H2_STYLE_KEYS)
STRUCTURAL_H3_STRESSORS = tuple(H3_STYLE_KEYS)
EXPECTED_STRUCTURAL_SUMMARY_ROWS = 55


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _finite_number(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{where} must be finite")
    return numeric


def _validate_summary(value: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(SUMMARY_FIELDS):
        raise ValueError(f"{where} has an unexpected summary schema")
    summary = dict(value)
    n = summary["n"]
    if isinstance(n, bool) or not isinstance(n, int) or n != 100:
        raise ValueError(f"{where}.n must equal the 100 structural design points")
    for field in SUMMARY_FIELDS[1:]:
        _finite_number(summary[field], where=f"{where}.{field}")
    ordered = [summary[field] for field in ("min", "q05", "median", "q95", "max")]
    if ordered != sorted(ordered):
        raise ValueError(f"{where} quantiles are not monotone")
    return summary


def validate_analysis(payload: Mapping[str, Any]) -> None:
    """Fail closed unless payload is a self-hashed analysis-v1 report."""

    if not isinstance(payload, dict):
        raise ValueError("structural analysis must be a JSON object")
    unsigned = dict(payload)
    claimed = unsigned.pop("analysis_sha256", None)
    if claimed != canonical_sha256(unsigned):
        raise ValueError("structural analysis SHA-256 does not match its content")
    expected = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_label": "structural sensitivity",
        "probability_interpretation": False,
        "n_design_points": 100,
        "n_parameters": 29,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"structural analysis has invalid {field}")
    h1 = payload.get("h1_sign_stability")
    h2 = payload.get("h2_sign_stability")
    if not isinstance(h1, dict) or set(h1) != set(STRUCTURAL_SCENARIOS):
        raise ValueError("structural analysis lacks the exact five-scenario H1 panel")
    if not isinstance(h2, dict) or set(h2) != set(STRUCTURAL_SCENARIOS):
        raise ValueError("structural analysis lacks the exact five-scenario H2 panel")
    for scenario in STRUCTURAL_SCENARIOS:
        if not isinstance(h1[scenario], dict):
            raise ValueError(f"H1 scenario {scenario!r} is malformed")
        if (
            not isinstance(h2[scenario], dict)
            or set(h2[scenario]) != set(STRUCTURAL_H2_CONTRASTS)
        ):
            raise ValueError(
                f"H2 scenario {scenario!r} lacks the exact five contrasts"
            )
    h3 = payload.get("h3_margin_stability")
    if not isinstance(h3, dict) or h3.get("strict_margin") != 0.01 or not isinstance(
        h3.get("cells"), dict,
    ):
        raise ValueError("structural analysis has invalid h3_margin_stability")
    h3_cells = h3["cells"]
    if set(h3_cells) != set(STRUCTURAL_SCENARIOS):
        raise ValueError("structural analysis lacks the exact five-scenario H3 panel")
    for scenario in STRUCTURAL_SCENARIOS:
        if (
            not isinstance(h3_cells[scenario], dict)
            or set(h3_cells[scenario]) != set(STRUCTURAL_H3_STRESSORS)
        ):
            raise ValueError(
                f"H3 scenario {scenario!r} lacks the exact five stressors"
            )


def summary_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten the existing descriptive statistics without recomputing them."""

    validate_analysis(payload)
    rows: list[dict[str, Any]] = []

    def add_row(
        family: str, scenario: str, contrast: str, cell: Mapping[str, Any],
        *, stressor: str = "",
    ) -> None:
        summary = _validate_summary(
            cell.get("descriptive_over_structural_points"),
            where=f"{family}.{scenario}.{contrast}",
        )
        row = {field: "" for field in CSV_FIELDS}
        row.update({
            "family": family, "scenario": scenario,
            "contrast": contrast, "stressor": stressor, **summary,
        })
        for field in CSV_FIELDS[len(SUMMARY_FIELDS) + 4:]:
            if field in cell:
                value = cell[field]
                if field == "all_cells_have_nonzero_exposure":
                    if not isinstance(value, bool):
                        raise ValueError(f"{family}.{scenario}.{contrast}.{field} is invalid")
                else:
                    _finite_number(
                        value, where=f"{family}.{scenario}.{contrast}.{field}",
                    )
                row[field] = value
        rows.append(row)

    for scenario in sorted(payload["h1_sign_stability"]):
        cell = payload["h1_sign_stability"][scenario]
        add_row("H1", scenario, str(cell.get("contrast", "")), cell)
    for scenario in sorted(payload["h2_sign_stability"]):
        contrasts = payload["h2_sign_stability"][scenario]
        if not isinstance(contrasts, dict) or not contrasts:
            raise ValueError(f"H2 scenario {scenario!r} is invalid")
        for contrast in sorted(contrasts):
            add_row("H2", scenario, contrast, contrasts[contrast])
    for scenario in sorted(payload["h3_margin_stability"]["cells"]):
        stressors = payload["h3_margin_stability"]["cells"][scenario]
        if not isinstance(stressors, dict) or not stressors:
            raise ValueError(f"H3 scenario {scenario!r} is invalid")
        for stressor in sorted(stressors):
            cell = stressors[stressor]
            add_row(
                "H3", scenario, str(cell.get("contrast", "")), cell,
                stressor=stressor,
            )
    if len(rows) != EXPECTED_STRUCTURAL_SUMMARY_ROWS:
        raise ValueError(
            "structural summary does not contain the exact 5 H1 + 25 H2 + "
            "25 H3 rows"
        )
    return rows


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _ordered_values(
    rows: list[dict[str, Any]], field: str, preferred: Mapping[str, str],
) -> list[str]:
    observed = {str(row[field]) for row in rows}
    return [value for value in preferred if value in observed] + sorted(
        observed - set(preferred)
    )


def _pretty_label(value: str) -> str:
    replacements = {
        "agribrain": "AGRI-BRAIN",
        "mcp": "MCP",
        "pirag": "PIR",
        "ari": "ARI",
    }
    words = value.replace("-", "_").split("_")
    return " ".join(replacements.get(word.lower(), word.capitalize()) for word in words)


def _style_key(value: str, index: int, preferred: Mapping[str, str]) -> str:
    if value in preferred:
        return preferred[value]
    available = tuple(SEMANTIC_COLORS)
    return available[index % len(available)]


def _offsets(n_series: int) -> list[float]:
    if n_series <= 1:
        return [0.0]
    width = 0.52
    return [
        -width / 2.0 + width * index / (n_series - 1)
        for index in range(n_series)
    ]


def _panel_key(axis: Any, *, ncol: int | None = None) -> None:
    """Draw the panel key in reserved space above the axes.

    The same contract the rest of the figure suite uses: a frameless
    horizontal key under the panel title, never inside the data area, so it
    cannot sit on a marker or an error bar.
    """
    handles, labels = axis.get_legend_handles_labels()
    if not handles:
        return
    if ncol is None:
        ncol = len(handles)
    rows = -(-len(handles) // ncol)
    legend = accessible_legend(
        axis, handles=handles, labels=labels,
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol,
        frameon=False, borderaxespad=0.0, handlelength=1.8,
        handletextpad=0.5, columnspacing=1.6, labelspacing=0.3,
    )
    # The key inherits the shared bold weight; the figure is printed bold
    # throughout and the hierarchy is carried by size.
    # These panels title with loc="left"; get_title() defaults to the centre
    # slot and would hand back an empty string, silently dropping the title.
    axis.set_title(axis.get_title(loc="left"), loc="left", fontweight="bold",
                   pad=25.0 * rows + 12.0)


def _style_horizontal_axis(
    axis: Any, scenarios: list[str], *, title: str, xlabel: str,
) -> None:
    axis.set_yticks(
        list(range(len(scenarios))),
        [_pretty_label(scenario) for scenario in scenarios],
    )
    axis.invert_yaxis()
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_xlabel(xlabel, fontweight="normal")
    style_axes(axis)
    axis.grid(False)
    axis.grid(
        axis="x", color="#C7C7C7", linewidth=0.7, alpha=0.6,
    )
    axis.axvline(
        0.0, color=REFERENCE_COLOR, linestyle="-", linewidth=1.4, zorder=1,
    )


def _draw_figure(rows: list[dict[str, Any]]) -> plt.Figure:
    """Draw the existing H1/H2/H3 summaries without recomputing any values."""

    apply_publication_style()
    figure, axes = plt.subplots(
        3, 1, figsize=(18.0, 11.5), constrained_layout=True,
    )
    figure.get_layout_engine().set(hspace=0.16, h_pad=0.10)

    h1_rows = [row for row in rows if row["family"] == "H1"]
    h1_scenarios = sorted({str(row["scenario"]) for row in h1_rows})
    h1_lookup = {str(row["scenario"]): row for row in h1_rows}
    h1_values = [float(h1_lookup[scenario]["mean"]) for scenario in h1_scenarios]
    h1_errors = [
        [
            h1_values[index] - float(h1_lookup[scenario]["q05"])
            for index, scenario in enumerate(h1_scenarios)
        ],
        [
            float(h1_lookup[scenario]["q95"]) - h1_values[index]
            for index, scenario in enumerate(h1_scenarios)
        ],
    ]
    h1_style = "agribrain"
    axes[0].errorbar(
        h1_values,
        list(range(len(h1_scenarios))),
        xerr=h1_errors,
        fmt=SEMANTIC_MARKERS[h1_style],
        linestyle="none",
        color=SEMANTIC_COLORS[h1_style],
        ecolor=SEMANTIC_COLORS[h1_style],
        markeredgecolor="white",
        markeredgewidth=0.9,
        markersize=10,
        elinewidth=2.0,
        capsize=5,
        capthick=1.8,
        zorder=3,
    )
    _style_horizontal_axis(
        axes[0],
        h1_scenarios,
        title="(a) H1: Full − No-context",
        xlabel="ΔARI (5th–95th percentile over 100 points)",
    )

    h2_rows = [row for row in rows if row["family"] == "H2"]
    h2_scenarios = sorted({str(row["scenario"]) for row in h2_rows})
    h2_series = _ordered_values(h2_rows, "contrast", H2_STYLE_KEYS)
    h2_lookup = {
        (str(row["scenario"]), str(row["contrast"])): row for row in h2_rows
    }
    for offset, contrast, series_index in zip(
        _offsets(len(h2_series)), h2_series, range(len(h2_series)), strict=True,
    ):
        observations = [
            (scenario_index, h2_lookup[(scenario, contrast)])
            for scenario_index, scenario in enumerate(h2_scenarios)
            if (scenario, contrast) in h2_lookup
        ]
        values = [float(row["mean"]) for _, row in observations]
        errors = [
            [values[index] - float(row["q05"]) for index, (_, row) in enumerate(observations)],
            [float(row["q95"]) - values[index] for index, (_, row) in enumerate(observations)],
        ]
        style_key = _style_key(contrast, series_index, H2_STYLE_KEYS)
        axes[1].errorbar(
            values,
            [scenario_index + offset for scenario_index, _ in observations],
            xerr=errors,
            fmt=SEMANTIC_MARKERS[style_key],
            linestyle="none",
            color=SEMANTIC_COLORS[style_key],
            ecolor=SEMANTIC_COLORS[style_key],
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=9,
            elinewidth=1.8,
            capsize=4,
            capthick=1.6,
            label=H2_LABELS.get(contrast, _pretty_label(contrast)),
            zorder=3,
        )
    _style_horizontal_axis(
        axes[1],
        h2_scenarios,
        title="(b) H2: Channel Contrasts",
        xlabel="ΔARI (5th–95th percentile over 100 points)",
    )
    _panel_key(axes[1], ncol=5)

    h3_rows = [row for row in rows if row["family"] == "H3"]
    h3_scenarios = sorted({str(row["scenario"]) for row in h3_rows})
    h3_series = _ordered_values(h3_rows, "stressor", H3_STYLE_KEYS)
    h3_lookup = {
        (str(row["scenario"]), str(row["stressor"])): row for row in h3_rows
    }
    for offset, stressor, series_index in zip(
        _offsets(len(h3_series)), h3_series, range(len(h3_series)), strict=True,
    ):
        observations = [
            (scenario_index, h3_lookup[(scenario, stressor)])
            for scenario_index, scenario in enumerate(h3_scenarios)
            if (scenario, stressor) in h3_lookup
        ]
        style_key = _style_key(stressor, series_index, H3_STYLE_KEYS)
        axes[2].plot(
            [float(row["max_absolute_delta"]) for _, row in observations],
            [scenario_index + offset for scenario_index, _ in observations],
            linestyle="none",
            marker=SEMANTIC_MARKERS[style_key],
            color=SEMANTIC_COLORS[style_key],
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=9,
            label=H3_LABELS.get(stressor, _pretty_label(stressor)),
            zorder=3,
        )
    _style_horizontal_axis(
        axes[2],
        h3_scenarios,
        title="(c) H3: Stressed − Nominal, Worst Case",
        xlabel="Max |ΔARI| over 100 points",
    )
    axes[2].axvline(
        0.01,
        color=SEMANTIC_COLORS["mcp_only"],
        linestyle=SEMANTIC_LINESTYLES["hybrid_rl"],
        linewidth=2.0,
        label="Margin 0.01",
        zorder=2,
    )
    _panel_key(axes[2], ncol=6)

    figure.suptitle(
        "Structural Sensitivity across the 100-Point Factor Box",
        fontweight="bold",
    )
    return figure


_VECTOR_OPERATOR = re.compile(
    rb"(?:^|\s)(?:m|l|c|v|y|h|re|S|s|f\*?|B\*?|b\*?|n)(?=\s|$)",
)


def _inspect_png(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        dpi = image.info.get("dpi")
        if image.format != "PNG" or dpi is None or len(dpi) != 2:
            raise ValueError("structural publication PNG lacks valid DPI metadata")
        observed_dpi = tuple(float(value) for value in dpi)
        if min(observed_dpi) < PUBLICATION_DPI - 1.0:
            raise ValueError("structural publication PNG is not rendered at 800 DPI")
        if min(image.size) < 3000:
            raise ValueError("structural publication PNG pixel dimensions are too small")
        if image.mode not in {"RGB", "RGBA"}:
            raise ValueError("structural publication PNG uses an unexpected colour mode")
        return {
            "format": image.format,
            "width_px": image.width,
            "height_px": image.height,
            "dpi": [round(value, 3) for value in observed_dpi],
            "colour_mode": image.mode,
        }


def _inspect_pdf(path: Path) -> dict[str, Any]:
    reader = PdfReader(path, strict=True)
    if reader.is_encrypted or len(reader.pages) != 1:
        raise ValueError("structural publication PDF must be one unencrypted page")
    page = reader.pages[0]
    resources = page["/Resources"].get_object()
    fonts = resources.get("/Font", {}).get_object()
    subtypes: set[str] = set()
    embedded_programs = 0
    for reference in fonts.values():
        font = reference.get_object()
        candidates = [font]
        candidates.extend(
            descendant.get_object() for descendant in font.get("/DescendantFonts", [])
        )
        for candidate in candidates:
            subtypes.add(str(candidate.get("/Subtype", "")))
            descriptor = candidate.get("/FontDescriptor")
            if descriptor is not None:
                descriptor = descriptor.get_object()
                embedded_programs += int(any(
                    key in descriptor for key in ("/FontFile", "/FontFile2", "/FontFile3")
                ))
    if "/Type3" in subtypes:
        raise ValueError("structural publication PDF contains Type 3 fonts")
    if not ({"/TrueType", "/CIDFontType2"} & subtypes) or embedded_programs <= 0:
        raise ValueError("structural publication PDF lacks embedded TrueType fonts")
    contents = page.get_contents()
    content_bytes = b"" if contents is None else contents.get_data()
    vector_present = _VECTOR_OPERATOR.search(content_bytes) is not None
    if not vector_present:
        raise ValueError("structural publication PDF lacks vector drawing operators")
    xobjects = resources.get("/XObject", {}).get_object()
    raster_images = sum(
        reference.get_object().get("/Subtype") == "/Image"
        for reference in xobjects.values()
    )
    if raster_images:
        raise ValueError("structural publication PDF unexpectedly rasterizes the figure")
    return {
        "page_count": 1,
        "width_points": round(float(page.mediabox.width), 3),
        "height_points": round(float(page.mediabox.height), 3),
        "font_subtypes": sorted(subtypes),
        "embedded_font_programs": embedded_programs,
        "type3_font_count": 0,
        "vector_drawing_operators_present": True,
        "raster_image_xobjects": raster_images,
    }


def _render_figure_pair(
    figure: plt.Figure, output: Path, png_path: Path, pdf_path: Path,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        dir=output, prefix=".structural_figure_render_",
    ) as temporary_name:
        staging = Path(temporary_name)
        save_figure_pair(
            figure, staging, "structural_sensitivity_summary", dpi=PUBLICATION_DPI,
        )
        staged_png = staging / PNG_NAME
        staged_pdf = staging / PDF_NAME
        quality = {
            "png": _inspect_png(staged_png),
            "pdf": _inspect_pdf(staged_pdf),
        }
        _atomic_bytes(png_path, staged_png.read_bytes())
        _atomic_bytes(pdf_path, staged_pdf.read_bytes())
    return quality


def _figure_style_record(
    rows: list[dict[str, Any]], quality: Mapping[str, Any],
) -> dict[str, Any]:
    h2_rows = [row for row in rows if row["family"] == "H2"]
    h3_rows = [row for row in rows if row["family"] == "H3"]

    def series_record(
        values: list[str], preferred: Mapping[str, str],
    ) -> dict[str, dict[str, str]]:
        return {
            value: {
                "colour": SEMANTIC_COLORS[
                    style_key := _style_key(value, index, preferred)
                ],
                "marker": SEMANTIC_MARKERS[style_key],
            }
            for index, value in enumerate(values)
        }

    return {
        "contract": FIGURE_STYLE_CONTRACT,
        "shared_style_module": (
            "mvp.simulation.analysis.publication_figure_style"
        ),
        "requested_png_dpi": PUBLICATION_DPI,
        "minimum_png_dpi": MINIMUM_PUBLICATION_DPI,
        "pdf_fonttype": 42,
        "background": "white",
        "typography": {
            "panel_and_figure_titles": "bold",
            "axis_tick_and_legend_text": "normal",
        },
        "series": {
            "H1": {
                "agribrain_minus_no_context": {
                    "colour": SEMANTIC_COLORS["agribrain"],
                    "marker": SEMANTIC_MARKERS["agribrain"],
                },
            },
            "H2": series_record(
                _ordered_values(h2_rows, "contrast", H2_STYLE_KEYS), H2_STYLE_KEYS,
            ),
            "H3": series_record(
                _ordered_values(h3_rows, "stressor", H3_STYLE_KEYS), H3_STYLE_KEYS,
            ),
        },
        "reference_lines": {
            "zero": {"colour": REFERENCE_COLOR, "linestyle": "solid"},
            "strict_0p01_margin": {
                "colour": SEMANTIC_COLORS["mcp_only"],
                "linestyle": SEMANTIC_LINESTYLES["hybrid_rl"],
            },
        },
        "redundant_encodings": {
            "data_series": ["high-contrast colour", "marker shape"],
            "reference_lines": ["colour", "line style"],
            "uncertainty": ["horizontal interval", "end caps"],
        },
        "quality_checks": dict(quality),
    }


def publish_analysis(analysis_path: Path | str, output_dir: Path | str) -> dict[str, Any]:
    """Validate one analysis and atomically publish its derived artifacts."""

    source = Path(analysis_path).resolve()
    output = Path(output_dir).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = summary_rows(payload)
    csv_path, png_path, pdf_path = (
        output / CSV_NAME, output / PNG_NAME, output / PDF_NAME,
    )
    _atomic_bytes(csv_path, _csv_bytes(rows))
    figure = _draw_figure(rows)
    try:
        figure_quality = _render_figure_pair(
            figure, output, png_path, pdf_path,
        )
    finally:
        plt.close(figure)
    artifacts = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": _file_sha256(path)}
        for path in (csv_path, png_path, pdf_path)
    ]
    receipt: dict[str, Any] = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "receipt_type": "structural_sensitivity_publication_receipt",
        "analysis_label": "structural sensitivity",
        "probability_interpretation": False,
        "source": {
            "name": source.name,
            "bytes": source.stat().st_size,
            "literal_sha256": _file_sha256(source),
            "analysis_sha256": payload["analysis_sha256"],
            "source_commit": payload.get("source_commit"),
            "design_sha256": payload.get("design_sha256"),
            "manifest_sha256": payload.get("manifest_sha256"),
        },
        "derivation": (
            "Direct deterministic export of the existing descriptive H1/H2/H3 "
            "statistics; no simulation, refitting, or statistical recomputation."
        ),
        "row_count": len(rows),
        "figure_style": _figure_style_record(rows, figure_quality),
        "artifacts": artifacts,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    receipt_path = output / RECEIPT_NAME
    _atomic_bytes(
        receipt_path,
        (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8"),
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    receipt = publish_analysis(args.analysis, args.output_dir)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
