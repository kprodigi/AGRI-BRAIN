"""Strict validation for the canonical publication-figure artifact set.

The PNG/PDF pairs are generated into a new staging directory.  This module
checks the exact inventory, parses every image, and verifies that the
provenance record hashes the literal bytes before the publisher is allowed to
promote the set into ``results/``.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image
from pypdf import PdfReader

from mvp.simulation.analysis.publication_figure_style import (
    PUBLICATION_DPI,
    publication_style_contract,
)

# The validated PNGs are rendered by this pipeline itself at PUBLICATION_DPI
# (800) with a tight bounding box, so an 18x13-inch panel grid legitimately
# reaches ~182 MP — above Pillow's default decompression-bomb error ceiling
# (~179 MP), which is calibrated for untrusted downloads, not self-rendered
# publication artifacts. Keep an explicit bound (never None) with headroom:
# 800 dpi x ~400 in^2.
Image.MAX_IMAGE_PIXELS = 260_000_000

EXPECTED_FIGURE_STEMS = (
    "heatwave",
    "overproduction",
    "cyber_outage",
    "adaptive_pricing",
    "cross_scenario",
    "ablation",
    "transport_emissions",
    "performance_efficiency",
    "context_value",
    "stress_robustness",
)
EXPECTED_FIGURE_FILES = tuple(
    f"{stem}.{extension}"
    for stem in EXPECTED_FIGURE_STEMS
    for extension in ("png", "pdf")
)
EXPECTED_SEEDS = (
    7, 42, 99, 101, 202, 303, 404, 505, 606, 707,
    808, 909, 1010, 1111, 1212, 1313, 1337, 1414, 1515, 2024,
)
EXPECTED_PANEL_GROUPS = {
    "heatwave",
    "overproduction",
    "cyber_outage",
    "adaptive_pricing",
    "cross_scenario_and_secondary",
}
EXPECTED_AGGREGATE_INPUTS = (
    "benchmark_summary.json",
    "benchmark_significance.json",
    "channel_attribution_aggregate.json",
    "stress_passfail.csv",
)
PROVENANCE_NAME = "figure_provenance.json"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
MINIMUM_PNG_EDGE_PIXELS = 900
MINIMUM_PDF_EDGE_POINTS = 144.0
_VECTOR_OPERATOR = re.compile(
    rb"(?<![A-Za-z])(?:m|l|c|v|y|h|re|S|s|f|F|f\*|B|B\*|b|b\*|n)(?![A-Za-z])"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def figure_records(directory: Path) -> list[dict[str, Any]]:
    """Return deterministic literal-byte records for all 20 figure files."""

    return [
        {
            "file": name,
            "bytes": (directory / name).stat().st_size,
            "sha256": sha256_file(directory / name),
        }
        for name in sorted(EXPECTED_FIGURE_FILES)
    ]


def _validate_png(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:  # Pillow exposes multiple decoder exceptions.
        raise ValueError(f"publication PNG cannot be decoded: {path.name}: {exc}") from exc
    try:
        with Image.open(path) as image:
            image.load()
            if image.width <= 0 or image.height <= 0:
                raise ValueError("non-positive image dimensions")
            if min(image.width, image.height) < MINIMUM_PNG_EDGE_PIXELS:
                raise ValueError(
                    "short edge is below the publication minimum "
                    f"({min(image.width, image.height)} < "
                    f"{MINIMUM_PNG_EDGE_PIXELS} pixels)"
                )
            if image.mode not in {"RGB", "RGBA"}:
                raise ValueError(f"unsupported publication color mode {image.mode!r}")
            dpi = image.info.get("dpi")
            if (
                not isinstance(dpi, tuple)
                or len(dpi) != 2
                or min(float(value) for value in dpi)
                < PUBLICATION_DPI - 1.0
            ):
                raise ValueError(
                    "missing or insufficient publication DPI metadata "
                    f"(canonical render requires {PUBLICATION_DPI} DPI)"
                )
    except Exception as exc:
        raise ValueError(
            f"publication PNG fails the quality gate: {path.name}: {exc}"
        ) from exc


def _validate_pdf(path: Path) -> None:
    try:
        reader = PdfReader(path, strict=True)
        if reader.is_encrypted:
            raise ValueError("encrypted PDF")
        if len(reader.pages) != 1:
            raise ValueError(f"expected one page, found {len(reader.pages)}")
        page = reader.pages[0]
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        if width <= 0.0 or height <= 0.0:
            raise ValueError("non-positive page dimensions")
        if min(width, height) < MINIMUM_PDF_EDGE_POINTS:
            raise ValueError(
                "short page edge is below the publication minimum "
                f"({min(width, height):.1f} < {MINIMUM_PDF_EDGE_POINTS:.1f} points)"
            )
        contents = page.get_contents()
        if contents is None:
            raise ValueError("PDF page has no content stream")
        content_bytes = contents.get_data()
        if not content_bytes or not _VECTOR_OPERATOR.search(content_bytes):
            raise ValueError("PDF page lacks vector drawing primitives")
        resources = page.get("/Resources")
        if resources is None:
            raise ValueError("PDF page lacks resources")
        resources = resources.get_object()
        fonts = resources.get("/Font")
        if fonts is None:
            raise ValueError("PDF page lacks embedded or referenced fonts")
        fonts = fonts.get_object()
        if not fonts:
            raise ValueError("PDF page has an empty font resource dictionary")
        font_subtypes: set[str] = set()
        embedded_font_programs = 0
        for font_reference in fonts.values():
            font = font_reference.get_object()
            candidates = [font]
            candidates.extend(
                reference.get_object()
                for reference in font.get("/DescendantFonts", [])
            )
            for candidate in candidates:
                subtype = str(candidate.get("/Subtype", ""))
                font_subtypes.add(subtype)
                if subtype == "/Type3":
                    raise ValueError("PDF uses disallowed Type 3 fonts")
                descriptor = candidate.get("/FontDescriptor")
                if descriptor is not None:
                    descriptor = descriptor.get_object()
                    embedded_font_programs += int(any(
                        key in descriptor
                        for key in ("/FontFile", "/FontFile2", "/FontFile3")
                    ))
        if not ({"/TrueType", "/CIDFontType2"} & font_subtypes):
            raise ValueError("PDF does not reference TrueType-compatible fonts")
        if embedded_font_programs <= 0:
            raise ValueError("PDF does not embed its TrueType font program")
        xobjects = resources.get("/XObject")
        if xobjects is not None:
            xobjects = xobjects.get_object()
            if any(
                str(reference.get_object().get("/Subtype", "")) == "/Image"
                for reference in xobjects.values()
            ):
                raise ValueError("PDF contains raster image XObjects")
    except Exception as exc:  # pypdf raises several parser-specific classes.
        raise ValueError(f"publication PDF cannot be parsed: {path.name}: {exc}") from exc


def validate_figure_directory(
    directory: Path,
    *,
    source_commit: str,
    run_tag: str,
    staging_only: bool = False,
) -> dict[str, Any]:
    """Validate figure bytes and their self-binding provenance record.

    ``staging_only`` additionally requires that the directory contain no files
    other than the 20 canonical figures and ``figure_provenance.json``.
    Publication results directories may contain unrelated tables and ledgers,
    so their exactness check is limited to PNG/PDF files.
    """

    directory = directory.resolve(strict=True)
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"figure directory is not a real directory: {directory}")
    if not _HEX40.fullmatch(source_commit):
        raise ValueError("source_commit must be a full lowercase Git SHA-1")
    if not run_tag.strip() or Path(run_tag).name != run_tag:
        raise ValueError("run_tag must be a non-empty path-safe name")

    expected_images = set(EXPECTED_FIGURE_FILES)
    observed_images = {
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".pdf"}
    }
    if observed_images != expected_images:
        raise ValueError(
            "publication figure inventory is not the exact 10 PNG/PDF pairs: "
            f"missing={sorted(expected_images - observed_images)}, "
            f"unexpected={sorted(observed_images - expected_images)}"
        )
    if staging_only:
        observed_files = {path.name for path in directory.iterdir() if path.is_file()}
        expected_files = {PROVENANCE_NAME, *expected_images}
        if observed_files != expected_files or any(
            not path.is_file() for path in directory.iterdir()
        ):
            raise ValueError(
                "figure staging directory contains undeclared entries: "
                f"missing={sorted(expected_files - observed_files)}, "
                f"unexpected={sorted(observed_files - expected_files)}"
            )

    for name in sorted(expected_images):
        path = directory / name
        if path.is_symlink():
            raise ValueError(f"publication figure must not be a symlink: {name}")
        if path.stat().st_size <= 0:
            raise ValueError(f"publication figure is empty: {name}")
        if path.suffix == ".png":
            _validate_png(path)
        else:
            _validate_pdf(path)

    provenance_path = directory / PROVENANCE_NAME
    if not provenance_path.is_file() or provenance_path.is_symlink():
        raise ValueError("figure_provenance.json is missing or is a symlink")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"invalid figure_provenance.json: {exc}") from exc
    if not isinstance(provenance, dict):
        raise ValueError("figure_provenance.json is not an object")
    if provenance.get("schema_version") != 3:
        raise ValueError("figure provenance uses an obsolete schema")
    if provenance.get("source_commit") != source_commit:
        raise ValueError("figure provenance source_commit mismatch")
    if (
        provenance.get("source_commit_semantics")
        != "raw_input_simulation_commit"
        or provenance.get("simulation_source_commit") != source_commit
    ):
        raise ValueError("figure provenance raw-input identity is ambiguous")
    renderer_commit = provenance.get("renderer_code_commit")
    if not isinstance(renderer_commit, str) or not _HEX40.fullmatch(renderer_commit):
        raise ValueError("figure provenance renderer commit is invalid")
    if provenance.get("dual_provenance") is not (renderer_commit != source_commit):
        raise ValueError("figure provenance dual-provenance flag is inconsistent")
    if provenance.get("run_tag") != run_tag:
        raise ValueError("figure provenance run_tag mismatch")
    if provenance.get("seed_panel") != list(EXPECTED_SEEDS):
        raise ValueError("figure provenance does not name the exact sorted seed panel")
    if provenance.get("n_seed_envelopes_loaded") != len(EXPECTED_SEEDS):
        raise ValueError("figure provenance does not declare 20 loaded seed envelopes")
    expected_seed_files = {
        f"benchmark_seeds/seed_{seed}.json" for seed in EXPECTED_SEEDS
    }
    seed_records = provenance.get("seed_input_artifacts")
    if not isinstance(seed_records, list) or len(seed_records) != len(EXPECTED_SEEDS):
        raise ValueError("figure provenance lacks the exact seed-input byte records")
    if {
        record.get("file") for record in seed_records if isinstance(record, dict)
    } != expected_seed_files:
        raise ValueError("figure provenance seed-input file inventory is incomplete")
    for record in seed_records:
        if (
            not isinstance(record, dict)
            or record.get("seed") not in EXPECTED_SEEDS
            or record.get("file") != f"benchmark_seeds/seed_{record.get('seed')}.json"
            or not isinstance(record.get("bytes"), int)
            or record["bytes"] <= 0
            or not _HEX64.fullmatch(str(record.get("sha256", "")))
        ):
            raise ValueError("figure provenance contains a malformed seed-input record")
    aggregate_records = provenance.get("aggregate_input_artifacts")
    if not isinstance(aggregate_records, list) or len(aggregate_records) != len(
        EXPECTED_AGGREGATE_INPUTS
    ):
        raise ValueError("figure provenance lacks the exact aggregate-input records")
    if {
        record.get("file") for record in aggregate_records
        if isinstance(record, dict)
    } != set(EXPECTED_AGGREGATE_INPUTS):
        raise ValueError("figure provenance aggregate-input inventory is incomplete")
    for record in aggregate_records:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("bytes"), int)
            or record["bytes"] <= 0
            or not _HEX64.fullmatch(str(record.get("sha256", "")))
        ):
            raise ValueError(
                "figure provenance contains a malformed aggregate-input record"
            )
    if provenance.get("render_input_isolated_snapshot") is not True:
        raise ValueError("figure provenance does not attest isolated input rendering")
    if provenance.get("illustrative_seed") != 42:
        raise ValueError("figure provenance illustrative seed must be 42")
    if provenance.get("publication_style") != publication_style_contract():
        raise ValueError(
            "figure provenance does not declare the exact publication style contract"
        )
    renderer_environment = provenance.get("renderer_environment")
    if not isinstance(renderer_environment, dict) or any(
        not isinstance(renderer_environment.get(name), str)
        or not renderer_environment[name].strip()
        for name in ("matplotlib", "numpy", "pillow")
    ):
        raise ValueError("figure provenance lacks renderer package versions")
    font_record = renderer_environment.get("resolved_font")
    if (
        not isinstance(font_record, dict)
        or not isinstance(font_record.get("resolved_family"), str)
        or not font_record["resolved_family"].strip()
        or not isinstance(font_record.get("resolved_path"), str)
        or not font_record["resolved_path"].strip()
        or not isinstance(font_record.get("file"), str)
        or not font_record["file"].strip()
        or not isinstance(font_record.get("bytes"), int)
        or font_record["bytes"] <= 0
        or not _HEX64.fullmatch(str(font_record.get("sha256", "")))
    ):
        raise ValueError("figure provenance lacks the resolved publication-font record")
    panels = provenance.get("panels")
    if not isinstance(panels, dict) or set(panels) != EXPECTED_PANEL_GROUPS:
        raise ValueError("figure provenance panel inventory is incomplete")
    cross = panels.get("cross_scenario_and_secondary")
    if not isinstance(cross, dict) or cross.get("fields") != list(
        EXPECTED_AGGREGATE_INPUTS
    ):
        raise ValueError("figure provenance cross-panel input inventory is incorrect")

    declared_records = provenance.get("rendered_artifacts")
    actual_records = figure_records(directory)
    if not isinstance(declared_records, list) or declared_records != actual_records:
        raise ValueError(
            "figure provenance does not hash-bind the exact rendered artifact bytes"
        )
    for record in declared_records:
        if (
            not isinstance(record.get("bytes"), int)
            or record["bytes"] <= 0
            or not _HEX64.fullmatch(str(record.get("sha256", "")))
        ):
            raise ValueError("figure provenance contains a malformed byte record")
    return provenance
