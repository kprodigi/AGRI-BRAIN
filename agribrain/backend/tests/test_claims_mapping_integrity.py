"""Claims-to-evidence mapping integrity guard.

The 2026-05 audit found that ``docs/CLAIMS_TO_EVIDENCE.md`` did not
list a source for fig1; reviewers looking for the architecture
diagram had no path to follow. The fix added
``docs/figures/fig1_architecture.{md,svg,pdf}`` and updated the
crosswalk. This test pins:

* every figure mentioned in the crosswalk has an artifact path,
* fig1 explicitly points at ``docs/figures/`` and the source ``.md``,
* every simulator-produced figure id (fig2..fig10) corresponds to a
  tracked PNG + PDF in ``mvp/simulation/results/``.

Future drift between the table and the actual artifact tree fails CI.
"""
from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLAIMS = _REPO_ROOT / "docs" / "CLAIMS_TO_EVIDENCE.md"
_RESULTS = _REPO_ROOT / "mvp" / "simulation" / "results"
_FIGURES = _REPO_ROOT / "docs" / "figures"


def _crosswalk_rows() -> list[tuple[str, str]]:
    """Return (paper_element, artifact_source) pairs from the crosswalk table."""
    text = _CLAIMS.read_text(encoding="utf-8")
    section = re.search(r"## Figure/Table Crosswalk[\s\S]+?(?=\n##\s|\Z)", text)
    assert section, "CLAIMS_TO_EVIDENCE.md missing 'Figure/Table Crosswalk' section"
    rows = re.findall(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$", section.group(0), re.MULTILINE)
    # Drop the header rows
    return [(p, a) for p, a in rows if not re.match(r"-+", p) and "Paper element" not in p]


def test_crosswalk_has_fig1_entry():
    rows = _crosswalk_rows()
    fig1_rows = [r for r in rows if "Fig 1" in r[0] or "fig1" in r[1]]
    assert fig1_rows, "CLAIMS_TO_EVIDENCE.md crosswalk does not mention fig1"


def test_crosswalk_fig1_points_at_figures_dir():
    rows = _crosswalk_rows()
    fig1_rows = [r for r in rows if "Fig 1" in r[0]]
    assert fig1_rows, "fig1 row not found"
    artifact_text = fig1_rows[0][1]
    assert "docs/figures/" in artifact_text or "docs%2Ffigures" in artifact_text, (
        f"fig1 artifact must point at docs/figures/; got: {artifact_text!r}"
    )


def test_fig1_source_files_exist():
    """The Mermaid source must be present; SVG/PDF are renderable on demand."""
    md = _FIGURES / "fig1_architecture.md"
    assert md.exists(), f"fig1 architecture source missing at {md}"
    text = md.read_text(encoding="utf-8")
    assert "```mermaid" in text, "fig1 source must contain a fenced ```mermaid block"


def test_simulator_figs_present_in_results():
    """fig2..fig10 must each have a tracked PNG and PDF in results/."""
    rows = _crosswalk_rows()
    missing = []
    for elem, artifact in rows:
        m = re.search(r"\b(fig\d+_[a-z_]+)", artifact)
        if not m:
            continue
        fig_stem = m.group(1)
        if fig_stem.startswith("fig1_"):
            # Hand-authored; lives under docs/figures, checked separately.
            continue
        png = _RESULTS / f"{fig_stem}.png"
        pdf = _RESULTS / f"{fig_stem}.pdf"
        if not png.exists() or not pdf.exists():
            missing.append(f"{fig_stem}: png_exists={png.exists()} pdf_exists={pdf.exists()}")
    assert not missing, "Simulator figures referenced in crosswalk missing from results/:\n" + "\n".join(missing)


def test_no_orphan_figures_in_results():
    """Every fig*.png in results/ must appear in the crosswalk."""
    rows = _crosswalk_rows()
    crosswalk_text = " ".join(a for _e, a in rows)
    orphans = []
    for png in _RESULTS.glob("fig*.png"):
        stem = png.stem  # e.g. fig2_heatwave
        # Drop any known suffix variants (e.g. fig2_heatwave_dark vs main)
        if stem in crosswalk_text:
            continue
        # Tolerate fig*.png where the base stem (without trailing _<variant>) is named.
        base = re.sub(r"_(dark|light|small|large)$", "", stem)
        if base in crosswalk_text:
            continue
        orphans.append(stem)
    assert not orphans, (
        "Tracked figures in mvp/simulation/results/ are not mentioned "
        "in CLAIMS_TO_EVIDENCE.md crosswalk:\n" + "\n".join(orphans)
    )
