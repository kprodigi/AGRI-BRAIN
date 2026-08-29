"""Repository metadata consistency guard.

Locks the contract that the version sources in this repo track each other:

  1. ``agribrain/backend/pyproject.toml`` ``[project] version`` -- what
     ``pip show agri-brain-backend`` reports (the canonical version)
  2. ``agribrain/frontend/package.json`` ``version`` -- what the
     dashboard footer renders
  3. ``README.md`` BibTeX block ``version = {...}`` -- what downstream
     researchers copy when citing (optional; tests skip if absent)
  4. ``CITATION.cff`` ``version:`` -- the machine-readable citation
     surface GitHub renders (optional; test skips if absent)

History: ``CITATION.cff`` and the README citation/BibTeX block were
temporarily removed in 2026-06 (author identity withheld during
review) and restored for publication in 2026-07; the
CITATION-vs-pyproject check below skips gracefully when the file is
absent so the guard survives either state.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_PYPROJECT = _REPO_ROOT / "agribrain" / "backend" / "pyproject.toml"
_FRONTEND_PKG = _REPO_ROOT / "agribrain" / "frontend" / "package.json"
_README = _REPO_ROOT / "README.md"


def _read_pyproject_version() -> str:
    text = _PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r"^\s*version\s*=\s*['\"]([^'\"]+)['\"]", text, re.MULTILINE)
    assert match, "pyproject.toml has no [project] version"
    return match.group(1).strip()


def test_readme_omits_doi_in_bibtex():
    """README BibTeX block must not carry a ``doi = {...}`` line.

    Same rationale as test_citation_omits_doi_field: a stale or
    placeholder DOI in the headline citation block is worse than no
    DOI, because copy-pasters propagate it without re-checking.
    """
    text = _README.read_text(encoding="utf-8")
    if "@software{" not in text and "@misc{" not in text:
        pytest.skip("README has no BibTeX block (intentional)")
    bib_block = re.search(
        r"@(?:software|misc)\{[^}]*?\}",
        text,
        flags=re.DOTALL,
    )
    if bib_block is None:
        pytest.skip("README BibTeX block could not be parsed")
    assert "doi" not in bib_block.group(0).lower(), (
        "README BibTeX block contains a ``doi`` field. The post-2026-05 "
        "rule keeps the citation BibTeX DOI-free; cite via version + "
        "commit SHA + repository URL only."
    )


# ---------------------------------------------------------------------------
# 2026-05 audit pass: frontend + README BibTeX version pins.
# ---------------------------------------------------------------------------
def _read_frontend_version() -> str:
    pkg = json.loads(_FRONTEND_PKG.read_text(encoding="utf-8"))
    return str(pkg.get("version", ""))


def _read_readme_bibtex_version() -> str | None:
    """Return the version inside the BibTeX block in README.md, or
    None if the README has no BibTeX block (signals the test should
    skip rather than fail).
    """
    text = _README.read_text(encoding="utf-8")
    if "@software{" not in text and "@misc{" not in text:
        return None
    match = re.search(
        r"@(?:software|misc)\{[\s\S]*?version\s*=\s*\{([^}]+)\}",
        text,
    )
    if match is None:
        return None
    return match.group(1).strip()


def test_frontend_version_matches_pyproject():
    """The dashboard's package.json must track the backend release."""
    fe = _read_frontend_version()
    pyp = _read_pyproject_version()
    assert fe == pyp, (
        f"Frontend package.json version ({fe!r}) does not match "
        f"pyproject.toml [project].version ({pyp!r}). The dashboard "
        f"footer would render a different version than the backend "
        f"reports. Bump frontend/package.json:version when bumping "
        f"the backend release."
    )


def test_readme_bibtex_version_matches_pyproject():
    """README BibTeX is the most-copied citation surface; lock it to
    the backend release so external researchers cite the right version.
    """
    readme_v = _read_readme_bibtex_version()
    if readme_v is None:
        pytest.skip("README has no BibTeX block (intentional)")
    pyp = _read_pyproject_version()
    assert readme_v == pyp, (
        f"README BibTeX version = {{{readme_v}}} does not match "
        f"pyproject.toml [project].version ({pyp!r}). Update the "
        f"BibTeX block when bumping the release version."
    )


_CITATION_CFF = _REPO_ROOT / "CITATION.cff"


def test_citation_cff_version_matches_pyproject():
    """CITATION.cff is the machine-readable citation surface GitHub
    renders; lock its pinned version to the backend release."""
    if not _CITATION_CFF.exists():
        pytest.skip("CITATION.cff absent (anonymized-review state)")
    text = _CITATION_CFF.read_text(encoding="utf-8")
    match = re.search(r"^version:\s*['\"]?([^'\"\n]+)['\"]?\s*$", text, re.MULTILINE)
    assert match, "CITATION.cff has no version: field"
    cff_v = match.group(1).strip()
    pyp = _read_pyproject_version()
    assert cff_v == pyp, (
        f"CITATION.cff version ({cff_v!r}) does not match "
        f"pyproject.toml [project].version ({pyp!r}). Bump both "
        f"together with the release tag."
    )


def test_citation_cff_omits_doi_field():
    """Same DOI policy as the README BibTeX: no top-level doi: field;
    a future DOI belongs in an ``identifiers:`` block instead."""
    if not _CITATION_CFF.exists():
        pytest.skip("CITATION.cff absent (anonymized-review state)")
    text = _CITATION_CFF.read_text(encoding="utf-8")
    assert not re.search(r"^doi:", text, re.MULTILINE), (
        "CITATION.cff carries a top-level doi: field; per the citation "
        "policy, mint DOIs into an identifiers: block instead."
    )
