"""Repository metadata consistency guard.

Locks the contract that the four version sources in this repo track
each other:

  1. ``CITATION.cff`` ``version:`` field -- what reviewers cite
  2. ``agribrain/backend/pyproject.toml`` ``[project] version`` -- what
     ``pip show agri-brain-backend`` reports
  3. ``agribrain/frontend/package.json`` ``version`` -- what the
     dashboard footer renders
  4. ``README.md`` BibTeX block ``version = {...}`` -- what
     downstream researchers copy when citing

Pre-2026-05 only (1) and (2) were locked here, so the frontend and
README BibTeX silently drifted (frontend stuck at 1.1.0 while CITATION
+ pyproject moved to 1.2.0). The 2026-05 audit pass added the
remaining two as hard assertions.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_CITATION = _REPO_ROOT / "CITATION.cff"
_PYPROJECT = _REPO_ROOT / "agribrain" / "backend" / "pyproject.toml"
_FRONTEND_PKG = _REPO_ROOT / "agribrain" / "frontend" / "package.json"
_README = _REPO_ROOT / "README.md"


def _read_pyproject_version() -> str:
    text = _PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r"^\s*version\s*=\s*['\"]([^'\"]+)['\"]", text, re.MULTILINE)
    assert match, "pyproject.toml has no [project] version"
    return match.group(1).strip()


def _read_citation_version() -> str:
    text = _CITATION.read_text(encoding="utf-8")
    match = re.search(r"^\s*version\s*:\s*['\"]?([^'\"\n#]+?)['\"]?\s*(#.*)?$", text, re.MULTILINE)
    assert match, "CITATION.cff has no version field"
    return match.group(1).strip()


def test_citation_version_matches_pyproject():
    cff = _read_citation_version()
    pyp = _read_pyproject_version()
    assert cff == pyp, (
        f"CITATION.cff version ({cff!r}) does not match "
        f"pyproject.toml [project].version ({pyp!r}). "
        f"Bump both together when releasing a new version."
    )


def test_citation_omits_doi_field():
    """DOI is intentionally absent from CITATION.cff (post-2026-05 rule).

    Citation pattern is: version + git_commit + repository-code, no DOI.
    Re-introducing a top-level ``doi:`` field would silently land an
    empty / placeholder DOI in downstream tools that read CFF
    machine-readably (Zenodo, GitHub citation widget, BibTeX
    converters). If a Zenodo DOI is minted later, the canonical
    CFF 1.2 path is the ``identifiers:`` block, not a top-level
    ``doi:`` -- this guard catches accidental re-introduction of
    either.
    """
    text = _CITATION.read_text(encoding="utf-8")
    # Match a top-level ``doi:`` key (no leading whitespace beyond the
    # YAML root) so a future ``identifiers: - type: doi`` block under
    # a list element would not falsely fire this guard.
    assert not re.search(r"^doi\s*:", text, re.MULTILINE), (
        "CITATION.cff has a top-level `doi:` field. The post-2026-05 "
        "rule keeps this file DOI-free; cite via version + commit SHA + "
        "repository URL. If a DOI must be added in a later release, "
        "use the CFF 1.2 ``identifiers:`` block instead of the legacy "
        "top-level key."
    )


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
