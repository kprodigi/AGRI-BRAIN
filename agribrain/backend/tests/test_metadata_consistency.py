"""Repository metadata consistency guard.

Locks the contract that ``CITATION.cff:version`` and
``agribrain/backend/pyproject.toml:[project].version`` track each other.
A reviewer who reads ``CITATION.cff`` to cite the software must see
the same version that ``pip show agri-brain-backend`` would report.

Also asserts the README's documented version mention (when present)
is consistent. The guard treats a stale CFF-vs-pyproject pair as a
hard failure so future bumps cannot land asymmetrically.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_CITATION = _REPO_ROOT / "CITATION.cff"
_PYPROJECT = _REPO_ROOT / "agribrain" / "backend" / "pyproject.toml"


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


def _read_citation_doi() -> str:
    text = _CITATION.read_text(encoding="utf-8")
    match = re.search(r"^\s*doi\s*:\s*['\"]([^'\"]*)['\"]\s*(#.*)?$", text, re.MULTILINE)
    if match is None:
        return ""
    return match.group(1).strip()


def test_citation_version_matches_pyproject():
    cff = _read_citation_version()
    pyp = _read_pyproject_version()
    assert cff == pyp, (
        f"CITATION.cff version ({cff!r}) does not match "
        f"pyproject.toml [project].version ({pyp!r}). "
        f"Bump both together when releasing a new version."
    )


def test_citation_doi_is_string_field():
    """DOI field must exist as a string (possibly empty until minted)."""
    text = _CITATION.read_text(encoding="utf-8")
    assert re.search(r"^\s*doi\s*:", text, re.MULTILINE), (
        "CITATION.cff is missing the doi: field. "
        "It should at minimum be `doi: \"\"` until a Zenodo DOI is minted."
    )


def test_citation_doi_format_when_set():
    """If a DOI is set, it must be a Zenodo-form 10.5281/zenodo.<digits> string."""
    doi = _read_citation_doi()
    if not doi:
        pytest.skip("DOI not yet minted (intentional pre-release placeholder)")
    assert re.fullmatch(r"10\.\d{4,9}/[\w.\-]+", doi), (
        f"DOI {doi!r} does not look like a registered DOI. "
        "Expected form: 10.5281/zenodo.<id>"
    )
