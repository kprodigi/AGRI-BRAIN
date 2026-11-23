"""Doc-vs-code drift guard for the stochastic-layer env-var defaults.

The 2026-05 review found that ``HOW_TO_RUN.md``'s STOCH_* table
declared values (1.5 / 5.0 / 0.18 / ...) that did not match the
defaults baked into ``mvp/simulation/stochastic.py`` (2.5 / 7.0 / 0.25
/ ...). Reviewers following the doc reproduced a different noise
envelope than the published HPC numbers.

The fix elevates ``stochastic.canonical_defaults()`` to single source
of truth. This test asserts the documented env-var tables in
``HOW_TO_RUN.md``, ``README.md``, ``.env.example``, and
``.env.prod.example`` agree with the canonical mapping. Future drift
between the tables and the code defaults fails CI.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SIM = _REPO_ROOT / "mvp" / "simulation"

# Make the simulator package importable without an editable install.
if str(_SIM) not in sys.path:
    sys.path.insert(0, str(_SIM))

import stochastic  # type: ignore  # noqa: E402


_HOW_TO_RUN = _REPO_ROOT / "HOW_TO_RUN.md"
_README = _REPO_ROOT / "README.md"
_ENV_DEV = _REPO_ROOT / ".env.example"
_ENV_PROD = _REPO_ROOT / ".env.prod.example"


def _normalise(raw: str) -> str:
    """Strip surrounding whitespace and trailing zeros so 0.10 == 0.1."""
    s = str(raw).strip().strip("`").strip()
    try:
        return f"{float(s):g}"
    except ValueError:
        return s


def _extract_md_table_defaults(path: Path) -> dict[str, str]:
    """Pull `| `KEY` | `VAL` | ...` rows out of a markdown env-var table."""
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"^\|\s*`(STOCH_[A-Z_]+)`\s*\|\s*`([^`]+)`\s*\|", re.MULTILINE
    )
    return {m.group(1): _normalise(m.group(2)) for m in pattern.finditer(text)}


def _extract_env_file_defaults(path: Path) -> dict[str, str]:
    """Pull KEY=VALUE assignments out of a dotenv-style file."""
    text = path.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.split("#", 1)[0].strip().strip('"').strip("'")
        if key.startswith("STOCH_"):
            out[key] = _normalise(value)
    return out


def _canonical_normalised() -> dict[str, str]:
    return {k: _normalise(v) for k, v in stochastic.canonical_defaults().items()}


def test_canonical_defaults_returns_independent_copy():
    a = stochastic.canonical_defaults()
    a["STOCH_TEMP_STD_C"] = "999"
    assert stochastic.canonical_defaults()["STOCH_TEMP_STD_C"] != "999"


@pytest.mark.parametrize(
    "doc_path",
    [_HOW_TO_RUN, _README],
    ids=["HOW_TO_RUN.md", "README.md"],
)
def test_md_table_matches_canonical(doc_path: Path):
    canonical = _canonical_normalised()
    documented = _extract_md_table_defaults(doc_path)
    if not documented:
        pytest.skip(f"{doc_path.name} has no STOCH_ table; nothing to check")
    drifts = []
    for key, doc_value in documented.items():
        if key not in canonical:
            drifts.append(f"  {doc_path.name} declares {key} but stochastic.py does not")
            continue
        if doc_value != canonical[key]:
            drifts.append(
                f"  {doc_path.name}:{key} = {doc_value!r}; "
                f"canonical = {canonical[key]!r}"
            )
    assert not drifts, (
        "Documented STOCH defaults disagree with stochastic.canonical_defaults():\n"
        + "\n".join(drifts)
    )


@pytest.mark.parametrize(
    "env_path",
    [_ENV_DEV, _ENV_PROD],
    ids=[".env.example", ".env.prod.example"],
)
def test_env_file_matches_canonical(env_path: Path):
    canonical = _canonical_normalised()
    documented = _extract_env_file_defaults(env_path)
    if not documented:
        pytest.skip(f"{env_path.name} has no STOCH_ assignments; nothing to check")
    drifts = []
    for key, doc_value in documented.items():
        if key not in canonical:
            drifts.append(f"  {env_path.name} declares {key} but stochastic.py does not")
            continue
        if doc_value != canonical[key]:
            drifts.append(
                f"  {env_path.name}:{key} = {doc_value!r}; "
                f"canonical = {canonical[key]!r}"
            )
    assert not drifts, (
        "Env-example STOCH defaults disagree with canonical_defaults():\n"
        + "\n".join(drifts)
    )


def test_no_seven_source_phrase_remains():
    """Lock the wording: the calibration is 8 sources + 1 orthogonal lag."""
    forbidden = re.compile(r"7[- ]source|seven[- ]source", re.IGNORECASE)
    offenders: list[str] = []
    for path in [_HOW_TO_RUN, _README]:
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if forbidden.search(line):
                offenders.append(f"{path.name}:{i}: {line.strip()}")
    assert not offenders, (
        "Stale '7-source' phrasing remains; the calibration is 8 sources:\n"
        + "\n".join(offenders)
    )
