"""Dataset schema drift guard.

The README's "Dataset" table documents the columns of
``agribrain/backend/src/data_spinach.csv``. The pre-2026-05 audit
found the table had ``demand_rate`` while the file's actual header
was ``demand_units``; the column count was also wrong. The fix
landed inline; this test locks the contract so any future CSV edit
or README edit that diverges fails CI.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA_CSV = _REPO_ROOT / "agribrain" / "backend" / "src" / "data_spinach.csv"
_README = _REPO_ROOT / "README.md"


_EXPECTED_COLUMNS = [
    "timestamp",
    "tempC",
    "RH",
    "shockG",
    "ambientC",
    "inventory_units",
    "demand_units",
    "quality_preference",
    "regulatory_temp_max",
]


def _csv_columns() -> list[str]:
    with _DATA_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return next(reader)


def _readme_documented_columns() -> list[str]:
    """Return the column-name backticks from the README's Dataset table.

    Parses rows of the form ``| `timestamp` | ISO 8601 ... |`` after
    the "## Dataset" heading. The order matters: the README is a
    single source of truth for the documented order.
    """
    text = _README.read_text(encoding="utf-8")
    # Locate the "## Dataset" section.
    match = re.search(r"##\s+Dataset[\s\S]*", text)
    assert match, "README.md has no '## Dataset' section"
    section = match.group(0)
    # Stop at the next H2 to avoid pulling in unrelated tables.
    section = re.split(r"\n##\s+", section, maxsplit=1)[0]
    rows = re.findall(r"^\|\s*`([^`]+)`\s*\|\s*([^|]+)\s*\|", section, re.MULTILINE)
    return [name.strip() for name, _desc in rows]


def test_csv_header_matches_expected_set():
    actual = _csv_columns()
    assert set(actual) == set(_EXPECTED_COLUMNS), (
        f"data_spinach.csv columns drifted.\n"
        f"  expected: {sorted(_EXPECTED_COLUMNS)}\n"
        f"  actual:   {sorted(actual)}"
    )


def test_csv_header_order_matches_expected():
    """Order matters because some tools index by position. Lock it."""
    actual = _csv_columns()
    assert actual == _EXPECTED_COLUMNS, (
        f"data_spinach.csv column order drifted.\n"
        f"  expected: {_EXPECTED_COLUMNS}\n"
        f"  actual:   {actual}"
    )


def test_readme_documents_every_csv_column():
    documented = _readme_documented_columns()
    csv_cols = _csv_columns()
    missing = [c for c in csv_cols if c not in documented]
    assert not missing, (
        f"README.md '## Dataset' table is missing columns from the CSV:\n"
        f"  missing: {missing}\n"
        f"  documented: {documented}"
    )


def test_readme_does_not_invent_columns():
    documented = _readme_documented_columns()
    csv_cols = set(_csv_columns())
    extra = [c for c in documented if c not in csv_cols]
    assert not extra, (
        f"README.md '## Dataset' table lists columns that are not in the CSV:\n"
        f"  extra: {extra}"
    )


def test_no_legacy_demand_rate_phrase_in_readme():
    """Lock out the pre-2026-05 ``demand_rate`` regression."""
    text = _README.read_text(encoding="utf-8")
    assert "`demand_rate`" not in text, (
        "README.md still contains the legacy `demand_rate` column "
        "(canonical name is `demand_units`)."
    )


def test_record_count_matches_advertised():
    """README says 288 records; the CSV must have 288 data rows."""
    with _DATA_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header
        rows = sum(1 for _ in reader)
    assert rows == 288, f"data_spinach.csv has {rows} rows; README claims 288"
