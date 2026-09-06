#!/usr/bin/env python3
"""Retired compatibility entry point for the obsolete H2 hash-seed comparison.

The former module executed during import, consumed superseded
``decision_ledger_h2*`` directories, and wrote a noncanonical diagnostic into
the publication results directory. It cannot support the locked H2 evidence.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence


RETIRED = True
EXIT_RETIRED = 2
MIGRATION_MESSAGE = """\
RETIRED: mvp/simulation/_h2_stability_compare.py cannot run or write H2 evidence.
The former diagnostic used obsolete decision_ledger_h2 hash-seed directories
and is not part of the locked H2 analysis.

Use the canonical workflow from the repository root:
  AGRIBRAIN_PARTITION=<partition> bash hpc/hpc_run.sh

Its dependent hpc/hpc_publish.sh stage reads the run-scoped
decision_ledger_per_seed tree and produces validated H2 artifacts.
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Fail closed without parsing arguments, launching work, or writing files."""
    del argv
    print(MIGRATION_MESSAGE, file=sys.stderr, end="")
    return EXIT_RETIRED


if __name__ == "__main__":
    raise SystemExit(main())
