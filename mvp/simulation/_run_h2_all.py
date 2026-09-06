#!/usr/bin/env python3
"""Retired compatibility entry point for the obsolete standalone H2 pool.

The former launcher created an independent 20-seed treatment outside the
locked HPC dependency chain. Canonical H2 evidence now comes from the same
per-seed ledgers as the primary benchmark, so this path must remain inert.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence


RETIRED = True
EXIT_RETIRED = 2
MIGRATION_MESSAGE = """\
RETIRED: mvp/simulation/_run_h2_all.py cannot run or write H2 evidence.

Launch the canonical commit-bound treatment from the repository root:
  AGRIBRAIN_PARTITION=<partition> bash hpc/hpc_run.sh

Its dependent hpc/hpc_publish.sh stage consolidates the normal per-seed
decision ledgers and produces the validated H2/channel-attribution artifacts.
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Fail closed without launching processes, reading results, or writing."""
    del argv
    print(MIGRATION_MESSAGE, file=sys.stderr, end="")
    return EXIT_RETIRED


if __name__ == "__main__":
    raise SystemExit(main())
