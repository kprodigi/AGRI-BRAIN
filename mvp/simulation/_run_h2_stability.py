#!/usr/bin/env python3
"""Retired compatibility entry point for obsolete H2 hash-seed runs.

The former helper generated noncanonical side treatments under alternate
hash seeds. Those outputs are not part of the locked H2 estimand or evidence
panel and must not be regenerated into the publication results tree.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence


RETIRED = True
EXIT_RETIRED = 2
MIGRATION_MESSAGE = """\
RETIRED: mvp/simulation/_run_h2_stability.py cannot run or write H2 evidence.

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
