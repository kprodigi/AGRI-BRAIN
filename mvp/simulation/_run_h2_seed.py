#!/usr/bin/env python3
"""Retired compatibility entry point for the obsolete standalone H2 run.

H2 channel evidence is derived from the ordinary, commit-bound per-seed
decision ledgers. Running a second instrumented treatment would create a
different evidence population and could leave plausible-looking side
artifacts beside the canonical results, so this path now fails closed.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence


RETIRED = True
EXIT_RETIRED = 2
MIGRATION_MESSAGE = """\
RETIRED: mvp/simulation/_run_h2_seed.py cannot run or write H2 evidence.

Launch the canonical commit-bound treatment from the repository root:
  AGRIBRAIN_PARTITION=<partition> bash hpc/hpc_run.sh

Its dependent hpc/hpc_publish.sh stage consolidates the normal per-seed
decision ledgers and produces the validated H2/channel-attribution artifacts.
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Fail closed without reading results, running simulations, or writing."""
    del argv
    print(MIGRATION_MESSAGE, file=sys.stderr, end="")
    return EXIT_RETIRED


if __name__ == "__main__":
    raise SystemExit(main())
