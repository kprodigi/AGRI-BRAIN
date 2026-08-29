#!/usr/bin/env python3
"""Retired compatibility entry point for the obsolete temporal-slice runner.

The former implementation used different stochastic seeds across policy arms,
ran learned modes for one updating episode, and wrote its within-trace
diagnostics into the publication results directory. It therefore cannot support
the locked paired, episode-indexed, frozen-evaluation protocol or an external
validity claim.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence


RETIRED = True
EXIT_RETIRED = 2
MIGRATION_MESSAGE = """\
RETIRED: run_temporal_stability.py cannot execute simulations or write evidence.
The former runner used unpaired stochastic seeds and did not implement the
episode-indexed adaptation/frozen-evaluation protocol. Its within-trace slices
also are not external validation.

Use the canonical workflow instead:
  AGRIBRAIN_PARTITION=<partition> bash hpc/hpc_run.sh
  bash hpc/hpc_publish.sh <run_tag>
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Fail closed without parsing arguments, launching work, or writing files."""
    del argv
    print(MIGRATION_MESSAGE, file=sys.stderr, end="")
    return EXIT_RETIRED


if __name__ == "__main__":
    raise SystemExit(main())
