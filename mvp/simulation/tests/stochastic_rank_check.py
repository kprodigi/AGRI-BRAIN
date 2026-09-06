#!/usr/bin/env python3
"""Retired one-pass rank-check runner.

The former executable used a different stochastic and episode protocol and
wrote directly into the canonical results directory. It is retained only as a
fail-closed migration pointer, not as a test or evidence generator.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence


RETIRED = True
EXIT_RETIRED = 2
MIGRATION_MESSAGE = """\
RETIRED: stochastic_rank_check.py cannot execute or write benchmark results.
Its one-pass stochastic design does not match the locked 3-adaptation plus
1-frozen-evaluation protocol.

Use the canonical workflow instead:
  AGRIBRAIN_PARTITION=<partition> bash hpc/hpc_run.sh
  bash hpc/hpc_publish.sh <run_tag>
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Fail closed without executing simulations or writing output."""
    del argv
    print(MIGRATION_MESSAGE, file=sys.stderr, end="")
    return EXIT_RETIRED


if __name__ == "__main__":
    raise SystemExit(main())
