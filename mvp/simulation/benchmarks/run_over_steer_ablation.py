#!/usr/bin/env python3
"""Retired compatibility entry point for the obsolete over-steering runner.

The former implementation did not satisfy the locked publication protocol: it
reused sequential stochastic state across warm-start calls, left retained
evaluation adaptive, and did not identify fresh episode-indexed streams. Its
outputs therefore cannot be interpreted as paired publication evidence.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence


RETIRED = True
EXIT_RETIRED = 2
MIGRATION_MESSAGE = """\
RETIRED: run_over_steer_ablation.py cannot execute simulations or write evidence.
The former runner did not implement fresh episode-indexed stochastic streams or
frozen retained evaluation, so its outputs are not publication evidence.

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
