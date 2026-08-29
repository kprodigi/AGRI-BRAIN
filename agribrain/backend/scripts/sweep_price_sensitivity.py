#!/usr/bin/env python3
"""Retired price-sensitivity shortcut based on the pre-final methodology."""
from __future__ import annotations

import sys
from collections.abc import Sequence


RETIRED = True
EXIT_RETIRED = 2
MIGRATION_MESSAGE = """\
RETIRED: sweep_price_sensitivity.py cannot execute or write results.
The former sweep used obsolete one-pass equations and was not a prespecified
publication analysis.

Use the declared structural-sensitivity workflow instead:
  AGRIBRAIN_PARTITION=<partition> bash hpc/hpc_sensitivity_run.sh
  bash hpc/hpc_sensitivity_publish.sh <run_tag>
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Fail closed without executing a sweep or writing any artifact."""
    del argv
    print(MIGRATION_MESSAGE, file=sys.stderr, end="")
    return EXIT_RETIRED


if __name__ == "__main__":
    raise SystemExit(main())
