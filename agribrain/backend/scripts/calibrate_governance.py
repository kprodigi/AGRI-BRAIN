#!/usr/bin/env python3
"""Retired calibration shortcut based on the pre-final benchmark design."""
from __future__ import annotations

import sys
from collections.abc import Sequence


RETIRED = True
EXIT_RETIRED = 2
MIGRATION_MESSAGE = """\
RETIRED: calibrate_governance.py cannot execute or write results.
The former single-pass calibration was not part of the locked methodology.

Use the declared structural-sensitivity workflow instead:
  AGRIBRAIN_PARTITION=<partition> bash hpc/hpc_sensitivity_run.sh
  bash hpc/hpc_sensitivity_publish.sh <run_tag>
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Fail closed without calibrating or writing any artifact."""
    del argv
    print(MIGRATION_MESSAGE, file=sys.stderr, end="")
    return EXIT_RETIRED


if __name__ == "__main__":
    raise SystemExit(main())
