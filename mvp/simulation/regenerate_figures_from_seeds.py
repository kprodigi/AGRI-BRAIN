#!/usr/bin/env python3
"""Retired figure shortcut that could overwrite canonical publication output.

The former script chose an arbitrary trace seed and regenerated figures without
the complete commit-bound artifact set or final validation. That output cannot
represent the locked cross-seed publication analysis.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence


RETIRED = True
EXIT_RETIRED = 2
MIGRATION_MESSAGE = """\
RETIRED: regenerate_figures_from_seeds.py cannot write publication figures.
It did not require the complete validated cross-seed artifact set.

Use the canonical workflow instead:
  AGRIBRAIN_PARTITION=<partition> bash hpc/hpc_run.sh
  bash hpc/hpc_publish.sh <run_tag>
  python mvp/simulation/generate_figures.py --help
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Fail closed without reading a seed cache or writing a figure."""
    del argv
    print(MIGRATION_MESSAGE, file=sys.stderr, end="")
    return EXIT_RETIRED


if __name__ == "__main__":
    raise SystemExit(main())
