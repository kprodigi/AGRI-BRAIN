#!/usr/bin/env python3
"""Retired compatibility entry point for the obsolete local core runner.

The former runner encoded a different stage order and an obsolete deterministic
regression gate. Publication execution is now defined only by the commit-bound
SLURM dependency chain launched by ``hpc/hpc_run.sh``. Keeping this path as a
fail-closed stub prevents an old command from creating noncanonical results.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence


RETIRED = True
EXIT_RETIRED = 2
MIGRATION_MESSAGE = """\
RETIRED: mvp/simulation/reproduce_core.py cannot run or publish experiments.

Use the canonical workflow from the repository root:
  AGRIBRAIN_PARTITION=<partition> bash hpc/hpc_run.sh

That orchestrator enforces the locked seed -> stress -> publish dependency
chain, source identity, raw-panel validation, aggregation, rendering, manifest
verification, and publication-artifact validation.
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Fail closed without launching subprocesses or writing any artifact."""
    del argv
    print(MIGRATION_MESSAGE, file=sys.stderr, end="")
    return EXIT_RETIRED


if __name__ == "__main__":
    raise SystemExit(main())
