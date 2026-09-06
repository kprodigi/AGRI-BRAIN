#!/usr/bin/env python3
"""Retired compatibility entry point; it must never create paper evidence.

The former conference-specific renderer pooled scenario means, clipped signed
effects, and reported a non-causal per-intervention ratio. Those estimands are
not part of the locked publication protocol. This file remains only so stale
commands fail with an explicit migration message instead of silently producing
plausible-looking figures or tables.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence


RETIRED = True
EXIT_RETIRED = 2
MIGRATION_MESSAGE = """\
RETIRED: mvp/simulation/focapo_figures.py is not a publication pipeline and
cannot generate figures or tables.

Use the canonical SLURM workflow from the repository root:
  AGRIBRAIN_PARTITION=<partition> bash hpc/hpc_run.sh

Its dependent hpc/hpc_publish.sh stage runs the canonical aggregation and
rendering path, including:
  mvp/simulation/benchmarks/aggregate_seeds.py
  mvp/simulation/generate_figures.py (via cache-bound figure regeneration)
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Fail closed without reading results or writing any artifact."""
    del argv
    print(MIGRATION_MESSAGE, file=sys.stderr, end="")
    return EXIT_RETIRED


if __name__ == "__main__":
    raise SystemExit(main())
