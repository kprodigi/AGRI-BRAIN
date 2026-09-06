"""Retired compatibility stub for a non-canonical exploratory runner.

This path previously implemented different scenarios, actions, waste, carbon,
and social-score equations from the publication benchmark while describing
itself as a reproducible large-scale experiment. Executing those equations
could create artifacts that looked publication-relevant but were not
methodologically comparable. The divergent implementation has therefore been
removed rather than hidden behind an opt-in flag.

Use the canonical runners instead:

* ``mvp/simulation/generate_results.py`` for the core simulator; or
* ``hpc/hpc_run.sh`` for the multi-seed publication workflow.
"""
from __future__ import annotations

from typing import NoReturn


DEPRECATION_MESSAGE = (
    "agribrain/backend/experiments/run_experiments.py is retired because its "
    "legacy equations do not match the publication methodology. No results "
    "were generated. Use mvp/simulation/generate_results.py for the canonical "
    "simulator or hpc/hpc_run.sh for the multi-seed publication workflow."
)


def main() -> NoReturn:
    """Fail fast so this legacy path cannot emit publication-like outputs."""
    raise SystemExit(DEPRECATION_MESSAGE)


if __name__ == "__main__":
    main()
