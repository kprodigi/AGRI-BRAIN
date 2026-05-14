"""Regenerate all figures from existing canonical seed JSONs — no new simulation.

The default ``python generate_figures.py`` invokes ``run_all()`` which kicks off
a single deterministic seed-42 simulation purely to populate the per-step trace
data for figs 2/3/4/5/8. That is wasteful and slightly misleading: the 20-seed
canonical benchmark already wrote those traces to disk under
``mvp/simulation/results/benchmark_seeds/seed_*.json``, and seed_42 is one of
them.

This script loads the canonical seed_42.json (full per-step traces for all 5
scenarios x static/hybrid_rl/agribrain) and constructs the ``data`` dict shape
that ``generate_figures.generate_all_figures(data=data)`` expects, then renders
every figure without touching the simulator. The bootstrap CIs for figs
6/7/8/9/10 come from ``benchmark_summary.json`` / ``benchmark_significance.json``
on disk as usual, so the cross-method bars and paired-test results remain the
canonical 20-seed values.

Usage (run from repo root):
    python mvp/simulation/regenerate_figures_from_seeds.py

If a seed file other than seed_42 should be the trace exemplar (e.g. for a
sensitivity check), pass ``--trace-seed <N>``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

RESULTS_DIR = HERE / "results"
SEEDS_DIR = RESULTS_DIR / "benchmark_seeds"


def build_data_dict(trace_seed: int = 42) -> dict:
    """Construct the data dict generate_figures expects from one canonical seed."""
    seed_path = SEEDS_DIR / f"seed_{trace_seed}.json"
    if not seed_path.exists():
        candidates = sorted(SEEDS_DIR.glob("seed_*.json"))
        raise FileNotFoundError(
            f"Trace seed file {seed_path} not found. Available seeds: "
            f"{[p.stem for p in candidates[:5]]}..."
        )

    with seed_path.open() as f:
        seed_blob = json.load(f)

    scenarios = seed_blob.get("scenarios", {})
    traces = seed_blob.get("traces", {})

    data: dict = {"results": {}}
    for scenario, modes in scenarios.items():
        data["results"][scenario] = {}
        for mode, scalar_fields in modes.items():
            # Start with the scalar metrics (ari, waste, rle, slca, carbon, ...)
            merged = dict(scalar_fields)
            # Merge per-step traces for the modes that have them
            # (static, hybrid_rl, agribrain). Ablation modes have only scalars.
            mode_traces = traces.get(scenario, {}).get(mode)
            if mode_traces:
                merged.update(mode_traces)
            data["results"][scenario][mode] = merged

    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-seed", type=int, default=42,
                        help="Which seed JSON to use for per-step trace data "
                             "(default 42 — the canonical exemplar)")
    args = parser.parse_args()

    print(f"Building data dict from seed_{args.trace_seed}.json...")
    data = build_data_dict(args.trace_seed)
    n_scenarios = len(data["results"])
    n_modes_per_scenario = {
        sc: len(modes) for sc, modes in data["results"].items()
    }
    print(f"  scenarios: {n_scenarios} -> {list(data['results'].keys())}")
    print(f"  modes/scenario: {n_modes_per_scenario}")
    print()

    print("Rendering figures (no simulation)...")
    # generate_figures imports generate_results at module-import time. The
    # latter validates DATA_CSV existence inside run_all(), not at import, so
    # importing generate_figures itself is cheap.
    import generate_figures  # noqa: E402

    generate_figures.generate_all_figures(data=data)
    print()
    print(f"All figures saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
