#!/usr/bin/env python3
"""Run simulation for a single seed and save metrics to benchmark_seeds/.

Usage:
    python run_single_seed.py 42
    python run_single_seed.py 1337
    python run_single_seed.py 42 --output-dir /scratch/run_abc123/seed_42
"""
import argparse
import json
from pathlib import Path

try:
    from ..generate_results import run_all, SCENARIOS, MODES
except ImportError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from generate_results import run_all, SCENARIOS, MODES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("seed", type=int, help="Seed for this run")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write seed_<seed>.json into. Defaults to "
             "mvp/simulation/results/benchmark_seeds/ when omitted.",
    )
    args = parser.parse_args()

    seed = args.seed
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        out_dir = Path(__file__).resolve().parent.parent / "results" / "benchmark_seeds"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running full simulation with seed={seed}...")
    data = run_all(seed=seed)

    # Drive the per-seed metric dump off the canonical MODES list in
    # generate_results so cold_start and the three sensitivity-perturbation
    # modes land in seed_<N>.json alongside the eight legacy modes. Keeping
    # a second hardcoded list here would silently drop the new ablation
    # data from every downstream aggregator stage.
    metrics = {}
    for sc in SCENARIOS:
        metrics[sc] = {}
        for mode in MODES:
            ep = data["results"][sc].get(mode)
            if ep is None:
                continue
            metrics[sc][mode] = {
                "ari": float(ep["ari"]),
                "waste": float(ep["waste"]),
                # Single canonical RLE: EU-hierarchy + severity-weighted.
                "rle": float(ep["rle"]),
                "slca": float(ep["slca"]),
                "carbon": float(ep["carbon"]),
                "equity": float(ep["equity"]),
            }
            # Required for validate_results.py (checks DecisionLatencyMs and
            # ConstraintViolationRate bounds) and for table1/table2 CSV
            # rewrites that preserve the legacy column set.
            metrics[sc][mode]["mean_decision_latency_ms"] = float(
                ep.get("mean_decision_latency_ms", 0.0)
            )
            metrics[sc][mode]["constraint_violation_rate"] = float(
                ep.get("constraint_violation_rate", 0.0)
            )
            metrics[sc][mode]["compliance_violation_rate"] = float(
                ep.get("compliance_violation_rate", 0.0)
            )
            # Also capture the new §4.7 diagnostic metrics when present so
            # the aggregator has the raw per-seed numbers for bootstrap CIs
            # without re-running the simulator. Empty/None when the mode
            # does not produce the metric (e.g. static has no honor rate).
            for extra in (
                "operational_violation_rate", "regulatory_violation_rate",
                "context_active_steps", "context_active_fraction",
                "context_honored_steps", "context_honor_rate",
                # 2026-05 fig9-c headline: context-influence rate
                # (% of context-active steps where the modifier
                # changed the chosen action). Honor rate is retained
                # above as a supplementary-methods companion.
                "context_influenced_steps", "context_influence_rate",
                # Outcome-side violation disposition: cross-method-honest
                # policy-quality score on the env-driven violation event
                # set. See resilience.compute_violation_disposition for
                # the canonical definition. The three rates sum to 1.0
                # whenever violation_event_count > 0 and are all 0.0
                # otherwise (no events to score disposition on).
                "downstream_violation_rate",
                "redistribute_violation_rate",
                "contained_violation_rate",
                "violation_event_count",
            ):
                if extra in ep:
                    metrics[sc][mode][extra] = ep[extra] if not isinstance(
                        ep[extra], (list, tuple)
                    ) else list(ep[extra])

    out_file = out_dir / f"seed_{seed}.json"
    out_file.write_text(json.dumps(metrics, indent=2))
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
