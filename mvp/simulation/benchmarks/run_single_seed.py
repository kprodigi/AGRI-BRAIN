#!/usr/bin/env python3
"""Run simulation for a single seed and save metrics to benchmark_seeds/.

Usage:
    python run_single_seed.py 42
    python run_single_seed.py 1337
    python run_single_seed.py 42 --output-dir /scratch/run_abc123/seed_42

Output JSON envelope (post 2026-05):

    {
      "seed": <int>,
      "scenarios": {<sc>: {<mode>: {<scalar metric>: float, ...}}},
      "traces":    {<sc>: {<mode>: {<trace name>: [floats]}}}
    }

The "scenarios" block carries the scalar metrics that
``aggregate_seeds.py`` bootstrap-CIs over the seed dimension. The
"traces" block carries per-step arrays (currently ``ari_trace`` only)
for the ``static``, ``hybrid_rl``, ``agribrain`` modes -- the canonical
paper trio plotted in fig 2 panel (d) -- so the figure can render
seed-stacked CI ribbons without re-running the simulator. Other modes
and other trace fields can be added by extending TRACE_MODES /
TRACE_FIELDS below.

Backward compatibility: the previous JSON format dumped the
"scenarios" block at the root with no envelope. The
``_load_per_seed_summary`` loader in ``generate_figures.py`` already
prefers ``obj.get("scenarios")`` over the legacy root-as-scenarios
fallback, so old benchmark snapshots aggregate the same way and new
ones expose traces additively.
"""
import argparse
import json
from pathlib import Path

try:
    from ..generate_results import run_all, MODES
except ImportError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from generate_results import run_all, MODES  # noqa: E402


#: Modes that get per-step traces dumped. The canonical paper trio for
#: fig 2 panel (d) and fig 4 panel (a). Adding more modes is cheap
#: (each mode adds ~3 KB of JSON per seed at 4-decimal precision) but
#: deliberately limited here so the per-seed JSONs stay tractable.
TRACE_MODES = ("static", "hybrid_rl", "agribrain")

#: Trace fields to dump. The 2026-05 extension covers every per-step
#: field the figure code reads from `data["results"][sc][mode]`, so a
#: completed HPC run produces a self-contained cache that
#: ``regenerate_figures_from_cache.py`` can re-render every figure
#: from without running the simulator again. Field-by-field rationale:
#:
#:  ari_trace                   fig 2 panel D, fig 4 panel A/D
#:  waste_trace                 fig 3 panel B, fig 4 panel D
#:  rho_trace                   fig 2 panel B (fallback), fig 3 panel C
#:  action_trace                fig 3 panel C, fig 4 panel B/C/D, fig 5 panel B
#:  prob_trace                  fig 2 panel B (fallback), fig 2 panel C
#:  carbon_trace                fig 8 panel A
#:  hours                       every per-step plot (x-axis index)
#:  batch_effective_rho_trace   fig 2 panel B
#:  effective_rho_trace         fig 2 panel B (fallback)
#:  temp_trace                  fig 2 panel A
#:  rh_trace                    fig 2 panel A
#:  inventory_trace             fig 3 panel A
#:  demand_trace                fig 3 panel A, fig 5 panel A
#:  slca_component_trace        fig 3 panel D (list[dict[str,float]] -- handled below)
#:  equity_trace                fig 5 panel C
#:  reward_trace                fig 5 panel D
#:
#: Total per-seed envelope at 4-decimal precision, 3 trace modes,
#: 5 scenarios: ~120 KB. 20 seeds: ~2.4 MB total. Negligible relative
#: to the simulator's runtime.
TRACE_FIELDS = (
    "ari_trace",
    "waste_trace",
    "rho_trace",
    "action_trace",
    "prob_trace",
    "carbon_trace",
    "hours",
    "batch_effective_rho_trace",
    "effective_rho_trace",
    "temp_trace",
    "rh_trace",
    "inventory_trace",
    "demand_trace",
    "slca_component_trace",
    "equity_trace",
    "reward_trace",
)


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

    # Drive the per-seed metric dump off the data dict's actual keys
    # rather than the imported SCENARIOS / MODES module-level
    # constants. Two reasons:
    #   1. Robustness against in-process patching: callers that
    #      monkeypatch gr.SCENARIOS / gr.MODES (e.g. limit-to-one-
    #      scenario probes) only mutate the *generate_results*
    #      module's bindings; this module imported the names at
    #      import time and would otherwise iterate the un-patched
    #      original lists, then crash on `data["results"][sc]`
    #      when sc isn't present.
    #   2. Future-proofing: when a new ablation mode is added to
    #      generate_results.MODES the script picks it up automatically
    #      without a parallel edit here, which is the original spirit
    #      of "single source of truth in generate_results".
    results = data["results"]
    scenarios_run = list(results.keys())
    modes_seen: set[str] = set()
    for sc in scenarios_run:
        modes_seen.update((results.get(sc) or {}).keys())
    # Preserve canonical ordering when the run touched the full set.
    modes_run = [m for m in MODES if m in modes_seen] + sorted(
        modes_seen.difference(MODES)
    )
    metrics = {}
    for sc in scenarios_run:
        metrics[sc] = {}
        for mode in modes_run:
            ep = (results.get(sc) or {}).get(mode)
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

    # Per-step trace dump for seed-CI ribbon panels (fig 2 panel d
    # ARI ribbon at present; extend by editing TRACE_MODES /
    # TRACE_FIELDS at the top of this file). Rounded to 4 decimal
    # places; that is well below the per-step measurement noise
    # (theta-perturbation sigma 0.15, source 7) and keeps the per-
    # seed JSON in the ~50 KB range.
    # Iterate the actual scenarios run (see the metrics-loop comment
    # above for the rationale: robustness against in-process
    # SCENARIOS patching).
    traces: dict = {}
    for sc in scenarios_run:
        sc_traces: dict = {}
        for mode in TRACE_MODES:
            ep = (results.get(sc) or {}).get(mode)
            if ep is None:
                continue
            cell: dict = {}
            for field in TRACE_FIELDS:
                if field not in ep:
                    continue
                arr = ep[field]
                if hasattr(arr, "tolist"):
                    arr = arr.tolist()
                if not arr:
                    cell[field] = []
                    continue
                # Two trace shapes are supported:
                #   (a) list[float]            -- the common case
                #   (b) list[dict[str,float]]  -- slca_component_trace
                # Round every numeric leaf to 4 decimals; non-numeric
                # leaves (e.g. the action_trace's ints) are preserved
                # as ints since round(float(int), 4) == int.
                first = arr[0]
                if isinstance(first, dict):
                    cell[field] = [
                        {k: round(float(v), 4) for k, v in entry.items()}
                        for entry in arr
                    ]
                else:
                    cell[field] = [round(float(x), 4) for x in arr]
            if cell:
                sc_traces[mode] = cell
        if sc_traces:
            traces[sc] = sc_traces

    out_file = out_dir / f"seed_{seed}.json"
    payload = {
        "seed": int(seed),
        "scenarios": metrics,
        "traces": traces,
    }
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
