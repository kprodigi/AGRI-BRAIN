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


def _to_jsonable(obj, _decimals: int = 4):
    """Recursively convert a per-step trace value into a JSON-friendly form.

    Replaces the 2026-05 dispatch-by-first-element scheme that lost
    ~50 hours of HPC compute to three "I didn't anticipate that
    shape" bugs in a row (prob_trace as nested list, mixed-type
    dicts in slca_component_trace, list[np.float64] from numpy ops
    inside Python list appends).

    The structural problem with the dispatch approach: it inferred
    the whole structure's shape from a single element's type, then
    applied a uniform transformation. That fails whenever:

      (i)   the first element doesn't represent the rest
            (heterogeneous dicts, mixed numeric/string values),
      (ii)  the type-introspection check is ambiguous (numpy scalars
            have a ``.tolist()`` method, the same attribute used to
            detect numpy arrays), or
      (iii) a future field has a shape never seen before.

    This visitor doesn't dispatch -- it descends. Every node is
    handled by its actual local type:

      * Anything with ``.tolist()``  ->  recurse into ``.tolist()``.
        Covers numpy scalars (``np.float64.tolist()`` -> Python
        scalar), numpy arrays of any rank
        (``np.ndarray.tolist()`` -> nested Python list), and any
        future tensor type that follows the same convention
        (torch tensors, jax arrays, etc.).
      * ``dict``                     ->  recurse on each value,
                                          preserve string keys verbatim.
      * ``list`` / ``tuple``         ->  recurse on each element,
                                          return as a list.
      * ``bool``                     ->  preserved as True / False
                                          (NOT folded into 0 / 1 even
                                          though ``bool`` is an
                                          ``int`` subclass).
      * Other ``int`` / ``float``    ->  rounded to ``_decimals``
                                          decimal places.
      * Anything else (str, None,
        custom objects, NaN/Inf, ...) -> preserved verbatim.

    No dispatch ambiguity, no shape enumeration, no first-element
    dependence. New TRACE_FIELDS shapes that nobody has seen yet
    are handled correctly by construction as long as their leaves
    round-trip through ``json.dumps``.

    Args:
        obj: The value to convert. May be a Python scalar / list /
            dict, a numpy scalar / array, or any nested combination.
        _decimals: Decimal places to round numeric leaves to.
            Default 4 keeps the per-seed JSON in the ~120 KB range
            across the full 16-field TRACE_FIELDS set; below the
            per-step measurement noise floor.

    Returns:
        A value composed solely of JSON-native Python types
        (None, bool, int, float, str, list, dict). Pass directly
        to ``json.dumps`` -- no custom encoder needed.
    """
    # numpy / tensor types: ``.tolist()`` is a uniform interface
    # that converts to native Python -- scalars become Python
    # scalars, ndarrays of any rank become nested Python lists.
    # Recurse into the converted form so the rules below apply.
    if hasattr(obj, "tolist"):
        return _to_jsonable(obj.tolist(), _decimals)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v, _decimals) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v, _decimals) for v in obj]
    # bool intentionally before int -- bool IS an int subclass in
    # Python, but a Boolean field should stay True / False rather
    # than collapsing to 1 / 0 after a round-trip.
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        return round(float(obj), _decimals)
    # str, None, NaN floats (covered above), and anything else
    # the caller hands us. JSON refuses NaN by default, but the
    # simulator does not produce NaN in trace fields and the
    # ``round(float(...))`` branch above would have caught it
    # anyway. Custom objects round-trip iff they implement
    # ``__str__`` or are JSON-encodable -- we don't try to
    # second-guess.
    return obj


# Public alias kept for back-compat with any code outside this module
# that imported ``_serialise_trace`` directly. The contract is now
# ``recursively make this JSON-jsonable, rounding numeric leaves``,
# which is strictly more permissive than the old shape-dispatch.
def _serialise_trace(arr):
    return _to_jsonable(arr, _decimals=4)


def _self_test_trace_dispatch():
    """Fail-fast self-test for the trace serialiser.

    Called once at the top of ``main()`` so any regression crashes
    in milliseconds instead of after 2.5 h of simulator runtime.
    Exercises every shape category that has ever appeared in the
    simulator's TRACE_FIELDS, plus a handful of pathological
    nestings (dict-of-dict, list-of-dict-with-array-value, etc.)
    that the visitor must handle uniformly even though no real
    trace currently uses them. The point is structural robustness:
    the assertion is that the visitor's behaviour is determined
    by element TYPE, not by element POSITION (the failure mode
    of the pre-2026-05 dispatcher).
    """
    import json as _json
    import math as _math
    import numpy as _np
    cases = [
        # Flat numeric (most TRACE_FIELDS).
        ("list[float]",       [0.123456, 0.789012],                  [0.1235, 0.7890]),
        ("list[int]",         [0, 1, 2],                              [0, 1, 2]),
        # Dicts with mixed leaf types (slca_component_trace).
        ("list[dict_mixed]",  [{"C": 0.7, "action_family": "cold_chain"}],
                              [{"C": 0.7, "action_family": "cold_chain"}]),
        # Nested numeric (prob_trace).
        ("list[list_float]",  [[0.3, 0.5, 0.2]],                     [[0.3, 0.5, 0.2]]),
        # Numpy at every depth (the bug class that lost compute twice).
        ("ndarray_1d",         _np.array([0.5, 1.5]),                  [0.5, 1.5]),
        ("ndarray_2d",         _np.array([[0.3, 0.5], [0.4, 0.4]]),    [[0.3, 0.5], [0.4, 0.4]]),
        ("list[np_scalar]",   [_np.float64(0.5), _np.float64(1.5)],   [0.5, 1.5]),
        ("list[np_array]",    [_np.array([0.3, 0.5])],                 [[0.3, 0.5]]),
        # bool preservation (not collapsed to 0/1).
        ("list[dict_bool]",   [{"flag": True, "x": 0.123456}],         [{"flag": True, "x": 0.1235}]),
        # Empty.
        ("empty",              [],                                     []),
        # Pathological-but-valid combinations the visitor must handle
        # uniformly even though no real trace uses them today: this
        # is the structural-robustness assertion (vs the old shape-
        # dispatch).
        ("dict_of_dict",       {"outer": {"C": _np.float64(0.7), "tag": "x"}},
                              {"outer": {"C": 0.7, "tag": "x"}}),
        ("list_of_dict_with_array_value",
                               [{"v": _np.array([0.1, 0.2])}],
                              [{"v": [0.1, 0.2]}]),
    ]
    for label, val, expected in cases:
        out = _to_jsonable(val)
        assert out == expected, f"{label}: expected {expected!r}, got {out!r}"
        # Round-trip JSON-encode + decode confirms every leaf is
        # JSON-native. NaN / Inf handling: json.dumps would raise
        # ValueError on those by default; the simulator does not
        # emit NaN in TRACE_FIELDS, but this round-trip is the
        # canonical proof that the visitor's output is strictly
        # JSON-clean.
        roundtrip = _json.loads(_json.dumps(out))
        assert roundtrip == expected, f"{label} json round-trip diverged"
    # NaN does NOT round-trip through json.dumps by default -- this
    # documents that boundary so a future caller is not surprised.
    nan_out = _to_jsonable([_math.nan, 0.5])
    assert _math.isnan(nan_out[0]) and nan_out[1] == 0.5


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

    # Fail-fast self-test for the trace-dump dispatch. Catches
    # regressions in milliseconds instead of after 2.5 h of
    # simulator runtime per seed task. The 2026-05 HPC runs lost
    # ~50 hours of compute to bugs that this guard would have
    # caught at job start.
    print("Self-testing trace-dump dispatch...")
    _self_test_trace_dispatch()
    print("OK.")

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

    # ---- Partial-save guard: write metrics-only first ----
    # The 2026-05 HPC pipeline lost ~50 hours of compute when a
    # bug in the trace-dump path raised an exception, the seed
    # task exited non-zero, the aggregator's afterok dependency
    # failed, and 20 seeds * 2.5 h of valid metrics evaporated
    # because they were never written to disk. Defensive change:
    # write the metrics block FIRST so the canonical published
    # numbers (which the aggregator's bootstrap CIs are computed
    # from) are durable regardless of what happens in the
    # subsequent trace-dump path.
    out_file = out_dir / f"seed_{seed}.json"
    metrics_only_payload = {
        "seed": int(seed),
        "scenarios": metrics,
        "traces": {},
        "_note": (
            "Metrics-only checkpoint. The trace-dump pass runs next. "
            "If it succeeds, this file is overwritten with the full "
            "envelope. If it crashes, the metrics block here survives "
            "and the aggregator can still produce benchmark_summary.json "
            "and benchmark_significance.json from the canonical 20-seed "
            "scalars."
        ),
    }
    out_file.write_text(json.dumps(metrics_only_payload, indent=2))
    print(f"Saved metrics-only checkpoint: {out_file}")

    # ---- Per-step trace dump (best-effort, separate from metrics) ----
    # Per-step traces drive figure code's line plots and any
    # per-step uncertainty band. The serialiser is now a recursive
    # visitor (``_to_jsonable``) that handles arbitrary nesting and
    # any future trace shape, so the failure modes the previous
    # dispatch suffered (prob_trace as nested list,
    # slca_component_trace's mixed-type dict, list[np.float64])
    # cannot recur. Even so, the dump runs inside a try/except so
    # an unanticipated future shape can never destroy the metrics
    # checkpoint above. A serialisation failure logs the offending
    # (scenario, mode, field) and continues to the next field --
    # the figures fall back to their single-seed-line code paths
    # for any field the cache lacks.
    traces: dict = {}
    trace_failures: list[str] = []
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
                try:
                    cell[field] = _to_jsonable(ep[field])
                except Exception as exc:  # noqa: BLE001 -- log + continue
                    trace_failures.append(
                        f"{sc}/{mode}/{field}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    print(
                        f"WARN: trace serialisation failed for "
                        f"{sc}/{mode}/{field} -- {type(exc).__name__}: {exc}"
                    )
            if cell:
                sc_traces[mode] = cell
        if sc_traces:
            traces[sc] = sc_traces

    full_payload = {
        "seed": int(seed),
        "scenarios": metrics,
        "traces": traces,
    }
    if trace_failures:
        full_payload["_trace_failures"] = trace_failures
        print(
            f"WARN: {len(trace_failures)} trace fields failed to "
            f"serialise; metrics block is intact."
        )
    out_file.write_text(json.dumps(full_payload, indent=2))
    print(f"Saved full envelope: {out_file}")


if __name__ == "__main__":
    main()
