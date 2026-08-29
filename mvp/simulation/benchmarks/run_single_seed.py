#!/usr/bin/env python3
"""Run simulation for a single seed and save metrics to benchmark_seeds/.

Usage:
    python run_single_seed.py 42
    python run_single_seed.py 1337
    python run_single_seed.py 42 --output-dir /scratch/run_abc123/seed_42

Output JSON envelope (post 2026-05):

    {
      "_meta": {
        "source_commit": <full Git SHA>,
        "run_tag": <publication run tag>,
        "episode_scope": "final episode per scenario-mode-seed arm",
        "decision_history_scope": "earlier decisions in the same episode only"
      },
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
import math
import os
from pathlib import Path

try:
    from ..analysis.experiment_accounting import (
        PRIMARY_PUBLICATION_MODES,
        build_episode_accounting,
    )
except ImportError:
    import sys as _accounting_sys

    _ACCOUNTING_REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(_ACCOUNTING_REPO_ROOT) not in _accounting_sys.path:
        _accounting_sys.path.insert(0, str(_ACCOUNTING_REPO_ROOT))
    from mvp.simulation.analysis.experiment_accounting import (  # noqa: E402
        PRIMARY_PUBLICATION_MODES,
        build_episode_accounting,
    )

from hpc.slurm_execution_provenance import (  # noqa: E402
    CORE_SEEDS,
    build_array_execution_provenance,
)

try:
    from .. import generate_results as _generate_results
except ImportError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    import generate_results as _generate_results  # noqa: E402

run_all = _generate_results.run_all
MODES = _generate_results.MODES
TRACE_SCHEMA_VERSION = _generate_results.TRACE_SCHEMA_VERSION

try:
    from .trace_contract import TRACE_FIELDS, TRACE_MODES
except ImportError:
    from benchmarks.trace_contract import TRACE_FIELDS, TRACE_MODES


#: Modes that get per-step traces dumped. The canonical paper trio for
#: fig 2 panel (d) and fig 4 panel (a). Adding more modes is cheap
#: (each mode adds ~3 KB of JSON per seed at 4-decimal precision) but
#: deliberately limited here so the per-seed JSONs stay tractable.
#: Trace fields to dump. The 2026-05 extension covers every per-step
#: field the figure code reads from `data["results"][sc][mode]`, so a
#: completed HPC run produces a self-contained cache that
#: ``regenerate_figures_from_cache.py`` can re-render every figure
#: from without running the simulator again. Field-by-field rationale:
#:
#:  ari_trace                   fig 2 panel D, fig 4 panel A/D
#:  waste_trace                 fig 3 panel B, fig 4 panel D
#:  rho_policy_observed_trace   policy input / fig 2 panel B
#:  rho_outcome_environmental_trace latent endpoint state / fig 2 panel B
#:  action_trace                fig 3 panel C, fig 4 panel B/C/D, fig 5 panel B
#:  prob_trace                  fig 2 panel B (fallback), fig 2 panel C
#:  carbon_trace                fig 8 panel A
#:  hours                       every per-step plot (x-axis index)
#:  explicit *_policy_observed / *_outcome_environmental traces preserve the
#:  confirmatory two-world state boundary; legacy aliases are also retained.
#:  slca_component_trace        fig 3 panel D (list[dict[str,float]] -- handled below)
#:  equity_trace                fig 5 panel C
#:  reward_trace                fig 5 panel D
#:
#: Total per-seed envelope at 4-decimal precision, 3 trace modes,
#: 5 scenarios: ~120 KB. 20 seeds: ~2.4 MB total. Negligible relative
#: to the simulator's runtime.
def _to_jsonable(obj, _decimals: int | None = 4):
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
      * ``int``                      ->  preserved as an integer.
      * ``float``                    ->  rounded to ``_decimals`` decimal
                                          places.
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
    if isinstance(obj, int):
        # Discrete trace fields (especially action_trace) must remain JSON
        # integers.  Converting through float made the preserved d3286ae HPC
        # run encode valid action indices as 0.0/1.0/2.0.
        return obj
    if isinstance(obj, float):
        value = float(obj)
        if not math.isfinite(value):
            raise ValueError(f"non-finite numeric trace value: {value!r}")
        return value if _decimals is None else round(value, _decimals)
    # Strings and None pass through. Any unsupported custom object will be
    # rejected by the final strict json.dumps call.
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
        roundtrip = _json.loads(_json.dumps(out, allow_nan=False))
        assert roundtrip == expected, f"{label} json round-trip diverged"
    try:
        _to_jsonable([_math.nan, 0.5])
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite trace values must be rejected")


def _enforce_strict_trace_completion(trace_failures: list[str], seed: int) -> None:
    """Fail the seed task after preserving its diagnostic envelope."""
    if trace_failures and os.environ.get("STRICT_VALIDATION", "0") == "1":
        raise RuntimeError(
            "Strict publication seed run retained trace serialization failures; "
            f"rerun seed {seed} after fixing them: {trace_failures!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("seed", type=int, help="Seed for this run")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Explicit run-scoped directory for seed_<seed>.json. The option "
            "is mandatory so an exploratory invocation cannot overwrite "
            "canonical publication evidence."
        ),
    )
    args = parser.parse_args()

    seed = args.seed
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    execution_provenance = None
    if os.environ.get("STRICT_VALIDATION", "0") == "1":
        if seed not in CORE_SEEDS:
            raise RuntimeError(
                f"strict publication seed {seed} is outside the locked 20-seed panel"
            )
        logical_task_index = CORE_SEEDS.index(seed)
        execution_provenance = build_array_execution_provenance(
            stage="core_seed_array",
            logical_task_index=logical_task_index,
        )
        if execution_provenance["slurm_array_task_id"] != logical_task_index:
            raise RuntimeError(
                "SLURM_ARRAY_TASK_ID does not map to the requested canonical seed"
            )

    # ``run_all`` also writes representative protocol/context artifacts through
    # its module-level RESULTS_DIR.  Parallel seed jobs previously raced on
    # those shared filenames even though their final seed envelopes were
    # isolated.  Redirect auxiliary artifacts to a seed-specific directory;
    # the publication figures use the traces embedded in seed_<seed>.json.
    auxiliary_dir = out_dir / "auxiliary" / f"seed_{seed}"
    auxiliary_dir.mkdir(parents=True, exist_ok=True)
    _generate_results.RESULTS_DIR = auxiliary_dir

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
    episode_budget_by_mode = {
        mode: int(_generate_results._MULTI_EPISODE_MODES.get(mode, 1))
        for mode in modes_run
    }
    primary_modes_run = [
        mode for mode in PRIMARY_PUBLICATION_MODES if mode in modes_run
    ]
    episode_accounting = build_episode_accounting(
        scenarios=scenarios_run,
        configured_modes=modes_run,
        episode_budget_by_mode=episode_budget_by_mode,
        n_seeds=1,
        primary_modes=primary_modes_run,
    )
    episode_accounting["complete_primary_mode_panel"] = (
        tuple(primary_modes_run) == PRIMARY_PUBLICATION_MODES
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
                # Exploratory ratio retained with its literal units.  It is
                # computed per seed before aggregation so its BCa interval
                # preserves the ARI/carbon covariance; it is not reconstructed
                # from two marginal confidence intervals.
                "carbon_efficiency_ari_per_kgco2e_proxy": float(
                    ep["carbon_efficiency_ari_per_kgco2e_proxy"]
                ),
            }
            # Retained as a descriptive, hardware-dependent diagnostic only.
            # Canonical publication CSVs deliberately exclude latency from
            # inferential seed summaries; Green-AI reporting declares its
            # measured timer boundary separately.
            metrics[sc][mode]["mean_decision_latency_ms"] = float(
                ep.get("mean_decision_latency_ms", 0.0)
            )
            metrics[sc][mode]["p95_decision_latency_ms"] = float(
                ep.get("p95_decision_latency_ms", 0.0)
            )
            metrics[sc][mode]["latency_penalty_usd"] = float(
                ep.get("latency_penalty_usd", 0.0)
            )
            metrics[sc][mode]["mean_decision_latency_ms_descriptive_only"] = (
                ep.get("mean_decision_latency_ms_descriptive_only") is True
            )
            metrics[sc][mode]["latency_penalty_usd_descriptive_only"] = (
                ep.get("latency_penalty_usd_descriptive_only") is True
            )
            metrics[sc][mode]["constraint_violation_rate"] = float(
                ep.get("constraint_violation_rate", 0.0)
            )
            metrics[sc][mode]["compliance_violation_rate"] = float(
                ep.get("compliance_violation_rate", 0.0)
            )
            metrics[sc][mode]["message_count"] = int(
                ep.get("message_count", 0)
            )
            # Also capture the new §4.7 diagnostic metrics when present so
            # the aggregator has the raw per-seed numbers for bootstrap CIs
            # without re-running the simulator. Empty/None when the mode
            # does not produce the metric (e.g. static has no honor rate).
            for extra in (
                "operational_violation_rate", "regulatory_violation_rate",
                "operating_envelope_violation_rate",
                "context_active_steps", "context_active_fraction",
                # 2026-05 apples-to-apples cross-mode dispatch
                # counters. Always-equal-to-n_steps for context-
                # enabled modes (288 on canonical 72-hr episodes) and
                # zero for static / hybrid_rl / no_context. Used as
                # the denominator for context_dispatch_influence_rate.
                "context_dispatch_attempt_steps",
                "context_dispatch_attempt_fraction",
                "context_honored_steps", "context_honor_rate",
                # Fig. 9 context-influence rate: percentage of
                # context-active steps where paired live and
                # context-ablated calls, using the same saved pre-selection
                # RNG state, selected different actions. Stochastic calls
                # consume the same categorical variate even if the live
                # probability-gap override discards its sampled action. Honor rate
                # is retained above
                # as a supplementary-methods companion.
                "context_influenced_steps", "context_influence_rate",
                # 2026-05 cross-mode-comparable influence rate
                # (numerator unchanged, denominator switched to
                # context_dispatch_attempt_steps). Resolves the
                # heatwave activation-regime confound between
                # agribrain (72-step retrieval-gated) and mcp_only
                # (~168-step retrieval-free) by sharing the
                # 288-step dispatch denominator.
                "context_dispatch_influence_rate",
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
                # Fail-closed MCP execution evidence. Canonical strict runs
                # abort on any JSON-RPC error, real tool isError response, or
                # recorder truncation; the retained zero counts make that
                # execution invariant inspectable per final episode.
                "protocol_interaction_count",
                "protocol_tools_call_count",
                "protocol_prompts_get_count",
                "protocol_jsonrpc_error_count",
                "protocol_tool_iserror_count",
                "protocol_real_tool_iserror_count",
                "protocol_error_count",
                "protocol_dropped_interaction_count",
                "dispatcher_tool_failure_count",
                "context_execution_error_count",
                "mcp_calls_per_episode", "pirag_queries_per_episode",
                "fault_injection_scheduled_opportunity_steps",
                "fault_injection_trigger_steps",
                "fault_injected_tool_result_count",
                "trace_schema_version", "benchmark_seed", "episode_index",
                "learning_enabled", "episode_phase",
                "environment_stream_id", "policy_stream_id",
                "stochastic_stream_id", "context_prior_sha256",
                "policy_theta_initial_sha256", "spoilage_estimator",
                "latent_spoilage_model",
                "latent_environment_sha256",
                "observed_policy_input_sha256", "demand_observation_sha256",
                "demand_forecast_method", "supply_forecast_method",
                "dispatch_opportunity_count",
                "dispatch_cadence_hours", "endpoint_unit", "waste_definition",
                "carbon_definition", "scenario_onset_offset_hours",
                "effective_k_ref", "effective_Ea_R",
            ):
                if extra in ep:
                    metrics[sc][mode][extra] = ep[extra] if not isinstance(
                        ep[extra], (list, tuple)
                    ) else list(ep[extra])
            if "footprint" in ep:
                metrics[sc][mode]["footprint"] = _to_jsonable(
                    ep["footprint"], _decimals=12,
                )
                footprint = ep["footprint"]
                # Hardware-dependent descriptive estimates for the explicitly
                # bounded action-selection timer.  Promote scalar copies for
                # cross-seed summaries while preserving the complete nested
                # footprint record for equation-level audit.
                metrics[sc][mode][
                    "decision_path_compute_energy_estimate_j"
                ] = float(footprint["cumulative_energy_J"])
                metrics[sc][mode][
                    "decision_path_compute_water_estimate_l"
                ] = float(footprint["cumulative_water_L"])
                metrics[sc][mode][
                    "decision_path_elapsed_seconds"
                ] = float(footprint["cumulative_elapsed_seconds"])
                metrics[sc][mode][
                    "decision_step_count_energy_proxy_j"
                ] = float(footprint["cumulative_energy_per_step_proxy_J"])
                metrics[sc][mode][
                    "decision_step_count_water_proxy_l"
                ] = float(footprint["cumulative_water_per_step_proxy_L"])
            for context_object in (
                "context_active_per_recommendation",
                "context_ignored_per_recommendation",
                "context_threshold_counters",
            ):
                if context_object in ep:
                    metrics[sc][mode][context_object] = _to_jsonable(
                        ep[context_object], _decimals=12,
                    )
            # Retain compact, hash-stamped learner provenance for the final
            # episode. These nested records are not inferential endpoints, but
            # they prove which adaptive components actually updated and keep
            # per-role policy learning from being hidden behind one scalar.
            for learner_key in (
                "context_summary",
                "evaluator_summary",
                "learner_summary",
                "theta_learner_summary",
                "reward_shaping_learner_summary",
                "learner_freeze_summary",
            ):
                if learner_key in ep:
                    metrics[sc][mode][learner_key] = _to_jsonable(
                        ep[learner_key], _decimals=None,
                    )

    # ---- Partial-save guard: write metrics-only first ----
    # The 2026-05 HPC pipeline lost ~50 hours of compute when a
    # bug in the trace-dump path raised an exception, the seed
    # task exited non-zero, the aggregator's afterok dependency
    # failed, and 20 seeds * 2.5 h of valid metrics evaporated
    # because they were never written to disk. Defensive change:
    # write the metrics block FIRST so the canonical published
    # numbers are durable for diagnosis/recovery regardless of what happens in
    # the subsequent trace-dump path. Canonical publication validation still
    # rejects a metrics-only checkpoint or any trace failure; it is never
    # silently promoted into the 20-seed evidence panel.
    out_file = out_dir / f"seed_{seed}.json"
    partial_file = out_dir / f"seed_{seed}.partial.json"
    metrics_only_payload = {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "seed": int(seed),
        "scenarios": metrics,
        "traces": {},
        "_note": (
            "Metrics-only checkpoint. The trace-dump pass runs next. "
            "If it succeeds, this file is overwritten with the full "
            "envelope. If it crashes, the metrics block here survives "
            "and the aggregator can still produce benchmark_summary.json "
            "and benchmark_significance.json after the trace failure is fixed "
            "and this seed is rerun. Canonical validators reject this "
            "checkpoint as an incomplete publication envelope."
        ),
    }
    partial_file.write_text(
        json.dumps(metrics_only_payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Saved metrics-only checkpoint: {partial_file}")

    # ---- Per-step trace dump (recoverable checkpoint, strict publication gate) ----
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
    # the figures may fall back during exploratory work. Strict publication
    # runs fail after writing the diagnostic envelope, and the downstream raw
    # gate rejects `_trace_failures` in every environment.
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
        "_meta": {
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "source_commit": os.environ.get("AGRIBRAIN_GIT_COMMIT", "").strip(),
            "run_tag": os.environ.get("RUN_TAG", "").strip(),
            "episode_scope": "final episode per scenario-mode-seed arm",
            "decision_history_scope": "earlier decisions in the same episode only",
            "episode_accounting": episode_accounting,
            "execution_provenance": execution_provenance,
            "state_design": (
                "routing uses *_policy_observed; scored endpoints use "
                "*_outcome_environmental"
            ),
        },
        "trace_schema_version": TRACE_SCHEMA_VERSION,
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
    serialized = json.dumps(full_payload, indent=2, allow_nan=False)
    if trace_failures and os.environ.get("STRICT_VALIDATION", "0") == "1":
        # Preserve the diagnostic envelope only under the explicitly partial
        # name. Never replace a valid final envelope with a failed strict run.
        partial_file.write_text(serialized, encoding="utf-8")
        _enforce_strict_trace_completion(trace_failures, seed)

    final_temp = out_dir / f".seed_{seed}.full.tmp"
    final_temp.write_text(serialized, encoding="utf-8")
    os.replace(final_temp, out_file)
    partial_file.unlink(missing_ok=True)
    print(f"Atomically promoted full envelope: {out_file}")


if __name__ == "__main__":
    main()
