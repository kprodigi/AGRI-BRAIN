#!/usr/bin/env python3
"""Replay every figure from on-disk artifacts using the same clean source.

This is a deterministic same-commit verification/rendering step, not a path
for applying changed figure code to old simulation artifacts. The executing
checkout must be clean outside the run-output tree and its HEAD must equal the
simulation source commit. Direct execution of ``generate_figures.py`` is
retired and fails closed.

Two on-disk caches drive every panel:

  1. ``benchmark_summary.json`` -- 20-seed bootstrap means / stds /
     CI bounds per (scenario, mode, metric). Aggregate panels read these
     directly.

  2. ``benchmark_seeds/<RUN_TAG>/seed_*.json`` -- per-seed envelope
     ``{seed, scenarios, traces}``. Per-step ``traces[sc][mode][field]``
     arrays drive every line plot, distribution shift, and
     window-aggregated panel:

       fig 2 panels A / B / C / D
       fig 3 panels A / B / C / D
       fig 4 panels A / B / C / D
       fig 5 panels A / B / C / D
       fig 8 panel A

     This script picks one canonical seed (default 42, falls back to
     the smallest seed on disk) as the "single-seed representative"
     for ``ab[X_trace]`` reads inside the figure code; the figure
     code's own ``_load_per_seed_traces`` helper still consumes the
     full multi-seed envelope where it needs cross-seed CIs (fig 2
     panel D mean line, fig 4 panels B/C/D bars).

The HPC seed runner (``run_single_seed.py``) was extended in 2026-05
to dump every per-step field the figure code reads, so a completed
HPC run produces a self-contained cache. If the cache is partial
(an older run that only dumped ari_trace, say), this script will
emit "FAIL: fig N: KeyError <field>" for the figures whose required
trace is missing -- the rest still re-render.

Total runtime: a few seconds per figure (read JSON + matplotlib),
~30-60 s for all ten figures.

Usage::

    python mvp/simulation/regenerate_figures_from_cache.py

An explicit ``FIGURE_OUTPUT_DIR`` is mandatory. A normal post-archive render
must point it at a separate derived-output directory. Only the canonical HPC
publisher may point it at ``mvp/simulation/results``; that path also requires
``AGRIBRAIN_PUBLICATION_RENDER=1``. Identity validation is unconditional::

    STRICT_VALIDATION=1 AGRIBRAIN_GIT_COMMIT=<full-sha> RUN_TAG=<run-tag> \
      BENCHMARK_SEEDS=<locked-seed-list> \
      FIGURE_SEED_ROOT=<validated-seed-directory> \
      FIGURE_OUTPUT_DIR=<separate-derived-directory> \
      python mvp/simulation/regenerate_figures_from_cache.py
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

_SIM_DIR = Path(__file__).resolve().parent
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))
_REPO_ROOT = _SIM_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hpc.validate_source_checkout import (  # noqa: E402
    validation_errors as _source_validation_errors,
)
from mvp.simulation.analysis.publication_figure_style import (  # noqa: E402
    publication_style_contract,
)

_RESULTS_DIR = _SIM_DIR / "results"
_SEEDS_DIR = Path(
    os.environ.get(
        "FIGURE_SEED_ROOT", str(_RESULTS_DIR / "benchmark_seeds"),
    )
)

#: Seed treated as the "single-seed representative" for figure code
#: that reads ``ab["X_trace"]`` directly (most line plots).
#: Falls back to the smallest available seed if 42 isn't on disk.
_PREFERRED_SINGLE_SEED = 42

# These are the complete set of non-seed files read by generate_figures.py.
# Keeping the inventory explicit makes a source-code change that adds another
# aggregate input fail the provenance review until this contract is updated.
_AGGREGATE_FIGURE_INPUTS = (
    "benchmark_summary.json",
    "benchmark_significance.json",
    "channel_attribution_aggregate.json",
    "stress_passfail.csv",
)


def _require_renderer_source_identity() -> None:
    """Require a clean source checkout authorized for this derivation.

    Run artifacts are the sole status exception because aggregation and figure
    promotion necessarily populate that output tree. Their literal input bytes
    are independently snapshotted below and the final manifest binds them.
    """

    validation_environment = dict(os.environ)
    if any(
        os.environ.get(name, "").strip()
        for name in (
            "AGRIBRAIN_RECOVERY_RECEIPT",
            "AGRIBRAIN_SIMULATION_COMMIT",
            "AGRIBRAIN_PUBLICATION_CODE_COMMIT",
        )
    ):
        from mvp.simulation.analysis.recovery_provenance import (
            recovery_context_from_environment,
        )

        recovery = recovery_context_from_environment(
            results_dir=_RESULTS_DIR,
            repo_root=_REPO_ROOT,
        )
        if recovery is None:
            raise RuntimeError("incomplete publication-recovery renderer identity")
        # Git checkout validation describes the code executing the renderer;
        # figure provenance below continues to stamp the simulation commit on
        # its raw inputs and the artifact manifest records both identities.
        validation_environment["AGRIBRAIN_GIT_COMMIT"] = str(
            recovery["publication_code_commit"]
        )
    errors = _source_validation_errors(
        environ=validation_environment,
        repo_root=_REPO_ROOT,
        allow_run_artifacts=True,
    )
    if errors:
        raise RuntimeError(
            "renderer source identity is not the clean simulation commit: "
            + "; ".join(errors)
        )


def _figure_source_identity(
    environ: dict[str, str] | None = None,
) -> dict[str, object]:
    """Separate immutable simulation inputs from the executing renderer."""

    env = os.environ if environ is None else environ
    simulation_commit = (
        str(env.get("AGRIBRAIN_SIMULATION_COMMIT", "")).strip()
        or str(env.get("AGRIBRAIN_GIT_COMMIT", "")).strip()
    )
    raw_input_commit = str(env.get("AGRIBRAIN_GIT_COMMIT", "")).strip()
    renderer_commit = (
        str(env.get("AGRIBRAIN_PUBLICATION_CODE_COMMIT", "")).strip()
        or simulation_commit
    )
    if not simulation_commit or raw_input_commit != simulation_commit:
        raise RuntimeError("figure raw-input commit is not the simulation commit")
    return {
        "source_commit": simulation_commit,
        "source_commit_semantics": "raw_input_simulation_commit",
        "simulation_source_commit": simulation_commit,
        "renderer_code_commit": renderer_commit,
        "dual_provenance": renderer_commit != simulation_commit,
    }


def _load_summary_scalars(results_dir: Path = _RESULTS_DIR) -> dict:
    """Build ``{scenario: {mode: {metric: scalar}}}`` from the
    20-seed bootstrap means in benchmark_summary.json. Per-cell std /
    ci_low / ci_high blocks collapse to their ``mean`` value because
    the figure code's scalar reads expect plain floats.
    """
    summary_path = results_dir / "benchmark_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"{summary_path} not found. Run the HPC aggregator "
            f"first; without 20-seed scalars the aggregate panels cannot render."
        )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    meta = payload.get("_meta") if isinstance(payload, dict) else None
    if not isinstance(meta, dict):
        raise RuntimeError("benchmark_summary.json lacks run provenance")
    expected_commit = os.environ.get("AGRIBRAIN_GIT_COMMIT", "").strip()
    expected_tag = os.environ.get("RUN_TAG", "").strip()
    if not expected_commit or not expected_tag:
        raise RuntimeError("AGRIBRAIN_GIT_COMMIT and RUN_TAG are required")
    if meta.get("source_commit") != expected_commit:
        raise RuntimeError("benchmark_summary.json commit mismatch")
    if meta.get("run_tag") != expected_tag:
        raise RuntimeError("benchmark_summary.json run-tag mismatch")
    if int(meta.get("n_seeds", -1)) != 20:
        raise RuntimeError("benchmark_summary.json is not the exact 20-seed panel")
    summary = payload.get("summary", payload)
    expected_scenarios = {
        "heatwave", "overproduction", "cyber_outage",
        "adaptive_pricing", "baseline",
    }
    expected_modes = {
        "static", "hybrid_rl", "no_pinn", "no_slca", "no_context", "mcp_only",
        "pirag_only", "agribrain", "agribrain_standard_rag",
        "agribrain_no_peer", "agribrain_sign_unconstrained",
    }
    required_metrics = {"ari", "waste", "rle", "slca", "carbon", "equity"}
    if not isinstance(summary, dict) or set(summary) != expected_scenarios:
        raise RuntimeError("benchmark_summary.json lacks the exact scenario panel")
    for scenario, modes in summary.items():
        if not isinstance(modes, dict) or set(modes) != expected_modes:
            raise RuntimeError(
                f"benchmark_summary.json {scenario} lacks the exact 11-mode panel"
            )
        for mode, metrics in modes.items():
            if not isinstance(metrics, dict) or not required_metrics.issubset(metrics):
                raise RuntimeError(
                    f"benchmark_summary.json {scenario}/{mode} lacks core metrics"
                )
            for metric in required_metrics:
                node = metrics[metric]
                if (
                    not isinstance(node, dict)
                    or int(node.get("n_seeds", -1)) != 20
                    or any(key not in node for key in ("mean", "ci_low", "ci_high"))
                ):
                    raise RuntimeError(
                        f"benchmark_summary.json {scenario}/{mode}/{metric} "
                        "is not a complete 20-seed interval"
                    )
    out: dict = {}
    for sc, modes in summary.items():
        out[sc] = {}
        for mode, metrics in modes.items():
            ep: dict = {}
            for k, v in metrics.items():
                if isinstance(v, dict) and "mean" in v:
                    ep[k] = v["mean"]
                else:
                    ep[k] = v
            out[sc][mode] = ep
    return out


def _load_seed_traces() -> dict[int, dict]:
    """Load the single seed scope selected by the figure renderer and
    load every per-seed envelope's ``traces`` block. Returns
    ``{seed: {scenario: {mode: {field: list}}}}``. Missing
    ``traces`` keys (older envelopes that pre-date the trace dump)
    are skipped.
    """
    import generate_figures as gf  # type: ignore

    out: dict[int, dict] = {}
    for _path, obj in gf._load_seed_payloads():
        seed = obj.get("seed")
        traces = obj.get("traces")
        if not isinstance(seed, int) or not isinstance(traces, dict):
            continue
        out[seed] = traces
    return out


def _byte_record(path: Path, *, name: str, seed: int | None = None) -> dict:
    payload_bytes = path.read_bytes()
    record = {
        "file": name,
        "bytes": len(payload_bytes),
        "sha256": hashlib.sha256(payload_bytes).hexdigest(),
    }
    if seed is not None:
        record["seed"] = int(seed)
    return record


def _snapshot_render_inputs(
    snapshot_root: Path,
    *,
    source_seed_root: Path,
    seeds: list[int],
) -> tuple[list[dict], list[dict]]:
    """Copy every renderer input to an isolated, byte-stable snapshot.

    The second source pass prevents a concurrently changing publication tree
    from producing a mixed snapshot. Figures read only from the snapshot; the
    returned records describe those literal bytes and are later checked
    against both the canonical files and the final artifact manifest.
    """

    if (
        not source_seed_root.is_dir()
        or source_seed_root.is_symlink()
        or _RESULTS_DIR.is_symlink()
    ):
        raise RuntimeError("figure input roots must be real, non-symlink directories")
    snapshot_seed_root = snapshot_root / "benchmark_seeds"
    snapshot_seed_root.mkdir(parents=True)

    aggregate_records: list[dict] = []
    seed_records: list[dict] = []
    source_pairs: list[tuple[Path, Path, dict]] = []
    for name in _AGGREGATE_FIGURE_INPUTS:
        source = _RESULTS_DIR / name
        destination = snapshot_root / name
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"figure aggregate input is missing or unsafe: {source}")
        shutil.copyfile(source, destination)
        record = _byte_record(destination, name=name)
        aggregate_records.append(record)
        source_pairs.append((source, destination, record))

    for seed in seeds:
        name = f"benchmark_seeds/seed_{seed}.json"
        source = source_seed_root / f"seed_{seed}.json"
        destination = snapshot_seed_root / f"seed_{seed}.json"
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"figure seed input is missing or unsafe: {source}")
        shutil.copyfile(source, destination)
        record = _byte_record(destination, name=name, seed=seed)
        seed_records.append(record)
        source_pairs.append((source, destination, record))

    for source, destination, expected in source_pairs:
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"figure input changed type during snapshot: {source}")
        actual_source = _byte_record(source, name=str(expected["file"]))
        if any(
            actual_source[key] != expected[key]
            for key in ("file", "bytes", "sha256")
        ):
            raise RuntimeError(f"figure input changed during snapshot: {source}")
        if _byte_record(destination, name=str(expected["file"])) != actual_source:
            raise RuntimeError(f"figure input snapshot changed after copy: {destination}")
    return aggregate_records, seed_records


def _verify_render_inputs_unchanged(
    snapshot_root: Path,
    *,
    source_seed_root: Path,
    aggregate_records: list[dict],
    seed_records: list[dict],
) -> None:
    """Rehash the complete source and snapshot after all panels render."""

    for record in [*aggregate_records, *seed_records]:
        relative = Path(str(record["file"]))
        snapshot_path = snapshot_root / relative
        source_path = (
            source_seed_root / relative.name
            if relative.parts[0] == "benchmark_seeds"
            else _RESULTS_DIR / relative
        )
        for label, path in (("snapshot", snapshot_path), ("source", source_path)):
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(
                    f"figure {label} input changed type during render: {path}"
                )
            actual = _byte_record(path, name=str(record["file"]))
            if any(actual[key] != record[key] for key in ("file", "bytes", "sha256")):
                raise RuntimeError(
                    f"figure {label} input bytes changed during render: {path}"
                )


def _write_figure_provenance(
    single_seed: int,
    seeds: list[int],
    *,
    source_seed_root: Path,
    aggregate_input_artifacts: list[dict],
    seed_input_artifacts: list[dict],
) -> None:
    """Record the exact run, raw fields, and display transforms per panel."""
    from importlib.metadata import version as _package_version

    from matplotlib import font_manager as _font_manager

    output_dir = Path(os.environ["FIGURE_OUTPUT_DIR"])
    figure_names = sorted(
        f"{stem}.{extension}"
        for stem in (
            "heatwave", "overproduction", "cyber_outage",
            "adaptive_pricing", "cross_scenario", "ablation",
            "transport_emissions", "performance_efficiency", "context_value",
            "stress_robustness",
        )
        for extension in ("png", "pdf")
    )
    rendered_artifacts = []
    for name in figure_names:
        path = output_dir / name
        payload_bytes = path.read_bytes()
        rendered_artifacts.append({
            "file": name,
            "bytes": len(payload_bytes),
            "sha256": hashlib.sha256(payload_bytes).hexdigest(),
        })
    font_path = Path(
        _font_manager.findfont(
            _font_manager.FontProperties(
                family=["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
            ),
            fallback_to_default=True,
        )
    ).resolve(strict=True)
    font_record = _byte_record(font_path, name=font_path.name)
    font_record.update({
        "resolved_family": _font_manager.FontProperties(
            fname=str(font_path),
        ).get_name(),
        "resolved_path": str(font_path),
    })
    payload = {
        "schema_version": 3,
        # The source_commit compatibility alias is explicitly scoped to raw
        # simulation inputs rather than the code that rendered the figures.
        **_figure_source_identity(),
        "run_tag": os.environ.get("RUN_TAG", "").strip() or None,
        "seed_root": str(source_seed_root),
        "render_input_isolated_snapshot": True,
        "seed_panel": seeds,
        "n_seed_envelopes_loaded": len(seeds),
        "seed_input_artifacts": seed_input_artifacts,
        "aggregate_input_artifacts": aggregate_input_artifacts,
        "illustrative_seed": int(single_seed),
        "illustrative_seed_posture": (
            "predeclared trace illustration only; population summaries and "
            "uncertainty use the complete seed panel"
        ),
        "publication_style": publication_style_contract(),
        "renderer_environment": {
            "matplotlib": _package_version("matplotlib"),
            "numpy": _package_version("numpy"),
            "pillow": _package_version("Pillow"),
            "resolved_font": font_record,
        },
        "rendered_artifacts": rendered_artifacts,
        "panels": {
            "heatwave": {
                "a": {"fields": ["temp_outcome_environmental_trace", "rh_outcome_environmental_trace"], "aggregation": "illustrative seed, raw", "n_seeds": 1},
                "b": {"fields": ["rho_outcome_environmental_trace", "rho_policy_observed_trace"], "aggregation": "illustrative seed, raw", "n_seeds": 1},
                "c": {"fields": ["prob_trace", "rho_policy_observed_trace"], "aggregation": "illustrative seed, raw action probabilities with RLE-event guide", "n_seeds": 1},
                "d": {"fields": ["ari_trace"], "aggregation": "per-step seed mean, edge-truncated 3-hour display smoother", "n_seeds": 20},
            },
            "overproduction": {
                "a": {"fields": ["inventory_outcome_environmental_trace", "demand_outcome_environmental_trace"], "aggregation": "illustrative seed, raw", "n_seeds": 1},
                "b": {"fields": ["waste_trace"], "aggregation": "illustrative seed per method, edge-truncated 3-hour rolling mean", "n_seeds": 1},
                "c": {"fields": ["rho_outcome_environmental_trace", "action_trace"], "aggregation": "illustrative seed per method, trailing 3-hour RLE", "n_seeds": 1},
                "d": {"fields": ["slca_component_trace"], "aggregation": "per-seed component means with cross-seed uncertainty", "n_seeds": 20},
            },
            "cyber_outage": {
                "a": {"fields": ["ari_trace"], "aggregation": "illustrative seed, edge-truncated 3-hour rolling mean", "n_seeds": 1},
                "b": {"fields": ["action_trace"], "aggregation": "pre/during proportions", "n_seeds": 20},
                "c": {"fields": ["action_trace"], "aggregation": "seed-level pre/during behavior", "n_seeds": 20},
                "d": {"fields": ["ari_trace", "waste_trace", "action_trace"], "aggregation": "outage-window seed means", "n_seeds": 20},
            },
            "adaptive_pricing": {
                "a": {"fields": ["demand_trace"], "aggregation": "illustrative seed policy-observed demand forecast, trailing policy Bollinger window", "n_seeds": 1},
                "b": {"fields": ["action_trace"], "aggregation": "illustrative seed, twelve equal time bins", "n_seeds": 1},
                "c": {"fields": ["equity_trace"], "aggregation": "illustrative seed per method, edge-truncated 3-hour rolling mean", "n_seeds": 1},
                "d": {"fields": ["reward_trace"], "aggregation": "illustrative seed per method, edge-truncated 3-hour rolling mean", "n_seeds": 1},
            },
            "cross_scenario_and_secondary": {
                "fields": list(_AGGREGATE_FIGURE_INPUTS),
                "aggregation": "validated seed-level summaries; see each source artifact metadata",
                "n_seeds": 20,
            },
        },
    }
    (output_dir / "figure_provenance.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )


def _build_data_dict(scalars: dict, seed_traces: dict[int, dict],
                     single_seed: int) -> dict:
    """Build the ``data["results"]`` dict figure code expects.

    For each (scenario, mode) cell, merges the 20-seed scalar metrics
    (from benchmark_summary.json) with the chosen single-seed's
    per-step traces (from benchmark_seeds/seed_<single_seed>.json).
    The figure code then reads the merged dict the same way it reads
    a fresh ``run_all()`` payload -- single-seed line plots come from
    the trace fields, scalar bars / scatter markers come from the
    20-seed bootstrap means.
    """
    seed_block = seed_traces.get(single_seed, {})
    data: dict = {"results": {}}
    for sc, modes in scalars.items():
        data["results"][sc] = {}
        for mode, ep_scalars in modes.items():
            ep = dict(ep_scalars)  # copy so we can extend in place
            traces_for_cell = seed_block.get(sc, {}).get(mode, {})
            for field, seq in traces_for_cell.items():
                ep[field] = seq
            data["results"][sc][mode] = ep
    return data


def main() -> int:
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    if os.environ.get("STRICT_VALIDATION") != "1":
        print("ERROR: STRICT_VALIDATION=1 is mandatory for cached rendering")
        return 2
    required_env = (
        "AGRIBRAIN_GIT_COMMIT", "RUN_TAG", "BENCHMARK_SEEDS",
        "FIGURE_SEED_ROOT", "FIGURE_OUTPUT_DIR",
    )
    missing_env = [
        name for name in required_env if not os.environ.get(name, "").strip()
    ]
    if missing_env:
        print(
            "ERROR: required render identity/output variables missing: "
            + ", ".join(missing_env)
        )
        return 2
    expected_seeds = {
        42, 1337, 2024, 7, 99, 101, 202, 303, 404, 505,
        606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515,
    }
    try:
        declared_seeds = {
            int(value) for value in os.environ["BENCHMARK_SEEDS"].split(",")
            if value.strip()
        }
    except ValueError:
        print("ERROR: BENCHMARK_SEEDS contains a non-integer value")
        return 2
    if declared_seeds != expected_seeds:
        print("ERROR: BENCHMARK_SEEDS is not the exact locked 20-seed panel")
        return 2
    try:
        _require_renderer_source_identity()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 2
    output_dir = Path(os.environ["FIGURE_OUTPUT_DIR"]).resolve()
    canonical_results = _RESULTS_DIR.resolve()
    if (
        output_dir == canonical_results
        and os.environ.get("AGRIBRAIN_PUBLICATION_RENDER") != "1"
    ):
        print(
            "ERROR: only the canonical HPC publisher may overwrite the "
            "publication results directory"
        )
        return 2

    source_seed_root = _SEEDS_DIR.resolve()
    original_figure_seed_root = os.environ.get("FIGURE_SEED_ROOT")
    with tempfile.TemporaryDirectory(prefix="agribrain_figure_inputs_") as temp_name:
        snapshot_root = Path(temp_name)
        try:
            aggregate_records, seed_records = _snapshot_render_inputs(
                snapshot_root,
                source_seed_root=source_seed_root,
                seeds=sorted(expected_seeds),
            )
        except (OSError, RuntimeError) as exc:
            print(f"ERROR: {exc}")
            return 2
        snapshot_seed_root = snapshot_root / "benchmark_seeds"
        os.environ["FIGURE_SEED_ROOT"] = str(snapshot_seed_root)

        log("Loading 20-seed scalars from the isolated input snapshot...")
        try:
            scalars = _load_summary_scalars(snapshot_root)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"ERROR: {exc}")
            return 2

        log("Loading per-seed traces from the isolated input snapshot...")
        seed_traces = _load_seed_traces()
        if not seed_traces:
            print(
                "ERROR: no valid per-seed JSONs found in the isolated figure "
                "input snapshot"
            )
            return 1

        if _PREFERRED_SINGLE_SEED in seed_traces:
            single_seed = _PREFERRED_SINGLE_SEED
        else:
            single_seed = min(seed_traces.keys())
            log(
                f"Note: seed {_PREFERRED_SINGLE_SEED} not on disk; "
                f"using smallest available seed {single_seed} as the "
                f"single-seed representative."
            )
        log(
            f"Cached seeds available: {sorted(seed_traces.keys())} "
            f"({len(seed_traces)} total). Single-seed representative: "
            f"seed {single_seed}."
        )

        data = _build_data_dict(scalars, seed_traces, single_seed)

        log("Rendering figures from the isolated input snapshot...")
        import generate_figures as gf  # type: ignore

        original_gf_results_dir = gf.RESULTS_DIR
        gf.RESULTS_DIR = snapshot_root
        figs = [
            ("heatwave",            lambda: gf.fig2_heatwave(data)),
            ("overproduction",      lambda: gf.fig3_overproduction(data)),
            ("cyber_outage",        lambda: gf.fig4_cyber(data)),
            ("adaptive_pricing",    lambda: gf.fig5_pricing(data)),
            ("cross_scenario",      lambda: gf.fig6_cross(data)),
            ("ablation",            lambda: gf.fig7_ablation(data)),
            ("transport_emissions", lambda: gf.fig8_transport_emissions(data)),
            ("performance_efficiency", lambda: gf.fig11_performance_efficiency(data)),
            ("context_value",       lambda: gf.fig12_context_channels(data)),
            ("stress_robustness",   lambda: gf.fig13_stress_robustness(data)),
        ]
        failures: list[tuple[str, str]] = []
        try:
            for name, fn in figs:
                log(f"  {name}...")
                try:
                    fn()
                except Exception as exc:  # noqa: BLE001 - log + continue
                    print(f"  FAIL: {name}: {type(exc).__name__}: {exc}")
                    failures.append((name, str(exc)))
        finally:
            gf.RESULTS_DIR = original_gf_results_dir
            if original_figure_seed_root is None:
                os.environ.pop("FIGURE_SEED_ROOT", None)
            else:
                os.environ["FIGURE_SEED_ROOT"] = original_figure_seed_root

        if not failures:
            try:
                _verify_render_inputs_unchanged(
                    snapshot_root,
                    source_seed_root=source_seed_root,
                    aggregate_records=aggregate_records,
                    seed_records=seed_records,
                )
            except (OSError, RuntimeError) as exc:
                print(f"ERROR: {exc}")
                return 2
            _write_figure_provenance(
                single_seed,
                sorted(seed_traces),
                source_seed_root=source_seed_root,
                aggregate_input_artifacts=aggregate_records,
                seed_input_artifacts=seed_records,
            )
            log("Saved figure_provenance.json")

    if failures:
        print()
        print("=" * 60)
        print(
            f"WARNING: {len(failures)} figure(s) failed to render. "
            "Most likely cause: required trace fields are missing "
            "from the per-seed JSONs. The HPC seed runner "
            "(run_single_seed.py) writes the canonical TRACE_FIELDS "
            "set; if you see KeyError on a particular trace, the "
            "cached HPC run was produced before that field was added."
        )
        for name, msg in failures:
            print(f"  {name}: {msg}")
        print("=" * 60)

    try:
        _require_renderer_source_identity()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 2

    if output_dir == canonical_results:
        log("DONE. Canonical publisher will build and verify the artifact manifest.")
    else:
        log(f"DONE. Derived figures are isolated under {output_dir}.")
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
