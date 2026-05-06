#!/usr/bin/env python3
"""Re-render every figure from on-disk artifacts (no simulator run).

Use this when the figure source code has changed but the underlying
simulation data hasn't -- e.g. a styling tweak after an HPC run, where
the canonical re-render path (``python generate_figures.py``) would
otherwise trigger a fresh ~80-minute ``run_all()`` simulation.

Two on-disk caches drive every panel:

  1. ``benchmark_summary.json`` -- 20-seed bootstrap means / stds /
     CI bounds per (scenario, mode, metric). Fig 6 / 7 / 8 panel B /
     fig 9 panels A & B / fig 10 read these directly.

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
~30-60 s for all nine figures.

Usage::

    python mvp/simulation/regenerate_figures_from_cache.py

Outputs land at ``mvp/simulation/results/fig*.{png,pdf}`` (overwriting
the previous render). Re-stamp the artifact manifest afterwards::

    python mvp/simulation/analysis/build_artifact_manifest.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


_SIM_DIR = Path(__file__).resolve().parent
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))

_RESULTS_DIR = _SIM_DIR / "results"
_SEEDS_DIR = _RESULTS_DIR / "benchmark_seeds"

#: Seed treated as the "single-seed representative" for figure code
#: that reads ``ab["X_trace"]`` directly (most line plots).
#: Falls back to the smallest available seed if 42 isn't on disk.
_PREFERRED_SINGLE_SEED = 42


def _load_summary_scalars() -> dict:
    """Build ``{scenario: {mode: {metric: scalar}}}`` from the
    20-seed bootstrap means in benchmark_summary.json. Per-cell std /
    ci_low / ci_high blocks collapse to their ``mean`` value because
    the figure code's scalar reads expect plain floats.
    """
    summary_path = _RESULTS_DIR / "benchmark_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"{summary_path} not found. Run the HPC aggregator "
            f"first; without 20-seed scalars fig 6 / 7 / 8 panel B / "
            f"fig 9 / fig 10 cannot render."
        )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", payload)
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
    """Walk benchmark_seeds/ (flat or RUN_TAG sub-folder layout) and
    load every per-seed envelope's ``traces`` block. Returns
    ``{seed: {scenario: {mode: {field: list}}}}``. Missing
    ``traces`` keys (older envelopes that pre-date the trace dump)
    are skipped.
    """
    if not _SEEDS_DIR.exists():
        return {}
    seed_files = list(_SEEDS_DIR.glob("seed_*.json"))
    if not seed_files:
        for sub in _SEEDS_DIR.iterdir():
            if sub.is_dir():
                seed_files.extend(sub.glob("seed_*.json"))
    out: dict[int, dict] = {}
    for sp in seed_files:
        try:
            obj = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        seed = obj.get("seed")
        traces = obj.get("traces")
        if not isinstance(seed, int) or not isinstance(traces, dict):
            continue
        out[seed] = traces
    return out


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

    log("Loading 20-seed scalars from benchmark_summary.json...")
    try:
        scalars = _load_summary_scalars()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    log("Loading per-seed traces from benchmark_seeds/...")
    seed_traces = _load_seed_traces()
    if not seed_traces:
        print(
            "ERROR: no per-seed JSONs found under "
            f"{_SEEDS_DIR}. Run the HPC seed array first; without "
            "per-step traces, fig 2 / 3 / 4 / 5 / 8A cannot render."
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

    log("Rendering figures...")
    import generate_figures as gf  # type: ignore

    figs = [
        ("fig2_heatwave",                lambda: gf.fig2_heatwave(data)),
        ("fig3_overproduction",          lambda: gf.fig3_overproduction(data)),
        ("fig4_cyber",                   lambda: gf.fig4_cyber(data)),
        ("fig5_pricing",                 lambda: gf.fig5_pricing(data)),
        ("fig6_cross",                   lambda: gf.fig6_cross(data)),
        ("fig7_ablation",                lambda: gf.fig7_ablation(data)),
        ("fig8_green_ai",                lambda: gf.fig8_green_ai(data)),
        ("fig9_fault_degradation",       lambda: gf.fig9_fault_degradation()),
        ("fig10_latency_quality_frontier",
            lambda: gf.fig10_latency_quality_frontier(data)),
    ]
    failures: list[tuple[str, str]] = []
    for name, fn in figs:
        log(f"  {name}...")
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - log + continue
            print(f"  FAIL: {name}: {type(exc).__name__}: {exc}")
            failures.append((name, str(exc)))

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

    log("DONE. Refresh the artifact manifest:")
    log("  python mvp/simulation/analysis/build_artifact_manifest.py")
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
