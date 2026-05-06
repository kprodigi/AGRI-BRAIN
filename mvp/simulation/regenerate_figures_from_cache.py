#!/usr/bin/env python3
"""Re-render figures from existing on-disk artifacts (no simulator run).

Use this when the figure source code has changed but the underlying
simulation data hasn't -- e.g. a styling tweak after an HPC run, where
the canonical re-render path (``python generate_figures.py``) would
otherwise trigger a fresh ~80-minute ``run_all()`` simulation.

Synthesises ``data`` from ``benchmark_summary.json`` (multi-seed
bootstrap means → per-(scenario, mode) scalar metrics) and runs a
single deterministic episode per (scenario, mode) for the per-step
trace fields the figure code reads (``ari_trace``, ``waste_trace``,
``rho_trace``, ``carbon_trace``, ``action_trace``, ``prob_trace``,
``hours``). The seed-CI ribbon path in fig 2 / fig 4 picks up the
multi-seed traces from ``benchmark_seeds/<RUN_TAG>/`` automatically
via ``_load_per_seed_traces`` -- no per-seed loading needed here.

Total runtime: ~3-5 minutes per scenario × mode combination, so
~5-15 minutes for the canonical paper-trio rendering set.

Usage::

    python mvp/simulation/regenerate_figures_from_cache.py

Outputs land at ``mvp/simulation/results/fig*.{png,pdf}`` (overwriting
the previous render). Re-stamp the artifact manifest afterwards::

    python mvp/simulation/analysis/build_artifact_manifest.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np


_SIM_DIR = Path(__file__).resolve().parent
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))


def _scalars_from_summary(summary_path: Path) -> dict:
    """Build a ``data['results'][sc][mode] = {metric: value}`` dict from
    benchmark_summary.json's bootstrap means. fig8 panel b and fig10
    read these directly; the per-step trace fields are added below.
    """
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", payload)
    out: dict = {"results": {}}
    for sc, modes in summary.items():
        out["results"][sc] = {}
        for mode, metrics in modes.items():
            ep: dict = {}
            for k, v in metrics.items():
                if isinstance(v, dict) and "mean" in v:
                    ep[k] = v["mean"]
                else:
                    ep[k] = v
            out["results"][sc][mode] = ep
    return out


def main() -> int:
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    # Force deterministic mode for reproducibility of the per-step traces
    # this helper synthesises. The seed-CI ribbon (when traces exist) and
    # the bootstrap CIs in figures (which read from benchmark_summary
    # directly) are unaffected by this -- they come from the canonical
    # 20-seed stochastic HPC run.
    os.environ["DETERMINISTIC_MODE"] = "true"

    log("Loading benchmark_summary.json scalars...")
    summary_path = _SIM_DIR / "results" / "benchmark_summary.json"
    if not summary_path.exists():
        print(
            f"ERROR: {summary_path} not found. Run the HPC aggregator "
            f"first or fall back to ``python generate_figures.py`` "
            f"(which runs the full simulator)."
        )
        return 1
    data = _scalars_from_summary(summary_path)

    log("Importing simulator...")
    from generate_results import (  # type: ignore
        DATA_CSV, Policy,
        apply_scenario, run_episode, pd,
    )

    # Required (scenario, mode) combinations for the per-step traces
    # the figure code expects. Other figures (fig 6 / 7 / 8 panel b /
    # fig 9 / fig 10) read scalars only and use the data dict above.
    targets = {
        "heatwave": ["static", "hybrid_rl", "agribrain"],
        "overproduction": ["static", "hybrid_rl", "agribrain"],
        "cyber_outage": ["static", "hybrid_rl", "agribrain"],
        "adaptive_pricing": ["static", "hybrid_rl", "agribrain"],
    }

    df_base = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])
    policy = Policy()

    # Trace fields figure code reads from the episode dict.
    TRACE_FIELDS = (
        "ari_trace", "waste_trace", "rho_trace", "carbon_trace",
        "action_trace", "prob_trace", "hours",
        "batch_effective_rho_trace", "effective_rho_trace",
        "temp_trace",
    )

    for sc, modes in targets.items():
        log(f"apply_scenario({sc})...")
        df_scen = apply_scenario(df_base, sc, policy, np.random.default_rng(47))
        for mode in modes:
            log(f"  run_episode {sc} x {mode}...")
            rng = np.random.default_rng(42)
            ep = run_episode(df_scen, mode, policy, rng, sc, seed=42)
            target = data["results"][sc].setdefault(mode, {})
            for k, v in ep.items():
                if k.endswith("_trace") or k in TRACE_FIELDS or k not in target:
                    target[k] = v

    log("Rendering figures...")
    import generate_figures as gf  # type: ignore

    figs = [
        ("fig2_heatwave", lambda: gf.fig2_heatwave(data)),
        ("fig3_overproduction", lambda: gf.fig3_overproduction(data)),
        ("fig4_cyber", lambda: gf.fig4_cyber(data)),
        ("fig5_pricing", lambda: gf.fig5_pricing(data)),
        ("fig6_cross", lambda: gf.fig6_cross(data)),
        ("fig7_ablation", lambda: gf.fig7_ablation(data)),
        ("fig8_green_ai", lambda: gf.fig8_green_ai(data)),
        ("fig9_fault_degradation", lambda: gf.fig9_fault_degradation()),
        ("fig10_latency_quality_frontier",
            lambda: gf.fig10_latency_quality_frontier(data)),
    ]
    for name, fn in figs:
        log(f"  {name}...")
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - log + continue
            print(f"  FAIL: {name}: {exc}")

    log("DONE. Refresh the artifact manifest:")
    log("  python mvp/simulation/analysis/build_artifact_manifest.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
