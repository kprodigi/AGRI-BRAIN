#!/usr/bin/env python3
"""#1b over-steering ablation: gated agribrain vs ungated/uncapped (over-steer).

Tests whether the conservative *gated + capped* context design is optimal, or
whether forcing the context modifier onto every decision at full strength would
do better. The ONLY difference between the two arms is the
``context_to_logits.OVER_STEER`` flag, which (a) bypasses the
retrieval-quality guard so the modifier is applied on every decision instead of
the guard-selected ~25%, and (b) removes the per-element magnitude clip. Every
other factor is held identical -- operating mode (``agribrain``), the 20
canonical seeds, the per-seed RNG streams, the 4-iteration warm-start learning,
and the scenario perturbations -- so the *paired, per-seed* ARI difference
isolates the effect of the gating + cap.

Interpretation of ``mean_diff = over_steer - gated``:
  * mean_diff < 0, CI excludes 0  -> gating is OPTIMAL (over-steering hurts);
  * CI includes 0                 -> neutral (gating costs nothing, removes risk);
  * mean_diff > 0, CI excludes 0  -> over-steering helps (challenges the design).

Honest by construction: the script reports whichever way the data falls.

Saved results / reuse (so an HPC or local rerun can reuse what's done):
  * Every (scenario, seed) cell -- the raw {gated, over_steer} metrics for both
    arms -- is checkpointed to ``over_steer_cells/<scenario>__seed<N>.json`` the
    moment it finishes. The run is therefore crash-resilient and resumable.
  * ``--resume`` (default) skips any cell whose checkpoint already exists and
    loads it from disk; ``--fresh`` recomputes everything. So a rerun reuses the
    saved cells and only computes the missing ones.
  * ``--aggregate-only`` recomputes ``over_steer_ablation.json`` from the saved
    cells without running any episode (e.g. copy the cells off HPC, aggregate
    locally). The final JSON also embeds the raw per-seed arrays, so it is
    self-contained for re-aggregation.
  * Statistics reuse the canonical ``aggregate_seeds`` BCa bootstrap + Wilcoxon.

Run::

    PYTHONHASHSEED=0 python mvp/simulation/benchmarks/run_over_steer_ablation.py
    # resume an interrupted run (default behaviour):
    PYTHONHASHSEED=0 python .../run_over_steer_ablation.py --resume
    # just re-aggregate saved cells, no episodes:
    python .../run_over_steer_ablation.py --aggregate-only
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SIM = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SIM))
sys.path.insert(0, str(_SIM / "benchmarks"))
sys.path.insert(0, str(_SIM.parent.parent / "agribrain" / "backend"))

from generate_results import (  # noqa: E402
    DATA_CSV, SCENARIOS, Policy, apply_scenario, run_episode,
)
from stochastic import make_stochastic_layer  # noqa: E402
import pirag.context_to_logits as ctx  # noqa: E402  (holds the OVER_STEER flag)
from aggregate_seeds import (  # noqa: E402
    bootstrap_mean_diff_ci, wilcoxon_signed_rank_pvalue,
)

# The canonical 20 benchmark seeds (the dab51b1 run that backs Table 1).
SEEDS = [7, 42, 99, 101, 202, 303, 404, 505, 606, 707,
         808, 909, 1010, 1111, 1212, 1313, 1337, 1414, 1515, 2024]
PERTURBED = ("heatwave", "overproduction", "cyber_outage", "adaptive_pricing")
N_ITER = 4              # matches _MULTI_EPISODE_MODES["agribrain"]
METRICS = ("ari", "waste", "slca", "rle")
RESULTS = _SIM / "results"
DEFAULT_CELLS = RESULTS / "over_steer_cells"
DEFAULT_OUTPUT = RESULTS / "over_steer_ablation.json"


def _run_agribrain(df_scenario, scenario: str, seed: int, over_steer: bool) -> dict:
    """One agribrain run (4 warm-start iterations) under the given flag.

    Fresh policy + learner cache + RNG per call, seeded by ``seed`` so the two
    flag conditions draw identical stochastic streams (paired design)."""
    ctx.OVER_STEER = over_steer
    try:
        policy = Policy()
        cache: dict = {}
        mode_rng = np.random.default_rng(seed)
        stoch = make_stochastic_layer(np.random.default_rng(seed + 1))
        episode = None
        for _ in range(N_ITER):
            episode = run_episode(
                df_scenario, "agribrain", policy, mode_rng, scenario,
                stoch=stoch, seed=seed, learner_state_cache=cache,
                context_learner_overrides=None,
            )
        return {m: float(episode[m]) for m in METRICS}
    finally:
        ctx.OVER_STEER = False  # never leave the global flag set


def _cell_path(cells_dir: Path, scenario: str, seed: int) -> Path:
    return cells_dir / f"{scenario}__seed{seed}.json"


def _compute_cell(df_scenario, scenario: str, seed: int) -> dict:
    return {
        "scenario": scenario, "seed": int(seed), "n_iter": N_ITER,
        "gated": _run_agribrain(df_scenario, scenario, seed, over_steer=False),
        "over_steer": _run_agribrain(df_scenario, scenario, seed, over_steer=True),
        "saved_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return "unknown"


def _paired_stats(over: np.ndarray, gated: np.ndarray, key) -> dict:
    lo, hi = bootstrap_mean_diff_ci(over, gated, n_boot=10000, paired=True, cell_key=key)
    p = wilcoxon_signed_rank_pvalue(over, gated, cell_key=key)
    diff = over - gated
    return {
        "gated_mean": float(gated.mean()),
        "over_steer_mean": float(over.mean()),
        "mean_diff": float(diff.mean()),
        "ci_low": float(lo), "ci_high": float(hi),
        "wilcoxon_p": float(p),
        "n": int(over.size),
    }


def _aggregate_from_cells(cells_dir: Path, scen_list, seeds) -> dict:
    """Build the by-scenario + pooled stats (and embed raw arrays) from the
    saved per-(scenario, seed) checkpoint files. No episodes are run."""
    raw = {}   # scenario -> metric -> {"gated":[...], "over":[...], "seeds":[...]}
    for scenario in scen_list:
        acc = {m: {"gated": [], "over": []} for m in METRICS}
        acc_seeds = []
        for seed in seeds:
            cp = _cell_path(cells_dir, scenario, seed)
            if not cp.exists():
                continue
            cell = json.loads(cp.read_text())
            acc_seeds.append(int(seed))
            for m in METRICS:
                acc[m]["gated"].append(cell["gated"][m])
                acc[m]["over"].append(cell["over_steer"][m])
        if acc_seeds:
            for m in METRICS:
                acc[m]["seeds"] = acc_seeds
            raw[scenario] = acc

    by_scn = {}
    for scenario, acc in raw.items():
        cell = {"n_seeds": len(acc["ari"]["gated"]), "seeds": acc["ari"]["seeds"]}
        for m in METRICS:
            cell[m] = _paired_stats(np.array(acc[m]["over"]),
                                    np.array(acc[m]["gated"]),
                                    key=("over_steer", scenario, m))
        by_scn[scenario] = cell

    pooled = {}
    present = [s for s in PERTURBED if s in raw]
    for m in METRICS:
        if not present:
            break
        over = np.concatenate([np.array(raw[s][m]["over"]) for s in present])
        gated = np.concatenate([np.array(raw[s][m]["gated"]) for s in present])
        pooled[m] = _paired_stats(over, gated, key=("over_steer", "pooled", m))

    # Embed raw per-seed arrays so the JSON is self-contained for re-aggregation.
    raw_out = {s: {m: {"seeds": raw[s][m]["seeds"],
                       "gated": raw[s][m]["gated"], "over_steer": raw[s][m]["over"]}
                   for m in METRICS} for s in raw}
    return {"by_scenario": by_scn, "pooled_perturbed": pooled, "raw": raw_out}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=len(SEEDS),
                    help="use the first N canonical seeds (default: all 20)")
    ap.add_argument("--scenarios", type=str, default=None,
                    help="comma-separated subset (default: all)")
    ap.add_argument("--cells-dir", type=Path, default=DEFAULT_CELLS)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--fresh", action="store_true",
                    help="recompute every cell even if a checkpoint exists")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="re-aggregate existing cell checkpoints; run no episodes")
    args = ap.parse_args()

    seeds = SEEDS[: args.seeds]
    scen_list = (args.scenarios.split(",") if args.scenarios else list(SCENARIOS))
    args.cells_dir.mkdir(parents=True, exist_ok=True)

    if not args.aggregate_only:
        if not DATA_CSV.exists():
            raise SystemExit(f"data not found: {DATA_CSV}")
        base_df = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])
        for scenario in scen_list:
            sc_df = apply_scenario(base_df, scenario, Policy(), np.random.default_rng(7))
            for seed in seeds:
                cp = _cell_path(args.cells_dir, scenario, seed)
                if cp.exists() and not args.fresh:
                    print(f"  resume {scenario}/seed{seed} (cached)")
                    continue
                cell = _compute_cell(sc_df, scenario, seed)
                cp.write_text(json.dumps(cell, indent=2))   # checkpoint NOW
                g, o = cell["gated"]["ari"], cell["over_steer"]["ari"]
                print(f"  saved {scenario}/seed{seed}: gated={g:.4f} over={o:.4f} "
                      f"diff={o - g:+.4f}")

    agg = _aggregate_from_cells(args.cells_dir, scen_list, seeds)
    out = {
        "_meta": {
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "seeds": seeds, "n_seeds": len(seeds), "n_iter": N_ITER,
            "scenarios": scen_list,
            "cells_dir": str(args.cells_dir),
            "design": ("paired per-seed; agribrain mode under "
                       "context_to_logits.OVER_STEER False (gated, production) "
                       "vs True (guard bypassed + per-step cap removed). "
                       "mean_diff = over_steer - gated."),
            "reuse": ("per-(scenario,seed) checkpoints under cells_dir; rerun "
                      "with --resume (default) to reuse them, or --aggregate-only "
                      "to re-aggregate without running episodes."),
        },
        **agg,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))

    for scenario, cell in agg["by_scenario"].items():
        a = cell["ari"]
        print(f"  {scenario:16s} ARI gated={a['gated_mean']:.4f} "
              f"over={a['over_steer_mean']:.4f} diff={a['mean_diff']:+.4f} "
              f"CI[{a['ci_low']:+.4f},{a['ci_high']:+.4f}] p={a['wilcoxon_p']:.3g}")
    print(f"\nSaved: {args.output}  (cells: {args.cells_dir})")
    pa = agg["pooled_perturbed"].get("ari")
    if pa:
        verdict = ("gating OPTIMAL (over-steering hurts)" if pa["ci_high"] < 0
                   else "over-steering HELPS" if pa["ci_low"] > 0
                   else "neutral (gating costs nothing)")
        print(f"POOLED ARI: over-steer - gated = {pa['mean_diff']:+.4f} "
              f"CI[{pa['ci_low']:+.4f},{pa['ci_high']:+.4f}] -> {verdict}")


if __name__ == "__main__":
    main()
