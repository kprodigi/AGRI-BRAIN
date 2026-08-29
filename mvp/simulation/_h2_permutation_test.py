"""Seed-level inference for conditional observed-state feature-group masking.

Question: among recorded decisions whose modal route differs between the full
observed modifier and a zeroed modifier, how often does algebraically masking
each context feature group change that modal route, and how do those two
mask-effect indicators overlap across decisions?

For each changed decision we form the 2x2 of (MCP-feature mask effect,
retrieval-feature mask effect): both, MCP group only, retrieval group only, or
neither. ``Neither`` means both single-group reconstructions match the observed
modal route.

The conditional distinctness index D = 1 - P(neither) is the share of changed
modal routes for which the two single-group reconstructions do not both match
the observed route.

This analysis reuses retrieval results and guards from the observed execution.
It therefore describes algebraic sensitivity of the recorded policy surface;
it does not estimate the effect of disabling a communication, tool, or retrieval
channel. Separate experimental-mode comparisons provide channel-arm evidence.

Two inferential checks, both reusing the exact decision reconstruction from
aggregate_channel_attribution.py (validated to reproduce the recorded softmax
probabilities to <=1e-6):

  1. Seed-cluster bootstrap 95% CI on D (resample the 20 seeds).
  2. Seed-cluster sign-flip tests on the mean seed-level phi association and on
     D minus its within-seed independence expectation. Decisions are never
     treated as independent replicates.
"""
import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "benchmarks"))
sys.path.insert(0, str(HERE.parents[1] / "agribrain" / "backend"))
from aggregate_channel_attribution import (  # noqa: E402
    _decision,
    _walk,
    _load_episode,
    evidence_scope_metadata,
)

PERT = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing"]
B = 10000
rng = np.random.default_rng(20260605)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ledger-root", type=Path, required=True,
        help=(
            "Exact run-scoped seed_<N>/ ledger root, normally "
            "mvp/simulation/results/decision_ledger_per_seed/<RUN_TAG>"
        ),
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Explicit output path for channel_complementarity_test.json",
    )
    return parser


def build(ledger_root: Path):
    by_cell, seeds = _walk(ledger_root)
    per_seed = {}
    for (scn, mode), paths in by_cell.items():
        if mode != "agribrain" or scn not in PERT:
            continue
        for seed_n, path in paths:
            L = per_seed.setdefault(seed_n, [])
            for r in _load_episode(path):
                base = r.get("base_logits"); cm = r.get("context_modifier")
                ss = r.get("slca_shaping"); amp = r.get("slca_amp"); T = r.get("policy_temperature")
                mm = r.get("modifier_mcp"); mp = r.get("modifier_pirag")
                if None in (base, cm, ss, amp, T, mm, mp):
                    continue
                T = float(T)
                if T <= 0:
                    continue
                zero = np.zeros(3)
                d_zero = _decision(base, zero, ss, amp, T, False)
                d_observed = _decision(base, cm, ss, amp, T, True)
                if d_observed == d_zero:
                    continue  # only context-changed decisions
                d_mcp_features = _decision(base, mm, ss, amp, T, True)
                d_pirag_features = _decision(base, mp, ss, amp, T, True)
                L.append((
                    d_observed != d_pirag_features,
                    d_observed != d_mcp_features,
                ))
    return per_seed


def phi(a, b):
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a.astype(float), b.astype(float))[0, 1])


def main(
    argv: Sequence[str] | None = None,
    *,
    ledger_root: Path | None = None,
    output: Path | None = None,
) -> int:
    """Run only with explicit, run-scoped input and output paths."""
    global rng
    rng = np.random.default_rng(20260605)
    if (ledger_root is None) != (output is None):
        raise ValueError("ledger_root and output must be supplied together")
    if ledger_root is None:
        args = _build_parser().parse_args(argv)
        ledger_root = args.ledger_root
        output = args.output
    elif argv is not None:
        raise ValueError("argv cannot be combined with ledger_root/output")

    ledger_root = Path(ledger_root)
    output = Path(output)
    if not ledger_root.is_dir():
        raise RuntimeError(f"Ledger root is not a directory: {ledger_root}")

    per_seed = build(ledger_root)
    seeds = sorted(per_seed)
    if not seeds:
        raise RuntimeError(f"No eligible decision ledgers found under {ledger_root}")
    allp = [p for s in seeds for p in per_seed[s]]
    M = np.array([p[0] for p in allp], dtype=bool)
    P = np.array([p[1] for p in allp], dtype=bool)
    n = len(M)
    # Seed is the inferential unit. Equal-weight seed summaries avoid giving
    # seeds with more context-changed decisions disproportionate influence.
    seed_stats = []
    for seed_n in seeds:
        pairs = per_seed[seed_n]
        ms = np.array([p[0] for p in pairs], dtype=bool)
        ps = np.array([p[1] for p in pairs], dtype=bool)
        if len(ms) == 0:
            continue
        both_match_s = float(np.mean(~ms & ~ps))
        d_s = 1.0 - both_match_s
        e_s = 1.0 - float(np.mean(~ms)) * float(np.mean(~ps))
        seed_stats.append({
            "seed": int(seed_n), "n": int(len(ms)),
            "mcp_feature_group_only": float(np.mean(ms & ~ps)),
            "pirag_feature_group_only": float(np.mean(~ms & ps)),
            "both_feature_groups": float(np.mean(ms & ps)),
            "neither_feature_group": both_match_s,
            "conditional_distinctness_index": d_s,
            "phi": phi(ms, ps),
            "distinctness_minus_independence": d_s - e_s,
            "distinctness_independence_baseline": e_s,
        })
    if not seed_stats:
        raise RuntimeError("No seeds contained context-changed decisions")
    mcp_only = float(np.mean([s["mcp_feature_group_only"] for s in seed_stats]))
    pir_only = float(np.mean([s["pirag_feature_group_only"] for s in seed_stats]))
    both = float(np.mean([s["both_feature_groups"] for s in seed_stats]))
    neither = float(np.mean([s["neither_feature_group"] for s in seed_stats]))
    D = float(np.mean([s["conditional_distinctness_index"] for s in seed_stats]))
    print(f"n changed decisions = {n:,}  ({len(seed_stats)} contributing seeds, {PERT})")
    print(f"2x2 mask effects: MCP-group-only={mcp_only:.3f}  "
          f"retrieval-group-only={pir_only:.3f}  both={both:.3f}  neither={neither:.3f}")
    print(f"Conditional distinctness index D = {D:.4f}")

    # Seed-cluster bootstrap CI on the equal-weight mean seed D.
    boots = []
    for _ in range(B):
        idx = rng.integers(0, len(seed_stats), len(seed_stats))
        boots.append(float(np.mean([
            seed_stats[i]["conditional_distinctness_index"] for i in idx
        ])))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  seed-cluster bootstrap 95% CI on D = [{lo:.4f}, {hi:.4f}]  "
          f"(all > 0.5: {lo > 0.5})")

    # (2) cluster-level sign-flip tests. The number of independent units is
    # the number of seeds, not the number of decisions.
    phi_by_seed = np.asarray([s["phi"] for s in seed_stats], dtype=float)
    d_delta_by_seed = np.asarray(
        [s["distinctness_minus_independence"] for s in seed_stats], dtype=float
    )
    phi_obs = float(np.mean(phi_by_seed))
    d_delta_obs = float(np.mean(d_delta_by_seed))
    E_D_indep = float(np.mean([
        s["distinctness_independence_baseline"] for s in seed_stats
    ]))
    rng2 = np.random.default_rng(7)
    phinull = np.empty(B); Dnull = np.empty(B)
    for i in range(B):
        signs = rng2.choice((-1.0, 1.0), size=len(seed_stats))
        phinull[i] = float(np.mean(phi_by_seed * signs))
        Dnull[i] = float(np.mean(d_delta_by_seed * signs))
    p_phi_two = (np.sum(np.abs(phinull) >= abs(phi_obs)) + 1) / (B + 1)
    p_D_upper = (np.sum(Dnull >= d_delta_obs) + 1) / (B + 1)
    print(f"  phi(feature-group mask effects) = {phi_obs:+.4f}  "
          f"(null mean {phinull.mean():+.4f}, two-sided p = {p_phi_two:.4g})")
    print(f"  D vs independence: E[D_indep] = {E_D_indep:.4f}; "
          f"mean seed delta={d_delta_obs:+.4f}; sign-flip null "
          f"[{np.percentile(Dnull,2.5):.4f},{np.percentile(Dnull,97.5):.4f}]; "
          f"one-sided p = {p_D_upper:.4g}")

    out = {
        "_meta": {
            **evidence_scope_metadata(ledger_root, len(seed_stats)),
            "source_seed_count": len(seeds),
            "analysis_kind": "conditional_observed_state_feature_group_masking",
            "legacy_filename_notice": (
                "The configured artifact filename predates the current estimand label."
            ),
            "interpretation_limit": (
                "Retrieval results and guards are reused from the observed execution; "
                "the estimates cannot represent disabled communication channels."
            ),
        },
        "n_changed": n, "n_seeds": len(seed_stats), "scenarios": PERT,
        "conditional_distinctness_index": D,
        "bootstrap_ci": [float(lo), float(hi)], "bootstrap_ci_above_0p5": bool(lo > 0.5),
        "cells": {
            "mcp_feature_group_only": mcp_only,
            "pirag_feature_group_only": pir_only,
            "both_feature_groups": both,
            "neither_feature_group": neither,
        },
        "inferential_unit": "seed", "per_seed": seed_stats,
        "phi_feature_group_mask_effects": phi_obs,
        "phi_cluster_signflip_p_two_sided": float(p_phi_two),
        "distinctness_independence_baseline": E_D_indep,
        "distinctness_minus_independence": d_delta_obs,
        "distinctness_cluster_signflip_p_upper": float(p_D_upper),
        "n_perm": B,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2))
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
