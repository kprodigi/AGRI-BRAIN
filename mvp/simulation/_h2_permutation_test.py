"""Permutation test + bootstrap CI for the §5.8 complementarity claim.

Question: among the agribrain decisions the integrated context layer changes
(d_full != d_none), are the decisions where MCP is *necessary* (dropping MCP
flips the routing) and where piRAG is *necessary* genuinely DIFFERENT decisions
(complementary) rather than the same decisions (redundant)?

For each changed decision we form the 2x2 of (mcp_nec, pir_nec):
  (1,1) synergy  (both needed)     (1,0) MCP-only     (0,1) piRAG-only
  (0,0) redundant (neither needed: each channel alone already reproduces d_full)

Complementarity index C = 1 - P(0,0) = share of changed decisions attributable
to a single channel or to synergy (i.e. NOT redundantly produced by both).

Two inferential checks, both reusing the exact decision reconstruction from
aggregate_channel_attribution.py (validated to reproduce the recorded softmax
probabilities to <=1e-6):

  1. Seed-cluster bootstrap 95% CI on C (resample the 20 seeds).
  2. Permutation test on the association between mcp_nec and pir_nec across the
     pooled changed decisions: shuffle pir_nec (preserving its marginal),
     B=10000 perms. Report (a) the phi-correlation of the two necessity
     indicators vs its permutation null (direction of association: <0 disjoint
     / complementary, ~0 independent, >0 co-firing), and (b) whether C is more
     extreme than its independence null.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "benchmarks"))
sys.path.insert(0, str(HERE.parents[1] / "agribrain" / "backend"))
from aggregate_channel_attribution import _decision, _walk, _load_episode  # noqa: E402

import argparse  # noqa: E402

PERT = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing"]
B = 10000
rng = np.random.default_rng(20260605)

_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_ap.add_argument("--ledger-root", type=Path,
                 default=HERE / "results" / "decision_ledger_h2",
                 help="seed_<N>/ root of instrumented agribrain ledgers "
                      "(HPC: pass mvp/simulation/results/decision_ledger_per_seed)")
_ap.add_argument("--output", type=Path,
                 default=HERE / "results" / "channel_complementarity_test.json")
_ARGS, _ = _ap.parse_known_args()
LEDGER = _ARGS.ledger_root


def build():
    by_cell, seeds = _walk(LEDGER)
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
                d_none = _decision(base, zero, ss, amp, T, False)
                d_full = _decision(base, cm, ss, amp, T, True)
                if d_full == d_none:
                    continue  # only context-changed decisions
                d_mcp = _decision(base, mm, ss, amp, T, True)
                d_pir = _decision(base, mp, ss, amp, T, True)
                L.append((d_full != d_pir, d_full != d_mcp))  # (mcp_nec, pir_nec)
    return per_seed


def phi(a, b):
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a.astype(float), b.astype(float))[0, 1])


def main():
    per_seed = build()
    seeds = sorted(per_seed)
    allp = [p for s in seeds for p in per_seed[s]]
    M = np.array([p[0] for p in allp], dtype=bool)
    P = np.array([p[1] for p in allp], dtype=bool)
    n = len(M)
    redundant = float(np.mean(~M & ~P))
    synergy = float(np.mean(M & P))
    mcp_only = float(np.mean(M & ~P))
    pir_only = float(np.mean(~M & P))
    C = 1.0 - redundant
    print(f"n changed decisions = {n:,}  ({len(seeds)} seeds, {PERT})")
    print(f"2x2 among changed:  MCP-only={mcp_only:.3f}  piRAG-only={pir_only:.3f}  "
          f"synergy={synergy:.3f}  redundant={redundant:.3f}")
    print(f"Complementarity index C = {C:.4f}")

    # (1) seed-cluster bootstrap CI on C
    boots = []
    for _ in range(B):
        idx = rng.integers(0, len(seeds), len(seeds))
        samp = [p for i in idx for p in per_seed[seeds[i]]]
        sm = np.array([x[0] for x in samp], dtype=bool)
        sp = np.array([x[1] for x in samp], dtype=bool)
        boots.append(1.0 - np.mean(~sm & ~sp))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  seed-cluster bootstrap 95% CI on C = [{lo:.4f}, {hi:.4f}]  "
          f"(all > 0.5: {lo > 0.5})")

    # (2) permutation test on association
    phi_obs = phi(M, P)
    E_C_indep = 1.0 - float(np.mean(~M)) * float(np.mean(~P))
    rng2 = np.random.default_rng(7)
    phinull = np.empty(B); Cnull = np.empty(B)
    for i in range(B):
        Pp = rng2.permutation(P)
        phinull[i] = phi(M, Pp)
        Cnull[i] = 1.0 - np.mean(~M & ~Pp)
    p_phi_two = (np.sum(np.abs(phinull) >= abs(phi_obs)) + 1) / (B + 1)
    p_C_upper = (np.sum(Cnull >= C) + 1) / (B + 1)
    print(f"  phi(mcp_nec, pir_nec) = {phi_obs:+.4f}  "
          f"(null mean {phinull.mean():+.4f}, two-sided p = {p_phi_two:.4g})")
    print(f"  C vs independence null: E[C_indep] = {E_C_indep:.4f}; "
          f"null mean {Cnull.mean():.4f} [{np.percentile(Cnull,2.5):.4f},{np.percentile(Cnull,97.5):.4f}]; "
          f"p(C >= obs) = {p_C_upper:.4g}")
    direction = ("DISJOINT / complementary" if phi_obs < 0 else
                 "co-firing" if phi_obs > 0 else "independent")
    print(f"  => necessity sets are {direction} "
          f"(observed C {C:.3f} vs independence baseline {E_C_indep:.3f})")

    out = {
        "n_changed": n, "n_seeds": len(seeds), "scenarios": PERT,
        "complementarity_index": C,
        "bootstrap_ci": [float(lo), float(hi)], "bootstrap_ci_above_0p5": bool(lo > 0.5),
        "cells": {"mcp_only": mcp_only, "pirag_only": pir_only,
                  "synergy": synergy, "redundant": redundant},
        "phi_mcp_pirag": phi_obs, "phi_perm_p_two_sided": float(p_phi_two),
        "C_independence_baseline": E_C_indep,
        "C_perm_p_upper": float(p_C_upper),
        "n_perm": B,
    }
    _ARGS.output.write_text(json.dumps(out, indent=2))
    print(f"Saved {_ARGS.output}")


if __name__ == "__main__":
    main()
