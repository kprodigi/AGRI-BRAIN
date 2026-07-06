#!/usr/bin/env python3
r"""Specification-curve / multiverse robustness of the §5.8 channel analysis.

Pure re-analysis of the instrumented agribrain decision ledgers
(decision_ledger_h2, four perturbed scenarios, 20 seeds) -- no simulation, no
sim-code change. It re-derives the §5.8 headline conclusions under every
reasonable setting of the analysis's discretionary choices and shows they do
not depend on those choices:

  * complementarity (non-redundancy) index  > 0.5
  * emergent synergy rate                    > 0
  * necessity association  phi(mcp_nec, pir_nec) over changed decisions  > 0
  * conditional context-decisive rate  P(flip | context active) ~ 0.42

Analytic degrees of freedom swept (one at a time, others held at default):
  1. context-active threshold ``active_atol`` -- how small |modifier| counts as
     "inactive" (the denominator of the conditional rate).
  2. governance ceiling ``_CEIL`` and advantage ``_ADV`` -- the override
     thresholds in the decision reconstruction.
  3. compliance-event cutoff -- psi_0 > c defining an "MCP-governed" decision.
  4. bootstrap B -- resamples for the complementarity seed-cluster CI.

If the headline signs/inequalities hold across the whole grid, the conclusions
are not an artifact of analytic choices. Writes channel_spec_curve.json.

Run::
    python mvp/simulation/_h2_spec_curve.py --ledger-root mvp/simulation/results/decision_ledger_h2
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "benchmarks"))
from aggregate_channel_attribution import _walk, _load_episode  # noqa: E402

# Defaults (match the canonical aggregator / permutation test).
DEF_CEIL, DEF_ADV = 0.005, 0.80
DEF_ATOL = 1e-9            # |cm| above this on any element => context "active"
DEF_COMP_CUT = 0.0        # psi_0 > cut => compliance-relevant (>1e-9 in aggregator)
PERTURBED = ("heatwave", "overproduction", "cyber_outage", "adaptive_pricing")
_RNG = np.random.default_rng(20260605)


def _probs(base, m, ss, amp, T):
    m = np.asarray(m, float)
    slca = np.asarray(ss, float) * (amp * min(abs(m[1]), 1.0))
    z = (np.asarray(base, float) + m + slca) / T
    e = np.exp(z - z.max())
    return e / e.sum()


def _dec(base, m, ss, amp, T, governed, ceil, adv):
    p = _probs(base, m, ss, amp, T)
    if governed and p[0] < ceil and (p[1] - p[0]) > adv:
        return 1
    return int(np.argmax(p))


def _load(ledger_root: Path):
    """Return per-seed lists of usable step records for agribrain perturbed."""
    by_cell, seeds = _walk(ledger_root)
    per_seed: dict[int, list] = {}
    for (scenario, mode), paths in by_cell.items():
        if mode != "agribrain" or scenario not in PERTURBED:
            continue
        for seed_n, path in paths:
            for r in _load_episode(path):
                if any(r.get(k) is None for k in
                       ("base_logits", "context_modifier", "slca_shaping",
                        "slca_amp", "policy_temperature", "modifier_mcp",
                        "modifier_pirag")):
                    continue
                if float(r["policy_temperature"]) <= 0:
                    continue
                per_seed.setdefault(seed_n, []).append(r)
    return per_seed


def _metrics(per_seed, ceil, adv, atol, comp_cut):
    """Compute the headline metrics over the pooled agribrain perturbed set."""
    n_changed = 0
    cat = {"mcp_only": 0, "pir_only": 0, "redundant": 0, "synergy": 0}
    n_dec = 0
    n_active = 0
    n_dec_active = 0
    n_synergy = 0
    n_comp = 0
    n_mcp_nec_comp = 0
    mcp_nec_changed: list[int] = []
    pir_nec_changed: list[int] = []
    per_seed_compl = {}  # seed -> (changed, nonredundant) for cluster bootstrap
    for seed, recs in per_seed.items():
        s_changed = s_nonredund = 0
        for r in recs:
            base = r["base_logits"]; cm = r["context_modifier"]
            ss = r["slca_shaping"]; amp = r["slca_amp"]; T = float(r["policy_temperature"])
            mm = r["modifier_mcp"]; mp = r["modifier_pirag"]; psi = r.get("psi")
            d_none = _dec(base, np.zeros(3), ss, amp, T, False, ceil, adv)
            d_full = _dec(base, cm, ss, amp, T, True, ceil, adv)
            d_mcp = _dec(base, mm, ss, amp, T, True, ceil, adv)
            d_pir = _dec(base, mp, ss, amp, T, True, ceil, adv)
            n_dec += 1
            decisive = d_full != d_none
            active = float(np.max(np.abs(np.asarray(cm, float)))) > atol
            if active:
                n_active += 1
                if decisive:
                    n_dec_active += 1
            comp_on = psi is not None and float(np.asarray(psi, float)[0]) > comp_cut
            if comp_on:
                n_comp += 1
                if d_full != d_pir:
                    n_mcp_nec_comp += 1
            if d_mcp == d_none and d_pir == d_none and decisive:
                n_synergy += 1
            if decisive:
                n_changed += 1
                s_changed += 1
                mcp_suff = (d_mcp == d_full)
                pir_suff = (d_pir == d_full)
                if mcp_suff and pir_suff:
                    cat["redundant"] += 1
                elif mcp_suff:
                    cat["mcp_only"] += 1; s_nonredund += 1
                elif pir_suff:
                    cat["pir_only"] += 1; s_nonredund += 1
                else:
                    cat["synergy"] += 1; s_nonredund += 1
                mcp_nec_changed.append(int(d_full != d_pir))
                pir_nec_changed.append(int(d_full != d_mcp))
        per_seed_compl[seed] = (s_changed, s_nonredund)
    compl = ((cat["mcp_only"] + cat["pir_only"] + cat["synergy"]) / n_changed
             if n_changed else 0.0)
    M = np.array(mcp_nec_changed); P = np.array(pir_nec_changed)
    if M.size and len(set(M.tolist())) > 1 and len(set(P.tolist())) > 1:
        phi = float(np.corrcoef(M.astype(float), P.astype(float))[0, 1])
    else:
        phi = 0.0
    return {
        "complementarity_index": compl,
        "synergy_rate": (n_synergy / n_dec if n_dec else 0.0),
        "phi_changed": phi,
        "context_decisive_rate": (n_changed / n_dec if n_dec else 0.0),
        "context_decisive_given_active": (n_dec_active / n_active if n_active else 0.0),
        "mcp_necessary_given_compliance": (n_mcp_nec_comp / n_comp if n_comp else 0.0),
        "n_decisions": n_dec, "n_changed": n_changed, "n_active": n_active,
        "_per_seed_compl": per_seed_compl,
    }


def _compl_ci(per_seed_compl, B):
    """Seed-cluster bootstrap CI for the complementarity index."""
    seeds = list(per_seed_compl)
    boots = []
    for _ in range(B):
        idx = _RNG.integers(0, len(seeds), len(seeds))
        ch = sum(per_seed_compl[seeds[i]][0] for i in idx)
        nr = sum(per_seed_compl[seeds[i]][1] for i in idx)
        if ch:
            boots.append(nr / ch)
    if not boots:
        return (0.0, 0.0)
    return tuple(float(x) for x in np.percentile(boots, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ledger-root", type=Path,
                    default=Path("mvp/simulation/results/decision_ledger_h2"))
    ap.add_argument("--output", type=Path,
                    default=Path("mvp/simulation/results/channel_spec_curve.json"))
    args = ap.parse_args()
    root = args.ledger_root.resolve()
    if not root.exists():
        raise SystemExit(f"ledger root not found: {root}")
    per_seed = _load(root)
    print(f"Loaded {sum(len(v) for v in per_seed.values())} agribrain perturbed "
          f"decisions over {len(per_seed)} seeds")

    HEAD = ("complementarity_index", "synergy_rate", "phi_changed",
            "context_decisive_given_active", "mcp_necessary_given_compliance")

    def strip(d):
        return {k: v for k, v in d.items() if not k.startswith("_")}

    base = _metrics(per_seed, DEF_CEIL, DEF_ADV, DEF_ATOL, DEF_COMP_CUT)
    print("\nDEFAULT:", {k: round(base[k], 4) for k in HEAD})

    grid = {
        "active_atol": [1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2],
        "_CEIL": [0.001, 0.0025, 0.005, 0.01, 0.02],
        "_ADV": [0.70, 0.75, 0.80, 0.85, 0.90],
        "compliance_cutoff": [0.0, 0.1, 0.25, 0.5],
    }
    sweeps = {}
    for dof, vals in grid.items():
        rows = []
        for v in vals:
            ceil = v if dof == "_CEIL" else DEF_CEIL
            adv = v if dof == "_ADV" else DEF_ADV
            atol = v if dof == "active_atol" else DEF_ATOL
            cut = v if dof == "compliance_cutoff" else DEF_COMP_CUT
            m = _metrics(per_seed, ceil, adv, atol, cut)
            rows.append({"value": v, **{k: round(strip(m)[k], 4) for k in HEAD}})
        sweeps[dof] = rows

    # bootstrap-B sweep on the complementarity CI (point estimate is B-invariant).
    b_rows = []
    for B in (1000, 2000, 5000, 10000):
        lo, hi = _compl_ci(base["_per_seed_compl"], B)
        b_rows.append({"B": B, "compl_ci_low": round(lo, 4), "compl_ci_high": round(hi, 4)})
    sweeps["bootstrap_B"] = b_rows

    # Stability summary: min/max of each headline metric across ALL single-DOF specs.
    allspecs = [r for dof in grid for r in sweeps[dof]]
    stability = {}
    for k in HEAD:
        vals = [r[k] for r in allspecs]
        stability[k] = {"min": min(vals), "max": max(vals),
                        "range": round(max(vals) - min(vals), 4)}
    # The qualitative conclusions that must hold everywhere.
    verdict = {
        "complementarity_gt_0.5": all(r["complementarity_index"] > 0.5 for r in allspecs),
        "synergy_gt_0": all(r["synergy_rate"] > 0 for r in allspecs),
        "phi_gt_0": all(r["phi_changed"] > 0 for r in allspecs),
        "conditional_decisive_gt_0.30": all(
            r["context_decisive_given_active"] > 0.30 for r in allspecs),
    }

    out = {
        "_meta": {
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            # As-given (repo-relative) path, not resolve(): the resolved
            # absolute path leaks local machine state into a committed
            # evidence artifact.
            "ledger_root": str(args.ledger_root),
            "n_seeds": len(per_seed),
            "defaults": {"_CEIL": DEF_CEIL, "_ADV": DEF_ADV,
                         "active_atol": DEF_ATOL, "compliance_cutoff": DEF_COMP_CUT},
            "headline_metrics": list(HEAD),
        },
        "default": {k: strip(base)[k] for k in
                    list(HEAD) + ["context_decisive_rate", "n_decisions",
                                  "n_changed", "n_active"]},
        "sweeps": sweeps,
        "stability_across_all_specs": stability,
        "conclusions_hold_everywhere": verdict,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"\nStability (min..max across all specs):")
    for k in HEAD:
        print(f"  {k:34s} {stability[k]['min']:.4f} .. {stability[k]['max']:.4f}")
    print(f"\nConclusions hold across EVERY spec: {all(verdict.values())}  {verdict}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
