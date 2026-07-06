#!/usr/bin/env python3
r"""Decision-level channel-attribution aggregator for the §5.8 H2 test.

This supersedes the earlier logit-shift channel aggregator (removed 2026-06),
which measured the *signed logit shift on the chosen action* and a
"super-additivity" fraction defined as ``|joint| > max(|mcp|,|piRAG|)``.
That framing had three defects a referee would catch immediately:

  1. The context layer is **linear-additive in logit space** -- the modifier
     is ``clip(tau * Theta_context @ psi)`` and the two channels own disjoint
     psi components, so ``modifier_mcp + modifier_piRAG == modifier_full``
     *by construction*. Super-additivity in logit space is therefore
     impossible, and the old metric's median ``joint - max(single)`` was
     negative in every scenario -- the data contradicted the claim.
  2. The MCP-channel median was reported over *all* steps, ~75 % of which sit
     outside the retrieval-guard window where the modifier is zero, so the
     median collapsed to 0.000 and made MCP look inert.
  3. The "joint Delta-z on the chosen action" was measured on the *endogenous*
     chosen action, which is exactly the action the modifier pushed toward,
     so it carried little information.

The right place to measure two channels' value is the **argmax (decision)
level**, where the softmax is non-linear: removing a channel can flip the
routing decision even though logits add linearly. This script reconstructs,
for every agribrain decision, the action the policy would have taken under
four context configurations, using the observer-only ingredients the
instrumented policy records in the ledger (``base_logits``, ``slca_shaping``,
``slca_amp``, ``policy_temperature``, ``modifier_mcp``, ``modifier_pirag``,
``context_modifier``):

    decision(m) = argmax softmax( (base_logits + m + slca_boost(m)) / T )
    slca_boost(m) = slca_shaping * (slca_amp * min(|m_LR|, 1))
    + the governance override (pi_CC < ceiling and pi_LR - pi_CC > advantage)

      d_none  : m = 0            (context layer ablated; no slca_boost, no override)
      d_mcp   : m = modifier_mcp   (piRAG dropped -- MCP channel only)
      d_pirag : m = modifier_pirag (MCP dropped   -- piRAG channel only)
      d_full  : m = context_modifier (both channels, as the policy ran)

From the four decisions per step it derives (per scenario x mode, pooled over
seeds, with a seed-cluster bootstrap 95 % CI on the headline rates):

  * context_decisive_rate  P(d_full != d_none)          -- context changes routing
  * mcp_necessary_rate      P(d_full != d_pirag)         -- dropping MCP changes it
  * pirag_necessary_rate    P(d_full != d_mcp)           -- dropping piRAG changes it
  * synergy_rate            P(d_mcp == d_none == d_pirag  -- emergent: neither channel
                              and d_full != d_none)          alone moves it, together they do
  * attribution of every context-changed decision into
        {mcp_sufficient_only, pirag_sufficient_only, redundant, synergy}
  * complementarity_index   share of context-changed decisions carried by a
                            single channel or by synergy (i.e. NOT redundant)
  * activation orthogonality (Jaccard / phi correlation of the two channels'
    psi-activation on the applied-modifier steps)
  * directional conflict    among applied steps with both channels active, the
                            share where the two channels' argmax push differs
  * conditional channel magnitude (median/IQR of |modifier| when the channel
    is active, fixing the "MCP median 0" dilution artefact)
  * outcome linkage         mean realised reward/waste/slca/rho on the
                            context-decisive decisions vs the rest.

Run::

    python mvp/simulation/benchmarks/aggregate_channel_attribution.py \
        --ledger-root mvp/simulation/results/decision_ledger_per_seed \
        --output mvp/simulation/results/channel_attribution_aggregate.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# Governance override thresholds (probability space). Imported from the live
# policy so this script stays in lockstep with action_selection.py; the
# literals are the documented fallbacks if the backend is not importable.
try:
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[3] / "agribrain" / "backend")
    )
    from src.models.action_selection import (  # type: ignore
        GOVERNANCE_CC_PROB_CEILING as _CEIL,
        GOVERNANCE_LOCAL_ADVANTAGE_MIN as _ADV,
    )
except Exception:  # pragma: no cover - defensive
    _CEIL, _ADV = 0.005, 0.80

MCP_PSI = (0, 1, 4)    # compliance severity, forecast urgency, recovery saturation
PIRAG_PSI = (2, 3)     # retrieval confidence, regulatory pressure
_RNG = np.random.default_rng(20260605)
_N_BOOT = 2000


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _probs(base, m, slca_shaping, amp, T):
    """Softmax decision distribution under modifier ``m`` (no governance).

    Shared by ``_decision`` (argmax + override) and the decision-movement
    concentration metric, which needs the continuous distribution rather
    than the discrete argmax.
    """
    m = np.asarray(m, dtype=float)
    slca_boost = np.asarray(slca_shaping, dtype=float) * (amp * min(abs(m[1]), 1.0))
    logits = (np.asarray(base, dtype=float) + m + slca_boost) / T
    return _softmax(logits)


def _decision(base, m, slca_shaping, amp, T, governed):
    """Reconstruct the policy's argmax under modifier ``m``.

    ``governed`` toggles the governance override (active whenever the policy
    runs with a context modifier, i.e. for every config except d_none).
    """
    p = _probs(base, m, slca_shaping, amp, T)
    if governed and p[0] < _CEIL and (p[1] - p[0]) > _ADV:
        return 1
    return int(np.argmax(p))


def _gini(x: np.ndarray) -> float:
    """Gini concentration coefficient for a non-negative 1-D array.

    0 = every decision moves equally; 1 = all movement concentrated in a
    single decision. Uses the sorted O(n log n) formula (no O(n^2) matrix).
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n == 0:
        return 0.0
    total = float(x.sum())
    if total <= 0.0:
        return 0.0
    cum = float(np.sum(np.cumsum(x)))
    return float((n + 1 - 2.0 * cum / total) / n)


def _ci(values, stat=np.mean):
    """Seed-cluster bootstrap 95 % CI for a per-seed list of arrays.

    ``values`` is a list (one per seed) of 1-D boolean/float arrays. The
    bootstrap resamples seeds with replacement and recomputes the pooled
    statistic, which respects the within-seed correlation of the 288-step
    trajectory (the honest unit of replication is the seed, not the step).
    """
    if not values:
        return (0.0, 0.0, 0.0)
    pooled = np.concatenate(values)
    point = float(stat(pooled)) if pooled.size else 0.0
    n = len(values)
    boots = []
    for _ in range(_N_BOOT):
        idx = _RNG.integers(0, n, n)
        sample = np.concatenate([values[i] for i in idx])
        if sample.size:
            boots.append(float(stat(sample)))
    if not boots:
        return (point, point, point)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (point, float(lo), float(hi))


def _stat_block(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return {"median": 0.0, "q25": 0.0, "q75": 0.0, "mean": 0.0, "n": 0}
    return {
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "mean": float(np.mean(arr)),
        "n": int(arr.size),
    }


def _aggregate_cell(per_seed_records):
    """per_seed_records: list over seeds of list-of-step-dicts (one episode)."""
    # Per-seed boolean arrays for the headline rates (for the cluster bootstrap)
    s_decisive, s_mcp_nec, s_pirag_nec, s_synergy = [], [], [], []
    # 1a: conditional context-decisive rate -- restricted to context-ACTIVE
    # decisions (modifier applied: retrieval guard passed AND the modifier
    # was non-negligible, i.e. context "had something to say"). This is the
    # honest P(flip | context active) rather than P(flip | any decision).
    s_decisive_active = []
    n_active = 0
    # 2a: MCP/piRAG necessity restricted to MCP-GOVERNED events (compliance
    # severity psi0 > 0) -- the population MCP is designed to act on.
    s_mcp_nec_comp, s_pirag_nec_comp = [], []
    # 1c: decision-movement concentration. Per step, the total-variation
    # distance between the context-on and context-off softmax decision
    # distributions (how much context moved the decision), pooled cell-wide.
    cell_moves, cell_decisive_move = [], []
    # Pooled accumulators
    cat = {"mcp_sufficient_only": 0, "pirag_sufficient_only": 0,
           "redundant": 0, "synergy": 0}
    _ATTR_KEYS = ("mcp_sufficient_only", "pirag_sufficient_only", "redundant", "synergy")
    # Per-seed category indicators over the context-changed decisions, so each
    # attribution fraction gets the same seed-cluster bootstrap 95% CI as the
    # headline rates (one decision is in exactly one category, summing to 1).
    s_attr = {k: [] for k in _ATTR_KEYS}
    n_changed = 0
    n_instrumented = 0
    n_applied = 0
    # activation (applied steps)
    act_mcp, act_pirag, act_both, act_either = 0, 0, 0, 0
    act_mcp_vec, act_pirag_vec = [], []   # for phi correlation
    conflict_both = 0
    conflict_diff = 0
    # MCP-exclusive safety/governance mechanisms (these are MCP's real value:
    # discrete verified interventions piRAG cannot produce).
    n_gov_override = 0
    n_compliance_active = 0          # MCP check_compliance flagged a violation (psi0>0)
    n_compliance_decisive = 0        # of those, the routing differs from no-context
    # conditional magnitudes
    mag_mcp_active, mag_pirag_active = [], []
    # outcome linkage accumulators (decisive vs not)
    out_keys = ("reward", "waste", "slca", "rho")
    out_decisive = {k: [] for k in out_keys}
    out_rest = {k: [] for k in out_keys}

    for steps in per_seed_records:
        d_dec, d_mcpn, d_pirn, d_syn = [], [], [], []
        d_attr = {k: [] for k in _ATTR_KEYS}
        d_dec_active = []            # 1a: decisive | context active (this seed)
        d_mcpn_comp, d_pirn_comp = [], []   # 2a: necessity | compliance event
        for r in steps:
            base = r.get("base_logits")
            cm = r.get("context_modifier")
            ss = r.get("slca_shaping")
            amp = r.get("slca_amp")
            T = r.get("policy_temperature")
            mm = r.get("modifier_mcp")
            mp = r.get("modifier_pirag")
            psi = r.get("psi")
            if (base is None or cm is None or ss is None or amp is None
                    or T is None or mm is None or mp is None):
                continue
            T = float(T)
            if T <= 0:
                continue
            n_instrumented += 1
            zero = np.zeros(3)
            d_none = _decision(base, zero, ss, amp, T, governed=False)
            d_full = _decision(base, cm, ss, amp, T, governed=True)
            d_mcp = _decision(base, mm, ss, amp, T, governed=True)
            d_pirag = _decision(base, mp, ss, amp, T, governed=True)

            decisive = d_full != d_none
            d_dec.append(decisive)
            d_mcpn.append(d_full != d_pirag)   # dropping MCP changes the decision
            d_pirn.append(d_full != d_mcp)     # dropping piRAG changes the decision
            synergy = (d_mcp == d_none and d_pirag == d_none and d_full != d_none)
            d_syn.append(synergy)

            # 1a: context-active gating (modifier applied = retrieval guard
            # passed AND the modifier was non-negligible). The honest
            # denominator for "how often does context flip a decision".
            applied_now = not np.allclose(np.asarray(cm, dtype=float), 0.0)
            if applied_now:
                n_active += 1
                d_dec_active.append(decisive)

            # 2a: MCP-governed event = compliance severity (psi0) > 0. On this
            # population, report how often dropping a channel flips routing.
            comp_on = (psi is not None
                       and float(np.asarray(psi, dtype=float)[0]) > 1e-9)
            if comp_on:
                d_mcpn_comp.append(d_full != d_pirag)
                d_pirn_comp.append(d_full != d_mcp)

            # 1c: decision movement = total-variation distance between the
            # context-on and context-off softmax decision distributions.
            p_none = _probs(base, zero, ss, amp, T)
            p_full = _probs(base, cm, ss, amp, T)
            cell_moves.append(0.5 * float(np.sum(np.abs(p_full - p_none))))
            cell_decisive_move.append(decisive)

            if decisive:
                n_changed += 1
                mcp_suff = (d_mcp == d_full)
                pir_suff = (d_pirag == d_full)
                if mcp_suff and pir_suff:
                    _hit = "redundant"
                elif mcp_suff and not pir_suff:
                    _hit = "mcp_sufficient_only"
                elif pir_suff and not mcp_suff:
                    _hit = "pirag_sufficient_only"
                else:
                    _hit = "synergy"
                cat[_hit] += 1
                for _k in _ATTR_KEYS:
                    d_attr[_k].append(1.0 if _k == _hit else 0.0)

            # MCP-exclusive governance / compliance interventions.
            if r.get("governance_override"):
                n_gov_override += 1
            if psi is not None and float(np.asarray(psi)[0]) > 1e-9:
                n_compliance_active += 1
                if decisive:
                    n_compliance_decisive += 1

            # outcome linkage
            for k in out_keys:
                v = r.get(k)
                if v is None:
                    continue
                (out_decisive if decisive else out_rest)[k].append(float(v))

            # activation / conflict / magnitude on applied-modifier steps
            applied = not np.allclose(cm, 0.0)
            if applied and psi is not None:
                n_applied += 1
                psi = np.asarray(psi, dtype=float)
                mcp_on = bool(np.any(np.abs(psi[list(MCP_PSI)]) > 1e-9))
                pir_on = bool(np.any(np.abs(psi[list(PIRAG_PSI)]) > 1e-9))
                act_mcp += int(mcp_on)
                act_pirag += int(pir_on)
                act_both += int(mcp_on and pir_on)
                act_either += int(mcp_on or pir_on)
                act_mcp_vec.append(int(mcp_on))
                act_pirag_vec.append(int(pir_on))
                if mcp_on:
                    mag_mcp_active.append(float(np.max(np.abs(mm))))
                if pir_on:
                    mag_pirag_active.append(float(np.max(np.abs(mp))))
                if mcp_on and pir_on:
                    conflict_both += 1
                    if int(np.argmax(mm)) != int(np.argmax(mp)):
                        conflict_diff += 1

        if d_dec:
            s_decisive.append(np.array(d_dec))
            s_mcp_nec.append(np.array(d_mcpn))
            s_pirag_nec.append(np.array(d_pirn))
            s_synergy.append(np.array(d_syn))
        for _k in _ATTR_KEYS:
            if d_attr[_k]:
                s_attr[_k].append(np.array(d_attr[_k]))
        if d_dec_active:
            s_decisive_active.append(np.array(d_dec_active))
        if d_mcpn_comp:
            s_mcp_nec_comp.append(np.array(d_mcpn_comp))
            s_pirag_nec_comp.append(np.array(d_pirn_comp))

    def rate_ci(per_seed):
        pt, lo, hi = _ci(per_seed, np.mean)
        return {"rate": pt, "ci_low": lo, "ci_high": hi}

    # phi (Pearson) correlation of the two activation indicators
    phi = 0.0
    if act_mcp_vec and len(set(act_mcp_vec)) > 1 and len(set(act_pirag_vec)) > 1:
        phi = float(np.corrcoef(act_mcp_vec, act_pirag_vec)[0, 1])

    def mean_or0(xs):
        return float(np.mean(xs)) if xs else 0.0

    # 1c: decision-movement concentration. Gini of the per-decision TV
    # movement, plus the share of total movement carried by the decisive
    # flips and by the top decile. A high concentration with a low overall
    # decisive rate means the layer acts sparingly but where it matters.
    moves_arr = np.asarray(cell_moves, dtype=float)
    dec_mask = np.asarray(cell_decisive_move, dtype=bool)
    total_move = float(moves_arr.sum())
    if moves_arr.size and total_move > 0.0:
        share_decisive = float(moves_arr[dec_mask].sum() / total_move)
        k_top = max(1, int(np.ceil(0.10 * moves_arr.size)))
        top_decile_share = float(np.sort(moves_arr)[::-1][:k_top].sum() / total_move)
    else:
        share_decisive, top_decile_share = 0.0, 0.0
    concentration = {
        "gini": _gini(moves_arr),
        "share_carried_by_decisive": share_decisive,
        "top_decile_share": top_decile_share,
        "mean_move": float(moves_arr.mean()) if moves_arr.size else 0.0,
        "n": int(moves_arr.size),
    }

    return {
        "n_instrumented_decisions": n_instrumented,
        "n_seeds": len(s_decisive),
        "context_decisive": rate_ci(s_decisive),
        # 1a: the honest conditional -- decisive among context-ACTIVE steps.
        "context_decisive_given_active": rate_ci(s_decisive_active),
        "n_context_active": n_active,
        "mcp_necessary": rate_ci(s_mcp_nec),
        "pirag_necessary": rate_ci(s_pirag_nec),
        # 2a: necessity conditioned on MCP-governed (compliance) events.
        "mcp_necessary_given_compliance": rate_ci(s_mcp_nec_comp),
        "pirag_necessary_given_compliance": rate_ci(s_pirag_nec_comp),
        "synergy": rate_ci(s_synergy),
        # 1c: where the context layer concentrates its decision influence.
        "decision_movement_concentration": concentration,
        # Private raw arrays for exact cross-scenario pooling; popped by
        # main() before JSON serialization (numpy is not JSON-native).
        "_raw_move": moves_arr,
        "_raw_decisive": dec_mask,
        "n_context_changed": n_changed,
        "attribution_counts": cat,
        "attribution_fraction": (
            {k: (v / n_changed if n_changed else 0.0) for k, v in cat.items()}
        ),
        "attribution_fraction_ci": {k: rate_ci(s_attr[k]) for k in _ATTR_KEYS},
        "complementarity_index": (
            (cat["mcp_sufficient_only"] + cat["pirag_sufficient_only"] + cat["synergy"])
            / n_changed if n_changed else 0.0
        ),
        "activation": {
            "n_applied_steps": n_applied,
            "p_mcp": (act_mcp / n_applied if n_applied else 0.0),
            "p_pirag": (act_pirag / n_applied if n_applied else 0.0),
            "p_both": (act_both / n_applied if n_applied else 0.0),
            "p_either": (act_either / n_applied if n_applied else 0.0),
            "jaccard": (act_both / act_either if act_either else 0.0),
            "phi_correlation": phi,
        },
        "directional_conflict": {
            "n_both_active": conflict_both,
            "rate_channels_disagree": (
                conflict_diff / conflict_both if conflict_both else 0.0
            ),
        },
        "mcp_governance": {
            "governance_override_rate": (
                n_gov_override / n_instrumented if n_instrumented else 0.0
            ),
            "n_governance_override": n_gov_override,
            "compliance_active_rate": (
                n_compliance_active / n_instrumented if n_instrumented else 0.0
            ),
            "n_compliance_active": n_compliance_active,
            "compliance_decisive_rate": (
                n_compliance_decisive / n_compliance_active
                if n_compliance_active else 0.0
            ),
        },
        "conditional_magnitude": {
            "mcp_when_active": _stat_block(mag_mcp_active),
            "pirag_when_active": _stat_block(mag_pirag_active),
        },
        "outcome_linkage": {
            k: {
                "decisive_mean": mean_or0(out_decisive[k]),
                "rest_mean": mean_or0(out_rest[k]),
                "delta": mean_or0(out_decisive[k]) - mean_or0(out_rest[k]),
                "n_decisive": len(out_decisive[k]),
                "n_rest": len(out_rest[k]),
            }
            for k in out_keys
        },
    }


def _walk(ledger_root: Path):
    by_cell = {}
    seeds = set()
    for seed_dir in sorted(ledger_root.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        try:
            seed_n = int(seed_dir.name.split("_")[-1])
        except ValueError:
            continue
        seeds.add(seed_n)
        for jsonl in seed_dir.glob("*.jsonl"):
            stem = jsonl.stem
            if "__" not in stem:
                continue
            mode, scenario = stem.split("__", 1)
            by_cell.setdefault((scenario, mode), []).append((seed_n, jsonl))
    return by_cell, sorted(seeds)


def _load_episode(path: Path):
    out = []
    for i, line in enumerate(path.open()):
        if i == 0:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ledger-root", type=Path,
                    default=Path("mvp/simulation/results/decision_ledger_per_seed"))
    ap.add_argument("--output", type=Path,
                    default=Path("mvp/simulation/results/channel_attribution_aggregate.json"))
    ap.add_argument("--modes", type=str, default="agribrain",
                    help="Comma-separated modes to aggregate (default: agribrain).")
    ap.add_argument("--scenarios", type=str, default=None)
    args = ap.parse_args()

    root = args.ledger_root.resolve()
    if not root.exists():
        print(f"ERROR: ledger root not found: {root}")
        sys.exit(1)
    modes_filter = set(args.modes.split(",")) if args.modes else None
    scen_filter = set(args.scenarios.split(",")) if args.scenarios else None

    by_cell, seeds = _walk(root)
    print(f"Found {len(seeds)} seeds, {len(by_cell)} (scenario, mode) cells")

    out = {
        "_meta": {
            "generated_at": _dt.datetime.utcnow().isoformat() + "Z",
            "git_commit": _git_commit(),
            # As-given (repo-relative) path, not resolve(): the resolved
            # absolute path leaks local machine state into a committed
            # evidence artifact.
            "ledger_root": str(args.ledger_root),
            "n_seeds": len(seeds),
            "seeds": seeds,
            "governance_ceiling": _CEIL,
            "governance_advantage": _ADV,
            "n_bootstrap": _N_BOOT,
            "mcp_psi_indices": list(MCP_PSI),
            "pirag_psi_indices": list(PIRAG_PSI),
        },
        "by_scenario_mode": {},
    }

    scen_seen = set()
    for (scenario, mode), paths in sorted(by_cell.items()):
        if modes_filter and mode not in modes_filter:
            continue
        if scen_filter and scenario not in scen_filter:
            continue
        per_seed = [_load_episode(p) for _, p in sorted(paths)]
        # only keep seeds that actually carry instrumentation
        per_seed = [s for s in per_seed if s]
        if not per_seed:
            continue
        print(f"  {scenario}/{mode}: {len(per_seed)} seeds")
        cell = _aggregate_cell(per_seed)
        if cell["n_instrumented_decisions"] == 0:
            print(f"    (no instrumented decisions -- skipping {scenario}/{mode})")
            continue
        out["by_scenario_mode"].setdefault(scenario, {})[mode] = cell
        scen_seen.add(scenario)

    # Perturbed-pooled summary for agribrain (the 4 non-baseline scenarios).
    perturbed = [s for s in ("heatwave", "overproduction", "cyber_outage",
                             "adaptive_pricing") if s in out["by_scenario_mode"]]
    if perturbed and all("agribrain" in out["by_scenario_mode"][s] for s in perturbed):
        out["agribrain_perturbed_pooled"] = _pool_summary(
            [out["by_scenario_mode"][s]["agribrain"] for s in perturbed], perturbed
        )

    # Drop the private raw-movement arrays (numpy; not JSON-native) now that
    # _pool_summary has consumed them for the exact pooled concentration.
    for scen_cells in out["by_scenario_mode"].values():
        for cell in scen_cells.values():
            cell.pop("_raw_move", None)
            cell.pop("_raw_decisive", None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"Saved: {args.output}")


def _pool_summary(cells, scenarios):
    """n-weighted pooled headline numbers across scenarios (point estimates;
    CIs are reported per-scenario where the seed-cluster bootstrap is valid)."""
    tot_changed = sum(c["n_context_changed"] for c in cells)
    tot_instr = sum(c["n_instrumented_decisions"] for c in cells)
    cat = {k: sum(c["attribution_counts"][k] for c in cells)
           for k in ("mcp_sufficient_only", "pirag_sufficient_only",
                     "redundant", "synergy")}

    def _dig(node, key_path):
        for k in key_path:
            node = node[k]
        return node

    def wmean(key_path, weight_path=("n_instrumented_decisions",)):
        """Weighted mean of a per-cell rate, weighted by an arbitrary
        per-cell count (default n_instrumented). Conditional rates must be
        weighted by their own conditional denominator, not by all decisions."""
        num = 0.0
        den = 0.0
        for c in cells:
            w = float(_dig(c, weight_path))
            num += _dig(c, key_path) * w
            den += w
        return num / den if den else 0.0

    # 1c: exact pooled concentration -- concatenate the raw per-decision
    # movement across scenarios (the private arrays attached by
    # _aggregate_cell) so Gini / top-decile pool exactly rather than being
    # averaged from per-scenario summaries.
    raw_moves = (np.concatenate([c["_raw_move"] for c in cells])
                 if cells else np.array([], dtype=float))
    raw_dec = (np.concatenate([c["_raw_decisive"] for c in cells])
               if cells else np.array([], dtype=bool))
    tot_move = float(raw_moves.sum())
    if raw_moves.size and tot_move > 0.0:
        pooled_share_decisive = float(raw_moves[raw_dec].sum() / tot_move)
        k_top = max(1, int(np.ceil(0.10 * raw_moves.size)))
        pooled_top_decile = float(np.sort(raw_moves)[::-1][:k_top].sum() / tot_move)
    else:
        pooled_share_decisive, pooled_top_decile = 0.0, 0.0

    tot_active = sum(c.get("n_context_active", 0) for c in cells)
    tot_comp = sum(c["mcp_governance"]["n_compliance_active"] for c in cells)

    return {
        "scenarios": scenarios,
        "n_instrumented_decisions": tot_instr,
        "n_context_changed": tot_changed,
        "context_decisive_rate": wmean(["context_decisive", "rate"]),
        # 1a: pooled P(flip | context active), weighted by active-step count.
        "n_context_active": tot_active,
        "context_decisive_given_active_rate": wmean(
            ["context_decisive_given_active", "rate"], ("n_context_active",)),
        "mcp_necessary_rate": wmean(["mcp_necessary", "rate"]),
        "pirag_necessary_rate": wmean(["pirag_necessary", "rate"]),
        # 2a: pooled necessity | compliance event, weighted by compliance count.
        "n_compliance_active": tot_comp,
        "mcp_necessary_given_compliance_rate": wmean(
            ["mcp_necessary_given_compliance", "rate"],
            ("mcp_governance", "n_compliance_active")),
        "pirag_necessary_given_compliance_rate": wmean(
            ["pirag_necessary_given_compliance", "rate"],
            ("mcp_governance", "n_compliance_active")),
        "synergy_rate": wmean(["synergy", "rate"]),
        "attribution_counts": cat,
        "attribution_fraction": (
            {k: (v / tot_changed if tot_changed else 0.0) for k, v in cat.items()}
        ),
        "complementarity_index": (
            (cat["mcp_sufficient_only"] + cat["pirag_sufficient_only"] + cat["synergy"])
            / tot_changed if tot_changed else 0.0
        ),
        # 1c: pooled decision-movement concentration.
        "decision_movement_concentration": {
            "gini": _gini(raw_moves),
            "share_carried_by_decisive": pooled_share_decisive,
            "top_decile_share": pooled_top_decile,
            "mean_move": float(raw_moves.mean()) if raw_moves.size else 0.0,
            "n": int(raw_moves.size),
        },
    }


if __name__ == "__main__":
    main()
