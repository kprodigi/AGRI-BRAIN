#!/usr/bin/env python3
r"""Conditional observed-state feature-group analysis for H2.

This supersedes the earlier logit-shift channel aggregator (removed 2026-06),
which measured the *signed logit shift on the chosen action* and a
"super-additivity" fraction defined as ``|joint| > max(|mcp|,|piRAG|)``.
That framing had three defects a referee would catch immediately:

  1. Before the total cap, the context layer is linear-additive across a
     persistent MCP term and a separately gated piRAG term. The post-sum clip
     makes independently masked recomputations non-additive at saturated rows,
     so they must be interpreted as policy-surface diagnostics rather than
     decomposed causal effects. Super-additivity is not a model parameter.
  2. Retrieval guards regulate only piRAG-derived evidence. MCP operating-envelope,
     modeled-forecast, and history terms can remain active when retrieval is withheld,
     so whole-layer activity must not be inferred from the retrieval gate.
  3. The "joint Delta-z on the chosen action" was measured on the *endogenous*
     chosen action, which is exactly the action the modifier pushed toward,
     so it carried little information.

This script describes nonlinear **argmax sensitivity** to algebraically masking
two feature groups in the context vector. It does not remove or redispatch a
communication channel: retrieval was generated from the observed MCP results,
the same retrieval guard is reused, and all other recorded state is held fixed.
Accordingly, its estimates describe only the recorded policy surface; they do
not estimate effects of disabling transport, tool, or retrieval channels. Genuine channel-arm
comparisons are performed separately with the ``mcp_only``, ``pirag_only``, and
``no_context`` experimental modes.

For every agribrain decision, the script reconstructs the modal action under
four feature configurations, using the observer-only ingredients the
instrumented policy records in the ledger (``base_logits``, ``slca_shaping``,
``slca_amp``, ``policy_temperature``, ``modifier_mcp``, ``modifier_pirag``,
``context_modifier``):

    decision(m) = argmax softmax( (base_logits + m + slca_boost(m)) / T )
    slca_boost(m) = slca_shaping * (slca_amp * min(|m_LR|, 1))
    + the author-declared probability-gap override (pi_CC < ceiling and pi_LR - pi_CC > advantage)

      d_zero : m = 0 (algebraically zeroed context modifier)
      d_mcp_features : m = modifier_mcp (MCP-derived feature group retained)
      d_pirag_features : m = modifier_pirag (retrieval-derived group retained)
      d_observed : m = context_modifier (the observed full modifier)

From the four decisions per step it derives (per scenario x mode, pooled over
seeds, with a seed-cluster bootstrap 95 % CI on the headline rates):

  * context_route_change: P(d_observed != d_zero)
  * mcp_feature_group_mask_effect: P(d_observed != d_pirag_features)
  * pirag_feature_group_mask_effect: P(d_observed != d_mcp_features)
  * joint_only_route_change: neither single feature group changes the zeroed
    modal route, whereas their observed joint modifier does
  * partition of context-changed decisions by whether each single feature
    group reproduces the observed modal route
  * conditional_distinctness_index: share of changed modal routes not
    reproduced by both single feature groups
  * activation overlap (Jaccard / phi correlation of the two feature groups'
    psi-activation on the applied-modifier steps)
  * directional conflict among applied steps with both feature groups active
  * conditional feature-group magnitude (median/IQR of |modifier| when active)
  * outcome linkage: mean realised reward/waste/slca/rho on modal-route
    changes versus the remaining recorded decisions.

Run::

    python mvp/simulation/benchmarks/aggregate_channel_attribution.py \
        --ledger-root mvp/simulation/results/decision_ledger_per_seed/<RUN_TAG> \
        --output mvp/simulation/results/channel_attribution_aggregate.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Probability-gap override thresholds (probability space). Imported from the live
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

MCP_PSI = (0, 1, 4)    # envelope exceedance, forecast signal, recovery-history signal
PIRAG_PSI = (2, 3)     # retrieval-score signal, retrieved-policy signal
_RNG = np.random.default_rng(20260605)
_N_BOOT = 2000

EPISODE_SCOPE = "final episode per scenario-mode-seed arm"
DECISION_HISTORY_SCOPE = "earlier decisions in the same episode only"


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

    ``governed`` toggles the probability-gap override (active whenever the policy
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
    # Per-seed boolean arrays for the conditional masking rates.
    s_route_change, s_mcp_mask, s_pirag_mask, s_joint_only = [], [], [], []
    # Conditional modal-route-change rate restricted to context-active
    # decisions (combined modifier was non-negligible, i.e. at least one
    # context channel "had something to say"). This is the
    # honest P(flip | context active) rather than P(flip | any decision).
    s_decisive_active = []
    n_active = 0
    # Conditional feature-group masking when the operating-envelope feature is active.
    s_mcp_mask_comp, s_pirag_mask_comp = [], []
    # 1c: decision-movement concentration. Per step, the total-variation
    # distance between the context-on and context-off softmax decision
    # distributions (how much context moved the decision), pooled cell-wide.
    cell_moves, cell_decisive_move = [], []
    # Pooled accumulators
    cat = {
        "mcp_group_matches_observed_only": 0,
        "pirag_group_matches_observed_only": 0,
        "both_groups_match_observed": 0,
        "neither_group_matches_observed": 0,
    }
    _ATTR_KEYS = tuple(cat)
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
    # MCP-exclusive author-declared policy mechanisms: operating-envelope
    # checks and recorded probability-gap interventions not produced by the
    # retrieval channel. Their empirical contribution is estimated downstream.
    n_gov_override = 0
    # Variable names below are retained as legacy artifact-schema keys.
    n_compliance_active = 0          # operating-envelope feature active (psi0>0)
    n_compliance_decisive = 0        # among those, routing differs from no-context
    # conditional magnitudes
    mag_mcp_active, mag_pirag_active = [], []
    # outcome linkage accumulators (decisive vs not)
    out_keys = ("reward", "waste", "slca", "rho")
    out_decisive = {k: [] for k in out_keys}
    out_rest = {k: [] for k in out_keys}

    for steps in per_seed_records:
        d_change, d_mcp_mask, d_pirag_mask, d_joint_only = [], [], [], []
        d_attr = {k: [] for k in _ATTR_KEYS}
        d_dec_active = []            # 1a: decisive | context active (this seed)
        d_mcp_mask_comp, d_pirag_mask_comp = [], []
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
            d_zero = _decision(base, zero, ss, amp, T, governed=False)
            d_observed = _decision(base, cm, ss, amp, T, governed=True)
            d_mcp_features = _decision(base, mm, ss, amp, T, governed=True)
            d_pirag_features = _decision(base, mp, ss, amp, T, governed=True)

            route_changed = d_observed != d_zero
            d_change.append(route_changed)
            d_mcp_mask.append(d_observed != d_pirag_features)
            d_pirag_mask.append(d_observed != d_mcp_features)
            joint_only = (
                d_mcp_features == d_zero
                and d_pirag_features == d_zero
                and d_observed != d_zero
            )
            d_joint_only.append(joint_only)

            # 1a: context-active gating (the combined modifier was
            # non-negligible). Retrieval failure alone is not an inactivity
            # condition because the MCP term remains independent. The honest
            # denominator for "how often does context flip a decision".
            applied_now = not np.allclose(np.asarray(cm, dtype=float), 0.0)
            if applied_now:
                n_active += 1
                d_dec_active.append(route_changed)

            # 2a: MCP operating-envelope feature (psi0) > 0. On this
            # population, report how often dropping a channel flips routing.
            comp_on = (psi is not None
                       and float(np.asarray(psi, dtype=float)[0]) > 1e-9)
            if comp_on:
                d_mcp_mask_comp.append(d_observed != d_pirag_features)
                d_pirag_mask_comp.append(d_observed != d_mcp_features)

            # 1c: decision movement = total-variation distance between the
            # context-on and context-off softmax decision distributions.
            p_none = _probs(base, zero, ss, amp, T)
            p_full = _probs(base, cm, ss, amp, T)
            cell_moves.append(0.5 * float(np.sum(np.abs(p_full - p_none))))
            cell_decisive_move.append(route_changed)

            if route_changed:
                n_changed += 1
                mcp_matches = (d_mcp_features == d_observed)
                pirag_matches = (d_pirag_features == d_observed)
                if mcp_matches and pirag_matches:
                    _hit = "both_groups_match_observed"
                elif mcp_matches and not pirag_matches:
                    _hit = "mcp_group_matches_observed_only"
                elif pirag_matches and not mcp_matches:
                    _hit = "pirag_group_matches_observed_only"
                else:
                    _hit = "neither_group_matches_observed"
                cat[_hit] += 1
                for _k in _ATTR_KEYS:
                    d_attr[_k].append(1.0 if _k == _hit else 0.0)

            # MCP-side probability-gap-rule and operating-envelope activations.
            if r.get("governance_override"):
                n_gov_override += 1
            if psi is not None and float(np.asarray(psi)[0]) > 1e-9:
                n_compliance_active += 1
                if route_changed:
                    n_compliance_decisive += 1

            # outcome linkage
            for k in out_keys:
                v = r.get(k)
                if v is None:
                    continue
                (out_decisive if route_changed else out_rest)[k].append(float(v))

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

        if d_change:
            s_route_change.append(np.array(d_change))
            s_mcp_mask.append(np.array(d_mcp_mask))
            s_pirag_mask.append(np.array(d_pirag_mask))
            s_joint_only.append(np.array(d_joint_only))
        for _k in _ATTR_KEYS:
            if d_attr[_k]:
                s_attr[_k].append(np.array(d_attr[_k]))
        if d_dec_active:
            s_decisive_active.append(np.array(d_dec_active))
        if d_mcp_mask_comp:
            s_mcp_mask_comp.append(np.array(d_mcp_mask_comp))
            s_pirag_mask_comp.append(np.array(d_pirag_mask_comp))

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
        "n_seeds": len(s_route_change),
        "context_route_change": rate_ci(s_route_change),
        # Conditional modal-route change among context-active steps.
        "context_route_change_given_active": rate_ci(s_decisive_active),
        "n_context_active": n_active,
        "mcp_feature_group_mask_effect": rate_ci(s_mcp_mask),
        "pirag_feature_group_mask_effect": rate_ci(s_pirag_mask),
        "mcp_feature_group_mask_effect_given_compliance": rate_ci(s_mcp_mask_comp),
        "pirag_feature_group_mask_effect_given_compliance": rate_ci(s_pirag_mask_comp),
        "joint_only_route_change": rate_ci(s_joint_only),
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
        "conditional_distinctness_index": (
            (cat["mcp_group_matches_observed_only"]
             + cat["pirag_group_matches_observed_only"]
             + cat["neither_group_matches_observed"])
            / n_changed if n_changed else 0.0
        ),
        "activation": {
            "n_applied_steps": n_applied,
            "p_mcp_feature_group": (act_mcp / n_applied if n_applied else 0.0),
            "p_pirag_feature_group": (act_pirag / n_applied if n_applied else 0.0),
            "p_both_feature_groups": (act_both / n_applied if n_applied else 0.0),
            "p_either_feature_group": (act_either / n_applied if n_applied else 0.0),
            "jaccard": (act_both / act_either if act_either else 0.0),
            "phi_correlation": phi,
        },
        "directional_conflict": {
            "n_both_active": conflict_both,
            "rate_feature_groups_disagree": (
                conflict_diff / conflict_both if conflict_both else 0.0
            ),
        },
        "mcp_governance": {
            "legacy_schema_note": (
                "legacy key names; values describe the probability-gap rule "
                "and synthetic operating-envelope feature, not governance or compliance determinations"
            ),
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
            "mcp_feature_group_when_active": _stat_block(mag_mcp_active),
            "pirag_feature_group_when_active": _stat_block(mag_pirag_active),
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
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parents[3],
        ).stdout.strip()
        return commit if re.fullmatch(r"[0-9a-f]{40}", commit) else None
    except Exception:
        return None


def evidence_scope_metadata(ledger_root: Path | str, seed_count: int) -> dict:
    """Return the shared provenance/scope contract for ledger-derived evidence.

    Publication jobs export the full source commit and run tag. Local analysis
    falls back to the checked-out Git commit and records a null run tag rather
    than inventing one. A present but malformed/mismatched environment stamp is
    rejected because silently labelling evidence with the wrong run is worse
    than failing the aggregation.
    """
    env_commit = os.environ.get("AGRIBRAIN_GIT_COMMIT", "").strip()
    if env_commit and not re.fullmatch(r"[0-9a-f]{40}", env_commit):
        raise RuntimeError(
            "AGRIBRAIN_GIT_COMMIT must be a full lowercase 40-character SHA-1"
        )
    head_commit = _git_commit()
    if env_commit and head_commit and env_commit != head_commit:
        # Publication-only recovery intentionally computes deterministic
        # derivatives at a clean repair commit while preserving the simulation
        # commit on raw-result metadata.  Accept that split only after the full
        # run-scoped recovery authorization has been validated.  A normal run
        # still fails on any env/HEAD mismatch exactly as before.
        try:
            from mvp.simulation.analysis.recovery_provenance import (
                recovery_context_from_environment,
            )

            recovery = recovery_context_from_environment(
                results_dir=_REPO_ROOT / "mvp" / "simulation" / "results",
                repo_root=_REPO_ROOT,
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                "AGRIBRAIN_GIT_COMMIT does not match the checked-out source "
                f"commit and recovery authorization is invalid: {exc}"
            ) from exc
        if (
            recovery is None
            or recovery.get("simulation_source_commit") != env_commit
            or recovery.get("publication_code_commit") != head_commit
        ):
            raise RuntimeError(
                "AGRIBRAIN_GIT_COMMIT does not match the checked-out source commit"
            )
    source_commit = env_commit or head_commit

    run_tag = os.environ.get("RUN_TAG", "").strip()
    artifact_run_tag = os.environ.get("ARTIFACT_RUN_TAG", "").strip()
    if run_tag and artifact_run_tag and run_tag != artifact_run_tag:
        raise RuntimeError("RUN_TAG and ARTIFACT_RUN_TAG identify different runs")
    run_tag = run_tag or artifact_run_tag or None

    seed_count = int(seed_count)
    if seed_count < 0:
        raise ValueError("seed_count must be non-negative")

    ledger_root_id = (
        ledger_root.as_posix()
        if isinstance(ledger_root, Path)
        else str(ledger_root).replace("\\", "/")
    )
    return {
        "source_commit": source_commit,
        "ledger_root": ledger_root_id,
        "seed_count": seed_count,
        "run_tag": run_tag,
        "episode_scope": EPISODE_SCOPE,
        "decision_history_scope": DECISION_HISTORY_SCOPE,
    }


def main(argv=None):
    global _RNG
    # A callable producer must be deterministic even when invoked twice in one
    # interpreter (tests and archive tooling need not always spawn a process).
    _RNG = np.random.default_rng(20260605)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ledger-root", type=Path,
                    default=Path("mvp/simulation/results/decision_ledger_per_seed"))
    ap.add_argument("--output", type=Path,
                    default=Path("mvp/simulation/results/channel_attribution_aggregate.json"))
    ap.add_argument("--modes", type=str, default="agribrain",
                    help="Comma-separated modes to aggregate (default: agribrain).")
    ap.add_argument("--scenarios", type=str, default=None)
    args = ap.parse_args(argv)

    root = args.ledger_root.resolve()
    if not root.exists():
        print(f"ERROR: ledger root not found: {root}")
        sys.exit(1)
    modes_filter = set(args.modes.split(",")) if args.modes else None
    scen_filter = set(args.scenarios.split(",")) if args.scenarios else None

    by_cell, seeds = _walk(root)
    print(f"Found {len(seeds)} seeds, {len(by_cell)} (scenario, mode) cells")

    scope_meta = evidence_scope_metadata(args.ledger_root, len(seeds))
    out = {
        "_meta": {
            # Compatibility aliases retained for existing figure consumers.
            "git_commit": scope_meta["source_commit"],
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
            "analysis_kind": "conditional_observed_state_feature_group_masking",
            "estimand": (
                "modal routing sensitivity to algebraically masking MCP-derived "
                "or retrieval-derived context features in the recorded state"
            ),
            "interpretation_limit": (
                "Retrieval and guards are reused from the observed full-context "
                "execution; this analysis cannot estimate effects of disabling "
                "transport, tool, or retrieval channels."
            ),
            **scope_meta,
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
    cat = {
        k: sum(c["attribution_counts"][k] for c in cells)
        for k in (
            "mcp_group_matches_observed_only",
            "pirag_group_matches_observed_only",
            "both_groups_match_observed",
            "neither_group_matches_observed",
        )
    }

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
        "context_route_change_rate": wmean(["context_route_change", "rate"]),
        # P(modal route change | context active), weighted by active-step count.
        "n_context_active": tot_active,
        "context_route_change_given_active_rate": wmean(
            ["context_route_change_given_active", "rate"], ("n_context_active",)),
        "mcp_feature_group_mask_effect_rate": wmean(
            ["mcp_feature_group_mask_effect", "rate"]
        ),
        "pirag_feature_group_mask_effect_rate": wmean(
            ["pirag_feature_group_mask_effect", "rate"]
        ),
        # Conditional feature-group masking on operating-envelope-feature-active
        # observations. Public field names remain legacy schema aliases.
        "n_compliance_active": tot_comp,
        "mcp_feature_group_mask_effect_given_compliance_rate": wmean(
            ["mcp_feature_group_mask_effect_given_compliance", "rate"],
            ("mcp_governance", "n_compliance_active")),
        "pirag_feature_group_mask_effect_given_compliance_rate": wmean(
            ["pirag_feature_group_mask_effect_given_compliance", "rate"],
            ("mcp_governance", "n_compliance_active")),
        "joint_only_route_change_rate": wmean(["joint_only_route_change", "rate"]),
        "attribution_counts": cat,
        "attribution_fraction": (
            {k: (v / tot_changed if tot_changed else 0.0) for k, v in cat.items()}
        ),
        "conditional_distinctness_index": (
            (cat["mcp_group_matches_observed_only"]
             + cat["pirag_group_matches_observed_only"]
             + cat["neither_group_matches_observed"])
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
