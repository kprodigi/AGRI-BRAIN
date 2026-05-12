#!/usr/bin/env python3
"""Cross-seed channel-attribution aggregator for the §5.8 H3 hypothesis test.

Reads every per-seed DecisionLedger JSONL file produced by the canonical
HPC run (post-2026-05 ``hpc/hpc_seed.sh`` writes them to a seed-isolated
directory, then ``hpc/hpc_aggregate.sh`` consolidates them under
``mvp/simulation/results/decision_ledger_per_seed/seed_<N>/``).

For each (scenario, mode) cell the aggregator computes the per-channel
logit-contribution statistics that the manuscript §5.8 reports as
evidence for H3 (component complementarity):

  - MCP-channel logit shift on the chosen action: the dot product of
    Θ_context[a, :] with ψ restricted to the MCP-derived components
    {ψ₀ = compliance severity, ψ₁ = forecast urgency, ψ₄ = recovery
    saturation}, with the piRAG-derived components masked to zero.

  - piRAG-channel logit shift on the chosen action: the same dot
    product with the mask inverted (ψ₂ = retrieval confidence, ψ₃ =
    regulatory pressure kept; the rest masked).

  - Joint (full ψ) shift on the chosen action: the canonical
    Θ_context @ ψ followed by the temporal modulation, clipped to
    [-1, +1] (same operation the policy applies at run time).

The aggregator emits cross-seed median, 25th-, 75th-percentile, and
mean of each of these three series per (scenario, mode), plus the
sub-additivity check ``joint > max(mcp_only, pirag_only)`` per step
and the fraction of steps where the integrated channel exceeds the
better single channel by at least 0.005 ARI-equivalent logit shift.

The output JSON has the shape:

    {
      "_meta": {
        "n_seeds": int,
        "seeds": [list],
        "n_scenarios": int,
        "scenarios": [list],
        "n_modes": int,
        "modes": [list],
        "generated_at": "ISO-8601 timestamp",
        "git_commit": str,
        "ledger_root": str (absolute path),
      },
      "by_scenario_mode": {
        <scenario>: {
          <mode>: {
            "n_records": int,                    # = n_seeds * 288 typically
            "mcp_channel_logit_shift": {
              "median": float, "q25": float, "q75": float, "mean": float,
              "n_negative": int, "n_positive": int,
            },
            "pirag_channel_logit_shift": {...same shape...},
            "joint_logit_shift":       {...same shape...},
            "feature_attribution": {
              "psi0_compliance":   {"median": float, "mean_abs": float},
              "psi1_forecast":     {...},
              "psi2_retrieval":    {...},
              "psi3_regulatory":   {...},
              "psi4_recovery_sat": {...},
            },
            "sub_additivity": {
              "n_steps_joint_exceeds_max_single": int,
              "fraction_joint_dominates":           float,
              "median_joint_minus_max_single":      float,
            },
            "governance_override_rate": float,
          }
        }
      }
    }

Run as::

    python mvp/simulation/benchmarks/aggregate_decision_ledgers.py \\
        --ledger-root mvp/simulation/results/decision_ledger_per_seed \\
        --output mvp/simulation/results/decision_ledger_aggregate.json

CLI flags:
  --ledger-root  directory containing seed_<N>/ subdirs of *.jsonl files
  --output       output JSON path
  --scenarios    optional comma-separated subset of scenarios
  --modes        optional comma-separated subset of modes
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# Try absolute then relative import so the script runs both from the
# repo root and from inside mvp/simulation/benchmarks/.
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "agribrain" / "backend"))
    from pirag.context_to_logits import THETA_CONTEXT  # type: ignore
except Exception:
    # Fallback: hard-code the canonical matrix from Eq. 16 of the manuscript.
    # This keeps the aggregator runnable even if the backend package is not
    # importable (e.g. when the HPC venv is not active). Kept in sync with
    # agribrain/backend/pirag/context_to_logits.py:55-60.
    THETA_CONTEXT = np.array([
        # psi0   psi1   psi2   psi3   psi4
        [-0.40, -0.30, -0.10, -0.15, +0.12],   # Cold Chain
        [+0.30, +0.25, +0.15, +0.18, +0.08],   # Local Redistribute
        [+0.15, +0.10, -0.05, +0.05, -0.20],   # Recovery
    ])


# Channel masks: which ψ components are produced by which channel.
# Component-to-source mapping per Table 2 of the manuscript:
#   psi0 compliance severity -> MCP (check_compliance tool)
#   psi1 forecast urgency    -> MCP (spoilage_forecast tool)
#   psi2 retrieval confidence -> piRAG (top-citation RRF score)
#   psi3 regulatory pressure  -> piRAG (top-doc-id regulatory pattern)
#   psi4 recovery saturation  -> MCP (chain_query / blockchain history)
MCP_MASK = np.array([1, 1, 0, 0, 1], dtype=float)
PIRAG_MASK = np.array([0, 0, 1, 1, 0], dtype=float)


def _stat_block(values: np.ndarray) -> dict:
    """Return median / q25 / q75 / mean plus simple sign counts."""
    if values.size == 0:
        return {
            "median": 0.0, "q25": 0.0, "q75": 0.0, "mean": 0.0,
            "n_negative": 0, "n_positive": 0,
        }
    return {
        "median": float(np.median(values)),
        "q25":    float(np.percentile(values, 25)),
        "q75":    float(np.percentile(values, 75)),
        "mean":   float(np.mean(values)),
        "n_negative": int(np.sum(values < 0)),
        "n_positive": int(np.sum(values > 0)),
    }


def _feature_stat(values: np.ndarray) -> dict:
    """Median and mean-absolute for a per-feature attribution series."""
    if values.size == 0:
        return {"median": 0.0, "mean_abs": 0.0}
    return {
        "median":   float(np.median(values)),
        "mean_abs": float(np.mean(np.abs(values))),
    }


def _walk_ledger(ledger_root: Path) -> dict:
    """Walk seed_<N>/ subdirs, yield per-seed (scenario, mode, jsonl_path)."""
    by_cell: dict = {}  # {(scenario, mode): [(seed, path), ...]}
    seeds_found: set = set()
    for seed_dir in sorted(ledger_root.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        try:
            seed_n = int(seed_dir.name.split("_")[-1])
        except ValueError:
            print(f"WARN: skipping non-integer seed dir {seed_dir}")
            continue
        seeds_found.add(seed_n)
        for jsonl in seed_dir.glob("*.jsonl"):
            stem = jsonl.stem  # "{mode}__{scenario}"
            if "__" not in stem:
                continue
            mode, scenario = stem.split("__", 1)
            by_cell.setdefault((scenario, mode), []).append((seed_n, jsonl))
    return by_cell, sorted(seeds_found)


def _aggregate_cell(jsonl_paths: list[tuple[int, Path]]) -> dict:
    """Compute per-channel logit-shift statistics for one (scenario, mode) cell."""
    mcp_shifts: list[float] = []
    pirag_shifts: list[float] = []
    joint_shifts: list[float] = []
    # Per-feature contributions to the CHOSEN action's logit (signed).
    per_feature: list[list[float]] = [[] for _ in range(5)]
    n_records = 0
    n_override = 0

    for seed_n, jsonl in jsonl_paths:
        with jsonl.open() as fh:
            for i, line in enumerate(fh):
                # First line is the header / metadata record.
                if i == 0:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                psi = rec.get("psi")
                a_idx = rec.get("action_idx")
                if psi is None or a_idx is None:
                    continue
                psi_arr = np.asarray(psi, dtype=float)
                if psi_arr.shape != (5,):
                    continue
                # MCP-only shift on chosen action.
                mcp_psi = psi_arr * MCP_MASK
                mcp_shift = float(THETA_CONTEXT[a_idx] @ mcp_psi)
                # piRAG-only shift on chosen action.
                pirag_psi = psi_arr * PIRAG_MASK
                pirag_shift = float(THETA_CONTEXT[a_idx] @ pirag_psi)
                # Joint shift = the actual Δz on chosen action that the
                # policy used at runtime (already includes τ_mod and clip).
                cm = rec.get("context_modifier")
                joint_shift = float(cm[a_idx]) if cm is not None else (
                    float(THETA_CONTEXT[a_idx] @ psi_arr)
                )
                mcp_shifts.append(mcp_shift)
                pirag_shifts.append(pirag_shift)
                joint_shifts.append(joint_shift)
                # Per-feature contributions to chosen action logit.
                for j in range(5):
                    per_feature[j].append(float(THETA_CONTEXT[a_idx, j] * psi_arr[j]))
                n_records += 1
                if rec.get("governance_override"):
                    n_override += 1

    mcp_arr = np.asarray(mcp_shifts)
    pirag_arr = np.asarray(pirag_shifts)
    joint_arr = np.asarray(joint_shifts)

    # Sub-additivity check: at each step, does the joint shift exceed
    # the better-of-the-two single-channel shifts?
    if mcp_arr.size > 0:
        max_single = np.maximum(np.abs(mcp_arr), np.abs(pirag_arr))
        joint_minus_max = np.abs(joint_arr) - max_single
        n_dominates = int(np.sum(joint_minus_max > 0.005))
        frac_dominates = float(np.mean(joint_minus_max > 0.005))
        median_diff = float(np.median(joint_minus_max))
    else:
        n_dominates = 0
        frac_dominates = 0.0
        median_diff = 0.0

    return {
        "n_records": n_records,
        "n_seeds": len(jsonl_paths),
        "mcp_channel_logit_shift":   _stat_block(mcp_arr),
        "pirag_channel_logit_shift": _stat_block(pirag_arr),
        "joint_logit_shift":         _stat_block(joint_arr),
        "feature_attribution": {
            "psi0_compliance":    _feature_stat(np.asarray(per_feature[0])),
            "psi1_forecast":      _feature_stat(np.asarray(per_feature[1])),
            "psi2_retrieval":     _feature_stat(np.asarray(per_feature[2])),
            "psi3_regulatory":    _feature_stat(np.asarray(per_feature[3])),
            "psi4_recovery_sat":  _feature_stat(np.asarray(per_feature[4])),
        },
        "sub_additivity": {
            "n_steps_joint_exceeds_max_single": n_dominates,
            "fraction_joint_dominates":           frac_dominates,
            "median_joint_minus_max_single":      median_diff,
        },
        "governance_override_rate": (
            float(n_override / n_records) if n_records > 0 else 0.0
        ),
    }


def _get_git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--ledger-root", type=Path,
        default=Path("mvp/simulation/results/decision_ledger_per_seed"),
        help="Root directory containing seed_<N>/ subdirs",
    )
    ap.add_argument(
        "--output", type=Path,
        default=Path("mvp/simulation/results/decision_ledger_aggregate.json"),
        help="Output JSON path",
    )
    ap.add_argument(
        "--scenarios", type=str, default=None,
        help="Optional comma-separated subset of scenarios",
    )
    ap.add_argument(
        "--modes", type=str, default=None,
        help="Optional comma-separated subset of modes",
    )
    args = ap.parse_args()

    ledger_root = args.ledger_root.resolve()
    if not ledger_root.exists():
        # Fallback: try the legacy single-seed location so the aggregator
        # is still runnable on a local dev machine for smoke-testing.
        legacy = Path("mvp/simulation/results/decision_ledger").resolve()
        if legacy.exists():
            print(f"WARN: {ledger_root} not found; falling back to {legacy} "
                  f"(single-seed mode; H3 statistics will be n=1)")
            ledger_root = legacy.parent / "_single_seed_view"
            ledger_root.mkdir(exist_ok=True)
            # Use seed_0 so it matches the seed_<int> pattern the walker
            # accepts; the "0" is a sentinel for "local single-seed
            # smoke-test, not from a canonical 20-seed run".
            seed_link = ledger_root / "seed_0"
            # Use copy (not symlink) for cross-platform compatibility.
            # Symlinks fail on Windows without admin and on some Lustre /
            # NFS tiers. ~14 MB copy is cheap relative to a full run.
            if not seed_link.exists():
                import shutil
                shutil.copytree(legacy, seed_link)
        else:
            print(f"ERROR: ledger root not found: {ledger_root}")
            sys.exit(1)

    print(f"Reading ledger files under {ledger_root}...")
    by_cell, seeds_found = _walk_ledger(ledger_root)
    print(f"  {len(seeds_found)} seeds, {len(by_cell)} (scenario, mode) cells")

    scenarios_filter = (
        set(args.scenarios.split(",")) if args.scenarios else None
    )
    modes_filter = set(args.modes.split(",")) if args.modes else None

    out: dict = {
        "_meta": {
            "n_seeds": len(seeds_found),
            "seeds": seeds_found,
            "ledger_root": str(ledger_root),
            "generated_at": _dt.datetime.utcnow().isoformat() + "Z",
            "git_commit": _get_git_commit(),
        },
        "by_scenario_mode": {},
    }

    scenarios_seen: set = set()
    modes_seen: set = set()

    for (scenario, mode), paths in sorted(by_cell.items()):
        if scenarios_filter and scenario not in scenarios_filter:
            continue
        if modes_filter and mode not in modes_filter:
            continue
        scenarios_seen.add(scenario)
        modes_seen.add(mode)
        print(f"  aggregating {scenario}/{mode} ({len(paths)} seeds)...")
        cell = _aggregate_cell(paths)
        out["by_scenario_mode"].setdefault(scenario, {})[mode] = cell

    out["_meta"]["n_scenarios"] = len(scenarios_seen)
    out["_meta"]["scenarios"] = sorted(scenarios_seen)
    out["_meta"]["n_modes"] = len(modes_seen)
    out["_meta"]["modes"] = sorted(modes_seen)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"Saved aggregate: {args.output}")


if __name__ == "__main__":
    main()
