"""Compute explainability assessment metrics from a benchmark run.

This script measures three internal trace-integrity properties. It does not
measure whether an explanation is causally correct or understandable to users:

    1. **Policy-trace coverage** — fraction of context-influenced
       decisions whose ledger record carries every component the
       explanation engine needs (effective psi vector, effective learned
       context matrix, final-modifier feature allocation and residual,
       dominant feature index, chosen action, merkle leaf).
    2. **Sign consistency** — fraction of context-influenced decisions
       where the dominant recorded chosen-action feature allocation or
       non-feature residual shares a sign with the modifier component for
       that action. In English: when the explanation says "feature X
       drove action Y", that statement is internally consistent with
       the matrix that produced the logit shift.
    3. **Provenance integrity** — fraction of episodes whose recorded
       Merkle root recomputes byte-identically from the per-record
       leaf hashes. Catches a ledger that was edited after writing.

The script walks every ``*.jsonl`` file under the requested ledger root,
computes the three
metrics per (mode, scenario), aggregates them, and writes
``mvp/simulation/results/explainability_metrics.json``. It also
prints a one-screen summary for audit.

Usage::

    python -m mvp.simulation.analysis.explainability_metrics
    python -m mvp.simulation.analysis.explainability_metrics \\
        --ledger mvp/simulation/results/decision_ledger_per_seed/<RUN_TAG> \\
        --output mvp/simulation/results/explainability_metrics.json \\
        --threshold 0.10
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make repository packages importable when this file is run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "agribrain" / "backend"))

from mvp.simulation.benchmarks.aggregate_channel_attribution import (  # noqa: E402
    evidence_scope_metadata,
)

def _canonical_leaf(record: Dict[str, Any]) -> str:
    """Recompute the merkle leaf hash for a ledger record.

    Mirrors the logic in ``backend/src/chain/decision_ledger.py``: drop
    the ``_leaf`` field, sort keys, and SHA-256 the canonical JSON.
    """
    record_for_hash = {k: v for k, v in record.items() if k != "_leaf"}
    canonical = json.dumps(record_for_hash, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _merkle_root(leaves: List[str]) -> str:
    """Binary Merkle root over hex leaf hashes; mirrors DecisionLedger.merkle_root."""
    if not leaves:
        return ""
    layer = [bytes.fromhex(h) for h in leaves]
    while len(layer) > 1:
        nxt: List[bytes] = []
        for i in range(0, len(layer), 2):
            left = layer[i]
            right = layer[i + 1] if i + 1 < len(layer) else left
            nxt.append(hashlib.sha256(left + right).digest())
        layer = nxt
    return layer[0].hex()


def _read_ledger(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    header: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("_header"):
                header = obj
            else:
                rows.append(obj)
    return header, rows


def _is_context_active(rec: Dict[str, Any], threshold: float) -> bool:
    mod = rec.get("context_modifier")
    if not mod:
        return False
    try:
        return max(abs(float(x)) for x in mod) > threshold
    except (TypeError, ValueError):
        return False


def _has_full_chain(rec: Dict[str, Any]) -> bool:
    """Return whether a row carries every required policy-trace field."""
    psi = rec.get("psi")
    mod = rec.get("context_modifier")
    if not isinstance(psi, list) or len(psi) != 5:
        return False
    if not isinstance(mod, list) or len(mod) != 3:
        return False
    if (rec.get("dominant_psi_idx") is None
            and rec.get("dominant_context_component") is None):
        return False
    if rec.get("action_idx") is None:
        return False
    if _attribution_reconstruction_consistent(rec) is not True:
        return False
    if not rec.get("_leaf"):
        return False
    return True


def _attribution_reconstruction_consistent(rec: Dict[str, Any]) -> Optional[bool]:
    """Check that feature allocation plus residual equals the final modifier."""
    psi = rec.get("psi")
    theta = rec.get("effective_context_theta")
    full_contributions = rec.get("context_feature_contributions")
    full_residual = rec.get("context_nonfeature_residual")
    chosen_contributions = rec.get("chosen_action_context_contributions")
    chosen_residual = rec.get("chosen_action_context_residual")
    modifier = rec.get("context_modifier")
    action = rec.get("action_idx")
    if not isinstance(psi, list) or len(psi) != 5:
        return None
    if not isinstance(theta, list) or len(theta) != 3:
        return None
    if not all(isinstance(row, list) and len(row) == 5 for row in theta):
        return None
    if not isinstance(full_contributions, list) or len(full_contributions) != 3:
        return None
    if not all(
        isinstance(row, list) and len(row) == 5 for row in full_contributions
    ):
        return None
    if not isinstance(full_residual, list) or len(full_residual) != 3:
        return None
    if not isinstance(chosen_contributions, list) or len(chosen_contributions) != 5:
        return None
    if not isinstance(modifier, list) or len(modifier) != 3:
        return None
    if action is None:
        return None
    try:
        action = int(action)
        matrix = [[float(v) for v in row] for row in full_contributions]
        residual = [float(v) for v in full_residual]
        final_modifier = [float(v) for v in modifier]
        chosen = [float(v) for v in chosen_contributions]
        chosen_residual = float(chosen_residual)
    except (IndexError, TypeError, ValueError):
        return None
    full_ok = all(
        abs(sum(matrix[i]) + residual[i] - final_modifier[i])
        <= 1e-12 + 1e-10 * abs(final_modifier[i])
        for i in range(3)
    )
    chosen_ok = all(
        abs(chosen[j] - matrix[action][j])
        <= 1e-12 + 1e-10 * abs(matrix[action][j])
        for j in range(5)
    ) and (
        abs(chosen_residual - residual[action])
        <= 1e-12 + 1e-10 * abs(residual[action])
    )
    return full_ok and chosen_ok


def _sign_consistent(rec: Dict[str, Any], theta_context=None) -> Optional[bool]:
    """Does the dominant recorded contribution agree in sign with the
    final modifier component for the chosen action?

    Returns None when the row does not carry the data needed to check.

    ``theta_context`` is accepted only for API compatibility with older test
    callers. Historical decisions are not reconstructed from that process-wide
    default: the ledger's per-decision contribution vector is authoritative.
    """
    mod = rec.get("context_modifier")
    contributions = rec.get("chosen_action_context_contributions")
    residual = rec.get("chosen_action_context_residual")
    a = rec.get("action_idx")
    if not isinstance(contributions, list) or len(contributions) != 5:
        return None
    if not isinstance(mod, list):
        return None
    if a is None:
        return None
    try:
        a_int = int(a)
        values = [float(x) for x in contributions]
        feature = values[max(range(len(values)), key=lambda k: abs(values[k]))]
        residual_value = float(residual or 0.0)
        contribution = (
            residual_value if abs(residual_value) > abs(feature) else feature
        )
        mod_a = float(mod[a_int])
    except Exception:  # noqa: BLE001
        return None
    if abs(contribution) < 1e-9:
        return True
    return (contribution >= 0) == (mod_a >= 0)


def _summarise_episode(
    path: Path,
    threshold: float,
    theta_context,
) -> Dict[str, Any]:
    header, rows = _read_ledger(path)
    metadata = header.get("metadata", {})
    mode = metadata.get("mode", "unknown")
    scenario = metadata.get("scenario", "unknown")
    seed = metadata.get("seed", -1)

    n_total = len(rows)
    n_active = 0
    n_covered = 0
    n_sign_checked = 0
    n_sign_consistent = 0
    n_reconstruction_checked = 0
    n_reconstruction_consistent = 0

    leaves: List[str] = []
    leaf_mismatches = 0
    for rec in rows:
        recomputed = _canonical_leaf(rec)
        recorded = rec.get("_leaf", "")
        if recorded and recorded != recomputed:
            leaf_mismatches += 1
        leaves.append(recomputed)

        if _is_context_active(rec, threshold):
            n_active += 1
            if _has_full_chain(rec):
                n_covered += 1
            sc = _sign_consistent(rec, theta_context)
            if sc is not None:
                n_sign_checked += 1
                if sc:
                    n_sign_consistent += 1
            reconstruction = _attribution_reconstruction_consistent(rec)
            if reconstruction is not None:
                n_reconstruction_checked += 1
                if reconstruction:
                    n_reconstruction_consistent += 1

    recorded_root = header.get("merkle_root", "")
    actual_root = _merkle_root(leaves) if leaves else ""
    root_ok = bool(recorded_root) and recorded_root == actual_root and leaf_mismatches == 0

    return {
        "file": (
            path.relative_to(_REPO_ROOT).as_posix()
            if path.is_relative_to(_REPO_ROOT) else path.as_posix()
        ),
        "mode": mode,
        "scenario": scenario,
        "seed": seed,
        "n_records": n_total,
        "n_context_active": n_active,
        "n_policy_trace_covered": n_covered,
        "coverage_rate": (n_covered / n_active) if n_active else None,
        "n_sign_checked": n_sign_checked,
        "n_sign_consistent": n_sign_consistent,
        "sign_consistency_rate": (n_sign_consistent / n_sign_checked) if n_sign_checked else None,
        "n_final_modifier_reconstruction_checked": n_reconstruction_checked,
        "n_final_modifier_reconstruction_consistent": n_reconstruction_consistent,
        "final_modifier_reconstruction_rate": (
            n_reconstruction_consistent / n_reconstruction_checked
            if n_reconstruction_checked else None
        ),
        "merkle_recorded": recorded_root,
        "merkle_recomputed": actual_root,
        "leaf_mismatches": leaf_mismatches,
        "provenance_ok": root_ok,
    }


def aggregate(per_file: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_active = sum(r["n_context_active"] for r in per_file)
    n_covered = sum(r["n_policy_trace_covered"] for r in per_file)
    n_sign_checked = sum(r["n_sign_checked"] for r in per_file)
    n_sign_consistent = sum(r["n_sign_consistent"] for r in per_file)
    n_reconstruction_checked = sum(
        r["n_final_modifier_reconstruction_checked"] for r in per_file
    )
    n_reconstruction_consistent = sum(
        r["n_final_modifier_reconstruction_consistent"] for r in per_file
    )
    n_episodes = len(per_file)
    n_prov_ok = sum(1 for r in per_file if r["provenance_ok"])
    return {
        "episodes": n_episodes,
        "context_active_decisions": n_active,
        "policy_trace_coverage": (n_covered / n_active) if n_active else None,
        "sign_consistency": (n_sign_consistent / n_sign_checked) if n_sign_checked else None,
        "final_modifier_reconstruction": (
            n_reconstruction_consistent / n_reconstruction_checked
            if n_reconstruction_checked else None
        ),
        "provenance_integrity": (n_prov_ok / n_episodes) if n_episodes else None,
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ledger",
        default=str(_REPO_ROOT / "mvp" / "simulation" / "results" / "decision_ledger"),
        help="Directory containing per-episode *.jsonl decision ledgers.",
    )
    p.add_argument(
        "--output",
        default=str(_REPO_ROOT / "mvp" / "simulation" / "results" / "explainability_metrics.json"),
        help="Path to write the aggregated metrics JSON.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help=(
            "max(abs(modifier)) above which a decision counts as "
            "'context-influenced'. Default 0.10 matches the headline "
            "context-honor threshold used elsewhere."
        ),
    )
    p.add_argument(
        "--include-mode",
        action="append",
        default=None,
        help=(
            "If supplied, restrict to decisions whose `mode` field matches. "
            "Repeat for multiple modes. Defaults to all modes that record psi."
        ),
    )
    args = p.parse_args(argv)

    ledger_dir = Path(args.ledger)
    if not ledger_dir.exists():
        print(f"error: ledger dir not found: {ledger_dir}", file=sys.stderr)
        return 2

    per_file: List[Dict[str, Any]] = []
    evidence_seeds = set()
    for path in sorted(ledger_dir.rglob("*.jsonl")):
        try:
            row = _summarise_episode(path, args.threshold, None)
        except Exception as exc:  # noqa: BLE001
            if os.environ.get("STRICT_VALIDATION", "0") == "1":
                raise RuntimeError(
                    f"cannot summarize publication ledger {path}: {exc}"
                ) from exc
            print(f"warn: skipping {path.name}: {exc}", file=sys.stderr)
            continue
        if args.include_mode and row["mode"] not in args.include_mode:
            continue
        per_file.append(row)
        # Publication ledgers are consolidated as seed_<canonical-seed>/...
        # while the ledger header's legacy ``seed`` field is the mode RNG seed.
        # Count the independent experimental units from the directory identity;
        # use the header only for the standalone flat-ledger fallback.
        panel_seed = None
        try:
            relative_parts = path.relative_to(ledger_dir).parts[:-1]
        except ValueError:
            relative_parts = path.parts[:-1]
        for part in relative_parts:
            match = re.fullmatch(r"seed_(\d+)", part)
            if match:
                panel_seed = int(match.group(1))
                break
        if panel_seed is None:
            try:
                header_seed = int(row.get("seed", -1))
            except (TypeError, ValueError):
                header_seed = -1
            if header_seed >= 0:
                panel_seed = header_seed
        if panel_seed is not None:
            evidence_seeds.add(panel_seed)

    aggregate_metrics = aggregate(per_file)
    out = {
        "_meta": evidence_scope_metadata(args.ledger, len(evidence_seeds)),
        "threshold": args.threshold,
        "include_modes": args.include_mode,
        "aggregate": aggregate_metrics,
        "per_file": per_file,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False so a non-finite statistic fails here, at the producer,
    # instead of surfacing later in the semantic gate's strict JSON parser.
    out_path.write_text(
        json.dumps(out, indent=2, allow_nan=False), encoding="utf-8",
    )

    # Headline summary for the terminal.
    cov = aggregate_metrics["policy_trace_coverage"]
    sgn = aggregate_metrics["sign_consistency"]
    recon = aggregate_metrics["final_modifier_reconstruction"]
    prov = aggregate_metrics["provenance_integrity"]
    fmt = lambda v: f"{100*v:6.2f}%" if v is not None else "    n/a"
    print("explainability_metrics:")
    print(f"  episodes                : {aggregate_metrics['episodes']}")
    print(f"  context-active decisions: {aggregate_metrics['context_active_decisions']}")
    print(f"  policy-trace coverage   : {fmt(cov)}")
    print(f"  sign consistency        : {fmt(sgn)}")
    print(f"  modifier reconstruction : {fmt(recon)}")
    print(f"  provenance integrity    : {fmt(prov)}")
    print(f"  written to              : {out_path.relative_to(_REPO_ROOT) if out_path.is_relative_to(_REPO_ROOT) else out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
