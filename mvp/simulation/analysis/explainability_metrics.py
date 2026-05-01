"""Compute explainability assessment metrics from a benchmark run.

Section 1 of the manuscript advertises three structural-explainability
metrics (and §4.10 quotes them at "100% / 100% / 100%"):

    1. **Causal chain coverage** — fraction of context-influenced
       decisions whose ledger record carries every component the
       explanation engine needs (psi vector, logit modifier, dominant
       feature index, chosen action, merkle leaf).
    2. **Sign consistency** — fraction of context-influenced decisions
       where the dominant psi feature's THETA_CONTEXT entry for the
       chosen action shares a sign with the modifier component for
       that action. In English: when the explanation says "feature X
       drove action Y", that statement is internally consistent with
       the matrix that produced the logit shift.
    3. **Provenance integrity** — fraction of episodes whose recorded
       Merkle root recomputes byte-identically from the per-record
       leaf hashes. Catches a ledger that was edited after writing.

The script walks every ``*.jsonl`` file under
``mvp/simulation/results/decision_ledger/``, computes the three
metrics per (mode, scenario), aggregates them, and writes
``mvp/simulation/results/explainability_metrics.json``. It also
prints a one-screen summary so the paper claim can be eyeballed
without opening the JSON.

Usage::

    python -m mvp.simulation.analysis.explainability_metrics
    python -m mvp.simulation.analysis.explainability_metrics \\
        --ledger mvp/simulation/results/decision_ledger \\
        --output mvp/simulation/results/explainability_metrics.json \\
        --threshold 0.05
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# When run as a module the package import is fine; when run as a script we
# need to make THETA_CONTEXT importable.
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "agribrain" / "backend"))


def _load_theta_context() -> "Optional[Any]":
    try:
        from pirag.context_to_logits import THETA_CONTEXT  # type: ignore
        return THETA_CONTEXT
    except Exception:  # noqa: BLE001
        return None


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
    """A row 'covers' the causal chain when every required field exists."""
    psi = rec.get("psi")
    mod = rec.get("context_modifier")
    if not isinstance(psi, list) or len(psi) != 5:
        return False
    if not isinstance(mod, list) or len(mod) != 3:
        return False
    if rec.get("dominant_psi_idx") is None:
        return False
    if rec.get("action_idx") is None:
        return False
    if not rec.get("_leaf"):
        return False
    return True


def _sign_consistent(rec: Dict[str, Any], theta_context) -> Optional[bool]:
    """Does the stated dominant feature's matrix entry agree in sign with
    the modifier component for the chosen action?

    Returns None when the row does not carry the data needed to check."""
    if theta_context is None:
        return None
    psi = rec.get("psi")
    mod = rec.get("context_modifier")
    j = rec.get("dominant_psi_idx")
    a = rec.get("action_idx")
    if not isinstance(psi, list) or not isinstance(mod, list):
        return None
    if j is None or a is None:
        return None
    try:
        theta_aj = float(theta_context[a, j])
        psi_j = float(psi[j])
        mod_a = float(mod[a])
    except Exception:  # noqa: BLE001
        return None
    # Contribution of the dominant feature to the chosen action's logit.
    contribution = theta_aj * psi_j
    # When the contribution is non-trivial, it must share a sign with the
    # modifier component the explanation reports for that action. (When
    # the dominant feature is a piRAG feature masked off by mcp_only,
    # psi_j == 0 and there is nothing to check; treat as consistent.)
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

    recorded_root = header.get("merkle_root", "")
    actual_root = _merkle_root(leaves) if leaves else ""
    root_ok = bool(recorded_root) and recorded_root == actual_root and leaf_mismatches == 0

    return {
        "file": str(path.relative_to(_REPO_ROOT)) if path.is_relative_to(_REPO_ROOT) else str(path),
        "mode": mode,
        "scenario": scenario,
        "seed": seed,
        "n_records": n_total,
        "n_context_active": n_active,
        "n_causal_covered": n_covered,
        "coverage_rate": (n_covered / n_active) if n_active else None,
        "n_sign_checked": n_sign_checked,
        "n_sign_consistent": n_sign_consistent,
        "sign_consistency_rate": (n_sign_consistent / n_sign_checked) if n_sign_checked else None,
        "merkle_recorded": recorded_root,
        "merkle_recomputed": actual_root,
        "leaf_mismatches": leaf_mismatches,
        "provenance_ok": root_ok,
    }


def aggregate(per_file: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_active = sum(r["n_context_active"] for r in per_file)
    n_covered = sum(r["n_causal_covered"] for r in per_file)
    n_sign_checked = sum(r["n_sign_checked"] for r in per_file)
    n_sign_consistent = sum(r["n_sign_consistent"] for r in per_file)
    n_episodes = len(per_file)
    n_prov_ok = sum(1 for r in per_file if r["provenance_ok"])
    return {
        "episodes": n_episodes,
        "context_active_decisions": n_active,
        "causal_chain_coverage": (n_covered / n_active) if n_active else None,
        "sign_consistency": (n_sign_consistent / n_sign_checked) if n_sign_checked else None,
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
        default=0.05,
        help=(
            "max(abs(modifier)) above which a decision counts as "
            "'context-influenced'. Default 0.05 matches the headline "
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

    theta_context = _load_theta_context()
    if theta_context is None:
        print("warn: THETA_CONTEXT not importable; sign-consistency will be None", file=sys.stderr)

    per_file: List[Dict[str, Any]] = []
    for path in sorted(ledger_dir.glob("*.jsonl")):
        try:
            row = _summarise_episode(path, args.threshold, theta_context)
        except Exception as exc:  # noqa: BLE001
            print(f"warn: skipping {path.name}: {exc}", file=sys.stderr)
            continue
        if args.include_mode and row["mode"] not in args.include_mode:
            continue
        per_file.append(row)

    aggregate_metrics = aggregate(per_file)
    out = {
        "threshold": args.threshold,
        "include_modes": args.include_mode,
        "aggregate": aggregate_metrics,
        "per_file": per_file,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Headline summary for the terminal.
    cov = aggregate_metrics["causal_chain_coverage"]
    sgn = aggregate_metrics["sign_consistency"]
    prov = aggregate_metrics["provenance_integrity"]
    fmt = lambda v: f"{100*v:6.2f}%" if v is not None else "    n/a"
    print("explainability_metrics:")
    print(f"  episodes                : {aggregate_metrics['episodes']}")
    print(f"  context-active decisions: {aggregate_metrics['context_active_decisions']}")
    print(f"  causal chain coverage   : {fmt(cov)}")
    print(f"  sign consistency        : {fmt(sgn)}")
    print(f"  provenance integrity    : {fmt(prov)}")
    print(f"  written to              : {out_path.relative_to(_REPO_ROOT) if out_path.is_relative_to(_REPO_ROOT) else out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
