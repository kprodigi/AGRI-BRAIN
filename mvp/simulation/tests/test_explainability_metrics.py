"""Tests for mvp.simulation.analysis.explainability_metrics.

These exercise the metric computations on a synthetic ledger so the
script can be relied on without a full HPC run.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from mvp.simulation.analysis import explainability_metrics as em  # noqa: E402


def _leaf_for(record):
    rec_for_hash = {k: v for k, v in record.items() if k != "_leaf"}
    return hashlib.sha256(json.dumps(rec_for_hash, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _make_episode(tmp_path: Path, mode: str, scenario: str, rows, *, corrupt_root: bool = False, corrupt_leaf: int = -1):
    """Write a synthetic decision-ledger jsonl mirroring the simulator's format."""
    enriched = []
    for r in rows:
        rec = dict(r)
        rec["_leaf"] = _leaf_for(rec)
        enriched.append(rec)
    if corrupt_leaf >= 0 and corrupt_leaf < len(enriched):
        enriched[corrupt_leaf]["_leaf"] = "deadbeef" * 8
    leaves = [r["_leaf"] for r in enriched]
    actual_root = em._merkle_root(leaves)
    header = {
        "_header": True,
        "merkle_root": actual_root if not corrupt_root else "00" * 32,
        "metadata": {"mode": mode, "scenario": scenario, "seed": 0},
        "n_records": len(enriched),
    }
    path = tmp_path / f"{mode}__{scenario}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header, separators=(",", ":")) + "\n")
        for rec in enriched:
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
    return path


def _row(action_idx: int, *, psi=None, mod=None, dom=None):
    return {
        "ts": 0,
        "hour": 0.0,
        "agent": "farm",
        "role": "farm",
        "action": "cold_chain",
        "action_idx": int(action_idx),
        "probs": [0.5, 0.3, 0.2],
        "reward": 0.0,
        "waste": 0.0,
        "rho": 0.05,
        "slca": 0.5,
        "carbon_kg": 1.0,
        "mode": "agribrain",
        "scenario": "baseline",
        "psi": list(psi) if psi is not None else None,
        "context_modifier": list(mod) if mod is not None else None,
        "dominant_psi_idx": dom,
        "dominant_action_idx": (int(max(range(len(mod)), key=lambda i: mod[i])) if mod is not None else None),
        "governance_override": False,
    }


def test_provenance_integrity_pass(tmp_path):
    rows = [_row(0), _row(1), _row(2)]
    _make_episode(tmp_path, "agribrain", "baseline", rows)
    out = em.aggregate([em._summarise_episode(p, 0.05, None) for p in tmp_path.glob("*.jsonl")])
    assert out["provenance_integrity"] == 1.0


def test_provenance_integrity_detects_root_corruption(tmp_path):
    rows = [_row(0), _row(1)]
    _make_episode(tmp_path, "agribrain", "baseline", rows, corrupt_root=True)
    summary = em._summarise_episode(next(tmp_path.glob("*.jsonl")), 0.05, None)
    assert summary["provenance_ok"] is False


def test_provenance_integrity_detects_leaf_corruption(tmp_path):
    rows = [_row(0), _row(1), _row(2)]
    _make_episode(tmp_path, "agribrain", "baseline", rows, corrupt_leaf=1)
    summary = em._summarise_episode(next(tmp_path.glob("*.jsonl")), 0.05, None)
    assert summary["leaf_mismatches"] == 1
    assert summary["provenance_ok"] is False


def test_causal_chain_coverage_full(tmp_path):
    # All three rows have a non-trivial modifier and all explainability fields.
    psi = [0.8, 0.1, 0.0, 0.0, 0.0]
    mod = [-0.6, 0.4, 0.1]
    rows = [_row(1, psi=psi, mod=mod, dom=0) for _ in range(3)]
    _make_episode(tmp_path, "agribrain", "baseline", rows)
    summary = em._summarise_episode(next(tmp_path.glob("*.jsonl")), 0.05, None)
    assert summary["n_context_active"] == 3
    assert summary["coverage_rate"] == 1.0


def test_causal_chain_coverage_partial(tmp_path):
    psi = [0.8, 0.1, 0.0, 0.0, 0.0]
    mod = [-0.6, 0.4, 0.1]
    full = _row(1, psi=psi, mod=mod, dom=0)
    no_psi = _row(1, psi=None, mod=mod, dom=None)
    rows = [full, full, no_psi]
    _make_episode(tmp_path, "agribrain", "baseline", rows)
    summary = em._summarise_episode(next(tmp_path.glob("*.jsonl")), 0.05, None)
    assert summary["n_context_active"] == 3
    assert summary["coverage_rate"] == pytest.approx(2 / 3, rel=1e-6)


def test_sign_consistency_uses_theta_context(tmp_path):
    import numpy as np
    theta = np.array([
        [-0.8, -0.6, -0.15, -0.30, +0.25],   # cold_chain
        [+0.5, +0.4, +0.20, +0.25, +0.10],   # local_redistribute
        [+0.3, +0.2, -0.05, +0.05, -0.35],   # recovery
    ])
    psi = [0.9, 0.0, 0.0, 0.0, 0.0]
    mod = (theta @ np.asarray(psi)).tolist()  # consistent by construction
    consistent = _row(1, psi=psi, mod=mod, dom=0)  # action_idx=1 -> theta[1,0] > 0, mod[1] > 0
    inconsistent = dict(consistent)
    inconsistent["context_modifier"] = [-x for x in mod]  # flip modifier sign artificially
    rows = [consistent, consistent, inconsistent]
    _make_episode(tmp_path, "agribrain", "baseline", rows)
    summary = em._summarise_episode(next(tmp_path.glob("*.jsonl")), 0.05, theta)
    assert summary["n_sign_checked"] == 3
    assert summary["sign_consistency_rate"] == pytest.approx(2 / 3, rel=1e-6)


def test_aggregate_handles_empty_input():
    out = em.aggregate([])
    assert out["episodes"] == 0
    assert out["causal_chain_coverage"] is None
    assert out["sign_consistency"] is None
    assert out["provenance_integrity"] is None


def test_main_writes_output(tmp_path, capsys):
    psi = [0.8, 0.1, 0.0, 0.0, 0.0]
    mod = [-0.6, 0.4, 0.1]
    rows = [_row(1, psi=psi, mod=mod, dom=0)]
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    _make_episode(ledger_dir, "agribrain", "baseline", rows)
    out_path = tmp_path / "metrics.json"
    rc = em.main([
        "--ledger", str(ledger_dir),
        "--output", str(out_path),
    ])
    assert rc == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["aggregate"]["episodes"] == 1
    assert payload["aggregate"]["provenance_integrity"] == 1.0
