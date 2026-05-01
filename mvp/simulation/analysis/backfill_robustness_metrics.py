"""Backfill robustness-variant metrics from the canonical decision_ledger.

The published benchmark run (`hpc_results.tar.gz`, 2026-04) was produced
before the robustness-variant metrics (RLE_w, ARI_geom, equity_sen)
were added to `generate_results.py`. The per-seed bootstrap CIs reported
in `benchmark_summary.json` therefore cover only the original primary
metrics.

This script reconstructs single-seed point estimates of the new metrics
from the per-step decision_ledger files (one canonical seed per
(mode, scenario) pair). The output is *not* a substitute for the
multi-seed CIs that the next HPC run will produce — it is a
deterministic point-estimate companion useful for sanity-checking the
new metrics' magnitudes and for early manuscript drafts.

Usage:
    python -m mvp.simulation.analysis.backfill_robustness_metrics

Writes:
    mvp/simulation/results/benchmark_robustness_singleseed.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Make backend models importable
ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "agribrain" / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from src.models.resilience import (  # noqa: E402
    compute_ari,
    compute_ari_geom,
    compute_equity,
    compute_equity_sen,
    compute_rle,
    compute_rle_uniform,
)


LEDGER_DIR = ROOT / "mvp" / "simulation" / "results" / "decision_ledger"
OUT_PATH = ROOT / "mvp" / "simulation" / "results" / "benchmark_robustness_singleseed.json"


def _load_ledger(path: Path) -> Dict[str, Any]:
    """Read a ledger jsonl file. Returns header metadata plus a list of
    per-step dicts."""
    header: Dict[str, Any] = {}
    steps: List[Dict[str, Any]] = []
    with open(path) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            if i == 0 and "_header" in d:
                header = d.get("metadata", {}) or {}
            elif "_leaf" in d:
                steps.append(d)
            else:
                # Data line without leaf marker
                steps.append(d)
    return {"header": header, "steps": steps}


def _backfill_one(path: Path) -> Dict[str, Any]:
    """Compute robustness metrics for a single ledger file."""
    bundle = _load_ledger(path)
    header = bundle["header"]
    steps = bundle["steps"]
    rho_vals = [float(s["rho"]) for s in steps]
    actions = [str(s["action"]) for s in steps]
    slca_vals = [float(s["slca"]) for s in steps]
    waste_vals = [float(s["waste"]) for s in steps]

    # Primary metrics: single canonical RLE (EU-hierarchy +
    # severity-weighted form, resilience.compute_rle).
    ari_per_step = [
        compute_ari(w, s, r) for w, s, r in zip(waste_vals, slca_vals, rho_vals)
    ]
    ari_mean = float(sum(ari_per_step) / max(len(ari_per_step), 1))
    rle_value = compute_rle(rho_vals, actions)
    rle_uniform_value = compute_rle_uniform(rho_vals, actions)
    eq_primary = compute_equity(slca_vals)

    # Robustness variants: ari_geom + Sen-welfare equity + EU-agnostic
    # rle_uniform companion (uniform action weights so the metric does
    # not encode the EU 2008/98/EC hierarchy ordering; defends against
    # the "EU-shaped policy wins on EU-shaped metric" attack).
    ari_geom_per_step = [
        compute_ari_geom(w, s, r) for w, s, r in zip(waste_vals, slca_vals, rho_vals)
    ]
    ari_geom_mean = float(sum(ari_geom_per_step) / max(len(ari_geom_per_step), 1))
    eq_sen = compute_equity_sen(slca_vals)

    return {
        "mode": header.get("mode", "unknown"),
        "scenario": header.get("scenario", "unknown"),
        "seed": header.get("seed", -1),
        "n_steps": len(steps),
        "primary": {
            "ari": ari_mean,
            "rle": rle_value,
            "equity": eq_primary,
        },
        "robustness": {
            "ari_geom": ari_geom_mean,
            "equity_sen": eq_sen,
            "rle_uniform": rle_uniform_value,
        },
    }


def main() -> None:
    if not LEDGER_DIR.exists():
        print(f"Ledger directory not found: {LEDGER_DIR}", file=sys.stderr)
        sys.exit(1)

    files = sorted(LEDGER_DIR.glob("*.jsonl"))
    print(f"Backfilling robustness metrics from {len(files)} ledger files...")

    by_scenario_mode: Dict[str, Dict[str, Any]] = {}
    for fp in files:
        try:
            rec = _backfill_one(fp)
        except Exception as exc:
            print(f"  skipped {fp.name}: {exc}", file=sys.stderr)
            continue
        scenario = rec["scenario"]
        mode = rec["mode"]
        by_scenario_mode.setdefault(scenario, {})[mode] = rec

    out = {
        "_doc": (
            "Single-seed point estimates of robustness-variant metrics "
            "(ARI_geom, RLE_weighted, equity_sen) reconstructed from the "
            "canonical decision_ledger. Multi-seed CIs require re-running "
            "the HPC benchmark with the updated generate_results.py."
        ),
        "results": by_scenario_mode,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_PATH}")

    # Console summary: agribrain vs static across scenarios for the
    # three new metrics. This is the headline supplementary table.
    print("\nAgriBrain vs static (single-seed point estimates):")
    print(f"{'scenario':<18} {'metric':<14} {'agribrain':>10} {'static':>10}")
    print("-" * 60)
    for scen, modes in by_scenario_mode.items():
        if "agribrain" not in modes or "static" not in modes:
            continue
        ab = modes["agribrain"]["robustness"]
        st = modes["static"]["robustness"]
        for k in ("ari_geom", "equity_sen"):
            print(f"{scen:<18} {k:<14} {ab[k]:>10.4f} {st[k]:>10.4f}")


if __name__ == "__main__":
    main()
