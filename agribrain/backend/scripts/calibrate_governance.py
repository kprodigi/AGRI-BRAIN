"""Derive governance override thresholds from a calibration rollout.

Runs the simulator across benchmark scenarios under the full AgriBrain
mode, collects every decision's softmax probability vector, then calls
``calibrate_governance_thresholds`` on the stacked (N, 3) array. The
returned quantiles are what
``GOVERNANCE_CC_PROB_CEILING`` and
``GOVERNANCE_LOCAL_ADVANTAGE_MIN`` should be set to for the main
benchmark run.

Run from the repository root:

    python agribrain/backend/scripts/calibrate_governance.py

This writes a JSON report to
``mvp/simulation/results/governance_calibration.json`` and prints the
two threshold values to stdout. Paste the values into
``action_selection.py`` (or import the JSON programmatically) before
the HPC sweep.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SIM_DIR = REPO_ROOT / "mvp" / "simulation"
BACKEND = REPO_ROOT / "agribrain" / "backend"
for _p in (str(SIM_DIR), str(BACKEND)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from generate_results import run_episode, DATA_CSV, SCENARIOS  # noqa: E402
from src.models.policy import Policy  # noqa: E402
from src.models.action_selection import calibrate_governance_thresholds  # noqa: E402


def collect_probs(seed: int = 42, n_steps: int = 48) -> np.ndarray:
    """Run agribrain across all scenarios and stack the per-step probs."""
    df = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])
    df_short = df.iloc[:n_steps].reset_index(drop=True)
    all_probs = []
    for scenario in SCENARIOS:
        rng = np.random.default_rng(seed)
        result = run_episode(
            df_short, mode="agribrain", policy=Policy(),
            rng=rng, scenario=scenario, seed=seed,
        )
        probs = np.asarray(result["prob_trace"], dtype=np.float64)
        if probs.ndim == 2 and probs.shape[-1] == 3:
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 3))


def main() -> None:
    rollouts = collect_probs()
    if len(rollouts) == 0:
        print("ERROR: no probability rollouts collected", file=sys.stderr)
        sys.exit(1)
    out = calibrate_governance_thresholds(
        rollouts, cc_quantile=0.05, local_quantile=0.50,
    )
    report = {
        "n_decisions": int(len(rollouts)),
        "scenarios": SCENARIOS,
        "cc_quantile": 0.05,
        "local_quantile": 0.50,
        "cc_prob_ceiling": out["cc_prob_ceiling"],
        "local_advantage_min": out["local_advantage_min"],
        "cc_prob_distribution_summary": {
            "min": float(rollouts[:, 0].min()),
            "q05": float(np.quantile(rollouts[:, 0], 0.05)),
            "q50": float(np.quantile(rollouts[:, 0], 0.50)),
            "q95": float(np.quantile(rollouts[:, 0], 0.95)),
            "max": float(rollouts[:, 0].max()),
        },
        "local_minus_cc_summary": {
            "min": float((rollouts[:, 1] - rollouts[:, 0]).min()),
            "q25": float(np.quantile(rollouts[:, 1] - rollouts[:, 0], 0.25)),
            "q50": float(np.quantile(rollouts[:, 1] - rollouts[:, 0], 0.50)),
            "q75": float(np.quantile(rollouts[:, 1] - rollouts[:, 0], 0.75)),
            "max": float((rollouts[:, 1] - rollouts[:, 0]).max()),
        },
    }
    results_dir = REPO_ROOT / "mvp" / "simulation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "governance_calibration.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote calibration report: {out_path}")
    print(f"cc_prob_ceiling     = {report['cc_prob_ceiling']:.6f}")
    print(f"local_advantage_min = {report['local_advantage_min']:.6f}")


if __name__ == "__main__":
    main()
