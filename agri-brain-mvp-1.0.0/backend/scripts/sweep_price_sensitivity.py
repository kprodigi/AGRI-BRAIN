"""Sensitivity sweep over the hand-picked price column of THETA.

``THETA[:, 9]`` was introduced in the 10-dim state commit with values
``[+0.30, -0.30, -0.05]`` (ColdChain, LocalRedistribute, Recovery).
Those numbers are a first-order economic intuition: positive demand
z-score (shortage) favours preservation, negative (oversupply) favours
redistribution, recovery is price-neutral. Since the intuition is not
calibrated, this script asks how sensitive the benchmark metrics are
to the specific values by scaling the column from zero (the
"price-unaware" ablation) through 2x the hand-picked magnitude.

For each scale we run ``agribrain`` across all five scenarios and
report the mean reward, mean waste, and mean ARI. If metrics are flat
across scales the column does not matter for this benchmark. If they
shift monotonically with scale the sign is well-chosen and the
magnitude is defensibly in the right range. Non-monotonic behaviour is
the warning sign that would say the hand-picked values are off and
need calibration.

Run from the repository root:

    python agri-brain-mvp-1.0.0/backend/scripts/sweep_price_sensitivity.py

Writes ``mvp/simulation/results/price_sensitivity.json`` and prints a
one-line-per-scale summary to stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SIM_DIR = REPO_ROOT / "mvp" / "simulation"
BACKEND = REPO_ROOT / "agri-brain-mvp-1.0.0" / "backend"
for _p in (str(SIM_DIR), str(BACKEND)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from generate_results import run_episode, DATA_CSV, SCENARIOS  # noqa: E402
from src.models import action_selection as _as  # noqa: E402
from src.models.policy import Policy  # noqa: E402

SCALES = [0.0, 0.5, 1.0, 1.5, 2.0]
"""Multipliers applied to the hand-calibrated ``THETA[:, 9]``.

``0.0`` is the price-unaware ablation (column zeroed). ``1.0`` is the
shipped values. ``2.0`` doubles the magnitude. Scales beyond 2.0 would
push per-entry drift outside the learner's 25 percent cap even if
learning were disabled for this run, so the sweep stays in [0, 2].
"""


def _run_one_scale(scale: float, seed: int = 42, n_steps: int = 48) -> dict:
    """Run agribrain across all scenarios at a given price-column scale.

    Returns a dict with mean metrics over the 5 scenarios.
    """
    original_price_column = _as.THETA[:, 9].copy()
    _as.THETA[:, 9] = scale * original_price_column
    try:
        df = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])
        df_short = df.iloc[:n_steps].reset_index(drop=True)

        rewards, wastes, aris, effective_prices = [], [], [], []
        for scenario in SCENARIOS:
            rng = np.random.default_rng(seed)
            result = run_episode(
                df_short, mode="agribrain", policy=Policy(),
                rng=rng, scenario=scenario, seed=seed,
            )
            rewards.append(float(np.mean(result["reward_trace"])))
            wastes.append(float(result.get("waste", 0.0)))
            aris.append(float(result.get("ari", 0.0)))
            effective_prices.append(float(np.mean(result.get("prob_trace", [[0]])[0])))

        return {
            "scale": scale,
            "effective_price_column": _as.THETA[:, 9].tolist(),
            "mean_reward": float(np.mean(rewards)),
            "mean_waste": float(np.mean(wastes)),
            "mean_ari": float(np.mean(aris)),
            "per_scenario_reward": dict(zip(SCENARIOS, rewards)),
            "per_scenario_waste": dict(zip(SCENARIOS, wastes)),
            "per_scenario_ari": dict(zip(SCENARIOS, aris)),
        }
    finally:
        _as.THETA[:, 9] = original_price_column


def _interpret(results: list[dict]) -> dict:
    """Classify monotonicity and rank sensitivity magnitudes."""
    scales = [r["scale"] for r in results]
    rewards = [r["mean_reward"] for r in results]
    wastes = [r["mean_waste"] for r in results]
    aris = [r["mean_ari"] for r in results]

    def _monotonic(xs):
        diffs = np.diff(xs)
        if np.all(diffs >= -1e-9):
            return "monotone_up"
        if np.all(diffs <= 1e-9):
            return "monotone_down"
        return "non_monotone"

    reward_range = max(rewards) - min(rewards)
    waste_range = max(wastes) - min(wastes)
    ari_range = max(aris) - min(aris)

    return {
        "scales": scales,
        "reward_monotonicity": _monotonic(rewards),
        "waste_monotonicity": _monotonic(wastes),
        "ari_monotonicity": _monotonic(aris),
        "reward_range": reward_range,
        "waste_range": waste_range,
        "ari_range": ari_range,
        "reward_at_scale_0": rewards[scales.index(0.0)] if 0.0 in scales else None,
        "reward_at_scale_1": rewards[scales.index(1.0)] if 1.0 in scales else None,
        "verdict": (
            "insensitive"
            if max(reward_range, ari_range) < 0.01
            else "sensitive"
        ),
    }


def main() -> None:
    print(f"Sweeping THETA[:, 9] over scales {SCALES}")
    print(f"Hand-calibrated shipping values: {_as.THETA[:, 9].tolist()}")

    results = []
    for scale in SCALES:
        summary = _run_one_scale(scale)
        print(
            f"scale={scale:.2f}: "
            f"reward={summary['mean_reward']:+.4f}  "
            f"waste={summary['mean_waste']:+.4f}  "
            f"ari={summary['mean_ari']:+.4f}  "
            f"col={summary['effective_price_column']}"
        )
        results.append(summary)

    report = {
        "shipping_price_column": _as.THETA[:, 9].tolist(),
        "scales": SCALES,
        "per_scale": results,
        "interpretation": _interpret(results),
    }

    out_dir = REPO_ROOT / "mvp" / "simulation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "price_sensitivity.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote sensitivity report: {out_path}")
    interp = report["interpretation"]
    print(f"verdict: {interp['verdict']} "
          f"(reward range {interp['reward_range']:.4f}, "
          f"ari range {interp['ari_range']:.4f})")
    print(f"monotonicity: reward={interp['reward_monotonicity']} "
          f"waste={interp['waste_monotonicity']} "
          f"ari={interp['ari_monotonicity']}")


if __name__ == "__main__":
    main()
