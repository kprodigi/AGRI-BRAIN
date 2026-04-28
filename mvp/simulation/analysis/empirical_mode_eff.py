"""Reproduction script for docs/MODE_EFF_EMPIRICAL.md.

Reads the published `benchmark_summary.json` and prints the
predicted-vs-empirical save-efficiency comparison table that defends
the capability-additive MODE_EFF calibration.

Usage:
    python -m mvp.simulation.analysis.empirical_mode_eff
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "agribrain" / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from src.models.waste import MODE_EFF  # noqa: E402

SUMMARY_PATH = ROOT / "mvp" / "simulation" / "results" / "benchmark_summary.json"


def main() -> None:
    with open(SUMMARY_PATH) as f:
        data = json.load(f)

    scenarios = list(data.keys())

    # Mean empirical save per mode across scenarios
    mode_saves: dict[str, list[float]] = {}
    for s in scenarios:
        if "static" not in data[s]:
            continue
        w_static = data[s]["static"]["waste"]["mean"]
        if w_static <= 0:
            continue
        for m in data[s]:
            if m == "static":
                continue
            w_m = data[s][m]["waste"]["mean"]
            mode_saves.setdefault(m, []).append(1.0 - w_m / w_static)

    # Per-mode aggregate
    summary = []
    for m in sorted(mode_saves.keys()):
        saves = mode_saves[m]
        if not saves:
            continue
        mean = sum(saves) / len(saves)
        predicted = MODE_EFF.get(m, float("nan"))
        delta = mean - predicted if predicted == predicted else float("nan")
        summary.append((m, predicted, mean, delta, min(saves), max(saves), len(saves)))

    print(f"\nMODE_EFF: predicted vs. empirical save efficiency\n")
    print(f"{'mode':<14} {'predicted':>10} {'observed':>10} {'delta':>8} "
          f"{'obs_min':>9} {'obs_max':>9} {'n_scen':>7}")
    print("-" * 70)
    print(f"{'static':<14} {0.0:>10.3f} {0.0:>10.3f} {0.0:>+8.3f} "
          f"{0.0:>9.3f} {0.0:>9.3f} {5:>7d}")
    for m, pred, obs, delta, lo, hi, n in summary:
        print(f"{m:<14} {pred:>10.3f} {obs:>10.3f} {delta:>+8.3f} "
              f"{lo:>9.3f} {hi:>9.3f} {n:>7d}")
    print()

    # Per-scenario per-mode detail
    print("Per-scenario detail (effective_save = 1 - waste_m / waste_static):\n")
    print(f"{'scenario':<18}", end="")
    modes_in_order = sorted(set().union(*(set(data[s].keys()) for s in scenarios)) - {"static"})
    for m in modes_in_order:
        print(f"{m:>13}", end="")
    print()
    print("-" * (18 + 13 * len(modes_in_order)))
    for s in scenarios:
        if "static" not in data[s]:
            continue
        w_static = data[s]["static"]["waste"]["mean"]
        print(f"{s:<18}", end="")
        for m in modes_in_order:
            if m in data[s]:
                w_m = data[s][m]["waste"]["mean"]
                save = 1.0 - w_m / w_static if w_static > 0 else 0.0
                print(f"{save:>13.3f}", end="")
            else:
                print(f"{'-':>13}", end="")
        print()


if __name__ == "__main__":
    main()
