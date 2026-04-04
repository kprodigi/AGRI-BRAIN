#!/usr/bin/env python3
"""Run a stochastic multi-seed benchmark summary for AGRIBRAIN only."""
from __future__ import annotations

import io
import json
import os
from contextlib import redirect_stdout
from statistics import mean, pstdev

from generate_results import SCENARIOS, run_all


def main() -> None:
    os.environ["DETERMINISTIC_MODE"] = "false"
    seeds = [42, 1337, 2024, 7, 99]
    vals = {s: {"ari": [], "waste": []} for s in SCENARIOS}

    for seed in seeds:
        with redirect_stdout(io.StringIO()):
            out = run_all(seed=seed)
        for s in SCENARIOS:
            ep = out["results"][s]["agribrain"]
            vals[s]["ari"].append(float(ep["ari"]))
            vals[s]["waste"].append(float(ep["waste"]))

    summary = {}
    for s in SCENARIOS:
        summary[s] = {
            "ari_mean": mean(vals[s]["ari"]),
            "ari_std": pstdev(vals[s]["ari"]),
            "waste_mean": mean(vals[s]["waste"]),
            "waste_std": pstdev(vals[s]["waste"]),
        }

    print("FRESH_STOCH_MULTI_SEED_DONE")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
