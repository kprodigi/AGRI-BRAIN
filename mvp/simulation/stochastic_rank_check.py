#!/usr/bin/env python3
"""Compute stochastic multi-seed defensibility table vs deterministic reference.

Runs AGRIBRAIN mode across all scenarios for multiple seeds and reports:
- mean/std by scenario (ARI, Waste)
- delta vs deterministic reference table1
- whether scenario rank ordering is preserved
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import pandas as pd

from generate_results import (
    DATA_CSV,
    SCENARIOS,
    Policy,
    apply_scenario,
    run_episode,
)
from stochastic import make_stochastic_layer


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    os.environ["DETERMINISTIC_MODE"] = "false"
    seeds = [42, 1337, 2024, 7, 99]

    det_path = RESULTS_DIR / "table1_summary_deterministic_ref.csv"
    if not det_path.exists():
        raise FileNotFoundError(f"Missing deterministic reference: {det_path}")
    det = pd.read_csv(det_path)
    det_ag = det[det["Method"] == "agribrain"].set_index("Scenario")

    base_df = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])
    policy = Policy()

    collected = {s: {"ari": [], "waste": []} for s in SCENARIOS}
    for seed in seeds:
        rng_master = np.random.default_rng(seed)
        for scenario in SCENARIOS:
            scenario_rng = np.random.default_rng(rng_master.integers(0, 2**31))
            df_scenario = apply_scenario(base_df, scenario, policy, scenario_rng)
            stoch = make_stochastic_layer(np.random.default_rng(rng_master.integers(0, 2**31)))
            ep = run_episode(
                df_scenario,
                "agribrain",
                policy,
                np.random.default_rng(rng_master.integers(0, 2**31)),
                scenario,
                stoch=stoch,
            )
            collected[scenario]["ari"].append(float(ep["ari"]))
            collected[scenario]["waste"].append(float(ep["waste"]))

    rows = []
    for s in SCENARIOS:
        ari_m = mean(collected[s]["ari"])
        ari_sd = pstdev(collected[s]["ari"])
        w_m = mean(collected[s]["waste"])
        w_sd = pstdev(collected[s]["waste"])
        d_ari = float(det_ag.loc[s, "ARI"])
        d_w = float(det_ag.loc[s, "Waste"])
        rows.append(
            {
                "Scenario": s,
                "Det_ARI": d_ari,
                "Stoch_ARI_Mean": ari_m,
                "Stoch_ARI_Std": ari_sd,
                "Delta_ARI": ari_m - d_ari,
                "Det_Waste": d_w,
                "Stoch_Waste_Mean": w_m,
                "Stoch_Waste_Std": w_sd,
                "Delta_Waste": w_m - d_w,
            }
        )

    out_df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "stochastic_rank_check.csv"
    out_df.to_csv(out_csv, index=False)

    det_order = (
        out_df[["Scenario", "Det_ARI"]]
        .sort_values("Det_ARI", ascending=False)["Scenario"]
        .tolist()
    )
    stoch_order = (
        out_df[["Scenario", "Stoch_ARI_Mean"]]
        .sort_values("Stoch_ARI_Mean", ascending=False)["Scenario"]
        .tolist()
    )
    payload = {
        "seeds": seeds,
        "deterministic_ari_order": det_order,
        "stochastic_ari_order": stoch_order,
        "order_preserved": det_order == stoch_order,
        "rows": rows,
    }
    out_json = RESULTS_DIR / "stochastic_rank_check.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved {out_csv}")
    print(f"Saved {out_json}")
    print(f"Order preserved: {payload['order_preserved']}")
    for r in rows:
        print(
            f"{r['Scenario']:<18s} det_ari={r['Det_ARI']:.3f} stoch_ari={r['Stoch_ARI_Mean']:.3f} "
            f"std={r['Stoch_ARI_Std']:.4f} det_w={r['Det_Waste']:.3f} stoch_w={r['Stoch_Waste_Mean']:.3f}"
        )


if __name__ == "__main__":
    main()
