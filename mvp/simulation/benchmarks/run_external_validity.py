#!/usr/bin/env python3
"""Within-trace temporal stability check via early/mid/late slices.

This script evaluates whether method ordering and relative gains persist
across early, mid, and late thirds of the SAME 288-row sensor trace.

This is NOT external validity in the methodological sense — it is a
within-trace temporal-stability hold-out. Genuine external validity
(different crop, different region, real field telemetry) is out of
scope for this manuscript and is flagged as future work in
future work. Filenames (`external_validity_*.{json,csv}`)
are preserved for backward compatibility with the artifact manifest;
every prose reference to this check should use the
"within-trace temporal stability" wording.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ..generate_results import DATA_CSV, SCENARIOS, Policy, apply_scenario, run_episode
    from ..stochastic import make_stochastic_layer
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from generate_results import DATA_CSV, SCENARIOS, Policy, apply_scenario, run_episode
    from stochastic import make_stochastic_layer


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
WINDOWS = ("early", "mid", "late")
MODES = ("static", "hybrid_rl", "agribrain", "mcp_only", "pirag_only", "no_context")


def _split_windows(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    n = len(df)
    a = n // 3
    b = 2 * n // 3
    return {
        "early": df.iloc[:a].copy(),
        "mid": df.iloc[a:b].copy(),
        "late": df.iloc[b:].copy(),
    }


def main() -> None:
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Data CSV not found: {DATA_CSV}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    base_df = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])
    rows = []
    summary = {}
    rng = np.random.default_rng(2026)

    for scenario in SCENARIOS:
        sc_df = apply_scenario(base_df, scenario, Policy(), np.random.default_rng(7))
        windows = _split_windows(sc_df)
        summary[scenario] = {}

        for wname in WINDOWS:
            wdf = windows[wname]
            summary[scenario][wname] = {}
            policy = Policy()
            for mode in MODES:
                mode_seed = int(rng.integers(0, 2**31))
                stoch = make_stochastic_layer(np.random.default_rng(mode_seed + 1))
                ep = run_episode(
                    wdf,
                    mode,
                    policy,
                    np.random.default_rng(mode_seed),
                    scenario,
                    stoch=stoch,
                )
                rec = {
                    "ari": float(ep["ari"]),
                    "waste": float(ep["waste"]),
                    # Headline RLE = realistic match-quality.
                    "rle": float(ep.get("rle_realistic", ep["rle"])),
                    "rle_binary": float(ep["rle"]),
                    "rle_weighted": float(ep.get("rle_weighted", ep["rle"])),
                    "rle_capacity_constrained": float(
                        ep.get("rle_capacity_constrained",
                               ep.get("rle_realistic", ep["rle"]))
                    ),
                    "slca": float(ep["slca"]),
                    "carbon": float(ep["carbon"]),
                    "equity": float(ep["equity"]),
                }
                summary[scenario][wname][mode] = rec
                rows.append(
                    {
                        "Scenario": scenario,
                        "Window": wname,
                        "Method": mode,
                        "ARI": rec["ari"],
                        "Waste": rec["waste"],
                        "RLE": rec["rle"],
                        "SLCA": rec["slca"],
                        "Carbon": rec["carbon"],
                        "Equity": rec["equity"],
                    }
                )

    # Add delta rows vs static to make publication interpretation easier.
    delta_rows = []
    for scenario in SCENARIOS:
        for wname in WINDOWS:
            stat = summary[scenario][wname]["static"]
            for mode in ("hybrid_rl", "agribrain", "mcp_only", "pirag_only", "no_context"):
                rec = summary[scenario][wname][mode]
                delta_rows.append(
                    {
                        "Scenario": scenario,
                        "Window": wname,
                        "Method": mode,
                        "dARI_vs_static": rec["ari"] - stat["ari"],
                        "dWaste_vs_static": rec["waste"] - stat["waste"],
                        "dSLCA_vs_static": rec["slca"] - stat["slca"],
                        "dRLE_vs_static": rec["rle"] - stat["rle"],
                    }
                )

    (RESULTS_DIR / "external_validity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "external_validity_summary.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(RESULTS_DIR / "external_validity_deltas.csv", index=False)
    print(f"Saved {RESULTS_DIR / 'external_validity_summary.json'}")
    print(f"Saved {RESULTS_DIR / 'external_validity_summary.csv'}")
    print(f"Saved {RESULTS_DIR / 'external_validity_deltas.csv'}")


if __name__ == "__main__":
    main()
