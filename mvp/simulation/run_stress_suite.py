#!/usr/bin/env python3
"""Stress-test suite for C&CE robustness reporting."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from generate_results import DATA_CSV, SCENARIOS, Policy, apply_scenario, run_episode


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _perturb_df(df: pd.DataFrame, stressor: str, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    if stressor == "sensor_noise":
        out["tempC"] = out["tempC"] + rng.normal(0.0, 0.9, size=len(out))
        out["RH"] = np.clip(out["RH"] + rng.normal(0.0, 4.0, size=len(out)), 15.0, 100.0)
    elif stressor == "missing_data":
        miss = rng.random(len(out)) < 0.08
        out.loc[miss, "tempC"] = np.nan
        out.loc[miss, "RH"] = np.nan
        out["tempC"] = out["tempC"].ffill().bfill()
        out["RH"] = out["RH"].ffill().bfill()
    elif stressor == "telemetry_delay":
        delay_steps = 4
        out["tempC"] = out["tempC"].shift(delay_steps).bfill()
        out["RH"] = out["RH"].shift(delay_steps).bfill()
    return out


def _run_pair(
    df: pd.DataFrame,
    scenario: str,
    seed: int,
    with_faults: bool,
    modes: Iterable[str],
) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    policy = Policy()
    if with_faults:
        policy.enable_failure_injection = True
        policy.enable_mcp_reliability = True
    results: Dict[str, Dict[str, float]] = {}
    for mode in modes:
        print(f"  running mode={mode} scenario={scenario} faults={with_faults}")
        ep = run_episode(df, mode, policy, np.random.default_rng(rng.integers(0, 2**31)), scenario)
        if not np.isfinite(ep["ari"]) or not np.isfinite(ep["waste"]) or not np.isfinite(ep["slca"]):
            raise ValueError(f"Non-finite episode metrics for mode={mode}, scenario={scenario}")
        results[mode] = {
            "ari": float(ep["ari"]),
            "waste": float(ep["waste"]),
            "slca": float(ep["slca"]),
            "rle": float(ep["rle"]),
            "constraint_violation_rate": float(ep.get("constraint_violation_rate", 0.0)),
            "decision_latency_ms": float(ep.get("mean_decision_latency_ms", 0.0)),
        }
    return results


def _degrade(nom: Dict[str, float], stressed: Dict[str, float]) -> Dict[str, float]:
    return {
        "ari_delta": float(stressed["ari"] - nom["ari"]),
        "waste_delta": float(stressed["waste"] - nom["waste"]),
        "slca_delta": float(stressed["slca"] - nom["slca"]),
        "rle_delta": float(stressed["rle"] - nom["rle"]),
        "constraint_violation_delta": float(stressed["constraint_violation_rate"] - nom["constraint_violation_rate"]),
        "latency_ms_delta": float(stressed["decision_latency_ms"] - nom["decision_latency_ms"]),
    }


def main() -> None:
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Data CSV not found: {DATA_CSV}")
    df_base = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)

    stressors = ("sensor_noise", "missing_data", "telemetry_delay", "mcp_fault_injection")
    scenarios_env = os.environ.get("STRESS_SCENARIOS", "").strip()
    if scenarios_env:
        scenarios = [s.strip() for s in scenarios_env.split(",") if s.strip()]
    else:
        scenarios = list(SCENARIOS)
    max_rows_env = os.environ.get("STRESS_MAX_ROWS", "").strip()
    max_rows = int(max_rows_env) if max_rows_env else 0
    stress_modes = {
        "sensor_noise": ("agribrain", "hybrid_rl"),
        "missing_data": ("agribrain", "hybrid_rl"),
        "telemetry_delay": ("agribrain", "hybrid_rl"),
        # MCP-specific robustness should compare MCP-dependent variants only.
        "mcp_fault_injection": ("agribrain", "mcp_only"),
    }
    summary: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    rows = []

    for scenario in scenarios:
        print(f"\n[stress] scenario={scenario}")
        scenario_df = apply_scenario(df_base, scenario, Policy(), np.random.default_rng(7))
        if max_rows > 0:
            if max_rows < 8:
                raise ValueError("STRESS_MAX_ROWS must be >= 8 to avoid degenerate dynamics.")
            scenario_df = scenario_df.head(max_rows).copy()
        baseline_union_modes = tuple(sorted({m for modes in stress_modes.values() for m in modes}))
        baseline = _run_pair(
            scenario_df,
            scenario,
            seed=42,
            with_faults=False,
            modes=baseline_union_modes,
        )
        summary[scenario] = {"baseline": baseline}
        for stressor in stressors:
            print(f" [stress] stressor={stressor}")
            stressed_df = _perturb_df(scenario_df, stressor, rng) if stressor != "mcp_fault_injection" else scenario_df
            modes = stress_modes[stressor]
            stressed = _run_pair(
                stressed_df,
                scenario,
                seed=42,
                with_faults=(stressor == "mcp_fault_injection"),
                modes=modes,
            )
            summary[scenario][stressor] = stressed
            for mode in modes:
                d = _degrade(baseline[mode], stressed[mode])
                row = {"Scenario": scenario, "Stressor": stressor, "Method": mode}
                row.update({k: round(v, 6) for k, v in d.items()})
                rows.append(row)

    out_payload = {
        "meta": {
            "scenarios": scenarios,
            "max_rows": max_rows if max_rows > 0 else None,
        },
        "results": summary,
    }
    (RESULTS_DIR / "stress_summary.json").write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "stress_degradation.csv", index=False)
    print(f"Saved {RESULTS_DIR / 'stress_summary.json'}")
    print(f"Saved {RESULTS_DIR / 'stress_degradation.csv'}")


if __name__ == "__main__":
    main()
