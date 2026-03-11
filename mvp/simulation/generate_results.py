#!/usr/bin/env python3
"""
AGRI-BRAIN Results Generation
==============================
Runs all 5 scenarios × 5 modes, computes per-run metrics, and saves
CSV summary tables to mvp/simulation/results/.

Uses an AgentCoordinator to dispatch decisions to role-specific agents
(farm, processor, cooperative, distributor, recovery) at each lifecycle stage.

Standalone usage:
    cd mvp/simulation
    python generate_results.py

Callable from backend:
    from mvp.simulation.generate_results import run_all, get_summary_json

This module is a **Layer 3 orchestrator**.  All scientific models, equations,
and scoring functions live in the backend model files (Layer 1):

    src.models.spoilage           — Arrhenius decay, Baranyi lag phase
    src.models.forecast           — Holt-Winters demand forecasting (fallback)
    src.models.lstm_demand        — Numpy-only LSTM demand forecasting (default)
    src.models.yield_forecast     — Holt-Winters yield/supply forecasting
    src.models.slca               — 4-component Social LCA scoring
    src.models.policy             — Policy configuration
    src.models.waste              — Operational waste model
    src.models.carbon             — Transport carbon emissions + COP degradation
    src.models.resilience         — ARI, RLE, equity metrics
    src.models.reward             — Multi-objective reward function
    src.models.action_selection   — Softmax policy, feature vectors
    src.models.reverse_logistics  — Circular economy scoring
    src.agents.coordinator        — Multi-agent coordination (5 agents)
"""
from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure backend models are importable
# ---------------------------------------------------------------------------
_BACKEND_SRC = Path(__file__).resolve().parent.parent.parent / "agri-brain-mvp-1.0.0" / "backend"
if str(_BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(_BACKEND_SRC))

import os
import numpy as np
import pandas as pd

# Layer 1 imports — all scientific logic lives here
from src.models.spoilage import compute_spoilage, compute_spoilage_pinn, arrhenius_k, volatility_flags
from src.models.forecast import yield_demand_forecast
from src.models.lstm_demand import lstm_demand_forecast
from src.models.yield_forecast import yield_supply_forecast
from src.models.slca import slca_score
from src.models.policy import Policy
from src.models.waste import (
    INV_BASELINE, compute_waste_rate, compute_save_factor,
)
from src.models.carbon import compute_transport_carbon
from src.models.resilience import compute_ari, RLETracker, compute_equity
from src.models.reward import compute_reward
from src.models.action_selection import (
    ACTIONS, ACTION_KM_KEYS, select_action,
    compute_thermal_stress, compute_slca_attenuation,
)
from src.models.reverse_logistics import evaluate_recovery_options, compute_circular_economy_score
from src.models.policy_learner import PolicyLearner
from src.models.action_selection import build_feature_vector
from src.agents.coordinator import AgentCoordinator

try:
    from pirag.context_provider import get_policy_context as _get_policy_context
except Exception:
    _get_policy_context = None

# Forecast method selection (default: LSTM, fallback: Holt-Winters)
FORECAST_METHOD = os.environ.get("FORECAST_METHOD", "lstm")

# Online learning toggle (default: disabled to preserve deterministic results)
ONLINE_LEARNING = os.environ.get("ONLINE_LEARNING", "false").lower() == "true"

# RAG context toggle (default: enabled; set to "false" for fast batch runs)
RAG_CONTEXT_ENABLED = os.environ.get("RAG_CONTEXT_ENABLED", "true").lower() != "false"

def _demand_forecast(df, horizon=1, **kwargs):
    """Dispatch to LSTM or Holt-Winters demand forecaster based on config."""
    if FORECAST_METHOD == "holt_winters":
        return yield_demand_forecast(df, horizon=horizon, **kwargs)
    return lstm_demand_forecast(df, horizon=horizon, **kwargs)

# ---------------------------------------------------------------------------
# Constants (orchestration-level only — no physics here)
# ---------------------------------------------------------------------------
SEED = 42

SCENARIOS = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing", "baseline"]
MODES = ["static", "hybrid_rl", "no_pinn", "no_slca", "agribrain"]

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DATA_CSV = _BACKEND_SRC / "src" / "data_spinach.csv"


# ---------------------------------------------------------------------------
# Scenario perturbation (applies environmental stress to the base data)
# ---------------------------------------------------------------------------
def _hours_from_start(df: pd.DataFrame) -> np.ndarray:
    ts = pd.to_datetime(df["timestamp"])
    return ((ts - ts.iloc[0]).dt.total_seconds() / 3600.0).to_numpy(dtype=np.float64)


def _recompute_derived(df: pd.DataFrame, policy: Policy) -> pd.DataFrame:
    df = compute_spoilage(df, k_ref=policy.k_ref, Ea_R=policy.Ea_R,
                          T_ref_K=policy.T_ref_K, beta=policy.beta_humidity,
                          lag_lambda=policy.lag_lambda)
    df["volatility"] = volatility_flags(df, window=policy.boll_window, k=policy.boll_k)
    return df


def apply_scenario(df: pd.DataFrame, name: str, policy: Policy,
                   rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    for col in ("tempC", "RH", "inventory_units", "demand_units"):
        if col in df.columns:
            df[col] = df[col].astype(float)
    hours = _hours_from_start(df)
    n = len(df)

    if name == "heatwave":
        temp_add = np.zeros(n)
        rh_add = np.zeros(n)
        for i in range(n):
            h = hours[i]
            if 24.0 <= h <= 48.0:
                # Sigmoid onset: reaches ~95 % of peak by h ≈ 30 (6 h ramp).
                # More realistic than a linear ramp — heatwaves build
                # rapidly once they onset (WMO, 2018).
                onset = 1.0 - np.exp(-0.5 * (h - 24.0))
                temp_add[i] = 20.0 * onset
                rh_add[i] = 10.0 * onset
            elif h > 48.0:
                temp_add[i] = 20.0 * np.exp(-0.1 * (h - 48.0))
                rh_add[i] = 10.0 * np.exp(-0.1 * (h - 48.0))
        df["tempC"] = df["tempC"] + temp_add
        df["RH"] = (df["RH"] + rh_add).clip(0, 100)
    elif name == "overproduction":
        mask = (hours >= 12.0) & (hours <= 60.0)
        df.loc[mask, "inventory_units"] = df.loc[mask, "inventory_units"] * 2.5
        # Overloaded cold storage: progressive temperature creep.
        # At 2.5× capacity, reduced airflow and compressor strain raise
        # cold-room temperature by up to +8 °C (James & James, 2010).
        # Sigmoid onset (~95 % by h ≈ 22), exponential recovery after.
        temp_add = np.zeros(n)
        for i in range(n):
            h = hours[i]
            if 12.0 <= h <= 60.0:
                onset = 1.0 - np.exp(-0.3 * (h - 12.0))
                temp_add[i] = 8.0 * onset
            elif h > 60.0:
                temp_add[i] = 8.0 * np.exp(-0.15 * (h - 60.0))
        df["tempC"] = df["tempC"] + temp_add
    elif name == "cyber_outage":
        mask = hours >= 24.0
        df.loc[mask, "demand_units"] = df.loc[mask, "demand_units"] * 0.15
        df.loc[mask, "inventory_units"] = df.loc[mask, "inventory_units"] * 0.25
        # Refrigeration degradation: IT-controlled cooling partially fails
        temp_add = np.zeros(n)
        for i in range(n):
            h = hours[i]
            if h >= 24.0:
                onset = 1.0 - np.exp(-0.2 * (h - 24.0))
                temp_add[i] = 5.0 * onset
        df["tempC"] = df["tempC"] + temp_add
    elif name == "adaptive_pricing":
        oscillation = 45.0 * np.sin(2.0 * np.pi * np.arange(n) / 60.0)
        noise = rng.normal(0.0, 14.0, size=n)
        df["demand_units"] = (df["demand_units"] + oscillation + noise).clip(0)
        # Cold-storage stress from demand-supply mismatch
        demand = df["demand_units"].to_numpy()
        inv = df["inventory_units"].to_numpy()
        surplus_signal = np.clip((inv / 12000.0 - 1.0), 0, 2.0)
        temp_add = 2.0 * surplus_signal
        df["tempC"] = df["tempC"] + temp_add

    return _recompute_derived(df, policy)


# ---------------------------------------------------------------------------
# Single episode runner (orchestration only — calls Layer 1 models)
# ---------------------------------------------------------------------------
_PINN_MODES = {"agribrain", "no_slca"}
"""Modes that use PINN-enhanced spoilage prediction."""


def run_episode(
    df: pd.DataFrame, mode: str, policy: Policy,
    rng: np.random.Generator, scenario: str = "baseline",
) -> dict:
    n = len(df)
    hours = _hours_from_start(df)

    # --- Multi-agent coordinator ---
    coordinator = AgentCoordinator()
    coordinator.reset()

    # --- PolicyLearner (optional, off by default) ---
    learner = PolicyLearner() if ONLINE_LEARNING else None

    # --- PINN-enhanced spoilage for eligible modes ---
    if mode in _PINN_MODES:
        df = compute_spoilage_pinn(
            df, k_ref=policy.k_ref, Ea_R=policy.Ea_R,
            T_ref_K=policy.T_ref_K, beta=policy.beta_humidity,
            lag_lambda=policy.lag_lambda,
        )

    ari_vals, waste_vals, slca_vals = [], [], []
    rle_tracker = RLETracker()
    carbon_total, cum_r = 0.0, 0.0
    cumulative_reward = []
    rho_trace, action_trace, prob_trace = [], [], []
    reward_trace, carbon_trace, slca_component_trace = [], [], []
    active_agent_trace = []
    circular_scores = []
    supply_hats = []

    for idx in range(n):
        row = df.iloc[idx]
        rho = float(row.get("spoilage_risk", 1.0 - row["shelf_left"]))
        inv = float(row.get("inventory_units", 100.0))
        temp = float(row["tempC"])
        rh_val = float(row["RH"])
        tau = 1.0 if str(row.get("volatility", "normal")) == "anomaly" else 0.0

        lookback = min(idx + 1, 48)
        hist_slice = df.iloc[max(0, idx + 1 - lookback):idx + 1]
        yf = _demand_forecast(hist_slice, horizon=1)
        y_hat = float(yf["forecast"][0]) if yf["forecast"] else 100.0

        # Yield/supply forecast (Section 4 — short-term yield forecaster)
        sf = yield_supply_forecast(hist_slice, horizon=1, series_col="inventory_units")
        supply_hat = float(sf["forecast"][0]) if sf["forecast"] else inv
        supply_hats.append(supply_hat)

        # Surplus ratio (computed before env_state for coordinator)
        surplus_ratio = max(0.0, inv / INV_BASELINE - 1.0)

        # RAG context (gated for performance in batch runs)
        rag_context = None
        if RAG_CONTEXT_ENABLED and _get_policy_context is not None:
            try:
                rag_context = _get_policy_context(scenario=scenario, spoilage_risk=rho, temperature=temp)
            except Exception:
                pass

        # Build env_state for the coordinator
        env_state = {
            "rho": rho, "inv": inv, "temp": temp, "rh": rh_val,
            "y_hat": y_hat, "tau": tau, "surplus_ratio": surplus_ratio,
            "supply_hat": supply_hat,
        }

        # Action selection via AgentCoordinator
        action_idx, probs, active_agent = coordinator.step(
            env_state, hours[idx], mode, policy, rng, scenario,
            rag_context=rag_context,
        )
        action = ACTIONS[action_idx]
        active_agent_trace.append(active_agent.role)

        # Carbon emissions (Layer 1: carbon.py)
        km = getattr(policy, ACTION_KM_KEYS[action])
        thermal_stress = compute_thermal_stress(temp)
        carbon = compute_transport_carbon(km, policy.carbon_per_km, thermal_stress)

        # SLCA scoring (Layer 1: slca.py) with stress attenuation
        slca_result = slca_score(carbon, action,
                                 w_c=policy.w_c, w_l=policy.w_l,
                                 w_r=policy.w_r, w_p=policy.w_p)
        slca_raw = slca_result["composite"]
        slca_quality = compute_slca_attenuation(thermal_stress, surplus_ratio)
        slca_c = slca_raw * slca_quality

        # Waste computation (Layer 1: waste.py + spoilage.py)
        k_inst = arrhenius_k(temp, policy.k_ref, policy.Ea_R,
                             policy.T_ref_K, rh_val / 100.0,
                             policy.beta_humidity)
        waste_raw = compute_waste_rate(k_inst, surplus_ratio)
        save = compute_save_factor(action, mode, surplus_ratio)
        waste = float(waste_raw * (1.0 - save))

        # Circular economy score (Layer 1: reverse_logistics.py)
        recovery_opts = evaluate_recovery_options(rho, inv, temp)
        circular = compute_circular_economy_score(action, recovery_opts)
        circular_scores.append(circular)

        # ARI (Layer 1: resilience.py)
        ari = compute_ari(waste, slca_c, rho)

        # RLE tracking (Layer 1: resilience.py)
        rle_tracker.update(rho, action)

        # Reward (Layer 1: reward.py)
        reward = compute_reward(slca_c, waste, eta=policy.eta)
        cum_r += reward

        # Post-step: update agent state and route messages
        obs = active_agent.observe(env_state, hours[idx])
        outcome = {"waste": waste, "rho": rho}
        coordinator.post_step(active_agent, action_idx, obs, outcome, hour=hours[idx])

        # PolicyLearner: record experience for optional online learning
        if learner is not None:
            phi = build_feature_vector(rho, inv, y_hat, temp)
            learner.record(phi, action_idx, reward)

        # Collect traces
        ari_vals.append(ari)
        waste_vals.append(waste)
        slca_vals.append(slca_c)
        carbon_total += carbon
        cumulative_reward.append(cum_r)
        rho_trace.append(rho)
        action_trace.append(action_idx)
        prob_trace.append(probs.tolist())
        reward_trace.append(reward)
        carbon_trace.append(carbon)
        slca_component_trace.append(slca_result)

    # PolicyLearner: apply gradient update at episode end
    if learner is not None:
        from src.models.action_selection import THETA
        updated_theta = learner.update(THETA)
        print(f"  Policy weights updated via REINFORCE (delta norm: {np.linalg.norm(updated_theta - THETA):.6f})")

    # Episode-level metrics (Layer 1: resilience.py)
    rle = rle_tracker.rle
    equity = compute_equity(slca_vals)

    # Rolling equity (6-hour window = 24 steps at 15-min resolution)
    eq_window = 24
    equity_trace = []
    for idx in range(n):
        start = max(0, idx - eq_window + 1)
        window_slca = slca_vals[start:idx + 1]
        if len(window_slca) > 1:
            eq_val = 1.0 - float(np.std(window_slca))
        else:
            eq_val = 1.0
        equity_trace.append(eq_val)

    return {
        "ari": float(np.mean(ari_vals)), "rle": float(rle),
        "waste": float(np.mean(waste_vals)), "slca": float(np.mean(slca_vals)),
        "carbon": float(carbon_total), "equity": float(equity),
        "circular_economy": float(np.mean(circular_scores)),
        "mean_supply_forecast": float(np.mean(supply_hats)),
        "ari_trace": ari_vals, "waste_trace": waste_vals,
        "rho_trace": rho_trace, "action_trace": action_trace,
        "prob_trace": prob_trace, "reward_trace": reward_trace,
        "cumulative_reward": cumulative_reward, "carbon_trace": carbon_trace,
        "slca_component_trace": slca_component_trace, "slca_trace": slca_vals,
        "equity_trace": equity_trace,
        "hours": hours.tolist(), "temp_trace": df["tempC"].tolist(),
        "rh_trace": df["RH"].tolist(), "demand_trace": df["demand_units"].tolist(),
        "inventory_trace": df["inventory_units"].tolist(),
        "active_agent_trace": active_agent_trace,
        "agent_summaries": coordinator.agent_summaries(),
        "message_count": len(coordinator.message_log),
    }


# ---------------------------------------------------------------------------
# Full run across all scenarios × modes
# ---------------------------------------------------------------------------
def run_all(seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    policy = Policy()

    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Data CSV not found: {DATA_CSV}")

    df_base = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])

    results: dict[str, dict[str, dict]] = {}
    df_scenarios: dict[str, pd.DataFrame] = {}

    for scenario in SCENARIOS:
        results[scenario] = {}
        scenario_rng = np.random.default_rng(rng.integers(0, 2**31))
        df_scenario = apply_scenario(df_base, scenario, policy, scenario_rng)
        df_scenarios[scenario] = df_scenario

        for mode in MODES:
            mode_rng = np.random.default_rng(rng.integers(0, 2**31))
            episode = run_episode(df_scenario, mode, policy, mode_rng, scenario)
            results[scenario][mode] = episode
            print(f"  [{scenario:>20s}] [{mode:>12s}] ARI={episode['ari']:.3f}  "
                  f"waste={episode['waste']:.3f}  RLE={episode['rle']:.3f}  "
                  f"SLCA={episode['slca']:.3f}  carbon={episode['carbon']:.0f}  "
                  f"equity={episode['equity']:.3f}")

    table1_methods = ["static", "hybrid_rl", "agribrain"]
    table1_rows = []
    for scenario in SCENARIOS:
        for method in table1_methods:
            ep = results[scenario][method]
            table1_rows.append({
                "Scenario": scenario, "Method": method,
                "ARI": round(ep["ari"], 3), "RLE": round(ep["rle"], 3),
                "Waste": round(ep["waste"], 3), "SLCA": round(ep["slca"], 3),
                "Carbon": round(ep["carbon"], 0), "Equity": round(ep["equity"], 3),
            })
    table1 = pd.DataFrame(table1_rows)

    table2_rows = []
    for scenario in SCENARIOS:
        for mode in MODES:
            ep = results[scenario][mode]
            table2_rows.append({
                "Scenario": scenario, "Variant": mode,
                "ARI": round(ep["ari"], 3), "RLE": round(ep["rle"], 3),
                "Waste": round(ep["waste"], 3), "SLCA": round(ep["slca"], 3),
            })
    table2 = pd.DataFrame(table2_rows)

    return {"results": results, "table1": table1, "table2": table2,
            "df_scenarios": df_scenarios}


def save_tables(table1: pd.DataFrame, table2: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t1_path = RESULTS_DIR / "table1_summary.csv"
    t2_path = RESULTS_DIR / "table2_ablation.csv"
    table1.to_csv(t1_path, index=False)
    table2.to_csv(t2_path, index=False)
    print(f"Saved {t1_path}")
    print(f"Saved {t2_path}")


def get_summary_json(run_data: dict | None = None) -> dict:
    if run_data is None:
        run_data = run_all()
    summary = {}
    for scenario in SCENARIOS:
        summary[scenario] = {}
        for mode in MODES:
            ep = run_data["results"][scenario][mode]
            summary[scenario][mode] = {
                "ari": round(ep["ari"], 4), "rle": round(ep["rle"], 4),
                "waste": round(ep["waste"], 4), "slca": round(ep["slca"], 4),
                "carbon": round(ep["carbon"], 2), "equity": round(ep["equity"], 4),
            }
    return summary


if __name__ == "__main__":
    print("=" * 70)
    print("AGRI-BRAIN Results Generation")
    print("=" * 70)
    print(f"Seed: {SEED}")
    print(f"Scenarios: {SCENARIOS}")
    print(f"Modes: {MODES}")
    print()

    data = run_all()
    save_tables(data["table1"], data["table2"])

    print()
    print("=" * 70)
    print("Table 1 — Summary (Scenario × Method)")
    print("=" * 70)
    print(data["table1"].to_string(index=False))

    print()
    print("=" * 70)
    print("Table 2 — Ablation (Scenario × Variant)")
    print("=" * 70)
    print(data["table2"].to_string(index=False))

    print()
    print("Done. Results saved to", RESULTS_DIR)
