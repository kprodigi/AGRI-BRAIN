#!/usr/bin/env python3
"""
AGRI-BRAIN Results Generation
==============================
Runs all 5 scenarios × 8 modes, computes per-run metrics, and saves
CSV summary tables to mvp/simulation/results/.

Uses an AgentCoordinator to dispatch decisions to role-specific agents
(farm, processor, cooperative, distributor, recovery) at each lifecycle stage.

MCP/piRAG context injection is enabled for ``agribrain``, ``mcp_only``,
and ``pirag_only`` modes.  Disabled for all others (including ``no_context``
which uses the same logits as agribrain but without context modifier).

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

import json
import os
import time
import numpy as np
import pandas as pd

# Layer 1 imports — all scientific logic lives here
from src.models.spoilage import compute_spoilage, compute_spoilage_pinn, arrhenius_k, volatility_flags
from src.models.footprint import FootprintMeter
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
    from .stochastic import DETERMINISTIC_MODE, make_stochastic_layer, _DISABLED as _STOCH_DISABLED
except ImportError:
    from stochastic import DETERMINISTIC_MODE, make_stochastic_layer, _DISABLED as _STOCH_DISABLED

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

# Stochastic by default; set DETERMINISTIC_MODE=true for strict reproducibility.
DETERMINISTIC_MODE = os.environ.get("DETERMINISTIC_MODE", "false").lower() == "true"
STOCH_TEMP_STD_C = float(os.environ.get("STOCH_TEMP_STD_C", "0.35"))
STOCH_RH_STD = float(os.environ.get("STOCH_RH_STD", "1.5"))
STOCH_DEMAND_FRAC_STD = float(os.environ.get("STOCH_DEMAND_FRAC_STD", "0.04"))
STOCH_INVENTORY_FRAC_STD = float(os.environ.get("STOCH_INVENTORY_FRAC_STD", "0.03"))

def _demand_forecast(df, horizon=1, **kwargs):
    """Dispatch to LSTM or Holt-Winters demand forecaster based on config."""
    if FORECAST_METHOD == "holt_winters":
        return yield_demand_forecast(df, horizon=horizon, **kwargs)
    return lstm_demand_forecast(df, horizon=horizon, **kwargs)


def _apply_stochastic_variability(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """No-op: global DataFrame perturbation removed to prevent double-layer noise.

    Stochastic perturbation is applied exclusively per-step via StochasticLayer
    (perturb_temperature, perturb_humidity, etc.) inside run_episode(). This
    avoids compounding noise (global + per-step) and keeps the effective sigma
    equal to the documented single-layer assumptions.
    """
    return df

# ---------------------------------------------------------------------------
# Constants (orchestration-level only — no physics here)
# ---------------------------------------------------------------------------
SEED = 42

SCENARIOS = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing", "baseline"]
MODES = ["static", "hybrid_rl", "no_pinn", "no_slca",
         "agribrain", "no_context", "mcp_only", "pirag_only"]

# Modes that enable MCP/piRAG context infrastructure
_CONTEXT_ENABLED_MODES = {"agribrain", "mcp_only", "pirag_only"}

# Modes that use agribrain logits for action selection
_AGRIBRAIN_LOGIT_MODES = {"agribrain", "no_context", "mcp_only", "pirag_only"}

# Modes where MCP compliance data feeds waste penalty
_MCP_WASTE_MODES = {"agribrain", "mcp_only"}

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DATA_CSV = Path(os.environ.get("DATA_CSV", "")) if os.environ.get("DATA_CSV") else _BACKEND_SRC / "src" / "data_spinach.csv"


# ---------------------------------------------------------------------------
# Scenario perturbation — delegates to backend canonical implementations
# ---------------------------------------------------------------------------
from src.routers.scenarios import (
    _apply_heatwave, _apply_overproduction,
    _apply_cyber_outage, _apply_adaptive_pricing,
    _hours_from_start, register_app_state as _register_scenario_state,
)

_SCENARIO_FN = {
    "heatwave": _apply_heatwave,
    "overproduction": _apply_overproduction,
    "cyber_outage": _apply_cyber_outage,
    "adaptive_pricing": _apply_adaptive_pricing,
}


def apply_scenario(df: pd.DataFrame, name: str, policy: Policy,
                   rng: np.random.Generator) -> pd.DataFrame:
    # Ensure scenario functions use our policy for _recompute_derived
    _register_scenario_state({"policy": policy})
    fn = _SCENARIO_FN.get(name)
    if fn is None:
        # baseline — just recompute derived columns
        df = compute_spoilage(df.copy(), k_ref=policy.k_ref, Ea_R=policy.Ea_R,
                              T_ref_K=policy.T_ref_K, beta=policy.beta_humidity,
                              lag_lambda=policy.lag_lambda)
        df["volatility"] = volatility_flags(df, window=policy.boll_window, k=policy.boll_k)
        return df
    return fn(df)


# ---------------------------------------------------------------------------
# Single episode runner (orchestration only — calls Layer 1 models)
# ---------------------------------------------------------------------------
_PINN_MODES = {"agribrain", "no_slca", "no_context", "mcp_only", "pirag_only"}
"""Modes that use PINN-enhanced spoilage prediction."""


def run_episode(
    df: pd.DataFrame, mode: str, policy: Policy,
    rng: np.random.Generator, scenario: str = "baseline",
    stoch=None,
) -> dict:
    if stoch is None:
        stoch = _STOCH_DISABLED
    n = len(df)
    hours = _hours_from_start(df)

    # --- Multi-agent coordinator ---
    context_mode = mode in _CONTEXT_ENABLED_MODES
    coordinator = AgentCoordinator(context_enabled=context_mode)
    coordinator.reset()

    # --- Green AI footprint meter ---
    meter = FootprintMeter()

    # --- PolicyLearner (optional, off by default) ---
    learner = PolicyLearner() if ONLINE_LEARNING else None

    # --- PINN-enhanced spoilage for eligible modes ---
    effective_mode = "agribrain" if mode in _AGRIBRAIN_LOGIT_MODES else mode
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
    decision_latency_ms = []
    observed_temp_trace, observed_rh_trace = [], []
    observed_demand_trace, observed_inv_trace = [], []
    constraint_violation_steps = 0
    compliance_violation_steps = 0
    temperature_violation_steps = 0
    quality_violation_steps = 0

    prev_temp, prev_rh = float(df.iloc[0]["tempC"]), float(df.iloc[0]["RH"])

    for idx in range(n):
        row = df.iloc[idx]
        rho = float(row.get("spoilage_risk", 1.0 - row["shelf_left"]))
        inv = float(row.get("inventory_units", 100.0))
        temp = float(row["tempC"])
        rh_val = float(row["RH"])
        tau = 1.0 if str(row.get("volatility", "normal")) == "anomaly" else 0.0

        # Stochastic perturbation (no-op when DETERMINISTIC_MODE=true)
        temp = stoch.perturb_temperature(temp)
        rh_val = stoch.perturb_humidity(rh_val)
        inv = stoch.perturb_inventory(inv)

        # Telemetry delay: carry over previous perturbed step's readings
        if idx > 0 and stoch.should_delay():
            temp = prev_temp
            rh_val = prev_rh

        # Recompute spoilage risk from perturbed temp/RH.
        # Uses instantaneous Arrhenius k-ratio as proxy since the full
        # spoilage pipeline (compute_spoilage_pinn) requires the entire
        # DataFrame and produces cumulative values, not per-step updates.
        if stoch.enabled:
            k_perturbed = arrhenius_k(temp, policy.k_ref, policy.Ea_R,
                                      policy.T_ref_K, rh_val / 100.0,
                                      policy.beta_humidity)
            rho = min(1.0, max(0.0, k_perturbed / (policy.k_ref + 1e-12)))

        # Track perturbed values for next step's potential delay event
        prev_temp, prev_rh = temp, rh_val
        observed_temp_trace.append(temp)
        observed_rh_trace.append(rh_val)
        observed_inv_trace.append(inv)

        lookback = min(idx + 1, 48)
        hist_slice = df.iloc[max(0, idx + 1 - lookback):idx + 1]
        yf = _demand_forecast(hist_slice, horizon=1)
        y_hat = float(yf["forecast"][0]) if yf["forecast"] else 100.0
        y_hat = stoch.perturb_demand(y_hat)
        observed_demand_trace.append(y_hat)

        # Yield/supply forecast (Section 4 — short-term yield forecaster)
        sf = yield_supply_forecast(hist_slice, horizon=1, series_col="inventory_units")
        supply_hat = float(sf["forecast"][0]) if sf["forecast"] else inv
        supply_hats.append(supply_hat)

        # Surplus ratio (computed before env_state for coordinator)
        surplus_ratio = max(0.0, inv / INV_BASELINE - 1.0)

        # RAG context (legacy path — coordinator now handles MCP/piRAG internally)
        rag_context = None
        if RAG_CONTEXT_ENABLED and not context_mode and _get_policy_context is not None:
            try:
                rag_context = _get_policy_context(scenario=scenario, spoilage_risk=rho, temperature=temp)
            except Exception:
                pass

        # Build env_state for the coordinator
        env_state = {
            "rho": rho, "inv": inv, "temp": temp, "rh": rh_val,
            "y_hat": y_hat, "tau": tau, "surplus_ratio": surplus_ratio,
            "supply_hat": supply_hat,
            "policy_flags": {
                "enable_mcp_qos_routing": bool(getattr(policy, "enable_mcp_qos_routing", False)),
                "enable_mcp_reliability": bool(getattr(policy, "enable_mcp_reliability", False)),
                "enable_pirag_counterfactual_eval": bool(getattr(policy, "enable_pirag_counterfactual_eval", False)),
                "enable_physics_consistency_gate": bool(getattr(policy, "enable_physics_consistency_gate", False)),
                "enable_heterogeneous_profiles": bool(getattr(policy, "enable_heterogeneous_profiles", False)),
                "enable_temporal_retrieval_weighting": bool(getattr(policy, "enable_temporal_retrieval_weighting", True)),
                "enable_dynamic_knowledge_feedback": bool(getattr(policy, "enable_dynamic_knowledge_feedback", True)),
                "enable_failure_injection": bool(getattr(policy, "enable_failure_injection", False)),
                "enable_research_metrics": bool(getattr(policy, "enable_research_metrics", False)),
            },
        }

        # Action selection via AgentCoordinator
        # Pass the actual mode name so the coordinator can apply context_mode mapping
        step_t0 = time.perf_counter()
        action_idx, probs, active_agent = coordinator.step(
            env_state, hours[idx], effective_mode if mode not in _CONTEXT_ENABLED_MODES else mode,
            policy, rng, scenario, rag_context=rag_context,
        )
        # Latency is recorded as observed wall-clock time (deterministic observation);
        # no synthetic latency perturbation is applied.
        decision_latency_ms.append((time.perf_counter() - step_t0) * 1000.0)
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

        # Pass MCP compliance data to waste model for MCP-enabled modes
        compliance_data = None
        compliance_violation = False
        if mode in _MCP_WASTE_MODES and hasattr(coordinator, '_step_mcp_results'):
            compliance_data = coordinator._step_mcp_results.get("check_compliance")
            if isinstance(compliance_data, dict):
                compliance_violation = not bool(compliance_data.get("compliant", True))
                if compliance_violation:
                    compliance_violation_steps += 1

        save = compute_save_factor(
            action, "agribrain" if mode in _AGRIBRAIN_LOGIT_MODES else mode,
            surplus_ratio, compliance_data=compliance_data,
        )
        waste = float(waste_raw * (1.0 - save))

        temp_violation = temp > float(policy.max_temp_c)
        shelf_left = 1.0 - rho  # derived from (possibly perturbed) rho
        quality_violation = shelf_left < float(policy.min_shelf_expedite)
        if temp_violation:
            temperature_violation_steps += 1
        if quality_violation:
            quality_violation_steps += 1
        if temp_violation or quality_violation or compliance_violation:
            constraint_violation_steps += 1

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

        # Green AI footprint tracking (Section 4.12)
        fp = meter.compute_footprint(steps=1)

        # Post-step: update agent state and route messages
        obs = active_agent.observe(env_state, hours[idx])
        outcome = {"waste": waste, "rho": rho, "slca": slca_c, "carbon_kg": carbon}
        coordinator.post_step(active_agent, action_idx, obs, outcome,
                              hour=hours[idx], reward=reward)

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

    # PolicyLearner: apply gradient update at episode end (disabled by default)
    if learner is not None:
        import src.models.action_selection as _as_module
        updated_theta = learner.update(_as_module.THETA.copy())
        delta_norm = np.linalg.norm(updated_theta - _as_module.THETA)
        _as_module.THETA = updated_theta  # persist update for next episode
        print(f"  Policy weights updated via REINFORCE (delta norm: {delta_norm:.6f})")

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
            eq_val = compute_equity(window_slca)
        else:
            eq_val = 1.0
        equity_trace.append(eq_val)

    latency_arr = np.array(decision_latency_ms, dtype=float) if decision_latency_ms else np.array([0.0])
    latency_penalty_usd = float(np.sum(np.maximum(latency_arr - 50.0, 0.0)) * 0.0002)
    result = {
        "ari": float(np.mean(ari_vals)), "rle": float(rle),
        "waste": float(np.mean(waste_vals)), "slca": float(np.mean(slca_vals)),
        "carbon": float(carbon_total), "equity": float(equity),
        "circular_economy": float(np.mean(circular_scores)),
        "mean_supply_forecast": float(np.mean(supply_hats)),
        "mean_decision_latency_ms": float(np.mean(latency_arr)),
        "p95_decision_latency_ms": float(np.percentile(latency_arr, 95)),
        "latency_penalty_usd": latency_penalty_usd,
        "constraint_violation_rate": float(constraint_violation_steps / max(n, 1)),
        "compliance_violation_rate": float(compliance_violation_steps / max(n, 1)),
        "temperature_violation_rate": float(temperature_violation_steps / max(n, 1)),
        "quality_violation_rate": float(quality_violation_steps / max(n, 1)),
        "ari_trace": ari_vals, "waste_trace": waste_vals,
        "rho_trace": rho_trace, "action_trace": action_trace,
        "prob_trace": prob_trace, "reward_trace": reward_trace,
        "cumulative_reward": cumulative_reward, "carbon_trace": carbon_trace,
        "slca_component_trace": slca_component_trace, "slca_trace": slca_vals,
        "decision_latency_ms_trace": decision_latency_ms,
        "equity_trace": equity_trace,
        "hours": hours.tolist(),
        "temp_trace": observed_temp_trace,
        "rh_trace": observed_rh_trace,
        "demand_trace": observed_demand_trace,
        "inventory_trace": observed_inv_trace,
        "active_agent_trace": active_agent_trace,
        "footprint": meter.summary(),
        "agent_summaries": coordinator.agent_summaries(),
        "message_count": len(coordinator.message_log),
    }

    # Context diagnostics for context-enabled modes
    if context_mode:
        result["context_summary"] = coordinator.context_summary()
        result["learner_summary"] = coordinator.learner_summary()
        result["evaluator_summary"] = coordinator.evaluator_summary()

        # Trace export for paper evidence
        if coordinator.trace_exporter is not None:
            result["trace_summary"] = coordinator.trace_exporter.summary()
            result["_trace_exporter"] = coordinator.trace_exporter

        # Protocol recorder for genuine MCP interactions
        if coordinator.protocol_recorder is not None:
            result["_protocol_recorder"] = coordinator.protocol_recorder

    return result


# ---------------------------------------------------------------------------
# Full run across all scenarios × modes
# ---------------------------------------------------------------------------
def run_all(seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    policy = Policy()
    # Optional experiment toggles from environment.
    policy.enable_failure_injection = os.environ.get("FAILURE_INJECTION", "false").lower() == "true"
    policy.enable_mcp_reliability = os.environ.get("MCP_RELIABILITY", "false").lower() == "true"
    policy.enable_mcp_qos_routing = os.environ.get("MCP_QOS_ROUTING", "false").lower() == "true"
    policy.enable_pirag_counterfactual_eval = os.environ.get("PIRAG_COUNTERFACTUAL", "false").lower() == "true"
    policy.enable_physics_consistency_gate = os.environ.get("PHYSICS_CONSISTENCY_GATE", "false").lower() == "true"
    policy.enable_heterogeneous_profiles = os.environ.get("HETEROGENEOUS_PROFILES", "false").lower() == "true"
    policy.enable_research_metrics = os.environ.get("RESEARCH_METRICS", "false").lower() == "true"

    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Data CSV not found: {DATA_CSV}")

    df_base = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])

    results: dict[str, dict[str, dict]] = {}
    df_scenarios: dict[str, pd.DataFrame] = {}

    for scenario in SCENARIOS:
        results[scenario] = {}
        scenario_rng = np.random.default_rng(rng.integers(0, 2**31))
        df_scenario = apply_scenario(df_base, scenario, policy, scenario_rng)
        df_scenario = _apply_stochastic_variability(df_scenario, scenario_rng)
        df_scenarios[scenario] = df_scenario

        # Shared seed for context ablation modes so any ARI difference
        # comes exclusively from the context modifier, not stochastic noise.
        ablation_seed = rng.integers(0, 2**31)
        mode_seeds: dict[str, int] = {}
        for mode in MODES:
            if mode in _AGRIBRAIN_LOGIT_MODES:
                mode_seeds[mode] = int(ablation_seed)
            else:
                mode_seeds[mode] = int(rng.integers(0, 2**31))

        for mode in MODES:
            mode_rng = np.random.default_rng(mode_seeds[mode])
            # Stochastic layer gets an independent RNG stream (seed offset +1)
            stoch = make_stochastic_layer(np.random.default_rng(mode_seeds[mode] + 1))
            episode = run_episode(df_scenario, mode, policy, mode_rng, scenario, stoch=stoch)
            results[scenario][mode] = episode
            print(f"  [{scenario:>20s}] [{mode:>12s}] ARI={episode['ari']:.3f}  "
                  f"waste={episode['waste']:.3f}  RLE={episode['rle']:.3f}  "
                  f"SLCA={episode['slca']:.3f}  carbon={episode['carbon']:.0f}  "
                  f"equity={episode['equity']:.3f}  "
                  f"lat_ms={episode['mean_decision_latency_ms']:.2f}  "
                  f"cvr={episode['constraint_violation_rate']:.3f}")
            if mode in _CONTEXT_ENABLED_MODES and "context_summary" in episode:
                ctx = episode["context_summary"]
                evl = episode.get("evaluator_summary", {})
                lrn = episode.get("learner_summary", {})
                print(f"    Context: {ctx.get('total_mcp_tool_calls', 0)} MCP calls, "
                      f"{ctx.get('total_context_steps', 0)} piRAG queries, "
                      f"modifier nonzero {ctx.get('nonzero_modifier_steps', 0)}/{ctx.get('total_context_steps', 0)} steps, "
                      f"guard failures {ctx.get('guard_failures', 0)}, "
                      f"governance overrides {ctx.get('governance_overrides', 0)}")
                if evl:
                    print(f"    Evaluator: action changed {evl.get('context_changed_action_count', 0)}/{evl.get('total_steps', 0)} steps")
                if mode == "agribrain" and lrn.get("final_theta"):
                    print(f"    Learned THETA_CONTEXT (change norm={lrn['theta_change_norm']:.4f}):")
                    for i, row_name in enumerate(["ColdChain", "Redist  ", "Recovery"]):
                        final = lrn["final_theta"][i]
                        initial = lrn["initial_theta"][i]
                        delta = [f - ini for ini, f in zip(initial, final)]
                        print(f"      {row_name}: [{', '.join(f'{v:+.3f}' for v in final)}] "
                              f"(delta=[{', '.join(f'{d:+.3f}' for d in delta)}])")
                    print(f"    SLCA amp: {lrn['initial_slca_amp']:.3f} -> {lrn['final_slca_amp']:.3f}  "
                          f"Signs preserved: {lrn['sign_preserved']}")

            # Export traces for agribrain mode
            if mode == "agribrain" and "_trace_exporter" in episode:
                exporter = episode["_trace_exporter"]
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                trace_path = RESULTS_DIR / f"traces_{scenario}.json"
                exporter.export_json(str(trace_path))

                role_table = exporter.export_role_comparison_table()
                if role_table:
                    print(f"    Role context comparison:")
                    for row in role_table:
                        kw_str = ", ".join(row.get("top_keywords", [])[:3]) or "none"
                        print(f"      {row['role']:12s}: MCP={row['mcp_tools']}, "
                              f"KB={row['primary_kb_document']}, "
                              f"guidance={row['primary_guidance_type']}, "
                              f"keywords=[{kw_str}]")

                chains = exporter.export_provenance_chains()
                print(f"    Provenance: {len(chains)} verifiable chains")

                if exporter._traces:
                    sample = exporter._traces[0]
                    if sample.explanation_summary:
                        print(f"    Sample explanation (hour {sample.hour}, {sample.role}):")
                        print(f"      {sample.explanation_summary[:120]}")

                # Save interoperability traces
                interop = exporter.export_interoperability_trace()
                if interop:
                    interop_path = RESULTS_DIR / f"mcp_interop_{scenario}.json"
                    with open(interop_path, "w") as f:
                        json.dump(interop, f, indent=2, default=str)

            # Export real MCP protocol recordings for agribrain
            if mode == "agribrain" and "_protocol_recorder" in episode:
                proto = episode["_protocol_recorder"]
                proto_summary = proto.summary()
                if proto_summary["total_interactions"] > 0:
                    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                    proto_path = RESULTS_DIR / f"mcp_protocol_{scenario}.json"
                    proto.export_json(str(proto_path))
                    print(f"    Protocol: {proto_summary['total_interactions']} real MCP interactions, "
                          f"methods={proto_summary['methods']}")

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
                "DecisionLatencyMs": round(ep["mean_decision_latency_ms"], 3),
                "ConstraintViolationRate": round(ep["constraint_violation_rate"], 4),
                "ComplianceViolationRate": round(ep["compliance_violation_rate"], 4),
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
                "DecisionLatencyMs": round(ep["mean_decision_latency_ms"], 3),
                "ConstraintViolationRate": round(ep["constraint_violation_rate"], 4),
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
                "decision_latency_ms": round(ep["mean_decision_latency_ms"], 4),
                "constraint_violation_rate": round(ep["constraint_violation_rate"], 6),
            }
    return summary


if __name__ == "__main__":
    print("=" * 70)
    print("AGRI-BRAIN Results Generation")
    print("=" * 70)
    print(f"Seed: {SEED}")
    print(f"Deterministic mode: {DETERMINISTIC_MODE}")
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

    # Print context summaries for agribrain mode
    print()
    print("=" * 70)
    print("Context Integration Summary (agribrain mode)")
    print("=" * 70)
    for scenario in SCENARIOS:
        ep = data["results"][scenario].get("agribrain", {})
        ctx = ep.get("context_summary", {})
        lrn = ep.get("learner_summary", {})
        evl = ep.get("evaluator_summary", {})
        if ctx:
            print(f"\n  [{scenario}]")
            print(f"    MCP tool calls: {ctx.get('total_mcp_tool_calls', 0)}")
            print(f"    Mean modifier magnitude: {ctx.get('mean_modifier_magnitude', 0):.4f}")
            print(f"    Guard failures: {ctx.get('guard_failures', 0)}")
            print(f"    Governance overrides: {ctx.get('governance_overrides', 0)}")
            if lrn:
                print(f"    Learner updates: {lrn.get('n_updates', 0)}")
                print(f"    Mean advantage: {lrn.get('mean_advantage', 0):.4f}")
            if evl:
                print(f"    Context change rate: {evl.get('context_change_rate', 0):.3f}")

    print()
    print("Done. Results saved to", RESULTS_DIR)
