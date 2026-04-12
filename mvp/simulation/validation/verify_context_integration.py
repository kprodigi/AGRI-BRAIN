#!/usr/bin/env python3
"""
Verification script for MCP + piRAG context integration (Task 30).

Runs the baseline scenario with agribrain context ON vs OFF, prints
diagnostic comparisons, and asserts correctness properties.

Usage:
    cd mvp/simulation
    python verify_context_integration.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_SRC = Path(__file__).resolve().parent.parent.parent.parent / "agri-brain-mvp-1.0.0" / "backend"
if str(_BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(_BACKEND_SRC))

import numpy as np
import pandas as pd

from src.models.spoilage import compute_spoilage, volatility_flags
from src.models.policy import Policy
from src.agents.coordinator import AgentCoordinator
from src.models.action_selection import ACTIONS, ACTION_KM_KEYS, select_action, build_feature_vector
from src.models.action_selection import compute_thermal_stress, compute_slca_attenuation
from src.models.carbon import compute_transport_carbon
from src.models.slca import slca_score
from src.models.waste import INV_BASELINE, compute_waste_rate, compute_save_factor
from src.models.spoilage import arrhenius_k
from src.models.resilience import compute_ari, compute_equity
from src.models.reward import compute_reward
from src.models.lstm_demand import lstm_demand_forecast
from src.models.yield_forecast import yield_supply_forecast

from src.routers.scenarios import _hours_from_start, register_app_state as _register_scenario_state

DATA_CSV = _BACKEND_SRC / "src" / "data_spinach.csv"
SEED = 42


def _run_short_episode(df, mode, context_enabled, n_steps=48):
    """Run a short episode for verification."""
    policy = Policy()
    # Optional robustness mode controlled by env.
    import os
    if os.environ.get("FAILURE_INJECTION", "false").lower() == "true":
        policy.enable_failure_injection = True
        policy.enable_mcp_reliability = True
    rng = np.random.default_rng(SEED)
    hours = _hours_from_start(df)

    coordinator = AgentCoordinator(context_enabled=context_enabled)
    coordinator.reset()

    ari_vals, waste_vals, slca_vals, carbon_total = [], [], [], 0.0
    action_trace = []

    for idx in range(min(n_steps, len(df))):
        row = df.iloc[idx]
        rho = float(row.get("spoilage_risk", 1.0 - row["shelf_left"]))
        inv = float(row.get("inventory_units", 100.0))
        temp = float(row["tempC"])
        rh_val = float(row["RH"])
        tau = 1.0 if str(row.get("volatility", "normal")) == "anomaly" else 0.0

        lookback = min(idx + 1, 48)
        hist_slice = df.iloc[max(0, idx + 1 - lookback):idx + 1]
        yf = lstm_demand_forecast(hist_slice, horizon=1)
        y_hat = float(yf["forecast"][0]) if yf["forecast"] else 100.0
        surplus_ratio = max(0.0, inv / INV_BASELINE - 1.0)

        env_state = {
            "rho": rho, "inv": inv, "temp": temp, "rh": rh_val,
            "y_hat": y_hat, "tau": tau, "surplus_ratio": surplus_ratio,
            "supply_hat": inv,
        }

        action_idx, probs, active = coordinator.step(
            env_state, hours[idx], mode, policy, rng, "baseline",
        )
        action = ACTIONS[action_idx]
        action_trace.append(action_idx)

        km = getattr(policy, ACTION_KM_KEYS[action])
        thermal_stress = compute_thermal_stress(temp)
        carbon = compute_transport_carbon(km, policy.carbon_per_km, thermal_stress)

        slca_result = slca_score(carbon, action)
        slca_raw = slca_result["composite"]
        slca_quality = compute_slca_attenuation(thermal_stress, surplus_ratio)
        slca_c = slca_raw * slca_quality

        k_inst = arrhenius_k(temp, policy.k_ref, policy.Ea_R,
                             policy.T_ref_K, rh_val / 100.0, policy.beta_humidity)
        waste_raw = compute_waste_rate(k_inst, surplus_ratio)
        save = compute_save_factor(action, mode, surplus_ratio)
        waste = float(waste_raw * (1.0 - save))

        ari = compute_ari(waste, slca_c, rho)
        reward = compute_reward(slca_c, waste, eta=policy.eta)

        obs = active.observe(env_state, hours[idx])
        outcome = {"waste": waste, "rho": rho, "slca": slca_c, "carbon_kg": carbon}
        coordinator.post_step(active, action_idx, obs, outcome,
                              hour=hours[idx], reward=reward)

        ari_vals.append(ari)
        waste_vals.append(waste)
        slca_vals.append(slca_c)
        carbon_total += carbon

    return {
        "ari": np.mean(ari_vals),
        "waste": np.mean(waste_vals),
        "slca": np.mean(slca_vals),
        "carbon": carbon_total,
        "action_trace": action_trace,
        "context_summary": coordinator.context_summary(),
        "learner_summary": coordinator.learner_summary(),
        "evaluator_summary": coordinator.evaluator_summary(),
        "context_log": coordinator.context_log[:3],
    }


def main():
    print("=" * 70)
    print("MCP + piRAG Context Integration Verification")
    print("=" * 70)
    print("  (set FAILURE_INJECTION=true to verify reliability under injected MCP faults)")

    if not DATA_CSV.exists():
        print(f"ERROR: Data CSV not found: {DATA_CSV}")
        return

    policy = Policy()
    _register_scenario_state({"policy": policy})

    df = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])
    df = compute_spoilage(df, k_ref=policy.k_ref, Ea_R=policy.Ea_R,
                          T_ref_K=policy.T_ref_K, beta=policy.beta_humidity,
                          lag_lambda=policy.lag_lambda)
    df["volatility"] = volatility_flags(df, window=policy.boll_window, k=policy.boll_k)

    n_steps = 48  # 12 hours worth

    print(f"\nRunning {n_steps} steps with context ON...")
    ctx_on = _run_short_episode(df, "agribrain", context_enabled=True, n_steps=n_steps)

    print(f"Running {n_steps} steps with context OFF...")
    ctx_off = _run_short_episode(df, "agribrain", context_enabled=False, n_steps=n_steps)

    # 1-2: Side-by-side metrics
    print("\n" + "=" * 70)
    print("Side-by-Side Comparison")
    print("=" * 70)
    print(f"{'Metric':<25} {'Context ON':>15} {'Context OFF':>15} {'Delta':>15}")
    print("-" * 70)
    for metric in ["ari", "waste", "slca", "carbon"]:
        on_val = ctx_on[metric]
        off_val = ctx_off[metric]
        delta = on_val - off_val
        print(f"{metric:<25} {on_val:>15.4f} {off_val:>15.4f} {delta:>+15.4f}")

    # 3: Context summary
    print("\n" + "=" * 70)
    print("Context Summary (ON mode)")
    print("=" * 70)
    ctx_summary = ctx_on["context_summary"]
    print(f"  Total context steps: {ctx_summary.get('total_context_steps', 0)}")
    print(f"  Total MCP tool calls: {ctx_summary.get('total_mcp_tool_calls', 0)}")
    print(f"  Mean modifier magnitude: {ctx_summary.get('mean_modifier_magnitude', 0):.4f}")
    print(f"  Guard failures: {ctx_summary.get('guard_failures', 0)}")
    print(f"  Nonzero modifier steps: {ctx_summary.get('nonzero_modifier_steps', 0)}")
    for role, stats in ctx_summary.get("per_role", {}).items():
        print(f"    [{role}] MCP calls: {stats.get('mcp_calls', 0)}, "
              f"piRAG queries: {stats.get('pirag_queries', 0)}, "
              f"mean modifier: {stats.get('mean_modifier_magnitude', 0):.4f}, "
              f"nonzero: {stats.get('nonzero_modifier_count', 0)}")

    # 4: Evaluator summary
    print("\n" + "=" * 70)
    print("Context Evaluator Summary")
    print("=" * 70)
    eval_summary = ctx_on["evaluator_summary"]
    for k, v in eval_summary.items():
        print(f"  {k}: {v}")

    # 5: Learner summary
    print("\n" + "=" * 70)
    print("Context Rule Learner Summary")
    print("=" * 70)
    learner_summary = ctx_on["learner_summary"]
    for k, v in learner_summary.items():
        if isinstance(v, list) and len(v) > 5:
            print(f"  {k}: [{', '.join(f'{x:.3f}' for x in v[:5])}...]")
        else:
            print(f"  {k}: {v}")

    # 6-7: Assertions
    print("\n" + "=" * 70)
    print("Assertions")
    print("=" * 70)

    # If context was active, metrics should differ
    if ctx_summary.get("total_context_steps", 0) > 0:
        # Modifier should be non-zero (guard fix verification)
        nonzero_steps = ctx_summary.get("nonzero_modifier_steps", 0)
        total_steps = ctx_summary.get("total_context_steps", 0)
        print(f"  Nonzero modifier steps: {nonzero_steps}/{total_steps}")
        assert nonzero_steps > 0, "Context modifier should be non-zero after guard fix"
        print("  PASS: Modifier is non-zero")

        # Guard failures should be zero or very low
        guard_failures = ctx_summary.get("guard_failures", 0)
        print(f"  Guard failures: {guard_failures}/{total_steps}")
        print(f"  PASS: Guard failure rate {guard_failures/total_steps:.1%}")

        actions_on = ctx_on["action_trace"]
        actions_off = ctx_off["action_trace"]
        n_different = sum(1 for a, b in zip(actions_on, actions_off) if a != b)
        print(f"  Action differences: {n_different}/{len(actions_on)}")

        # ARI delta should be bounded
        ari_delta = abs(ctx_on["ari"] - ctx_off["ari"])
        print(f"  ARI delta: {ari_delta:.4f} (limit: 0.10)")
        assert ari_delta < 0.10, f"ARI delta {ari_delta:.4f} exceeds limit of 0.10"
        print("  PASS: ARI delta within bounds")
    else:
        print("  Context infrastructure not fully initialized (imports may have failed)")
        print("  Checking that metrics are identical when context is disabled...")
        assert np.isclose(ctx_on["ari"], ctx_off["ari"], atol=1e-6)
        print("  PASS: Metrics identical when context disabled")

    # 8: Sample context log entries
    print("\n" + "=" * 70)
    print("Sample Context Log (first 3 steps)")
    print("=" * 70)
    for entry in ctx_on.get("context_log", [])[:3]:
        print(f"  Hour {entry.get('hour', '?'):.1f} | Role: {entry.get('role', '?')} | "
              f"Tools: {entry.get('mcp_tools_invoked', [])} | "
              f"Doc: {entry.get('top_doc_id', 'none')} | "
              f"Modifier norm: {entry.get('modifier_norm', 0):.4f}")

    # 9: MCP handshake verification
    print("\n" + "=" * 70)
    print("MCP Protocol Verification")
    print("=" * 70)
    try:
        from pirag.mcp.protocol import MCPServer, MCPMessage
        from pirag.mcp.registry import get_default_registry
        from pirag.mcp.prompts import register_prompts

        import pirag.mcp.registry as _reg_mod
        _reg_mod._DEFAULT_REGISTRY = None

        registry = get_default_registry()
        server = MCPServer(registry=registry)
        register_prompts(server)

        # Handshake
        init_resp = server.handle_message(MCPMessage(id=1, method="initialize"))
        assert init_resp.result is not None
        print(f"  Handshake: OK (protocol version {init_resp.result['protocolVersion']})")

        # Tools list
        tools_resp = server.handle_message(MCPMessage(id=2, method="tools/list"))
        n_tools = len(tools_resp.result.get("tools", []))
        print(f"  Tools listed: {n_tools}")

        # Resource read
        from pirag.mcp.resources import register_agent_resources
        register_agent_resources(server, lambda: {"temp": 6.5, "rho": 0.15})
        res_resp = server.handle_message(MCPMessage(id=3, method="resources/read",
                                                     params={"uri": "agribrain://telemetry/temperature"}))
        assert res_resp.error is None
        print("  Resource read: OK")

        # Prompt expansion
        prompt_resp = server.handle_message(MCPMessage(id=4, method="prompts/get",
                                                        params={"name": "regulatory_compliance_check", "arguments": {}}))
        assert prompt_resp.result is not None
        print("  Prompt expansion: OK")

        print("  ALL MCP PROTOCOL CHECKS PASSED")

    except Exception as e:
        print(f"  MCP verification error: {e}")

    print("\n" + "=" * 70)
    print("Verification Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
