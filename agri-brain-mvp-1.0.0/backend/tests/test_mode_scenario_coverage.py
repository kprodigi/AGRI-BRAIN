"""End-to-end smoke tests: every declared mode runs against every scenario.

Previous audits found silent bugs where new modes added to ``MODES`` in
``mvp/simulation/generate_results.py`` were missing from other mode-indexed
structures elsewhere (``VALID_MODES`` in ``action_selection.py``,
``_PINN_MODES`` and ``_MCP_WASTE_MODES`` in ``generate_results.py``,
``CYBER_REROUTE_PROB`` in ``action_selection.py``). Each gap was only caught
by a manual sweep, because no test exercised the full (scenario, mode)
matrix.

This module adds a cheap end-to-end matrix so future additions to MODES
automatically surface the same class of gap in CI. The dataframe is
truncated so the full 5 x 8 matrix finishes in a few seconds on a laptop.
A second focused test verifies the invariant that cyber_outage reports
the Bernoulli policy distribution rather than a sampled one-hot.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def sim_runtime():
    """Lazy-import the simulation modules with the sim dir on sys.path."""
    backend = Path(__file__).resolve().parents[1]
    sim = backend.parent.parent / "mvp" / "simulation"
    if str(sim) not in sys.path:
        sys.path.insert(0, str(sim))
    import generate_results as gr
    import stochastic as st
    return gr, st


@pytest.fixture(scope="module")
def short_df(sim_runtime):
    """8-step dataframe (2 h of telemetry) keeps every smoke test under a
    few seconds while still exercising the per-step MCP + piRAG + policy
    path end-to-end."""
    gr, _ = sim_runtime
    df = pd.read_csv(gr.DATA_CSV, parse_dates=["timestamp"]).head(8)
    return df


SCENARIOS = ("heatwave", "overproduction", "cyber_outage", "adaptive_pricing", "baseline")
# Eight canonical modes + seven paper-§4.7 ablation modes. cold_start and
# the six pert_* / pert_*_static modes are paper-defense ablations (zero-init
# learning and prior-perturbation sensitivity with and without REINFORCE,
# respectively) but they run through the same coordinator and policy code
# paths as agribrain, so they must satisfy the same VALID_MODES /
# CYBER_REROUTE_PROB / context-enabled invariants the other agribrain-family
# modes satisfy.
MODES = ("agribrain", "mcp_only", "pirag_only", "no_context",
         "static", "hybrid_rl", "no_pinn", "no_slca",
         "agribrain_cold_start",
         "agribrain_pert_10", "agribrain_pert_25", "agribrain_pert_50",
         "agribrain_pert_10_static", "agribrain_pert_25_static",
         "agribrain_pert_50_static")


# ---------------------------------------------------------------------------
# Fast consistency tests. No episode run; these finish in milliseconds and
# catch the class of silent bug where a new mode was added to ``MODES`` in
# the simulator but forgotten in one of the downstream mode-indexed
# structures. They run in the default ``pytest`` invocation.
# ---------------------------------------------------------------------------

def test_simulator_modes_match_canonical_list(sim_runtime):
    """``generate_results.MODES`` must match the canonical 8-mode list
    above; if it drifts, the slow matrix below would silently skip the
    new mode and downstream lookups would not be exercised."""
    gr, _ = sim_runtime
    assert tuple(sorted(gr.MODES)) == tuple(sorted(MODES))


def test_valid_modes_covers_every_simulator_mode(sim_runtime):
    """Every mode the simulator iterates over must be accepted by
    ``select_action.VALID_MODES``; otherwise run_episode raises
    ``ValueError`` the first time that mode reaches the policy."""
    from src.models.action_selection import VALID_MODES
    gr, _ = sim_runtime
    missing = set(gr.MODES) - set(VALID_MODES)
    assert not missing, f"simulator modes missing from VALID_MODES: {sorted(missing)}"


def test_cyber_reroute_prob_covers_every_non_static_mode(sim_runtime):
    """Every simulator mode that can reach the cyber_outage branch (i.e.
    every non-static mode) must have an entry in ``CYBER_REROUTE_PROB``;
    otherwise the ``.get(mode, 0.50)`` default silently handicaps that
    mode's reroute success rate."""
    from src.models.action_selection import CYBER_REROUTE_PROB
    gr, _ = sim_runtime
    # static returns cold-chain before reaching the cyber branch, so it is
    # intentionally absent from the lookup.
    expected = set(gr.MODES) - {"static"}
    missing = expected - set(CYBER_REROUTE_PROB)
    assert not missing, (
        f"non-static modes missing from CYBER_REROUTE_PROB: {sorted(missing)}"
    )


def test_core_context_modes_wired_together(sim_runtime):
    """The four context-enabled modes share agribrain's infrastructure;
    they must appear in every mode-indexed set that agribrain does."""
    gr, _ = sim_runtime
    must_contain_agribrain = {
        "_CONTEXT_ENABLED_MODES": gr._CONTEXT_ENABLED_MODES,
        "_AGRIBRAIN_LOGIT_MODES": gr._AGRIBRAIN_LOGIT_MODES,
        "_PINN_MODES": gr._PINN_MODES,
        "_MCP_WASTE_MODES": gr._MCP_WASTE_MODES,
    }
    for name, values in must_contain_agribrain.items():
        assert "agribrain" in values, f"{name} missing agribrain"
    # mcp_only / pirag_only / no_context share agribrain logits but
    # their membership in the context-enabled set is stricter. The paper-
    # §4.7 ablation modes (cold_start, pert_*) also join the context-
    # enabled set because they exercise the same MCP + piRAG + learner
    # pipeline, differing only in THETA_CONTEXT initialization.
    assert gr._AGRIBRAIN_LOGIT_MODES >= {
        "agribrain", "no_context", "mcp_only", "pirag_only",
    }
    assert gr._CONTEXT_ENABLED_MODES >= {"agribrain", "mcp_only", "pirag_only"}
    # no_context is NOT in the context-enabled set (by design — it is the
    # Theta=0 frozen counterfactual for §4.7).
    assert "no_context" not in gr._CONTEXT_ENABLED_MODES


# ---------------------------------------------------------------------------
# Slow end-to-end matrix. Each test instantiates the full coordinator,
# dispatches MCP tools and piRAG retrieval per step, and runs the real
# scoring pipeline. Opt-in via ``pytest -m slow``.
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("mode", MODES)
def test_every_mode_runs_on_baseline(sim_runtime, short_df, mode):
    """Running each declared mode on the baseline scenario must not crash.

    This is the minimal smoke test that catches silent bugs where a new
    mode is missing from ``VALID_MODES``, ``_PINN_MODES``, or similar
    mode-indexed sets.
    """
    gr, st = sim_runtime
    policy = gr.Policy()
    scen_rng = np.random.default_rng(0)
    scen_stoch = st.make_stochastic_layer(np.random.default_rng(1))
    df = gr.apply_scenario(short_df, "baseline", policy, scen_rng, stoch=scen_stoch)

    ep = gr.run_episode(df, mode, policy, np.random.default_rng(42),
                         "baseline", stoch=scen_stoch)

    for metric in ("ari", "waste", "slca", "equity", "rle"):
        value = float(ep[metric])
        assert 0.0 <= value <= 1.0, f"{mode} baseline {metric}={value} outside [0, 1]"
    assert float(ep["carbon"]) >= 0.0
    assert len(ep["prob_trace"]) == len(df)
    for probs in ep["prob_trace"]:
        assert abs(sum(probs) - 1.0) < 1e-6
        assert all(p >= 0.0 for p in probs)


@pytest.mark.slow
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_scenario_runs_under_agribrain(sim_runtime, short_df, scenario):
    """Running agribrain against each scenario must not crash.

    Catches scenario-mode wiring bugs in the opposite direction: a new
    scenario handler that forgets to handle the context-enabled modes.
    """
    gr, st = sim_runtime
    policy = gr.Policy()
    scen_rng = np.random.default_rng(0)
    scen_stoch = st.make_stochastic_layer(np.random.default_rng(1))
    df = gr.apply_scenario(short_df, scenario, policy, scen_rng, stoch=scen_stoch)

    ep = gr.run_episode(df, "agribrain", policy, np.random.default_rng(42),
                         scenario, stoch=scen_stoch)

    assert 0.0 <= float(ep["ari"]) <= 1.0
    assert len(ep["prob_trace"]) == len(df)


@pytest.mark.slow
def test_agribrain_under_cyber_outage_runs(sim_runtime, short_df):
    """Explicit coverage for agribrain x cyber_outage."""
    gr, st = sim_runtime
    policy = gr.Policy()
    scen_rng = np.random.default_rng(0)
    scen_stoch = st.make_stochastic_layer(np.random.default_rng(1))
    df = gr.apply_scenario(short_df, "cyber_outage", policy, scen_rng, stoch=scen_stoch)
    ep = gr.run_episode(df, "agribrain", policy, np.random.default_rng(42),
                         "cyber_outage", stoch=scen_stoch)
    assert 0.0 <= float(ep["ari"]) <= 1.0


@pytest.mark.slow
def test_cyber_outage_reports_bernoulli_policy(sim_runtime):
    """Post-fix invariant: during a cyber outage the reported policy
    distribution must be the Bernoulli reroute distribution [1 - p, p, 0],
    not a sampled one-hot. Regression guard for commit 59dbc1c.

    agribrain's reroute success probability is 0.82, so every outage-step
    ``prob_trace`` entry must equal [0.18, 0.82, 0.0] with zero recovery
    mass.
    """
    gr, st = sim_runtime
    policy = gr.Policy()
    scen_rng = np.random.default_rng(0)
    scen_stoch = st.make_stochastic_layer(np.random.default_rng(1))
    # Need enough rows past the 24 h outage onset. 100 rows at 15-min
    # cadence = 25 h, so at least a handful of steps fall in the outage.
    full = pd.read_csv(gr.DATA_CSV, parse_dates=["timestamp"]).head(100)
    df = gr.apply_scenario(full, "cyber_outage", policy, scen_rng, stoch=scen_stoch)
    ep = gr.run_episode(df, "agribrain", policy, np.random.default_rng(42),
                         "cyber_outage", stoch=scen_stoch)

    outage_probs = [
        probs for probs, hour in zip(ep["prob_trace"], ep["hours"])
        if hour >= 24.0
    ]
    assert outage_probs, "expected at least one step under outage"
    for probs in outage_probs:
        assert abs(sum(probs) - 1.0) < 1e-9
        assert probs[2] == 0.0, "recovery mass must be zero under outage"
        assert abs(probs[0] - 0.18) < 1e-9, f"cold-chain mass {probs[0]} != 0.18"
        assert abs(probs[1] - 0.82) < 1e-9, f"local mass {probs[1]} != 0.82"
