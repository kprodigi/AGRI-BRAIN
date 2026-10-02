"""End-to-end contracts for waste, carbon, SLCA, and compute footprint."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
SIM_ROOT = REPO_ROOT / "mvp" / "simulation"
for candidate in (REPO_ROOT, SIM_ROOT, REPO_ROOT / "agribrain" / "backend"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from mvp.simulation.generate_results import (  # noqa: E402
    DATA_CSV,
    Policy,
    _stream_id,
    apply_scenario,
    decision_ledger_scope,
    run_episode,
)
from mvp.simulation.stochastic import _DISABLED  # noqa: E402


def test_episode_uses_one_15_minute_routing_unit_and_per_unit_slca_cap(tmp_path):
    policy = Policy(carbon_cap=25.0)
    base = pd.read_csv(DATA_CSV, parse_dates=["timestamp"]).head(8)
    frame = apply_scenario(
        base,
        "baseline",
        policy,
        np.random.default_rng(10),
    )
    with decision_ledger_scope(tmp_path, reset=True):
        episode = run_episode(
            frame,
            "static",
            policy,
            np.random.default_rng(11),
            seed=11,
            benchmark_seed=11,
            episode_index=3,
            stoch=_DISABLED,
            environment_stream_id=_stream_id(11, "baseline", 3, "environment"),
            policy_stream_id=_stream_id(11, "baseline", 3, "policy"),
            stochastic_stream_id=_stream_id(11, "baseline", 3, "environment"),
        )

    assert episode["dispatch_cadence_hours"] == 0.25
    assert episode["dispatch_opportunity_count"] == len(frame)
    assert episode["endpoint_unit"] == "standardized_routing_opportunity"
    assert episode["functional_unit"] == (
        "one standardized batch-routing opportunity per 15-minute row"
    )
    assert episode["shipment_interpretation"] == (
        "synthetic activity unit; not evidence of 288 measured shipments"
    )
    assert episode[
        "waste_cap_fraction_after_surplus_amplification"
    ] == 0.15
    assert episode["slca_carbon_cap_kg_per_routing_opportunity"] == 25.0
    assert episode["slca_carbon_basis"] == (
        "per_routing_opportunity_action_emissions_proxy_kgCO2e"
    )
    assert episode["carbon_efficiency_definition"] == (
        "episode_mean_ari/episode_summed_modeled_transport_emissions_"
        "indicator_kgCO2e; no factor of 1000"
    )
    assert episode["carbon_efficiency_ari_per_kgco2e_proxy"] == (
        episode["ari"] / episode["carbon"]
    )
    carbon_model = episode["transport_carbon_model"]
    assert carbon_model["route_km_by_action_before_stochastic_multiplier"] == {
        "cold_chain": 120.0,
        "local_redistribute": 45.0,
        "recovery": 80.0,
    }
    assert carbon_model["carbon_per_km_kgCO2e"] == 0.12
    assert carbon_model["physical_efficiency_factor"] == 1.0
    assert carbon_model["experimental_mode_multiplier_present"] is False

    for carbon, components in zip(
        episode["carbon_trace"], episode["slca_component_trace"], strict=True,
    ):
        # slca_score deliberately serializes components to four decimals.
        assert np.isclose(
            components["C"],
            max(0.0, 1.0 - carbon / 25.0),
            rtol=0.0,
            atol=5e-5,
        )
    for social, components in zip(
        episode["slca_trace"], episode["slca_component_trace"], strict=True,
    ):
        assert components["composite_attenuated"] == social
    assert episode["equity_trace"][0] == episode["slca_trace"][0]


def test_episode_compute_footprint_uses_timed_action_selection_only(tmp_path):
    policy = Policy()
    base = pd.read_csv(DATA_CSV, parse_dates=["timestamp"]).head(8)
    frame = apply_scenario(
        base,
        "baseline",
        policy,
        np.random.default_rng(20),
    )
    with decision_ledger_scope(tmp_path, reset=True):
        episode = run_episode(
            frame,
            "static",
            policy,
            np.random.default_rng(21),
            seed=21,
            benchmark_seed=21,
            episode_index=3,
            stoch=_DISABLED,
            environment_stream_id=_stream_id(21, "baseline", 3, "environment"),
            policy_stream_id=_stream_id(21, "baseline", 3, "policy"),
            stochastic_stream_id=_stream_id(21, "baseline", 3, "environment"),
        )

    footprint = episode["footprint"]
    assert footprint["timed_call_count"] == len(frame)
    assert footprint["total_steps"] == len(frame)
    assert footprint["time_based_estimate_available"] is True
    assert footprint["measurement_scope"].startswith(
        "coordinator.step action-selection wall time only"
    )
    assert footprint["proxy_step_unit"] == "standardized routing opportunity"
    assert np.isclose(
        footprint["cumulative_energy_J"],
        footprint["assumed_active_power_W"]
        * footprint["cumulative_elapsed_seconds"],
        rtol=0.0,
        atol=1e-7,
    )
    assert np.isclose(
        footprint["cumulative_water_L"],
        footprint["water_rate_L_per_server_second"]
        * footprint["cumulative_elapsed_seconds"],
        rtol=0.0,
        atol=1e-12,
    )
    assert footprint["cumulative_water_per_step_proxy_L"] == (
        len(frame) * 1.8e-6
    )
    assert footprint["cumulative_water_L"] != (
        footprint["cumulative_water_per_step_proxy_L"]
    )
