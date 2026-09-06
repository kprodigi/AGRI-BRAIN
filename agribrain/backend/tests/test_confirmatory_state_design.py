"""Invariants for the confirmatory observed/latent state design."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SIM_ROOT = REPO_ROOT / "mvp" / "simulation"
for candidate in (REPO_ROOT, SIM_ROOT, REPO_ROOT / "agribrain" / "backend"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from mvp.simulation.benchmarks.run_stress_suite import _perturb_df
from mvp.simulation.generate_results import (
    DATA_CSV,
    Policy,
    _restore_policy_theta_after_call,
    apply_scenario,
    decision_ledger_scope,
    run_episode,
)
from mvp.simulation.stochastic import _DISABLED
from src.models import scenario_engine
from src.models.spoilage import (
    advance_spoilage_risk_midpoint,
    compute_spoilage,
)


def _base() -> pd.DataFrame:
    return pd.read_csv(DATA_CSV, parse_dates=["timestamp"])


def test_unknown_scenario_identifiers_fail_closed():
    frame = _base().head(8)
    policy = Policy()
    with pytest.raises(ValueError, match="unknown scenario"):
        scenario_engine.apply("basline_typo", frame, policy)
    with pytest.raises(ValueError, match="unknown scenario"):
        apply_scenario(
            frame,
            "basline_typo",
            policy,
            np.random.default_rng(1),
        )


def test_midpoint_transition_matches_vector_integrator_and_is_monotone():
    frame = _base().head(40)
    computed = compute_spoilage(frame)
    hours = (
        (computed["timestamp"] - computed["timestamp"].iloc[0])
        .dt.total_seconds().to_numpy() / 3600.0
    )
    online = [0.0]
    for index in range(1, len(computed)):
        online.append(advance_spoilage_risk_midpoint(
            online[-1],
            previous_temp_C=float(computed.iloc[index - 1]["tempC"]),
            current_temp_C=float(computed.iloc[index]["tempC"]),
            previous_rh_pct=float(computed.iloc[index - 1]["RH"]),
            current_rh_pct=float(computed.iloc[index]["RH"]),
            previous_hour=float(hours[index - 1]),
            current_hour=float(hours[index]),
        ))
    assert np.allclose(online, computed["spoilage_risk"].to_numpy(), atol=1e-14)
    assert np.all(np.diff(online) >= -1e-15)


class _OnsetStub:
    enabled = True

    def __init__(self, offset: float):
        self.offset = offset

    def jitter_onset_hour(self, base: float, *, counter: int | None = None) -> float:
        return float(base + self.offset)


def test_scenario_intensity_zero_is_baseline_and_onset_offset_is_signed():
    frame = _base()
    policy = Policy()
    zero_heat = apply_scenario(
        frame, "heatwave", policy, np.random.default_rng(1), intensity=0.0,
    )
    zero_over = apply_scenario(
        frame, "overproduction", policy, np.random.default_rng(1),
        intensity=0.0,
    )
    zero_cyber = apply_scenario(
        frame, "cyber_outage", policy, np.random.default_rng(1),
        intensity=0.0,
    )
    zero_pricing = apply_scenario(
        frame, "adaptive_pricing", policy, np.random.default_rng(1),
        intensity=0.0,
    )
    assert np.allclose(zero_heat["tempC"], frame["tempC"])
    assert np.allclose(zero_over["inventory_units"], frame["inventory_units"])
    assert np.allclose(zero_over["tempC"], frame["tempC"])
    assert np.allclose(zero_cyber["demand_units"], frame["demand_units"])
    assert np.allclose(zero_cyber["tempC"], frame["tempC"])
    assert np.allclose(zero_pricing["demand_units"], frame["demand_units"])
    assert np.allclose(zero_pricing["tempC"], frame["tempC"])

    earlier = apply_scenario(
        frame, "heatwave", policy, np.random.default_rng(2),
        stoch=_OnsetStub(-3.0),
    )
    later = apply_scenario(
        frame, "heatwave", policy, np.random.default_rng(2),
        stoch=_OnsetStub(3.0),
    )
    earlier_changed = np.flatnonzero(
        np.abs(earlier["tempC"].to_numpy() - frame["tempC"].to_numpy()) > 1e-12
    )[0]
    later_changed = np.flatnonzero(
        np.abs(later["tempC"].to_numpy() - frame["tempC"].to_numpy()) > 1e-12
    )[0]
    assert earlier_changed < later_changed
    assert np.array_equal(earlier["timestamp"], frame["timestamp"])
    assert np.array_equal(later["timestamp"], frame["timestamp"])

    hours = (
        (frame["timestamp"] - frame["timestamp"].iloc[0])
        .dt.total_seconds().to_numpy() / 3600.0
    )
    for scenario, field, expected_early, expected_late in (
        ("overproduction", "inventory_units", 6.0, 18.0),
        ("cyber_outage", "demand_units", 18.0, 30.0),
    ):
        shifted_early = apply_scenario(
            frame, scenario, policy, np.random.default_rng(2),
            stoch=_OnsetStub(-6.0),
        )
        shifted_late = apply_scenario(
            frame, scenario, policy, np.random.default_rng(2),
            stoch=_OnsetStub(6.0),
        )
        early_idx = np.flatnonzero(
            np.abs(shifted_early[field].to_numpy() - frame[field].to_numpy())
            > 1e-12
        )[0]
        late_idx = np.flatnonzero(
            np.abs(shifted_late[field].to_numpy() - frame[field].to_numpy())
            > 1e-12
        )[0]
        assert hours[early_idx] == pytest.approx(expected_early)
        assert hours[late_idx] == pytest.approx(expected_late)


def test_scenario_intensity_interpolates_declared_dose():
    frame = _base()
    policy = Policy()
    hours = (
        (frame["timestamp"] - frame["timestamp"].iloc[0])
        .dt.total_seconds().to_numpy() / 3600.0
    )
    at_30 = int(np.flatnonzero(np.isclose(hours, 30.0))[0])

    over_half = apply_scenario(
        frame, "overproduction", policy, np.random.default_rng(1),
        intensity=0.5,
    )
    assert over_half["inventory_units"].iloc[at_30] == pytest.approx(
        frame["inventory_units"].iloc[at_30] * 1.75,
    )
    cyber_half = apply_scenario(
        frame, "cyber_outage", policy, np.random.default_rng(1),
        intensity=0.5,
    )
    assert cyber_half["demand_units"].iloc[at_30] == pytest.approx(
        frame["demand_units"].iloc[at_30] * 0.575,
    )
    heat_half = apply_scenario(
        frame, "heatwave", policy, np.random.default_rng(1), intensity=0.5,
    )
    heat_full = apply_scenario(
        frame, "heatwave", policy, np.random.default_rng(1), intensity=1.0,
    )
    half_delta = heat_half["tempC"].iloc[at_30] - frame["tempC"].iloc[at_30]
    full_delta = heat_full["tempC"].iloc[at_30] - frame["tempC"].iloc[at_30]
    assert half_delta == pytest.approx(0.5 * full_delta)


@pytest.mark.parametrize("invalid", [-0.01, float("nan"), float("inf")])
def test_scenario_controls_reject_invalid_intensity(invalid):
    with pytest.raises(ValueError, match="intensity"):
        apply_scenario(
            _base().head(8), "heatwave", Policy(), np.random.default_rng(1),
            intensity=invalid,
        )


def test_h3_perturbations_encode_observation_only_dose_without_lookahead():
    frame = _base().head(24)
    delayed = _perturb_df(frame, "telemetry_delay", np.random.default_rng(4))
    assert np.array_equal(delayed["tempC"], frame["tempC"])
    assert np.array_equal(delayed["RH"], frame["RH"])
    # ``_perturb_df`` records only the primitive H3 dose. ``run_episode``
    # applies it after constructing the canonical stochastic observation
    # stream, so the stress layer never mutates latent truth or bypasses the
    # declared observation order.
    assert "temp_policy_observed" not in delayed
    assert "rh_policy_observed" not in delayed
    assert delayed["h3_telemetry_source_step_index"].tolist() == [
        max(index - 4, 0) for index in range(len(delayed))
    ]

    missing = _perturb_df(frame, "missing_data", np.random.default_rng(9))
    assert not bool(missing["h3_missing_observation"].iloc[0])
    assert np.array_equal(missing["tempC"], frame["tempC"])
    for latent_field in (
        "timestamp", "tempC", "RH", "inventory_units", "demand_units",
    ):
        assert np.array_equal(missing[latent_field], frame[latent_field])
    missing_dose = missing.attrs["observation_treatment"]
    assert missing_dose["stressor"] == "missing_data"
    assert missing_dose["missing_count"] > 0
    assert len(missing_dose["missing_mask_sha256"]) == 64
    assert len(missing_dose["treatment_sha256"]) == 64
    assert int(missing["h3_missing_observation"].sum()) == (
        missing_dose["missing_count"]
    )

    compounded = _perturb_df(frame, "compounded", np.random.default_rng(12))
    dose = compounded.attrs["observation_treatment"]
    assert dose["delay_steps"] == 4
    assert dose["missing_count"] > 0
    assert len(dose["temp_noise_sha256"]) == 64
    assert len(dose["rh_noise_sha256"]) == 64
    assert {
        "h3_temp_noise_c", "h3_rh_noise_pct",
        "h3_missing_observation", "h3_telemetry_source_step_index",
    }.issubset(compounded.columns)


def test_episode_scores_latent_truth_and_emits_unambiguous_aliases(tmp_path):
    frame = apply_scenario(
        _base().head(24), "baseline", Policy(), np.random.default_rng(1),
    )
    stressed = _perturb_df(frame, "sensor_noise", np.random.default_rng(5))
    with decision_ledger_scope(tmp_path, reset=True):
        episode = run_episode(
            stressed, "static", Policy(), np.random.default_rng(6),
            seed=42, benchmark_seed=42, episode_index=3, stoch=_DISABLED,
        )
    assert episode["rho_trace"] == episode["rho_policy_observed_trace"]
    assert episode["temp_trace"] == episode["temp_policy_observed_trace"]
    assert episode["rho_policy_observed_trace"] != (
        episode["rho_outcome_environmental_trace"]
    )
    for ari, waste, social, rho in zip(
        episode["ari_trace"], episode["waste_trace"], episode["slca_trace"],
        episode["rho_outcome_environmental_trace"],
    ):
        assert np.isclose(ari, (1.0 - waste) * social * (1.0 - rho))
    assert "effective_rho_trace" not in episode
    assert "batch_effective_rho_trace" not in episode
    assert episode["dispatch_opportunity_count"] == len(frame)
    assert episode["simulated_dispatch_accounted_trace"] == [True] * len(frame)
    ledger_lines = Path(episode["decision_ledger_path"]).read_text(
        encoding="utf-8"
    ).splitlines()
    header = json.loads(ledger_lines[0])
    records = [json.loads(line) for line in ledger_lines[1:]]
    assert header["metadata"]["observation_treatment"] == (
        stressed.attrs["observation_treatment"]
    )
    assert all(record["h3_stressor"] == "sensor_noise" for record in records)
    assert [record["h3_temp_noise_c"] for record in records] == (
        stressed["h3_temp_noise_c"].tolist()
    )


def test_failed_run_restores_process_global_policy_prior():
    import src.models.action_selection as action_selection

    original = action_selection.THETA.copy()

    @_restore_policy_theta_after_call
    def fail_after_swap():
        action_selection.THETA = original + 123.0
        raise RuntimeError("synthetic failure")

    with pytest.raises(RuntimeError, match="synthetic failure"):
        fail_after_swap()
    np.testing.assert_array_equal(action_selection.THETA, original)
