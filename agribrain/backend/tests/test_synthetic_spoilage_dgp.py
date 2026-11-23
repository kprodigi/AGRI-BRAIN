"""Exact contracts for the reusable independent synthetic spoilage DGP."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mvp.simulation.pinn.generate_synthetic_spoilage_data import _trajectory
from src.models.synthetic_spoilage_dgp import (
    DEFAULT_PACKAGING_INDEX,
    HANDLING_SHOCK_LOG_RATE_COEFFICIENT,
    PACKAGING_CENTER,
    PACKAGING_LOG_RATE_COEFFICIENT,
    RH_TRANSIENT_LOG_RATE_COEFFICIENT,
    SYNTHETIC_DGP_KIND,
    compute_spoilage_independent_synthetic_dgp,
    synthetic_dgp_provenance,
)


def test_backend_dgp_is_bit_exact_for_all_documented_generator_trajectories(
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    manifest = json.loads((
        repo_root
        / "mvp/simulation/pinn/artifacts/synthetic_spoilage_residual_v1_manifest.json"
    ).read_text(encoding="utf-8"))
    regimes = ("cold_chain", "heat_excursion", "oscillatory")

    for index in range(36):
        trajectory_id = f"trajectory_{index + 1:03d}"
        generated = _trajectory(
            trajectory_id,
            int(manifest["trajectory_seeds"][trajectory_id]),
            regimes[index % len(regimes)],
        )
        replay_input = generated.loc[:, [
            "timestamp", "tempC", "RH", "shockG", "packaging_index",
        ]].copy()
        replay = compute_spoilage_independent_synthetic_dgp(
            replay_input,
            k_ref=float(generated["k_ref_per_h"].iloc[0]),
            Ea_R=float(generated["ea_over_r_kelvin"].iloc[0]),
            packaging_index=None,
        )

        expected_quality = generated["latent_quality_fraction"].to_numpy(
            dtype=np.float64,
        )
        actual_quality = replay["shelf_left"].to_numpy(dtype=np.float64)
        np.testing.assert_array_equal(actual_quality, expected_quality)
        np.testing.assert_array_equal(
            replay["latent_quality_fraction"].to_numpy(dtype=np.float64),
            expected_quality,
        )
        np.testing.assert_array_equal(
            replay["spoilage_risk"].to_numpy(dtype=np.float64),
            1.0 - expected_quality,
        )
        np.testing.assert_array_equal(
            replay["latent_spoilage_risk"].to_numpy(dtype=np.float64),
            1.0 - expected_quality,
        )


def test_synthetic_dgp_provenance_is_exact_and_json_native() -> None:
    assert SYNTHETIC_DGP_KIND == "independent_synthetic_dgp_v1"
    assert DEFAULT_PACKAGING_INDEX == PACKAGING_CENTER == 0.50
    assert PACKAGING_LOG_RATE_COEFFICIENT == 0.44
    assert HANDLING_SHOCK_LOG_RATE_COEFFICIENT == 0.80
    assert RH_TRANSIENT_LOG_RATE_COEFFICIENT == 0.0040
    assert synthetic_dgp_provenance() == {
        "schema_version": 1,
        "kind": "independent_synthetic_dgp_v1",
        "role": "common_mode_invariant_noise_free_outcome_reference",
        "target_origin": "independent_synthetic_dgp",
        "synthetic_only": True,
        "external_validation": False,
        "empirical_claims_permitted": False,
        "noise_free": True,
        "state_variable": "remaining_quality_fraction",
        "initial_quality_fraction": 1.0,
        "integration": "midpoint_exponential_state_update",
        "state_equation": (
            "C_i=C_(i-1)*exp(-k_base(T_mid,RH_mid)*alpha(t_mid)*"
            "exp(u_i)*delta_t_i)"
        ),
        "lag_equation": "alpha(t)=t/(t+lag_lambda)",
        "log_rate_multiplier_equation": (
            "u=0.44*(packaging_index-0.50)+0.80*handling_shock_G_mid+"
            "0.0040*abs_dRH_dt_mid"
        ),
        "coefficients": {
            "packaging_center": 0.50,
            "packaging_log_rate": 0.44,
            "handling_shock_log_rate_per_g": 0.80,
            "rh_transient_log_rate_per_pct_per_hour": 0.0040,
        },
        "parameters": {
            "k_ref_per_h": 0.0021,
            "ea_over_r_kelvin": 8000.0,
            "reference_temperature_kelvin": 277.15,
            "humidity_coupling": 0.25,
            "lag_lambda_hours": 12.0,
            "packaging_index": 0.50,
        },
    }


def test_dgp_packaging_is_trajectory_level_and_frame_attrs_are_preserved() -> None:
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=4, freq="15min"),
        "tempC": [4.0, 5.0, 6.0, 7.0],
        "RH": [90.0, 88.0, 91.0, 87.0],
        "shockG": [0.01, 0.03, 0.02, 0.08],
        "packaging_index": [0.30, 0.30, 0.30, 0.30],
    })
    frame.attrs["observation_treatment"] = {"stressor": "nominal"}
    result = compute_spoilage_independent_synthetic_dgp(frame)
    assert result.attrs["observation_treatment"] == {"stressor": "nominal"}
    assert result.attrs["synthetic_spoilage_dgp"] == (
        synthetic_dgp_provenance(packaging_index=0.30)
    )
    assert result["packaging_index"].tolist() == [0.30] * 4

    invalid = frame.copy()
    invalid["packaging_index"] = [0.30, 0.30, 0.31, 0.30]
    with pytest.raises(ValueError, match="trajectory-level"):
        compute_spoilage_independent_synthetic_dgp(invalid)
