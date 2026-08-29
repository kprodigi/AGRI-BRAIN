"""Scientific and provenance contracts for the frozen synthetic PINN."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import src.models.action_selection as action_selection
from src.models.mode_capabilities import capabilities_for
from src.models.outcome_equation_contract import (
    validate_recorded_spoilage_trajectories,
)
from src.models.pinn_residual import (
    FEATURE_NAMES,
    LossWeights,
    build_training_objective,
    compute_spoilage_with_frozen_residual,
    load_frozen_checkpoint,
    pack_parameters,
    residual_prediction_and_jacobian,
)
from src.models.spoilage import compute_spoilage
from src.models.synthetic_spoilage_dgp import (
    compute_spoilage_independent_synthetic_dgp,
    synthetic_dgp_provenance,
)
from mvp.simulation.generate_results import (
    DATA_CSV,
    Policy,
    _stream_id,
    _stream_seed,
    apply_scenario,
    decision_ledger_scope,
    policy_theta_for_seed,
    run_episode,
)
from mvp.simulation.stochastic import make_stochastic_layer


REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = REPO_ROOT / "mvp" / "simulation" / "pinn" / "artifacts"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_synthetic_dataset_is_group_split_and_hash_bound() -> None:
    manifest_path = ARTIFACTS / "synthetic_spoilage_residual_v1_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = ARTIFACTS / manifest["dataset_file"]
    assert _sha256(dataset) == manifest["dataset_sha256"]
    assert manifest["target_origin"] == "independent_synthetic_dgp"
    assert manifest["synthetic_only"] is True
    assert manifest["external_validation"] is False
    assert manifest["empirical_claims_permitted"] is False
    splits = manifest["trajectory_splits"]
    train, validation, test = map(set, (
        splits["train"], splits["validation"], splits["test"],
    ))
    assert (len(train), len(validation), len(test)) == (24, 6, 6)
    assert train.isdisjoint(validation | test)
    assert validation.isdisjoint(test)
    assert len(train | validation | test) == 36


def test_documented_loss_jacobian_matches_finite_difference() -> None:
    rng = np.random.default_rng(918273)
    X = rng.normal(size=(8, len(FEATURE_NAMES)))
    W = rng.normal(0.0, 0.15, size=(len(FEATURE_NAMES), 4))
    b = rng.normal(0.0, 0.05, size=4)
    v = rng.normal(0.0, 0.05, size=4)
    theta = pack_parameters(W, b, v, 0.01)
    _, analytic = residual_prediction_and_jacobian(X, theta, hidden_size=4)
    epsilon = 1e-6
    for column in (0, 7, 13, len(theta) - 1):
        plus = theta.copy()
        minus = theta.copy()
        plus[column] += epsilon
        minus[column] -= epsilon
        y_plus, _ = residual_prediction_and_jacobian(X, plus, hidden_size=4)
        y_minus, _ = residual_prediction_and_jacobian(X, minus, hidden_size=4)
        numerical = (y_plus - y_minus) / (2.0 * epsilon)
        np.testing.assert_allclose(analytic[:, column], numerical, rtol=2e-5, atol=2e-8)


def test_frozen_checkpoint_and_heldout_metrics_are_valid() -> None:
    checkpoint = load_frozen_checkpoint()
    manifest = json.loads(
        (ARTIFACTS / "spoilage_pinn_v1_manifest.json").read_text(encoding="utf-8")
    )
    assert checkpoint.checkpoint_sha256 == manifest["checkpoint_sha256"]
    test = manifest["metrics"]["test"]
    assert test["latent_rmse"] < test["mechanistic_latent_rmse"]
    assert test["residual_abs_max"] <= 0.08
    assert test["deployed_unit_interval_violation_count"] == 0
    assert test["deployed_monotonicity_violation_count"] == 0


def test_no_pinn_is_a_clean_one_factor_ablation() -> None:
    full = capabilities_for("agribrain")
    no_pinn = capabilities_for("no_pinn")
    full_dict = dict(full.__dict__)
    ablated_dict = dict(no_pinn.__dict__)
    assert full_dict.pop("spoilage_residual") is True
    assert ablated_dict.pop("spoilage_residual") is False
    assert ablated_dict == full_dict


def test_frozen_residual_inference_is_deterministic_and_bounded() -> None:
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=12, freq="15min"),
        "tempC": np.linspace(4.0, 10.0, 12),
        "RH": np.linspace(90.0, 84.0, 12),
        "shockG": np.linspace(0.01, 0.08, 12),
    })
    first = compute_spoilage_with_frozen_residual(frame)
    second = compute_spoilage_with_frozen_residual(frame)
    np.testing.assert_array_equal(first["shelf_left"], second["shelf_left"])
    correction = first["pinn_residual_correction"].to_numpy(float)
    assert np.max(np.abs(correction)) <= 0.08 + 1e-12
    assert np.all(np.diff(first["shelf_left"].to_numpy(float)) <= 1e-12)


def test_ledger_replay_requires_frozen_provenance_and_clean_no_pinn() -> None:
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=12, freq="15min"),
        "tempC": np.linspace(4.0, 10.0, 12),
        "RH": np.linspace(90.0, 84.0, 12),
        "shockG": np.linspace(0.01, 0.08, 12),
    })
    checkpoint = load_frozen_checkpoint()
    corrected = compute_spoilage_with_frozen_residual(frame, checkpoint=checkpoint)
    mechanistic = compute_spoilage(frame)
    latent = compute_spoilage_independent_synthetic_dgp(frame)
    hours = np.arange(len(frame), dtype=float) * 0.25

    def records_for(
        outcome_rho: np.ndarray,
        policy_rho: np.ndarray,
    ) -> list[dict[str, float]]:
        return [{
            "hour": float(hours[index]),
            "temp_outcome_environmental": float(frame["tempC"].iloc[index]),
            "rh_outcome_environmental": float(frame["RH"].iloc[index]),
            "temp_policy_observed": float(frame["tempC"].iloc[index]),
            "rh_policy_observed": float(frame["RH"].iloc[index]),
            "shock_g": float(frame["shockG"].iloc[index]),
            "rho_outcome_environmental": float(outcome_rho[index]),
            "rho_policy_observed": float(policy_rho[index]),
        } for index in range(len(frame))]

    contract = {"arrhenius": {
        "effective_k_ref": 0.0021,
        "effective_ea_over_r": 8000.0,
        "reference_temperature_k": 277.15,
        "humidity_coupling": 0.25,
        "rational_lag_hours": 12.0,
    }}
    residual_metadata = {
        "kind": "mechanistic_plus_frozen_synthetic_pinn_residual",
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "training_dataset_sha256": checkpoint.dataset_sha256,
        "training_target_origin": "independent_synthetic_dgp",
        "residual_bound_abs": 0.08,
        "deployment_transform": (
            "clip_quality_to_unit_interval_then_cumulative_minimum"
        ),
        "synthetic_only": True,
        "external_validation": False,
    }
    latent_metadata = synthetic_dgp_provenance()
    validate_recorded_spoilage_trajectories(
        records_for(
            latent["spoilage_risk"].to_numpy(float),
            corrected["spoilage_risk"].to_numpy(float),
        ),
        contract,
        spoilage_estimator=residual_metadata,
        latent_spoilage_model=latent_metadata,
        contract_validated=True,
    )
    invalid_metadata = dict(residual_metadata, checkpoint_sha256="0" * 64)
    with np.testing.assert_raises_regex(ValueError, "checkpoint SHA-256"):
        validate_recorded_spoilage_trajectories(
            records_for(
                latent["spoilage_risk"].to_numpy(float),
                corrected["spoilage_risk"].to_numpy(float),
            ),
            contract,
            spoilage_estimator=invalid_metadata,
            latent_spoilage_model=latent_metadata,
            contract_validated=True,
        )
    validate_recorded_spoilage_trajectories(
        records_for(
            latent["spoilage_risk"].to_numpy(float),
            mechanistic["spoilage_risk"].to_numpy(float),
        ),
        contract,
        spoilage_estimator={
            "kind": "mechanistic_only_no_pinn",
            "checkpoint_sha256": None,
            "training_dataset_sha256": None,
            "training_target_origin": None,
            "residual_bound_abs": None,
            "deployment_transform": None,
            "synthetic_only": True,
            "external_validation": False,
        },
        latent_spoilage_model=latent_metadata,
        contract_validated=True,
    )
    with np.testing.assert_raises_regex(
        ValueError, "independent synthetic DGP",
    ):
        validate_recorded_spoilage_trajectories(
            records_for(
                latent["spoilage_risk"].to_numpy(float),
                mechanistic["spoilage_risk"].to_numpy(float),
            ),
            contract,
            spoilage_estimator={
                "kind": "mechanistic_only_no_pinn",
                "checkpoint_sha256": None,
                "training_dataset_sha256": None,
                "training_target_origin": None,
                "residual_bound_abs": None,
                "deployment_transform": None,
                "synthetic_only": True,
                "external_validation": False,
            },
            latent_spoilage_model={**latent_metadata, "noise_free": False},
            contract_validated=True,
        )


def test_paired_agribrain_no_pinn_share_dgp_but_not_policy_estimator(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The no-PINN arm changes policy information, never scored truth."""

    seed = 17
    scenario = "baseline"
    policy = Policy()
    base = pd.read_csv(DATA_CSV, parse_dates=["timestamp"]).iloc[:24].copy()
    environment_seed = _stream_seed(seed, scenario, 3, "environment")
    layer = make_stochastic_layer(
        np.random.default_rng(environment_seed), stream_seed=environment_seed,
    )
    frame = apply_scenario(
        base,
        scenario,
        policy,
        np.random.default_rng(_stream_seed(seed, scenario, 3, "scenario")),
        stoch=layer,
    )
    monkeypatch.setattr(
        action_selection,
        "THETA",
        policy_theta_for_seed(
            np.asarray(action_selection.DECLARED_THETA, dtype=float), seed,
        ),
    )
    results = {}
    with decision_ledger_scope(tmp_path, reset=True):
        for mode in ("agribrain", "no_pinn"):
            results[mode] = run_episode(
                frame.copy(),
                mode,
                policy,
                np.random.default_rng(_stream_seed(seed, scenario, 3, "policy")),
                scenario=scenario,
                stoch=layer,
                seed=seed,
                benchmark_seed=seed,
                episode_index=3,
                environment_stream_id=_stream_id(
                    seed, scenario, 3, "environment",
                ),
                policy_stream_id=_stream_id(seed, scenario, 3, "policy"),
                stochastic_stream_id=_stream_id(
                    seed, scenario, 3, "environment",
                ),
                learning_enabled=False,
            )

    agribrain = results["agribrain"]
    no_pinn = results["no_pinn"]
    assert agribrain["latent_spoilage_model"] == no_pinn[
        "latent_spoilage_model"
    ]
    assert agribrain["latent_spoilage_model"]["kind"] == (
        "independent_synthetic_dgp_v1"
    )
    assert agribrain["latent_environment_sha256"] == no_pinn[
        "latent_environment_sha256"
    ]
    np.testing.assert_array_equal(
        agribrain["rho_outcome_environmental_trace"],
        no_pinn["rho_outcome_environmental_trace"],
    )
    assert agribrain["rho_policy_observed_trace"] != no_pinn[
        "rho_policy_observed_trace"
    ]
    assert agribrain["observed_policy_input_sha256"] != no_pinn[
        "observed_policy_input_sha256"
    ]


def test_full_loss_jacobian_propagates_all_terms() -> None:
    dataset_manifest = json.loads(
        (ARTIFACTS / "synthetic_spoilage_residual_v1_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    frame = pd.read_csv(ARTIFACTS / dataset_manifest["dataset_file"])
    trajectory = dataset_manifest["trajectory_splits"]["train"][0]
    selected = frame[frame["trajectory_id"] == trajectory]
    all_train = frame[frame["split"] == "train"]
    feature_mean = all_train.loc[:, FEATURE_NAMES].to_numpy(float).mean(axis=0)
    feature_scale = all_train.loc[:, FEATURE_NAMES].to_numpy(float).std(axis=0)
    objective = build_training_objective(
        selected,
        trajectory_ids=[trajectory],
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        hidden_size=3,
        weights=LossWeights(),
    )
    rng = np.random.default_rng(17)
    theta = pack_parameters(
        rng.normal(0.0, 0.08, size=(len(FEATURE_NAMES), 3)),
        np.zeros(3),
        rng.normal(0.0, 0.03, size=3),
        0.0,
    )
    residuals, jacobian = objective.residuals_and_jacobian(theta)
    assert residuals.shape[0] == jacobian.shape[0]
    epsilon = 1e-6
    for column in (0, len(theta) - 1):
        plus, minus = theta.copy(), theta.copy()
        plus[column] += epsilon
        minus[column] -= epsilon
        numerical = (objective.residuals(plus) - objective.residuals(minus)) / (
            2.0 * epsilon
        )
        np.testing.assert_allclose(
            jacobian[:, column], numerical, rtol=3e-4, atol=3e-8,
        )
