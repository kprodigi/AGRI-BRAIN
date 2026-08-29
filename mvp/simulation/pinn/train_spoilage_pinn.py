#!/usr/bin/env python3
"""Train and freeze the synthetic spoilage PINN residual exactly once."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "agribrain" / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.models.pinn_residual import (  # noqa: E402
    FEATURE_NAMES,
    LossWeights,
    build_training_objective,
    pack_parameters,
    residual_prediction_and_jacobian,
    unpack_parameters,
)


TRAINING_SEEDS = (104729, 130363, 155921)
HIDDEN_SIZE = 12
MAX_ITERATIONS = 150
LOSS_WEIGHTS = LossWeights(
    data=1.0,
    physics=0.20,
    boundary=1.0,
    regularization=0.01,
    monotonicity=0.20,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initialize(seed: int, n_features: int, hidden_size: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    limit = np.sqrt(6.0 / (n_features + hidden_size))
    W = rng.uniform(-limit, limit, size=(n_features, hidden_size))
    b = np.zeros(hidden_size, dtype=np.float64)
    v = rng.normal(0.0, 0.015, size=hidden_size)
    return pack_parameters(W, b, v, 0.0)


def objective_value_and_gradient(objective, theta: np.ndarray) -> tuple[float, np.ndarray]:
    """Return 1/2 ||r(theta)||^2 and the exact chain-rule gradient J^T r."""

    residuals, jacobian = objective.residuals_and_jacobian(theta)
    return 0.5 * float(np.dot(residuals, residuals)), jacobian.T @ residuals


def _load_dataset(artifact_dir: Path) -> tuple[pd.DataFrame, dict]:
    manifest_path = artifact_dir / "synthetic_spoilage_residual_v1_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_path = artifact_dir / manifest["dataset_file"]
    if sha256(dataset_path) != manifest["dataset_sha256"]:
        raise ValueError("synthetic PINN dataset hash mismatch")
    if manifest.get("target_origin") != "independent_synthetic_dgp":
        raise ValueError("synthetic PINN target provenance is invalid")
    if manifest.get("synthetic_only") is not True:
        raise ValueError("synthetic PINN dataset lacks the synthetic-only label")
    frame = pd.read_csv(dataset_path)
    required = {
        "trajectory_id", "split", "observed_quality_fraction",
        "latent_quality_fraction", "mechanistic_quality_fraction", *FEATURE_NAMES,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"synthetic PINN dataset is missing columns: {missing}")
    if len(frame) != manifest["n_rows"]:
        raise ValueError("synthetic PINN dataset row count changed")
    return frame, manifest


def _split_ids(manifest: dict, key: str) -> list[str]:
    return [str(value) for value in manifest["trajectory_splits"][key]]


def train(artifact_dir: Path, *, force: bool = False) -> tuple[Path, Path, Path]:
    checkpoint_path = artifact_dir / "spoilage_pinn_v1.npz"
    checkpoint_manifest_path = artifact_dir / "spoilage_pinn_v1_manifest.json"
    history_path = artifact_dir / "spoilage_pinn_v1_training_history.json"
    if not force and any(
        path.exists() for path in (checkpoint_path, checkpoint_manifest_path, history_path)
    ):
        raise FileExistsError("frozen PINN artifacts already exist; pass --force to replace")

    frame, dataset_manifest = _load_dataset(artifact_dir)
    train_ids = _split_ids(dataset_manifest, "train")
    validation_ids = _split_ids(dataset_manifest, "validation")
    test_ids = _split_ids(dataset_manifest, "test")
    train_frame = frame[frame["trajectory_id"].astype(str).isin(train_ids)]
    train_features = train_frame.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
    feature_mean = train_features.mean(axis=0)
    feature_scale = train_features.std(axis=0, ddof=0)
    if np.any(feature_scale <= 1e-12):
        raise ValueError("PINN training feature has zero scale")

    objectives = {
        split: build_training_objective(
            frame,
            trajectory_ids=identifiers,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            hidden_size=HIDDEN_SIZE,
            weights=LOSS_WEIGHTS,
        )
        for split, identifiers in (
            ("train", train_ids),
            ("validation", validation_ids),
            ("test", test_ids),
        )
    }
    latent_by_split = {
        split: frame[frame["trajectory_id"].astype(str).isin(identifiers)]
        .sort_values(["trajectory_id", "time_h"], kind="stable")[
            "latent_quality_fraction"
        ].to_numpy(dtype=np.float64)
        for split, identifiers in (
            ("train", train_ids),
            ("validation", validation_ids),
            ("test", test_ids),
        )
    }

    candidates = []
    fitted_parameters: dict[int, np.ndarray] = {}
    for seed in TRAINING_SEEDS:
        initial = initialize(seed, len(FEATURE_NAMES), HIDDEN_SIZE)
        initial_train = objectives["train"].metrics(
            initial, latent_quality=latent_by_split["train"],
        )
        result = minimize(
            lambda parameters: objective_value_and_gradient(
                objectives["train"], parameters,
            ),
            initial,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": MAX_ITERATIONS,
                "ftol": 1e-14,
                "gtol": 1e-9,
                "maxls": 40,
            },
        )
        fitted_parameters[seed] = result.x.copy()
        metrics = {
            split: objectives[split].metrics(
                result.x, latent_quality=latent_by_split[split],
            )
            for split in ("train", "validation")
        }
        candidates.append({
            "initialization_seed": seed,
            "optimizer_success": bool(result.success),
            "optimizer_status": int(result.status),
            "optimizer_message": str(result.message),
            "iterations": int(result.nit),
            "function_and_gradient_evaluations": int(result.nfev),
            "initial_train_metrics": initial_train,
            "final_metrics": metrics,
            "weighted_loss": float(result.fun),
        })
        print(
            f"initialization {seed}: iterations={result.nit}, "
            f"validation_rmse={metrics['validation']['observed_rmse']:.8f}",
            flush=True,
        )

    # Validation selects the initialization.  The held-out test split is read
    # only after this selection and is never used to choose hyperparameters.
    selected = min(
        candidates,
        key=lambda candidate: (
            candidate["final_metrics"]["validation"]["observed_rmse"],
            candidate["initialization_seed"],
        ),
    )
    selected_seed = int(selected["initialization_seed"])
    theta = fitted_parameters[selected_seed]
    selected_metrics = {
        split: objectives[split].metrics(
            theta, latent_quality=latent_by_split[split],
        )
        for split in ("train", "validation", "test")
    }
    if not selected_metrics["test"]["latent_rmse"] < selected_metrics["test"][
        "mechanistic_latent_rmse"
    ]:
        raise RuntimeError("frozen residual does not improve held-out latent RMSE")
    if selected_metrics["test"]["residual_abs_max"] > 0.08 + 1e-12:
        raise RuntimeError("frozen residual violates its structural output bound")

    W, b, v, c = unpack_parameters(
        theta, n_features=len(FEATURE_NAMES), hidden_size=HIDDEN_SIZE,
    )
    np.savez_compressed(
        checkpoint_path,
        W=W,
        b=b,
        v=v,
        c=np.asarray(c, dtype=np.float64),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
    )
    checkpoint_hash = sha256(checkpoint_path)
    history = {
        "schema_version": 1,
        "dataset_sha256": dataset_manifest["dataset_sha256"],
        "training_seeds": list(TRAINING_SEEDS),
        "selection_rule": "minimum validation observed-quality RMSE; seed breaks ties",
        "selected_initialization_seed": selected_seed,
        "test_split_used_for_selection": False,
        "optimizer": {
            "name": "scipy.optimize.minimize",
            "method": "L-BFGS-B",
            "gradient": "analytic J^T r for all documented loss terms",
            "max_iterations": MAX_ITERATIONS,
            "ftol": 1e-14,
            "gtol": 1e-9,
            "max_line_search_steps": 40,
        },
        "candidates": candidates,
        "selected_metrics": selected_metrics,
    }
    history_path.write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "status": "frozen_for_confirmatory_simulation",
        "target_origin": "independent_synthetic_dgp",
        "synthetic_only": True,
        "external_validation": False,
        "empirical_claims_permitted": False,
        "dataset_file": dataset_manifest["dataset_file"],
        "dataset_sha256": dataset_manifest["dataset_sha256"],
        "dataset_manifest_file": "synthetic_spoilage_residual_v1_manifest.json",
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_hash,
        "training_history_file": history_path.name,
        "training_history_sha256": sha256(history_path),
        "feature_names": list(FEATURE_NAMES),
        "feature_normalization": "train-trajectories mean and population SD only",
        "architecture": {
            "kind": "one_hidden_layer_tanh_residual_network",
            "hidden_size": HIDDEN_SIZE,
            "parameter_count": int(len(theta)),
            "residual_equation": "delta_C=0.08*tanh(v^T*tanh(W*x_norm+b)+c)",
            "residual_bound": 0.08,
        },
        "loss": {
            "equation": (
                "lambda_data*MSE(C_pred,C_syn_obs) + "
                "lambda_phys*MSE(r_trapezoidal) "
                "+ lambda_BC*MSE(C_pred(t0),1) + lambda_reg*MSE(delta_C,0) "
                "+ lambda_mono*MSE(ReLU(C_pred[j]-C_pred[j-1]),0)"
            ),
            "weights": LOSS_WEIGHTS.as_dict(),
            "physics_discretization": "trajectory-wise trapezoidal first-order ODE residual",
        },
        "trajectory_splits": dataset_manifest["trajectory_splits"],
        "training_seeds": list(TRAINING_SEEDS),
        "selected_initialization_seed": selected_seed,
        "test_split_used_for_selection": False,
        "metrics": selected_metrics,
        "limitations": [
            "The checkpoint is validated only against held-out synthetic DGP trajectories.",
            "It is not evidence of empirical spinach shelf-life accuracy.",
            "The estimator is frozen before and shared across simulation arms.",
        ],
    }
    checkpoint_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return checkpoint_path, checkpoint_manifest_path, history_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir", type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    paths = train(args.artifact_dir.resolve(), force=args.force)
    for path in paths:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
