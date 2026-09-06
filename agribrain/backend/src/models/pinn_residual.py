"""Frozen physics-informed residual for the synthetic spoilage benchmark.

The residual is trained *offline* against quality trajectories emitted by the
versioned synthetic DGP under ``mvp/simulation/pinn``.  It is never fitted to
the 288-row simulation input and is never re-trained inside an episode.  The
target is therefore independent of the mechanistic integrator used as the
baseline, while remaining explicitly synthetic (it is not empirical spinach
validation).

For a feature vector ``x`` the one-hidden-layer network predicts

    delta_C(x) = 0.08 tanh(v^T tanh(W x + b) + c)

and ``C_PINN = C_mech + delta_C``.  Training minimizes data, first-order ODE,
initial-boundary, residual-size and monotonicity terms.  The implementation
below exposes the exact residual vector and analytic Jacobian used by
``scipy.optimize.least_squares`` so every declared loss term differentiates
through the network.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .spoilage import arrhenius_k, compute_spoilage


MAX_RESIDUAL = 0.08
FEATURE_NAMES: tuple[str, ...] = (
    "time_h",
    "tempC",
    "RH",
    "shockG",
    "packaging_index",
    "rh_transient_per_h",
    "k_ref_per_h",
    "ea_over_r_kelvin",
)
DEFAULT_PACKAGING_INDEX = 0.50

_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ARTIFACT_DIR = _REPO_ROOT / "mvp" / "simulation" / "pinn" / "artifacts"
DEFAULT_CHECKPOINT = DEFAULT_ARTIFACT_DIR / "spoilage_pinn_v1.npz"
DEFAULT_CHECKPOINT_MANIFEST = DEFAULT_ARTIFACT_DIR / "spoilage_pinn_v1_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ResidualCheckpoint:
    """Portable frozen network and training-feature normalization."""

    W: np.ndarray
    b: np.ndarray
    v: np.ndarray
    c: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    checkpoint_path: str
    checkpoint_sha256: str
    dataset_sha256: str
    schema_version: int

    @property
    def hidden_size(self) -> int:
        return int(self.W.shape[1])


def _validate_checkpoint_arrays(checkpoint: ResidualCheckpoint) -> None:
    n_features = len(FEATURE_NAMES)
    if checkpoint.W.ndim != 2 or checkpoint.W.shape[0] != n_features:
        raise ValueError("PINN checkpoint W has the wrong feature dimension")
    hidden = checkpoint.W.shape[1]
    if checkpoint.b.shape != (hidden,) or checkpoint.v.shape != (hidden,):
        raise ValueError("PINN checkpoint hidden-layer shapes are inconsistent")
    if checkpoint.feature_mean.shape != (n_features,):
        raise ValueError("PINN checkpoint feature_mean shape is invalid")
    if checkpoint.feature_scale.shape != (n_features,):
        raise ValueError("PINN checkpoint feature_scale shape is invalid")
    arrays = (
        checkpoint.W, checkpoint.b, checkpoint.v,
        checkpoint.feature_mean, checkpoint.feature_scale,
    )
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("PINN checkpoint contains non-finite values")
    if not np.isfinite(checkpoint.c):
        raise ValueError("PINN checkpoint output bias is non-finite")
    if np.any(checkpoint.feature_scale <= 0.0):
        raise ValueError("PINN checkpoint feature scales must be positive")


@lru_cache(maxsize=4)
def load_frozen_checkpoint(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    manifest_path: str | Path = DEFAULT_CHECKPOINT_MANIFEST,
) -> ResidualCheckpoint:
    """Load and cryptographically bind the frozen synthetic checkpoint.

    The manifest must explicitly state that the target is synthetic and that
    no external validation claim is permitted.  This fail-closed check keeps a
    later checkpoint swap from silently changing the scientific claim.
    """

    checkpoint_file = Path(checkpoint_path).resolve()
    manifest_file = Path(manifest_path).resolve()
    if not checkpoint_file.is_file() or not manifest_file.is_file():
        raise FileNotFoundError("frozen PINN checkpoint or manifest is missing")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported PINN checkpoint-manifest schema")
    if manifest.get("status") != "frozen_for_confirmatory_simulation":
        raise ValueError("PINN checkpoint is not frozen for the confirmatory run")
    if manifest.get("target_origin") != "independent_synthetic_dgp":
        raise ValueError("PINN target origin is not the declared independent synthetic DGP")
    if manifest.get("synthetic_only") is not True:
        raise ValueError("PINN checkpoint must be labelled synthetic-only")
    if manifest.get("external_validation") is not False:
        raise ValueError("PINN checkpoint must not claim external validation")
    actual_hash = _sha256(checkpoint_file)
    if manifest.get("checkpoint_sha256") != actual_hash:
        raise ValueError("PINN checkpoint SHA-256 does not match its manifest")

    with np.load(checkpoint_file, allow_pickle=False) as payload:
        checkpoint = ResidualCheckpoint(
            W=np.asarray(payload["W"], dtype=np.float64),
            b=np.asarray(payload["b"], dtype=np.float64),
            v=np.asarray(payload["v"], dtype=np.float64),
            c=float(np.asarray(payload["c"], dtype=np.float64).reshape(())),
            feature_mean=np.asarray(payload["feature_mean"], dtype=np.float64),
            feature_scale=np.asarray(payload["feature_scale"], dtype=np.float64),
            checkpoint_path=str(checkpoint_file),
            checkpoint_sha256=actual_hash,
            dataset_sha256=str(manifest["dataset_sha256"]),
            schema_version=int(manifest["schema_version"]),
        )
    _validate_checkpoint_arrays(checkpoint)
    return checkpoint


def _hours_from_start(frame: pd.DataFrame) -> np.ndarray:
    timestamps = pd.to_datetime(frame["timestamp"])
    return (
        (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy(dtype=np.float64)
        / 3600.0
    )


def build_residual_features(
    frame: pd.DataFrame,
    *,
    k_ref: float,
    ea_over_r: float,
    packaging_index: float | np.ndarray | None = None,
) -> np.ndarray:
    """Build the exact eight features used in offline training and inference."""

    required = {"timestamp", "tempC", "RH", "shockG"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"PINN residual input is missing columns: {missing}")
    time_h = _hours_from_start(frame)
    temp = frame["tempC"].to_numpy(dtype=np.float64)
    rh = frame["RH"].to_numpy(dtype=np.float64)
    shock = frame["shockG"].to_numpy(dtype=np.float64)
    if packaging_index is None:
        if "packaging_index" in frame.columns:
            packaging = frame["packaging_index"].to_numpy(dtype=np.float64)
        else:
            packaging = np.full(len(frame), DEFAULT_PACKAGING_INDEX, dtype=np.float64)
    else:
        packaging = np.broadcast_to(
            np.asarray(packaging_index, dtype=np.float64), (len(frame),),
        ).copy()
    if len(frame) == 0:
        raise ValueError("PINN residual requires at least one row")
    rh_transient = np.zeros(len(frame), dtype=np.float64)
    if len(frame) > 1:
        delta_t = np.diff(time_h)
        if np.any(delta_t <= 0.0):
            raise ValueError("PINN residual timestamps must be strictly increasing")
        rh_transient[1:] = np.abs(np.diff(rh)) / delta_t
    features = np.column_stack((
        time_h,
        temp,
        rh,
        shock,
        packaging,
        rh_transient,
        np.full(len(frame), float(k_ref), dtype=np.float64),
        np.full(len(frame), float(ea_over_r), dtype=np.float64),
    ))
    if not np.isfinite(features).all():
        raise ValueError("PINN residual features contain non-finite values")
    return features


def build_residual_feature_row(
    *,
    time_h: float,
    temp_c: float,
    rh_pct: float,
    shock_g: float,
    rh_transient_per_h: float,
    k_ref: float,
    ea_over_r: float,
    packaging_index: float = DEFAULT_PACKAGING_INDEX,
) -> np.ndarray:
    """Build one online feature row without resetting elapsed time to zero."""

    row = np.asarray([[
        time_h, temp_c, rh_pct, shock_g, packaging_index,
        rh_transient_per_h, k_ref, ea_over_r,
    ]], dtype=np.float64)
    if not np.isfinite(row).all():
        raise ValueError("PINN residual online feature row is non-finite")
    return row


def pack_parameters(W: np.ndarray, b: np.ndarray, v: np.ndarray, c: float) -> np.ndarray:
    return np.concatenate((W.ravel(), b.ravel(), v.ravel(), np.asarray([c])))


def unpack_parameters(
    theta: np.ndarray, *, n_features: int, hidden_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    theta = np.asarray(theta, dtype=np.float64)
    expected = n_features * hidden_size + 2 * hidden_size + 1
    if theta.shape != (expected,):
        raise ValueError(f"expected {expected} PINN parameters, received {theta.shape}")
    offset = n_features * hidden_size
    W = theta[:offset].reshape(n_features, hidden_size)
    b = theta[offset:offset + hidden_size]
    offset += hidden_size
    v = theta[offset:offset + hidden_size]
    c = float(theta[-1])
    return W, b, v, c


def residual_prediction_and_jacobian(
    normalized_features: np.ndarray,
    theta: np.ndarray,
    *,
    hidden_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return bounded residual and its exact Jacobian wrt every parameter."""

    X = np.asarray(normalized_features, dtype=np.float64)
    W, b, v, c = unpack_parameters(
        theta, n_features=X.shape[1], hidden_size=hidden_size,
    )
    hidden = np.tanh(X @ W + b)
    output = np.tanh(hidden @ v + c)
    outer_slope = MAX_RESIDUAL * (1.0 - output * output)
    hidden_slope = (1.0 - hidden * hidden) * v[None, :]

    n, d = X.shape
    jac_W = (
        outer_slope[:, None, None]
        * X[:, :, None]
        * hidden_slope[:, None, :]
    ).reshape(n, d * hidden_size)
    jac_b = outer_slope[:, None] * hidden_slope
    jac_v = outer_slope[:, None] * hidden
    jac_c = outer_slope[:, None]
    jacobian = np.concatenate((jac_W, jac_b, jac_v, jac_c), axis=1)
    return MAX_RESIDUAL * output, jacobian


def predict_residual(
    features: np.ndarray,
    checkpoint: ResidualCheckpoint,
) -> np.ndarray:
    _validate_checkpoint_arrays(checkpoint)
    X = (
        np.asarray(features, dtype=np.float64) - checkpoint.feature_mean
    ) / checkpoint.feature_scale
    theta = pack_parameters(checkpoint.W, checkpoint.b, checkpoint.v, checkpoint.c)
    delta, _ = residual_prediction_and_jacobian(
        X, theta, hidden_size=checkpoint.hidden_size,
    )
    if np.max(np.abs(delta), initial=0.0) > MAX_RESIDUAL + 1e-12:
        raise RuntimeError("PINN residual exceeded its structural bound")
    return delta


def compute_spoilage_with_frozen_residual(
    frame: pd.DataFrame,
    *,
    k_ref: float = 0.0021,
    Ea_R: float = 8000.0,
    T_ref_K: float = 277.15,
    beta: float = 0.25,
    lag_lambda: float = 12.0,
    checkpoint: ResidualCheckpoint | None = None,
    packaging_index: float | np.ndarray | None = None,
) -> pd.DataFrame:
    """Apply the same pre-trained residual to a mechanistic trajectory.

    No optimization or target construction occurs here.  The frozen estimator
    is therefore identical across modes and seeds; only declared environmental
    inputs and episode-level kinetic draws can change its prediction.
    """

    fitted = compute_spoilage(
        frame,
        k_ref=k_ref,
        Ea_R=Ea_R,
        T_ref_K=T_ref_K,
        beta=beta,
        lag_lambda=lag_lambda,
    )
    checkpoint = checkpoint or load_frozen_checkpoint()
    features = build_residual_features(
        fitted,
        k_ref=k_ref,
        ea_over_r=Ea_R,
        packaging_index=packaging_index,
    )
    delta = predict_residual(features, checkpoint)
    quality = fitted["shelf_left"].to_numpy(dtype=np.float64) + delta
    quality = np.minimum.accumulate(np.clip(quality, 0.0, 1.0))
    result = fitted.copy()
    result["mechanistic_shelf_left"] = fitted["shelf_left"].to_numpy(dtype=np.float64)
    result["pinn_residual_correction"] = (
        quality - result["mechanistic_shelf_left"].to_numpy(dtype=np.float64)
    )
    result["shelf_left"] = quality
    result["spoilage_risk"] = 1.0 - quality
    return result


@dataclass(frozen=True)
class LossWeights:
    data: float = 1.0
    physics: float = 0.20
    boundary: float = 1.0
    regularization: float = 0.01
    monotonicity: float = 0.20

    def as_dict(self) -> dict[str, float]:
        return {
            "data": self.data,
            "physics": self.physics,
            "boundary": self.boundary,
            "regularization": self.regularization,
            "monotonicity": self.monotonicity,
        }


class ResidualTrainingObjective:
    """Documented loss and analytic Jacobian on complete trajectories."""

    def __init__(
        self,
        *,
        normalized_features: np.ndarray,
        mechanistic_quality: np.ndarray,
        observed_quality: np.ndarray,
        time_h: np.ndarray,
        temp_c: np.ndarray,
        rh_pct: np.ndarray,
        trajectory_ids: Sequence[str],
        hidden_size: int,
        weights: LossWeights,
        k_ref: np.ndarray,
        ea_over_r: np.ndarray,
        t_ref_k: float = 277.15,
        beta: float = 0.25,
        lag_lambda: float = 12.0,
    ) -> None:
        self.X = np.asarray(normalized_features, dtype=np.float64)
        self.mechanistic = np.asarray(mechanistic_quality, dtype=np.float64)
        self.observed = np.asarray(observed_quality, dtype=np.float64)
        self.time_h = np.asarray(time_h, dtype=np.float64)
        self.temp_c = np.asarray(temp_c, dtype=np.float64)
        self.rh_pct = np.asarray(rh_pct, dtype=np.float64)
        self.trajectory_ids = np.asarray(trajectory_ids, dtype=str)
        self.hidden_size = int(hidden_size)
        self.weights = weights
        self.k_ref = np.asarray(k_ref, dtype=np.float64)
        self.ea_over_r = np.asarray(ea_over_r, dtype=np.float64)
        self.t_ref_k = float(t_ref_k)
        self.beta = float(beta)
        self.lag_lambda = float(lag_lambda)
        n = len(self.X)
        vectors = (
            self.mechanistic, self.observed, self.time_h, self.temp_c,
            self.rh_pct, self.trajectory_ids, self.k_ref, self.ea_over_r,
        )
        if n == 0 or any(len(vector) != n for vector in vectors):
            raise ValueError("PINN training arrays must have one common nonzero length")
        if not all(value >= 0.0 for value in weights.as_dict().values()):
            raise ValueError("PINN loss weights must be nonnegative")

        previous: list[int] = []
        current: list[int] = []
        first: list[int] = []
        for trajectory in dict.fromkeys(self.trajectory_ids.tolist()):
            indices = np.flatnonzero(self.trajectory_ids == trajectory)
            if len(indices) < 2:
                raise ValueError("each PINN training trajectory needs at least two rows")
            order = indices[np.argsort(self.time_h[indices], kind="stable")]
            if np.any(np.diff(self.time_h[order]) <= 0.0):
                raise ValueError("PINN trajectory times must be strictly increasing")
            first.append(int(order[0]))
            previous.extend(int(value) for value in order[:-1])
            current.extend(int(value) for value in order[1:])
        self.first = np.asarray(first, dtype=int)
        self.previous = np.asarray(previous, dtype=int)
        self.current = np.asarray(current, dtype=int)

        dt = self.time_h[self.current] - self.time_h[self.previous]
        mid_time = 0.5 * (
            self.time_h[self.current] + self.time_h[self.previous]
        )
        mid_temp = 0.5 * (
            self.temp_c[self.current] + self.temp_c[self.previous]
        )
        mid_rh = 0.005 * (
            self.rh_pct[self.current] + self.rh_pct[self.previous]
        )
        mid_k_ref = 0.5 * (
            self.k_ref[self.current] + self.k_ref[self.previous]
        )
        mid_ea = 0.5 * (
            self.ea_over_r[self.current] + self.ea_over_r[self.previous]
        )
        rate = arrhenius_k(
            mid_temp,
            k_ref=mid_k_ref,
            Ea_R=mid_ea,
            T_ref_K=self.t_ref_k,
            rh_frac=mid_rh,
            beta=self.beta,
        )
        alpha = (
            mid_time / (mid_time + self.lag_lambda)
            if self.lag_lambda > 0.0 else np.ones_like(mid_time)
        )
        k_eff = np.asarray(rate, dtype=np.float64) * alpha
        self.q = 1.0 / dt + 0.5 * k_eff
        self.r = -1.0 / dt + 0.5 * k_eff

    @staticmethod
    def _scale(weight: float, count: int) -> float:
        return float(np.sqrt(weight / max(count, 1)))

    def residuals_and_jacobian(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        delta, jac_delta = residual_prediction_and_jacobian(
            self.X, theta, hidden_size=self.hidden_size,
        )
        quality = self.mechanistic + delta

        data_error = quality - self.observed
        physics_error = (
            self.q * quality[self.current] + self.r * quality[self.previous]
        )
        boundary_error = quality[self.first] - 1.0
        regularization_error = delta
        monotonicity_raw = quality[self.current] - quality[self.previous]
        monotonicity_active = monotonicity_raw > 0.0
        monotonicity_error = np.maximum(monotonicity_raw, 0.0)

        data_scale = self._scale(self.weights.data, len(data_error))
        phys_scale = self._scale(self.weights.physics, len(physics_error))
        bc_scale = self._scale(self.weights.boundary, len(boundary_error))
        reg_scale = self._scale(
            self.weights.regularization, len(regularization_error),
        )
        mono_scale = self._scale(
            self.weights.monotonicity, len(monotonicity_error),
        )

        residuals = np.concatenate((
            data_scale * data_error,
            phys_scale * physics_error,
            bc_scale * boundary_error,
            reg_scale * regularization_error,
            mono_scale * monotonicity_error,
        ))
        physics_jac = (
            self.q[:, None] * jac_delta[self.current]
            + self.r[:, None] * jac_delta[self.previous]
        )
        monotonicity_jac = (
            jac_delta[self.current] - jac_delta[self.previous]
        )
        monotonicity_jac[~monotonicity_active] = 0.0
        jacobian = np.vstack((
            data_scale * jac_delta,
            phys_scale * physics_jac,
            bc_scale * jac_delta[self.first],
            reg_scale * jac_delta,
            mono_scale * monotonicity_jac,
        ))
        return residuals, jacobian

    def residuals(self, theta: np.ndarray) -> np.ndarray:
        return self.residuals_and_jacobian(theta)[0]

    def jacobian(self, theta: np.ndarray) -> np.ndarray:
        return self.residuals_and_jacobian(theta)[1]

    def metrics(
        self,
        theta: np.ndarray,
        *,
        latent_quality: np.ndarray | None = None,
    ) -> dict[str, float | int]:
        delta, _ = residual_prediction_and_jacobian(
            self.X, theta, hidden_size=self.hidden_size,
        )
        quality = self.mechanistic + delta
        physics = self.q * quality[self.current] + self.r * quality[self.previous]
        increases = quality[self.current] - quality[self.previous]
        deployed_quality = quality.copy()
        for trajectory in dict.fromkeys(self.trajectory_ids.tolist()):
            indices = np.flatnonzero(self.trajectory_ids == trajectory)
            order = indices[np.argsort(self.time_h[indices], kind="stable")]
            deployed_quality[order] = np.minimum.accumulate(
                np.clip(deployed_quality[order], 0.0, 1.0)
            )
        deployed_increases = (
            deployed_quality[self.current] - deployed_quality[self.previous]
        )
        metrics: dict[str, float | int] = {
            "n_rows": int(len(quality)),
            "n_trajectories": int(len(self.first)),
            "observed_rmse": float(np.sqrt(np.mean((quality - self.observed) ** 2))),
            "observed_mae": float(np.mean(np.abs(quality - self.observed))),
            "physics_residual_rmse_per_h": float(np.sqrt(np.mean(physics ** 2))),
            "boundary_rmse": float(np.sqrt(np.mean((quality[self.first] - 1.0) ** 2))),
            "residual_rms": float(np.sqrt(np.mean(delta ** 2))),
            "residual_abs_max": float(np.max(np.abs(delta))),
            "monotonicity_violation_count": int(np.count_nonzero(increases > 1e-9)),
            "unit_interval_violation_count": int(
                np.count_nonzero((quality < 0.0) | (quality > 1.0))
            ),
            "deployed_monotonicity_violation_count": int(
                np.count_nonzero(deployed_increases > 1e-12)
            ),
            "deployed_unit_interval_violation_count": int(np.count_nonzero(
                (deployed_quality < 0.0) | (deployed_quality > 1.0)
            )),
            "deployed_observed_rmse": float(np.sqrt(np.mean(
                (deployed_quality - self.observed) ** 2
            ))),
        }
        if latent_quality is not None:
            latent = np.asarray(latent_quality, dtype=np.float64)
            if latent.shape != quality.shape:
                raise ValueError("latent quality shape does not match predictions")
            metrics["latent_rmse"] = float(
                np.sqrt(np.mean((quality - latent) ** 2))
            )
            metrics["mechanistic_latent_rmse"] = float(
                np.sqrt(np.mean((self.mechanistic - latent) ** 2))
            )
            metrics["deployed_latent_rmse"] = float(
                np.sqrt(np.mean((deployed_quality - latent) ** 2))
            )
        return metrics


def build_training_objective(
    frame: pd.DataFrame,
    *,
    trajectory_ids: Sequence[str],
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    hidden_size: int,
    weights: LossWeights,
) -> ResidualTrainingObjective:
    """Construct the objective for an explicit trajectory-ID split."""

    feature_mean = np.asarray(feature_mean, dtype=np.float64)
    feature_scale = np.asarray(feature_scale, dtype=np.float64)
    if feature_mean.shape != (len(FEATURE_NAMES),):
        raise ValueError("PINN feature mean has the wrong shape")
    if feature_scale.shape != (len(FEATURE_NAMES),) or np.any(feature_scale <= 0.0):
        raise ValueError("PINN feature scales must be positive and complete")
    subset = frame[frame["trajectory_id"].astype(str).isin(trajectory_ids)].copy()
    subset = subset.sort_values(["trajectory_id", "time_h"], kind="stable")
    if subset.empty:
        raise ValueError("PINN split selects no trajectories")
    X = subset.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
    X = (X - feature_mean) / feature_scale
    return ResidualTrainingObjective(
        normalized_features=X,
        mechanistic_quality=subset["mechanistic_quality_fraction"].to_numpy(float),
        observed_quality=subset["observed_quality_fraction"].to_numpy(float),
        time_h=subset["time_h"].to_numpy(float),
        temp_c=subset["tempC"].to_numpy(float),
        rh_pct=subset["RH"].to_numpy(float),
        trajectory_ids=subset["trajectory_id"].astype(str).to_numpy(),
        hidden_size=hidden_size,
        weights=weights,
        k_ref=subset["k_ref_per_h"].to_numpy(float),
        ea_over_r=subset["ea_over_r_kelvin"].to_numpy(float),
    )


__all__ = [
    "DEFAULT_ARTIFACT_DIR", "DEFAULT_CHECKPOINT", "DEFAULT_CHECKPOINT_MANIFEST",
    "DEFAULT_PACKAGING_INDEX", "FEATURE_NAMES", "LossWeights", "MAX_RESIDUAL",
    "ResidualCheckpoint", "ResidualTrainingObjective", "build_residual_feature_row",
    "build_residual_features",
    "build_training_objective", "compute_spoilage_with_frozen_residual",
    "load_frozen_checkpoint", "pack_parameters", "predict_residual",
    "residual_prediction_and_jacobian", "unpack_parameters",
]
